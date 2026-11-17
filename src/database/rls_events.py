"""Phase D RLS — re-bind the tenant GUC at the start of every transaction.

``bind_session_tenant`` (``src/database/session.py``) sets ``app.current_tenant``
transaction-scoped (``set_config(..., is_local=true)``, PgBouncer-safe), and the
request path binds it exactly once via the ``apply_tenant_rls`` router dep. But a
single request handler can COMMIT before it is finished reading — most notably
``get_run`` (``src/api/routes/runs.py``) drives ``RunUsageReconciler.reconcile``,
which commits the request session mid-handler. That commit ends the transaction
the GUC was bound in, so the next RLS-gated read (``_project_id_for_deliverable``)
runs in a fresh, UNBOUND transaction → under enforce the GUC reads back as ``''``
→ ``NULLIF`` → NULL → the RLS predicate is false → 0 rows → a spurious
``404 "run not found"`` on a run the caller genuinely owns. The same
commit-before-RLS-read hazard exists in the admin and cancel/pause/resume routes.

This installs an ``after_begin`` listener that re-applies ``app.current_tenant``
at the start of EVERY transaction — from ``session.info[SESSION_TENANT_KEY]`` (the
tenant this session was explicitly bound to, which is the EFFECTIVE tenant
including a super_admin ``X-Active-Tenant-Id`` scope-switch), falling back to the
identity ContextVar (which ``AuthMiddleware`` sets per request from the token
CLAIM, and the dispatcher sets around background work) — so any
number of intra-request or background commits self-heal. Inert unless
``TENANT_RLS_MODE=enforce`` and a tenant is in context, so pre-auth / no-tenant
background sessions stay no-ops. The dispatcher's explicit ``bind_run_tenant``
calls remain load-bearing for its bootstrap (only a ``run_id`` in hand, ContextVar
not yet set) — this listener is additive. Scope is this process only.

Mirrors the proven fix in ``analysis_service``
(``src/shared/database/rls_events.py``, PR #66).
"""

from __future__ import annotations

from sqlalchemy import Connection, event
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, SessionTransaction
from w2v_common.auth.identity_context import get_tenant_id_from_context

from src.database.session import (
    BIND_TENANT_DRIVER_SQL,
    RESET_READ_WRITE_SQL,
    ROW_SECURITY_PIN_SQL,
    SESSION_TENANT_KEY,
    _rls_enforced,
    backend_is_in_recovery,
    note_read_only_reset_rejected,
    warn_if_read_only_was_leaked,
    warn_if_row_security_was_leaked,
)

_installed = False


def install_rls_tenant_rebind() -> None:
    """Register the after_begin GUC re-bind listener once per process."""
    global _installed
    if _installed:
        return

    @event.listens_for(Session, "after_begin")
    def _rebind_tenant(
        session: Session, transaction: SessionTransaction, connection: Connection
    ) -> None:
        if not _rls_enforced():
            return
        # PREFER the tenant this session was explicitly bound to (recorded by
        # bind_session_tenant / bind_run_tenant in session.info). The identity
        # ContextVar carries the token's CLAIM tenant, which is NOT the effective
        # tenant when a super_admin scope-switches with X-Active-Tenant-Id: using
        # it here silently rebound transaction 2+ of such a request to the admin's
        # own tenant → 0 rows under enforce → 200/204 with wrong (empty) data.
        # Falls back to the ContextVar for sessions nobody bound explicitly
        # (unchanged behaviour for the ordinary claim==effective request path).
        # FIRST STATEMENT OF THE TRANSACTION, and it has to be. A leaked
        # session-level ``default_transaction_read_only = on`` on the borrowed
        # PgBouncer backend makes every write fail with ``ReadOnlySqlTransaction:
        # cannot execute UPDATE in a read-only transaction``; Postgres only lets us
        # clear it before the transaction's first snapshot, so this must precede the
        # pin/bind SELECT below. Moving it after that SELECT does not merely weaken
        # the fix — it raises ``transaction read-write mode must be set before any
        # query`` and breaks EVERY transaction. See ``RESET_READ_WRITE_SQL``.
        # ⛔ …but ONLY when the backend can legally accept it. On a read-only STANDBY
        # Postgres rejects this statement outright (0A000 "cannot set transaction
        # read-write mode during recovery"), and since it is the first statement of
        # every transaction, issuing it unconditionally would take the WHOLE service
        # down on a standby — reads and health probes included — where it previously
        # degraded to read-only-but-working. ``backend_is_in_recovery`` is answered
        # once, by the probe at engine init, so the healthy path pays nothing here.
        # The except-branch is the safety net for a failover the probe could not have
        # seen: it latches, so only the discovering transaction dies.
        if not backend_is_in_recovery():
            try:
                connection.exec_driver_sql(RESET_READ_WRITE_SQL)
            except DBAPIError as exc:
                note_read_only_reset_rejected(exc)
                raise
        bound = session.info.get(SESSION_TENANT_KEY)
        tenant = bound if isinstance(bound, str) and bound else get_tenant_id_from_context()
        if not tenant:
            # No tenant to bind, but the row_security pin is still load-bearing: a
            # leaked ``row_security = off`` on the pooled server connection makes
            # every RLS-table query RAISE rather than return 0 rows, so an unbound
            # background transaction (the reconciler's cross-tenant orphan scan) dies
            # instead of reading through its SECURITY DEFINER resolver. See
            # ``ROW_SECURITY_PIN_SQL`` in ``src/database/session.py``.
            #
            # DELIBERATE: with the pin in place an unbound read of an RLS table now
            # returns 0 rows instead of raising. That is the CORRECT semantics — an
            # unbound tenant makes the policy predicate false, which is exactly what
            # every other unbound read in the service does, and raising here is what
            # crash-looped the reconciler. It is NOT a silent failure: the poisoning
            # itself is named by the WARNING below, which fires on this same
            # statement. ``bind_run_tenant``'s fail-loud contract ("must not execute
            # silently against 0 rows") applies to a run whose tenant cannot be
            # RESOLVED — that still raises, in session.py — not to this scan, which
            # has no tenant BY DESIGN.
            row = connection.exec_driver_sql(ROW_SECURITY_PIN_SQL).first()
        else:
            # ``str(tenant)`` is LOAD-BEARING, not cosmetic. Since w2v-common
            # 0.67.0 ``get_tenant_id_from_context()`` returns a ``uuid.UUID``,
            # and psycopg adapts a UUID to the postgres ``uuid`` type — but
            # ``set_config(text, text, bool)`` has no ``uuid`` overload, so the
            # bind fails with `function set_config(unknown, uuid, boolean) does
            # not exist` and takes the WHOLE transaction with it. That is the
            # tenant-isolation rebind on every multi-commit background path, so
            # the failure is a hard error rather than a scoping hole — but it is
            # invisible to `ty`, because `exec_driver_sql` params are untyped.
            row = connection.exec_driver_sql(BIND_TENANT_DRIVER_SQL, (str(tenant),)).first()
        # ``prior_row_security`` / ``prior_read_only`` are the values the pooled
        # connection arrived with, read in the SAME round-trip (see session.py).
        # Warn when either was poisoned so the sibling services that share this pool
        # and have NO pin/reset get named. Both are reported: they are separate
        # leaks and an incident can carry either or both.
        if row is not None:
            warn_if_row_security_was_leaked(row[0])
            warn_if_read_only_was_leaked(row[1])

    _installed = True
