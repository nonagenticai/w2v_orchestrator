"""Database session management module."""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .connection import get_engine

logger = logging.getLogger(__name__)

_async_session_maker: async_sessionmaker[AsyncSession] | None = None
# Serialises lazy init in `get_session_maker()` so concurrent
# first-callers don't each construct a separate sessionmaker —
# same race shape as `get_engine`.
_session_maker_init_lock: asyncio.Lock | None = None


async def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Get or create the global async session maker.

    Concurrent first-callers serialise on ``_session_maker_init_lock``;
    only one ``async_sessionmaker`` ever exists per process.
    """
    global _async_session_maker, _session_maker_init_lock
    if _async_session_maker is not None:
        return _async_session_maker
    if _session_maker_init_lock is None:
        _session_maker_init_lock = asyncio.Lock()
    async with _session_maker_init_lock:
        if _async_session_maker is None:
            engine = await get_engine()
            _async_session_maker = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )
    return _async_session_maker


async def get_db() -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency for database sessions."""
    session_maker = await get_session_maker()
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession]:
    """Context manager for manual database session management."""
    session_maker = await get_session_maker()
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def close_session_maker() -> None:
    """Reset the global session maker instance."""
    global _async_session_maker
    _async_session_maker = None


# ─── Phase D RLS — bind app.current_tenant on a session (single source) ────────
# Used by BOTH the request path (apply_tenant_rls dep) and every background
# session. Transaction-scoped (is_local=true) → PgBouncer-pool-safe and reset at
# commit. Gated by TENANT_RLS_MODE so it is a hard no-op until the cutover, which
# is what keeps all of Phase D increment 3 inert until the role switch + enforce
# flip land together (see src/database/rls.py).
def _rls_enforced() -> bool:
    # Imported lazily: src.config imports settle after this module at startup.
    from src.config import get_settings

    return get_settings().tenant_rls_mode == "enforce"


#: ``Session.info`` key recording the tenant a session was EXPLICITLY bound to.
#: The GUC itself is transaction-scoped (``is_local=true``) and dies at COMMIT, so
#: the ``after_begin`` re-bind listener (``src/database/rls_events.py``) needs a
#: durable, session-lifetime record of *which* tenant to re-bind. The identity
#: ContextVar alone is not that record: it holds the token's CLAIM tenant, which
#: differs from the effective tenant whenever a ``super_admin`` scope-switches via
#: ``X-Active-Tenant-Id`` — so from the second transaction onward the request
#: silently rebound to the ADMIN's own tenant and every RLS-gated read matched 0
#: rows. ``Session.info`` is per-session (a request session is created and dropped
#: per request; every background unit of work opens its own), so it is
#: request-scoped by construction — no ContextVar reset/leak surface, and a
#: background task that copies the request's context is unaffected.
SESSION_TENANT_KEY = "w2v_bound_tenant"

#: ``row_security = off`` makes Postgres RAISE (42501 "query would be affected by
#: row-level security policy for table …") instead of applying a policy, for ANY
#: query touching an RLS-enabled table — reads included. It is a plain USERSET
#: session GUC, and under PgBouncer ``pool_mode = transaction`` a session-level
#: ``SET`` issued by ANY client sharing the pool (``pg_dump`` opens with exactly
#: ``SET row_security = off``) stays on the server connection: the reset query is
#: not run between transactions. A poisoned connection then fails every statement
#: the orchestrator makes on ``runs`` / ``projects`` / ``stage_outputs`` — which is
#: how the reconciler loop and the resumed ``RunDispatcher._run`` task both died on
#: 2026-08-20 while ``/api/runs/{id}/status`` 500'd on the same pod.
#:
#: So pin it back ON, transaction-locally, at the head of every transaction via the
#: ``after_begin`` listener (``src/database/rls_events.py``, installed by
#: ``asgi.py``). ``on`` is the Postgres default and the STRICTER value — it means
#: "apply the policy", never "skip it" — so this cannot widen what the restricted
#: runtime role sees; it only stops another client's leaked GUC from turning our
#: tenant-scoped query into a hard error.
#:
#: ⚠️ The listener is the SINGLE source of the pin. ``bind_session_tenant`` /
#: ``bind_run_tenant`` deliberately do NOT re-pin: ``after_begin`` fires on the
#: FIRST statement of a transaction, i.e. strictly before their own ``execute``,
#: so a second copy is unreachable by construction (it survives its own mutation
#: test — deleting it changes nothing). One source, one place to read.
#:
#: ⚠️ Pinning silently would DELETE THE ONLY ALARM. At least seven sibling w2v
#: services share this PgBouncer pool under ``TENANT_RLS_MODE=enforce`` and have no
#: pin; a leaked GUC still 500s them. If the orchestrator self-healed quietly, the
#: one service that could NAME the root cause would say nothing. So read the prior
#: value in the SAME statement (zero extra round-trips — ``set_config`` returns the
#: NEW value and can never tell you the old one) and WARN when it was not ``on``.
#: The read lives in a ``FROM`` subquery, not beside ``set_config`` in the target
#: list, because target-list evaluation order is not guaranteed: the ``FROM`` is
#: evaluated first by definition, so ``prior_row_security`` is always the
#: pre-pin value.
#:
#: ``prior_read_only`` rides along in the SAME subquery for the SECOND leaked-GUC
#: class (``default_transaction_read_only``; see ``RESET_READ_WRITE_SQL``). It
#: deliberately reads ``default_transaction_read_only`` and NOT
#: ``transaction_read_only``: the reset below has already forced the latter ``off``
#: transaction-locally by the time this statement runs, so it would always read
#: back ``off`` and the alarm would be dead. ``default_transaction_read_only`` is
#: the SESSION-level setting the poisoning client actually left behind, and
#: ``SET LOCAL transaction_read_only`` does not touch it — measured, not assumed.
_PRIOR_GUCS = (
    " FROM (SELECT current_setting('row_security', true) AS prior_row_security, "
    "current_setting('default_transaction_read_only', true) AS prior_read_only) AS _rs"
)

#: Paramless pin + probe, for a transaction with no tenant to bind.
ROW_SECURITY_PIN_SQL = (
    "SELECT _rs.prior_row_security, _rs.prior_read_only, "
    "set_config('row_security', 'on', true)" + _PRIOR_GUCS
)
#: Pin + probe + tenant bind, DBAPI positional form (the ``after_begin`` listener
#: runs on a raw ``Connection``, which has no named-param binding).
BIND_TENANT_DRIVER_SQL = (
    "SELECT _rs.prior_row_security, _rs.prior_read_only, "
    "set_config('row_security', 'on', true), "
    "set_config('app.current_tenant', %s, true)" + _PRIOR_GUCS
)

#: Undo a leaked session-level ``default_transaction_read_only = on`` for THIS
#: transaction. The exact sibling of the ``row_security`` pin above, for the other
#: GUC that PgBouncer's ``server_reset_query`` does not scrub: the pooler's reset
#: covers only ``RESET row_security``, so ANY session-level ``SET`` left on a server
#: connection leaks to every later borrower of that backend, permanently and
#: per-connection — which reads exactly like a flaky application bug because only
#: the transactions that happen to borrow the poisoned backend fail.
#:
#: Live symptom this exists for: ``psycopg.errors.ReadOnlySqlTransaction: cannot
#: execute UPDATE in a read-only transaction`` on the preview-proxy sweeper's
#: bookkeeping write, ~15/hr/replica on 2026-08-23, AFTER the transcode and the
#: MinIO upload had already succeeded — so the pod burned the whole expensive job
#: and then could not record that it had done it.
#:
#: ⚠️ ORDER IS LOAD-BEARING and is the reason this is a separate statement rather
#: than another ``set_config`` folded into the SELECT above. Postgres
#: (``check_transaction_read_only``) refuses to clear read-only once the
#: transaction has taken its first snapshot. MEASURED on postgres 18:
#:
#:   BEGIN; SET LOCAL transaction_read_only = off; SELECT set_config(...); UPDATE …
#:     -> UPDATE 1                                          (works)
#:   BEGIN; SELECT set_config(...); SET LOCAL transaction_read_only = off; …
#:     -> ERROR: transaction read-write mode must be set before any query
#:
#: Any SELECT — including the pin/bind statement itself — takes that snapshot, so
#: folding this in as a third ``set_config`` in the target list CANNOT work: the
#: SELECT's own snapshot is established before its target list is evaluated. This
#: MUST be issued first, and ``after_begin`` is the only hook that fires after BEGIN
#: and before the transaction's first statement.
#:
#: ``SET LOCAL`` (not ``SET``) so it dies at COMMIT and never becomes a leak of our
#: own on the shared pool — the same discipline as ``set_config(..., is_local=true)``
#: everywhere else in this module. Verified: the session-level
#: ``default_transaction_read_only`` still reads ``on`` after such a transaction
#: commits, i.e. we override the poison without inheriting or spreading it.
RESET_READ_WRITE_SQL = "SET LOCAL transaction_read_only = off"

#: ⛔ …and it is ILLEGAL on a read-only STANDBY, which is why nothing above may
#: issue it unconditionally. ``check_transaction_read_only`` (postgres
#: ``src/backend/commands/variable.c``) refuses to clear read-only while
#: ``RecoveryInProgress()``:
#:
#:   ERROR:  cannot set transaction read-write mode during recovery   (SQLSTATE 0A000)
#:
#: A standby is not hypothetical in this fleet: ``postgres.test`` answers on the
#: write PRIMARY, while a standby node runs a read-only standby on its
#: OWN local ``:5432`` (``pg_is_in_recovery() = t``). A failover, a misroute or a
#: copied DSN is enough to point a pod at it. Since ``after_begin`` fires on the
#: FIRST statement of EVERY transaction, an unconditional reset would turn that
#: standby from "reads work, writes fail" into "every transaction dies at BEGIN" —
#: pure reads and the DB-touching health probe included. That is strictly worse than
#: the degradation it replaces, so the reset is gated on ``backend_is_in_recovery``.
RECOVERY_PROBE_SQL = "SELECT pg_is_in_recovery()"

#: SQLSTATE (``feature_not_supported``) of the rejection above. Matched instead of
#: the message text, which is localised. Nothing ELSE a bare
#: ``SET LOCAL transaction_read_only = off`` can return carries this code.
RECOVERY_REJECTION_SQLSTATE = "0A000"

#: Process-global: is the backend this engine talks to in recovery? Written ONCE,
#: by the probe ``get_engine`` runs on a fresh connection right after the engine is
#: built (``src/database/connection.py``) — deliberately NOT re-probed per
#: transaction, which would add a round-trip to every transaction in the service to
#: answer a question whose answer changes at most once per pod lifetime.
#:
#: Default ``False`` = "assume primary", so a process that never ran the probe (the
#: test suite builds its own engines) keeps the pre-existing behaviour exactly.
_backend_in_recovery: bool = False


def set_backend_in_recovery(in_recovery: bool) -> None:
    """Record whether the backend is a read-only standby (see ``RECOVERY_PROBE_SQL``)."""
    global _backend_in_recovery
    _backend_in_recovery = in_recovery


def backend_is_in_recovery() -> bool:
    """Whether the read-write reset must be SKIPPED for this backend."""
    return _backend_in_recovery


def reset_backend_recovery_state() -> None:
    """Back to the 'assume primary' default (process-global; used by tests)."""
    set_backend_in_recovery(False)


def note_read_only_reset_rejected(exc: DBAPIError) -> bool:
    """Latch 'this backend is a standby' from a REJECTED read-write reset.

    The init probe answers the question for the connection the engine was built on;
    this is the safety net for the case it cannot cover — the DSN starting to answer
    from a standby LATER (failover, pgbouncer repoint). Without it a stale
    ``False`` would kill every transaction for the rest of the pod's life.

    Returns whether the rejection was the recovery one. The caller re-raises either
    way: the transaction that DISCOVERS the flip is already aborted server-side and
    cannot be rescued from here (the reset is the transaction's first statement, so
    a SAVEPOINT would need a round-trip on every healthy transaction to save this
    one). Latching means exactly ONE transaction pays for the discovery and every
    later one degrades to read-only-but-working.
    """
    if getattr(exc.orig, "sqlstate", None) != RECOVERY_REJECTION_SQLSTATE:
        return False
    if not _backend_in_recovery:
        logger.warning(
            "the backend rejected '%s' with SQLSTATE %s — it is a read-only STANDBY "
            "(pg_is_in_recovery() = t), which the startup probe did not see. The DSN "
            "now answers from a standby: writes cannot succeed here at all. Skipping "
            "the leaked-read-only reset from now on so reads keep working; this one "
            "transaction is lost.",
            RESET_READ_WRITE_SQL,
            RECOVERY_REJECTION_SQLSTATE,
        )
    set_backend_in_recovery(True)
    return True


#: Tenant bind only — the pin is already applied by the ``after_begin`` listener
#: that fired on this very statement (see above).
_BIND_TENANT_SQL = "SELECT set_config('app.current_tenant', :tenant, true)"

#: Minimum seconds between leaked-GUC warnings. A poisoned pooled connection is
#: borrowed by every transaction until PgBouncer retires it, so an unthrottled
#: warning would be one line per transaction — thousands per minute from the
#: reconciler alone. Throttling keeps the diagnosis loud enough to find and quiet
#: enough to read; the suppressed count is carried in the next line so the volume
#: is never lost.
_ROW_SECURITY_WARN_INTERVAL_S = 60.0
_row_security_warn_state: tuple[float, int] = (0.0, 0)
#: Independent throttle for the read-only leak. Deliberately NOT shared with the
#: ``row_security`` one: they are different poisons from (potentially) different
#: clients, and a shared throttle would let whichever fired first silence the
#: other for a minute — hiding half the diagnosis in exactly the incident where
#: both matter.
_read_only_warn_state: tuple[float, int] = (0.0, 0)


def reset_row_security_warn_throttle() -> None:
    """Clear the leaked-GUC warning throttles (process-global; used by tests)."""
    global _row_security_warn_state, _read_only_warn_state
    _row_security_warn_state = (0.0, 0)
    _read_only_warn_state = (0.0, 0)


def warn_if_read_only_was_leaked(prior: str | None) -> bool:
    """Warn when ``default_transaction_read_only`` was ``on`` before we reset it.

    ``prior`` is the ``prior_read_only`` column of ``ROW_SECURITY_PIN_SQL`` /
    ``BIND_TENANT_DRIVER_SQL``. Returns whether a line was emitted, so tests can
    assert the DECISION rather than log plumbing. Never raises, for the same
    reason as its ``row_security`` sibling: an alarm that can kill the transaction
    it describes is worse than the silence it replaces.
    """
    global _read_only_warn_state
    if prior is None or prior == "off":
        return False
    now = time.monotonic()
    last, suppressed = _read_only_warn_state
    if last and now - last < _ROW_SECURITY_WARN_INTERVAL_S:
        _read_only_warn_state = (last, suppressed + 1)
        return False
    _read_only_warn_state = (now, 0)
    logger.warning(
        "default_transaction_read_only was %r on a pooled server connection and has "
        "been overridden to read-write for this transaction. Root cause: PgBouncer "
        "runs pool_mode=transaction and its server_reset_query scrubs only "
        "'RESET row_security', so a session-level 'SET default_transaction_read_only "
        "= on' from ANY client sharing the pool sticks on that server connection and "
        "every later borrower inherits it — permanently, and only on that one "
        "backend, which is why it presents as a flaky application bug. Every write "
        "on the poisoned backend fails with 'cannot execute UPDATE in a read-only "
        "transaction'. This service self-heals via SET LOCAL; the OTHER w2v services "
        "on the same pool have no such reset and are failing writes right now. Find "
        "and stop the client issuing the SET. "
        "(%d further occurrence(s) suppressed in the last %.0fs)",
        prior,
        suppressed,
        _ROW_SECURITY_WARN_INTERVAL_S,
    )
    return True


def warn_if_row_security_was_leaked(prior: str | None) -> bool:
    """Warn when ``row_security`` was NOT ``on`` before we pinned it.

    ``prior`` is the ``prior_row_security`` column of ``ROW_SECURITY_PIN_SQL`` /
    ``BIND_TENANT_DRIVER_SQL``. Returns whether a line was emitted (the throttle
    can swallow one), so callers/tests can assert on the decision rather than on
    log plumbing. Never raises: an alarm that can kill the transaction it is
    describing is worse than the silence it replaces.
    """
    global _row_security_warn_state
    if prior is None or prior == "on":
        return False
    now = time.monotonic()
    last, suppressed = _row_security_warn_state
    if last and now - last < _ROW_SECURITY_WARN_INTERVAL_S:
        _row_security_warn_state = (last, suppressed + 1)
        return False
    _row_security_warn_state = (now, 0)
    logger.warning(
        "row_security was %r on a pooled server connection and has been pinned back "
        "to 'on' for this transaction. Root cause: PgBouncer runs pool_mode="
        "transaction with no server_reset_query_always, so a session-level "
        "'SET row_security = off' from ANY client sharing the pool (pg_dump issues "
        "exactly that) sticks on the server connection and every later borrower "
        "inherits it. Under TENANT_RLS_MODE=enforce that makes Postgres RAISE 42501 "
        "'query would be affected by row-level security policy' on every RLS-gated "
        "statement instead of applying the policy. The orchestrator self-heals via "
        "this pin; the OTHER w2v services on the same pool (analysis, "
        "video-compiler, frontend-test-runner, …) run enforce WITHOUT a pin and are "
        "500ing right now. Find and stop the client issuing the SET. "
        "(%d further occurrence(s) suppressed in the last %.0fs)",
        prior,
        suppressed,
        _ROW_SECURITY_WARN_INTERVAL_S,
    )
    return True


async def bind_session_tenant(session: AsyncSession, tenant_id: UUID | None) -> None:
    """Bind ``app.current_tenant`` to ``tenant_id`` for this transaction (no-op
    unless TENANT_RLS_MODE=enforce). Pass the tenant a background unit of work
    belongs to BEFORE its first query so the RLS policies admit its rows.

    Also records the tenant in ``session.info[SESSION_TENANT_KEY]`` so the
    ``after_begin`` listener re-binds THIS tenant (not the claim) after a commit."""
    if tenant_id is None or not _rls_enforced():
        return
    session.info[SESSION_TENANT_KEY] = str(tenant_id)
    await session.execute(text(_BIND_TENANT_SQL), {"tenant": str(tenant_id)})


async def bind_run_tenant(session: AsyncSession, run_id: UUID) -> None:
    """Resolve a run's tenant via the ``run_tenant`` SECURITY DEFINER resolver and
    bind the GUC — the dispatcher bootstrap (it has only a run_id, and reading the
    run row is itself RLS-gated under enforce). No-op unless enforce. Fail-loud if
    the run has no resolvable tenant under enforce (a run that can't be scoped must
    not execute silently against 0 rows)."""
    if not _rls_enforced():
        return
    tenant = (
        await session.execute(text("SELECT run_tenant(:rid)"), {"rid": str(run_id)})
    ).scalar_one_or_none()
    if tenant is None:
        raise RuntimeError(
            f"run {run_id} has no resolvable tenant; cannot execute under RLS enforce"
        )
    session.info[SESSION_TENANT_KEY] = str(tenant)
    await session.execute(text(_BIND_TENANT_SQL), {"tenant": str(tenant)})
