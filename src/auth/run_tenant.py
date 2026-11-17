"""The tenant a background run is BILLED as — parsed strictly, never defaulted.

Every LLM call a run makes goes to worker_management, which
``w2v_common.auth.outbound`` classifies as a METERED destination. On that hop
``identity_event_hook`` stamps ``X-Effective-Tenant-Id: <bound tenant>``
**verbatim**, and worker_management + inference-router roll their cost events up
by that header. The value bound at dispatch is therefore not an internal detail:
it IS the billing record.

w2v-common 0.67.0 deleted the nil-uuid *substitution* (``auth.default_tenant_id``
/ ``UNATTRIBUTED_TENANT_ID``) so the library can no longer INVENT a sentinel. It
did not, at that version, reject one it was HANDED: ``set_identity_context`` runs
the value through ``parse_tenant_id``, and ``00000000-0000-0000-0000-000000000000``
is a perfectly valid UUID. Measured on w2v-common 0.67.2, binding the nil uuid
with a vouched provenance and firing the real ``identity_event_hook`` at
``…/workers/growth-bundle/message``::

    nil-uuid   -> X-Effective-Tenant-Id='00000000-0000-0000-0000-000000000000'
    absent     -> X-Effective-Tenant-Id=None   (billed to the w2v service account)

**Updated for w2v-common 0.68.0:** ``parse_tenant_id`` now refuses the nil uuid
itself, on both of its paths, so the first line above is no longer reachable —
``set_identity_context(tenant_id=<nil>)`` raises ``InvalidTenantIdError``. That
does NOT retire this guard, for two reasons. It still fires EARLIER, at dispatch,
so an unbillable legacy snapshot is a graceful skip (the parent run completes and
only the discretionary re-render is dropped) rather than an exception thrown
inside the spawn. And the second line — an ABSENT tenant silently billed to the
w2v service account — is untouched by the library change and remains this
helper's own responsibility.

Both are unattributable spend. The orchestrator is the only place that can stop
them, because it is the only place that knows the run's tenant came out of a
``config_snapshot`` written (for legacy rows) while the nil uuid was still the
configured fallback — see ``src/auth/tenant_scope.py`` for that history.

The second line above is the quieter half and is why "just drop the header" is
NOT an acceptable degradation: an absent tenant does not fail, it silently bills
the shared w2v service-account tenant. Refusing is the only honest option, so
this helper RAISES and never returns a placeholder.

Contrast with ``tenant_scope_key`` (``src/auth/tenant_scope.py``), which answers
a different question — "may I use this as a cache/storage prefix?" — and answers
``None`` for "no, fall back to unscoped". Unscoped is a safe cache; unattributed
is not a safe invoice. Do not swap one for the other.

Scope of this guard — the API request lane is deliberately NOT covered
---------------------------------------------------------------------
Measured (2026-08-27), not estimated. ``src/`` has **40** ``.message(`` call
sites on a ``WMClient`` (``grep -rn '\\.message(' --include=*.py src/`` returns
45; five are prose — ``batch_ledger.py:9``, ``llm_budget.py:181/224/293``,
``run_dispatcher.py:529``). **38** are DAG stage handlers under
``services/stages/``, reached only through ``RunDispatcher``, so this guard's
one binding attributes all of them. The remaining **2** —
``src/services/storyboard/assist.py:322`` (``_DRAFT_WORKER``) and ``:507``
(``_REWRITE_WORKER``) — are on the SYNCHRONOUS API lane: their only importer is
``src/api/routes/storyboard.py:87``, so their tenant comes from
``AuthMiddleware``, not from here. No stage module carrying a ``.message(`` is
imported by anything under ``src/api/``.

Those two are NOT guarded, on purpose:

* There is nowhere small to put it. ``src/auth/middleware.py`` is a 26-line
  re-export of ``w2v_common.auth.middleware.AuthMiddleware`` (a pinned wheel).
  Every inbound lane funnels through ONE helper there,
  ``middleware._tenant_or_unauthorized`` — the natural home for a nil check, and
  a fleet-wide library change affecting every service, not an orchestrator one.
* The premise that an inbound ``X-Tenant-Id: <nil uuid>`` reaches the wire is
  FALSE. Firing the real ``identity_event_hook`` at the worker_management URL
  with the nil uuid bound, per provenance rung: ``unvouched-header`` /
  ``trusted-network-default`` / ``auth-disabled`` → ``UnvouchedTenantError``
  (nothing delegated); ``license-token`` / ``jwt-claim`` / ``vouched-onbehalf``
  → ``X-Effective-Tenant-Id: 00000000-…``. The trusted-network lane is the ONLY
  one that reads the raw ``X-Tenant-Id`` VALUE, and it is unvouched — so a
  caller-settable header already fails closed. The residual exposure is narrower
  and needs a credential actually ISSUED for the nil tenant (it is registered in
  ``licenses.tenants`` as ``default-anon``), which no caller can mint.
* Forcing it would trade a narrow billing hole for a broad availability one. A
  refusal at the auth boundary is a blanket 401 on EVERY endpoint, not just the
  two metered ones — including reads, where an unattributable tenant is merely a
  data scope and costs nothing. That is a real error-contract change for callers
  that work today.

If the nil tenant must be refused inbound too, do it in ``w2v_common``'s
``_tenant_or_unauthorized`` as a fleet decision, not as a local wrapper here.
"""

from __future__ import annotations

from uuid import UUID

from w2v_common.tenant import InvalidTenantIdError, parse_tenant_id

__all__ = ["billable_tenant_id"]

#: Named so the refusal message says what it refused without echoing a value that
#: might be a leaked ``Bearer`` token (``parse_tenant_id`` takes the same care).
_NIL_UUID = UUID(int=0)


def billable_tenant_id(value: object, *, source: str = "run config_snapshot") -> UUID:
    """Return ``value`` as the run's billing tenant, or raise.

    Raises :class:`w2v_common.tenant.MissingTenantError` when absent/blank and
    :class:`w2v_common.tenant.InvalidTenantIdError` when unparseable or nil.
    Both subclass :class:`w2v_common.tenant.TenantError`, so a dispatcher that
    treats "no billable tenant" as one refusal can catch the base.
    """
    parsed = parse_tenant_id(value, source=source)
    if parsed == _NIL_UUID:
        raise InvalidTenantIdError(
            f"{source}: tenant id is the nil uuid, which is a sentinel and not a "
            "customer. It was the pre-0.67 default fallback, so legacy rows still "
            "hold it; delegating it bills nobody and dead-letters the cost event."
        )
    return parsed
