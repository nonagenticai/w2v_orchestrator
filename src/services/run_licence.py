"""Which licence a run's cost events are attributed to.

Every metered S2S hop of a run carries ``X-License-Id`` from the run snapshot's
``license_id`` (``run_dispatcher`` re-binds it on the background task), and the
downstream emitters (``w2v_common.metering.cost_bus``, worker-management /
inference-router) write it as the cost event's ``license_id`` on
``billing:costs`` (billing-service ``docs/EMITTER_GUIDE.md``). So the snapshot
must hold the run TENANT's own w2v licence.

Before this, the snapshot copied the request's ContextVar, which is EMPTY on the
human (membership) lane: every run a person started went out licence-less, so
downstream either attributed it to the licence the hop itself authenticated with
(the shared ``DOWNSTREAM_LICENSE_TOKEN``'s — one umbrella licence for every
tenant) or left billing to derive one by tenant.

Resolution, in order:

1. ``tenants.licence_ref`` — the tenant's entitling w2v licence (held by
   ``svc:<tenant>:w2v``, kept current by ``licence_refresh``);
2. the request's own licence, when the identity bound one (licence-token or
   API-key lane: introspected/recorded for THIS tenant — w2v_common never lets a
   licence survive a tenant re-scope);
3. nothing → ``""``, logged at ERROR and counted
   (``w2v_orchestrator_run_licence_unattributed_total``). An honest absence that
   billing counts as unattributed — never a fake or an umbrella licence.

The nil UUID is never a licence (billing's ``NULL_LICENSE_ID`` sentinel).

Request time uses the SAME resolution (:func:`resolve_request_licence`): the
membership (human JWT) lane binds no licence token, so ``TenancyMiddleware``
binds the resolved tenant's ``licence_ref`` as the request's ``license_id`` —
the request-time S2S hops (``X-License-Id``) are billed to the tenant's own
licence, and an absence is logged at ERROR + counted
(``w2v_orchestrator_request_licence_unattributed_total``), never papered over.
"""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.database.repositories.tenant_licence_repository import TenantLicenceRepository
from src.observability.metrics import (
    request_licence_unattributed_total,
    run_licence_unattributed_total,
)

_log = logging.getLogger("w2v_orchestrator.run_licence")
_request_log = logging.getLogger("w2v_orchestrator.request_licence")

_NIL = UUID(int=0)


def _licence(raw: object) -> UUID | None:
    """``raw`` as a real licence id, or None (blank, malformed, or the nil UUID)."""
    if raw is None or raw == "":
        return None
    try:
        value = raw if isinstance(raw, UUID) else UUID(str(raw))
    except ValueError:
        return None
    return None if value == _NIL else value


async def _tenant_own_licence(session: AsyncSession, tenant_id: UUID) -> tuple[UUID | None, str]:
    """``tenants.licence_ref`` as a real licence, else None + the reason."""
    stored = await TenantLicenceRepository(session).stored(tenant_id)
    if stored is None:
        return None, "no_tenant_row"
    own = _licence(stored.licence_ref)
    return own, "" if own is not None else "no_licence_ref"


async def resolve_run_licence(
    session: AsyncSession, tenant_id: UUID, context_licence: str | None
) -> str:
    """The licence id (string form) a run of ``tenant_id`` is attributed to, or
    ``""`` (logged + counted) when the tenant has none. See the module docstring."""
    own, reason = await _tenant_own_licence(session, tenant_id)
    if own is not None:
        return str(own)
    bound = _licence(context_licence)
    if bound is not None:
        return str(bound)
    run_licence_unattributed_total.labels(reason=reason).inc()
    _log.error(
        "run for tenant %s has NO licence to attribute its cost events to (%s): its "
        "S2S hops carry an empty X-License-Id. Record the tenant's w2v licence "
        "(licence_ref) — never fall back to a shared licence.",
        tenant_id,
        reason,
    )
    return ""


def request_licence_unattributed(tenant_id: UUID, reason: str) -> str:
    """Log (ERROR) + count a request with no attributable licence; returns ``""``."""
    request_licence_unattributed_total.labels(reason=reason).inc()
    _request_log.error(
        "request for tenant %s has NO licence to attribute its cost events to (%s): "
        "its S2S hops carry an empty X-License-Id. Record the tenant's w2v licence "
        "(licence_ref) — never fall back to a shared licence.",
        tenant_id,
        reason,
    )
    return ""


async def resolve_request_licence(session: AsyncSession, tenant_id: UUID) -> str:
    """The licence id a membership-resolved request of ``tenant_id`` is attributed
    to: the tenant's own ``licence_ref`` (as :func:`resolve_run_licence` step 1 —
    the human lane carries no licence of its own), or ``""`` logged + counted."""
    own, reason = await _tenant_own_licence(session, tenant_id)
    if own is not None:
        return str(own)
    return request_licence_unattributed(tenant_id, reason)
