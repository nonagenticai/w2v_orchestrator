"""``POST /api/internal/provider-slots/resolve`` — a run's resolved provider slot, for w2v satellites.

A w2v satellite service (html_render, content, video_production, …) that calls a
model provider on a run's behalf asks HERE which provider that run uses, so the
tenant's own connection (or the platform) is honoured outside the orchestrator too.

Auth: the P7 **cell lane ONLY** — a verified ``X-W2V-Cell-Token`` (tenant scope)
whose issuer is in ``CELL_ALLOWED_CALLERS`` (the middleware's ``verify_cell_token``
enforces that list) AND in this endpoint's own explicit allowlist
``W2V_PROVIDER_SLOT_CALLERS`` (no wildcard; else 403 ``caller_not_allowed``), tenant
provenance ``CELL_JWT``. A token that carries a ``run`` claim may only ask for THAT
run (``run_id`` must equal it, and may not be null) — else 403 ``run_mismatch``. A licence-lane / Keycloak /
API-key request — or a dual-mode request where the legacy lane authenticated — is
403 ``cell_lane_required``. The tenant is the SIGNED one from the identity context.

Body ``{"slot": "<registry slot>", "run_id": "<uuid>" | null}``:

* ``run_id`` set — the run is loaded under RLS and must belong to the caller's
  tenant (else 404 ``run_not_found``); its ``config_snapshot.provider_slots`` pin is
  resolved EXACTLY (a later tenant edit does not change a running job). A pinned
  version that is purged / unusable is 409 ``provider_unavailable`` — never a silent
  platform fallback.
* ``run_id`` null — live resolution for the tenant (direct API / n8n path).

Both apply the uniform ``byo_mode`` policy (``providers/access.py``):
``byo_required`` with no tenant connection (or a platform pin) → 403
``byo_provider_required`` naming the slot; ``platform_only`` → the platform default.

Response: the P5 ``provider_slot`` shape (``providers/slot_payload.slot_payload``) plus
``source``. A ``tenant`` answer carries the DECRYPTED secret; a ``deployment`` answer
carries none (``secret``/``base_url``/``ref`` null) — satellites call the platform
with their own service licence. Every answer is ``Cache-Control: no-store``. Neither
the body nor the secret is ever logged.
"""

from __future__ import annotations

import logging
from typing import Annotated, Final
from uuid import UUID

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from w2v_common.auth.identity_context import (
    get_tenant_id_from_context,
    get_tenant_source_from_context,
)
from w2v_common.tenant import TenantSource

from src.api.routes._common import SessionDep, TenantIdDep
from src.api.routes.runs._shared import _tenant_owns_run
from src.config import get_settings
from src.observability.metrics import provider_slot_resolve_total
from src.services.providers.adapters import build_resolver_deps
from src.services.providers.byo_posture import ByoPostureReader, get_byo_posture
from src.services.providers.errors import (
    ProviderKeyringUnavailableError,
    SlotNotConfiguredError,
    SlotUnavailableError,
)
from src.services.providers.registry import PINNED_SLOT_NAMES
from src.services.providers.resolver import ResolvedSlot, resolve_slot
from src.services.providers.slot_payload import slot_payload

__all__ = ["router"]

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/internal/provider-slots", tags=["internal"])

_NO_STORE: Final = {"Cache-Control": "no-store"}
#: ``request.state.cell_auth_lane`` when the CELL lane (not dual) authenticated.
_CELL_LANE: Final = "cell"
_TENANT_SCOPE: Final = "tenant"


def get_resolve_byo_posture() -> ByoPostureReader:
    """The licences ``byo_mode`` reader (a dependency so tests can pin a posture)."""
    return get_byo_posture()


ByoPostureDep = Annotated[ByoPostureReader, Depends(get_resolve_byo_posture)]


class ResolveSlotBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slot: str = Field(min_length=1, max_length=64)
    run_id: UUID | None = None


def _answer(
    status_code: int, body: dict[str, object], *, slot: str, source: str, caller: str, outcome: str
) -> JSONResponse:
    provider_slot_resolve_total.labels(
        slot=slot, source=source, caller=caller, outcome=outcome
    ).inc()
    return JSONResponse(status_code=status_code, content=body, headers=_NO_STORE)


def _cell_caller(request: Request) -> str | None:
    """The verified cell issuer when the CELL lane authenticated this request, else None."""
    state = request.state
    if getattr(state, "cell_auth_lane", None) != _CELL_LANE:
        return None
    if getattr(state, "cell_scope", None) != _TENANT_SCOPE:
        return None
    if get_tenant_source_from_context() is not TenantSource.CELL_JWT:
        return None
    caller = getattr(state, "cell_caller", None)
    return caller if isinstance(caller, str) and caller else None


def _view(resolved: ResolvedSlot) -> dict[str, object]:
    if resolved.source == "tenant":
        return {"source": "tenant", **slot_payload(resolved)}
    return {
        "source": "deployment",
        "slot": resolved.ref.slot,
        "kind": resolved.ref.kind,
        "base_url": None,
        "model": None,
        "options": {},
        "secret": None,
        "ref": None,
    }


@router.post("/resolve", operation_id="internal.provider_slots.resolve")
async def resolve_provider_slot(
    body: ResolveSlotBody,
    request: Request,
    tenant_id: TenantIdDep,
    session: SessionDep,
    posture: ByoPostureDep,
) -> JSONResponse:
    """The run's (or the tenant's live) resolved provider for ``slot`` — see module doc."""
    caller = _cell_caller(request)
    slot = body.slot if body.slot in PINNED_SLOT_NAMES else "unknown"
    if caller is None or str(get_tenant_id_from_context()) != str(tenant_id):
        return _answer(
            status.HTTP_403_FORBIDDEN,
            {
                "detail": {
                    "code": "cell_lane_required",
                    "message": "This endpoint is reachable only on the in-cell service lane.",
                }
            },
            slot=slot,
            source="none",
            caller="none",
            outcome="forbidden",
        )
    if caller not in get_settings().provider_slot_resolve.provider_slot_callers:
        return _answer(
            status.HTTP_403_FORBIDDEN,
            {
                "detail": {
                    "code": "caller_not_allowed",
                    "message": "This service is not allowed to resolve provider slots "
                    "(W2V_PROVIDER_SLOT_CALLERS).",
                }
            },
            slot=slot,
            source="none",
            caller=caller,
            outcome="caller_not_allowed",
        )
    run_claim = getattr(getattr(request.state, "cell_claims", None), "run", None)
    if run_claim and str(body.run_id) != str(run_claim):
        # A token minted FOR a run may only read THAT run's pins — never another run's,
        # and never the tenant's live connections (run_id null).
        return _answer(
            status.HTTP_403_FORBIDDEN,
            {
                "detail": {
                    "code": "run_mismatch",
                    "message": "run_id must equal the run the cell token was minted for.",
                }
            },
            slot=slot,
            source="none",
            caller=caller,
            outcome="run_mismatch",
        )
    if body.slot not in PINNED_SLOT_NAMES:
        return _answer(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            {"detail": {"code": "unknown_slot", "message": "not a resolvable provider slot"}},
            slot=slot,
            source="none",
            caller=caller,
            outcome="unknown_slot",
        )

    snapshot: dict[str, object] | None = None
    if body.run_id is not None:
        run = await _tenant_owns_run(session, body.run_id, tenant_id)
        if run is None:
            return _answer(
                status.HTTP_404_NOT_FOUND,
                {"detail": {"code": "run_not_found", "message": "run not found"}},
                slot=slot,
                source="none",
                caller=caller,
                outcome="run_not_found",
            )
        snapshot = dict(run.config_snapshot or {})

    deps = build_resolver_deps(session, get_settings())
    try:
        resolved = await resolve_slot(
            deps.repo,
            deps.decryptor,
            tenant_id,
            body.slot,
            run_snapshot=snapshot,
            policy=deps.policy,
            posture=posture,
        )
    except SlotNotConfiguredError as exc:
        return _answer(
            status.HTTP_403_FORBIDDEN,
            {"detail": exc.failure_detail},
            slot=slot,
            source="deployment",
            caller=caller,
            outcome="byo_provider_required",
        )
    except ProviderKeyringUnavailableError as exc:
        return _answer(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            {"detail": {"code": "provider_keyring_unavailable", "message": str(exc)}},
            slot=slot,
            source="tenant",
            caller=caller,
            outcome="keyring_unavailable",
        )
    except SlotUnavailableError as exc:
        return _answer(
            status.HTTP_409_CONFLICT,
            {"detail": {"code": "provider_unavailable", "slot": exc.slot, "message": str(exc)}},
            slot=slot,
            source="tenant",
            caller=caller,
            outcome="provider_unavailable",
        )
    log.info(
        "provider slot resolved slot=%s source=%s caller=%s run=%s",
        body.slot,
        resolved.source,
        caller,
        body.run_id,
    )
    return _answer(
        status.HTTP_200_OK,
        _view(resolved),
        slot=slot,
        source=resolved.source,
        caller=caller,
        outcome="ok",
    )
