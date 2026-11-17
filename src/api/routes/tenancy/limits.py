"""Tenant limits, read-only (P6 ORCH-3).

- GET /api/tenant/limits → what bounds this tenant, each limit labelled with whether
  anything enforces it today. ``TenantAdminDep`` + ``LICENSE_GATE_ONLY`` (the
  tenant-admin tier, same as ``/api/admin/usage``). It reads no tenant table, so
  there is no RLS bind.

What is and is not enforced (verified in code, not assumed):

* licence ``quota_monthly`` / ``token_budget_monthly`` — NOT enforced by w2v. The
  orchestrator never reads them (no ``start_run`` check); the licences service
  provisions the quota into billing, and nothing in this service consults it.
  Which limits get enforced, and where, is an open owner decision (P6 blueprint).
* licences ``byo_mode`` (app ``w2v``) — enforced for every model-provider slot
  (``providers/access.py``): ``byo_required`` ⇒ run-start preflight + resolver refuse
  the platform fallback (``SlotNotConfiguredError``), and modality_gateway answers
  402 for the same tenant on every platform-GPU call; ``platform_only`` ⇒ the
  tenant's own connections are ignored and unwritable.
* allowed tenant kinds (``PROVIDER_ALLOWED_KINDS`` ∩ the registry) — enforced on
  connection save and again when a pinned connection is resolved (``check_kind``).
* provider-test rate limit — enforced per tenant per window by
  ``ProviderTestRateLimiter`` when the limit is > 0; it fails OPEN while Redis is
  unavailable (abuse damping, not an authorization boundary).

No write route exists (Decision 12: limits are super_admin / operator owned).
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter

from src.api.routes._common import (
    LICENSE_GATE_ONLY,
    W2V_APP_KEY,
    LicensesClientDep,
    TenantAdminDep,
)
from src.api.schemas.tenant_limits import LicenceLookup, TenantLimit, TenantLimitsView
from src.clients.licenses_client import (
    LicensesClient,
    LicensesClientError,
    TenantNotOnboardedError,
)
from src.clients.schemas.licenses import PlanView
from src.services.providers.access import required_slots
from src.services.providers.byo_posture import LicencesByoPosture
from src.services.providers.errors import ByoPostureUnavailableError
from src.services.providers.registry import SLOTS, writable_slots
from src.services.providers.settings import ProviderPolicySettings, get_provider_policy

router = APIRouter(prefix="/api/tenant/limits", tags=["tenancy"])

_log = logging.getLogger(__name__)

_LICENCE_NOT_ENFORCED = (
    "Not enforced: w2v never checks it before starting a run. The licences service "
    "provisions it into billing; whether any metered upstream refuses on it is not "
    "visible to this service."
)


async def _licence_plan(
    licenses: LicensesClient, tenant_id: str
) -> tuple[LicenceLookup, PlanView | None, str | None]:
    """The tenant's active w2v licence plan: ``(lookup, plan, plan_code)``."""
    try:
        active = await licenses.get_active_license(tenant_id=tenant_id, app_key=W2V_APP_KEY)
        if active is None:
            return "no_active_licence", None, None
        if active.conflict:
            return "multiple_active_licences", None, None
        plan_ref, app_ref = active.plan, active.app
        if plan_ref is None or plan_ref.id is None or app_ref is None or app_ref.id is None:
            return "unavailable", None, plan_ref.code if plan_ref else None
        plans = await licenses.list_plans(app_ref.id, tenant_id=tenant_id)
    except TenantNotOnboardedError:
        # licences 403 ``tenant_not_accessible``: an answer about the tenant (not
        # onboarded to w2v), not a lookup failure.
        return "not_onboarded", None, None
    except (LicensesClientError, httpx.HTTPError) as exc:
        _log.warning("tenant limits: licence plan lookup failed for %s: %s", tenant_id, exc)
        return "unavailable", None, None
    plan = next((p for p in plans if p.id == plan_ref.id), None)
    return ("ok" if plan else "unavailable"), plan, plan_ref.code


def _licence_limits(plan: PlanView | None) -> list[TenantLimit]:
    return [
        TenantLimit(
            key="licence.quota_monthly",
            source="licence",
            value=plan.quota_monthly if plan else None,
            unit="requests per month",
            enforced=False,
            note=_LICENCE_NOT_ENFORCED,
        ),
        TenantLimit(
            key="licence.token_budget_monthly",
            source="licence",
            value=plan.token_budget_monthly if plan else None,
            unit="tokens per month",
            enforced=False,
            note=_LICENCE_NOT_ENFORCED,
        ),
    ]


def _policy_limits(
    policy: ProviderPolicySettings, byo_required: frozenset[str] | None
) -> list[TenantLimit]:
    allowed = sorted(
        {
            kind
            for slot in writable_slots()
            for kind in SLOTS[slot].tenant_kinds
            if policy.kind_allowed(slot, kind)
        }
    )
    rate_on = policy.test_rate_limit > 0
    return [
        TenantLimit(
            key="provider.byo_required",
            source="licence",
            value=None if byo_required is None else sorted(byo_required),
            unit="slots",
            enforced=True,
            enforced_at="run start, slot resolution and modality_gateway (402)",
            note=(
                "The organisation's provider policy could not be read from licences right "
                "now; model-provider slots fail closed until it can."
                if byo_required is None
                else "Slots this tenant must configure itself; a stage that needs one fails "
                "with 'Configure your own provider to continue'."
                if byo_required
                else "No slot requires a tenant provider; the platform default is used."
            ),
        ),
        TenantLimit(
            key="provider.allowed_kinds",
            source="provider_policy",
            value=allowed,
            unit="provider kinds",
            enforced=True,
            enforced_at="connection save and slot resolution",
            note=(
                "Provider kinds a tenant connection may use (operator-restricted)."
                if policy.allowed_kinds
                else "Every provider kind the registry supports is allowed."
            ),
        ),
        TenantLimit(
            key="provider.test_rate_limit",
            source="provider_policy",
            value=policy.test_rate_limit,
            unit=f"live provider tests per {policy.test_rate_window_seconds}s",
            enforced=rate_on,
            enforced_at="provider connection test" if rate_on else None,
            note=(
                "Per tenant, shared by the test route and the test inside a save. "
                "Not applied while the rate-limit store (Redis) is unavailable."
                if rate_on
                else "Disabled by the operator (limit 0)."
            ),
        ),
    ]


@router.get(
    "",
    response_model=TenantLimitsView,
    operation_id="tenant.limits.get",
    dependencies=LICENSE_GATE_ONLY,
)
async def get_tenant_limits(
    tenant_id: TenantAdminDep, licenses: LicensesClientDep
) -> TenantLimitsView:
    lookup, plan, plan_code = await _licence_plan(licenses, str(tenant_id))
    # Read-only view: an unreadable posture is SHOWN as unknown (value None), not a
    # 503 — the view still answers, exactly like the licence-plan lookup above.
    byo: frozenset[str] | None
    try:
        mode = await LicencesByoPosture(licenses).byo_mode(tenant_id, slot="tts")
        byo = required_slots(mode)
    except ByoPostureUnavailableError:
        byo = None
    return TenantLimitsView(
        tenant_id=tenant_id,
        licence_lookup=lookup,
        plan_code=plan_code,
        limits=[*_licence_limits(plan), *_policy_limits(get_provider_policy(), byo)],
    )
