"""P6 ORCH-3 schemas: the read-only tenant limits view (``GET /api/tenant/limits``).

Every limit says whether something actually refuses work when it is exceeded
(``enforced``) and where (``enforced_at``). Nothing here can be changed by a tenant;
limits are set by the operator (provider policy env) or the licence plan.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from ._shared import _camel

LicenceLookup = Literal[
    "ok", "no_active_licence", "multiple_active_licences", "not_onboarded", "unavailable"
]


class TenantLimit(BaseModel):
    model_config = _camel
    #: Stable id, e.g. ``licence.quota_monthly`` / ``provider.byo_required``.
    key: str
    source: Literal["licence", "provider_policy"]
    #: The limit itself: a number, a list of names, or ``None`` (unset / unknown).
    value: int | list[str] | None = None
    unit: str | None = None
    #: True only when a code path in this deployment refuses work on it today.
    enforced: bool
    #: Where it is enforced (``None`` when ``enforced`` is false).
    enforced_at: str | None = None
    note: str


class TenantLimitsView(BaseModel):
    model_config = _camel
    tenant_id: UUID
    #: Whether the licence plan could be read (licence limits are ``None`` otherwise).
    licence_lookup: LicenceLookup
    plan_code: str | None = None
    #: Always false: there is no write route (limits are operator/super-admin owned).
    editable: Literal[False] = False
    limits: list[TenantLimit]
