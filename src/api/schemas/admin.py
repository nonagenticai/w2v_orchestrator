"""Auth + meta + Admin (Phase B/C) + usage schemas."""

from __future__ import annotations

import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from ._shared import _camel
from .tenancy import MyTenant, PendingInvite

# ─── Auth + meta ────────────────────────────────────────────────────────


class AuthMe(BaseModel):
    model_config = _camel
    # None only for a membership-resolved caller (W2V_TENANCY_SOURCE=membership,
    # or a claimless token under shadow) with no membership (the SPA's
    # no-workspace screen).
    tenant_id: UUID | None
    subject_id: UUID
    email: str | None
    name: str | None
    roles: list[str]
    # P2 WP4a: the caller's membership role in `tenant_id` (None: no membership,
    # a super_admin acting into a foreign tenant, or a non-JWT caller), every
    # active membership (tenant switcher), and the pending invites addressed to
    # the caller's VERIFIED email. /me never accepts one: the SPA offers them and
    # the caller joins explicitly via POST /api/invites/{invite_id}/accept.
    role: str | None = None
    memberships: list[MyTenant] = Field(default_factory=list)
    pending_invites: list[PendingInvite] = Field(default_factory=list)
    # The requested active tenant (``X-Active-Tenant-Id``) the backend REFUSED
    # for this request — unknown, suspended or deleted — so the SPA drops the
    # stored choice instead of rendering a shell the backend will not serve.
    # None when nothing was requested or the request was honoured.
    active_tenant_refused: UUID | None = None


class LicenseStatus(BaseModel):
    """Rich, app-scoped license status for the SPA (GET /api/license/status).

    ``state ∈ active|expiring_soon|expired|revoked|none|unknown``. Fail-soft:
    ``unknown`` means an upstream lookup blip, not a block. Mirrors fr's
    ``/api/license/status`` contract so the frontend gate ports directly.
    """

    model_config = _camel
    app_key: str
    gate_enabled: bool
    has_license: bool
    state: str
    expires_at: str | None
    days_remaining: int | None
    plan_code: str | None


# ─── Admin tier (Phase B) ───────────────────────────────────────────────


class AdminTenantSummary(BaseModel):
    model_config = _camel
    tenant_id: UUID
    name: str | None = None
    status: str = "active"
    project_count: int
    last_activity: datetime.datetime | None = None
    # Per-tenant usage rollup (list view). Default 0 so the PATCH response path
    # (get_tenant_summary, which does not compute usage) still validates; the
    # list refetches after a write so its 0s are never displayed.
    run_count: int = 0
    total_cost_usd: float = 0.0
    #: P9: the partner licence that manages this tenant (``PUT …/partner``,
    #: or partner creation/adoption); None = not partner-managed.
    partner_licence_id: UUID | None = None
    #: Can a super_admin scope-switch INTO this tenant? Computed by calling
    #: ``AdminRepository.tenant_switchable`` — the function ``TenancyMiddleware``
    #: enforces the switch with (operator-held ``suspended``/``deleted`` and
    #: unknown tenants are refused). The tenant picker offers only these, so the
    #: UI never offers what the backend refuses. Required: never defaulted.
    switchable: bool


class AdminTenantList(BaseModel):
    model_config = _camel
    items: list[AdminTenantSummary]
    total: int


class AdminTenantUpdate(BaseModel):
    """Super-admin rename / status change for a tenant registry row. Both fields
    optional but the route rejects an all-None body (nothing to change)."""

    model_config = _camel
    name: str | None = None
    status: Literal["active", "suspended"] | None = None


class AdminRunListItem(BaseModel):
    """One run in the cross-tenant operator run pane."""

    model_config = _camel
    run_id: UUID
    tenant_id: UUID
    project_id: UUID
    project_title: str
    deliverable_id: UUID
    blueprint_id: str
    status: str
    current_stage: str | None = None
    started_at: datetime.datetime | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    # When a PAUSED run began waiting for its reviewer (``runs.paused_at``); None
    # for every run that is not paused. The pane ages a paused row from this, not
    # from ``started_at`` — otherwise a run parked at a review gate for a day reads
    # as a run that has been executing for a day.
    paused_at: datetime.datetime | None = None
    # ``checkpoint`` (waiting for a reviewer) | ``manual`` (someone pressed Pause) |
    # ``unknown`` (paused before 0102); None unless paused. Same as RunSummary.
    pause_reason: Literal["checkpoint", "manual", "unknown"] | None = None
    # Only meaningful for active (pending/running) rows: True if some pod holds
    # the run's execution lease (live), False if it's an orphan no pod is running.
    lease_present: bool | None = None


class AdminRunList(BaseModel):
    model_config = _camel
    items: list[AdminRunListItem]
    total: int


class AdminOverview(BaseModel):
    model_config = _camel
    tenant_id: UUID
    project_count: int
    run_count: int
    total_cost_usd: float
    # "unavailable" when billing could not be read for not-yet-mirrored runs
    # (cost_unavailable_reason says why): total_cost_usd then covers only the
    # cost already mirrored locally and is a LOWER BOUND, not the tenant's cost.
    cost_status: Literal["ok", "unavailable"] = "ok"
    cost_unavailable_reason: str | None = None


class AdminAuditEntry(BaseModel):
    model_config = _camel
    id: UUID
    created_at: datetime.datetime
    actor_subject_id: UUID | None = None
    action: str
    target_tenant_id: UUID | None = None
    detail: dict[str, object] = Field(default_factory=dict)


class AdminAuditList(BaseModel):
    model_config = _camel
    items: list[AdminAuditEntry]
    total: int
    limit: int
    offset: int


class ScopeSwitchAuditRequest(BaseModel):
    """Record a deliberate super-admin scope switch. ``to_tenant_id`` None = the
    super-admin reset back to their own tenant (action ``scope.reset``)."""

    model_config = _camel
    to_tenant_id: UUID | None = None


class RequestAuditEntry(BaseModel):
    """One ``request_audit`` row (the middleware-wide per-request audit)."""

    model_config = _camel
    id: UUID
    created_at: datetime.datetime
    actor_subject_id: UUID | None = None
    claim_tenant_id: UUID | None = None
    effective_tenant_id: UUID | None = None
    cross_tenant: bool
    event: str
    method: str
    path: str
    resource: str | None = None
    status_code: int
    ip: str | None = None
    duration_ms: float | None = None
    credential_fingerprint: str | None = None
    request_id: str | None = None
    detail: dict[str, object] = Field(default_factory=dict)


class RequestAuditList(BaseModel):
    model_config = _camel
    items: list[RequestAuditEntry]
    total: int
    limit: int
    offset: int


# ─── Admin provisioning (Phase C, C-2; P2 tenancy WP4b) ──────────────────


LlmAccessMode = Literal["local_allowed", "platform_only", "byo_required"]


class AdminProvisionRequest(BaseModel):
    """Super-admin: provision a new tenant + invite its first user.

    ``tenant_slug`` becomes the new tenant's registry name. The tenant id is
    minted server-side; the user is INVITED into it with ``workspace_role`` (P2:
    w2v owns membership — nothing tenant-related is written to Keycloak)."""

    model_config = _camel
    email: str = Field(min_length=3, max_length=320)
    name: str = Field(min_length=1, max_length=200)
    tenant_slug: str = Field(pattern=r"^[a-z0-9-]{2,40}$")
    workspace_role: Literal["member", "tenant_admin"] = "member"
    apps: list[str] | None = None
    #: The tenant's LLM/GPU posture (licences ``byo_mode`` for app ``w2v``), set
    #: AFTER the licences are ensured. Licences' own default for a new tenant is
    #: ``byo_required``, so w2v always sets it explicitly. All three postures are
    #: accepted (owner decision 2026-10-22: the posture is chosen AT provisioning,
    #: CommonPractice-style); ``byo_required`` = the tenant brings every model provider.
    llm_access: LlmAccessMode = "local_allowed"


class AdminTenantLlmAccess(BaseModel):
    """``GET|PUT /api/admin/tenants/{id}/llm-access`` — the tenant's effective w2v
    posture and where it came from (``app_override`` = the per-app ``w2v`` row,
    ``tenant`` = tenant-wide value, ``default`` = not in the licences registry)."""

    model_config = _camel
    mode: LlmAccessMode
    source: Literal["tenant", "app_override", "default"]


class AdminTenantLlmAccessUpdate(BaseModel):
    model_config = _camel
    mode: LlmAccessMode


class AdminProvisionLlmAccess(BaseModel):
    """Outcome of setting the new tenant's LLM posture during provisioning.
    ``status='failed'`` ⇒ the tenant is still on licences' ``byo_required``
    default (every model provider must be its own) until an operator sets it
    (``PUT /api/admin/tenants/{id}/llm-access``); ``detail`` says why."""

    model_config = _camel
    mode: LlmAccessMode
    status: Literal["ok", "failed"]
    detail: str | None = None


class AdminProvisionResult(BaseModel):
    """One-time provisioning result (licence tokens returned ONCE — never
    re-fetchable). ``invite_id`` is the first user's pending invite (None if the
    identity was somehow already an active member). The invitee's Keycloak user
    id is deliberately NOT returned — it is recorded in provisioning_audit."""

    model_config = _camel
    tenant_id: UUID
    invite_id: UUID | None = None
    tokens: dict[str, str] = Field(default_factory=dict)
    license_ids: dict[str, str] = Field(default_factory=dict)
    llm_access: AdminProvisionLlmAccess | None = None


class AdminUsageTotals(BaseModel):
    model_config = _camel
    run_count: int
    project_count: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


class AdminUsageByType(BaseModel):
    model_config = _camel
    type: str
    run_count: int
    cost_usd: float
    input_tokens: int
    output_tokens: int


class AdminUsageByStage(BaseModel):
    model_config = _camel
    stage_name: str
    call_count: int
    cost_usd: float
    input_tokens: int
    output_tokens: int


class AdminUsageByModel(BaseModel):
    model_config = _camel
    model: str | None = None
    provider: str | None = None
    call_count: int
    cost_usd: float
    input_tokens: int
    output_tokens: int


class AdminUsageByUser(BaseModel):
    """Usage attributed to one subject (P6 ORCH-3).

    ``subject_id`` ``None`` is the ``"unattributed"`` group: rows written before
    attribution existed that the backfill could not attribute. ``member_status`` is
    the subject's membership in this tenant (``removed`` members are kept; ``None`` =
    never a member, e.g. a service account)."""

    model_config = _camel
    subject_id: UUID | None = None
    email: str | None = None
    member_status: Literal["active", "removed"] | None = None
    #: Email, else the subject id, else ``"unattributed"``.
    label: str
    call_count: int
    cost_usd: float
    input_tokens: int
    output_tokens: int


class AdminUsageBySlot(BaseModel):
    """Usage per provider slot (P6 ORCH-3).

    ``slot`` is the tenant connection's slot for in-process (``tenant_byo``) calls,
    ``"platform"`` for billing-mirrored calls (the platform route — not who paid), and
    ``"unattributed"`` for rows with neither. ``funding`` is the agreed taxonomy
    (``local_gpu``/``platform_external``/``tenant_byo``) or ``"unattributed"`` when not
    known (billing-mirrored rows: billing cannot tell platform-paid from tenant BYO).
    ``units`` sums the billable quantity per unit kind
    (``tokens``/``chars``/``images``/``seconds``)."""

    model_config = _camel
    slot: str
    funding: Literal["local_gpu", "platform_external", "tenant_byo", "unattributed"]
    call_count: int
    cost_usd: float
    input_tokens: int
    output_tokens: int
    units: dict[str, float] = {}


class AdminUsagePoint(BaseModel):
    """One bucket of the usage time-series.

    ``bucket`` is the bucket-start date (the day itself / the Monday for a week /
    the 1st for a month). The series is gap-filled — zero-cost buckets appear
    between the range ends — and optionally windowed to the last N days.
    """

    model_config = _camel
    bucket: str  # ISO date (YYYY-MM-DD), bucket start
    cost_usd: float
    input_tokens: int
    output_tokens: int


class AdminUsageDetail(BaseModel):
    """The drill-down behind ``AdminOverview`` — cost/token totals plus
    breakdowns by project type, usage stage, and model/provider, and a
    cost/token time-series. All figures reflect the requested ``window_days``
    (all-time when omitted)."""

    model_config = _camel
    tenant_id: UUID
    totals: AdminUsageTotals
    by_type: list[AdminUsageByType]
    by_stage: list[AdminUsageByStage]
    by_model: list[AdminUsageByModel]
    by_user: list[AdminUsageByUser] = []
    by_slot: list[AdminUsageBySlot] = []
    time_series: list[AdminUsagePoint] = []
    # Same contract as AdminOverview.cost_status: "unavailable" ⇒ the figures
    # exclude not-yet-mirrored runs whose cost could not be read.
    cost_status: Literal["ok", "unavailable"] = "ok"
    cost_unavailable_reason: str | None = None
