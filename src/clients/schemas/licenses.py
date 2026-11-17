"""Typed responses for the licenses service.

Source of truth (read, not guessed) — `/nonagenticai/licenses/main.py`. All
snake_case, no aliases. The licenses service is the license-token/entitlement
authority, so these models gate real provisioning + the 402 license gate.

⚠️ Two DISTINCT license shapes the consumers used to conflate (each mismatch was
a live bug typing now prevents):
- ``GET /licenses/active`` returns **TokenIntrospectResponse** — its license id
  field is ``license_id`` (NOT ``id``), and it has **no** ``revoked_at``. The
  client also synthesises ``{"conflict": True}`` on a 409 multi-active.
- ``GET /licenses/user/{id}`` + ``POST`` return **LicenseResponse** — which DOES
  have ``id`` and ``revoked_at`` (and no bare ``revoked`` bool).
"""

from __future__ import annotations

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Apps & plans
# ---------------------------------------------------------------------------


class AppView(BaseModel):
    """``GET /apps`` item — mirrors ``AppResponse``."""

    id: str
    key: str
    name: str
    description: str | None = None
    revoked: bool = False
    created_at: str | None = None


class PlanView(BaseModel):
    """``GET /apps/{app_id}/plans`` item — mirrors ``PlanResponse``."""

    id: str
    app_id: str
    code: str
    name: str
    description: str | None = None
    region_id: str | None = None
    duration_days: int | None = None
    rate_limit_rpm: int | None = None
    rate_limit_burst: int | None = None
    quota_monthly: int | None = None
    token_budget_monthly: int | None = None
    revoked: bool = False


# ---------------------------------------------------------------------------
# Licenses (LicenseResponse shape — has id + revoked_at)
# ---------------------------------------------------------------------------


class AppInfoView(BaseModel):
    """Nested app on a LicenseView — mirrors ``AppInfo``."""

    id: str
    key: str
    name: str


class PlanInfoView(BaseModel):
    """Nested plan on a LicenseView — mirrors ``PlanInfo``."""

    id: str
    code: str
    name: str
    rate_limit_rpm: int | None = None
    rate_limit_burst: int | None = None
    quota_monthly: int | None = None
    token_budget_monthly: int | None = None


class LicenseView(BaseModel):
    """``GET /licenses/user/{id}`` item + ``POST`` result — mirrors ``LicenseResponse``.

    Has ``id`` and ``revoked_at`` (revocation is a timestamp, NOT a bare
    ``revoked`` bool — reading ``.revoked`` was a live idempotency bug).
    """

    id: str
    subject_id: str | None = None
    tenant_id: str | None = None
    app_id: str | None = None
    plan_id: str | None = None
    region_id: str | None = None
    status: str | None = None
    starts_at: str | None = None
    expires_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    revoked_at: str | None = None
    metadata: dict[str, object] | None = None
    app: AppInfoView | None = None
    plan: PlanInfoView | None = None
    #: licences migration 059: a DEPENDENCY licence (provisioned only because a
    #: bundled app requires this app; shared, ``bundle_id`` NULL). ``None`` when
    #: the listing does not carry the field (``GET /licenses`` did not, until
    #: licences exposed it) — then ``metadata.dependency_of`` is the marker.
    is_dependency: bool | None = None
    bundle_id: str | None = None


# ---------------------------------------------------------------------------
# Active-license introspection (TokenIntrospectResponse shape — has license_id)
# ---------------------------------------------------------------------------


class ActiveAppRef(BaseModel):
    """The ``app`` blob on the introspect shape (a dict downstream)."""

    id: str | None = None
    key: str | None = None
    name: str | None = None


class ActivePlanRef(BaseModel):
    """The ``plan`` blob on the introspect shape (a dict downstream)."""

    id: str | None = None
    code: str | None = None
    name: str | None = None


class ActiveLicenseView(BaseModel):
    """``GET /licenses/active`` — mirrors ``TokenIntrospectResponse`` PLUS the
    client's synthetic ``{"conflict": True}`` 409 sentinel.

    The license id field is ``license_id`` (there is no ``id`` here — reading
    ``id`` off this shape always yielded ``None``, the bug this PR fixes). There
    is no ``revoked_at`` on the introspect shape (an active license is not
    revoked by definition).
    """

    active: bool = False
    #: Synthetic sentinel the client sets on a 409 multi-active (not a wire field).
    conflict: bool = False
    tenant_id: str | None = None
    license_id: str | None = None
    subject_id: str | None = None
    byo_mode: str | None = None
    app: ActiveAppRef | None = None
    plan: ActivePlanRef | None = None
    status: str | None = None
    starts_at: str | None = None
    expires_at: str | None = None
    region_id: str | None = None
    metadata: dict[str, object] | None = None
    reason: str | None = None
    #: ``GET /licenses/active`` only: how many active licences the (tenant, app)
    #: pair matched. The returned ``license_id`` is a deterministic PICK when >1, so
    #: a caller may treat it as "the tenant's licence" only when this is exactly 1.
    candidate_count: int | None = None
    ambiguous: bool | None = None
    #: ``/license/introspect`` only: the presented token's own expiry.
    token_expires_at: str | None = None


class TenantByoModeView(BaseModel):
    """``GET /tenants/{id}/byo-mode?app=<key>`` — the resolved BYO posture.

    With ``app`` set, licences returns the per-product ``tenant_app_byo_mode``
    override when one exists, else the tenant-wide value. ``/licenses/active``'s
    ``byo_mode`` (above) is ALWAYS tenant-wide — licences deliberately skips the
    per-app override on the introspect path — so it cannot answer "what applies
    to w2v" and is not used for that.
    """

    tenant_id: str
    byo_mode: str
    app: str | None = None
    #: ``row`` (a real read) or ``fault`` (licences' fail-closed fabrication —
    #: only returned when the caller asked for ``on_unresolved=fail_closed``).
    resolution: str = "row"


class TenantAppByoOverrideView(BaseModel):
    """``GET /tenants/{id}/apps/{app_key}/byo-mode`` — the per-app OVERRIDE row
    (licences ``TenantOfferingRecord``). NOT the effective posture: a 404 there
    means "no override, the tenant-wide value applies"."""

    app_key: str
    byo_mode: str
    offering_code: str | None = None


class TenantAppByoModeSetView(BaseModel):
    """``PUT /tenants/{id}/apps/{app_key}/byo-mode`` — licences
    ``TenantAppByoModeResponse``. ``prior_byo_mode`` is the override this write
    overwrote (None when it created the row)."""

    tenant_id: str
    app_key: str
    byo_mode: str
    prior_byo_mode: str | None = None


# ---------------------------------------------------------------------------
# Tenant bundle ensure (POST /internal/tenants/{id}/bundles/{bundle}/ensure)
# ---------------------------------------------------------------------------


class BundleEnsureAppView(BaseModel):
    """One ensured app — mirrors licences ``BundleEnsureApp``. The licence is
    TENANT-level: held by ``svc:<tenant-uuid>:<app>`` (realm ``service``)."""

    decision: str
    license_id: str | None = None
    plan: str | None = None
    metering: str | None = None


class BundleEnsureRefusalView(BaseModel):
    """One refused app — mirrors licences ``BundleEnsureRefusal`` (nothing written
    for it): ``several_candidates`` | ``member_licence`` | ``no_plan`` |
    ``other_bundle`` | ``app_key_too_long`` | ``lost_race``."""

    app: str
    reason: str


class BundleEnsureView(BaseModel):
    """``BundleEnsureResponse`` — 200, or 409 when at least one app was refused."""

    tenant_id: str
    bundle: str
    budget_currency: str | None = None
    apps: dict[str, BundleEnsureAppView] = {}
    refused: list[BundleEnsureRefusalView] = []
