"""P2 tenancy WP4b schemas: tenant API keys + partner tenant creation."""

from __future__ import annotations

import datetime
from typing import Annotated, Final, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from ._shared import _camel

#: A tenant's w2v licence token on the WIRE (request bodies only). ``SecretStr``
#: renders ``**********`` in every repr / ``model_dump``; it is never part of any
#: response model, log line, exception message or audit payload.
LicenceToken = Annotated[SecretStr, Field(min_length=1, max_length=4096)]

# ─── Tenant API keys (/api/tenant/api-keys) ──────────────────────────────

#: Grants a key may carry beyond the ``member`` role (P8). Mirrors the DB CHECK
#: ``ck_tenant_api_keys_scopes`` (migration 0086).
ApiKeyScope = Literal["connections:write"]
SCOPE_CONNECTIONS_WRITE: Final = "connections:write"


class ApiKeyCreate(BaseModel):
    model_config = _camel
    name: str = Field(min_length=1, max_length=200)
    expires_at: datetime.datetime | None = None
    #: ``connections:write`` lets the key bulk-write the tenant's provider
    #: connections (``PUT /api/tenant/connections``). Default: none.
    scopes: list[ApiKeyScope] = Field(default_factory=list, max_length=8)

    @field_validator("scopes")
    @classmethod
    def _dedupe(cls, value: list[ApiKeyScope]) -> list[ApiKeyScope]:
        return sorted(set(value))


class ApiKeySummary(BaseModel):
    """A key as listed — NEVER the secret or its hash; ``prefix`` is public."""

    model_config = _camel
    id: UUID
    name: str
    prefix: str
    created_by: UUID | None = None
    created_at: datetime.datetime
    last_used_at: datetime.datetime | None = None
    expires_at: datetime.datetime | None = None
    scopes: list[str] = Field(default_factory=list)
    #: ``partner``: minted by the tenant's partner through the partner lane (P9) —
    #: a machine credential the partner manages; ``tenant_admin``: minted in the UI.
    created_via: Literal["tenant_admin", "partner"] = "tenant_admin"


class ApiKeyList(BaseModel):
    model_config = _camel
    items: list[ApiKeySummary]
    total: int


class ApiKeyCreated(ApiKeySummary):
    """Create response: the only time ``key`` (``w2v_sk_…``) is ever returned."""

    key: str


# ─── Partner tenant creation (POST /api/tenants) ─────────────────────────


class PartnerTenantCreate(BaseModel):
    """``tenant_id`` makes the call idempotent (omit → a new id is minted).

    ``admin_email`` is OPTIONAL (P9-4): absent → no first-admin invite; the
    tenant is managed by the partner's machine keys (a super_admin can still
    invite a human later)."""

    model_config = _camel
    tenant_id: UUID | None = None
    name: str = Field(min_length=1, max_length=200)
    admin_email: str | None = Field(
        default=None, min_length=3, max_length=320, pattern=r"^[^@\s]+@[^@\s]+$"
    )
    #: The tenant's OWN w2v licence token (optional). Introspected: it must be
    #: active, of THIS ``tenant_id`` and of app ``w2v`` — else 422
    #: ``LICENCE_TOKEN_MISMATCH``. Stored sealed; its licence becomes ``licence_ref``.
    #: A replay with a DIFFERENT valid token replaces the stored one (lease + CAS);
    #: the same token again is a no-op.
    licence_token: LicenceToken | None = None


class PartnerTenantResult(BaseModel):
    model_config = _camel
    tenant_id: UUID
    name: str | None
    created: bool
    invite_id: UUID | None
    already_member: bool
    #: The tenant's billing licence (from the token, or a single active licence).
    licence_ref: UUID | None = None


# ─── Partner-minted tenant API keys (P9-3) ───────────────────────────────


class PartnerApiKeyCreate(BaseModel):
    """``POST /api/tenants/{tenant_id}/api-keys`` — a partner mints a machine key
    for a tenant bound to its partner licence. ``name`` is the idempotency key
    (a second mint of a LIVE name is 409 ``API_KEY_NAME_EXISTS``)."""

    model_config = _camel
    name: str = Field(min_length=1, max_length=200)
    scopes: list[ApiKeyScope] = Field(default_factory=list, max_length=8)

    @field_validator("scopes")
    @classmethod
    def _dedupe(cls, value: list[ApiKeyScope]) -> list[ApiKeyScope]:
        return sorted(set(value))


# ─── Super-admin partner binding (P9-5) ──────────────────────────────────


class TenantPartnerBind(BaseModel):
    """``PUT /api/admin/tenants/{tenant_id}/partner`` — bind an EXISTING tenant to
    an allow-listed partner licence so that partner can mint its keys. Unknown
    fields are refused: the retired ``licence_id`` must not be silently ignored."""

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True, extra="forbid")
    partner_licence_id: UUID
    #: The tenant's OWN w2v licence TOKEN (optional). Introspected exactly as on
    #: ``POST /api/tenants`` (active ∧ this tenant ∧ app w2v, else 422
    #: ``LICENCE_TOKEN_MISMATCH``); stored sealed, its licence becomes
    #: ``licence_ref`` (bills the API-key lane).
    licence_token: LicenceToken | None = None


class TenantPartnerBinding(BaseModel):
    model_config = _camel
    tenant_id: UUID
    partner_licence_id: UUID
    #: False on an idempotent re-bind to the same partner (nothing changed).
    bound: bool
    #: The tenant's entitling w2v licence (bills its API-key lane); None until
    #: the licences service reports one (the refresh sweep sets it later).
    licence_ref: UUID | None = None


class TenantLicenceTokenSet(BaseModel):
    """``PUT /api/tenants/{tenant_id}/licence-token`` — rotate / hand over the
    tenant's stored licence token (the owning partner, or super_admin)."""

    model_config = _camel
    licence_token: LicenceToken


class TenantLicenceTokenResult(BaseModel):
    """Never carries the token — only what it resolved to."""

    model_config = _camel
    tenant_id: UUID
    licence_ref: UUID | None
    #: False when the SAME token was already stored (nothing written).
    changed: bool
    licence_token_set_at: datetime.datetime | None = None
    licence_token_expires_at: datetime.datetime | None = None
