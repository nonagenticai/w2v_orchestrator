"""P2 tenancy (WP4a) — members, invites, the caller's tenants, tenant rename.

Membership is w2v's own ``tenant_members`` table (``role`` ∈ member|tenant_admin,
``status`` ∈ active|removed), not a Keycloak realm role. API keys + partner
tenant creation live in ``tenancy_keys.py`` (WP4b).
"""

from __future__ import annotations

import datetime
import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from ._shared import _camel

MemberRole = Literal["member", "tenant_admin"]
MemberStatus = Literal["active", "removed"]
InviteStatus = Literal["pending", "accepted", "revoked", "expired"]

# Deliberately loose (one @, a dot in the domain, no whitespace): Keycloak and
# the invitee's mailbox are the real validators; this only rejects garbage
# before a Keycloak user is created for it.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ─── Members ─────────────────────────────────────────────────────────────


class TenantMemberOut(BaseModel):
    model_config = _camel
    subject_id: UUID
    email: str | None
    display_name: str | None
    role: MemberRole
    status: MemberStatus
    first_seen_at: datetime.datetime
    last_seen_at: datetime.datetime


class TenantMemberList(BaseModel):
    model_config = _camel
    items: list[TenantMemberOut]
    total: int
    limit: int
    offset: int


class TenantMemberRoleUpdate(BaseModel):
    model_config = _camel
    role: MemberRole


class TenantMemberRemoved(BaseModel):
    model_config = _camel
    subject_id: UUID
    status: Literal["removed"] = "removed"


# ─── Invites ─────────────────────────────────────────────────────────────


class InviteCreate(BaseModel):
    model_config = _camel
    email: str = Field(min_length=3, max_length=320)
    role: MemberRole = "member"

    @field_validator("email")
    @classmethod
    def _email_shape(cls, v: str) -> str:
        v = v.strip()
        if not _EMAIL_RE.match(v):
            raise ValueError("not an email address")
        return v


EmailDelivery = Literal["requested", "failed"]


class InviteOut(BaseModel):
    model_config = _camel
    id: UUID
    email: str
    role: MemberRole
    status: InviteStatus
    expires_at: datetime.datetime
    created_at: datetime.datetime
    resent_at: datetime.datetime | None
    invited_by: UUID | None


class InviteCreated(InviteOut):
    """The invite + its one-time link. The token is shown ONLY here (create and
    resend); the database keeps its SHA-256."""

    invite_url: str
    #: ``"requested"`` — provisioning accepted the identity invite (it decides
    #: whether an email is needed and never says; w2v cannot know if one was
    #: sent). ``"failed"`` — provisioning could not be reached or refused; the
    #: invite stands, so share ``invite_url`` or resend. Never ``"sent"``:
    #: whether the address has an account must not be observable by a
    #: tenant_admin in a realm shared with fr/kag.
    email_delivery: EmailDelivery


class InviteList(BaseModel):
    model_config = _camel
    items: list[InviteOut]


class InviteAccept(BaseModel):
    model_config = _camel
    token: str = Field(min_length=16, max_length=256)


class InviteAccepted(BaseModel):
    model_config = _camel
    tenant_id: UUID
    tenant_name: str | None
    #: The caller's role in the tenant after acceptance. An existing active
    #: membership keeps its role — accepting an invite never demotes or promotes.
    role: MemberRole
    #: False when this call created/reactivated the membership; True when the
    #: invite was already accepted by the caller (idempotent replay) or the caller
    #: was already an active member.
    already_member: bool


class PendingInvite(BaseModel):
    """A pending, unexpired invite addressed to the caller's VERIFIED email
    (``AuthMe.pending_invites``). Listing it accepts nothing — the caller joins
    explicitly via ``POST /api/invites/{id}/accept``. Deliberately minimal: no
    inviter, no email, no token."""

    model_config = _camel
    id: UUID
    tenant_id: UUID
    tenant_name: str | None
    role: MemberRole
    expires_at: datetime.datetime


# ─── The caller's tenants ────────────────────────────────────────────────


class MyTenant(BaseModel):
    model_config = _camel
    tenant_id: UUID
    name: str | None
    role: MemberRole
    #: ``tenants.status`` (None when the tenant has no registry row yet).
    tenant_status: str | None
    last_seen_at: datetime.datetime | None


class MyTenantList(BaseModel):
    model_config = _camel
    items: list[MyTenant]


# ─── Tenant ──────────────────────────────────────────────────────────────


class TenantRename(BaseModel):
    model_config = _camel
    name: str = Field(min_length=1, max_length=200)


class TenantOut(BaseModel):
    model_config = _camel
    tenant_id: UUID
    name: str | None
    status: str
