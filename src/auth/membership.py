"""Membership-based tenant resolution (P2 tenancy, WP3) — pure helpers.

w2v's own ``tenant_members`` table — not the Keycloak ``tenant_id`` claim — is
the tenancy authority once ``W2V_TENANCY_SOURCE=membership``. This module holds
the decision rules; ``src/auth/tenancy_middleware.py`` does the I/O and applies
them per request.

Resolution for a human (JWT ``sub``) caller, given their ACTIVE memberships in
non-suspended tenants (most recently used first):

1. A requested tenant (``X-Active-Tenant-Id`` header, else ``?active_tenant=``)
   that is one of the caller's memberships → that tenant.
2. A requested tenant the caller is NOT a member of:
   * ``super_admin`` → that tenant (the caller must still confirm it exists and
     is not suspended — I/O, done by the middleware), audited;
   * anyone else → refused (``NO_MEMBERSHIP``). Never silently served a
     different tenant than the one the client said it is acting in: a write
     would land in the wrong workspace.
3. No request: exactly one membership → it; several → the most recently used.
4. Nothing → refused (``NO_MEMBERSHIP``).

No I/O here: every input is passed in, so the rules are testable in isolation.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

#: ``W2V_TENANCY_SOURCE`` values (``src/config.py``):
#: * ``claim`` — today: tenant from the JWT ``tenant_id`` claim (default).
#: * ``shadow`` — serve by claim, compute the membership answer, count mismatches.
#: * ``membership`` — tenant from w2v's ``tenant_members`` table; claim ignored.
TenancySource = Literal["claim", "shadow", "membership"]

#: ``request.state.tenant_resolved_by`` value stamped on a membership-resolved
#: request. ``_common.get_tenant_id`` / ``require_known_tenant`` /
#: ``require_tenant_admin`` read it.
RESOLVED_BY_MEMBERSHIP = "membership"

#: ``request.state.tenant_resolved_by`` value stamped when ``TenancyMiddleware``
#: consulted w2v's membership for a JWT caller (``membership`` mode, or a
#: claimless ``shadow`` token) and found NONE usable, on a
#: ``NO_MEMBERSHIP_EXEMPT_PATHS`` route. The request carries no tenant on
#: purpose: ``_common.get_optional_tenant_id`` answers ``None`` for it instead of
#: demanding the claim the caller never had (live 2026-10-15: a claimless shadow
#: portal user got 401 ``missing tenant_id claim`` on ``/api/auth/me``).
RESOLVED_NO_MEMBERSHIP = "membership:none"

#: Error code for an authenticated caller with no usable membership.
NO_MEMBERSHIP = "NO_MEMBERSHIP"

SUPER_ADMIN = "super_admin"
TENANT_ADMIN = "tenant_admin"
MEMBER = "member"

#: Request channels naming the tenant the caller wants to act in. The same two
#: channels the claim-mode super_admin scope-switch reads (``_common.get_tenant_id``).
ACTIVE_TENANT_HEADER = "x-active-tenant-id"
ACTIVE_TENANT_QUERY = "active_tenant"

#: Membership roles (``tenant_members.role`` CHECK constraint, migration 0072).
MembershipRole = Literal["member", "tenant_admin"]

ResolutionVia = Literal["requested", "single", "most_recent", "super_admin"]


@dataclass(frozen=True, slots=True)
class MembershipRow:
    """One active membership of the caller in a non-suspended tenant."""

    tenant_id: UUID
    role: str
    last_seen_at: datetime.datetime | None = None


@dataclass(frozen=True, slots=True)
class Resolution:
    """The tenant a request acts in, and why.

    ``role`` is the caller's membership role in that tenant; ``None`` only for
    a super_admin acting into a tenant they are not a member of (their
    authority there is the realm role, not a membership).
    """

    tenant_id: UUID
    role: str | None
    via: ResolutionVia


@dataclass(frozen=True, slots=True)
class Refusal:
    """No tenant can be resolved. ``reason`` is a stable label (metrics/logs)."""

    reason: Literal["no_membership", "not_a_member"]


class MalformedRequestedTenantError(ValueError):
    """A super_admin named a requested tenant that is not a UUID (→ 422)."""


def parse_requested_tenant(raw: str | None, *, is_super_admin: bool) -> UUID | None:
    """The requested tenant, or ``None`` when absent.

    Malformed: a super_admin gets a strict error (parity with the claim-mode
    scope-switch 422); anyone else's malformed value is treated as absent.
    """
    if not raw:
        return None
    try:
        return UUID(raw)
    except ValueError as exc:
        if is_super_admin:
            raise MalformedRequestedTenantError("malformed active tenant: not a UUID") from exc
        return None


def membership_roles(resolution: Resolution, realm_roles: list[str] | set[str]) -> list[str]:
    """The effective roles for a membership-resolved request.

    The membership role in the resolved tenant, plus ``super_admin`` when the
    realm grants it. A realm ``tenant_admin``/``member`` role is deliberately
    DROPPED: it is a realm-wide (fr/kag) grant and says nothing about this w2v
    tenant.
    """
    roles: list[str] = []
    if resolution.role:
        roles.append(resolution.role)
    if SUPER_ADMIN in realm_roles:
        roles.append(SUPER_ADMIN)
    return roles


def resolve_tenant(
    memberships: list[MembershipRow],
    requested: UUID | None,
    *,
    is_super_admin: bool,
) -> Resolution | Refusal:
    """Apply the resolution rules (module docstring). ``memberships`` must be
    ordered most recently used first (``MembersRepository.list_resolvable_memberships``).

    A ``via="super_admin"`` result is PROVISIONAL: the caller must confirm the
    tenant exists and is not suspended before acting in it.
    """
    by_tenant = {m.tenant_id: m for m in memberships}
    if requested is not None:
        own = by_tenant.get(requested)
        if own is not None:
            return Resolution(tenant_id=own.tenant_id, role=own.role, via="requested")
        if is_super_admin:
            return Resolution(tenant_id=requested, role=None, via="super_admin")
        return Refusal(reason="not_a_member")
    if not memberships:
        return Refusal(reason="no_membership")
    first = memberships[0]
    return Resolution(
        tenant_id=first.tenant_id,
        role=first.role,
        via="single" if len(memberships) == 1 else "most_recent",
    )


def lazy_membership_role(realm_roles: list[str] | set[str]) -> MembershipRole:
    """Role for a membership lazily created from a claim (shadow mode).

    Mirrors migration 0072 / ``scripts/backfill_tenancy.py``: only an explicit
    realm ``tenant_admin`` maps to ``tenant_admin``, and never for a
    super_admin (whose composite role lists tenant_admin too — their authority
    is the realm role, not a membership).
    """
    if TENANT_ADMIN in realm_roles and SUPER_ADMIN not in realm_roles:
        return "tenant_admin"
    return "member"
