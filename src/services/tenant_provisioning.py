"""Tenant creation + first-admin invite (P2 tenancy, WP4b).

Shared by the two tenant-creating routes:

* ``POST /api/admin/provision`` (super_admin) — ``created_via='super_admin'``;
* ``POST /api/tenants`` (partner licence lane) — ``created_via='partner'`` +
  ``partner_licence_id``.

w2v owns tenancy now: a tenant is a ``tenants`` row and its people are
``tenant_members`` rows reached through ``tenant_invites``. Keycloak only holds
the IDENTITY — no ``tenant_id``/``tenant_slug`` attribute and no realm-role
grant is ever written here (the membership middleware ignores both).

Nothing here commits: the caller's session owns the transaction. Identity side
effects are external (provisioning's identity invite API) and are not rolled
back with it — every step is idempotent so a retry converges (the identity
invite finds the existing account; a pending invite is reused; an existing
tenant row is left untouched).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.clients.identity_inviter import tenant_context
from src.database.models import Tenant, TenantMember
from src.database.repositories.admin_audit_repository import AuditRepository
from src.database.repositories.admin_repository import AdminRepository
from src.database.repositories.invites_repository import (
    InviteRole,
    InvitesRepository,
    new_invite_token,
)
from src.database.tenant_lock import lock_tenant_id

#: How long a first-admin invite stays acceptable.
INVITE_TTL = datetime.timedelta(days=7)

CreatedVia = Literal["super_admin", "partner"]


class _Inviter(Protocol):
    """The identity invite surface (``IdentityInviter``): find-or-create by
    email + the action email, and nothing that writes tenancy into Keycloak."""

    async def invite(
        self,
        email: str,
        *,
        redirect_uri: str | None = None,
        actor: UUID | None = None,
        context: str | None = None,
    ) -> None: ...


class TenantConflictError(Exception):
    """The tenant id is not available to this creator (exists under another
    creator, or — partner lane — is already in use)."""


@dataclass(frozen=True, slots=True)
class TenantCreation:
    tenant: Tenant
    created: bool


@dataclass(frozen=True, slots=True)
class InviteOutcome:
    """``invite_id`` is None when the email is already an active member (no
    invite needed). There is no subject id: provisioning never returns one, and
    the membership binds when the invitee accepts with its verified email."""

    invite_id: UUID | None
    already_member: bool


async def tenant_has_occupants(session: AsyncSession, tenant_id: UUID) -> bool:
    """True when ``tenant_id`` is in use: any project, member, invite or API key
    (see ``tenant_is_occupied`` in ``src/database/rls.py`` for what counts and
    what deliberately does not). Read through the SECURITY DEFINER resolver — the
    caller's session is RLS-bound to ANOTHER tenant, so a plain count would
    answer 0 under enforce."""
    occupied = (
        await session.execute(text("SELECT tenant_is_occupied(:t)"), {"t": tenant_id})
    ).scalar_one()
    return bool(occupied)


async def create_tenant(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    name: str,
    created_via: CreatedVia,
    actor_subject_id: UUID | None,
    partner_licence_id: UUID | None = None,
) -> TenantCreation:
    """Insert the ``tenants`` row if absent and audit ``tenant.create``.

    ``super_admin``: idempotent on ``tenant_id`` for the same creator; anything
    else is :class:`TenantConflictError`.

    ``partner`` (the caller picks ``tenant_id``, so this is the takeover
    surface) — owner rule, all evaluated under one advisory lock + the row lock,
    in the same transaction as the insert:

    (a) no ``tenants`` row, and the tenant is not occupied (no projects,
        members, invites or API keys) → create it;
    (b) a row whose ``partner_licence_id`` is the caller's → idempotent replay;
    (c) a ``created_via='backfill'`` row with no ``partner_licence_id`` that is
        not occupied → adopt it (``partner_licence_id`` set);

    everything else → :class:`TenantConflictError` (``TENANT_ID_UNAVAILABLE``).
    """
    if created_via == "partner":
        return await _partner_create_tenant(
            session,
            tenant_id=tenant_id,
            name=name,
            actor_subject_id=actor_subject_id,
            partner_licence_id=partner_licence_id,
        )
    admin = AdminRepository(session)
    created = await admin.ensure_tenant(
        tenant_id,
        created_via=created_via,
        name=name,
        partner_licence_id=partner_licence_id,
    )
    tenant = await admin.get_tenant(tenant_id)
    if tenant is None:  # pragma: no cover - the row was just ensured
        raise RuntimeError(f"tenant {tenant_id} vanished after ensure_tenant")
    if not created and tenant.status == "deleted":  # an offboarded id is never re-provisioned
        raise TenantConflictError(f"tenant {tenant_id} was deleted")
    if not created and (
        tenant.created_via != created_via or tenant.partner_licence_id != partner_licence_id
    ):
        raise TenantConflictError(f"tenant {tenant_id} exists and was created by another party")
    if created:
        await _audit_create(session, tenant_id, name, created_via, actor_subject_id, None)
    return TenantCreation(tenant=tenant, created=created)


async def _partner_create_tenant(
    session: AsyncSession,
    *,
    tenant_id: UUID,
    name: str,
    actor_subject_id: UUID | None,
    partner_licence_id: UUID | None,
) -> TenantCreation:
    if partner_licence_id is None:
        raise TenantConflictError("a partner creation needs the partner licence id")
    await lock_tenant_id(session, tenant_id)
    tenant = (
        await session.execute(select(Tenant).where(Tenant.id == tenant_id).with_for_update())
    ).scalar_one_or_none()

    if tenant is not None:
        if tenant.status == "deleted":  # an offboarded id is never re-provisioned
            raise TenantConflictError(f"tenant {tenant_id} was deleted")
        if tenant.partner_licence_id == partner_licence_id:  # (b) replay
            return TenantCreation(tenant=tenant, created=False)
        if (  # (c) adopt an empty backfill row
            tenant.created_via == "backfill"
            and tenant.partner_licence_id is None
            and not await tenant_has_occupants(session, tenant_id)
        ):
            tenant.partner_licence_id = partner_licence_id
            if tenant.name is None:
                tenant.name = name
            await session.flush()
            await AuditRepository(session).record(
                actor_subject_id=actor_subject_id,
                action="tenant.update",
                target_tenant_id=tenant_id,
                detail={"partner_adopt": True, "partner_licence_id": str(partner_licence_id)},
            )
            return TenantCreation(tenant=tenant, created=True)
        raise TenantConflictError(f"tenant {tenant_id} is not available to this partner")

    if await tenant_has_occupants(session, tenant_id):  # licensed, unregistered tenant
        raise TenantConflictError(f"tenant {tenant_id} already holds data")
    admin = AdminRepository(session)
    created = await admin.ensure_tenant(
        tenant_id, created_via="partner", name=name, partner_licence_id=partner_licence_id
    )
    tenant = await admin.get_tenant(tenant_id)
    if tenant is None:  # pragma: no cover - the row was just ensured
        raise RuntimeError(f"tenant {tenant_id} vanished after ensure_tenant")
    if not created:  # a non-partner writer (backfill / invite FK repair) won the race
        raise TenantConflictError(f"tenant {tenant_id} was registered concurrently")
    await _audit_create(session, tenant_id, name, "partner", actor_subject_id, partner_licence_id)
    return TenantCreation(tenant=tenant, created=True)


async def _audit_create(
    session: AsyncSession,
    tenant_id: UUID,
    name: str,
    created_via: CreatedVia,
    actor_subject_id: UUID | None,
    partner_licence_id: UUID | None,
) -> None:
    await AuditRepository(session).record(
        actor_subject_id=actor_subject_id,
        action="tenant.create",
        target_tenant_id=tenant_id,
        detail={
            "name": name,
            "created_via": created_via,
            "partner_licence_id": str(partner_licence_id) if partner_licence_id else None,
        },
    )


async def invite_user(
    session: AsyncSession,
    inviter: _Inviter,
    *,
    tenant_id: UUID,
    email: str,
    role: InviteRole,
    invited_by: UUID | None,
) -> InviteOutcome:
    """Make sure ``email`` has a pending invite into ``tenant_id`` with
    ``role`` and that provisioning has been asked to provide its identity.

    * already an ACTIVE member (by email) → nothing (``already_member``).
    * a pending invite for the email → reused (idempotent replay; its identity
      invite was already accepted when it was created, so no re-send).
    * else a new invite row, then provisioning's identity invite (find or create
      by email with no attributes/credentials/roles, plus the action email it
      needs) with ``actor`` + ``context`` audit fields. An
      :class:`~src.clients.identity_inviter.IdentityInviteError` propagates;
      the caller maps it and rolls the row back.

    The raw token is not returned: the invitee accepts by logging in with the
    verified email and accepting the invite ``/api/auth/me`` lists
    (``POST /api/invites/{id}/accept``), or via a WP4a resend link.
    """
    email = email.strip()
    member = (
        await session.execute(
            select(TenantMember.subject_id).where(
                TenantMember.tenant_id == tenant_id,
                TenantMember.status == "active",
                func.lower(TenantMember.email) == email.lower(),
            )
        )
    ).first()
    if member is not None:
        return InviteOutcome(None, True)

    invites = InvitesRepository(session)
    pending = await invites.get_pending_for_tenant_email(tenant_id, email)
    if pending is not None:
        return InviteOutcome(pending.id, False)

    _token, token_hash = new_invite_token()
    invite = await invites.create(
        tenant_id=tenant_id,
        email=email,
        role=role,
        token_hash=token_hash,
        expires_at=datetime.datetime.now(datetime.UTC) + INVITE_TTL,
        invited_by=invited_by,
    )
    await inviter.invite(email, actor=invited_by, context=tenant_context(tenant_id))
    await AuditRepository(session).record(
        actor_subject_id=invited_by,
        action="member.invite",
        target_tenant_id=tenant_id,
        detail={"invite_id": str(invite.id), "email": email, "role": role},
    )
    return InviteOutcome(invite.id, False)
