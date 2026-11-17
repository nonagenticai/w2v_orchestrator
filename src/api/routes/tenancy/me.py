"""The caller's own tenancy (P2 WP4a).

- GET /api/me/tenants — the caller's ACTIVE memberships (+ tenant names), for
  the tenant switcher. Reachable with no membership at all (empty list) — it
  is in ``NO_MEMBERSHIP_EXEMPT_PATHS``.

Also the ``/api/auth/me`` helper :func:`memberships_and_pending_invites`: the
caller's memberships plus every pending, unexpired invite addressed to the
caller's VERIFIED email (any tenant). READ-ONLY — /me never accepts an invite
or writes a membership (owner decision); the caller joins explicitly via
``POST /api/invites/{invite_id}/accept`` or the token link.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes._common import SessionDep, SubjectIdDep
from src.api.routes.tenancy.invites import verified_email
from src.api.schemas.tenancy import MyTenant, MyTenantList, PendingInvite
from src.database.models import Tenant, TenantMember
from src.database.repositories.invites_repository import InvitesRepository

router = APIRouter(prefix="/api/me", tags=["tenancy"])


async def my_tenants(session: AsyncSession, subject_id: UUID) -> list[MyTenant]:
    rows = await session.execute(
        select(
            TenantMember.tenant_id,
            TenantMember.role,
            TenantMember.last_seen_at,
            Tenant.name,
            Tenant.status,
        )
        .outerjoin(Tenant, Tenant.id == TenantMember.tenant_id)
        .where(TenantMember.subject_id == subject_id, TenantMember.status == "active")
        .order_by(TenantMember.last_seen_at.desc(), TenantMember.tenant_id)
    )
    return [
        MyTenant(
            tenant_id=r.tenant_id,
            name=r.name,
            role=r.role,
            tenant_status=r.status,
            last_seen_at=r.last_seen_at,
        )
        for r in rows
    ]


async def pending_invites_for(session: AsyncSession, request: Request) -> list[PendingInvite]:
    """Pending, unexpired invites for the caller's VERIFIED email (case-
    insensitive), across tenants. An unverified email lists nothing (and so
    reveals nothing). No locks, no writes."""
    email = verified_email(request)
    if email is None:
        return []
    pending = await InvitesRepository(session).list_pending_for_email(email)
    if not pending:
        return []
    rows = await session.execute(
        select(Tenant.id, Tenant.name).where(Tenant.id.in_({i.tenant_id for i in pending}))
    )
    # A Result has .keys(), so dict(result) would treat it as a mapping — iterate.
    names = {tid: name for tid, name in rows.tuples()}
    return [
        PendingInvite(
            id=inv.id,
            tenant_id=inv.tenant_id,
            tenant_name=names.get(inv.tenant_id),
            role=inv.role,  # ty: ignore[invalid-argument-type]  # why: DB CHECK constrains it
            expires_at=inv.expires_at,
        )
        for inv in pending
    ]


async def memberships_and_pending_invites(
    session: AsyncSession, request: Request, subject_id: UUID
) -> tuple[list[MyTenant], list[PendingInvite]]:
    return await my_tenants(session, subject_id), await pending_invites_for(session, request)


@router.get("/tenants", response_model=MyTenantList, operation_id="me.tenants")
async def list_my_tenants(subject_id: SubjectIdDep, session: SessionDep) -> MyTenantList:
    return MyTenantList(items=await my_tenants(session, subject_id))
