"""Tenant members (P2 WP4a) — the w2v-owned roster of the EFFECTIVE tenant.

- GET    /api/tenant/members                   (tenant_admin) roster, paginated
- PATCH  /api/tenant/members/{subject_id}      (tenant_admin) role member|tenant_admin
- DELETE /api/tenant/members/{subject_id}      (tenant_admin) status → removed

A membership role lives in ``tenant_members.role`` — never a Keycloak realm role
(a realm role is identity-wide, so a w2v tenant_admin granting one would change
the person in fr/kag too). Every mutation is scoped to the acting tenant
(``TenantAdminDep`` → the membership-resolved / scope-switched tenant), so a
tenant_admin can never touch another tenant's members by subject id.

Last-admin guard: a tenant always keeps ≥1 active tenant_admin. The active admin
rows are locked ``FOR UPDATE`` before counting, so two admins demoting each
other concurrently serialize instead of both succeeding.
"""

from __future__ import annotations

import logging
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Query
from sqlalchemy import ColumnElement, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes._common import (
    LICENSE_GATE_ONLY,
    AuditRepoDep,
    MembersRepoDep,
    SessionDep,
    SubjectIdDep,
    TenantAdminDep,
)
from src.api.routes.tenancy.errors import TenancyError
from src.api.schemas.tenancy import (
    TenantMemberList,
    TenantMemberOut,
    TenantMemberRemoved,
    TenantMemberRoleUpdate,
)
from src.database.models import TenantMember

router = APIRouter(prefix="/api/tenant/members", tags=["tenancy"])

_log = logging.getLogger("w2v_orchestrator.tenancy")

LAST_TENANT_ADMIN = "LAST_TENANT_ADMIN"


def _out(row: TenantMember) -> TenantMemberOut:
    return TenantMemberOut(
        subject_id=row.subject_id,
        email=row.email,
        display_name=row.display_name,
        role=row.role,  # ty: ignore[invalid-argument-type]  # why: DB CHECK constrains it to the Literal
        status=row.status,  # ty: ignore[invalid-argument-type]  # why: DB CHECK constrains it to the Literal
        first_seen_at=row.first_seen_at,
        last_seen_at=row.last_seen_at,
    )


async def _locked_active_admins(session: AsyncSession, tenant_id: UUID) -> set[UUID]:
    """The tenant's active tenant_admins, row-locked for this transaction."""
    rows = await session.execute(
        select(TenantMember.subject_id)
        .where(
            TenantMember.tenant_id == tenant_id,
            TenantMember.status == "active",
            TenantMember.role == "tenant_admin",
        )
        .with_for_update()
    )
    return {r.subject_id for r in rows}


def _refuse_last_admin(subject_id: UUID, admins: set[UUID]) -> None:
    if admins == {subject_id}:
        raise TenancyError(
            409, LAST_TENANT_ADMIN, "A workspace must keep at least one active tenant admin."
        )


@router.get(
    "",
    response_model=TenantMemberList,
    operation_id="tenant.members.list",
    dependencies=LICENSE_GATE_ONLY,
)
async def list_members(
    tenant_id: TenantAdminDep,
    session: SessionDep,
    member_status: Literal["active", "removed", "all"] = Query("active", alias="status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> TenantMemberList:
    conditions: list[ColumnElement[bool]] = [TenantMember.tenant_id == tenant_id]
    if member_status != "all":
        conditions.append(TenantMember.status == member_status)
    rows = (
        (
            await session.execute(
                select(TenantMember)
                .where(*conditions)
                .order_by(
                    TenantMember.last_seen_at.desc(),
                    TenantMember.subject_id,
                )
                .limit(limit)
                .offset(offset)
            )
        )
        .scalars()
        .all()
    )
    total = (
        await session.execute(select(func.count()).select_from(TenantMember).where(*conditions))
    ).scalar_one()
    return TenantMemberList(items=[_out(r) for r in rows], total=total, limit=limit, offset=offset)


@router.patch(
    "/{subject_id}",
    response_model=TenantMemberOut,
    operation_id="tenant.members.set_role",
    dependencies=LICENSE_GATE_ONLY,
)
async def set_member_role(
    subject_id: UUID,
    body: TenantMemberRoleUpdate,
    tenant_id: TenantAdminDep,
    actor: SubjectIdDep,
    members: MembersRepoDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> TenantMemberOut:
    admins = await _locked_active_admins(session, tenant_id)
    member = await members.get_active_membership(subject_id, tenant_id)
    if member is None:
        raise TenancyError(404, "MEMBER_NOT_FOUND", "member not found in this workspace")
    before = member.role
    if before == body.role:
        return _out(member)
    if before == "tenant_admin":
        _refuse_last_admin(subject_id, admins)
    await members.set_role(subject_id, tenant_id, body.role)
    await audit.record(
        actor_subject_id=actor,
        action="member.set_role",
        target_tenant_id=tenant_id,
        detail={"subject_id": str(subject_id), "from": before, "to": body.role},
    )
    await session.commit()
    await session.refresh(member)
    return _out(member)


@router.delete(
    "/{subject_id}",
    response_model=TenantMemberRemoved,
    operation_id="tenant.members.remove",
    dependencies=LICENSE_GATE_ONLY,
)
async def remove_member(
    subject_id: UUID,
    tenant_id: TenantAdminDep,
    actor: SubjectIdDep,
    members: MembersRepoDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> TenantMemberRemoved:
    admins = await _locked_active_admins(session, tenant_id)
    member = await members.get_active_membership(subject_id, tenant_id)
    if member is None:
        raise TenancyError(404, "MEMBER_NOT_FOUND", "member not found in this workspace")
    if member.role == "tenant_admin":
        _refuse_last_admin(subject_id, admins)
    await members.remove_member(subject_id, tenant_id)
    await audit.record(
        actor_subject_id=actor,
        action="member.remove",
        target_tenant_id=tenant_id,
        detail={"subject_id": str(subject_id), "role": member.role},
    )
    # w2v-side only: the Keycloak identity is not w2v's to disable (provisioning
    # owns identities; another app may still rely on the account).
    await session.commit()
    return TenantMemberRemoved(subject_id=subject_id)
