"""The effective tenant itself (P2 WP4a).

- PATCH /api/tenant {name} — a tenant_admin renames their own workspace.

Status changes (suspend/reactivate) stay super_admin-only on
``PATCH /api/admin/tenants/{id}``; a tenant_admin can change the name only.
Audited as ``tenant.update`` with the before/after name.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.routes._common import (
    LICENSE_GATE_ONLY,
    AdminRepoDep,
    AuditRepoDep,
    SessionDep,
    SubjectIdDep,
    TenantAdminDep,
)
from src.api.schemas.tenancy import TenantOut, TenantRename

router = APIRouter(prefix="/api/tenant", tags=["tenancy"])


@router.patch(
    "", response_model=TenantOut, operation_id="tenant.rename", dependencies=LICENSE_GATE_ONLY
)
async def rename_tenant(
    body: TenantRename,
    tenant_id: TenantAdminDep,
    actor: SubjectIdDep,
    repo: AdminRepoDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> TenantOut:
    existing = await repo.get_tenant(tenant_id)
    before = existing.name if existing is not None else None
    row = await repo.upsert_tenant(tenant_id, name=body.name, created_via="backfill")
    await audit.record(
        actor_subject_id=actor,
        action="tenant.update",
        target_tenant_id=tenant_id,
        detail={"before": {"name": before}, "after": {"name": row.name}, "by": "tenant_admin"},
    )
    await session.commit()
    return TenantOut(tenant_id=tenant_id, name=row.name, status=row.status)
