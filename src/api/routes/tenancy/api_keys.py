"""Tenant API keys (P2 tenancy, WP4b) — tenant_admin manages their tenant's keys.

- GET    /api/tenant/api-keys           → live keys (never the secret or hash).
- POST   /api/tenant/api-keys           → mint a key; the ``w2v_sk_…`` secret is
  returned ONCE in this response and is unrecoverable afterwards.
- DELETE /api/tenant/api-keys/{key_id}  → revoke (the next request with it is 401).

A key authenticates as a ``member`` of its tenant under a synthetic subject
(``src.auth.tenancy_middleware.api_key_subject``) — so a key can never manage
keys (tenant_admin only). ``scopes`` (P8) add narrow grants on top:
``connections:write`` = the bulk ``PUT /api/tenant/connections``. Mutations are audited (``apikey.create`` /
``apikey.revoke``) atomically with the write.

RLS (0090): ``tenant_api_keys`` carries ``tenant_isolation``, so every route here
binds ``app.current_tenant`` to the AUTHORIZED effective tenant (after the
tenant_admin + scope-switch checks) before touching the table — the ``after_begin``
fallback is the token CLAIM, which is the wrong tenant for a super_admin acting
via ``X-Active-Tenant-Id``.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, params, status

from src.api.routes._common import (
    LICENSE_GATE_ONLY,
    AuditRepoDep,
    SessionDep,
    SubjectIdDep,
    TenantAdminDep,
)
from src.api.schemas.tenancy_keys import ApiKeyCreate, ApiKeyCreated, ApiKeyList, ApiKeySummary
from src.database.models import TenantApiKey
from src.database.repositories.api_keys_repository import ApiKeysRepository
from src.database.session import bind_session_tenant


async def bind_api_keys_tenant_rls(tenant_id: TenantAdminDep, session: SessionDep) -> None:
    """Bind ``app.current_tenant`` to the authorized tenant (module doc). Also
    recorded in ``session.info`` so ``after_begin`` re-binds it after a commit."""
    await bind_session_tenant(session, tenant_id)


#: Entitlement + the RLS bind, on every key-management route.
_KEYS_GATE: list[params.Depends] = [*LICENSE_GATE_ONLY, Depends(bind_api_keys_tenant_rls)]

router = APIRouter(prefix="/api/tenant/api-keys", tags=["tenancy"])


def api_key_summary(row: TenantApiKey) -> ApiKeySummary:
    """The listable shape of a key — never the secret or its hash."""
    return ApiKeySummary(
        id=row.id,
        name=row.name,
        prefix=row.prefix,
        created_by=row.created_by,
        created_at=row.created_at,
        last_used_at=row.last_used_at,
        expires_at=row.expires_at,
        scopes=list(row.scopes or []),
        created_via="partner" if row.partner_licence_id is not None else "tenant_admin",
    )


@router.get(
    "",
    response_model=ApiKeyList,
    operation_id="tenant.api_keys.list",
    dependencies=_KEYS_GATE,
)
async def list_api_keys(tenant_id: TenantAdminDep, session: SessionDep) -> ApiKeyList:
    rows = await ApiKeysRepository(session).list_for_tenant(tenant_id)
    return ApiKeyList(items=[api_key_summary(r) for r in rows], total=len(rows))


@router.post(
    "",
    response_model=ApiKeyCreated,
    status_code=status.HTTP_201_CREATED,
    operation_id="tenant.api_keys.create",
    dependencies=_KEYS_GATE,
)
async def create_api_key(
    body: ApiKeyCreate,
    tenant_id: TenantAdminDep,
    actor: SubjectIdDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> ApiKeyCreated:
    row, full_key = await ApiKeysRepository(session).create(
        tenant_id=tenant_id,
        name=body.name,
        created_by=actor,
        expires_at=body.expires_at,
        scopes=list(body.scopes),
    )
    await session.refresh(row)  # load server-default created_at (async: no lazy load)
    await audit.record(
        actor_subject_id=actor,
        action="apikey.create",
        target_tenant_id=tenant_id,
        # The public prefix only — never the secret.
        detail={
            "key_id": str(row.id),
            "name": row.name,
            "prefix": row.prefix,
            "scopes": list(row.scopes or []),
        },
    )
    await session.commit()
    return ApiKeyCreated(**api_key_summary(row).model_dump(), key=full_key)


@router.delete(
    "/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="tenant.api_keys.revoke",
    dependencies=_KEYS_GATE,
)
async def revoke_api_key(
    key_id: UUID,
    tenant_id: TenantAdminDep,
    actor: SubjectIdDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> Response:
    # Scoped to the acting tenant: another tenant's key id is a 404, not a revoke.
    if not await ApiKeysRepository(session).revoke(key_id, tenant_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "API key not found")
    await audit.record(
        actor_subject_id=actor,
        action="apikey.revoke",
        target_tenant_id=tenant_id,
        detail={"key_id": str(key_id)},
    )
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
