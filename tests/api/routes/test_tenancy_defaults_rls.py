"""P6 ORCH-1 — ``tenant_defaults`` RLS under the prod runtime posture.

The test DB user OWNS the tables and RLS is ENABLE-not-FORCE, so only a non-owner
``NOBYPASSRLS`` role makes rows depend on ``app.current_tenant`` (the prod
``w2v_orch_app`` posture). Two claims:

1. The POLICY (not the repository's WHERE clause) isolates tenants: a raw, unfiltered
   ``SELECT`` bound to tenant A sees none of B's row, and a write of B's row while
   bound to A is refused by ``WITH CHECK``.
2. The PUT binds the AUTHORIZED effective tenant: a super_admin whose claim is A and
   who scope-switches into B writes B's row (without the bind the INSERT would hit
   the RLS ``WITH CHECK`` under A); without the switch the same admin reads A.
"""

from __future__ import annotations

from uuid import UUID, uuid4, uuid7

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import insert, select, text
from sqlalchemy.exc import DBAPIError
from starlette.middleware.base import BaseHTTPMiddleware
from w2v_common.auth.identity_context import set_identity_context

from src.api.routes.tenancy.defaults import router
from src.api.routes.tenancy.errors import register_tenancy_exception_handlers
from src.database.models import Tenant, TenantDefaults
from tests.api.routes._licence_stubs import override_entitled

# Re-exported fixtures (pytest resolves them by name in this module); the test
# parameters below that take them are therefore F811 "redefinitions" by design.
from tests.api.routes.test_tenancy_connections_rls import (  # noqa: F401 — fixtures
    _enforce,
    restricted_factory,
)

pytestmark = pytest.mark.asyncio


async def _tenants(owner_factory, *ids: UUID) -> None:
    async with owner_factory() as owner:
        for tid in ids:
            await owner.execute(insert(Tenant).values(id=tid, name=f"t-{tid}"))
        await owner.commit()


async def _bind(session, tenant: UUID) -> None:
    await session.execute(
        text("SELECT set_config('app.current_tenant', :t, true)"), {"t": str(tenant)}
    )


async def test_policy_isolates_tenants_without_any_where_clause(
    db_session_factory,
    restricted_factory,  # noqa: F811 — the imported fixture
) -> None:
    a, b = uuid7(), uuid7()
    await _tenants(db_session_factory, a, b)
    async with db_session_factory() as owner:
        await owner.execute(
            insert(TenantDefaults).values(
                tenant_id=b, retention_mode="off", voice={"voice_id": "x"}
            )
        )
        await owner.commit()

    async with restricted_factory() as s:
        await _bind(s, a)
        assert (await s.execute(select(TenantDefaults))).scalars().all() == []
        await s.rollback()
        # Control: the same unfiltered read bound to B sees B's row.
        await _bind(s, b)
        rows = (await s.execute(select(TenantDefaults))).scalars().all()
        assert [r.tenant_id for r in rows] == [b]
        await s.rollback()

    async with restricted_factory() as s:
        await _bind(s, a)
        with pytest.raises(DBAPIError, match="row-level security"):
            await s.execute(insert(TenantDefaults).values(tenant_id=b, retention_mode="auto"))
        await s.rollback()
    async with restricted_factory() as s:
        await _bind(s, a)
        updated = await s.execute(
            text("UPDATE tenant_defaults SET retention_mode = 'auto' WHERE tenant_id = :b"),
            {"b": b},
        )
        assert updated.rowcount == 0, "A must not be able to touch B's row"
        await s.commit()
    async with db_session_factory() as owner:
        row = (
            await owner.execute(select(TenantDefaults).where(TenantDefaults.tenant_id == b))
        ).scalar_one()
        assert row.retention_mode == "off"


def _app(factory, *, claim: UUID) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = factory
    app.include_router(router)
    register_tenancy_exception_handlers(app)
    override_entitled(app)
    subject = str(uuid4())

    class _ClaimIdentity(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = str(claim)
            request.state.roles = ["super_admin"]
            request.state.subject_id = subject
            set_identity_context(str(claim), None, subject)
            return await call_next(request)

    app.add_middleware(_ClaimIdentity)
    return app


async def test_super_admin_scope_switch_writes_and_reads_the_selected_tenant(
    db_session_factory,
    restricted_factory,  # noqa: F811 — the imported fixture
) -> None:
    claim, target = uuid7(), uuid7()
    await _tenants(db_session_factory, claim, target)
    app = _app(restricted_factory, claim=claim)
    switch = {"X-Active-Tenant-Id": str(target)}
    body = {"expected_version": 0, "voice_direction": {"tone": "calm"}}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.put("/api/tenant/defaults", json=body, headers=switch)
        assert r.status_code == 200, r.text
        assert r.json()["version"] == 1

        r = await c.get("/api/tenant/defaults", headers=switch)
        assert r.status_code == 200, r.text
        assert r.json()["voice_direction"] == {"tone": "calm"}

        # Control: without the switch the same admin reads its OWN (empty) tenant.
        r = await c.get("/api/tenant/defaults")
        assert r.status_code == 200, r.text
        assert r.json()["version"] == 0

    async with db_session_factory() as owner:
        tenants = (await owner.execute(select(TenantDefaults.tenant_id))).scalars().all()
    assert target in tenants and claim not in tenants
