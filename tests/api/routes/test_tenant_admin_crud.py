"""Admin tier (Phase B, 3rd increment) — tenant registry CRUD.

PATCH /api/admin/tenants/{id} is the first admin *write*: a super_admin renames
a tenant and toggles its status (upsert — creates the registry row if the tenant
was only known via its projects). Same per-test-app + middleware-seeded-roles
idiom as `test_admin_authz.py`.
"""

from __future__ import annotations

from uuid import UUID, uuid4, uuid7

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes.admin import router as admin_router
from src.database.models import Project

pytestmark = pytest.mark.asyncio


def _build_app(db_session_factory, *, tenant_id: str, roles: list[str]) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = db_session_factory
    app.include_router(admin_router)
    subject_id = str(uuid4())

    class _SeedRoles(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = tenant_id
            request.state.roles = roles
            request.state.subject_id = subject_id
            return await call_next(request)

    app.add_middleware(_SeedRoles)
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


async def _seed_project(session, tenant_id: UUID) -> None:
    session.add(
        Project(
            id=uuid7(),
            tenant_id=tenant_id,
            owner_subject_id=uuid7(),
            visibility="tenant",
            type="generic",
            title="P",
        )
    )
    await session.flush()


async def test_super_admin_renames_and_suspends(db_session, db_session_factory) -> None:
    target = uuid4()
    await _seed_project(db_session, target)
    await db_session.commit()

    app = _build_app(db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"])
    async with _client(app) as c:
        r = await c.patch(
            f"/api/admin/tenants/{target}",
            json={"name": "Renamed Inc", "status": "suspended"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["tenant_id"] == str(target)
        assert body["name"] == "Renamed Inc"
        assert body["status"] == "suspended"
        assert body["project_count"] == 1  # registry merged with project aggregate

        # The change is visible on the list endpoint too.
        listing = await c.get("/api/admin/tenants")
    item = {i["tenant_id"]: i for i in listing.json()["items"]}[str(target)]
    assert item["name"] == "Renamed Inc"
    assert item["status"] == "suspended"


async def test_patch_upserts_a_project_only_tenant(db_session, db_session_factory) -> None:
    """A tenant known only via its projects (no registry row) gets one created."""
    target = uuid4()
    await _seed_project(db_session, target)
    await db_session.commit()

    app = _build_app(db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"])
    async with _client(app) as c:
        r = await c.patch(f"/api/admin/tenants/{target}", json={"name": "First Name"})
    assert r.status_code == 200
    assert r.json()["name"] == "First Name"
    assert r.json()["status"] == "active"  # default on insert


async def test_patch_forbidden_for_tenant_admin(db_session_factory) -> None:
    app = _build_app(db_session_factory, tenant_id=str(uuid4()), roles=["tenant_admin"])
    async with _client(app) as c:
        r = await c.patch(f"/api/admin/tenants/{uuid4()}", json={"name": "Nope"})
    assert r.status_code == 403


async def test_patch_empty_body_is_422(db_session_factory) -> None:
    app = _build_app(db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"])
    async with _client(app) as c:
        r = await c.patch(f"/api/admin/tenants/{uuid4()}", json={})
    assert r.status_code == 422


async def test_patch_invalid_status_is_422(db_session_factory) -> None:
    app = _build_app(db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"])
    async with _client(app) as c:
        r = await c.patch(f"/api/admin/tenants/{uuid4()}", json={"status": "frozen"})
    assert r.status_code == 422


async def test_patch_status_only_keeps_existing_name(db_session, db_session_factory) -> None:
    """status-only PATCH leaves an existing name untouched (None = unchanged)."""
    target = uuid4()
    await _seed_project(db_session, target)
    await db_session.commit()

    app = _build_app(db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"])
    async with _client(app) as c:
        await c.patch(f"/api/admin/tenants/{target}", json={"name": "Keep Me"})
        r = await c.patch(f"/api/admin/tenants/{target}", json={"status": "suspended"})
    assert r.status_code == 200
    assert r.json()["name"] == "Keep Me"
    assert r.json()["status"] == "suspended"
