"""Studio v3 — auth, blueprints, health route tests (T10)."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.routes.blueprints import router as blueprints_router
from src.api.routes.health import router as health_router

pytestmark = pytest.mark.asyncio


async def test_auth_me_returns_seeded_uuids(client: AsyncClient) -> None:
    from tests.conftest import SeededClient

    assert isinstance(client, SeededClient)
    r = await client.get("/api/auth/me")
    assert r.status_code == 200
    body = r.json()
    assert body["tenant_id"] == client.tenant_id
    assert body["subject_id"] == client.subject_id
    # The seeded fixture sets no roles on request.state -> empty default.
    assert body["roles"] == []


async def test_auth_me_forwards_roles_email_name(db_session_factory) -> None:
    """When AuthMiddleware (w2v_common >=0.18.10) stashes roles/email/name on
    request.state, /auth/me must surface them — the wire that lights up the
    admin tier. Seeds request.state directly (no real Keycloak) to lock the
    endpoint contract independent of the middleware's introspection paths. The
    session factory backs /me's best-effort members-roster upsert (Phase C C-1)."""
    from uuid import uuid4

    from starlette.middleware.base import BaseHTTPMiddleware

    from src.api.routes.auth import router as auth_router

    tenant_id = str(uuid4())
    subject_id = str(uuid4())

    class _SeedRoles(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = tenant_id
            request.state.subject_id = subject_id
            request.state.roles = ["super_admin", "tenant_admin"]
            request.state.email = "ada@acme.test"
            request.state.name = "ada"
            return await call_next(request)

    app = FastAPI()
    app.state.session_factory = db_session_factory
    app.include_router(auth_router)
    app.add_middleware(_SeedRoles)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        r = await c.get("/api/auth/me")
    assert r.status_code == 200
    body = r.json()
    assert body["roles"] == ["super_admin", "tenant_admin"]
    assert body["email"] == "ada@acme.test"
    assert body["name"] == "ada"


async def test_blueprints_returns_registered_blueprints() -> None:
    app = FastAPI()
    app.include_router(blueprints_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        r = await c.get("/api/blueprints")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    ids = {bp["id"] for bp in data}
    assert "generic_v1" in ids


async def test_blueprint_detail_returns_dag_for_known_id() -> None:
    app = FastAPI()
    app.include_router(blueprints_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        r = await c.get("/api/blueprints/seo_audit_v1")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "seo_audit_v1"
    assert body["project_type"] == "seo_audit"
    names = {s["name"] for s in body["stages"]}
    assert "seo_audit_scan" in names
    assert "seo_audit_compile" in names


async def test_blueprint_detail_404_for_unknown_id() -> None:
    app = FastAPI()
    app.include_router(blueprints_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        r = await c.get("/api/blueprints/does_not_exist_v9")
    assert r.status_code == 404


async def test_healthz_always_ok() -> None:
    app = FastAPI()
    app.include_router(health_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        r = await c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_readyz_503_when_engine_missing() -> None:
    app = FastAPI()
    app.include_router(health_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        r = await c.get("/readyz")
    assert r.status_code == 503


async def test_readyz_ok_with_engine(db_engine) -> None:
    app = FastAPI()
    app.state.db_engine = db_engine
    app.include_router(health_router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        r = await c.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"
