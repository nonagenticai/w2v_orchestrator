"""P3 — ``/api/tenant/connections`` binds ``app.current_tenant`` to the EFFECTIVE tenant.

Claim mode, ``TENANT_RLS_MODE=enforce``, the handler session running as a
throwaway non-owner ``NOBYPASSRLS`` role (the prod ``w2v_orch_app`` posture — the
test DB user OWNS the tables and RLS is ENABLE-not-FORCE, so only a non-owner role
makes rows depend on the GUC). A ``super_admin`` whose token claims tenant A
scope-switches into tenant B with ``X-Active-Tenant-Id``.

Without the route-level bind the ``after_begin`` listener falls back to the identity
ContextVar — the CLAIM tenant A — so B's rows are invisible (GET shows no
connection) and the PUT's INSERT dies on the RLS ``WITH CHECK``. With the bind, the
admin reads and writes B, and A is untouched.
"""

from __future__ import annotations

import os
from typing import Any
from uuid import UUID, uuid4, uuid7

import fakeredis.aioredis
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event, func, insert, select, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.middleware.base import BaseHTTPMiddleware
from w2v_common.auth.identity_context import clear_identity_context, set_identity_context

from src.api.routes.tenancy.connections import (
    get_connections_keyring,
    get_connections_policy,
    get_provider_tester,
    router,
)
from src.api.routes.tenancy.errors import register_tenancy_exception_handlers
from src.config import clear_settings_cache
from src.database.models import Tenant, TenantConnection
from src.database.rls_events import install_rls_tenant_rebind
from src.services.providers.crypto import Keyring
from src.services.providers.settings import ProviderPolicySettings
from src.services.providers.tester import ProviderTestResult
from tests.api.routes._licence_stubs import override_entitled
from tests.conftest import TEST_DATABASE_URL, require_createrole

pytestmark = pytest.mark.asyncio

BASE = "https://llm-a.example.test/v1"
_POLICY = ProviderPolicySettings(
    ssrf_mode="private_ok", egress_allowlist=frozenset({"llm-a.example.test"})
)


class _OkTester:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def test(self, **kwargs: Any) -> ProviderTestResult:
        self.calls.append(kwargs)
        return ProviderTestResult(True, "ok", 200)


@pytest.fixture(autouse=True)
def _enforce(monkeypatch):
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    monkeypatch.setenv("W2V_TENANCY_SOURCE", "claim")
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "false")
    clear_settings_cache()
    clear_identity_context()
    install_rls_tenant_rebind()  # idempotent; asgi installs it in prod
    yield
    clear_settings_cache()
    clear_identity_context()


@pytest.fixture
async def restricted_factory(db_session_factory):
    """Sessions that run as a non-owner NOBYPASSRLS role for their whole life."""
    role = f"conn_rls_probe_{uuid7().hex[:12]}"
    async with db_session_factory() as owner:
        schema = (await owner.execute(text("SELECT current_schema()"))).scalar_one()
        try:
            await owner.execute(text(f'CREATE ROLE "{role}" NOLOGIN NOSUPERUSER NOBYPASSRLS'))
        except ProgrammingError as exc:
            await owner.rollback()
            require_createrole(exc)
        for stmt in (
            f'GRANT "{role}" TO CURRENT_USER',
            f'GRANT USAGE ON SCHEMA "{schema}" TO "{role}"',
            f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA "{schema}" TO "{role}"',
            f'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA "{schema}" TO "{role}"',
            f'GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA "{schema}" TO "{role}"',
        ):
            await owner.execute(text(stmt))
        await owner.commit()

    engine = create_async_engine(TEST_DATABASE_URL)  # inherits PGOPTIONS (run schema)

    @event.listens_for(engine.sync_engine, "connect")
    def _set_role(dbapi_conn, _record) -> None:  # type: ignore[no-untyped-def]
        cur = dbapi_conn.cursor()
        cur.execute(f'SET ROLE "{role}"')
        cur.close()
        dbapi_conn.commit()  # a SET inside a rolled-back txn would revert

    try:
        yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    finally:
        await engine.dispose()
        async with db_session_factory() as owner:
            for stmt in (
                f'REVOKE ALL ON ALL TABLES IN SCHEMA "{schema}" FROM "{role}"',
                f'REVOKE ALL ON ALL SEQUENCES IN SCHEMA "{schema}" FROM "{role}"',
                f'REVOKE ALL ON ALL FUNCTIONS IN SCHEMA "{schema}" FROM "{role}"',
                f'REVOKE ALL ON SCHEMA "{schema}" FROM "{role}"',
                f'REVOKE "{role}" FROM CURRENT_USER',
                f'DROP OWNED BY "{role}"',
                f'DROP ROLE IF EXISTS "{role}"',
            ):
                try:
                    await owner.execute(text(stmt))
                    await owner.commit()
                except ProgrammingError:
                    await owner.rollback()


def _app(factory, *, claim: UUID, tester: _OkTester) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = factory
    app.state.redis = fakeredis.aioredis.FakeRedis()
    app.include_router(router)
    register_tenancy_exception_handlers(app)
    override_entitled(app)
    app.dependency_overrides[get_provider_tester] = lambda: tester
    app.dependency_overrides[get_connections_policy] = lambda: _POLICY
    keyring = Keyring([("k1", os.urandom(32))])
    app.dependency_overrides[get_connections_keyring] = lambda: keyring
    subject = str(uuid4())

    class _ClaimIdentity(BaseHTTPMiddleware):
        """What AuthMiddleware does in claim mode: request.state from the token and
        the identity ContextVar set to the CLAIM tenant."""

        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = str(claim)
            request.state.roles = ["super_admin"]
            request.state.subject_id = subject
            set_identity_context(str(claim), None, subject)
            return await call_next(request)

    app.add_middleware(_ClaimIdentity)
    return app


async def _rows(owner_factory, tenant: UUID) -> int:
    async with owner_factory() as s:
        return (
            await s.execute(
                select(func.count())
                .select_from(TenantConnection)
                .where(TenantConnection.tenant_id == tenant)
            )
        ).scalar_one()


async def test_super_admin_scope_switch_reads_and_writes_the_selected_tenant(
    db_session_factory, restricted_factory
) -> None:
    claim, target = uuid7(), uuid7()
    async with db_session_factory() as owner:
        await owner.execute(insert(Tenant).values(id=target, name=f"t-{target}"))
        await owner.commit()
    tester = _OkTester()
    app = _app(restricted_factory, claim=claim, tester=tester)
    switch = {"X-Active-Tenant-Id": str(target)}
    body = {
        "kind": "openai-compatible",
        "base_url": BASE,
        "model": "m-1",
        "secret": "sk-test",  # pragma: allowlist secret
        "expected_version": 0,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.put("/api/tenant/connections/llm", json=body, headers=switch)
        assert r.status_code == 200, r.text
        assert r.json()["connection"]["version"] == 1

        r = await c.get("/api/tenant/connections", headers=switch)
        assert r.status_code == 200, r.text
        llm = next(i for i in r.json()["items"] if i["slot"] == "llm")
        assert llm["source"] == "tenant"
        assert llm["connection"]["version"] == 1

        # Re-test the stored version (rollback + commit mid-request, re-bound).
        r = await c.post("/api/tenant/connections/llm/test", headers=switch)
        assert r.status_code == 200, r.text
        assert r.json()["version"] == 1

        # Control: without the switch the same admin sees its OWN (empty) tenant.
        r = await c.get("/api/tenant/connections")
        assert r.status_code == 200, r.text
        own = next(i for i in r.json()["items"] if i["slot"] == "llm")
        assert own["connection"] is None

    assert await _rows(db_session_factory, target) == 1
    assert await _rows(db_session_factory, claim) == 0
    assert len(tester.calls) == 2
