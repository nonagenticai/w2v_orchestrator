"""0090 — ``tenant_api_keys`` RLS under the prod runtime posture.

Only a non-owner ``NOBYPASSRLS`` role (prod ``w2v_orch_app``) makes rows depend on
``app.current_tenant`` (the test DB user OWNS the table and RLS is ENABLE-not-FORCE).
So every test here runs its sessions through ``restricted_factory``.

What is proven:
* the POLICY, not a WHERE clause, isolates tenants: bound to A, the repository
  asked for B's keys BY B's tenant_id lists nothing and revokes nothing;
* key AUTHENTICATION still works with NO tenant bound (the ``resolve_api_key``
  SECURITY DEFINER resolver) — and the control shows a plain SELECT by prefix
  under the same posture finds nothing, so the resolver is load-bearing;
* minting sees another tenant's prefix (``api_key_prefix_taken``) and re-rolls;
* the registered verifier resolves + bumps ``last_used_at`` (binds the key's tenant);
* the key-management routes bind the AUTHORIZED effective tenant.
"""

from __future__ import annotations

from uuid import UUID, uuid4, uuid7

import fakeredis.aioredis
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import insert, select, text
from sqlalchemy.exc import DBAPIError
from starlette.middleware.base import BaseHTTPMiddleware
from w2v_common.auth.identity_context import set_identity_context

from src.api.routes.tenancy import api_keys
from src.api.routes.tenancy.errors import register_tenancy_exception_handlers
from src.auth import api_key_format
from src.auth.tenancy_middleware import make_api_key_verifier
from src.database.models import Tenant, TenantApiKey
from src.database.repositories.api_keys_repository import ApiKeysRepository
from tests.api.routes._licence_stubs import override_entitled

# Re-exported fixtures (pytest resolves them by name in this module).
from tests.api.routes.test_tenancy_connections_rls import (  # noqa: F401 — fixtures
    _enforce,
    restricted_factory,
)

pytestmark = pytest.mark.asyncio


async def _bind(session, tenant: UUID) -> None:
    await session.execute(
        text("SELECT set_config('app.current_tenant', :t, true)"), {"t": str(tenant)}
    )


async def _two_tenants_with_keys(owner_factory) -> tuple[UUID, UUID, UUID, str, UUID, str]:
    """Tenants A and B, one key each, minted as the OWNER (RLS-inert)."""
    a, b = uuid7(), uuid7()
    async with owner_factory() as owner:
        for tid in (a, b):
            await owner.execute(insert(Tenant).values(id=tid, name=f"t-{tid}", status="active"))
        repo = ApiKeysRepository(owner)
        row_a, key_a = await repo.create(tenant_id=a, name="a-ci", created_by=None)
        row_b, key_b = await repo.create(tenant_id=b, name="b-ci", created_by=None)
        await owner.commit()
        return a, b, row_a.id, key_a, row_b.id, key_b


async def test_policy_isolates_keys_even_when_asked_by_the_other_tenant_id(
    db_session_factory,
    restricted_factory,  # noqa: F811 — the imported fixture
) -> None:
    a, b, _, _, key_b_id, _ = await _two_tenants_with_keys(db_session_factory)

    async with restricted_factory() as s:
        await _bind(s, a)
        assert (await s.execute(select(TenantApiKey.tenant_id))).scalars().all() == [a], (
            "an unfiltered read bound to A sees only A's key"
        )
        repo = ApiKeysRepository(s)
        assert await repo.list_for_tenant(b, include_revoked=True) == [], (
            "bound to A, B's keys are invisible even when asked for BY B's tenant_id"
        )
        assert not await repo.revoke(key_b_id, b), "A cannot revoke B's key"
        assert await repo.live_scopes(key_b_id, b) is None
        await s.commit()

    async with restricted_factory() as s:  # control: bound to B, B's key is there
        await _bind(s, b)
        assert [k.id for k in await ApiKeysRepository(s).list_for_tenant(b)] == [key_b_id]
        await s.rollback()

    async with restricted_factory() as s:
        await _bind(s, a)
        with pytest.raises(DBAPIError, match="row-level security"):
            await ApiKeysRepository(s).create(tenant_id=b, name="smuggled", created_by=None)
        await s.rollback()

    async with db_session_factory() as owner:
        revoked = (
            await owner.execute(select(TenantApiKey.revoked_at).where(TenantApiKey.id == key_b_id))
        ).scalar_one()
    assert revoked is None, "B's key survived A's revoke attempt"


async def test_key_auth_by_hash_works_with_no_tenant_bound(
    db_session_factory,
    restricted_factory,  # noqa: F811
) -> None:
    _, b, _, _, key_b_id, key_b = await _two_tenants_with_keys(db_session_factory)
    prefix = key_b.split("_")[2]

    async with restricted_factory() as s:
        # Control: the same posture, no tenant → a plain SELECT finds nothing, so
        # authentication would 401 every key without the resolver.
        plain = await s.execute(select(TenantApiKey.id).where(TenantApiKey.prefix == prefix))
        assert plain.first() is None, "control: the table really is RLS-gated here"

        repo = ApiKeysRepository(s)
        resolved = await repo.resolve(key_b)
        assert resolved is not None
        assert (resolved.id, resolved.tenant_id, resolved.scopes) == (key_b_id, b, [])
        # Holding the prefix without the secret matches nothing.
        assert await repo.resolve(f"w2v_sk_{prefix}_{'A' * 43}") is None
        await s.rollback()


async def test_resolve_refuses_revoked_and_expired_keys_without_a_tenant(
    db_session_factory,
    restricted_factory,  # noqa: F811
) -> None:
    _, b, _, _, key_b_id, key_b = await _two_tenants_with_keys(db_session_factory)
    async with db_session_factory() as owner:
        await ApiKeysRepository(owner).revoke(key_b_id, b)
        await owner.commit()
    async with restricted_factory() as s:
        assert await ApiKeysRepository(s).resolve(key_b) is None
        await s.rollback()


async def test_mint_rerolls_a_prefix_held_by_another_tenant(
    db_session_factory,
    restricted_factory,  # noqa: F811
    monkeypatch,
) -> None:
    a, _, _, _, _, key_b = await _two_tenants_with_keys(db_session_factory)
    b_prefix = key_b.split("_")[2]
    real = api_key_format.generate_api_key
    calls: list[str] = []

    def _collide_once() -> api_key_format.GeneratedApiKey:
        if not calls:
            calls.append(b_prefix)
            return api_key_format.GeneratedApiKey(
                full_key=f"w2v_sk_{b_prefix}_{'B' * 43}",
                prefix=b_prefix,
                key_hash=api_key_format.hash_api_key(f"w2v_sk_{b_prefix}_{'B' * 43}"),
            )
        fresh = real()
        calls.append(fresh.prefix)
        return fresh

    monkeypatch.setattr(
        "src.database.repositories.api_keys_repository.generate_api_key", _collide_once
    )
    async with restricted_factory() as s:
        await _bind(s, a)
        row, _ = await ApiKeysRepository(s).create(tenant_id=a, name="a-2", created_by=None)
        await s.commit()
    assert calls[0] == b_prefix and len(calls) == 2, "B's prefix was seen across RLS"
    assert row.prefix == calls[1]


async def test_verifier_resolves_and_bumps_last_used_under_enforce(
    db_session_factory,
    restricted_factory,  # noqa: F811
) -> None:
    _, b, _, _, key_b_id, key_b = await _two_tenants_with_keys(db_session_factory)
    verdict = await make_api_key_verifier(restricted_factory)(key_b)
    assert verdict == (b, str(key_b_id))
    async with db_session_factory() as owner:
        used = (
            await owner.execute(
                select(TenantApiKey.last_used_at).where(TenantApiKey.id == key_b_id)
            )
        ).scalar_one()
    assert used is not None, "the bump ran bound to the key's tenant (else UPDATE 0)"


def _app(factory, *, claim: UUID) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = factory
    app.state.redis = fakeredis.aioredis.FakeRedis()
    app.include_router(api_keys.router)
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


async def test_routes_bind_the_effective_tenant(
    db_session_factory,
    restricted_factory,  # noqa: F811
) -> None:
    a, b, key_a_id, _, key_b_id, _ = await _two_tenants_with_keys(db_session_factory)
    app = _app(restricted_factory, claim=a)
    switch = {"X-Active-Tenant-Id": str(b)}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/api/tenant/api-keys")
        assert r.status_code == 200, r.text
        assert [i["id"] for i in r.json()["items"]] == [str(key_a_id)]

        r = await c.delete(f"/api/tenant/api-keys/{key_b_id}")
        assert r.status_code == 404, "A's admin cannot revoke B's key"

        # Scope-switched into B: the route binds B (not the claim A) → B's key.
        r = await c.get("/api/tenant/api-keys", headers=switch)
        assert r.status_code == 200, r.text
        assert [i["id"] for i in r.json()["items"]] == [str(key_b_id)]

        r = await c.post("/api/tenant/api-keys", json={"name": "b-2"}, headers=switch)
        assert r.status_code == 201, r.text
        assert r.json()["key"].startswith("w2v_sk_")

        r = await c.delete(f"/api/tenant/api-keys/{key_b_id}", headers=switch)
        assert r.status_code == 204, r.text

    async with db_session_factory() as owner:
        rows = (
            await owner.execute(
                select(TenantApiKey.tenant_id, TenantApiKey.name, TenantApiKey.revoked_at)
                .where(TenantApiKey.tenant_id.in_([a, b]))
                .order_by(TenantApiKey.name)
            )
        ).all()
    assert [(r.tenant_id, r.name, r.revoked_at is not None) for r in rows] == [
        (a, "a-ci", False),
        (b, "b-2", False),
        (b, "b-ci", True),
    ]
