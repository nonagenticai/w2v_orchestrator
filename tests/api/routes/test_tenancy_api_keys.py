"""P2 WP4b — tenant API keys on the REAL request path.

Same production stack as ``test_membership_resolution.py``: a real RS256 JWT
verified by w2v_common's ``AuthMiddleware`` (``create_app``), the orchestrator's
``TenancyMiddleware`` in membership mode, the registered ``w2v_sk_`` verifier,
and real ``tenant_api_keys`` / ``tenant_members`` / ``admin_audit_log`` rows.
Only the JWKS socket is fake.

Non-vacuity: the synthetic-subject test reads ``SubjectIdDep`` — which 401s on
w2v_common's ``api-key:<id>`` subject unless the orchestrator maps it — and
asserts the exact ``uuid5`` value; the revoke test uses the SAME key before and
after the DELETE.
"""

from __future__ import annotations

import time
import types
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import APIRouter, FastAPI, Request
from jwt.algorithms import RSAAlgorithm
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from w2v_common.asgi import create_app
from w2v_common.auth import keycloak as keycloak_mod
from w2v_common.auth import register_api_key_verifier
from w2v_common.auth.identity_context import (
    get_actor_id_from_context,
    get_subject_id_from_context,
)

from src.api.routes._common import SubjectIdDep, TenantIdDep
from src.api.routes.tenancy import api_keys
from src.auth.tenancy_middleware import (
    W2V_API_KEY_NS,
    install_tenancy_middleware,
    make_api_key_verifier,
)
from src.config import clear_settings_cache, get_settings
from src.database.models import AdminAuditLog, Tenant, TenantApiKey
from src.database.repositories.members_repository import MembersRepository

pytestmark = pytest.mark.asyncio

KEYCLOAK_URL = "https://keycloak.test"
REALM = "portal"
ISSUER = f"{KEYCLOAK_URL}/realms/{REALM}"
JWKS_URI = f"{ISSUER}/protocol/openid-connect/certs"
AUDIENCE = "w2v-frontend"
KID = "wp4b-kid"
_PUBLIC_PEER = ("203.0.113.9", 50000)


@pytest.fixture(scope="module")
def signing_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@dataclass
class Harness:
    app: FastAPI
    key: rsa.RSAPrivateKey
    factory: async_sessionmaker[AsyncSession]

    def jwt(self, sub: uuid.UUID) -> str:
        now = int(time.time())
        claims = {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": str(sub),
            "iat": now,
            "exp": now + 300,
            "email": "ada@acme.test",
            "email_verified": True,
            "preferred_username": "ada",
            "realm_access": {"roles": []},
        }
        return jwt.encode(claims, self.key, algorithm="RS256", headers={"kid": KID})

    async def call(self, method: str, path: str, bearer: str, json: Any = None) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app, client=_PUBLIC_PEER)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.request(
                method, path, json=json, headers={"Authorization": f"Bearer {bearer}"}
            )

    async def tenant_with(self, role: str) -> tuple[uuid.UUID, uuid.UUID]:
        tid, sub = uuid.uuid7(), uuid.uuid4()
        async with self.factory() as s:
            await s.execute(insert(Tenant).values(id=tid, name=f"t-{tid}", status="active"))
            await MembersRepository(s).add_member(subject_id=sub, tenant_id=tid, role=role)  # ty: ignore[invalid-argument-type]  # why: test passes a valid literal as plain str
            await s.commit()
        return tid, sub

    async def audit_actions(self, tid: uuid.UUID) -> list[tuple[str, uuid.UUID | None]]:
        async with self.factory() as s:
            rows = await s.execute(
                select(AdminAuditLog.action, AdminAuditLog.actor_subject_id)
                .where(AdminAuditLog.target_tenant_id == tid)
                .order_by(AdminAuditLog.created_at)
            )
            return [(r.action, r.actor_subject_id) for r in rows]


@pytest.fixture
def harness(monkeypatch, signing_key, db_session_factory) -> Iterator[Harness]:
    monkeypatch.setenv("KEYCLOAK_URL", KEYCLOAK_URL)
    monkeypatch.setenv("KEYCLOAK_REALM", REALM)
    monkeypatch.setenv("KEYCLOAK_AUDIENCES", AUDIENCE)
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "false")
    monkeypatch.setenv("W2V_TENANCY_SOURCE", "membership")
    monkeypatch.setenv("AUTH_JWT_TENANT_REQUIRED", "false")
    for name in ("KEYCLOAK_ISSUER", "KEYCLOAK_JWT_LEEWAY", "KEYCLOAK_JWKS_TTL"):
        monkeypatch.delenv(name, raising=False)
    clear_settings_cache()

    jwk = RSAAlgorithm.to_jwk(signing_key.public_key(), as_dict=True)
    jwk.update({"kid": KID, "alg": "RS256", "use": "sig"})
    real_async_client = httpx.AsyncClient

    def _serve_jwks(request: httpx.Request) -> httpx.Response:
        if str(request.url) != JWKS_URI:
            return httpx.Response(404)
        return httpx.Response(200, json={"keys": [jwk]})

    def _mock_async_client(**kwargs: Any) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(_serve_jwks), **kwargs)

    monkeypatch.setattr(
        keycloak_mod,
        "httpx",
        types.SimpleNamespace(AsyncClient=_mock_async_client, HTTPError=httpx.HTTPError),
    )
    keycloak_mod.reset_jwt_validator()
    register_api_key_verifier(make_api_key_verifier(db_session_factory))

    probe = APIRouter(prefix="/api/probe")

    @probe.get("")
    async def whoami(
        request: Request, tenant_id: TenantIdDep, subject: SubjectIdDep
    ) -> dict[str, object]:
        return {
            "tenant": str(tenant_id),
            "subject": str(subject),
            "context_subject": get_subject_id_from_context(),
            "actor": get_actor_id_from_context(),
            "actor_type": getattr(request.state, "actor_type", None),
            "roles": list(request.state.roles),
        }

    app = create_app(get_settings(), routers=[api_keys.router, probe], cors_enabled=False)
    install_tenancy_middleware(app)
    app.state.session_factory = db_session_factory
    yield Harness(app=app, key=signing_key, factory=db_session_factory)

    register_api_key_verifier(None)
    keycloak_mod.reset_jwt_validator()
    clear_settings_cache()


async def test_admin_creates_lists_and_secret_is_shown_once(harness: Harness) -> None:
    tid, admin = await harness.tenant_with("tenant_admin")
    token = harness.jwt(admin)

    created = await harness.call("POST", "/api/tenant/api-keys", token, {"name": "ci"})
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["key"].startswith(f"w2v_sk_{body['prefix']}_")
    assert body["created_by"] == str(admin)
    assert body["created_via"] == "tenant_admin"

    listed = await harness.call("GET", "/api/tenant/api-keys", token)
    assert listed.status_code == 200, listed.text
    [item] = listed.json()["items"]
    assert item["id"] == body["id"] and item["prefix"] == body["prefix"]
    assert "key" not in item and "key_hash" not in item
    assert body["key"] not in listed.text
    assert await harness.audit_actions(tid) == [("apikey.create", admin)]


async def test_key_acts_as_synthetic_member_subject(harness: Harness) -> None:
    tid, admin = await harness.tenant_with("tenant_admin")
    created = await harness.call("POST", "/api/tenant/api-keys", harness.jwt(admin), {"name": "k"})
    key_id, secret = created.json()["id"], created.json()["key"]
    expected = str(uuid.uuid5(W2V_API_KEY_NS, key_id))

    r = await harness.call("GET", "/api/probe", secret)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["tenant"] == str(tid)
    assert out["subject"] == expected
    assert out["context_subject"] == expected and out["actor"] == expected
    assert out["subject"] != str(admin), "a key never acts as its creator"
    assert out["actor_type"] == "api_key"
    assert out["roles"] == ["member"]
    # A key is a member: it cannot manage keys (or mint more of itself).
    assert (await harness.call("GET", "/api/tenant/api-keys", secret)).status_code == 403
    assert (
        await harness.call("POST", "/api/tenant/api-keys", secret, {"name": "x"})
    ).status_code == 403


async def test_revoked_key_is_401_and_revoke_is_tenant_scoped(harness: Harness) -> None:
    tid, admin = await harness.tenant_with("tenant_admin")
    other_tid, other_admin = await harness.tenant_with("tenant_admin")
    token = harness.jwt(admin)
    created = await harness.call("POST", "/api/tenant/api-keys", token, {"name": "k"})
    key_id, secret = created.json()["id"], created.json()["key"]
    assert (await harness.call("GET", "/api/probe", secret)).status_code == 200

    # Another tenant's admin cannot revoke it (404, key still works).
    foreign = await harness.call(
        "DELETE", f"/api/tenant/api-keys/{key_id}", harness.jwt(other_admin)
    )
    assert foreign.status_code == 404, foreign.text
    assert (await harness.call("GET", "/api/probe", secret)).status_code == 200

    revoked = await harness.call("DELETE", f"/api/tenant/api-keys/{key_id}", token)
    assert revoked.status_code == 204, revoked.text
    after = await harness.call("GET", "/api/probe", secret)
    assert after.status_code == 401, after.text
    assert after.json()["error"]["code"] == "INVALID_API_KEY"
    assert (await harness.call("GET", "/api/tenant/api-keys", token)).json()["items"] == []
    again = await harness.call("DELETE", f"/api/tenant/api-keys/{key_id}", token)
    assert again.status_code == 404
    assert await harness.audit_actions(tid) == [
        ("apikey.create", admin),
        ("apikey.revoke", admin),
    ]
    assert await harness.audit_actions(other_tid) == []
    async with harness.factory() as s:
        row = (
            await s.execute(select(TenantApiKey).where(TenantApiKey.id == uuid.UUID(key_id)))
        ).scalar_one()
    assert row.revoked_at is not None


async def test_plain_member_cannot_create_keys(harness: Harness) -> None:
    _tid, member = await harness.tenant_with("member")
    r = await harness.call("POST", "/api/tenant/api-keys", harness.jwt(member), {"name": "k"})
    assert r.status_code == 403, r.text


async def test_scopes_are_stored_listed_and_closed(harness: Harness) -> None:
    """P8: ``scopes`` round-trips through create + list; default is none; an
    unknown scope is 422 (the API Literal AND the DB CHECK close the vocabulary)."""
    tid, admin = await harness.tenant_with("tenant_admin")
    token = harness.jwt(admin)
    plain = await harness.call("POST", "/api/tenant/api-keys", token, {"name": "plain"})
    assert plain.status_code == 201, plain.text
    assert plain.json()["scopes"] == []
    scoped = await harness.call(
        "POST",
        "/api/tenant/api-keys",
        token,
        {"name": "merqi", "scopes": ["connections:write", "connections:write"]},
    )
    assert scoped.status_code == 201, scoped.text
    assert scoped.json()["scopes"] == ["connections:write"]
    listed = {
        i["name"]: i["scopes"]
        for i in (await harness.call("GET", "/api/tenant/api-keys", token)).json()["items"]
    }
    assert listed == {"plain": [], "merqi": ["connections:write"]}
    bad = await harness.call(
        "POST", "/api/tenant/api-keys", token, {"name": "x", "scopes": ["connections:admin"]}
    )
    assert bad.status_code == 422, bad.text
    async with harness.factory() as s:
        audit = (
            (
                await s.execute(
                    select(AdminAuditLog.detail).where(
                        AdminAuditLog.target_tenant_id == tid,
                        AdminAuditLog.action == "apikey.create",
                    )
                )
            )
            .scalars()
            .all()
        )
    scopes = [d["scopes"] for d in audit]
    assert all(isinstance(x, list) for x in scopes)
    assert sorted(map(str, scopes)) == ["['connections:write']", "[]"]


async def test_db_check_refuses_an_unknown_scope(harness: Harness) -> None:
    from sqlalchemy.exc import IntegrityError

    from src.database.repositories.api_keys_repository import ApiKeysRepository

    tid, _admin = await harness.tenant_with("tenant_admin")
    async with harness.factory() as s:
        with pytest.raises(IntegrityError, match="ck_tenant_api_keys_scopes"):
            await ApiKeysRepository(s).create(
                tenant_id=tid, name="x", created_by=None, scopes=["root"]
            )
