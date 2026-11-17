"""Claimless JWT under ``W2V_TENANCY_SOURCE=shadow`` — END TO END through the app.

Live defect (2026-10-15, strato, #1105 / eae1c5fc): a portal user with no
``tenant_id`` claim logged in, ``TenancyMiddleware`` logged "tenancy shadow: no
tenant claim — resolved by membership" and counted ``claim_missing`` — and then
``GET /api/auth/me`` answered 401 ``missing tenant_id claim``. #1105 tested only
the middleware; the route dependency (``get_optional_tenant_id``) still keyed
"no tenant is fine" on ``W2V_TENANCY_SOURCE == "membership"`` and so demanded the
claim in shadow.

Every request here goes through the production stack the way ``src/asgi.py``
builds it: a real RS256 token verified by w2v_common's ``AuthMiddleware``
(``create_app``), ``TenancyMiddleware`` installed by ``install_tenancy_middleware``
(which aligns ``AUTH_JWT_TENANT_REQUIRED`` from the mode, exactly as on boot),
the REAL ``/api/auth/me``, invite-acceptance and ``/api/projects`` routers —
the last behind the real ``TENANT_GATE`` — against a real Postgres. The only
fakes are the JWKS socket and the licences client (which a membership-resolved
request must never need).
"""

from __future__ import annotations

import datetime
import os
import time
import types
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from jwt.algorithms import RSAAlgorithm
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from w2v_common.asgi import create_app
from w2v_common.auth import keycloak as keycloak_mod

from src.api.routes._common import TENANT_GATE, get_licenses_client
from src.api.routes.auth import router as auth_router
from src.api.routes.projects import router as projects_router
from src.api.routes.tenancy import ROUTERS
from src.api.routes.tenancy.errors import register_tenancy_exception_handlers
from src.auth.tenancy_middleware import install_tenancy_middleware
from src.config import clear_settings_cache, get_settings
from src.database.models import Project, Tenant, TenantInvite, TenantMember
from src.database.repositories.members_repository import MembersRepository

pytestmark = pytest.mark.asyncio

KEYCLOAK_URL = "https://keycloak.test"
REALM = "portal"
ISSUER = f"{KEYCLOAK_URL}/realms/{REALM}"
JWKS_URI = f"{ISSUER}/protocol/openid-connect/certs"
AUDIENCE = "w2v-frontend"
KID = "shadow-claimless-kid"
# Outside TRUSTED_NETWORKS so no request is decided by the trusted-network lane.
_PUBLIC_PEER = ("203.0.113.11", 50000)


class _NoLicenceLookups:
    """A membership-resolved request is "known" without asking licences; a
    claim-carrying shadow request here hits a registered tenant first."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"licences client used ({name})")


@dataclass
class App:
    app: FastAPI
    key: rsa.RSAPrivateKey
    factory: async_sessionmaker[AsyncSession]

    def token(
        self,
        sub: uuid.UUID,
        *,
        email: str,
        claim: uuid.UUID | None = None,
        roles: list[str] | None = None,
    ) -> str:
        now = int(time.time())
        claims: dict[str, object] = {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": str(sub),
            "iat": now,
            "exp": now + 300,
            "email": email,
            "email_verified": True,
            "preferred_username": email.split("@")[0],
            "realm_access": {"roles": roles or []},
        }
        if claim is not None:
            claims["tenant_id"] = str(claim)
        return jwt.encode(claims, self.key, algorithm="RS256", headers={"kid": KID})

    async def call(
        self, method: str, path: str, bearer: str, headers: dict[str, str] | None = None
    ) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app, client=_PUBLIC_PEER)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.request(
                method, path, headers={"Authorization": f"Bearer {bearer}", **(headers or {})}
            )

    async def tenant(self) -> uuid.UUID:
        tid = uuid.uuid7()
        async with self.factory() as s:
            await s.execute(insert(Tenant).values(id=tid, name=f"t-{tid}"))
            await s.commit()
        return tid

    async def member(self, sub: uuid.UUID, tenant: uuid.UUID, role: str = "member") -> None:
        async with self.factory() as s:
            await MembersRepository(s).add_member(subject_id=sub, tenant_id=tenant, role=role)  # ty: ignore[invalid-argument-type]  # why: tests pass the valid literals as str
            await s.commit()

    async def invite(self, tenant: uuid.UUID, email: str, role: str = "member") -> uuid.UUID:
        iid = uuid.uuid7()
        async with self.factory() as s:
            s.add(
                TenantInvite(
                    id=iid,
                    tenant_id=tenant,
                    email=email,
                    role=role,
                    token_hash=os.urandom(32),
                    expires_at=datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=7),
                )
            )
            await s.commit()
        return iid

    async def project(self, tenant: uuid.UUID, title: str) -> uuid.UUID:
        pid = uuid.uuid7()
        async with self.factory() as s:
            s.add(
                Project(
                    id=pid,
                    tenant_id=tenant,
                    owner_subject_id=uuid.uuid7(),
                    visibility="tenant",
                    type="generic",
                    title=title,
                )
            )
            await s.commit()
        return pid

    async def rows(self, sub: uuid.UUID) -> dict[uuid.UUID, tuple[str, str]]:
        async with self.factory() as s:
            result = await s.execute(
                select(TenantMember.tenant_id, TenantMember.role, TenantMember.status).where(
                    TenantMember.subject_id == sub
                )
            )
            return {r.tenant_id: (r.role, r.status) for r in result}


@pytest.fixture(scope="module")
def signing_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def build(monkeypatch, signing_key, db_session_factory) -> Iterator[Callable[[str], App]]:
    """``build(mode)`` boots the app for ``W2V_TENANCY_SOURCE=mode`` the way the
    pod does: AUTH_JWT_TENANT_REQUIRED left at its deployed default (required)
    and aligned by ``install_tenancy_middleware`` itself."""
    monkeypatch.setenv("KEYCLOAK_URL", KEYCLOAK_URL)
    monkeypatch.setenv("KEYCLOAK_REALM", REALM)
    monkeypatch.setenv("KEYCLOAK_AUDIENCES", AUDIENCE)
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "false")
    # Recorded by monkeypatch so the alignment's direct os.environ write is undone.
    monkeypatch.setenv("AUTH_JWT_TENANT_REQUIRED", "true")
    for name in ("KEYCLOAK_ISSUER", "KEYCLOAK_JWT_LEEWAY", "KEYCLOAK_JWKS_TTL"):
        monkeypatch.delenv(name, raising=False)

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

    def _build(mode: str) -> App:
        monkeypatch.setenv("W2V_TENANCY_SOURCE", mode)
        clear_settings_cache()
        keycloak_mod.reset_jwt_validator()
        app = create_app(get_settings(), routers=[auth_router, *ROUTERS], cors_enabled=False)
        app.include_router(projects_router, dependencies=TENANT_GATE)
        install_tenancy_middleware(app)
        register_tenancy_exception_handlers(app)
        app.state.session_factory = db_session_factory
        app.dependency_overrides[get_licenses_client] = _NoLicenceLookups
        return App(app=app, key=signing_key, factory=db_session_factory)

    yield _build

    keycloak_mod.reset_jwt_validator()
    clear_settings_cache()


def _email() -> str:
    return f"invitee-{uuid.uuid4().hex[:10]}@acme.test"


# ─── (a) /api/auth/me ────────────────────────────────────────────────────


async def test_a_shadow_claimless_non_member_me_is_200_with_invites_and_no_tenant(
    build: Callable[[str], App],
) -> None:
    """The live defect: was 401 ``missing tenant_id claim``."""
    h = build("shadow")
    sub, email = uuid.uuid7(), _email()
    t = await h.tenant()
    invite_id = await h.invite(t, email)

    r = await h.call("GET", "/api/auth/me", h.token(sub, email=email))

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tenant_id"] is None
    assert body["subject_id"] == str(sub)
    assert body["memberships"] == []
    assert [i["id"] for i in body["pending_invites"]] == [str(invite_id)]
    assert body["pending_invites"][0]["tenant_id"] == str(t)
    assert await h.rows(sub) == {}, "/me never joins a claimless caller"


async def test_a_shadow_claimless_member_me_lists_the_membership_tenant(
    build: Callable[[str], App],
) -> None:
    h = build("shadow")
    sub, email = uuid.uuid7(), _email()
    t = await h.tenant()
    await h.member(sub, t, role="tenant_admin")

    r = await h.call("GET", "/api/auth/me", h.token(sub, email=email))

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tenant_id"] == str(t)
    assert body["role"] == "tenant_admin"
    assert [m["tenant_id"] for m in body["memberships"]] == [str(t)]


# ─── (b)+(c) accept the invite, then act in its tenant ───────────────────


async def test_bc_shadow_claimless_accepts_invite_then_lists_projects_of_that_tenant(
    build: Callable[[str], App],
) -> None:
    h = build("shadow")
    sub, email = uuid.uuid7(), _email()
    t, other = await h.tenant(), await h.tenant()
    invite_id = await h.invite(t, email, role="tenant_admin")
    mine = await h.project(t, "in the invited tenant")
    await h.project(other, "someone else's")
    tok = h.token(sub, email=email)

    # Before joining: no workspace, so tenant-scoped data is NO_MEMBERSHIP.
    before = await h.call("GET", "/api/projects", tok)
    assert before.status_code == 403, before.text
    assert before.json()["error"]["code"] == "NO_MEMBERSHIP"

    # (b) the SPA's flow: /me lists the invite, the caller accepts it by id.
    me = await h.call("GET", "/api/auth/me", tok)
    assert me.status_code == 200, me.text
    assert [i["id"] for i in me.json()["pending_invites"]] == [str(invite_id)]
    accepted = await h.call("POST", f"/api/invites/{invite_id}/accept", tok)
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["tenant_id"] == str(t)
    assert await h.rows(sub) == {t: ("tenant_admin", "active")}

    # (c) the same claimless token now acts in the membership tenant.
    projects = await h.call("GET", "/api/projects", tok)
    assert projects.status_code == 200, projects.text
    assert [p["id"] for p in projects.json()["items"]] == [str(mine)]

    me_after = await h.call("GET", "/api/auth/me", tok)
    assert me_after.status_code == 200, me_after.text
    assert me_after.json()["tenant_id"] == str(t)
    assert me_after.json()["pending_invites"] == []


# ─── (d) claim mode is unchanged ─────────────────────────────────────────


async def test_d_claim_mode_claimless_token_is_still_401(build: Callable[[str], App]) -> None:
    h = build("claim")
    sub, email = uuid.uuid7(), _email()
    t = await h.tenant()
    await h.member(sub, t)
    tok = h.token(sub, email=email)

    for path in ("/api/auth/me", "/api/projects"):
        r = await h.call("GET", path, tok)
        assert r.status_code == 401, (path, r.text)


# ─── (e) a claim-carrying token under shadow is unchanged ────────────────


async def test_e_shadow_token_with_claim_is_served_by_the_claim(
    build: Callable[[str], App],
) -> None:
    h = build("shadow")
    sub, email = uuid.uuid7(), _email()
    claim, membership = await h.tenant(), await h.tenant()
    await h.member(sub, membership)
    p_claim = await h.project(claim, "claim tenant")
    await h.project(membership, "membership tenant")
    tok = h.token(sub, email=email, claim=claim)

    me = await h.call("GET", "/api/auth/me", tok)
    projects = await h.call("GET", "/api/projects", tok)

    assert me.status_code == 200, me.text
    assert me.json()["tenant_id"] == str(claim), "shadow still SERVES by the claim"
    assert projects.status_code == 200, projects.text
    assert [p["id"] for p in projects.json()["items"]] == [str(p_claim)]
    # The pre-cutover roster fill for the caller's OWN claim tenant still happens.
    assert (await h.rows(sub)).get(claim) == ("member", "active")


# ─── /me roster write: a membership-resolved visit is not a membership ───


async def test_shadow_claimless_super_admin_scope_switch_me_creates_no_membership(
    build: Callable[[str], App],
) -> None:
    """A claimless super_admin's switch resolves via membership (super_admin
    path), so state.tenant_id IS the visited tenant — /me's "own claim tenant"
    test was true by construction and minted a membership for a visit."""
    h = build("shadow")
    sub, email = uuid.uuid7(), _email()
    home, visited = await h.tenant(), await h.tenant()
    await h.member(sub, home)
    tok = h.token(sub, email=email, roles=["super_admin"])

    me = await h.call("GET", "/api/auth/me", tok, {"X-Active-Tenant-Id": str(visited)})

    assert me.status_code == 200, me.text
    assert me.json()["tenant_id"] == str(visited)
    assert set(await h.rows(sub)) == {home}, "a scope-switch visit is not a membership"
