"""Shared harness for the P2 WP4a tenancy route tests.

Every request goes through the production auth stack: a real RS256 token,
verified by w2v_common's ``AuthMiddleware`` (``create_app``), then
``TenancyMiddleware`` in ``membership`` mode, then the REAL tenancy routers
(``src.api.routes.tenancy.ROUTERS`` + ``/api/auth/me``) against a real Postgres.
The only fakes are the JWKS socket and provisioning's identity invite API, which
is faked at the HTTP boundary (``FakeProvisioning`` — the contract's behaviour,
in memory — behind an ``httpx.MockTransport`` handed to the real
``IdentityInviter``).
"""

from __future__ import annotations

import json
import time
import types
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
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

from src.api.routes._common import get_inviter
from src.api.routes.auth import router as auth_router
from src.api.routes.tenancy import ROUTERS
from src.api.routes.tenancy.errors import register_tenancy_exception_handlers
from src.auth.tenancy_middleware import install_tenancy_middleware
from src.clients.identity_inviter import IdentityInviter
from src.config import clear_settings_cache, get_settings
from src.database.models import AdminAuditLog, Tenant, TenantInvite, TenantMember
from src.database.repositories.members_repository import MembersRepository

KEYCLOAK_URL = "https://keycloak.test"
REALM = "portal"
ISSUER = f"{KEYCLOAK_URL}/realms/{REALM}"
JWKS_URI = f"{ISSUER}/protocol/openid-connect/certs"
AUDIENCE = "w2v-frontend"
KID = "wp4a-kid"
_PUBLIC_PEER = ("203.0.113.9", 50000)
PROVISIONING_URL = "https://provisioning.test"
INVITE_PATH = "/api/identity/invites"


class _Auth:
    async def token(self) -> str:
        return "sa-token"


@dataclass
class FakeProvisioning:
    """Provisioning's ``POST /api/identity/invites`` (the identity invite
    contract), in memory. Records every request and every action email."""

    users: dict[str, dict[str, Any]] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)
    #: (email, actions, redirect_uri, client_id) per action email sent.
    mails: list[tuple[str, list[str], str, str]] = field(default_factory=list)
    fail_actions_email: bool = False

    def add_user(
        self,
        email: str,
        *,
        verified: bool = True,
        has_password: bool = True,
        enabled: bool = True,
    ) -> str:
        uid = str(uuid.uuid4())
        self.users[uid] = {
            "id": uid,
            "email": email.strip().lower(),
            "emailVerified": verified,
            "has_password": has_password,
            "enabled": enabled,
        }
        return uid

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.url.path == INVITE_PATH]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method != "POST" or request.url.path != INVITE_PATH:
            return httpx.Response(404, json={"detail": "Not Found"})
        if request.headers.get("authorization") != "Bearer sa-token":
            return httpx.Response(401, json={"detail": "unauthorized"})
        body = json.loads(request.content)
        required = {"email", "client_id", "redirect_uri"}
        if not required <= set(body) <= required | {"actor", "context"}:
            return httpx.Response(422, json={"detail": "bad body"})
        email = body["email"].strip().lower()
        matches = [u for u in self.users.values() if u["email"] == email]
        actions: list[str] = []
        if not matches:
            self.add_user(email, verified=False, has_password=False)
            actions = ["UPDATE_PASSWORD", "VERIFY_EMAIL"]
        elif len(matches) == 1 and matches[0]["enabled"]:
            user = matches[0]
            actions = [] if user["has_password"] else ["UPDATE_PASSWORD"]
            if not user["emailVerified"]:
                actions.append("VERIFY_EMAIL")
        # (ambiguous or disabled: nothing is sent, and the answer is the same)
        if actions:
            if self.fail_actions_email:
                return httpx.Response(502, json={"detail": "IDP_UNAVAILABLE"})
            self.mails.append((email, actions, body["redirect_uri"], body["client_id"]))
        # The contract answers every accepted invite identically: no account state.
        return httpx.Response(202, json={"accepted": True})


@dataclass
class Harness:
    app: FastAPI
    key: rsa.RSAPrivateKey
    factory: async_sessionmaker[AsyncSession]
    idp: FakeProvisioning

    def token(
        self,
        sub: uuid.UUID,
        *,
        email: str = "someone@acme.test",
        verified: bool = True,
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
            "email_verified": verified,
            "preferred_username": email.split("@")[0],
            "realm_access": {"roles": roles or []},
        }
        return jwt.encode(claims, self.key, algorithm="RS256", headers={"kid": KID})

    async def call(
        self,
        method: str,
        path: str,
        bearer: str,
        *,
        json_body: object | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app, client=_PUBLIC_PEER)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.request(
                method,
                path,
                json=json_body,
                headers={"Authorization": f"Bearer {bearer}", **(headers or {})},
            )

    async def tenant(self, name: str | None = None) -> uuid.UUID:
        tid = uuid.uuid7()
        async with self.factory() as s:
            await s.execute(insert(Tenant).values(id=tid, name=name or f"t-{tid}"))
            await s.commit()
        return tid

    async def member(
        self, sub: uuid.UUID, tenant: uuid.UUID, role: str = "member", email: str | None = None
    ) -> None:
        async with self.factory() as s:
            await MembersRepository(s).add_member(
                subject_id=sub,
                tenant_id=tenant,
                role=role,  # ty: ignore[invalid-argument-type]  # why: tests pass the two valid literals as str
                email=email,
            )
            await s.commit()

    async def membership(self, sub: uuid.UUID, tenant: uuid.UUID) -> tuple[str, str] | None:
        async with self.factory() as s:
            row = (
                await s.execute(
                    select(TenantMember.role, TenantMember.status).where(
                        TenantMember.subject_id == sub, TenantMember.tenant_id == tenant
                    )
                )
            ).first()
            return (row.role, row.status) if row else None

    async def invites(self, tenant: uuid.UUID) -> list[TenantInvite]:
        async with self.factory() as s:
            return list(
                (await s.execute(select(TenantInvite).where(TenantInvite.tenant_id == tenant)))
                .scalars()
                .all()
            )

    async def audit_actions(self, tenant: uuid.UUID) -> list[str]:
        async with self.factory() as s:
            return list(
                (
                    await s.execute(
                        select(AdminAuditLog.action)
                        .where(AdminAuditLog.target_tenant_id == tenant)
                        .order_by(AdminAuditLog.created_at)
                    )
                )
                .scalars()
                .all()
            )


@pytest.fixture(scope="module", name="signing_key")
def signing_key_fixture() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(name="harness")
def harness_fixture(monkeypatch, signing_key, db_session_factory) -> Iterator[Harness]:
    monkeypatch.setenv("KEYCLOAK_URL", KEYCLOAK_URL)
    monkeypatch.setenv("KEYCLOAK_REALM", REALM)
    monkeypatch.setenv("KEYCLOAK_AUDIENCES", AUDIENCE)
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "false")
    monkeypatch.setenv("W2V_TENANCY_SOURCE", "membership")
    monkeypatch.setenv("AUTH_JWT_TENANT_REQUIRED", "false")
    monkeypatch.setenv("W2V_FRONTEND_URL", "https://w2v.example.test")
    monkeypatch.setenv("IDENTITY_INVITE_CLIENT_ID", "w2v-frontend")
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

    idp = FakeProvisioning()
    app = create_app(get_settings(), routers=[auth_router, *ROUTERS], cors_enabled=False)
    install_tenancy_middleware(app)
    register_tenancy_exception_handlers(app)
    app.state.session_factory = db_session_factory
    app.dependency_overrides[get_inviter] = lambda: IdentityInviter(
        authenticator=_Auth(),
        base_url=PROVISIONING_URL,
        client_id="w2v-frontend",
        default_redirect_uri="https://w2v.example.test/",
        client=real_async_client(transport=httpx.MockTransport(idp.handler)),
    )
    yield Harness(app=app, key=signing_key, factory=db_session_factory, idp=idp)

    keycloak_mod.reset_jwt_validator()
    clear_settings_cache()
