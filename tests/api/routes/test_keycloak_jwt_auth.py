"""Keycloak RS256 JWT lane, end to end through the service's real auth stack.

Why this file exists: the orchestrator authenticates browser callers with a
Keycloak RS256 bearer, verified by w2v_common's ``KeycloakJWTValidator``
(``jwt.decode`` against a ``PyJWK`` built from the realm JWKS) inside the
``AuthMiddleware`` that ``w2v_common.asgi.create_app`` registers from
``settings.auth``. Every other auth test in this repo either seeds
``request.state`` directly or patches ``_introspect_license`` — none ever minted
and verified a real token. So a PyJWT bump (the CVE-2026-101918 lock move to
2.15.1) could break RS256 verification and the suite would stay green.

Hermetic by construction: a fresh RSA keypair is generated per module, the JWKS
is served by an ``httpx.MockTransport`` swapped in for the validator's OWN
JWKS fetch (the ``httpx`` name inside ``w2v_common.auth.keycloak``), and the
roster write behind ``/api/auth/me`` is a fake. Nothing patches ``jwt.decode``
or ``PyJWK`` — those are the code under test. No network, no database.
"""

from __future__ import annotations

import time
import types
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from jwt.algorithms import RSAAlgorithm
from w2v_common.asgi import create_app
from w2v_common.auth import keycloak as keycloak_mod
from w2v_common.auth.middleware import AuthMiddleware

from src.api.routes._common import get_db_session, get_members_repo
from src.api.routes.auth import router as auth_router
from src.config import clear_settings_cache, get_settings

pytestmark = pytest.mark.asyncio

KEYCLOAK_URL = "https://keycloak.test"
REALM = "portal"
ISSUER = f"{KEYCLOAK_URL}/realms/{REALM}"
JWKS_URI = f"{ISSUER}/protocol/openid-connect/certs"
AUDIENCE = "w2v-frontend"
KID = "test-kid-1"

TENANT_ID = str(uuid.uuid7())
SUBJECT_ID = str(uuid.uuid7())

# A public, non-private peer address. ASGITransport's default client host is
# 127.0.0.1, which is inside TRUSTED_NETWORKS — a no-token request from there
# would be decided by the trusted-network lane's env flag, not by "no token".
_PUBLIC_PEER = ("203.0.113.7", 50000)


def _new_private_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def signing_key() -> rsa.RSAPrivateKey:
    return _new_private_key()


@pytest.fixture(scope="module")
def foreign_key() -> rsa.RSAPrivateKey:
    """A key the realm never published — signs with the TRUSTED kid anyway."""
    return _new_private_key()


@pytest.fixture(scope="module")
def jwks(signing_key: rsa.RSAPrivateKey) -> dict[str, list[dict[str, object]]]:
    jwk = RSAAlgorithm.to_jwk(signing_key.public_key(), as_dict=True)
    jwk.update({"kid": KID, "alg": "RS256", "use": "sig"})
    return {"keys": [jwk]}


def _claims(**overrides: object) -> dict[str, object]:
    now = int(time.time())
    claims: dict[str, object] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": SUBJECT_ID,
        "iat": now,
        "exp": now + 300,
        "tenant_id": TENANT_ID,
        "email": "ada@acme.test",
        "preferred_username": "ada",
        "realm_access": {"roles": ["tenant_admin"]},
    }
    claims.update(overrides)
    return claims


def _mint(key: rsa.RSAPrivateKey, **overrides: object) -> str:
    return jwt.encode(_claims(**overrides), key, algorithm="RS256", headers={"kid": KID})


class _FakeMembers:
    """Stands in for MembersRepository so /me's roster upsert needs no DB."""

    def __init__(self) -> None:
        self.seen: list[dict[str, object]] = []

    async def record_seen(self, **kwargs: object) -> None:
        self.seen.append(kwargs)


class _FakeSession:
    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


class _Harness:
    def __init__(self, app: FastAPI, members: _FakeMembers, jwks_calls: list[str]) -> None:
        self.app = app
        self.members = members
        self.jwks_calls = jwks_calls

    async def get_me(self, headers: dict[str, str] | None = None) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app, client=_PUBLIC_PEER)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.get("/api/auth/me", headers=headers or {})


@pytest.fixture
def harness(monkeypatch, jwks) -> Iterator[_Harness]:
    # Keycloak lane configuration — the same env vars the deployment sets.
    monkeypatch.setenv("KEYCLOAK_URL", KEYCLOAK_URL)
    monkeypatch.setenv("KEYCLOAK_REALM", REALM)
    monkeypatch.setenv("KEYCLOAK_AUDIENCES", AUDIENCE)
    monkeypatch.delenv("KEYCLOAK_ISSUER", raising=False)
    monkeypatch.delenv("KEYCLOAK_JWT_LEEWAY", raising=False)
    monkeypatch.delenv("KEYCLOAK_JWKS_TTL", raising=False)
    monkeypatch.setenv("AUTH_ENABLED", "true")
    clear_settings_cache()

    # The ONE seam: the validator's own JWKS HTTP fetch. Rebinding the `httpx`
    # name inside w2v_common.auth.keycloak (rather than httpx.AsyncClient
    # globally) leaves the test client's own httpx untouched.
    jwks_calls: list[str] = []

    def _serve_jwks(request: httpx.Request) -> httpx.Response:
        jwks_calls.append(str(request.url))
        if str(request.url) != JWKS_URI:
            return httpx.Response(404)
        return httpx.Response(200, json=jwks)

    real_async_client = httpx.AsyncClient

    def _mock_async_client(**kwargs: Any) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(_serve_jwks), **kwargs)

    fake_httpx = types.SimpleNamespace(AsyncClient=_mock_async_client, HTTPError=httpx.HTTPError)
    monkeypatch.setattr(keycloak_mod, "httpx", fake_httpx)

    # Process-wide validator singleton: rebuild it from THIS test's env, and
    # never leak a validator wired to the mock into another test.
    keycloak_mod.reset_jwt_validator()

    # Built with the same factory + settings src/asgi.py uses, so auth comes
    # from settings.auth exactly as in production.
    app = create_app(get_settings(), routers=[auth_router], cors_enabled=False)
    members = _FakeMembers()

    async def _fake_session() -> AsyncIterator[_FakeSession]:
        yield _FakeSession()

    app.dependency_overrides[get_db_session] = _fake_session
    app.dependency_overrides[get_members_repo] = lambda: members

    yield _Harness(app, members, jwks_calls)

    keycloak_mod.reset_jwt_validator()
    clear_settings_cache()


def _error_code(response: httpx.Response) -> str:
    return response.json()["error"]["code"]


async def test_auth_middleware_is_wired_from_settings(harness: _Harness) -> None:
    """Guard the harness itself: without AuthMiddleware every test below is vacuous."""
    assert any(m.cls is AuthMiddleware for m in harness.app.user_middleware)


async def test_valid_rs256_token_authenticates(harness: _Harness, signing_key) -> None:
    token = _mint(signing_key)

    r = await harness.get_me({"Authorization": f"Bearer {token}"})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tenant_id"] == TENANT_ID
    assert body["subject_id"] == SUBJECT_ID
    assert body["email"] == "ada@acme.test"
    assert body["roles"] == ["tenant_admin"]
    # The key really came through the JWKS seam, from the configured realm.
    assert harness.jwks_calls == [JWKS_URI]
    # And the route actually ran with the verified identity.
    assert len(harness.members.seen) == 1
    assert str(harness.members.seen[0]["tenant_id"]) == TENANT_ID


async def test_token_signed_by_unpublished_key_is_rejected(harness: _Harness, foreign_key) -> None:
    """Same kid, wrong key: only signature verification can refuse this."""
    token = _mint(foreign_key)

    r = await harness.get_me({"Authorization": f"Bearer {token}"})

    assert r.status_code == 401
    assert _error_code(r) == "JWT_INVALID"
    assert harness.members.seen == []


async def test_tampered_payload_is_rejected(harness: _Harness, signing_key) -> None:
    """A validly signed token whose payload was swapped (tenant escalation)."""
    header, _payload, signature = _mint(signing_key).split(".")
    _h, forged_payload, _s = _mint(signing_key, tenant_id=str(uuid.uuid7())).split(".")
    tampered = f"{header}.{forged_payload}.{signature}"

    r = await harness.get_me({"Authorization": f"Bearer {tampered}"})

    assert r.status_code == 401
    assert _error_code(r) == "JWT_INVALID"
    assert harness.members.seen == []


async def test_expired_token_is_rejected(harness: _Harness, signing_key) -> None:
    # Well past the validator's default 10s clock-skew leeway.
    past = int(time.time()) - 3600
    token = _mint(signing_key, iat=past - 300, exp=past)

    r = await harness.get_me({"Authorization": f"Bearer {token}"})

    assert r.status_code == 401
    assert _error_code(r) == "JWT_INVALID"
    assert harness.members.seen == []


async def test_wrong_audience_is_rejected(harness: _Harness, signing_key) -> None:
    token = _mint(signing_key, aud="some-other-client")

    r = await harness.get_me({"Authorization": f"Bearer {token}"})

    assert r.status_code == 401
    assert _error_code(r) == "JWT_INVALID"
    assert harness.members.seen == []


async def test_missing_token_is_rejected(harness: _Harness) -> None:
    r = await harness.get_me()

    assert r.status_code == 401
    assert _error_code(r) == "AUTHENTICATION_REQUIRED"
    assert harness.members.seen == []
    # No credential means the JWT lane never ran.
    assert harness.jwks_calls == []
