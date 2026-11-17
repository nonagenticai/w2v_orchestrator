"""P7 C2: the orchestrator's in-cell calls carry a cell token naming the RIGHT tenant.

Covers the two call paths that bypass the cached ``BaseServiceClient`` identity:

* ``routes/_proxy.py`` (critique → analysis-service, redesign → html-render-service)
  used raw ``httpx.AsyncClient``s with no event hook, so no proxied request ever
  carried ``X-W2V-Cell-Token``. Its tenant must be ``TenantIdDep`` (incl. a
  claim-mode ``super_admin`` scope switch), its actor the human ``sub``.
* ``YouTubeUploadClient`` acts for an EXPLICIT owner tenant; the hook minted the
  ambient one. A mismatch would make youtube_upload read the wrong tenant once
  it trusts the token. Operator (``/api/admin``) calls act for no tenant and get
  a ``scope=service`` token.

Tokens are verified for real (Ed25519 key + a JWKS file, as in w2v_common's own
tests), never decoded unverified.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from jwt.algorithms import OKPAlgorithm
from w2v_common.auth import cell_jwt
from w2v_common.auth.cell_jwt import (
    CELL_TOKEN_HEADER,
    CellClaims,
    CellScope,
    CellTokenError,
    reset_cell_state,
    verify_cell_token,
)
from w2v_common.auth.identity_context import (
    clear_identity_context,
    get_tenant_id_from_context,
    set_identity_context,
)
from w2v_common.discovery import service_url
from w2v_common.tenant import TenantSource

import src.api.routes._proxy as proxy_mod
from src.api.routes._proxy import proxy_get
from src.api.routes.critique import router as critique_router
from src.api.routes.redesign import router as redesign_router
from src.clients.youtube_upload_client import YouTubeUploadClient

pytestmark = pytest.mark.asyncio

ORCH = "w2v-orchestrator"
DEPLOYMENT = "strato-w2v"
TENANT = uuid.UUID("0198f1d2-0000-7000-8000-0000000c3111")
OTHER_TENANT = uuid.UUID("0198f1d2-0000-7000-8000-0000000c3222")
ACTOR = "0198f1d2-0000-7000-8000-0000000ac701"

_REAL_ASYNC_CLIENT = httpx.AsyncClient


@pytest.fixture(scope="module")
def orch_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


@pytest.fixture(autouse=True)
def cell_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, orch_key: Ed25519PrivateKey
) -> Iterator[None]:
    """This process signs as the orchestrator; verification switches the service name."""
    jwk = json.loads(OKPAlgorithm.to_jwk(orch_key.public_key()))
    jwk.update({"kid": "orch-1", "iss": ORCH})
    jwks = tmp_path / "jwks.json"
    jwks.write_text(json.dumps({"keys": [jwk]}))
    pem = orch_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    monkeypatch.setenv("CELL_JWT_SIGNING_KEY", pem)
    monkeypatch.setenv("CELL_JWT_KID", "orch-1")
    monkeypatch.setenv("CELL_SERVICE_NAME", ORCH)
    monkeypatch.setenv("CELL_DEPLOYMENT_ID", DEPLOYMENT)
    monkeypatch.setenv("CELL_JWKS_PATH", str(jwks))
    monkeypatch.setenv("CELL_ALLOWED_CALLERS", "*")
    reset_cell_state()
    clear_identity_context()
    yield
    reset_cell_state()
    clear_identity_context()


def _verify(
    monkeypatch: pytest.MonkeyPatch,
    token: str,
    callee: str,
    *,
    now: float | None = None,
    service_callers: frozenset[str] | None = None,
) -> CellClaims:
    """Verify ``token`` AS ``callee`` would (aud, cell, kid→iss, exp)."""
    with monkeypatch.context() as m:
        m.setenv("CELL_SERVICE_NAME", callee)
        return verify_cell_token(token, now=now, service_callers=service_callers)


def _capture_outbound(monkeypatch: pytest.MonkeyPatch, module: Any) -> list[httpx.Request]:
    """Route every ``httpx.AsyncClient`` built via ``module.httpx`` to a recorder.

    Unlike the critique tests' stub this keeps every constructor kwarg —
    ``event_hooks`` in particular — so the production hooks really run.
    """
    seen: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    def _factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(_handler)
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", _factory)
    return seen


def _seeded_app(*, claim: uuid.UUID, roles: list[str]) -> FastAPI:
    """critique + redesign routers behind a pure-ASGI seeder of a JWT-lane identity."""

    class _Seed:
        def __init__(self, app: Any) -> None:
            self.app = app

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            if scope["type"] != "http":
                await self.app(scope, receive, send)
                return
            state = scope.setdefault("state", {})
            state.update({"tenant_id": str(claim), "subject_id": ACTOR, "roles": roles})
            set_identity_context(
                claim, None, ACTOR, tenant_source=TenantSource.JWT_CLAIM, actor_id=ACTOR
            )
            try:
                await self.app(scope, receive, send)
            finally:
                clear_identity_context()

    app = FastAPI()
    app.add_middleware(_Seed)
    app.include_router(critique_router)
    app.include_router(redesign_router)
    return app


# --------------------------------------------------------------------------- #
# routes/_proxy.py
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("path", "callee", "host"),
    [("/api/critique/job-1", "analysis-service", "analysis")],
)
async def test_proxied_request_carries_a_cell_token_for_the_request_tenant(
    monkeypatch: pytest.MonkeyPatch, path: str, callee: str, host: str
) -> None:
    seen = _capture_outbound(monkeypatch, proxy_mod)
    app = _seeded_app(claim=TENANT, roles=["member"])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get(path)
    assert resp.status_code == 200, resp.text
    (outbound,) = seen
    assert str(outbound.url).startswith(service_url(host))
    claims = _verify(monkeypatch, outbound.headers[CELL_TOKEN_HEADER], callee)
    assert claims.tenant_id == TENANT
    assert claims.actor == ACTOR
    assert claims.audience == callee
    assert claims.issuer == ORCH
    # Dual-send: the licence lane's tenant side-channel is still there.
    assert outbound.headers["X-Effective-Tenant-Id"] == str(TENANT)


async def test_super_admin_scope_switch_is_the_cell_token_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TenantIdDep (the switched tenant), not the ambient claim, is what gets signed."""
    seen = _capture_outbound(monkeypatch, proxy_mod)
    app = _seeded_app(claim=TENANT, roles=["super_admin"])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/api/critique/job-1", headers={"X-Active-Tenant-Id": str(OTHER_TENANT)})
    assert resp.status_code == 200, resp.text
    (outbound,) = seen
    claims = _verify(monkeypatch, outbound.headers[CELL_TOKEN_HEADER], "analysis-service")
    assert claims.tenant_id == OTHER_TENANT
    assert claims.actor == ACTOR
    assert outbound.headers["X-Effective-Tenant-Id"] == str(OTHER_TENANT)


async def test_rebinding_for_one_send_does_not_leak_into_the_request_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_outbound(monkeypatch, proxy_mod)
    set_identity_context(TENANT, None, ACTOR, tenant_source=TenantSource.JWT_CLAIM, actor_id=ACTOR)
    await proxy_get("/api/critique/x", effective_tenant=OTHER_TENANT)
    assert get_tenant_id_from_context() == TENANT


async def test_a_poll_past_the_ttl_mints_a_fresh_token_per_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No token is held: the second poll, 120 s (> 60 s TTL) later, is signed anew."""
    seen = _capture_outbound(monkeypatch, proxy_mod)
    clock = [1_900_000_000.0]
    monkeypatch.setattr(cell_jwt, "time", SimpleNamespace(time=lambda: clock[0]))
    set_identity_context(TENANT, None, ACTOR, tenant_source=TenantSource.JWT_CLAIM, actor_id=ACTOR)

    await proxy_get("/api/critique/job-1", effective_tenant=TENANT)
    clock[0] += 120
    await proxy_get("/api/critique/job-1", effective_tenant=TENANT)

    first, second = (r.headers[CELL_TOKEN_HEADER] for r in seen)
    t0 = 1_900_000_000.0
    # The first token is dead by the second poll ...
    with pytest.raises(CellTokenError):
        _verify(monkeypatch, first, "analysis-service", now=t0 + 120)
    # ... and the second one, minted then, is valid then.
    late = _verify(monkeypatch, second, "analysis-service", now=t0 + 120)
    early = _verify(monkeypatch, first, "analysis-service", now=t0 + 1)
    assert late.issued_at - early.issued_at == 120
    assert late.jti != early.jti
    assert late.tenant_id == TENANT


# --------------------------------------------------------------------------- #
# YouTubeUploadClient
# --------------------------------------------------------------------------- #


@pytest.fixture
def yt(monkeypatch: pytest.MonkeyPatch) -> tuple[YouTubeUploadClient, list[httpx.Request]]:
    import w2v_common.clients.base as base_mod

    seen = _capture_outbound(monkeypatch, base_mod)
    client = YouTubeUploadClient(license_token="sa-token")  # pragma: allowlist secret
    assert client.base_url.startswith(service_url("youtube_upload"))
    return client, seen


@pytest.mark.parametrize(
    "call",
    [lambda c: c.get_upload_job("job-1", owner_tenant_id=str(OTHER_TENANT))],
    ids=["get_upload_job"],
)
async def test_yt_cell_token_tenant_is_the_explicit_owner_not_the_ambient_tenant(
    monkeypatch: pytest.MonkeyPatch,
    yt: tuple[YouTubeUploadClient, list[httpx.Request]],
    call: Any,
) -> None:
    client, seen = yt
    set_identity_context(TENANT, None, ACTOR, tenant_source=TenantSource.MEMBERSHIP, actor_id=ACTOR)
    await call(client)
    (outbound,) = seen
    claims = _verify(monkeypatch, outbound.headers[CELL_TOKEN_HEADER], "youtube-upload-service")
    assert claims.tenant_id == OTHER_TENANT
    assert claims.actor == ACTOR
    # Dual-send: the legacy owner header names the same tenant.
    assert outbound.headers["X-Owner-Tenant-Id"] == str(OTHER_TENANT)
    # And the request's own identity is untouched.
    assert get_tenant_id_from_context() == TENANT


async def test_yt_operator_lane_sends_a_service_scope_token_with_no_tenant(
    monkeypatch: pytest.MonkeyPatch,
    yt: tuple[YouTubeUploadClient, list[httpx.Request]],
) -> None:
    """``/api/admin/*`` acts for NO tenant: the operator's own tenant must not be signed."""
    client, seen = yt
    set_identity_context(TENANT, None, ACTOR, tenant_source=TenantSource.MEMBERSHIP, actor_id=ACTOR)
    await client.operator_call("GET", "/platform-channel")
    (outbound,) = seen
    assert "X-Owner-Tenant-Id" not in outbound.headers
    token = outbound.headers[CELL_TOKEN_HEADER]
    # A receiver route that did not opt in refuses it ...
    with pytest.raises(CellTokenError):
        _verify(monkeypatch, token, "youtube-upload-service")
    # ... one that admits the orchestrator as a service caller accepts it.
    claims = _verify(
        monkeypatch, token, "youtube-upload-service", service_callers=frozenset({ORCH})
    )
    assert claims.scope is CellScope.SERVICE
    assert claims.tenant_id is None
    assert claims.actor is None
