"""P5-3 — the orchestrator proxies youtube_upload's OAuth grant routes.

``/api/youtube/auth/{connect,status,disconnect}`` are served by the orchestrator
and forwarded on the service-account lane with ``X-Owner-Tenant-Id`` = the
membership-resolved tenant. Role gating is the orchestrator's (TenantAdminDep for
connect/disconnect, any member for status).

The REAL ``YouTubeUploadClient`` is used with an ``httpx.MockTransport`` standing
in for youtube_upload's socket, so the assertions are about the bytes on the wire
(path, method, ``X-Owner-Tenant-Id``), not about a mocked method signature.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes.youtube import router as youtube_router
from src.auth.membership import RESOLVED_BY_MEMBERSHIP
from src.clients.youtube_upload_client import YouTubeUploadClient

pytestmark = pytest.mark.asyncio

_UPSTREAM = "http://youtube-upload.test"


class _Upstream:
    """Records every request youtube_upload would have received; answers canned."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.reply: Callable[[httpx.Request], httpx.Response] = lambda req: httpx.Response(
            200, json={"ok": True, "path": req.url.path}
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.reply(request)


def _build_app(db_session_factory, upstream: _Upstream, *, tenant_id: str, roles: list[str]):
    app = FastAPI()
    app.state.session_factory = db_session_factory
    yt = YouTubeUploadClient(
        base_url=_UPSTREAM, license_token="sa-token"
    )  # pragma: allowlist secret
    yt.set_transport(httpx.MockTransport(upstream.handler))
    app.state.youtube_client = yt
    app.include_router(youtube_router)

    class _SeedMembership(BaseHTTPMiddleware):
        """What TenancyMiddleware leaves behind on a membership-resolved request."""

        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = tenant_id
            request.state.subject_id = str(uuid4())
            request.state.roles = roles
            request.state.tenant_resolved_by = RESOLVED_BY_MEMBERSHIP
            return await call_next(request)

    app.add_middleware(_SeedMembership)
    return app


@pytest.fixture
def upstream() -> _Upstream:
    return _Upstream()


@pytest.fixture
def make_client(db_session_factory, upstream):
    def _make(*, roles: list[str], tenant_id: str | None = None) -> tuple[AsyncClient, str]:
        tid = tenant_id or str(uuid4())
        app = _build_app(db_session_factory, upstream, tenant_id=tid, roles=roles)
        c = AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")
        return c, tid

    return _make


# --------------------------------------------------------------------------- gating


async def test_member_can_read_status_scoped_to_their_tenant(make_client, upstream) -> None:
    client, tenant = make_client(roles=["member"])
    upstream.reply = lambda req: httpx.Response(
        200, json={"connected": False, "status": "not_connected", "tenant_id": tenant}
    )
    async with client:
        r = await client.get("/api/youtube/auth/status")
    assert r.status_code == 200, r.text
    assert r.json() == {"connected": False, "status": "not_connected", "tenant_id": tenant}
    (sent,) = upstream.requests
    assert sent.method == "GET"
    assert sent.url.path == "/api/auth/status"
    assert sent.headers["X-Owner-Tenant-Id"] == tenant


@pytest.mark.parametrize(
    ("method", "path"),
    [("POST", "/api/youtube/auth/connect")],
)
async def test_member_is_403_on_connect_and_disconnect(make_client, upstream, method, path) -> None:
    client, _ = make_client(roles=["member"])
    async with client:
        r = await client.request(method, path)
    assert r.status_code == 403, r.text
    assert upstream.requests == [], "a refused member must never reach youtube_upload"


@pytest.mark.parametrize(
    ("method", "path", "upstream_path"),
    [("POST", "/api/youtube/auth/connect", "/api/auth/connect")],
)
async def test_tenant_admin_forwards_own_tenant_not_a_spoofed_one(
    make_client, upstream, method, path, upstream_path
) -> None:
    """Inbound spoof headers must not steer the owner tenant sent upstream."""
    client, tenant = make_client(roles=["tenant_admin"])
    foreign = str(uuid4())
    upstream.reply = lambda req: httpx.Response(
        200,
        json={"authorization_url": "https://accounts.google.com/o/oauth2/auth?x=1", "state": "s"},
    )
    async with client:
        r = await client.request(
            method,
            path,
            headers={"X-Owner-Tenant-Id": foreign, "X-Active-Tenant-Id": foreign},
        )
    assert r.status_code == 200, r.text
    (sent,) = upstream.requests
    assert sent.method == method
    assert sent.url.path == upstream_path
    assert sent.headers["X-Owner-Tenant-Id"] == tenant
    assert sent.headers["X-Owner-Tenant-Id"] != foreign


async def test_connect_mirrors_the_json_contract(make_client, upstream) -> None:
    client, _ = make_client(roles=["tenant_admin"])
    body = {
        "authorization_url": "https://accounts.google.com/o/oauth2/auth?state=abc",
        "state": "abc",
    }
    upstream.reply = lambda req: httpx.Response(200, json=body)
    async with client:
        r = await client.post("/api/youtube/auth/connect")
    assert r.status_code == 200
    assert r.json() == body
    assert r.headers["content-type"].startswith("application/json")


async def test_production_client_carries_sa_token() -> None:
    """The SA lane: the baked license token rides ``X-Tenant-Id: Bearer``.

    ``set_transport`` (the seam used above) rebuilds the httpx client without the
    base headers, so the SA header is asserted on the production-built client.
    """
    yt = YouTubeUploadClient(
        base_url=_UPSTREAM, license_token="sa-token"
    )  # pragma: allowlist secret
    try:
        c = await yt._get_client()
        assert c.headers["X-Tenant-Id"] == "Bearer sa-token"
    finally:
        await yt.close()


# ---------------------------------------------------------------------- error mapping


async def test_upstream_404_passes_through(make_client, upstream) -> None:
    client, _ = make_client(roles=["tenant_admin"])
    upstream.reply = lambda req: httpx.Response(
        404, json={"detail": "No YouTube connection found for this tenant"}
    )
    async with client:
        r = await client.delete("/api/youtube/auth/disconnect")
    assert r.status_code == 404
    assert r.json() == {"detail": "No YouTube connection found for this tenant"}


@pytest.mark.parametrize("code", [500])
async def test_upstream_5xx_is_502(make_client, upstream, code) -> None:
    client, _ = make_client(roles=["member"])
    upstream.reply = lambda req: httpx.Response(code, json={"detail": "boom"})
    async with client:
        r = await client.get("/api/youtube/auth/status")
    assert r.status_code == 502, r.text
    assert f"HTTP {code}" in r.json()["detail"]


@pytest.mark.parametrize("code", [401])
async def test_upstream_sa_refusal_is_502_not_mirrored(make_client, upstream, code) -> None:
    """A 401/403 upstream is about the SA credential; mirroring it logs the user out."""
    client, _ = make_client(roles=["member"])
    upstream.reply = lambda req: httpx.Response(code, json={"detail": "nope"})
    async with client:
        r = await client.get("/api/youtube/auth/status")
    assert r.status_code == 502, r.text


async def test_upstream_unreachable_is_502(make_client, upstream) -> None:
    client, _ = make_client(roles=["member"])

    def _boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=req)

    upstream.reply = _boom
    async with client:
        r = await client.get("/api/youtube/auth/status")
    assert r.status_code == 502, r.text
    assert "unreachable" in r.json()["detail"]


async def test_upstream_redirect_is_not_followed(make_client, upstream) -> None:
    client, _ = make_client(roles=["tenant_admin"])
    upstream.reply = lambda req: httpx.Response(
        302, headers={"location": "https://accounts.google.com/o/oauth2/auth"}
    )
    async with client:
        r = await client.post("/api/youtube/auth/connect")
    assert r.status_code == 302
    assert r.headers["location"] == "https://accounts.google.com/o/oauth2/auth"
    assert len(upstream.requests) == 1
