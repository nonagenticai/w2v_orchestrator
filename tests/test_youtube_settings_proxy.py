"""Per-tenant YouTube settings + operator platform channel — orchestrator proxy.

youtube_upload owns the rows; the orchestrator owns WHO may touch them:

- ``GET /api/youtube/settings`` — any member of the active tenant;
- ``PUT|DELETE /api/youtube/settings/oauth-client`` — tenant_admin;
- ``GET|PUT /api/admin/youtube/platform-channel`` and
  ``PUT /api/admin/tenants/{tenant_id}/youtube`` — super_admin, and sent with NO
  ``X-Owner-Tenant-Id`` (operator lane).

The REAL ``YouTubeUploadClient`` runs over an ``httpx.MockTransport``, so the
assertions are about the bytes youtube_upload would receive (path, method, owner
header, JSON body) — same harness as ``test_p5_youtube_auth_proxy``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from starlette.middleware.base import BaseHTTPMiddleware
from w2v_common.auth.identity_context import clear_identity_context, set_identity_context

from src.api.routes import _common
from src.api.routes.youtube import admin_router as youtube_admin_router
from src.api.routes.youtube import router as youtube_router
from src.auth.membership import RESOLVED_BY_MEMBERSHIP
from src.clients.youtube_upload_client import YouTubeUploadClient
from src.database.models import AdminAuditLog

pytestmark = pytest.mark.asyncio

_UPSTREAM = "http://youtube-upload.test"
_CLIENT_SECRET = "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"  # pragma: allowlist secret
_REFRESH_TOKEN = "1//s3nt1nel-refresh-token-VALUE"  # pragma: allowlist secret


class _Upstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.reply: Callable[[httpx.Request], httpx.Response] = lambda req: httpx.Response(
            200, json={"ok": True}
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.reply(request)


def _build_app(
    db_session_factory,
    upstream: _Upstream,
    *,
    tenant_id: str,
    roles: list[str],
    subject_id: str | None = None,
):
    app = FastAPI()
    app.state.session_factory = db_session_factory
    yt = YouTubeUploadClient(
        base_url=_UPSTREAM, license_token="sa-token"
    )  # pragma: allowlist secret
    yt.set_transport(httpx.MockTransport(upstream.handler))
    app.state.youtube_client = yt
    app.include_router(youtube_router)
    app.include_router(youtube_admin_router)

    # Entitlement is not what these tests are about; the wiring of LICENSE_GATE_ONLY
    # on the admin routes is pinned by tests/unit/test_tenant_gate_coverage.py.
    async def _licensed() -> None:
        return None

    app.dependency_overrides[_common.require_license] = _licensed

    class _SeedMembership(BaseHTTPMiddleware):
        """TenancyMiddleware's leftovers + the AMBIENT identity AuthMiddleware binds."""

        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = tenant_id
            request.state.subject_id = subject_id or str(uuid4())
            request.state.roles = roles
            request.state.tenant_resolved_by = RESOLVED_BY_MEMBERSHIP
            # Worst case for the actor tests: the AMBIENT actor is whatever the
            # client sent, as if a lane had trusted it. identity_event_hook would
            # stamp that onto every outbound hop unless the route overrides it.
            set_identity_context(
                tenant_id=tenant_id,
                license_id="lic",
                subject_id="sub",
                actor_id=request.headers.get("X-Actor-Id"),
            )
            try:
                return await call_next(request)
            finally:
                clear_identity_context()

    app.add_middleware(_SeedMembership)
    return app


@pytest.fixture
def upstream() -> _Upstream:
    return _Upstream()


@pytest.fixture
def make_client(db_session_factory, upstream):
    def _make(*, roles: list[str], subject: str | None = None) -> tuple[AsyncClient, str]:
        tid = str(uuid4())
        app = _build_app(
            db_session_factory, upstream, tenant_id=tid, roles=roles, subject_id=subject
        )
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver"), tid

    return _make


def _sent_json(req: httpx.Request) -> dict[str, object]:
    body = json.loads(req.content)
    assert isinstance(body, dict)
    return body


# ------------------------------------------------------------------ tenant settings


async def test_member_reads_settings_for_own_tenant(make_client, upstream) -> None:
    client, tenant = make_client(roles=["member"])
    settings = {
        "oauth_client_id": "test-client-id.apps.example",
        "has_oauth_client_secret": True,
        "redirect_uri": "https://studio.nonagentic.ai/api/youtube/auth/callback",
        "platform_channel_allowed": None,
        "effective_platform_channel_allowed": True,
        "platform_channel": {"channel_id": "UCx", "channel_title": "Platform"},
    }
    upstream.reply = lambda req: httpx.Response(200, json=settings)
    async with client:
        r = await client.get("/api/youtube/settings", headers={"X-Owner-Tenant-Id": str(uuid4())})
    assert r.status_code == 200, r.text
    assert r.json() == settings
    (sent,) = upstream.requests
    assert (sent.method, sent.url.path) == ("GET", "/api/settings")
    assert sent.headers["X-Owner-Tenant-Id"] == tenant


@pytest.mark.parametrize("method", ["PUT"])
async def test_member_is_403_on_oauth_client_writes(make_client, upstream, method) -> None:
    client, _ = make_client(roles=["member"])
    async with client:
        r = await client.request(
            method,
            "/api/youtube/settings/oauth-client",
            json={"client_id": "c", "client_secret": _CLIENT_SECRET},
        )
    assert r.status_code == 403, r.text
    assert upstream.requests == []


async def test_tenant_admin_puts_oauth_client_secret_never_echoed_or_logged(
    make_client, upstream, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    client, tenant = make_client(roles=["tenant_admin"])
    foreign = str(uuid4())
    upstream.reply = lambda req: httpx.Response(
        200, json={"oauth_client_id": "cid", "has_oauth_client_secret": True}
    )
    async with client:
        r = await client.put(
            "/api/youtube/settings/oauth-client",
            json={"client_id": "cid", "client_secret": _CLIENT_SECRET},
            headers={"X-Owner-Tenant-Id": foreign, "X-Active-Tenant-Id": foreign},
        )
    assert r.status_code == 200, r.text
    (sent,) = upstream.requests
    assert (sent.method, sent.url.path) == ("PUT", "/api/settings/oauth-client")
    assert sent.headers["X-Owner-Tenant-Id"] == tenant
    assert _sent_json(sent) == {"client_id": "cid", "client_secret": _CLIENT_SECRET}
    assert _CLIENT_SECRET not in r.text
    assert _CLIENT_SECRET not in caplog.text


async def test_oauth_client_put_requires_both_fields(make_client, upstream) -> None:
    client, _ = make_client(roles=["tenant_admin"])
    async with client:
        r = await client.put("/api/youtube/settings/oauth-client", json={"client_id": "cid"})
    assert r.status_code == 422
    assert upstream.requests == []


async def test_tenant_admin_deletes_oauth_client_204_passthrough(make_client, upstream) -> None:
    client, tenant = make_client(roles=["tenant_admin"])
    upstream.reply = lambda req: httpx.Response(204)
    async with client:
        r = await client.delete("/api/youtube/settings/oauth-client")
    assert r.status_code == 204
    assert r.content == b""
    (sent,) = upstream.requests
    assert (sent.method, sent.url.path) == ("DELETE", "/api/settings/oauth-client")
    assert sent.headers["X-Owner-Tenant-Id"] == tenant


async def test_upstream_422_passes_through_with_echoed_secret_scrubbed(
    make_client, upstream, caplog: pytest.LogCaptureFixture
) -> None:
    """FastAPI's 422 echoes the rejected payload under ``input`` — scrub it."""
    caplog.set_level(logging.DEBUG)
    client, _ = make_client(roles=["tenant_admin"])
    upstream.reply = lambda req: httpx.Response(
        422,
        json={
            "detail": [
                {
                    "type": "value_error",
                    "loc": ["body", "client_secret"],
                    "msg": "Value error, not a Google client secret",
                    "input": _CLIENT_SECRET,
                },
                {"loc": ["body"], "msg": "bad", "input": {"client_secret": _CLIENT_SECRET}},
            ]
        },
    )
    async with client:
        r = await client.put(
            "/api/youtube/settings/oauth-client",
            json={"client_id": "cid", "client_secret": _CLIENT_SECRET},
        )
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail[0]["msg"] == "Value error, not a Google client secret"
    assert detail[0]["loc"] == ["body", "client_secret"]
    assert detail[0]["input"] == "[REDACTED]"
    assert _CLIENT_SECRET not in r.text
    assert _CLIENT_SECRET not in caplog.text


@pytest.mark.parametrize(
    ("code", "detail"),
    [(404, "no settings for this tenant")],
)
async def test_upstream_4xx_passes_through(make_client, upstream, code, detail) -> None:
    client, _ = make_client(roles=["member"])
    upstream.reply = lambda req: httpx.Response(code, json={"detail": detail})
    async with client:
        r = await client.get("/api/youtube/settings")
    assert r.status_code == code
    assert r.json() == {"detail": detail}


@pytest.mark.parametrize("code", [500])
async def test_upstream_5xx_and_sa_refusal_are_502(make_client, upstream, code) -> None:
    client, _ = make_client(roles=["member"])
    upstream.reply = lambda req: httpx.Response(code, json={"detail": "boom"})
    async with client:
        r = await client.get("/api/youtube/settings")
    assert r.status_code == 502, r.text
    assert f"HTTP {code}" in r.json()["detail"]


async def test_upstream_unreachable_is_502(make_client, upstream) -> None:
    client, _ = make_client(roles=["tenant_admin"])

    def _boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=req)

    upstream.reply = _boom
    async with client:
        r = await client.delete("/api/youtube/settings/oauth-client")
    assert r.status_code == 502, r.text
    assert "unreachable" in r.json()["detail"]


async def test_status_passthrough_keeps_resolution_fields(make_client, upstream) -> None:
    """The status proxy mirrors the body verbatim — mode/channel_title/needs_reconnect survive."""
    client, _ = make_client(roles=["member"])
    body = {
        "connected": True,
        "mode": "platform",
        "channel_title": "George Dekker",
        "needs_reconnect": False,
    }
    upstream.reply = lambda req: httpx.Response(200, json=body)
    async with client:
        r = await client.get("/api/youtube/auth/status")
    assert r.status_code == 200
    assert r.json() == body


# ---------------------------------------------------------------- operator (super_admin)

# Fixed, NOT uuid4(): this value is part of a parametrize node id, so generating
# it at import time makes every pytest-xdist worker collect a different test id
# and aborts the run with "Different tests were collected between gw0 and gwN".
_FIXED_TENANT_ID = "00000000-0000-4000-8000-00000000b001"

_ADMIN_CALLS = [
    ("GET", "/api/admin/youtube/platform-channel", None),
    (
        "PUT",
        "/api/admin/youtube/platform-channel",
        {
            "client_id": "cid",
            "channel_id": "UCx",
            "channel_title": "Platform",
            "default_for_tenants": True,
        },
    ),
    ("PUT", f"/api/admin/tenants/{_FIXED_TENANT_ID}/youtube", {"platform_channel_allowed": False}),
]


@pytest.mark.parametrize("roles", [["member"]])
@pytest.mark.parametrize(
    ("method", "path", "body"), [("GET", "/api/admin/youtube/platform-channel", None)]
)
async def test_non_super_admin_is_403_on_operator_routes(
    make_client, upstream, roles, method, path, body
) -> None:
    client, _ = make_client(roles=roles)
    async with client:
        r = await client.request(method, path, json=body)
    assert r.status_code == 403, r.text
    assert upstream.requests == []


async def test_super_admin_reads_platform_channel_without_owner_tenant(
    make_client, upstream
) -> None:
    client, _ = make_client(roles=["super_admin", "tenant_admin"])
    channel = {
        "client_id": "cid",
        "channel_id": "UCCswvznb9fLCzlaetPPkhLg",
        "channel_title": "George Dekker",
        "default_for_tenants": True,
        "has_client_secret": True,
        "has_refresh_token": True,
    }
    upstream.reply = lambda req: httpx.Response(200, json=channel)
    async with client:
        r = await client.get("/api/admin/youtube/platform-channel")
    assert r.status_code == 200, r.text
    assert r.json() == channel
    (sent,) = upstream.requests
    assert (sent.method, sent.url.path) == ("GET", "/api/admin/platform-channel")
    # The ambient identity tenant IS bound (see _SeedMembership) — the operator
    # lane must still send no owner tenant.
    assert "X-Owner-Tenant-Id" not in sent.headers


async def test_super_admin_puts_platform_channel_secrets_forwarded_not_echoed(
    make_client, upstream, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    client, _ = make_client(roles=["super_admin"])
    upstream.reply = lambda req: httpx.Response(
        200,
        json={**_sent_json(req), "refresh_token": _REFRESH_TOKEN},
    )
    body = {
        "client_id": "cid",
        "client_secret": _CLIENT_SECRET,
        "refresh_token": _REFRESH_TOKEN,
        "channel_id": "UCx",
        "channel_title": "Platform",
        "default_for_tenants": False,
    }
    async with client:
        r = await client.put("/api/admin/youtube/platform-channel", json=body)
    assert r.status_code == 200, r.text
    (sent,) = upstream.requests
    assert (sent.method, sent.url.path) == ("PUT", "/api/admin/platform-channel")
    assert "X-Owner-Tenant-Id" not in sent.headers
    assert _sent_json(sent) == body
    # Even an upstream that (wrongly) echoed the secrets does not reach the caller.
    assert _CLIENT_SECRET not in r.text
    assert _REFRESH_TOKEN not in r.text
    assert r.json()["channel_title"] == "Platform"
    assert _CLIENT_SECRET not in caplog.text
    assert _REFRESH_TOKEN not in caplog.text


async def test_platform_channel_put_omitted_secrets_are_not_sent(make_client, upstream) -> None:
    """Secrets optional on PUT = keep: absent from the upstream body, not ``null``."""
    client, _ = make_client(roles=["super_admin"])
    async with client:
        r = await client.put(
            "/api/admin/youtube/platform-channel",
            json={
                "client_id": "cid",
                "channel_id": "UCx",
                "channel_title": "Platform",
                "default_for_tenants": True,
            },
        )
    assert r.status_code == 200, r.text
    (sent,) = upstream.requests
    assert _sent_json(sent) == {
        "client_id": "cid",
        "channel_id": "UCx",
        "channel_title": "Platform",
        "default_for_tenants": True,
    }


@pytest.mark.parametrize("allowed", [True])
async def test_super_admin_sets_other_tenants_platform_policy(
    make_client, upstream, allowed
) -> None:
    client, own_tenant = make_client(roles=["super_admin"])
    other = str(uuid4())
    upstream.reply = lambda req: httpx.Response(
        200, json={"tenant_id": other, "platform_channel_allowed": allowed}
    )
    async with client:
        r = await client.put(
            f"/api/admin/tenants/{other}/youtube", json={"platform_channel_allowed": allowed}
        )
    assert r.status_code == 200, r.text
    (sent,) = upstream.requests
    assert (sent.method, sent.url.path) == (
        "PUT",
        f"/api/admin/tenants/{other}/platform-channel",
    )
    assert _sent_json(sent) == {"allowed": allowed}
    assert "X-Owner-Tenant-Id" not in sent.headers
    assert other != own_tenant


async def test_tenant_policy_field_is_required(make_client, upstream) -> None:
    """``null`` means "follow the default"; an ABSENT field is a client bug, not null."""
    client, _ = make_client(roles=["super_admin"])
    async with client:
        r = await client.put(f"/api/admin/tenants/{uuid4()}/youtube", json={})
    assert r.status_code == 422
    assert upstream.requests == []


async def test_operator_upstream_404_passes_through(make_client, upstream) -> None:
    client, _ = make_client(roles=["super_admin"])
    upstream.reply = lambda req: httpx.Response(404, json={"detail": "no platform channel"})
    async with client:
        r = await client.get("/api/admin/youtube/platform-channel")
    assert r.status_code == 404
    assert r.json() == {"detail": "no platform channel"}


# --------------------------------------------------------------------- client lane


async def test_operator_call_strips_ambient_and_explicit_owner() -> None:
    upstream = _Upstream()
    yt = YouTubeUploadClient(
        base_url=_UPSTREAM, license_token="sa-token"
    )  # pragma: allowlist secret
    yt.set_transport(httpx.MockTransport(upstream.handler))
    tenant = str(uuid4())
    set_identity_context(tenant_id=tenant, license_id="lic", subject_id="sub")
    try:
        await yt.operator_call("GET", "/platform-channel")
        await yt.settings_call("GET", "", tenant)
    finally:
        clear_identity_context()
        await yt.close()
    operator, tenant_call = upstream.requests
    assert "X-Owner-Tenant-Id" not in operator.headers
    # The suppression is scoped to the operator call — the next call stamps again.
    assert tenant_call.headers["X-Owner-Tenant-Id"] == tenant


# ------------------------------------------------------------------ admin audit log
# Every other /api/admin and tenant-admin mutation writes ``admin_audit_log``
# (tenant.update, connection.set, apikey.create ...). These four config writes
# must too — recorded only once youtube_upload ACCEPTED the change, and never
# with a secret in ``detail``.


async def _audit_rows(db_session_factory, *, action: str) -> list[AdminAuditLog]:
    async with db_session_factory() as s:
        return list(
            (
                await s.execute(
                    select(AdminAuditLog)
                    .where(AdminAuditLog.action == action)
                    .order_by(AdminAuditLog.created_at)
                )
            )
            .scalars()
            .all()
        )


async def test_oauth_client_put_and_delete_are_audited_without_the_secret(
    make_client, upstream, db_session_factory
) -> None:
    client, tenant = make_client(roles=["tenant_admin"])
    async with client:
        r = await client.put(
            "/api/youtube/settings/oauth-client",
            json={"client_id": "cid-audit", "client_secret": _CLIENT_SECRET},
        )
        assert r.status_code == 200, r.text
        upstream.reply = lambda req: httpx.Response(204)
        r = await client.delete("/api/youtube/settings/oauth-client")
        assert r.status_code == 204, r.text
    [put] = [
        e
        for e in await _audit_rows(db_session_factory, action="youtube.oauth_client.set")
        if str(e.target_tenant_id) == tenant
    ]
    assert put.detail == {"client_id": "cid-audit"}
    assert put.actor_subject_id is not None
    assert _CLIENT_SECRET not in json.dumps(put.detail)
    [cleared] = [
        e
        for e in await _audit_rows(db_session_factory, action="youtube.oauth_client.clear")
        if str(e.target_tenant_id) == tenant
    ]
    assert cleared.detail == {}


async def test_rejected_upstream_write_is_not_audited(
    make_client, upstream, db_session_factory
) -> None:
    client, tenant = make_client(roles=["tenant_admin"])
    upstream.reply = lambda req: httpx.Response(422, json={"detail": "bad client"})
    async with client:
        r = await client.put(
            "/api/youtube/settings/oauth-client",
            json={"client_id": "cid", "client_secret": _CLIENT_SECRET},
        )
    assert r.status_code == 422
    rows = await _audit_rows(db_session_factory, action="youtube.oauth_client.set")
    assert [e for e in rows if str(e.target_tenant_id) == tenant] == []


async def test_platform_channel_put_is_audited_naming_rotated_secrets_only(
    make_client, upstream, db_session_factory
) -> None:
    client, _ = make_client(roles=["super_admin"])
    title = f"Platform {uuid4()}"
    async with client:
        r = await client.put(
            "/api/admin/youtube/platform-channel",
            json={
                "client_id": "cid",
                "refresh_token": _REFRESH_TOKEN,
                "channel_id": "UCx",
                "channel_title": title,
                "default_for_tenants": True,
            },
        )
    assert r.status_code == 200, r.text
    [entry] = [
        e
        for e in await _audit_rows(db_session_factory, action="youtube.platform_channel.set")
        if e.detail.get("channel_title") == title
    ]
    assert entry.target_tenant_id is None
    assert entry.detail == {
        "client_id": "cid",
        "channel_id": "UCx",
        "channel_title": title,
        "default_for_tenants": True,
        "rotated": ["refresh_token"],
    }
    assert _REFRESH_TOKEN not in json.dumps(entry.detail)


async def test_tenant_policy_put_is_audited_against_the_target_tenant(
    make_client, upstream, db_session_factory
) -> None:
    client, _ = make_client(roles=["super_admin"])
    other = str(uuid4())
    async with client:
        r = await client.put(
            f"/api/admin/tenants/{other}/youtube", json={"platform_channel_allowed": False}
        )
    assert r.status_code == 200, r.text
    [entry] = [
        e
        for e in await _audit_rows(db_session_factory, action="youtube.tenant_policy.set")
        if str(e.target_tenant_id) == other
    ]
    assert entry.detail == {"platform_channel_allowed": False}


# ------------------------------------------------------ operator tenant-settings list


@pytest.mark.parametrize("roles", [["member"]])
async def test_non_super_admin_is_403_on_tenant_settings_list(make_client, upstream, roles) -> None:
    client, _ = make_client(roles=roles)
    async with client:
        r = await client.get("/api/admin/youtube/tenants")
    assert r.status_code == 403, r.text
    assert upstream.requests == []


async def test_super_admin_lists_tenant_settings_passthrough_without_owner(
    make_client, upstream
) -> None:
    client, _ = make_client(roles=["super_admin"])
    rows = [
        {
            "tenant_id": str(uuid4()),
            "platform_channel_allowed": None,
            "has_grant": True,
            "has_oauth_client": False,
        },
        {
            "tenant_id": str(uuid4()),
            "platform_channel_allowed": False,
            "has_grant": False,
            "has_oauth_client": True,
        },
    ]
    upstream.reply = lambda req: httpx.Response(200, json=rows)
    async with client:
        r = await client.get("/api/admin/youtube/tenants")
    assert r.status_code == 200, r.text
    assert r.json() == rows
    (sent,) = upstream.requests
    assert (sent.method, sent.url.path) == ("GET", "/api/admin/tenant-settings")
    assert "X-Owner-Tenant-Id" not in sent.headers


async def test_tenant_settings_list_upstream_5xx_is_502(make_client, upstream) -> None:
    client, _ = make_client(roles=["super_admin"])
    upstream.reply = lambda req: httpx.Response(503, json={"detail": "down"})
    async with client:
        r = await client.get("/api/admin/youtube/tenants")
    assert r.status_code == 502, r.text


# ------------------------------------------------------------- acting user (X-Actor-Id)
# youtube_upload records ``updated_by`` from ``X-Actor-Id`` (else the SA subject).
# Every write names the authenticated human — the SAME subject the audit row
# carries — and a client-supplied X-Actor-Id never reaches upstream, even when
# the ambient identity context holds it (the seed middleware binds it there).

_FORGED_ACTOR = "00000000-0000-4000-8000-00000000f0f0"


@pytest.mark.parametrize(
    ("roles", "method", "path", "body", "upstream_path"),
    [
        (
            ["tenant_admin"],
            "PUT",
            "/api/youtube/settings/oauth-client",
            {"client_id": "cid", "client_secret": _CLIENT_SECRET},
            "/api/settings/oauth-client",
        )
    ],
)
@pytest.mark.parametrize("client_actor", [None])
async def test_writes_forward_the_authenticated_subject_as_actor(
    make_client, upstream, roles, method, path, body, upstream_path, client_actor
) -> None:
    subject = str(uuid4())
    other = str(uuid4())
    client, _ = make_client(roles=roles, subject=subject)
    headers = {"X-Actor-Id": client_actor} if client_actor else {}
    async with client:
        r = await client.request(method, path.format(other=other), json=body, headers=headers)
    assert r.status_code == 200, r.text
    (sent,) = upstream.requests
    assert (sent.method, sent.url.path) == (method, upstream_path.format(other=other))
    assert sent.headers.get_list("X-Actor-Id") == [subject]
    assert _FORGED_ACTOR not in str(sent.headers)


async def test_actor_header_matches_the_audit_row_actor(
    make_client, upstream, db_session_factory
) -> None:
    subject = str(uuid4())
    client, tenant = make_client(roles=["tenant_admin"], subject=subject)
    async with client:
        r = await client.put(
            "/api/youtube/settings/oauth-client",
            json={"client_id": "cid-actor", "client_secret": _CLIENT_SECRET},
            headers={"X-Actor-Id": _FORGED_ACTOR},
        )
    assert r.status_code == 200, r.text
    [entry] = [
        e
        for e in await _audit_rows(db_session_factory, action="youtube.oauth_client.set")
        if str(e.target_tenant_id) == tenant
    ]
    (sent,) = upstream.requests
    assert sent.headers["X-Actor-Id"] == str(entry.actor_subject_id) == subject


async def test_reads_do_not_carry_an_explicit_actor(make_client, upstream) -> None:
    """GETs change nothing upstream; the override is scoped to writes only."""
    client, _ = make_client(roles=["super_admin"], subject=str(uuid4()))
    async with client:
        r = await client.get("/api/admin/youtube/platform-channel")
    assert r.status_code == 200, r.text
    (sent,) = upstream.requests
    assert sent.headers.get("X-Actor-Id") is None
