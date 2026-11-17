"""Tests for the /api/critique and /api/redesign proxy routers.

The orchestrator owns w2v.nonagentic.ai's API surface; these proxies
forward critique/redesign calls to analysis_service. We mock the outbound
httpx call (patched in the shared ``_proxy`` module) and assert the
page_size→per_page translation, create-body passthrough, status
passthrough, and license-token header forwarding.

The proxies are DB-free, so this test builds a minimal FastAPI app
(critique + redesign routers + a tenant-id-seeding middleware) instead of
using the conftest's heavy ``app`` fixture, which requires a live postgres.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID, uuid4, uuid7

import httpx
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient
from w2v_common.auth.identity_context import clear_identity_context, set_identity_context
from w2v_common.tenant import MissingTenantError

from src.api.routes.brand import router as brand_router
from src.api.routes.critique import router as critique_router
from src.api.routes.redesign import router as redesign_router


def _identity_seeder(*, tenant_id: str, roles: list[str] | None = None, subject_id: str):
    """Build a PURE ASGI middleware that seeds request.state + the identity ContextVar.

    Pure ASGI, never ``BaseHTTPMiddleware``: the latter runs the inner app in a
    separate anyio task, so a ContextVar bound in ``dispatch`` does NOT reach the
    route handler and the binding silently no-ops. Since w2v-common 0.67.0 that
    is fatal rather than cosmetic — ``get_identity_headers()`` RAISES when no
    tenant is bound ("an outbound call may not carry a blank X-Tenant-Id"), so
    seeding ``request.state`` alone no longer stands in for the real app (which
    binds the ContextVar in ``src/auth/dependencies.py``).

    The binding uses the CLAIMED tenant. That is deliberate: the scope-switch
    tests assert that a ``super_admin``'s ``?active_tenant=`` override is what
    reaches the wire, which only proves the routes resolve through
    ``get_tenant_id`` if the ambient var holds the claim instead.
    """

    class _Seed:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                await self.app(scope, receive, send)
                return
            state = scope.setdefault("state", {})
            state["tenant_id"] = tenant_id
            state["subject_id"] = subject_id
            if roles is not None:
                state["roles"] = roles
            set_identity_context(tenant_id=tenant_id, license_id="lic-test", subject_id=subject_id)
            try:
                await self.app(scope, receive, send)
            finally:
                clear_identity_context()

    return _Seed


@pytest.fixture
def app_with_seeded_tenant() -> FastAPI:
    """Minimal FastAPI app with both proxy routers + tenant_id seeded on request.state."""
    tenant_id = str(uuid4())

    subject_id = str(uuid4())

    app = FastAPI()
    app.add_middleware(_identity_seeder(tenant_id=tenant_id, subject_id=subject_id))
    app.include_router(critique_router)
    app.include_router(redesign_router)
    return app


@pytest.fixture
async def client(app_with_seeded_tenant):
    transport = ASGITransport(app=app_with_seeded_tenant)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest.fixture
def mock_analysis(monkeypatch):
    """Patch httpx.AsyncClient inside the shared proxy module to a MockTransport.

    Returns a dict with ``captured`` (every outbound request, incl. parsed JSON
    body) and ``state`` (mutable per-test response payload + status + content-type).
    """
    captured: list[dict[str, Any]] = []
    state: dict[str, Any] = {
        "payload": {"items": [], "total": 0, "page": 1, "per_page": 20},
        "status": 200,
        "content_type": "application/json",
        "raw": None,  # when set, returned verbatim instead of json.dumps(payload)
    }

    def _handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode() if request.content else ""
        captured.append(
            {
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "params": dict(request.url.params),
                "headers": dict(request.headers),
                "json": json.loads(body) if body else None,
            }
        )
        content = state["raw"] if state["raw"] is not None else json.dumps(state["payload"])
        return httpx.Response(
            state["status"],
            content=content,
            headers={"content-type": state["content_type"]},
        )

    real_async_client = httpx.AsyncClient

    class _StubAsyncClient:
        def __init__(self, *args, **kwargs):
            self._inner = real_async_client(
                transport=httpx.MockTransport(_handler),
                timeout=kwargs.get("timeout"),
            )

        async def __aenter__(self):
            await self._inner.__aenter__()
            return self

        async def __aexit__(self, *exc):
            await self._inner.__aexit__(*exc)

        async def get(self, url, *, params=None, headers=None):
            return await self._inner.get(url, params=params, headers=headers)

        async def post(self, url, *, json=None, headers=None):
            return await self._inner.post(url, json=json, headers=headers)

    import src.api.routes._proxy as proxy_mod

    monkeypatch.setattr(proxy_mod.httpx, "AsyncClient", _StubAsyncClient)
    return {"captured": captured, "state": state}


# --------------------------------------------------------------------------- #
# Critique reads (unchanged behavior after the _proxy extraction)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_list_translates_page_size_to_per_page(client, mock_analysis) -> None:
    mock_analysis["state"]["payload"] = {
        "items": [{"job_id": "j1", "status": "completed", "url": "https://x"}],
        "total": 1,
        "page": 1,
        "per_page": 20,
    }
    r = await client.get("/api/critique?page=1&page_size=20")
    assert r.status_code == 200, r.text
    assert r.json()["total"] == 1

    sent = mock_analysis["captured"][-1]
    assert sent["path"] == "/api/critique"
    assert sent["params"]["page"] == "1"
    assert sent["params"]["per_page"] == "20"
    # Frontend's page_size param name must NOT leak through to the downstream.
    assert "page_size" not in sent["params"]


@pytest.mark.asyncio
async def test_list_forwards_status_filter(client, mock_analysis) -> None:
    mock_analysis["state"]["payload"] = {"items": [], "total": 0, "page": 1, "per_page": 20}
    r = await client.get("/api/critique?page=1&page_size=20&status=processing")
    assert r.status_code == 200
    assert mock_analysis["captured"][-1]["params"]["status"] == "processing"


@pytest.mark.asyncio
async def test_get_single_critique(client, mock_analysis) -> None:
    mock_analysis["state"]["payload"] = {
        "job_id": "abc-123",
        "status": "completed",
        "url": "https://example.com",
    }
    r = await client.get("/api/critique/abc-123")
    assert r.status_code == 200
    assert r.json()["job_id"] == "abc-123"
    assert mock_analysis["captured"][-1]["path"] == "/api/critique/abc-123"


@pytest.mark.asyncio
async def test_downstream_error_surfaces_status(client, mock_analysis) -> None:
    mock_analysis["state"]["status"] = 503
    mock_analysis["state"]["payload"] = {"detail": "upstream busy"}
    r = await client.get("/api/critique/abc-123")
    assert r.status_code == 503


# --------------------------------------------------------------------------- #
# Critique create (new)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_create_critique_posts_body_to_analysis(client, mock_analysis) -> None:
    mock_analysis["state"]["payload"] = {"job_id": "new-1", "status": "pending"}
    r = await client.post("/api/critique", json={"url": "https://example.com"})
    assert r.status_code == 200, r.text
    assert r.json()["job_id"] == "new-1"

    sent = mock_analysis["captured"][-1]
    assert sent["method"] == "POST"
    assert sent["path"] == "/api/critique"
    assert sent["json"] == {"url": "https://example.com"}


@pytest.mark.asyncio
async def test_create_critique_drops_none_optional_fields(client, mock_analysis) -> None:
    mock_analysis["state"]["payload"] = {"job_id": "new-2", "status": "pending"}
    r = await client.post(
        "/api/critique",
        json={"url": "https://example.com", "max_pages": 3, "categories": ["seo", "a11y"]},
    )
    assert r.status_code == 200
    sent = mock_analysis["captured"][-1]["json"]
    # exclude_none means unset optionals (instructions) must not be forwarded.
    assert sent == {
        "url": "https://example.com",
        "max_pages": 3,
        "categories": ["seo", "a11y"],
    }


@pytest.mark.asyncio
async def test_create_critique_forwards_license_token(client, mock_analysis) -> None:
    mock_analysis["state"]["payload"] = {"job_id": "new-3", "status": "pending"}
    await client.post("/api/critique", json={"url": "https://example.com"})
    headers = mock_analysis["captured"][-1]["headers"]
    # Orchestrator's own DOWNSTREAM_LICENSE_TOKEN fallback fires (no inbound license).
    assert headers.get("authorization", "").startswith("Bearer ")
    assert headers.get("x-tenant-id", "").startswith("Bearer ")


# --------------------------------------------------------------------------- #
# Critique create — analysis CritiqueRequest mirror (P9-7: html / artifact_ref)
# --------------------------------------------------------------------------- #

# A caller-steered LLM endpoint + env-var NAME (not a secret value) — the attack
# shape the proxy must refuse.
_EXFIL_LLM_CONFIG = {
    "base_url": "https://evil.test",
    "api_key_env": "OPENAI_API_KEY",  # pragma: allowlist secret
}
_HTML = "<!doctype html><html><head><title>t</title></head><body><h1>Hi</h1></body></html>"


@pytest.mark.asyncio
async def test_create_critique_html_body_forwarded_unchanged(client, mock_analysis) -> None:
    """merqi marketing's branding-render critique sends ``{html, categories}``."""
    mock_analysis["state"]["payload"] = {"job_id": "h-1", "status": "pending"}
    body = {"html": _HTML, "categories": ["seo", "accessibility", "performance", "ux"]}
    r = await client.post("/api/critique", json=body)
    assert r.status_code == 200, r.text
    sent = mock_analysis["captured"][-1]
    assert sent["method"] == "POST"
    assert sent["path"] == "/api/critique"
    assert sent["json"] == body


@pytest.mark.asyncio
async def test_create_critique_artifact_ref_forwarded_unchanged(client, mock_analysis) -> None:
    mock_analysis["state"]["payload"] = {"job_id": "a-1", "status": "pending"}
    body = {"artifact_ref": "renders/r-1/index.html", "categories": ["seo"]}
    r = await client.post("/api/critique", json=body)
    assert r.status_code == 200, r.text
    assert mock_analysis["captured"][-1]["json"] == body


@pytest.mark.asyncio
async def test_create_critique_full_contract_forwarded(client, mock_analysis) -> None:
    """Every mirrored analysis field survives the hop byte-for-byte."""
    mock_analysis["state"]["payload"] = {"job_id": "f-1", "status": "pending"}
    body = {
        "url": "https://example.com",
        "categories": ["seo"],
        "max_pages": 2,
        "instructions": "focus on the hero",
        "github_repo": "https://github.com/o/r",
        "force_rescan": True,
        "callback_url": "https://hooks.example.com/cb",
        "tags": ["p9"],
        "webhook_config": {"events": ["completed"], "severity_threshold": "high"},
        "auth_info": {"type": "bearer", "token": "t0k"},
    }
    r = await client.post("/api/critique", json=body)
    assert r.status_code == 200, r.text
    assert mock_analysis["captured"][-1]["json"] == body


@pytest.mark.parametrize(
    "body",
    [{}],
    ids=["empty"],
)
@pytest.mark.asyncio
async def test_create_critique_requires_exactly_one_source(client, mock_analysis, body) -> None:
    r = await client.post("/api/critique", json=body)
    assert r.status_code == 422, r.text
    assert "exactly one of" in r.text
    assert mock_analysis["captured"] == [], "a rejected body never reaches analysis"


@pytest.mark.parametrize(
    "extra",
    [{"llm_config": _EXFIL_LLM_CONFIG}],
    ids=["llm_config"],
)
@pytest.mark.asyncio
async def test_create_critique_rejects_unexposed_fields(client, mock_analysis, extra) -> None:
    """Not an open passthrough: analysis-internal / unknown fields are a loud 422."""
    r = await client.post("/api/critique", json={"html": _HTML, **extra})
    assert r.status_code == 422, r.text
    assert mock_analysis["captured"] == []


@pytest.mark.asyncio
async def test_create_critique_auth_info_rules_mirror_analysis(client, mock_analysis) -> None:
    r = await client.post(
        "/api/critique", json={"url": "https://example.com", "auth_info": {"type": "basic"}}
    )
    assert r.status_code == 422, r.text
    assert "username and password" in r.text
    assert mock_analysis["captured"] == []


def _capped_app() -> FastAPI:
    from src.api.routes.brand import router as brand_router
    from src.middleware.proxy_body_limit import ProxyBodyLimitMiddleware

    app = FastAPI()
    app.add_middleware(_identity_seeder(tenant_id=str(uuid4()), subject_id=str(uuid4())))
    app.add_middleware(ProxyBodyLimitMiddleware)
    app.include_router(critique_router)
    app.include_router(brand_router)
    return app


def _chunked(payload: bytes):
    async def gen():  # no Content-Length -> the byte-counting path
        for i in range(0, len(payload), 256 * 1024):
            yield payload[i : i + 256 * 1024]

    return gen()


@pytest.mark.asyncio
async def test_critique_create_cap_is_analysis_html_limit(mock_analysis) -> None:
    """POST /api/critique is capped at analysis's own MAX_HTML_SIZE (+ envelope),
    not the 256 KiB default: a 1 MB rendered page is forwarded intact; a body
    over analysis's limit is 413 (declared AND chunked) and never forwarded."""
    from src.middleware.proxy_body_limit import (
        ANALYSIS_CRITIQUE_MAX_HTML_BYTES,
        PROXY_BODY_LIMIT_OVERRIDES,
    )

    assert ANALYSIS_CRITIQUE_MAX_HTML_BYTES == 10 * 1024 * 1024  # analysis MAX_HTML_SIZE
    cap = PROXY_BODY_LIMIT_OVERRIDES["/api/critique"]
    mock_analysis["state"]["payload"] = {"job_id": "ok", "status": "pending"}
    async with AsyncClient(transport=ASGITransport(app=_capped_app()), base_url="http://t") as c:
        one_mb = "x" * (1024 * 1024)
        ok = await c.post("/api/critique", json={"html": one_mb, "categories": ["seo"]})
        assert ok.status_code == 200, ok.text
        assert mock_analysis["captured"][-1]["json"]["html"] == one_mb

        # Full analysis-size html still fits inside the envelope allowance.
        full = await c.post("/api/critique", json={"html": "x" * ANALYSIS_CRITIQUE_MAX_HTML_BYTES})
        assert full.status_code == 200, full.text
        sent_before = len(mock_analysis["captured"])

        over = {"html": "x" * (cap + 1)}
        declared = await c.post("/api/critique", json=over)
        assert declared.status_code == 413, declared.text
        assert declared.json()["error"]["code"] == "REQUEST_BODY_TOO_LARGE"
        chunked = await c.post(
            "/api/critique",
            content=_chunked(json.dumps(over).encode()),
            headers={"Content-Type": "application/json"},
        )
        assert chunked.status_code == 413, chunked.text
        assert len(mock_analysis["captured"]) == sent_before, "over-cap never forwarded"


@pytest.mark.asyncio
async def test_brand_route_keeps_the_256k_default_cap(mock_analysis) -> None:
    """The critique override is exact-path: brand/* stay at 256 KiB."""
    from src.middleware.proxy_body_limit import PROXY_BODY_LIMIT_BYTES

    body = {"url": "https://a.test", "pad": "x" * (PROXY_BODY_LIMIT_BYTES + 1)}
    async with AsyncClient(transport=ASGITransport(app=_capped_app()), base_url="http://t") as c:
        declared = await c.post("/api/brand/from-url", json=body)
        assert declared.status_code == 413, declared.text
        chunked = await c.post(
            "/api/brand/from-url",
            content=_chunked(json.dumps(body).encode()),
            headers={"Content-Type": "application/json"},
        )
        assert chunked.status_code == 413, chunked.text
    assert mock_analysis["captured"] == []


# --------------------------------------------------------------------------- #
# Redesign proxy (new)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_create_redesign_posts_body_with_defaults(client, mock_analysis) -> None:
    mock_analysis["state"]["payload"] = {"job_id": "rd-1", "status": "pending"}
    r = await client.post(
        "/api/redesign",
        json={"url": "https://example.com", "critique_findings": [{"id": "f1"}]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["job_id"] == "rd-1"

    sent = mock_analysis["captured"][-1]
    assert sent["method"] == "POST"
    # Inbound public path is /api/redesign; forwarded to html_render's /api/designs.
    assert sent["path"] == "/api/designs"
    # Defaults (scope/style) are sent; critique_findings passes through.
    assert sent["json"]["url"] == "https://example.com"
    assert sent["json"]["scope"] == "full_page"
    assert sent["json"]["style"] == "modern"
    assert sent["json"]["critique_findings"] == [{"id": "f1"}]


@pytest.mark.asyncio
async def test_get_redesign_status(client, mock_analysis) -> None:
    mock_analysis["state"]["payload"] = {"job_id": "rd-1", "status": "running"}
    r = await client.get("/api/redesign/jobs/rd-1")
    assert r.status_code == 200
    assert r.json()["status"] == "running"
    assert mock_analysis["captured"][-1]["path"] == "/api/designs/jobs/rd-1"


@pytest.mark.asyncio
async def test_get_redesign_result(client, mock_analysis) -> None:
    mock_analysis["state"]["payload"] = {"job_id": "rd-1", "variants": ["modern"]}
    r = await client.get("/api/redesign/jobs/rd-1/result")
    assert r.status_code == 200
    assert mock_analysis["captured"][-1]["path"] == "/api/designs/jobs/rd-1/result"


@pytest.mark.asyncio
async def test_download_variant_html_streams_text(client, mock_analysis) -> None:
    mock_analysis["state"]["raw"] = "<html><body>redesigned</body></html>"
    mock_analysis["state"]["content_type"] = "text/html; charset=utf-8"
    r = await client.get("/api/redesign/jobs/rd-1/variants/modern/html")
    assert r.status_code == 200, r.text
    assert "redesigned" in r.text
    assert r.headers["content-type"].startswith("text/html")
    assert mock_analysis["captured"][-1]["path"] == "/api/designs/jobs/rd-1/variants/modern/html"


@pytest.mark.asyncio
async def test_redesign_downstream_error_surfaces_status(client, mock_analysis) -> None:
    mock_analysis["state"]["status"] = 404
    mock_analysis["state"]["payload"] = {"detail": "no such job"}
    r = await client.get("/api/redesign/jobs/missing")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Browser scope-switch adoption — ?active_tenant= on the redesign downloads
# --------------------------------------------------------------------------- #
# `/jobs/{id}/html` renders in an <iframe src> and `/jobs/{id}/pdf` behind an
# <a href download>; neither can send X-Active-Tenant-Id, so the switch rides
# `?active_tenant=` — read by `get_tenant_id`, the single effective-tenant dep.
# These handlers do NOT run an ownership query (they are thin proxies to
# html_render), so there is no 200-vs-404 delta to assert here. The observable
# signal is the SCOPE_SWITCH audit line + the strict 422: BOTH are emitted by
# `get_tenant_id` and by NOTHING else on this path, so a header-only resolution
# would silently ignore the param and produce neither.

_AUDIT_LOGGER = "w2v_orchestrator.audit"


def _roles_seeded_redesign_app(*, tenant_id: str, roles: list[str]) -> FastAPI:
    app = FastAPI()
    app.add_middleware(_identity_seeder(tenant_id=tenant_id, roles=roles, subject_id=str(uuid4())))
    app.include_router(redesign_router)
    return app


async def _redesign_get(app: FastAPI, path: str):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        return await c.get(path)


def _scope_switches(caplog) -> list:
    return [rec for rec in caplog.records if rec.message.startswith("SCOPE_SWITCH")]


@pytest.mark.parametrize(
    ("suffix", "raw", "content_type", "needle"),
    [("html", "<html><body>redesigned</body></html>", "text/html; charset=utf-8", "redesigned")],
)
@pytest.mark.asyncio
async def test_redesign_download_active_tenant_param_super_vs_non_super(
    mock_analysis, caplog, suffix, raw, content_type, needle
) -> None:
    """Same URL, two role sets: only super_admin's `active_tenant` re-scopes.

    Non-super's param must be ignored silently (probe-resistant) — asserted by
    the ABSENCE of the audit line, not by a status code, since the proxy returns
    200 either way.
    """
    mock_analysis["state"]["raw"] = raw
    mock_analysis["state"]["content_type"] = content_type

    tenant_a, tenant_b = str(uuid4()), str(uuid4())
    path = f"/api/redesign/jobs/rd-1/{suffix}?active_tenant={tenant_b}"

    super_app = _roles_seeded_redesign_app(tenant_id=tenant_a, roles=["super_admin"])
    with caplog.at_level(logging.INFO, logger=_AUDIT_LOGGER):
        allowed = await _redesign_get(super_app, path)
        switched = _scope_switches(caplog)
    assert allowed.status_code == 200, allowed.text
    assert needle in allowed.text  # the artifact really streamed through
    assert len(switched) == 1
    msg = switched[0].getMessage()
    assert tenant_a in msg and tenant_b in msg
    assert f"/api/redesign/jobs/rd-1/{suffix}" in msg

    caplog.clear()
    plain_app = _roles_seeded_redesign_app(tenant_id=tenant_a, roles=["tenant_admin"])
    with caplog.at_level(logging.INFO, logger=_AUDIT_LOGGER):
        ignored = await _redesign_get(plain_app, path)
    assert ignored.status_code == 200
    assert not _scope_switches(caplog)  # no switch happened → no escape, no audit


@pytest.mark.parametrize("suffix", ["html"])
@pytest.mark.asyncio
async def test_redesign_download_malformed_active_tenant_is_422(mock_analysis, suffix) -> None:
    """A super_admin's unparseable `active_tenant` is rejected by the dep itself."""
    app = _roles_seeded_redesign_app(tenant_id=str(uuid4()), roles=["super_admin"])
    resp = await _redesign_get(app, f"/api/redesign/jobs/rd-1/{suffix}?active_tenant=not-a-uuid")
    assert resp.status_code == 422
    assert resp.json()["detail"] == "malformed active tenant: not a UUID"


# --------------------------------------------------------------------------- #
# X-Effective-Tenant-Id — the browser caller's real tenant must reach downstream
# --------------------------------------------------------------------------- #
# A browser user authenticates to the orchestrator with a Keycloak JWT, so
# `get_caller_license_token()` is empty and `_proxy` falls back to the
# orchestrator's OWN service-account DOWNSTREAM_LICENSE_TOKEN. Downstream
# introspects that to the SA tenant — so without a side-channel, every customer's
# redesign/critique job lands in the single service-account tenant (measured:
# 891/894 rows in html_render's redesign_jobs carry the SA tenant).
#
# The side-channel is `X-Effective-Tenant-Id`, honoured by w2v_common's
# AuthMiddleware ONLY when the introspected subject is in AUTH_DELEGATING_SUBJECTS
# (the SA subject is; a customer's is not — see the unit test below).
#
# The value MUST come from `get_tenant_id` (TenantIdDep) — the single server-side
# effective-tenant chokepoint — never from an inbound client-supplied header.


def _effective_tenant_of(mock) -> str | None:
    return mock["captured"][-1]["headers"].get("x-effective-tenant-id")


@pytest.mark.asyncio
async def test_redesign_forwards_caller_tenant_as_effective_tenant(mock_analysis) -> None:
    """A JWT browser caller's own tenant rides X-Effective-Tenant-Id."""
    tenant = str(uuid7())
    app = _roles_seeded_redesign_app(tenant_id=tenant, roles=["tenant_admin"])
    mock_analysis["state"]["payload"] = {"job_id": "rd-1", "status": "running"}
    r = await _redesign_get(app, "/api/redesign/jobs/rd-1")
    assert r.status_code == 200, r.text

    headers = mock_analysis["captured"][-1]["headers"]
    assert headers.get("x-effective-tenant-id") == tenant
    # The SA license token still rides X-Tenant-Id: Bearer <token> — that is the
    # ONLY channel w2v_common's AuthMiddleware accepts a license token on
    # (`_extract_license_token`), so it must NOT be removed.
    assert headers.get("x-tenant-id", "").startswith("Bearer ")
    assert headers.get("authorization", "").startswith("Bearer ")


@pytest.mark.asyncio
async def test_critique_forwards_caller_tenant_as_effective_tenant(mock_analysis) -> None:
    """Same for the critique proxy (analysis_service peer), on a POST."""
    tenant = str(uuid7())

    app = FastAPI()
    app.add_middleware(
        _identity_seeder(tenant_id=tenant, roles=["tenant_admin"], subject_id=str(uuid7()))
    )
    app.include_router(critique_router)
    mock_analysis["state"]["payload"] = {"job_id": "c-1", "status": "pending"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        r = await c.post("/api/critique", json={"url": "https://example.com"})
    assert r.status_code == 200, r.text
    assert _effective_tenant_of(mock_analysis) == tenant


@pytest.mark.asyncio
async def test_effective_tenant_follows_the_scope_switch(mock_analysis) -> None:
    """A super_admin's `?active_tenant=B` must re-scope the DOWNSTREAM job too.

    This is why the value comes from `get_tenant_id` (TenantIdDep) and not from
    the raw `_tenant_id_var` ContextVar / raw claim: the ContextVar holds the
    admin's OWN claim, so a hook-based stamp would file the job under A while the
    orchestrator's own gates ran under B.
    """
    tenant_a, tenant_b = str(uuid7()), str(uuid7())
    app = _roles_seeded_redesign_app(tenant_id=tenant_a, roles=["super_admin"])
    mock_analysis["state"]["payload"] = {"job_id": "rd-1", "status": "running"}
    r = await _redesign_get(app, f"/api/redesign/jobs/rd-1?active_tenant={tenant_b}")
    assert r.status_code == 200, r.text
    assert _effective_tenant_of(mock_analysis) == tenant_b


@pytest.mark.asyncio
async def test_non_super_caller_cannot_steer_the_effective_tenant(mock_analysis) -> None:
    """Security: a client-supplied `?active_tenant=` from a non-super caller is
    ignored — the effective tenant stays the caller's own server-side claim."""
    tenant_a, tenant_b = str(uuid7()), str(uuid7())
    app = _roles_seeded_redesign_app(tenant_id=tenant_a, roles=["tenant_admin"])
    mock_analysis["state"]["payload"] = {"job_id": "rd-1", "status": "running"}
    r = await _redesign_get(app, f"/api/redesign/jobs/rd-1?active_tenant={tenant_b}")
    assert r.status_code == 200, r.text
    assert _effective_tenant_of(mock_analysis) == tenant_a


# ── EVERY proxied route must stamp it — mutation guard ────────────────────────
# The single-route tests above are not enough: dropping `effective_tenant=` from
# ONE handler killed none of them (mutation M3, `redesign.variant.html`). Each
# entry below is one reachable proxied route; the coverage guard right after
# fails if a route is ever added to either router without being listed here, so
# a new proxy endpoint cannot silently ship unstamped.

_PROXIED_ROUTES: list[tuple[str, str, dict | None]] = [
    ("POST", "/api/critique", {"url": "https://example.com"}),
    ("GET", "/api/critique", None),
    ("GET", "/api/critique/c-1", None),
    ("GET", "/api/critique/c-1/report/pdf", None),
    ("GET", "/api/critique/compare?job_id_1=c-1&job_id_2=c-2", None),
    ("POST", "/api/brand/from-url", {"url": "https://example.com"}),
    ("POST", "/api/brand/extract", {"analysis": {}}),
    ("POST", "/api/brand/icp-from-url", {"url": "https://example.com"}),
    ("POST", "/api/redesign", {"url": "https://example.com"}),
    ("GET", "/api/redesign/jobs/rd-1", None),
    ("GET", "/api/redesign/jobs/rd-1/result", None),
    ("GET", "/api/redesign/jobs/rd-1/html", None),
    ("GET", "/api/redesign/jobs/rd-1/pdf", None),
    ("GET", "/api/redesign/jobs/rd-1/variants/modern/html", None),
]


def test_the_route_table_above_covers_every_proxied_route() -> None:
    """Guard the guard: every route on both proxy routers must be exercised.

    Compares the parametrize table to the routers' own declared paths, so adding
    a proxy endpoint without a case here is a RED test rather than a silent hole.
    """
    declared = {
        (method, route.path)
        for router in (critique_router, redesign_router, brand_router)
        for route in router.routes
        if isinstance(route, APIRoute)
        for method in route.methods or ()
    }
    # Map each concrete test URL back to its templated path.
    covered = {
        ("POST", "/api/critique"),
        ("GET", "/api/critique"),
        ("GET", "/api/critique/{job_id}"),
        ("GET", "/api/critique/{job_id}/report/pdf"),
        ("GET", "/api/critique/compare"),
        ("POST", "/api/brand/from-url"),
        ("POST", "/api/brand/extract"),
        ("POST", "/api/brand/icp-from-url"),
        ("POST", "/api/redesign"),
        ("GET", "/api/redesign/jobs/{job_id}"),
        ("GET", "/api/redesign/jobs/{job_id}/result"),
        ("GET", "/api/redesign/jobs/{job_id}/html"),
        ("GET", "/api/redesign/jobs/{job_id}/pdf"),
        ("GET", "/api/redesign/jobs/{job_id}/variants/{style}/html"),
    }
    assert declared == covered, f"un-exercised proxy routes: {declared - covered}"
    assert len(_PROXIED_ROUTES) == len(covered)


def _both_routers_app(*, tenant_id: str, roles: list[str]) -> FastAPI:
    subject_id = str(uuid7())

    app = FastAPI()
    app.add_middleware(_identity_seeder(tenant_id=tenant_id, roles=roles, subject_id=subject_id))
    app.include_router(critique_router)
    app.include_router(redesign_router)
    app.include_router(brand_router)
    return app


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [("POST", "/api/critique", {"url": "https://example.com"})],
    ids=lambda v: str(v),
)
@pytest.mark.asyncio
async def test_every_proxied_route_stamps_the_effective_tenant(
    mock_analysis, method, path, body
) -> None:
    """Assert on the REAL intercepted ``httpx.Request`` that leaves the process."""
    tenant = str(uuid7())
    app = _both_routers_app(tenant_id=tenant, roles=["tenant_admin"])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        r = await c.request(method, path, json=body)
    assert r.status_code == 200, r.text

    sent = mock_analysis["captured"][-1]["headers"]
    assert sent.get("x-effective-tenant-id") == tenant, f"{method} {path} lost the stamp"
    # The license token still rides its canonical channel on every route.
    assert sent.get("x-tenant-id", "").startswith("Bearer ")


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [("POST", "/api/critique", {"url": "https://example.com"})],
    ids=lambda v: str(v),
)
@pytest.mark.asyncio
async def test_every_proxied_route_follows_the_scope_switch(
    mock_analysis, method, path, body
) -> None:
    """...and every route resolves it through `get_tenant_id`, not the raw claim."""
    tenant_a, tenant_b = str(uuid7()), str(uuid7())
    app = _both_routers_app(tenant_id=tenant_a, roles=["super_admin"])
    sep = "&" if "?" in path else "?"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        r = await c.request(method, f"{path}{sep}active_tenant={tenant_b}", json=body)
    assert r.status_code == 200, r.text
    assert mock_analysis["captured"][-1]["headers"].get("x-effective-tenant-id") == tenant_b


@pytest.mark.parametrize("no_tenant", [None])
def test_forward_headers_refuses_a_missing_tenant(no_tenant) -> None:
    """No caller tenant → REFUSE the outbound call, don't stringify the absence.

    The stringification is the failure mode this has always guarded: the natural
    one-liner ``headers[EFFECTIVE_TENANT_HEADER] = str(effective_tenant)`` sends
    ``X-Effective-Tenant-Id: None``, which downstream's
    ``_apply_effective_tenant`` reads as a NON-EMPTY value differing from the
    token's tenant — so a delegating SA would re-scope every unauthenticated
    background call to a tenant literally named "None", and a non-delegating one
    would emit a fleet-wide rejection warning on every request.

    w2v-common 0.67.0 STRENGTHENED this from "omit the header" to "raise": with
    no ambient tenant bound, ``get_identity_headers()`` refuses outright, because
    an outbound call may not carry a blank tenant at all. The assertion is
    therefore inverted to pin the refusal — asserting the old silent-omission
    behaviour would now be asserting a weaker contract than the one we ship.
    """
    from src.api.routes._proxy import forward_headers

    clear_identity_context()
    with pytest.raises(MissingTenantError):
        forward_headers(no_tenant)


def test_forward_headers_stamps_the_effective_tenant_when_one_is_bound() -> None:
    """The positive control for the test above: a bound tenant IS forwarded.

    Without this, the refusal test alone would still pass if ``forward_headers``
    raised unconditionally.
    """
    from src.api.routes._proxy import forward_headers

    caller = str(uuid7())
    set_identity_context(tenant_id=caller, license_id="lic", subject_id="sub")
    try:
        headers = forward_headers(caller)
        present = {k.lower(): v for k, v in headers.items()}
        assert present["x-effective-tenant-id"] == caller
        assert "None" not in present.values()
    finally:
        clear_identity_context()


# NOT coverage of this diff — a CONTRACT test pinned on w2v_common. This asserts
# the downstream half of the mechanism (`AuthMiddleware._apply_effective_tenant`)
# that makes stamping the header unconditionally safe. It stays green if the
# `_proxy` change is deleted entirely, by design: it exists so that a w2v_common
# bump which quietly widens delegation (or drops the AUTH_DELEGATING_SUBJECTS
# gate) fails HERE rather than in prod. The coverage of this diff is
# `test_every_proxied_route_stamps_the_effective_tenant` above.
def test_w2v_common_contract_only_delegating_subjects_may_assert_effective_tenant(
    monkeypatch,
) -> None:
    """Negative control against the REAL downstream contract.

    `_proxy` stamps X-Effective-Tenant-Id unconditionally (as
    ``identity_event_hook`` does fleet-wide). That is safe because w2v_common's
    `AuthMiddleware._apply_effective_tenant` honours the header ONLY when the
    INTROSPECTED subject is in AUTH_DELEGATING_SUBJECTS. When a caller presents
    their OWN license token, the introspected subject is the customer's — not the
    orchestrator SA — so the header is dropped and the token's own tenant stands.
    """
    from starlette.requests import Request as StarletteRequest
    from w2v_common.auth import middleware as wm

    # Subjects stay `str` (they are free-form introspection subjects); tenants are
    # real `uuid.UUID` values — w2v-common 0.67.0 types both the `tenant_id`
    # parameter and the returned effective tenant as `uuid.UUID`, not `str`.
    sa_subject, customer_subject = str(uuid7()), str(uuid7())
    sa_tenant, victim_tenant = uuid7(), uuid7()
    monkeypatch.setattr(wm, "_delegating_subjects", lambda: frozenset({sa_subject}))

    def _req(effective: UUID) -> StarletteRequest:
        # The header is still WIRE text; only the parsed value is a UUID.
        return StarletteRequest(
            {
                "type": "http",
                "method": "GET",
                "path": "/",
                "headers": [(b"x-effective-tenant-id", str(effective).encode())],
            }
        )

    # The orchestrator SA delegating → the effective tenant is honoured.
    assert (
        wm._apply_effective_tenant(_req(victim_tenant), sa_subject, sa_tenant, sa_subject, None)[0]
        == victim_tenant
    )

    # A customer's own license token → subject is not delegating → header dropped.
    assert (
        wm._apply_effective_tenant(
            _req(victim_tenant), customer_subject, sa_tenant, customer_subject, None
        )[0]
        == sa_tenant
    )
