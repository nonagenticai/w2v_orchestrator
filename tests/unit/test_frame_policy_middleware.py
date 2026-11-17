"""The redesign html previews must be frameable by the SPA; nothing else may be.

Prod symptom (website_review run 01a09f82-7179-73a5-a23a-40c15d148b37): the Studio's
``<iframe sandbox="allow-scripts" src="/api/redesign/jobs/{id}/html">`` stayed blank
because every orchestrator response carried ``X-Frame-Options: DENY``, stamped by the
canonical ``w2v_common`` SecurityHeadersMiddleware (which OVERWRITES the header, so a
route cannot opt out on its own).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient
from w2v_common.middleware import SecurityHeadersMiddleware

from src.api.routes.redesign import router as redesign_router
from src.middleware.security import FramePolicyMiddleware, is_frameable_path


def _app() -> FastAPI:
    app = FastAPI()

    async def _html_stream():
        yield b"<html><body>redesigned</body></html>"

    @app.get("/api/redesign/jobs/{job_id}/html")
    async def primary(job_id: str) -> StreamingResponse:
        return StreamingResponse(_html_stream(), media_type="text/html")

    @app.get("/api/redesign/jobs/{job_id}/variants/{style}/html")
    async def variant(job_id: str, style: str) -> StreamingResponse:
        return StreamingResponse(_html_stream(), media_type="text/html")

    @app.get("/api/redesign/jobs/{job_id}/pdf")
    async def pdf(job_id: str) -> PlainTextResponse:
        return PlainTextResponse("pdf")

    @app.get("/api/projects")
    async def projects() -> PlainTextResponse:
        return PlainTextResponse("[]")

    # Same registration ORDER as create_app + src/asgi.py: canonical first (inner),
    # extra_middleware after (outer).
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(FramePolicyMiddleware)
    return app


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://t") as c:
        yield c


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/redesign/jobs/rd-1/html"])
async def test_redesign_html_previews_are_same_origin_frameable(client, path: str) -> None:
    r = await client.get(path)
    assert r.status_code == 200
    assert r.headers["x-frame-options"] == "SAMEORIGIN"
    assert r.headers["content-security-policy"] == "frame-ancestors 'self'"
    # exactly one X-Frame-Options — a duplicate DENY would still block the frame
    assert r.headers.get_list("x-frame-options") == ["SAMEORIGIN"]
    # the rest of the canonical headers survive
    assert r.headers["x-content-type-options"] == "nosniff"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/redesign/jobs/rd-1/pdf"])
async def test_everything_else_stays_deny(client, path: str) -> None:
    r = await client.get(path)
    assert r.status_code == 200
    assert r.headers["x-frame-options"] == "DENY"
    assert "content-security-policy" not in r.headers


def test_pattern_matches_exactly_the_redesign_html_routes() -> None:
    """Pin the path pattern to the router's REAL route paths — no drift either way."""
    paths = {route.path for route in redesign_router.routes if isinstance(route, APIRoute)}
    html_routes = {p for p in paths if p.endswith("/html")}
    assert html_routes == {
        "/api/redesign/jobs/{job_id}/html",
        "/api/redesign/jobs/{job_id}/variants/{style}/html",
    }
    for p in paths:
        concrete = p.replace("{job_id}", "rd-1").replace("{style}", "modern")
        assert is_frameable_path(concrete) is (p in html_routes), p
    assert not is_frameable_path("/api/redesign/jobs/a/b/html")
    assert not is_frameable_path("/api/redesign/jobs/rd-1/html/extra")


def test_frame_policy_is_outer_of_the_canonical_security_headers() -> None:
    from src.asgi import app

    order = [getattr(m.cls, "__name__", "") for m in app.user_middleware]
    assert order.index("FramePolicyMiddleware") < order.index("SecurityHeadersMiddleware")
