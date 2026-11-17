"""The two signup service doors bypass AuthMiddleware in the REAL app — and only they.

Through ``src.asgi.app`` (no lifespan): with the flag off, a tokenless POST to a
door reaches the route and answers the route's 503 ``signup_not_configured``
(AuthMiddleware would have answered 401 first); a super_admin route, a GET on a
door path, and a look-alike path still meet AuthMiddleware.
"""

from __future__ import annotations

from uuid import uuid7

import pytest
from httpx import ASGITransport, AsyncClient

from src.auth.signup_doors import is_signup_service_door

pytestmark = pytest.mark.asyncio


async def test_door_matcher_is_exact() -> None:
    sid = str(uuid7())
    assert is_signup_service_door("POST", "/api/signup-approvals/intake")
    assert is_signup_service_door("POST", f"/api/signup-approvals/{sid}/telegram-decision")
    assert not is_signup_service_door("GET", "/api/signup-approvals/intake")
    assert not is_signup_service_door("POST", "/api/signup-approvals/a/b/telegram-decision")
    assert not is_signup_service_door("POST", "/api/admin/signup-approvals/invite")
    assert not is_signup_service_door("POST", "/api/signup-approvals/intake/extra")
    assert not is_signup_service_door("POST", "/api/projects")


async def test_doors_reach_the_route_everything_else_meets_auth(monkeypatch) -> None:
    from src.asgi import app

    monkeypatch.delenv("W2V_SIGNUP_ENABLED", raising=False)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        door = await c.post("/api/signup-approvals/intake", json={})
        decision = await c.post(f"/api/signup-approvals/{uuid7()}/telegram-decision", json={})
        admin = await c.get("/api/admin/signup-approvals")
    assert door.status_code == 503 and door.json()["error"] == "signup_not_configured"
    assert decision.status_code == 503
    assert admin.status_code == 401, "super_admin routes stay behind AuthMiddleware"


async def test_enabled_tokenless_door_is_refused_by_the_route_not_auth(monkeypatch) -> None:
    from src.asgi import app

    monkeypatch.setenv("W2V_SIGNUP_ENABLED", "true")
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        r = await c.post("/api/signup-approvals/intake", json={})
    # The route's own flat body — AuthMiddleware's 401 has a different envelope.
    assert r.status_code == 401 and r.json()["error"] == "unauthorized"


async def test_door_bodies_are_capped_before_parsing(monkeypatch) -> None:
    """The doors are public and FastAPI parses JSON before the route's token
    check: a body over 16 KiB is 413'd — on its declared Content-Length before a
    byte is read, and on the streamed count when chunked (no Content-Length)."""
    from src.asgi import app
    from src.auth.signup_doors import SIGNUP_DOOR_BODY_LIMIT_BYTES

    monkeypatch.setenv("W2V_SIGNUP_ENABLED", "true")
    big = b'{"email": "' + b"a" * (SIGNUP_DOOR_BODY_LIMIT_BYTES + 1) + b'"}'

    async def chunked():  # type: ignore[no-untyped-def]
        for i in range(0, len(big), 4096):
            yield big[i : i + 4096]

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    headers = {"Content-Type": "application/json"}
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        declared = await c.post("/api/signup-approvals/intake", content=big, headers=headers)
        streamed = await c.post(
            f"/api/signup-approvals/{uuid7()}/telegram-decision",
            content=chunked(),
            headers=headers,
        )
        small = await c.post("/api/signup-approvals/intake", json={"email": "a@b.test"})
    assert declared.status_code == 413
    assert declared.json()["error"]["code"] == "REQUEST_BODY_TOO_LARGE"
    assert "content-length" not in {k.lower() for k in streamed.request.headers}
    assert streamed.status_code == 413
    # Under the cap the route's own token check answers (the cap changes nothing else).
    assert small.status_code == 401 and small.json()["error"] == "unauthorized"
