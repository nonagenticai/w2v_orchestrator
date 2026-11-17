"""Unit tests for ShutdownMiddleware."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.middleware.shutdown import ShutdownMiddleware


def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(ShutdownMiddleware)
    app.state.shutting_down = False

    @app.get("/ping")
    def ping() -> dict[str, str]:
        return {"ok": "true"}

    @app.post("/echo")
    def echo() -> dict[str, str]:
        return {"ok": "true"}

    return app


def test_normal_post_passes_through() -> None:
    app = _make_app()
    with TestClient(app) as client:
        resp = client.post("/echo")
    assert resp.status_code == 200


def test_post_during_drain_returns_503_with_retry_after() -> None:
    app = _make_app()
    app.state.shutting_down = True
    with TestClient(app) as client:
        resp = client.post("/echo")
    assert resp.status_code == 503
    assert resp.headers["Retry-After"] == "5"
    body = resp.json()
    assert body["code"] == "shutting_down"


def test_get_during_drain_still_passes() -> None:
    app = _make_app()
    app.state.shutting_down = True
    with TestClient(app) as client:
        resp = client.get("/ping")
    assert resp.status_code == 200


@pytest.mark.parametrize("method", ["POST"])
def test_all_drainable_methods_return_503(method: str) -> None:
    app = FastAPI()
    app.add_middleware(ShutdownMiddleware)
    app.state.shutting_down = True

    @app.api_route("/r", methods=["POST", "PUT", "PATCH", "DELETE"])
    def handler() -> dict[str, str]:  # pragma: no cover — middleware blocks
        return {"ok": "true"}

    with TestClient(app) as client:
        resp = client.request(method, "/r")
    assert resp.status_code == 503
