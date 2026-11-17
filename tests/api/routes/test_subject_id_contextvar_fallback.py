"""Regression: get_subject_id falls back to w2v_common ContextVar.

AuthMiddleware (w2v_common) only stores subject_id in a ContextVar via
set_identity_context — it never assigns request.state.subject_id. Before
this fallback existed, every authenticated request to a SubjectIdDep route
returned 401 "missing subject_id claim" in production.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware
from w2v_common.auth.identity_context import (
    clear_identity_context,
    set_identity_context,
)

from src.api.routes._common import SubjectIdDep, TenantIdDep

pytestmark = pytest.mark.asyncio


def _make_app(*, set_state_subject: bool, contextvar_subject: str | None) -> FastAPI:
    app = FastAPI()

    class _SeedFromContextVar(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.tenant_id = str(uuid4())
            if set_state_subject:
                request.state.subject_id = str(uuid4())
            if contextvar_subject is not None:
                set_identity_context(
                    tenant_id=str(request.state.tenant_id),
                    license_id="lic",
                    subject_id=contextvar_subject,
                )
            try:
                return await call_next(request)
            finally:
                clear_identity_context()

    app.add_middleware(_SeedFromContextVar)

    @app.get("/probe")
    async def probe(tenant_id: TenantIdDep, subject_id: SubjectIdDep) -> dict[str, str]:
        return {"tenant_id": str(tenant_id), "subject_id": str(subject_id)}

    return app


async def _get(app: FastAPI) -> tuple[int, dict | None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        r = await c.get("/probe")
    return r.status_code, (
        r.json() if r.headers.get("content-type", "").startswith("application/json") else None
    )


async def test_subject_id_falls_back_to_contextvar_when_state_unset() -> None:
    expected = str(uuid4())
    app = _make_app(set_state_subject=False, contextvar_subject=expected)
    status, body = await _get(app)
    assert status == 200
    assert body is not None
    assert body["subject_id"] == expected


async def test_subject_id_state_wins_over_contextvar() -> None:
    contextvar_value = str(uuid4())
    app = _make_app(set_state_subject=True, contextvar_subject=contextvar_value)
    status, body = await _get(app)
    assert status == 200
    assert body is not None
    assert body["subject_id"] != contextvar_value


async def test_subject_id_401_when_neither_state_nor_contextvar_set() -> None:
    app = _make_app(set_state_subject=False, contextvar_subject=None)
    status, body = await _get(app)
    assert status == 401
    assert body is not None
    assert "missing subject_id" in body["detail"]


async def test_subject_id_400ish_when_contextvar_value_is_not_uuid() -> None:
    app = _make_app(set_state_subject=False, contextvar_subject="not-a-uuid")
    status, body = await _get(app)
    assert status == 401
    assert body is not None
    assert "malformed subject_id" in body["detail"]
