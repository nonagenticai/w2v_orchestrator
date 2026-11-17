"""Middleware-wide persistent request audit (the step before Phase D RLS).

``AuditMiddleware`` persists one ``request_audit`` row per mutation / auth-failure
/ cross-tenant access, off the hot path and fail-safe. These tests mount it OUTER
to a seed-identity middleware (mirroring the real Audit-outer-to-Auth arrangement,
which is the load-bearing mechanic: an outer BaseHTTPMiddleware reads request.state
that an inner one set, after call_next returns). The read side is GET
/api/admin/request-audit (super_admin), tested directly against seeded rows.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4, uuid7

import pytest
from fastapi import APIRouter, FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.api.routes.admin import router as admin_router
from src.database.models import RequestAudit
from src.middleware.audit import AuditMiddleware
from tests.api.routes._licence_stubs import override_entitled

pytestmark = pytest.mark.asyncio


class _Identity:
    """Minimal stand-in for w2v_common's User (only ``sub`` is read here)."""

    def __init__(self, sub: str) -> None:
        self.sub = sub


def _toy_router() -> APIRouter:
    r = APIRouter()

    @r.get("/api/projects")
    async def list_projects() -> dict[str, bool]:
        return {"ok": True}

    @r.post("/api/projects")
    async def create_project() -> dict[str, bool]:
        return {"created": True}

    @r.get("/api/forbidden")
    async def forbidden() -> dict[str, bool]:
        raise HTTPException(403, "nope")

    @r.post("/api/projects/boom")
    async def boom_post() -> dict[str, bool]:
        raise RuntimeError(
            "kaboom at postgres://svc:hunter2@db.test/x"  # pragma: allowlist secret
        )

    @r.get("/api/projects/boom")
    async def boom_get() -> dict[str, bool]:
        raise RuntimeError("kaboom")

    return r


def _build_audited_app(
    session_factory,
    *,
    tenant_id: str,
    subject_id: str,
    roles: list[str],
) -> FastAPI:
    """App with a toy router, a seed-identity middleware, and AuditMiddleware
    mounted OUTER to the seed (added last = outermost)."""
    app = FastAPI()
    if session_factory is not None:
        app.state.session_factory = session_factory
    app.include_router(_toy_router())

    class _Seed(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = tenant_id
            request.state.subject_id = subject_id
            request.state.roles = roles
            request.state.user = _Identity(subject_id)
            request.state.request_id = "test-req-id"
            return await call_next(request)

    app.add_middleware(_Seed)  # inner
    app.add_middleware(AuditMiddleware)  # outer — sees state set by _Seed
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


async def _rows(session_factory) -> list[RequestAudit]:
    async with session_factory() as s:
        return list((await s.execute(select(RequestAudit))).scalars().all())


@pytest.fixture(autouse=True)
async def _clean_request_audit(db_session_factory):
    async with db_session_factory() as s:
        await s.execute(text("TRUNCATE request_audit"))
        await s.commit()
    yield


# ─── middleware persistence ──────────────────────────────────────────────


async def test_mutation_persists_row(db_session_factory) -> None:
    tenant, subject = str(uuid4()), str(uuid4())
    app = _build_audited_app(
        db_session_factory, tenant_id=tenant, subject_id=subject, roles=["member"]
    )
    async with _client(app) as c:
        resp = await c.post("/api/projects")
    assert resp.status_code == 200
    rows = await _rows(db_session_factory)
    assert len(rows) == 1
    row = rows[0]
    assert row.event == "mutation"
    assert row.method == "POST"
    assert row.resource == "project"
    assert row.cross_tenant is False
    assert row.effective_tenant_id == UUID(tenant)
    assert row.claim_tenant_id == UUID(tenant)
    assert row.actor_subject_id == UUID(subject)
    assert row.request_id == "test-req-id"
    assert row.status_code == 200


async def test_auth_failure_persists_row(db_session_factory) -> None:
    tenant, subject = str(uuid4()), str(uuid4())
    app = _build_audited_app(
        db_session_factory, tenant_id=tenant, subject_id=subject, roles=["member"]
    )
    async with _client(app) as c:
        resp = await c.get("/api/forbidden")
    assert resp.status_code == 403
    rows = await _rows(db_session_factory)
    assert len(rows) == 1
    assert rows[0].event == "auth_failure"
    assert rows[0].status_code == 403


async def test_cross_tenant_read_persists_row(db_session_factory) -> None:
    tenant, subject, other = str(uuid4()), str(uuid4()), str(uuid4())
    app = _build_audited_app(
        db_session_factory, tenant_id=tenant, subject_id=subject, roles=["super_admin"]
    )
    async with _client(app) as c:
        resp = await c.get("/api/projects", headers={"X-Active-Tenant-Id": other})
    assert resp.status_code == 200
    rows = await _rows(db_session_factory)
    assert len(rows) == 1
    row = rows[0]
    assert row.event == "read"
    assert row.cross_tenant is True
    assert row.effective_tenant_id == UUID(other)
    assert row.claim_tenant_id == UUID(tenant)
    assert row.detail.get("from_tenant") == tenant


async def test_plain_same_tenant_get_persists_nothing(db_session_factory) -> None:
    tenant, subject = str(uuid4()), str(uuid4())
    app = _build_audited_app(
        db_session_factory, tenant_id=tenant, subject_id=subject, roles=["member"]
    )
    async with _client(app) as c:
        resp = await c.get("/api/projects")
    assert resp.status_code == 200
    assert await _rows(db_session_factory) == []


async def test_non_super_active_tenant_header_is_not_cross_tenant(db_session_factory) -> None:
    """A non-super caller sending X-Active-Tenant-Id is probe-ignored: a plain GET
    stays a no-op (no scope-switch, no row)."""
    tenant, subject, other = str(uuid4()), str(uuid4()), str(uuid4())
    app = _build_audited_app(
        db_session_factory, tenant_id=tenant, subject_id=subject, roles=["member"]
    )
    async with _client(app) as c:
        resp = await c.get("/api/projects", headers={"X-Active-Tenant-Id": other})
    assert resp.status_code == 200
    assert await _rows(db_session_factory) == []


async def test_db_write_failure_does_not_break_request(db_session_factory, monkeypatch) -> None:
    async def _boom(self, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("db down")

    monkeypatch.setattr(
        "src.database.repositories.request_audit_repository.RequestAuditRepository.record",
        _boom,
    )
    tenant, subject = str(uuid4()), str(uuid4())
    app = _build_audited_app(
        db_session_factory, tenant_id=tenant, subject_id=subject, roles=["member"]
    )
    async with _client(app) as c:
        resp = await c.post("/api/projects")
    # The request still succeeds even though the audit write blew up.
    assert resp.status_code == 200
    assert resp.json() == {"created": True}
    assert await _rows(db_session_factory) == []


# ─── unhandled exceptions (the 5xx blind spot) ───────────────────────────


async def test_unhandled_exception_persists_500_row_and_reraises(db_session_factory) -> None:
    """A handler that raises must (a) still propagate the ORIGINAL exception and
    (b) leave a durable 500 row. Before 2026-08-23 `call_next` had no `except`,
    so the exception bypassed the persist entirely and the table could not hold
    a single 5xx."""
    tenant, subject = str(uuid4()), str(uuid4())
    app = _build_audited_app(
        db_session_factory, tenant_id=tenant, subject_id=subject, roles=["member"]
    )
    async with _client(app) as c:
        with pytest.raises(RuntimeError) as excinfo:
            await c.post("/api/projects/boom")
    # Propagated unchanged — not swallowed, not re-wrapped.
    assert type(excinfo.value) is RuntimeError
    assert excinfo.value.args[0].startswith("kaboom at ")

    rows = await _rows(db_session_factory)
    assert len(rows) == 1
    row = rows[0]
    assert row.status_code == 500
    assert row.event == "mutation"
    assert row.method == "POST"
    assert row.request_id == "test-req-id"
    assert row.effective_tenant_id == UUID(tenant)
    assert row.detail["exception_type"] == "RuntimeError"
    # Type + message only — and the DSN password never reaches the column.
    message = row.detail["exception_message"]
    assert isinstance(message, str)
    assert message.startswith("kaboom at postgres://[redacted]")
    assert "hunter2" not in message  # pragma: allowlist secret


async def test_unhandled_exception_on_get_is_recorded_as_read(db_session_factory) -> None:
    """A failing GET is not a mutation, not an auth failure and not cross-tenant,
    so only the 5xx arm of `should_record` keeps it. Without that arm the most
    common shape of server error would still vanish."""
    tenant, subject = str(uuid4()), str(uuid4())
    app = _build_audited_app(
        db_session_factory, tenant_id=tenant, subject_id=subject, roles=["member"]
    )
    async with _client(app) as c:
        with pytest.raises(RuntimeError):
            await c.get("/api/projects/boom")
    rows = await _rows(db_session_factory)
    assert len(rows) == 1
    assert rows[0].status_code == 500
    assert rows[0].event == "read"


async def test_audit_write_failure_does_not_mask_the_original_exception(
    db_session_factory, monkeypatch
) -> None:
    """If persisting the 500 itself blows up, the caller must still see the
    ORIGINAL RuntimeError — never the audit machinery's own error."""

    async def _boom(self, **kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("audit down")

    monkeypatch.setattr(
        "src.database.repositories.request_audit_repository.RequestAuditRepository.record",
        _boom,
    )
    tenant, subject = str(uuid4()), str(uuid4())
    app = _build_audited_app(
        db_session_factory, tenant_id=tenant, subject_id=subject, roles=["member"]
    )
    async with _client(app) as c:
        with pytest.raises(RuntimeError) as excinfo:
            await c.post("/api/projects/boom")
    assert type(excinfo.value) is RuntimeError
    assert "audit down" not in str(excinfo.value)
    assert await _rows(db_session_factory) == []


async def test_audit_classification_failure_does_not_mask_the_original_exception(
    db_session_factory, monkeypatch
) -> None:
    """`_persist` swallows its own DB errors, but everything BEFORE it (tenant
    resolution, header reads, message scrubbing) can raise too — and on the
    exception path a substitute error would silently replace the real one. The
    guard in `_record_unhandled` is what stops that."""

    def _boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("classifier down")

    monkeypatch.setattr("src.middleware.audit.resolve_active_tenant", _boom)
    tenant, subject = str(uuid4()), str(uuid4())
    app = _build_audited_app(
        db_session_factory, tenant_id=tenant, subject_id=subject, roles=["member"]
    )
    async with _client(app) as c:
        with pytest.raises(RuntimeError) as excinfo:
            await c.post("/api/projects/boom")
    assert type(excinfo.value) is RuntimeError
    assert "classifier down" not in str(excinfo.value)


async def test_cancellation_at_the_middleware_boundary_writes_nothing(db_session_factory) -> None:
    """A cancellation delivered to `dispatch`'s own await (client disconnect /
    shutdown) is a control-flow signal, not a 500: it must propagate untouched
    and leave NO row. Driven at the middleware boundary rather than by raising
    inside a route, because an inner BaseHTTPMiddleware converts a route-raised
    CancelledError into `RuntimeError("No response returned.")` — which IS a
    genuine server error and is correctly recorded as one."""
    tenant, subject = str(uuid4()), str(uuid4())
    app = _build_audited_app(
        db_session_factory, tenant_id=tenant, subject_id=subject, roles=["member"]
    )
    middleware = AuditMiddleware(app=app)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/projects",
            "raw_path": b"/api/projects",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "app": app,
            "state": {"tenant_id": tenant, "roles": ["member"], "request_id": "test-req-id"},
        }
    )

    async def _cancelled(_request: Request) -> Response:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await middleware.dispatch(request, _cancelled)
    assert await _rows(db_session_factory) == []


async def test_missing_session_factory_is_noop(db_session_factory) -> None:
    tenant, subject = str(uuid4()), str(uuid4())
    app = _build_audited_app(None, tenant_id=tenant, subject_id=subject, roles=["member"])
    async with _client(app) as c:
        resp = await c.post("/api/projects")
    assert resp.status_code == 200
    # No factory → nothing written anywhere; the table stays empty.
    assert await _rows(db_session_factory) == []


# ─── read endpoint (GET /api/admin/request-audit) ────────────────────────


def _build_admin_app(
    session_factory, *, tenant_id: str, subject_id: str, roles: list[str]
) -> FastAPI:
    """Admin app WITHOUT AuditMiddleware (so reads don't pollute the table)."""
    app = FastAPI()
    app.state.session_factory = session_factory
    app.include_router(admin_router)
    # The route carries LICENSE_GATE_ONLY (like /audit); hand the gate an
    # entitled answer — it stays mounted, this file tests the read itself.
    override_entitled(app)

    class _Seed(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = tenant_id
            request.state.subject_id = subject_id
            request.state.roles = roles
            return await call_next(request)

    app.add_middleware(_Seed)
    return app


async def _seed_audit_rows(session_factory, tenant: UUID, actor: UUID) -> None:
    base = datetime.now(UTC)
    async with session_factory() as s:
        s.add_all(
            [
                RequestAudit(
                    id=uuid7(),
                    created_at=base - timedelta(seconds=3),
                    actor_subject_id=actor,
                    claim_tenant_id=tenant,
                    effective_tenant_id=tenant,
                    cross_tenant=False,
                    event="mutation",
                    method="POST",
                    path="/api/projects",
                    resource="project",
                    status_code=201,
                    ip="1.2.3.4",
                    duration_ms=12.3,
                    credential_fingerprint="abc",
                    request_id="r1",
                    detail={},
                ),
                RequestAudit(
                    id=uuid7(),
                    created_at=base - timedelta(seconds=2),
                    actor_subject_id=actor,
                    claim_tenant_id=tenant,
                    effective_tenant_id=tenant,
                    cross_tenant=False,
                    event="auth_failure",
                    method="GET",
                    path="/api/forbidden",
                    resource="forbidden",
                    status_code=403,
                    ip="1.2.3.4",
                    duration_ms=1.1,
                    credential_fingerprint=None,
                    request_id="r2",
                    detail={},
                ),
            ]
        )
        await s.commit()


async def test_read_endpoint_lists_and_filters(db_session_factory) -> None:
    tenant, subject = uuid4(), uuid4()
    await _seed_audit_rows(db_session_factory, tenant, subject)
    app = _build_admin_app(
        db_session_factory, tenant_id=str(tenant), subject_id=str(subject), roles=["super_admin"]
    )
    async with _client(app) as c:
        resp = await c.get("/api/admin/request-audit")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        # newest-first
        assert body["items"][0]["event"] == "auth_failure"

        # filter by event
        resp = await c.get("/api/admin/request-audit", params={"event": "mutation"})
        assert resp.status_code == 200
        assert resp.json()["total"] == 1
        assert resp.json()["items"][0]["method"] == "POST"

        # filter by min_status (only the 403)
        resp = await c.get("/api/admin/request-audit", params={"min_status": 400})
        assert resp.json()["total"] == 1
        assert resp.json()["items"][0]["status_code"] == 403


async def test_read_endpoint_requires_super_admin(db_session_factory) -> None:
    tenant, subject = uuid4(), uuid4()
    app = _build_admin_app(
        db_session_factory, tenant_id=str(tenant), subject_id=str(subject), roles=["tenant_admin"]
    )
    async with _client(app) as c:
        resp = await c.get("/api/admin/request-audit")
    assert resp.status_code == 403
