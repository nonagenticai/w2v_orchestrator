"""Admin tier — POST /api/admin/runs/{run_id}/cancel (owner decision 2026-10-20).

A super_admin stops ANY tenant's run from the cross-tenant operator pane. The
tenant is resolved through the ``run_tenant`` SECURITY DEFINER resolver and the
session bound to it (never ``row_security=off``), the cancel is the tenant route's
own ``cancel_run_tree``, and every cancel writes a ``run.cancel`` audit row.
"""

from __future__ import annotations

from uuid import UUID, uuid4, uuid7

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes.admin import router as admin_router
from src.database.models import AdminAuditLog, Deliverable, Project, Run
from tests.api.routes._licence_stubs import override_entitled

pytestmark = pytest.mark.asyncio


class _NoLiveTask:
    """No pod owns the run's task → the mark-cancelled fallback path runs."""

    def __init__(self) -> None:
        self.broadcast: list[UUID] = []

    async def cancel(self, run_id: UUID) -> bool:
        return False

    async def broadcast_interrupt(self, run_id: UUID) -> None:
        self.broadcast.append(run_id)


def _build_app(db_session_factory, *, roles: list[str], dispatcher) -> tuple[FastAPI, str]:
    app = FastAPI()
    app.state.session_factory = db_session_factory
    app.state.dispatcher = dispatcher
    app.include_router(admin_router)
    override_entitled(app)
    subject_id = str(uuid4())
    own_tenant = str(uuid4())  # the operator's OWN tenant — never the run's

    class _SeedRoles(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = own_tenant
            request.state.roles = roles
            request.state.subject_id = subject_id
            return await call_next(request)

    app.add_middleware(_SeedRoles)
    return app, subject_id


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


async def _seed_run(session, tenant_id: UUID, *, status: str) -> UUID:
    proj = Project(
        id=uuid7(),
        tenant_id=tenant_id,
        owner_subject_id=uuid7(),
        visibility="tenant",
        type="generic",
        title="other tenant's film",
    )
    session.add(proj)
    deliv = Deliverable(id=uuid7(), project_id=proj.id, name="d", format="16x9")
    session.add(deliv)
    await session.flush()
    run = Run(id=uuid7(), deliverable_id=deliv.id, blueprint_id="generic_v1", status=status)
    session.add(run)
    await session.commit()
    return run.id


async def _status(db_session_factory, run_id: UUID) -> str:
    async with db_session_factory() as s:
        return (await s.execute(select(Run.status).where(Run.id == run_id))).scalar_one()


async def _audit_rows(db_session_factory) -> list[AdminAuditLog]:
    async with db_session_factory() as s:
        return list(
            (
                await s.execute(select(AdminAuditLog).where(AdminAuditLog.action == "run.cancel"))
            ).scalars()
        )


async def test_non_super_admin_is_403_and_nothing_changes(db_session, db_session_factory) -> None:
    run_id = await _seed_run(db_session, uuid4(), status="paused")
    app, _ = _build_app(db_session_factory, roles=["tenant_admin"], dispatcher=_NoLiveTask())
    async with _client(app) as c:
        r = await c.post(f"/api/admin/runs/{run_id}/cancel")
    assert r.status_code == 403, r.text
    assert await _status(db_session_factory, run_id) == "paused"
    assert await _audit_rows(db_session_factory) == []


async def test_super_admin_cancels_another_tenants_paused_run(
    db_session, db_session_factory
) -> None:
    other_tenant = uuid4()
    run_id = await _seed_run(db_session, other_tenant, status="paused")
    dispatcher = _NoLiveTask()
    app, actor = _build_app(db_session_factory, roles=["super_admin"], dispatcher=dispatcher)
    async with _client(app) as c:
        r = await c.post(f"/api/admin/runs/{run_id}/cancel")
    assert r.status_code == 204, r.text
    assert await _status(db_session_factory, run_id) == "cancelled"
    assert dispatcher.broadcast == [run_id]  # cross-pod interrupt delivered
    [row] = await _audit_rows(db_session_factory)
    assert str(row.actor_subject_id) == actor
    assert row.target_tenant_id == other_tenant
    assert row.detail == {"run_id": str(run_id), "prior_status": "paused"}


async def test_already_terminal_run_is_idempotent_204(db_session, db_session_factory) -> None:
    """Same convention as POST /api/runs/{id}/cancel: terminal is not an error."""
    run_id = await _seed_run(db_session, uuid4(), status="completed")
    dispatcher = _NoLiveTask()
    app, _ = _build_app(db_session_factory, roles=["super_admin"], dispatcher=dispatcher)
    async with _client(app) as c:
        r = await c.post(f"/api/admin/runs/{run_id}/cancel")
    assert r.status_code == 204, r.text
    assert await _status(db_session_factory, run_id) == "completed"  # not rewritten
    assert dispatcher.broadcast == []
    [row] = await _audit_rows(db_session_factory)
    assert row.detail["prior_status"] == "completed"


async def test_unknown_run_is_404(db_session_factory) -> None:
    app, _ = _build_app(db_session_factory, roles=["super_admin"], dispatcher=_NoLiveTask())
    async with _client(app) as c:
        r = await c.post(f"/api/admin/runs/{uuid7()}/cancel")
    assert r.status_code == 404, r.text
    assert await _audit_rows(db_session_factory) == []
