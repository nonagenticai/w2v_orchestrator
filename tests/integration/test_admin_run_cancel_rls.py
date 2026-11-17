"""POST /api/admin/runs/{run_id}/cancel under the PROD RLS posture.

``tests/api/routes/test_admin_run_cancel.py`` runs with RLS NOT enforced, so it
cannot tell a route that binds the session to the run's tenant from one that
never binds at all — both see every row. Here every connection the app opens is a
non-owner ``NOBYPASSRLS`` role with ``TENANT_RLS_MODE=enforce``, the operator's
own claim tenant (A) is on the identity ContextVar exactly as AuthMiddleware puts
it there, and the run belongs to tenant B. The only way the run row can move to
``cancelled`` is the route's ``run_tenant()`` SECURITY DEFINER resolve followed by
``bind_session_tenant(B)``.

Pattern: ``tests/integration/test_step_ledger_kill_and_resume.py``
(``_enforce_rls`` + the runtime-role sessionmaker).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any
from uuid import UUID, uuid4, uuid7

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event, select, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.middleware.base import BaseHTTPMiddleware
from w2v_common.auth.identity_context import clear_identity_context, set_identity_context

from src.api.routes.admin import router as admin_router
from src.config import clear_settings_cache
from src.database.models import AdminAuditLog, Deliverable, Project, Run
from src.database.rls_events import install_rls_tenant_rebind
from src.database.session import bind_session_tenant
from tests.api.routes._licence_stubs import override_entitled
from tests.conftest import TEST_DATABASE_URL, require_createrole

pytestmark = pytest.mark.asyncio

SessionFactory = async_sessionmaker[AsyncSession]


class _NoLiveTask:
    """No pod owns the run's task → the mark-cancelled fallback path runs."""

    def __init__(self) -> None:
        self.broadcast: list[UUID] = []

    async def cancel(self, run_id: UUID) -> bool:
        return False

    async def broadcast_interrupt(self, run_id: UUID) -> None:
        self.broadcast.append(run_id)


@pytest.fixture(autouse=True)
def _enforce_rls(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The prod RLS posture for the app's own binders; restored afterwards."""
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    clear_settings_cache()
    clear_identity_context()
    install_rls_tenant_rebind()  # idempotent — asgi installs it at startup in prod
    yield
    clear_settings_cache()
    clear_identity_context()


@pytest.fixture
async def runtime_sessions(db_session_factory: SessionFactory) -> AsyncIterator[SessionFactory]:
    """A sessionmaker whose EVERY connection is a non-owner ``NOBYPASSRLS`` role."""
    role = f"cancel_runtime_{uuid4().hex[:12]}"
    async with db_session_factory() as owner:
        schema = (await owner.execute(text("SELECT current_schema()"))).scalar_one()
        try:
            await owner.execute(text(f'CREATE ROLE "{role}" NOLOGIN NOSUPERUSER NOBYPASSRLS'))
        except ProgrammingError as exc:  # insufficient_privilege → no CREATEROLE
            await owner.rollback()
            require_createrole(exc)  # fail under CI, skip for a locally-restricted role
        for grant in (
            f'GRANT "{role}" TO CURRENT_USER',
            f'GRANT USAGE ON SCHEMA "{schema}" TO "{role}"',
            f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA "{schema}" TO "{role}"',
            f'GRANT USAGE ON ALL SEQUENCES IN SCHEMA "{schema}" TO "{role}"',
            # run_tenant() is not PUBLIC-executable; prod grants it to the runtime role.
            f'GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA "{schema}" TO "{role}"',
        ):
            await owner.execute(text(grant))
        await owner.commit()

    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    @event.listens_for(engine.sync_engine, "connect")
    def _become_runtime_role(dbapi_connection: Any, _record: object) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute(f'SET ROLE "{role}"')
        cursor.close()
        dbapi_connection.commit()

    try:
        yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    finally:
        await engine.dispose()
        async with db_session_factory() as owner:
            for stmt in (
                f'REVOKE ALL ON ALL TABLES IN SCHEMA "{schema}" FROM "{role}"',
                f'REVOKE ALL ON ALL SEQUENCES IN SCHEMA "{schema}" FROM "{role}"',
                f'REVOKE ALL ON ALL FUNCTIONS IN SCHEMA "{schema}" FROM "{role}"',
                f'REVOKE ALL ON SCHEMA "{schema}" FROM "{role}"',
                f'REVOKE "{role}" FROM CURRENT_USER',
                f'DROP OWNED BY "{role}"',
                f'DROP ROLE IF EXISTS "{role}"',
            ):
                try:
                    await owner.execute(text(stmt))
                    await owner.commit()
                except ProgrammingError:
                    await owner.rollback()


def _build_app(
    sessions: SessionFactory, *, operator_tenant: UUID, dispatcher
) -> tuple[FastAPI, str]:
    app = FastAPI()
    app.state.session_factory = sessions
    app.state.dispatcher = dispatcher
    app.include_router(admin_router)
    override_entitled(app)
    subject_id = str(uuid4())

    class _SeedIdentity(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = str(operator_tenant)
            request.state.roles = ["super_admin"]
            request.state.subject_id = subject_id
            # AuthMiddleware seeds the CLAIM tenant (A) here in prod — the rebind
            # listener falls back to it, so a route that forgot to bind B would
            # query as A and see nothing.
            set_identity_context(str(operator_tenant), None, subject_id)
            return await call_next(request)

    app.add_middleware(_SeedIdentity)
    return app, subject_id


async def _seed_paused_run(owner: SessionFactory, tenant_id: UUID) -> UUID:
    async with owner() as session:
        proj = Project(
            id=uuid7(),
            tenant_id=tenant_id,
            owner_subject_id=uuid7(),
            visibility="tenant",
            type="generic",
            title="tenant B's film",
        )
        session.add(proj)
        deliv = Deliverable(id=uuid7(), project_id=proj.id, name="d", format="16x9")
        session.add(deliv)
        await session.flush()
        run = Run(id=uuid7(), deliverable_id=deliv.id, blueprint_id="generic_v1", status="paused")
        session.add(run)
        await session.commit()
        return run.id


async def _status_as(runtime: SessionFactory, run_id: UUID, tenant: UUID) -> str | None:
    async with runtime() as session:
        await bind_session_tenant(session, tenant)
        return (
            await session.execute(select(Run.status).where(Run.id == run_id))
        ).scalar_one_or_none()


async def test_super_admin_of_a_cancels_tenant_b_run_under_enforced_rls(
    db_session_factory: SessionFactory, runtime_sessions: SessionFactory
) -> None:
    tenant_a, tenant_b = uuid4(), uuid4()
    run_id = await _seed_paused_run(db_session_factory, tenant_b)

    # Controls: RLS really is enforced on these sessions — B sees its paused run,
    # A sees nothing. Without the second, the cancel assertion below is vacuous.
    assert await _status_as(runtime_sessions, run_id, tenant_b) == "paused"
    assert await _status_as(runtime_sessions, run_id, tenant_a) is None

    dispatcher = _NoLiveTask()
    app, actor = _build_app(runtime_sessions, operator_tenant=tenant_a, dispatcher=dispatcher)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        r = await c.post(f"/api/admin/runs/{run_id}/cancel")
    assert r.status_code == 204, r.text

    clear_identity_context()
    assert await _status_as(runtime_sessions, run_id, tenant_b) == "cancelled"
    assert dispatcher.broadcast == [run_id]

    async with db_session_factory() as owner:
        rows = list(
            (
                await owner.execute(
                    select(AdminAuditLog).where(
                        AdminAuditLog.action == "run.cancel",
                        AdminAuditLog.target_tenant_id == tenant_b,
                    )
                )
            ).scalars()
        )
    [row] = rows
    assert str(row.actor_subject_id) == actor
    assert row.detail == {"run_id": str(run_id), "prior_status": "paused"}


async def test_unknown_run_is_404_under_enforced_rls(
    runtime_sessions: SessionFactory,
) -> None:
    app, _ = _build_app(runtime_sessions, operator_tenant=uuid4(), dispatcher=_NoLiveTask())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        r = await c.post(f"/api/admin/runs/{uuid7()}/cancel")
    assert r.status_code == 404, r.text
