"""Admin tier — GET /api/admin/runs (cross-tenant operator run pane).

The platform "what is running right now, across all tenants" view. Cross-tenant
super-admin read via the ``admin_active_runs`` / ``admin_recent_runs`` SECURITY
DEFINER resolvers; each active row is enriched with ``leasePresent``.
"""

from __future__ import annotations

import datetime
from uuid import UUID, uuid4, uuid7

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from psycopg.errors import InsufficientPrivilege
from sqlalchemy.exc import ProgrammingError
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes._common import get_admin_repo
from src.api.routes.admin import router as admin_router
from src.database.models import Deliverable, Project, Run
from tests.api.routes._licence_stubs import override_entitled

pytestmark = pytest.mark.asyncio


class _FakeDispatcher:
    """lease_present → True for every run (simulates a live execution lease)."""

    async def lease_present(self, run_id: UUID) -> bool:
        return True


def _build_app(db_session_factory, *, tenant_id: str, roles: list[str], dispatcher=None) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = db_session_factory
    app.state.dispatcher = dispatcher
    app.include_router(admin_router)
    # require_license is ON by default and fails CLOSED, so the licence-gated
    # admin read routes would 503 here without a licences client and mask what
    # this file tests. Hand the gate an entitled answer; it stays mounted.
    override_entitled(app)
    subject_id = str(uuid4())

    class _SeedRoles(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = tenant_id
            request.state.roles = roles
            request.state.subject_id = subject_id
            return await call_next(request)

    app.add_middleware(_SeedRoles)
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


async def _seed_run(
    session,
    tenant_id: UUID,
    *,
    status: str,
    title: str,
    current_stage: str | None = None,
    updated_at: datetime.datetime | None = None,
    paused_at: datetime.datetime | None = None,
    pause_reason: str | None = None,
) -> UUID:
    proj = Project(
        id=uuid7(),
        tenant_id=tenant_id,
        owner_subject_id=uuid7(),
        visibility="tenant",
        type="generic",
        title=title,
    )
    session.add(proj)
    deliv = Deliverable(id=uuid7(), project_id=proj.id, name="d", format="16x9")
    session.add(deliv)
    await session.flush()
    run = Run(
        id=uuid7(),
        deliverable_id=deliv.id,
        blueprint_id="generic_v1",
        status=status,
        current_stage=current_stage,
        paused_at=paused_at,
        pause_reason=pause_reason,
    )
    session.add(run)
    await session.flush()
    if updated_at is not None:
        run.updated_at = updated_at
        await session.flush()
    return run.id


async def test_active_runs_cross_tenant_with_lease(db_session, db_session_factory) -> None:
    tenant_a, tenant_b = uuid4(), uuid4()
    running_a = await _seed_run(
        db_session, tenant_a, status="running", title="A-run", current_stage="narrate"
    )
    pending_b = await _seed_run(db_session, tenant_b, status="pending", title="B-pend")
    await _seed_run(db_session, tenant_a, status="completed", title="A-done")
    await db_session.commit()

    app = _build_app(
        db_session_factory,
        tenant_id=str(uuid4()),
        roles=["super_admin"],
        dispatcher=_FakeDispatcher(),
    )
    async with _client(app) as c:
        r = await c.get("/api/admin/runs?status=running")
    assert r.status_code == 200, r.text
    body = r.json()
    by_id = {item["run_id"]: item for item in body["items"]}
    # BOTH tenants' in-flight runs are visible (cross-tenant); the completed one is not.
    assert str(running_a) in by_id
    assert str(pending_b) in by_id
    assert body["total"] == 2
    # Rich row: tenant, project title, stage; leasePresent True for the running row.
    assert by_id[str(running_a)]["tenant_id"] == str(tenant_a)
    assert by_id[str(running_a)]["project_title"] == "A-run"
    assert by_id[str(running_a)]["current_stage"] == "narrate"
    assert by_id[str(running_a)]["lease_present"] is True
    # A pending row is not lease-checked → leasePresent stays null.
    assert by_id[str(pending_b)]["lease_present"] is None


async def test_active_runs_orphan_when_no_lease(db_session, db_session_factory) -> None:
    """A running row whose lease no pod holds is surfaced as leasePresent=False."""
    tenant = uuid4()
    run_id = await _seed_run(db_session, tenant, status="running", title="orphan")
    await db_session.commit()

    class _NoLease:
        async def lease_present(self, run_id: UUID) -> bool:
            return False

    app = _build_app(
        db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"], dispatcher=_NoLease()
    )
    async with _client(app) as c:
        r = await c.get("/api/admin/runs?status=running")
    assert r.status_code == 200, r.text
    item = next(i for i in r.json()["items"] if i["run_id"] == str(run_id))
    assert item["lease_present"] is False


async def test_recent_runs_cross_tenant_ordered(db_session, db_session_factory) -> None:
    tenant_a, tenant_b = uuid4(), uuid4()
    now = datetime.datetime.now(datetime.UTC)
    older = await _seed_run(
        db_session,
        tenant_a,
        status="completed",
        title="older",
        updated_at=now - datetime.timedelta(hours=2),
    )
    newer = await _seed_run(
        db_session,
        tenant_b,
        status="failed",
        title="newer",
        updated_at=now - datetime.timedelta(minutes=5),
    )
    # An in-flight run must NOT appear in the recent (terminal) view.
    await _seed_run(db_session, tenant_a, status="running", title="live")
    await db_session.commit()

    app = _build_app(
        db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"], dispatcher=None
    )
    async with _client(app) as c:
        r = await c.get("/api/admin/runs?status=recent&limit=10")
    assert r.status_code == 200, r.text
    ids = [i["run_id"] for i in r.json()["items"]]
    assert ids[:2] == [str(newer), str(older)], "recent runs ordered by updated_at desc"
    assert str(newer) in ids and str(older) in ids


async def test_recent_runs_respects_limit(db_session, db_session_factory) -> None:
    tenant = uuid4()
    now = datetime.datetime.now(datetime.UTC)
    for i in range(5):
        await _seed_run(
            db_session,
            tenant,
            status="completed",
            title=f"r{i}",
            updated_at=now - datetime.timedelta(minutes=i),
        )
    await db_session.commit()
    app = _build_app(
        db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"], dispatcher=None
    )
    async with _client(app) as c:
        r = await c.get("/api/admin/runs?status=recent&limit=3")
    assert r.status_code == 200, r.text
    assert len(r.json()["items"]) == 3


async def test_runs_requires_super_admin(db_session_factory) -> None:
    app = _build_app(
        db_session_factory, tenant_id=str(uuid4()), roles=["tenant_admin"], dispatcher=None
    )
    async with _client(app) as c:
        r = await c.get("/api/admin/runs?status=running")
    assert r.status_code == 403


class _PermissionDeniedRepo:
    """Simulates the runtime role lacking EXECUTE on the SECURITY DEFINER resolvers
    (the grant drifted / a fresh DB not yet migrated) — psycopg raises
    InsufficientPrivilege, wrapped by SQLAlchemy as ProgrammingError."""

    @staticmethod
    def _denied() -> ProgrammingError:
        return ProgrammingError(
            "SELECT admin_active_runs()",
            {},
            InsufficientPrivilege("permission denied for function"),
        )

    async def list_active_runs(self):  # type: ignore[no-untyped-def]
        raise self._denied()

    async def list_recent_runs(self, limit: int):  # type: ignore[no-untyped-def]
        raise self._denied()


@pytest.mark.parametrize("status_filter", ["running"])
async def test_runs_grant_missing_degrades_to_503(db_session_factory, status_filter: str) -> None:
    """A missing EXECUTE grant must surface as a clean 503, never a raw 500."""
    app = _build_app(
        db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"], dispatcher=None
    )
    app.dependency_overrides[get_admin_repo] = lambda: _PermissionDeniedRepo()
    async with _client(app) as c:
        r = await c.get(f"/api/admin/runs?status={status_filter}")
    assert r.status_code == 503, r.text


async def test_active_runs_include_paused_without_lease_probe(
    db_session, db_session_factory
) -> None:
    """0092: a PAUSED run (awaiting human review — the kind that stalls) is in-flight
    work and must appear in the Running view. It holds no execution lease, so the
    lease probe is skipped for it (``leasePresent`` is None), as for a pending row."""
    tenant = uuid4()
    paused = await _seed_run(db_session, tenant, status="paused", title="P", current_stage="gate")
    await db_session.commit()

    app = _build_app(
        db_session_factory,
        tenant_id=str(uuid4()),
        roles=["super_admin"],
        dispatcher=_FakeDispatcher(),
    )
    async with _client(app) as c:
        r = await c.get("/api/admin/runs?status=running")
    assert r.status_code == 200, r.text
    by_id = {item["run_id"]: item for item in r.json()["items"]}
    assert str(paused) in by_id, "paused runs must be listed as in-flight"
    assert by_id[str(paused)]["status"] == "paused"
    assert by_id[str(paused)].get("lease_present", by_id[str(paused)].get("leasePresent")) is None


async def test_paused_run_reports_when_it_started_waiting(db_session, db_session_factory) -> None:
    """REGRESSION (run 01a1770d, 2026-10-26/27): the pane aged every row from
    ``started_at``, so a run parked at a storyboard review gate for 29h sat in the
    Running view as a 29h-old run and was cancelled as stuck. 0102 returns
    ``paused_at``; the row must carry it for a paused run and only for a paused run."""
    tenant = uuid4()
    waiting_since = datetime.datetime(2026, 10, 26, 7, 15, 21, tzinfo=datetime.UTC)
    paused = await _seed_run(
        db_session,
        tenant,
        status="paused",
        title="gate",
        current_stage="promo_storyboard",
        paused_at=waiting_since,
        pause_reason="checkpoint",
    )
    manual = await _seed_run(
        db_session,
        tenant,
        status="paused",
        title="held",
        paused_at=waiting_since,
        pause_reason="manual",
    )
    running = await _seed_run(db_session, tenant, status="running", title="live")
    # A cancel does not clear runs.paused_at; the recent view must not show it as waiting.
    cancelled = await _seed_run(
        db_session,
        tenant,
        status="cancelled",
        title="gone",
        paused_at=waiting_since,
        pause_reason="checkpoint",
    )
    await db_session.commit()

    app = _build_app(
        db_session_factory,
        tenant_id=str(uuid4()),
        roles=["super_admin"],
        dispatcher=_FakeDispatcher(),
    )
    async with _client(app) as c:
        active = await c.get("/api/admin/runs?status=running")
        recent = await c.get("/api/admin/runs?status=recent&limit=50")
    assert active.status_code == 200, active.text
    assert recent.status_code == 200, recent.text
    by_id = {item["run_id"]: item for item in active.json()["items"]}
    assert datetime.datetime.fromisoformat(by_id[str(paused)]["paused_at"]) == waiting_since
    assert by_id[str(paused)]["pause_reason"] == "checkpoint"
    assert by_id[str(manual)]["pause_reason"] == "manual", "a manual pause is not a review wait"
    assert by_id[str(running)]["paused_at"] is None
    assert by_id[str(running)]["pause_reason"] is None
    gone = next(i for i in recent.json()["items"] if i["run_id"] == str(cancelled))
    assert gone["paused_at"] is None and gone["pause_reason"] is None
