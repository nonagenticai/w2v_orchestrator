"""Admin tier (Phase B, 4th increment) — persistent admin audit log.

Deliberate admin actions are recorded to ``admin_audit_log``:
- PATCH /api/admin/tenants/{id} writes a ``tenant.update`` row (before/after).
- POST  /api/admin/scope-switch writes ``scope.switch`` / ``scope.reset``.
- GET   /api/admin/audit (super_admin) reads them back, newest-first, paginated.

Same per-test-app + middleware-seeded-roles idiom as the sibling admin tests.
"""

from __future__ import annotations

from typing import get_args
from uuid import UUID, uuid4, uuid7

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes.admin import router as admin_router
from src.api.schemas.tenant_audit import AuditAction
from src.database.models import AdminAuditLog, Project
from src.database.repositories.admin_audit_repository import AuditRepository
from tests.api.routes._licence_stubs import override_entitled

pytestmark = pytest.mark.asyncio


def _build_app(db_session_factory, *, tenant_id: str, roles: list[str], subject_id: str) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = db_session_factory
    app.include_router(admin_router)
    # require_license is ON by default and fails CLOSED, so the licence-gated
    # admin read routes would 503 here without a licences client and mask what
    # this file tests. Hand the gate an entitled answer; it stays mounted.
    override_entitled(app)

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


async def _seed_project(session, tenant_id: UUID) -> None:
    session.add(
        Project(
            id=uuid7(),
            tenant_id=tenant_id,
            owner_subject_id=uuid7(),
            visibility="tenant",
            type="generic",
            title="P",
        )
    )
    await session.flush()


# ─── PATCH tenant → audit ────────────────────────────────────────────────


async def test_patch_records_tenant_update_audit(db_session, db_session_factory) -> None:
    target = uuid4()
    actor = str(uuid4())
    await _seed_project(db_session, target)
    await db_session.commit()

    app = _build_app(
        db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"], subject_id=actor
    )
    async with _client(app) as c:
        r = await c.patch(
            f"/api/admin/tenants/{target}",
            json={"name": "Acme", "status": "suspended"},
        )
        assert r.status_code == 200, r.text
        audit = await c.get("/api/admin/audit")
    assert audit.status_code == 200
    body = audit.json()
    assert body["total"] == 1
    entry = body["items"][0]
    assert entry["action"] == "tenant.update"
    assert entry["actor_subject_id"] == actor
    assert entry["target_tenant_id"] == str(target)
    # project-only tenant → no registry row before, so before is null.
    assert entry["detail"]["before"] is None
    assert entry["detail"]["after"] == {"name": "Acme", "status": "suspended"}


async def test_patch_before_after_on_existing_row(db_session, db_session_factory) -> None:
    """A second PATCH captures the prior registry state as ``before``."""
    target = uuid4()
    actor = str(uuid4())
    await _seed_project(db_session, target)
    await db_session.commit()

    app = _build_app(
        db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"], subject_id=actor
    )
    async with _client(app) as c:
        await c.patch(f"/api/admin/tenants/{target}", json={"name": "First"})
        await c.patch(f"/api/admin/tenants/{target}", json={"status": "suspended"})
        audit = await c.get("/api/admin/audit")
    items = audit.json()["items"]  # newest-first
    assert items[0]["detail"]["before"] == {"name": "First", "status": "active"}
    assert items[0]["detail"]["after"] == {"name": "First", "status": "suspended"}


async def test_rejected_patch_writes_no_audit(db_session_factory) -> None:
    """An all-None PATCH is 422 and leaves the audit log empty."""
    app = _build_app(
        db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"], subject_id=str(uuid4())
    )
    async with _client(app) as c:
        r = await c.patch(f"/api/admin/tenants/{uuid4()}", json={})
        assert r.status_code == 422
        audit = await c.get("/api/admin/audit")
    assert audit.json()["total"] == 0


# ─── POST scope-switch → audit ───────────────────────────────────────────


async def test_scope_switch_records_switch(db_session_factory) -> None:
    claim = uuid4()
    actor = str(uuid4())
    to = uuid4()
    app = _build_app(
        db_session_factory, tenant_id=str(claim), roles=["super_admin"], subject_id=actor
    )
    async with _client(app) as c:
        r = await c.post("/api/admin/scope-switch", json={"to_tenant_id": str(to)})
        assert r.status_code == 200, r.text
        entry = r.json()
        assert entry["action"] == "scope.switch"
        assert entry["target_tenant_id"] == str(to)
        assert entry["actor_subject_id"] == actor
        assert entry["detail"]["from_tenant"] == str(claim)
        audit = await c.get("/api/admin/audit")
    assert audit.json()["total"] == 1


async def test_scope_reset_records_reset(db_session_factory) -> None:
    claim = uuid4()
    app = _build_app(
        db_session_factory, tenant_id=str(claim), roles=["super_admin"], subject_id=str(uuid4())
    )
    async with _client(app) as c:
        r = await c.post("/api/admin/scope-switch", json={"to_tenant_id": None})
    assert r.status_code == 200
    entry = r.json()
    assert entry["action"] == "scope.reset"
    assert entry["target_tenant_id"] is None
    assert entry["detail"]["from_tenant"] == str(claim)


async def test_scope_switch_forbidden_for_tenant_admin(db_session_factory) -> None:
    app = _build_app(
        db_session_factory, tenant_id=str(uuid4()), roles=["tenant_admin"], subject_id=str(uuid4())
    )
    async with _client(app) as c:
        r = await c.post("/api/admin/scope-switch", json={"to_tenant_id": str(uuid4())})
    assert r.status_code == 403


# ─── GET audit authz + pagination ────────────────────────────────────────


async def test_audit_forbidden_for_tenant_admin(db_session_factory) -> None:
    app = _build_app(
        db_session_factory, tenant_id=str(uuid4()), roles=["tenant_admin"], subject_id=str(uuid4())
    )
    async with _client(app) as c:
        r = await c.get("/api/admin/audit")
    assert r.status_code == 403


async def test_audit_pagination_and_order(db_session_factory) -> None:
    claim = uuid4()
    app = _build_app(
        db_session_factory, tenant_id=str(claim), roles=["super_admin"], subject_id=str(uuid4())
    )
    async with _client(app) as c:
        # Three deliberate switches → three rows, newest-first.
        targets = [uuid4() for _ in range(3)]
        for t in targets:
            await c.post("/api/admin/scope-switch", json={"to_tenant_id": str(t)})
        page1 = await c.get("/api/admin/audit", params={"limit": 2, "offset": 0})
        page2 = await c.get("/api/admin/audit", params={"limit": 2, "offset": 2})
    b1, b2 = page1.json(), page2.json()
    assert b1["total"] == 3 and b2["total"] == 3
    assert b1["limit"] == 2 and b1["offset"] == 0
    assert len(b1["items"]) == 2 and len(b2["items"]) == 1
    # newest-first: the last-created target is first.
    assert b1["items"][0]["target_tenant_id"] == str(targets[-1])
    assert b2["items"][0]["target_tenant_id"] == str(targets[0])


async def test_audit_validates_limit(db_session_factory) -> None:
    app = _build_app(
        db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"], subject_id=str(uuid4())
    )
    async with _client(app) as c:
        assert (await c.get("/api/admin/audit", params={"limit": 0})).status_code == 422
        assert (await c.get("/api/admin/audit", params={"limit": 201})).status_code == 422


# ─── GET audit filters ───────────────────────────────────────────────────


async def test_audit_filter_by_action_and_tenant(db_session, db_session_factory) -> None:
    claim = uuid4()
    t1, t2, t3 = uuid4(), uuid4(), uuid4()
    await _seed_project(db_session, t3)  # so PATCH targets a project-only tenant
    await db_session.commit()

    app = _build_app(
        db_session_factory, tenant_id=str(claim), roles=["super_admin"], subject_id=str(uuid4())
    )
    async with _client(app) as c:
        await c.post("/api/admin/scope-switch", json={"to_tenant_id": str(t1)})  # scope.switch t1
        await c.post("/api/admin/scope-switch", json={"to_tenant_id": str(t2)})  # scope.switch t2
        await c.post("/api/admin/scope-switch", json={"to_tenant_id": None})  # scope.reset
        await c.patch(f"/api/admin/tenants/{t3}", json={"name": "Acme"})  # tenant.update t3

        by_switch = (await c.get("/api/admin/audit", params={"action": "scope.switch"})).json()
        by_update = (await c.get("/api/admin/audit", params={"action": "tenant.update"})).json()
        by_tenant = (await c.get("/api/admin/audit", params={"tenant_id": str(t1)})).json()

    assert by_switch["total"] == 2
    assert {i["action"] for i in by_switch["items"]} == {"scope.switch"}
    assert by_update["total"] == 1 and by_update["items"][0]["target_tenant_id"] == str(t3)
    assert by_tenant["total"] == 1 and by_tenant["items"][0]["target_tenant_id"] == str(t1)


async def test_audit_filter_by_actor(db_session_factory) -> None:
    actor1, actor2 = str(uuid4()), str(uuid4())
    app1 = _build_app(
        db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"], subject_id=actor1
    )
    app2 = _build_app(
        db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"], subject_id=actor2
    )
    async with _client(app1) as c:
        await c.post("/api/admin/scope-switch", json={"to_tenant_id": str(uuid4())})
    async with _client(app2) as c:
        await c.post("/api/admin/scope-switch", json={"to_tenant_id": str(uuid4())})
        only1 = (await c.get("/api/admin/audit", params={"actor_subject_id": actor1})).json()
        all_rows = (await c.get("/api/admin/audit")).json()
    assert all_rows["total"] == 2
    assert only1["total"] == 1 and only1["items"][0]["actor_subject_id"] == actor1


async def test_audit_invalid_action_filter_is_422(db_session_factory) -> None:
    app = _build_app(
        db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"], subject_id=str(uuid4())
    )
    async with _client(app) as c:
        r = await c.get("/api/admin/audit", params={"action": "bogus"})
    assert r.status_code == 422


async def test_audit_action_filter_accepts_every_stored_action(
    db_session, db_session_factory
) -> None:
    """Every ``AuditAction`` (== the ``ck_admin_audit_action`` CHECK, pinned by
    test_tenant_audit_vocabulary) is a 200 filter that returns exactly its row.

    The frontend found ``offboarding.*`` rejected with 422 by an inline copy of the
    vocabulary that had drifted; this walks the single source so a new action can
    never be unfilterable on the admin pane."""
    actions = get_args(AuditAction)
    # Non-vacuity: the families the frontend filters on are really in the walk.
    for family in ("offboarding.", "music.", "youtube.", "defaults.", "connection."):
        assert any(a.startswith(family) for a in actions), family
    repo = AuditRepository(db_session)
    for action in actions:
        await repo.record(actor_subject_id=uuid7(), action=action, detail={})
    await db_session.commit()

    app = _build_app(
        db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"], subject_id=str(uuid4())
    )
    async with _client(app) as c:
        for action in actions:
            r = await c.get("/api/admin/audit", params={"action": action})
            assert r.status_code == 200, (action, r.text)
            body = r.json()
            assert body["total"] == 1, (action, body)
            assert body["items"][0]["action"] == action


# ─── Repository unit ─────────────────────────────────────────────────────


async def test_repo_record_and_list(db_session) -> None:
    repo = AuditRepository(db_session)
    actor = uuid7()
    await repo.record(actor_subject_id=actor, action="scope.reset", detail={"from_tenant": "x"})
    await repo.record(
        actor_subject_id=actor,
        action="tenant.update",
        target_tenant_id=uuid7(),
        detail={"before": None},
    )
    await db_session.flush()

    rows, total = await repo.list_events(limit=10, offset=0)
    assert total == 2
    assert [r.action for r in rows] == ["tenant.update", "scope.reset"]  # newest-first

    # The rows are really persisted.
    persisted = (await db_session.execute(select(AdminAuditLog.action))).scalars().all()
    assert set(persisted) == {"tenant.update", "scope.reset"}
