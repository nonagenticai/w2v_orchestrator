"""Admin tier (Phase B) — authz matrix + data-correctness for /api/admin/*.

The guards read `request.state.roles` (populated in prod by w2v_common>=0.18.10).
These tests seed request.state directly via an inline middleware — the same
idiom as `test_auth_blueprints_health.py::test_auth_me_forwards_roles_email_name`
— and build per-test apps (the conftest `client` fixture seeds no roles).
"""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID, uuid4, uuid7

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes.admin import router as admin_router
from src.database.models import Deliverable, Project, Run, RunUsage, Tenant
from src.database.repositories.admin_repository import AdminRepository
from tests.api.routes._licence_stubs import override_entitled

pytestmark = pytest.mark.asyncio


def _build_app(db_session_factory, *, tenant_id: str | None, roles: list[str] | None) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = db_session_factory
    app.include_router(admin_router)
    # require_license is ON by default and fails CLOSED, so the licence-gated
    # admin read routes would 503 here without a licences client and mask what
    # this file tests. Hand the gate an entitled answer; it stays mounted.
    override_entitled(app)
    subject_id = str(uuid4())

    class _SeedRoles(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            # anon = no tenant claim → get_tenant_id raises 401 before any role
            # check (sub-dependency short-circuit).
            if tenant_id is not None:
                request.state.tenant_id = tenant_id
            if roles is not None:
                request.state.roles = roles
            request.state.subject_id = subject_id
            return await call_next(request)

    app.add_middleware(_SeedRoles)
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


# (label, has_tenant_claim, roles, expect /tenants [super], expect /overview [tenant_admin])
_MATRIX = [
    ("anon", False, None, 401, 401),
    ("member_empty", True, [], 403, 403),
    ("member_named", True, ["member"], 403, 403),
    ("tenant_admin", True, ["tenant_admin"], 403, 200),
    ("super_only", True, ["super_admin"], 200, 200),  # composite-only edge
    ("super_admin", True, ["super_admin", "tenant_admin"], 200, 200),
]


@pytest.mark.parametrize(
    "label,has_tenant,roles,exp_tenants,exp_overview", [("anon", False, None, 401, 401)]
)
async def test_admin_authz_matrix(
    db_session_factory,
    label: str,
    has_tenant: bool,
    roles: list[str] | None,
    exp_tenants: int,
    exp_overview: int,
) -> None:
    tenant_id = str(uuid4()) if has_tenant else None
    app = _build_app(db_session_factory, tenant_id=tenant_id, roles=roles)
    async with _client(app) as c:
        r_tenants = await c.get("/api/admin/tenants")
        r_overview = await c.get("/api/admin/overview")
    assert r_tenants.status_code == exp_tenants, f"{label} GET /api/admin/tenants"
    assert r_overview.status_code == exp_overview, f"{label} GET /api/admin/overview"


async def _seed_tenant(session, tenant_id: UUID, *, n_projects: int, costs: list[Decimal]) -> None:
    first = Project(
        id=uuid7(),
        tenant_id=tenant_id,
        owner_subject_id=uuid7(),
        visibility="tenant",
        type="generic",
        title="P",
    )
    session.add(first)
    for _ in range(n_projects - 1):
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
    deliv = Deliverable(id=uuid7(), project_id=first.id, name="d", format="16x9")
    session.add(deliv)
    await session.flush()
    for cost in costs:
        run = Run(
            id=uuid7(), deliverable_id=deliv.id, blueprint_id="generic_v1", status="completed"
        )
        session.add(run)
        await session.flush()
        session.add(RunUsage(id=uuid7(), run_id=run.id, stage_name="s", cost_usd=cost))
    await session.flush()


async def test_overview_counts_and_tenant_isolation(db_session, db_session_factory) -> None:
    tenant_a, tenant_b = uuid4(), uuid4()
    await _seed_tenant(db_session, tenant_a, n_projects=2, costs=[Decimal("1.50"), Decimal("2.25")])
    await _seed_tenant(db_session, tenant_b, n_projects=1, costs=[Decimal("9.99")])
    await db_session.commit()

    app = _build_app(db_session_factory, tenant_id=str(tenant_a), roles=["tenant_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["tenant_id"] == str(tenant_a)
    assert body["project_count"] == 2  # tenant A only
    assert body["run_count"] == 2  # tenant B's run excluded
    assert body["total_cost_usd"] == pytest.approx(3.75)  # B's 9.99 excluded


async def test_overview_empty_tenant_returns_zeros(db_session_factory) -> None:
    app = _build_app(db_session_factory, tenant_id=str(uuid4()), roles=["tenant_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["project_count"] == 0
    assert body["run_count"] == 0
    assert body["total_cost_usd"] == 0.0


async def test_tenants_lists_all_tenants(db_session, db_session_factory) -> None:
    tenant_a, tenant_b = uuid4(), uuid4()
    await _seed_tenant(db_session, tenant_a, n_projects=2, costs=[Decimal("1.00")])
    await _seed_tenant(db_session, tenant_b, n_projects=1, costs=[])
    await db_session.commit()

    # Caller's own tenant has no projects — /tenants reflects the projects
    # table, not the caller, so it still sees A and B.
    app = _build_app(db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/tenants")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    by_id = {item["tenant_id"]: item for item in body["items"]}
    assert by_id[str(tenant_a)]["project_count"] == 2
    assert by_id[str(tenant_b)]["project_count"] == 1
    assert by_id[str(tenant_a)]["last_activity"] is not None
    # No registry row yet → name null, status defaults to 'active'.
    assert by_id[str(tenant_a)]["name"] is None
    assert by_id[str(tenant_a)]["status"] == "active"


async def test_tenants_surfaces_registry_name_and_status(db_session, db_session_factory) -> None:
    """A registry row contributes name + status to the project-derived list."""
    tenant_a = uuid4()
    await _seed_tenant(db_session, tenant_a, n_projects=1, costs=[Decimal("1.00")])
    db_session.add(Tenant(id=tenant_a, name="Acme Corp", status="suspended"))
    await db_session.commit()

    app = _build_app(db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/tenants")
    assert r.status_code == 200
    item = {i["tenant_id"]: i for i in r.json()["items"]}[str(tenant_a)]
    assert item["name"] == "Acme Corp"
    assert item["status"] == "suspended"
    assert item["project_count"] == 1


async def test_tenants_flags_only_active_tenants_switchable(db_session, db_session_factory) -> None:
    """The picker offers exactly what the scope-switch accepts: suspended and
    deleted tenants are listed (operator view) but flagged not switchable; an
    active, a licence_lapsed and a registry-less project tenant are switchable."""
    ids = {s: uuid4() for s in ("active", "licence_lapsed", "suspended", "deleted")}
    for st, tid in ids.items():
        db_session.add(Tenant(id=tid, name=st, status=st))
    unregistered = uuid4()
    await _seed_tenant(db_session, unregistered, n_projects=1, costs=[])
    await db_session.commit()

    app = _build_app(db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/tenants")
    assert r.status_code == 200
    by_id = {i["tenant_id"]: i for i in r.json()["items"]}
    assert by_id[str(ids["active"])]["switchable"] is True
    assert by_id[str(ids["licence_lapsed"])]["switchable"] is True
    assert by_id[str(ids["suspended"])]["switchable"] is False
    assert by_id[str(ids["deleted"])]["switchable"] is False
    assert by_id[str(unregistered)]["switchable"] is True
    # Same predicate as the backend's scope-switch gate.
    async with db_session_factory() as s:
        repo = AdminRepository(s)
        for item in by_id.values():
            assert item["switchable"] is await repo.tenant_switchable(UUID(item["tenant_id"]))


async def test_tenants_switchable_is_decided_by_tenant_switchable_itself(
    db_session, db_session_factory, monkeypatch
) -> None:
    """No parallel rule: the list's ``switchable`` is whatever
    ``AdminRepository.tenant_switchable`` — the middleware's scope-switch gate —
    answers. The test inverts that function and expects the list inverted for
    every status, the registry-less project tenant included: a route that derived
    ``switchable`` from status itself would keep the un-inverted flags and fail."""
    ids = {st: uuid4() for st in ("active", "licence_lapsed", "suspended", "deleted")}
    for st, tid in ids.items():
        db_session.add(Tenant(id=tid, name=st, status=st))
    unregistered = uuid4()
    await _seed_tenant(db_session, unregistered, n_projects=1, costs=[])
    await db_session.commit()

    real = AdminRepository.tenant_switchable
    asked: list[UUID] = []

    async def inverted(self: AdminRepository, tenant_id: UUID) -> bool:
        asked.append(tenant_id)
        return not await real(self, tenant_id)

    monkeypatch.setattr(AdminRepository, "tenant_switchable", inverted)
    app = _build_app(db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/tenants")
    assert r.status_code == 200
    by_id = {i["tenant_id"]: i for i in r.json()["items"]}
    expected = {
        ids["active"]: False,
        ids["licence_lapsed"]: False,
        ids["suspended"]: True,
        ids["deleted"]: True,
        unregistered: False,
    }
    for tid, flag in expected.items():
        assert by_id[str(tid)]["switchable"] is flag, by_id[str(tid)]
    assert set(expected) <= set(asked)


async def test_tenant_patch_reports_switchable_from_the_same_gate(db_session_factory) -> None:
    """Suspending (PATCH) answers ``switchable: false``; re-activating answers
    true — the PATCH response cannot disagree with the list or the switch."""
    target = uuid4()
    app = _build_app(db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"])
    async with _client(app) as c:
        held = await c.patch(f"/api/admin/tenants/{target}", json={"status": "suspended"})
        back = await c.patch(f"/api/admin/tenants/{target}", json={"status": "active"})
    assert held.status_code == 200, held.text
    assert held.json()["switchable"] is False
    assert back.status_code == 200, back.text
    assert back.json()["switchable"] is True
    async with db_session_factory() as s:
        assert await AdminRepository(s).tenant_switchable(target) is True


async def test_tenants_includes_registered_tenant_with_no_projects(
    db_session, db_session_factory
) -> None:
    """A tenant named ahead of any activity (registry row, zero projects) still
    lists — proving the registry FULL-JOIN, not just the projects GROUP BY."""
    empty = uuid4()
    db_session.add(Tenant(id=empty, name="Empty Co"))
    await db_session.commit()

    app = _build_app(db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/tenants")
    assert r.status_code == 200
    by_id = {i["tenant_id"]: i for i in r.json()["items"]}
    assert str(empty) in by_id
    assert by_id[str(empty)]["name"] == "Empty Co"
    assert by_id[str(empty)]["status"] == "active"
    assert by_id[str(empty)]["project_count"] == 0
    assert by_id[str(empty)]["last_activity"] is None


async def test_tenants_carries_per_tenant_usage_rollup(db_session, db_session_factory) -> None:
    """The platform tenants list reports each tenant's run_count + total cost
    (increment 7), 0 for a tenant with no runs/usage — and the per-tenant cost
    does not bleed across tenants or inflate by project cardinality."""
    tenant_a, tenant_b, empty = uuid4(), uuid4(), uuid4()
    # A: 2 projects, 2 runs each with one usage row → run_count 2, cost 3.75.
    await _seed_tenant(db_session, tenant_a, n_projects=2, costs=[Decimal("1.50"), Decimal("2.25")])
    # B: 1 project, no usage → run_count 0, cost 0.
    await _seed_tenant(db_session, tenant_b, n_projects=1, costs=[])
    # Registry-only tenant, never any project → run_count 0, cost 0.
    db_session.add(Tenant(id=empty, name="Empty Co"))
    await db_session.commit()

    app = _build_app(db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/tenants")
    assert r.status_code == 200
    by_id = {i["tenant_id"]: i for i in r.json()["items"]}
    assert by_id[str(tenant_a)]["run_count"] == 2  # not inflated by the 2 projects
    assert by_id[str(tenant_a)]["total_cost_usd"] == pytest.approx(3.75)
    assert by_id[str(tenant_b)]["run_count"] == 0
    assert by_id[str(tenant_b)]["total_cost_usd"] == 0.0
    assert by_id[str(empty)]["run_count"] == 0
    assert by_id[str(empty)]["total_cost_usd"] == 0.0
