"""Admin tier (Phase B, 2nd increment) — X-Active-Tenant-Id scope-switch.

A `super_admin` may act INTO another tenant by sending `X-Active-Tenant-Id`.
Resolution happens in `get_tenant_id` (the single chokepoint every tenant-scoped
route depends on), so `/api/admin/overview` — which scopes to the effective
tenant via `TenantAdminDep` — proves the global re-scope. Matrix:

  super_admin + valid foreign header  → 200, scoped to the FOREIGN tenant + audit
  super_admin + no header             → 200, scoped to own claim
  super_admin + malformed header      → 422
  tenant_admin + foreign header       → 200, scoped to OWN claim (ignored, no escape)
  super_admin + header == own claim   → 200, own claim, NO audit line

Same seed/middleware idiom as `test_admin_authz.py` (per-test apps; the conftest
`client` fixture seeds no roles).

Two later sections cover the second channel the same switch may ride —
`?active_tenant=`, for requests the browser issues itself (EventSource, <img>,
<a download>, <iframe>), which cannot set a header. `get_tenant_id` reads BOTH,
and the last section pins that the ROUTER-LEVEL gates see it too.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from uuid import UUID, uuid4, uuid7

import pytest
from fastapi import APIRouter, Depends, FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes import _common
from src.api.routes._common import (
    TenantIdDep,
    apply_tenant_rls,
    require_known_tenant,
    require_license,
)
from src.api.routes.admin import router as admin_router
from src.clients.schemas.licenses import ActiveLicenseView
from src.database.models import Deliverable, Project, Run, RunUsage, Tenant
from tests.api.routes._licence_stubs import EntitledLicenses, override_entitled

pytestmark = pytest.mark.asyncio


def _build_app(db_session_factory, *, tenant_id: str, roles: list[str]) -> FastAPI:
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
            request.state.tenant_id = tenant_id
            request.state.roles = roles
            request.state.subject_id = subject_id
            return await call_next(request)

    app.add_middleware(_SeedRoles)
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


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


async def test_super_admin_switches_into_foreign_tenant(db_session, db_session_factory) -> None:
    """super_admin + valid foreign X-Active-Tenant-Id → /overview scoped to B."""
    tenant_a, tenant_b = uuid4(), uuid4()
    await _seed_tenant(db_session, tenant_a, n_projects=2, costs=[Decimal("1.50")])
    await _seed_tenant(db_session, tenant_b, n_projects=3, costs=[Decimal("4.00"), Decimal("2.00")])
    await db_session.commit()

    app = _build_app(
        db_session_factory, tenant_id=str(tenant_a), roles=["super_admin", "tenant_admin"]
    )
    async with _client(app) as c:
        r = await c.get("/api/admin/overview", headers={"X-Active-Tenant-Id": str(tenant_b)})
    assert r.status_code == 200
    body = r.json()
    assert body["tenant_id"] == str(tenant_b)  # effective tenant = B, not the A claim
    assert body["project_count"] == 3
    assert body["run_count"] == 2
    assert body["total_cost_usd"] == pytest.approx(6.00)


async def test_super_admin_no_header_scopes_to_own_claim(db_session, db_session_factory) -> None:
    tenant_a, tenant_b = uuid4(), uuid4()
    await _seed_tenant(db_session, tenant_a, n_projects=2, costs=[Decimal("1.50")])
    await _seed_tenant(db_session, tenant_b, n_projects=3, costs=[Decimal("4.00")])
    await db_session.commit()

    app = _build_app(
        db_session_factory, tenant_id=str(tenant_a), roles=["super_admin", "tenant_admin"]
    )
    async with _client(app) as c:
        r = await c.get("/api/admin/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["tenant_id"] == str(tenant_a)
    assert body["project_count"] == 2


async def test_super_admin_malformed_header_is_422(db_session_factory) -> None:
    app = _build_app(db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/overview", headers={"X-Active-Tenant-Id": "not-a-uuid"})
    assert r.status_code == 422


async def test_tenant_admin_cannot_escape_own_tenant(db_session, db_session_factory) -> None:
    """A non-super caller sending the header is silently ignored — no escape, no 422."""
    tenant_a, tenant_b = uuid4(), uuid4()
    await _seed_tenant(db_session, tenant_a, n_projects=2, costs=[Decimal("1.50")])
    await _seed_tenant(db_session, tenant_b, n_projects=5, costs=[Decimal("9.99")])
    await db_session.commit()

    app = _build_app(db_session_factory, tenant_id=str(tenant_a), roles=["tenant_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/overview", headers={"X-Active-Tenant-Id": str(tenant_b)})
    assert r.status_code == 200
    body = r.json()
    assert body["tenant_id"] == str(tenant_a)  # ignored → still scoped to A
    assert body["project_count"] == 2


async def test_tenant_admin_malformed_header_ignored_not_422(db_session_factory) -> None:
    """Probe-resistance: a non-super caller's malformed header is ignored, not 422."""
    app = _build_app(db_session_factory, tenant_id=str(uuid4()), roles=["tenant_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/overview", headers={"X-Active-Tenant-Id": "not-a-uuid"})
    assert r.status_code == 200


async def test_scope_switch_emits_audit_line(db_session, db_session_factory, caplog) -> None:
    tenant_a, tenant_b = uuid4(), uuid4()
    await _seed_tenant(db_session, tenant_a, n_projects=1, costs=[])
    await _seed_tenant(db_session, tenant_b, n_projects=1, costs=[])
    await db_session.commit()

    app = _build_app(db_session_factory, tenant_id=str(tenant_a), roles=["super_admin"])
    with caplog.at_level(logging.INFO, logger="w2v_orchestrator.audit"):
        async with _client(app) as c:
            r = await c.get("/api/admin/overview", headers={"X-Active-Tenant-Id": str(tenant_b)})
    assert r.status_code == 200
    switch_logs = [rec for rec in caplog.records if rec.message.startswith("SCOPE_SWITCH")]
    assert len(switch_logs) == 1
    msg = switch_logs[0].getMessage()
    assert str(tenant_a) in msg  # from_tenant
    assert str(tenant_b) in msg  # to_tenant


async def test_super_admin_switch_into_unknown_tenant_is_404(db_session_factory) -> None:
    """Fail-closed: a super_admin may not act into a tenant with no registry row
    and no projects (unknown UUID) — 404, not silent empty data."""
    app = _build_app(db_session_factory, tenant_id=str(uuid4()), roles=["super_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/overview", headers={"X-Active-Tenant-Id": str(uuid4())})
    assert r.status_code == 404


async def test_super_admin_switch_into_suspended_tenant_is_404(
    db_session, db_session_factory
) -> None:
    """A suspended registry tenant is blocked even though it has projects —
    the registry row is authoritative over the project-existence fallback."""
    tenant_a, tenant_b = uuid4(), uuid4()
    await _seed_tenant(db_session, tenant_b, n_projects=2, costs=[Decimal("1.00")])
    db_session.add(Tenant(id=tenant_b, name="Suspended Co", status="suspended"))
    await db_session.commit()

    app = _build_app(db_session_factory, tenant_id=str(tenant_a), roles=["super_admin"])
    async with _client(app) as c:
        r = await c.get("/api/admin/overview", headers={"X-Active-Tenant-Id": str(tenant_b)})
    assert r.status_code == 404


async def test_same_tenant_header_emits_no_audit(db_session, db_session_factory, caplog) -> None:
    """Header == own claim is a no-op switch — scoped to self, no audit noise."""
    tenant_a = uuid4()
    await _seed_tenant(db_session, tenant_a, n_projects=1, costs=[])
    await db_session.commit()

    app = _build_app(db_session_factory, tenant_id=str(tenant_a), roles=["super_admin"])
    with caplog.at_level(logging.INFO, logger="w2v_orchestrator.audit"):
        async with _client(app) as c:
            r = await c.get("/api/admin/overview", headers={"X-Active-Tenant-Id": str(tenant_a)})
    assert r.status_code == 200
    assert r.json()["tenant_id"] == str(tenant_a)
    assert not [rec for rec in caplog.records if rec.message.startswith("SCOPE_SWITCH")]


# ── header-less scope-switch via the `active_tenant` query param ───────────────
# EventSource cannot set the X-Active-Tenant-Id header — and neither can an
# <img>/<video>/<a download>/<iframe> load — so the same super_admin scope-switch
# may ride the `active_tenant` query param. There is exactly ONE dep that reads
# both channels (`get_tenant_id` / `TenantIdDep`); a tiny probe route mounted on
# it proves the resolution without needing Redis/streaming. The router-level
# gates resolve through that same dep — see the section at the bottom of this
# file, which is the test that would have caught the prod 404s.


def _build_sse_probe_app(*, tenant_id: str, roles: list[str]) -> FastAPI:
    app = FastAPI()
    subject_id = str(uuid4())

    @app.get("/probe")
    async def _probe(tenant: TenantIdDep) -> dict[str, str]:
        return {"tenant_id": str(tenant)}

    class _SeedRoles(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = tenant_id
            request.state.roles = roles
            request.state.subject_id = subject_id
            return await call_next(request)

    app.add_middleware(_SeedRoles)
    return app


async def test_sse_super_admin_switches_via_query_param() -> None:
    tenant_a, tenant_b = uuid4(), uuid4()
    app = _build_sse_probe_app(tenant_id=str(tenant_a), roles=["super_admin"])
    async with _client(app) as c:
        r = await c.get(f"/probe?active_tenant={tenant_b}")
    assert r.status_code == 200
    assert r.json()["tenant_id"] == str(tenant_b)  # effective = foreign tenant B


async def test_sse_no_param_scopes_to_own_claim() -> None:
    tenant_a = uuid4()
    app = _build_sse_probe_app(tenant_id=str(tenant_a), roles=["super_admin"])
    async with _client(app) as c:
        r = await c.get("/probe")
    assert r.status_code == 200
    assert r.json()["tenant_id"] == str(tenant_a)


async def test_sse_non_super_query_param_ignored() -> None:
    """Probe-resistance: a non-super caller's active_tenant param is ignored."""
    tenant_a, tenant_b = uuid4(), uuid4()
    app = _build_sse_probe_app(tenant_id=str(tenant_a), roles=["tenant_admin"])
    async with _client(app) as c:
        r = await c.get(f"/probe?active_tenant={tenant_b}")
    assert r.status_code == 200
    assert r.json()["tenant_id"] == str(tenant_a)  # ignored → no escape


async def test_sse_super_admin_malformed_query_param_is_422() -> None:
    app = _build_sse_probe_app(tenant_id=str(uuid4()), roles=["super_admin"])
    async with _client(app) as c:
        r = await c.get("/probe?active_tenant=not-a-uuid")
    assert r.status_code == 422


async def test_sse_header_still_wins_over_query_param() -> None:
    """If both are somehow present the header takes precedence (header-first)."""
    tenant_a, tenant_b, tenant_c = uuid4(), uuid4(), uuid4()
    app = _build_sse_probe_app(tenant_id=str(tenant_a), roles=["super_admin"])
    async with _client(app) as c:
        r = await c.get(
            f"/probe?active_tenant={tenant_c}",
            headers={"X-Active-Tenant-Id": str(tenant_b)},
        )
    assert r.status_code == 200
    assert r.json()["tenant_id"] == str(tenant_b)


# ── the ROUTER-LEVEL gates must see the switch too ─────────────────────────────
# THE REGRESSION THIS FILE EXISTS TO PREVENT. The tenant-scoped data routers are
# mounted in `src/asgi.py` with router-level dependencies — require_known_tenant
# → apply_tenant_rls → require_license — that each resolve the effective tenant
# themselves. While the query-param channel lived in a SEPARATE dep that only the
# handlers adopted, a `?active_tenant=` request resolved to TWO different tenants
# in one request: the handler saw the switched tenant, the gates saw the admin's
# own claim. apply_tenant_rls then bound Postgres `app.current_tenant` to the
# wrong tenant, RLS hid the row from the handler's ownership SELECT, and every
# caption/artifact/redesign download 404'd in prod — while the handler-only unit
# tests above stayed green, because they build probe apps WITHOUT these gates.
#
# So this probe app mounts the gate list exactly as asgi.py does, and asserts the
# switch is honoured BY THE GATES: the tenant apply_tenant_rls binds, and
# require_known_tenant admitting a tenant that is known while the CLAIM is not.


def _build_gated_probe_app(db_session_factory, *, tenant_id: str, roles: list[str]) -> FastAPI:
    """A probe route mounted with `src/asgi.py`'s real `_tenant_gate` list."""
    app = FastAPI()
    app.state.session_factory = db_session_factory
    probe_router = APIRouter()

    @probe_router.get("/api/probe")
    async def _probe(tenant: TenantIdDep) -> dict[str, str]:
        return {"tenant_id": str(tenant)}

    # Same deps, same order as asgi.py's `_tenant_gate`.
    app.include_router(
        probe_router,
        dependencies=[
            Depends(require_known_tenant),
            Depends(apply_tenant_rls),
            Depends(require_license),
        ],
    )
    # require_license is ON by default and fails CLOSED, so without a licenses
    # client it would 503 every probe here and mask what this file tests. Hand it
    # an entitled answer: the gate stays mounted in its real position and order,
    # and the assertions below stay about the SCOPE SWITCH, not entitlement.
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


async def test_router_level_gates_see_query_param_scope_switch(
    db_session, db_session_factory, monkeypatch
) -> None:
    """`?active_tenant=B` must be the tenant the ROUTER-LEVEL deps resolve.

    Tenant B is provisioned (has projects); the admin's own claim A is NOT. So:
      * require_known_tenant returning 200 (not 403 "tenant not provisioned")
        proves the fail-closed gate resolved B, not the A claim.
      * the tenant apply_tenant_rls hands to bind_session_tenant is asserted
        directly — that GUC is what RLS filters the handler's query by.
    Header-only resolution in `get_tenant_id` fails BOTH assertions (403, and A).
    """
    tenant_a, tenant_b = uuid4(), uuid4()
    await _seed_tenant(db_session, tenant_b, n_projects=1, costs=[])
    await db_session.commit()

    bound: list[UUID | None] = []

    async def _spy(session, tenant_id):  # type: ignore[no-untyped-def]
        bound.append(tenant_id)

    # apply_tenant_rls calls bind_session_tenant through this module's namespace.
    monkeypatch.setattr(_common, "bind_session_tenant", _spy)

    app = _build_gated_probe_app(db_session_factory, tenant_id=str(tenant_a), roles=["super_admin"])
    async with _client(app) as c:
        r = await c.get(f"/api/probe?active_tenant={tenant_b}")

    # require_known_tenant (enforce, the default) admitted B — A is unprovisioned.
    assert r.status_code == 200, r.text
    assert r.json()["tenant_id"] == str(tenant_b)  # the handler agrees
    # apply_tenant_rls bound the RLS GUC to the SWITCHED tenant, not the claim.
    assert bound == [tenant_b]


async def test_router_level_gates_reject_unprovisioned_claim_without_switch(
    db_session, db_session_factory
) -> None:
    """Control for the test above: without the param the gate sees the claim.

    Same app, same unprovisioned claim A — the only difference is the missing
    `?active_tenant=`. A 403 here proves the 200 above is the query-param switch
    reaching require_known_tenant, and not the gate being inert in this harness.

    Claim A must also hold NO w2v licence: an entitled tenant is JIT-registered
    by require_known_tenant (tenant-level licence = provisioning), so the
    harness-wide "everyone entitled" stub would admit A for the wrong reason.
    """
    tenant_a, tenant_b = uuid4(), uuid4()
    await _seed_tenant(db_session, tenant_b, n_projects=1, costs=[])
    await db_session.commit()

    entitled = EntitledLicenses()

    class _EntitledOnlyB:
        async def get_active_license(
            self, *, tenant_id: str, app_key: str
        ) -> ActiveLicenseView | None:
            if tenant_id != str(tenant_b):
                return None
            return await entitled.get_active_license(tenant_id=tenant_id, app_key=app_key)

        async def list_user_licenses(self, subject_id: str | None, *, tenant_id: str) -> list:
            return []

    app = _build_gated_probe_app(db_session_factory, tenant_id=str(tenant_a), roles=["super_admin"])
    app.dependency_overrides[_common.get_licenses_client] = lambda: _EntitledOnlyB()
    async with _client(app) as c:
        r = await c.get("/api/probe")
    assert r.status_code == 403
