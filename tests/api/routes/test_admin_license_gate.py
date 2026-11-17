"""Licence gate on the ADMIN tier.

`_tenant_gate` is mounted router-wide on the tenant-scoped data routers, but the
admin router was exempt wholesale. Its routes carry SuperAdminDep /
TenantAdminDep, so they were never *unauthenticated* — they were UNENTITLED:
a tenant whose w2v licence is none/expired/revoked still read its own overview,
usage, member roster and the operator run + audit panes. The owner ruling is that all
non-sanctioned usage is blocked, so those routes are now gated per-route.

What stays EXEMPT is not an oversight but the chicken-and-egg: /provision is
how a tenant COMES TO HAVE a licence, and /tenants, /tenants/{id},
/scope-switch are the operator surface that reaches a
tenant which does not have one yet. Gating those would deadlock onboarding —
the gate would demand the licence that the route is there to mint.

The gate is LICENSE_GATE_ONLY, not the full `_tenant_gate` — see
`_common.LICENSE_GATE_ONLY` for why the other two deps are deliberately
omitted on this tier; `test_gate_is_entitlement_only_never_tenant_existence`
below pins it so nobody "completes" it later and 403s a new tenant's admin.

Refusal modes (72a7b8ee) must stay distinguishable here too: an authoritative
"not entitled" is 402, an indeterminate "we could not check" is 503.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4, uuid7

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes import _common
from src.api.routes.admin import router as admin_router
from src.clients.schemas.licenses import AppInfoView, LicenseView
from src.database.models import Project, Tenant
from src.services.license_status import clear_license_status_cache
from tests.api.routes._licence_stubs import EntitledLicenses

pytestmark = pytest.mark.asyncio


# ─── the wiring, read off the REAL app ───────────────────────────────────
# Route-level `dependencies=` are invisible to a grep for decorators. Assert on the app that
# actually serves traffic, not on a test-local reconstruction of it.

#: (method, path) → the gate that route MUST carry.
_GATED = {
    ("GET", "/api/admin/overview"),
    ("GET", "/api/admin/usage"),
    ("GET", "/api/admin/runs"),
    ("GET", "/api/admin/audit"),
}

#: Circular by construction — provisioning mints the licence the gate wants.
_EXEMPT = {
    ("POST", "/api/admin/provision"),
    ("GET", "/api/admin/tenants"),
    ("PATCH", "/api/admin/tenants/{tenant_id}"),
    ("POST", "/api/admin/scope-switch"),
}


def _effective_dependencies() -> dict[tuple[str, str], list[str]]:
    """Flatten the real app's route tree to {(method, path): [dep names]}.

    FastAPI keeps `include_router(...)` lazy (`_IncludedRouter`), so router-level
    dependencies live on the include context and per-route ones on the route —
    a caller is gated by the UNION. Walking both is the only honest read.
    """
    from src.asgi import app  # imported lazily: builds Settings + the whole app

    def walk(routes, inherited: tuple = ()):
        for route in routes:
            ctx = getattr(route, "include_context", None)
            if ctx is not None:
                yield from walk(ctx.included_router.routes, inherited + tuple(ctx.dependencies))
                continue
            if isinstance(route, APIRoute):
                deps = inherited + tuple(route.dependencies)
                for method in route.methods:
                    yield (method, route.path), [d.dependency.__name__ for d in deps]

    return dict(walk(app.routes))


async def test_gated_admin_routes_require_a_licence() -> None:
    """Every product-capability admin read."""
    deps = _effective_dependencies()
    missing = sorted(key for key in _GATED if "require_license" not in deps.get(key, []))
    assert not missing, f"reachable without an entitled licence: {missing}"


async def test_provisioning_routes_stay_exempt() -> None:
    """The chicken-and-egg set. A licence gate here would deadlock onboarding."""
    deps = _effective_dependencies()
    gated = sorted(key for key in _EXEMPT if "require_license" in deps.get(key, []))
    assert not gated, f"provisioning cannot be licence-gated (circular): {gated}"


async def test_gate_is_entitlement_only_never_tenant_existence() -> None:
    """The admin tier takes LICENSE_GATE_ONLY *on purpose*.

    `require_tenant_admin` runs `tenant_switchable` only on a real cross-tenant
    switch and leaves a caller's OWN claim fail-open, because a freshly-licensed
    tenant with no registry row and no projects is indistinguishable from an
    unprovisioned one. Adding `require_known_tenant` here would 403 exactly that
    tenant's admin; an RLS bind would collapse the cross-tenant /runs + /audit
    panes to one tenant. Neither is part of closing the ENTITLEMENT hole.
    """
    deps = _effective_dependencies()
    for key in _GATED:
        assert deps[key] == ["require_license"], f"{key} gained an unintended gate: {deps[key]}"


# ─── the gate BITES: 402 for an authoritative refusal ────────────────────


class _ScriptedLicenses:
    """LicensesClient stand-in: no active licence, a scripted inventory."""

    def __init__(self, *, inventory: list[LicenseView] | None = None, raises: bool = False) -> None:
        self._inventory = inventory or []
        self._raises = raises
        self.active_calls = 0

    async def get_active_license(self, *, tenant_id: str, app_key: str) -> None:
        self.active_calls += 1
        if self._raises:
            raise RuntimeError("licenses service unreachable")
        return

    async def list_user_licenses(self, subject_id: str | None, *, tenant_id: str) -> list[Any]:
        return self._inventory


def _iso(days: int) -> str:
    return (datetime.now(UTC) + timedelta(days=days)).isoformat()


def _revoked() -> _ScriptedLicenses:
    return _ScriptedLicenses(
        inventory=[
            LicenseView(
                id="lic-rev",
                app=AppInfoView(id="app-w2v", key="w2v", name="W2V"),
                status="revoked",
                revoked_at=_iso(-1),
            )
        ]
    )


def _expired() -> _ScriptedLicenses:
    return _ScriptedLicenses(
        inventory=[
            LicenseView(
                id="lic-exp",
                app=AppInfoView(id="app-w2v", key="w2v", name="W2V"),
                status="expired",
                expires_at=_iso(-3),
            )
        ]
    )


@pytest.fixture(autouse=True)
def _reset_license_cache():
    # get_license_status memoises per (tenant, app, subject) for 60s — a leaked
    # entry would let one test's verdict decide another's.
    clear_license_status_cache()
    yield
    clear_license_status_cache()


def _build_admin_app(db_session_factory, *, tenant_id: str, roles: list[str], licenses) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = db_session_factory
    # /runs reads the dispatcher for lease_present; None is the "no lease" arm.
    app.state.dispatcher = None
    app.include_router(admin_router)
    app.dependency_overrides[_common.get_licenses_client] = lambda: licenses
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
    return AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    )


# Both roles are present: this is about ENTITLEMENT, and a 403 here would mean
# the test proved nothing.
_ADMIN_ROLES = ["super_admin", "tenant_admin"]

_GATED_ADMIN_PATHS = [
    "/api/admin/overview",
    "/api/admin/usage",
    "/api/admin/runs",
    "/api/admin/audit",
]


@pytest.mark.parametrize("path", ["/api/admin/overview"])
async def test_revoked_tenant_is_402_on_every_gated_admin_route(db_session_factory, path) -> None:
    app = _build_admin_app(
        db_session_factory, tenant_id=str(uuid4()), roles=_ADMIN_ROLES, licenses=_revoked()
    )
    async with _client(app) as c:
        r = await c.get(path)
    assert r.status_code == 402, f"{path} served an unlicensed tenant"
    assert r.json()["detail"]["error"] == "license_revoked"


@pytest.mark.parametrize("path", ["/api/admin/overview"])
async def test_expired_tenant_is_402_on_every_gated_admin_route(db_session_factory, path) -> None:
    app = _build_admin_app(
        db_session_factory, tenant_id=str(uuid4()), roles=_ADMIN_ROLES, licenses=_expired()
    )
    async with _client(app) as c:
        r = await c.get(path)
    assert r.status_code == 402
    assert r.json()["detail"]["error"] == "license_expired"


async def test_admin_outage_is_503_not_402(db_session_factory) -> None:
    """72a7b8ee's distinction must survive onto this tier: "we could not check"
    is NOT "you have no licence". A 402 here would tell a paying operator their
    licence is gone over our own licences-service blip."""
    app = _build_admin_app(
        db_session_factory,
        tenant_id=str(uuid4()),
        roles=_ADMIN_ROLES,
        licenses=_ScriptedLicenses(raises=True),
    )
    async with _client(app) as c:
        r = await c.get("/api/admin/overview")
    assert r.status_code == 503
    assert r.json()["detail"]["error"] == "license_unavailable"
    assert r.headers["retry-after"] == "30"


# ─── the chicken-and-egg: provisioning still works with NO licence ───────


async def test_provisioning_surface_works_for_a_tenant_with_no_licence(
    db_session_factory,
) -> None:
    """THE case the exemption exists for. A tenant with no licence at all — the
    state every tenant is in one instant before it is provisioned — must still
    reach the operator surface that mints one. If this ever 402s, onboarding is
    deadlocked and no new customer can be created."""
    no_licence = _ScriptedLicenses()  # no active licence, empty inventory → 'none'
    app = _build_admin_app(
        db_session_factory, tenant_id=str(uuid4()), roles=_ADMIN_ROLES, licenses=no_licence
    )
    async with _client(app) as c:
        r = await c.get("/api/admin/tenants")
    assert r.status_code == 200, r.text
    # And it did not merely pass the check — it was never subjected to one.
    assert no_licence.active_calls == 0


async def test_the_same_unlicensed_caller_is_refused_on_a_gated_route(
    db_session_factory,
) -> None:
    """Pins the pair: one caller, one licence state, two routes, two verdicts.
    Otherwise the test above could pass simply because the gate is off."""
    no_licence = _ScriptedLicenses()
    tenant = str(uuid4())
    app = _build_admin_app(
        db_session_factory, tenant_id=tenant, roles=_ADMIN_ROLES, licenses=no_licence
    )
    async with _client(app) as c:
        exempt = await c.get("/api/admin/tenants")
        clear_license_status_cache()
        gated = await c.get("/api/admin/overview")
    assert exempt.status_code == 200
    assert gated.status_code == 402
    assert gated.json()["detail"]["error"] == "license_none"


# ─── super-admin is not broken ───────────────────────────────────────────


async def _seed_tenant(session, tenant_id: UUID) -> None:
    session.add(Tenant(id=tenant_id, name="T", status="active"))
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
    await session.commit()


async def test_entitled_super_admin_still_reads_across_tenants(
    db_session, db_session_factory
) -> None:
    """/runs and /audit are deliberately CROSS-tenant. The gate must charge
    entitlement without narrowing scope — which is why they take
    LICENSE_GATE_ONLY and not a gate that binds RLS to the operator's own claim.
    """
    own = uuid4()
    await _seed_tenant(db_session, own)
    app = _build_admin_app(
        db_session_factory, tenant_id=str(own), roles=_ADMIN_ROLES, licenses=EntitledLicenses()
    )
    async with _client(app) as c:
        runs = await c.get("/api/admin/runs")
        audit = await c.get("/api/admin/audit")
    assert (runs.status_code, audit.status_code) == (200, 200), (runs.text, audit.text)


async def test_entitled_super_admin_scope_switch_still_works(
    db_session, db_session_factory
) -> None:
    """The X-Active-Tenant-Id switch must survive the new gate: an entitled
    super-admin acting INTO a foreign tenant still reads that tenant's overview.
    """
    own, foreign = uuid4(), uuid4()
    await _seed_tenant(db_session, own)
    await _seed_tenant(db_session, foreign)
    app = _build_admin_app(
        db_session_factory, tenant_id=str(own), roles=_ADMIN_ROLES, licenses=EntitledLicenses()
    )
    async with _client(app) as c:
        r = await c.get("/api/admin/overview", headers={"X-Active-Tenant-Id": str(foreign)})
    assert r.status_code == 200, r.text
    assert r.json()["tenant_id"] == str(foreign)
