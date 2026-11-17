"""Admin tier (Phase C, increment C-3) — fail-closed tenant-existence.

``require_known_tenant`` gates every tenant-scoped data router: a request is
served only if its EFFECTIVE tenant (claim, or a super-admin X-Active-Tenant-Id
switch) is known — a ``tenants`` registry row OR ≥1 project, and not suspended.
Mode (off|warn|enforce, default enforce) is a ConfigMap dial-back. These tests
mount a minimal router behind the gate (the dependency is router-agnostic).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4, uuid7

import pytest
from fastapi import APIRouter, Depends, FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes import _common
from src.api.routes._common import get_licenses_client, require_known_tenant
from src.api.routes.tenancy.errors import register_tenancy_exception_handlers
from src.clients.licenses_client import TenantNotOnboardedError
from src.clients.schemas.licenses import ActiveLicenseView
from src.config import clear_settings_cache
from src.database.models import Project, Tenant
from src.database.repositories.admin_repository import AdminRepository
from src.services.license_status import clear_license_status_cache

pytestmark = pytest.mark.asyncio

_gated = APIRouter()


@_gated.get("/api/_probe/ping")
async def _ping() -> dict[str, bool]:
    return {"ok": True}


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Each test starts with an empty known-tenant cache + default (enforce) mode."""
    _common._known_tenant_cache.clear()
    clear_license_status_cache()
    clear_settings_cache()
    yield
    _common._known_tenant_cache.clear()
    clear_license_status_cache()
    clear_settings_cache()


class _FakeLicenses:
    """Licenses client stub. ``verdict``: ``"none"`` (404 → no active licence),
    ``"active"`` (200 for this tenant + w2v), or ``"error"`` (upstream blip →
    ``unknown``). Counts calls so a test can prove a lookup was / was not made."""

    def __init__(self, verdict: str = "none") -> None:
        self.verdict = verdict
        self.calls = 0
        self.subjects: list[str | None] = []

    async def get_active_license(self, *, tenant_id: str, app_key: str) -> ActiveLicenseView | None:
        self.calls += 1
        if self.verdict == "error":
            raise RuntimeError("licenses service unreachable")
        if self.verdict == "not_onboarded":
            # licences migration 050: another app owns the tenant, no grant to w2v.
            raise TenantNotOnboardedError(
                "get_active_license returned 403: tenant_not_accessible",
                status_code=403,
                code="tenant_not_accessible",
            )
        if self.verdict == "active":
            return ActiveLicenseView(active=True, tenant_id=tenant_id, status="active")
        return None

    async def list_user_licenses(self, subject_id: str | None, *, tenant_id: str) -> list[Any]:
        self.subjects.append(subject_id)
        return []


def _set_mode(monkeypatch, mode: str) -> None:
    monkeypatch.setenv("TENANT_ENFORCEMENT_MODE", mode)
    clear_settings_cache()


def _build_app(
    db_session_factory,
    *,
    tenant_id: str,
    roles: list[str] | None = None,
    subject_id: str | None = None,
    tenant_delegated: bool | None = None,
    licenses: _FakeLicenses | None = None,
) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = db_session_factory
    app.include_router(_gated, dependencies=[Depends(require_known_tenant)])
    register_tenancy_exception_handlers(app)  # as src/asgi.py does
    # Default: no active licence, so every pre-existing "unknown → 403" case keeps
    # its meaning and never reaches a real licenses service.
    fake = licenses if licenses is not None else _FakeLicenses("none")
    app.dependency_overrides[get_licenses_client] = lambda: fake

    class _SeedIdentity(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = tenant_id
            request.state.roles = roles or []
            request.state.subject_id = subject_id or str(uuid4())
            # `tenant_delegated=None` (the default) deliberately leaves the
            # attribute UNSET — mirrors an orchestrator running against an
            # OLDER w2v_common that has never heard of the S95 marker, so
            # `require_known_tenant`'s `getattr(..., False)` is exercised for
            # real rather than always reading an explicit False.
            if tenant_delegated is not None:
                request.state.tenant_delegated = tenant_delegated
            return await call_next(request)

    app.add_middleware(_SeedIdentity)
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://testserver"
    )


async def _ping_status(app: FastAPI, **kw) -> int:
    async with _client(app) as c:
        return (await c.get("/api/_probe/ping", **kw)).status_code


# ─── known tenants pass ──────────────────────────────────────────────────


async def test_known_via_registry_row_allowed(db_session, db_session_factory) -> None:
    tenant = uuid4()
    db_session.add(Tenant(id=tenant, status="active"))
    await db_session.commit()
    app = _build_app(db_session_factory, tenant_id=str(tenant))
    assert await _ping_status(app) == 200


async def test_known_via_project_only_allowed(db_session, db_session_factory) -> None:
    tenant = uuid4()
    db_session.add(
        Project(
            id=uuid7(),
            tenant_id=tenant,
            owner_subject_id=uuid7(),
            visibility="tenant",
            type="generic",
            title="P",
        )
    )
    await db_session.commit()
    app = _build_app(db_session_factory, tenant_id=str(tenant))
    assert await _ping_status(app) == 200


# ─── unknown tenant: mode matrix ─────────────────────────────────────────


async def test_unknown_enforced_is_403(db_session_factory) -> None:
    # default mode = enforce
    app = _build_app(db_session_factory, tenant_id=str(uuid4()))
    assert await _ping_status(app) == 403


async def test_unknown_warn_allows_but_logs(db_session_factory, monkeypatch, caplog) -> None:
    _set_mode(monkeypatch, "warn")
    app = _build_app(db_session_factory, tenant_id=str(uuid4()))
    with caplog.at_level("WARNING", logger="w2v_orchestrator.audit"):
        assert await _ping_status(app) == 200
    assert any("TENANT_UNKNOWN" in r.message for r in caplog.records)


async def test_off_allows_without_querying(db_session_factory, monkeypatch) -> None:
    _set_mode(monkeypatch, "off")
    calls = {"n": 0}

    async def _counting(self, tid):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return False

    monkeypatch.setattr(AdminRepository, "tenant_switchable", _counting)
    app = _build_app(db_session_factory, tenant_id=str(uuid4()))
    assert await _ping_status(app) == 200
    assert calls["n"] == 0  # off short-circuits before any DB work


async def test_suspended_tenant_blocked(db_session, db_session_factory) -> None:
    tenant = uuid4()
    db_session.add(Tenant(id=tenant, status="suspended"))
    await db_session.commit()
    app = _build_app(db_session_factory, tenant_id=str(tenant))
    assert await _ping_status(app) == 403


# ─── scope-switch validates the EFFECTIVE tenant ─────────────────────────


async def test_scope_switch_into_known_tenant_allowed(db_session, db_session_factory) -> None:
    claim, foreign = uuid4(), uuid4()
    db_session.add(Tenant(id=foreign, status="active"))
    await db_session.commit()
    app = _build_app(db_session_factory, tenant_id=str(claim), roles=["super_admin"])
    assert await _ping_status(app, headers={"X-Active-Tenant-Id": str(foreign)}) == 200


async def test_scope_switch_into_unknown_tenant_blocked(db_session, db_session_factory) -> None:
    claim, foreign = uuid4(), uuid4()
    app = _build_app(db_session_factory, tenant_id=str(claim), roles=["super_admin"])
    assert await _ping_status(app, headers={"X-Active-Tenant-Id": str(foreign)}) == 403


# ─── positive-result cache ───────────────────────────────────────────────


async def test_known_tenant_is_cached(db_session, db_session_factory, monkeypatch) -> None:
    tenant = uuid4()
    db_session.add(Tenant(id=tenant, status="active"))
    await db_session.commit()
    calls = {"n": 0}
    real = AdminRepository.tenant_switchable

    async def _counting(self, tid: UUID) -> bool:
        calls["n"] += 1
        return await real(self, tid)

    monkeypatch.setattr(AdminRepository, "tenant_switchable", _counting)
    app = _build_app(db_session_factory, tenant_id=str(tenant))
    assert await _ping_status(app) == 200
    assert await _ping_status(app) == 200
    assert calls["n"] == 1  # second request served from the positive cache


# ─── S95: delegated-tenant marker ─────────────────────────────────────────
#
# marketing sends `X-Effective-Tenant-Id: <real end tenant>` so cost attributes
# to the real customer, not the service account. w2v_common's
# `_apply_effective_tenant` stamps `request.state.tenant_delegated = True` when
# (and only when) that assertion came from an allowlisted delegating subject.
# `require_known_tenant` must treat that marker as "known" WITHOUT ever writing
# the delegated tenant into `_known_tenant_cache` (see the cache-poisoning test
# below for why).


async def test_delegated_marker_allows_unknown_tenant(db_session_factory) -> None:
    """The whole point of S95: a delegated, unregistered tenant is served, not
    403'd. No Tenant row, no Project row — `tenant_switchable` would say False."""
    tenant = uuid4()
    app = _build_app(db_session_factory, tenant_id=str(tenant), tenant_delegated=True)
    assert await _ping_status(app) == 200


async def test_no_marker_unknown_tenant_still_403(db_session_factory) -> None:
    """Fail-closed is preserved: an unregistered tenant with NO delegation
    marker at all (the attribute is unset, exactly like an orchestrator running
    against a pre-S95 w2v_common) is still rejected. Distinct from
    `test_unknown_enforced_is_403` in that it pins the `getattr` default path
    specifically, not just "no special header was sent"."""
    app = _build_app(db_session_factory, tenant_id=str(uuid4()), tenant_delegated=None)
    assert await _ping_status(app) == 403


async def test_delegated_marker_false_unknown_tenant_still_403(db_session_factory) -> None:
    """`tenant_delegated=False` (explicitly set, e.g. a rejected assertion or an
    ordinary non-re-scoped delegator call) must be treated the same as absent —
    still fail-closed."""
    app = _build_app(db_session_factory, tenant_id=str(uuid4()), tenant_delegated=False)
    assert await _ping_status(app) == 403


async def test_delegated_request_does_not_poison_cache_for_later_direct_caller(
    db_session_factory,
) -> None:
    """⛔ The cache-poisoning guard.

    `_known_tenant_cache` is a process-global ``dict[UUID, float]`` keyed ONLY
    by tenant_id — it has no notion of caller or provenance. If the delegated
    branch wrote the unknown tenant into that cache, a LATER, NON-delegated
    request for the SAME tenant (e.g. a spoofed or misrouted direct
    customer-token call that was never vouched for by a delegator) would be
    waved through by `_tenant_known_cached` with no re-check — a privilege
    leak. The delegated branch must return without ever touching the cache.
    """
    tenant = uuid4()

    # 1) A delegated request for the (unregistered) tenant — allowed via the
    #    marker, must NOT populate `_known_tenant_cache`.
    delegated_app = _build_app(db_session_factory, tenant_id=str(tenant), tenant_delegated=True)
    assert await _ping_status(delegated_app) == 200
    assert tenant not in _common._known_tenant_cache

    # 2) A later, non-delegated request for the SAME still-unregistered tenant
    #    MUST still be rejected — not waved through by a poisoned cache entry.
    direct_app = _build_app(db_session_factory, tenant_id=str(tenant), tenant_delegated=False)
    assert await _ping_status(direct_app) == 403


# ─── Tenant-level w2v licence admits (fr parity, owner ruling 2026-09-28) ──
#
# provision_tenant_bundle.py (licenses repo) is THE provisioning path and writes
# only a tenant-level licence, so a freshly provisioned tenant has no registry
# row and no projects. Like fr, an entitled tenant-level licence admits it —
# with NO subject scoping and NO write on the request path. A registry row, when
# present, stays authoritative (suspended → 403 whatever the licence says).


async def _registry_status(db_session_factory, tenant: UUID) -> str | None:
    async with db_session_factory() as s:
        row = await s.get(Tenant, tenant)
        return row.status if row is not None else None


async def test_unregistered_tenant_with_entitled_licence_is_admitted_without_a_write(
    db_session_factory,
) -> None:
    tenant = uuid4()
    fake = _FakeLicenses("active")
    app = _build_app(db_session_factory, tenant_id=str(tenant), licenses=fake)
    assert await _ping_status(app) == 200
    assert fake.calls == 1
    assert await _registry_status(db_session_factory, tenant) is None  # nothing written
    # Positive verdict cached like any known tenant: no second lookup.
    assert await _ping_status(app) == 200
    assert fake.calls == 1


async def test_unregistered_tenant_without_licence_is_403(db_session_factory) -> None:
    tenant = uuid4()
    fake = _FakeLicenses("none")
    app = _build_app(db_session_factory, tenant_id=str(tenant), licenses=fake)
    assert await _ping_status(app) == 403
    assert fake.calls == 1
    # Tenant-level only: the request carries a subject, yet the verdict never
    # consulted a subject's inventory (fr parity — no subject scoping).
    assert fake.subjects == []
    assert tenant not in _common._known_tenant_cache


async def test_suspended_tenant_with_entitled_licence_is_403(
    db_session, db_session_factory
) -> None:
    tenant = uuid4()
    db_session.add(Tenant(id=tenant, status="suspended"))
    await db_session.commit()
    fake = _FakeLicenses("active")
    app = _build_app(db_session_factory, tenant_id=str(tenant), licenses=fake)
    assert await _ping_status(app) == 403
    assert fake.calls == 0  # the registry row is authoritative; no lookup at all
    assert await _registry_status(db_session_factory, tenant) == "suspended"


async def test_tenant_not_onboarded_to_w2v_is_403_not_onboarded(db_session_factory) -> None:
    """403 ``tenant_not_accessible`` from licences = "not onboarded to w2v": the
    auto-registration path answers 403 ``TENANT_NOT_ONBOARDED`` (distinct from the
    generic "tenant not provisioned"), consults no subject inventory (that would be
    refused the same way), and caches nothing as known."""
    tenant = uuid4()
    fake = _FakeLicenses("not_onboarded")
    app = _build_app(db_session_factory, tenant_id=str(tenant), licenses=fake)
    async with _client(app) as c:
        r = await c.get("/api/_probe/ping")
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "TENANT_NOT_ONBOARDED"
    assert fake.calls == 1
    assert fake.subjects == []
    assert tenant not in _common._known_tenant_cache
    assert await _registry_status(db_session_factory, tenant) is None
    # A definitive (cached) verdict: a second request does not re-ask licences.
    assert await _ping_status(app) == 403
    assert fake.calls == 1


async def test_unregistered_tenant_licence_lookup_error_is_403(db_session_factory) -> None:
    tenant = uuid4()
    fake = _FakeLicenses("error")
    app = _build_app(db_session_factory, tenant_id=str(tenant), licenses=fake)
    assert await _ping_status(app) == 403
    assert fake.calls == 1
    assert tenant not in _common._known_tenant_cache
