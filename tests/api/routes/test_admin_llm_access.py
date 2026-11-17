"""Tenant LLM/GPU access (licences ``byo_mode`` for app ``w2v``) — operator surface.

- ``GET|PUT /api/admin/tenants/{id}/llm-access`` (super_admin);
- ``POST /api/admin/provision`` sets ``llm_access`` (default ``local_allowed``)
  AFTER the licences are ensured, and REPORTS a failure instead of raising
  (raising would discard the one-time tokens; silently passing would leave the
  tenant on licences' ``byo_required`` default with every run blocked).
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes import _common
from src.api.routes.admin import router as admin_router
from src.api.routes.tenancy.errors import register_tenancy_exception_handlers
from src.clients.licenses_client import (
    LicensesClient,
    LicensesClientError,
    TenantNotOnboardedError,
)
from src.clients.schemas.licenses import (
    TenantAppByoModeSetView,
    TenantAppByoOverrideView,
    TenantByoModeView,
)
from src.database.models import AdminAuditLog, ProvisioningAudit
from src.services.providers import byo_posture
from src.services.tenant_llm_access import apply_tenant_llm_access, read_tenant_llm_access
from tests._tenant_licences_fake import TenantLicencesFake
from tests.api.routes._p9_licences import TEST_KEYRING, override_licence_keyring

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _no_real_backoff(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The posture write's retry backoff, recorded instead of slept."""
    from src.services import license_provisioning

    slept: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(license_provisioning, "_sleep", _fake_sleep)
    return slept


class _NoopInviter:
    async def invite(self, email: str, **_: Any) -> None:
        return None


class _ByoLicences(TenantLicencesFake):
    """The byo-mode reads on top of the provisioning fake: ``tenant_mode`` is the
    tenant-wide value (None ⇒ 404 tenant_not_found_in_registry)."""

    def __init__(self, *, tenant_mode: str | None = "byo_required", **kw: Any) -> None:
        super().__init__(**kw)
        self.tenant_mode = tenant_mode
        self.read_error: Exception | None = None

    async def get_tenant_app_byo_override(self, *, tenant_id, app_key):  # type: ignore[no-untyped-def]
        if self.read_error is not None:
            raise self.read_error
        mode = self.byo_overrides.get((str(tenant_id), app_key))
        return None if mode is None else TenantAppByoOverrideView(app_key=app_key, byo_mode=mode)

    async def get_tenant_byo_mode(self, *, tenant_id, app_key):  # type: ignore[no-untyped-def]
        if self.tenant_mode is None:
            raise LicensesClientError(
                "get_tenant_byo_mode returned 404",
                status_code=404,
                code="tenant_not_found_in_registry",
            )
        return TenantByoModeView(tenant_id=str(tenant_id), byo_mode=self.tenant_mode, app=app_key)


def _build_app(db_session_factory, *, roles: list[str], lic: _ByoLicences) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = db_session_factory
    app.include_router(admin_router)
    register_tenancy_exception_handlers(app)

    class _SeedIdentity(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = str(uuid4())
            request.state.roles = roles
            request.state.subject_id = str(uuid4())
            return await call_next(request)

    app.add_middleware(_SeedIdentity)
    app.dependency_overrides[_common.get_inviter] = lambda: _NoopInviter()
    app.dependency_overrides[_common.get_licenses_client] = lambda: lic
    override_licence_keyring(app, TEST_KEYRING)
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    )


_BODY = {
    "email": "llm-access@example.com",
    "name": "LLM Access",
    "tenant_slug": "llm-access-co",
    "workspace_role": "tenant_admin",
}


# ─── provision ─────────────────────────────────────────────────────────────


async def test_provision_sets_local_allowed_by_default(db_session, db_session_factory) -> None:
    lic = _ByoLicences()
    app = _build_app(db_session_factory, roles=["super_admin"], lic=lic)
    async with _client(app) as c:
        r = await c.post("/api/admin/provision", json=_BODY)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["llm_access"] == {"mode": "local_allowed", "status": "ok", "detail": None}
    # The PER-APP w2v override was written (other apps untouched), after the ensure.
    assert lic.byo_overrides == {(body["tenant_id"], "w2v"): "local_allowed"}
    assert lic.ensure_calls, "licences must be ensured before the posture is set"
    audit = (
        await db_session.execute(
            select(ProvisioningAudit).where(ProvisioningAudit.tenant_id == UUID(body["tenant_id"]))
        )
    ).scalar_one()
    assert audit.status == "success"
    assert audit.result["llm_access"]["status"] == "ok"


async def test_provision_honours_platform_only(db_session_factory) -> None:
    lic = _ByoLicences()
    app = _build_app(db_session_factory, roles=["super_admin"], lic=lic)
    async with _client(app) as c:
        r = await c.post("/api/admin/provision", json={**_BODY, "llm_access": "platform_only"})
    assert r.status_code == 201, r.text
    assert r.json()["llm_access"]["mode"] == "platform_only"
    assert list(lic.byo_overrides.values()) == ["platform_only"]


async def test_provision_accepts_byo_required(db_session_factory) -> None:
    """Owner decision 2026-10-22: the posture is chosen AT provisioning — all three."""
    lic = _ByoLicences()
    app = _build_app(db_session_factory, roles=["super_admin"], lic=lic)
    async with _client(app) as c:
        r = await c.post("/api/admin/provision", json={**_BODY, "llm_access": "byo_required"})
    assert r.status_code == 201, r.text
    assert r.json()["llm_access"] == {"mode": "byo_required", "status": "ok", "detail": None}
    assert list(lic.byo_overrides.values()) == ["byo_required"]


@pytest.mark.parametrize("mode", ["anything"])
async def test_provision_rejects_a_disallowed_llm_access(db_session_factory, mode: str) -> None:
    lic = _ByoLicences()
    app = _build_app(db_session_factory, roles=["super_admin"], lic=lic)
    async with _client(app) as c:
        r = await c.post("/api/admin/provision", json={**_BODY, "llm_access": mode})
    assert r.status_code == 422
    assert lic.ensure_calls == []
    assert lic.registered == []


async def test_provision_reports_a_failed_llm_access_and_keeps_the_tokens(
    db_session, db_session_factory
) -> None:
    lic = _ByoLicences()
    lic.byo_set_error = LicensesClientError(
        "set_tenant_app_byo_mode returned 403: secret-ish body",
        status_code=403,
        code="app_not_administrable_by_caller",
    )
    app = _build_app(db_session_factory, roles=["super_admin"], lic=lic)
    async with _client(app) as c:
        r = await c.post("/api/admin/provision", json=_BODY)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["tokens"], "the one-time tokens must still be handed out"
    la = body["llm_access"]
    assert la["status"] == "failed"
    assert la["mode"] == "local_allowed"
    assert "403" in la["detail"] and "app_not_administrable_by_caller" in la["detail"]
    assert "secret-ish body" not in r.text  # never the upstream body
    # A licences REFUSAL is deterministic: written once, never retried.
    assert lic.byo_set_attempts == 1
    audit = (
        await db_session.execute(
            select(ProvisioningAudit).where(ProvisioningAudit.tenant_id == UUID(body["tenant_id"]))
        )
    ).scalar_one()
    # Not plain `success`: the tenant is still on byo_required (runs blocked).
    assert audit.status == "partial"
    assert "llm_access" in audit.error and "403" in audit.error
    assert "secret-ish body" not in audit.error
    assert audit.result["llm_access"]["status"] == "failed"
    assert audit.result["license_ids"], "the full result is kept for the operator's retry"


async def test_provision_retries_a_transient_llm_access_failure(
    db_session, db_session_factory, _no_real_backoff: list[float]
) -> None:
    lic = _ByoLicences()
    lic.byo_set_error = LicensesClientError("boom", status_code=503)
    lic.byo_set_fail_times = 2  # fails twice, the third attempt lands
    app = _build_app(db_session_factory, roles=["super_admin"], lic=lic)
    async with _client(app) as c:
        r = await c.post("/api/admin/provision", json=_BODY)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["llm_access"] == {"mode": "local_allowed", "status": "ok", "detail": None}
    assert lic.byo_set_attempts == 3
    assert _no_real_backoff == [0.5, 1.0]
    assert lic.byo_overrides == {(body["tenant_id"], "w2v"): "local_allowed"}
    audit = (
        await db_session.execute(
            select(ProvisioningAudit).where(ProvisioningAudit.tenant_id == UUID(body["tenant_id"]))
        )
    ).scalar_one()
    assert audit.status == "success" and audit.error is None


async def test_provision_gives_up_after_three_transient_failures(
    db_session, db_session_factory, _no_real_backoff: list[float]
) -> None:
    lic = _ByoLicences()
    lic.byo_set_error = httpx.ReadTimeout("slow")
    app = _build_app(db_session_factory, roles=["super_admin"], lic=lic)
    async with _client(app) as c:
        r = await c.post("/api/admin/provision", json=_BODY)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["tokens"]
    assert body["llm_access"] == {
        "mode": "local_allowed",
        "status": "failed",
        "detail": "ReadTimeout",
    }
    assert lic.byo_set_attempts == 3
    assert _no_real_backoff == [0.5, 1.0]  # ~1.5s total: inside the request budget
    audit = (
        await db_session.execute(
            select(ProvisioningAudit).where(ProvisioningAudit.tenant_id == UUID(body["tenant_id"]))
        )
    ).scalar_one()
    assert audit.status == "partial"
    assert audit.error == "llm_access 'local_allowed' not set: ReadTimeout"


# ─── GET / PUT /tenants/{id}/llm-access ─────────────────────────────────────


async def test_get_llm_access_sources(db_session_factory) -> None:
    tid = uuid4()
    lic = _ByoLicences(tenant_mode="platform_only")
    app = _build_app(db_session_factory, roles=["super_admin"], lic=lic)
    async with _client(app) as c:
        r = await c.get(f"/api/admin/tenants/{tid}/llm-access")
        assert r.status_code == 200, r.text
        assert r.json() == {"mode": "platform_only", "source": "tenant"}

        lic.byo_overrides[(str(tid), "w2v")] = "local_allowed"
        r = await c.get(f"/api/admin/tenants/{tid}/llm-access")
        assert r.json() == {"mode": "local_allowed", "source": "app_override"}

        lic.byo_overrides.clear()
        lic.tenant_mode = None
        r = await c.get(f"/api/admin/tenants/{tid}/llm-access")
        assert r.json() == {"mode": "byo_required", "source": "default"}


@pytest.mark.parametrize("upstream", [401])
async def test_get_llm_access_licences_refusal_is_502(db_session_factory, upstream: int) -> None:
    lic = _ByoLicences()
    lic.read_error = LicensesClientError("nope", status_code=upstream, code="x_code")
    app = _build_app(db_session_factory, roles=["super_admin"], lic=lic)
    async with _client(app) as c:
        r = await c.get(f"/api/admin/tenants/{uuid4()}/llm-access")
    assert r.status_code == 502, r.text
    err = r.json()["error"]
    assert err["code"] == "LICENCES_REFUSED"
    assert err["details"] == {"upstream_status": upstream, "upstream_code": "x_code"}


async def test_get_llm_access_for_a_tenant_another_app_owns_is_not_onboarded(
    db_session_factory,
) -> None:
    """Live 2026-10-20: /tenants showed a Cloudflare 502 for every tenant another
    app owns (system, aa-agents, ...), because licences' 403 tenant_not_accessible
    was mapped to LICENCES_REFUSED. It is an answer about the tenant."""
    lic = _ByoLicences()
    lic.read_error = TenantNotOnboardedError(
        "get_tenant_app_byo_override returned 403", status_code=403, code="tenant_not_accessible"
    )
    app = _build_app(db_session_factory, roles=["super_admin"], lic=lic)
    async with _client(app) as c:
        r = await c.get(f"/api/admin/tenants/{uuid4()}/llm-access")
    assert r.status_code == 403, r.text
    assert r.json()["error"]["code"] == "TENANT_NOT_ONBOARDED"


async def test_get_llm_access_outage_is_503(db_session_factory) -> None:
    lic = _ByoLicences()
    lic.read_error = httpx.ConnectError("down")
    app = _build_app(db_session_factory, roles=["super_admin"], lic=lic)
    async with _client(app) as c:
        r = await c.get(f"/api/admin/tenants/{uuid4()}/llm-access")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "LICENCES_UNAVAILABLE"


async def test_put_llm_access_writes_audits_and_invalidates_cache(
    db_session, db_session_factory
) -> None:
    tid = uuid4()
    byo_posture._cache[(tid, "w2v")] = ("byo_required", float("inf"))
    lic = _ByoLicences()
    lic.byo_overrides[(str(tid), "w2v")] = "byo_required"
    app = _build_app(db_session_factory, roles=["super_admin"], lic=lic)
    async with _client(app) as c:
        r = await c.put(f"/api/admin/tenants/{tid}/llm-access", json={"mode": "platform_only"})
    assert r.status_code == 200, r.text
    assert r.json() == {"mode": "platform_only", "source": "app_override"}
    assert lic.byo_overrides[(str(tid), "w2v")] == "platform_only"
    assert (tid, "w2v") not in byo_posture._cache
    row = (
        await db_session.execute(select(AdminAuditLog).where(AdminAuditLog.target_tenant_id == tid))
    ).scalar_one()
    assert row.action == "tenant.update"
    assert row.detail == {
        "llm_access": {"app": "w2v", "before_override": "byo_required", "after": "platform_only"}
    }


async def test_put_llm_access_refusal_is_502_and_not_audited(
    db_session, db_session_factory
) -> None:
    tid = uuid4()
    lic = _ByoLicences()
    lic.byo_set_error = LicensesClientError("x", status_code=403, code=None)
    app = _build_app(db_session_factory, roles=["super_admin"], lic=lic)
    async with _client(app) as c:
        r = await c.put(f"/api/admin/tenants/{tid}/llm-access", json={"mode": "local_allowed"})
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "LICENCES_REFUSED"
    rows = (
        await db_session.execute(select(AdminAuditLog).where(AdminAuditLog.target_tenant_id == tid))
    ).all()
    assert rows == []


async def test_put_llm_access_validates_and_requires_super_admin(db_session_factory) -> None:
    lic = _ByoLicences()
    async with _client(_build_app(db_session_factory, roles=["super_admin"], lic=lic)) as c:
        r = await c.put(f"/api/admin/tenants/{uuid4()}/llm-access", json={"mode": "cloud"})
    assert r.status_code == 422
    async with _client(_build_app(db_session_factory, roles=["tenant_admin"], lic=lic)) as c:
        r = await c.put(f"/api/admin/tenants/{uuid4()}/llm-access", json={"mode": "local_allowed"})
        g = await c.get(f"/api/admin/tenants/{uuid4()}/llm-access")
    assert r.status_code == 403 and g.status_code == 403
    assert lic.byo_overrides == {}


# ─── service helper + client (no DB) ───────────────────────────────────────


async def test_apply_refuses_unknown_mode_without_calling_licences() -> None:
    from src.services.tenant_llm_access import UnknownLlmAccessModeError

    lic = _ByoLicences()
    with pytest.raises(UnknownLlmAccessModeError):
        await apply_tenant_llm_access(uuid4(), "cloud", licenses_client=lic)  # ty: ignore[invalid-argument-type]
    assert lic.byo_overrides == {}


async def test_shared_core_refuses_an_unknown_mode_before_any_side_effect() -> None:
    from src.services.license_provisioning import provision_tenant_licences

    lic = _ByoLicences()
    with pytest.raises(ValueError, match="cloud"):
        await provision_tenant_licences(
            lic,  # ty: ignore[invalid-argument-type]
            tenant_id=uuid4(),
            llm_access="cloud",  # ty: ignore[invalid-argument-type]
        )
    assert lic.ensure_calls == [] and lic.byo_overrides == {}


async def test_shared_core_sets_byo_required_for_a_new_tenant() -> None:
    from src.services.license_provisioning import provision_tenant_licences

    tid = uuid4()
    lic = _ByoLicences()
    got = await provision_tenant_licences(
        lic,  # ty: ignore[invalid-argument-type]
        tenant_id=tid,
        llm_access="byo_required",
    )
    assert got.llm_access.mode == "byo_required" and got.llm_access.status == "ok"
    assert lic.byo_overrides[(str(tid), "w2v")] == "byo_required"


async def test_shared_core_ensures_then_sets_and_retry_helper_reports() -> None:
    from src.services.license_provisioning import (
        provision_tenant_licences,
        set_new_tenant_llm_access,
    )

    tid = uuid4()
    lic = _ByoLicences()
    lic.byo_set_error = httpx.ReadTimeout("slow")
    out = await provision_tenant_licences(lic, tenant_id=tid)  # ty: ignore[invalid-argument-type]
    assert out.licences.license_ids and out.llm_access.status == "failed"
    assert out.llm_access.as_record() == {
        "mode": "local_allowed",
        "status": "failed",
        "detail": "ReadTimeout",
    }
    lic.byo_set_error = None  # the approve's retry of just this step
    step = await set_new_tenant_llm_access(
        lic,  # ty: ignore[invalid-argument-type]
        tenant_id=tid,
        llm_access="platform_only",
    )
    assert step.status == "ok"
    assert lic.byo_overrides == {(str(tid), "w2v"): "platform_only"}


class _StaticToken:
    async def token(self) -> str:
        return "sa-token"


def _mock_client(handler) -> LicensesClient:  # type: ignore[no-untyped-def]
    return LicensesClient(
        authenticator=_StaticToken(),  # type: ignore[arg-type]
        base_url="http://licenses.test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def test_client_set_and_read_per_app_byo_mode() -> None:
    tid = uuid4()
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if request.method == "PUT" and path == f"/tenants/{tid}/apps/w2v/byo-mode":
            assert json.loads(request.content) == {"byo_mode": "local_allowed"}
            return httpx.Response(
                200,
                json={
                    "tenant_id": str(tid),
                    "tenant_name": "t",
                    "app_id": "a",
                    "app_key": "w2v",
                    "byo_mode": "local_allowed",
                    "prior_byo_mode": None,
                },
            )
        if request.method == "GET" and path == f"/tenants/{tid}/apps/w2v/byo-mode":
            return httpx.Response(404, json={"detail": "No per-product BYO override"})
        if request.method == "GET" and path == f"/tenants/{tid}/byo-mode":
            return httpx.Response(
                200, json={"tenant_id": str(tid), "name": "t", "byo_mode": "byo_required"}
            )
        return httpx.Response(500)

    client = _mock_client(handler)
    out = await client.set_tenant_app_byo_mode(
        tenant_id=tid, app_key="w2v", byo_mode="local_allowed"
    )
    assert isinstance(out, TenantAppByoModeSetView) and out.byo_mode == "local_allowed"
    assert seen[0].headers["Authorization"] == "Bearer sa-token"
    assert seen[0].headers["X-Tenant-Id"] == str(tid)
    access = await read_tenant_llm_access(tid, licenses_client=client)
    assert (access.mode, access.source) == ("byo_required", "tenant")


async def test_client_set_byo_mode_raises_with_status_and_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, json={"detail": {"error": "app_not_administrable_by_caller", "message": "m"}}
        )

    with pytest.raises(LicensesClientError) as info:
        await _mock_client(handler).set_tenant_app_byo_mode(
            tenant_id=uuid4(), app_key="w2v", byo_mode="local_allowed"
        )
    assert info.value.status_code == 403
    assert info.value.code == "app_not_administrable_by_caller"


async def test_invalidate_byo_posture_drops_only_that_tenant() -> None:
    a, b = uuid4(), uuid4()
    byo_posture._cache[(a, "w2v")] = ("local_allowed", float("inf"))
    byo_posture._cache[(b, "w2v")] = ("local_allowed", float("inf"))
    try:
        byo_posture.invalidate_byo_posture(a)
        assert (a, "w2v") not in byo_posture._cache
        assert (b, "w2v") in byo_posture._cache
    finally:
        byo_posture.clear_byo_posture_cache()
