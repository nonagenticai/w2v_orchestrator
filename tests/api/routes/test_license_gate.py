"""Gate 2 — server-side w2v license enforcement (``require_license``).

Mirrors ``test_tenant_failclosed.py``: mounts a minimal probe router behind the
dependency (it is router-agnostic) and drives it with a faked LicensesClient.

Contract:
  * gate ON BY DEFAULT — with no env set at all it enforces;
  * gate ON + entitled (active / expiring_soon) → pass (200);
  * gate ON + none / expired / revoked → 402 ``license_<state>`` (an
    AUTHORITATIVE refusal: we asked, and the tenant is not entitled);
  * gate ON + licenses-upstream error → FAIL-CLOSED to **503**
    ``license_unavailable`` (we could NOT check — a different thing, and a
    caller must be able to tell the two apart);
  * gate OFF (explicit W2V_LICENSE_GATE_ENABLED=false) → break-glass no-op,
    never calls the licenses service.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from fastapi import APIRouter, Depends, FastAPI
from httpx import ASGITransport, AsyncClient, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes._common import get_licenses_client, require_license
from src.api.routes.tenancy.errors import register_tenancy_exception_handlers
from src.clients.licenses_client import LicensesClientError, TenantNotOnboardedError
from src.clients.schemas.licenses import (
    ActiveLicenseView,
    ActivePlanRef,
    AppInfoView,
    LicenseView,
)
from src.config import clear_settings_cache, get_settings
from src.services.license_status import clear_license_status_cache

pytestmark = pytest.mark.asyncio

_gated = APIRouter()


@_gated.get("/api/_probe/ping")
async def _ping() -> dict[str, bool]:
    return {"ok": True}


class _FakeLicensesClient:
    """Stands in for LicensesClient with scripted active/inventory responses."""

    def __init__(
        self,
        *,
        active: ActiveLicenseView | None = None,
        inventory: list[LicenseView] | None = None,
        raises: bool = False,
    ) -> None:
        self._active = active
        self._inventory = inventory or []
        self._raises = raises
        self.active_calls = 0

    async def get_active_license(self, *, tenant_id: str, app_key: str) -> ActiveLicenseView | None:
        self.active_calls += 1
        if self._raises:
            raise RuntimeError("licenses service unreachable")
        return self._active

    async def list_user_licenses(
        self, subject_id: str | None, *, tenant_id: str
    ) -> list[LicenseView]:
        return self._inventory


@pytest.fixture(autouse=True)
def _reset_settings():
    clear_settings_cache()
    # get_license_status memoises per (tenant, app, subject) for 60s — a leaked
    # entry would let one test's verdict decide another's.
    clear_license_status_cache()
    yield
    clear_settings_cache()
    clear_license_status_cache()


def _build_app(fake: _FakeLicensesClient, *, tenant_id: str | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(_gated, dependencies=[Depends(require_license)])
    app.dependency_overrides[get_licenses_client] = lambda: fake

    tid = tenant_id or str(uuid4())

    class _SeedIdentity(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = tid
            request.state.subject_id = str(uuid4())
            request.state.roles = []
            return await call_next(request)

    app.add_middleware(_SeedIdentity)
    return app


async def _probe_response(app: FastAPI) -> Response:
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        return await c.get("/api/_probe/ping")


async def _probe(app: FastAPI) -> tuple[int, dict[str, Any]]:
    # Every arm returns JSON: 200 → {"ok": true}; 402/503 → {"detail": {...}}.
    r = await _probe_response(app)
    return r.status_code, r.json()


def _enable_gate(monkeypatch) -> None:
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    clear_settings_cache()


def _disable_gate(monkeypatch) -> None:
    """Break-glass OFF. Explicit now that ON is the default."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "false")
    clear_settings_cache()


def _unset_gate_env(monkeypatch) -> None:
    """Strip BOTH aliases so the DECLARED default decides, not the runner's env.

    tests/conftest.py's `_clear_settings` scrubs only DATABASE_/AUTH_/... prefixes,
    so an ambient W2V_LICENSE_GATE_ENABLED would otherwise answer for the code.
    """
    monkeypatch.delenv("W2V_LICENSE_GATE_ENABLED", raising=False)
    monkeypatch.delenv("LICENSE_GATE_ENABLED", raising=False)
    clear_settings_cache()


def _iso(days: int) -> str:
    return (datetime.now(UTC) + timedelta(days=days)).isoformat()


# ─── the DEFAULT is ON ───────────────────────────────────────────────────
# The owner ruling is that all w2v usage is licence-gated. A default of False
# made `require_license` return before it looked at anything, so the gate was
# mounted on every tenant-scoped router and enforced nothing.


async def test_gate_is_enabled_by_default(monkeypatch) -> None:
    """No env at all → the DECLARED default, and it must be ON."""
    _unset_gate_env(monkeypatch)
    assert get_settings().license_gate_enabled is True


async def test_gate_enforces_with_no_env_set(monkeypatch) -> None:
    """The default is not just declared — it BITES. With nothing exported, an
    unlicensed tenant is refused and the licenses service really was consulted."""
    _unset_gate_env(monkeypatch)
    fake = _FakeLicensesClient(active=None, inventory=[])
    status, body = await _probe(_build_app(fake))
    assert status == 402
    assert body["detail"]["error"] == "license_none"
    assert fake.active_calls == 1  # it did not short-circuit


# ─── gate OFF (explicit break-glass): pure no-op ─────────────────────────


async def test_gate_off_is_noop_and_never_calls_licenses(monkeypatch) -> None:
    _disable_gate(monkeypatch)
    fake = _FakeLicensesClient(active=None, inventory=[])  # would be 'none' if consulted
    app = _build_app(fake)
    status, _ = await _probe(app)
    assert status == 200
    assert fake.active_calls == 0  # short-circuits before any licenses call


# ─── gate ON: entitled passes ────────────────────────────────────────────


async def test_gate_on_active_allows(monkeypatch) -> None:
    _enable_gate(monkeypatch)
    fake = _FakeLicensesClient(
        active=ActiveLicenseView(expires_at=None, plan=ActivePlanRef(code="standard"))
    )
    status, _ = await _probe(_build_app(fake))
    assert status == 200
    assert fake.active_calls == 1


async def test_gate_on_expiring_soon_allows(monkeypatch) -> None:
    _enable_gate(monkeypatch)
    fake = _FakeLicensesClient(
        active=ActiveLicenseView(expires_at=_iso(5), plan=ActivePlanRef(code="standard"))
    )
    status, _ = await _probe(_build_app(fake))
    assert status == 200


# ─── gate ON: unentitled 402s with the right code ────────────────────────


async def test_gate_on_none_is_402(monkeypatch) -> None:
    _enable_gate(monkeypatch)
    fake = _FakeLicensesClient(active=None, inventory=[])
    status, body = await _probe(_build_app(fake))
    assert status == 402
    assert body["detail"]["error"] == "license_none"


async def test_gate_on_expired_is_402(monkeypatch) -> None:
    _enable_gate(monkeypatch)
    fake = _FakeLicensesClient(
        active=None,
        inventory=[
            LicenseView(
                id="lic-exp",
                app=AppInfoView(id="app-w2v", key="w2v", name="W2V"),
                status="expired",
                expires_at=_iso(-3),
            )
        ],
    )
    status, body = await _probe(_build_app(fake))
    assert status == 402
    assert body["detail"]["error"] == "license_expired"


async def test_gate_on_revoked_is_402(monkeypatch) -> None:
    _enable_gate(monkeypatch)
    fake = _FakeLicensesClient(
        active=None,
        inventory=[
            LicenseView(
                id="lic-rev",
                app=AppInfoView(id="app-w2v", key="w2v", name="W2V"),
                status="revoked",
                revoked_at=_iso(-1),
            )
        ],
    )
    status, body = await _probe(_build_app(fake))
    assert status == 402
    assert body["detail"]["error"] == "license_revoked"


# ─── gate ON: fail-CLOSED on an upstream error → 503, not 200 ────────────
# The old contract allowed 'unknown'. An introspection outage therefore meant
# ALLOW, fleet-wide, silently — the gate's own failure mode was to stop gating.


async def test_gate_on_upstream_error_fails_closed(monkeypatch) -> None:
    _enable_gate(monkeypatch)
    fake = _FakeLicensesClient(raises=True)
    status, body = await _probe(_build_app(fake))
    assert status == 503, "an unverifiable entitlement must DENY, never allow"
    assert body["detail"]["error"] == "license_unavailable"


async def test_upstream_error_response_is_retryable(monkeypatch) -> None:
    """503 is the honest shape: server-side, transient, retryable — so it carries
    Retry-After and names the indeterminate state it could not resolve."""
    _enable_gate(monkeypatch)
    r = await _probe_response(_build_app(_FakeLicensesClient(raises=True)))
    assert r.status_code == 503
    assert r.headers["retry-after"] == "30"
    assert r.json()["detail"]["state"] == "unknown"


async def test_refusal_modes_are_distinguishable(monkeypatch) -> None:
    """A caller MUST be able to tell "you have no licence" from "we could not
    check". Same gate, same tenant, two upstream conditions — and the status
    code AND the error code differ on both axes, so neither a status-code-only
    nor an error-code-only client can confuse them."""
    _enable_gate(monkeypatch)
    unlicensed_status, unlicensed_body = await _probe(
        _build_app(_FakeLicensesClient(active=None, inventory=[]))
    )
    clear_license_status_cache()
    outage_status, outage_body = await _probe(_build_app(_FakeLicensesClient(raises=True)))

    assert (unlicensed_status, unlicensed_body["detail"]["error"]) == (402, "license_none")
    assert (outage_status, outage_body["detail"]["error"]) == (503, "license_unavailable")
    assert unlicensed_status != outage_status
    assert unlicensed_body["detail"]["error"] != outage_body["detail"]["error"]


# ─── G7: licences 403 tenant_not_accessible = "not onboarded to w2v" ─────


class _NotOnboardedLicenses(_FakeLicensesClient):
    """licences migration 050: another app owns the tenant and granted w2v no
    access — every tenant-scoped read answers 403 ``tenant_not_accessible``."""

    def __init__(self, *, on_inventory_only: bool = False) -> None:
        super().__init__()
        self.on_inventory_only = on_inventory_only

    def _refuse(self) -> TenantNotOnboardedError:
        return TenantNotOnboardedError(
            "returned 403: tenant_not_accessible", status_code=403, code="tenant_not_accessible"
        )

    async def get_active_license(self, *, tenant_id: str, app_key: str) -> ActiveLicenseView | None:
        self.active_calls += 1
        if self.on_inventory_only:
            # The primary route refuses the SA (401/403 role) → inventory fallback.
            raise LicensesClientError("get_active_license returned 403", status_code=403)
        raise self._refuse()

    async def list_user_licenses(
        self, subject_id: str | None, *, tenant_id: str
    ) -> list[LicenseView]:
        raise self._refuse()


@pytest.mark.parametrize("on_inventory_only", [False])
async def test_gate_on_not_onboarded_is_403_tenant_not_onboarded(
    monkeypatch, on_inventory_only: bool
) -> None:
    """Distinct from 402 ``license_none`` (buy a licence), 503 (could not check)
    and 502 (our misconfiguration): a 403 naming the tenant as not onboarded —
    whichever licences read reported it."""
    _enable_gate(monkeypatch)
    fake = _NotOnboardedLicenses(on_inventory_only=on_inventory_only)
    app = _build_app(fake)
    register_tenancy_exception_handlers(app)
    r = await _probe_response(app)
    assert r.status_code == 403, r.text
    assert r.json()["error"]["code"] == "TENANT_NOT_ONBOARDED"
    assert "retry-after" not in r.headers
