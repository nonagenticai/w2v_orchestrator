"""Gate 2 regression — the license gate must survive a 403 on ``/licenses/active``.

WHY THIS FILE EXISTS (the live defect it pins):
``GET /licenses/active`` is guarded by the licenses service's
``require_internal_service`` — one of {service-internal, metering-internal,
onboarding-internal, license-admin}. The ``w2v-backend`` service account holds
only ``license-reader`` + ``license-manager``, so that route answered **403 on
every call, continuously, in production**. The 403 unwound past the inventory
fallback into ``get_license_status``'s blanket ``except``, which returned
``unknown`` — a state ``_LICENSE_BLOCKING_STATES`` does not block. The gate was
mounted on every tenant-scoped router, the flag was on, and it enforced
NOTHING.

Every pre-existing test in ``test_license_gate.py`` drives a ``_FakeLicensesClient``
that returns scripted objects and therefore *cannot fail authorization* — which
is exactly why the hole survived. So these tests drive the **real**
``LicensesClient`` (real HTTP verbs, real status handling, real error type)
against an ASGI stub that reproduces the licenses service's auth contract:

  * ``GET /licenses/active``       → 403 ``{"detail": "Internal service role required"}``
    unless the principal holds an internal-service role (main.py:812 +
    auth/dependencies.py ``INTERNAL_SERVICE_ROLES``);
  * ``GET /licenses/user/{sub}``   → 200 ``list[LicenseResponse]`` for
    ``license-admin`` OR ``license-reader`` (main.py:963), returning the
    subject's licenses across **ALL tenants** (main.py:979) — hence the
    cross-tenant leak tests below.

⚠️ FOLLOW-UP: the licenses repo (``/Movies/nonagenticai/licenses``) is a separate
repo and is NOT an installed dependency here, so this cannot import the real
``licenses.main`` app. The stub is hand-derived from that source. A true
cross-repo contract test (running the real licenses ASGI app in-process) is
still owed; if the licenses service ever changes its guard on either route,
this stub goes stale silently.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid7

import pytest
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes._common import W2V_APP_KEY, get_licenses_client, require_license
from src.clients.licenses_client import LicensesClient, LicensesClientError
from src.config import clear_settings_cache
from src.services.license_status import clear_license_status_cache, get_license_status

pytestmark = pytest.mark.asyncio

# The licenses roles the w2v-backend SA actually holds (gitops #1035). NOT an
# internal-service role — that is the whole point of this file.
W2V_BACKEND_ROLES = ("license-reader", "license-manager")

_INTERNAL_SERVICE_ROLES = frozenset(
    {"service-internal", "metering-internal", "onboarding-internal", "license-admin"}
)
_INVENTORY_ROLES = frozenset({"license-admin", "license-reader"})


def _iso(days: int) -> str:
    return (datetime.now(UTC) + timedelta(days=days)).isoformat()


def _license_row(
    *,
    tenant_id: str,
    app_key: str = W2V_APP_KEY,
    status: str = "active",
    expires_at: str | None = None,
    revoked_at: str | None = None,
    plan_code: str = "standard",
) -> dict[str, Any]:
    """A ``LicenseResponse``-shaped row (licenses/main.py:612-629)."""
    return {
        "id": str(uuid7()),
        "subject_id": str(uuid7()),
        "tenant_id": tenant_id,
        "status": status,
        "starts_at": _iso(-30),
        "expires_at": expires_at,
        "revoked_at": revoked_at,
        "app": {"id": str(uuid7()), "key": app_key, "name": app_key.upper()},
        "plan": {"id": str(uuid7()), "code": plan_code, "name": plan_code.title()},
    }


class _LicensesServiceStub:
    """Reproduces the licenses service's AUTH contract for the two gate routes.

    Roles ride in the Bearer token as ``roles:a,b`` so the stub can authorize the
    way the real service does (from the SA JWT's realm roles) without a Keycloak.
    """

    def __init__(
        self,
        *,
        inventory: list[dict[str, Any]] | None = None,
        inventory_status: int = 200,
        tenant_inventory: dict[str, list[dict[str, Any]]] | None = None,
        svc_route_missing: bool = False,
    ) -> None:
        self.svc_route_missing = svc_route_missing
        self.inventory = inventory or []
        self.inventory_status = inventory_status
        # Tenant-level holders, keyed ``svc:<tenant>:<app>``. A svc subject not in
        # here answers 404 "Subject not found" — exactly what the licenses
        # service does for a tenant that was never minted a tenant-level licence.
        self.tenant_inventory = tenant_inventory or {}
        self.active_calls = 0
        self.inventory_calls = 0
        self.tenant_inventory_calls: list[str] = []
        self.app = FastAPI()
        self.app.get("/licenses/active")(self._active)
        self.app.get("/licenses/user/{subject_id}")(self._user)

    @staticmethod
    def _roles(request: Request) -> frozenset[str]:
        auth = request.headers.get("authorization", "")
        token = auth.removeprefix("Bearer ")
        if not token.startswith("roles:"):
            return frozenset()
        return frozenset(token.removeprefix("roles:").split(","))

    async def _active(self, request: Request) -> JSONResponse:
        self.active_calls += 1
        if not self._roles(request) & _INTERNAL_SERVICE_ROLES:
            # Verbatim from licenses/auth/dependencies.py require_internal_service.
            return JSONResponse(
                status_code=403, content={"detail": "Internal service role required"}
            )
        return JSONResponse(status_code=404, content={"detail": "No active license"})

    async def _user(self, subject_id: str, request: Request) -> JSONResponse:
        if not self._roles(request) & _INVENTORY_ROLES:
            return JSONResponse(status_code=403, content={"detail": "Insufficient role"})
        if subject_id.startswith("svc:"):
            self.tenant_inventory_calls.append(subject_id)
            if self.svc_route_missing:
                # FastAPI's bare unmatched-route body — NOT "Subject not found".
                return JSONResponse(status_code=404, content={"detail": "Not Found"})
            rows = self.tenant_inventory.get(subject_id)
            if rows is None:
                return JSONResponse(status_code=404, content={"detail": "Subject not found"})
            return JSONResponse(status_code=200, content=rows)
        self.inventory_calls += 1
        if self.inventory_status != 200:
            return JSONResponse(
                status_code=self.inventory_status, content={"detail": "Subject not found"}
            )
        return JSONResponse(status_code=200, content=self.inventory)


class _RoleToken:
    """TokenProvider yielding a principal with the given licenses roles."""

    def __init__(self, roles: tuple[str, ...]) -> None:
        self._roles = roles

    async def token(self) -> str:
        return "roles:" + ",".join(self._roles)


def _real_client(stub: _LicensesServiceStub, roles: tuple[str, ...]) -> LicensesClient:
    """The REAL LicensesClient, wired to the stub over an in-process ASGI transport."""
    http = AsyncClient(
        transport=ASGITransport(app=stub.app), base_url="http://licenses.test", timeout=5.0
    )
    return LicensesClient(
        authenticator=_RoleToken(roles), base_url="http://licenses.test", client=http
    )


@pytest.fixture(autouse=True)
def _reset_state():
    clear_settings_cache()
    clear_license_status_cache()
    yield
    clear_settings_cache()
    clear_license_status_cache()


async def _status(
    stub: _LicensesServiceStub,
    *,
    tenant_id: str,
    subject_id: str | None,
    roles: tuple[str, ...] = W2V_BACKEND_ROLES,
):
    return await get_license_status(
        client=_real_client(stub, roles),
        app_key=W2V_APP_KEY,
        tenant_id=tenant_id,
        subject_id=subject_id,
        gate_enabled=True,
    )


# ─── the bug itself: /licenses/active really does 403 this principal ─────


async def test_w2v_backend_principal_is_403ed_by_licenses_active() -> None:
    """Pins the live root cause: the SA's roles do not satisfy require_internal_service."""
    stub = _LicensesServiceStub()
    client = _real_client(stub, W2V_BACKEND_ROLES)
    with pytest.raises(LicensesClientError) as exc:
        await client.get_active_license(tenant_id=str(uuid7()), app_key=W2V_APP_KEY)
    assert exc.value.status_code == 403
    assert "Internal service role required" in str(exc.value)
    assert stub.active_calls == 1


# ─── THE anti-402-storm proof ────────────────────────────────────────────


async def test_active_license_reachable_only_via_inventory_is_entitled() -> None:
    """A genuinely ACTIVE license, visible ONLY through the inventory route, must
    resolve to an ENTITLED state.

    This is the assertion that matters most in this PR. The naive fix — let the
    403 fall through to ``derive_inactive_state`` — returns ``none`` here, and
    ``none`` is blocking, so every paying customer would be 402'd.
    """
    tenant = str(uuid7())
    stub = _LicensesServiceStub(inventory=[_license_row(tenant_id=tenant, expires_at=_iso(180))])
    result = await _status(stub, tenant_id=tenant, subject_id=str(uuid7()))
    assert result.state == "active"
    assert result.has_license is True
    assert result.plan_code == "standard"
    assert stub.inventory_calls == 1


async def test_inventory_expiring_soon_is_still_entitled() -> None:
    tenant = str(uuid7())
    stub = _LicensesServiceStub(inventory=[_license_row(tenant_id=tenant, expires_at=_iso(5))])
    result = await _status(stub, tenant_id=tenant, subject_id=str(uuid7()))
    assert result.state == "expiring_soon"
    assert result.has_license is True


async def test_inventory_perpetual_license_is_entitled() -> None:
    tenant = str(uuid7())
    stub = _LicensesServiceStub(inventory=[_license_row(tenant_id=tenant, expires_at=None)])
    result = await _status(stub, tenant_id=tenant, subject_id=str(uuid7()))
    assert result.state == "active"


# ─── unentitled still resolves correctly through the fallback ────────────


async def test_inventory_revoked_is_blocking() -> None:
    tenant = str(uuid7())
    stub = _LicensesServiceStub(
        inventory=[
            _license_row(
                tenant_id=tenant, status="revoked", revoked_at=_iso(-1), expires_at=_iso(90)
            )
        ]
    )
    result = await _status(stub, tenant_id=tenant, subject_id=str(uuid7()))
    assert result.state == "revoked"
    assert result.has_license is False


async def test_inventory_expired_is_blocking() -> None:
    tenant = str(uuid7())
    stub = _LicensesServiceStub(
        inventory=[_license_row(tenant_id=tenant, status="expired", expires_at=_iso(-3))]
    )
    result = await _status(stub, tenant_id=tenant, subject_id=str(uuid7()))
    assert result.state == "expired"


async def test_empty_inventory_is_none() -> None:
    stub = _LicensesServiceStub(inventory=[])
    result = await _status(stub, tenant_id=str(uuid7()), subject_id=str(uuid7()))
    assert result.state == "none"


# ─── scope: neither a foreign tenant nor a foreign app may entitle ───────


async def test_foreign_tenant_license_does_not_entitle() -> None:
    """/licenses/user/{sub} returns the subject's licenses across ALL tenants
    (licenses/main.py:979), so an unscoped read would let tenant A's license
    entitle a request made in tenant B."""
    stub = _LicensesServiceStub(
        inventory=[_license_row(tenant_id=str(uuid7()), expires_at=_iso(180))]
    )
    result = await _status(stub, tenant_id=str(uuid7()), subject_id=str(uuid7()))
    assert result.state == "none"
    assert result.has_license is False


async def test_other_app_license_does_not_entitle() -> None:
    tenant = str(uuid7())
    stub = _LicensesServiceStub(
        inventory=[_license_row(tenant_id=tenant, app_key="kanban", expires_at=_iso(180))]
    )
    result = await _status(stub, tenant_id=tenant, subject_id=str(uuid7()))
    assert result.state == "none"
    assert result.has_license is False


async def test_license_without_tenant_id_does_not_entitle() -> None:
    """A row we cannot positively scope to this tenant must never entitle."""
    row = _license_row(tenant_id="", expires_at=_iso(180))
    row["tenant_id"] = None
    stub = _LicensesServiceStub(inventory=[row])
    result = await _status(stub, tenant_id=str(uuid7()), subject_id=str(uuid7()))
    assert result.state == "none"


# ─── loud failure + fail-soft ────────────────────────────────────────────


async def test_403_logs_the_license_gate_misconfigured_marker(caplog) -> None:
    tenant = str(uuid7())
    stub = _LicensesServiceStub(inventory=[_license_row(tenant_id=tenant, expires_at=_iso(90))])
    with caplog.at_level(logging.ERROR, logger="w2v_orchestrator.license_status"):
        await _status(stub, tenant_id=tenant, subject_id=str(uuid7()))
    misconfigured = [
        r
        for r in caplog.records
        if r.levelno >= logging.ERROR and "license_gate_misconfigured" in r.getMessage()
    ]
    assert misconfigured, "the 403 must be loud — an ERROR with a greppable marker"
    message = misconfigured[0].getMessage()
    assert tenant in message
    assert W2V_APP_KEY in message


async def test_403_then_failing_inventory_is_unknown_not_none() -> None:
    """The trap: if the inventory lookup ITSELF fails we know NOTHING, so the
    state must be 'unknown' (non-blocking). Returning 'none' here — the naive
    reading — would 402 every tenant during a licenses outage."""
    stub = _LicensesServiceStub(inventory_status=404)
    result = await _status(stub, tenant_id=str(uuid7()), subject_id=str(uuid7()))
    assert result.state == "unknown"
    assert result.has_license is False


async def test_403_with_no_subject_and_no_tenant_holder_is_unknown_not_none() -> None:
    """No subject (the S2S ``require_license`` path) reads the tenant's
    ``svc:<tenant>:<app>`` holder. When that answers 404 Subject not found, the
    tenant may still hold per-MEMBER licences (minting is still per-user) that we
    cannot see without a subject — so this is ``unknown`` (503), never ``none``
    (402). origin/main returned ``unknown`` here; deriving ``none`` would 402 a
    tenant whose licences all sit on its members."""
    tenant = str(uuid7())
    stub = _LicensesServiceStub()
    result = await _status(stub, tenant_id=tenant, subject_id=None)
    assert result.state == "unknown"
    assert result.has_license is False
    assert stub.tenant_inventory_calls == [f"svc:{tenant}:{W2V_APP_KEY}"]
    assert stub.inventory_calls == 0


async def test_403_no_tenant_holder_and_empty_caller_inventory_is_none() -> None:
    """The ONLY way the tenant holder's Subject-not-found yields ``none``: the
    caller's own inventory was also fetched and holds nothing for this tenant."""
    tenant = str(uuid7())
    stub = _LicensesServiceStub(inventory=[])
    result = await _status(stub, tenant_id=tenant, subject_id=str(uuid7()))
    assert result.state == "none"
    assert stub.tenant_inventory_calls == [f"svc:{tenant}:{W2V_APP_KEY}"]
    assert stub.inventory_calls == 1


async def test_403_with_no_subject_and_non_uuid_tenant_is_unknown_not_none() -> None:
    """Neither key usable ⇒ the inventory cannot be consulted at all ⇒ unknown."""
    stub = _LicensesServiceStub()
    result = await _status(stub, tenant_id="not-a-uuid", subject_id=None)
    assert result.state == "unknown"


# ─── BACKLOG #3: the fallback is keyed by TENANT, not subject ─────────────


async def test_403_tenant_only_licence_is_entitled() -> None:
    """The tenant holds ONLY ``svc:<tenant>:w2v``; the caller's own inventory is
    empty. Before the fix the fallback read only ``/licenses/user/{sub}`` and
    derived ``none`` ⇒ 402 for a paying tenant."""
    tenant = str(uuid7())
    svc = f"svc:{tenant}:{W2V_APP_KEY}"
    stub = _LicensesServiceStub(
        inventory=[],
        tenant_inventory={svc: [_license_row(tenant_id=tenant, expires_at=_iso(180))]},
    )
    result = await _status(stub, tenant_id=tenant, subject_id=str(uuid7()))
    assert result.state == "active"
    assert result.has_license is True
    assert stub.tenant_inventory_calls == [svc]


async def test_403_tenant_only_licence_entitles_the_subjectless_admission_path() -> None:
    tenant = str(uuid7())
    svc = f"svc:{tenant}:{W2V_APP_KEY}"
    stub = _LicensesServiceStub(
        tenant_inventory={svc: [_license_row(tenant_id=tenant, expires_at=_iso(180))]}
    )
    result = await _status(stub, tenant_id=tenant, subject_id=None)
    assert result.state == "active"


async def test_403_another_tenants_holder_does_not_entitle() -> None:
    """Only THIS tenant's holder is read — another tenant's svc licence is never consulted."""
    tenant, other = str(uuid7()), str(uuid7())
    stub = _LicensesServiceStub(
        tenant_inventory={
            f"svc:{other}:{W2V_APP_KEY}": [_license_row(tenant_id=other, expires_at=_iso(180))]
        }
    )
    result = await _status(stub, tenant_id=tenant, subject_id=str(uuid7()))
    assert result.state == "none"
    assert result.has_license is False


async def test_403_tenant_holder_upstream_error_is_unknown_not_none() -> None:
    """Only a 404 (Subject not found) is an empty tenant inventory; a 403 on the
    tenant-level read is a failure ⇒ unknown, never a fabricated ``none``."""
    stub = _LicensesServiceStub()
    result = await _status(stub, tenant_id=str(uuid7()), subject_id=None, roles=("other",))
    assert result.state == "unknown"


async def test_403_tenant_holder_route_missing_404_is_unknown_not_none() -> None:
    """A 404 that is NOT ``Subject not found`` (route missing / proxy) proves
    nothing about the tenant's licences ⇒ unknown, never a fabricated ``none``."""
    stub = _LicensesServiceStub(svc_route_missing=True)
    result = await _status(stub, tenant_id=str(uuid7()), subject_id=None)
    assert result.state == "unknown"


# ─── caching ─────────────────────────────────────────────────────────────


async def test_result_is_cached_within_ttl() -> None:
    tenant, subject = str(uuid7()), str(uuid7())
    stub = _LicensesServiceStub(inventory=[_license_row(tenant_id=tenant, expires_at=_iso(180))])
    first = await _status(stub, tenant_id=tenant, subject_id=subject)
    second = await _status(stub, tenant_id=tenant, subject_id=subject)
    assert first.state == second.state == "active"
    assert stub.active_calls == 1, "second call inside the TTL must not hit upstream"
    assert stub.inventory_calls == 1


async def test_distinct_tenants_do_not_share_a_cache_entry() -> None:
    tenant_a, tenant_b, subject = str(uuid7()), str(uuid7()), str(uuid7())
    stub = _LicensesServiceStub(inventory=[_license_row(tenant_id=tenant_a, expires_at=_iso(180))])
    assert (await _status(stub, tenant_id=tenant_a, subject_id=subject)).state == "active"
    # Same subject, different tenant: the cache key must not collide, and the
    # tenant filter must deny.
    assert (await _status(stub, tenant_id=tenant_b, subject_id=subject)).state == "none"


async def test_upstream_error_after_a_good_hit_serves_stale_not_unknown() -> None:
    """Stale-while-error: a tenant that resolved cleanly moments ago must not be
    downgraded to 'unknown' by a transient blip."""
    from src.services import license_status

    tenant, subject = str(uuid7()), str(uuid7())
    good = _LicensesServiceStub(inventory=[_license_row(tenant_id=tenant, expires_at=_iso(180))])
    assert (await _status(good, tenant_id=tenant, subject_id=subject)).state == "active"

    # Expire the positive entry only; the longer-lived stale entry survives.
    license_status._status_cache.clear()

    broken = _LicensesServiceStub(inventory_status=503)
    result = await _status(broken, tenant_id=tenant, subject_id=subject)
    assert result.state == "active", "must serve last-known-good, not 'unknown'"
    assert result.has_license is True


# ─── end-to-end through the actual gate dependency ───────────────────────

_gated = APIRouter()


@_gated.get("/api/_probe/ping")
async def _ping() -> dict[str, bool]:
    return {"ok": True}


def _build_app(stub: _LicensesServiceStub, tenant_id: str) -> FastAPI:
    app = FastAPI()
    app.include_router(_gated, dependencies=[Depends(require_license)])
    app.dependency_overrides[get_licenses_client] = lambda: _real_client(stub, W2V_BACKEND_ROLES)

    class _SeedIdentity(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = tenant_id
            request.state.subject_id = str(uuid7())
            request.state.roles = []
            return await call_next(request)

    app.add_middleware(_SeedIdentity)
    return app


async def _probe(app: FastAPI) -> tuple[int, dict[str, Any]]:
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        r = await c.get("/api/_probe/ping")
        return r.status_code, r.json()


async def test_gate_allows_an_inventory_only_active_license(monkeypatch) -> None:
    """THE regression: gate ON, /licenses/active 403s, license is real → ALLOW."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    clear_settings_cache()
    tenant = str(uuid7())
    stub = _LicensesServiceStub(inventory=[_license_row(tenant_id=tenant, expires_at=_iso(180))])
    status_code, _ = await _probe(_build_app(stub, tenant))
    assert status_code == 200
    # A bare 200 also happens when the gate is silently unenforced (the bug), so
    # pin that the 403 really was refined through the inventory route.
    assert stub.active_calls == 1
    assert stub.inventory_calls == 1


async def test_gate_402s_a_revoked_license_through_the_fallback(monkeypatch) -> None:
    """And the gate must still BITE — proving the fallback is not blanket-allow."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    clear_settings_cache()
    tenant = str(uuid7())
    stub = _LicensesServiceStub(
        inventory=[_license_row(tenant_id=tenant, status="revoked", revoked_at=_iso(-1))]
    )
    status_code, body = await _probe(_build_app(stub, tenant))
    assert status_code == 402
    assert body["detail"]["error"] == "license_revoked"


async def test_gate_402s_a_foreign_tenants_license(monkeypatch) -> None:
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    clear_settings_cache()
    stub = _LicensesServiceStub(
        inventory=[_license_row(tenant_id=str(uuid7()), expires_at=_iso(180))]
    )
    status_code, body = await _probe(_build_app(stub, str(uuid7())))
    assert status_code == 402
    assert body["detail"]["error"] == "license_none"


# ─── request-local fast path (request.state.license) ─────────────────────
# AuthMiddleware stashes the licenses service's introspection response there;
# when it proves BOTH the app and the tenant, the gate needs no HTTP hop.


def _introspect(
    *,
    tenant_id: str,
    app_key: str = W2V_APP_KEY,
    active: bool = True,
    expires_at: str | None = None,
) -> dict[str, Any]:
    """A ``TokenIntrospectResponse``-shaped stash (licenses/main.py:457-472).
    ``app`` is ``App.to_dict()``, so ``key`` is always present."""
    return {
        "active": active,
        "tenant_id": tenant_id,
        "license_id": str(uuid7()),
        "subject_id": str(uuid7()),
        "app": {"id": str(uuid7()), "key": app_key, "name": app_key.upper()},
        "plan": {"id": str(uuid7()), "code": "standard", "name": "Standard"},
        "status": "active",
        "expires_at": expires_at,
    }


def _build_app_with_stash(
    stub: _LicensesServiceStub, tenant_id: str, stash: dict[str, Any] | None
) -> FastAPI:
    app = FastAPI()
    app.include_router(_gated, dependencies=[Depends(require_license)])
    app.dependency_overrides[get_licenses_client] = lambda: _real_client(stub, W2V_BACKEND_ROLES)

    class _SeedIdentity(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = tenant_id
            request.state.subject_id = str(uuid7())
            request.state.roles = []
            request.state.license = stash
            return await call_next(request)

    app.add_middleware(_SeedIdentity)
    return app


async def test_fast_path_allows_and_skips_the_http_hop(monkeypatch) -> None:
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    clear_settings_cache()
    tenant = str(uuid7())
    stub = _LicensesServiceStub(inventory=[])  # would derive 'none' if consulted
    stash = _introspect(tenant_id=tenant, expires_at=_iso(180))
    status_code, _ = await _probe(_build_app_with_stash(stub, tenant, stash))
    assert status_code == 200
    assert stub.active_calls == 0 and stub.inventory_calls == 0


async def test_fast_path_402s_an_expired_stashed_license(monkeypatch) -> None:
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    clear_settings_cache()
    tenant = str(uuid7())
    stub = _LicensesServiceStub(inventory=[])
    stash = _introspect(tenant_id=tenant, expires_at=_iso(-1))
    status_code, body = await _probe(_build_app_with_stash(stub, tenant, stash))
    assert status_code == 402
    assert body["detail"]["error"] == "license_expired"


async def test_fast_path_ignores_a_license_for_another_app(monkeypatch) -> None:
    """A license minted for a DIFFERENT product must never entitle w2v — it falls
    through to the upstream lookup, which here finds nothing → 402."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    clear_settings_cache()
    tenant = str(uuid7())
    stub = _LicensesServiceStub(inventory=[])
    stash = _introspect(tenant_id=tenant, app_key="kanban", expires_at=_iso(180))
    status_code, body = await _probe(_build_app_with_stash(stub, tenant, stash))
    assert status_code == 402
    assert body["detail"]["error"] == "license_none"
    assert stub.inventory_calls == 1, "must fall through, not shortcut on a foreign app"


async def test_fast_path_ignores_a_license_for_another_tenant(monkeypatch) -> None:
    """The effective tenant can be a super-admin scope-switch, so the token's own
    tenant is not automatically the tenant being gated."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    clear_settings_cache()
    stub = _LicensesServiceStub(inventory=[])
    stash = _introspect(tenant_id=str(uuid7()), expires_at=_iso(180))
    status_code, body = await _probe(_build_app_with_stash(stub, str(uuid7()), stash))
    assert status_code == 402
    assert body["detail"]["error"] == "license_none"


async def test_fast_path_ignores_an_inactive_stash(monkeypatch) -> None:
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    clear_settings_cache()
    tenant = str(uuid7())
    stub = _LicensesServiceStub(inventory=[_license_row(tenant_id=tenant, expires_at=_iso(180))])
    stash = _introspect(tenant_id=tenant, active=False)
    status_code, _ = await _probe(_build_app_with_stash(stub, tenant, stash))
    # Falls through to the inventory fallback, which finds the real license.
    assert status_code == 200
    assert stub.inventory_calls == 1
