"""Partner create registers the tenant in licences (licences migration 050).

``POST /api/tenants`` must ``PUT /tenants/{id}`` ``{name}`` in licences — w2v becomes
the owning app — before anything is written in w2v, and must turn licences'
ownership refusals into clear, distinct answers with no w2v row left behind.
A tenant licences reports as ``tenant_not_accessible`` (another app's, no grant)
is "not onboarded", never an outage.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4, uuid7

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes import _common
from src.api.routes.tenancy import partner
from src.api.routes.tenancy.errors import register_tenancy_exception_handlers
from src.clients.licenses_client import (
    LicencesAccessRefusedError,
    LicencesTenantOwnedByOtherAppError,
)
from src.database.models import Tenant
from tests.api.routes._p9_licences import P9Licences
from tests.api.routes.test_tenancy_partner import (
    _allowlist,  # noqa: F401 — autouse: the partner licence allowlist
    _body,
    _FakeInviter,
    _licence,
)

pytestmark = pytest.mark.asyncio


def _app(db_session_factory, lic: P9Licences) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = db_session_factory
    app.include_router(partner.router)
    register_tenancy_exception_handlers(app)
    licence = _licence()

    class _SeedLicenceLane(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.license = licence
            return await call_next(request)

    app.add_middleware(_SeedLicenceLane)
    app.dependency_overrides[_common.get_inviter] = lambda: _FakeInviter()
    app.dependency_overrides[_common.get_licenses_client] = lambda: lic
    return app


async def _post(app: FastAPI, body: dict[str, Any]):
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        return await c.post("/api/tenants", json=body)


async def _tenant_row(db_session_factory, tid: UUID) -> Tenant | None:
    async with db_session_factory() as s:
        return (await s.execute(select(Tenant).where(Tenant.id == tid))).scalar_one_or_none()


async def test_licences_name_is_the_slug_of_the_tenant_name() -> None:
    tid = uuid7()
    assert partner.licences_tenant_name("Acme Partner Co", tid) == "acme-partner-co"
    assert partner.licences_tenant_name("  Ørsted & Café — NL ", tid) == "rsted-cafe-nl"
    assert partner.licences_tenant_name("東京", tid) == f"tenant-{tid}"


async def test_create_registers_the_tenant_in_licences_and_replay_re_registers(
    db_session_factory,
) -> None:
    lic = P9Licences()
    tid = uuid7()
    app = _app(db_session_factory, lic)

    first = await _post(app, _body(tid))
    assert first.status_code == 201, first.text
    assert lic.registered == [(str(tid), "acme-partner-co")]
    row = await _tenant_row(db_session_factory, tid)
    assert row is not None and row.name == "Acme Partner Co"  # w2v keeps the display name

    second = await _post(app, _body(tid))
    assert second.status_code == 200, second.text
    # Idempotent at licences too (200 there): a redrive also repairs a missed registration.
    assert lic.registered == [(str(tid), "acme-partner-co")] * 2


async def test_a_tenant_not_onboarded_to_w2v_is_not_an_outage(db_session_factory) -> None:
    """``/licenses/active`` 403 ``tenant_not_accessible`` = no w2v licence visible;
    the create proceeds to registration instead of answering 502."""
    tid = uuid7()
    lic = P9Licences()
    lic.not_onboarded.add(tid)

    r = await _post(_app(db_session_factory, lic), _body(tid))

    assert r.status_code == 201, r.text
    assert lic.registered == [(str(tid), "acme-partner-co")]


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (
            LicencesTenantOwnedByOtherAppError(
                "register_tenant returned 409", status_code=409, code="tenant_owned_by_other_app"
            ),
            409,
            "TENANT_OWNED_BY_OTHER_APP",
        )
    ],
)
async def test_a_registration_refusal_creates_nothing_in_w2v(
    db_session_factory, error: Exception, status: int, code: str
) -> None:
    lic = P9Licences()
    lic.register_error = error
    tid = uuid7()

    r = await _post(_app(db_session_factory, lic), _body(tid))

    assert r.status_code == status, r.text
    assert r.json()["error"]["code"] == code
    assert await _tenant_row(db_session_factory, tid) is None, "nothing created in w2v"
    if code == "LICENCES_TENANT_NAME_TAKEN":
        assert r.json()["error"]["details"] == {"licences_name": "acme-partner-co"}


async def test_registration_refusal_never_leaks_the_upstream_body(db_session_factory) -> None:
    lic = P9Licences()
    lic.register_error = LicencesAccessRefusedError(
        'register_tenant returned 403: {"detail":{"error":"unknown_api_client",'
        '"message":"Bearer eyJhbGc.eyJzdWI.sig"}}',
        status_code=403,
        code="unknown_api_client",
    )
    r = await _post(_app(db_session_factory, lic), _body(uuid7()))
    assert r.status_code == 502, r.text
    assert "eyJ" not in r.text


async def test_no_tenant_id_registers_the_minted_id(db_session_factory) -> None:
    lic = P9Licences()
    body = {"name": "Fresh Co", "admin_email": f"a-{uuid4()}@fresh.test"}

    r = await _post(_app(db_session_factory, lic), body)

    assert r.status_code == 201, r.text
    assert lic.registered == [(r.json()["tenant_id"], "fresh-co")]
