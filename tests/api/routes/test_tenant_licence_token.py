"""Tenant licence token on the partner + admin surfaces (introspection, mig 0089).

* ``POST /api/tenants`` ``licence_token``: introspected (active ∧ this tenant ∧
  app w2v, else 422 ``LICENCE_TOKEN_MISMATCH``), stored SEALED, its licence is
  ``licence_ref``; a replay with a NEW token replaces it (lease + CAS), the same
  token is a no-op; without a token a first party's ref is set only when the
  tenant holds exactly ONE active licence.
* ``PUT /api/tenants/{id}/licence-token``: owning partner or super_admin; 409
  ``licence_busy`` (lease held) / ``licence_changed_concurrently`` (CAS lost);
  committed before the response.
* The raw token never appears in a response, the audit log or the logs.

The licences service is the ``P9Licences`` fake; tenant / audit rows are real.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, cast
from uuid import UUID, uuid4, uuid7

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, update
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes import _common
from src.api.routes.admin import router as admin_router
from src.api.routes.tenancy import partner
from src.api.routes.tenancy.errors import register_tenancy_exception_handlers
from src.database.models import AdminAuditLog, Tenant
from src.database.repositories import tenant_licence_repository as repo_mod
from src.services.providers.crypto import Keyring
from src.services.tenant_licence import open_licence_token
from tests.api.routes._p9_licences import TEST_KEYRING, P9Licences, override_licence_keyring
from tests.api.routes.test_tenancy_partner import (  # noqa: F401 — autouse fixture
    _OTHER_PARTNER,
    _PARTNER,
    _allowlist,
    _FakeInviter,
    _licence,
    _project,
)

pytestmark = pytest.mark.asyncio


def _app(
    factory,
    lic: P9Licences,
    *,
    licence: dict[str, Any] | None = None,
    roles: list[str] | None = None,
    keyring: Keyring = TEST_KEYRING,
) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = factory
    app.include_router(partner.router)
    app.include_router(admin_router)
    register_tenancy_exception_handlers(app)

    class _Seed(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.license = licence
            request.state.roles = roles or []
            request.state.tenant_id = str(uuid4())
            request.state.subject_id = str(uuid4())
            return await call_next(request)

    app.add_middleware(_Seed)
    app.dependency_overrides[_common.get_inviter] = lambda: _FakeInviter()
    app.dependency_overrides[_common.get_licenses_client] = lambda: lic
    override_licence_keyring(app, keyring)
    return app


async def _call(app: FastAPI, method: str, path: str, json: Any = None):
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        return await c.request(method, path, json=json)


async def _tenant(factory, tid: UUID) -> Tenant | None:
    async with factory() as s:
        return (await s.execute(select(Tenant).where(Tenant.id == tid))).scalar_one_or_none()


async def _actions(factory, tid: UUID) -> list[tuple[str, dict]]:
    async with factory() as s:
        rows = await s.execute(
            select(AdminAuditLog.action, AdminAuditLog.detail)
            .where(AdminAuditLog.target_tenant_id == tid)
            .order_by(AdminAuditLog.created_at)
        )
        return [(r.action, r.detail) for r in rows]


def _stored_token(tenant: Tenant) -> str:
    assert tenant.licence_token_ciphertext is not None
    return open_licence_token(TEST_KEYRING, tenant.id, bytes(tenant.licence_token_ciphertext))


# ─── POST /api/tenants with a licence token ────────────────────────────────


async def test_create_with_token_stores_it_sealed_and_records_the_ref(
    db_session_factory, caplog
) -> None:
    tid, licence = uuid7(), uuid7()
    lic = P9Licences()
    token = lic.token(tid, licence, token_expires_at="2027-10-01T00:00:00+00:00")
    app = _app(db_session_factory, lic, licence=_licence())
    with caplog.at_level(logging.DEBUG):
        r = await _call(
            app,
            "POST",
            "/api/tenants",
            {"tenant_id": str(tid), "name": "t", "licence_token": token},
        )
    assert r.status_code == 201, r.text
    assert r.json()["licence_ref"] == str(licence)
    tenant = await _tenant(db_session_factory, tid)
    assert tenant is not None and tenant.licence_ref == licence and tenant.status == "active"
    assert tenant.licence_token_kid == "lt1" and tenant.licence_token_set_at is not None
    assert tenant.licence_token_expires_at == datetime.datetime(2027, 10, 1, tzinfo=datetime.UTC)
    assert token.encode() not in bytes(tenant.licence_token_ciphertext or b"")
    assert _stored_token(tenant) == token
    # Sealed for THIS tenant: the blob does not open as any other tenant's.
    with pytest.raises(Exception, match="authentication"):
        open_licence_token(TEST_KEYRING, uuid7(), bytes(tenant.licence_token_ciphertext or b""))
    # Never in a response, the audit or a log line.
    assert token not in r.text
    assert token not in str(await _actions(db_session_factory, tid))
    assert token not in caplog.text
    assert "licence_token" not in repr(tenant) and token not in repr(tenant)


@pytest.mark.parametrize(
    ("case", "reason"),
    [("other-tenant", "tenant")],
)
async def test_create_refuses_a_token_that_is_not_this_tenants_w2v(
    db_session_factory, case: str, reason: str
) -> None:
    tid, licence = uuid7(), uuid7()
    lic = P9Licences()
    token = {
        "other-tenant": lic.token(uuid7(), licence),
        "other-app": lic.token(tid, licence, app="fr"),
        "inactive": lic.token(tid, licence, active=False, reason="license_inactive"),
    }[case]
    r = await _call(
        _app(db_session_factory, lic, licence=_licence()),
        "POST",
        "/api/tenants",
        {"tenant_id": str(tid), "name": "t", "licence_token": token},
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "LICENCE_TOKEN_MISMATCH"
    assert r.json()["error"]["details"] == {"reason": reason}
    assert token not in r.text
    assert await _tenant(db_session_factory, tid) is None, "a refused token creates nothing"


async def test_token_without_tenant_id_cannot_match_a_fresh_id(db_session_factory) -> None:
    lic = P9Licences()
    token = lic.token(uuid7(), uuid7())
    r = await _call(
        _app(db_session_factory, lic, licence=_licence()),
        "POST",
        "/api/tenants",
        {"name": "t", "licence_token": token},
    )
    assert r.status_code == 422 and r.json()["error"]["details"] == {"reason": "tenant"}


async def test_holding_the_token_lets_a_third_party_claim_a_licensed_id(
    db_session_factory,
) -> None:
    """Without a token a licensed id is 409 for a third party (no ownership proof);
    WITH the tenant's own token, possession is the proof."""
    tid, licence = uuid7(), uuid7()
    lic = P9Licences({tid: [licence]})
    app = _app(db_session_factory, lic, licence=_licence(_OTHER_PARTNER))
    bare = await _call(app, "POST", "/api/tenants", {"tenant_id": str(tid), "name": "t"})
    assert bare.status_code == 409, bare.text
    token = lic.token(tid, licence)
    r = await _call(
        app, "POST", "/api/tenants", {"tenant_id": str(tid), "name": "t", "licence_token": token}
    )
    assert r.status_code == 201, r.text
    tenant = await _tenant(db_session_factory, tid)
    assert tenant is not None and tenant.licence_ref == licence


async def test_replay_with_same_token_is_a_noop_and_a_new_token_replaces_it(
    db_session_factory,
) -> None:
    """Provisioning mints a NEW token on every attempt (resume / redrive): the
    replay installs it through the rotation path; the same token writes nothing."""
    tid, licence, licence2 = uuid7(), uuid7(), uuid7()
    lic = P9Licences()
    first = lic.token(tid, licence)
    app = _app(db_session_factory, lic, licence=_licence())
    body = {"tenant_id": str(tid), "name": "t", "licence_token": first}
    assert (await _call(app, "POST", "/api/tenants", body)).status_code == 201
    t1 = await _tenant(db_session_factory, tid)
    assert t1 is not None

    same = await _call(app, "POST", "/api/tenants", body)
    assert same.status_code == 200, same.text
    t2 = await _tenant(db_session_factory, tid)
    assert t2 is not None
    assert t2.licence_token_set_at == t1.licence_token_set_at, "same token: nothing written"
    assert bytes(t2.licence_token_ciphertext or b"") == bytes(t1.licence_token_ciphertext or b"")

    second = lic.token(tid, licence2)
    again = await _call(app, "POST", "/api/tenants", {**body, "licence_token": second})
    assert again.status_code == 200, again.text
    assert again.json()["licence_ref"] == str(licence2)
    t3 = await _tenant(db_session_factory, tid)
    assert t3 is not None and _stored_token(t3) == second and t3.licence_ref == licence2
    actions = [a for a, _ in await _actions(db_session_factory, tid)]
    assert actions.count("tenant.licence_token_set") == 2, actions  # create + replacement
    assert t3.licence_op_id is None and t3.licence_op_until is None, "lease released"


@pytest.mark.parametrize(("candidates", "ref_set"), [(1, True)])
async def test_first_party_without_token_records_a_ref_only_for_a_single_licence(
    db_session_factory, monkeypatch, candidates: int, ref_set: bool
) -> None:
    """One live licence → it is the ref (licences' bundle ensure answers
    ``exists``). Several → licences refuses ``several_candidates`` and the create
    FAILS LOUDLY (409 ``LICENCE_REFUSED``) instead of leaving a null ref."""
    from src.config import clear_settings_cache

    monkeypatch.setenv("W2V_PARTNER_FIRST_PARTY_LICENSE_IDS", str(_PARTNER))
    clear_settings_cache()
    tid = uuid7()
    licences = [uuid7() for _ in range(candidates)]
    r = await _call(
        _app(db_session_factory, P9Licences({tid: licences}), licence=_licence()),
        "POST",
        "/api/tenants",
        {"tenant_id": str(tid), "name": "t"},
    )
    clear_settings_cache()
    if ref_set:
        assert r.status_code == 201, r.text
    else:
        assert r.status_code == 409, r.text
        assert r.json()["error"]["code"] == "LICENCE_REFUSED"
        assert r.json()["error"]["details"]["reason"] == "several_candidates"
    tenant = await _tenant(db_session_factory, tid)
    assert tenant is not None
    assert tenant.licence_ref == (licences[0] if ref_set else None)
    assert tenant.licence_token_ciphertext is None


async def test_first_party_tokenless_replay_never_touches_a_token_owned_ref(
    db_session_factory, monkeypatch
) -> None:
    """A stored token owns ``licence_ref``: a token-less first-party create/replay
    whose fallback finds ONE (other) licence must not overwrite it — only the token
    path (lease + CAS) changes it."""
    from src.config import clear_settings_cache

    lic = P9Licences()
    tid, token_ref, token = await _partner_tenant(db_session_factory, lic)
    other = uuid7()
    lic.live[tid] = [other]  # the fallback's single candidate is NOT the token's licence
    monkeypatch.setenv("W2V_PARTNER_FIRST_PARTY_LICENSE_IDS", str(_PARTNER))
    clear_settings_cache()
    try:
        r = await _call(
            _app(db_session_factory, lic, licence=_licence()),
            "POST",
            "/api/tenants",
            {"tenant_id": str(tid), "name": "t"},
        )
    finally:
        clear_settings_cache()
    assert r.status_code == 200, r.text
    assert r.json()["licence_ref"] == str(token_ref), "the response reports the token's ref"
    tenant = await _tenant(db_session_factory, tid)
    assert tenant is not None and tenant.licence_ref == token_ref
    assert _stored_token(tenant) == token


async def test_first_party_tokenless_create_respects_a_held_lease(
    db_session_factory, monkeypatch
) -> None:
    """No token stored, but a licence-change (token install) is in flight: the
    token-less write must not race it."""
    from src.config import clear_settings_cache

    monkeypatch.setenv("W2V_PARTNER_FIRST_PARTY_LICENSE_IDS", str(_PARTNER))
    clear_settings_cache()
    tid, first = uuid7(), uuid7()
    lic = P9Licences({tid: [first]})
    app = _app(db_session_factory, lic, licence=_licence())
    try:
        body = {"tenant_id": str(tid), "name": "t"}
        assert (await _call(app, "POST", "/api/tenants", body)).status_code == 201
        async with db_session_factory() as s:
            await s.execute(
                update(Tenant)
                .where(Tenant.id == tid)
                .values(
                    licence_op_id=uuid7(),
                    licence_op_until=datetime.datetime.now(datetime.UTC)
                    + datetime.timedelta(minutes=5),
                )
            )
            await s.commit()
        lic.live[tid] = [uuid7()]
        r = await _call(app, "POST", "/api/tenants", body)
    finally:
        clear_settings_cache()
    assert r.status_code == 200, r.text
    tenant = await _tenant(db_session_factory, tid)
    assert tenant is not None and tenant.licence_ref == first
    assert r.json()["licence_ref"] == str(first)


async def test_create_with_token_needs_a_keyring(db_session_factory) -> None:
    tid = uuid7()
    lic = P9Licences()
    r = await _call(
        _app(db_session_factory, lic, licence=_licence(), keyring=Keyring([])),
        "POST",
        "/api/tenants",
        {"tenant_id": str(tid), "name": "t", "licence_token": lic.token(tid, uuid7())},
    )
    assert r.status_code == 503 and r.json()["error"]["code"] == "PROVIDER_KEYRING_UNAVAILABLE"
    assert await _tenant(db_session_factory, tid) is None


# ─── PUT /api/tenants/{id}/licence-token ───────────────────────────────────


async def _partner_tenant(factory, lic: P9Licences) -> tuple[UUID, UUID, str]:
    tid, licence = uuid7(), uuid7()
    token = lic.token(tid, licence)
    r = await _call(
        _app(factory, lic, licence=_licence()),
        "POST",
        "/api/tenants",
        {"tenant_id": str(tid), "name": "t", "licence_token": token},
    )
    assert r.status_code == 201, r.text
    return tid, licence, token


async def test_owning_partner_and_super_admin_rotate_the_token(db_session_factory) -> None:
    lic = P9Licences()
    tid, _old, _ = await _partner_tenant(db_session_factory, lic)
    new_licence = uuid7()
    token = lic.token(tid, new_licence)
    r = await _call(
        _app(db_session_factory, lic, licence=_licence()),
        "PUT",
        f"/api/tenants/{tid}/licence-token",
        {"licence_token": token},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["changed"] is True and out["licence_ref"] == str(new_licence)
    assert token not in r.text
    tenant = await _tenant(db_session_factory, tid)
    assert tenant is not None and _stored_token(tenant) == token

    admin_token = lic.token(tid, new_licence)
    a = await _call(
        _app(db_session_factory, lic, roles=["super_admin"]),
        "PUT",
        f"/api/tenants/{tid}/licence-token",
        {"licence_token": admin_token},
    )
    assert a.status_code == 200, a.text
    same = await _call(
        _app(db_session_factory, lic, roles=["super_admin"]),
        "PUT",
        f"/api/tenants/{tid}/licence-token",
        {"licence_token": admin_token},
    )
    assert same.status_code == 200 and same.json()["changed"] is False
    sets = [
        d for a, d in await _actions(db_session_factory, tid) if a == "tenant.licence_token_set"
    ]
    assert [d["by"] for d in sets] == ["partner", "partner", "super_admin"]
    assert sets[1]["previous_licence_ref"] == str(_old)


async def test_rotation_refusals(db_session_factory) -> None:
    lic = P9Licences()
    tid, _, _ = await _partner_tenant(db_session_factory, lic)
    good = lic.token(tid, uuid7())
    foreign = await _call(
        _app(db_session_factory, lic, licence=_licence(_OTHER_PARTNER)),
        "PUT",
        f"/api/tenants/{tid}/licence-token",
        {"licence_token": good},
    )
    assert foreign.status_code == 403, foreign.text
    unknown = await _call(
        _app(db_session_factory, lic, licence=_licence()),
        "PUT",
        f"/api/tenants/{uuid7()}/licence-token",
        {"licence_token": good},
    )
    assert unknown.status_code == 404, unknown.text
    nobody = await _call(
        _app(db_session_factory, lic, roles=["tenant_admin"]),
        "PUT",
        f"/api/tenants/{tid}/licence-token",
        {"licence_token": good},
    )
    assert nobody.status_code == 403, nobody.text
    other_tenant = await _call(
        _app(db_session_factory, lic, licence=_licence()),
        "PUT",
        f"/api/tenants/{tid}/licence-token",
        {"licence_token": lic.token(uuid7(), uuid7())},
    )
    assert other_tenant.status_code == 422
    assert other_tenant.json()["error"]["code"] == "LICENCE_TOKEN_MISMATCH"
    tenant = await _tenant(db_session_factory, tid)
    assert tenant is not None and _stored_token(tenant) != good


async def test_lease_busy_is_409_and_writes_nothing(db_session_factory) -> None:
    lic = P9Licences()
    tid, licence, token = await _partner_tenant(db_session_factory, lic)
    async with db_session_factory() as s:
        await s.execute(
            update(Tenant)
            .where(Tenant.id == tid)
            .values(
                licence_op_id=uuid7(),
                licence_op_until=datetime.datetime.now(datetime.UTC)
                + datetime.timedelta(minutes=5),
            )
        )
        await s.commit()
    r = await _call(
        _app(db_session_factory, lic, licence=_licence()),
        "PUT",
        f"/api/tenants/{tid}/licence-token",
        {"licence_token": lic.token(tid, uuid7())},
    )
    assert r.status_code == 409 and r.json()["error"]["code"] == "licence_busy", r.text
    tenant = await _tenant(db_session_factory, tid)
    assert tenant is not None and _stored_token(tenant) == token and tenant.licence_ref == licence


async def test_an_expired_lease_is_taken_over(db_session_factory) -> None:
    lic = P9Licences()
    tid, _, _ = await _partner_tenant(db_session_factory, lic)
    async with db_session_factory() as s:
        await s.execute(
            update(Tenant)
            .where(Tenant.id == tid)
            .values(
                licence_op_id=uuid7(),
                licence_op_until=datetime.datetime.now(datetime.UTC)
                - datetime.timedelta(seconds=1),
            )
        )
        await s.commit()
    r = await _call(
        _app(db_session_factory, lic, licence=_licence()),
        "PUT",
        f"/api/tenants/{tid}/licence-token",
        {"licence_token": lic.token(tid, uuid7())},
    )
    assert r.status_code == 200, r.text


async def test_cas_loss_is_409_changed_concurrently(db_session_factory, monkeypatch) -> None:
    """Another writer changes the stored blob between the lease and the swap: the
    CAS on the old ciphertext must lose, and the interloper's token must stay."""
    lic = P9Licences()
    tid, _, _ = await _partner_tenant(db_session_factory, lic)
    real_swap = repo_mod.TenantLicenceRepository.swap_token
    interloper = b"v1|lt1|" + bytes(40)

    async def racing_swap(self, tenant_id, **kw):  # type: ignore[no-untyped-def]
        async with db_session_factory() as other:
            await other.execute(
                update(Tenant)
                .where(Tenant.id == tenant_id)
                .values(licence_token_ciphertext=interloper)
            )
            await other.commit()
        return await real_swap(self, tenant_id, **kw)

    monkeypatch.setattr(repo_mod.TenantLicenceRepository, "swap_token", racing_swap)
    r = await _call(
        _app(db_session_factory, lic, licence=_licence()),
        "PUT",
        f"/api/tenants/{tid}/licence-token",
        {"licence_token": lic.token(tid, uuid7())},
    )
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "licence_changed_concurrently"
    tenant = await _tenant(db_session_factory, tid)
    assert tenant is not None and bytes(tenant.licence_token_ciphertext or b"") == interloper
    assert tenant.licence_op_id is None, "the lease is released after a lost CAS"
    actions = [a for a, _ in await _actions(db_session_factory, tid)]
    assert actions.count("tenant.licence_token_set") == 1, "only the create's: none for a lost swap"


async def test_rotation_is_committed_before_the_response_is_built(db_session_factory) -> None:
    """Driven through the handler directly (the test client's teardown commit
    would hide a missing commit — CLAUDE.md)."""
    from types import SimpleNamespace

    from starlette.requests import Request

    from src.api.schemas.tenancy_keys import TenantLicenceTokenSet
    from src.database.repositories.admin_audit_repository import AuditRepository

    lic = P9Licences()
    tid, _, _ = await _partner_tenant(db_session_factory, lic)
    token = lic.token(tid, uuid7())
    request = cast("Request", SimpleNamespace(state=SimpleNamespace(license=_licence(), roles=[])))
    async with db_session_factory() as session:
        out = await partner.set_tenant_licence_token(
            tenant_id=tid,
            body=TenantLicenceTokenSet.model_validate({"licence_token": token}),
            request=request,
            licences=cast("Any", lic),
            keyring=TEST_KEYRING,
            audit=AuditRepository(session),
            session=session,
        )
        assert out.changed is True
        async with db_session_factory() as probe:
            row = (await probe.execute(select(Tenant).where(Tenant.id == tid))).scalar_one()
            assert _stored_token(row) == token, "rotation not durable at response time"


async def test_no_response_model_can_carry_the_token(db_session_factory) -> None:
    """Scan every serialized tenant / admin response for the raw token."""
    lic = P9Licences()
    tid, _, token = await _partner_tenant(db_session_factory, lic)
    admin = _app(db_session_factory, lic, roles=["super_admin"])
    listed = await _call(admin, "GET", "/api/admin/tenants")
    assert listed.status_code == 200, listed.text
    assert str(tid) in listed.text and token not in listed.text
    for field in ("licence_token", "ciphertext", "licence_token_kid"):
        assert field not in listed.text
    from src.api.schemas import tenancy_keys

    for model in (
        tenancy_keys.PartnerTenantResult,
        tenancy_keys.TenantLicenceTokenResult,
        tenancy_keys.TenantPartnerBinding,
    ):
        assert not any(
            "token" in f and "set_at" not in f and "expires" not in f for f in model.model_fields
        ), model
    body = tenancy_keys.TenantLicenceTokenSet.model_validate({"licence_token": token})
    assert token not in repr(body) and token not in str(body.model_dump())
