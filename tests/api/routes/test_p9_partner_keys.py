"""P9 — the partner lane mints machine keys; super_admin binds existing tenants.

* ``POST /api/tenants`` without ``admin_email`` (P9-4): no invite, ``licence_ref``
  recorded from the live w2v licence (P9-6).
* ``POST|DELETE /api/tenants/{tenant_id}/api-keys`` (P9-3): only for a tenant bound
  to the CALLER's partner licence; 404 unknown tenant, 403 another party's tenant;
  scopes closed; secret returned once; idempotent by name (409 with metadata, no
  secret); a partner revokes only keys it minted.
* ``PUT /api/admin/tenants/{tenant_id}/partner`` (P9-5): super_admin only; binds an
  occupied tenant; idempotent for the same partner; refuses another partner.

Licence lane seeded the way w2v_common leaves it (``request.state.license`` = the
introspection dict); Keycloak + licences are fakes; tenant / key / audit rows hit
the real test DB. Non-vacuity: every refusal test asserts NO key row was written,
and the happy paths assert on columns only the new code writes
(``partner_licence_id``, ``created_via=partner``, ``licence_ref``).
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID, uuid4, uuid7

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, insert, select, update
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes import _common
from src.api.routes.admin import router as admin_router
from src.api.routes.tenancy import partner
from src.api.routes.tenancy.errors import register_tenancy_exception_handlers
from src.database.models import AdminAuditLog, Project, Tenant, TenantApiKey, TenantInvite
from src.database.repositories.api_keys_repository import ApiKeysRepository
from tests.api.routes._p9_licences import TEST_KEYRING, P9Licences, override_licence_keyring
from tests.api.routes.test_tenancy_partner import (  # noqa: F401 — autouse fixture
    _OTHER_PARTNER,
    _PARTNER,
    _allowlist,
    _FakeInviter,
    _licence,
)

pytestmark = pytest.mark.asyncio


def _lic(live: dict[UUID, UUID] | None = None, **kw: Any) -> P9Licences:
    return P9Licences({t: [lic] for t, lic in (live or {}).items()}, **kw)


def _partner_app(factory, licence: dict[str, Any] | None, lic: P9Licences | None = None) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = factory
    app.include_router(partner.router)
    register_tenancy_exception_handlers(app)

    class _SeedLicenceLane(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.license = licence
            return await call_next(request)

    app.add_middleware(_SeedLicenceLane)
    app.dependency_overrides[_common.get_inviter] = lambda: _FakeInviter()
    app.dependency_overrides[_common.get_licenses_client] = lambda: lic or _lic()
    override_licence_keyring(app)
    return app


def _admin_app(factory, *, roles: list[str], lic: P9Licences | None = None) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = factory
    app.include_router(admin_router)
    register_tenancy_exception_handlers(app)

    class _SeedIdentity(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = str(uuid4())
            request.state.roles = roles
            request.state.subject_id = str(uuid4())
            return await call_next(request)

    app.add_middleware(_SeedIdentity)
    app.dependency_overrides[_common.get_licenses_client] = lambda: lic or _lic()
    override_licence_keyring(app)
    return app


async def _call(app: FastAPI, method: str, path: str, json: Any = None):
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        return await c.request(method, path, json=json)


async def _registry(factory, *, partner_licence: UUID | None, created_via="partner") -> UUID:
    tid = uuid7()
    async with factory() as s:
        await s.execute(
            insert(Tenant).values(
                id=tid, name="t", created_via=created_via, partner_licence_id=partner_licence
            )
        )
        await s.commit()
    return tid


async def _keys(factory, tid: UUID) -> list[TenantApiKey]:
    async with factory() as s:
        return list(
            (await s.execute(select(TenantApiKey).where(TenantApiKey.tenant_id == tid))).scalars()
        )


async def _audit(factory, tid: UUID) -> list[tuple[str, UUID | None, dict]]:
    async with factory() as s:
        rows = await s.execute(
            select(AdminAuditLog.action, AdminAuditLog.actor_subject_id, AdminAuditLog.detail)
            .where(AdminAuditLog.target_tenant_id == tid)
            .order_by(AdminAuditLog.created_at)
        )
        return [(r.action, r.actor_subject_id, r.detail) for r in rows]


# ─── P9-4: admin_email optional; review blocker: no licensed-tenant takeover ─


async def test_partner_creates_unlicensed_tenant_without_admin_email(db_session_factory) -> None:
    tid = uuid7()
    lic = _lic({uuid7(): uuid7()})
    r = await _call(
        _partner_app(db_session_factory, _licence(), lic),
        "POST",
        "/api/tenants",
        {"tenant_id": str(tid), "name": "merqi customer"},
    )
    assert r.status_code == 201, r.text
    assert r.json()["invite_id"] is None and r.json()["already_member"] is False
    async with db_session_factory() as s:
        tenant = (await s.execute(select(Tenant).where(Tenant.id == tid))).scalar_one()
        invites = (
            await s.execute(
                select(func.count()).select_from(TenantInvite).where(TenantInvite.tenant_id == tid)
            )
        ).scalar_one()
    assert invites == 0, "no admin_email → no invite"
    assert tenant.partner_licence_id == _PARTNER
    # Live 2026-10-15: a partner-created tenant came back with licence_ref=null and
    # was unusable. Creation now ENSURES the tenant's own w2v bundle and records it.
    [ensured] = lic.live[tid]
    assert tenant.licence_ref == ensured, "creation ensures the tenant's own w2v licence"
    assert r.json()["licence_ref"] == str(ensured)
    assert lic.ensure_calls == [str(tid)]
    assert [a for a, _, _ in await _audit(db_session_factory, tid)] == ["tenant.create"]


@pytest.mark.parametrize("registry", [None], ids=["no-row"])
async def test_licensed_but_empty_tenant_cannot_be_claimed(
    db_session_factory, registry: str | None
) -> None:
    """The review blocker: a tenant with a LIVE w2v licence and no w2v data. If a
    partner could claim it, its keys would spend on the victim's licence."""
    if registry is None:
        tid = uuid7()
    else:
        tid = await _registry(db_session_factory, partner_licence=None, created_via=registry)
    victim_licence = uuid7()
    lic = _lic({tid: victim_licence})
    app = _partner_app(db_session_factory, _licence(), lic)
    r = await _call(app, "POST", "/api/tenants", {"tenant_id": str(tid), "name": "squat"})
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "TENANT_ID_UNAVAILABLE"
    async with db_session_factory() as s:
        row = (await s.execute(select(Tenant).where(Tenant.id == tid))).scalar_one_or_none()
    if registry is None:
        assert row is None, "no registry row minted"
    else:
        assert row is not None and row.partner_licence_id is None, "not adopted"
    assert row is None or row.licence_ref is None, "the victim's licence was not adopted"

    mint = await _call(app, "POST", f"/api/tenants/{tid}/api-keys", {"name": "k"})
    assert mint.status_code in (403, 404), mint.text
    assert await _keys(db_session_factory, tid) == []


async def test_partner_create_fails_closed_when_licences_are_down(db_session_factory) -> None:
    tid = uuid7()
    r = await _call(
        _partner_app(db_session_factory, _licence(), _lic(down=True)),
        "POST",
        "/api/tenants",
        {"tenant_id": str(tid), "name": "x"},
    )
    assert r.status_code == 503, r.text
    assert r.json()["error"]["code"] == "LICENCES_UNAVAILABLE"
    assert r.headers.get("retry-after") == "30"
    async with db_session_factory() as s:
        assert (await s.execute(select(Tenant).where(Tenant.id == tid))).first() is None


@pytest.mark.parametrize("status_code", [400])
async def test_partner_create_says_refused_not_retry_on_a_definite_4xx(
    db_session_factory, status_code: int
) -> None:
    """R3 gate: off-cluster, the ingress refused ``/internal/*`` with 400
    ``missing_app_key``. That is a configuration fault — 502 LICENCES_REFUSED with
    no Retry-After, never the misleading 503 "retry"; no upstream body leaks."""
    tid = uuid7()
    r = await _call(
        _partner_app(db_session_factory, _licence(), _lic(refuse=status_code)),
        "POST",
        "/api/tenants",
        {"tenant_id": str(tid), "name": "x"},
    )
    assert r.status_code == 502, r.text
    err = r.json()["error"]
    assert err["code"] == "LICENCES_REFUSED"
    assert err["details"] == {"upstream_status": status_code}
    assert "retry-after" not in r.headers
    assert "missing_app_key" not in r.text and "eyJ" not in r.text
    async with db_session_factory() as s:
        assert (await s.execute(select(Tenant).where(Tenant.id == tid))).first() is None


async def test_own_replay_stays_200_after_the_bundle_licenses_the_tenant(
    db_session_factory,
) -> None:
    """Create ensures the bundle; once the partner holds the id, a re-POST is its
    replay — 200, the SAME licence (licences answers ``exists``), no second one."""
    tid = uuid7()
    body = {"tenant_id": str(tid), "name": "merqi customer"}
    lic = _lic()
    app = _partner_app(db_session_factory, _licence(), lic)
    first = await _call(app, "POST", "/api/tenants", body)
    assert first.status_code == 201, first.text
    again = await _call(app, "POST", "/api/tenants", body)
    assert again.status_code == 200, again.text
    [ensured] = lic.live[tid]
    assert first.json()["licence_ref"] == again.json()["licence_ref"] == str(ensured)
    assert lic.ensure_calls == [str(tid), str(tid)], "the replay re-ensures (idempotent)"
    async with db_session_factory() as s:
        tenant = (await s.execute(select(Tenant).where(Tenant.id == tid))).scalar_one()
    assert tenant.licence_ref == ensured


# ─── P9-3: partner mints machine keys ──────────────────────────────────────


async def test_partner_mints_key_for_own_tenant_secret_once_audited(db_session_factory) -> None:
    tid = await _registry(db_session_factory, partner_licence=_PARTNER)
    licence = _licence()
    subject = UUID(licence["subject_id"])
    app = _partner_app(db_session_factory, licence)

    r = await _call(
        app,
        "POST",
        f"/api/tenants/{tid}/api-keys",
        {"name": "merqi-control", "scopes": ["connections:write", "connections:write"]},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["key"].startswith(f"w2v_sk_{body['prefix']}_")
    assert body["scopes"] == ["connections:write"]
    assert body["created_via"] == "partner"
    assert body["created_by"] == str(subject)

    [row] = await _keys(db_session_factory, tid)
    assert row.partner_licence_id == _PARTNER and row.created_by == subject
    async with db_session_factory() as s:
        assert (await ApiKeysRepository(s).resolve(body["key"])) is not None, "the key works"

    # Idempotent by name: a re-mint is 409 with metadata and NEVER a secret.
    again = await _call(app, "POST", f"/api/tenants/{tid}/api-keys", {"name": "merqi-control"})
    assert again.status_code == 409, again.text
    err = again.json()["error"]
    assert err["code"] == "API_KEY_NAME_EXISTS"
    assert err["details"]["key_id"] == body["id"] and err["details"]["prefix"] == body["prefix"]
    assert err["details"]["created_via"] == "partner"
    assert body["key"] not in again.text and "w2v_sk_" not in again.text
    assert len(await _keys(db_session_factory, tid)) == 1

    [(action, actor, detail)] = await _audit(db_session_factory, tid)
    assert (action, actor) == ("apikey.create", subject)
    assert detail["by"] == "partner" and detail["partner_licence_id"] == str(_PARTNER)
    assert body["key"] not in str(detail)


async def test_unknown_tenant_is_404_and_foreign_tenant_is_403(db_session_factory) -> None:
    other = await _registry(db_session_factory, partner_licence=_OTHER_PARTNER)
    operator = await _registry(db_session_factory, partner_licence=None, created_via="super_admin")
    missing = uuid7()
    app = _partner_app(db_session_factory, _licence())

    r404 = await _call(app, "POST", f"/api/tenants/{missing}/api-keys", {"name": "k"})
    assert r404.status_code == 404, r404.text
    assert r404.json()["error"]["code"] == "TENANT_NOT_FOUND"
    for tid in (other, operator):
        r = await _call(app, "POST", f"/api/tenants/{tid}/api-keys", {"name": "k"})
        assert r.status_code == 403, r.text
        assert r.json()["error"]["code"] == "TENANT_NOT_PARTNER_BOUND"
        assert await _keys(db_session_factory, tid) == []
        assert await _audit(db_session_factory, tid) == []


@pytest.mark.parametrize(
    "licence",
    [pytest.param(None, id="no-licence-lane")],
)
async def test_non_partner_caller_cannot_mint_or_revoke(db_session_factory, licence) -> None:
    tid = await _registry(db_session_factory, partner_licence=_PARTNER)
    app = _partner_app(db_session_factory, licence)
    r = await _call(app, "POST", f"/api/tenants/{tid}/api-keys", {"name": "k"})
    assert r.status_code == 403, r.text
    d = await _call(app, "DELETE", f"/api/tenants/{tid}/api-keys/{uuid7()}")
    assert d.status_code == 403, d.text
    assert await _keys(db_session_factory, tid) == []


@pytest.mark.parametrize(
    "body",
    [pytest.param({"name": "k", "scopes": ["connections:admin"]}, id="unknown-scope")],
)
async def test_scope_and_name_validation(db_session_factory, body) -> None:
    tid = await _registry(db_session_factory, partner_licence=_PARTNER)
    r = await _call(
        _partner_app(db_session_factory, _licence()), "POST", f"/api/tenants/{tid}/api-keys", body
    )
    assert r.status_code == 422, r.text
    assert await _keys(db_session_factory, tid) == []


async def test_concurrent_same_name_mints_land_once(db_session_factory) -> None:
    tid = await _registry(db_session_factory, partner_licence=_PARTNER)
    app = _partner_app(db_session_factory, _licence())
    results = await asyncio.gather(
        *(_call(app, "POST", f"/api/tenants/{tid}/api-keys", {"name": "race"}) for _ in range(4))
    )
    codes = sorted(r.status_code for r in results)
    assert codes == [201, 409, 409, 409], [r.text for r in results]
    assert len(await _keys(db_session_factory, tid)) == 1


async def test_partner_revokes_only_its_own_keys_and_can_rotate(db_session_factory) -> None:
    tid = await _registry(db_session_factory, partner_licence=_PARTNER)
    licence = _licence()
    app = _partner_app(db_session_factory, licence)
    minted = (await _call(app, "POST", f"/api/tenants/{tid}/api-keys", {"name": "m"})).json()

    # A tenant_admin's key in the same tenant is NOT the partner's to revoke.
    async with db_session_factory() as s:
        human, _secret = await ApiKeysRepository(s).create(
            tenant_id=tid, name="laptop", created_by=uuid4()
        )
        human_id = human.id
        await s.commit()
    r = await _call(app, "DELETE", f"/api/tenants/{tid}/api-keys/{human_id}")
    assert r.status_code == 404, r.text

    # Another partner cannot even reach the tenant.
    other = _partner_app(db_session_factory, _licence(_OTHER_PARTNER))
    r = await _call(other, "DELETE", f"/api/tenants/{tid}/api-keys/{minted['id']}")
    assert r.status_code == 403, r.text

    ok = await _call(app, "DELETE", f"/api/tenants/{tid}/api-keys/{minted['id']}")
    assert ok.status_code == 204, ok.text
    again = await _call(app, "DELETE", f"/api/tenants/{tid}/api-keys/{minted['id']}")
    assert again.status_code == 404, again.text
    async with db_session_factory() as s:
        assert await ApiKeysRepository(s).resolve(minted["key"]) is None, "revoked key is dead"

    # Rotation: the name is free again once the old key is revoked.
    rotated = await _call(app, "POST", f"/api/tenants/{tid}/api-keys", {"name": "m"})
    assert rotated.status_code == 201, rotated.text
    assert rotated.json()["id"] != minted["id"]

    keys = {k.id: k for k in await _keys(db_session_factory, tid)}
    assert keys[human_id].revoked_at is None, "the human key survived"
    actions = [(a, actor) for a, actor, _ in await _audit(db_session_factory, tid)]
    subject = UUID(licence["subject_id"])
    assert actions == [
        ("apikey.create", subject),
        ("apikey.revoke", subject),
        ("apikey.create", subject),
    ]


# ─── P9-5: super_admin binds an existing tenant ───────────────────────────


async def _project(factory, tenant_id: UUID) -> None:
    async with factory() as s:
        s.add(
            Project(
                id=uuid7(),
                tenant_id=tenant_id,
                owner_subject_id=uuid7(),
                visibility="private",
                type="promo",
                title="dogfood data",
            )
        )
        await s.commit()


async def test_bind_occupied_tenant_then_partner_mints(db_session_factory) -> None:
    """merqi's dogfood tenant: data, no registry row, the partner create path 409s.
    The operator binds it (with the tenant's verified w2v licence as its billing
    ref); the partner can then mint its keys."""
    tid, live_licence = uuid7(), uuid7()
    await _project(db_session_factory, tid)
    admin = _admin_app(db_session_factory, roles=["super_admin"], lic=_lic({tid: live_licence}))
    r = await _call(
        admin, "PUT", f"/api/admin/tenants/{tid}/partner", {"partner_licence_id": str(_PARTNER)}
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["bound"] is True and out["partner_licence_id"] == str(_PARTNER)
    assert out["licence_ref"] is None, "no licence_id given → nothing picked automatically"
    async with db_session_factory() as s:
        tenant = (await s.execute(select(Tenant).where(Tenant.id == tid))).scalar_one()
    assert tenant.partner_licence_id == _PARTNER and tenant.created_via == "super_admin"
    assert tenant.licence_ref is None
    listed = await _call(admin, "GET", "/api/admin/tenants")
    [item] = [i for i in listed.json()["items"] if i["tenant_id"] == str(tid)]
    assert item["partner_licence_id"] == str(_PARTNER), "operator list shows the binding"

    # Idempotent for the same partner: no second audit row …
    again = await _call(
        admin, "PUT", f"/api/admin/tenants/{tid}/partner", {"partner_licence_id": str(_PARTNER)}
    )
    assert again.status_code == 200 and again.json()["bound"] is False, again.text
    # … and the tenant's own, introspected token is stored and its licence becomes
    # the billing ref (audited as a token set, never the token itself).
    lic = admin.dependency_overrides[_common.get_licenses_client]()
    token = lic.token(tid, live_licence)
    with_ref = await _call(
        admin,
        "PUT",
        f"/api/admin/tenants/{tid}/partner",
        {"partner_licence_id": str(_PARTNER), "licence_token": token},
    )
    assert with_ref.status_code == 200, with_ref.text
    assert with_ref.json()["licence_ref"] == str(live_licence)
    assert token not in with_ref.text
    async with db_session_factory() as s:
        tenant = (await s.execute(select(Tenant).where(Tenant.id == tid))).scalar_one()
    assert tenant.licence_ref == live_licence
    assert tenant.licence_token_ciphertext is not None and token.encode() not in bytes(
        tenant.licence_token_ciphertext
    )
    audit = await _audit(db_session_factory, tid)
    assert [a for a, _, _ in audit] == ["tenant.partner_bind", "tenant.licence_token_set"]
    assert audit[0][2] == {"partner_licence_id": str(_PARTNER), "created_registry_row": True}
    assert audit[1][2]["licence_ref"] == str(live_licence)
    assert audit[1][2]["by"] == "super_admin"
    assert audit[1][1] is not None, "the route records the acting super_admin subject"
    assert token not in str(audit)

    # The retired licence_id field is refused loudly, never silently ignored.
    old = await _call(
        admin,
        "PUT",
        f"/api/admin/tenants/{tid}/partner",
        {"partner_licence_id": str(_PARTNER), "licence_id": str(live_licence)},
    )
    assert old.status_code == 422, old.text

    minted = await _call(
        _partner_app(db_session_factory, _licence()),
        "POST",
        f"/api/tenants/{tid}/api-keys",
        {"name": "merqi-marketing"},
    )
    assert minted.status_code == 201, minted.text


@pytest.mark.parametrize(
    "case",
    ["other-tenants"],
)
async def test_bind_refuses_a_token_that_is_not_the_tenants_active_w2v(
    db_session_factory, case: str
) -> None:
    tid, other, lic_id = uuid7(), uuid7(), uuid7()
    await _project(db_session_factory, tid)
    lic = {"licences-down": _lic(down=True), "licences-refused": _lic(refuse=403)}.get(case, _lic())
    token = {
        "other-tenants": lic.token(other, lic_id),
        "other-app": lic.token(tid, lic_id, app="fr"),
        "inactive": lic.token(tid, lic_id, active=False, reason="license_inactive"),
    }.get(case, "lic_tok_unknown")
    r = await _call(
        _admin_app(db_session_factory, roles=["super_admin"], lic=lic),
        "PUT",
        f"/api/admin/tenants/{tid}/partner",
        {"partner_licence_id": str(_PARTNER), "licence_token": token},
    )
    want = {
        "licences-down": (503, "LICENCES_UNAVAILABLE"),
        "licences-refused": (502, "LICENCES_REFUSED"),
    }.get(case, (422, "LICENCE_TOKEN_MISMATCH"))
    assert (r.status_code, r.json()["error"]["code"]) == want, r.text
    reason = {"other-tenants": "tenant", "other-app": "app", "inactive": "license_inactive"}
    if case in reason:
        assert r.json()["error"]["details"] == {"reason": reason[case]}
    assert token not in r.text
    async with db_session_factory() as s:
        row = (await s.execute(select(Tenant).where(Tenant.id == tid))).scalar_one_or_none()
    assert row is None, "a refused bind writes nothing"


async def test_name_clash_with_someone_elses_key_is_a_bare_409(db_session_factory) -> None:
    """Fix 4: metadata of a key the partner did not mint is never disclosed."""
    tid = await _registry(db_session_factory, partner_licence=_PARTNER)
    async with db_session_factory() as s:
        human, _secret = await ApiKeysRepository(s).create(
            tenant_id=tid, name="merqi-sales", created_by=uuid4()
        )
        human_id, human_prefix = human.id, human.prefix
        await s.commit()
    r = await _call(
        _partner_app(db_session_factory, _licence()),
        "POST",
        f"/api/tenants/{tid}/api-keys",
        {"name": "merqi-sales"},
    )
    assert r.status_code == 409, r.text
    err = r.json()["error"]
    assert err["code"] == "API_KEY_NAME_EXISTS"
    assert not err.get("details"), err
    assert str(human_id) not in r.text and human_prefix not in r.text
    # Another partner's own key in a tenant it no longer manages is equally opaque —
    # covered by the 403 binding gate before any name lookup.


async def test_bind_refuses_a_different_partner(db_session_factory) -> None:
    tid = await _registry(db_session_factory, partner_licence=_OTHER_PARTNER)
    r = await _call(
        _admin_app(db_session_factory, roles=["super_admin"]),
        "PUT",
        f"/api/admin/tenants/{tid}/partner",
        {"partner_licence_id": str(_PARTNER)},
    )
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "TENANT_BOUND_TO_OTHER_PARTNER"
    async with db_session_factory() as s:
        tenant = (await s.execute(select(Tenant).where(Tenant.id == tid))).scalar_one()
    assert tenant.partner_licence_id == _OTHER_PARTNER, "never overwritten"
    assert await _audit(db_session_factory, tid) == []


async def test_bind_refusals(db_session_factory) -> None:
    occupied = uuid7()
    await _project(db_session_factory, occupied)
    admin = _admin_app(db_session_factory, roles=["super_admin"])
    body = {"partner_licence_id": str(_PARTNER)}

    not_partner = await _call(
        admin,
        "PUT",
        f"/api/admin/tenants/{occupied}/partner",
        {"partner_licence_id": str(uuid4())},
    )
    assert not_partner.status_code == 422, not_partner.text
    assert not_partner.json()["error"]["code"] == "NOT_A_PARTNER_LICENCE"

    unknown = await _call(admin, "PUT", f"/api/admin/tenants/{uuid7()}/partner", body)
    assert unknown.status_code == 404, unknown.text

    tenant_admin = _admin_app(db_session_factory, roles=["tenant_admin"])
    forbidden = await _call(tenant_admin, "PUT", f"/api/admin/tenants/{occupied}/partner", body)
    assert forbidden.status_code == 403, forbidden.text

    async with db_session_factory() as s:
        row = (await s.execute(select(Tenant).where(Tenant.id == occupied))).scalar_one_or_none()
    assert row is None, "no refused bind wrote a registry row"


# ─── durability: committed before the response is built (CLAUDE.md) ────────


async def test_mint_and_bind_are_committed_before_the_response_is_built(
    db_session_factory,
) -> None:
    """Driven through the handlers directly: the test client would run the session
    teardown (which commits) before an assertion could look — proving nothing."""
    from types import SimpleNamespace
    from typing import cast

    from starlette.requests import Request

    from src.api.routes.admin import bind_tenant_partner
    from src.api.schemas.tenancy_keys import PartnerApiKeyCreate, TenantPartnerBind
    from src.database.repositories.admin_audit_repository import AuditRepository

    tid = uuid7()
    await _project(db_session_factory, tid)
    async with db_session_factory() as session:
        bound = await bind_tenant_partner(
            tenant_id=tid,
            body=TenantPartnerBind(partner_licence_id=_PARTNER),
            actor=uuid4(),
            _=None,
            audit=AuditRepository(session),
            licences=cast("Any", _lic()),
            keyring=TEST_KEYRING,
            session=session,
        )
        assert bound.bound is True
        async with db_session_factory() as probe:
            row = (await probe.execute(select(Tenant).where(Tenant.id == tid))).scalar_one()
            assert row.partner_licence_id == _PARTNER, "bind not durable at response time"

    request = cast("Request", SimpleNamespace(state=SimpleNamespace(license=_licence())))
    async with db_session_factory() as session:
        created = await partner.partner_create_api_key(
            tenant_id=tid,
            body=PartnerApiKeyCreate(name="durable"),
            request=request,
            audit=AuditRepository(session),
            session=session,
        )
        async with db_session_factory() as probe:
            found = (
                await probe.execute(select(TenantApiKey.id).where(TenantApiKey.id == created.id))
            ).first()
            assert found is not None, "minted key not durable at response time"


# ─── re-review: first-party partners (our own provisioning) ────────────────


@pytest.fixture
def first_party(monkeypatch):
    from src.config import clear_settings_cache

    monkeypatch.setenv("W2V_PARTNER_FIRST_PARTY_LICENSE_IDS", str(_PARTNER))
    clear_settings_cache()
    yield _PARTNER
    clear_settings_cache()


async def test_provisioning_shaped_create_first_party_claims_and_records_licence_ref(
    db_session_factory, first_party
) -> None:
    """Live provisioning mints the tenant's w2v licence FIRST, then creates the w2v
    tenant (and redrives that step). A first-party partner gets 201 and the
    tenant's licence becomes its billing ref; a redrive is a 200 replay that keeps
    (or repairs) the ref."""
    tid, minted = uuid7(), uuid7()
    lic = _lic({tid: minted})
    app = _partner_app(db_session_factory, _licence(first_party), lic)
    body = {"tenant_id": str(tid), "name": "new deal"}
    r = await _call(app, "POST", "/api/tenants", body)
    assert r.status_code == 201, r.text
    async with db_session_factory() as s:
        tenant = (await s.execute(select(Tenant).where(Tenant.id == tid))).scalar_one()
    assert tenant.partner_licence_id == first_party and tenant.licence_ref == minted

    # Redrive after the ref was lost (e.g. created before this fix): repaired.
    async with db_session_factory() as s:
        await s.execute(update(Tenant).where(Tenant.id == tid).values(licence_ref=None))
        await s.commit()
    again = await _call(app, "POST", "/api/tenants", body)
    assert again.status_code == 200, again.text
    async with db_session_factory() as s:
        tenant = (await s.execute(select(Tenant).where(Tenant.id == tid))).scalar_one()
    assert tenant.licence_ref == minted

    # And its keys work end to end: mint is allowed for the claimed tenant.
    key = await _call(app, "POST", f"/api/tenants/{tid}/api-keys", {"name": "merqi-control"})
    assert key.status_code == 201, key.text


async def test_provisioning_shaped_create_third_party_is_still_409(
    db_session_factory, first_party
) -> None:
    tid = uuid7()
    third_party = _partner_app(db_session_factory, _licence(_OTHER_PARTNER), _lic({tid: uuid7()}))
    r = await _call(third_party, "POST", "/api/tenants", {"tenant_id": str(tid), "name": "x"})
    assert r.status_code == 409, r.text
    async with db_session_factory() as s:
        assert (await s.execute(select(Tenant).where(Tenant.id == tid))).first() is None


async def test_first_party_create_of_an_unlicensed_tenant_ensures_its_ref(
    db_session_factory, first_party
) -> None:
    tid = uuid7()
    lic = _lic()
    r = await _call(
        _partner_app(db_session_factory, _licence(first_party), lic),
        "POST",
        "/api/tenants",
        {"tenant_id": str(tid), "name": "x"},
    )
    assert r.status_code == 201, r.text
    async with db_session_factory() as s:
        tenant = (await s.execute(select(Tenant).where(Tenant.id == tid))).scalar_one()
    assert tenant.licence_ref == lic.live[tid][0]


async def test_first_party_ids_must_be_partner_ids(monkeypatch, caplog) -> None:
    from src.config import clear_settings_cache, get_settings

    stray = uuid4()
    monkeypatch.setenv("W2V_PARTNER_FIRST_PARTY_LICENSE_IDS", f"{_PARTNER},{stray}")
    clear_settings_cache()
    try:
        with caplog.at_level("WARNING"):
            settings = get_settings()
        assert settings.w2v_partner_first_party_license_ids == frozenset({_PARTNER})
        assert str(stray) in caplog.text
    finally:
        clear_settings_cache()


# ─── licence ensure on create (live 2026-10-15: licence_ref came back null) ──


async def test_create_with_a_refused_licence_fails_loudly_then_a_replay_recovers(
    db_session_factory,
) -> None:
    """licences refuses the tenant's w2v licence (e.g. a legacy person-held one):
    the create is a typed 409, never a 201 with ``licence_ref: null``. The row it
    committed stays; once the refusal is resolved the partner's replay ensures
    and records the ref."""
    tid = uuid7()
    body = {"tenant_id": str(tid), "name": "merqi customer"}
    lic = _lic()
    lic.ensure_refuse = {"w2v": "member_licence"}
    app = _partner_app(db_session_factory, _licence(), lic)

    refused = await _call(app, "POST", "/api/tenants", body)

    assert refused.status_code == 409, refused.text
    err = refused.json()["error"]
    assert err["code"] == "LICENCE_REFUSED"
    assert err["details"] == {"tenant_id": str(tid), "app": "w2v", "reason": "member_licence"}
    async with db_session_factory() as s:
        tenant = (await s.execute(select(Tenant).where(Tenant.id == tid))).scalar_one()
    assert tenant.licence_ref is None and tenant.partner_licence_id == _PARTNER

    lic.ensure_refuse = {}
    replay = await _call(app, "POST", "/api/tenants", body)

    assert replay.status_code == 200, replay.text
    [ensured] = lic.live[tid]
    assert replay.json()["licence_ref"] == str(ensured)


async def test_create_when_the_licence_ensure_cannot_reach_licences_is_503(
    db_session_factory,
) -> None:
    from src.clients.licenses_client import LicensesClientError

    tid = uuid7()
    lic = _lic()
    lic.ensure_error = LicensesClientError("ensure_tenant_bundle returned 503", status_code=503)
    r = await _call(
        _partner_app(db_session_factory, _licence(), lic),
        "POST",
        "/api/tenants",
        {"tenant_id": str(tid), "name": "x"},
    )
    assert r.status_code == 503, r.text
    assert r.json()["error"]["code"] == "LICENCES_UNAVAILABLE"


async def test_create_whose_ensure_times_out_after_licences_finished_records_the_licence(
    db_session_factory,
) -> None:
    """Live 2026-10-15: licences #275 made the ensure ~16 s; it committed while
    the client raised ReadTimeout, and the create answered 503 for a tenant that
    was in fact licensed. The re-read finds ``svc:<tenant>:w2v``'s licence."""
    import httpx

    tid = uuid7()
    lic = _lic()
    lic.ensure_error_after_commit = httpx.ReadTimeout("read timed out")
    r = await _call(
        _partner_app(db_session_factory, _licence(), lic),
        "POST",
        "/api/tenants",
        {"tenant_id": str(tid), "name": "x"},
    )
    assert r.status_code == 201, r.text
    [ensured] = lic.live[tid]
    assert r.json()["licence_ref"] == str(ensured)
    assert f"svc:{tid}:w2v" in lic.reread_subjects
    async with db_session_factory() as s:
        tenant = (await s.execute(select(Tenant).where(Tenant.id == tid))).scalar_one()
    assert tenant.licence_ref == ensured


async def test_create_whose_ensure_times_out_with_no_licence_written_is_503(
    db_session_factory,
) -> None:
    import httpx

    tid = uuid7()
    lic = _lic()
    lic.ensure_error = httpx.ReadTimeout("read timed out")
    lic.ensure_error_after_commit = lic.ensure_error  # serve the (empty) re-read
    r = await _call(
        _partner_app(db_session_factory, _licence(), lic),
        "POST",
        "/api/tenants",
        {"tenant_id": str(tid), "name": "x"},
    )
    assert r.status_code == 503, r.text
    assert r.json()["error"]["code"] == "LICENCES_UNAVAILABLE"
    assert f"svc:{tid}:w2v" in lic.reread_subjects
    async with db_session_factory() as s:
        tenant = (await s.execute(select(Tenant).where(Tenant.id == tid))).scalar_one()
    assert tenant.licence_ref is None


async def test_create_when_licences_definitely_refuses_the_ensure_is_502(
    db_session_factory,
) -> None:
    from src.clients.licenses_client import LicensesClientError

    lic = _lic()
    lic.ensure_error = LicensesClientError("ensure_tenant_bundle returned 403", status_code=403)
    r = await _call(
        _partner_app(db_session_factory, _licence(), lic),
        "POST",
        "/api/tenants",
        {"tenant_id": str(uuid7()), "name": "x"},
    )
    assert r.status_code == 502, r.text
    assert r.json()["error"]["code"] == "LICENCES_REFUSED"
