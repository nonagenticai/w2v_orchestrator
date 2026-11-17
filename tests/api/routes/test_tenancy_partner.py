"""P2 WP4b — partner tenant creation, ``POST /api/tenants``.

The licence lane is seeded the way w2v_common's licence-token lane leaves it
(``request.state.license`` = the introspection dict); Keycloak is an
identity-only fake that RAISES on any tenancy write. Tenant / invite / audit
rows hit the real test DB.

Non-vacuity: the allowlist + activity + app checks each have their own refusal
test against an otherwise-valid licence, and the happy path asserts on rows a
missing route could never produce (``created_via='partner'`` +
``partner_licence_id``).
"""

from __future__ import annotations

import asyncio
import base64
import os
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4, uuid7

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, insert, select, text
from sqlalchemy.exc import ProgrammingError
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes import _common
from src.api.routes.tenancy import partner
from src.api.routes.tenancy.errors import register_tenancy_exception_handlers
from src.clients.identity_inviter import IDENTITY_RATE_LIMITED, IdentityInviteError
from src.config import clear_settings_cache
from src.database.models import (
    AdminAuditLog,
    Project,
    Tenant,
    TenantApiKey,
    TenantInvite,
    TenantMember,
)
from src.database.repositories.members_repository import MembersRepository
from src.database.tenant_lock import lock_tenant_id
from src.services.tenant_provisioning import tenant_has_occupants
from tests.api.routes._p9_licences import P9Licences
from tests.conftest import require_createrole

pytestmark = pytest.mark.asyncio

_PARTNER = UUID("33333333-3333-4333-8333-333333333333")
_OTHER_PARTNER = UUID("44444444-4444-4444-8444-444444444444")


class _FakeInviter:
    """provisioning's identity invite, in memory. Answers nothing about the
    account (the 202 contract); records each request for the assertions."""

    def __init__(self) -> None:
        self.users: dict[str, UUID] = {}
        self.calls: list[dict[str, Any]] = []

    async def invite(
        self,
        email: str,
        *,
        redirect_uri: str | None = None,
        actor: UUID | None = None,
        context: str | None = None,
    ) -> None:
        self.users.setdefault(email, uuid4())
        self.calls.append({"email": email, "actor": actor, "context": context})


def _licence(licence_id: UUID | None = _PARTNER, **over: Any) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "active": True,
        "license_id": str(licence_id) if licence_id else None,
        "tenant_id": str(uuid4()),  # the PARTNER's own tenant — never the created one
        "subject_id": str(uuid4()),
        "app": {"key": "w2v"},
        "status": "active",
        "expires_at": (datetime.now(UTC) + timedelta(days=200)).isoformat(),
    }
    detail.update(over)
    return detail


@pytest.fixture(autouse=True)
def _allowlist(monkeypatch):
    monkeypatch.setenv("W2V_PARTNER_LICENSE_IDS", f"{_PARTNER},{_OTHER_PARTNER}")
    clear_settings_cache()
    yield
    clear_settings_cache()


def _app(db_session_factory, idp: _FakeInviter, licence: dict[str, Any] | None) -> FastAPI:
    app = FastAPI()
    app.state.session_factory = db_session_factory
    app.include_router(partner.router)
    register_tenancy_exception_handlers(app)

    class _SeedLicenceLane(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.license = licence
            return await call_next(request)

    app.add_middleware(_SeedLicenceLane)
    app.dependency_overrides[_common.get_inviter] = lambda: idp
    # P9: creation refuses an id that holds a live w2v licence; none live here.
    app.dependency_overrides[_common.get_licenses_client] = lambda: P9Licences()
    return app


async def _post(app: FastAPI, body: dict[str, Any]):
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        return await c.post("/api/tenants", json=body)


def _body(tenant_id: UUID | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"name": "Acme Partner Co", "admin_email": f"boss-{uuid4()}@acme.test"}
    if tenant_id is not None:
        body["tenant_id"] = str(tenant_id)
    return body


@pytest.mark.parametrize(
    "licence",
    [pytest.param(None, id="no-licence-lane")],
)
async def test_non_partner_caller_is_403_and_creates_nothing(db_session_factory, licence) -> None:
    idp = _FakeInviter()
    tid = uuid7()
    r = await _post(_app(db_session_factory, idp, licence), _body(tid))
    assert r.status_code == 403, r.text
    async with db_session_factory() as s:
        assert (await s.execute(select(Tenant).where(Tenant.id == tid))).first() is None
    assert idp.users == {}


async def test_partner_creates_tenant_and_first_admin_invite(db_session_factory) -> None:
    idp = _FakeInviter()
    tid = uuid7()
    body = _body(tid)
    r = await _post(_app(db_session_factory, idp, _licence()), body)
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["tenant_id"] == str(tid) and out["created"] is True
    assert [c["email"] for c in idp.calls] == [body["admin_email"]], "one identity invite"
    assert idp.calls[0]["context"] == f"w2v:tenant:{tid}"
    async with db_session_factory() as s:
        tenant = (await s.execute(select(Tenant).where(Tenant.id == tid))).scalar_one()
        invite = (
            await s.execute(select(TenantInvite).where(TenantInvite.tenant_id == tid))
        ).scalar_one()
        actions = set(
            (
                await s.execute(
                    select(AdminAuditLog.action).where(AdminAuditLog.target_tenant_id == tid)
                )
            ).scalars()
        )
    assert tenant.created_via == "partner"
    assert tenant.partner_licence_id == _PARTNER
    assert tenant.name == "Acme Partner Co"
    assert invite.role == "tenant_admin" and invite.status == "pending"
    assert invite.email == body["admin_email"]
    assert str(invite.id) == out["invite_id"]
    assert actions == {"tenant.create", "member.invite"}


async def test_partner_replay_is_idempotent(db_session_factory) -> None:
    idp = _FakeInviter()
    tid = uuid7()
    body = _body(tid)
    app = _app(db_session_factory, idp, _licence())
    first = await _post(app, body)
    second = await _post(app, body)
    assert first.status_code == 201, first.text
    assert second.status_code == 200, second.text
    assert second.json()["created"] is False
    assert second.json()["invite_id"] == first.json()["invite_id"]
    for resp in (first, second):
        assert "admin_user_id" not in resp.json(), "no Keycloak user id in the response"
        assert not any("user" in k for k in resp.json()), resp.json()
    async with db_session_factory() as s:
        n_invites = (
            await s.execute(
                select(func.count()).select_from(TenantInvite).where(TenantInvite.tenant_id == tid)
            )
        ).scalar_one()
        n_creates = (
            await s.execute(
                select(func.count())
                .select_from(AdminAuditLog)
                .where(
                    AdminAuditLog.target_tenant_id == tid, AdminAuditLog.action == "tenant.create"
                )
            )
        ).scalar_one()
    assert n_invites == 1 and n_creates == 1
    # The replay reuses the pending invite: provisioning is asked once.
    assert [c["email"] for c in idp.calls] == [body["admin_email"]]


async def test_another_partners_tenant_id_is_409(db_session_factory) -> None:
    tid = uuid7()
    owner = await _post(_app(db_session_factory, _FakeInviter(), _licence()), _body(tid))
    assert owner.status_code == 201, owner.text
    intruder_idp = _FakeInviter()
    r = await _post(_app(db_session_factory, intruder_idp, _licence(_OTHER_PARTNER)), _body(tid))
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "TENANT_ID_UNAVAILABLE"
    assert intruder_idp.users == {}, "no identity created for a refused tenant"
    async with db_session_factory() as s:
        n = (
            await s.execute(
                select(func.count()).select_from(TenantInvite).where(TenantInvite.tenant_id == tid)
            )
        ).scalar_one()
    assert n == 1


# ─── takeover guard: a partner may only claim an EMPTY tenant id ─────────────


async def _project(factory, tenant_id: UUID) -> None:
    async with factory() as s:
        s.add(
            Project(
                id=uuid7(),
                tenant_id=tenant_id,
                owner_subject_id=uuid7(),
                visibility="private",
                type="promo",
                title="licensed tenant data",
            )
        )
        await s.commit()


async def _member(factory, tenant_id: UUID) -> None:
    async with factory() as s:
        s.add(TenantMember(subject_id=uuid7(), tenant_id=tenant_id, role="tenant_admin"))
        await s.commit()


async def _registry(factory, tenant_id: UUID, created_via: str, licence: UUID | None = None):
    async with factory() as s:
        await s.execute(
            insert(Tenant).values(
                id=tenant_id, name=None, created_via=created_via, partner_licence_id=licence
            )
        )
        await s.commit()


async def _tenant_row(factory, tenant_id: UUID) -> Tenant | None:
    async with factory() as s:
        return (await s.execute(select(Tenant).where(Tenant.id == tenant_id))).scalar_one_or_none()


async def _invite_count(factory, tenant_id: UUID) -> int:
    async with factory() as s:
        return (
            await s.execute(
                select(func.count())
                .select_from(TenantInvite)
                .where(TenantInvite.tenant_id == tenant_id)
            )
        ).scalar_one()


@pytest.mark.parametrize("occupant", ["project"])
async def test_unregistered_tenant_with_data_cannot_be_claimed(
    db_session_factory, occupant: str
) -> None:
    """A licensed tenant with no ``tenants`` row (licence-bundle path) that owns
    a project or a member must NOT be claimable — the partner would invite its
    own tenant_admin into someone else's data."""
    tid = uuid7()
    await (_project if occupant == "project" else _member)(db_session_factory, tid)
    idp = _FakeInviter()
    r = await _post(_app(db_session_factory, idp, _licence()), _body(tid))
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "TENANT_ID_UNAVAILABLE"
    assert await _tenant_row(db_session_factory, tid) is None, "no registry row minted"
    assert await _invite_count(db_session_factory, tid) == 0
    assert idp.users == {}, "no identity created for a refused tenant"


async def test_empty_backfill_row_is_adopted_then_replayable(db_session_factory) -> None:
    tid = uuid7()
    await _registry(db_session_factory, tid, "backfill")
    app = _app(db_session_factory, _FakeInviter(), _licence())
    first = await _post(app, _body(tid))
    assert first.status_code == 201, first.text
    row = await _tenant_row(db_session_factory, tid)
    assert row is not None
    assert row.partner_licence_id == _PARTNER
    assert row.created_via == "backfill"
    assert row.name == "Acme Partner Co"
    again = await _post(app, _body(tid))
    assert again.status_code == 200, again.text
    # Adopted — another partner can no longer take it.
    other = await _post(
        _app(db_session_factory, _FakeInviter(), _licence(_OTHER_PARTNER)), _body(tid)
    )
    assert other.status_code == 409, other.text


@pytest.mark.parametrize("occupant", ["project"])
async def test_backfill_row_with_data_is_not_adoptable(db_session_factory, occupant: str) -> None:
    tid = uuid7()
    await _registry(db_session_factory, tid, "backfill")
    await (_project if occupant == "project" else _member)(db_session_factory, tid)
    r = await _post(_app(db_session_factory, _FakeInviter(), _licence()), _body(tid))
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "TENANT_ID_UNAVAILABLE"
    row = await _tenant_row(db_session_factory, tid)
    assert row is not None and row.partner_licence_id is None
    assert await _invite_count(db_session_factory, tid) == 0


async def _invite(factory, tenant_id: UUID) -> None:
    async with factory() as s:
        s.add(
            TenantInvite(
                id=uuid7(),
                tenant_id=tenant_id,
                email=f"someone-{uuid4()}@licensed.test",
                role="member",
                token_hash=os.urandom(32),
                expires_at=datetime.now(UTC) + timedelta(days=7),
                # Settled, not pending: a revoked invite still means the id is in use.
                status="revoked",
            )
        )
        await s.commit()


async def _api_key(factory, tenant_id: UUID) -> None:
    async with factory() as s:
        s.add(
            TenantApiKey(
                id=uuid7(),
                tenant_id=tenant_id,
                name="ci key",
                prefix=base64.b32encode(os.urandom(5)).decode().lower(),
                key_hash=os.urandom(32),
            )
        )
        await s.commit()


@pytest.mark.parametrize("occupant", ["invite"])
async def test_backfill_row_with_invite_or_api_key_is_not_adoptable(
    db_session_factory, occupant: str
) -> None:
    """An invite (any status) or an API key means the tenant is in use even with
    no project and no member yet — a partner must not adopt it. (Both FK to
    ``tenants``, so only the adopt lane can meet them.)"""
    tid = uuid7()
    await _registry(db_session_factory, tid, "backfill")
    await (_invite if occupant == "invite" else _api_key)(db_session_factory, tid)
    idp = _FakeInviter()
    r = await _post(_app(db_session_factory, idp, _licence()), _body(tid))
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "TENANT_ID_UNAVAILABLE"
    row = await _tenant_row(db_session_factory, tid)
    assert row is not None and row.partner_licence_id is None
    assert idp.users == {}, "no identity created for a refused tenant"


async def test_occupancy_resolver_pins_search_path_with_pg_temp_last(db_session_factory) -> None:
    """A SECURITY DEFINER function must not resolve unqualified names through the
    caller's temp schema: ``search_path`` is pinned, ``pg_temp`` LAST."""
    async with db_session_factory() as s:
        schema = (await s.execute(text("SELECT current_schema()"))).scalar_one()
        row = (
            await s.execute(
                text(
                    "SELECT p.prosecdef, p.proconfig FROM pg_proc p "
                    "WHERE p.proname = 'tenant_is_occupied' "
                    "AND p.pronamespace = to_regnamespace(current_schema())"
                )
            )
        ).one()
    assert row.prosecdef is True
    assert row.proconfig == [f"search_path=pg_catalog, {schema}, pg_temp"]


async def _wait_for_blocked_advisory_lock(factory) -> bool:
    async with factory() as probe:
        for _ in range(100):
            waiting = (
                await probe.execute(
                    text(
                        "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND NOT granted"
                    )
                )
            ).scalar_one()
            await probe.rollback()
            if waiting:
                return True
            await asyncio.sleep(0.05)
    return False


async def test_record_seen_insert_waits_for_the_partner_tenant_lock(db_session_factory) -> None:
    """A claim-mode first login (member INSERT) must queue behind a partner
    creation holding the tenant lock — otherwise it can land between the
    partner's occupancy check and its tenant insert."""
    tid, subject = uuid7(), uuid7()
    async with db_session_factory() as partner_tx:
        await lock_tenant_id(partner_tx, tid)

        async def _login() -> None:
            async with db_session_factory() as s:
                await MembersRepository(s).record_seen(
                    subject_id=subject, tenant_id=tid, email=None, display_name=None, roles=[]
                )
                await s.commit()

        login = asyncio.create_task(_login())
        try:
            assert await _wait_for_blocked_advisory_lock(db_session_factory), (
                "the member insert did not wait on the tenant lock"
            )
            assert not login.done()
        finally:
            await partner_tx.commit()  # releases the xact lock
        await asyncio.wait_for(login, timeout=10)
    async with db_session_factory() as s:
        assert await MembersRepository(s).get_member(subject, tid) is not None


async def test_record_seen_refresh_does_not_take_the_tenant_lock(db_session_factory) -> None:
    """The last_seen refresh of an EXISTING member occupies nothing new — it must
    not queue behind a partner creation (keeps the /me hot path lock-free)."""
    tid, subject = uuid7(), uuid7()
    async with db_session_factory() as s:
        await MembersRepository(s).record_seen(
            subject_id=subject, tenant_id=tid, email="a@x.test", display_name=None, roles=[]
        )
        await s.commit()
    async with db_session_factory() as partner_tx:
        await lock_tenant_id(partner_tx, tid)
        try:
            async with db_session_factory() as s:
                await asyncio.wait_for(
                    MembersRepository(s).record_seen(
                        subject_id=subject,
                        tenant_id=tid,
                        email=None,
                        display_name="Ann",
                        roles=["member"],
                    ),
                    timeout=5,
                )
                await s.commit()
        finally:
            await partner_tx.rollback()
    async with db_session_factory() as s:
        row = await MembersRepository(s).get_member(subject, tid)
    assert row is not None
    assert (row.email, row.display_name, row.roles) == ("a@x.test", "Ann", ["member"])


async def test_operator_created_tenant_is_not_claimable(db_session_factory) -> None:
    tid = uuid7()
    await _registry(db_session_factory, tid, "super_admin")
    r = await _post(_app(db_session_factory, _FakeInviter(), _licence()), _body(tid))
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "TENANT_ID_UNAVAILABLE"


async def test_occupancy_check_sees_projects_across_rls(db_session_factory) -> None:
    """The partner's session is RLS-bound to the PARTNER's tenant (or none). As a
    non-owner NOBYPASSRLS role — the runtime ``w2v_orch_app`` posture under
    enforce — a plain projects count is blind to the target tenant; the
    occupancy check must still see its project (SECURITY DEFINER resolver)."""
    role = f"occ_probe_{uuid7().hex[:12]}"
    async with db_session_factory() as owner:
        schema = (await owner.execute(text("SELECT current_schema()"))).scalar_one()
        try:
            await owner.execute(text(f'CREATE ROLE "{role}" NOLOGIN NOSUPERUSER NOBYPASSRLS'))
        except ProgrammingError as exc:
            await owner.rollback()
            require_createrole(exc)
        await owner.execute(text(f'GRANT "{role}" TO CURRENT_USER'))
        await owner.execute(text(f'GRANT USAGE ON SCHEMA "{schema}" TO "{role}"'))
        await owner.execute(text(f'GRANT SELECT ON ALL TABLES IN SCHEMA "{schema}" TO "{role}"'))
        await owner.execute(text(f'GRANT EXECUTE ON FUNCTION tenant_is_occupied(uuid) TO "{role}"'))
        await owner.commit()
    try:
        target, partners_own = uuid7(), uuid7()
        await _project(db_session_factory, target)
        async with db_session_factory() as s:
            await s.execute(text(f'SET LOCAL ROLE "{role}"'))
            await s.execute(
                text("SELECT set_config('app.current_tenant', :t, true)"), {"t": str(partners_own)}
            )
            blind = (
                await s.execute(
                    select(func.count()).select_from(Project).where(Project.tenant_id == target)
                )
            ).scalar_one()
            assert blind == 0, "precondition: RLS hides the target's projects from this role"
            assert await tenant_has_occupants(s, target) is True
            assert await tenant_has_occupants(s, uuid7()) is False
            await s.rollback()
    finally:
        async with db_session_factory() as owner:
            await owner.execute(text(f'DROP OWNED BY "{role}"'))
            await owner.execute(text(f'DROP ROLE IF EXISTS "{role}"'))
            await owner.commit()


async def test_identity_failure_rolls_the_tenant_back_and_answers_its_code(
    db_session_factory,
) -> None:
    class _Refused(_FakeInviter):
        async def invite(self, email: str, **kw: Any) -> None:
            raise IdentityInviteError(
                IDENTITY_RATE_LIMITED, "slow down", upstream_status=429, retry_after=5
            )

    tid = uuid7()
    r = await _post(_app(db_session_factory, _Refused(), _licence()), _body(tid))
    assert r.status_code == 429, r.text
    assert r.json()["error"]["code"] == "IDENTITY_RATE_LIMITED"
    assert r.headers["retry-after"] == "5"
    async with db_session_factory() as s:
        assert (await s.execute(select(Tenant).where(Tenant.id == tid))).first() is None
        n = (
            await s.execute(
                select(func.count()).select_from(TenantInvite).where(TenantInvite.tenant_id == tid)
            )
        ).scalar_one()
    assert n == 0
