"""Super-admin provisioning (Phase C-2, reworked in P2 tenancy WP4b).

POST /api/admin/provision orchestrates: tenants row (``create_tenant``,
``created_via='super_admin'``) → identity find-or-create through provisioning's
identity invite API (the ONLY identity surface w2v has — no attributes, roles or
passwords can even be expressed) → pending first-user invite → license
mint+token → tenant ``licence_ref`` → provisioning_audit (pending→success/partial).
The inviter + licenses are dependency-overridden fakes; tenant / invite / audit
writes hit the real test DB.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.routes import _common
from src.api.routes.admin import router as admin_router
from src.api.routes.tenancy.errors import register_tenancy_exception_handlers
from src.clients.identity_inviter import (
    IDENTITY_PROVIDER_FORBIDDEN,
    IdentityInviteError,
)
from src.clients.licenses_client import (
    LicencesAccessRefusedError,
)
from src.database.models import (
    AdminAuditLog,
    ProvisioningAudit,
    Tenant,
    TenantInvite,
    TenantMember,
)
from src.services.providers.crypto import Keyring
from src.services.tenant_licence import open_licence_token
from tests._tenant_licences_fake import TenantLicencesFake
from tests.api.routes._p9_licences import TEST_KEYRING, override_licence_keyring

pytestmark = pytest.mark.asyncio

_NEW_USER_ID = "11111111-1111-1111-1111-111111111111"


class _FakeInviter:
    """provisioning's identity invite (``IdentityInviter.invite`` → nothing
    about the account). There is NO Keycloak reader: the provision resolves the
    invitee from w2v's own ``tenant_members`` (G2)."""

    def __init__(self, *, error: IdentityInviteError | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def invite(
        self,
        email: str,
        *,
        redirect_uri: str | None = None,
        actor: UUID | None = None,
        context: str | None = None,
    ) -> None:
        self.calls.append(
            {"email": email, "redirect_uri": redirect_uri, "actor": actor, "context": context}
        )
        if self.error is not None:
            raise self.error


class _FakeLicenses(TenantLicencesFake):
    """licences, modelling WHO holds each licence (tests/_tenant_licences_fake.py).
    ``mint_error`` fails the bundle ensure; ``pre_held`` makes the tenant-level
    licences exist already (a retried provision)."""

    def __init__(
        self, *, pre_held: bool = False, member_holder: str | None = None, **kw: Any
    ) -> None:
        super().__init__(**kw)
        self.pre_held = pre_held
        #: A PERSON already holds the tenant's w2v licence (a legacy tenant).
        self.member_holder = member_holder
        self.mint_error: Exception | None = None

    async def ensure_tenant_bundle(self, tenant_id, bundle_key, *, plans):  # type: ignore[no-untyped-def]
        if self.mint_error is not None:
            self.ensure_error = self.mint_error
        if self.member_holder is not None and not self.ensure_calls:
            await self.create_license(
                subject_id=self.member_holder, app_id="app-w2v", plan_id="p", tenant_id=tenant_id
            )
        if self.pre_held and not self.ensure_calls:
            await super().ensure_tenant_bundle(tenant_id, bundle_key, plans=plans)
            self.ensure_calls.clear()
        return await super().ensure_tenant_bundle(tenant_id, bundle_key, plans=plans)


def _build_app(
    db_session_factory,
    *,
    roles: list[str],
    idp: _FakeInviter,
    lic: _FakeLicenses,
    keyring: Keyring = TEST_KEYRING,
) -> FastAPI:
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
    app.dependency_overrides[_common.get_inviter] = lambda: idp
    app.dependency_overrides[_common.get_licenses_client] = lambda: lic
    override_licence_keyring(app, keyring)
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    )


_BODY = {
    "email": "newuser@example.com",
    "name": "New User",
    "tenant_slug": "acme-corp",
    "workspace_role": "tenant_admin",
}


async def test_provision_happy_path(db_session, db_session_factory) -> None:
    idp, lic = _FakeInviter(), _FakeLicenses()
    app = _build_app(db_session_factory, roles=["super_admin"], idp=idp, lic=lic)
    async with _client(app) as c:
        r = await c.post("/api/admin/provision", json=_BODY)
    assert r.status_code == 201, r.text
    body = r.json()
    # The invitee's Keycloak id never leaves the server (provisioning_audit keeps it).
    assert "user_id" not in body
    assert "already_existed" not in body and "magic_link_sent" not in body
    assert "temporary_password" not in body
    [lic_id] = body["license_ids"].values()
    assert body["license_ids"] == {"w2v": lic_id}
    assert body["tokens"] == {"w2v": f"TOKEN-{lic_id}-1"}

    # ONE identity invite for the email; the default redirect (no per-invite link).
    [call] = idp.calls
    assert call["email"] == "newuser@example.com"
    assert call["redirect_uri"] is None
    assert call["actor"] is not None
    assert call["context"] == f"w2v:tenant:{body['tenant_id']}"
    # The licence is TENANT-level: ensured through the w2v bundle on the new
    # tenant and held by its service subject — never by the invited person.
    assert [c["tenant_id"] for c in lic.ensure_calls] == [body["tenant_id"]]
    assert lic.holder_of(lic_id) == f"svc:{body['tenant_id']}:w2v"
    assert all(h.startswith("svc:") for (_t, _a, h) in lic.holders.values())

    tenant_uuid = UUID(body["tenant_id"])
    # Tenant registry row created through create_tenant, licence recorded.
    tenant = (await db_session.execute(select(Tenant).where(Tenant.id == tenant_uuid))).scalar_one()
    assert tenant.name == "acme-corp"
    assert tenant.created_via == "super_admin"
    assert tenant.licence_ref == UUID(lic_id)
    # …and a DEDICATED token (not the one handed out) is stored sealed. Both are
    # create_token — regenerate would revoke the other.
    assert lic.token_calls == [lic_id, lic_id]
    assert tenant.licence_token_ciphertext is not None
    assert (
        open_licence_token(TEST_KEYRING, tenant_uuid, bytes(tenant.licence_token_ciphertext))
        == f"TOKEN-{lic_id}-2"
    )
    assert tenant.licence_token_expires_at is not None
    assert f"TOKEN-{lic_id}-2" not in r.text
    # The first user is INVITED, not pre-seeded as a member.
    invite = (
        await db_session.execute(select(TenantInvite).where(TenantInvite.tenant_id == tenant_uuid))
    ).scalar_one()
    assert str(invite.id) == body["invite_id"]
    assert invite.status == "pending"
    assert invite.role == "tenant_admin"
    assert invite.email == "newuser@example.com"
    assert invite.kc_user_created is False  # account state is never known to w2v
    members = (
        await db_session.execute(select(TenantMember).where(TenantMember.tenant_id == tenant_uuid))
    ).all()
    assert members == []
    actions = set(
        (
            await db_session.execute(
                select(AdminAuditLog.action).where(AdminAuditLog.target_tenant_id == tenant_uuid)
            )
        ).scalars()
    )
    assert actions == {"tenant.create", "member.invite"}
    # Provisioning audit row = success.
    audit = (
        await db_session.execute(
            select(ProvisioningAudit).where(ProvisioningAudit.tenant_id == tenant_uuid)
        )
    ).scalar_one()
    assert audit.status == "success"
    assert audit.target_email == "newuser@example.com"
    assert audit.result["license_ids"] == {"w2v": lic_id}
    # A person w2v has never seen has no subject yet: it binds when they log in
    # with the verified email and accept the invite (no Keycloak read-back).
    assert audit.target_subject_id is None
    assert audit.result["user_id"] is None


async def test_provision_reuses_the_tenant_level_licence_no_double_mint(
    db_session, db_session_factory
) -> None:
    """Retry after a partial provision must NOT double-mint: licences answers
    ``exists`` for the tenant-level licence and it is reused."""
    idp, lic = _FakeInviter(), _FakeLicenses(pre_held=True)
    app = _build_app(db_session_factory, roles=["super_admin"], idp=idp, lic=lic)
    async with _client(app) as c:
        r = await c.post("/api/admin/provision", json=_BODY)
    assert r.status_code == 201, r.text
    body = r.json()
    assert len(lic.holders) == 2  # w2v + w2v-image, held before — nothing new
    [lic_id] = body["license_ids"].values()
    assert lic.holder_of(lic_id) == f"svc:{body['tenant_id']}:w2v"


async def test_provision_never_falls_back_to_a_person_held_licence(
    db_session, db_session_factory
) -> None:
    """licences refuses the tenant-level licence (``member_licence``): the
    provision fails 409 LICENCE_REFUSED — it never mints for the invitee instead."""
    idp, lic = _FakeInviter(), _FakeLicenses(member_holder=_NEW_USER_ID)
    app = _build_app(db_session_factory, roles=["super_admin"], idp=idp, lic=lic)
    async with _client(app) as c:
        r = await c.post("/api/admin/provision", json=_BODY)
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "LICENCE_REFUSED"
    assert "member_licence" in r.json()["error"]["message"]
    assert lic.token_calls == []
    audit = (
        await db_session.execute(
            select(ProvisioningAudit).where(ProvisioningAudit.target_email == _BODY["email"])
        )
    ).scalar_one()
    assert audit.status == "partial"


async def _seed_member(db_session, *, email: str, subject: UUID | None = None) -> UUID:
    """An identity that already signed in to ANOTHER w2v tenant under ``email``."""
    subject = subject or uuid4()
    db_session.add(TenantMember(subject_id=subject, tenant_id=uuid4(), email=email, role="member"))
    await db_session.commit()
    return subject


async def _audit_for(db_session, email: str) -> ProvisioningAudit:
    return (
        await db_session.execute(
            select(ProvisioningAudit).where(ProvisioningAudit.target_email == email)
        )
    ).scalar_one()


async def test_provision_records_a_known_invitee_from_w2v_members(
    db_session, db_session_factory
) -> None:
    """G2: the invitee's subject comes from w2v's OWN ``tenant_members`` (any
    tenant, case-insensitive email) — never from a Keycloak email search."""
    email = f"known-{uuid4().hex[:8]}@example.com"
    known = await _seed_member(db_session, email=email.upper())
    idp, lic = _FakeInviter(), _FakeLicenses()
    app = _build_app(db_session_factory, roles=["super_admin"], idp=idp, lic=lic)
    async with _client(app) as c:
        r = await c.post("/api/admin/provision", json={**_BODY, "email": email})
    assert r.status_code == 201, r.text
    assert str(known) not in r.text  # the subject id never leaves the server
    audit = await _audit_for(db_session, email)
    assert audit.status == "success"
    assert audit.target_subject_id == known
    assert audit.result is not None
    assert audit.result["user_id"] == str(known)
    # Still only INVITED into the new tenant — not pre-seeded as a member there.
    tenant_uuid = UUID(r.json()["tenant_id"])
    rows = (
        await db_session.execute(select(TenantMember).where(TenantMember.tenant_id == tenant_uuid))
    ).all()
    assert rows == []


async def test_provision_does_not_guess_between_two_subjects_on_one_email(
    db_session, db_session_factory
) -> None:
    """Two identities on one address (an account was recreated): not guessed
    between — the subject binds at invite acceptance instead."""
    email = f"twice-{uuid4().hex[:8]}@example.com"
    await _seed_member(db_session, email=email)
    await _seed_member(db_session, email=email)
    app = _build_app(
        db_session_factory, roles=["super_admin"], idp=_FakeInviter(), lic=_FakeLicenses()
    )
    async with _client(app) as c:
        r = await c.post("/api/admin/provision", json={**_BODY, "email": email})
    assert r.status_code == 201, r.text
    assert (await _audit_for(db_session, email)).target_subject_id is None


async def test_provision_partial_on_missing_app(db_session, db_session_factory) -> None:
    """Requested app key not in w2v's licences bundle → 404 + a `partial` audit row."""
    idp = _FakeInviter()
    lic = _FakeLicenses(bundle_apps=("not-w2v",))
    app = _build_app(db_session_factory, roles=["super_admin"], idp=idp, lic=lic)
    async with _client(app) as c:
        r = await c.post("/api/admin/provision", json=_BODY)
    assert r.status_code == 404, r.text
    # The KC user was created before the failure — audit must show partial.
    audit = (
        await db_session.execute(
            select(ProvisioningAudit).where(ProvisioningAudit.target_email == "newuser@example.com")
        )
    ).scalar_one()
    assert audit.status == "partial"
    assert audit.error
    assert audit.target_subject_id is None  # nobody w2v knows yet


class _MaskingProv:
    """Wraps the real ``ProvisioningRepository`` but makes ``finish(status=
    "partial")`` throw — simulating a DB write failure while recording the
    provisioning audit trail for a real failure (here, the missing-app 404).
    """

    def __init__(self, real) -> None:
        self._real = real

    async def start(self, **kw):
        return await self._real.start(**kw)

    async def finish(self, *a, **kw):
        if kw.get("status") == "partial":
            raise RuntimeError("db write failed while flipping audit to partial")
        return await self._real.finish(*a, **kw)


async def test_provision_finish_failure_does_not_mask_the_real_error(
    db_session_factory,
) -> None:
    """A ``prov.finish`` write failure inside the route's ``except Exception``
    handler (admin.py) must not replace the real provisioning exception (here,
    the 404 for an unknown licenses app) that triggered it.

    FastAPI's own ExceptionMiddleware catches a re-raised ``HTTPException``
    and turns it into the matching HTTP response — it never reaches the ASGI
    transport as a raised exception. A masking ``RuntimeError`` has no
    registered handler, so it WOULD escape as an unhandled 500 (verified by a
    throwaway probe before this test was written). So the observable proof
    that the real exception type propagated, not the masking one, is that the
    client still sees the original 404 rather than an unhandled-exception 500.

    Mutation check: reverting admin.py's inner try/except around that
    ``prov.finish`` call back to an unguarded call turns this red — the
    response becomes an ASGI 500 raised from the injected ``RuntimeError``
    instead of the original 404 ``HTTPException``.
    """
    from src.api.routes._common import get_provisioning_repo
    from src.database.repositories.provisioning_repository import ProvisioningRepository

    idp = _FakeInviter()
    lic = _FakeLicenses(bundle_apps=("not-w2v",))
    app = _build_app(db_session_factory, roles=["super_admin"], idp=idp, lic=lic)
    real_repo = ProvisioningRepository(db_session_factory)
    app.dependency_overrides[get_provisioning_repo] = lambda: _MaskingProv(real_repo)

    async with _client(app) as c:
        r = await c.post("/api/admin/provision", json=_BODY)
    assert r.status_code == 404, r.text
    assert "not found" in r.text


async def test_provision_forbidden_for_tenant_admin(db_session_factory) -> None:
    app = _build_app(
        db_session_factory, roles=["tenant_admin"], idp=_FakeInviter(), lic=_FakeLicenses()
    )
    async with _client(app) as c:
        r = await c.post("/api/admin/provision", json=_BODY)
    assert r.status_code == 403


async def test_provision_rejects_bad_slug(db_session_factory) -> None:
    app = _build_app(
        db_session_factory, roles=["super_admin"], idp=_FakeInviter(), lic=_FakeLicenses()
    )
    async with _client(app) as c:
        r = await c.post("/api/admin/provision", json={**_BODY, "tenant_slug": "Bad Slug!"})
    assert r.status_code == 422


async def test_provision_without_a_keyring_refuses_before_any_side_effect(
    db_session_factory,
) -> None:
    idp, lic = _FakeInviter(), _FakeLicenses()
    app = _build_app(
        db_session_factory, roles=["super_admin"], idp=idp, lic=lic, keyring=Keyring([])
    )
    async with _client(app) as c:
        r = await c.post("/api/admin/provision", json=_BODY)
    assert r.status_code == 503, r.text
    assert r.json()["error"]["code"] == "PROVIDER_KEYRING_UNAVAILABLE"
    assert idp.calls == [] and lic.ensure_calls == [] and lic.holders == {}


# ─── licences migration 050: tenants owned per app ──────────────────────


async def test_provision_registers_the_new_tenant_in_licences_first(
    db_session_factory,
) -> None:
    idp, lic = _FakeInviter(), _FakeLicenses()
    app = _build_app(db_session_factory, roles=["super_admin"], idp=idp, lic=lic)
    async with _client(app) as c:
        r = await c.post("/api/admin/provision", json=_BODY)
    assert r.status_code == 201, r.text
    # Registered as w2v's own under the slug, on the SAME id the licence is ensured on.
    assert lic.registered == [(r.json()["tenant_id"], "acme-corp")]
    assert [c["tenant_id"] for c in lic.ensure_calls] == [r.json()["tenant_id"]]


@pytest.mark.parametrize(
    ("error", "status_code", "code"),
    [
        (
            LicencesAccessRefusedError(
                "ensure_tenant_bundle returned 403",
                status_code=403,
                code="app_not_issuable_by_caller",
            ),
            403,
            "LICENCE_APP_NOT_ISSUABLE",
        )
    ],
)
async def test_provision_licences_ownership_refusals_are_clear(
    db_session, db_session_factory, error: Exception, status_code: int, code: str
) -> None:
    idp, lic = _FakeInviter(), _FakeLicenses()
    lic.mint_error = error
    app = _build_app(db_session_factory, roles=["super_admin"], idp=idp, lic=lic)
    async with _client(app) as c:
        r = await c.post("/api/admin/provision", json=_BODY)
    assert r.status_code == status_code, r.text
    assert r.json()["error"]["code"] == code
    # The audit row still records the attempt as partial (unchanged contract).
    audit = (
        await db_session.execute(
            select(ProvisioningAudit).where(ProvisioningAudit.target_email == _BODY["email"])
        )
    ).scalar_one()
    assert audit.status == "partial"


@pytest.mark.parametrize(
    ("error", "status", "retry_after"),
    [
        (
            IdentityInviteError(IDENTITY_PROVIDER_FORBIDDEN, "no role", upstream_status=403),
            502,
            None,
        )
    ],
)
async def test_provision_identity_failure_maps_its_code_and_marks_partial(
    db_session, db_session_factory, error: IdentityInviteError, status: int, retry_after: str | None
) -> None:
    lic = _FakeLicenses()
    app = _build_app(
        db_session_factory, roles=["super_admin"], idp=_FakeInviter(error=error), lic=lic
    )
    async with _client(app) as c:
        r = await c.post("/api/admin/provision", json=_BODY)
    assert r.status_code == status, r.text
    assert r.json()["error"]["code"] == error.code
    assert r.headers.get("retry-after") == retry_after
    assert lic.ensure_calls == []
    audit = (
        await db_session.execute(
            select(ProvisioningAudit).where(ProvisioningAudit.target_email == "newuser@example.com")
        )
    ).scalar_one()
    assert audit.status == "partial"
