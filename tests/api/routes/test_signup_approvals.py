"""Early-access signup (v2): service doors, approve core, steps, tick, Telegram.

Real test postgres for every row; licences / the Keycloak admin API (the
approve's own identity + its one email) are dependency-overridden fakes; service tokens are signed with a local RSA key and
verified by w2v_common's real ``KeycloakJWTValidator`` (its JWKS cache seeded with
that key). Telegram never leaves the process (``set_telegram_post_for_tests``).
"""

from __future__ import annotations

import datetime
import json
import time
from typing import Any, cast
from uuid import UUID, uuid7

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from jwt.algorithms import RSAAlgorithm
from sqlalchemy import func, select, update
from starlette.middleware.base import BaseHTTPMiddleware
from w2v_common.auth.keycloak import KeycloakJWTValidator

from src.api.routes import _common
from src.api.routes.signup_approvals import (
    admin_router,
    door_router,
    register_signup_exception_handlers,
)
from src.api.routes.tenancy.errors import register_tenancy_exception_handlers
from src.clients.keycloak_admin import KeycloakAdminClient, KeycloakAdminError
from src.clients.licenses_client import (
    LicencesTenantNameTakenError,
    LicensesClient,
    RegisteredTenant,
)
from src.clients.schemas.keycloak import KeycloakUser
from src.clients.schemas.licenses import ActiveAppRef, ActiveLicenseView, ActivePlanRef
from src.config import SignupSettings, clear_settings_cache, get_settings
from src.database.models import AdminAuditLog, SignupApproval, Tenant, TenantInvite, TenantMember
from src.database.repositories.invites_repository import InvitesRepository, new_invite_token
from src.database.repositories.members_repository import MembersRepository
from src.services.license_status import clear_license_status_cache
from src.services.signup import core, service_token
from src.services.signup import identity as ident
from src.services.signup import telegram as tg
from src.services.signup.service_token import (
    jwks_realm_for_issuer,
    set_signup_token_validator_for_tests,
)
from tests._tenant_licences_fake import TenantLicencesFake
from tests.api.routes._p9_licences import TEST_KEYRING, override_licence_keyring

pytestmark = pytest.mark.asyncio

ISSUER = "https://auth.example.test/realms/portal"
FRONTEND = "https://w2v.example.test"
KID = "test-kid"
_PRIVATE = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_OTHER_PRIVATE = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _validator(issuer: str = ISSUER) -> KeycloakJWTValidator:
    v = KeycloakJWTValidator(
        keycloak_url="http://keycloak.invalid",
        realm=issuer.rsplit("/realms/", 1)[-1],
        audiences=[],
        issuer=issuer,
    )
    jwk = json.loads(RSAAlgorithm.to_jwk(_PRIVATE.public_key()))
    jwk["kid"] = KID
    v._jwks_by_kid = {KID: jwk}
    v._jwks_fetched_at = time.monotonic()
    return v


def token(
    *,
    azp: str = "w2v-signup-intake",
    aud: list[str] | str | None = None,
    roles: tuple[str, ...] = ("w2v-signup-intake",),
    iss: str = ISSUER,
    exp_in: int = 300,
    key: Any = _PRIVATE,
    drop: tuple[str, ...] = (),
) -> str:
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": iss,
        "sub": "sa-subject",
        "azp": azp,
        "aud": aud if aud is not None else [azp, "account"],
        "realm_access": {"roles": list(roles)},
        "iat": now,
        "exp": now + exp_in,
    }
    for k in drop:
        claims.pop(k, None)
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": KID})


def intake_token(**kw: Any) -> dict[str, str]:
    return {"Authorization": f"Bearer {token(**kw)}"}


def decider_token(**kw: Any) -> dict[str, str]:
    kw.setdefault("azp", "chat-bridge-signup-decider")
    kw.setdefault("roles", ("w2v-signup-decide",))
    return {"Authorization": f"Bearer {token(**kw)}"}


class FakeKeycloak:
    """The Keycloak admin surface the approve uses, in memory. ``calls`` are the
    ``execute_actions_email`` sends; ``error`` is raised by the send."""

    def __init__(self) -> None:
        self.users: dict[str, dict[str, Any]] = {}
        self.created: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.attr_writes: list[tuple[str, dict[str, str | None]]] = []
        self.calls: list[dict[str, Any]] = []
        self.error: BaseException | None = None
        self.drop_attrs: set[str] = set()

    def seed(self, email: str, attributes: dict[str, list[str]] | None = None) -> str:
        uid = str(uuid7())
        self.users[uid] = {"id": uid, "email": email, "attributes": attributes or {}}
        return uid

    async def create_user(self, representation: dict[str, Any]) -> str | None:
        if any(u["email"] == representation["email"] for u in self.users.values()):
            return None
        self.created.append(representation)
        uid = str(uuid7())
        attrs = {
            k: list(v) for k, v in representation["attributes"].items() if k not in self.drop_attrs
        }
        self.users[uid] = {**representation, "id": uid, "attributes": attrs}
        return uid

    async def find_user_by_email(self, email: str) -> KeycloakUser | None:
        for u in self.users.values():
            if u["email"] == email.strip().lower():
                return KeycloakUser.model_validate(u)
        return None

    async def get_user(self, user_id: str) -> KeycloakUser | None:
        u = self.users.get(user_id)
        return KeycloakUser.model_validate(u) if u else None

    async def update_user_attributes(self, user_id: str, attrs: dict[str, str | None]) -> None:
        self.attr_writes.append((user_id, attrs))
        for k, v in attrs.items():
            if v is None:
                self.users[user_id]["attributes"].pop(k, None)
            else:
                self.users[user_id]["attributes"][k] = [v]

    async def execute_actions_email(self, user_id: str, actions: list[str], **kw: Any) -> None:
        self.calls.append(
            {"user_id": user_id, "email": self.users[user_id]["email"], "actions": actions, **kw}
        )
        if self.error is not None:
            raise self.error

    async def count_credentials(self, user_id: str) -> int:
        return len(self.users[user_id].get("credentials") or [])

    async def count_federated_identities(self, user_id: str) -> int:
        return len(self.users[user_id].get("federatedIdentities") or [])

    async def delete_user(self, user_id: str) -> bool:
        self.deleted.append(user_id)
        return self.users.pop(user_id, None) is not None


class FakeLicences(TenantLicencesFake):
    """TenantLicencesFake + the gate's read (``get_active_license``) + name clashes."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.taken_names: set[str] = set()
        self.gate_inactive = False

    async def register_tenant(self, tenant_id, *, name: str):  # type: ignore[no-untyped-def]
        if name in self.taken_names:
            raise LicencesTenantNameTakenError("register_tenant 409", status_code=409)
        self.registered.append((str(tenant_id), name))
        return RegisteredTenant(tenant_id=str(tenant_id), name=name, created=True)

    async def get_active_license(self, *, tenant_id, app_key: str):  # type: ignore[no-untyped-def]
        if self.gate_inactive:
            return None
        holder = f"svc:{tenant_id}:{app_key}"
        lids = [lid for lid, (t, a, h) in self.holders.items() if h == holder and a == app_key]
        if not lids:
            return None
        return ActiveLicenseView(
            active=True,
            tenant_id=str(tenant_id),
            license_id=lids[0],
            app=ActiveAppRef(key=app_key),
            plan=ActivePlanRef(code="standard"),
            candidate_count=1,
        )


def row_actor(row: SignupApproval) -> object:
    """Who the recorded email send was made for (audit-only; Keycloak never sees it)."""
    return row.steps["invite_email"]["detail"]["actor"]


class Sent:
    def __init__(self) -> None:
        self.messages: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, url: str, payload: dict[str, Any]) -> httpx.Response:
        self.messages.append((url, payload))
        return httpx.Response(200, json={"ok": True})


@pytest.fixture
async def signup_env(monkeypatch):
    monkeypatch.setenv("W2V_SIGNUP_ENABLED", "true")
    monkeypatch.setenv("KEYCLOAK_ISSUER", ISSUER)
    monkeypatch.setenv("W2V_FRONTEND_URL", FRONTEND)
    monkeypatch.setenv(
        "W2V_SIGNUP_TELEGRAM_BOT_TOKEN", "123:SECRET-BOT-TOKEN"
    )  # pragma: allowlist secret
    monkeypatch.setenv("W2V_SIGNUP_TELEGRAM_CHAT_ID", "-1003934173351")
    monkeypatch.setenv("W2V_SIGNUP_TELEGRAM_THREAD_ID", "4242")
    clear_settings_cache()
    clear_license_status_cache()
    set_signup_token_validator_for_tests(_validator())
    sent = Sent()
    tg.set_telegram_post_for_tests(sent)
    yield sent
    await tg.drain_telegram_for_tests()
    set_signup_token_validator_for_tests(None)
    tg.set_telegram_post_for_tests(None)
    clear_license_status_cache()


class Harness:
    def __init__(self, app: FastAPI, sf: Any, lic: FakeLicences, idp: FakeKeycloak, sent: Sent):
        self.app, self.sf, self.lic, self.idp, self.sent = app, sf, lic, idp, sent
        self.admin_subject = uuid7()

    def client(self) -> AsyncClient:
        return AsyncClient(
            transport=ASGITransport(app=self.app, raise_app_exceptions=False),
            base_url="http://testserver",
        )

    @property
    def deps(self) -> core.SignupDeps:
        return core.SignupDeps(
            licenses=cast("LicensesClient", self.lic),
            identity=cast("KeycloakAdminClient", self.idp),
            keyring=TEST_KEYRING,
        )

    async def row(self, signup_id: str | UUID) -> SignupApproval:
        got = await core.load(self.sf, UUID(str(signup_id)))
        assert got is not None
        return got

    async def intake(self, email: str = "ann@acme.example", **body: Any) -> httpx.Response:
        payload = {"email": email, "name": "Ann Example", "source": "web", **body}
        async with self.client() as c:
            return await c.post(
                "/api/signup-approvals/intake", json=payload, headers=intake_token()
            )

    async def pending_id(self, email: str = "ann@acme.example") -> UUID:
        got = await core.pending_id_for_email(self.sf, email)
        assert got is not None
        return got


@pytest.fixture
def h(db_session_factory, signup_env) -> Harness:
    lic, idp = FakeLicences(), FakeKeycloak()
    app = FastAPI()
    app.state.session_factory = db_session_factory
    app.include_router(door_router)
    app.include_router(admin_router)
    register_signup_exception_handlers(app)
    register_tenancy_exception_handlers(app)
    harness = Harness(app, db_session_factory, lic, idp, signup_env)

    class _Seed(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.tenant_id = str(uuid7())
            request.state.subject_id = str(harness.admin_subject)
            request.state.roles = list(getattr(app.state, "roles", ["super_admin"]))
            return await call_next(request)

    app.add_middleware(_Seed)
    app.dependency_overrides[_common.get_signup_keycloak_admin] = lambda: idp
    app.dependency_overrides[_common.get_licenses_client] = lambda: lic
    override_licence_keyring(app, TEST_KEYRING)
    return harness


# ─── disabled flag ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("method", "path"),
    [("POST", "/api/signup-approvals/intake")],
)
async def test_every_route_503s_unless_enabled(h: Harness, monkeypatch, method, path) -> None:
    monkeypatch.setenv("W2V_SIGNUP_ENABLED", "false")
    async with h.client() as c:
        r = await c.request(method, path, json={})
    assert r.status_code == 503, r.text
    assert r.json()["error"] == "signup_not_configured"
    # Read per call: flipping the env back opens it without a restart.
    monkeypatch.setenv("W2V_SIGNUP_ENABLED", "true")
    async with h.client() as c:
        r = await c.request(method, path, json={})
    assert r.status_code != 503


# ─── auth matrix ─────────────────────────────────────────────────────────


_INTAKE_BODY = {"email": "ann@acme.example", "name": "Ann", "source": "web"}


# Headers are built INSIDE the test: a token minted at import time would expire
# before a long suite reaches it and turn every 403 case into a 401.
_RAW = "raw"


def _headers(spec: tuple[str, Any]) -> dict[str, str]:
    kind, arg = spec
    if kind == _RAW:
        return dict(arg)
    return intake_token(**arg) if kind == "intake" else decider_token(**arg)


@pytest.mark.parametrize(
    ("spec", "status", "code"),
    [((_RAW, {}), 401, "unauthorized")],
)
async def test_intake_auth_matrix(h: Harness, spec, status, code) -> None:
    async with h.client() as c:
        r = await c.post("/api/signup-approvals/intake", json=_INTAKE_BODY, headers=_headers(spec))
    assert r.status_code == status, r.text
    assert r.json()["error"] == code
    async with h.sf() as s:
        assert (await s.execute(select(SignupApproval))).scalars().all() == []


@pytest.mark.parametrize(
    ("spec", "status", "code"),
    [((_RAW, {}), 401, "unauthorized")],
)
async def test_decider_auth_matrix(h: Harness, spec, status, code) -> None:
    await h.intake()
    sid = await h.pending_id()
    async with h.client() as c:
        r = await c.post(
            f"/api/signup-approvals/{sid}/telegram-decision",
            json={"action": "approve", "telegramUserId": 1},
            headers=_headers(spec),
        )
    assert r.status_code == status, r.text
    assert r.json()["error"] == code
    assert (await h.row(sid)).status == "pending"


async def test_no_issuer_configured_refuses_every_token(h: Harness, monkeypatch) -> None:
    monkeypatch.delenv("KEYCLOAK_ISSUER")
    async with h.client() as c:
        r = await c.post("/api/signup-approvals/intake", json=_INTAKE_BODY, headers=intake_token())
    assert r.status_code == 401


# ─── realm-per-app P0: per-door issuer pins ──────────────────────────────

SECURE_ISSUER = "https://auth.example.test/realms/secure-apps"


async def test_door_issuers_default_to_the_shared_pin(monkeypatch) -> None:
    for name in (
        "W2V_SIGNUP_TOKEN_ISSUER",
        "W2V_SIGNUP_INTAKE_ISSUER",
        "W2V_SIGNUP_DECIDER_ISSUER",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KEYCLOAK_ISSUER", ISSUER + "/")
    cfg = SignupSettings()
    assert cfg.intake_token_issuer == ISSUER
    assert cfg.decider_token_issuer == ISSUER
    monkeypatch.setenv("W2V_SIGNUP_DECIDER_ISSUER", SECURE_ISSUER)
    cfg = SignupSettings()
    assert (cfg.intake_token_issuer, cfg.decider_token_issuer) == (ISSUER, SECURE_ISSUER)


async def test_jwks_realm_comes_from_the_pinned_issuer() -> None:
    """KEYCLOAK_REALM (portal) must not decide which realm's keys verify a secure-apps pin."""
    assert jwks_realm_for_issuer(SECURE_ISSUER, "portal") == "secure-apps"
    assert jwks_realm_for_issuer(ISSUER + "/", "portal") == "portal"
    assert jwks_realm_for_issuer("https://auth.example.test", "portal") == "portal"


async def test_validator_fetches_jwks_from_the_issuer_realm(monkeypatch) -> None:
    set_signup_token_validator_for_tests(None)
    monkeypatch.setenv("KEYCLOAK_URL", "http://keycloak.test.test:8080")
    monkeypatch.setenv("KEYCLOAK_REALM", "portal")
    clear_settings_cache()
    try:
        prov = get_settings().provisioning
        assert prov.keycloak_url and prov.keycloak_realm == "portal"
        v = service_token._validator(SECURE_ISSUER)
        assert isinstance(v, KeycloakJWTValidator)
        assert v.issuer == SECURE_ISSUER
        assert v._jwks_uri == (
            f"{prov.keycloak_url.rstrip('/')}/realms/secure-apps/protocol/openid-connect/certs"
        )
        p = service_token._validator(ISSUER)
        assert isinstance(p, KeycloakJWTValidator)
        assert p._jwks_uri.endswith("/realms/portal/protocol/openid-connect/certs")
    finally:
        service_token._validators.clear()
        clear_settings_cache()


async def test_decider_pinned_to_its_own_issuer(h: Harness, monkeypatch) -> None:
    assert (await h.intake()).status_code == 202
    sid = await h.pending_id()
    monkeypatch.setenv("W2V_SIGNUP_DECIDER_ISSUER", SECURE_ISSUER)
    body = {"action": "approve", "telegramUserId": 1}
    url = f"/api/signup-approvals/{sid}/telegram-decision"
    async with h.client() as c:
        # A portal-issued decider token no longer passes the door's pin.
        r = await c.post(url, json=body, headers=decider_token())
        assert r.status_code == 401, r.text
        # The intake door keeps the shared (portal) pin.
        r = await c.post("/api/signup-approvals/intake", json=_INTAKE_BODY, headers=intake_token())
        assert r.status_code == 202, r.text
    set_signup_token_validator_for_tests(_validator(SECURE_ISSUER))
    async with h.client() as c:
        r = await c.post(url, json=body, headers=decider_token(iss=SECURE_ISSUER))
    assert r.status_code != 401, r.text
    assert r.json().get("error") != "signup_decider_forbidden", r.text


async def test_intake_pinned_to_its_own_issuer(h: Harness, monkeypatch) -> None:
    monkeypatch.setenv("W2V_SIGNUP_INTAKE_ISSUER", SECURE_ISSUER)
    async with h.client() as c:
        r = await c.post("/api/signup-approvals/intake", json=_INTAKE_BODY, headers=intake_token())
        assert r.status_code == 401, r.text
    set_signup_token_validator_for_tests(_validator(SECURE_ISSUER))
    async with h.client() as c:
        r = await c.post(
            "/api/signup-approvals/intake",
            json=_INTAKE_BODY,
            headers=intake_token(iss=SECURE_ISSUER),
        )
    assert r.status_code == 202, r.text


# ─── intake ──────────────────────────────────────────────────────────────


async def test_intake_queues_once_and_is_idempotent(h: Harness) -> None:
    r = await h.intake(company="Acme", message="Hi\nthere", locale="de-DE")
    assert r.status_code == 202 and r.json() == {"status": "queued"}
    r2 = await h.intake(email="ANN@acme.example")  # same address, any case
    assert r2.status_code == 202 and r2.json() == {"status": "already_pending"}
    async with h.sf() as s:
        rows = (await s.execute(select(SignupApproval))).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert (row.status, row.locale, row.company, row.message) == (
        "pending",
        "de",
        "Acme",
        "Hi\nthere",
    )
    assert row.intake_client == "w2v-signup-intake" and row.source == "web"
    await tg.drain_telegram_for_tests()
    assert len(h.sent.messages) == 1, "a de-duplicated re-submit sends nothing"


async def test_intake_strips_control_and_bidi_characters_and_422s_bad_input(h: Harness) -> None:
    r = await h.intake(email="bob@acme.example", name="Bob‮\nSignup id: x")
    assert r.status_code == 202
    row = await h.row(await h.pending_id("bob@acme.example"))
    assert row.name == "Bob Signup id: x"
    async with h.client() as c:
        bad = await c.post(
            "/api/signup-approvals/intake",
            json={"email": "nope", "name": "x", "source": "web"},
            headers=intake_token(),
        )
        extra = await c.post(
            "/api/signup-approvals/intake",
            json={**_INTAKE_BODY, "role": "super_admin"},
            headers=intake_token(),
        )
        long_msg = await c.post(
            "/api/signup-approvals/intake",
            json={**_INTAKE_BODY, "message": "x" * 2001},
            headers=intake_token(),
        )
    assert (bad.status_code, extra.status_code, long_msg.status_code) == (422, 422, 422)


async def test_unknown_locale_is_not_told_never_refused(h: Harness) -> None:
    r = await h.intake(locale="xx-YY")
    assert r.status_code == 202
    assert (await h.row(await h.pending_id())).locale is None


async def test_queue_full_refuses(h: Harness, monkeypatch) -> None:
    monkeypatch.setenv("W2V_SIGNUP_QUEUE_MAX_PENDING", "1")
    assert (await h.intake()).status_code == 202
    r = await h.intake(email="second@acme.example")
    assert r.status_code == 503 and r.json()["error"] == "signup_queue_full"


# ─── Telegram decision (C3) ──────────────────────────────────────────────


async def _decide(h: Harness, sid: object, action: str, **body: Any) -> httpx.Response:
    async with h.client() as c:
        return await c.post(
            f"/api/signup-approvals/{sid}/telegram-decision",
            json={"action": action, "telegramUserId": 7001, **body},
            headers=decider_token(),
        )


async def test_telegram_approve_provisions_seats_and_emails_last(h: Harness) -> None:
    await h.intake(company="Acme Corp", locale="nl")
    sid = await h.pending_id()
    r = await _decide(h, sid, "approve")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "approved"
    assert len(body["summary"]) <= 200 and "\n" not in body["summary"]
    assert "set-password email sent" in body["summary"]

    row = await h.row(sid)
    assert row.status == "approved" and row.decided_via == "telegram"
    assert row.decider_actor == "telegram:7001" and row.decided_by is None
    assert (row.tenant_name, row.tenant_slug) == ("Acme Corp", "acme-corp")
    assert body["tenantId"] == str(row.tenant_id)
    steps = row.steps
    assert [steps[k]["status"] for k in core.STEPS] == ["ok", "ok", "ok", "ok"]
    assert row.steps_next_attempt_at is None

    # identity: w2v minted the realm user itself — credential-less, unverified,
    # named (no "Update account information" form), stamped with this row.
    [created] = h.idp.created
    assert created == {
        "username": "ann@acme.example",
        "email": "ann@acme.example",
        "emailVerified": False,
        "enabled": True,
        "firstName": "Ann",
        "lastName": "Example",
        "attributes": {
            "locale": ["nl"],
            "signup_app": ["w2v"],
            "w2v_signup_id": [str(sid)],
        },
    }
    assert "credentials" not in created
    user_id = steps["identity"]["detail"]["user_id"]
    assert steps["identity"]["detail"]["minted"] is True

    # provision: the SHARED core — registered under the slug, tenant row, tenant-level
    # licences held by svc:<tenant>:<app>, LLM posture local_allowed, sealed token.
    tid = str(row.tenant_id)
    assert h.lic.registered == [(tid, "acme-corp")]
    assert all(holder.startswith(f"svc:{tid}:") for (_t, _a, holder) in h.lic.holders.values())
    assert h.lic.byo_overrides[(tid, "w2v")] == "local_allowed"
    assert steps["provision"]["detail"]["licence_state"] in ("active", "expiring_soon")
    async with h.sf() as s:
        tenant = (await s.execute(select(Tenant).where(Tenant.id == row.tenant_id))).scalar_one()
        invite = (
            await s.execute(select(TenantInvite).where(TenantInvite.tenant_id == row.tenant_id))
        ).scalar_one()
    assert tenant.name == "acme-corp" and tenant.created_via == "super_admin"
    assert tenant.licence_token_ciphertext is not None and tenant.licence_ref is not None
    # membership: SEATED at approval (CommonPractice) — the tenant_admin invite is
    # created AND accepted for the minted Keycloak user (= the JWT sub they sign in
    # with), leaving exactly what an accepted invite leaves: an active member row.
    assert (invite.role, invite.status, invite.email) == ("tenant_admin", "accepted", row.email)
    assert invite.accepted_subject_id == UUID(user_id)
    async with h.sf() as s:
        member = (
            await s.execute(select(TenantMember).where(TenantMember.tenant_id == row.tenant_id))
        ).scalar_one()
    assert (member.subject_id, member.role, member.status) == (
        UUID(user_id),
        "tenant_admin",
        "active",
    )
    assert (member.email, member.display_name) == (row.email, "Ann Example")
    assert steps["membership"]["detail"]["subject_id"] == user_id
    # the Telegram approve takes the deployment default LLM access, recorded on the row
    assert row.llm_access == "local_allowed"

    # invite_email: ONE Keycloak execute-actions-email to the minted user, for the
    # public login client, 14 days, landing on the APP ROOT (no token: they are
    # seated already). Attributes were set at create: nothing re-written.
    [call] = h.idp.calls
    assert call["user_id"] == user_id and call["email"] == "ann@acme.example"
    assert call["actions"] == ["UPDATE_PASSWORD", "VERIFY_EMAIL"]
    assert call["client_id"] == "w2v-frontend"
    assert call["lifespan_seconds"] == 14 * 24 * 3600
    assert h.idp.attr_writes == []
    assert steps["invite_email"]["detail"]["attributes"] is True
    assert call["redirect_uri"] == f"{FRONTEND}/"
    assert steps["invite_email"]["detail"]["redirect_uri"] == f"{FRONTEND}/"


async def test_telegram_decision_is_idempotent_and_404s(h: Harness) -> None:
    await h.intake()
    sid = await h.pending_id()
    assert (await _decide(h, sid, "approve")).json()["status"] == "approved"
    again = await _decide(h, sid, "reject")
    assert again.status_code == 200
    assert again.json()["status"] == "already_decided" and again.json()["outcome"] == "approved"
    assert again.json()["tenantId"] == str((await h.row(sid)).tenant_id)
    assert len(h.idp.calls) == 1, "a redelivered tap never sends a second email"
    assert (await _decide(h, uuid7(), "approve")).status_code == 404
    assert (await _decide(h, "not-a-uuid", "approve")).status_code == 404


async def test_telegram_reject_marks_only(h: Harness) -> None:
    await h.intake()
    sid = await h.pending_id()
    r = await _decide(h, sid, "reject")
    assert r.status_code == 200 and r.json()["status"] == "rejected"
    row = await h.row(sid)
    assert (row.status, row.tenant_id, row.decider_actor) == ("rejected", None, "telegram:7001")
    assert h.lic.registered == [] and h.idp.calls == []
    assert (await _decide(h, sid, "approve")).json()["status"] == "already_decided"
    # A decided row no longer blocks a new request from the same address.
    assert (await h.intake()).json() == {"status": "queued"}


async def test_telegram_decision_body_is_strict(h: Harness) -> None:
    await h.intake()
    sid = await h.pending_id()
    for body in (
        {"action": "approve", "telegramUserId": "7001"},
        {"action": "maybe", "telegramUserId": 7001},
        {"action": "approve", "telegramUserId": 7001, "extra": 1},
        {"action": "approve"},
    ):
        async with h.client() as c:
            r = await c.post(
                f"/api/signup-approvals/{sid}/telegram-decision", json=body, headers=decider_token()
            )
        assert r.status_code == 422, body
    assert (await h.row(sid)).status == "pending"


async def test_concurrent_claim_answers_in_progress(h: Harness) -> None:
    await h.intake()
    sid = await h.pending_id()
    async with h.sf() as s:
        await s.execute(
            update(SignupApproval)
            .where(SignupApproval.id == sid)
            .values(
                approving_until=datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=5),
                approving_claim=uuid7(),
            )
        )
        await s.commit()
    r = await _decide(h, sid, "approve")
    # RETRYABLE for chat-bridge (non-200, not 404 → it keeps the buttons): the
    # concurrent approve may still fail and leave the row pending.
    assert r.status_code == 409 and r.json()["status"] == "in_progress"
    assert r.json()["summary"] == "Another approval is in progress - try again in a minute."
    r2 = await _decide(h, sid, "reject")
    assert r2.status_code == 409 and r2.json()["status"] == "in_progress"
    assert (await h.row(sid)).status == "pending"


# ─── step ordering / failure / resumability ──────────────────────────────


async def test_email_never_sent_before_provision_ok_and_resumes(h: Harness) -> None:
    await h.intake()
    sid = await h.pending_id()
    h.lic.ensure_error = RuntimeError("licences down")
    r = await _decide(h, sid, "approve")
    # The decision stands; the failure is recorded and retried by the tick.
    assert r.status_code == 200 and r.json()["status"] == "approved"
    assert "provision failed" in r.json()["summary"]
    row = await h.row(sid)
    assert row.status == "approved"
    assert row.steps["provision"]["status"] == "failed"
    assert row.steps["membership"]["status"] == "waiting"
    assert row.steps["invite_email"]["status"] == "waiting"
    assert h.idp.calls == [], "no email while provisioning is not ok"
    assert row.steps_next_attempt_at is not None

    # Resume: the SAME tenant id, nothing duplicated.
    h.lic.ensure_error = None
    tenant_id = row.tenant_id
    run = await core.run_steps(h.sf, h.deps, sid, actor=core.SYSTEM_ACTOR)
    assert run.order == ["provision", "membership", "invite_email"]
    row = await h.row(sid)
    assert row.tenant_id == tenant_id
    assert [row.steps[k]["status"] for k in core.STEPS] == ["ok", "ok", "ok", "ok"]
    assert len(h.idp.created) == 1, "the identity is minted once"
    assert len(h.idp.calls) == 1
    # Once ok, nothing re-runs.
    again = await core.run_steps(h.sf, h.deps, sid, actor=core.SYSTEM_ACTOR)
    assert again.order == [] and len(h.idp.calls) == 1


async def test_gate_check_fails_the_provision_step(h: Harness) -> None:
    """Issue exactly what the gate checks: a licence the gate does not read as
    entitled fails provision, so the email never goes."""
    h.lic.gate_inactive = True
    await h.intake()
    sid = await h.pending_id()
    await _decide(h, sid, "approve")
    row = await h.row(sid)
    assert row.steps["provision"]["status"] == "failed"
    assert row.steps["provision"]["code"] == "licence_not_entitled"
    assert h.idp.calls == []


async def test_gate_key_is_the_request_gates_key() -> None:
    from src.services.license_provisioning import W2V_LICENCE_APP_KEY

    assert W2V_LICENCE_APP_KEY == _common.W2V_APP_KEY == "w2v"
    assert "w2v" in core._app_keys()


async def test_email_timeout_is_unknown_and_never_auto_resent(h: Harness) -> None:
    await h.intake()
    sid = await h.pending_id()
    h.idp.error = KeycloakAdminError("timed out", uncertain=True)
    await _decide(h, sid, "approve")
    row = await h.row(sid)
    assert row.steps["invite_email"]["status"] == "unknown"
    assert row.steps_next_attempt_at is None, "an uncertain send parks the row for the operator"
    h.idp.error = None
    tick = await core.run_steps(h.sf, h.deps, sid, actor=core.SYSTEM_ACTOR)
    assert tick.order == [] and len(h.idp.calls) == 1
    # Only the super_admin Retry re-sends it.
    async with h.client() as c:
        r = await c.post(f"/api/admin/signup-approvals/{sid}/retry")
    assert r.status_code == 200, r.text
    assert r.json()["approval"]["email_state"] == "sent"
    assert len(h.idp.calls) == 2


async def test_definite_email_failure_is_failed_and_retried(h: Harness) -> None:
    await h.intake()
    sid = await h.pending_id()
    h.idp.error = KeycloakAdminError("execute_actions_email returned 500", status_code=500)
    await _decide(h, sid, "approve")
    row = await h.row(sid)
    assert row.steps["invite_email"]["status"] == "failed"
    assert row.steps["invite_email"]["code"] == "invite_email_failed"
    h.idp.error = None
    async with h.sf() as s:
        await s.execute(
            update(SignupApproval)
            .where(SignupApproval.id == sid)
            .values(steps_next_attempt_at=func.now())
        )
        await s.commit()
    assert await core.run_tick(h.sf, h.deps) == 1
    assert (await h.row(sid)).steps["invite_email"]["status"] == "ok"


async def test_licences_name_clash_moves_to_the_next_slug(h: Harness) -> None:
    h.lic.taken_names = {"acme"}
    await h.intake(company="Acme")
    sid = await h.pending_id()
    await _decide(h, sid, "approve")
    row = await h.row(sid)
    assert row.tenant_slug == "acme-2"
    assert row.steps["provision"]["status"] == "ok"


async def test_two_signups_for_one_company_get_distinct_slugs(h: Harness) -> None:
    await h.intake(company="Acme")
    await h.intake(email="bob@acme.example", company="Acme")
    a, b = await h.pending_id(), await h.pending_id("bob@acme.example")
    await _decide(h, a, "approve")
    await _decide(h, b, "approve")
    assert ((await h.row(a)).tenant_slug, (await h.row(b)).tenant_slug) == ("acme", "acme-2")


# ─── tick: lease + backoff ───────────────────────────────────────────────


async def test_backoff_is_5min_doubling_capped_6h() -> None:
    got = [core.steps_backoff_seconds(n) for n in range(1, 11)]
    assert got[:4] == [300, 600, 1200, 2400]
    assert max(got) == 6 * 3600 and got[-1] == 6 * 3600


async def test_tick_backs_off_and_gives_up_after_ten(h: Harness) -> None:
    h.lic.ensure_error = RuntimeError("licences down")
    await h.intake()
    sid = await h.pending_id()
    await _decide(h, sid, "approve")
    for attempt in range(1, 11):
        async with h.sf() as s:
            await s.execute(
                update(SignupApproval)
                .where(SignupApproval.id == sid)
                .values(steps_next_attempt_at=func.now())
            )
            await s.commit()
        before = datetime.datetime.now(datetime.UTC)
        assert await core.run_tick(h.sf, h.deps) == 1
        row = await h.row(sid)
        assert row.steps_attempts == attempt
        if attempt < 10:
            assert row.steps_next_attempt_at is not None
            delay = (row.steps_next_attempt_at - before).total_seconds()
            assert abs(delay - core.steps_backoff_seconds(attempt)) < 60
        else:
            assert row.steps_next_attempt_at is None, "10 attempts, then the operator's Retry"
    assert await core.run_tick(h.sf, h.deps) == 0


async def test_tick_skips_a_row_whose_lease_is_held(h: Harness) -> None:
    h.lic.ensure_error = RuntimeError("licences down")
    await h.intake()
    sid = await h.pending_id()
    await _decide(h, sid, "approve")
    h.lic.ensure_error = None
    future = datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=5)
    async with h.sf() as s:
        await s.execute(
            update(SignupApproval)
            .where(SignupApproval.id == sid)
            .values(
                steps_next_attempt_at=func.now(),
                steps_lease_until=future,
                steps_lease_token=uuid7(),
            )
        )
        await s.commit()
    assert await core.run_tick(h.sf, h.deps) == 0, "another replica holds the lease"
    assert (await h.row(sid)).steps["provision"]["status"] == "failed"
    async with h.sf() as s:
        await s.execute(
            update(SignupApproval)
            .where(SignupApproval.id == sid)
            .values(steps_lease_until=func.now())
        )
        await s.commit()
    assert await core.run_tick(h.sf, h.deps) == 1
    row = await h.row(sid)
    assert row.steps["invite_email"]["status"] == "ok" and row.steps_lease_token is None


async def test_tick_is_inert_unless_enabled(h: Harness, monkeypatch) -> None:
    from src.lifecycle.signup_tick import signup_tick_once

    monkeypatch.setenv("W2V_SIGNUP_ENABLED", "false")
    assert await signup_tick_once(h.sf) == 0


# ─── super_admin routes ──────────────────────────────────────────────────


async def test_admin_routes_need_super_admin(h: Harness) -> None:
    h.app.state.roles = ["tenant_admin"]
    async with h.client() as c:
        assert (await c.get("/api/admin/signup-approvals")).status_code == 403
        r = await c.post(
            "/api/admin/signup-approvals/invite", json={"email": "a@b.example", "name": "A"}
        )
        assert r.status_code == 403


async def test_admin_list_get_approve_reject(h: Harness) -> None:
    await h.intake()
    await h.intake(email="bob@other.example")
    a, b = await h.pending_id(), await h.pending_id("bob@other.example")
    async with h.client() as c:
        listed = (await c.get("/api/admin/signup-approvals", params={"status": "pending"})).json()
        assert {i["id"] for i in listed["items"]} == {str(a), str(b)}
        assert (
            await c.get("/api/admin/signup-approvals", params={"status": "x"})
        ).status_code == 422
        got = await c.get(f"/api/admin/signup-approvals/{a}")
        assert got.json()["data"]["suggested_tenant_name"] == "Ann Example"
        appr = await c.post(
            f"/api/admin/signup-approvals/{a}/approve", json={"tenantName": "Ann Studio"}
        )
        assert appr.status_code == 200, appr.text
        assert appr.json()["tenant"]["slug"] == "ann-studio" and appr.json()["emailState"] == "sent"
        rej = await c.post(f"/api/admin/signup-approvals/{b}/reject")
        assert rej.status_code == 200 and rej.json()["approval"]["status"] == "rejected"
        again = await c.post(f"/api/admin/signup-approvals/{b}/approve")
        assert again.status_code == 400 and again.json()["error"] == "approval_not_pending"
        assert (await c.get(f"/api/admin/signup-approvals/{uuid7()}")).status_code == 404
    row = await h.row(a)
    assert row.decided_via == "onboarding" and row.decided_by == str(h.admin_subject)
    # Admin decisions post a short outcome to Telegram.
    await tg.drain_telegram_for_tests()
    texts = [p["text"] for _u, p in h.sent.messages]
    assert any(t.startswith("Studio signup approved") for t in texts)
    assert any(t.startswith("Studio signup rejected") for t in texts)


async def test_invite_creates_an_approved_row(h: Harness) -> None:
    async with h.client() as c:
        r = await c.post(
            "/api/admin/signup-approvals/invite",
            json={"email": "zoe@zeta.example", "name": "Zoe", "company": "Zeta", "locale": "fr"},
        )
    assert r.status_code == 200, r.text
    assert r.json()["reusedPending"] is False
    row = await h.row(r.json()["approval"]["id"])
    assert (row.source, row.status, row.locale, row.tenant_slug) == (
        "invite",
        "approved",
        "fr",
        "zeta",
    )
    assert h.idp.created[0]["attributes"]["locale"] == ["fr"]
    assert (h.idp.created[0]["firstName"], h.idp.created[0]["lastName"]) == ("Zoe", "Zoe")
    assert len(h.idp.calls) == 1


async def test_invite_approves_the_existing_pending_row(h: Harness) -> None:
    await h.intake(locale="de")
    sid = await h.pending_id()
    async with h.client() as c:
        r = await c.post(
            "/api/admin/signup-approvals/invite",
            json={"email": "Ann@Acme.example", "name": "Ann", "locale": "pl"},
        )
    assert r.status_code == 200, r.text
    assert r.json()["reusedPending"] is True and r.json()["approval"]["id"] == str(sid)
    row = await h.row(sid)
    assert (row.source, row.status, row.locale) == ("web", "approved", "pl")
    async with h.sf() as s:
        assert len((await s.execute(select(SignupApproval))).scalars().all()) == 1
    async with h.client() as c:
        bad = await c.post(
            "/api/admin/signup-approvals/invite",
            json={"email": "x@y.example", "name": "X", "locale": "xx"},
        )
    assert bad.status_code == 422


async def test_retry_only_for_approved_rows(h: Harness) -> None:
    await h.intake()
    sid = await h.pending_id()
    async with h.client() as c:
        r = await c.post(f"/api/admin/signup-approvals/{sid}/retry")
    assert r.status_code == 400 and r.json()["error"] == "approval_not_approved"


async def test_needs_attention_filter(h: Harness) -> None:
    h.lic.ensure_error = RuntimeError("down")
    await h.intake()
    sid = await h.pending_id()
    await _decide(h, sid, "approve")
    async with h.client() as c:
        items = (
            await c.get("/api/admin/signup-approvals", params={"status": "needs_attention"})
        ).json()["items"]
    assert [i["id"] for i in items] == [str(sid)]
    assert items[0]["steps_not_ok"][0]["step"] == "provision"


# ─── review fixes: lease at decision, in_flight email, repost, one workspace ──


class Killed(BaseException):
    """A pod kill mid-call: not an ``Exception``, so nothing records an outcome."""


async def _invite_and_audit(h: Harness, sid: UUID) -> tuple[TenantInvite, AdminAuditLog]:
    row = await h.row(sid)
    async with h.sf() as s:
        invite = (
            await s.execute(select(TenantInvite).where(TenantInvite.tenant_id == row.tenant_id))
        ).scalar_one()
        audit = (
            await s.execute(
                select(AdminAuditLog).where(
                    AdminAuditLog.action == "member.invite",
                    AdminAuditLog.target_tenant_id == row.tenant_id,
                )
            )
        ).scalar_one()
    return invite, audit


async def test_the_approver_holds_the_steps_lease_from_the_decision(h: Harness, monkeypatch):
    """A tick on another replica, right between the decision and the approver's
    own steps, can neither find the row due nor take its lease: the steps run as
    the approver (invite + audit carry the super_admin), and the reply is final."""
    await h.intake()
    sid = await h.pending_id()
    real_release = core._release_claim
    seen: dict[str, Any] = {}

    async def release_then_race(sf, signup_id, token):  # type: ignore[no-untyped-def]
        await real_release(sf, signup_id, token)
        row = await h.row(signup_id)
        seen["status"] = row.status
        assert row.steps_next_attempt_at is not None
        seen["due_in"] = (
            row.steps_next_attempt_at - datetime.datetime.now(datetime.UTC)
        ).total_seconds()
        seen["lease"] = row.steps_lease_token
        # Even a FORCED-due tick cannot take the lease the decision handed over.
        async with h.sf() as s:
            await s.execute(
                update(SignupApproval)
                .where(SignupApproval.id == signup_id)
                .values(steps_next_attempt_at=func.now())
            )
            await s.commit()
        seen["tick"] = await core.run_tick(h.sf, h.deps)

    monkeypatch.setattr(core, "_release_claim", release_then_race)
    async with h.client() as c:
        r = await c.post(f"/api/admin/signup-approvals/{sid}/approve")
    assert r.status_code == 200, r.text
    assert seen["status"] == "approved" and seen["lease"] is not None
    assert seen["due_in"] > core.STEPS_LEASE_SECONDS - 60, "first tick a lease-length out"
    assert seen["tick"] == 0, "no tick runs the steps before the approver"
    assert r.json()["emailState"] == "sent" and len(h.idp.calls) == 1
    assert row_actor(await h.row(sid)) == str(h.admin_subject)
    invite, audit = await _invite_and_audit(h, sid)
    assert invite.invited_by == h.admin_subject
    assert audit.actor_subject_id == h.admin_subject
    assert audit.detail["actor"] == str(h.admin_subject)
    row = await h.row(sid)
    assert row.steps_lease_token is None and row.steps_next_attempt_at is None


async def test_telegram_approver_is_recorded_on_the_invite_audit(h: Harness) -> None:
    await h.intake()
    sid = await h.pending_id()
    assert (await _decide(h, sid, "approve")).json()["status"] == "approved"
    _invite, audit = await _invite_and_audit(h, sid)
    assert audit.detail["actor"] == "telegram:7001"
    assert audit.detail["via"] == "chat-bridge-signup-decider"


async def test_the_tick_resumes_as_the_decider_not_the_system(h: Harness) -> None:
    h.lic.ensure_error = RuntimeError("licences down")
    await h.intake()
    sid = await h.pending_id()
    async with h.client() as c:
        await c.post(f"/api/admin/signup-approvals/{sid}/approve")
    assert (await h.row(sid)).steps["provision"]["status"] == "failed"
    h.lic.ensure_error = None
    async with h.sf() as s:
        await s.execute(
            update(SignupApproval)
            .where(SignupApproval.id == sid)
            .values(steps_next_attempt_at=func.now())
        )
        await s.commit()
    assert await core.run_tick(h.sf, h.deps) == 1
    invite, audit = await _invite_and_audit(h, sid)
    assert invite.invited_by == h.admin_subject
    assert audit.detail["actor"] == str(h.admin_subject)
    assert row_actor(await h.row(sid)) == str(h.admin_subject)


async def test_a_kill_mid_send_leaves_in_flight_and_is_never_auto_resent(h: Harness) -> None:
    await h.intake()
    sid = await h.pending_id()
    h.idp.error = Killed()
    with pytest.raises(Killed):
        await core.approve(h.sf, h.deps, sid, core.SignupActor(kind="telegram", telegram_user_id=1))
    assert len(h.idp.calls) == 1
    row = await h.row(sid)
    rec = row.steps["invite_email"]
    assert rec["status"] == "in_flight", "persisted BEFORE the Keycloak call"
    assert rec["detail"]["redirect_uri"] == f"{FRONTEND}/"
    invite, _audit = await _invite_and_audit(h, sid)
    assert invite.status == "accepted", "seated before the email, whatever the send does"

    # The pod is gone: its lease lapses, the tick comes round — and sends nothing.
    h.idp.error = None
    async with h.sf() as s:
        await s.execute(
            update(SignupApproval)
            .where(SignupApproval.id == sid)
            .values(steps_next_attempt_at=func.now(), steps_lease_until=func.now())
        )
        await s.commit()
    assert await core.run_tick(h.sf, h.deps) == 1
    row = await h.row(sid)
    assert len(h.idp.calls) == 1, "an in_flight send is never re-sent automatically"
    assert row.steps["invite_email"]["status"] == "in_flight"
    assert row.steps_next_attempt_at is None, "parked for the operator"
    assert [r.id for r in await core.list_signups(h.sf, "needs_attention")] == [sid]

    # Only the super_admin Retry re-sends.
    async with h.client() as c:
        r = await c.post(f"/api/admin/signup-approvals/{sid}/retry")
    assert r.status_code == 200 and r.json()["approval"]["email_state"] == "sent"
    assert len(h.idp.calls) == 2


async def test_an_unexpected_send_error_is_unknown_not_failed(h: Harness) -> None:
    await h.intake()
    sid = await h.pending_id()
    h.idp.error = RuntimeError("connection reset mid-response")
    await _decide(h, sid, "approve")
    row = await h.row(sid)
    assert row.steps["invite_email"]["status"] == "unknown"
    assert row.steps_next_attempt_at is None


async def test_a_failed_approve_reposts_the_buttons(h: Harness, monkeypatch) -> None:
    await h.intake()
    sid = await h.pending_id()
    await tg.drain_telegram_for_tests()
    h.sent.messages.clear()

    async def no_name(*_a: Any, **_k: Any) -> SignupApproval:
        raise core.SignupError(409, "tenant_name_unavailable", "no free name")

    monkeypatch.setattr(core, "_decide_approved", no_name)
    r = await _decide(h, sid, "approve")
    assert r.status_code == 502 and r.json()["status"] == "failed"
    row = await h.row(sid)
    assert row.status == "pending" and row.approving_claim is None
    await tg.drain_telegram_for_tests()
    [(_url, payload)] = h.sent.messages
    assert payload["text"].startswith(tg.STILL_PENDING_HEADLINE)
    assert "tenant_name_unavailable" in payload["text"]
    buttons = [b for line in payload["reply_markup"]["inline_keyboard"] for b in line]
    assert {b.get("callback_data") for b in buttons} >= {
        tg.callback_data(str(sid), "approve"),
        tg.callback_data(str(sid), "reject"),
    }


async def test_an_approved_address_gets_no_second_workspace(h: Harness) -> None:
    await h.intake()
    sid = await h.pending_id()
    await _decide(h, sid, "approve")
    # Intake: the contract's idempotent answer, no new row.
    again = await h.intake(email="ANN@acme.example")
    assert again.status_code == 202 and again.json() == {"status": "already_pending"}
    async with h.sf() as s:
        n = (await s.execute(select(func.count()).select_from(SignupApproval))).scalar_one()
    assert n == 1
    # super_admin invite: a clear 409, nothing created.
    async with h.client() as c:
        r = await c.post(
            "/api/admin/signup-approvals/invite",
            json={"email": "ann@acme.example", "name": "Ann"},
        )
    assert r.status_code == 409 and r.json()["error"] == "early_access_workspace_exists"
    assert "already has an early-access workspace" in r.json()["detail"]
    assert r.json()["signupId"] == str(sid)
    assert len(h.lic.registered) == 1 and len(h.idp.calls) == 1


async def test_a_rejected_address_can_be_invited(h: Harness) -> None:
    await h.intake()
    sid = await h.pending_id()
    await _decide(h, sid, "reject")
    async with h.client() as c:
        r = await c.post(
            "/api/admin/signup-approvals/invite",
            json={"email": "ann@acme.example", "name": "Ann"},
        )
    assert r.status_code == 200, r.text
    assert r.json()["reusedPending"] is False and r.json()["approval"]["status"] == "approved"


# ─── an uncertain welcome email is told to Telegram once per attempt ─────

UNCERTAIN_PREFIX = "Studio early access: the welcome email for "


async def _uncertain_notices(h: Harness) -> list[dict[str, Any]]:
    await tg.drain_telegram_for_tests()
    return [p for _u, p in h.sent.messages if p["text"].startswith(UNCERTAIN_PREFIX)]


async def _make_due(h: Harness, sid: object) -> None:
    async with h.sf() as s:
        await s.execute(
            update(SignupApproval)
            .where(SignupApproval.id == sid)
            .values(steps_next_attempt_at=func.now(), steps_lease_until=func.now())
        )
        await s.commit()


async def test_an_unknown_email_is_told_once_and_not_on_later_ticks(h: Harness) -> None:
    await h.intake(name="Ann <https://evil.example>")
    sid = await h.pending_id()
    h.idp.error = KeycloakAdminError("timed out", uncertain=True)
    await _decide(h, sid, "approve")
    [notice] = await _uncertain_notices(h)
    text = notice["text"]
    assert "ann@acme.example" in text and "may not have been delivered" in text
    assert f"{FRONTEND}/admin/onboarding?signup={sid}" in text
    units = text.encode("utf-16-le")
    codes = [
        units[2 * e["offset"] : 2 * (e["offset"] + e["length"])].decode("utf-16-le")
        for e in notice["entities"]
    ]
    assert codes == ["Ann <https://evil.example>", "ann@acme.example"]
    [[button]] = notice["reply_markup"]["inline_keyboard"]
    assert button == {"text": "Open in Studio", "url": f"{FRONTEND}/admin/onboarding?signup={sid}"}
    row = await h.row(sid)
    assert row.steps["invite_email"]["status"] == "unknown"
    assert row.steps["invite_email"]["operator_notified_at"]

    # Later ticks (and direct step runs) say nothing more.
    for _ in range(2):
        await _make_due(h, sid)
        await core.run_tick(h.sf, h.deps)
        await core.run_steps(h.sf, h.deps, sid, actor=core.SYSTEM_ACTOR)
    assert len(await _uncertain_notices(h)) == 1
    assert len(h.idp.calls) == 1

    # A Retry that ends uncertain again is a NEW occurrence: told once more.
    async with h.client() as c:
        r = await c.post(f"/api/admin/signup-approvals/{sid}/retry")
    assert r.status_code == 200, r.text
    assert len(await _uncertain_notices(h)) == 2


async def test_a_resumed_in_flight_email_is_told_once(h: Harness) -> None:
    await h.intake()
    sid = await h.pending_id()
    h.idp.error = Killed()
    with pytest.raises(Killed):
        await core.approve(h.sf, h.deps, sid, core.SignupActor(kind="telegram", telegram_user_id=1))
    assert await _uncertain_notices(h) == [], "the killed process told nobody"
    h.idp.error = None
    await _make_due(h, sid)
    assert await core.run_tick(h.sf, h.deps) == 1
    [notice] = await _uncertain_notices(h)
    assert "Send status: in_flight" in notice["text"]
    row = await h.row(sid)
    assert row.steps["invite_email"]["status"] == "in_flight"
    assert row.steps["invite_email"]["operator_notified_at"]
    for _ in range(2):
        await _make_due(h, sid)
        await core.run_tick(h.sf, h.deps)
    assert len(await _uncertain_notices(h)) == 1
    assert len(h.idp.calls) == 1, "still never re-sent automatically"


async def test_a_telegram_failure_does_not_fail_the_step(h: Harness, caplog) -> None:
    async def broken(_url: str, _payload: dict[str, Any]) -> httpx.Response:
        raise httpx.ConnectError("telegram down")

    tg.set_telegram_post_for_tests(broken)
    sleeps: list[float] = []

    async def no_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    original_sleep = tg._sleep
    tg._sleep = no_sleep
    try:
        await h.intake()
        sid = await h.pending_id()
        h.idp.error = RuntimeError("connection reset mid-response")
        r = await _decide(h, sid, "approve")
        assert r.status_code == 200, r.text
        await tg.drain_telegram_for_tests()
    finally:
        tg._sleep = original_sleep
    row = await h.row(sid)
    assert row.status == "approved"
    assert row.steps["invite_email"]["status"] == "unknown"
    assert row.steps["invite_email"]["operator_notified_at"], "told once, even if it failed"
    assert row.steps_next_attempt_at is None
    assert "SECRET-BOT-TOKEN" not in caplog.text


async def test_a_notice_crash_never_fails_the_run(h: Harness, monkeypatch) -> None:
    def boom(_n: Any) -> bool:
        raise RuntimeError("notice crashed")

    monkeypatch.setattr(core, "notify_email_uncertain", boom)
    await h.intake()
    sid = await h.pending_id()
    h.idp.error = RuntimeError("connection reset mid-response")
    r = await _decide(h, sid, "approve")
    assert r.status_code == 200, r.text
    row = await h.row(sid)
    assert row.status == "approved" and row.steps["invite_email"]["status"] == "unknown"


# ─── the identity step (George 2026-10-20: exactly like CommonPractice / fr) ──


async def _force_due(h: Harness, sid: UUID) -> None:
    async with h.sf() as s:
        await s.execute(
            update(SignupApproval)
            .where(SignupApproval.id == sid)
            .values(steps_next_attempt_at=func.now())
        )
        await s.commit()


async def test_a_foreign_existing_login_is_refused_before_anything_is_founded(h: Harness) -> None:
    foreign = h.idp.seed("ann@acme.example", {"tenant_id": ["someone-else"]})
    await h.intake()
    sid = await h.pending_id()
    r = await _decide(h, sid, "approve")
    assert r.status_code == 200 and r.json()["status"] == "approved"
    row = await h.row(sid)
    assert row.steps["identity"]["status"] == "failed"
    assert row.steps["identity"]["code"] == ident.IDENTITY_EXISTS
    assert row.steps["identity"]["detail"]["user_id"] == foreign
    for later in ("provision", "membership", "invite_email"):
        assert row.steps[later]["status"] == "waiting"
    assert h.lic.registered == [], "no tenant for a person this path cannot onboard"
    assert h.idp.created == [] and h.idp.calls == [] and h.idp.attr_writes == []
    assert h.idp.deleted == [] and foreign in h.idp.users, "never touched"
    assert [r.id for r in await core.list_signups(h.sf, "needs_attention")] == [sid]
    # The tick retries it and still refuses it.
    await _force_due(h, sid)
    assert await core.run_tick(h.sf, h.deps) == 1
    assert (await h.row(sid)).steps["identity"]["code"] == ident.IDENTITY_EXISTS
    assert h.idp.calls == []


async def test_a_login_stamped_with_this_row_is_adopted(h: Harness) -> None:
    """A crash between Keycloak's create and the step record: the resume finds
    the address taken by a user carrying THIS row's id, and adopts it."""
    await h.intake()
    sid = await h.pending_id()
    own = h.idp.seed("ann@acme.example", {"w2v_signup_id": [str(sid)]})
    await _decide(h, sid, "approve")
    row = await h.row(sid)
    assert row.steps["identity"]["status"] == "ok"
    assert row.steps["identity"]["detail"] == {"user_id": own, "minted": False}
    assert h.idp.created == []
    [call] = h.idp.calls
    assert call["user_id"] == own
    # locale + signup_app were missing on the adopted user: set before the send.
    assert h.idp.attr_writes == [(own, {"locale": "en", "signup_app": "w2v"})]


async def test_a_login_stamped_with_another_row_is_refused(h: Harness) -> None:
    await h.intake()
    sid = await h.pending_id()
    h.idp.seed("ann@acme.example", {"w2v_signup_id": [str(uuid7())]})
    await _decide(h, sid, "approve")
    assert (await h.row(sid)).steps["identity"]["code"] == ident.IDENTITY_EXISTS
    assert h.idp.calls == [] and h.lic.registered == []


async def test_a_dropped_stamp_fails_and_removes_the_login_it_minted(h: Harness) -> None:
    h.idp.drop_attrs = {"w2v_signup_id"}
    await h.intake()
    sid = await h.pending_id()
    await _decide(h, sid, "approve")
    row = await h.row(sid)
    assert row.steps["identity"]["code"] == ident.IDENTITY_ATTR_DROPPED
    assert row.steps["identity"]["detail"]["deleted"] is True
    assert h.idp.users == {} and len(h.idp.deleted) == 1
    assert h.lic.registered == [] and h.idp.calls == []


async def test_a_keycloak_failure_is_identity_failed(h: Harness, monkeypatch) -> None:
    async def forbidden(_rep: dict[str, Any]) -> str | None:
        raise KeycloakAdminError("create_user returned 403", status_code=403)

    monkeypatch.setattr(h.idp, "create_user", forbidden)
    await h.intake()
    sid = await h.pending_id()
    await _decide(h, sid, "approve")
    row = await h.row(sid)
    assert row.steps["identity"]["status"] == "failed"
    assert row.steps["identity"]["code"] == ident.IDENTITY_FAILED
    assert h.lic.registered == [] and h.idp.calls == []


async def test_the_email_never_goes_to_a_login_that_lost_its_stamp(h: Harness) -> None:
    h.lic.ensure_error = RuntimeError("licences down")
    await h.intake()
    sid = await h.pending_id()
    await _decide(h, sid, "approve")
    row = await h.row(sid)
    user_id = row.steps["identity"]["detail"]["user_id"]
    h.idp.users[user_id]["attributes"].pop("w2v_signup_id")
    h.lic.ensure_error = None
    await core.run_steps(h.sf, h.deps, sid, actor=core.SYSTEM_ACTOR)
    row = await h.row(sid)
    assert row.steps["provision"]["status"] == "ok"
    assert row.steps["invite_email"]["status"] == "failed"
    assert row.steps["invite_email"]["code"] == ident.IDENTITY_NOT_OWNED
    assert h.idp.calls == [] and h.idp.attr_writes == []
    # It EXISTS but is not ours: the recorded user is kept and keeps being refused.
    await core.retry(h.sf, h.deps, sid, core.SYSTEM_ACTOR)
    row = await h.row(sid)
    assert row.steps["identity"]["detail"]["user_id"] == user_id
    assert row.steps["invite_email"]["code"] == ident.IDENTITY_NOT_OWNED
    assert h.idp.created[1:] == [] and h.idp.calls == [] and h.idp.deleted == []


async def test_a_login_deleted_after_the_identity_step_is_minted_again(h: Harness) -> None:
    """fr: the recorded user was deleted out of band — forget it, and the next
    attempt / Retry re-runs the identity step (a fresh user) and mails THAT one,
    instead of failing identity_not_owned forever."""
    h.lic.ensure_error = RuntimeError("licences down")
    await h.intake()
    sid = await h.pending_id()
    await _decide(h, sid, "approve")
    old = (await h.row(sid)).steps["identity"]["detail"]["user_id"]
    del h.idp.users[old]
    h.lic.ensure_error = None
    await core.run_steps(h.sf, h.deps, sid, actor=core.SYSTEM_ACTOR)
    row = await h.row(sid)
    assert row.steps["invite_email"]["code"] == ident.IDENTITY_GONE
    assert row.steps["identity"]["status"] == "failed"
    assert row.steps["identity"]["code"] == ident.IDENTITY_GONE
    assert h.idp.calls == []
    await core.retry(h.sf, h.deps, sid, core.SYSTEM_ACTOR)
    row = await h.row(sid)
    new = row.steps["identity"]["detail"]["user_id"]
    assert row.steps["identity"]["status"] == "ok" and new != old
    assert row.steps["invite_email"]["status"] == "ok"
    [call] = h.idp.calls
    assert call["user_id"] == new
    assert len(h.idp.created) == 2


async def test_a_login_deleted_after_its_seat_is_unseated_and_the_new_one_seated(
    h: Harness,
) -> None:
    """The seat follows the login: a seat bound to a login deleted out of band is
    ended, and the re-minted login is seated before its email goes."""
    h.idp.error = KeycloakAdminError("execute_actions_email returned 500", status_code=500)
    await h.intake()
    sid = await h.pending_id()
    await _decide(h, sid, "approve")
    row = await h.row(sid)
    assert row.steps["membership"]["status"] == "ok"
    assert row.steps["invite_email"]["status"] == "failed"
    old = row.steps["identity"]["detail"]["user_id"]
    del h.idp.users[old]
    h.idp.error = None
    await core.retry(h.sf, h.deps, sid, core.SYSTEM_ACTOR)  # identity gone: unseat
    await core.retry(h.sf, h.deps, sid, core.SYSTEM_ACTOR)  # re-mint, seat, send
    row = await h.row(sid)
    new = row.steps["identity"]["detail"]["user_id"]
    assert new != old and row.steps["invite_email"]["status"] == "ok"
    async with h.sf() as s:
        members = {
            (m.subject_id, m.status)
            for m in (
                await s.execute(select(TenantMember).where(TenantMember.tenant_id == row.tenant_id))
            ).scalars()
        }
    assert members == {(UUID(old), "removed"), (UUID(new), "active")}
    # The membership step re-ran for the new login, so its record names it.
    assert row.steps["membership"]["status"] == "ok"
    assert row.steps["membership"]["detail"]["subject_id"] == new
    assert [c["user_id"] for c in h.idp.calls] == [old, new]
    assert all(c["redirect_uri"] == f"{FRONTEND}/" for c in h.idp.calls)


async def test_a_resend_reuses_the_same_tokenless_link(h: Harness) -> None:
    """CP: the email lands on the app root, so a Retry never invalidates an
    earlier email — there is no per-send token to rotate."""
    h.idp.error = RuntimeError("connection reset mid-response")
    await h.intake()
    sid = await h.pending_id()
    await _decide(h, sid, "approve")
    assert (await h.row(sid)).steps["invite_email"]["status"] == "unknown"
    h.idp.error = None
    await core.retry(h.sf, h.deps, sid, core.SYSTEM_ACTOR)
    assert [c["redirect_uri"] for c in h.idp.calls] == [f"{FRONTEND}/", f"{FRONTEND}/"]
    invite, _audit = await _invite_and_audit(h, sid)
    assert invite.status == "accepted"


async def test_approve_and_invite_record_the_chosen_llm_access(h: Harness) -> None:
    """CP's per-person plan choice: the approver picks the LLM access on approve
    and on the invite form; it is recorded on the row and provisioned."""
    await h.intake()
    sid = await h.pending_id()
    async with h.client() as c:
        r = await c.post(
            f"/api/admin/signup-approvals/{sid}/approve", json={"llmAccess": "platform_only"}
        )
        assert r.status_code == 200, r.text
        assert r.json()["approval"]["llm_access"] == "platform_only"
        inv = await c.post(
            "/api/admin/signup-approvals/invite",
            json={"email": "zoe@zeta.example", "name": "Zoe", "llmAccess": "platform_only"},
        )
        assert inv.status_code == 200, inv.text
        dflt = await c.post(
            "/api/admin/signup-approvals/invite",
            json={"email": "max@mu.example", "name": "Max"},
        )
        assert dflt.status_code == 200, dflt.text
    for body in (r.json(), inv.json()):
        tid = body["tenantId"]
        assert h.lic.byo_overrides[(tid, "w2v")] == "platform_only"
        row = await h.row(body["approval"]["id"])
        assert row.llm_access == "platform_only"
        assert row.steps["provision"]["detail"]["llm_access"]["mode"] == "platform_only"
    assert h.lic.byo_overrides[(dflt.json()["tenantId"], "w2v")] == "local_allowed"
    assert dflt.json()["approval"]["llm_access"] == "local_allowed"


async def test_approve_and_invite_accept_byo_required(h: Harness) -> None:
    """Owner decision 2026-10-22: all three postures are offered on approve / invite
    (like ``POST /api/admin/provision``); ``byo_required`` is recorded + provisioned."""
    await h.intake()
    sid = await h.pending_id()
    async with h.client() as c:
        r = await c.post(
            f"/api/admin/signup-approvals/{sid}/approve", json={"llmAccess": "byo_required"}
        )
        assert r.status_code == 200, r.text
        inv = await c.post(
            "/api/admin/signup-approvals/invite",
            json={"email": "zoe@zeta.example", "name": "Zoe", "llmAccess": "byo_required"},
        )
        assert inv.status_code == 200, inv.text
    for body in (r.json(), inv.json()):
        assert h.lic.byo_overrides[(body["tenantId"], "w2v")] == "byo_required"
        row = await h.row(body["approval"]["id"])
        assert row.llm_access == "byo_required"
        assert row.steps["provision"]["detail"]["llm_access"]["mode"] == "byo_required"


async def test_telegram_approve_passes_local_allowed_explicitly(h: Harness, monkeypatch) -> None:
    """The one-tap Telegram approve is "Platform models allowed" (owner choice) —
    EXPLICITLY, never the deployment default it used to fall back to."""
    monkeypatch.setenv("W2V_SIGNUP_LLM_ACCESS", "byo_required")
    clear_settings_cache()
    await h.intake()
    sid = await h.pending_id()
    seen: list[object] = []
    real = core.approve

    async def _spy(*a: Any, **k: Any) -> Any:
        seen.append(k.get("llm_access"))
        return await real(*a, **k)

    monkeypatch.setattr(core, "approve", _spy)
    r = await _decide(h, sid, "approve")
    assert r.status_code == 200, r.text
    assert seen == ["local_allowed"]
    row = await h.row(sid)
    assert row.llm_access == "local_allowed"
    assert h.lic.byo_overrides[(str(row.tenant_id), "w2v")] == "local_allowed"


@pytest.mark.parametrize("bad", ["anything"])
async def test_an_unknown_llm_access_is_refused_on_approve_and_invite(h: Harness, bad: str) -> None:
    """An unknown posture is a 422, nothing decided."""
    await h.intake()
    sid = await h.pending_id()
    async with h.client() as c:
        r = await c.post(f"/api/admin/signup-approvals/{sid}/approve", json={"llmAccess": bad})
        inv = await c.post(
            "/api/admin/signup-approvals/invite",
            json={"email": "zoe@zeta.example", "name": "Zoe", "llmAccess": bad},
        )
    assert r.status_code == 422 and inv.status_code == 422
    assert (await h.row(sid)).status == "pending"
    assert h.lic.byo_overrides == {}


async def test_a_row_decided_without_a_recorded_choice_takes_the_default(
    h: Harness, monkeypatch
) -> None:
    """A row approved before the column existed (NULL) provisions the deployment
    default; the default itself is configurable."""
    monkeypatch.setenv("W2V_SIGNUP_LLM_ACCESS", "platform_only")
    clear_settings_cache()
    await h.intake()
    sid = await h.pending_id()
    row = await h.row(sid)
    assert core.llm_access_of(row) == "platform_only"
    row.llm_access = "local_allowed"
    assert core.llm_access_of(row) == "local_allowed"
    async with h.client() as c:  # an admin approve WITHOUT llmAccess: the default
        r = await c.post(f"/api/admin/signup-approvals/{sid}/approve")
    assert r.status_code == 200, r.text
    row = await h.row(sid)
    assert row.llm_access == "platform_only"
    assert h.lic.byo_overrides[(str(row.tenant_id), "w2v")] == "platform_only"


async def test_a_stamped_login_with_a_credential_is_not_adopted(h: Harness) -> None:
    await h.intake()
    sid = await h.pending_id()
    own = h.idp.seed("ann@acme.example", {"w2v_signup_id": [str(sid)]})
    h.idp.users[own]["credentials"] = [{"type": "password"}]
    await _decide(h, sid, "approve")
    assert (await h.row(sid)).steps["identity"]["code"] == ident.IDENTITY_EXISTS
    assert h.idp.calls == [] and h.lic.registered == [] and own in h.idp.users


async def test_a_failed_attribute_write_does_not_block_the_email(h: Harness, monkeypatch) -> None:
    h.lic.ensure_error = RuntimeError("licences down")
    await h.intake(locale="de")
    sid = await h.pending_id()
    await _decide(h, sid, "approve")
    user_id = (await h.row(sid)).steps["identity"]["detail"]["user_id"]
    h.idp.users[user_id]["attributes"].pop("signup_app")

    async def broken(_uid: str, _attrs: dict[str, str | None]) -> None:
        raise KeycloakAdminError("update_user_attributes returned 500", status_code=500)

    monkeypatch.setattr(h.idp, "update_user_attributes", broken)
    h.lic.ensure_error = None
    await core.run_steps(h.sf, h.deps, sid, actor=core.SYSTEM_ACTOR)
    rec = (await h.row(sid)).steps["invite_email"]
    assert rec["status"] == "ok" and rec["detail"]["attributes"] is False
    assert len(h.idp.calls) == 1


async def test_a_seat_removed_after_seating_is_never_re_seated_by_the_email(h: Harness) -> None:
    """A super_admin removed the seat after the membership step (the email had not
    gone yet): the Retry must not undo that — the email step fails
    ``seat_removed``, sends nothing, and the membership stays removed."""
    h.idp.error = KeycloakAdminError("execute_actions_email returned 500", status_code=500)
    await h.intake()
    sid = await h.pending_id()
    await _decide(h, sid, "approve")
    row = await h.row(sid)
    assert row.steps["membership"]["status"] == "ok"
    user_id = row.steps["identity"]["detail"]["user_id"]
    assert row.tenant_id is not None
    async with h.sf() as s:
        assert await MembersRepository(s).remove_member(UUID(user_id), row.tenant_id)
        await s.commit()
    h.idp.error = None
    await core.retry(h.sf, h.deps, sid, core.SYSTEM_ACTOR)
    row = await h.row(sid)
    rec = row.steps["invite_email"]
    assert (rec["status"], rec["code"]) == ("failed", core.SEAT_REMOVED)
    assert rec["detail"]["subject_id"] == user_id
    assert row.steps["membership"]["status"] == "ok"
    assert len(h.idp.calls) == 1, "only the first (refused) send — no email after removal"
    async with h.sf() as s:
        member = (
            await s.execute(select(TenantMember).where(TenantMember.tenant_id == row.tenant_id))
        ).scalar_one()
    assert (member.subject_id, member.status) == (UUID(user_id), "removed")


async def test_a_removed_seat_whose_login_was_deleted_is_not_re_seated(h: Harness) -> None:
    """A super_admin removed the seat AND the login was deleted out of band: the
    identity-gone path finds no seat to end, so it must not reopen membership or
    forget the login — re-seating a re-minted login would undo the removal. The
    step records ``seat_removed`` and the operator decides."""
    h.idp.error = KeycloakAdminError("execute_actions_email returned 500", status_code=500)
    await h.intake()
    sid = await h.pending_id()
    await _decide(h, sid, "approve")
    row = await h.row(sid)
    old = row.steps["identity"]["detail"]["user_id"]
    assert row.tenant_id is not None
    async with h.sf() as s:
        assert await MembersRepository(s).remove_member(UUID(old), row.tenant_id)
        await s.commit()
    del h.idp.users[old]
    h.idp.error = None
    await core.retry(h.sf, h.deps, sid, core.SYSTEM_ACTOR)
    await core.retry(h.sf, h.deps, sid, core.SYSTEM_ACTOR)
    row = await h.row(sid)
    rec = row.steps["invite_email"]
    assert (rec["status"], rec["code"]) == ("failed", core.SEAT_REMOVED)
    assert row.steps["identity"]["status"] == "ok"
    assert row.steps["identity"]["detail"]["user_id"] == old
    assert row.steps["membership"]["status"] == "ok"
    assert row.steps_next_attempt_at is None
    assert len(h.idp.created) == 1, "no login re-minted"
    assert len(h.idp.calls) == 1, "only the first (refused) send"
    async with h.sf() as s:
        members = {
            (m.subject_id, m.status)
            for m in (
                await s.execute(select(TenantMember).where(TenantMember.tenant_id == row.tenant_id))
            ).scalars()
        }
    assert members == {(UUID(old), "removed")}


async def test_seat_removed_is_parked_and_never_auto_retried(h: Harness) -> None:
    """``seat_removed`` is operator-terminal like ``unknown``: the tick parks the
    row (no next attempt) and never re-runs the email step on its own."""
    h.idp.error = KeycloakAdminError("execute_actions_email returned 500", status_code=500)
    await h.intake()
    sid = await h.pending_id()
    await _decide(h, sid, "approve")
    row = await h.row(sid)
    user_id = row.steps["identity"]["detail"]["user_id"]
    assert row.tenant_id is not None
    async with h.sf() as s:
        assert await MembersRepository(s).remove_member(UUID(user_id), row.tenant_id)
        await s.execute(
            update(SignupApproval)
            .where(SignupApproval.id == sid)
            .values(steps_next_attempt_at=func.now())
        )
        await s.commit()
    h.idp.error = None
    assert await core.run_tick(h.sf, h.deps) == 1
    row = await h.row(sid)
    rec = row.steps["invite_email"]
    assert (rec["status"], rec["code"]) == ("failed", core.SEAT_REMOVED)
    assert row.steps_next_attempt_at is None, "seat_removed parks the row for the operator"
    assert core.awaiting_operator(row)
    run = await core.run_steps(h.sf, h.deps, sid, actor=core.SYSTEM_ACTOR)
    assert run.order == [] and len(h.idp.calls) == 1
    assert (await h.row(sid)).steps["invite_email"]["at"] == rec["at"]


async def test_the_seat_re_arms_an_expired_pending_invite_before_accepting(h: Harness) -> None:
    """``accept_invite_row`` binds only an unexpired invite and leaves the check
    to its caller: the seat path re-arms an expired pending invite (audited)
    before it accepts it — never accepts it while expired."""
    h.idp.error = KeycloakAdminError("execute_actions_email returned 500", status_code=500)
    await h.intake()
    sid = await h.pending_id()
    await _decide(h, sid, "approve")
    row = await h.row(sid)
    stale = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=1)
    other = uuid7()
    assert row.tenant_id is not None
    async with h.sf() as s:
        pending = await InvitesRepository(s).create(
            tenant_id=row.tenant_id,
            email=row.email,
            role="tenant_admin",
            token_hash=new_invite_token()[1],
            expires_at=stale,
            invited_by=None,
        )
        await s.commit()
    before = datetime.datetime.now(datetime.UTC)
    async with h.sf() as s:
        invite_id, already = await core._seat(s, row, str(other), core.SYSTEM_ACTOR)
        await s.commit()
    assert (invite_id, already) == (pending.id, False)
    async with h.sf() as s:
        inv = (
            await s.execute(select(TenantInvite).where(TenantInvite.id == pending.id))
        ).scalar_one()
        audits = (
            (await s.execute(select(AdminAuditLog).where(AdminAuditLog.action == "member.invite")))
            .scalars()
            .all()
        )
    [audit] = [a for a in audits if a.detail.get("rearmed")]
    assert inv.status == "accepted" and inv.accepted_subject_id == other
    assert inv.expires_at > before
    assert inv.accepted_at is not None and inv.accepted_at < inv.expires_at
    assert audit.detail["invite_id"] == str(pending.id)


async def test_the_list_exposes_the_deployment_default_llm_access(h: Harness, monkeypatch) -> None:
    """The page pre-selects this and sends nothing unless the operator changes it,
    so ``W2V_SIGNUP_LLM_ACCESS`` governs the onboarding page as well as Telegram."""
    async with h.client() as c:
        assert (await c.get("/api/admin/signup-approvals")).json()[
            "defaultLlmAccess"
        ] == "local_allowed"
    monkeypatch.setenv("W2V_SIGNUP_LLM_ACCESS", "platform_only")
    clear_settings_cache()
    async with h.client() as c:
        assert (await c.get("/api/admin/signup-approvals")).json()[
            "defaultLlmAccess"
        ] == "platform_only"
