"""P2 tenancy WP3 — membership resolution on the REAL request path.

Every request here goes through the production auth stack: a real RS256 token
minted per test, verified by w2v_common's ``KeycloakJWTValidator`` inside the
``AuthMiddleware`` that ``create_app`` registers, then ``TenancyMiddleware``
(installed exactly as ``src/asgi.py`` does), then real routes against a real
Postgres ``tenant_members`` / ``tenants`` / ``tenant_api_keys``. The only fakes
are the JWKS socket and the outbound peer's socket.

Non-vacuity: every assertion about a resolved tenant is about a tenant that
differs from the token's claim, or a status the claim lane would not produce,
so ``W2V_TENANCY_SOURCE=claim`` (or a missing middleware) fails them. One test
pins the control directly: without the middleware, the SA-bearer hop of a
tenant-less inbound request raises ``MissingTenantError``.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import sys
import time
import types
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import APIRouter, Depends, FastAPI, Request
from jwt.algorithms import RSAAlgorithm
from prometheus_client import REGISTRY
from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from w2v_common.asgi import create_app
from w2v_common.auth import keycloak as keycloak_mod
from w2v_common.auth import register_api_key_verifier
from w2v_common.auth.identity_context import (
    get_actor_id_from_context,
    get_license_id_from_context,
    get_project_id_from_context,
    get_tenant_id_from_context,
    get_tenant_source_from_context,
)
from w2v_common.tenant import MissingTenantError

from src.api.routes._common import (
    TenantAdminDep,
    TenantIdDep,
    get_licenses_client,
    require_known_tenant,
)
from src.api.routes.auth import router as auth_router
from src.auth.membership import (
    MembershipRow,
    Refusal,
    Resolution,
    lazy_membership_role,
    resolve_tenant,
)
from src.auth.tenancy_middleware import (
    align_jwt_tenant_requirement,
    install_tenancy_middleware,
    make_api_key_verifier,
)
from src.clients.design_client import DesignClient
from src.config import clear_settings_cache, get_settings
from src.database.models import RequestAudit, Tenant, TenantApiKey, TenantMember
from src.database.repositories.api_keys_repository import ApiKeysRepository
from src.database.repositories.members_repository import MembersRepository
from src.middleware.audit import AuditMiddleware

pytestmark = pytest.mark.asyncio

KEYCLOAK_URL = "https://keycloak.test"
REALM = "portal"
ISSUER = f"{KEYCLOAK_URL}/realms/{REALM}"
JWKS_URI = f"{ISSUER}/protocol/openid-connect/certs"
AUDIENCE = "w2v-frontend"
KID = "wp3-kid"
_SA_TOKEN = "sa-license-token-not-a-jwt"  # a header value, authenticates nothing
# Outside TRUSTED_NETWORKS so no request is decided by the trusted-network lane.
_PUBLIC_PEER = ("203.0.113.9", 50000)
_MISMATCH = "w2v_orchestrator_tenancy_shadow_mismatch_total"
_CHECKED = "w2v_orchestrator_tenancy_shadow_checked_total"
_REQ_UNATTRIBUTED = "w2v_orchestrator_request_licence_unattributed_total"


@pytest.fixture(scope="module")
def signing_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@dataclass
class Harness:
    app: FastAPI
    key: rsa.RSAPrivateKey
    factory: async_sessionmaker[AsyncSession]
    monkeypatch: pytest.MonkeyPatch
    outbound: list[httpx.Request] = field(default_factory=list)

    def mode(self, source: str, *, tenant_required: bool = True) -> None:
        self.monkeypatch.setenv("W2V_TENANCY_SOURCE", source)
        self.monkeypatch.setenv("AUTH_JWT_TENANT_REQUIRED", "true" if tenant_required else "false")
        clear_settings_cache()

    def token(
        self,
        sub: uuid.UUID,
        *,
        claim: uuid.UUID | None = None,
        roles: list[str] | None = None,
        extra: dict[str, object] | None = None,
    ) -> str:
        now = int(time.time())
        claims: dict[str, object] = {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": str(sub),
            "iat": now,
            "exp": now + 300,
            "email": "ada@acme.test",
            "email_verified": True,
            "preferred_username": "ada",
            "realm_access": {"roles": roles or []},
        }
        if claim is not None:
            claims["tenant_id"] = str(claim)
        claims.update(extra or {})
        return jwt.encode(claims, self.key, algorithm="RS256", headers={"kid": KID})

    async def get(
        self, path: str, bearer: str, headers: dict[str, str] | None = None
    ) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app, client=_PUBLIC_PEER)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.get(
                path, headers={"Authorization": f"Bearer {bearer}", **(headers or {})}
            )

    async def tenant(
        self, status: str = "active", *, licence_ref: uuid.UUID | None = None
    ) -> uuid.UUID:
        tid = uuid.uuid7()
        async with self.factory() as s:
            await s.execute(
                insert(Tenant).values(
                    id=tid, name=f"t-{tid}", status=status, licence_ref=licence_ref
                )
            )
            await s.commit()
        return tid

    async def member(
        self,
        sub: uuid.UUID,
        tenant: uuid.UUID,
        role: str = "member",
        *,
        seen_minutes_ago: int = 0,
    ) -> None:
        async with self.factory() as s:
            await MembersRepository(s).add_member(subject_id=sub, tenant_id=tenant, role=role)  # ty: ignore[invalid-argument-type]  # why: test passes the two valid literals as plain str
            await s.execute(
                update(TenantMember)
                .where(TenantMember.subject_id == sub, TenantMember.tenant_id == tenant)
                .values(
                    last_seen_at=datetime.datetime.now(datetime.UTC)
                    - datetime.timedelta(minutes=seen_minutes_ago)
                )
            )
            await s.commit()

    async def rows(self, sub: uuid.UUID) -> dict[uuid.UUID, tuple[str, str]]:
        async with self.factory() as s:
            result = await s.execute(
                select(TenantMember.tenant_id, TenantMember.role, TenantMember.status).where(
                    TenantMember.subject_id == sub
                )
            )
            return {r.tenant_id: (r.role, r.status) for r in result}


class _NoLicenceLookups:
    """require_known_tenant must not need the licences service for a member."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"licences client used ({name}) for a membership-resolved tenant")


def _build_app(
    factory: async_sessionmaker[AsyncSession], sink: list[httpx.Request], *, tenancy: bool
) -> FastAPI:
    probe = APIRouter(prefix="/api/probe")

    @probe.get("")
    async def whoami(request: Request, tenant_id: TenantIdDep) -> dict[str, object]:
        source = get_tenant_source_from_context()
        return {
            "tenant": str(tenant_id),
            "context_tenant": str(get_tenant_id_from_context()),
            "roles": list(request.state.roles),
            "source": source.value if source else None,
            "actor": get_actor_id_from_context(),
            "project": get_project_id_from_context(),
            "license": get_license_id_from_context(),
        }

    @probe.get("/admin")
    async def admin_only(tenant_id: TenantAdminDep) -> dict[str, str]:
        return {"tenant": str(tenant_id)}

    @probe.get("/known", dependencies=[Depends(require_known_tenant)])
    async def known(tenant_id: TenantIdDep) -> dict[str, str]:
        return {"tenant": str(tenant_id)}

    @probe.get("/outbound")
    async def outbound() -> dict[str, str]:
        # A REAL production client (SA bearer baked in, identity_event_hook
        # wired) aimed at a METERED peer; only the socket is fake.
        client = DesignClient(base_url="http://html-render.test:6319", license_token=_SA_TOKEN)

        def handler(request: httpx.Request) -> httpx.Response:
            sink.append(request)
            return httpx.Response(200, json={"job_id": "job-1", "status": "pending"})

        inner = await client._get_client()
        inner._transport = httpx.MockTransport(handler)
        try:
            await client.start_design(url="https://example.com", scope="redesign")
        finally:
            await client.close()
        return {"ok": "1"}

    app = create_app(get_settings(), routers=[auth_router, probe], cors_enabled=False)
    if tenancy:
        install_tenancy_middleware(app)
    app.state.session_factory = factory
    app.dependency_overrides[get_licenses_client] = _NoLicenceLookups
    return app


@pytest.fixture
def harness(monkeypatch, signing_key, db_session_factory) -> Iterator[Harness]:
    monkeypatch.setenv("KEYCLOAK_URL", KEYCLOAK_URL)
    monkeypatch.setenv("KEYCLOAK_REALM", REALM)
    monkeypatch.setenv("KEYCLOAK_AUDIENCES", AUDIENCE)
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "false")
    for name in ("KEYCLOAK_ISSUER", "KEYCLOAK_JWT_LEEWAY", "KEYCLOAK_JWKS_TTL"):
        monkeypatch.delenv(name, raising=False)
    clear_settings_cache()

    jwk = RSAAlgorithm.to_jwk(signing_key.public_key(), as_dict=True)
    jwk.update({"kid": KID, "alg": "RS256", "use": "sig"})
    real_async_client = httpx.AsyncClient

    def _serve_jwks(request: httpx.Request) -> httpx.Response:
        if str(request.url) != JWKS_URI:
            return httpx.Response(404)
        return httpx.Response(200, json={"keys": [jwk]})

    def _mock_async_client(**kwargs: Any) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(_serve_jwks), **kwargs)

    monkeypatch.setattr(
        keycloak_mod,
        "httpx",
        types.SimpleNamespace(AsyncClient=_mock_async_client, HTTPError=httpx.HTTPError),
    )
    keycloak_mod.reset_jwt_validator()
    register_api_key_verifier(make_api_key_verifier(db_session_factory))

    sink: list[httpx.Request] = []
    app = _build_app(db_session_factory, sink, tenancy=True)
    h = Harness(app=app, key=signing_key, factory=db_session_factory, monkeypatch=monkeypatch)
    h.outbound = sink
    yield h

    register_api_key_verifier(None)
    keycloak_mod.reset_jwt_validator()
    clear_settings_cache()


def _mismatches(reason: str) -> float:
    return REGISTRY.get_sample_value(_MISMATCH, {"reason": reason}) or 0.0


def _checked() -> dict[str, float]:
    return {
        result: REGISTRY.get_sample_value(_CHECKED, {"result": result}) or 0.0
        for result in ("match", "mismatch", "error", "claim_missing")
    }


def _checked_delta(before: dict[str, float]) -> dict[str, float]:
    after = _checked()
    return {k: after[k] - before[k] for k in after if after[k] != before[k]}


# ─── pure rules ──────────────────────────────────────────────────────────


async def test_rules_requested_single_most_recent_and_refusals() -> None:
    a, b, x = uuid.uuid7(), uuid.uuid7(), uuid.uuid7()
    rows = [MembershipRow(b, "member"), MembershipRow(a, "tenant_admin")]  # b most recent

    assert resolve_tenant(rows, a, is_super_admin=False) == Resolution(
        a, "tenant_admin", "requested"
    )
    assert resolve_tenant(rows, None, is_super_admin=False) == Resolution(
        b, "member", "most_recent"
    )
    assert resolve_tenant(rows[:1], None, is_super_admin=False) == Resolution(b, "member", "single")
    assert resolve_tenant(rows, x, is_super_admin=False) == Refusal("not_a_member")
    assert resolve_tenant([], None, is_super_admin=False) == Refusal("no_membership")
    assert resolve_tenant([], x, is_super_admin=True) == Resolution(x, None, "super_admin")
    assert resolve_tenant([], None, is_super_admin=True) == Refusal("no_membership")
    # Lazy role mirrors 0072: super_admin never becomes tenant_admin.
    assert lazy_membership_role({"tenant_admin"}) == "tenant_admin"
    assert lazy_membership_role({"tenant_admin", "super_admin"}) == "member"


# ─── membership mode ─────────────────────────────────────────────────────


async def test_membership_mode_ignores_the_claim(harness: Harness) -> None:
    harness.mode("membership")
    sub, fr_tenant = uuid.uuid7(), uuid.uuid7()
    w2v_tenant = await harness.tenant()
    await harness.member(sub, w2v_tenant)

    project = uuid.uuid7()
    r = await harness.get(
        "/api/probe", harness.token(sub, claim=fr_tenant), {"X-Project-Id": str(project)}
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tenant"] == str(w2v_tenant)
    assert body["context_tenant"] == str(w2v_tenant), "identity context must be re-seeded"
    assert body["source"] == "membership"
    assert body["actor"] == str(sub), "set_identity_context cleared the actor; it must be re-bound"
    assert body["roles"] == ["member"]
    assert body["project"] == str(project), "the X-Project-Id binding must survive the re-seed"


async def test_switch_only_among_own_memberships(harness: Harness) -> None:
    harness.mode("membership", tenant_required=False)
    sub = uuid.uuid7()
    a, b, foreign = await harness.tenant(), await harness.tenant(), await harness.tenant()
    await harness.member(sub, a, seen_minutes_ago=60)
    await harness.member(sub, b, seen_minutes_ago=1)
    tok = harness.token(sub)

    default = await harness.get("/api/probe", tok)
    by_header = await harness.get("/api/probe", tok, {"X-Active-Tenant-Id": str(a)})
    by_query = await harness.get(f"/api/probe?active_tenant={a}", tok)
    other = await harness.get("/api/probe", tok, {"X-Active-Tenant-Id": str(foreign)})

    assert default.json()["tenant"] == str(b), "no request → most recently used"
    assert by_header.json()["tenant"] == str(a)
    assert by_query.json()["tenant"] == str(a), "EventSource/<img> channel must switch too"
    assert other.status_code == 403
    assert other.json()["error"]["code"] == "NO_MEMBERSHIP"


async def test_removed_member_is_refused_but_me_still_answers(harness: Harness) -> None:
    harness.mode("membership")
    sub = uuid.uuid7()
    t = await harness.tenant()
    await harness.member(sub, t)
    tok = harness.token(sub, claim=t)  # the claim still names the tenant
    assert (await harness.get("/api/probe", tok)).status_code == 200

    async with harness.factory() as s:
        assert await MembersRepository(s).remove_member(sub, t)
        await s.commit()

    r = await harness.get("/api/probe", tok)
    me = await harness.get("/api/auth/me", tok)

    assert r.status_code == 403
    assert r.json()["error"]["code"] == "NO_MEMBERSHIP"
    assert me.status_code == 200, me.text
    assert me.json()["tenant_id"] is None, "a removed member's claim must not resurface on /me"
    assert (await harness.rows(sub))[t] == ("member", "removed"), "/me must not resurrect"


async def test_suspended_tenant_membership_is_not_resolvable(harness: Harness) -> None:
    harness.mode("membership")
    sub = uuid.uuid7()
    t = await harness.tenant(status="suspended")
    await harness.member(sub, t)

    r = await harness.get("/api/probe", harness.token(sub, claim=t))

    assert r.status_code == 403
    assert r.json()["error"]["code"] == "NO_MEMBERSHIP"


async def test_super_admin_may_select_any_active_tenant(harness: Harness, caplog) -> None:
    harness.mode("membership", tenant_required=False)
    sub = uuid.uuid7()
    t, suspended = await harness.tenant(), await harness.tenant(status="suspended")
    tok = harness.token(sub, roles=["super_admin", "tenant_admin"])

    with caplog.at_level(logging.INFO, logger="w2v_orchestrator.audit"):
        r = await harness.get("/api/probe", tok, {"X-Active-Tenant-Id": str(t)})
    unknown = await harness.get("/api/probe", tok, {"X-Active-Tenant-Id": str(uuid.uuid7())})
    blocked = await harness.get("/api/probe", tok, {"X-Active-Tenant-Id": str(suspended)})
    nothing = await harness.get("/api/probe", tok)
    malformed = await harness.get("/api/probe", tok, {"X-Active-Tenant-Id": "nope"})

    assert r.status_code == 200, r.text
    assert r.json()["tenant"] == str(t)
    # Realm tenant_admin is dropped; super_admin (which implies admin) is kept.
    assert r.json()["roles"] == ["super_admin"]
    assert any(
        "SCOPE_SWITCH" in rec.getMessage() and str(t) in rec.getMessage() for rec in caplog.records
    )
    assert unknown.status_code == 404
    assert blocked.status_code == 404
    assert nothing.status_code == 403
    assert malformed.status_code == 422
    assert await harness.rows(sub) == {}, "a super_admin visit is not a membership"
    # And the admin surface accepts them in the switched tenant.
    admin = await harness.get("/api/probe/admin", tok, {"X-Active-Tenant-Id": str(t)})
    assert admin.status_code == 200


@pytest.mark.parametrize("status", ["unknown"])
async def test_refused_active_tenant_never_404s_auth_me(harness: Harness, status: str) -> None:
    """A super_admin whose stored active tenant is unknown / suspended / deleted
    must still get /api/auth/me (its roles are what the SPA uses to show the
    picker that can undo the choice). Data routes still refuse 404."""
    harness.mode("membership", tenant_required=False)
    sub = uuid.uuid7()
    own = await harness.tenant()
    await harness.member(sub, own, role="tenant_admin")
    bad = uuid.uuid7() if status == "unknown" else await harness.tenant(status=status)
    tok = harness.token(sub, roles=["super_admin", "tenant_admin"])
    hdr = {"X-Active-Tenant-Id": str(bad)}

    me = await harness.get("/api/auth/me", tok, hdr)
    data = await harness.get("/api/probe", tok, hdr)

    assert me.status_code == 200, me.text
    body = me.json()
    assert "super_admin" in body["roles"], "the caller must still look super_admin"
    assert body["active_tenant_refused"] == str(bad)
    # Resolved as if no tenant were requested: the caller's own membership.
    assert body["tenant_id"] == str(own)
    assert data.status_code == 404
    assert data.json()["error"]["code"] == "TENANT_NOT_FOUND"
    assert data.json()["error"]["details"] == {"active_tenant": str(bad)}


async def test_refused_active_tenant_without_membership_me_is_tenantless(
    harness: Harness,
) -> None:
    harness.mode("membership", tenant_required=False)
    sub = uuid.uuid7()
    bad = await harness.tenant(status="suspended")
    good = await harness.tenant()
    tok = harness.token(sub, roles=["super_admin"])

    me = await harness.get("/api/auth/me", tok, {"X-Active-Tenant-Id": str(bad)})
    ok = await harness.get("/api/auth/me", tok, {"X-Active-Tenant-Id": str(good)})

    assert me.status_code == 200, me.text
    assert me.json()["tenant_id"] is None
    assert me.json()["roles"] == ["super_admin"]
    assert me.json()["active_tenant_refused"] == str(bad)
    assert ok.status_code == 200, ok.text
    assert ok.json()["tenant_id"] == str(good)
    assert ok.json()["active_tenant_refused"] is None


async def test_admin_and_known_gates_read_membership_not_realm_roles(harness: Harness) -> None:
    """B21: realm tenant_admin (fr-wide) must not make you admin of a w2v tenant,
    and a member's (registered) tenant is 'known' with no project or licence."""
    harness.mode("membership", tenant_required=False)
    member_sub, admin_sub = uuid.uuid7(), uuid.uuid7()
    t = await harness.tenant()  # registry row only: no projects, no licence lookup allowed
    await harness.member(member_sub, t, "member")
    await harness.member(admin_sub, t, "tenant_admin")

    realm_admin_but_member = harness.token(member_sub, roles=["tenant_admin"])
    plain_member_is_tenant_admin = harness.token(admin_sub, roles=["member"])

    assert (await harness.get("/api/probe/admin", realm_admin_but_member)).status_code == 403
    ok = await harness.get("/api/probe/admin", plain_member_is_tenant_admin)
    assert ok.status_code == 200 and ok.json()["tenant"] == str(t)
    known = await harness.get("/api/probe/known", realm_admin_but_member)
    assert known.status_code == 200, known.text


async def test_membership_in_an_unregistered_tenant_is_not_resolvable(harness: Harness) -> None:
    """A membership row alone (e.g. lazily created from a bare fr claim) must not
    make an unprovisioned tenant reachable: resolution skips require_known_tenant,
    so the tenants registry row is the provisioning proof."""
    harness.mode("membership", tenant_required=False)
    sub = uuid.uuid7()
    unregistered = uuid.uuid7()  # NO tenants row
    await harness.member(sub, unregistered)
    tok = harness.token(sub)

    default = await harness.get("/api/probe/known", tok)
    requested = await harness.get(
        "/api/probe/known", tok, {"X-Active-Tenant-Id": str(unregistered)}
    )

    for r in (default, requested):
        assert r.status_code == 403, r.text
        assert r.json()["error"]["code"] == "NO_MEMBERSHIP"


# ─── A2 identity seam: the realm names no tenant and no tenant role ──────


async def test_realm_tenant_admin_with_a_claim_but_no_membership_is_refused(
    harness: Harness,
) -> None:
    """CP pattern: Keycloak authenticates, w2v authorizes. A realm tenant_admin
    whose token's tenant_id claim names a REGISTERED w2v tenant, but who holds no
    tenant_members row there, is refused on every gate — whether they rely on
    the claim or name the tenant explicitly. Claim mode serves this caller 200
    (the control below), so the 403s are the membership authority's."""
    sub = uuid.uuid7()
    t = await harness.tenant()
    tok = harness.token(sub, claim=t, roles=["tenant_admin", "member"])

    harness.mode("claim")
    control = await harness.get("/api/probe/admin", tok)
    assert control.status_code == 200, "control: claim mode trusts the claim + realm role"

    harness.mode("membership")
    for path in ("/api/probe", "/api/probe/admin", "/api/probe/known"):
        for headers in (None, {"X-Active-Tenant-Id": str(t)}):
            r = await harness.get(path, tok, headers)
            assert r.status_code == 403, (path, headers, r.text)
            assert r.json()["error"]["code"] == "NO_MEMBERSHIP"
    me = await harness.get("/api/auth/me", tok)
    assert me.status_code == 200, me.text
    assert me.json()["tenant_id"] is None, "the claim must not surface as the /me tenant"
    assert me.json()["roles"] == [], "realm tenant_admin/member are not w2v roles"
    assert await harness.rows(sub) == {}, "/me in membership mode never mints a member"


async def test_member_tenant_ids_claim_is_ignored(harness: Harness) -> None:
    """A ``member_tenant_ids`` claim (fr's multi-tenant hint) grants nothing:
    only tenant_members rows are switchable, and the claim's list is neither a
    default nor an allowed switch target."""
    harness.mode("membership")
    sub = uuid.uuid7()
    own, listed = await harness.tenant(), await harness.tenant()
    await harness.member(sub, own)
    tok = harness.token(
        sub,
        claim=listed,
        roles=["tenant_admin"],
        extra={"member_tenant_ids": [str(listed), str(own)]},
    )

    default = await harness.get("/api/probe", tok)
    switched = await harness.get("/api/probe", tok, {"X-Active-Tenant-Id": str(listed)})
    admin = await harness.get("/api/probe/admin", tok)

    assert default.status_code == 200, default.text
    assert default.json()["tenant"] == str(own)
    assert default.json()["roles"] == ["member"]
    assert switched.status_code == 403
    assert switched.json()["error"]["code"] == "NO_MEMBERSHIP"
    assert admin.status_code == 403, "realm tenant_admin must not make a member an admin"


async def test_super_admin_without_membership_acts_only_in_the_tenant_it_names(
    harness: Harness, caplog
) -> None:
    """super_admin is the ONE honoured realm role: with no membership it may act
    in any provisioned tenant it names (audited), but its token's tenant claim is
    never a fallback — no named tenant is a 403, not "the claim tenant"."""
    harness.mode("membership")
    sub = uuid.uuid7()
    claim_tenant, target = await harness.tenant(), await harness.tenant()
    tok = harness.token(sub, claim=claim_tenant, roles=["super_admin", "tenant_admin"])

    with caplog.at_level(logging.INFO, logger="w2v_orchestrator.audit"):
        r = await harness.get("/api/probe", tok, {"X-Active-Tenant-Id": str(target)})
    admin = await harness.get("/api/probe/admin", tok, {"X-Active-Tenant-Id": str(target)})
    no_target = await harness.get("/api/probe", tok)

    assert r.status_code == 200, r.text
    assert r.json()["tenant"] == str(target)
    assert r.json()["roles"] == ["super_admin"]
    assert admin.status_code == 200 and admin.json()["tenant"] == str(target)
    assert any(
        "SCOPE_SWITCH" in rec.getMessage() and str(target) in rec.getMessage()
        for rec in caplog.records
    )
    assert no_target.status_code == 403, "the claim tenant is never a super_admin default"
    assert no_target.json()["error"]["code"] == "NO_MEMBERSHIP"
    assert await harness.rows(sub) == {}


async def test_membership_mode_does_not_require_the_claim_at_the_auth_edge(
    harness: Harness,
) -> None:
    """A member whose token carries NO tenant_id claim (an invitee: provisioning
    sets no realm tenant attribute) is 401'd by w2v_common while the claim is
    required; ``install_tenancy_middleware`` aligns the requirement for
    membership mode so the DB, not the claim, decides."""
    harness.mode("membership", tenant_required=True)
    sub = uuid.uuid7()
    t = await harness.tenant()
    await harness.member(sub, t)
    tok = harness.token(sub)  # no tenant_id claim at all

    before = await harness.get("/api/probe", tok)
    assert before.status_code == 401, "control: required claim + no claim = 401 at the edge"

    assert align_jwt_tenant_requirement("membership") is True
    after = await harness.get("/api/probe", tok)
    assert after.status_code == 200, after.text
    assert after.json()["tenant"] == str(t)
    assert align_jwt_tenant_requirement("membership") is False, "idempotent"


async def test_claim_mode_keeps_the_claim_required(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_JWT_TENANT_REQUIRED", "true")
    assert align_jwt_tenant_requirement("claim") is False
    assert os.environ["AUTH_JWT_TENANT_REQUIRED"] == "true"


async def test_shadow_mode_relaxes_the_absent_claim_too(monkeypatch) -> None:
    """Shadow must SEE claimless identities (users that exist only in w2v's DB),
    not have w2v_common 401 them before the comparison runs."""
    monkeypatch.setenv("AUTH_JWT_TENANT_REQUIRED", "true")
    assert align_jwt_tenant_requirement("shadow") is True
    assert os.environ["AUTH_JWT_TENANT_REQUIRED"] == "false"


async def test_membership_mode_claimless_token_gets_its_membership_tenant_and_role(
    harness: Harness,
) -> None:
    harness.mode("membership")
    align_jwt_tenant_requirement("membership")
    sub = uuid.uuid7()
    t = await harness.tenant()
    await harness.member(sub, t, role="tenant_admin")

    r = await harness.get("/api/probe", harness.token(sub))

    assert r.status_code == 200, r.text
    assert r.json()["tenant"] == str(t)
    assert r.json()["roles"] == ["tenant_admin"]
    assert r.json()["source"] == "membership"


async def test_shadow_claimless_member_is_served_by_membership_and_counted(
    harness: Harness,
) -> None:
    """Live 2026-10-15: a portal user with no tenant attribute was 401
    ``JWT has no valid tenant`` on /api/auth/me under shadow. Now: served by its
    w2v membership, counted ``claim_missing`` (not a mismatch, not an error)."""
    harness.mode("shadow")
    sub = uuid.uuid7()
    t = await harness.tenant()
    await harness.member(sub, t, role="tenant_admin")
    tok = harness.token(sub)  # no tenant_id claim

    control = await harness.get("/api/probe", tok)
    assert control.status_code == 401, "control: claim still required before alignment"

    assert align_jwt_tenant_requirement("shadow") is True
    mismatch_before = {
        reason: _mismatches(reason)
        for reason in ("no_membership", "not_a_member", "tenant_differs", "error")
    }
    checked_before = _checked()
    r = await harness.get("/api/probe", tok)
    me = await harness.get("/api/auth/me", tok)

    assert r.status_code == 200, r.text
    assert r.json()["tenant"] == str(t)
    assert r.json()["roles"] == ["tenant_admin"]
    assert r.json()["source"] == "membership"
    assert me.status_code == 200, me.text
    assert _checked_delta(checked_before) == {"claim_missing": 2.0}
    assert {reason: _mismatches(reason) for reason in mismatch_before} == mismatch_before


async def test_shadow_claimless_non_member_is_403_not_200(harness: Harness) -> None:
    harness.mode("shadow")
    align_jwt_tenant_requirement("shadow")
    sub = uuid.uuid7()
    await harness.tenant()  # a tenant exists; this subject is not in it

    r = await harness.get("/api/probe", harness.token(sub))

    assert r.status_code == 403, r.text
    assert r.json()["error"]["code"] == "NO_MEMBERSHIP"
    assert await harness.rows(sub) == {}, "a claimless caller is never lazily joined"


async def test_claim_mode_claimless_token_is_401_as_before(harness: Harness) -> None:
    harness.mode("claim")  # AUTH_JWT_TENANT_REQUIRED left at its default (required)
    align_jwt_tenant_requirement("claim")
    sub = uuid.uuid7()
    await harness.member(sub, await harness.tenant())

    r = await harness.get("/api/probe", harness.token(sub))

    assert r.status_code == 401
    assert r.json()["error"]["code"] == "JWT_INVALID"


async def test_super_admin_non_member_read_writes_a_cross_tenant_audit_row(
    harness: Harness, db_session_factory
) -> None:
    """Membership mode rewrites state.tenant_id to the selected tenant, so the
    audit middleware cannot infer the switch by comparison — TenancyMiddleware
    must flag it. A super_admin GET into a non-member tenant is durably recorded;
    the same super_admin's GET in a tenant they ARE a member of is not."""
    harness.mode("membership", tenant_required=False)
    probe = APIRouter(prefix="/api/probe")

    @probe.get("")
    async def whoami(tenant_id: TenantIdDep) -> dict[str, str]:
        return {"tenant": str(tenant_id)}

    app = create_app(
        get_settings(),
        routers=[probe],
        cors_enabled=False,
        extra_middleware=[(AuditMiddleware, {})],
    )
    install_tenancy_middleware(app)
    app.state.session_factory = db_session_factory
    harness.app = app

    sub = uuid.uuid7()
    home, visited = await harness.tenant(), await harness.tenant()
    await harness.member(sub, home)
    tok = harness.token(sub, roles=["super_admin"])

    async def audit_rows() -> list[RequestAudit]:
        async with db_session_factory() as s:
            result = await s.execute(
                select(RequestAudit).where(RequestAudit.actor_subject_id == sub)
            )
            return list(result.scalars().all())

    own = await harness.get("/api/probe", tok, {"X-Active-Tenant-Id": str(home)})
    assert own.status_code == 200, own.text
    assert await audit_rows() == [], "a GET in a tenant you are a member of is not cross-tenant"

    r = await harness.get("/api/probe", tok, {"X-Active-Tenant-Id": str(visited)})

    assert r.status_code == 200, r.text
    rows = await audit_rows()
    assert len(rows) == 1
    assert (rows[0].event, rows[0].cross_tenant, rows[0].effective_tenant_id) == (
        "read",
        True,
        visited,
    )


async def test_outbound_effective_tenant_is_the_resolved_tenant(harness: Harness) -> None:
    """The SA-bearer hop from a membership-resolved request names the MEMBERSHIP
    tenant — for a claim-less token (AUTH_JWT_TENANT_REQUIRED=false) and for a
    token whose claim names a different (fr) tenant."""
    harness.mode("membership", tenant_required=False)
    sub, fr_tenant = uuid.uuid7(), uuid.uuid7()
    t = await harness.tenant()
    await harness.member(sub, t)

    claimless = await harness.get("/api/probe/outbound", harness.token(sub))
    with_claim = await harness.get("/api/probe/outbound", harness.token(sub, claim=fr_tenant))

    assert claimless.status_code == 200, claimless.text
    assert with_claim.status_code == 200, with_claim.text
    assert len(harness.outbound) == 2
    for sent in harness.outbound:
        assert sent.headers["X-Tenant-Id"] == f"Bearer {_SA_TOKEN}", "SA credential preserved"
        assert sent.headers["X-Effective-Tenant-Id"] == str(t)


def _req_unattributed(reason: str) -> float:
    return REGISTRY.get_sample_value(_REQ_UNATTRIBUTED, {"reason": reason}) or 0.0


async def test_membership_request_bills_the_resolved_tenants_own_licence(
    harness: Harness,
) -> None:
    """Request-time cost attribution on the human lane: the resolved tenant's
    ``licence_ref`` is bound as ``license_id`` and rides every outbound hop as
    ``X-License-Id`` — the same resolution a run snapshot uses. The claim (an fr
    tenant) and the SA bearer's licence are never what bills."""
    harness.mode("membership", tenant_required=False)
    sub, fr_tenant, ref = uuid.uuid7(), uuid.uuid7(), uuid.uuid7()
    t = await harness.tenant(licence_ref=ref)
    await harness.member(sub, t)
    before = {r: _req_unattributed(r) for r in ("no_tenant_row", "no_licence_ref")}

    probe = await harness.get("/api/probe", harness.token(sub, claim=fr_tenant))
    hop = await harness.get("/api/probe/outbound", harness.token(sub))

    assert probe.status_code == 200, probe.text
    assert probe.json()["license"] == str(ref)
    assert hop.status_code == 200, hop.text
    assert [r.headers.get("X-License-Id") for r in harness.outbound] == [str(ref)]
    assert {r: _req_unattributed(r) for r in before} == before


async def test_membership_request_without_a_licence_ref_is_unattributed_loudly(
    harness: Harness, caplog
) -> None:
    """No ``licence_ref`` → NO licence (empty, never the nil UUID, never an
    umbrella), logged at ERROR and counted; the request itself is still served."""
    harness.mode("membership", tenant_required=False)
    sub = uuid.uuid7()
    t = await harness.tenant(licence_ref=uuid.UUID(int=0))  # nil is not a licence
    await harness.member(sub, t)
    before = _req_unattributed("no_licence_ref")

    with caplog.at_level(logging.ERROR, logger="w2v_orchestrator.request_licence"):
        probe = await harness.get("/api/probe", harness.token(sub))
        hop = await harness.get("/api/probe/outbound", harness.token(sub))

    assert probe.status_code == 200, probe.text
    assert probe.json()["license"] == ""
    assert hop.status_code == 200, hop.text
    assert [r.headers.get("X-License-Id") for r in harness.outbound] == [None]
    assert _req_unattributed("no_licence_ref") - before == 2
    assert sum(str(t) in r.getMessage() for r in caplog.records if r.levelno == logging.ERROR) == 2


async def test_membership_licence_read_failure_is_counted_not_guessed(
    harness: Harness, monkeypatch, caplog
) -> None:
    """A failed licence read binds NO licence (counted ``read_failed``, ERROR) —
    and does not refuse a request whose membership already resolved."""
    from src.auth import tenancy_middleware as tm

    async def _boom(_session: object, _tenant: object) -> str:
        raise RuntimeError("db down")

    monkeypatch.setattr(tm, "resolve_request_licence", _boom)
    harness.mode("membership", tenant_required=False)
    sub = uuid.uuid7()
    t = await harness.tenant(licence_ref=uuid.uuid7())
    await harness.member(sub, t)
    before = _req_unattributed("read_failed")

    with caplog.at_level(logging.ERROR, logger="w2v_orchestrator.request_licence"):
        r = await harness.get("/api/probe", harness.token(sub))

    assert r.status_code == 200, r.text
    assert r.json()["license"] == ""
    assert _req_unattributed("read_failed") - before == 1
    assert any(rec.levelno == logging.ERROR for rec in caplog.records)


async def test_without_the_middleware_the_sa_hop_fails_closed(
    harness: Harness, db_session_factory
) -> None:
    """Control for the test above: the same claim-less request through an app
    WITHOUT TenancyMiddleware reaches the route tenant-less, and w2v_common
    0.90.0 refuses the SA-bearer hop rather than act as the SA tenant."""
    harness.mode("membership", tenant_required=False)
    sub = uuid.uuid7()
    await harness.member(sub, await harness.tenant())
    sink: list[httpx.Request] = []
    bare = _build_app(db_session_factory, sink, tenancy=False)
    transport = httpx.ASGITransport(app=bare, client=_PUBLIC_PEER)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        with pytest.raises(MissingTenantError):
            await c.get(
                "/api/probe/outbound", headers={"Authorization": f"Bearer {harness.token(sub)}"}
            )
    assert sink == []


# ─── claim + shadow modes ────────────────────────────────────────────────


async def test_claim_mode_is_unchanged(harness: Harness) -> None:
    harness.mode("claim")
    sub, claim = uuid.uuid7(), uuid.uuid7()
    await harness.member(sub, await harness.tenant())  # a membership elsewhere is ignored

    r = await harness.get("/api/probe", harness.token(sub, claim=claim, roles=["tenant_admin"]))

    assert r.status_code == 200
    assert r.json()["tenant"] == str(claim)
    assert r.json()["source"] == "jwt-claim"
    assert r.json()["roles"] == ["tenant_admin"]


async def test_claim_mode_never_serves_a_claimless_token(harness: Harness) -> None:
    harness.mode("claim", tenant_required=False)
    sub = uuid.uuid7()
    await harness.member(sub, await harness.tenant())

    r = await harness.get("/api/probe", harness.token(sub))

    assert r.status_code == 401
    assert r.json()["error"]["code"] == "JWT_INVALID"


async def test_shadow_serves_claim_counts_mismatch_and_lazily_upserts(
    harness: Harness, caplog
) -> None:
    harness.mode("shadow")
    sub = uuid.uuid7()
    claim, other = await harness.tenant(), await harness.tenant()
    await harness.member(sub, other, seen_minutes_ago=5)
    tok = harness.token(sub, claim=claim, roles=["tenant_admin"])
    before = _mismatches("tenant_differs")
    checked_before = _checked()

    with caplog.at_level(logging.WARNING, logger="src.auth.tenancy_middleware"):
        first = await harness.get("/api/probe", tok)
    after_first = _mismatches("tenant_differs")
    assert _checked_delta(checked_before) == {"mismatch": 1.0}
    checked_mid = _checked()
    second = await harness.get("/api/probe", tok)
    assert _checked_delta(checked_mid) == {"match": 1.0}

    assert first.status_code == 200
    assert first.json()["tenant"] == str(claim), "shadow serves by claim"
    assert first.json()["source"] == "jwt-claim"
    assert after_first == before + 1
    assert any("TENANCY_SHADOW_MISMATCH" in rec.getMessage() for rec in caplog.records)
    # Lazily created for the claim tenant, role from the realm (0072 mapping).
    assert (await harness.rows(sub))[claim] == ("tenant_admin", "active")
    # Now the membership answer agrees (claim tenant is the most recently used).
    assert second.json()["tenant"] == str(claim)
    assert _mismatches("tenant_differs") == after_first


async def test_shadow_counts_no_membership_and_never_resurrects_a_removed_member(
    harness: Harness,
) -> None:
    harness.mode("shadow")
    sub = uuid.uuid7()
    claim = await harness.tenant()
    await harness.member(sub, claim)
    async with harness.factory() as s:
        await MembersRepository(s).remove_member(sub, claim)
        await s.commit()
    before = _mismatches("no_membership")
    checked_before = _checked()

    r = await harness.get("/api/probe", harness.token(sub, claim=claim))

    assert r.status_code == 200 and r.json()["tenant"] == str(claim)
    assert _mismatches("no_membership") == before + 1
    assert _checked_delta(checked_before) == {"mismatch": 1.0}
    assert (await harness.rows(sub))[claim] == ("member", "removed")


async def test_shadow_counts_a_match_once_and_no_mismatch(harness: Harness) -> None:
    """The denominator: an agreeing comparison is counted, so 0 mismatches is
    distinguishable from no shadow traffic at all."""
    harness.mode("shadow")
    sub = uuid.uuid7()
    claim = await harness.tenant()
    await harness.member(sub, claim)
    mismatch_before = {
        reason: _mismatches(reason)
        for reason in ("no_membership", "not_a_member", "tenant_differs", "error")
    }
    checked_before = _checked()

    r = await harness.get("/api/probe", harness.token(sub, claim=claim))

    assert r.status_code == 200 and r.json()["tenant"] == str(claim)
    assert _checked_delta(checked_before) == {"match": 1.0}
    assert {reason: _mismatches(reason) for reason in mismatch_before} == mismatch_before


async def test_shadow_check_failure_counts_error_once(harness: Harness) -> None:
    harness.mode("shadow")
    sub = uuid.uuid7()
    claim = await harness.tenant()
    checked_before = _checked()
    error_before = _mismatches("error")

    async def _boom(self: MembersRepository, subject_id: uuid.UUID) -> list[object]:
        raise RuntimeError("db down")

    harness.monkeypatch.setattr(MembersRepository, "list_resolvable_memberships", _boom)
    r = await harness.get("/api/probe", harness.token(sub, claim=claim))

    assert r.status_code == 200 and r.json()["tenant"] == str(claim), "shadow never breaks"
    assert _checked_delta(checked_before) == {"error": 1.0}
    assert _mismatches("error") == error_before + 1


async def test_shadow_upsert_failure_after_verdict_counts_the_verdict_not_an_error(
    harness: Harness,
) -> None:
    """Verdict reached (no_membership → mismatch), THEN the lazy upsert fails: the
    comparison is counted once under its verdict and is NOT also an error mismatch
    — else every failed upsert inflates the cutover's error rate and denominator."""
    harness.mode("shadow")
    sub = uuid.uuid7()
    claim = await harness.tenant()  # provisioned + no membership → lazy upsert runs
    checked_before = _checked()
    no_membership_before = _mismatches("no_membership")
    error_before = _mismatches("error")
    upserts: list[uuid.UUID] = []

    async def _boom(self: MembersRepository, **kwargs: Any) -> bool:
        upserts.append(kwargs["tenant_id"])
        raise RuntimeError("db down mid-upsert")

    harness.monkeypatch.setattr(MembersRepository, "ensure_member", _boom)
    r = await harness.get("/api/probe", harness.token(sub, claim=claim))

    assert r.status_code == 200 and r.json()["tenant"] == str(claim), "shadow never breaks"
    assert upserts == [claim], "the failure must come from the post-verdict upsert"
    assert _checked_delta(checked_before) == {"mismatch": 1.0}
    assert _mismatches("no_membership") == no_membership_before + 1
    assert _mismatches("error") == error_before, "an upsert failure is not a failed check"


async def test_shadow_checked_series_exist_at_zero_from_import() -> None:
    """Pre-created: a FRESH process exposes all four series at 0 before any
    comparison ran (this process's registry already holds counted samples)."""
    probe = (
        "from prometheus_client import REGISTRY\n"
        "import src.observability.metrics\n"
        f"print([REGISTRY.get_sample_value({_CHECKED!r}, {{'result': r}}) "
        "for r in ('match', 'mismatch', 'error', 'claim_missing')])"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", probe, stdout=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    assert proc.returncode == 0
    assert stdout.decode().strip() == "[0.0, 0.0, 0.0, 0.0]"


async def test_shadow_never_lazily_joins_an_unprovisioned_or_suspended_tenant(
    harness: Harness,
) -> None:
    """A bare fr claim (no tenants row) or a suspended tenant gets no lazy
    membership, and no tenants row is invented — else every fr-only portal user
    would be a resolvable member of an unprovisioned tenant after cutover."""
    harness.mode("shadow")
    sub = uuid.uuid7()
    fr_only, suspended = uuid.uuid7(), await harness.tenant(status="suspended")

    for claim in (fr_only, suspended):
        r = await harness.get("/api/probe", harness.token(sub, claim=claim))
        assert r.status_code == 200 and r.json()["tenant"] == str(claim), "shadow serves claim"

    assert await harness.rows(sub) == {}
    async with harness.factory() as s:
        assert (await s.execute(select(Tenant).where(Tenant.id == fr_only))).first() is None


async def test_shadow_super_admin_scope_switch_creates_no_membership(harness: Harness) -> None:
    harness.mode("shadow")
    sub = uuid.uuid7()
    home, visited = await harness.tenant(), await harness.tenant()
    await harness.member(sub, home)
    tok = harness.token(sub, claim=home, roles=["super_admin"])

    r = await harness.get("/api/probe", tok, {"X-Active-Tenant-Id": str(visited)})
    me = await harness.get("/api/auth/me", tok, {"X-Active-Tenant-Id": str(visited)})

    assert r.status_code == 200 and r.json()["tenant"] == str(visited)
    assert me.status_code == 200 and me.json()["tenant_id"] == str(visited)
    assert set(await harness.rows(sub)) == {home}, "a scope-switch visit is not a membership"


async def test_claim_mode_me_creates_only_for_the_own_tenant(harness: Harness) -> None:
    harness.mode("claim")
    sub = uuid.uuid7()
    home, visited = await harness.tenant(), await harness.tenant()
    tok = harness.token(sub, claim=home, roles=["super_admin"])

    assert (await harness.get("/api/auth/me", tok)).status_code == 200
    assert (
        await harness.get("/api/auth/me", tok, {"X-Active-Tenant-Id": str(visited)})
    ).status_code == 200

    assert set(await harness.rows(sub)) == {home}


# ─── tenant API keys ─────────────────────────────────────────────────────


async def _mint_key(
    harness: Harness, tenant: uuid.UUID, *, expires_at: datetime.datetime | None = None
) -> tuple[uuid.UUID, str]:
    async with harness.factory() as s:
        row, full = await ApiKeysRepository(s).create(
            tenant_id=tenant, name="ci", created_by=None, expires_at=expires_at
        )
        await s.commit()
        return row.id, full


async def test_api_key_resolves_its_tenant_as_member(harness: Harness) -> None:
    harness.mode("membership")
    t = await harness.tenant()
    key_id, full = await _mint_key(harness, t)

    r = await harness.get("/api/probe", full)
    admin = await harness.get("/api/probe/admin", full)

    assert r.status_code == 200, r.text
    assert r.json()["tenant"] == str(t)
    assert r.json()["source"] == "tenant-api-key"
    assert r.json()["roles"] == ["member"]
    assert admin.status_code == 403, "an API key is never an admin"
    async with harness.factory() as s:
        used = (
            await s.execute(select(TenantApiKey.last_used_at).where(TenantApiKey.id == key_id))
        ).scalar_one()
    assert used is not None


async def test_revoked_or_expired_api_key_is_401(harness: Harness) -> None:
    harness.mode("membership")
    t = await harness.tenant()
    key_id, revoked = await _mint_key(harness, t)
    async with harness.factory() as s:
        assert await ApiKeysRepository(s).revoke(key_id, t)
        await s.commit()
    _, expired = await _mint_key(
        harness, t, expires_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=1)
    )
    _, valid = await _mint_key(harness, t)
    tampered = valid[:-1] + ("A" if valid[-1] != "A" else "B")

    for bearer in (revoked, expired, tampered):
        r = await harness.get("/api/probe", bearer)
        assert r.status_code == 401, (bearer[:16], r.text)
        assert r.json()["error"]["code"] == "INVALID_API_KEY"


async def test_last_used_bump_failure_does_not_fail_a_valid_key(
    harness: Harness, monkeypatch, caplog
) -> None:
    """touch_last_used is bookkeeping: its failure must not turn a verified key
    into a 503, and the warning must not carry the secret."""
    harness.mode("membership")
    t = await harness.tenant()
    _, full = await _mint_key(harness, t)

    async def _boom(self: ApiKeysRepository, key_id: uuid.UUID) -> None:
        raise RuntimeError(f"db down while handling {full}")

    monkeypatch.setattr(ApiKeysRepository, "touch_last_used", _boom)
    with caplog.at_level(logging.WARNING, logger="src.auth.tenancy_middleware"):
        r = await harness.get("/api/probe", full)

    assert r.status_code == 200, r.text
    assert r.json()["tenant"] == str(t)
    warned = [rec.getMessage() for rec in caplog.records if "last_used_at" in rec.getMessage()]
    assert warned, "the failed bump must be logged"
    assert all(full not in m and full[:20] not in m for m in warned)


# ─── NO_MEMBERSHIP exemption is an EXACT match (P2 explicit invite accept) ───


@pytest.mark.parametrize(
    ("path", "exempt"),
    [("/api/auth/me", True)],
)
async def test_no_membership_exemption_is_exact(path: str, exempt: bool) -> None:
    from src.auth.tenancy_middleware import is_no_membership_exempt

    assert is_no_membership_exempt(path) is exempt
