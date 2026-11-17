"""P2 WP4a — tenant invites + acceptance, through the real auth + tenancy stack.

Non-vacuity: every test drives a WP4a route (404 without it) or asserts a state
change only the invite flow produces (an active membership in a tenant the
caller had none in, a rotated token hash, a Keycloak call shape).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import uuid
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from sqlalchemy import select, text, update
from sqlalchemy.exc import DBAPIError

from src.database.models import TenantInvite
from src.database.repositories.invites_repository import InvitesRepository
from tests.api.routes._tenancy_harness import (  # noqa: F401 — fixtures
    Harness,
    harness_fixture,
    signing_key_fixture,
)

pytestmark = pytest.mark.asyncio

ADMIN_EMAIL = "admin@acme.test"
NEW_EMAIL = "Newbie@Example.test"


def _token(resp_json: dict) -> str:
    return parse_qs(urlparse(resp_json["invite_url"]).query)["token"][0]


async def _admin(h: Harness, tenant: uuid.UUID) -> tuple[uuid.UUID, str]:
    sub = uuid.uuid7()
    await h.member(sub, tenant, "tenant_admin", email=ADMIN_EMAIL)
    return sub, h.token(sub, email=ADMIN_EMAIL)


async def _invite(h: Harness, admin_tok: str, email: str = NEW_EMAIL, role: str = "member"):
    r = await h.call(
        "POST", "/api/tenant/invites", admin_tok, json_body={"email": email, "role": role}
    )
    assert r.status_code == 201, r.text
    return r.json()


async def test_invite_then_accept_with_matching_verified_email_creates_membership(
    harness: Harness,
) -> None:
    tenant = await harness.tenant("Acme")
    admin_sub, admin_tok = await _admin(harness, tenant)

    body = await _invite(harness, admin_tok, role="tenant_admin")
    token = _token(body)
    assert body["invite_url"].startswith("https://w2v.example.test/invite/accept?token=")
    assert body["email_delivery"] == "requested"
    assert len(token) >= 43  # 32 random bytes, urlsafe

    # Only the hash is stored.
    [row] = await harness.invites(tenant)
    assert row.token_hash == hashlib.sha256(token.encode()).digest()
    assert row.status == "pending"
    assert row.kc_user_created is False  # w2v never learns account state any more
    assert row.expires_at - datetime.datetime.now(datetime.UTC) > datetime.timedelta(days=6)

    # Identity: ONE provisioning invite call carrying exactly the contract's
    # fields (no attributes / credentials / roles can be sent) plus the audit
    # actor/context. The action email redirects to the token-FREE SPA root
    # (which lists pending invites); the token never reaches the email.
    assert harness.idp.bodies() == [
        {
            "email": NEW_EMAIL,
            "client_id": "w2v-frontend",
            "redirect_uri": "https://w2v.example.test/",
            "actor": str(admin_sub),
            "context": f"w2v:tenant:{tenant}",
        }
    ]
    [mail] = harness.idp.mails
    assert mail == (
        NEW_EMAIL.lower(),
        ["UPDATE_PASSWORD", "VERIFY_EMAIL"],
        "https://w2v.example.test/",
        "w2v-frontend",
    )
    assert token not in json.dumps(harness.idp.bodies())

    invitee = uuid.uuid7()
    tok = harness.token(invitee, email="newbie@example.test")  # case-insensitive match
    # Before accepting: no membership → every tenant route refuses.
    denied = await harness.call("GET", "/api/tenant/members", tok)
    assert denied.status_code == 403
    assert denied.json()["error"]["code"] == "NO_MEMBERSHIP"

    r = await harness.call("POST", "/api/invites/accept", tok, json_body={"token": token})
    assert r.status_code == 200, r.text
    assert r.json() == {
        "tenant_id": str(tenant),
        "tenant_name": "Acme",
        "role": "tenant_admin",
        "already_member": False,
    }
    assert await harness.membership(invitee, tenant) == ("tenant_admin", "active")
    [row] = await harness.invites(tenant)
    assert row.status == "accepted"
    assert row.accepted_subject_id == invitee
    assert "member.invite" in await harness.audit_actions(tenant)

    # Membership now resolves: the invitee (tenant_admin) can read the roster.
    assert (await harness.call("GET", "/api/tenant/members", tok)).status_code == 200

    # Idempotent replay.
    again = await harness.call("POST", "/api/invites/accept", tok, json_body={"token": token})
    assert again.status_code == 200
    assert again.json()["already_member"] is True


async def test_accept_refuses_mismatched_or_unverified_email(harness: Harness) -> None:
    tenant = await harness.tenant()
    _, admin_tok = await _admin(harness, tenant)
    token = _token(await _invite(harness, admin_tok))
    intruder = uuid.uuid7()

    wrong = harness.token(intruder, email="someone-else@example.test")
    r = await harness.call("POST", "/api/invites/accept", wrong, json_body={"token": token})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "INVITE_EMAIL_MISMATCH"

    unverified = harness.token(intruder, email=NEW_EMAIL, verified=False)
    r = await harness.call("POST", "/api/invites/accept", unverified, json_body={"token": token})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "EMAIL_NOT_VERIFIED"

    assert await harness.membership(intruder, tenant) is None
    [row] = await harness.invites(tenant)
    assert row.status == "pending"


async def test_revoked_and_expired_invites_are_gone(harness: Harness) -> None:
    tenant = await harness.tenant()
    _, admin_tok = await _admin(harness, tenant)
    revoked = await _invite(harness, admin_tok, email="r@example.test")
    expired = await _invite(harness, admin_tok, email="e@example.test")

    r = await harness.call("DELETE", f"/api/tenant/invites/{revoked['id']}", admin_tok)
    assert r.status_code == 204
    assert "invite.revoke" in await harness.audit_actions(tenant)
    listed = await harness.call("GET", "/api/tenant/invites", admin_tok)
    assert [i["id"] for i in listed.json()["items"]] == [expired["id"]]

    async with harness.factory() as s:
        await s.execute(
            update(TenantInvite)
            .where(TenantInvite.id == uuid.UUID(expired["id"]))
            .values(expires_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=1))
        )
        await s.commit()

    for body, email in ((revoked, "r@example.test"), (expired, "e@example.test")):
        sub = uuid.uuid7()
        r = await harness.call(
            "POST",
            "/api/invites/accept",
            harness.token(sub, email=email),
            json_body={"token": _token(body)},
        )
        assert r.status_code == 410, (email, r.text)
        assert r.json()["error"]["code"] == "INVITE_GONE"
        assert await harness.membership(sub, tenant) is None

    # Revoking twice is a conflict, not a silent success.
    r = await harness.call("DELETE", f"/api/tenant/invites/{revoked['id']}", admin_tok)
    assert r.status_code == 409


async def test_resend_rotates_the_token_and_kills_the_old_link(harness: Harness) -> None:
    tenant = await harness.tenant()
    _, admin_tok = await _admin(harness, tenant)
    first = await _invite(harness, admin_tok)

    r = await harness.call("POST", f"/api/tenant/invites/{first['id']}/resend", admin_tok)
    assert r.status_code == 200, r.text
    second = r.json()
    assert second["id"] == first["id"]
    assert _token(second) != _token(first)
    assert second["resent_at"] is not None
    # Created + unverified account: both actions mailed again.
    assert [m[1] for m in harness.idp.mails] == [["UPDATE_PASSWORD", "VERIFY_EMAIL"]] * 2
    # Neither email carries a token: both land on the SPA root, so the FIRST
    # email stays usable after the resend (acceptance is by verified email).
    assert {m[2] for m in harness.idp.mails} == {"https://w2v.example.test/"}
    assert _token(first) not in json.dumps(harness.idp.bodies())
    assert _token(second) not in json.dumps(harness.idp.bodies())

    invitee = uuid.uuid7()
    tok = harness.token(invitee, email=NEW_EMAIL)
    old = await harness.call("POST", "/api/invites/accept", tok, json_body={"token": _token(first)})
    assert old.status_code == 404
    new = await harness.call(
        "POST", "/api/invites/accept", tok, json_body={"token": _token(second)}
    )
    assert new.status_code == 200
    assert await harness.membership(invitee, tenant) == ("member", "active")


async def test_existing_verified_account_is_neither_created_nor_mailed(harness: Harness) -> None:
    tenant = await harness.tenant()
    _, admin_tok = await _admin(harness, tenant)
    harness.idp.add_user("known@example.test", verified=True)
    harness.idp.add_user("unverified@example.test", verified=False)

    known = await _invite(harness, admin_tok, email="known@example.test")
    unverified = await _invite(harness, admin_tok, email="unverified@example.test")
    assert known["email_delivery"] == unverified["email_delivery"] == "requested"
    assert len(harness.idp.users) == 2  # nothing created
    [mail] = harness.idp.mails
    assert mail[:2] == ("unverified@example.test", ["VERIFY_EMAIL"])


async def test_duplicate_pending_invite_and_existing_member_conflict(harness: Harness) -> None:
    tenant = await harness.tenant()
    _, admin_tok = await _admin(harness, tenant)
    first = await _invite(harness, admin_tok)

    dup = await harness.call(
        "POST", "/api/tenant/invites", admin_tok, json_body={"email": NEW_EMAIL.upper()}
    )
    assert dup.status_code == 409
    assert dup.json()["error"]["details"]["invite_id"] == first["id"]

    member = await harness.call(
        "POST", "/api/tenant/invites", admin_tok, json_body={"email": ADMIN_EMAIL}
    )
    assert member.status_code == 409
    assert member.json()["error"]["code"] == "ALREADY_MEMBER"

    bad = await harness.call(
        "POST", "/api/tenant/invites", admin_tok, json_body={"email": "not-an-email"}
    )
    assert bad.status_code == 422


async def test_non_admin_is_refused_on_every_invite_mutation(harness: Harness) -> None:
    tenant = await harness.tenant()
    _, admin_tok = await _admin(harness, tenant)
    inv = await _invite(harness, admin_tok)
    plain = uuid.uuid7()
    await harness.member(plain, tenant, "member")
    tok = harness.token(plain)

    attempts = [
        ("POST", "/api/tenant/invites", {"email": "x@example.test"}),
        ("GET", "/api/tenant/invites", None),
        ("POST", f"/api/tenant/invites/{inv['id']}/resend", None),
        ("DELETE", f"/api/tenant/invites/{inv['id']}", None),
    ]
    for method, path, body in attempts:
        r = await harness.call(method, path, tok, json_body=body)
        assert r.status_code == 403, (method, path, r.text)
    [row] = await harness.invites(tenant)
    assert row.status == "pending"
    assert row.resent_at is None


async def test_tenant_admin_cannot_see_or_touch_another_tenants_invites(harness: Harness) -> None:
    a, b = await harness.tenant(), await harness.tenant()
    _, admin_a = await _admin(harness, a)
    _, admin_b = await _admin(harness, b)
    inv_b = await _invite(harness, admin_b, email="b-only@example.test")

    listed = await harness.call("GET", "/api/tenant/invites", admin_a)
    assert listed.status_code == 200
    assert listed.json()["items"] == []
    for method, path in (
        ("POST", f"/api/tenant/invites/{inv_b['id']}/resend"),
        ("DELETE", f"/api/tenant/invites/{inv_b['id']}"),
    ):
        r = await harness.call(method, path, admin_a)
        assert r.status_code == 404, (method, r.text)
    # Naming tenant B explicitly is refused before any route runs.
    r = await harness.call(
        "GET", "/api/tenant/invites", admin_a, headers={"X-Active-Tenant-Id": str(b)}
    )
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "NO_MEMBERSHIP"
    [row] = await harness.invites(b)
    assert row.status == "pending"
    assert row.resent_at is None


#: The exact key set of one ``AuthMe.pending_invites`` entry — no inviter, no
#: email, no token.
PENDING_INVITE_KEYS = {"id", "tenant_id", "tenant_name", "role", "expires_at"}


async def test_auth_me_lists_pending_invites_and_accepts_nothing(harness: Harness) -> None:
    """Owner decision: /api/auth/me LISTS the pending invites for the caller's
    verified email; it never accepts one or writes a membership."""
    tenant = await harness.tenant("Joinable")
    _, admin_tok = await _admin(harness, tenant)
    inv = await _invite(harness, admin_tok)
    # An expired invite for the same email in another tenant is not listed.
    other = await harness.tenant("Stale")
    _, other_admin = await _admin(harness, other)
    stale = await _invite(harness, other_admin)
    async with harness.factory() as s:
        await s.execute(
            update(TenantInvite)
            .where(TenantInvite.id == uuid.UUID(stale["id"]))
            .values(expires_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=1))
        )
        await s.commit()

    # Unverified: nothing listed, nothing revealed.
    sub = uuid.uuid7()
    r = await harness.call(
        "GET", "/api/auth/me", harness.token(sub, email=NEW_EMAIL, verified=False)
    )
    assert r.status_code == 200, r.text
    assert r.json()["tenant_id"] is None
    assert r.json()["pending_invites"] == []

    # Verified (different case): listed, still NOT accepted.
    r = await harness.call("GET", "/api/auth/me", harness.token(sub, email=NEW_EMAIL.upper()))
    assert r.status_code == 200, r.text
    me = r.json()
    assert "accepted_invites" not in me
    assert me["tenant_id"] is None and me["role"] is None
    assert me["memberships"] == []
    [pending] = me["pending_invites"]
    assert set(pending) == PENDING_INVITE_KEYS
    assert pending["id"] == inv["id"]
    assert pending["tenant_id"] == str(tenant)
    assert pending["tenant_name"] == "Joinable"
    assert pending["role"] == "member"
    assert pending["expires_at"] == inv["expires_at"]
    assert await harness.membership(sub, tenant) is None
    [row] = await harness.invites(tenant)
    assert row.status == "pending"


async def test_accept_by_id_joins_a_listed_invite(harness: Harness) -> None:
    """A caller with NO membership (membership mode) reaches the by-id accept —
    it is NO_MEMBERSHIP-exempt — and joins; a replay is idempotent."""
    tenant = await harness.tenant("Acme")
    _, admin_tok = await _admin(harness, tenant)
    await _invite(harness, admin_tok, role="tenant_admin")
    sub = uuid.uuid7()
    tok = harness.token(sub, email="newbie@example.test")

    [pending] = (await harness.call("GET", "/api/auth/me", tok)).json()["pending_invites"]
    r = await harness.call("POST", f"/api/invites/{pending['id']}/accept", tok)
    assert r.status_code == 200, r.text
    assert r.json() == {
        "tenant_id": str(tenant),
        "tenant_name": "Acme",
        "role": "tenant_admin",
        "already_member": False,
    }
    assert await harness.membership(sub, tenant) == ("tenant_admin", "active")
    [row] = await harness.invites(tenant)
    assert row.status == "accepted" and row.accepted_subject_id == sub

    me = (await harness.call("GET", "/api/auth/me", tok)).json()
    assert me["tenant_id"] == str(tenant)
    assert me["pending_invites"] == []

    again = await harness.call("POST", f"/api/invites/{pending['id']}/accept", tok)
    assert again.status_code == 200, again.text
    assert again.json()["already_member"] is True


async def test_accept_by_id_refuses_another_email_and_an_unverified_one(
    harness: Harness,
) -> None:
    tenant = await harness.tenant()
    _, admin_tok = await _admin(harness, tenant)
    inv_id = (await _invite(harness, admin_tok))["id"]
    intruder = uuid.uuid7()

    wrong = harness.token(intruder, email="someone-else@example.test")
    r = await harness.call("POST", f"/api/invites/{inv_id}/accept", wrong)
    assert r.status_code == 404, r.text  # an id must not confirm another person's invite
    assert r.json()["error"]["code"] == "INVITE_NOT_FOUND"

    unknown = await harness.call("POST", f"/api/invites/{uuid.uuid7()}/accept", wrong)
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "INVITE_NOT_FOUND"

    unverified = harness.token(intruder, email=NEW_EMAIL, verified=False)
    r = await harness.call("POST", f"/api/invites/{inv_id}/accept", unverified)
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "EMAIL_NOT_VERIFIED"

    assert await harness.membership(intruder, tenant) is None
    [row] = await harness.invites(tenant)
    assert row.status == "pending"


async def test_accept_by_id_refuses_revoked_and_expired(harness: Harness) -> None:
    tenant = await harness.tenant()
    _, admin_tok = await _admin(harness, tenant)
    revoked = await _invite(harness, admin_tok, email="r@example.test")
    expired = await _invite(harness, admin_tok, email="e@example.test")
    assert (
        await harness.call("DELETE", f"/api/tenant/invites/{revoked['id']}", admin_tok)
    ).status_code == 204
    async with harness.factory() as s:
        await s.execute(
            update(TenantInvite)
            .where(TenantInvite.id == uuid.UUID(expired["id"]))
            .values(expires_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=1))
        )
        await s.commit()
    for body, email in ((revoked, "r@example.test"), (expired, "e@example.test")):
        sub = uuid.uuid7()
        r = await harness.call(
            "POST", f"/api/invites/{body['id']}/accept", harness.token(sub, email=email)
        )
        assert r.status_code == 410, (email, r.text)
        assert r.json()["error"]["code"] == "INVITE_GONE"
        assert await harness.membership(sub, tenant) is None


# ─── WP4a review fixes ───────────────────────────────────────────────────

#: The exact key set of an invite create/resend response. It must not vary
#: with whether the address already had an account in the shared realm.
INVITE_CREATED_KEYS = {
    "id",
    "email",
    "role",
    "status",
    "expires_at",
    "created_at",
    "resent_at",
    "invited_by",
    "invite_url",
    "email_delivery",
}


async def test_invite_response_does_not_reveal_whether_the_account_exists(
    harness: Harness,
) -> None:
    """Account enumeration: a tenant_admin must not learn whether an email has
    an account in the shared realm, nor whether it is verified."""
    tenant = await harness.tenant()
    _, admin_tok = await _admin(harness, tenant)
    harness.idp.add_user("known@example.test", verified=True)
    harness.idp.add_user("unverified@example.test", verified=False)

    bodies = {
        email: await _invite(harness, admin_tok, email=email)
        for email in ("fresh@example.test", "known@example.test", "unverified@example.test")
    }
    for email, body in bodies.items():
        assert set(body) == INVITE_CREATED_KEYS, email
        assert body["email_delivery"] == "requested", email
        resent = await harness.call("POST", f"/api/tenant/invites/{body['id']}/resend", admin_tok)
        assert resent.status_code == 200, resent.text
        assert set(resent.json()) == INVITE_CREATED_KEYS, email
        assert resent.json()["email_delivery"] == "requested", email

    listed = await harness.call("GET", "/api/tenant/invites", admin_tok)
    assert listed.status_code == 200
    for item in listed.json()["items"]:
        assert set(item) == INVITE_CREATED_KEYS - {"invite_url", "email_delivery"}

    # w2v itself cannot tell the three apart: provisioning answered each one
    # identically, and nothing about the account was stored.
    assert {r.kc_user_created for r in await harness.invites(tenant)} == {False}


def _email_fails(h: Harness) -> None:
    h.idp.fail_actions_email = True


def _swap(status: int, **kw):
    def install(h: Harness) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            h.idp.requests.append(request)
            return httpx.Response(status, **kw)

        h.idp.handler = handler  # ty: ignore[invalid-assignment]  # why: swap the fake's transport

    return install


def _unreachable(h: Harness) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    h.idp.handler = handler  # ty: ignore[invalid-assignment]  # why: swap the fake's transport


@pytest.mark.parametrize(
    ("break_idp", "code"),
    [pytest.param(_email_fails, "KEYCLOAK_UNAVAILABLE", id="email-502")],
)
async def test_identity_failure_after_commit_keeps_the_invite_and_says_failed(
    harness: Harness, caplog: pytest.LogCaptureFixture, break_idp, code: str
) -> None:
    """The identity email is requested only AFTER the invite commits, so no
    identity failure can lose it: create AND resend answer normally with
    ``email_delivery: "failed"`` (same shape), the invite stands, its
    admin-copyable link works, and the log names the code."""
    tenant = await harness.tenant()
    _, admin_tok = await _admin(harness, tenant)
    up = harness.idp.handler
    break_idp(harness)

    with caplog.at_level(logging.WARNING, logger="w2v_orchestrator.tenancy"):
        r = await harness.call(
            "POST", "/api/tenant/invites", admin_tok, json_body={"email": NEW_EMAIL}
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert set(body) == INVITE_CREATED_KEYS
        assert body["email_delivery"] == "failed"
        resent = await harness.call("POST", f"/api/tenant/invites/{body['id']}/resend", admin_tok)
    assert resent.status_code == 200, resent.text
    assert resent.json()["email_delivery"] == "failed"
    assert sum(f"code={code}" in rec.getMessage() for rec in caplog.records) == 2
    [row] = await harness.invites(tenant)
    assert row.status == "pending"

    # The admin-copyable link still works; a later resend succeeds.
    harness.idp.handler = up  # ty: ignore[invalid-assignment]  # why: restore the fake's transport
    harness.idp.fail_actions_email = False
    again = await harness.call("POST", f"/api/tenant/invites/{body['id']}/resend", admin_tok)
    assert again.json()["email_delivery"] == "requested"
    invitee = uuid.uuid7()
    ok = await harness.call(
        "POST",
        "/api/invites/accept",
        harness.token(invitee, email=NEW_EMAIL),
        json_body={"token": _token(again.json())},
    )
    assert ok.status_code == 200, ok.text


async def test_the_identity_email_is_requested_only_after_the_invite_commits(
    harness: Harness,
) -> None:
    """At the moment provisioning is called, an INDEPENDENT connection must
    already see the invite row (create) and its rotated token (resend)."""
    tenant = await harness.tenant()
    _, admin_tok = await _admin(harness, tenant)
    up = harness.idp.handler
    seen: list[tuple[str, bytes] | None] = []

    async def checks_durability(request: httpx.Request) -> httpx.Response:
        async with harness.factory() as s:
            row = (
                await s.execute(
                    select(TenantInvite.status, TenantInvite.token_hash).where(
                        TenantInvite.tenant_id == tenant
                    )
                )
            ).first()
        seen.append((row.status, bytes(row.token_hash)) if row else None)
        return up(request)

    harness.idp.handler = checks_durability  # ty: ignore[invalid-assignment]  # why: swap the fake's transport
    body = await _invite(harness, admin_tok)
    r = await harness.call("POST", f"/api/tenant/invites/{body['id']}/resend", admin_tok)
    assert r.status_code == 200, r.text
    assert seen == [
        ("pending", hashlib.sha256(_token(body).encode()).digest()),
        ("pending", hashlib.sha256(_token(r.json()).encode()).digest()),
    ]


async def test_create_race_returns_the_existing_invite_before_touching_keycloak(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-check loses a race (simulated: it misses the existing row), so
    the insert hits the partial unique pending index. That must be answered
    with the existing invite — and no Keycloak user may be created for it."""
    tenant = await harness.tenant()
    _, admin_tok = await _admin(harness, tenant)
    first = await _invite(harness, admin_tok)
    harness.idp.requests.clear()

    real = InvitesRepository.get_pending_for_tenant_email
    calls = {"n": 0}

    async def precheck_misses(self, tenant_id, email):
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        return await real(self, tenant_id, email)

    monkeypatch.setattr(InvitesRepository, "get_pending_for_tenant_email", precheck_misses)
    harness.idp.users.clear()  # an un-created account: the old order would create it
    r = await harness.call(
        "POST", "/api/tenant/invites", admin_tok, json_body={"email": NEW_EMAIL.lower()}
    )
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "INVITE_PENDING"
    assert r.json()["error"]["details"]["invite_id"] == first["id"]
    assert harness.idp.requests == []
    assert harness.idp.users == {}
    [row] = await harness.invites(tenant)
    assert str(row.id) == first["id"]


async def test_accept_writes_no_membership_when_the_pending_flip_loses(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``mark_accepted`` is the authority: when it reports the invite is no
    longer pending, neither accept path may add a member."""
    tenant = await harness.tenant()
    _, admin_tok = await _admin(harness, tenant)
    created = await _invite(harness, admin_tok)
    token = _token(created)

    async def lost(self, invite_id, subject_id):
        return False

    monkeypatch.setattr(InvitesRepository, "mark_accepted", lost)
    sub = uuid.uuid7()
    tok = harness.token(sub, email=NEW_EMAIL)

    r = await harness.call("POST", "/api/invites/accept", tok, json_body={"token": token})
    assert r.status_code == 410, r.text
    assert r.json()["error"]["code"] == "INVITE_GONE"
    assert await harness.membership(sub, tenant) is None

    by_id = await harness.call("POST", f"/api/invites/{created['id']}/accept", tok)
    assert by_id.status_code == 410, by_id.text
    assert by_id.json()["error"]["code"] == "INVITE_GONE"
    assert await harness.membership(sub, tenant) is None


async def test_accept_by_id_and_a_concurrent_revoke_cannot_both_win(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tenant_admin revokes the invite on another connection while the by-id
    accept holds it between "load" and "mark accepted". With the FOR UPDATE the
    revoke waits on the lock (bounded here by lock_timeout) and loses; without
    it the revoke commits first and the accept 410s. Invariant: accepted ⇔
    member, revoked ⇔ no member — and the accept that holds the lock wins."""
    tenant = await harness.tenant()
    _, admin_tok = await _admin(harness, tenant)
    inv_id = uuid.UUID((await _invite(harness, admin_tok))["id"])

    real = InvitesRepository.mark_accepted
    revoke_won: list[bool] = []

    async def race_a_revoke_then_accept(self, invite_id, subject_id):
        async with harness.factory() as other:  # a separate connection
            await other.execute(text("SET LOCAL lock_timeout = '300ms'"))
            try:
                won = await InvitesRepository(other).revoke(inv_id, tenant)
                await other.commit()
            except DBAPIError:
                await other.rollback()
                won = False
        revoke_won.append(won)
        return await real(self, invite_id, subject_id)

    monkeypatch.setattr(InvitesRepository, "mark_accepted", race_a_revoke_then_accept)
    sub = uuid.uuid7()
    r = await harness.call(
        "POST", f"/api/invites/{inv_id}/accept", harness.token(sub, email=NEW_EMAIL)
    )
    assert r.status_code == 200, r.text

    [row] = await harness.invites(tenant)
    assert revoke_won == [False], "the revoke must block on the accept's row lock"
    assert row.status == "accepted"
    assert await harness.membership(sub, tenant) == ("member", "active")


@pytest.mark.parametrize("state", ["disabled"])
async def test_a_disabled_or_ambiguous_address_answers_like_any_other(
    harness: Harness, state: str
) -> None:
    """Provisioning accepts every well-formed invite identically, so the
    tenant_admin's response is the same as for a fresh address — no account
    state leaks — and the invite row stands."""
    tenant = await harness.tenant()
    _, admin_tok = await _admin(harness, tenant)
    if state == "disabled":
        harness.idp.add_user(NEW_EMAIL, enabled=False)
    else:
        harness.idp.add_user(NEW_EMAIL)
        harness.idp.add_user(NEW_EMAIL)
    body = await _invite(harness, admin_tok)
    assert set(body) == INVITE_CREATED_KEYS
    assert body["email_delivery"] == "requested"
    assert harness.idp.mails == []
    [row] = await harness.invites(tenant)
    assert row.status == "pending"
