"""P2 WP4a — tenant members roster, role change, removal (real auth + tenancy stack)."""

from __future__ import annotations

import datetime
import uuid

import pytest
from sqlalchemy import insert

from src.database.models import TenantInvite
from tests.api.routes._tenancy_harness import (  # noqa: F401 — fixtures
    Harness,
    harness_fixture,
    signing_key_fixture,
)

pytestmark = pytest.mark.asyncio


async def _admin(h: Harness, tenant: uuid.UUID) -> tuple[uuid.UUID, str]:
    sub = uuid.uuid7()
    await h.member(sub, tenant, "tenant_admin", email=f"admin-{sub}@acme.test")
    return sub, h.token(sub)


async def test_roster_lists_only_the_effective_tenant(harness: Harness) -> None:
    a, b = await harness.tenant(), await harness.tenant()
    admin, tok = await _admin(harness, a)
    alice, bob = uuid.uuid7(), uuid.uuid7()
    await harness.member(alice, a, email="alice@a.test")
    await harness.member(bob, b, email="bob@b.test")

    r = await harness.call("GET", "/api/tenant/members", tok)
    assert r.status_code == 200, r.text
    body = r.json()
    assert {m["subject_id"] for m in body["items"]} == {str(admin), str(alice)}
    assert body["total"] == 2
    assert {m["role"] for m in body["items"]} == {"tenant_admin", "member"}


async def test_role_change_is_scoped_audited_and_reversible(harness: Harness) -> None:
    tenant = await harness.tenant()
    _, tok = await _admin(harness, tenant)
    alice = uuid.uuid7()
    await harness.member(alice, tenant)

    r = await harness.call(
        "PATCH", f"/api/tenant/members/{alice}", tok, json_body={"role": "tenant_admin"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["role"] == "tenant_admin"
    assert await harness.membership(alice, tenant) == ("tenant_admin", "active")
    r = await harness.call(
        "PATCH", f"/api/tenant/members/{alice}", tok, json_body={"role": "member"}
    )
    assert r.json()["role"] == "member"
    assert (await harness.audit_actions(tenant)).count("member.set_role") == 2

    bad = await harness.call(
        "PATCH", f"/api/tenant/members/{alice}", tok, json_body={"role": "super_admin"}
    )
    assert bad.status_code == 422


async def test_last_active_tenant_admin_cannot_be_removed_or_demoted(harness: Harness) -> None:
    tenant = await harness.tenant()
    admin, tok = await _admin(harness, tenant)

    for method, body in (("PATCH", {"role": "member"}), ("DELETE", None)):
        r = await harness.call(method, f"/api/tenant/members/{admin}", tok, json_body=body)
        assert r.status_code == 409, (method, r.text)
        assert r.json()["error"]["code"] == "LAST_TENANT_ADMIN"
    assert await harness.membership(admin, tenant) == ("tenant_admin", "active")

    # With a second admin, the first may step down.
    second = uuid.uuid7()
    await harness.member(second, tenant, "tenant_admin")
    r = await harness.call("DELETE", f"/api/tenant/members/{admin}", tok)
    assert r.status_code == 200, r.text
    assert await harness.membership(admin, tenant) == ("tenant_admin", "removed")


async def test_removed_member_is_refused_and_removal_is_audited(harness: Harness) -> None:
    tenant = await harness.tenant()
    _, tok = await _admin(harness, tenant)
    alice = uuid.uuid7()
    await harness.member(alice, tenant)
    alice_tok = harness.token(alice)
    assert (await harness.call("GET", "/api/me/tenants", alice_tok)).json()["items"]

    r = await harness.call("DELETE", f"/api/tenant/members/{alice}", tok)
    assert r.status_code == 200, r.text
    assert r.json() == {"subject_id": str(alice), "status": "removed"}
    assert await harness.membership(alice, tenant) == ("member", "removed")
    assert "member.remove" in await harness.audit_actions(tenant)
    # Identity untouched: removal is w2v-side only.
    assert harness.idp.requests == []

    after = await harness.call("GET", "/api/me/tenants", alice_tok)
    assert after.json()["items"] == []
    denied = await harness.call(
        "GET", "/api/tenant/members", alice_tok, headers={"X-Active-Tenant-Id": str(tenant)}
    )
    assert denied.status_code == 403
    # Removing again: no active row.
    again = await harness.call("DELETE", f"/api/tenant/members/{alice}", tok)
    assert again.status_code == 404


async def _created_by_invite(h: Harness, sub: uuid.UUID, tenant: uuid.UUID) -> None:
    async with h.factory() as s:
        await s.execute(
            insert(TenantInvite).values(
                id=uuid.uuid7(),
                tenant_id=tenant,
                email="made@w2v.test",
                role="member",
                token_hash=uuid.uuid4().bytes * 2,
                expires_at=datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=1),
                status="accepted",
                accepted_subject_id=sub,
                kc_user_created=True,
            )
        )
        await s.commit()


async def test_removing_the_last_membership_of_an_invited_account_never_touches_identity(
    harness: Harness,
) -> None:
    """Removal is w2v-side only (template decision): even the last membership
    of an account provisioning created for a w2v invite leaves the identity
    alone — w2v holds no disable power, and another app may use the account."""
    a, b = await harness.tenant(), await harness.tenant()
    _, tok_a = await _admin(harness, a)
    _, tok_b = await _admin(harness, b)
    sub = uuid.uuid7()
    await harness.member(sub, a)
    await harness.member(sub, b)
    await _created_by_invite(harness, sub, a)

    for tok, tenant in ((tok_a, a), (tok_b, b)):
        r = await harness.call("DELETE", f"/api/tenant/members/{sub}", tok)
        assert r.status_code == 200, r.text
        assert r.json() == {"subject_id": str(sub), "status": "removed"}
        assert await harness.membership(sub, tenant) == ("member", "removed")
    assert harness.idp.requests == []


async def test_non_admin_is_refused_on_every_member_route(harness: Harness) -> None:
    tenant = await harness.tenant()
    admin, _ = await _admin(harness, tenant)
    plain = uuid.uuid7()
    await harness.member(plain, tenant)
    tok = harness.token(plain, roles=["tenant_admin"])  # a REALM role grants nothing here

    for method, path, body in (
        ("GET", "/api/tenant/members", None),
        ("PATCH", f"/api/tenant/members/{admin}", {"role": "member"}),
        ("PATCH", f"/api/tenant/members/{plain}", {"role": "tenant_admin"}),
        ("DELETE", f"/api/tenant/members/{admin}", None),
    ):
        r = await harness.call(method, path, tok, json_body=body)
        assert r.status_code == 403, (method, path, r.text)
    assert await harness.membership(admin, tenant) == ("tenant_admin", "active")
    assert await harness.membership(plain, tenant) == ("member", "active")


async def test_tenant_admin_cannot_modify_another_tenants_members(harness: Harness) -> None:
    a, b = await harness.tenant(), await harness.tenant()
    _, tok_a = await _admin(harness, a)
    victim = uuid.uuid7()
    await harness.member(victim, b, "tenant_admin")

    for method, body in (("PATCH", {"role": "member"}), ("DELETE", None)):
        r = await harness.call(method, f"/api/tenant/members/{victim}", tok_a, json_body=body)
        assert r.status_code == 404, (method, r.text)
        r = await harness.call(
            method,
            f"/api/tenant/members/{victim}",
            tok_a,
            json_body=body,
            headers={"X-Active-Tenant-Id": str(b)},
        )
        assert r.status_code == 403, (method, r.text)
    assert await harness.membership(victim, b) == ("tenant_admin", "active")
    assert await harness.audit_actions(b) == []
