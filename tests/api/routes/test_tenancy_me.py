"""P2 WP4a — the caller's tenants, /api/auth/me tenancy fields, tenant rename."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from src.database.models import Tenant
from tests.api.routes._tenancy_harness import (  # noqa: F401 — fixtures
    Harness,
    harness_fixture,
    signing_key_fixture,
)

pytestmark = pytest.mark.asyncio


async def test_my_tenants_lists_active_memberships_with_names(harness: Harness) -> None:
    a, b, c = await harness.tenant("Alpha"), await harness.tenant("Beta"), await harness.tenant()
    sub = uuid.uuid7()
    await harness.member(sub, a, "tenant_admin")
    await harness.member(sub, b)
    await harness.member(sub, c)
    other = uuid.uuid7()
    await harness.member(other, c, "tenant_admin")
    # Removed from c by its admin.
    r = await harness.call("DELETE", f"/api/tenant/members/{sub}", harness.token(other))
    assert r.status_code == 200, r.text

    r = await harness.call("GET", "/api/me/tenants", harness.token(sub))
    assert r.status_code == 200, r.text
    got = {i["tenant_id"]: (i["name"], i["role"]) for i in r.json()["items"]}
    assert got == {str(a): ("Alpha", "tenant_admin"), str(b): ("Beta", "member")}


async def test_my_tenants_is_reachable_with_no_membership(harness: Harness) -> None:
    sub = uuid.uuid7()
    r = await harness.call("GET", "/api/me/tenants", harness.token(sub))
    assert r.status_code == 200, r.text
    assert r.json() == {"items": []}


async def test_auth_me_reports_membership_role_and_memberships(harness: Harness) -> None:
    a, b = await harness.tenant("Alpha"), await harness.tenant("Beta")
    sub = uuid.uuid7()
    await harness.member(sub, a, "tenant_admin")
    await harness.member(sub, b)

    r = await harness.call(
        "GET", "/api/auth/me", harness.token(sub), headers={"X-Active-Tenant-Id": str(b)}
    )
    assert r.status_code == 200, r.text
    me = r.json()
    assert me["tenant_id"] == str(b)
    assert me["role"] == "member"
    assert {m["tenant_id"]: m["role"] for m in me["memberships"]} == {
        str(a): "tenant_admin",
        str(b): "member",
    }


async def test_tenant_admin_renames_their_own_tenant_only(harness: Harness) -> None:
    a, b = await harness.tenant("Old"), await harness.tenant("Other")
    admin = uuid.uuid7()
    await harness.member(admin, a, "tenant_admin")
    plain = uuid.uuid7()
    await harness.member(plain, a)

    r = await harness.call("PATCH", "/api/tenant", harness.token(admin), json_body={"name": "New"})
    assert r.status_code == 200, r.text
    assert r.json() == {"tenant_id": str(a), "name": "New", "status": "active"}
    assert "tenant.update" in await harness.audit_actions(a)

    denied = await harness.call(
        "PATCH", "/api/tenant", harness.token(plain), json_body={"name": "Hijack"}
    )
    assert denied.status_code == 403
    cross = await harness.call(
        "PATCH",
        "/api/tenant",
        harness.token(admin),
        json_body={"name": "Hijack"},
        headers={"X-Active-Tenant-Id": str(b)},
    )
    assert cross.status_code == 403

    async with harness.factory() as s:
        names = dict((await s.execute(select(Tenant.id, Tenant.name))).tuples().all())
    assert names == {a: "New", b: "Other"}
