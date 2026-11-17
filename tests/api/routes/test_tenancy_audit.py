"""P6 ORCH-5 — tenant-scoped audit view on the REAL request path.

Real RS256 JWT → w2v_common ``AuthMiddleware`` → ``TenancyMiddleware`` (membership
mode) → the REAL ``/api/tenant/audit`` router → real ``admin_audit_log`` /
``tenant_members`` rows in Postgres. Only the JWKS socket is faked.

Non-vacuity: every isolation/refusal assertion is paired with a control that sees the
row through the same call shape.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

import pytest

from src.database.repositories.admin_audit_repository import AuditRepository
from tests.api.routes._tenancy_harness import (  # noqa: F401 — fixtures
    Harness,
    harness_fixture,
    signing_key_fixture,
)

pytestmark = pytest.mark.asyncio

PATH = "/api/tenant/audit"
T0 = datetime.datetime(2026, 9, 1, tzinfo=datetime.UTC)


async def _user(h: Harness, tenant: uuid.UUID, role: str = "tenant_admin") -> uuid.UUID:
    sub = uuid.uuid7()
    await h.member(sub, tenant, role, email=f"{role}-{sub}@acme.test")
    return sub


async def _seed(
    h: Harness,
    tenant: uuid.UUID | None,
    action: str,
    actor: uuid.UUID | None,
    detail: dict[str, Any] | None = None,
    at: datetime.datetime | None = None,
) -> uuid.UUID:
    async with h.factory() as s:
        row = await AuditRepository(s).record(
            actor_subject_id=actor, action=action, target_tenant_id=tenant, detail=detail
        )
        if at is not None:
            row.created_at = at
        await s.commit()
        return row.id


async def _get(h: Harness, token: str, query: str = "") -> Any:
    r = await h.call("GET", f"{PATH}{query}", token)
    assert r.status_code == 200, r.text
    return r.json()


async def test_tenant_a_never_sees_tenant_b_rows(harness: Harness) -> None:
    a, b = await harness.tenant(), await harness.tenant()
    admin_a, admin_b = await _user(harness, a), await _user(harness, b)
    row_a = await _seed(harness, a, "defaults.update", admin_a, {"version": 1})
    row_b = await _seed(harness, b, "defaults.update", admin_b, {"version": 1})
    await _seed(harness, None, "youtube.platform_channel.set", uuid.uuid7(), {})

    body_a = await _get(harness, harness.token(admin_a))
    body_b = await _get(harness, harness.token(admin_b))
    assert [i["id"] for i in body_a["items"]] == [str(row_a)]
    assert [i["id"] for i in body_b["items"]] == [str(row_b)]

    # An admin of A cannot scope-switch into B (super_admin only) …
    r = await harness.call(
        "GET", PATH, harness.token(admin_a), headers={"X-Active-Tenant-Id": str(b)}
    )
    assert r.status_code in (403, 404), r.text
    # … and filtering on B's admin from A yields nothing (control: B sees that row).
    assert (await _get(harness, harness.token(admin_a), f"?actor={admin_b}"))["items"] == []
    assert len((await _get(harness, harness.token(admin_b), f"?actor={admin_b}"))["items"]) == 1


async def test_member_is_forbidden(harness: Harness) -> None:
    tenant = await harness.tenant()
    member = await _user(harness, tenant, "member")
    admin = await _user(harness, tenant)
    await _seed(harness, tenant, "defaults.update", admin, {"version": 1})
    r = await harness.call("GET", PATH, harness.token(member))
    assert r.status_code == 403, r.text
    assert "items" not in r.text
    assert len((await _get(harness, harness.token(admin)))["items"]) == 1  # control


async def test_super_admin_actions_are_redacted_to_platform_operator(harness: Harness) -> None:
    tenant, home = await harness.tenant(), await harness.tenant()
    admin = await _user(harness, tenant)
    super_sub = uuid.uuid7()
    await harness.member(super_sub, home, "tenant_admin", email="root@platform.test")

    # A REAL super_admin write: scope-switched into the tenant, PUT the defaults.
    super_tok = harness.token(super_sub, roles=["super_admin"], email="root@platform.test")
    r = await harness.call(
        "PUT",
        "/api/tenant/defaults",
        super_tok,
        json_body={"expected_version": 0, "retention_mode": "off"},
        headers={"X-Active-Tenant-Id": str(tenant)},
    )
    assert r.status_code == 200, r.text
    # Super-admin-only actions, including one by a super_admin who IS a member here.
    super_member = await _user(harness, tenant)
    await _seed(harness, tenant, "scope.switch", super_member, {"from_tenant": str(home)})
    await _seed(harness, tenant, "tenant.update", admin, {"after": {"status": "suspended"}})
    # Member actions (control): named, with email.
    await _seed(harness, tenant, "tenant.update", admin, {"by": "tenant_admin"})
    await _seed(harness, tenant, "member.set_role", admin, {"to": "member"})

    items = (await _get(harness, harness.token(admin)))["items"]
    by_action = [(i["action"], i["actor_kind"]) for i in items]
    assert by_action == [
        ("member.set_role", "member"),
        ("tenant.update", "member"),
        ("tenant.update", "platform_operator"),
        ("scope.switch", "platform_operator"),
        ("defaults.update", "platform_operator"),
    ]
    text = str(items)
    assert str(super_sub) not in text, "operator subject id leaked"
    assert str(super_member) not in text, "a member acting as platform operator was named"
    assert str(home) not in text, "operator's home tenant leaked via detail.from_tenant"
    assert "root@platform.test" not in text
    for i in items:
        if i["actor_kind"] == "platform_operator":
            assert i["actor_subject_id"] is None
            assert i["actor_label"] == "platform operator"
        else:
            assert i["actor_subject_id"] == str(admin)
            assert i["actor_label"] == f"tenant_admin-{admin}@acme.test"

    # The actor filter cannot be used as an oracle on the operator's id …
    for operator in (super_sub, super_member):
        assert (await _get(harness, harness.token(admin), f"?actor={operator}"))["items"] == []
    # … while actor_kind splits the two sets exactly.
    ops = (await _get(harness, harness.token(admin), "?actor_kind=platform_operator"))["items"]
    mem = (await _get(harness, harness.token(admin), "?actor_kind=member"))["items"]
    assert (len(ops), len(mem)) == (3, 2)


async def test_secret_named_detail_keys_never_reach_the_response(harness: Harness) -> None:
    tenant = await harness.tenant()
    admin = await _user(harness, tenant)
    await _seed(
        harness,
        tenant,
        "connection.set",
        admin,
        {
            "slot": "llm",
            "key_fingerprint": "abcdef012345",
            "api_key": "sk-LEAK-1",  # pragma: allowlist secret — fake, asserts the scrub
            "nested": {"client_secret": "LEAK-2", "kind": "openai"},  # pragma: allowlist secret
            "list": [{"Authorization": "Bearer LEAK-3"}],
        },
    )
    [item] = (await _get(harness, harness.token(admin)))["items"]
    assert "LEAK" not in str(item)
    assert item["detail"] == {
        "slot": "llm",
        "key_fingerprint": "abcdef012345",
        "nested": {"kind": "openai"},
        "list": [{}],
    }


async def test_keyset_pagination_and_bounds(harness: Harness) -> None:
    tenant = await harness.tenant()
    admin = await _user(harness, tenant)
    tok = harness.token(admin)
    ids = [
        await _seed(
            harness,
            tenant,
            "defaults.update",
            admin,
            {"version": n},
            T0 + datetime.timedelta(hours=n),
        )
        for n in range(7)
    ]
    newest_first = [str(i) for i in reversed(ids)]

    seen: list[str] = []
    cursor: str | None = None
    pages = 0
    while True:
        q = "?limit=3" + (f"&cursor={cursor}" if cursor else "")
        body = await _get(harness, tok, q)
        seen += [i["id"] for i in body["items"]]
        pages += 1
        cursor = body["next_cursor"]
        if cursor is None:
            break
    assert seen == newest_first
    assert pages == 3

    # Hard max page size, and bad inputs, are 422 — never a silently clamped page.
    for bad in (
        "?limit=101",
        "?limit=0",
        "?cursor=not-a-cursor",
        "?action=not.an.action",
        "?since=2026-09-01T00:00:00",  # naive
        "?since=2026-09-02T00:00:00Z&until=2026-09-01T00:00:00Z",
        "?since=2025-01-01T00:00:00Z&until=2026-09-01T00:00:00Z",  # > 366 days
    ):
        r = await harness.call("GET", f"{PATH}{bad}", tok)
        assert r.status_code == 422, (bad, r.text)
    assert len((await _get(harness, tok, "?limit=100"))["items"]) == 7

    # Date range: [since, until) — hours 2, 3, 4.
    window = await _get(harness, tok, "?since=2026-09-01T02:00:00Z&until=2026-09-01T05:00:00Z")
    assert [i["detail"]["version"] for i in window["items"]] == [4, 3, 2]
    assert (await _get(harness, tok, "?action=member.remove"))["items"] == []


async def test_secret_scrub_matches_any_key_spelling(harness: Harness) -> None:
    """camelCase / kebab-case / *_access_key spellings are secret names too."""
    tenant = await harness.tenant()
    admin = await _user(harness, tenant)
    await _seed(
        harness,
        tenant,
        "connection.set",
        admin,
        {
            "slot": "storage",
            "privateKey": "LEAK-1",  # pragma: allowlist secret — fake, asserts the scrub
            "x-api-key": "LEAK-2",
            "aws_access_key": "LEAK-3",
            "accessKeyId": "LEAK-4",
            "nested": [
                {
                    "Refresh-Token": "LEAK-5",
                    "client.secret": "LEAK-6",  # pragma: allowlist secret
                    "kind": "s3",
                }
            ],
        },
    )
    [item] = (await _get(harness, harness.token(admin)))["items"]
    assert "LEAK" not in str(item), item
    assert item["detail"] == {"slot": "storage", "nested": [{"kind": "s3"}]}


async def test_forged_cursors_and_extreme_ranges_never_500_or_cross_tenants(
    harness: Harness,
) -> None:
    import base64
    import json

    a, b = await harness.tenant(), await harness.tenant()
    admin_a, admin_b = await _user(harness, a), await _user(harness, b)
    await _seed(harness, b, "defaults.update", admin_b, {"version": 1})
    own = await _seed(harness, a, "defaults.update", admin_a, {"version": 1})
    tok = harness.token(admin_a)

    def enc(obj: object) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")

    # A well-formed forged cursor only moves the keyset bound; it cannot widen the tenant.
    far = enc({"t": "9999-12-31T23:59:59+00:00", "i": str(uuid.UUID(int=2**128 - 1))})
    assert [i["id"] for i in (await _get(harness, tok, f"?cursor={far}"))["items"]] == [str(own)]
    early = enc({"t": "0001-01-01T00:00:00+14:00", "i": str(uuid.UUID(int=0))})
    assert (await _get(harness, tok, f"?cursor={early}"))["items"] == []
    for bad in (
        enc([1, 2]),
        enc("x"),
        enc({"t": 1, "i": 2}),
        enc({"t": "2026-01-01T00:00:00", "i": str(uuid.uuid7())}),  # naive
        enc({"t": "2026-01-01T00:00:00Z", "i": "' OR 1=1 --"}),
        base64.urlsafe_b64encode(b"\xff\xfe\x00").decode(),
        "%00",
    ):
        r = await harness.call("GET", f"{PATH}?cursor={bad}", tok)
        assert r.status_code == 422, (bad, r.status_code, r.text)
    for q in (
        "?since=0001-01-01T00:00:00%2B14:00",
        "?until=9999-12-31T23:59:59-14:00",
        "?since=9999-12-31T00:00:00Z&until=9999-12-31T23:00:00Z",
    ):
        r = await harness.call("GET", f"{PATH}{q}", tok)
        assert r.status_code in (200, 422), (q, r.status_code, r.text)
