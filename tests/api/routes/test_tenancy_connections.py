"""P3 WP4 — tenant provider connections on the REAL request path.

Real RS256 JWT → w2v_common ``AuthMiddleware`` → ``TenancyMiddleware`` (membership
mode) → the REAL ``/api/tenant/connections`` router → real ``tenant_connections`` /
``tenant_connection_versions`` / ``admin_audit_log`` rows in Postgres. Faked: the
JWKS socket, and the live provider test (``get_provider_tester`` → a recorder that
returns a scripted verdict) — the HTTP tester itself is covered in
``tests/services/providers/test_tester.py``. Policy and keyring are injected so the
test controls SSRF mode and key material.

Non-vacuity: every "nothing persisted" assertion is paired with a sibling case that
DOES persist through the same call shape, and every "the key never left" assertion
reads the tester's recorded calls.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import fakeredis.aioredis
import pytest
from sqlalchemy import func, select

from src.api.routes.tenancy.connections import (
    get_connections_keyring,
    get_connections_policy,
    get_provider_tester,
)
from src.database.models import AdminAuditLog, TenantConnection, TenantConnectionVersion
from src.services.providers.crypto import Keyring
from src.services.providers.registry import SLOT_NAMES
from src.services.providers.settings import ProviderPolicySettings
from src.services.providers.tester import ProviderTestResult
from tests.api.routes._tenancy_harness import (  # noqa: F401 — fixtures
    Harness,
    harness_fixture,
    signing_key_fixture,
)

pytestmark = pytest.mark.asyncio

SECRET = "sk-live-NEVER-ECHO-7c1d2e"  # pragma: allowlist secret
BASE_A = "https://llm-a.example.test/v1"
BASE_B = "https://llm-b.example.test/v1"
_PRIVATE_OK = ProviderPolicySettings(
    ssrf_mode="private_ok",
    egress_allowlist=frozenset({"llm-a.example.test", "llm-b.example.test"}),
)


@dataclass
class FakeTester:
    """Records every live test (incl. the secret it was handed) and returns ``verdict``."""

    verdict: ProviderTestResult = field(
        default_factory=lambda: ProviderTestResult(True, "llm answered HTTP 200", 200)
    )
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def test(self, **kwargs: Any) -> ProviderTestResult:
        self.calls.append(kwargs)
        return self.verdict


@dataclass
class Ctx:
    h: Harness
    tester: FakeTester
    policy: list[ProviderPolicySettings]
    keyring: list[Keyring]


@pytest.fixture
def ctx(harness: Harness) -> Ctx:
    c = Ctx(
        h=harness,
        tester=FakeTester(),
        policy=[_PRIVATE_OK],
        keyring=[Keyring([("k1", os.urandom(32))])],
    )
    harness.app.dependency_overrides[get_provider_tester] = lambda: c.tester
    harness.app.dependency_overrides[get_connections_policy] = lambda: c.policy[0]
    harness.app.dependency_overrides[get_connections_keyring] = lambda: c.keyring[0]
    # The live-test rate limiter reads app.state.redis (fakeredis, as the suite does).
    harness.app.state.redis = fakeredis.aioredis.FakeRedis()
    return c


async def _user(h: Harness, tenant: uuid.UUID, role: str = "tenant_admin") -> str:
    sub = uuid.uuid7()
    await h.member(sub, tenant, role, email=f"{role}-{sub}@acme.test")
    return h.token(sub)


def _put(expected: int, **over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "kind": "openai-compatible",
        "base_url": BASE_A,
        "model": "m-1",
        "secret": SECRET,
        "expected_version": expected,
    }
    body.update(over)
    return {k: v for k, v in body.items() if v is not ...}


async def _llm(h: Harness, tok: str) -> dict[str, Any]:
    r = await h.call("GET", "/api/tenant/connections", tok)
    assert r.status_code == 200, r.text
    return next(i for i in r.json()["items"] if i["slot"] == "llm")


async def _connection_rows(h: Harness, tenant: uuid.UUID) -> int:
    async with h.factory() as s:
        return (
            await s.execute(
                select(func.count())
                .select_from(TenantConnection)
                .where(TenantConnection.tenant_id == tenant)
            )
        ).scalar_one()


async def _audit(h: Harness, tenant: uuid.UUID) -> list[AdminAuditLog]:
    async with h.factory() as s:
        return list(
            (
                await s.execute(
                    select(AdminAuditLog)
                    .where(AdminAuditLog.target_tenant_id == tenant)
                    .order_by(AdminAuditLog.created_at)
                )
            )
            .scalars()
            .all()
        )


# ─── authz ───────────────────────────────────────────────────────────────


async def test_member_is_forbidden_on_every_route(ctx: Ctx) -> None:
    tenant = await ctx.h.tenant()
    member = await _user(ctx.h, tenant, "member")
    for method, path, body in (
        ("GET", "/api/tenant/connections", None),
        ("PUT", "/api/tenant/connections/llm", _put(0)),
        ("POST", "/api/tenant/connections/llm/test", None),
        ("DELETE", "/api/tenant/connections/llm", None),
    ):
        r = await ctx.h.call(method, path, member, json_body=body)
        assert r.status_code == 403, (method, path, r.text)
    assert ctx.tester.calls == []
    assert await _connection_rows(ctx.h, tenant) == 0
    # Control: the same PUT as an admin persists.
    admin = await _user(ctx.h, tenant)
    r = await ctx.h.call("PUT", "/api/tenant/connections/llm", admin, json_body=_put(0))
    assert r.status_code == 200, r.text


async def test_tenant_a_cannot_see_or_modify_tenant_b(ctx: Ctx) -> None:
    a, b = await ctx.h.tenant(), await ctx.h.tenant()
    tok_a, tok_b = await _user(ctx.h, a), await _user(ctx.h, b)
    r = await ctx.h.call("PUT", "/api/tenant/connections/llm", tok_b, json_body=_put(0))
    assert r.status_code == 200, r.text

    assert (await _llm(ctx.h, tok_a))["connection"] is None
    # A's delete finds nothing of its own — B's connection is untouched.
    r = await ctx.h.call("DELETE", "/api/tenant/connections/llm", tok_a)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "CONNECTION_NOT_FOUND"
    # A's PUT creates A's OWN version 1; B stays at version 1 with its key.
    r = await ctx.h.call(
        "PUT", "/api/tenant/connections/llm", tok_a, json_body=_put(0, base_url=BASE_B)
    )
    assert r.status_code == 200, r.text
    b_llm = await _llm(ctx.h, tok_b)
    assert b_llm["connection"]["version"] == 1
    assert b_llm["connection"]["base_url"] == BASE_A
    # A tenant_admin cannot switch into B (super-admin only).
    r = await ctx.h.call(
        "GET", "/api/tenant/connections", tok_a, headers={"X-Active-Tenant-Id": str(b)}
    )
    assert r.status_code == 403
    assert {e.action for e in await _audit(ctx.h, b)} == {"connection.set"}


# ─── GET ─────────────────────────────────────────────────────────────────


async def test_get_lists_every_registry_slot_with_deployment_defaults(ctx: Ctx) -> None:
    tenant = await ctx.h.tenant()
    tok = await _user(ctx.h, tenant)
    r = await ctx.h.call("GET", "/api/tenant/connections", tok)
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    assert [i["slot"] for i in items] == list(SLOT_NAMES)
    llm = items[0]
    assert llm == {
        "slot": "llm",
        "type": "connection",
        "writable": True,
        "allowed_kinds": ["anthropic", "openai-compatible"],
        "source": "deployment",
        "default_kind": "nonagentic-wm",
        "byo_required": False,
        "kind_requirements": {
            "anthropic": {
                "base_url": "optional",
                "model_required": False,
                "secret_required": False,
                "required_options": [],
            },
            "openai-compatible": {
                "base_url": "required",
                "model_required": False,
                "secret_required": False,
                "required_options": [],
            },
        },
        "connection": None,
    }
    assert r.json()["llm_access"] == {"mode": "local_allowed", "label": "Platform models allowed"}
    writable = {"llm", "vlm", "tts", "asr", "image", "video_gen", "ocr", "device_farm"}
    assert {i["slot"] for i in items if i["writable"]} == writable
    assert all(i["allowed_kinds"] == [] for i in items if i["slot"] not in writable)


async def test_get_reports_the_tenants_licences_byo_posture(ctx: Ctx) -> None:
    """#1106: ``byo_required`` per slot is the tenant's licences ``byo_mode``."""
    from src.api.routes.tenancy.connections import get_connections_byo_posture
    from src.services.providers.byo_posture import StaticByoPosture

    posture = StaticByoPosture("byo_required")
    ctx.h.app.dependency_overrides[get_connections_byo_posture] = lambda: posture
    tenant = await ctx.h.tenant()
    tok = await _user(ctx.h, tenant)
    r = await ctx.h.call("GET", "/api/tenant/connections", tok)
    assert r.status_code == 200, r.text
    byo = {i["slot"]: i["byo_required"] for i in r.json()["items"]}
    # 2026-10-22: every model-provider slot, not only the platform-GPU trio.
    assert {s for s, required in byo.items() if required} == {
        "llm", "vlm", "tts", "asr", "image", "video_gen", "ocr",
    }  # fmt: skip
    assert posture.calls == [tenant]


async def test_get_fails_closed_when_the_posture_cannot_be_read(ctx: Ctx) -> None:
    from src.api.routes.tenancy.connections import get_connections_byo_posture
    from src.services.providers.errors import ByoPostureUnavailableError

    class _Down:
        async def byo_mode(self, tenant_id: uuid.UUID, *, slot: str):
            raise ByoPostureUnavailableError(slot, "licences answered 503")

    ctx.h.app.dependency_overrides[get_connections_byo_posture] = lambda: _Down()
    tenant = await ctx.h.tenant()
    tok = await _user(ctx.h, tenant)
    r = await ctx.h.call("GET", "/api/tenant/connections", tok)
    assert r.status_code == 503, r.text
    assert r.json()["error"]["code"] == "BYO_POSTURE_UNAVAILABLE"


# ─── PUT ─────────────────────────────────────────────────────────────────


async def test_put_tests_before_persisting_and_never_echoes_the_secret(ctx: Ctx) -> None:
    tenant = await ctx.h.tenant()
    tok = await _user(ctx.h, tenant)
    r = await ctx.h.call("PUT", "/api/tenant/connections/llm", tok, json_body=_put(0))
    assert r.status_code == 200, r.text
    assert SECRET not in r.text
    body = r.json()
    conn = body["connection"]
    assert body["source"] == "tenant"
    assert conn["version"] == 1 and conn["has_secret"] is True
    assert conn["last_test_ok"] is True and conn["last_test_at"]
    assert len(conn["key_fingerprint"]) == 12
    assert set(conn) == {
        "id",
        "kind",
        "base_url",
        "model",
        "options",
        "version",
        "has_secret",
        "key_fingerprint",
        "last_test_at",
        "last_test_ok",
        "last_test_error",
    }
    # The live test received the key it was meant to try.
    [call] = ctx.tester.calls
    assert call["secret"] == SECRET and call["base_url"] == BASE_A

    listing = await ctx.h.call("GET", "/api/tenant/connections", tok)
    assert SECRET not in listing.text
    # Stored sealed, never in the clear; the audit row carries the fingerprint only.
    async with ctx.h.factory() as s:
        stored = (
            await s.execute(
                select(TenantConnectionVersion.secret_ciphertext)
                .join(TenantConnection)
                .where(TenantConnection.tenant_id == tenant)
            )
        ).scalar_one()
    assert stored is not None and SECRET.encode() not in bytes(stored)
    [entry] = await _audit(ctx.h, tenant)
    assert entry.action == "connection.set"
    assert entry.detail == {
        "slot": "llm",
        "kind": "openai-compatible",
        "version": 1,
        "key_fingerprint": conn["key_fingerprint"],
        "probed": True,
    }
    assert SECRET not in json.dumps(entry.detail)


async def test_failed_live_test_persists_only_an_audit_row(ctx: Ctx) -> None:
    """Owner decision: a failed PUT live test writes ``connection.test_failed``
    (slot, kind, upstream_status — never the secret) and persists NOTHING else."""
    tenant = await ctx.h.tenant()
    tok = await _user(ctx.h, tenant)
    ctx.tester.verdict = ProviderTestResult(False, "llm answered HTTP 401", 401)
    r = await ctx.h.call("PUT", "/api/tenant/connections/llm", tok, json_body=_put(0))
    assert r.status_code == 422, r.text
    err = r.json()["error"]
    assert err["code"] == "PROVIDER_TEST_FAILED"
    assert err["details"] == {"upstream_status": 401}
    assert SECRET not in r.text
    assert len(ctx.tester.calls) == 1
    assert await _connection_rows(ctx.h, tenant) == 0
    [entry] = await _audit(ctx.h, tenant)
    assert entry.action == "connection.test_failed"
    assert entry.detail == {"slot": "llm", "kind": "openai-compatible", "upstream_status": 401}
    assert SECRET not in json.dumps(entry.detail)
    # Control: the identical request persists once the provider accepts it.
    ctx.tester.verdict = ProviderTestResult(True, "ok", 200)
    r = await ctx.h.call("PUT", "/api/tenant/connections/llm", tok, json_body=_put(0))
    assert r.status_code == 200, r.text
    assert await _connection_rows(ctx.h, tenant) == 1
    assert [e.action for e in await _audit(ctx.h, tenant)] == [
        "connection.test_failed",
        "connection.set",
    ]


async def test_concurrent_first_puts_lose_with_409_not_500(
    ctx: Ctx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two first PUTs (expected_version=0) race: both lock-read "no active row"
    (FOR UPDATE locks nothing that does not exist), both INSERT, and the partial
    unique index ``uq_tenant_connections_tenant_slot_active`` rejects the loser.
    That must be a 409 VERSION_CONFLICT, not an unhandled IntegrityError (500).

    The race is made deterministic: the winner commits during the loser's live
    test, and the loser's locked re-read is forced to see the pre-race state."""
    from src.database.repositories import connections_repository as repo_mod

    tenant = await ctx.h.tenant()
    tok = await _user(ctx.h, tenant)
    real_active = repo_mod.ConnectionsRepository._active
    stale_once = {"armed": False}

    async def _active(self, tenant_id, slot, *, lock):  # type: ignore[no-untyped-def]
        if lock and stale_once["armed"]:
            stale_once["armed"] = False
            return None  # the loser's snapshot: "no active connection yet"
        return await real_active(self, tenant_id, slot, lock=lock)

    monkeypatch.setattr(repo_mod.ConnectionsRepository, "_active", _active)

    class RacingTester(FakeTester):
        async def test(self, **kwargs: Any) -> ProviderTestResult:
            self.calls.append(kwargs)
            async with ctx.h.factory() as s:  # the winner commits mid-test
                await repo_mod.ConnectionsRepository(s).create_version(
                    tenant_id=tenant,
                    slot="llm",
                    kind="openai-compatible",
                    base_url=BASE_B,
                    secret=None,
                    expected_version=0,
                )
                await s.commit()
            stale_once["armed"] = True
            return self.verdict

    racing = RacingTester()
    ctx.h.app.dependency_overrides[get_provider_tester] = lambda: racing
    r = await ctx.h.call("PUT", "/api/tenant/connections/llm", tok, json_body=_put(0))
    assert r.status_code == 409, r.text
    err = r.json()["error"]
    assert err["code"] == "VERSION_CONFLICT"
    assert err["details"] == {"expected": 0, "current": 1}
    assert SECRET not in r.text
    # Only the winner's connection exists; the loser left no row and no audit.
    assert await _connection_rows(ctx.h, tenant) == 1
    assert (await _llm(ctx.h, tok))["connection"]["base_url"] == BASE_B
    assert await _audit(ctx.h, tenant) == []


async def test_version_conflict_is_409_and_does_not_test(ctx: Ctx) -> None:
    tenant = await ctx.h.tenant()
    tok = await _user(ctx.h, tenant)
    assert (
        await ctx.h.call("PUT", "/api/tenant/connections/llm", tok, json_body=_put(0))
    ).status_code == 200
    stale = await ctx.h.call(
        "PUT", "/api/tenant/connections/llm", tok, json_body=_put(0, model="m-2")
    )
    assert stale.status_code == 409, stale.text
    err = stale.json()["error"]
    assert err["code"] == "VERSION_CONFLICT"
    assert err["details"] == {"expected": 0, "current": 1}
    assert len(ctx.tester.calls) == 1  # the stale write never reached the provider
    fresh = await ctx.h.call(
        "PUT", "/api/tenant/connections/llm", tok, json_body=_put(1, model="m-2")
    )
    assert fresh.status_code == 200, fresh.text
    assert fresh.json()["connection"]["version"] == 2


async def test_non_writable_and_unknown_slots(ctx: Ctx) -> None:
    tenant = await ctx.h.tenant()
    tok = await _user(ctx.h, tenant)
    for method, body in (("PUT", _put(0)), ("DELETE", None), ("POST", None)):
        path = "/api/tenant/connections/scrape" + ("/test" if method == "POST" else "")
        r = await ctx.h.call(method, path, tok, json_body=body)
        assert r.status_code == 409, (method, r.text)
        assert r.json()["error"]["code"] == "SLOT_NOT_WRITABLE"
    r = await ctx.h.call("PUT", "/api/tenant/connections/nope", tok, json_body=_put(0))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "SLOT_NOT_FOUND"
    assert ctx.tester.calls == []


async def test_private_endpoints_are_refused_in_hosted_mode(ctx: Ctx) -> None:
    ctx.policy[0] = ProviderPolicySettings(ssrf_mode="hosted")
    tenant = await ctx.h.tenant()
    tok = await _user(ctx.h, tenant)
    for url in (
        "http://10.1.2.3/v1",
        "http://169.254.169.254/latest",
        "http://[::1]:8001/v1",
        "https://llm.test/v1",
    ):
        r = await ctx.h.call(
            "PUT", "/api/tenant/connections/llm", tok, json_body=_put(0, base_url=url)
        )
        assert r.status_code == 422, (url, r.text)
        assert r.json()["error"]["code"] == "EGRESS_REFUSED", url
    # A credential in the URL is refused and NOT echoed back.
    r = await ctx.h.call(
        "PUT",
        "/api/tenant/connections/llm",
        tok,
        json_body=_put(0, base_url="https://user:hunter2pw@llm-a.example.test/v1"),
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_BASE_URL"
    assert "hunter2pw" not in r.text
    assert ctx.tester.calls == []
    assert await _connection_rows(ctx.h, tenant) == 0


async def test_kind_policy_is_enforced(ctx: Ctx) -> None:
    ctx.policy[0] = ProviderPolicySettings(
        ssrf_mode="private_ok",
        egress_allowlist=frozenset({"llm-a.example.test"}),
        allowed_kinds=frozenset({"anthropic"}),
    )
    tenant = await ctx.h.tenant()
    tok = await _user(ctx.h, tenant)
    r = await ctx.h.call("PUT", "/api/tenant/connections/llm", tok, json_body=_put(0))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "PROVIDER_KIND_NOT_ALLOWED"
    r = await ctx.h.call(
        "PUT", "/api/tenant/connections/llm", tok, json_body=_put(0, base_url=None)
    )
    assert r.json()["error"]["code"] == "PROVIDER_KIND_NOT_ALLOWED"
    assert (await _llm(ctx.h, tok))["allowed_kinds"] == ["anthropic"]
    assert ctx.tester.calls == []


async def test_keeping_the_secret_across_an_endpoint_change_requires_reentry(ctx: Ctx) -> None:
    tenant = await ctx.h.tenant()
    tok = await _user(ctx.h, tenant)
    assert (
        await ctx.h.call("PUT", "/api/tenant/connections/llm", tok, json_body=_put(0))
    ).status_code == 200
    moved = await ctx.h.call(
        "PUT",
        "/api/tenant/connections/llm",
        tok,
        json_body=_put(1, base_url=BASE_B, secret=...),
    )
    assert moved.status_code == 422, moved.text
    assert moved.json()["error"]["code"] == "SECRET_REENTRY_REQUIRED"
    # The stored key was never sent to the new host.
    assert [c["base_url"] for c in ctx.tester.calls] == [BASE_A]
    # Same endpoint, new model: the key is kept and tested with the stored value.
    kept = await ctx.h.call(
        "PUT", "/api/tenant/connections/llm", tok, json_body=_put(1, model="m-9", secret=...)
    )
    assert kept.status_code == 200, kept.text
    assert kept.json()["connection"]["has_secret"] is True
    assert ctx.tester.calls[-1]["secret"] == SECRET


async def test_empty_secret_clears_it_and_is_audited(ctx: Ctx) -> None:
    tenant = await ctx.h.tenant()
    tok = await _user(ctx.h, tenant)
    await ctx.h.call("PUT", "/api/tenant/connections/llm", tok, json_body=_put(0))
    r = await ctx.h.call("PUT", "/api/tenant/connections/llm", tok, json_body=_put(1, secret=""))
    assert r.status_code == 200, r.text
    conn = r.json()["connection"]
    assert conn["has_secret"] is False and conn["key_fingerprint"] is None
    assert ctx.tester.calls[-1]["secret"] is None
    assert [e.action for e in await _audit(ctx.h, tenant)] == [
        "connection.set",
        "connection.set",
        "connection.clear_secret",
    ]


async def test_missing_keyring_is_503_and_nothing_is_tested(ctx: Ctx) -> None:
    ctx.keyring[0] = Keyring(())
    tenant = await ctx.h.tenant()
    tok = await _user(ctx.h, tenant)
    r = await ctx.h.call("PUT", "/api/tenant/connections/llm", tok, json_body=_put(0))
    assert r.status_code == 503, r.text
    assert r.json()["error"]["code"] == "PROVIDER_KEYRING_UNAVAILABLE"
    assert ctx.tester.calls == []
    assert await _connection_rows(ctx.h, tenant) == 0


# ─── POST /test ──────────────────────────────────────────────────────────


async def test_retest_of_the_stored_version_records_and_audits(ctx: Ctx) -> None:
    tenant = await ctx.h.tenant()
    tok = await _user(ctx.h, tenant)
    await ctx.h.call("PUT", "/api/tenant/connections/llm", tok, json_body=_put(0))

    ctx.tester.verdict = ProviderTestResult(False, "llm answered HTTP 401", 401)
    r = await ctx.h.call("POST", "/api/tenant/connections/llm/test", tok)
    assert r.status_code == 200, r.text
    assert r.json() == {
        "ok": False,
        "probed": True,
        "message": "llm answered HTTP 401",
        "upstream_status": 401,
        "version": 1,
    }
    assert ctx.tester.calls[-1]["secret"] == SECRET
    conn = (await _llm(ctx.h, tok))["connection"]
    assert conn["last_test_ok"] is False
    assert conn["last_test_error"] == "llm answered HTTP 401"

    # A draft is tested with its own parameters and changes nothing stored.
    ctx.tester.verdict = ProviderTestResult(True, "ok", 200)
    other = "sk-other"  # pragma: allowlist secret
    draft = {"kind": "openai-compatible", "base_url": BASE_B, "model": "x", "secret": other}
    r = await ctx.h.call("POST", "/api/tenant/connections/llm/test", tok, json_body=draft)
    assert r.status_code == 200, r.text
    assert r.json()["version"] is None and r.json()["ok"] is True
    assert ctx.tester.calls[-1]["base_url"] == BASE_B
    assert ctx.tester.calls[-1]["secret"] == other
    assert (await _llm(ctx.h, tok))["connection"]["version"] == 1

    actions = [e.action for e in await _audit(ctx.h, tenant)]
    assert actions == ["connection.set", "connection.test_failed", "connection.test"]
    assert all(SECRET not in json.dumps(e.detail) for e in await _audit(ctx.h, tenant))


async def test_retest_without_a_connection_is_404(ctx: Ctx) -> None:
    tenant = await ctx.h.tenant()
    tok = await _user(ctx.h, tenant)
    r = await ctx.h.call("POST", "/api/tenant/connections/llm/test", tok)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "CONNECTION_NOT_FOUND"


# ─── DELETE ──────────────────────────────────────────────────────────────


async def test_delete_falls_back_to_the_deployment_default(ctx: Ctx) -> None:
    tenant = await ctx.h.tenant()
    tok = await _user(ctx.h, tenant)
    await ctx.h.call("PUT", "/api/tenant/connections/llm", tok, json_body=_put(0))
    r = await ctx.h.call("DELETE", "/api/tenant/connections/llm", tok)
    assert r.status_code == 204, r.text
    llm = await _llm(ctx.h, tok)
    assert llm["source"] == "deployment" and llm["connection"] is None
    assert [e.action for e in await _audit(ctx.h, tenant)][-1] == "connection.delete"
    again = await ctx.h.call("DELETE", "/api/tenant/connections/llm", tok)
    assert again.status_code == 404
    # A new connection after delete starts a fresh one at version 1.
    r = await ctx.h.call("PUT", "/api/tenant/connections/llm", tok, json_body=_put(0))
    assert r.status_code == 200, r.text
    assert r.json()["connection"]["version"] == 1


# ─── live-test rate limit ────────────────────────────────────────────────


async def test_put_and_post_share_one_per_tenant_live_test_budget(ctx: Ctx) -> None:
    ctx.policy[0] = _PRIVATE_OK.model_copy(update={"test_rate_limit": 3})
    a, b = await ctx.h.tenant(), await ctx.h.tenant()
    tok_a, tok_b = await _user(ctx.h, a), await _user(ctx.h, b)

    # 1: PUT live test.
    r = await ctx.h.call("PUT", "/api/tenant/connections/llm", tok_a, json_body=_put(0))
    assert r.status_code == 200, r.text
    # A request refused BEFORE the live test (stale version) spends nothing.
    r = await ctx.h.call("PUT", "/api/tenant/connections/llm", tok_a, json_body=_put(0))
    assert r.status_code == 409, r.text
    # 2 + 3: POST re-test, PUT live test.
    r = await ctx.h.call("POST", "/api/tenant/connections/llm/test", tok_a)
    assert r.status_code == 200, r.text
    r = await ctx.h.call(
        "PUT", "/api/tenant/connections/llm", tok_a, json_body=_put(1, base_url=BASE_B)
    )
    assert r.status_code == 200, r.text
    assert len(ctx.tester.calls) == 3

    # 4th on EITHER route → 429 with Retry-After, and the tester is never reached.
    for method, path, body in (
        ("POST", "/api/tenant/connections/llm/test", None),
        ("PUT", "/api/tenant/connections/llm", _put(2)),
    ):
        r = await ctx.h.call(method, path, tok_a, json_body=body)
        assert r.status_code == 429, (method, r.text)
        assert r.json()["error"]["code"] == "PROVIDER_TEST_RATE_LIMITED"
        retry_after = int(r.headers["Retry-After"])
        assert 1 <= retry_after <= 60
        assert r.json()["error"]["details"]["retry_after"] == retry_after
    assert len(ctx.tester.calls) == 3
    # The refused PUT persisted nothing: still version 2.
    assert (await _llm(ctx.h, tok_a))["connection"]["version"] == 2

    # The budget is per tenant: B is unaffected.
    r = await ctx.h.call("PUT", "/api/tenant/connections/llm", tok_b, json_body=_put(0))
    assert r.status_code == 200, r.text
    assert len(ctx.tester.calls) == 4


async def test_rate_limit_fails_open_without_redis(ctx: Ctx) -> None:
    ctx.policy[0] = _PRIVATE_OK.model_copy(update={"test_rate_limit": 1})
    ctx.h.app.state.redis = None
    tenant = await ctx.h.tenant()
    tok = await _user(ctx.h, tenant)
    r = await ctx.h.call("PUT", "/api/tenant/connections/llm", tok, json_body=_put(0))
    assert r.status_code == 200, r.text
    r = await ctx.h.call("POST", "/api/tenant/connections/llm/test", tok)
    assert r.status_code == 200, r.text
    assert len(ctx.tester.calls) == 2


# ─── tenant GPU slots (asr/vlm writable, nonagentic_endpoint) ─────────────


async def test_gpu_slot_writes_are_tenant_admin_only_and_probe_the_new_kind(ctx: Ctx) -> None:
    tenant = await ctx.h.tenant()
    member = await _user(ctx.h, tenant, "member")
    body = _put(0, kind="nonagentic_endpoint", base_url="https://llm-a.example.test/asr",
                model=..., secret=SECRET)  # fmt: skip
    for method, payload in (("PUT", body), ("DELETE", None)):
        r = await ctx.h.call(method, "/api/tenant/connections/asr", member, json_body=payload)
        assert r.status_code == 403, (method, r.text)
    assert ctx.tester.calls == []

    admin = await _user(ctx.h, tenant)
    r = await ctx.h.call("PUT", "/api/tenant/connections/asr", admin, json_body=body)
    assert r.status_code == 200, r.text
    assert SECRET not in r.text
    conn = r.json()["connection"]
    assert (conn["kind"], conn["has_secret"]) == ("nonagentic_endpoint", True)
    [call] = ctx.tester.calls
    assert (call["slot"], call["kind"]) == ("asr", "nonagentic_endpoint")

    # The token is optional for this kind; the base URL is not.
    no_url = _put(0, kind="nonagentic_endpoint", base_url=None, model=..., secret=...)
    r = await ctx.h.call("PUT", "/api/tenant/connections/tts", admin, json_body=no_url)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "BASE_URL_REQUIRED"
    no_key = _put(0, kind="nonagentic_endpoint", base_url="https://llm-a.example.test/tts",
                  model=..., secret=...)  # fmt: skip
    r = await ctx.h.call("PUT", "/api/tenant/connections/tts", admin, json_body=no_key)
    assert r.status_code == 200, r.text

    vlm = _put(0, base_url="https://llm-a.example.test/v1", model="qwen-vl")
    r = await ctx.h.call("PUT", "/api/tenant/connections/vlm", admin, json_body=vlm)
    assert r.status_code == 200, r.text
