"""P8 — bulk subtree ``PUT /api/tenant/connections`` + the requirements manifest route,
on the REAL request path.

Same stack as ``test_tenancy_connections.py``: real RS256 JWT → w2v_common
``AuthMiddleware`` → ``TenancyMiddleware`` (membership mode) → the REAL tenancy
routers → real rows in Postgres; the ``w2v_sk_`` verifier is registered so tenant API
keys authenticate exactly as in production. Faked: the JWKS socket and the live
provider test (a recorder with a per-slot verdict).

Non-vacuity: every "nothing persisted" case has a sibling that DOES persist through
the same body shape; "zero probes" reads the recorder's call list after a first write
that provably made calls.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import fakeredis.aioredis
import pytest
from sqlalchemy import func, select
from w2v_common.auth import register_api_key_verifier

from src.api.routes.provider_requirements import router as provider_requirements_router
from src.api.routes.tenancy.connections import (
    get_connections_keyring,
    get_connections_policy,
    get_provider_tester,
)
from src.auth.tenancy_middleware import make_api_key_verifier
from src.database.models import AdminAuditLog, TenantConnection, TenantConnectionVersion
from src.database.repositories.api_keys_repository import ApiKeysRepository
from src.database.repositories.connections_repository import ConnectionsRepository
from src.services.providers.crypto import Keyring
from src.services.providers.settings import ProviderPolicySettings
from src.services.providers.tester import ProviderTestResult
from tests.api.routes._tenancy_harness import (  # noqa: F401 — fixtures
    Harness,
    harness_fixture,
    signing_key_fixture,
)

pytestmark = pytest.mark.asyncio

SECRET = "sk-live-NEVER-ECHO-p8-bulk"  # pragma: allowlist secret
LLM_URL = "https://llm-a.example.test/v1"
TTS_URL = "https://tts-a.example.test/v1"
_POLICY = ProviderPolicySettings(
    ssrf_mode="private_ok",
    egress_allowlist=frozenset({"llm-a.example.test", "tts-a.example.test"}),
)


@dataclass
class SlotTester:
    """Records every live test; ``fail`` names the slots whose test fails."""

    fail: set[str] = field(default_factory=set)
    calls: list[dict[str, Any]] = field(default_factory=list)
    #: Runs once inside the probe window (between planning and persisting).
    during_probe: Callable[[], Awaitable[None]] | None = None

    async def test(self, **kwargs: Any) -> ProviderTestResult:
        self.calls.append(kwargs)
        hook, self.during_probe = self.during_probe, None
        if hook is not None:
            await hook()
        if kwargs["slot"] in self.fail:
            return ProviderTestResult(False, f"{kwargs['slot']} answered HTTP 401", 401)
        return ProviderTestResult(True, "answered HTTP 200", 200)


@dataclass
class Ctx:
    h: Harness
    tester: SlotTester


@pytest.fixture
def ctx(harness: Harness) -> Iterator[Ctx]:
    c = Ctx(h=harness, tester=SlotTester())
    keyring = Keyring([("k1", os.urandom(32))])
    harness.app.dependency_overrides[get_provider_tester] = lambda: c.tester
    harness.app.dependency_overrides[get_connections_policy] = lambda: _POLICY
    harness.app.dependency_overrides[get_connections_keyring] = lambda: keyring
    harness.app.state.redis = fakeredis.aioredis.FakeRedis()
    harness.app.include_router(provider_requirements_router)
    register_api_key_verifier(make_api_key_verifier(harness.factory))
    yield c
    register_api_key_verifier(None)


async def _admin(h: Harness, tenant: uuid.UUID, role: str = "tenant_admin") -> str:
    sub = uuid.uuid7()
    await h.member(sub, tenant, role, email=f"{role}-{sub}@acme.test")
    return h.token(sub)


async def _api_key(h: Harness, tenant: uuid.UUID, scopes: list[str]) -> tuple[uuid.UUID, str]:
    async with h.factory() as s:
        row, full_key = await ApiKeysRepository(s).create(
            tenant_id=tenant, name="merqi", created_by=None, scopes=scopes
        )
        key_id = row.id
        await s.commit()
    return key_id, full_key


def _tree(expected: dict[str, int] | None = None, **over: Any) -> dict[str, Any]:
    """llm + tts + image, all valid; ``expected`` = per-slot expected_version."""
    exp = expected or {}
    conns: dict[str, Any] = {
        "llm": {
            "kind": "openai-compatible",
            "base_url": LLM_URL,
            "model": "m-1",
            "secret": SECRET,
            "expected_version": exp.get("llm", 0),
        },
        "tts": {
            "kind": "openai-compatible",
            "base_url": TTS_URL,
            "model": "kokoro",
            "options": {"voice": "af_heart"},
            "expected_version": exp.get("tts", 0),
        },
        "image": {"kind": "openai", "secret": SECRET, "expected_version": exp.get("image", 0)},
    }
    conns.update(over)
    return {"connections": conns}


async def _count(h: Harness, tenant: uuid.UUID, model: type) -> int:
    async with h.factory() as s:
        stmt = select(func.count()).select_from(model)
        if model is TenantConnectionVersion:
            stmt = stmt.join(
                TenantConnection, TenantConnection.id == TenantConnectionVersion.connection_id
            )
        return (await s.execute(stmt.where(TenantConnection.tenant_id == tenant))).scalar_one()


async def _audit(h: Harness, tenant: uuid.UUID) -> list[AdminAuditLog]:
    async with h.factory() as s:
        rows = await s.execute(
            select(AdminAuditLog)
            .where(AdminAuditLog.target_tenant_id == tenant)
            .order_by(AdminAuditLog.created_at)
        )
        return list(rows.scalars().all())


# ─── all or nothing ──────────────────────────────────────────────────────


async def test_one_failing_test_persists_nothing(ctx: Ctx) -> None:
    tenant = await ctx.h.tenant()
    tok = await _admin(ctx.h, tenant)
    ctx.tester.fail = {"image"}

    r = await ctx.h.call("PUT", "/api/tenant/connections", tok, json_body=_tree())
    assert r.status_code == 422, r.text
    err = r.json()["error"]
    assert err["code"] == "PROVIDER_TEST_FAILED"
    assert set(err["details"]["slots"]) == {"image"}
    assert sorted(c["slot"] for c in ctx.tester.calls) == ["image", "llm", "tts"]
    assert await _count(ctx.h, tenant, TenantConnection) == 0
    assert await _count(ctx.h, tenant, TenantConnectionVersion) == 0
    assert SECRET not in r.text
    # Only the failed attempt is audited — no connection.* write.
    assert [a.action for a in await _audit(ctx.h, tenant)] == ["connection.test_failed"]

    # Control: the SAME tree with every test passing persists all three, version 1.
    ctx.tester.fail = set()
    r = await ctx.h.call("PUT", "/api/tenant/connections", tok, json_body=_tree())
    assert r.status_code == 200, r.text
    items = {i["slot"]: i for i in r.json()["items"]}
    assert {s: i["change"] for s, i in items.items()} == {
        "llm": "created",
        "tts": "created",
        "image": "created",
    }
    assert all(i["connection"]["version"] == 1 for i in items.values())
    assert all(i["test"]["ok"] is True for i in items.values())
    assert await _count(ctx.h, tenant, TenantConnection) == 3
    [bulk] = [a for a in await _audit(ctx.h, tenant) if a.action == "connection.bulk_set"]
    changes = bulk.detail["changes"]
    assert isinstance(changes, list)
    assert {c["slot"] for c in changes} == {"llm", "tts", "image"}
    assert SECRET not in str(bulk.detail)


async def test_reput_of_the_identical_tree_is_zero_probes_zero_versions(ctx: Ctx) -> None:
    tenant = await ctx.h.tenant()
    tok = await _admin(ctx.h, tenant)
    r = await ctx.h.call("PUT", "/api/tenant/connections", tok, json_body=_tree())
    assert r.status_code == 200, r.text
    assert len(ctx.tester.calls) == 3  # the first write really probed
    versions, audits = (
        await _count(ctx.h, tenant, TenantConnectionVersion),
        len(await _audit(ctx.h, tenant)),
    )
    ones = {"llm": 1, "tts": 1, "image": 1}

    # Secrets re-sent verbatim (same fingerprint) AND secrets omitted (keep): both no-ops.
    for body in (_tree(ones), _tree(ones, image={"kind": "openai", "expected_version": 1})):
        r = await ctx.h.call("PUT", "/api/tenant/connections", tok, json_body=body)
        assert r.status_code == 200, r.text
        assert {i["change"] for i in r.json()["items"]} == {"unchanged"}
        assert all(i["test"] is None for i in r.json()["items"])
    assert len(ctx.tester.calls) == 3, "a no-op re-save must not re-probe"
    assert await _count(ctx.h, tenant, TenantConnectionVersion) == versions
    assert len(await _audit(ctx.h, tenant)) == audits

    # Control: one real change → exactly one probe, one new version.
    changed = _tree(
        ones, llm={**_tree()["connections"]["llm"], "model": "m-2", "expected_version": 1}
    )
    r = await ctx.h.call("PUT", "/api/tenant/connections", tok, json_body=changed)
    assert r.status_code == 200, r.text
    assert {i["slot"]: i["change"] for i in r.json()["items"]}["llm"] == "updated"
    assert [c["slot"] for c in ctx.tester.calls[3:]] == ["llm"]
    assert await _count(ctx.h, tenant, TenantConnectionVersion) == versions + 1


async def test_stale_expected_version_is_409_and_persists_nothing(ctx: Ctx) -> None:
    tenant = await ctx.h.tenant()
    tok = await _admin(ctx.h, tenant)
    r = await ctx.h.call("PUT", "/api/tenant/connections", tok, json_body=_tree())
    assert r.status_code == 200, r.text
    calls = len(ctx.tester.calls)
    stale = _tree({"llm": 1, "tts": 0, "image": 1})  # tts is at version 1, not 0
    stale["connections"]["llm"]["model"] = "m-9"
    r = await ctx.h.call("PUT", "/api/tenant/connections", tok, json_body=stale)
    assert r.status_code == 409, r.text
    assert r.json()["error"]["details"]["slot"] == "tts"
    assert len(ctx.tester.calls) == calls
    assert await _count(ctx.h, tenant, TenantConnectionVersion) == 3


def _concurrent_update(h: Harness, tenant: uuid.UUID, slot: str) -> Callable[[], Awaitable[None]]:
    """Another writer moves ``slot`` from version 1 to 2 on its OWN session."""

    async def _write() -> None:
        async with h.factory() as s:
            await ConnectionsRepository(s).create_version(
                tenant_id=tenant,
                slot=slot,
                kind="openai-compatible",
                base_url=TTS_URL if slot == "tts" else LLM_URL,
                model="concurrent",
                options={"voice": "af_heart"} if slot == "tts" else {},
                secret=None,
                expected_version=1,
            )
            await s.commit()

    return _write


async def _live_versions(h: Harness, tenant: uuid.UUID) -> dict[str, int]:
    async with h.factory() as s:
        rows = await ConnectionsRepository(s).list_for_tenant(tenant)
    return {conn.slot: ver.version for conn, ver in rows}


async def test_delete_of_a_version_updated_during_the_probe_is_409(ctx: Ctx) -> None:
    tenant = await ctx.h.tenant()
    tok = await _admin(ctx.h, tenant)
    assert (
        await ctx.h.call("PUT", "/api/tenant/connections", tok, json_body=_tree())
    ).status_code == 200
    llm = {**_tree()["connections"]["llm"], "model": "m-2", "expected_version": 1}
    # tts is deleted (seen at v1); while llm is being probed, tts moves to v2.
    ctx.tester.during_probe = _concurrent_update(ctx.h, tenant, "tts")
    r = await ctx.h.call(
        "PUT",
        "/api/tenant/connections",
        tok,
        json_body={"connections": {"llm": llm, "tts": None}},
    )
    assert r.status_code == 409, r.text
    err = r.json()["error"]
    assert (err["code"], err["details"]["slot"]) == ("VERSION_CONFLICT", "tts")
    assert (err["details"]["expected"], err["details"]["current"]) == (1, 2)
    # Nothing persisted: tts is NOT deleted (the concurrent v2 survives), llm not updated.
    assert await _live_versions(ctx.h, tenant) == {"image": 1, "llm": 1, "tts": 2}

    # Control: with no concurrent writer the same shape deletes (tts now at v2).
    r = await ctx.h.call(
        "PUT",
        "/api/tenant/connections",
        tok,
        json_body={"connections": {"llm": llm, "tts": None}},
    )
    assert r.status_code == 200, r.text
    assert await _live_versions(ctx.h, tenant) == {"image": 1, "llm": 2}


async def test_write_time_conflict_names_the_slot(ctx: Ctx) -> None:
    tenant = await ctx.h.tenant()
    tok = await _admin(ctx.h, tenant)
    assert (
        await ctx.h.call("PUT", "/api/tenant/connections", tok, json_body=_tree())
    ).status_code == 200
    ctx.tester.during_probe = _concurrent_update(ctx.h, tenant, "llm")
    llm = {**_tree()["connections"]["llm"], "model": "m-2", "expected_version": 1}
    r = await ctx.h.call(
        "PUT", "/api/tenant/connections", tok, json_body={"connections": {"llm": llm}}
    )
    assert r.status_code == 409, r.text
    err = r.json()["error"]
    assert (err["code"], err["details"]["slot"]) == ("VERSION_CONFLICT", "llm")
    assert await _live_versions(ctx.h, tenant) == {"image": 1, "llm": 2, "tts": 1}
    async with ctx.h.factory() as s:
        cur = await ConnectionsRepository(s).get_current(tenant, "llm")
    assert cur is not None and cur[1].model == "concurrent", "the concurrent write wins"


async def test_null_deletes_back_to_the_deployment_default(ctx: Ctx) -> None:
    tenant = await ctx.h.tenant()
    tok = await _admin(ctx.h, tenant)
    assert (
        await ctx.h.call("PUT", "/api/tenant/connections", tok, json_body=_tree())
    ).status_code == 200
    r = await ctx.h.call(
        "PUT",
        "/api/tenant/connections",
        tok,
        json_body={"connections": {"tts": None, "scrape": None}},
    )
    # scrape is platform-only: the whole request is refused, tts NOT deleted.
    assert r.status_code == 422, r.text
    assert set(r.json()["error"]["details"]["slots"]) == {"scrape"}
    assert await _count(ctx.h, tenant, TenantConnection) == 3
    r = await ctx.h.call(
        "PUT", "/api/tenant/connections", tok, json_body={"connections": {"tts": None}}
    )
    assert r.status_code == 200, r.text
    [item] = r.json()["items"]
    assert (item["change"], item["source"], item["connection"]) == ("deleted", "deployment", None)


@pytest.mark.parametrize(
    ("body", "code"),
    [({"connections": {"nope": None}}, "BULK_VALIDATION_FAILED")],
    ids=["unknown-slot"],
)
async def test_unknown_or_grant_slot_or_dependencies_is_422(
    ctx: Ctx, body: dict[str, Any], code: str | None
) -> None:
    tenant = await ctx.h.tenant()
    tok = await _admin(ctx.h, tenant)
    full = {**body, "connections": {**_tree()["connections"], **body["connections"]}}
    r = await ctx.h.call("PUT", "/api/tenant/connections", tok, json_body=full)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == (code or "DEPENDENCY_NOT_CONFIGURABLE")
    assert ctx.tester.calls == []
    assert await _count(ctx.h, tenant, TenantConnection) == 0


# ─── auth: tenant_admin OR an API key with connections:write ─────────────


async def test_api_key_needs_the_connections_write_scope(ctx: Ctx) -> None:
    tenant = await ctx.h.tenant()
    _plain_id, plain = await _api_key(ctx.h, tenant, [])
    scoped_id, scoped = await _api_key(ctx.h, tenant, ["connections:write"])

    r = await ctx.h.call("PUT", "/api/tenant/connections", plain, json_body=_tree())
    assert r.status_code == 403, r.text
    assert r.json()["error"]["code"] == "API_KEY_SCOPE_REQUIRED"
    member = await _admin(ctx.h, tenant, "member")
    r = await ctx.h.call("PUT", "/api/tenant/connections", member, json_body=_tree())
    assert r.status_code == 403, r.text
    assert ctx.tester.calls == []
    assert await _count(ctx.h, tenant, TenantConnection) == 0

    r = await ctx.h.call("PUT", "/api/tenant/connections", scoped, json_body=_tree())
    assert r.status_code == 200, r.text
    [bulk] = [a for a in await _audit(ctx.h, tenant) if a.action == "connection.bulk_set"]
    assert bulk.actor_subject_id == scoped_id, "actor = the key id for an API key"
    assert bulk.detail["actor_type"] == "api_key"
    assert bulk.detail["key_id"] == str(scoped_id)
    # The scope is NOT a tenant_admin grant: per-slot PUT stays tenant_admin only.
    r = await ctx.h.call(
        "PUT",
        "/api/tenant/connections/llm",
        scoped,
        json_body={**_tree()["connections"]["llm"], "expected_version": 1, "model": "m-3"},
    )
    assert r.status_code == 403, r.text


async def test_a_scoped_key_writes_only_its_own_tenant(ctx: Ctx) -> None:
    a, b = await ctx.h.tenant(), await ctx.h.tenant()
    _kid, key_a = await _api_key(ctx.h, a, ["connections:write"])
    r = await ctx.h.call(
        "PUT",
        "/api/tenant/connections",
        key_a,
        json_body=_tree(),
        headers={"X-Active-Tenant-Id": str(b)},
    )
    # Whatever the header says, a key only ever acts as its own tenant: the write
    # lands on A (the key's tenant), never on B.
    assert r.status_code == 200, r.text
    assert await _count(ctx.h, b, TenantConnection) == 0
    assert await _count(ctx.h, a, TenantConnection) == 3


# ─── the requirements manifest route ─────────────────────────────────────


async def test_manifest_is_readable_by_member_and_any_api_key(ctx: Ctx) -> None:
    tenant = await ctx.h.tenant()
    member = await _admin(ctx.h, tenant, "member")
    _kid, key = await _api_key(ctx.h, tenant, [])
    for bearer in (member, key):
        r = await ctx.h.call("GET", "/api/provider-requirements", bearer)
        assert r.status_code == 200, r.text
        body = r.json()
        assert [s["slot"] for s in body["slots"]] == [
            "llm",
            "vlm",
            "tts",
            "asr",
            "image",
            "video_gen",
            "ocr",
            "publish.youtube",
            "device_farm",
        ]
        assert [d["name"] for d in body["dependencies"]] == ["webscraper", "media_discovery"]
    r = await ctx.h.call("GET", "/api/provider-requirements", "not-a-token")
    assert r.status_code == 401, r.text
