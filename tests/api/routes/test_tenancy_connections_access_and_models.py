"""LLM access posture and model discovery on the REAL ``/api/tenant/connections``
request path, plus the 2026-10-20 owner decision that these connections are w2v's
OWN store: nothing is mirrored into worker_management and no response carries a
``runtime_sync`` field.

Real RS256 JWT → AuthMiddleware → TenancyMiddleware (membership) → the router →
real rows in Postgres. Faked: the live provider test and the model lister.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import fakeredis.aioredis
import httpx
import pytest
from sqlalchemy import func, select

from src.api.routes.tenancy.connections import (
    get_connections_byo_posture,
    get_connections_keyring,
    get_connections_policy,
    get_model_lister,
    get_provider_tester,
)
from src.database.models import TenantConnection
from src.services.providers.byo_posture import ByoMode, StaticByoPosture
from src.services.providers.crypto import Keyring
from src.services.providers.settings import ProviderPolicySettings
from src.services.providers.tester import ModelListResult, ProviderTestResult
from tests.api.routes._tenancy_harness import (  # noqa: F401 — fixtures
    Harness,
    harness_fixture,
    signing_key_fixture,
)

pytestmark = pytest.mark.asyncio

SECRET = "sk-live-NEVER-ECHO-41ab9c"  # pragma: allowlist secret
BASE = "https://llm-a.example.test/v1"
_POLICY = ProviderPolicySettings(
    ssrf_mode="private_ok", egress_allowlist=frozenset({"llm-a.example.test"})
)


@dataclass
class FakeTester:
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def test(self, **kwargs: Any) -> ProviderTestResult:
        self.calls.append(kwargs)
        return ProviderTestResult(True, "answered HTTP 200", 200)


@dataclass
class FakeLister:
    result: ModelListResult = field(
        default_factory=lambda: ModelListResult(["gpt-a", "gpt-b"], None, 200)
    )
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def list_models(self, **kwargs: Any) -> ModelListResult:
        self.calls.append(kwargs)
        return self.result


@dataclass
class Ctx:
    h: Harness
    tester: FakeTester
    lister: FakeLister
    posture: list[StaticByoPosture]


@pytest.fixture
def ctx(harness: Harness) -> Ctx:
    c = Ctx(
        h=harness,
        tester=FakeTester(),
        lister=FakeLister(),
        posture=[StaticByoPosture("local_allowed")],
    )
    keyring = Keyring([("k1", os.urandom(32))])
    app = harness.app
    app.dependency_overrides[get_provider_tester] = lambda: c.tester
    app.dependency_overrides[get_model_lister] = lambda: c.lister
    app.dependency_overrides[get_connections_policy] = lambda: _POLICY
    app.dependency_overrides[get_connections_keyring] = lambda: keyring
    app.dependency_overrides[get_connections_byo_posture] = lambda: c.posture[0]
    app.state.redis = fakeredis.aioredis.FakeRedis()
    return c


async def _admin(h: Harness) -> tuple[uuid.UUID, str]:
    tenant = await h.tenant()
    sub = uuid.uuid7()
    await h.member(sub, tenant, "tenant_admin", email=f"a-{sub}@acme.test")
    return tenant, h.token(sub)


def _put(expected: int = 0, **over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "kind": "openai-compatible",
        "base_url": BASE,
        "model": "m-1",
        "secret": SECRET,
        "expected_version": expected,
    }
    body.update(over)
    return body


async def _slot(h: Harness, tok: str, slot: str) -> dict[str, Any]:
    r = await h.call("GET", "/api/tenant/connections", tok)
    assert r.status_code == 200, r.text
    return next(i for i in r.json()["items"] if i["slot"] == slot)


async def _rows(h: Harness, tenant: uuid.UUID) -> int:
    async with h.factory() as s:
        return (
            await s.execute(
                select(func.count())
                .select_from(TenantConnection)
                .where(TenantConnection.tenant_id == tenant, TenantConnection.status == "active")
            )
        ).scalar_one()


# ─── llm_access ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("mode", "label"),
    [("local_allowed", "Platform models allowed")],
)
async def test_get_carries_the_llm_access_posture(ctx: Ctx, mode: ByoMode, label: str) -> None:
    ctx.posture[0] = StaticByoPosture(mode)
    _tenant, tok = await _admin(ctx.h)
    r = await ctx.h.call("GET", "/api/tenant/connections", tok)
    assert r.status_code == 200, r.text
    assert r.json()["llm_access"] == {"mode": mode, "label": label}


@pytest.mark.parametrize("slot", ["llm"])
async def test_platform_only_refuses_setting_a_byo_slot(ctx: Ctx, slot: str) -> None:
    """Every ``byo_governed`` slot (2026-10-22: video_gen and ocr included)."""
    ctx.posture[0] = StaticByoPosture("platform_only")
    tenant, tok = await _admin(ctx.h)
    r = await ctx.h.call("PUT", f"/api/tenant/connections/{slot}", tok, json_body=_put())
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "tenant_byo_disabled"
    assert await _rows(ctx.h, tenant) == 0
    assert ctx.tester.calls == [], "a refused write must not send the key anywhere"


@pytest.mark.parametrize("url", ["https://studio.nonagentic.ai/v1"])
async def test_a_connection_pointing_at_w2v_itself_is_refused(ctx: Ctx, url: str) -> None:
    """Loop guard: w2v's own public hosts are never a tenant provider endpoint."""
    tenant, tok = await _admin(ctx.h)
    r = await ctx.h.call("PUT", "/api/tenant/connections/llm", tok, json_body=_put(base_url=url))
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "EGRESS_REFUSED"
    assert await _rows(ctx.h, tenant) == 0
    assert ctx.tester.calls == []


async def test_platform_only_refuses_the_bulk_write_too(ctx: Ctx) -> None:
    ctx.posture[0] = StaticByoPosture("platform_only")
    tenant, tok = await _admin(ctx.h)
    r = await ctx.h.call(
        "PUT", "/api/tenant/connections", tok, json_body={"connections": {"llm": _put()}}
    )
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "tenant_byo_disabled"
    assert await _rows(ctx.h, tenant) == 0


async def test_platform_only_still_allows_deleting_and_the_same_put_passes_otherwise(
    ctx: Ctx,
) -> None:
    """Non-vacuity: the identical PUT succeeds under local_allowed; after the flip
    to platform_only the connection can still be removed."""
    tenant, tok = await _admin(ctx.h)
    r = await ctx.h.call("PUT", "/api/tenant/connections/llm", tok, json_body=_put())
    assert r.status_code == 200, r.text
    ctx.posture[0] = StaticByoPosture("platform_only")
    r = await ctx.h.call("DELETE", "/api/tenant/connections/llm", tok)
    assert r.status_code == 204, r.text
    assert await _rows(ctx.h, tenant) == 0


# ─── model discovery ────────────────────────────────────────────────────


async def test_models_lists_a_draft_endpoint(ctx: Ctx) -> None:
    _tenant, tok = await _admin(ctx.h)
    r = await ctx.h.call(
        "POST",
        "/api/tenant/connections/llm/models",
        tok,
        json_body={"kind": "openai-compatible", "base_url": BASE, "secret": SECRET},
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"models": ["gpt-a", "gpt-b"]}
    assert ctx.lister.calls == [
        {"slot": "llm", "kind": "openai-compatible", "base_url": BASE, "secret": SECRET}
    ]
    assert SECRET not in r.text


async def test_models_uses_the_stored_secret_when_omitted(ctx: Ctx) -> None:
    _tenant, tok = await _admin(ctx.h)
    assert (
        await ctx.h.call("PUT", "/api/tenant/connections/llm", tok, json_body=_put())
    ).status_code == 200
    r = await ctx.h.call(
        "POST",
        "/api/tenant/connections/llm/models",
        tok,
        json_body={"kind": "openai-compatible", "base_url": BASE},
    )
    assert r.status_code == 200, r.text
    assert ctx.lister.calls[-1]["secret"] == SECRET


async def test_models_never_carries_the_stored_secret_to_a_new_endpoint(ctx: Ctx) -> None:
    _tenant, tok = await _admin(ctx.h)
    assert (
        await ctx.h.call("PUT", "/api/tenant/connections/llm", tok, json_body=_put())
    ).status_code == 200
    r = await ctx.h.call(
        "POST",
        "/api/tenant/connections/llm/models",
        tok,
        json_body={"kind": "anthropic"},
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "SECRET_REENTRY_REQUIRED"
    assert ctx.lister.calls == []


async def test_models_refuses_an_egress_denied_endpoint_before_sending(ctx: Ctx) -> None:
    _tenant, tok = await _admin(ctx.h)
    r = await ctx.h.call(
        "POST",
        "/api/tenant/connections/llm/models",
        tok,
        json_body={
            "kind": "openai-compatible",
            "base_url": "https://not-allowed.example.test/v1",
            "secret": SECRET,
        },
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "EGRESS_REFUSED"
    assert ctx.lister.calls == []


async def test_models_upstream_failure_is_models_unavailable(ctx: Ctx) -> None:
    ctx.lister.result = ModelListResult(None, "llm-a answered HTTP 401: rejected", 401)
    _tenant, tok = await _admin(ctx.h)
    r = await ctx.h.call(
        "POST",
        "/api/tenant/connections/llm/models",
        tok,
        json_body={"kind": "openai-compatible", "base_url": BASE, "secret": SECRET},
    )
    assert r.status_code == 422, r.text
    err = r.json()["error"]
    assert err["code"] == "models_unavailable"
    assert err["details"]["upstream_status"] == 401
    assert SECRET not in r.text


async def test_models_under_platform_only_is_tenant_byo_disabled(ctx: Ctx) -> None:
    """Same refusal as PUT: a Local-only tenant's own keys are disabled, so the
    draft key is never sent to the provider either."""
    ctx.posture[0] = StaticByoPosture("platform_only")
    _tenant, tok = await _admin(ctx.h)
    r = await ctx.h.call(
        "POST",
        "/api/tenant/connections/llm/models",
        tok,
        json_body={"kind": "openai-compatible", "base_url": BASE, "secret": SECRET},
    )
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "tenant_byo_disabled"
    assert ctx.lister.calls == []
    # Non-vacuity: the identical request lists under local_allowed.
    ctx.posture[0] = StaticByoPosture("local_allowed")
    r = await ctx.h.call(
        "POST",
        "/api/tenant/connections/llm/models",
        tok,
        json_body={"kind": "openai-compatible", "base_url": BASE, "secret": SECRET},
    )
    assert r.status_code == 200, r.text


# ─── no worker_management mirror (owner decision 2026-10-20) ───────────


async def test_no_response_carries_runtime_sync(ctx: Ctx) -> None:
    """GET, per-slot PUT and bulk PUT views no longer expose the retired WM mirror
    status — the frontend stopped reading it in the same change."""
    _tenant, tok = await _admin(ctx.h)
    r = await ctx.h.call("PUT", "/api/tenant/connections/llm", tok, json_body=_put())
    assert r.status_code == 200, r.text
    assert r.json()["connection"]["version"] == 1
    assert "runtime_sync" not in r.json() and "runtimeSync" not in r.json()
    r = await ctx.h.call(
        "PUT", "/api/tenant/connections", tok, json_body={"connections": {"vlm": _put()}}
    )
    assert r.status_code == 200, r.text
    assert r.json()["items"] and all("runtime_sync" not in i for i in r.json()["items"])
    r = await ctx.h.call("GET", "/api/tenant/connections", tok)
    assert r.status_code == 200, r.text
    assert all("runtime_sync" not in i and "runtimeSync" not in i for i in r.json()["items"])


async def test_writes_and_deletes_never_call_worker_management(
    ctx: Ctx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The only outbound request a connection write may make is the live provider
    test (faked here). Any real httpx request — a WM ``/tenant-config`` mirror —
    fails the test."""

    async def _no_network(_self: Any, request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected outbound request {request.method} {request.url}")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _no_network)
    tenant, tok = await _admin(ctx.h)
    r = await ctx.h.call("PUT", "/api/tenant/connections/llm", tok, json_body=_put())
    assert r.status_code == 200, r.text
    r = await ctx.h.call("DELETE", "/api/tenant/connections/llm", tok)
    assert r.status_code == 204, r.text
    assert await _rows(ctx.h, tenant) == 0
    # A repeated DELETE is a plain 404: there is no WM removal left to retry.
    r = await ctx.h.call("DELETE", "/api/tenant/connections/llm", tok)
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "CONNECTION_NOT_FOUND"
