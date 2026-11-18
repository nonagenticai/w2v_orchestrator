"""P3 WP-LLM — the orchestrator's WMClient dispatches a run's ``llm`` slot.

* deployment / unpinned → the platform backend (``LLM_BACKEND``), never the tenant
  resolver;
* tenant → in-process pydantic-ai on the pinned connection: the fake provider must
  receive the pinned endpoint (connected by IP, Host/SNI = the name), the model, the
  key and the system prompt; structured output parsed; errors mapped into the
  orchestrator's taxonomy without ever leaking the key.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Iterator
from typing import Any
from uuid import UUID, uuid7

import httpx2
import pytest
from w2v_common.auth.identity_context import set_correlation_context, set_identity_context

from src.auth.run_providers import bind_run_provider_slots, clear_run_provider_slots
from src.models.failure_class import CUSTOMER_INPUT, EXTERNAL
from src.services.llm import WMClient, dispatch, in_process
from src.services.llm.in_process import PinnedTransport
from src.services.providers.resolver import ResolvedSlot, SlotRef
from src.services.stages.upstream import (
    PermanentUpstreamError,
    TransientUpstreamError,
    UpstreamAuthError,
)
from tests.services.llm._fakes import (
    CONN_ID,
    SECRET,
    FakeProvider,
    anthropic_reply,
    openai_reply,
    status_reply,
    tenant_slot,
)

pytestmark = pytest.mark.asyncio

TENANT = UUID("01990000-0000-7000-8000-0000000000aa")
RUN_ID = uuid7()
TENANT_PIN = {
    "provider_slots": {
        "llm": {
            "source": "tenant",
            "kind": "openai-compatible",
            "connection_id": str(CONN_ID),
            "version": 3,
        }
    }
}
DEPLOYMENT_PIN = {"provider_slots": {"llm": {"source": "deployment", "kind": "nonagentic-wm"}}}


@pytest.fixture(autouse=True)
def _context() -> Iterator[None]:
    clear_run_provider_slots()
    set_identity_context(str(TENANT), None, None)
    set_correlation_context(str(RUN_ID), "demo_video_script")
    yield
    clear_run_provider_slots()
    set_correlation_context(None, None)


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the in-process retry sleeps instead of sleeping."""
    slept: list[float] = []

    async def _sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(in_process, "_sleep", _sleep)
    return slept


@pytest.fixture
def usage_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def _record(**kwargs: Any) -> bool:
        calls.append(kwargs)
        return True

    monkeypatch.setattr(dispatch, "record_in_process_usage", _record)
    return calls


def _use_slot(monkeypatch: pytest.MonkeyPatch, slot: ResolvedSlot) -> list[SlotRef]:
    seen: list[SlotRef] = []

    async def _resolve(tenant_id: UUID, ref: SlotRef) -> ResolvedSlot:
        assert tenant_id == TENANT
        seen.append(ref)
        return slot

    monkeypatch.setattr(dispatch, "resolve_pinned_llm", _resolve)
    return seen


def _use_provider(monkeypatch: pytest.MonkeyPatch, fake: FakeProvider) -> None:
    monkeypatch.setattr(in_process, "inner_transport_factory", fake.transport)


CALL: dict[str, Any] = {
    "worker_name": "script-doctor-critique",
    "message": "critique this",
    "system_prompt": "be terse",
    "output_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
    "model_override": "some-model",
    "work_tier": "high",
    "max_tokens": 1200,
    "enable_reasoning": False,
    "context": {"k": "v"},
}


# ── deployment / unpinned ───────────────────────────────────────────────────
#
# The platform default (no pin, a legacy snapshot, or the ``nonagentic-wm``
# deployment kind) is pinned in ``test_vllm_direct.py`` / ``test_ir_direct.py``;
# the worker_management request it used to send was deleted in w2v_common 0.109.0.


@pytest.mark.parametrize("pins", [None])
async def test_unpinned_or_deployment_slot_never_touches_the_tenant_resolver(
    monkeypatch: pytest.MonkeyPatch, pins: dict[str, Any] | None
) -> None:
    async def _never(*_a: object, **_k: object) -> ResolvedSlot:
        raise AssertionError("a deployment slot must not touch the tenant resolver")

    seen: list[str] = []

    async def _platform(_settings: object, request: Any, *, timeout: float) -> Any:
        seen.append(request.worker_name)
        return dispatch.WMResponse({"ok": True}, "object", None, None, None)

    async def _no_ledger(**_k: object) -> bool:
        return True

    monkeypatch.setattr(dispatch, "resolve_pinned_llm", _never)
    monkeypatch.setattr(dispatch.vllm_direct, "call_platform_vllm", _platform)
    monkeypatch.setattr(dispatch, "record_platform_vllm_usage", _no_ledger)
    if pins is not None:
        bind_run_provider_slots(pins)

    resp = await WMClient(timeout=5).message(**CALL)

    assert resp.output == {"ok": True}
    assert seen == ["script-doctor-critique"]


# ── tenant: in-process ──────────────────────────────────────────────────────


async def test_tenant_slot_runs_in_process_on_the_pinned_endpoint(
    monkeypatch: pytest.MonkeyPatch, usage_calls: list[dict[str, Any]]
) -> None:
    slot = tenant_slot()
    seen = _use_slot(monkeypatch, slot)
    fake = FakeProvider(openai_reply(text="the critique"))
    _use_provider(monkeypatch, fake)
    bind_run_provider_slots(TENANT_PIN)

    resp = await WMClient(timeout=42).message(
        "script-doctor-critique",
        "critique this script",
        system_prompt="Per-call steering.",
        model_override="platform-model",
        work_tier="high",
    )

    assert seen == [SlotRef.from_dict("llm", TENANT_PIN["provider_slots"]["llm"])]
    (req,) = fake.requests
    assert req.url.host == "203.0.113.7", "connect to the resolver's pinned IP"
    assert req.url.path == "/v1/chat/completions"
    assert req.headers["host"] == "llm.tenant.example"
    assert req.extensions["sni_hostname"] == "llm.tenant.example"
    assert req.headers["authorization"] == f"Bearer {SECRET}"
    body = fake.bodies[0]
    assert body["model"] == "tenant-model-7", "the tenant slot's model wins over model_override"
    assert body["temperature"] == 0.1
    assert body["max_tokens"] == 8000, "WM_MIN_OUTPUT_TOKENS floor, as on the WM path"
    system = "\n".join(m["content"] for m in body["messages"] if m["role"] == "system")
    assert "Per-call steering." in system
    assert "You are a ruthless script doctor." in system, (
        "the vendored WM seed prompt for script-doctor-critique is applied too"
    )
    assert system.index("ruthless script doctor") < system.index("Per-call steering.")
    user = [m for m in body["messages"] if m["role"] == "user"]
    assert user[-1]["content"] == "critique this script"

    assert resp.output == "the critique"
    assert resp.output_type == "string"
    assert resp.usage_metrics is not None
    assert resp.usage_metrics["input_tokens"] == 11
    assert resp.usage_metrics["output_tokens"] == 7
    assert resp.metadata is not None
    assert resp.metadata["overrides_ignored"] == {
        "model_override": "platform-model",
        "work_tier": "high",
    }
    (usage,) = usage_calls
    assert usage["tenant_id"] == TENANT and usage["worker_name"] == "script-doctor-critique"
    assert usage["result"].input_tokens == 11 and usage["result"].output_tokens == 7


async def test_structured_output_is_parsed_and_contract_violations_reasked(
    monkeypatch: pytest.MonkeyPatch, usage_calls: list[dict[str, Any]]
) -> None:
    _use_slot(monkeypatch, tenant_slot())
    fake = FakeProvider(openai_reply([{"summary": "no title"}, {"title": "T", "score": 3}]))
    _use_provider(monkeypatch, fake)
    bind_run_provider_slots(TENANT_PIN)
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"title": {"type": "string"}, "score": {"type": "integer"}},
        "required": ["title"],
    }

    resp = await WMClient(timeout=5).message("promo-review", "review", output_schema=schema)

    assert resp.output == {"title": "T", "score": 3}
    assert resp.output_type == "object"
    assert resp.output_schema_used == schema
    assert len(fake.bodies) == 2, "the missing required key was re-asked once"
    assert fake.bodies[0]["tools"][0]["function"]["parameters"]["required"] == ["title"]


async def test_structured_output_that_never_validates_is_transient_like_wm(
    monkeypatch: pytest.MonkeyPatch, usage_calls: list[dict[str, Any]]
) -> None:
    _use_slot(monkeypatch, tenant_slot())
    _use_provider(monkeypatch, FakeProvider(openai_reply([{"nope": 1}])))
    bind_run_provider_slots(TENANT_PIN)
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }

    with pytest.raises(TransientUpstreamError) as err:
        await WMClient(timeout=5).message("promo-review", "review", output_schema=schema)

    assert err.value.failure_class == EXTERNAL
    assert "retries" in str(err.value).lower()
    assert usage_calls == []


async def test_content_blocks_become_multimodal_parts(
    monkeypatch: pytest.MonkeyPatch, usage_calls: list[dict[str, Any]]
) -> None:
    _use_slot(monkeypatch, tenant_slot())
    fake = FakeProvider(openai_reply(text="a cat"))
    _use_provider(monkeypatch, fake)
    bind_run_provider_slots(TENANT_PIN)
    png = b"\x89PNG\r\n\x1a\nfake"

    await WMClient(timeout=5).message(
        "film-critic",
        content_blocks=[
            {"type": "text", "text": "what is this?"},
            {"type": "image", "data": base64.b64encode(png).decode(), "media_type": "image/png"},
        ],
    )

    user = [m for m in fake.bodies[0]["messages"] if m["role"] == "user"][-1]["content"]
    assert {"type": "text", "text": "what is this?"} in user
    image = next(p for p in user if p["type"] == "image_url")
    assert image["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(png).decode()


async def test_anthropic_slot_uses_its_key_and_model(
    monkeypatch: pytest.MonkeyPatch, usage_calls: list[dict[str, Any]]
) -> None:
    _use_slot(
        monkeypatch,
        tenant_slot("anthropic", base_url=None, pinned_ip=None, model="claude-tenant"),
    )
    fake = FakeProvider(anthropic_reply("hello from claude"))
    _use_provider(monkeypatch, fake)
    bind_run_provider_slots(TENANT_PIN)

    resp = await WMClient(timeout=5).message("growth-bundle", "hi", system_prompt="Be kind.")

    (req,) = fake.requests
    assert req.url.host == "api.anthropic.com" and req.url.path == "/v1/messages"
    # anthropic>=2 authenticates with `Authorization: Bearer`; 1.x used `x-api-key`.
    # Assert the slot's key reaches the wire, whichever header the SDK uses.
    assert (req.headers.get("authorization") or req.headers.get("x-api-key")) in (
        f"Bearer {SECRET}",
        SECRET,
    )
    assert fake.bodies[0]["model"] == "claude-tenant"
    assert "Be kind." in json.dumps(fake.bodies[0]["system"])
    assert resp.output == "hello from claude"
    assert usage_calls[0]["result"].provider_kind == "anthropic"


# ── errors ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("status", "body", "expected", "fault"),
    [
        (
            401,
            {"error": {"message": f"Incorrect API key provided: {SECRET}"}},
            # A PermanentUpstreamError carrying failure_detail code=upstream_auth.
            UpstreamAuthError,
            CUSTOMER_INPUT,
        ),
        (
            404,
            {"error": {"message": f"model not found (key {SECRET[:12]}…)"}},
            PermanentUpstreamError,
            CUSTOMER_INPUT,
        ),
        (429, {"error": {"message": "rate limited"}}, TransientUpstreamError, EXTERNAL),
        (503, {"error": {"message": "overloaded"}}, TransientUpstreamError, EXTERNAL),
    ],
)
async def test_provider_http_errors_map_to_the_taxonomy_without_the_key(
    monkeypatch: pytest.MonkeyPatch,
    usage_calls: list[dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
    status: int,
    body: dict[str, Any],
    expected: type[TransientUpstreamError | PermanentUpstreamError],
    fault: str,
) -> None:
    caplog.set_level(logging.DEBUG)
    _use_slot(monkeypatch, tenant_slot())
    _use_provider(monkeypatch, FakeProvider(status_reply(status, body)))
    bind_run_provider_slots(TENANT_PIN)

    with pytest.raises(expected) as err:
        await WMClient(timeout=5).message("promo-review", "x")

    assert type(err.value) is expected
    assert err.value.failure_class == fault
    assert f"HTTP {status}" in str(err.value)
    assert SECRET not in str(err.value)
    assert SECRET[:12] not in str(err.value)
    assert err.value.__cause__ is None, "no chained provider exception carrying the body"
    assert SECRET not in caplog.text
    assert usage_calls == []


# ── retries: the WM path's BaseServiceClient policy ─────────────────────────


async def test_429_with_retry_after_is_retried_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, usage_calls: list[dict[str, Any]], sleeps: list[float]
) -> None:
    _use_slot(monkeypatch, tenant_slot())
    ok = openai_reply(text="after the limiter")

    def reply(body: dict[str, Any], n: int) -> httpx2.Response:
        if n == 0:
            return httpx2.Response(
                429, headers={"Retry-After": "7"}, json={"error": {"message": "slow down"}}
            )
        return ok(body, n)

    fake = FakeProvider(reply)
    _use_provider(monkeypatch, fake)
    bind_run_provider_slots(TENANT_PIN)

    resp = await WMClient(timeout=5).message("promo-review", "x")

    assert resp.output == "after the limiter"
    assert len(fake.requests) == 2
    assert sleeps == [7.0], "Retry-After (7s) wins over the 5s linear 429 backoff"
    assert len(usage_calls) == 1


async def test_5xx_retries_on_the_wm_budget_then_surfaces_transient(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    _use_slot(monkeypatch, tenant_slot())
    fake = FakeProvider(status_reply(503, {"error": {"message": "overloaded"}}))
    _use_provider(monkeypatch, fake)
    bind_run_provider_slots(TENANT_PIN)

    with pytest.raises(TransientUpstreamError):
        await WMClient(timeout=5).message("promo-review", "x")

    assert len(fake.requests) == 3, "BaseServiceClient's 5xx budget: 3 attempts"
    assert sleeps == [1.0, 2.0], "linear 1s backoff, as on the WM path"


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_4xx_is_never_retried(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], status: int
) -> None:
    _use_slot(monkeypatch, tenant_slot())
    fake = FakeProvider(status_reply(status, {"error": {"message": "nope"}}))
    _use_provider(monkeypatch, fake)
    bind_run_provider_slots(TENANT_PIN)

    with pytest.raises(PermanentUpstreamError):
        await WMClient(timeout=5).message("promo-review", "x")

    assert len(fake.requests) == 1
    assert sleeps == []


async def test_connect_timeout_is_retried_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    _use_slot(monkeypatch, tenant_slot())
    ok = openai_reply(text="second try")
    calls: list[int] = []

    def _flaky() -> httpx2.MockTransport:
        def handler(request: httpx2.Request) -> httpx2.Response:
            calls.append(1)
            if len(calls) == 1:
                raise httpx2.ConnectTimeout("connect timed out", request=request)
            return ok(json.loads(request.content), len(calls) - 1)

        return httpx2.MockTransport(handler)

    monkeypatch.setattr(in_process, "inner_transport_factory", _flaky)
    bind_run_provider_slots(TENANT_PIN)

    resp = await WMClient(timeout=5).message("promo-review", "x")

    assert resp.output == "second try"
    assert len(calls) == 2
    assert sleeps == [1.0]


async def test_read_timeout_is_not_resent(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    """A read timeout may mean the completion is still running (and billing)
    upstream; the non-idempotent POST is never re-sent at the transport level."""
    _use_slot(monkeypatch, tenant_slot())
    calls: list[int] = []

    def _slow() -> httpx2.MockTransport:
        def handler(request: httpx2.Request) -> httpx2.Response:
            calls.append(1)
            raise httpx2.ReadTimeout("read timed out", request=request)

        return httpx2.MockTransport(handler)

    monkeypatch.setattr(in_process, "inner_transport_factory", _slow)
    bind_run_provider_slots(TENANT_PIN)

    with pytest.raises(TransientUpstreamError):
        await WMClient(timeout=5).message("promo-review", "x")
    assert len(calls) == 1
    assert sleeps == []


async def test_unreachable_provider_is_transient(
    monkeypatch: pytest.MonkeyPatch, usage_calls: list[dict[str, Any]]
) -> None:
    _use_slot(monkeypatch, tenant_slot())

    def _down() -> httpx2.MockTransport:
        def handler(request: httpx2.Request) -> httpx2.Response:
            raise httpx2.ConnectError("connection refused", request=request)

        return httpx2.MockTransport(handler)

    monkeypatch.setattr(in_process, "inner_transport_factory", _down)
    bind_run_provider_slots(TENANT_PIN)

    with pytest.raises(TransientUpstreamError) as err:
        await WMClient(timeout=5).message("promo-review", "x")
    assert err.value.failure_class == EXTERNAL


async def test_tenant_slot_without_a_bound_tenant_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_slot(monkeypatch, tenant_slot())
    set_identity_context(None, None, None)
    bind_run_provider_slots(TENANT_PIN)
    with pytest.raises(PermanentUpstreamError, match="no tenant"):
        await WMClient(timeout=5).message("promo-review", "x")


# ── egress pin ──────────────────────────────────────────────────────────────


async def test_pinned_transport_refuses_any_other_origin() -> None:
    seen: list[httpx2.Request] = []
    inner = httpx2.MockTransport(lambda r: seen.append(r) or httpx2.Response(200))
    transport = PinnedTransport("https://llm.tenant.example/v1", "203.0.113.7", inner)
    async with httpx2.AsyncClient(transport=transport) as client:
        for url in (
            "https://evil.example/v1/chat/completions",
            "http://llm.tenant.example/v1/chat/completions",
            "https://llm.tenant.example:8443/v1/chat/completions",
        ):
            with pytest.raises(httpx2.ConnectError, match="origin"):
                await client.get(url)
        await client.get("https://llm.tenant.example/v1/models")
    assert [r.url.host for r in seen] == ["203.0.113.7"]


async def test_sanitise_strips_key_shaped_text() -> None:
    text = f"bad key {SECRET} and Bearer abc.def and sk-ant-api03-XYZ123 ok"
    out = in_process.sanitise(text, SECRET)
    assert SECRET not in out and "abc.def" not in out and "XYZ123" not in out
    assert out.endswith("ok")


# ── slot= (w2v_common 0.113.0) ──────────────────────────────────────────────

BOTH_TENANT_PIN = {
    "provider_slots": {
        "llm": TENANT_PIN["provider_slots"]["llm"],
        "vlm": {**TENANT_PIN["provider_slots"]["llm"], "version": 4},
    }
}
IMAGE_BLOCKS: list[dict[str, Any]] = [
    {"type": "text", "text": "describe"},
    {"type": "image", "data": base64.b64encode(b"x").decode(), "media_type": "image/png"},
]


@pytest.mark.parametrize(
    ("kwargs", "expected_slot"),
    [({"content_blocks": IMAGE_BLOCKS}, "vlm")],
)
async def test_slot_kwarg_selects_the_pinned_slot_in_process(
    monkeypatch: pytest.MonkeyPatch,
    usage_calls: list[dict[str, Any]],
    kwargs: dict[str, Any],
    expected_slot: str,
) -> None:
    """``slot=`` picks the run slot; the orchestrator's pins stay authoritative.

    ``W2V_PROVIDER_SLOTS=enforce`` must change nothing here: the base class's own
    slot resolution is never reached (``super().message`` is not called).
    """
    import w2v_common.providers.slots as common_slots

    def _base_never(*_a: object, **_k: object) -> bool:
        raise AssertionError("w2v_common's own slot resolution must not run in-process")

    monkeypatch.setenv("W2V_PROVIDER_SLOTS", "enforce")
    monkeypatch.setattr(common_slots, "provider_slots_enforced", _base_never)
    monkeypatch.setattr(common_slots, "resolve_provider_slot", _base_never)

    resolved_for: list[str] = []
    resolved = tenant_slot()

    async def _resolve(tenant_id: UUID, ref: SlotRef, *, slot: str = "llm") -> ResolvedSlot:
        assert tenant_id == TENANT
        assert ref.slot == slot
        resolved_for.append(slot)
        return resolved

    monkeypatch.setattr(dispatch, "resolve_pinned_llm", _resolve)
    _use_provider(monkeypatch, FakeProvider(openai_reply(text="ok")))
    bind_run_provider_slots(BOTH_TENANT_PIN)

    resp = await WMClient(timeout=5).message("script-doctor-critique", **kwargs)

    assert resp.output == "ok"
    assert resolved_for == [expected_slot]
    assert len(usage_calls) == 1
