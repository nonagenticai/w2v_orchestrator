"""LLM_BACKEND=vllm — the platform-default slot calls the pgx vLLM directly.

Stub transport only (``in_process.inner_transport_factory``), never the network.
Pins: backend selection (ir default, wm refused / vllm for unpinned + deployment-pinned runs,
tenant slots untouched), the request shape (URL, bearer token, model, seed prompt,
temperature, token clamp, thinking off, xgrammar-stripped json_schema that KEEPS
``prefixItems``), schema enforcement with re-ask, and error mapping without leaking
the token.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from uuid import UUID, uuid7

import httpx2
import pytest
from pydantic import ValidationError
from w2v_common.auth.identity_context import set_correlation_context, set_identity_context
from w2v_common.clients.ir_direct import IRNotConfiguredError

from src.auth.run_providers import bind_run_provider_slots, clear_run_provider_slots
from src.config import PlatformLLMSettings, clear_settings_cache
from src.models.failure_class import EXTERNAL, OUR_DEFECT
from src.services.llm import WMClient, dispatch, in_process, vllm_direct
from src.services.stages.primitives.script_stage import non_empty_array
from src.services.stages.upstream import PermanentUpstreamError, TransientUpstreamError
from tests.services.llm._fakes import FakeProvider

pytestmark = pytest.mark.asyncio

TOKEN = "vllm-licence-STATIC-0123456789"  # pragma: allowlist secret - fake fixture token
BASE = "http://vllm.test:8001/v1"
TENANT = UUID("01990000-0000-7000-8000-0000000000bb")
CONN_ID = UUID("01990000-0000-7000-8000-000000000001")


@pytest.fixture(autouse=True)
def _context() -> Iterator[None]:
    clear_run_provider_slots()
    set_identity_context(str(TENANT), None, None)
    set_correlation_context(str(uuid7()), "demo_video_script")
    yield
    clear_run_provider_slots()
    set_correlation_context(None, None)


@pytest.fixture(autouse=True)
def _no_ledger(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The run_usage write is pinned against real postgres in test_in_process_usage;
    here it is recorded, so these transport-only tests never open a DB session."""
    calls: list[dict[str, Any]] = []

    async def _record(**kwargs: Any) -> bool:
        calls.append(kwargs)
        return True

    monkeypatch.setattr(dispatch, "record_platform_vllm_usage", _record)
    return calls


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(in_process, "_sleep", _sleep)


@pytest.fixture
def vllm_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    monkeypatch.setenv("LLM_VLLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_VLLM_TOKEN", TOKEN)
    clear_settings_cache()
    yield
    clear_settings_cache()


def _completion(content: str, *, finish: str = "stop", usage: tuple[int, int] = (20, 9)) -> Any:
    return httpx2.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "local-llm",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish,
                }
            ],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
        },
    )


def _provider(monkeypatch: pytest.MonkeyPatch, *replies: Any) -> FakeProvider:
    def reply(_body: dict[str, Any], n: int) -> Any:
        return replies[min(n, len(replies) - 1)]

    fake = FakeProvider(reply=reply)
    monkeypatch.setattr(in_process, "inner_transport_factory", fake.transport)
    return fake


SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "lines": {**non_empty_array({"type": "string", "maxLength": 80}), "minItems": 1},
        "tone": {"type": "string", "enum": ["warm", "dry"]},
    },
    "required": ["lines", "tone"],
}


ENUM_SCHEMA: dict[str, object] = {"type": "string", "enum": ["a", "b"]}


# ── settings ────────────────────────────────────────────────────────────────


async def test_backend_defaults_to_ir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset ⇒ ``ir``, the ``w2v_common`` >= 0.109.0 default; boots without a token."""
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("LLM_IR_TOKEN", raising=False)
    assert PlatformLLMSettings().llm_backend == "ir"


async def test_wm_backend_is_refused_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """worker_management was deleted in w2v_common 0.109.0: ``wm`` is not a backend."""
    monkeypatch.setenv("LLM_BACKEND", "wm")
    with pytest.raises(ValidationError, match=r"llm_backend|LLM_BACKEND"):
        PlatformLLMSettings()


async def test_vllm_backend_refuses_to_boot_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    monkeypatch.delenv("LLM_VLLM_TOKEN", raising=False)
    with pytest.raises(ValidationError, match="LLM_VLLM_TOKEN"):
        PlatformLLMSettings()


async def test_fast_workers_route_to_fast_model_only_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_VLLM_FAST_WORKERS", "pacing-critic, seo-intent-classify")
    assert PlatformLLMSettings().model_for("pacing-critic") == "local-llm"  # no fast model
    monkeypatch.setenv("LLM_VLLM_MODEL_FAST", "small-llm")
    s = PlatformLLMSettings()
    assert s.model_for("pacing-critic") == "small-llm"
    assert s.model_for("film-critic") == "local-llm"


# ── dispatch ────────────────────────────────────────────────────────────────


async def test_defaulted_ir_without_licence_fails_the_call_and_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM_BACKEND unset + no LLM_IR_TOKEN: a loud IRNotConfiguredError, no request."""
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("LLM_IR_TOKEN", raising=False)
    clear_settings_cache()
    fake = _provider(monkeypatch, _completion("{}"))

    async def _boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("vllm path taken with LLM_BACKEND unset")

    monkeypatch.setattr(vllm_direct, "call_platform_vllm", _boom)
    with pytest.raises(IRNotConfiguredError, match="LLM_IR_TOKEN"):
        await WMClient().message("film-critic", "hi")
    assert fake.requests == []


@pytest.mark.parametrize("pin", [None])
async def test_platform_default_goes_to_vllm_with_static_token(
    monkeypatch: pytest.MonkeyPatch, vllm_env: None, pin: dict[str, Any] | None
) -> None:
    if pin is not None:
        bind_run_provider_slots({"provider_slots": {"llm": pin}})
    fake = _provider(monkeypatch, _completion(json.dumps({"lines": ["a", "b"], "tone": "dry"})))

    resp = await WMClient().message(
        "script-doctor-critique",
        "critique this",
        system_prompt="be terse",
        output_schema=SCHEMA,
        max_tokens=50_000,
        model_override="ignored-model",
        context={"stage": "x"},
    )

    assert resp.output == {"lines": ["a", "b"], "tone": "dry"}
    assert resp.output_type == "object"
    assert resp.output_schema_used == SCHEMA
    assert resp.usage_metrics is not None
    assert resp.usage_metrics["provider"] == "vllm"
    assert resp.usage_metrics["funding"] == "platform_direct"
    assert resp.usage_metrics["input_tokens"] == 20
    assert resp.metadata is not None
    assert resp.metadata["served_model"] == "local-llm"
    assert resp.metadata["overrides_ignored"] == {"model_override": "ignored-model"}

    (req,) = fake.requests
    assert str(req.url) == f"{BASE}/chat/completions"
    assert req.headers["authorization"] == f"Bearer {TOKEN}"
    assert "x-tenant-id" not in req.headers  # static token, no bundle mint
    body = fake.bodies[0]
    assert body["model"] == "local-llm"
    assert body["max_tokens"] == vllm_direct.VLLM_MAX_OUTPUT_TOKENS
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    system = body["messages"][0]
    assert system["role"] == "system"
    assert system["content"].endswith("be terse")  # seed prompt first, call prompt last
    assert len(system["content"]) > len("be terse")
    assert body["messages"][1] == {"role": "user", "content": "critique this"}
    grammar = body["response_format"]["json_schema"]["schema"]
    lines = grammar["properties"]["lines"]
    assert "prefixItems" in lines  # the non_empty_array guarantee reaches the grammar
    assert "minItems" not in lines  # xgrammar cannot compile it
    assert "maxLength" not in lines["items"]
    assert body["response_format"]["json_schema"]["strict"] is False


async def test_tenant_slot_is_not_routed_to_vllm(
    monkeypatch: pytest.MonkeyPatch, vllm_env: None
) -> None:
    bind_run_provider_slots(
        {
            "provider_slots": {
                "llm": {
                    "source": "tenant",
                    "kind": "openai-compatible",
                    "connection_id": str(CONN_ID),
                    "version": 1,
                }
            }
        }
    )

    async def _boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("tenant slot must not use the platform vLLM")

    sentinel = RuntimeError("tenant path")

    async def _resolve(*_a: Any, **_k: Any) -> Any:
        raise sentinel

    monkeypatch.setattr(vllm_direct, "call_platform_vllm", _boom)
    monkeypatch.setattr(dispatch, "resolve_pinned_llm", _resolve)
    with pytest.raises(RuntimeError, match="tenant path"):
        await WMClient().message("film-critic", "hi")


async def test_enable_reasoning_true_is_forwarded(
    monkeypatch: pytest.MonkeyPatch, vllm_env: None
) -> None:
    fake = _provider(monkeypatch, _completion("plain text"))
    resp = await WMClient().message("film-critic", "hi", enable_reasoning=True)
    assert resp.output == "plain text"
    assert fake.bodies[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert "response_format" not in fake.bodies[0]


# ── schema enforcement ──────────────────────────────────────────────────────


async def test_schema_violation_is_reasked_with_the_violations(
    monkeypatch: pytest.MonkeyPatch, vllm_env: None
) -> None:
    fake = _provider(
        monkeypatch,
        _completion("not json"),
        _completion(json.dumps({"lines": ["a"], "tone": "loud"})),
        _completion(json.dumps({"lines": ["a"], "tone": "warm"})),
    )
    resp = await WMClient().message("film-critic", "hi", output_schema=SCHEMA)
    assert resp.output == {"lines": ["a"], "tone": "warm"}
    assert len(fake.requests) == 3
    assert resp.usage_metrics is not None and resp.usage_metrics["requests"] == 3
    feedback = fake.bodies[2]["messages"][-1]["content"]
    assert "tone" in feedback and "'loud' is not one of" in feedback
    assert fake.bodies[2]["messages"][-2] == {
        "role": "assistant",
        "content": json.dumps({"lines": ["a"], "tone": "loud"}),
    }


async def test_schema_retries_exhausted_is_a_transient_bad_draw(
    monkeypatch: pytest.MonkeyPatch, vllm_env: None
) -> None:
    fake = _provider(monkeypatch, _completion(json.dumps({"tone": "warm"})))
    with pytest.raises(TransientUpstreamError, match="maximum output retries") as exc:
        await WMClient().message("film-critic", "hi", output_schema=SCHEMA)
    assert exc.value.failure_class == EXTERNAL
    assert len(fake.requests) == vllm_direct.OUTPUT_RETRIES + 1


async def test_bound_violation_is_reported_not_fatal(
    monkeypatch: pytest.MonkeyPatch, vllm_env: None
) -> None:
    _provider(monkeypatch, _completion(json.dumps({"lines": ["x" * 90], "tone": "dry"})))
    resp = await WMClient().message("film-critic", "hi", output_schema=SCHEMA)
    assert resp.usage_metrics is not None
    (bound,) = resp.usage_metrics["bound_violations"]
    assert bound["keyword"] == "maxLength" and bound["path"] == "/lines/0"


async def test_scalar_enum_root_is_wrapped_for_the_grammar_and_unwrapped(
    monkeypatch: pytest.MonkeyPatch, vllm_env: None
) -> None:
    fake = _provider(monkeypatch, _completion(json.dumps({"response": "b"})))
    resp = await WMClient().message("seo-intent-classify", "hi", output_schema=ENUM_SCHEMA)
    assert resp.output == "b"
    assert resp.output_type == "enum"
    grammar = fake.bodies[0]["response_format"]["json_schema"]["schema"]
    assert grammar["required"] == ["response"]
    assert grammar["properties"]["response"] == {"type": "string", "enum": ["a", "b"]}


async def test_content_blocks_become_openai_parts(
    monkeypatch: pytest.MonkeyPatch, vllm_env: None
) -> None:
    fake = _provider(monkeypatch, _completion("seen"))
    await WMClient().message(
        "video-frame-analyzer",
        content_blocks=[
            {"type": "text", "text": "describe"},
            {"type": "image", "media_type": "image/png", "data": "aGk="},
        ],
    )
    assert fake.bodies[0]["messages"][-1]["content"] == [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
    ]


# ── errors ──────────────────────────────────────────────────────────────────


async def test_auth_rejection_is_our_defect_and_never_leaks_the_token(
    monkeypatch: pytest.MonkeyPatch, vllm_env: None
) -> None:
    _provider(monkeypatch, httpx2.Response(403, json={"detail": f"bad Bearer {TOKEN}"}))
    with pytest.raises(PermanentUpstreamError) as exc:
        await WMClient().message("film-critic", "hi")
    assert exc.value.failure_class == OUR_DEFECT
    assert "vllm" in str(exc.value)
    assert TOKEN not in str(exc.value)


async def test_server_error_after_retries_is_transient(
    monkeypatch: pytest.MonkeyPatch, vllm_env: None
) -> None:
    fake = _provider(monkeypatch, httpx2.Response(503, json={"detail": "overloaded"}))
    with pytest.raises(TransientUpstreamError) as exc:
        await WMClient().message("film-critic", "hi")
    assert exc.value.failure_class == EXTERNAL
    assert len(fake.requests) == 3  # RetryTransport: the WM path's 5xx budget


async def test_platform_read_timeout_is_retried_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, vllm_env: None
) -> None:
    """The platform vLLM is unmetered: a read timeout keeps its transport retry
    (only TENANT providers drop it — see test_wm_dispatch read-timeout test)."""
    ok = _completion(json.dumps({"lines": ["a"], "tone": "dry"}))
    calls: list[int] = []

    def _flaky() -> httpx2.MockTransport:
        def handler(request: httpx2.Request) -> httpx2.Response:
            calls.append(1)
            if len(calls) == 1:
                raise httpx2.ReadTimeout("read timed out", request=request)
            return ok

        return httpx2.MockTransport(handler)

    monkeypatch.setattr(in_process, "inner_transport_factory", _flaky)
    resp = await WMClient().message("film-critic", "hi", output_schema=SCHEMA)
    assert resp.output == {"lines": ["a"], "tone": "dry"}
    assert len(calls) == 2


async def test_truncated_reply_is_a_retryable_model_failure_not_customer_input(
    monkeypatch: pytest.MonkeyPatch, vllm_env: None
) -> None:
    """``finish_reason=length`` is a bad DRAW of the platform model (live 2026-10-20:
    a repetition loop inside one ``text`` string), never the customer's input."""
    _provider(monkeypatch, _completion('{"lines": ["a"', finish="length", usage=(20, 10050)))
    with pytest.raises(TransientUpstreamError, match="token limit") as exc_info:
        await WMClient().message("film-critic", "hi", output_schema=SCHEMA)
    assert exc_info.value.failure_class == EXTERNAL
    assert "completion_tokens=10050" in str(exc_info.value)


async def test_script_draft_body_carries_its_anti_loop_frequency_penalty(
    monkeypatch: pytest.MonkeyPatch, vllm_env: None
) -> None:
    fake = _provider(monkeypatch, _completion('{"lines": ["a"], "tone": "dry"}'))
    await WMClient().message("script-draft", "hi", output_schema=SCHEMA)
    await WMClient().message("film-critic", "hi", output_schema=SCHEMA)
    assert fake.bodies[0]["frequency_penalty"] == 0.2
    # Only the measured worker gets it: every other body is byte-identical to before.
    assert "frequency_penalty" not in fake.bodies[1]
