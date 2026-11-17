"""LLM_BACKEND=ir — the platform-default slot calls inference-router directly.

Stub transport only (``in_process.inner_transport_factory``), never the network.
Pins: backend selection (ir for unpinned + deployment-pinned runs, tenant slots
untouched, vllm still the rollback), the request (URL ``/v1/chat/completions``,
static ``inference-router`` licence on ``X-Tenant-Id``, delegation + attribution
headers, top-level ``enable_reasoning``, token clamp, model ``local-llm``), the
metering split (IR meters: no local run_usage row, no ``funding``), the delegation
refusal for an unvouched tenant, no read-timeout re-send, and error mapping that
never leaks the token.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from uuid import UUID, uuid7

import httpx2
import pytest
from pydantic import ValidationError
from w2v_common.auth.identity_context import (
    set_correlation_context,
    set_identity_context,
    set_w2v_project_context,
)
from w2v_common.tenant import TenantSource, UnvouchedTenantError

from src.auth.run_providers import bind_run_provider_slots, clear_run_provider_slots
from src.config import PlatformLLMSettings, clear_settings_cache
from src.models.failure_class import EXTERNAL, OUR_DEFECT
from src.services.llm import WMClient, dispatch, in_process, ir_direct, vllm_direct
from src.services.stages.upstream import PermanentUpstreamError, TransientUpstreamError
from tests.services.llm._fakes import FakeProvider

pytestmark = pytest.mark.asyncio

TOKEN = "ir-licence-STATIC-0123456789"  # pragma: allowlist secret - fake fixture token
BASE = "http://inference-router.test:8150"
TENANT = UUID("01990000-0000-7000-8000-0000000000cc")
SUBJECT = "01990000-0000-7000-8000-0000000000dd"
CONN_ID = UUID("01990000-0000-7000-8000-000000000001")
RUN_ID = str(uuid7())

SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"tone": {"type": "string", "enum": ["warm", "dry"]}},
    "required": ["tone"],
}


@pytest.fixture(autouse=True)
def _context() -> Iterator[None]:
    clear_run_provider_slots()
    set_identity_context(str(TENANT), None, SUBJECT, tenant_source=TenantSource.LICENSE_TOKEN)
    set_correlation_context(RUN_ID, "demo_video_script")
    yield
    clear_run_provider_slots()
    set_correlation_context(None, None)


@pytest.fixture(autouse=True)
def _ledger(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
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
def ir_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LLM_BACKEND", "ir")
    monkeypatch.setenv("LLM_IR_BASE_URL", BASE)
    monkeypatch.setenv("LLM_IR_TOKEN", TOKEN)
    # w2v_common (0.109) refuses an IR base URL that is not the host discovery
    # resolves for inference_router (delegation keys on it): point discovery here.
    monkeypatch.setenv("W2V_INFERENCE_ROUTER_URL", BASE)
    clear_settings_cache()
    yield
    clear_settings_cache()


def _completion(content: str, *, finish: str = "stop") -> Any:
    return httpx2.Response(
        200,
        json={
            "id": "chatcmpl-ir",
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
            "usage": {"prompt_tokens": 31, "completion_tokens": 12, "total_tokens": 43},
        },
    )


def _provider(monkeypatch: pytest.MonkeyPatch, *replies: Any) -> FakeProvider:
    def reply(_body: dict[str, Any], n: int) -> Any:
        return replies[min(n, len(replies) - 1)]

    fake = FakeProvider(reply=reply)
    monkeypatch.setattr(in_process, "inner_transport_factory", fake.transport)
    return fake


# ── settings ────────────────────────────────────────────────────────────────


async def test_ir_backend_refuses_to_boot_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "ir")
    monkeypatch.delenv("LLM_IR_TOKEN", raising=False)
    with pytest.raises(ValidationError, match="LLM_IR_TOKEN"):
        PlatformLLMSettings()


async def test_ir_base_url_defaults_to_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_IR_BASE_URL", raising=False)
    s = PlatformLLMSettings()
    assert s.llm_ir_base_url.startswith("http") and "inference-router" in s.llm_ir_base_url
    assert s.llm_ir_model == "local-llm"


# ── dispatch + request shape ────────────────────────────────────────────────


@pytest.mark.parametrize("pin", [None])
async def test_platform_default_goes_to_ir_with_static_licence_and_delegation(
    monkeypatch: pytest.MonkeyPatch,
    ir_env: None,
    _ledger: list[dict[str, Any]],
    pin: dict[str, Any] | None,
) -> None:
    if pin is not None:
        bind_run_provider_slots({"provider_slots": {"llm": pin}})

    async def _no_vllm(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("vllm path taken under LLM_BACKEND=ir")

    monkeypatch.setattr(vllm_direct, "call_platform_vllm", _no_vllm)
    fake = _provider(monkeypatch, _completion(json.dumps({"tone": "dry"})))

    resp = await WMClient().message(
        "script-doctor-critique",
        "critique this",
        system_prompt="be terse",
        output_schema=SCHEMA,
        max_tokens=50_000,
    )

    assert resp.output == {"tone": "dry"}
    assert resp.usage_metrics is not None
    assert resp.usage_metrics["provider"] == ir_direct.PROVIDER
    assert resp.usage_metrics["input_tokens"] == 31
    assert resp.usage_metrics["output_tokens"] == 12
    assert resp.usage_metrics["total_tokens"] == 43
    assert "funding" not in resp.usage_metrics  # IR meters — not platform_direct
    assert resp.metadata is not None and resp.metadata["backend"] == ir_direct.PROVIDER
    # IR's cost event is mirrored by the billing reconciler; a local row would double count.
    assert _ledger == []

    (req,) = fake.requests
    assert str(req.url) == f"{BASE}/v1/chat/completions"
    assert req.headers["x-tenant-id"] == f"Bearer {TOKEN}"
    assert "authorization" not in req.headers
    assert req.headers["x-effective-tenant-id"] == str(TENANT)
    assert req.headers["x-app-id"] == "w2v"
    assert req.headers["x-worker-name"] == "script-doctor-critique"
    assert req.headers["x-job-id"] == RUN_ID
    assert req.headers["x-correlation-id"] == RUN_ID
    assert req.headers["x-stage"] == "demo_video_script"
    assert req.headers["x-subject-id"] == SUBJECT

    body = fake.bodies[0]
    assert body["model"] == "local-llm"
    assert body["max_tokens"] == 16_384
    assert body["enable_reasoning"] is False
    assert "chat_template_kwargs" not in body
    assert body["messages"][0]["content"].endswith("be terse")
    assert body["response_format"]["type"] == "json_schema"


async def test_enable_reasoning_is_top_level(monkeypatch: pytest.MonkeyPatch, ir_env: None) -> None:
    fake = _provider(monkeypatch, _completion("thought it through"))
    resp = await WMClient().message("film-critic", "hi", enable_reasoning=True)
    assert resp.output == "thought it through"
    assert fake.bodies[0]["enable_reasoning"] is True
    assert "chat_template_kwargs" not in fake.bodies[0]


async def test_explicit_project_id_is_sent_as_x_project_id(
    monkeypatch: pytest.MonkeyPatch, ir_env: None
) -> None:
    """A request-scope caller (storyboard assist) names its W2V project explicitly;
    IR attributes the call by it, exactly as the deleted WM path did."""
    project = uuid7()
    fake = _provider(monkeypatch, _completion("ok"))
    await WMClient().message("storyboard-draft", "hi", project_id=project)
    (req,) = fake.requests
    assert req.headers["x-project-id"] == str(project)


async def test_bound_w2v_project_is_sent_when_none_is_passed(
    monkeypatch: pytest.MonkeyPatch, ir_env: None
) -> None:
    project = str(uuid7())
    set_w2v_project_context(project)
    try:
        fake = _provider(monkeypatch, _completion("ok"))
        await WMClient().message("film-critic", "hi")
    finally:
        set_w2v_project_context(None)
    (req,) = fake.requests
    assert req.headers["x-project-id"] == project


async def test_malformed_project_id_is_refused_before_sending(
    monkeypatch: pytest.MonkeyPatch, ir_env: None
) -> None:
    fake = _provider(monkeypatch, _completion("ok"))
    with pytest.raises(ValueError, match="canonical"):
        await WMClient().message("film-critic", "hi", project_id="not-a-uuid")
    assert fake.requests == []


async def test_unvouched_tenant_is_refused_before_sending(
    monkeypatch: pytest.MonkeyPatch, ir_env: None
) -> None:
    set_identity_context(str(TENANT), None, None, tenant_source=None)
    fake = _provider(monkeypatch, _completion("never"))
    with pytest.raises(UnvouchedTenantError):
        await WMClient().message("film-critic", "hi")
    assert fake.requests == []


async def test_tenant_slot_is_not_routed_to_ir(
    monkeypatch: pytest.MonkeyPatch, ir_env: None
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
        raise AssertionError("tenant slot must not use inference-router")

    async def _resolve(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("tenant path")

    monkeypatch.setattr(ir_direct, "call_platform_ir", _boom)
    monkeypatch.setattr(dispatch, "resolve_pinned_llm", _resolve)
    with pytest.raises(RuntimeError, match="tenant path"):
        await WMClient().message("film-critic", "hi")


async def test_vllm_stays_the_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    monkeypatch.setenv("LLM_VLLM_TOKEN", "vllm-licence-fake-0123456789")  # pragma: allowlist secret
    clear_settings_cache()
    try:

        async def _no_ir(*_a: Any, **_k: Any) -> Any:
            raise AssertionError("ir path taken under LLM_BACKEND=vllm")

        monkeypatch.setattr(ir_direct, "call_platform_ir", _no_ir)
        fake = _provider(monkeypatch, _completion("from vllm"))
        resp = await WMClient().message("film-critic", "hi")
        assert resp.output == "from vllm"
        assert str(fake.requests[0].url).endswith("/v1/chat/completions")
        assert "chat_template_kwargs" in fake.bodies[0]
    finally:
        clear_settings_cache()


# ── errors ──────────────────────────────────────────────────────────────────


async def test_auth_rejection_names_the_ir_licence_and_never_leaks_the_token(
    monkeypatch: pytest.MonkeyPatch, ir_env: None
) -> None:
    _provider(monkeypatch, httpx2.Response(401, json={"detail": f"bad Bearer {TOKEN}"}))
    with pytest.raises(PermanentUpstreamError) as exc:
        await WMClient().message("film-critic", "hi")
    assert exc.value.failure_class == OUR_DEFECT
    assert "LLM_IR_TOKEN" in str(exc.value) and "inference-router" in str(exc.value)
    assert TOKEN not in str(exc.value)


async def test_read_timeout_is_not_resent_because_ir_meters(
    monkeypatch: pytest.MonkeyPatch, ir_env: None
) -> None:
    calls: list[int] = []

    def _slow() -> httpx2.MockTransport:
        def handler(request: httpx2.Request) -> httpx2.Response:
            calls.append(1)
            raise httpx2.ReadTimeout("read timed out", request=request)

        return httpx2.MockTransport(handler)

    monkeypatch.setattr(in_process, "inner_transport_factory", _slow)
    with pytest.raises(TransientUpstreamError):
        await WMClient().message("film-critic", "hi")
    assert len(calls) == 1


# ── truncated draw / anti-loop sampling (live 2026-10-20) ──────────────────


async def test_truncated_ir_reply_is_retryable_external_not_customer_input(
    monkeypatch: pytest.MonkeyPatch, ir_env: None
) -> None:
    """Runs 01a158a5-c749 / 01a158a9-5a9b: IR answered 200 with ``finish_reason=length``
    (a ~43k-char repetition loop) and the run was filed as ``customer_input``."""
    _provider(monkeypatch, _completion('{"tone": "dr', finish="length"))
    with pytest.raises(TransientUpstreamError, match="output token limit") as exc_info:
        await WMClient().message("script-draft", "hi", output_schema=SCHEMA)
    assert exc_info.value.failure_class == EXTERNAL
    assert not isinstance(exc_info.value, PermanentUpstreamError)


async def test_ir_body_for_script_draft_carries_frequency_penalty(
    monkeypatch: pytest.MonkeyPatch, ir_env: None
) -> None:
    fake = _provider(monkeypatch, _completion('{"tone": "dry"}'))
    await WMClient().message("script-draft", "hi", output_schema=SCHEMA, enable_reasoning=False)
    body = fake.bodies[0]
    # IR forwards frequency_penalty verbatim to vLLM; thinking stays a top-level OFF.
    assert body["frequency_penalty"] == 0.2
    assert body["enable_reasoning"] is False
    assert "chat_template_kwargs" not in body
