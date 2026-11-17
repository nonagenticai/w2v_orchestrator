"""The conftest egress guard: no test may reach a real LLM endpoint.

CI runs on in-cluster ARC runners, where the default endpoints resolve to
PRODUCTION. 2026-10-07: 10 ``POST /workers/script-draft/message`` from CI runner
pods in a day. worker_management's message path is gone (``w2v_common`` 0.109.0);
the platform LLM is now Chat Completions on the vLLM / inference-router, so the
guard in ``tests/conftest.py`` (``_install_llm_egress_guard``) sits on those
sockets. These tests pin it:

* an unpatched orchestrator ``WMClient`` call fails fast with a named error and
  opens NO socket;
* a stage's own default ``WMClient`` (the shape that leaked) is blocked the same way;
* the raw ``w2v_common`` client (its own vLLM pool, no transport) is blocked too;
* a test that installs a ``PlatformChat`` still drives the real body/parse path.
"""

from __future__ import annotations

import socket
import time

import pytest
from w2v_common.clients.wm_client import WMClient as RawWMClient

from src.services.llm import WMClient
from tests.conftest import LLM_EGRESS_BLOCKED, LLMEgressBlockedError
from tests.support.platform_llm import CHAT_PATH, PlatformChat, posted_schema


@pytest.fixture
def no_socket(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record (and refuse) any DNS lookup or TCP connect attempted during the test."""
    attempts: list[object] = []

    def _refuse_getaddrinfo(*args: object, **_kwargs: object) -> list[object]:
        attempts.append(("getaddrinfo", args[0] if args else None))
        raise OSError("network disabled by test")

    def _refuse_connect(_self: socket.socket, address: object) -> None:
        attempts.append(("connect", address))
        raise OSError("network disabled by test")

    monkeypatch.setattr(socket, "getaddrinfo", _refuse_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", _refuse_connect)
    return attempts


async def test_unpatched_wmclient_fails_fast_without_network(
    no_socket: list[object], _no_real_worker_management: list[str]
) -> None:
    wm = WMClient(timeout=600.0)

    started = time.monotonic()
    with pytest.raises(LLMEgressBlockedError, match="PlatformChat") as excinfo:
        await wm.message("script-draft", "hello")
    elapsed = time.monotonic() - started

    assert LLM_EGRESS_BLOCKED in str(excinfo.value)
    assert "vllm.test/v1/chat/completions" in str(excinfo.value), "error must name the target"
    assert no_socket == [], f"the guard must stop the call before any socket: {no_socket}"
    assert elapsed < 1.0, f"must fail fast, not ride a retry/backoff ladder ({elapsed:.2f}s)"
    # The per-test teardown check saw it; clear it so THIS test (which provoked the
    # guard on purpose) is not failed by it.
    assert _no_real_worker_management == ["http://vllm.test"]
    _no_real_worker_management.clear()


async def test_stage_default_wmclient_is_blocked(
    no_socket: list[object], _no_real_worker_management: list[str]
) -> None:
    """The exact leak: a stage built with no client constructs a real ``WMClient``."""
    from src.services.stages.generic.script_draft import ScriptDraftStage

    stage = ScriptDraftStage()
    assert type(stage._wm) is WMClient, "precondition: the stage built a REAL client"

    with pytest.raises(LLMEgressBlockedError):
        await stage._wm.message("script-draft", "hello")

    assert no_socket == []
    assert len(_no_real_worker_management) == 1
    _no_real_worker_management.clear()


async def test_raw_common_client_without_transport_is_blocked(
    no_socket: list[object], _no_real_worker_management: list[str]
) -> None:
    """``w2v_common``'s own vLLM pool (built with no transport) is guarded as well."""
    wm = RawWMClient(timeout=30.0)

    with pytest.raises(LLMEgressBlockedError):
        await wm.message("script-draft", "hello")

    assert no_socket == []
    assert _no_real_worker_management == ["http://vllm.test/v1"]
    _no_real_worker_management.clear()


async def test_installed_platform_chat_still_works(
    monkeypatch: pytest.MonkeyPatch, _no_real_worker_management: list[str]
) -> None:
    """The ``*_empty_draw_wire`` pattern must keep driving the real body/parse path."""

    async def _no_ledger(**_k: object) -> bool:
        return True

    monkeypatch.setattr("src.services.llm.dispatch.record_platform_vllm_usage", _no_ledger)
    chat = PlatformChat(lambda _body, _n: {"lines": []}).install(monkeypatch)
    schema: dict[str, object] = {"type": "object", "properties": {"lines": {"type": "array"}}}

    resp = await WMClient(timeout=30.0).message("script-draft", "hello", output_schema=schema)

    assert chat.paths == [CHAT_PATH]
    assert posted_schema(chat.bodies[0]) == schema
    assert resp.output == {"lines": []}
    assert _no_real_worker_management == []
