"""The platform LLM on the wire, for tests: Chat Completions behind a MockTransport.

The suite runs the platform-default slot on ``LLM_BACKEND=vllm`` (``tests/conftest.py``
``_clear_settings``), the backend prod runs. A platform call is ONE OpenAI Chat
Completions POST to ``{LLM_VLLM_BASE_URL}/chat/completions`` made through
``in_process.inner_transport_factory`` (the socket), so a test that wants the REAL
request body + response parsing (``*_empty_draw_wire``) installs a
:class:`PlatformChat` there instead of mocking ``WMClient.message``.

The worker_management ``POST /workers/{name}/message`` shape this replaced was
deleted in ``w2v_common`` 0.109.0. What that body carried as ``output_schema`` is
now ``response_format.json_schema.schema`` (xgrammar-stripped: ``minItems`` gone,
``prefixItems`` kept) — :func:`posted_schema` reads it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import httpx2
import pytest
from w2v_common.clients.wm_client import WMClient as RawWMClient

from src.services.llm import in_process

#: ``LLM_VLLM_BASE_URL`` the conftest sets; never resolvable, never production.
VLLM_TEST_BASE_URL = "http://vllm.test/v1"
#: The path every platform call POSTs to.
CHAT_PATH = "/v1/chat/completions"


def chat_completion(
    content: str, *, finish: str = "stop", usage: tuple[int, int] = (20, 9)
) -> httpx2.Response:
    """A vLLM Chat Completions 200 carrying ``content`` as the assistant message."""
    return httpx2.Response(
        200,
        json={
            "id": "chatcmpl-test",
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


def posted_schema(body: dict[str, Any]) -> dict[str, Any] | None:
    """The grammar a platform call sent (``response_format.json_schema.schema``)."""
    fmt = body.get("response_format")
    if not isinstance(fmt, dict):
        return None
    return fmt["json_schema"]["schema"]


def user_text(body: dict[str, Any]) -> str:
    """The user turn of the FIRST message pair (the prompt the stage built)."""
    for message in body["messages"]:
        if message["role"] == "user":
            content = message["content"]
            return content if isinstance(content, str) else json.dumps(content)
    return ""


def system_text(body: dict[str, Any]) -> str:
    return "\n\n".join(m["content"] for m in body["messages"] if m["role"] == "system")


Answer = Callable[[dict[str, Any], int], Any]


@dataclass
class PlatformChat:
    """Records every platform LLM request; answers with ``answer(body, n)``.

    ``answer`` returns the decoded OUTPUT (dict / list — JSON-encoded into the
    assistant content, as the model would emit it), a ``str`` sent VERBATIM as the
    assistant content (plain-text replies, or malformed JSON), or a ready
    ``httpx2.Response`` (errors, ``finish_reason="length"``). ``n`` is the 0-based
    call index.
    """

    answer: Answer
    requests: list[httpx2.Request] = field(default_factory=list)
    bodies: list[dict[str, Any]] = field(default_factory=list)

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content or b"{}")
        self.requests.append(request)
        self.bodies.append(body)
        out = self.answer(body, len(self.bodies) - 1)
        if isinstance(out, httpx2.Response):
            return out
        if isinstance(out, str):
            return chat_completion(out)  # raw assistant text, as the model emits it
        return chat_completion(json.dumps(out))

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self.handler)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> PlatformChat:
        monkeypatch.setattr(in_process, "inner_transport_factory", self.transport)
        return self

    def activate(self) -> PlatformChat:
        """Make THIS the platform socket until the next ``activate`` / test teardown.

        For helpers with no ``monkeypatch`` in reach (a wire object's ``client()``):
        the module MUST use the ``platform_llm_socket`` fixture (``tests/conftest.py``),
        which restores the guarded socket at teardown.
        """
        in_process.inner_transport_factory = self.transport
        return self

    @property
    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]

    @property
    def schemas(self) -> list[dict[str, Any] | None]:
        return [posted_schema(b) for b in self.bodies]


#: inference-router origin for tests. ``w2v_common`` refuses an IR base URL that is
#: not the host discovery resolves, so :func:`use_ir_backend` points discovery here.
IR_TEST_BASE_URL = "http://inference-router.test:8150"


def use_ir_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """``LLM_BACKEND=ir`` with a fake licence, discovery pointed at the test origin.

    For tests that observe what a ``w2v_common`` ``WMClient`` puts on the wire for
    inference-router — the backend whose pooled client carries the S2S identity
    hook (actor, project, subject attribution). Call after the conftest's
    ``_clear_settings`` (any test body or later fixture is).
    """
    from src.config import clear_settings_cache

    monkeypatch.setenv("LLM_BACKEND", "ir")
    monkeypatch.setenv("LLM_IR_TOKEN", "test-ir-licence-token")  # pragma: allowlist secret
    monkeypatch.setenv("W2V_INFERENCE_ROUTER_URL", IR_TEST_BASE_URL)
    monkeypatch.delenv("LLM_IR_BASE_URL", raising=False)
    clear_settings_cache()


def common_ir_client(
    seen: list[httpx.Request], output: object = None, *, timeout: float = 30.0
) -> RawWMClient:
    """A REAL ``w2v_common`` ``WMClient`` on the ir backend over a MockTransport.

    Records every request in ``seen`` and answers ``output`` (default
    ``{"ok": True}``) as a Chat Completions reply. Needs :func:`use_ir_backend`.
    """
    reply = {"ok": True} if output is None else output

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=json.loads(chat_completion(json.dumps(reply)).content))

    return RawWMClient(timeout=timeout, llm_backend="ir", transport=httpx.MockTransport(_handler))


__all__ = [
    "CHAT_PATH",
    "IR_TEST_BASE_URL",
    "VLLM_TEST_BASE_URL",
    "PlatformChat",
    "chat_completion",
    "common_ir_client",
    "posted_schema",
    "system_text",
    "use_ir_backend",
    "user_text",
]
