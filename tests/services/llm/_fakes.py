"""Fake tenant LLM providers (httpx2 transports) for the in-process LLM tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import httpx2
from pydantic import SecretStr

from src.services.providers.resolver import ResolvedSlot, SlotRef

SECRET = "sk-tenant-SECRET-0123456789"  # pragma: allowlist secret - fake fixture key
CONN_ID = UUID("01990000-0000-7000-8000-000000000001")


def tenant_slot(
    kind: str = "openai-compatible",
    *,
    base_url: str | None = "https://llm.tenant.example/v1",
    model: str = "tenant-model-7",
    pinned_ip: str | None = "203.0.113.7",
    version: int = 3,
) -> ResolvedSlot:
    return ResolvedSlot(
        ref=SlotRef(slot="llm", source="tenant", kind=kind, connection_id=CONN_ID, version=version),
        base_url=base_url,
        model=model,
        secret=SecretStr(SECRET),
        pinned_ip=pinned_ip,
    )


@dataclass
class FakeProvider:
    """Records every request; answers with ``reply(body, n)`` (n = 0-based call index)."""

    reply: Callable[[dict[str, Any], int], httpx2.Response]
    requests: list[httpx2.Request] = field(default_factory=list)
    bodies: list[dict[str, Any]] = field(default_factory=list)

    def transport(self) -> httpx2.MockTransport:
        def handler(request: httpx2.Request) -> httpx2.Response:
            body = json.loads(request.content or b"{}")
            self.requests.append(request)
            self.bodies.append(body)
            return self.reply(body, len(self.bodies) - 1)

        return httpx2.MockTransport(handler)


def openai_reply(
    outputs: list[Any] | None = None,
    *,
    text: str = "plain answer",
    usage: tuple[int, int] = (11, 7),
) -> Callable[[dict[str, Any], int], httpx2.Response]:
    """Chat Completions answer: a tool call carrying ``outputs[n]`` when tools were sent."""

    def reply(body: dict[str, Any], n: int) -> httpx2.Response:
        message: dict[str, Any] = {"role": "assistant", "content": text}
        finish = "stop"
        tools = body.get("tools") or []
        if tools and outputs is not None:
            out = outputs[min(n, len(outputs) - 1)]
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{n}",
                        "type": "function",
                        "function": {
                            "name": tools[0]["function"]["name"],
                            "arguments": json.dumps(out),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        return httpx2.Response(
            200,
            json={
                "id": f"chatcmpl-{n}",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model"),
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": usage[0],
                    "completion_tokens": usage[1],
                    "total_tokens": sum(usage),
                },
            },
        )

    return reply


def anthropic_reply(
    text: str = "anthropic answer", usage: tuple[int, int] = (5, 3)
) -> Callable[[dict[str, Any], int], httpx2.Response]:
    def reply(body: dict[str, Any], n: int) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "id": f"msg_{n}",
                "type": "message",
                "role": "assistant",
                "model": body.get("model"),
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
            },
        )

    return reply


def status_reply(
    status: int, body: dict[str, Any]
) -> Callable[[dict[str, Any], int], httpx2.Response]:
    def reply(_body: dict[str, Any], _n: int) -> httpx2.Response:
        return httpx2.Response(status, json=body)

    return reply
