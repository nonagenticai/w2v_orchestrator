"""A keyless openai-compatible llm/vlm connection (self-hosted vLLM, no auth) runs.

The registry declares the key optional (``secret_required=False``) for
``openai-compatible`` llm and vlm, but ``build_model`` refused any connection
without a key at runtime. It must now build, send NO ``Authorization`` header,
and never fall back to the process's ``OPENAI_API_KEY`` (a platform key must
never reach a tenant endpoint).
"""

from __future__ import annotations

from uuid import UUID

import httpx2
import pytest
from pydantic import SecretStr
from pydantic_ai.models.openai import OpenAIChatModel

from src.services.llm import in_process
from src.services.providers.errors import SlotUnavailableError
from src.services.providers.registry import KIND_ANTHROPIC, KIND_OPENAI_COMPATIBLE
from src.services.providers.resolver import ResolvedSlot, SlotRef

CONN = UUID("01990000-0000-7000-8000-000000000005")
BASE = "https://vllm.example.com/v1"


def _resolved(kind: str, secret: str | None, slot: str = "llm") -> ResolvedSlot:
    return ResolvedSlot(
        ref=SlotRef(slot=slot, source="tenant", kind=kind, connection_id=CONN, version=1),
        base_url=BASE,
        model="qwen",
        options={},
        secret=SecretStr(secret) if secret else None,
        pinned_ip=None,
    )


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> list[httpx2.Request]:
    requests: list[httpx2.Request] = []

    def _handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"object": "list", "data": []})

    monkeypatch.setattr(
        in_process, "inner_transport_factory", lambda: httpx2.MockTransport(_handler)
    )
    # A platform key in the env must NOT be picked up for a keyless tenant endpoint.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-platform-must-not-leak")  # pragma: allowlist secret
    return requests


@pytest.mark.asyncio
@pytest.mark.parametrize("slot", ["llm"])
async def test_keyless_openai_compatible_builds_and_sends_no_authorization(
    seen: list[httpx2.Request], slot: str
) -> None:
    model, http = in_process.build_model(_resolved(KIND_OPENAI_COMPATIBLE, None, slot), timeout=5.0)
    try:
        assert isinstance(model, OpenAIChatModel)
        await model.client.models.list()
    finally:
        await http.aclose()
    assert len(seen) == 1
    assert str(seen[0].url).startswith(BASE)
    assert "authorization" not in {k.lower() for k in seen[0].headers}


@pytest.mark.asyncio
async def test_a_configured_key_is_still_sent_as_bearer(seen: list[httpx2.Request]) -> None:
    model, http = in_process.build_model(
        _resolved(KIND_OPENAI_COMPATIBLE, "tenant-key-123"), timeout=5.0
    )
    try:
        assert isinstance(model, OpenAIChatModel)
        await model.client.models.list()
    finally:
        await http.aclose()
    assert seen[0].headers["authorization"] == "Bearer tenant-key-123"


def test_anthropic_still_requires_a_key() -> None:
    with pytest.raises(SlotUnavailableError, match="no API key"):
        in_process.build_model(_resolved(KIND_ANTHROPIC, None), timeout=5.0)
