"""HttpProviderTester — the live test a tenant connection must pass before it is stored.

The HTTP boundary is an ``httpx.MockTransport`` handed to the REAL tester, so the
requests built, the SSRF screen (``w2v_common.net.safe_request``) and the result
sanitising all run for real. ``private_ok`` + an allow-listed host keeps the tests
DNS-free; the hosted-mode case uses an IP literal (no lookup needed).
"""

from __future__ import annotations

import json

import httpx
import pytest

from src.services.providers.settings import ProviderPolicySettings
from src.services.providers.tester import HttpProviderTester

pytestmark = pytest.mark.asyncio

SECRET = "sk-test-DO-NOT-LEAK-91f3"  # pragma: allowlist secret
_ALLOWED = ProviderPolicySettings(ssrf_mode="private_ok", egress_allowlist=frozenset({"llm.test"}))


class _Upstream:
    def __init__(self, *responses: httpx.Response) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responses.pop(0)


def _tester(upstream: _Upstream, policy: ProviderPolicySettings = _ALLOWED) -> HttpProviderTester:
    return HttpProviderTester(policy, transport=httpx.MockTransport(upstream))


async def test_openai_compatible_lists_models_with_the_bearer_key() -> None:
    up = _Upstream(httpx.Response(200, json={"data": []}))
    result = await _tester(up).test(
        slot="llm",
        kind="openai-compatible",
        base_url="https://llm.test/v1/",
        model=None,
        options={},
        secret=SECRET,
    )
    assert result.ok, result
    assert result.upstream_status == 200
    [req] = up.requests
    assert (req.method, str(req.url)) == ("GET", "https://llm.test/v1/models")
    assert req.headers["authorization"] == f"Bearer {SECRET}"


async def test_openai_compatible_falls_back_to_a_one_token_chat_when_models_is_404() -> None:
    up = _Upstream(httpx.Response(404), httpx.Response(200, json={"choices": []}))
    result = await _tester(up).test(
        slot="llm",
        kind="openai-compatible",
        base_url="https://llm.test/v1",
        model="m-1",
        options={},
        secret=SECRET,
    )
    assert result.ok, result
    assert [r.method for r in up.requests] == ["GET", "POST"]
    chat = up.requests[1]
    assert str(chat.url) == "https://llm.test/v1/chat/completions"
    body = json.loads(chat.content)
    assert body["model"] == "m-1" and body["max_tokens"] == 1


async def test_models_404_without_a_model_fails_instead_of_guessing() -> None:
    up = _Upstream(httpx.Response(404))
    result = await _tester(up).test(
        slot="llm",
        kind="openai-compatible",
        base_url="https://llm.test/v1",
        model=None,
        options={},
        secret=None,
    )
    assert not result.ok
    assert result.upstream_status == 404
    assert len(up.requests) == 1


async def test_rejected_key_fails_without_echoing_the_key_or_the_upstream_body() -> None:
    # A real provider 401 body can quote part of the key; it must never be relayed.
    up = _Upstream(httpx.Response(401, json={"error": f"Incorrect API key provided: {SECRET}"}))
    result = await _tester(up).test(
        slot="llm",
        kind="openai-compatible",
        base_url="https://llm.test/v1",
        model="m",
        options={},
        secret=SECRET,
    )
    assert not result.ok
    assert result.upstream_status == 401
    assert SECRET not in result.message
    assert "credential was rejected" in result.message


async def test_anthropic_sends_a_minimal_messages_call() -> None:
    up = _Upstream(httpx.Response(200, json={"content": []}))
    result = await _tester(up).test(
        slot="llm",
        kind="anthropic",
        base_url="https://llm.test/v1",
        model="claude-x",
        options={},
        secret=SECRET,
    )
    assert result.ok, result
    [req] = up.requests
    assert (req.method, str(req.url)) == ("POST", "https://llm.test/v1/messages")
    assert req.headers["x-api-key"] == SECRET
    assert req.headers["anthropic-version"]
    assert json.loads(req.content) == {
        "model": "claude-x",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "ping"}],
    }


async def test_redirect_into_a_private_range_is_refused_and_not_followed() -> None:
    up = _Upstream(httpx.Response(302, headers={"Location": "http://10.0.0.7/steal"}))
    result = await _tester(up).test(
        slot="llm",
        kind="openai-compatible",
        base_url="https://llm.test/v1",
        model="m",
        options={},
        secret=SECRET,
    )
    assert not result.ok
    assert "refused by egress policy" in result.message
    assert [str(r.url) for r in up.requests] == ["https://llm.test/v1/models"]


async def test_hosted_mode_refuses_a_private_endpoint_before_any_request() -> None:
    up = _Upstream()
    result = await _tester(up, ProviderPolicySettings(ssrf_mode="hosted")).test(
        slot="llm",
        kind="openai-compatible",
        base_url="http://192.168.1.10:8001/v1",
        model="m",
        options={},
        secret=SECRET,
    )
    assert not result.ok
    assert up.requests == []


async def test_transport_error_is_sanitised() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    tester = HttpProviderTester(_ALLOWED, transport=httpx.MockTransport(boom))
    result = await tester.test(
        slot="llm",
        kind="openai-compatible",
        base_url="https://llm.test/v1",
        model="m",
        options={},
        secret=SECRET,
    )
    assert not result.ok
    assert result.upstream_status is None
    assert "ConnectTimeout" in result.message
    assert SECRET not in result.message


# ─── openai-compatible llm: the configured model must exist ───────────────────


async def _llm(up: _Upstream, model: str | None):
    return await _tester(up).test(
        slot="llm",
        kind="openai-compatible",
        base_url="https://llm.test/v1",
        model=model,
        options={},
        secret=SECRET,
    )


async def test_listed_model_passes_with_one_request() -> None:
    up = _Upstream(httpx.Response(200, json={"data": [{"id": "other"}, {"id": "m-1"}]}))
    result = await _llm(up, "m-1")
    assert result.ok, result
    assert "lists model 'm-1'" in result.message
    assert [r.method for r in up.requests] == ["GET"]


async def test_unlisted_model_that_chats_passes() -> None:
    """A listing that lacks the model is not proof it is unserved (Azure deployment
    names, Gemini ``models/`` prefix, Ollama ``:latest``): the chat call decides."""
    up = _Upstream(
        httpx.Response(200, json={"data": [{"id": "models/gemini-2.5-pro"}]}),
        httpx.Response(200, json={"choices": [{"message": {"content": "p"}}]}),
    )
    result = await _llm(up, "gemini-2.5-pro")
    assert result.ok, result
    assert result.upstream_status == 200
    assert [(r.method, r.url.path) for r in up.requests] == [
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
    ]
    chat = up.requests[1]
    assert json.loads(chat.content)["model"] == "gemini-2.5-pro"
    assert chat.headers["authorization"] == f"Bearer {SECRET}"


async def test_unlisted_model_whose_chat_404s_fails_naming_the_model() -> None:
    """The original bug stays fixed: a typo'd model is not stored — the chat call
    against it fails, and the message names the model and the chat error."""
    up = _Upstream(
        httpx.Response(200, json={"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]}),
        httpx.Response(404, json={"error": "UPSTREAM-BODY-xyzzy"}),
    )
    result = await _llm(up, "gpt-4o-typo")
    assert not result.ok
    assert result.upstream_status == 404
    assert "'gpt-4o-typo'" in result.message
    assert "chat completion failed" in result.message
    assert "HTTP 404" in result.message
    assert "xyzzy" not in result.message, "upstream body must not be echoed"
    assert SECRET not in result.message
    assert [r.method for r in up.requests] == ["GET", "POST"]


async def test_listed_model_passes_without_a_chat_call() -> None:
    up = _Upstream(httpx.Response(200, json={"models": [{"name": "llama3:8b"}]}))
    result = await _llm(up, "llama3:8b")
    assert result.ok, result
    assert [r.method for r in up.requests] == ["GET"], "listed ⇒ fast path, no chat"


async def test_paginated_listing_without_the_model_falls_through_to_chat() -> None:
    up = _Upstream(
        httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "m-a"}], "has_more": True, "last_id": "m-a"},
        ),
        httpx.Response(200, json={"choices": []}),
    )
    result = await _llm(up, "m-z")
    assert result.ok, result
    assert [r.method for r in up.requests] == ["GET", "POST"]
    assert json.loads(up.requests[1].content)["model"] == "m-z"


async def test_empty_listing_with_a_failing_chat_fails() -> None:
    up = _Upstream(httpx.Response(200, json={"data": []}), httpx.Response(400))
    result = await _llm(up, "m-1")
    assert not result.ok
    assert "'m-1'" in result.message
    assert "check the model name" in result.message


@pytest.mark.parametrize("status", [404])
async def test_unsupported_listing_falls_back_to_chat_against_the_model(status: int) -> None:
    up = _Upstream(httpx.Response(status), httpx.Response(200, json={"choices": []}))
    result = await _llm(up, "m-1")
    assert result.ok, result
    assert [r.method for r in up.requests] == ["GET", "POST"]
    assert json.loads(up.requests[1].content)["model"] == "m-1"


async def test_non_listing_200_body_falls_back_to_chat_and_its_404_fails() -> None:
    """A 2xx that is not a model listing proves nothing about the model: the chat
    completion against it decides — and a 404 there (unknown model) is a failure."""
    up = _Upstream(
        httpx.Response(200, text="<html>ok</html>"),
        httpx.Response(404, json={"error": "model not found"}),
    )
    result = await _llm(up, "ghost-model")
    assert not result.ok
    assert result.upstream_status == 404
    assert [r.method for r in up.requests] == ["GET", "POST"]


async def test_chat_fallback_rejecting_the_model_fails() -> None:
    up = _Upstream(httpx.Response(404), httpx.Response(400, json={"error": "bad model"}))
    result = await _llm(up, "m-bad")
    assert not result.ok
    assert result.upstream_status == 400
    assert "check the model name" in result.message


async def test_unsupported_listing_without_a_model_still_fails() -> None:
    up = _Upstream(httpx.Response(405))
    result = await _llm(up, None)
    assert not result.ok
    assert result.upstream_status == 405
    assert len(up.requests) == 1


async def test_listing_auth_failure_is_reported_before_any_model_check() -> None:
    up = _Upstream(httpx.Response(401))
    result = await _llm(up, "m-1")
    assert not result.ok
    assert "credential was rejected" in result.message
