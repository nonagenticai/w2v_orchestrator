"""``HttpProviderTester.list_models`` — ``POST /api/tenant/connections/{slot}/models``.

Same harness as ``test_tester.py``: a MockTransport behind the REAL tester, so the
SSRF screen and result sanitising run; allow-listed hosts keep it DNS-free.
"""

from __future__ import annotations

import httpx
import pytest

from src.services.providers.settings import ProviderPolicySettings
from src.services.providers.tester import HttpProviderTester

pytestmark = pytest.mark.asyncio

SECRET = "sk-models-DO-NOT-LEAK-77aa"  # pragma: allowlist secret
_ALLOWED = ProviderPolicySettings(
    ssrf_mode="private_ok",
    egress_allowlist=frozenset({"llm.test", "api.anthropic.com", "api.elevenlabs.io"}),
)


class _Upstream:
    def __init__(self, *responses: httpx.Response) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responses.pop(0)


def _tester(up: _Upstream, policy: ProviderPolicySettings = _ALLOWED) -> HttpProviderTester:
    return HttpProviderTester(policy, transport=httpx.MockTransport(up))


async def test_openai_compatible_lists_sorted_ids_with_the_bearer_key() -> None:
    up = _Upstream(httpx.Response(200, json={"data": [{"id": "b"}, {"id": "a"}]}))
    result = await _tester(up).list_models(
        slot="llm", kind="openai-compatible", base_url="https://llm.test/v1/", secret=SECRET
    )
    assert result.models == ["a", "b"]
    [req] = up.requests
    assert (req.method, str(req.url)) == ("GET", "https://llm.test/v1/models")
    assert req.headers["authorization"] == f"Bearer {SECRET}"


async def test_anthropic_uses_its_own_auth_and_v1_path() -> None:
    up = _Upstream(httpx.Response(200, json={"data": [{"id": "claude-x", "type": "model"}]}))
    result = await _tester(up).list_models(
        slot="llm", kind="anthropic", base_url=None, secret=SECRET
    )
    assert result.models == ["claude-x"]
    [req] = up.requests
    assert str(req.url) == "https://api.anthropic.com/v1/models"
    assert req.headers["x-api-key"] == SECRET
    assert "authorization" not in req.headers


async def test_elevenlabs_reads_model_id_only() -> None:
    up = _Upstream(
        httpx.Response(200, json=[{"model_id": "eleven_v2", "name": "Eleven Multilingual"}])
    )
    result = await _tester(up).list_models(
        slot="tts", kind="elevenlabs", base_url=None, secret=SECRET
    )
    assert result.models == ["eleven_v2"]
    assert up.requests[0].headers["xi-api-key"] == SECRET


async def test_an_upstream_error_is_a_message_without_the_key_or_body() -> None:
    up = _Upstream(httpx.Response(401, json={"error": f"bad key {SECRET}"}))
    result = await _tester(up).list_models(
        slot="llm", kind="openai-compatible", base_url="https://llm.test/v1", secret=SECRET
    )
    assert result.models is None and result.upstream_status == 401
    assert SECRET not in (result.message or "")
    assert "credential was rejected" in (result.message or "")


async def test_a_body_that_is_not_a_listing_is_unavailable() -> None:
    up = _Upstream(httpx.Response(200, text="hello"))
    result = await _tester(up).list_models(
        slot="llm", kind="openai-compatible", base_url="https://llm.test/v1", secret=None
    )
    assert result.models is None and "did not return a model listing" in (result.message or "")


async def test_egress_refusal_sends_nothing() -> None:
    up = _Upstream()
    hosted = ProviderPolicySettings(ssrf_mode="hosted")
    result = await _tester(up, hosted).list_models(
        slot="llm", kind="openai-compatible", base_url="http://10.0.0.5/v1", secret=SECRET
    )
    assert result.models is None and "refused by egress policy" in (result.message or "")
    assert up.requests == []


@pytest.mark.parametrize(("kind", "base_url"), [("runway", None)])
async def test_kinds_that_cannot_list(kind: str, base_url: str | None) -> None:
    up = _Upstream()
    result = await _tester(up).list_models(slot="llm", kind=kind, base_url=base_url, secret=SECRET)
    assert result.models is None
    assert up.requests == []
