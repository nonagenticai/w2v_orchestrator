"""P5 test-on-save probes — the REAL ``HttpProviderTester`` against an ``httpx.MockTransport``.

The SSRF-safe ``safe_request`` path runs for real. ``private_ok`` with the provider
hosts allow-listed keeps the tests DNS-free; one hosted-mode case proves a private
``base_url`` never receives the key.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from src.services.providers.settings import ProviderPolicySettings
from src.services.providers.tester import HttpProviderTester

pytestmark = pytest.mark.asyncio

SECRET = "sk-p5-DO-NOT-LEAK-c0ffee"  # pragma: allowlist secret
_POLICY = ProviderPolicySettings(
    ssrf_mode="private_ok",
    egress_allowlist=frozenset(
        {
            "api.openai.com",
            "api.elevenlabs.io",
            "generativelanguage.googleapis.com",
            "api.browserstack.com",
            "tts.selfhost.test",
        }
    ),
)


class Upstream:
    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json={"echo": "x"})


def _tester(up: Upstream, policy: ProviderPolicySettings = _POLICY) -> HttpProviderTester:
    return HttpProviderTester(policy, transport=httpx.MockTransport(up))


@pytest.mark.parametrize(
    ("slot", "kind", "base_url", "url", "header", "value"),
    [
        (
            "tts",
            "openai",
            None,
            "https://api.openai.com/v1/models",
            "authorization",
            f"Bearer {SECRET}",
        )
    ],
)
async def test_each_kind_probes_its_auth_only_endpoint(
    slot: str, kind: str, base_url: str | None, url: str, header: str, value: str
) -> None:
    up = Upstream(200)
    result = await _tester(up).test(
        slot=slot, kind=kind, base_url=base_url, model=None, options={"voice": "v"}, secret=SECRET
    )
    assert result.ok and result.probed, result
    [req] = up.requests
    assert (req.method, str(req.url)) == ("GET", url)
    assert req.headers[header] == value
    assert req.content == b""  # auth-only: nothing generated


async def test_browserstack_probe_uses_basic_auth_on_the_fixed_plan_endpoint() -> None:
    up = Upstream(200)
    result = await _tester(up).test(
        slot="device_farm",
        kind="browserstack",
        base_url=None,
        model=None,
        options={"username": "acme"},
        secret=SECRET,
    )
    assert result.ok and result.probed
    [req] = up.requests
    assert str(req.url) == "https://api.browserstack.com/automate/plan.json"
    expected = base64.b64encode(f"acme:{SECRET}".encode()).decode()
    assert req.headers["authorization"] == f"Basic {expected}"


async def test_browserstack_without_username_fails_without_a_request() -> None:
    up = Upstream(200)
    result = await _tester(up).test(
        slot="device_farm",
        kind="browserstack",
        base_url=None,
        model=None,
        options={},
        secret=SECRET,
    )
    assert not result.ok
    assert up.requests == []


@pytest.mark.parametrize("status", [401])
async def test_rejected_credential_is_a_failure_that_never_echoes_the_key(status: int) -> None:
    up = Upstream(status)
    result = await _tester(up).test(
        slot="tts", kind="elevenlabs", base_url=None, model=None, options={}, secret=SECRET
    )
    assert not result.ok and result.probed
    assert result.upstream_status == status
    assert SECRET not in result.message


@pytest.mark.parametrize("kind", ["runway"])
async def test_kinds_without_an_auth_endpoint_are_honestly_not_probed(kind: str) -> None:
    up = Upstream(200)
    result = await _tester(up).test(
        slot="video_gen", kind=kind, base_url=None, model=None, options={}, secret=SECRET
    )
    assert result.probed is False
    assert result.message.startswith("not probed")
    assert up.requests == []  # nothing was sent, so nothing is claimed


async def test_hosted_mode_refuses_a_private_base_url_before_sending_the_key() -> None:
    up = Upstream(200)
    result = await _tester(up, ProviderPolicySettings(ssrf_mode="hosted")).test(
        slot="tts",
        kind="openai-compatible",
        base_url="http://10.0.0.7/v1",
        model="m",
        options={"voice": "v"},
        secret=SECRET,
    )
    assert not result.ok
    assert up.requests == []


class _ModelsMissing:
    """A self-hosted speech server with no ``/models``: 404 there, ``speech`` on synth."""

    def __init__(self, speech_status: int) -> None:
        self.speech_status = speech_status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(404, json={"detail": "Not Found"})
        return httpx.Response(self.speech_status, content=b"RIFF")


@pytest.mark.parametrize(("speech_status", "ok"), [(200, True)])
async def test_tts_openai_compatible_without_models_falls_back_to_a_tiny_synth(
    speech_status: int, ok: bool
) -> None:
    """``GET /models`` 404 → a one-word ``POST /audio/speech``; only its 2xx passes."""
    up = _ModelsMissing(speech_status)
    tester = HttpProviderTester(_POLICY, transport=httpx.MockTransport(up))
    result = await tester.test(
        slot="tts",
        kind="openai-compatible",
        base_url="https://tts.selfhost.test/v1",
        model="kokoro",
        options={"voice": "af_heart"},
        secret=SECRET,
    )

    assert result.ok is ok and result.probed, result
    assert result.upstream_status == speech_status
    assert SECRET not in result.message
    get, post = up.requests
    assert (get.method, str(get.url)) == ("GET", "https://tts.selfhost.test/v1/models")
    assert (post.method, str(post.url)) == ("POST", "https://tts.selfhost.test/v1/audio/speech")
    assert post.headers["authorization"] == f"Bearer {SECRET}"
    body = json.loads(post.content)
    assert body["input"] == "test"
    assert body["model"] == "kokoro"
    assert body["voice"] == "af_heart"
