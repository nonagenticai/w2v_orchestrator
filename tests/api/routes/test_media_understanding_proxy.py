"""``POST /api/transcribe`` + ``/api/analyze*`` — merqi recon's A/V leg through the orchestrator.

Production stack (``test_p9_api_key_lane``'s harness: w2v_common ``AuthMiddleware``
with the ``w2v_sk_`` verifier, ``TenancyMiddleware``, the full ``TENANT_GATE``,
real tenants/keys rows); only the upstream socket is fake. Pinned:

* an API-key caller reaches both peers, with the key NEVER forwarded and the
  resolved tenant as ``X-Effective-Tenant-Id``;
* tenant isolation: tenant A's key cannot act for B (spoofed tenant headers are
  ignored; the effective tenant is the key's);
* upstream errors are MAPPED (caller errors pass through, a busy/unavailable
  429/503 keeps its status + ``Retry-After`` (#1111), credential refusals and
  other 5xx become 502, timeouts 504) and never echo the upstream body.
"""

from __future__ import annotations

import types
import uuid
from typing import Any

import httpx
import pytest

from src.api.byo_errors import register_byo_exception_handlers
from src.api.routes._common import TENANT_GATE
from src.api.routes.media_understanding import get_media_byo_posture
from src.api.routes.media_understanding import router as media_router
from src.services.providers.byo_posture import StaticByoPosture
from src.services.providers.errors import ByoPostureUnavailableError
from tests.api.routes.test_p9_api_key_lane import (  # noqa: F401 — the fixture
    _PUBLIC_PEER,
    _SA_TOKEN,
    Harness,
    harness,
)

pytestmark = pytest.mark.asyncio

#: Captured at import, before any fixture patches the shared ``httpx`` module
#: (the lane harness replaces ``httpx.AsyncClient`` process-wide).
_RealAsyncClient = httpx.AsyncClient


class _Upstream:
    """Records every proxied request; answers with ``status``/``body`` (or raises)."""

    def __init__(self) -> None:
        self.sent: list[httpx.Request] = []
        self.status = 200
        self.body: Any = {"text": "hello world", "segments": [{"start": 0.0, "text": "hello"}]}
        self.headers: dict[str, str] = {}
        self.raise_exc: Exception | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.sent.append(request)
        if self.raise_exc is not None:
            raise self.raise_exc
        return httpx.Response(self.status, json=self.body, headers=self.headers)


@pytest.fixture
def upstream(harness: Harness, monkeypatch) -> _Upstream:  # noqa: F811 — pytest fixture injection
    up = _Upstream()

    def _client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        return _RealAsyncClient(*args, transport=httpx.MockTransport(up.handler), **kwargs)

    import src.api.routes._proxy as proxy_mod

    fake_httpx = types.SimpleNamespace(**{n: getattr(httpx, n) for n in dir(httpx)})
    fake_httpx.AsyncClient = _client
    monkeypatch.setattr(proxy_mod, "httpx", fake_httpx)
    harness.app.include_router(media_router, dependencies=TENANT_GATE)
    register_byo_exception_handlers(harness.app)
    # Default posture: the tenant may use the platform ASR. BYO tests override it.
    harness.app.dependency_overrides[get_media_byo_posture] = lambda: StaticByoPosture(
        "local_allowed"
    )
    return up


async def _transcribe(
    harness: Harness,  # noqa: F811
    key: str,
    *,
    audio: bytes = b"ID3-fake-mp3",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=harness.app, client=_PUBLIC_PEER)
    async with _RealAsyncClient(transport=transport, base_url="http://testserver") as c:
        return await c.post(
            "/api/transcribe",
            files={"file": ("ep1.mp3", audio, "audio/mpeg")},
            data={"language": "en"},
            headers={"Authorization": f"Bearer {key}", **(headers or {})},
        )


def _assert_no_key_and_tenant(req: httpx.Request, key: str, tenant: uuid.UUID) -> None:
    leaked = {k: v for k, v in req.headers.items() if "w2v_sk_" in v or key in v}
    assert not leaked, f"tenant API key forwarded downstream: {sorted(leaked)}"
    assert req.headers["authorization"] == f"Bearer {_SA_TOKEN}"
    assert req.headers["x-effective-tenant-id"] == str(tenant)


async def test_api_key_caller_transcribes_through_the_proxy(
    harness: Harness,  # noqa: F811
    upstream: _Upstream,
) -> None:
    tid, key = await harness.tenant_with_key(uuid.uuid7())
    r = await _transcribe(harness, key)
    assert r.status_code == 200, r.text
    assert r.json() == upstream.body, "the ASR response is returned unchanged"
    [req] = upstream.sent
    assert req.method == "POST"
    assert req.url.path == "/api/transcribe"
    assert req.url.host == "asr", "the orchestrator's own ASR peer (discovery)"
    body = req.content
    assert b"ID3-fake-mp3" in body and b'name="file"; filename="ep1.mp3"' in body
    assert b'name="language"\r\n\r\nen' in body
    assert b'name="word_timestamps"\r\n\r\ntrue' in body
    _assert_no_key_and_tenant(req, key, tid)


async def test_api_key_caller_starts_polls_and_fetches_a_video_analysis(
    harness: Harness,  # noqa: F811
    upstream: _Upstream,
) -> None:
    tid, key = await harness.tenant_with_key(uuid.uuid7())
    upstream.status, upstream.body = 202, {"job_id": "abc123", "status": "pending"}
    start = await harness.call("POST", "/api/analyze", key, {"video_url": "https://v.test/a.mp4"})
    assert start.status_code == 202, start.text
    assert start.json()["job_id"] == "abc123"
    upstream.status, upstream.body = 200, {"status": "completed"}
    assert (await harness.call("GET", "/api/analyze/abc123", key)).status_code == 200
    assert (await harness.call("GET", "/api/analyze/abc123/result", key)).status_code == 200
    paths = [(r.method, r.url.path) for r in upstream.sent]
    assert paths == [
        ("POST", "/api/analyze"),
        ("GET", "/api/analyze/abc123"),
        ("GET", "/api/analyze/abc123/result"),
    ]
    assert upstream.sent[0].url.host == "video-analyzer-service.w2v"
    for req in upstream.sent:
        _assert_no_key_and_tenant(req, key, tid)


async def test_analyze_accepts_only_a_video_url(
    harness: Harness,  # noqa: F811
    upstream: _Upstream,
) -> None:
    _tid, key = await harness.tenant_with_key(uuid.uuid7())
    for body in (
        {"video_url": "https://v.test/a.mp4", "video_path": "/etc/passwd"},
        {"video_url": "file:///etc/passwd"},
        {},
    ):
        r = await harness.call("POST", "/api/analyze", key, body)
        assert r.status_code == 422, (body, r.text)
    assert upstream.sent == []


async def test_a_key_of_tenant_a_cannot_act_for_tenant_b(
    harness: Harness,  # noqa: F811
    upstream: _Upstream,
) -> None:
    tenant_a, key_a = await harness.tenant_with_key(uuid.uuid7())
    tenant_b, _key_b = await harness.tenant_with_key(uuid.uuid7())
    spoof = {
        "X-Effective-Tenant-Id": str(tenant_b),
        "X-Active-Tenant-Id": str(tenant_b),
        "X-Tenant-Id": str(tenant_b),
    }
    r = await _transcribe(harness, key_a, headers=spoof)
    assert r.status_code == 200, r.text
    transport = httpx.ASGITransport(app=harness.app, client=_PUBLIC_PEER)
    async with _RealAsyncClient(transport=transport, base_url="http://testserver") as c:
        poll = await c.get(
            "/api/analyze/job-of-b", headers={"Authorization": f"Bearer {key_a}", **spoof}
        )
    assert poll.status_code == 200, poll.text
    assert [req.headers["x-effective-tenant-id"] for req in upstream.sent] == [str(tenant_a)] * 2


@pytest.mark.parametrize(
    ("status", "body", "headers", "expected", "retry_after"),
    [(422, {"detail": "unsupported audio"}, {}, 422, None)],
)
async def test_upstream_errors_are_mapped_never_echoed(
    harness: Harness,  # noqa: F811
    upstream: _Upstream,
    status: int,
    body: Any,
    headers: dict[str, str],
    expected: int,
    retry_after: str | None,
) -> None:
    _tid, key = await harness.tenant_with_key(uuid.uuid7())
    upstream.status, upstream.body, upstream.headers = status, body, headers
    r = await _transcribe(harness, key)
    assert r.status_code == expected, r.text
    assert "eyJsecret" not in r.text
    assert r.headers.get("retry-after") == retry_after


async def test_busy_upstream_keeps_the_gpu_busy_contract_body(
    harness: Harness,  # noqa: F811
    upstream: _Upstream,
) -> None:
    """#1111: an upstream ``gpu_busy`` 503 reaches the caller as the same contract
    (503, ``Retry-After``, ``error``/``lane``/``retry_after``) — never a 502, and
    nothing beyond the whitelisted fields is echoed."""
    from src.api.error_details import register_object_detail_http_handler

    register_object_detail_http_handler(harness.app)  # as ``create_app`` does
    _tid, key = await harness.tenant_with_key(uuid.uuid7())
    upstream.status = 503
    upstream.body = {
        "detail": {"error": "gpu_busy", "lane": "asr", "retry_after": 30, "leak": "eyJsecret"}
    }
    upstream.headers = {"Retry-After": "30"}
    r = await _transcribe(harness, key)
    assert r.status_code == 503, r.text
    assert r.headers["retry-after"] == "30"
    assert r.json()["error"]["details"] == {
        "error": "gpu_busy",
        "message": "transcription_worker is busy; retry shortly",
        "lane": "asr",
        "retry_after": 30,
    }
    assert "eyJsecret" not in r.text


@pytest.mark.parametrize(
    ("exc", "expected"),
    [(httpx.ConnectError("refused"), 502)],
)
async def test_upstream_unreachable_or_slow(
    harness: Harness,  # noqa: F811
    upstream: _Upstream,
    exc: Exception,
    expected: int,
) -> None:
    _tid, key = await harness.tenant_with_key(uuid.uuid7())
    upstream.raise_exc = exc
    r = await harness.call("GET", "/api/analyze/abc123", key)
    assert r.status_code == expected, r.text


async def test_transcribe_body_over_the_cap_is_413_before_the_upstream(
    harness: Harness,  # noqa: F811
    upstream: _Upstream,
    monkeypatch,
) -> None:
    import src.api.routes.media_understanding as mu

    monkeypatch.setattr(mu, "TRANSCRIBE_MAX_AUDIO_BYTES", 1024)
    _tid, key = await harness.tenant_with_key(uuid.uuid7())
    r = await _transcribe(harness, key, audio=b"x" * 2048)
    assert r.status_code == 413, r.text
    assert upstream.sent == []


async def test_anonymous_caller_is_refused(
    harness: Harness,  # noqa: F811
    upstream: _Upstream,
) -> None:
    transport = httpx.ASGITransport(app=harness.app, client=_PUBLIC_PEER)
    async with _RealAsyncClient(transport=transport, base_url="http://testserver") as c:
        r = await c.post("/api/transcribe", files={"file": ("a.mp3", b"x", "audio/mpeg")})
        a = await c.get("/api/analyze/abc123")
    assert r.status_code == 401 and a.status_code == 401
    assert upstream.sent == []


# --- BYO: /api/transcribe hits a PLATFORM GPU (asr) — same #1106 refusal as runs ---


class _PostureDown:
    async def byo_mode(self, tenant_id: uuid.UUID, *, slot: str) -> str:
        raise ByoPostureUnavailableError(slot, "licences unreachable")


async def test_byo_required_tenant_is_refused_before_the_asr_peer_is_called(
    harness: Harness,  # noqa: F811
    upstream: _Upstream,
) -> None:
    tid, key = await harness.tenant_with_key(uuid.uuid7())
    posture = StaticByoPosture("byo_required")
    harness.app.dependency_overrides[get_media_byo_posture] = lambda: posture
    r = await _transcribe(harness, key)
    assert r.status_code == 402, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "byo_provider_required"
    assert detail["modality"] == "asr"
    assert upstream.sent == [], "a byo_required tenant must not reach the platform ASR"
    assert posture.calls == [tid], "the posture is read for the RESOLVED tenant"


async def test_posture_outage_fails_closed_with_503(
    harness: Harness,  # noqa: F811
    upstream: _Upstream,
) -> None:
    _tid, key = await harness.tenant_with_key(uuid.uuid7())
    harness.app.dependency_overrides[get_media_byo_posture] = _PostureDown
    r = await _transcribe(harness, key)
    assert r.status_code == 503, r.text
    assert r.headers["retry-after"] == "30"
    assert upstream.sent == [], "an unknown posture is never treated as local_allowed"


async def test_platform_only_and_local_allowed_tenants_still_transcribe(
    harness: Harness,  # noqa: F811
    upstream: _Upstream,
) -> None:
    for mode in ("local_allowed", "platform_only"):
        _tid, key = await harness.tenant_with_key(uuid.uuid7())
        harness.app.dependency_overrides[get_media_byo_posture] = lambda m=mode: StaticByoPosture(m)
        r = await _transcribe(harness, key)
        assert r.status_code == 200, (mode, r.text)
    assert len(upstream.sent) == 2
