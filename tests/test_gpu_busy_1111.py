"""#1111 — the three paths #1108 left without GPU busy handling.

* image-gen (ACG): an asset download answered busy (429/503) is waited out per
  ``Retry-After`` within the deadline, then ``GpuBusyError``; a concept-image job
  that failed because ACG's ``local_image`` backend stayed busy is ALSO a
  ``GpuBusyError`` (transient, ``external``) — never a terminal ``RuntimeError``;
* TTS voice preview: a busy lane is bounded by its own wall-clock budget (not the
  2-try transport cap), honouring ``Retry-After``; the exhausted error carries the
  backend's last hint so the route can pass it on.

(``/api/transcribe``'s 503 → 503 + ``Retry-After`` mapping is pinned in
``tests/api/routes/test_media_understanding_proxy.py``.)
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator

import httpx
import pytest

import src.clients.ai_content_generation_client as acg_mod
import src.clients.tts_client as tts_mod
from src.clients.ai_content_generation_client import (
    IMAGE_BUSY_MARKER,
    AIContentGenerationClient,
)
from src.clients.gpu_busy import GpuBusyError, get_gpu_admission_settings
from src.clients.tts_client import TTSClient
from src.models.failure_class import EXTERNAL, attribute_exception
from src.services.stages.upstream import PermanentUpstreamError, TransientUpstreamError


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
    """Instant (recorded) sleeps and a known 60s busy deadline per test."""
    slept: list[float] = []

    async def _sleep(seconds: float) -> None:
        slept.append(seconds)
        await asyncio.sleep(0)

    monkeypatch.setattr(acg_mod, "_sleep", _sleep)
    monkeypatch.setattr(tts_mod, "_sleep", _sleep)
    monkeypatch.setenv("GPU_BUSY_DEADLINE_SECONDS", "60")
    get_gpu_admission_settings.cache_clear()
    yield slept
    get_gpu_admission_settings.cache_clear()


def _busy(retry_after: str | None = None, *, status: int = 503, lane: str = "image"):
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return httpx.Response(
        status, json={"error": "gpu_busy", "lane": lane, "retry_after": 3}, headers=headers
    )


class _Seq:
    """MockTransport handler: answers ``responses`` in order (last repeats)."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses[min(len(self.requests), len(self.responses)) - 1]


def _acg(handler: Callable[[httpx.Request], httpx.Response]) -> AIContentGenerationClient:
    client = AIContentGenerationClient(base_url="http://acg.test", license_token="svc")
    client.set_transport(httpx.MockTransport(handler))
    return client


# ── image-gen (ACG) downloads ────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["download_concept_image"])
async def test_acg_download_busy_is_waited_out_then_succeeds(
    _fresh: list[float], method: str
) -> None:
    seq = _Seq(_busy("7"), _busy(status=429), httpx.Response(200, content=b"PNG"))

    data = await getattr(_acg(seq), method)("/api/concept-images/assets/a1/download")

    assert data == b"PNG"
    assert len(seq.requests) == 3
    assert _fresh == [7.0, 3.0]  # header, then the body's retry_after


async def test_acg_download_busy_past_deadline_is_a_transient_gpu_busy_error(
    _fresh: list[float],
) -> None:
    seq = _Seq(_busy("25"))

    with pytest.raises(GpuBusyError) as info:
        await asyncio.wait_for(_acg(seq).download_concept_image("/x"), timeout=10)

    err = info.value
    assert _fresh == [25.0, 25.0] and len(seq.requests) == 3  # a 3rd 25s does not fit 60s
    assert isinstance(err, TransientUpstreamError)
    assert not isinstance(err, PermanentUpstreamError)
    assert err.service == "ai_content_generation" and err.lane == "image"
    assert err.status_code == 503 and err.retry_after == 25.0
    assert attribute_exception(err) == EXTERNAL


async def test_acg_download_other_errors_still_raise_with_body(_fresh: list[float]) -> None:
    seq = _Seq(httpx.Response(404, json={"detail": "no such asset"}))

    with pytest.raises(httpx.HTTPStatusError, match="no such asset"):
        await _acg(seq).download_concept_image("/x")

    assert len(seq.requests) == 1 and _fresh == []


# ── image-gen (ACG) job that failed busy ─────────────────────────────────────


def _failed_job(error: str) -> Callable[[httpx.Request], httpx.Response]:
    def acg(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"job_id": "j1", "status": "failed", "error": error})

    return acg


@pytest.mark.parametrize("status", [503, 429])
async def test_acg_job_failed_on_a_busy_backend_is_gpu_busy_not_terminal(status: int) -> None:
    client = _acg(_failed_job(f"local_image: backend returned {status}"))

    with pytest.raises(GpuBusyError) as info:
        await client.wait_for_concept_image("j1", timeout=5)

    err = info.value
    assert isinstance(err, TransientUpstreamError)
    assert err.status_code == status and err.lane == "image"
    assert attribute_exception(err) == EXTERNAL


@pytest.mark.parametrize(
    "error", ["local_image: backend returned 500", "local_image: backend returned 5031"]
)
async def test_acg_job_failed_otherwise_stays_a_runtime_error(error: str) -> None:
    with pytest.raises(RuntimeError) as info:
        await _acg(_failed_job(error)).wait_for_concept_image("j1", timeout=5)

    assert not isinstance(info.value, GpuBusyError)


def test_vendored_acg_busy_marker_matches_local_image_wording() -> None:
    # ACG local_image: f"local_image: backend returned {status}" for 429/503.
    for status in (429, 503):
        assert IMAGE_BUSY_MARKER.search("local_image: backend returned " + str(status))


# ── TTS voice preview busy budget ────────────────────────────────────────────


def _tts(handler: _Seq) -> TTSClient:
    client = TTSClient(base_url="http://modality-gateway.test", license_token="svc")
    client.set_transport(httpx.MockTransport(handler))
    return client


def _tts_ok() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "audio_base64": "UklGRg==",
            "duration_sec": 1.0,
            "sample_rate": 24000,
            "provider": "kokoro",
            "voice": "af_heart",
            "format": "wav",
        },
        headers={"X-TTS-LUFS": "-18.0"},
    )


async def test_preview_busy_budget_outlasts_the_two_try_cap(_fresh: list[float]) -> None:
    """attempts=2 alone gave up on the 2nd busy answer; with a busy budget the
    preview sits out a full lane (here 2 x 25s, inside the 60s busy share) and plays."""
    seq = _Seq(_busy("25", lane="kokoro"), _busy("25"), _tts_ok())

    result = await asyncio.wait_for(
        _tts(seq).synthesize(text="hi", attempts=2, busy_deadline_seconds=90.0), timeout=10
    )

    assert result.audio_bytes == b"RIFF"
    assert _fresh == [25.0, 25.0] and len(seq.requests) == 3


async def test_preview_in_flight_call_is_capped_to_the_remaining_budget(
    _fresh: list[float],
) -> None:
    """Budget + the in-flight synth end by ~95s (Cloudflare cuts at ~100s): each
    attempt's timeout is what is left of 90 + 5, never the 300s default."""
    seq = _Seq(_busy("25", lane="kokoro"), _busy("25"), _tts_ok())

    await asyncio.wait_for(
        _tts(seq).synthesize(text="hi", attempts=2, busy_deadline_seconds=90.0), timeout=10
    )

    reads = [req.extensions["timeout"]["read"] for req in seq.requests]
    assert 90.0 < reads[0] <= 95.0
    assert 65.0 < reads[1] <= 70.0  # 95 - 25 slept
    assert 40.0 < reads[2] <= 45.0  # 95 - 50 slept
    assert all(sum(_fresh[:i]) + reads[i] <= 95.0 for i in range(len(reads)))


async def test_three_busy_30s_answers_end_as_gpu_busy_never_502(_fresh: list[float]) -> None:
    """Busy waits stop at budget - synth reserve (60s): the 3rd Retry-After: 30 does
    not fit, so the preview answers busy (GpuBusyError → 503) instead of waiting
    on and leaving the synth no time."""
    seq = _Seq(_busy("30", lane="kokoro"), _busy("30"), _busy("30"), _tts_ok())

    with pytest.raises(GpuBusyError) as info:
        await asyncio.wait_for(
            _tts(seq).synthesize(text="hi", attempts=2, busy_deadline_seconds=90.0), timeout=10
        )

    assert len(seq.requests) == 3 and sum(_fresh) <= 60.0
    assert info.value.retry_after == 30.0


async def test_slow_synth_after_busy_waits_has_real_time_then_is_busy_not_502(
    _fresh: list[float],
) -> None:
    """2 x Retry-After: 30, then a synth too slow for what is left: the synth gets
    >= 30s (the reserve), and its timeout ends as GpuBusyError (→ 503), not a
    TTSSynthesisError (→ 502)."""
    answers = iter([_busy("30", lane="kokoro"), _busy("30")])
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        nxt = next(answers, None)
        if nxt is None:
            raise httpx.ReadTimeout("synth too slow", request=request)
        return nxt

    client = TTSClient(base_url="http://modality-gateway.test", license_token="svc")
    client.set_transport(httpx.MockTransport(handler))

    with pytest.raises(GpuBusyError) as info:
        await asyncio.wait_for(
            client.synthesize(text="hi", attempts=2, busy_deadline_seconds=90.0), timeout=10
        )

    assert _fresh == [30.0, 30.0] and len(sent) == 3
    assert sent[2].extensions["timeout"]["read"] >= 30.0
    assert info.value.retry_after == 30.0


async def test_waiting_for_a_synth_slot_counts_against_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full ``_synth_gate`` cannot hold the preview past its ceiling: the slot
    wait is budgeted too, and running out is busy (503), not a hang."""
    monkeypatch.setattr(tts_mod, "_synth_semaphore", asyncio.Semaphore(0))
    monkeypatch.setattr(tts_mod, "_BUSY_CALL_MARGIN_SECONDS", 0.0)
    seq = _Seq(_tts_ok())

    with pytest.raises(GpuBusyError):
        await asyncio.wait_for(
            _tts(seq).synthesize(text="hi", attempts=2, busy_deadline_seconds=0.2), timeout=5
        )

    assert seq.requests == []


async def test_no_busy_budget_keeps_the_client_timeout(_fresh: list[float]) -> None:
    seq = _Seq(_tts_ok())

    await _tts(seq).synthesize(text="hi")

    assert seq.requests[0].extensions["timeout"]["read"] == tts_mod._REQUEST_TIMEOUT_SECONDS


async def test_preview_busy_budget_exhausted_carries_the_last_retry_after(
    _fresh: list[float],
) -> None:
    seq = _Seq(_busy("50", lane="kokoro"))

    with pytest.raises(GpuBusyError) as info:
        await asyncio.wait_for(
            _tts(seq).synthesize(text="hi", attempts=2, busy_deadline_seconds=90.0), timeout=10
        )

    # 50 fits the 60s busy share; a second 50 does not.
    assert _fresh == [50.0] and len(seq.requests) == 2
    assert info.value.retry_after == 50.0


async def test_attempts_without_a_busy_budget_keeps_the_old_cap(_fresh: list[float]) -> None:
    seq = _Seq(_busy("5", lane="kokoro"), _tts_ok())

    with pytest.raises(GpuBusyError):
        await _tts(seq).synthesize(text="hi", attempts=1)

    assert len(seq.requests) == 1 and _fresh == []


def test_passthrough_lane_is_bounded() -> None:
    """A ``gpu_busy`` lane from upstream is echoed at most 64 chars."""
    from src.api.routes._proxy import map_upstream_error

    resp = httpx.Response(503, json={"error": "gpu_busy", "lane": "x" * 500, "retry_after": 4})
    exc = map_upstream_error(resp, "transcription_worker")

    assert exc.status_code == 503 and exc.headers == {"Retry-After": "4"}
    assert isinstance(exc.detail, dict) and exc.detail["lane"] == "x" * 64
