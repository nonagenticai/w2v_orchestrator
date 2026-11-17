"""Regression tests for TTSClient transport resilience.

mini2 (single 16 GiB box running kokoro) drops connections mid-request when the
orchestrator over-drives it during concurrent narration — surfacing as an
empty-message ``httpx.TransportError``. The synthesize path calls the httpx
client directly (to set ``Accept: application/json``), which bypasses the base
client's transport retry, so ``_post_synthesize_json`` retries transport drops
itself with exponential backoff and a shared concurrency gate. These tests pin
that behaviour: transient drops recover, persistent drops fail loud, and real
errors (4xx/5xx) are NOT retried.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import httpx
import pytest

import src.clients.tts_client as tts_mod
from src.clients.tts_client import TTSClient, TTSSynthesisError


def _ok_response(
    headers: dict[str, str] | None = None, extra: dict[str, Any] | None = None
) -> httpx.Response:
    body = {
        "audio_base64": base64.b64encode(b"WAVDATA").decode(),
        "duration_sec": 1.0,
        "sample_rate": 24000,
        "provider": "kokoro",
        "voice": "af_heart",
        "format": "wav",
    }
    body.update(extra or {})
    return httpx.Response(
        200,
        json=body,
        headers=headers,
        request=httpx.Request("POST", "http://tts.test"),
    )


class _FakeClient:
    """Stand-in for the httpx AsyncClient ``_get_client`` returns."""

    def __init__(
        self, *, fail_times: int, exc: Exception | None = None, final: httpx.Response | None = None
    ) -> None:
        self.calls = 0
        self.last_json: dict[str, Any] | None = None
        self._fail_times = fail_times
        self._exc = exc or httpx.RemoteProtocolError(
            "", request=httpx.Request("POST", "http://tts.test")
        )
        self._final = final

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        self.calls += 1
        self.last_json = kwargs.get("json")
        if self.calls <= self._fail_times:
            raise self._exc
        return self._final if self._final is not None else _ok_response()


def _client_with(monkeypatch: pytest.MonkeyPatch, fake: _FakeClient) -> TTSClient:
    # base_url + license_token bypass settings/discovery; no real socket opens.
    client = TTSClient(base_url="http://tts.test", license_token="t")

    async def _fake_get_client() -> _FakeClient:
        return fake

    monkeypatch.setattr(client, "_get_client", _fake_get_client)
    # Don't actually sleep through backoff in tests.
    monkeypatch.setattr(tts_mod, "_TRANSPORT_RETRY_BASE_SECONDS", 0.0)
    return client


async def test_transient_transport_drop_is_retried_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeClient(fail_times=2)  # default attempts = 4, so 3rd call wins
    client = _client_with(monkeypatch, fake)

    result = await client.synthesize(text="hello", voice="af_heart")

    assert result.audio_bytes == b"WAVDATA"
    assert fake.calls == 3  # two drops + one success


async def test_persistent_transport_drop_raises_after_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tts_mod, "_TRANSPORT_RETRY_ATTEMPTS", 4)
    fake = _FakeClient(fail_times=99)
    client = _client_with(monkeypatch, fake)

    with pytest.raises(TTSSynthesisError, match="after 4 attempts"):
        await client.synthesize(text="hello")

    assert fake.calls == 4  # exhausted exactly the attempt budget


async def test_http_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    # A 500 (e.g. unknown voice) must surface immediately, not loop.
    err = httpx.Response(
        500, text="unknown voice", request=httpx.Request("POST", "http://tts.test")
    )
    fake = _FakeClient(fail_times=0, final=err)
    client = _client_with(monkeypatch, fake)

    with pytest.raises(TTSSynthesisError, match="returned 500"):
        await client.synthesize(text="hello")

    assert fake.calls == 1  # no retry on a real HTTP error


def test_synth_gate_uses_configured_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tts_mod, "_MAX_CONCURRENT_SYNTHESIZE", 2)
    monkeypatch.setattr(tts_mod, "_synth_semaphore", None)

    gate = tts_mod._synth_gate()

    assert gate._value == 2
    assert tts_mod._synth_gate() is gate  # memoized


# --- X-TTS-LUFS: the un-normalized-clip detector --------------------------------
#
# tts_backend masters every clip to -18 LUFS and reports the measured loudness in
# X-TTS-LUFS. It OMITS the header when its own premaster failed open (no ffmpeg on
# the device, ffmpeg error, kill-switch off) — so an ABSENT header means that clip
# is un-normalized and will jump level against its neighbours. That was silent.


async def test_synthesize_captures_lufs_header(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(fail_times=0, final=_ok_response({"X-TTS-LUFS": "-18.2"}))
    client = _client_with(monkeypatch, fake)

    result = await client.synthesize(text="hello", voice="af_heart")

    assert result.integrated_lufs == pytest.approx(-18.2)
    assert result.audio_bytes == b"WAVDATA"  # existing contract unchanged


async def test_synthesize_warns_when_lufs_header_absent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No header ⇒ integrated_lufs is None + a WARNING — never a crash."""
    fake = _FakeClient(fail_times=0, final=_ok_response())
    client = _client_with(monkeypatch, fake)

    with caplog.at_level("WARNING"):
        result = await client.synthesize(text="hello", voice="af_heart")

    assert result.integrated_lufs is None
    assert result.audio_bytes == b"WAVDATA"
    # The header's absence means the level is UNPROVEN — deliberately not the old
    # "NOT loudness-normalized" claim, which was wrong: tts_backend also omits the
    # header when only its post-premaster RE-MEASUREMENT failed, on audio that is
    # in fact normalized. `ensure_tts_loudness` re-masters either way.
    assert "UNPROVEN" in caplog.text


async def test_synthesize_tolerates_unparseable_lufs_header(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = _FakeClient(fail_times=0, final=_ok_response({"X-TTS-LUFS": "-inf dB"}))
    client = _client_with(monkeypatch, fake)

    with caplog.at_level("WARNING"):
        result = await client.synthesize(text="hello")

    assert result.integrated_lufs is None
    assert "unparseable" in caplog.text


# --- SSML pass-through -----------------------------------------------------------
#
# tts_backend parses ``text`` as SSML only when the request carries ``ssml: true``.
# The client adds that key ONLY when the caller opts in, so the plain-text request
# body stays byte-identical to every caller that never passes the flag.


async def test_synthesize_sets_ssml_flag_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeClient(fail_times=0)
    client = _client_with(monkeypatch, fake)

    await client.synthesize(text="<speak>hi</speak>", voice="af_heart", ssml=True)

    assert fake.last_json is not None
    assert fake.last_json["ssml"] is True


async def test_synthesize_omits_ssml_key_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeClient(fail_times=0)
    client = _client_with(monkeypatch, fake)

    await client.synthesize(text="hello", voice="af_heart")

    assert fake.last_json is not None
    assert "ssml" not in fake.last_json  # byte-identical to the pre-ssml body


def test_tts_client_uses_long_request_timeout() -> None:
    # The 30s BaseServiceClient default ReadTimeouts a real long-form chunk
    # (~20-40s synth) under any queueing; TTS must default to a long timeout.
    client = TTSClient(license_token="x")
    assert client.timeout >= 120.0
    # An explicit override still wins.
    assert TTSClient(license_token="x", timeout=45.0).timeout == 45.0


# ══ speech_seconds / pause_seconds — ABSENT must not read as ZERO ════════════
#
# tts_backend reports how a clip splits into speech vs. the fixed silence it
# spliced in, and `narration._fit_speed` subtracts that silence from the fit
# ratio (A/V quality wave 8). The producer declares both with a `0.0` default,
# but a tts_backend deployed BEFORE that change omits them entirely — and the
# two cases must stay distinguishable at this seam:
#
#   absent  → None → "this backend does not report the split; use the authored
#                     `<break>` alone", i.e. exactly the pre-wave-8 arithmetic
#   0.0     → 0.0  → "this backend measured, and there is genuinely no pacing
#                     silence" (kitten)
#
# Collapsing absent into 0.0 would tell the solver the whole clip is
# compressible speech on every backend that cannot answer — reintroducing the
# over-speeding bug under a field name that looks like it fixed it.


async def test_an_older_backend_omitting_the_split_yields_none_not_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeClient(fail_times=0, final=_ok_response())
    client = _client_with(monkeypatch, fake)

    result = await client.synthesize(text="hello", voice="af_heart")

    assert result.speech_seconds is None
    assert result.pause_seconds is None


async def test_a_reported_split_reaches_the_caller_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Including a genuine 0.0, which is a MEASUREMENT and not an absence."""
    fake = _FakeClient(
        fail_times=0,
        final=_ok_response(extra={"speech_seconds": 0.55, "pause_seconds": 0.45}),
    )
    client = _client_with(monkeypatch, fake)
    result = await client.synthesize(text="hello", voice="af_heart")
    assert result.speech_seconds == pytest.approx(0.55)
    assert result.pause_seconds == pytest.approx(0.45)

    fake = _FakeClient(
        fail_times=0,
        final=_ok_response(extra={"speech_seconds": 1.0, "pause_seconds": 0.0}),
    )
    client = _client_with(monkeypatch, fake)
    result = await client.synthesize(text="hello", voice="af_heart")
    assert result.pause_seconds == 0.0
    assert result.pause_seconds is not None


# ══ 429/502/503/504 — "not right now", not "never" ═══════════════════════════
#
# Prod run 01a0a054-14fa-72b9-a9a2-dc5a209522f7 died in website_review_record on
# ``tts /synthesize returned 503: … license_unavailable … retry shortly`` 26s in,
# after ~15 min of paid upstream work, attributed ``unknown``. The client raised
# on the FIRST 503; the stage's one per-item retry landed ~1s later in the same
# outage. These pin: a retryable status is retried with bounded backoff, the
# server's Retry-After is honoured (capped), exhaustion attributes ``external``,
# and a non-retryable 4xx still surfaces on the first call.


def _status_response(status: int, *, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        text=('{"error":{"code":"HTTP_503","message":"license_unavailable — retry shortly"}}'),
        headers=headers,
        request=httpx.Request("POST", "http://tts.test"),
    )


class _SequenceClient:
    """Returns the queued responses in order, then the last one forever."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.calls = 0
        self._responses = responses

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]


def _seq_client(
    monkeypatch: pytest.MonkeyPatch, fake: _SequenceClient
) -> tuple[TTSClient, list[float]]:
    client = TTSClient(base_url="http://tts.test", license_token="t")

    async def _fake_get_client() -> _SequenceClient:
        return fake

    sleeps: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        await asyncio.sleep(0)  # yield, so a runaway retry loop is cancellable

    monkeypatch.setattr(client, "_get_client", _fake_get_client)
    # raising=False so a pre-fix module (no `_sleep`) fails on BEHAVIOUR, not setup.
    monkeypatch.setattr(tts_mod, "_sleep", _fake_sleep, raising=False)
    return client, sleeps


async def test_503_then_200_is_retried_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _SequenceClient([_status_response(503), _status_response(503), _ok_response()])
    client, sleeps = _seq_client(monkeypatch, fake)

    result = await client.synthesize(text="hello", voice="af_heart")

    assert result.audio_bytes == b"WAVDATA"
    assert fake.calls == 3
    # No Retry-After anywhere: the busy contract's jittered exponential backoff
    # (2s then 4s, ±20%), not the 1.5s transport schedule.
    assert len(sleeps) == 2
    assert 1.6 <= sleeps[0] <= 2.4
    assert 3.2 <= sleeps[1] <= 4.8


@pytest.mark.parametrize("status", [429])
async def test_other_transient_statuses_are_retried(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    fake = _SequenceClient([_status_response(status), _ok_response()])
    client, _ = _seq_client(monkeypatch, fake)

    result = await client.synthesize(text="hello")

    assert result.audio_bytes == b"WAVDATA"
    assert fake.calls == 2


async def test_retry_after_seconds_is_honoured_and_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _SequenceClient(
        [
            _status_response(429, headers={"Retry-After": "4"}),
            _status_response(503, headers={"Retry-After": "600"}),
            _ok_response(),
        ]
    )
    client, sleeps = _seq_client(monkeypatch, fake)

    await client.synthesize(text="hello")

    assert sleeps == [4.0, 120.0]  # honoured verbatim, then clamped to the 120s contract max


async def test_persistent_503_exhausts_with_external_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from w2v_common.stages import StageRetriesExhaustedError

    from src.clients.gpu_busy import GpuBusyError, get_gpu_admission_settings
    from src.models.failure_class import EXTERNAL, attribute_exception

    monkeypatch.setenv("GPU_BUSY_DEADLINE_SECONDS", "120")
    get_gpu_admission_settings.cache_clear()
    fake = _SequenceClient([_status_response(503)])
    client, sleeps = _seq_client(monkeypatch, fake)

    try:
        # wait_for: a never-firing deadline must fail this test, not hang the suite.
        with pytest.raises(GpuBusyError, match="busy") as info:
            await asyncio.wait_for(client.synthesize(text="hello"), timeout=10)
    finally:
        get_gpu_admission_settings.cache_clear()

    assert fake.calls == len(sleeps) + 1
    # Bounded by the busy DEADLINE, not unbounded.
    assert 0 < sum(sleeps) <= 120.0
    exc = info.value
    assert attribute_exception(exc) == EXTERNAL
    # And through the stage wrapper that actually reaches runs.failure_class.
    try:
        raise StageRetriesExhaustedError(
            stage_name="website_review_record[0]", attempts=2, last_error=exc
        ) from exc
    except StageRetriesExhaustedError as wrapped:
        assert attribute_exception(wrapped) == EXTERNAL


async def test_400_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _SequenceClient([_status_response(400), _ok_response()])
    client, sleeps = _seq_client(monkeypatch, fake)

    with pytest.raises(TTSSynthesisError, match="returned 400"):
        await client.synthesize(text="hello")

    assert fake.calls == 1
    assert sleeps == []
