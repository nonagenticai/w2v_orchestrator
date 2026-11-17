"""Unit tests for the speech-rate calibration probe (``shared/speech_rate.py``).

No network, no TTS: a fake client returns the ``duration_seconds`` tts_backend would
report, and the module's only job is to turn that into words-per-minute — or to refuse
to, loudly and without ever raising.

Contracts pinned here:

* a probe measures ``CALIBRATION_WORDS / (duration/60)`` off the REAL reported duration;
* the memo probes ONCE per ``(provider, voice)``, including under concurrency;
* every failure shape (raise, non-positive duration, implausible rate) returns ``None``;
* ``resolve_natural_wpm`` returns the caller's ``default`` whenever the probe cannot answer;
* ``resolve_natural_wpm_detail`` reports WHETHER that happened and WHY, because a
  fallback rate and a measured one are the same float and differ by ~1% on a good day.
"""

from __future__ import annotations

import ast
import asyncio
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest

import src.services.stages.shared.speech_rate as speech_rate_module
from src.clients.tts_client import TTSClient
from src.services.runs.run_context import RunContext
from src.services.stages.shared.degradation import SPEECH_RATE_UNUSABLE
from src.services.stages.shared.speech_rate import (
    CALIBRATION_PASSAGE,
    CALIBRATION_SENTENCE_TERMINATORS,
    CALIBRATION_TEXT,
    CALIBRATION_WORDS,
    DEFAULT_SAMPLE_LINES,
    IMPLAUSIBLE_RATE,
    MAX_PLAUSIBLE_WPM,
    MAX_SAMPLE_LINES,
    MIN_PLAUSIBLE_WPM,
    RESOLUTION_ERROR,
    SAMPLE_LINES_TOO_SHORT,
    SAMPLE_NO_LINES,
    SAMPLE_TOO_FEW_MEASURED,
    SCRIPT_SAMPLE,
    ScriptSpeechRate,
    SpeechRate,
    clear_measured_wpm_cache,
    measure_natural_wpm,
    resolve_natural_wpm,
    resolve_natural_wpm_detail,
    sample_size,
    script_sample_degradation,
    select_sample_indices,
    speech_rate_fallback_degradation,
)
from src.services.stages.shared.time_budget import DEFAULT_WPM_NATURAL
from src.services.tts_storage import deterministic_clip_id

# The calibration passage spoken at 94.0 wpm — the rate measured off the live kokoro
# voice. Derived from the real word count so it tracks any reword of the passage.
_KOKORO_SECONDS = CALIBRATION_WORDS * 60.0 / 94.0


class _FakeTTS:
    """Records every synthesize call and answers with a fixed duration (or raises)."""

    def __init__(self, *, duration: float = _KOKORO_SECONDS, error: Exception | None = None):
        self.duration = duration
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def synthesize(self, **kwargs: Any) -> Any:
        # A real synthesis suspends. Without a suspension point the first caller runs
        # to completion inside `gather`'s first step and the concurrency test can never
        # exercise the in-lock re-check it exists to prove.
        await asyncio.sleep(0)
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return type("_Synth", (), {"duration_seconds": self.duration})()


def _client(fake: object) -> TTSClient:
    """Any object with an async ``synthesize`` stands in for the real client.

    Typed as ``object`` rather than ``_FakeTTS``: the module's contract is
    duck-typed (it calls exactly one method), and there is more than one fake —
    ``_PerVoiceTTS`` answers each voice with a different rate, which is the only
    way to exercise the cast-weighted budget at all.
    """
    return cast("TTSClient", fake)


@pytest.fixture(autouse=True)
def _passage_rate_is_the_prior(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the passage->narration ratio at 1.0: these tests prove the measured-rate
    PLUMBING with arithmetic written at the raw passage rate. The ratio itself is
    pinned by ``tests/services/stages/shared/test_speech_rate_narration_prior.py``."""
    from src.services.stages.shared import speech_rate

    monkeypatch.setattr(speech_rate, "PASSAGE_TO_NARRATION_RATIO", 1.0)


@pytest.fixture(autouse=True)
def _clean_cache() -> Any:
    clear_measured_wpm_cache()
    yield
    clear_measured_wpm_cache()


def _ctx(voice: dict[str, Any] | None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={} if voice is None else {"voice": voice},
        prior_run_outputs={},
    )


# ── the measurement itself ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_probe_measures_wpm_from_the_reported_duration() -> None:
    fake = _FakeTTS(duration=_KOKORO_SECONDS)

    wpm = await measure_natural_wpm(_client(fake), voice="af_heart")

    assert wpm == pytest.approx(94.0)
    # The probe must ask for the calibration sentence at UNMODIFIED speed: `speed`
    # scales duration linearly backend-side, so any other value measures the
    # multiplier rather than the voice.
    assert fake.calls == [
        {
            "text": CALIBRATION_TEXT,
            "voice": "af_heart",
            "provider": None,
            "speed": 1.0,
            "attempts": 1,
        }
    ]


def test_calibration_word_count_is_derived_from_the_text() -> None:
    # The count may never be hand-maintained beside the passage — the two would drift,
    # and a stale denominator biases every wpm this module ever reports. This is the
    # exact bug a second hard-coded constant would reintroduce.
    assert len(CALIBRATION_TEXT.split()) == CALIBRATION_WORDS


def _calibration_sentences() -> list[str]:
    """The passage split on sentence terminators, empties dropped."""
    pattern = f"[{re.escape(CALIBRATION_SENTENCE_TERMINATORS)}]+"
    return [s for s in (part.strip() for part in re.split(pattern, CALIBRATION_TEXT)) if s]


def test_calibration_passage_is_multi_sentence() -> None:
    """A one-sentence probe pays one sentence-terminal silence over its whole word
    count and so over-reports the rate of narration that stops every ~13 words. The
    probe must be structurally like what it predicts, so assert the PROPERTY — a
    future reword must not be able to silently restore a single-sentence probe.
    """
    assert len(_calibration_sentences()) >= 3


def test_calibration_words_per_sentence_matches_real_narration() -> None:
    """Measured off delivered narration: ~359 words over ~25-30 sentences, i.e.
    ~12-14 words per sentence. That ratio — not the passage's total length — is what
    fixes the boundary cost per word the probe pays.
    """
    sentences = _calibration_sentences()
    words_per_sentence = CALIBRATION_WORDS / len(sentences)
    assert 12.0 <= words_per_sentence <= 14.0


def test_calibration_passage_has_no_numerals_that_a_normalizer_would_expand() -> None:
    """A digit is spoken as one-or-more words that ``str.split`` counts as one token,
    which silently biases the denominator.
    """
    assert not any(ch.isdigit() for ch in CALIBRATION_TEXT)


# ── the memo: measured once per process, per voice ───────────────────────────


@pytest.mark.asyncio
async def test_cache_probes_once_for_two_calls() -> None:
    fake = _FakeTTS()

    first = await measure_natural_wpm(_client(fake), voice="af_heart")
    second = await measure_natural_wpm(_client(fake), voice="af_heart")

    assert first == second == pytest.approx(94.0)
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_cache_is_keyed_by_voice_and_provider() -> None:
    fake = _FakeTTS()

    await measure_natural_wpm(_client(fake), voice="af_heart")
    await measure_natural_wpm(_client(fake), voice="am_onyx")
    await measure_natural_wpm(_client(fake), voice="af_heart", provider="elevenlabs")

    assert len(fake.calls) == 3


@pytest.mark.asyncio
async def test_concurrent_callers_share_one_probe() -> None:
    """N parallel runs on one voice must not each pay for a synthesis."""
    fake = _FakeTTS()

    results = await asyncio.gather(
        *(measure_natural_wpm(_client(fake), voice="af_heart") for _ in range(5))
    )

    assert all(r == pytest.approx(94.0) for r in results)
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_a_failed_probe_is_suppressed_not_retried_every_call() -> None:
    fake = _FakeTTS(error=RuntimeError("tts down"))

    assert await measure_natural_wpm(_client(fake), voice="af_heart") is None
    assert await measure_natural_wpm(_client(fake), voice="af_heart") is None

    assert len(fake.calls) == 1


# ── every failure shape returns None, and none of them raise ─────────────────


@pytest.mark.asyncio
async def test_client_error_returns_none() -> None:
    fake = _FakeTTS(error=RuntimeError("tts_backend 500"))

    assert await measure_natural_wpm(_client(fake), voice="af_heart") is None


@pytest.mark.asyncio
async def test_timeout_returns_none() -> None:
    fake = _FakeTTS(error=TimeoutError())

    assert await measure_natural_wpm(_client(fake), voice="af_heart") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [0.0])
async def test_non_positive_duration_returns_none(duration: float) -> None:
    fake = _FakeTTS(duration=duration)

    assert await measure_natural_wpm(_client(fake), voice="af_heart") is None


@pytest.mark.asyncio
async def test_implausibly_fast_result_is_rejected() -> None:
    # 17 words in 1s ⇒ 1020 wpm: not a fast voice, a broken measurement.
    fake = _FakeTTS(duration=1.0)

    assert await measure_natural_wpm(_client(fake), voice="af_heart") is None


@pytest.mark.asyncio
async def test_implausibly_slow_result_is_rejected() -> None:
    # Just below the floor: 17 words at (MIN - 1) wpm.
    fake = _FakeTTS(duration=CALIBRATION_WORDS * 60.0 / (MIN_PLAUSIBLE_WPM - 1.0))

    assert await measure_natural_wpm(_client(fake), voice="af_heart") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("wpm", [MIN_PLAUSIBLE_WPM])
async def test_rates_inside_the_plausible_band_are_accepted(wpm: float) -> None:
    fake = _FakeTTS(duration=CALIBRATION_WORDS * 60.0 / wpm)

    assert await measure_natural_wpm(_client(fake), voice="af_heart") == pytest.approx(wpm)


# ── resolve_natural_wpm: reads the run's voice, never fails the run ──────────


@pytest.mark.asyncio
async def test_resolve_reads_the_runs_voice_id() -> None:
    fake = _FakeTTS()
    ctx = _ctx({"voice_id": "af_heart"})

    wpm = await resolve_natural_wpm(ctx, client=_client(fake), default=150.0)

    assert wpm == pytest.approx(94.0)
    assert fake.calls[0]["voice"] == "af_heart"


@pytest.mark.asyncio
async def test_resolve_with_no_voice_block_probes_the_backend_default_voice() -> None:
    fake = _FakeTTS()

    wpm = await resolve_natural_wpm(_ctx(None), client=_client(fake), default=150.0)

    assert wpm == pytest.approx(94.0)
    assert fake.calls[0]["voice"] is None


@pytest.mark.asyncio
async def test_resolve_falls_back_to_the_default_when_the_probe_fails() -> None:
    fake = _FakeTTS(error=RuntimeError("tts down"))

    wpm = await resolve_natural_wpm(
        _ctx({"voice_id": "af_heart"}), client=_client(fake), default=150.0
    )

    assert wpm == 150.0


@pytest.mark.asyncio
async def test_resolve_falls_back_when_the_measurement_is_implausible() -> None:
    fake = _FakeTTS(duration=0.01)

    wpm = await resolve_natural_wpm(
        _ctx({"voice_id": "af_heart"}), client=_client(fake), default=150.0
    )

    assert wpm == 150.0


@pytest.mark.asyncio
async def test_an_explicit_voice_lands_under_its_own_cache_key() -> None:
    """Two voices on one run must be two measurements, not one memo shared by both."""
    fake = _FakeTTS()
    ctx = _ctx({"voice_id": "af_heart"})

    await resolve_natural_wpm(ctx, client=_client(fake), default=150.0, voice="am_michael")
    await resolve_natural_wpm(ctx, client=_client(fake), default=150.0, voice="af_heart")
    await resolve_natural_wpm(ctx, client=_client(fake), default=150.0, voice="am_michael")

    assert [c["voice"] for c in fake.calls] == ["am_michael", "af_heart"]


@pytest.mark.asyncio
async def test_an_explicit_none_voice_probes_the_backend_default_not_the_config_voice() -> None:
    """``None`` is a meaningful voice argument, not "unspecified" — a caller that wants
    the backend's own default measured must be able to say so over a configured voice.
    """
    fake = _FakeTTS()
    ctx = _ctx({"voice_id": "af_heart"})

    await resolve_natural_wpm(ctx, client=_client(fake), default=150.0, voice=None)

    assert fake.calls[0]["voice"] is None


@pytest.mark.asyncio
async def test_omitting_the_voice_preserves_the_config_voice_behaviour() -> None:
    fake = _FakeTTS()
    ctx = _ctx({"voice_id": "af_heart"})

    wpm = await resolve_natural_wpm(ctx, client=_client(fake), default=150.0)

    assert wpm == pytest.approx(94.0)
    assert fake.calls[0]["voice"] == "af_heart"


@pytest.mark.asyncio
async def test_an_explicit_voice_still_falls_back_to_the_default_when_the_probe_fails() -> None:
    fake = _FakeTTS(error=RuntimeError("tts down"))

    wpm = await resolve_natural_wpm(
        _ctx({"voice_id": "af_heart"}), client=_client(fake), default=150.0, voice="am_michael"
    )

    assert wpm == 150.0


@pytest.mark.asyncio
async def test_a_probe_timeout_yields_the_hardcoded_default_wpm() -> None:
    """The graceful-fallback contract, against the REAL constant callers pass: a probe
    that cannot answer leaves the caller byte-identical to the pre-measurement
    behaviour rather than failing a paid run.
    """
    fake = _FakeTTS(error=TimeoutError())

    wpm = await resolve_natural_wpm(
        _ctx({"voice_id": "af_heart"}), client=_client(fake), default=DEFAULT_WPM_NATURAL
    )

    assert wpm == DEFAULT_WPM_NATURAL == 150.0


def test_plausible_band_brackets_real_speech() -> None:
    assert MIN_PLAUSIBLE_WPM < 94.0 < MAX_PLAUSIBLE_WPM


# ── the fallback is OBSERVABLE: measured vs assumed, and why ─────────────────
#
# tts.test is a launchd unit on mini2, outside the cluster and outside every
# dashboard. The measured rate today is 151.6 wpm and the constant is 150.0, so a
# fallback is nearly right on a good day and badly wrong on a different voice — and
# until these contracts existed nothing in the run record said which day you had.


class _ConnectError(Exception):
    """Stands in for ``httpx.ConnectError``: a transport failure that is NOT an
    ``OSError`` and NOT a ``TimeoutError``, so it can only be told apart
    structurally. Classifying it as a generic probe error would point an operator
    at the backend when the fault is the route to it.
    """


class _ConnectTimeout(Exception):  # noqa: N818 - the NAME is the thing under test
    """``httpx.ConnectTimeout``: a timeout that subclasses neither builtin."""


@pytest.mark.asyncio
async def test_a_successful_probe_reports_itself_as_measured() -> None:
    fake = _FakeTTS(duration=_KOKORO_SECONDS)

    rate = await resolve_natural_wpm_detail(
        _ctx({"voice_id": "af_heart"}), client=_client(fake), default=DEFAULT_WPM_NATURAL
    )

    assert rate.measured is True
    assert rate.assumed is False
    assert rate.reason is None
    assert rate.wpm == pytest.approx(94.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_reason"),
    [(TimeoutError(), "probe_timeout")],
)
async def test_a_failing_probe_reports_a_fallback_with_its_reason(
    error: Exception, expected_reason: str
) -> None:
    fake = _FakeTTS(error=error)

    rate = await resolve_natural_wpm_detail(
        _ctx({"voice_id": "af_heart"}), client=_client(fake), default=DEFAULT_WPM_NATURAL
    )

    assert rate.assumed is True
    assert rate.reason == expected_reason
    # Constraint 1: the rate is still usable. Availability of a calibration courtesy
    # must never fail a paid run, so this is a real float, never None and never NaN.
    assert rate.wpm == DEFAULT_WPM_NATURAL
    assert math.isfinite(rate.wpm)


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [0.0])
async def test_an_unusable_duration_reports_a_fallback_distinct_from_a_bad_rate(
    duration: float,
) -> None:
    """The audio path answered and its NUMBER is broken — a different fault from a
    dead backend and from a truncated clip, and it must not be reported as either.
    NaN in particular fails ``duration <= 0.0``, so without an explicit finiteness
    check it falls through to the band test and is misnamed ``implausible_rate``.
    """
    fake = _FakeTTS(duration=duration)

    rate = await resolve_natural_wpm_detail(
        _ctx({"voice_id": "af_heart"}), client=_client(fake), default=DEFAULT_WPM_NATURAL
    )

    assert rate.assumed is True
    assert rate.reason == "unusable_duration"
    assert rate.wpm == DEFAULT_WPM_NATURAL
    assert math.isfinite(rate.wpm)


@pytest.mark.asyncio
async def test_an_implausible_rate_reports_its_own_reason() -> None:
    # A finite duration that implies ~3900 wpm: truncated audio or a stub duration.
    fake = _FakeTTS(duration=1.0)

    rate = await resolve_natural_wpm_detail(
        _ctx({"voice_id": "af_heart"}), client=_client(fake), default=DEFAULT_WPM_NATURAL
    )

    assert rate.assumed is True
    assert rate.reason == "implausible_rate"


@pytest.mark.asyncio
async def test_a_failure_outside_the_probe_is_named_as_such(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing was ever asked of TTS — pointing an operator at mini2 would be wrong."""
    import src.services.stages.shared.speech_rate as speech_rate

    def _explode(**_: Any) -> Any:
        raise RuntimeError("no client")

    monkeypatch.setattr(speech_rate, "TTSClient", _explode)

    rate = await resolve_natural_wpm_detail(
        _ctx({"voice_id": "af_heart"}), default=DEFAULT_WPM_NATURAL
    )

    assert rate.assumed is True
    assert rate.reason == "resolution_error"
    assert rate.wpm == DEFAULT_WPM_NATURAL


@pytest.mark.asyncio
async def test_reason_is_none_if_and_only_if_measured() -> None:
    """The invariant that keeps ``reason`` unmisreadable: a measured rate never
    carries one, an assumed rate always does.
    """
    measured = await resolve_natural_wpm_detail(
        _ctx({"voice_id": "af_heart"}), client=_client(_FakeTTS()), default=150.0
    )
    assumed = await resolve_natural_wpm_detail(
        _ctx({"voice_id": "am_onyx"}),
        client=_client(_FakeTTS(error=TimeoutError())),
        default=150.0,
    )

    assert (measured.reason is None) is measured.measured
    assert (assumed.reason is None) is assumed.measured


# ── the memo must not cache a FALLBACK as if it were a measurement ───────────


@pytest.mark.asyncio
async def test_a_fallback_is_never_memoised_as_a_measurement() -> None:
    """Caching a failure as a measurement would poison the rate for the life of the
    process: every later caller would be told 150.0 was MEASURED off this voice. The
    suppression window is a separate, deliberate thing — it stops a dead backend
    being re-probed once per run — and it must keep reporting ``assumed``.
    """
    fake = _FakeTTS(error=TimeoutError())
    ctx = _ctx({"voice_id": "af_heart"})

    first = await resolve_natural_wpm_detail(ctx, client=_client(fake), default=150.0)
    second = await resolve_natural_wpm_detail(ctx, client=_client(fake), default=150.0)

    # Suppressed: one probe, not two.
    assert len(fake.calls) == 1
    # ...and STILL honest on the suppressed call, with the ORIGINATING reason —
    # not "suppressed", which names the memo rather than the fault.
    assert [r.measured for r in (first, second)] == [False, False]
    assert [r.reason for r in (first, second)] == ["probe_timeout", "probe_timeout"]
    assert second.wpm == 150.0


@pytest.mark.asyncio
async def test_a_measurement_after_a_failure_is_reported_as_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the suppression window lapses and a probe succeeds, the recorded fault
    must be dropped: a stale reason beside a live measurement would report a
    genuinely measured rate as assumed, which is the same silent-lie defect pointed
    the other way.
    """
    import src.services.stages.shared.speech_rate as speech_rate

    monkeypatch.setattr(speech_rate, "FAILURE_RETRY_AFTER_SECONDS", 0.0)
    ctx = _ctx({"voice_id": "af_heart"})

    failed = await resolve_natural_wpm_detail(
        ctx, client=_client(_FakeTTS(error=TimeoutError())), default=150.0
    )
    recovered = await resolve_natural_wpm_detail(
        ctx, client=_client(_FakeTTS(duration=_KOKORO_SECONDS)), default=150.0
    )

    assert failed.assumed is True
    assert recovered.measured is True
    assert recovered.reason is None
    assert recovered.wpm == pytest.approx(94.0)
    # White-box on purpose: the recorded fault is only ever READ when there is no
    # measurement, so leaving it behind changes nothing observable TODAY and is a
    # trap for the next reader (and an unbounded dict on a multi-day pod). Pin that
    # a measurement retires its key's fault record rather than shadowing it.
    assert ("deployment", None, "af_heart") not in speech_rate._failed_reason
    assert ("deployment", None, "af_heart") not in speech_rate._failed_until


@pytest.mark.asyncio
async def test_a_stale_success_is_re_probed_not_trusted_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SUCCESS must expire, exactly as a FAILURE does.

    Before ``SUCCESS_TTL_SECONDS`` existed, ``_measured`` had no TTL at all: a
    voice that measured plausible-but-wrong once (a degraded box, a stub duration
    that happened to clear the band) stayed wrong for the life of the pod, while a
    FAILURE self-healed in ``FAILURE_RETRY_AFTER_SECONDS``. This pins that a second
    probe fires once the TTL has passed, and that a fresh (different) answer is
    what callers see afterward — a cache that merely stopped returning a value
    would not prove the SAME stale number wasn't just being handed back.
    """
    import src.services.stages.shared.speech_rate as speech_rate

    monkeypatch.setattr(speech_rate, "SUCCESS_TTL_SECONDS", 0.0)
    ctx = _ctx({"voice_id": "af_heart"})

    first = await resolve_natural_wpm_detail(
        ctx, client=_client(_FakeTTS(duration=_KOKORO_SECONDS)), default=150.0
    )
    # A different duration on the second probe: if the stale entry were still being
    # served, this measurement would never be seen.
    second_duration = CALIBRATION_WORDS * 60.0 / 120.0
    second = await resolve_natural_wpm_detail(
        ctx, client=_client(_FakeTTS(duration=second_duration)), default=150.0
    )

    assert first.wpm == pytest.approx(94.0)
    assert first.measured is True
    assert second.wpm == pytest.approx(120.0)
    assert second.measured is True


@pytest.mark.asyncio
async def test_an_unrecorded_refusal_is_named_not_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sibling stage tests stub ``measure_natural_wpm`` to return ``None`` outright,
    so no reason is ever recorded. That must read as an explicitly unknown cause
    rather than borrowing some other voice's fault.
    """
    import src.services.stages.shared.speech_rate as speech_rate

    async def _no_answer(*_: Any, **__: Any) -> float | None:
        return None

    monkeypatch.setattr(speech_rate, "measure_natural_wpm", _no_answer)

    rate = await resolve_natural_wpm_detail(
        _ctx({"voice_id": "af_heart"}), client=_client(_FakeTTS()), default=150.0
    )

    assert rate.assumed is True
    assert rate.reason == "probe_unavailable"


# ── the float-returning entry point is unchanged ─────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("fake", [_FakeTTS()])
async def test_resolve_returns_exactly_the_detail_rate(fake: _FakeTTS) -> None:
    """The six existing call sites keep a plain float. If this ever diverges from
    ``detail.wpm`` there are two definitions of the run's speech rate.
    """
    ctx = _ctx({"voice_id": "af_heart"})

    detail = await resolve_natural_wpm_detail(ctx, client=_client(fake), default=150.0)
    plain = await resolve_natural_wpm(ctx, client=_client(fake), default=150.0)

    assert plain == detail.wpm
    assert isinstance(plain, float)


# ── the degradation entry a caller can record ────────────────────────────────


def test_fallback_degradation_reuses_the_existing_speech_rate_code() -> None:
    entry = speech_rate_fallback_degradation(
        stage="lesson_script",
        rate=SpeechRate(wpm=150.0, measured=False, reason="probe_timeout"),
        voice="af_heart",
    )

    assert entry["code"] == SPEECH_RATE_UNUSABLE
    # The run-specific facts live in `detail`, never in the queryable `code`.
    assert "probe_timeout" in entry["detail"]
    assert "af_heart" in entry["detail"]
    assert "150.0 wpm" in entry["detail"]


def test_fallback_degradation_refuses_a_measured_rate() -> None:
    """A channel that also carries successes is one people learn to ignore."""
    with pytest.raises(ValueError, match="MEASURED"):
        speech_rate_fallback_degradation(
            stage="lesson_script", rate=SpeechRate(wpm=94.0, measured=True)
        )


# ── the `voice=` override: documented, kept, and NOT wired ───────────────────


def _voice_passing_call_sites() -> list[str]:
    """Every call under ``src/`` that budgets against a NAMED voice.

    Two shapes count: a ``resolve_natural_wpm*`` call passing ``voice=``, and any
    call to ``resolve_cast_wpm`` (whose whole purpose is per-voice budgeting — it
    takes ``segments`` of ``(text, voice)`` and passes each one down as ``voice=``).
    Counting only the first shape would have reported "not wired" the day
    ``lesson/script.py`` started budgeting its cast correctly, because the
    ``voice=`` keyword then lives inside ``resolve_cast_wpm`` — i.e. inside the
    excluded module — and the guard would have gone on describing a closed gap as
    open.

    Parsed rather than grepped: a substring search for ``voice`` matches the
    ``voice_id`` reads and the ``voice_for_speaker`` casting helper, so it would say
    "wired" about code that does the opposite.

    ``speech_rate.py`` itself is excluded: ``resolve_natural_wpm`` forwards its own
    ``voice=`` to ``resolve_natural_wpm_detail``, and ``resolve_cast_wpm`` forwards
    one per segment — that is the plumbing of the seam, not a stage using it.
    """
    src_root = Path(speech_rate_module.__file__).parents[3]
    assert src_root.name == "src"
    sites: list[str] = []
    own_module = Path(speech_rate_module.__file__).resolve()
    for path in sorted(src_root.rglob("*.py")):
        if path.resolve() == own_module:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name == "resolve_cast_wpm":
                sites.append(f"{path.relative_to(src_root)}:{node.lineno}")
                continue
            if name not in {"resolve_natural_wpm", "resolve_natural_wpm_detail"}:
                continue
            if any(kw.arg == "voice" for kw in node.keywords):
                sites.append(f"{path.relative_to(src_root)}:{node.lineno}")
    return sites


def test_the_voice_override_is_documented_as_the_known_gap_it_is() -> None:
    """The docstring must state the TRUTH about ``voice=``, in either direction.

    The parameter's docstring used to instruct cast-mode stages to "pass it here",
    which reads as a description of how the code works. It is not: NO caller passes
    it, so ``lesson`` and ``demo_video`` — which cast paragraphs per speaker through
    ``voice_for_speaker`` — budget against ``config_snapshot["voice"]["voice_id"]``
    while a different voice speaks.

    This test is bidirectional on purpose. While nothing passes ``voice=``, the
    docstring must name the gap. The day someone wires it, this test goes red and
    forces the docstring to stop calling it a gap — so the prose cannot drift away
    from the code in either direction.
    """
    doc = resolve_natural_wpm.__doc__ or ""
    wired = _voice_passing_call_sites()

    if wired:
        assert "KNOWN GAP" not in doc, (
            f"`voice=` is now passed at {wired} — the docstring still calls it an unwired gap."
        )
        # ...and "not a gap any more" is not the whole truth either while OTHER
        # cast-mode stages still budget blind. Every wired module must be NAMED, so
        # a reader can tell which budgets are trustworthy from the docstring alone —
        # otherwise the prose drifts by going vague instead of by going stale.
        for site in wired:
            module = site.rsplit(":", 1)[0]
            assert module in doc, (
                f"{module} budgets against named voices but the docstring never says so"
            )
    else:
        assert "KNOWN GAP" in doc, (
            "no caller passes `voice=`, so cast-mode stages budget against the "
            "configured voice while another one speaks; the docstring must say so."
        )
        assert "voice_for_speaker" in doc


# ── resolve_cast_wpm: budget against the voices that ACTUALLY speak ──────────


class _PerVoiceTTS:
    """Answers each voice with the duration that voice's own rate implies.

    A single fixed duration cannot exercise this function at all: the whole defect
    it fixes is that two voices with DIFFERENT rates were budgeted at one.
    """

    def __init__(self, wpm_by_voice: dict[str | None, float]):
        self.wpm_by_voice = wpm_by_voice
        self.calls: list[str | None] = []

    async def synthesize(self, **kwargs: Any) -> Any:
        await asyncio.sleep(0)
        voice = kwargs.get("voice")
        self.calls.append(voice)
        wpm = self.wpm_by_voice.get(voice)
        if wpm is None:
            raise ConnectionError(f"no such voice: {voice}")
        return type("_Synth", (), {"duration_seconds": CALIBRATION_WORDS * 60.0 / wpm})()


@pytest.mark.asyncio
async def test_cast_rate_is_the_word_weighted_sum_not_one_global_rate() -> None:
    """Σ(words_i / wpm_i) * 60 — asserted with real numbers, not a shape.

    100 words at 200 wpm is 30.0s; 50 words at 100 wpm is another 30.0s; the script
    takes 60.0s. Budgeting all 150 words at the FIRST voice's 200 wpm says 45.0s —
    a 25% under-estimate, which is the exact shape of the live lesson defect (a
    +23.07% overrun that estimated at +7.6% and was therefore never trimmed).
    """
    fake = _PerVoiceTTS({"fast": 200.0, "slow": 100.0})

    rate = await speech_rate_module.resolve_cast_wpm(
        _ctx({"provider": None}),
        segments=[("w " * 100, "fast"), ("w " * 50, "slow")],
        default=DEFAULT_WPM_NATURAL,
        client=_client(fake),
    )

    assert rate.weighted is True
    assert rate.seconds == pytest.approx(60.0)
    # The blend expressed back as one rate: 150 words in 60.0s is 150.0 wpm — and
    # it re-derives the SAME estimate through the existing single-rate machinery.
    assert rate.wpm == pytest.approx(150.0)
    assert 150 / rate.wpm * 60.0 == pytest.approx(rate.seconds)
    # Strictly slower than budgeting everything at the faster voice.
    assert rate.wpm < 200.0
    # One probe per DISTINCT voice, not one per segment.
    assert sorted(str(c) for c in fake.calls) == ["fast", "slow"]
    assert all(r.measured for r in rate.per_voice.values())


@pytest.mark.asyncio
async def test_an_unprobeable_voice_is_reported_never_silently_substituted() -> None:
    """An unmeasurable voice is UNMEASURED — it contributes at the caller's default.

    The fallback is the caller's previous behaviour (byte-identical), and it is
    stated: the voice's `SpeechRate` carries `assumed` and the operational reason,
    which is what the caller turns into a degradation entry.
    """
    fake = _PerVoiceTTS({"fast": 200.0})  # "ghost" raises ConnectionError

    rate = await speech_rate_module.resolve_cast_wpm(
        _ctx(None),
        segments=[("w " * 100, "fast"), ("w " * 100, "ghost")],
        default=120.0,
        client=_client(fake),
    )

    assert rate.per_voice["fast"].measured is True
    ghost = rate.per_voice["ghost"]
    assert ghost.assumed is True
    assert ghost.wpm == 120.0
    assert ghost.reason == "probe_unreachable"
    # 100/200*60 + 100/120*60 = 30.0 + 50.0
    assert rate.seconds == pytest.approx(80.0)


@pytest.mark.asyncio
async def test_a_single_voice_cast_measures_exactly_that_voice() -> None:
    fake = _PerVoiceTTS({"solo": 96.0})

    rate = await speech_rate_module.resolve_cast_wpm(
        _ctx(None),
        segments=[("w " * 48, "solo"), ("w " * 48, "solo")],
        default=DEFAULT_WPM_NATURAL,
        client=_client(fake),
    )

    assert rate.wpm == pytest.approx(96.0)
    assert rate.seconds == pytest.approx(60.0)  # 96 words at 96 wpm
    assert fake.calls == ["solo"]


@pytest.mark.asyncio
async def test_a_wordless_script_returns_the_default_unweighted() -> None:
    """No words is no evidence — and it must not divide by zero on a paid run."""
    fake = _PerVoiceTTS({"solo": 96.0})

    rate = await speech_rate_module.resolve_cast_wpm(
        _ctx(None),
        segments=[("   ", "solo")],
        default=DEFAULT_WPM_NATURAL,
        client=_client(fake),
    )

    assert rate.weighted is False
    assert rate.wpm == DEFAULT_WPM_NATURAL
    assert fake.calls == [], "nothing to budget, so nothing is probed"


@pytest.mark.asyncio
async def test_a_non_positive_default_cannot_divide_by_zero() -> None:
    """A broken caller default returns unweighted rather than raising mid-run."""
    fake = _PerVoiceTTS({})  # every probe fails, so every voice takes the default

    rate = await speech_rate_module.resolve_cast_wpm(
        _ctx(None),
        segments=[("w " * 10, "ghost")],
        default=0.0,
        client=_client(fake),
    )

    assert rate.weighted is False
    assert rate.wpm == 0.0


# ── PHASE B: the rate of THIS SCRIPT, not of a fixed passage ─────────────────
#
# The defect these pin, measured on the live backend from inside the orchestrator pod
# (durations byte-identical to the first clips of live app_review run
# 01a091bc-eae4-724d-b278-7ee323344bf4, so this reproduces production exactly):
#
#     CALIBRATION_TEXT :  65w  22.25s  ->  175.2 wpm
#     real prod line   :  17w   8.37s  ->  121.9 wpm
#     real prod line   :  12w   6.62s  ->  108.8 wpm
#     real prod line   :  18w   8.62s  ->  125.3 wpm
#     PRODUCTION       :  47w  23.61s  ->  119.4 wpm
#
# The passage synthesizes ~47% faster than the narration it is used to budget. No
# fixed passage can close that, and neither can a fixed per-utterance constant (the
# measured one is NEGATIVE: -0.170s). The only thing that predicts this script is
# this script.

#: The three real production lines, as (word count, real seconds). The texts below
#: carry the measured WORD COUNTS; their wording is irrelevant to a words-per-minute
#: and the real ones are customer content.
LIVE_LINES: dict[str, float] = {
    " ".join(f"alpha{i}" for i in range(17)): 8.37,
    " ".join(f"bravo{i}" for i in range(12)): 6.62,
    " ".join(f"delta{i}" for i in range(18)): 8.62,
}
#: 47 words in 23.61s. What the script really delivers at.
LIVE_CONTENT_WPM = 47 / (23.61 / 60.0)
#: What CALIBRATION_TEXT reports for the same voice on the same box.
PASSAGE_WPM = 175.2


class _ScriptTTS:
    """Answers each TEXT with its own duration — the shape a real backend has.

    ``_FakeTTS`` answers one duration whatever it is asked, which is exactly the
    assumption Phase B exists to delete. Anything not in ``durations`` raises, so a
    test can make specific lines fail without a second fake.
    """

    def __init__(self, durations: dict[str, float]) -> None:
        # COPIED, not aliased: a test that adds a line to its own fake must not be
        # able to edit the module-level LIVE_LINES out from under every later test.
        self.durations = dict(durations)
        self.calls: list[dict[str, Any]] = []

    async def synthesize(self, **kwargs: Any) -> Any:
        await asyncio.sleep(0)
        self.calls.append(kwargs)
        text = str(kwargs["text"])
        if text not in self.durations:
            raise RuntimeError(f"no synthesis for {text!r}")
        return type("_Synth", (), {"duration_seconds": self.durations[text]})()

    @property
    def texts(self) -> list[str]:
        return [str(call["text"]) for call in self.calls]


@pytest.fixture
def stored() -> Any:
    """Every ``synthesize_and_store_clip`` the sampler made, as recorded kwargs.

    Patched in by :func:`_patch_store`, which each test that persists calls; the
    fixture only owns the list so the assertions read off one object.
    """
    return []


def _patch_store(monkeypatch: pytest.MonkeyPatch, stored: list[dict[str, Any]]) -> None:
    """Stand in for ``primitives.narration.synthesize_and_store_clip``.

    The real one needs MinIO. What matters to Phase B is that it is called with the
    narrate stage's name and the narrate speed (that is what makes the clip id
    collide with narrate's own, and therefore what makes the sample free), and that
    its ``seconds`` is what gets measured — so the fake records the kwargs and
    forwards to the injected client for the duration.
    """

    async def _fake(client: Any, **kwargs: Any) -> Any:
        stored.append(dict(kwargs))
        result = await client.synthesize(text=kwargs["text"], voice=kwargs.get("voice"))
        return type(
            "_Clip",
            (),
            {
                "clip_id": "clip",
                "artifact_url": "url",
                "seconds": result.duration_seconds,
                "speed": kwargs.get("speed"),
                "words": len(str(kwargs["text"]).split()),
            },
        )()

    monkeypatch.setattr(speech_rate_module, "synthesize_and_store_clip", _fake)


def _prior(wpm: float = PASSAGE_WPM) -> SpeechRate:
    """A LANDED Phase-A measurement — the thing Phase B is correcting."""
    return SpeechRate(wpm=wpm, measured=True)


# ── the measurement ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_sample_measures_this_script_not_the_calibration_passage() -> None:
    """THE defect. The same voice, the same box, a 47% different answer."""
    fake = _ScriptTTS(LIVE_LINES)

    sample = await speech_rate_module.resolve_script_sample_wpm(
        _ctx({"voice_id": "af_heart"}),
        texts=list(LIVE_LINES),
        prior=_prior(),
        client=_client(fake),
        sample_lines=3,
    )

    assert sample.rate.wpm == pytest.approx(LIVE_CONTENT_WPM)
    assert sample.rate.wpm == pytest.approx(119.44, abs=0.01)
    assert sample.rate.measured is True
    assert sample.rate.basis == SCRIPT_SAMPLE
    assert sample.reason is None
    assert sample.sampled is True
    # The prior is RETAINED, not overwritten: reporting both numbers is the whole
    # diagnostic, and a caller that only ever saw the winner could not state the gap.
    assert sample.prior.wpm == PASSAGE_WPM
    assert sample.prior.wpm / sample.rate.wpm == pytest.approx(1.467, abs=0.01)
    assert sample.sampled_words == 47
    assert sample.sampled_seconds == pytest.approx(23.61)


# ── sample selection: deterministic, and by word-count strata ────────────────

#: Lengths deliberately NOT in script order, so "the first k lines" and "the strata"
#: are different answers and a test can tell them apart.
_STRATA_TEXTS = [
    "w " * 14,  # 0
    "w " * 30,  # 1  longest
    "w " * 3,  # 2  shortest
    "w " * 21,  # 3
    "w " * 8,  # 4
    "w " * 17,  # 5
    "w " * 25,  # 6
    "w " * 11,  # 7
]


def test_the_sample_spans_word_count_strata_not_the_opening_lines() -> None:
    """Per-clip rate scatters 94-158 wpm on ONE voice and length is the largest single
    driver, so the sample must span length. The first k lines sample the OPENING.
    """
    picked = select_sample_indices(_STRATA_TEXTS, count=4)

    assert picked == (1, 2, 3, 7)
    lengths = sorted(len(_STRATA_TEXTS[index].split()) for index in picked)
    assert lengths == [3, 11, 21, 30], "shortest and longest both present"
    assert picked != (0, 1, 2, 3), "that is the opening, not a sample"


def test_a_one_line_sample_takes_the_median_not_an_extreme() -> None:
    """The two lines a single sample must NOT be drawn from are the extremes."""
    assert select_sample_indices(_STRATA_TEXTS, count=1) == (0,)
    assert len(_STRATA_TEXTS[0].split()) == 14


#: Six DISTINCT word counts in a deliberately jumbled script order. Six lines taken
#: three at a time puts the middle stratum at position ``1 * 5 / 2 == 2.5`` — the
#: only place where the module's explicit half-up rounding and Python's banker's
#: ``round`` disagree, and therefore the only corpus that can observe the rule at all.
_HALF_STEP_TEXTS = [" ".join(["w"] * n) for n in (3, 1, 6, 2, 5, 4)]


def test_a_half_integer_stratum_rounds_up_not_to_even() -> None:
    """``select_sample_indices`` documents explicit half-up rounding rather than
    ``round``. Every other corpus in this file lands on integer positions, so the rule
    was asserted by a docstring and by nothing else: swapping the line for ``round``
    left the whole suite green.

    Length-ordered, these six lines are ``[1, 3, 0, 5, 4, 2]``. Positions 0 and 5 are
    the extremes either way; the middle one is 2.5. Half-up takes position 3 (the
    four-word line, index 5); banker's ``round(2.5) == 2`` would take position 2 (the
    three-word line, index 0) — a DIFFERENT stratum, and one biased toward the short
    end on every script whose length is even.
    """
    picked = select_sample_indices(_HALF_STEP_TEXTS, count=3)

    assert picked == (1, 2, 5)
    assert [len(_HALF_STEP_TEXTS[index].split()) for index in picked] == [1, 6, 4]
    assert picked != (0, 1, 2), "that is what banker's rounding selects"


def test_an_empty_line_is_never_sampled() -> None:
    """A wordless line contributes nothing to a words-per-minute, and synthesizing
    one buys a clip nobody hears.
    """
    picked = select_sample_indices(["", "   ", "a b c", "d e", "\n"], count=4)

    assert picked == (2, 3)


def test_no_lines_at_all_selects_nothing_rather_than_raising() -> None:
    assert select_sample_indices([], count=4) == ()
    assert select_sample_indices(["", " "], count=4) == ()
    assert select_sample_indices(["a b"], count=0) == ()


#: Determinism corpus. ``_STRATA_TEXTS`` has eight DISTINCT lengths, so its ordering
#: never reaches the tie-break — and a tie-break is exactly where a hash dependency
#: would hide. These texts TIE (four at five words, two at twelve) and are distinct
#: strings, so only the index tie-break can order them stably.
_TIED_TEXTS = [
    "zulu " * 5,
    "alpha " * 12,
    "mike " * 5,
    "kilo " * 12,
    "echo " * 3,
    "romeo " * 5,
    "tango " * 20,
    "sierra " * 5,
]


def test_sample_selection_repeats_exactly_within_one_process() -> None:
    """Same draft, same answer — every time. A verdict that moves between two calls
    on the same text would make a re-ask and a trim unreproducible from the draft.
    """
    first = select_sample_indices(_TIED_TEXTS, count=4)

    assert all(select_sample_indices(_TIED_TEXTS, count=4) == first for _ in range(25))


def test_equal_length_lines_are_ordered_by_index_not_by_hash() -> None:
    """The tie-break must be positional. ``_TIED_TEXTS`` has four five-word lines, and
    anything that ranks them by their CONTENT (a hash, a set, a dict) reorders them
    between interpreters — which the cross-seed test below is what actually proves.
    """
    five_word = [index for index, text in enumerate(_TIED_TEXTS) if len(text.split()) == 5]

    assert five_word == [0, 2, 5, 7], "the corpus really does tie"
    assert select_sample_indices(_TIED_TEXTS, count=4) == (1, 2, 4, 6)


@pytest.mark.parametrize("seed", ["0"])
def test_sample_selection_is_identical_across_pythonhashseed_values(seed: str) -> None:
    """Out of process, under a different string-hash randomisation, byte for byte.

    An in-process repeat cannot catch a hash-, set- or dict-iteration dependency: the
    seed is fixed for the life of the interpreter. This spawns a fresh one per seed,
    on the corpus whose ties make the ordering rule observable at all.
    """
    program = (
        "from src.services.stages.shared.speech_rate import select_sample_indices\n"
        f"texts = {_TIED_TEXTS!r}\n"
        "print(select_sample_indices(texts, count=4))\n"
    )
    completed = subprocess.run(  # noqa: S603 -- fixed argv, no shell, our own interpreter
        [sys.executable, "-c", program],
        env={**os.environ, "PYTHONHASHSEED": seed},
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[4],
    )

    assert completed.stdout.strip() == str(select_sample_indices(_TIED_TEXTS, count=4))


def test_sample_size_is_four_by_default_and_grows_with_the_script() -> None:
    """k=4 puts the standard error of the pooled rate at ~3.5% (sd/sqrt(k) on a
    measured sd of ~7.1%), inside the +/-10% band the verdict is judged against.
    """
    assert sample_size(16) == DEFAULT_SAMPLE_LINES
    assert sample_size(3) == 3, "never more lines than the script has"
    assert sample_size(40) > DEFAULT_SAMPLE_LINES, "a long script is measured harder"
    assert sample_size(4000) == MAX_SAMPLE_LINES, "and never unboundedly so"
    assert sample_size(0) == 0


# ── failure: fall back to the prior, and SAY SO ──────────────────────────────


@pytest.mark.asyncio
async def test_a_sample_that_cannot_synthesize_keeps_the_prior_and_names_the_reason() -> None:
    fake = _ScriptTTS({})  # every line raises

    sample = await speech_rate_module.resolve_script_sample_wpm(
        _ctx({"voice_id": "af_heart"}),
        texts=list(LIVE_LINES),
        prior=_prior(),
        client=_client(fake),
        sample_lines=3,
    )

    assert sample.rate.wpm == PASSAGE_WPM
    assert sample.rate.basis == CALIBRATION_PASSAGE, "the basis says WHICH text was measured"
    assert sample.reason == SAMPLE_TOO_FEW_MEASURED
    assert sample.sampled is False
    # Selected but unmeasured: the caller can say how hard it tried.
    assert len(sample.sampled_indices) == 3
    assert sample.sampled_words == 0


@pytest.mark.asyncio
async def test_a_partial_sample_measures_off_the_lines_that_landed() -> None:
    """k-1 real measurements still beat a passage that is 47% off, so one line that
    times out costs its own contribution and nothing else.
    """
    texts = list(LIVE_LINES)
    landed = {texts[0]: LIVE_LINES[texts[0]], texts[2]: LIVE_LINES[texts[2]]}
    fake = _ScriptTTS(landed)

    sample = await speech_rate_module.resolve_script_sample_wpm(
        _ctx({"voice_id": "af_heart"}),
        texts=texts,
        prior=_prior(),
        client=_client(fake),
        sample_lines=3,
    )

    assert sample.reason is None
    assert sample.sampled_words == 35  # 17 + 18, the middle line dropped
    assert sample.sampled_seconds == pytest.approx(8.37 + 8.62)
    assert sample.rate.wpm == pytest.approx(35 / ((8.37 + 8.62) / 60.0))


@pytest.mark.asyncio
async def test_one_measured_line_is_refused_rather_than_believed() -> None:
    """Per-clip rate scatters 94-158 wpm on one voice, so a single clip's rate is
    that clip's rate. Below MIN_SAMPLE_LINES the sample is not a sample.
    """
    texts = list(LIVE_LINES)
    fake = _ScriptTTS({texts[0]: LIVE_LINES[texts[0]]})

    sample = await speech_rate_module.resolve_script_sample_wpm(
        _ctx({"voice_id": "af_heart"}),
        texts=texts,
        prior=_prior(),
        client=_client(fake),
        sample_lines=3,
    )

    assert sample.reason == SAMPLE_TOO_FEW_MEASURED
    assert sample.rate.wpm == PASSAGE_WPM


# ── failure: the sample landed but is DEGENERATE (short-scaffolding bias) ────


@pytest.mark.asyncio
async def test_a_short_boilerplate_sample_is_refused_not_believed() -> None:
    """THE PROVEN PRODUCTION BIAS. ``demo_video_gen`` samples 4 lines of bdd-drafted
    scaffolding text ("First, I am on the ... page.") that ``script_doctor`` rewrites
    entirely before anything ships — live run kb5 measured a 9.0 words/line sample at
    197.4 wpm against a 142.0 wpm delivery (+39%). This fixture reproduces that shape
    exactly: 4 near-identical 9-word lines, each paying the same fixed per-utterance
    overhead amortised over few words, landing WELL inside [MIN_PLAUSIBLE_WPM,
    MAX_PLAUSIBLE_WPM] — so the existing plausibility band alone cannot catch it.
    """
    texts = [
        "First, I am now on the React page.",
        "First, I am now on the Vue page.",
        "First, I am now on the Svelte page.",
        "First, I am now on the Angular page.",
    ]
    assert all(len(text.split()) == 8 for text in texts)  # 8 words/line — pin the fixture
    # 8 words/line at a real ~150 wpm voice, amortising a fixed overhead so the
    # MEASURED rate reads far higher than the voice's true rate — the same physics
    # documented for CALIBRATION_TEXT's own short-sentence correction.
    fake = _ScriptTTS({text: 8 * 60.0 / 260.0 for text in texts})

    sample = await speech_rate_module.resolve_script_sample_wpm(
        _ctx({"voice_id": "af_heart"}),
        texts=texts,
        prior=_prior(),
        client=_client(fake),
        sample_lines=4,
    )

    assert sample.reason == SAMPLE_LINES_TOO_SHORT
    assert sample.rate.wpm == PASSAGE_WPM
    assert sample.rate.basis == CALIBRATION_PASSAGE
    # Still measured (evidence is kept, per _split_evidence's own contract) even
    # though the caller must not use it as the rate.
    assert sample.sampled_words == 32
    assert sample.sampled is False


@pytest.mark.asyncio
async def test_a_realistic_length_sample_is_not_refused_by_the_new_guard() -> None:
    """The guard must not reject real narration-length content.
    :data:`MIN_SAMPLE_AVG_WORDS_PER_LINE` sits a safety margin below the 12-14
    words/sentence this module already treats as the realistic band, so a sample
    averaging comfortably inside that band (the live 17/12/18-word production lines)
    must still land.
    """
    sample = await speech_rate_module.resolve_script_sample_wpm(
        _ctx({"voice_id": "af_heart"}),
        texts=list(LIVE_LINES),
        prior=_prior(),
        client=_client(_ScriptTTS(LIVE_LINES)),
        sample_lines=3,
    )

    assert sample.reason is None
    assert sample.rate.basis == SCRIPT_SAMPLE
    assert sample.sampled_words / len(sample.sampled_indices) >= (
        speech_rate_module.MIN_SAMPLE_AVG_WORDS_PER_LINE
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [0.0])
async def test_a_line_with_an_unusable_duration_is_dropped_not_pooled(bad: float) -> None:
    """The audio path answered and the number it reported is not usable. Pooling it
    would corrupt every OTHER line's contribution, not just its own.
    """
    texts = list(LIVE_LINES)
    durations = dict(LIVE_LINES)
    durations[texts[1]] = bad
    fake = _ScriptTTS(durations)

    sample = await speech_rate_module.resolve_script_sample_wpm(
        _ctx({"voice_id": "af_heart"}),
        texts=texts,
        prior=_prior(),
        client=_client(fake),
        sample_lines=3,
    )

    assert sample.sampled_words == 35
    assert sample.sampled_seconds == pytest.approx(8.37 + 8.62)
    assert math.isfinite(sample.rate.wpm)


@pytest.mark.asyncio
async def test_an_implausible_content_rate_is_rejected_by_the_unchanged_band() -> None:
    """The band's job is unchanged: reject a BROKEN measurement (truncated audio, a
    stub duration), never a plausible-but-slow one. A content rate is EXPECTED to sit
    well under the passage rate, so nothing here may narrow toward the passage.
    """
    texts = list(LIVE_LINES)
    # 47 words in 0.5s -> 5640 wpm: a stub duration, not a fast voice.
    fake = _ScriptTTS(dict.fromkeys(texts, 0.5 / 3))

    sample = await speech_rate_module.resolve_script_sample_wpm(
        _ctx({"voice_id": "af_heart"}),
        texts=texts,
        prior=_prior(),
        client=_client(fake),
        sample_lines=3,
    )

    assert sample.reason == IMPLAUSIBLE_RATE
    assert sample.rate.wpm == PASSAGE_WPM
    assert sample.rate.basis == CALIBRATION_PASSAGE


@pytest.mark.asyncio
async def test_a_draft_with_nothing_to_say_reports_sample_no_lines() -> None:
    fake = _ScriptTTS({})

    sample = await speech_rate_module.resolve_script_sample_wpm(
        _ctx({"voice_id": "af_heart"}),
        texts=["", "   "],
        prior=_prior(),
        client=_client(fake),
    )

    assert sample.reason == SAMPLE_NO_LINES
    assert sample.sampled_indices == ()
    assert fake.calls == [], "nothing to measure, so nothing is synthesized"


@pytest.mark.asyncio
async def test_the_sample_never_raises_on_a_paid_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A calibration courtesy must never be the thing that kills a paid run."""

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise ZeroDivisionError("the arithmetic broke")

    monkeypatch.setattr(speech_rate_module, "_sample_totals", _boom)

    sample = await speech_rate_module.resolve_script_sample_wpm(
        _ctx({"voice_id": "af_heart"}),
        texts=list(LIVE_LINES),
        prior=_prior(),
        client=_client(_ScriptTTS(LIVE_LINES)),
        sample_lines=3,
    )

    assert sample.reason == RESOLUTION_ERROR
    assert sample.rate.wpm == PASSAGE_WPM


# ── the per-draft cache ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_content_rate_is_cached_per_run_and_per_draft() -> None:
    """Keyed by (run, voice, provider, sampled text) — NOT by voice alone. A re-ask
    is different text and must be measured afresh; another run's identical draft is
    another run and must not be answered from this one.
    """
    texts = list(LIVE_LINES)
    fake = _ScriptTTS(LIVE_LINES)
    ctx = _ctx({"voice_id": "af_heart"})

    async def _sample(target_ctx: RunContext, sampled: list[str]) -> float:
        result = await speech_rate_module.resolve_script_sample_wpm(
            target_ctx,
            texts=sampled,
            prior=_prior(),
            client=_client(fake),
            sample_lines=3,
        )
        return result.rate.wpm

    first = await _sample(ctx, texts)
    assert len(fake.calls) == 3

    again = await _sample(ctx, texts)
    assert again == first
    assert len(fake.calls) == 3, "the same draft of the same run is not re-synthesized"

    # A re-ask: same run, same voice, DIFFERENT words.
    reworded = [*texts[:2], " ".join(f"echo{i}" for i in range(18))]
    fake.durations[reworded[2]] = 9.9
    await _sample(ctx, reworded)
    assert len(fake.calls) == 6, "a new draft revision is measured, never assumed"

    # Another run, byte-identical draft: a different run is a different measurement.
    await _sample(_ctx({"voice_id": "af_heart"}), texts)
    assert len(fake.calls) == 9


# ── persistence: the sample IS the narration ─────────────────────────────────


@pytest.mark.asyncio
async def test_the_sample_is_stored_under_the_narrate_stage_at_the_narrate_speed(
    monkeypatch: pytest.MonkeyPatch, stored: list[dict[str, Any]]
) -> None:
    """THE thing that makes the sample free.

    ``deterministic_clip_id`` hashes ``(voice, speed, PREPROC_VERSION, text)`` and
    ``find_existing_clip`` looks under ``(run_id, stage_name, clip_id)``. Store the
    sample under the NARRATE stage's name at the speed narrate will request, and
    narrate's own existence probe hits: it synthesizes nothing for that line.

    A sample stored anywhere else, or at any other speed, is a clip nobody reuses.
    """
    _patch_store(monkeypatch, stored)
    fake = _ScriptTTS(LIVE_LINES)
    ctx = _ctx({"voice_id": "af_heart"})

    await speech_rate_module.resolve_script_sample_wpm(
        ctx,
        texts=list(LIVE_LINES),
        prior=_prior(),
        client=_client(fake),
        speed=None,
        narrate_stage_name="app_review_record",
        sample_lines=3,
    )

    assert len(stored) == 3
    assert {call["stage_name"] for call in stored} == {"app_review_record"}
    assert {call["run_id"] for call in stored} == {str(ctx.run_id)}
    assert {call["voice"] for call in stored} == {"af_heart"}
    assert [call["speed"] for call in stored] == [None, None, None], (
        "None is the backend default and 1.0 is an explicit request — different "
        "clip ids, so an invented 1.0 would be reused by nothing"
    )
    # The clip the sample stored is byte-for-byte the clip narrate will ask for.
    assert {
        deterministic_clip_id(str(call["text"]), call["voice"], call["speed"]) for call in stored
    } == {deterministic_clip_id(text, "af_heart", None) for text in LIVE_LINES}


@pytest.mark.asyncio
async def test_without_a_narrate_stage_the_sample_measures_and_stores_nothing(
    monkeypatch: pytest.MonkeyPatch, stored: list[dict[str, Any]]
) -> None:
    """A caller that re-synthesizes at a per-line rate (demo_video's ``vo_rate``)
    would reuse nothing, so it must be able to decline to store — and then the sample
    is genuinely spent, which is the honest answer rather than a contorted one.
    """
    _patch_store(monkeypatch, stored)
    fake = _ScriptTTS(LIVE_LINES)

    sample = await speech_rate_module.resolve_script_sample_wpm(
        _ctx({"voice_id": "af_heart"}),
        texts=list(LIVE_LINES),
        prior=_prior(),
        client=_client(fake),
        narrate_stage_name=None,
        sample_lines=3,
    )

    assert stored == []
    assert len(fake.calls) == 3, "still measured, just not persisted"
    assert sample.rate.wpm == pytest.approx(LIVE_CONTENT_WPM)


# ── the degradation ──────────────────────────────────────────────────────────


def test_script_sample_degradation_names_the_reason_and_the_rate_actually_used() -> None:
    sample = ScriptSpeechRate(
        rate=_prior(),
        prior=_prior(),
        sampled_indices=(0, 3, 7, 11),
        sampled_words=0,
        sampled_seconds=0.0,
        reason=SAMPLE_TOO_FEW_MEASURED,
    )

    entry = script_sample_degradation(stage="app_review_script", sample=sample)

    assert entry["code"] == SPEECH_RATE_UNUSABLE
    assert "sample_too_few_measured" in entry["detail"]
    assert "175.2 wpm" in entry["detail"], "the rate the run was actually budgeted at"
    assert "calibration_passage" in entry["detail"]
    assert "4 line(s) selected" in entry["detail"]


def test_script_sample_degradation_refuses_a_sample_that_landed() -> None:
    """A degradation channel that also carries successes is one people learn to
    ignore — ``mastering_gate_failed`` would have fired on 96.7% of masters.
    """
    landed = ScriptSpeechRate(
        rate=SpeechRate(wpm=119.4, measured=True, basis=SCRIPT_SAMPLE),
        prior=_prior(),
        reason=None,
    )

    with pytest.raises(ValueError, match="that MEASURED"):
        script_sample_degradation(stage="app_review_script", sample=landed)


def test_a_speech_rate_means_the_calibration_passage_unless_it_says_otherwise() -> None:
    """Every historical construction in this module is Phase A, and stays Phase A."""
    assert SpeechRate(wpm=150.0, measured=True).basis == CALIBRATION_PASSAGE
    assert CALIBRATION_PASSAGE != SCRIPT_SAMPLE


# ══ the two-term decomposition — MEASURED and REPORTED, never budgeted ═══════
#
# A clip's duration is not `words / rate`. It is roughly
# `words / speech_rate + n_pieces * per_piece_padding`, and the second term
# scales with SENTENCE COUNT, not word count — which is why one wpm scalar
# cannot express it. `tts_backend` now reports the split per synthesis
# (`speech_seconds` / `pause_seconds`), so the sampler pools and LOGS both terms.
#
# ⛔ It deliberately does NOT budget from them. Costed against the three real
# af_heart clips, a two-term budget spans s* 0.93-1.11 even granted perfect
# padding knowledge, and 0.83-1.24 once the piece count is off by +/-2 — which
# the planner cannot avoid, because the LLM picks the sentence structure after
# the budget is written. These tests therefore pin BOTH halves: the split is
# measured correctly, AND the rate the sampler returns is untouched by it.


class _SplitScriptTTS(_ScriptTTS):
    """`_ScriptTTS` whose responses also carry the backend's speech/pause split.

    `splits` maps text -> (speech_seconds, pause_seconds). A text ABSENT from it
    synthesizes fine but reports no split at all — the mixed-fleet case (one
    tts_backend upgraded, one not), which must contribute to the rate and to
    nothing else.
    """

    def __init__(self, durations: dict[str, float], splits: dict[str, tuple[float, float]]) -> None:
        super().__init__(durations)
        self.splits = dict(splits)

    async def synthesize(self, **kwargs: Any) -> Any:
        result = await super().synthesize(**kwargs)
        text = str(kwargs["text"])
        if text in self.splits:
            speech, pause = self.splits[text]
            result.speech_seconds = speech
            result.pause_seconds = pause
        return result


def test_approx_pieces_counts_sentence_terminators_with_a_floor_of_one() -> None:
    """The piece count is an APPROXIMATION of the backend's own segmentation.

    It is used only to report a per-piece padding figure, so what matters is that
    it tracks sentence structure and never returns 0 (which would divide by zero
    or silently claim one enormous piece).
    """
    approx = speech_rate_module._approx_pieces
    assert approx("No punctuation here") == 1
    assert approx("One. Two. Three.") == 3
    assert approx("Trailing prose after a stop. And more") == 2
    # Runs of terminators are ONE boundary, not several — "Wait!?" is one piece.
    assert approx("Wait!? Then this.") == 2
    assert approx("") == 1
    assert approx("...") == 1


@pytest.mark.asyncio
async def test_the_split_is_pooled_only_over_lines_that_reported_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The mixed-fleet case: a line with no split feeds the RATE and nothing else.

    Line one reports 6.00s speech + 2.37s pause (its 8.37s total); line three
    reports nothing. Pooling line three as zero pause would claim a measurement
    nobody took and drag the reported padding fraction toward zero — precisely
    the "absent means unknown, never zero" rule this seam exists to hold.

    So the decomposition must describe 17 words over 6.00s of speech
    (= 170.0 wpm), NOT 35 words over 14.62s, while the rate the sampler RETURNS
    still pools both lines exactly as before.
    """
    texts = list(LIVE_LINES)
    landed = {texts[0]: LIVE_LINES[texts[0]], texts[2]: LIVE_LINES[texts[2]]}
    fake = _SplitScriptTTS(landed, {texts[0]: (6.00, 2.37)})

    with caplog.at_level("INFO"):
        sample = await speech_rate_module.resolve_script_sample_wpm(
            _ctx({"voice_id": "af_heart"}),
            texts=texts,
            prior=_prior(),
            client=_client(fake),
            sample_lines=3,
        )

    # ── the budget is UNTOUCHED: both landed lines, exactly as before ──
    assert sample.sampled_words == 35
    assert sample.sampled_seconds == pytest.approx(8.37 + 8.62)
    assert sample.rate.wpm == pytest.approx(35 / ((8.37 + 8.62) / 60.0))

    # ── the diagnostic describes ONLY the reporting line ──
    line = next(
        r.getMessage() for r in caplog.records if "two-term decomposition" in r.getMessage()
    )
    assert "speech 170.0 wpm" in line, line
    assert "over 6.00s of speech" in line, line
    # 17 alpha-words, no terminators ⇒ one approximate piece ⇒ the whole 2.37s.
    assert "~2.370s per piece" in line, line
    assert "1 line(s) reported the split" in line, line
    # 2.37 / (6.00 + 2.37) — computed over the reporting line's OWN audio, not
    # over the pooled 17.0s the rate was measured across.
    assert "28.3% of the audio" in line, line


@pytest.mark.asyncio
async def test_a_backend_that_reports_no_split_logs_no_decomposition(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Silence, not a row of zeros.

    A zeroed decomposition reads exactly like "this voice inserts no padding",
    which is a finding — and it would be fabricated. A backend that cannot answer
    must produce no line at all, while the rate it DOES support is unaffected.
    """
    fake = _ScriptTTS(LIVE_LINES)

    with caplog.at_level("INFO"):
        sample = await speech_rate_module.resolve_script_sample_wpm(
            _ctx({"voice_id": "af_heart"}),
            texts=list(LIVE_LINES),
            prior=_prior(),
            client=_client(fake),
            sample_lines=3,
        )

    assert sample.rate.wpm == pytest.approx(LIVE_CONTENT_WPM)
    assert not [r for r in caplog.records if "two-term decomposition" in r.getMessage()]


@pytest.mark.asyncio
async def test_the_reported_speech_rate_differs_from_the_budgeted_rate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """THE finding, in one assertion: the two numbers are not the same number.

    The sampler budgets from words/TOTAL (what it has always done) while the
    decomposition reports words/SPEECH. Both lines here are 40.0% padding, so the
    two rates must differ by exactly 1/0.6 = 1.667x — and that gap is the entire
    reason a single wpm scalar over-fills every beat. Asserting they DIVERGE by
    the padding ratio is what makes this fail if the decomposition is ever
    accidentally computed off total duration again (it would then match exactly).
    """
    texts = list(LIVE_LINES)
    first, third = texts[0], texts[2]  # 17 words / 8.37s, 18 words / 8.62s
    fake = _SplitScriptTTS(
        {first: 8.37, third: 8.62},
        {first: (8.37 * 0.6, 8.37 * 0.4), third: (8.62 * 0.6, 8.62 * 0.4)},
    )

    with caplog.at_level("INFO"):
        sample = await speech_rate_module.resolve_script_sample_wpm(
            _ctx({"voice_id": "af_heart"}),
            texts=[first, third],
            prior=_prior(),
            client=_client(fake),
            sample_lines=2,
        )

    total_s = 8.37 + 8.62
    budgeted = 35 / (total_s / 60.0)  # ~123.6 wpm — unchanged behaviour
    speech_only = 35 / ((total_s * 0.6) / 60.0)  # ~205.9 wpm — what the voice does
    assert sample.rate.wpm == pytest.approx(budgeted)
    line = next(
        r.getMessage() for r in caplog.records if "two-term decomposition" in r.getMessage()
    )
    assert f"speech {speech_only:.1f} wpm" in line, line
    assert f"the {budgeted:.1f} wpm this sample will budget from" in line, line
    assert "40.0% of the audio" in line, line
    # The divergence IS the padding ratio — not incidentally close, exactly 1/0.6.
    assert speech_only / budgeted == pytest.approx(1.0 / 0.6)


# ══ …and now the split is PERSISTED, not only logged ════════════════════════
#
# The four `*_split` totals were computed and then dropped at
# `_log_two_term_decomposition` — explicitly "DIAGNOSTIC ONLY", with zero readers
# outside that function. So the fleet-wide sample a real two-term fit needs
# accumulated nowhere a query could reach. These tests pin that they now leave
# the function on `ScriptSpeechRate`.
#
# ⛔ MEASUREMENT ENABLEMENT ONLY. Every test here also asserts the RATE is
# untouched — the arithmetic, the budget and the fallbacks must be byte-identical
# to a build without these fields.


@pytest.mark.asyncio
async def test_the_split_leaves_the_function_on_the_result(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The same numbers the diagnostic log line prints, now on the return value.

    Line one reports 6.00s speech + 2.37s pause over one approximate piece; line
    three reports nothing, so `lines_split` is 1 while `sampled_seconds` pools
    both lines. That gap IS the record's meaning: the split describes a SUBSET.
    """
    texts = list(LIVE_LINES)
    landed = {texts[0]: LIVE_LINES[texts[0]], texts[2]: LIVE_LINES[texts[2]]}
    fake = _SplitScriptTTS(landed, {texts[0]: (6.00, 2.37)})

    with caplog.at_level("INFO"):
        sample = await speech_rate_module.resolve_script_sample_wpm(
            _ctx({"voice_id": "af_heart"}),
            texts=texts,
            prior=_prior(),
            client=_client(fake),
            sample_lines=3,
        )

    assert sample.speech_seconds_split == pytest.approx(6.00)
    assert sample.pause_seconds_split == pytest.approx(2.37)
    assert sample.approx_pieces_split == 1
    assert sample.lines_split == 1

    # ⛔ NOTHING BRANCHED. The rate is pooled over BOTH lines exactly as before.
    assert sample.sampled_words == 35
    assert sample.sampled_seconds == pytest.approx(8.37 + 8.62)
    assert sample.rate.wpm == pytest.approx(35 / ((8.37 + 8.62) / 60.0))
    assert sample.rate.basis == SCRIPT_SAMPLE


@pytest.mark.asyncio
async def test_a_backend_with_no_split_records_zero_lines_not_zero_pause() -> None:
    """`lines_split == 0` is "no evidence", and a reader must be able to see it.

    All four zeros are indistinguishable from "this voice inserts no padding" —
    a finding, and a fabricated one. `lines_split` is the field that tells them
    apart, so it is asserted here beside the zeroed seconds.
    """
    fake = _ScriptTTS(LIVE_LINES)

    sample = await speech_rate_module.resolve_script_sample_wpm(
        _ctx({"voice_id": "af_heart"}),
        texts=list(LIVE_LINES),
        prior=_prior(),
        client=_client(fake),
        sample_lines=3,
    )

    assert sample.lines_split == 0
    assert sample.speech_seconds_split == 0.0
    assert sample.pause_seconds_split == 0.0
    assert sample.approx_pieces_split == 0
    # The rate it DOES support is unaffected.
    assert sample.rate.wpm == pytest.approx(LIVE_CONTENT_WPM)


@pytest.mark.asyncio
async def test_a_rejected_sample_still_carries_its_split_evidence() -> None:
    """The split EXPLAINS a rejection, so it must survive one.

    A pooled rate outside the plausible band is exactly the sample whose
    speech/pause breakdown says whether the audio truncated or the padding
    exploded. Here 17 words in 300.00s pools to ~3.4 wpm — rejected — while the
    backend reported a perfectly ordinary 6.00s/2.37s split for that same line.
    Withholding the evidence on the failure path would drop it where it is worth
    most.
    """
    texts = list(LIVE_LINES)
    fake = _SplitScriptTTS({texts[0]: 300.0}, {texts[0]: (6.00, 2.37)})

    sample = await speech_rate_module.resolve_script_sample_wpm(
        _ctx({"voice_id": "af_heart"}),
        texts=texts,
        prior=_prior(),
        client=_client(fake),
        sample_lines=3,
    )

    # The rejection stands — this changes nothing about the verdict.
    assert sample.rate.basis == CALIBRATION_PASSAGE
    assert sample.reason is not None
    # …and the evidence came out with it.
    assert sample.lines_split == 1
    assert sample.speech_seconds_split == pytest.approx(6.00)
    assert sample.pause_seconds_split == pytest.approx(2.37)


@pytest.mark.asyncio
async def test_a_cache_hit_returns_the_same_evidence_as_the_miss_that_filled_it() -> None:
    """⭐ WAVE 14, N2 — THE HIT PATH USED TO PUBLISH FOUR ZEROS THAT MEANT "NO SAMPLE".

    The per-run content cache stored the wpm ALONE, so a second stage in the same run
    asking for the same rate got a ``ScriptSpeechRate`` with ``sampled_words=0``,
    ``sampled_seconds=0.0`` and the whole two-term split at zero — which is exactly the
    documented reading of "no backend reported a split / the sample never ran" (see
    ``test_a_backend_with_no_split_records_zero_lines_not_zero_pause``, which asserts
    those same zeros as the NO-EVIDENCE signal). The two cases were indistinguishable,
    and every cache hit was silently subtracted from the fleet-wide sample the split
    fields were persisted to accumulate.

    Same ``ctx`` (same ``run_id``), same texts ⇒ the second call is a HIT: the fake TTS
    is asserted to have been asked exactly once, so this cannot pass by accidentally
    re-measuring.
    """
    texts = list(LIVE_LINES)
    landed = {texts[0]: LIVE_LINES[texts[0]], texts[2]: LIVE_LINES[texts[2]]}
    fake = _SplitScriptTTS(landed, {texts[0]: (6.00, 2.37)})
    ctx = _ctx({"voice_id": "af_heart"})

    miss = await speech_rate_module.resolve_script_sample_wpm(
        ctx, texts=texts, prior=_prior(), client=_client(fake), sample_lines=3
    )
    calls_after_miss = len(fake.calls)
    hit = await speech_rate_module.resolve_script_sample_wpm(
        ctx, texts=texts, prior=_prior(), client=_client(fake), sample_lines=3
    )

    assert len(fake.calls) == calls_after_miss, "the second call re-measured; not a cache hit"

    # The rate was always right. It is the EVIDENCE that was dropped.
    assert hit.rate.wpm == pytest.approx(miss.rate.wpm)
    assert hit.rate.basis == SCRIPT_SAMPLE
    assert (hit.sampled_words, hit.sampled_seconds) == (
        miss.sampled_words,
        miss.sampled_seconds,
    )
    assert hit.speech_seconds_split == pytest.approx(miss.speech_seconds_split)
    assert hit.pause_seconds_split == pytest.approx(miss.pause_seconds_split)
    assert hit.approx_pieces_split == miss.approx_pieces_split
    assert hit.lines_split == miss.lines_split

    # …and pinned by VALUE, so a mutation that zeroes both paths alike cannot pass.
    assert hit.lines_split == 1
    assert hit.sampled_words == 35
    assert hit.speech_seconds_split == pytest.approx(6.00)
    assert hit.pause_seconds_split == pytest.approx(2.37)


@pytest.mark.asyncio
async def test_the_wpm_memo_is_scoped_to_the_runs_tts_pin() -> None:
    """A tenant BYO calibration must never answer for another tenant or the platform.

    The memo was process-wide on ``(provider, voice)``: tenant A's ElevenLabs voice
    "narrator" calibrated once, and tenant B (its own connection) or the platform asking
    for a same-named voice got A's rate. The tts pin (source, connection, version) is
    part of the key, so each pin probes its own voice.
    """
    from src.auth.run_providers import bind_run_provider_slots, clear_run_provider_slots

    def _tenant_pin(version: int) -> dict[str, object]:
        return {
            "provider_slots": {
                "tts": {
                    "source": "tenant",
                    "kind": "elevenlabs",
                    "connection_id": "0192f0c4-0000-7000-8000-0000000000a1",
                    "version": version,
                }
            }
        }

    tenant_a = _FakeTTS(duration=CALIBRATION_WORDS * 60.0 / 120.0)
    tenant_a_v2 = _FakeTTS(duration=CALIBRATION_WORDS * 60.0 / 160.0)
    platform = _FakeTTS(duration=_KOKORO_SECONDS)
    try:
        bind_run_provider_slots(_tenant_pin(1))
        a = await measure_natural_wpm(_client(tenant_a), voice="narrator")
        # Same pin → memo hit, no second call.
        assert await measure_natural_wpm(_client(tenant_a), voice="narrator") == a
        bind_run_provider_slots(_tenant_pin(2))
        a2 = await measure_natural_wpm(_client(tenant_a_v2), voice="narrator")
        clear_run_provider_slots()
        p = await measure_natural_wpm(_client(platform), voice="narrator")
    finally:
        clear_run_provider_slots()

    assert len(tenant_a.calls) == 1
    assert len(tenant_a_v2.calls) == 1, "a re-versioned connection is a new voice"
    assert len(platform.calls) == 1, "the platform must not be served the tenant's rate"
    assert a == pytest.approx(120.0)
    assert a2 == pytest.approx(160.0)
    assert p == pytest.approx(94.0)


class _HangingScriptTTS(_ScriptTTS):
    """``_ScriptTTS`` whose ``hang`` text never answers — a wedged synthesis."""

    def __init__(self, durations: dict[str, float], *, hang: str) -> None:
        super().__init__(durations)
        self.hang = hang

    async def synthesize(self, **kwargs: Any) -> Any:
        if str(kwargs["text"]) == self.hang:
            self.calls.append(kwargs)
            await asyncio.sleep(3600)
        return await super().synthesize(**kwargs)


@pytest.mark.asyncio
async def test_one_hung_sample_line_costs_only_its_own_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``_sample_totals`` docstring promises failures are PER LINE. It used to wrap
    the whole gather in ONE ``wait_for``: a single hung line raised out of the fan-out,
    discarded the two lines that had landed, and reported ``resolution_error`` with the
    prior rate. Now the hung line alone is dropped and the other two are measured.
    """
    monkeypatch.setattr(speech_rate_module, "_PROBE_TIMEOUT_SECONDS", 0.05)
    texts = list(LIVE_LINES)
    fake = _HangingScriptTTS(LIVE_LINES, hang=texts[1])

    sample = await speech_rate_module.resolve_script_sample_wpm(
        _ctx({"voice_id": "af_heart"}),
        texts=texts,
        prior=_prior(),
        client=_client(fake),
        sample_lines=3,
    )

    assert sample.reason is None
    assert sample.sampled_words == 35  # 17 + 18, the hung middle line dropped
    assert sample.sampled_seconds == pytest.approx(8.37 + 8.62)
    assert sample.rate.basis == SCRIPT_SAMPLE
