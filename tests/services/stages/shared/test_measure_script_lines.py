"""Unit tests for PHASE C — ``speech_rate.measure_script_lines``.

Phase B samples ``k`` lines and applies the pooled rate to every other line, which
re-imports ~5% of sampling error into a number the run is judged on. Phase C
synthesizes EVERY line, so the total is the sum of real clip seconds and a film's
delivered length is known before it is compiled.

Contracts pinned here:

* every line measured ⇒ ``total_seconds`` is the SUM of the real durations, basis
  ``measured_all``, and nothing anywhere is a rate applied to a word count;
* one line that fails is extrapolated at the rate of its OWN VOICE, not the pooled
  one — a cast's voices differ by far more than the error being removed;
* a voice with no measured line falls through to pooled, rather than to 0.0s;
* fewer than ``MIN_SAMPLE_LINES`` usable durations is REFUSED, not believed, and the
  refusal is reportable as a degradation;
* a mis-zipped ``texts``/``voices`` is a caller bug that must not raise;
* a blank line costs zero seconds and zero syntheses;
* a script with NO spoken line is measured exactly (0.0s, ``measured_all``, no
  degradation) rather than refused — a signal that fires on a correct result is not
  a signal;
* the recovery path is itself total: a ``texts``/``voices`` that is not iterable at
  all falls back rather than escaping the "NEVER RAISES" guarantee;
* the per-line timeout grows with queue position, counted over the lines actually
  SUBMITTED, because the pod-wide TTS gate makes a flat bound a queue-depth test
  rather than a timeout — and a junk gate size falls back to the default, not to 1;
* ``asyncio.CancelledError`` PROPAGATES — a cancelled run must stop synthesizing;
* clips are stored under the narrate stage's own name, which is what makes them free.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, cast
from uuid import uuid4

import pytest

import src.services.stages.shared.speech_rate as speech_rate_module
from src.clients.tts_client import TTSClient
from src.services.runs.run_context import RunContext
from src.services.stages.shared.degradation import SPEECH_RATE_UNUSABLE
from src.services.stages.shared.speech_rate import (
    BASIS_MEASURED_ALL,
    BASIS_MEASURED_PARTIAL,
    CALIBRATION_PASSAGE,
    IMPLAUSIBLE_RATE,
    MIN_SAMPLE_LINES,
    RESOLUTION_ERROR,
    SAMPLE_TOO_FEW_MEASURED,
    ScriptMeasurement,
    SpeechRate,
    measure_script_lines,
    script_measurement_degradation,
)

_STAGE = "app_review_record"


def _ctx() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"voice": {"voice_id": "af_heart", "provider": "kokoro"}},
        prior_run_outputs={},
    )


def _prior(wpm: float = 175.2) -> SpeechRate:
    """A LANDED Phase-A measurement — the passage rate Phase C is correcting."""
    return SpeechRate(wpm=wpm, measured=True)


def _client(fake: object) -> TTSClient:
    return cast("TTSClient", fake)


class _LineTTS:
    """Answers each (text) with its own duration; anything unknown raises.

    Mirrors ``test_speech_rate._ScriptTTS`` — a backend that answers ONE duration
    whatever it is asked is exactly the assumption this whole module deletes.
    ``hang`` names texts that never answer, so a timeout can be exercised without
    ever waiting a real timeout.
    """

    def __init__(self, durations: dict[str, float], *, hang: set[str] | None = None) -> None:
        self.durations = dict(durations)
        self.hang = set(hang or ())
        self.calls: list[dict[str, Any]] = []

    async def synthesize(self, **kwargs: Any) -> Any:
        await asyncio.sleep(0)
        self.calls.append(kwargs)
        text = str(kwargs["text"])
        if text in self.hang:
            await asyncio.sleep(3600)
        if text not in self.durations:
            raise RuntimeError(f"no synthesis for {text!r}")
        return type("_Synth", (), {"duration_seconds": self.durations[text]})()

    @property
    def texts(self) -> list[str]:
        return [str(call["text"]) for call in self.calls]


@pytest.fixture
def stored() -> list[dict[str, Any]]:
    """Every ``synthesize_and_store_clip`` call Phase C made, as recorded kwargs."""
    return []


@pytest.fixture(autouse=True)
def _patch_store(monkeypatch: pytest.MonkeyPatch, stored: list[dict[str, Any]]) -> None:
    """Stand in for ``primitives.narration.synthesize_and_store_clip`` (needs MinIO).

    What matters is that Phase C persists through the SAME call narrate uses, under
    narrate's stage name and speed — that is what makes ``deterministic_clip_id``
    collide and the measurement free.
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


def _line(prefix: str, words: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(words))


# Three real production lines' shapes: 17w/8.37s, 12w/6.62s, 18w/8.62s.
A = _line("alpha", 17)
B = _line("bravo", 12)
C = _line("delta", 18)
LIVE = {A: 8.37, B: 6.62, C: 8.62}


async def _measure(
    texts: list[str],
    voices: list[str | None],
    fake: object,
    **kwargs: Any,
) -> ScriptMeasurement:
    return await measure_script_lines(
        _ctx(),
        texts=texts,
        voices=voices,
        prior=_prior(),
        speed=1.0,
        stage_name=_STAGE,
        client=_client(fake),
        **kwargs,
    )


# ── the whole point: the total is a sum of measurements ──────────────────────


@pytest.mark.asyncio
async def test_every_line_measured_totals_the_real_seconds() -> None:
    """THE contract. No rate is applied to any word count anywhere."""
    fake = _LineTTS(LIVE)

    m = await _measure([A, B, C], ["af_heart"] * 3, fake)

    assert m.per_line_seconds == (8.37, 6.62, 8.62)
    assert m.total_seconds == pytest.approx(23.61)
    assert m.measured == (True, True, True)
    assert m.failed_indices == ()
    assert m.extrapolated_indices == ()
    assert m.measured_count == 3
    assert m.basis == BASIS_MEASURED_ALL
    assert m.complete is True
    assert m.reason is None
    assert m.wpm == pytest.approx(47 / (23.61 / 60.0))
    assert script_measurement_degradation(m, stage=_STAGE) is None


@pytest.mark.asyncio
async def test_total_seconds_is_the_sum_of_the_per_line_seconds() -> None:
    """Pinned separately: a total that drifts from its own parts is undetectable."""
    fake = _LineTTS(LIVE)

    m = await _measure([A, B, C], ["af_heart"] * 3, fake)

    assert m.total_seconds == pytest.approx(math.fsum(m.per_line_seconds))
    assert m.total_seconds == pytest.approx(sum([8.37, 6.62, 8.62]))


@pytest.mark.asyncio
async def test_clips_are_stored_under_the_narrate_stage_name(
    stored: list[dict[str, Any]],
) -> None:
    """The economic argument: a clip measured here is the clip record will reuse."""
    fake = _LineTTS(LIVE)

    await _measure([A, B, C], ["af_heart"] * 3, fake)

    assert len(stored) == 3
    assert {call["stage_name"] for call in stored} == {_STAGE}
    assert {call["speed"] for call in stored} == {1.0}
    assert [call["text"] for call in stored] == [A, B, C]


# ── extrapolation ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_timed_out_line_is_extrapolated_at_its_own_voices_rate() -> None:
    """SAME-VOICE, not pooled: a cast's voices differ by more than the sampling error.

    ``narrator`` measures 17w/8.37s + 18w/8.62s = 35w/16.99s = 123.6 wpm. ``villain``
    measures 12w/6.62s = 108.8 wpm, and the pooled rate is 119.4 — three different
    answers, so a villain line filled at the pooled or the narrator rate is visibly
    wrong and this assertion cannot pass by accident.
    """
    hung = _line("echo", 20)
    fake = _LineTTS({**LIVE, hung: 99.0}, hang={hung})

    m = await _measure(
        [A, B, C, hung],
        ["narrator", "villain", "narrator", "villain"],
        fake,
        timeout_per_line=0.05,
    )

    villain_wpm = 12 / (6.62 / 60.0)
    assert m.basis == BASIS_MEASURED_PARTIAL
    assert m.complete is False
    assert m.reason is None
    assert m.failed_indices == (3,)
    assert m.extrapolated_indices == (3,)
    assert m.measured == (True, True, True, False)
    assert m.measured_count == 3
    assert m.per_line_seconds[3] == pytest.approx(20 * 60.0 / villain_wpm)
    # The three rates in play are far enough apart that the wrong one is detectable.
    pooled_wpm = 47 / (23.61 / 60.0)
    narrator_wpm = 35 / (16.99 / 60.0)
    assert m.per_line_seconds[3] != pytest.approx(20 * 60.0 / pooled_wpm, abs=0.2)
    assert m.per_line_seconds[3] != pytest.approx(20 * 60.0 / narrator_wpm, abs=0.2)
    assert m.total_seconds == pytest.approx(math.fsum(m.per_line_seconds))


@pytest.mark.asyncio
async def test_a_voice_with_no_measured_line_falls_through_to_pooled() -> None:
    """No same-voice measurement exists ⇒ pooled, never zero and never a crash."""
    orphan = _line("echo", 20)
    fake = _LineTTS(LIVE, hang={orphan})

    m = await _measure(
        [A, B, C, orphan],
        ["af_heart", "af_heart", "af_heart", "never_spoke"],
        fake,
        timeout_per_line=0.05,
    )

    pooled_wpm = 47 / (23.61 / 60.0)
    assert m.basis == BASIS_MEASURED_PARTIAL
    assert m.failed_indices == (3,)
    assert m.per_line_seconds[3] == pytest.approx(20 * 60.0 / pooled_wpm)
    assert m.per_line_seconds[3] > 0.0


@pytest.mark.asyncio
async def test_an_unusable_duration_is_a_failed_line_not_a_measurement() -> None:
    """A backend that answers 0.0s is a DIFFERENT fault from one that does not answer."""
    zero = _line("echo", 20)
    fake = _LineTTS({**LIVE, zero: 0.0})

    m = await _measure([A, B, C, zero], ["af_heart"] * 4, fake)

    assert m.failed_indices == (3,)
    assert m.measured_count == 3
    assert m.per_line_seconds[3] > 0.0


# ── refusals ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_too_few_measured_lines_is_refused_and_reportable() -> None:
    """One clip's rate is one clip's rate — scatter on a single voice is 94-158 wpm."""
    fake = _LineTTS({A: 8.37})  # B and C raise

    m = await _measure([A, B, C], ["af_heart"] * 3, fake)

    assert MIN_SAMPLE_LINES == 2, "the threshold this test is written against"
    assert m.measured_count == 1
    assert m.reason == SAMPLE_TOO_FEW_MEASURED
    assert m.basis == CALIBRATION_PASSAGE
    assert m.complete is False
    assert m.wpm == pytest.approx(175.2)
    assert m.measured == (False, False, False)
    assert m.extrapolated_indices == (0, 1, 2)
    assert m.failed_indices == (1, 2)
    # Every line, INCLUDING the one that measured, is prior-derived: the measurement
    # was refused as a whole, so no part of it may leak into the answer.
    assert m.per_line_seconds[0] == pytest.approx(17 * 60.0 / 175.2)
    assert m.total_seconds == pytest.approx(math.fsum(m.per_line_seconds))

    entry = script_measurement_degradation(m, stage=_STAGE)
    assert entry is not None
    assert entry["code"] == SPEECH_RATE_UNUSABLE
    assert SAMPLE_TOO_FEW_MEASURED in entry["detail"]


@pytest.mark.asyncio
async def test_a_single_line_script_that_measured_is_measured_not_refused() -> None:
    """Run 01a156a8-978b: a ONE-paragraph lesson synthesized fine and the too-few
    gate threw its real seconds away for the calibration prior — the stage judged
    ~56s "in band" and the lesson delivered 70.37s. With nothing to extrapolate the
    clip's seconds ARE the delivery.
    """
    fake = _LineTTS({A: 8.37})

    m = await _measure([A], ["af_heart"], fake)

    assert m.reason is None
    assert m.basis == BASIS_MEASURED_ALL
    assert m.per_line_seconds == (8.37,)
    assert m.total_seconds == pytest.approx(8.37)
    assert m.measured == (True,)
    assert m.wpm == pytest.approx(17 / (8.37 / 60.0))
    assert script_measurement_degradation(m, stage=_STAGE) is None


@pytest.mark.asyncio
async def test_an_implausible_pooled_rate_is_rejected() -> None:
    """A padded or truncated clip implies a rate outside the band — refuse it."""
    fake = _LineTTS({A: 0.4, B: 0.3, C: 0.4})  # ~4000 wpm

    m = await _measure([A, B, C], ["af_heart"] * 3, fake)

    assert m.reason == IMPLAUSIBLE_RATE
    assert m.measured_count == 3, "the backend ANSWERED — that is not a too-few failure"
    assert m.basis == CALIBRATION_PASSAGE
    assert m.wpm == pytest.approx(175.2)
    assert script_measurement_degradation(m, stage=_STAGE) is not None


@pytest.mark.asyncio
async def test_a_length_mismatch_falls_back_instead_of_raising() -> None:
    """A mis-zipped cast is a caller bug; it must not fail a paid run."""
    fake = _LineTTS(LIVE)

    m = await _measure([A, B, C], ["af_heart", "af_heart"], fake)

    assert m.reason == RESOLUTION_ERROR
    assert m.basis == CALIBRATION_PASSAGE
    assert m.failed_indices == ()
    assert len(m.per_line_seconds) == 3
    assert fake.calls == [], "nothing may be synthesized against a voice list that is wrong"
    assert script_measurement_degradation(m, stage=_STAGE) is not None


# ── blank lines ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_blank_line_costs_zero_seconds_and_zero_syntheses() -> None:
    """Blank is 0.0s by arithmetic. Synthesizing one buys a clip nobody hears AND
    occupies a slot behind the pod-wide gate that a real line needs."""
    fake = _LineTTS(LIVE)

    m = await _measure([A, "   ", B, "", C], ["af_heart"] * 5, fake)

    assert fake.texts == [A, B, C], "the blank lines never reached TTS"
    assert m.per_line_seconds == (8.37, 0.0, 6.62, 0.0, 8.62)
    assert m.measured == (True, True, True, True, True), "an exact 0.0s IS measured"
    assert m.basis == BASIS_MEASURED_ALL
    assert m.measured_count == 3
    assert m.total_seconds == pytest.approx(23.61)


# ── the timeout, and cancellation ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_per_line_timeout_grows_with_queue_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flat bound is a queue-depth test, not a timeout.

    The pod-wide gate admits ``TTS_MAX_CONCURRENT_SYNTHESIZE`` syntheses at a time, so
    the line at queue position ``p`` waits ``p // gate`` batches before its own
    synthesis starts. Recorded off the real ``wait_for`` call, not recomputed.
    """
    monkeypatch.setenv("TTS_MAX_CONCURRENT_SYNTHESIZE", "2")
    seen: list[float] = []
    real_wait_for = asyncio.wait_for

    async def _record(awaitable: Any, timeout: float | None = None) -> Any:
        seen.append(float(timeout or 0.0))
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr(speech_rate_module.asyncio, "wait_for", _record)

    texts = [_line("l", n) for n in (5, 6, 7, 8, 9)]
    fake = _LineTTS(dict.fromkeys(texts, 3.0))

    await _measure(texts, ["af_heart"] * 5, fake, timeout_per_line=10.0)

    # gate=2 ⇒ positions 0,1 | 2,3 | 4 ⇒ multipliers 1,1,2,2,3
    assert seen == [10.0, 10.0, 20.0, 20.0, 30.0]


@pytest.mark.asyncio
async def test_cancellation_propagates() -> None:
    """A cancelled run must stop synthesizing, not be told TTS misbehaved."""

    class _Cancelling:
        async def synthesize(self, **kwargs: Any) -> Any:
            await asyncio.sleep(0)
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await _measure([A, B, C], ["af_heart"] * 3, _Cancelling())


@pytest.mark.asyncio
async def test_the_timeout_counts_submitted_position_not_script_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Position is counted over the lines actually SUBMITTED — blanks never queue.

    A blank line occupies no slot behind the pod-wide gate, so it must not advance
    the queue position either. With gate=2 and blanks interleaved, the four spoken
    lines hold submitted positions 0,1,2,3 ⇒ multipliers 1,1,2,2. Counting the raw
    script index instead would hand them 0,2,4,5 ⇒ 1,2,3,3 — i.e. it would inflate
    the tail's timeout on a queue that is not that deep, and the two sequences are
    disjoint after the first element.
    """
    monkeypatch.setenv("TTS_MAX_CONCURRENT_SYNTHESIZE", "2")
    seen: list[float] = []
    real_wait_for = asyncio.wait_for

    async def _record(awaitable: Any, timeout: float | None = None) -> Any:
        seen.append(float(timeout or 0.0))
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr(speech_rate_module.asyncio, "wait_for", _record)

    spoken = [_line("l", n) for n in (5, 6, 7, 8)]
    # Script indices of the spoken lines: 0, 2, 4, 5. Submitted positions: 0, 1, 2, 3.
    texts = [spoken[0], "", spoken[1], "   ", spoken[2], spoken[3], ""]
    fake = _LineTTS(dict.fromkeys(spoken, 3.0))

    m = await _measure(texts, ["af_heart"] * 7, fake, timeout_per_line=10.0)

    assert fake.texts == spoken, "only the spoken lines were ever submitted"
    assert seen == [10.0, 10.0, 20.0, 20.0]
    # The index-based sequence, spelled out so the mutation it guards is unmistakable.
    assert seen != [10.0, 20.0, 30.0, 30.0]
    assert m.basis == BASIS_MEASURED_ALL


def test_a_garbage_tts_gate_size_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Junk in the env is the DEFAULT gate, never 1 and never a crash.

    ``1`` is the dangerous wrong answer rather than a conservative one: it turns the
    per-line bound into ``timeout_per_line * (1 + position)``, growing without limit
    over a script's length against a pod that is in fact running three syntheses at a
    time — a queue-depth test wearing a timeout's name.
    """
    assert speech_rate_module._DEFAULT_TTS_GATE == 3, "the default this test is written against"

    monkeypatch.delenv("TTS_MAX_CONCURRENT_SYNTHESIZE", raising=False)
    assert speech_rate_module._tts_gate_size() == 3

    for junk in ("abc", "", " ", "-5", "0", "2.5", "3,0", "None"):
        monkeypatch.setenv("TTS_MAX_CONCURRENT_SYNTHESIZE", junk)
        assert speech_rate_module._tts_gate_size() == 3, f"{junk!r} must fall back to the default"

    # A real setting is still honoured — the fallback must not swallow everything.
    for good, expected in (("1", 1), ("2", 2), ("16", 16)):
        monkeypatch.setenv("TTS_MAX_CONCURRENT_SYNTHESIZE", good)
        assert speech_rate_module._tts_gate_size() == expected


# ── a script with nothing to say ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_script_with_no_spoken_lines_is_measured_exactly_not_refused() -> None:
    """0 spoken lines is a COMPLETE measurement: every line really is 0.0s.

    Refusing it would stamp ``sample_too_few_measured`` and raise a degradation
    reading "0 of 0 spoken line(s) usable" — a signal firing on a correct result.
    """
    fake = _LineTTS(LIVE)

    m = await _measure(["", "   ", "\n\t"], ["af_heart"] * 3, fake)

    assert fake.calls == [], "there was nothing to synthesize"
    assert m.reason is None
    assert script_measurement_degradation(m, stage=_STAGE) is None
    assert m.basis == BASIS_MEASURED_ALL
    assert m.complete is True
    assert m.per_line_seconds == (0.0, 0.0, 0.0)
    assert m.measured == (True, True, True)
    assert m.total_seconds == 0.0
    assert m.measured_count == 0
    assert m.failed_indices == ()
    assert m.extrapolated_indices == ()


@pytest.mark.asyncio
async def test_an_empty_script_is_measured_exactly_not_refused() -> None:
    """Zero lines at all — the degenerate case of the same fact."""
    fake = _LineTTS(LIVE)

    m = await _measure([], [], fake)

    assert fake.calls == []
    assert m.reason is None
    assert script_measurement_degradation(m, stage=_STAGE) is None
    assert m.basis == BASIS_MEASURED_ALL
    assert m.per_line_seconds == ()
    assert m.total_seconds == 0.0


@pytest.mark.asyncio
async def test_some_spoken_lines_too_few_usable_is_still_refused() -> None:
    """The genuinely-too-few case is UNCHANGED by the all-blank shortcut."""
    fake = _LineTTS({A: 8.37})  # B raises

    m = await _measure([A, "", B, "  "], ["af_heart"] * 4, fake)

    assert m.measured_count == 1
    assert m.reason == SAMPLE_TOO_FEW_MEASURED
    assert m.basis == CALIBRATION_PASSAGE
    assert script_measurement_degradation(m, stage=_STAGE) is not None


# ── the recovery path is itself total ────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_non_iterable_texts_falls_back_instead_of_raising() -> None:
    """NEVER RAISES covers the RECOVERY path too.

    ``_prior_measurement`` iterates ``texts``, so a ``texts`` that is not a sequence
    escapes the outer handler as a TypeError raised from inside the handler itself.
    """
    fake = _LineTTS(LIVE)

    m = await measure_script_lines(
        _ctx(),
        texts=cast("list[str]", 5),
        voices=["af_heart"],
        prior=_prior(),
        speed=1.0,
        stage_name=_STAGE,
        client=_client(fake),
    )

    assert m.reason == RESOLUTION_ERROR
    assert m.basis == CALIBRATION_PASSAGE
    assert m.wpm == pytest.approx(175.2)
    assert m.per_line_seconds == ()
    assert m.total_seconds == 0.0
    assert fake.calls == []
    assert script_measurement_degradation(m, stage=_STAGE) is not None


@pytest.mark.asyncio
async def test_a_non_iterable_voices_falls_back_instead_of_raising() -> None:
    """Same fact for ``voices`` — and the per-line prior estimate survives it."""
    fake = _LineTTS(LIVE)

    m = await measure_script_lines(
        _ctx(),
        texts=[A, B, C],
        voices=cast("list[str | None]", object()),
        prior=_prior(),
        speed=1.0,
        stage_name=_STAGE,
        client=_client(fake),
    )

    assert m.reason == RESOLUTION_ERROR
    assert m.basis == CALIBRATION_PASSAGE
    assert len(m.per_line_seconds) == 3
    assert m.per_line_seconds[0] == pytest.approx(17 * 60.0 / 175.2)
    assert fake.calls == []


@pytest.mark.asyncio
async def test_a_texts_that_raises_while_iterating_falls_back_instead_of_raising() -> None:
    """An iterable that blows up mid-iteration is the same class of caller bug."""

    def _explode() -> Any:
        yield A
        raise RuntimeError("bad generator")

    fake = _LineTTS(LIVE)

    m = await measure_script_lines(
        _ctx(),
        texts=cast("list[str]", _explode()),
        voices=["af_heart", "af_heart"],
        prior=_prior(),
        speed=1.0,
        stage_name=_STAGE,
        client=_client(fake),
    )

    assert m.reason == RESOLUTION_ERROR
    assert fake.calls == []


@pytest.mark.asyncio
async def test_an_outright_failure_returns_the_prior_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NEVER RAISES. A bug in here must not cost a paid run its narration."""

    async def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("bang")

    monkeypatch.setattr(speech_rate_module, "_measure_script_lines", _boom)
    fake = _LineTTS(LIVE)

    m = await _measure([A, B, C], ["af_heart"] * 3, fake)

    assert m.reason == RESOLUTION_ERROR
    assert m.basis == CALIBRATION_PASSAGE
    assert m.wpm == pytest.approx(175.2)
