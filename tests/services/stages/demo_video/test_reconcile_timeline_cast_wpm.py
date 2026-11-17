"""``reconcile_timeline`` budgets against the CAST, not against one voice.

``demo_video_narrate`` casts every line through ``voice_for_speaker`` (the ``speaker``
slot ``demo_video_gen`` stamps), while this stage used to size every word budget from a
single ``resolve_natural_wpm_detail`` probe of ``config_snapshot["voice"]["voice_id"]``
— a voice that, on a dialogue run, may never speak a word. Measured live: ``am_michael``
156.60 wpm against the default ``af_heart`` 175.24 wpm, a ~12% slower cast. On the lesson
stage the identical gap hid a +23% overrun behind a +7.6% estimate, the verdict came back
``ok``, and the corrective machinery was never reached.

Contracts pinned here:

* a cast containing a SLOWER voice budgets MORE seconds for the same words than the
  single-rate path does;
* a single-voice / dialogue-off run is untouched — one probe, the same floor, no
  degradation;
* a voice whose probe cannot answer contributes at the caller's default AND records
  ``SPEECH_RATE_UNUSABLE`` naming that voice — never a silent substitution;
* a genuine 150.0 measurement is a MEASUREMENT: the code branches on ``measured`` /
  ``assumed``, never on ``wpm != DEFAULT_WPM_NATURAL``.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pytest

from src.clients.tts_client import TTSClient
from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.reconcile_timeline import (
    ReconcileTimelineStage,
    _rescale_band,
)
from src.services.stages.primitives.dialogue import DEFAULT_CAST
from src.services.stages.shared.degradation import SPEECH_RATE_UNUSABLE, Degradation
from src.services.stages.shared.speech_rate import clear_measured_wpm_cache
from src.services.stages.shared.time_budget import DEFAULT_WPM_BAND, DEFAULT_WPM_NATURAL

# The two live-measured rates from the lesson investigation. `af_heart` is the
# configured/backend-default voice; `am_michael` is the one the cast actually puts on
# half the lines, and it is ~12% slower.
_AF_HEART_WPM = 175.24
_AM_MICHAEL_WPM = 156.60

# 9 words each. Question mark / statement is irrelevant here — demo_video_gen stamps the
# `speaker` slot explicitly and `voice_for_speaker` reads only that.
_LINE_A = "This beat introduces the product in one short line."
_LINE_B = "The answer covers the second half of that story."
_WORDS_A = len(_LINE_A.split())
_WORDS_B = len(_LINE_B.split())


class _CastTTS:
    """A backend that renders the calibration passage at a DIFFERENT rate per voice."""

    def __init__(
        self,
        *,
        wpm_by_voice: dict[str | None, float],
        errors: dict[str | None, Exception] | None = None,
    ) -> None:
        self.wpm_by_voice = wpm_by_voice
        self.errors = errors or {}
        self.calls: list[str | None] = []

    async def synthesize(self, **kwargs: Any) -> Any:
        voice = kwargs.get("voice")
        self.calls.append(voice)
        error = self.errors.get(voice)
        if error is not None:
            raise error
        # Word-count-proportional, not a fixed calibration-passage duration: Phase B
        # (see `speech_rate.py::resolve_script_sample_wpm`) samples THIS spine's own
        # lines, which are a different length than the calibration passage, so a fixed
        # duration would measure a nonsense (implausible) rate on those calls and this
        # fixture is meant to model one real voice answering ANY text at its own wpm.
        words = len(str(kwargs.get("text", "")).split())
        return type("_Synth", (), {"duration_seconds": words * 60.0 / self.wpm_by_voice[voice]})()


def _client(fake: _CastTTS) -> TTSClient:
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


def _ctx(*, dialogue: bool) -> RunContext:
    """Two narrated beats, speakers a/b. ``dialogue`` gates ``voice_for_speaker``.

    ``demo_video_record`` returns no clips, so nothing is footage-pinned and every beat's
    demand is pure word arithmetic — which is exactly the quantity the speaking rate
    sizes.
    """
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "direction": {"timing": {"mode": "auto"}},
            "deliverable": {"target_duration_seconds": 8.0},
            "voice": {"voice_id": "af_heart"},
            "brief": {"dialogue": {"enabled": dialogue}},
        },
        prior_run_outputs={
            "demo_video_gen": {
                "lines": [
                    {"id": "clip-1", "text": _LINE_A, "seconds": 3.0, "clip": None, "speaker": "a"},
                    {"id": "clip-2", "text": _LINE_B, "seconds": 3.0, "clip": None, "speaker": "b"},
                ],
                "bdd_job_id": "",
                "features_generated": 0,
                "package_url": "",
            },
            "demo_video_record": {"clips": []},
        },
    )


def _floor_at(natural: float, words: int) -> float:
    """``collect_demands``' ``min`` for ``words``: the band's TOP, rescaled to ``natural``."""
    return words / _rescale_band(DEFAULT_WPM_BAND, natural)[1] * 60.0


def _blended(pairs: list[tuple[int, float]]) -> float:
    """The word-weighted rate: ``Σwords / (Σ(words_i / wpm_i))``."""
    seconds = sum(words / wpm * 60.0 for words, wpm in pairs)
    return sum(words for words, _ in pairs) / (seconds / 60.0)


def _unusable(degradations: list[Degradation]) -> list[Degradation]:
    return [d for d in degradations if d.get("code") == SPEECH_RATE_UNUSABLE]


# ── the cast budget is LONGER than the single-rate budget ────────────────────


@pytest.mark.asyncio
async def test_cast_with_a_slower_voice_budgets_more_seconds_than_one_rate() -> None:
    """The defect, stated as arithmetic.

    Both spines draw the same words and probe the same backend. The only difference is
    whether the budget is taken at the configured ``af_heart`` alone or at the cast that
    will really speak, half of which is ``am_michael`` at 156.60 wpm. The cast rate is
    the word-weighted blend, and the slower it is the MORE seconds the same words need.
    """
    rates: dict[str | None, float] = {"af_heart": _AF_HEART_WPM, "am_michael": _AM_MICHAEL_WPM}
    single = ReconcileTimelineStage(tts_client=_client(_CastTTS(wpm_by_voice=rates)))
    single_out = await single.handle(single.input_model(), _ctx(dialogue=False))

    clear_measured_wpm_cache()
    cast_probe = _CastTTS(wpm_by_voice=rates)
    casted = ReconcileTimelineStage(tts_client=_client(cast_probe))
    cast_out = await casted.handle(casted.input_model(), _ctx(dialogue=True))

    # The cast IS two distinct voices — the premise of the whole test.
    assert set(cast_probe.calls) == {DEFAULT_CAST["a"], DEFAULT_CAST["b"]}
    blended = _blended([(_WORDS_A, _AF_HEART_WPM), (_WORDS_B, _AM_MICHAEL_WPM)])
    assert blended < _AF_HEART_WPM

    assert float(single_out.lines[0].min_seconds or 0.0) == pytest.approx(
        _floor_at(_AF_HEART_WPM, _WORDS_A), rel=1e-9
    )
    assert float(cast_out.lines[0].min_seconds or 0.0) == pytest.approx(
        _floor_at(blended, _WORDS_A), rel=1e-9
    )
    # The point: a slower cast demands strictly more time for identical copy.
    assert float(cast_out.lines[0].min_seconds or 0.0) > float(
        single_out.lines[0].min_seconds or 0.0
    )
    # Nothing was assumed — both voices answered.
    assert _unusable(cast_out.degradations) == []


# ── single-voice runs are untouched ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_voice_run_probes_once_and_is_unchanged() -> None:
    """Dialogue off ⇒ every beat resolves to ``voice.voice_id`` ⇒ the old path exactly.

    One probe (not one per beat, and not the cast's two), the floor the single measured
    rate implies, and no degradation of any kind.
    """
    probe = _CastTTS(wpm_by_voice={"af_heart": _AF_HEART_WPM})
    stage = ReconcileTimelineStage(tts_client=_client(probe))

    out = await stage.handle(stage.input_model(), _ctx(dialogue=False))

    assert probe.calls == ["af_heart"]
    assert float(out.lines[0].min_seconds or 0.0) == pytest.approx(
        _floor_at(_AF_HEART_WPM, _WORDS_A), rel=1e-9
    )
    assert _unusable(out.degradations) == []


# ── an unprobeable voice degrades HONESTLY ───────────────────────────────────


@pytest.mark.asyncio
async def test_an_unprobeable_cast_voice_degrades_at_the_default_and_names_itself() -> None:
    """``am_michael`` will not answer: it contributes at 150.0 and SAYS SO.

    The substitution itself is the pre-existing behaviour (the stage has always fallen
    back to ``DEFAULT_WPM_NATURAL``); what must never happen again is that it is silent.
    The degradation names the voice, so "which runs were budgeted blind, and on whom?"
    is a query rather than an archaeology project.
    """
    probe = _CastTTS(
        wpm_by_voice={"af_heart": _AF_HEART_WPM},
        errors={"am_michael": RuntimeError("tts down")},
    )
    stage = ReconcileTimelineStage(tts_client=_client(probe))

    out = await stage.handle(stage.input_model(), _ctx(dialogue=True))

    flagged = _unusable(out.degradations)
    assert len(flagged) == 1
    assert "am_michael" in flagged[0]["detail"]
    assert "af_heart" not in flagged[0]["detail"]
    # The unmeasurable half contributes at the caller's default, not at af_heart's rate.
    blended = _blended([(_WORDS_A, _AF_HEART_WPM), (_WORDS_B, DEFAULT_WPM_NATURAL)])
    assert float(out.lines[0].min_seconds or 0.0) == pytest.approx(
        _floor_at(blended, _WORDS_A), rel=1e-9
    )


# ── a genuine 150.0 is a MEASUREMENT, not a sentinel ─────────────────────────


@pytest.mark.asyncio
async def test_a_genuine_150_measurement_is_not_mistaken_for_a_fallback() -> None:
    """``af_heart`` really does speak at exactly ``DEFAULT_WPM_NATURAL``.

    The fallback constant and a real measurement are the same Python float, so any code
    that asks "is this rate the default?" instead of "did anybody measure it?" files this
    draw as a probe failure. Here that would raise a degradation for a voice that
    answered — ``speech_rate_fallback_degradation`` refuses a measured rate outright —
    and it would also discard the 150.0 from the blend.
    """
    probe = _CastTTS(wpm_by_voice={"af_heart": DEFAULT_WPM_NATURAL, "am_michael": _AM_MICHAEL_WPM})
    stage = ReconcileTimelineStage(tts_client=_client(probe))

    out = await stage.handle(stage.input_model(), _ctx(dialogue=True))

    assert _unusable(out.degradations) == []
    blended = _blended([(_WORDS_A, DEFAULT_WPM_NATURAL), (_WORDS_B, _AM_MICHAEL_WPM)])
    assert float(out.lines[0].min_seconds or 0.0) == pytest.approx(
        _floor_at(blended, _WORDS_A), rel=1e-9
    )
