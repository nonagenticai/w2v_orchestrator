"""An AUTHORED ``direction.timing.wpm.natural`` still WINS — but no longer silently.

An author who writes a rate means it, so the authored value governs the whole word
budget and the measurement is not substituted for it. What was wrong is that it won
INVISIBLY: a stale constant left in ``direction.timing.wpm.natural`` keeps budgeting
every beat at a rate the run's voice does not speak at, for as long as nobody notices,
and the run carried no record that a better number existed. Narration that "weirdly
slows down" is exactly that shape.

So the stage now probes anyway and puts the override on ``notes`` — which
``runs/_shared.py`` auto-translates into a ``degraded`` run — when the two disagree by
more than :data:`_AUTHORED_RATE_NOTE_FRACTION`.

⛔ NOT a change of precedence. Every assertion below also pins that the authored value
is the one actually budgeted against.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pytest

import src.services.stages.demo_video.reconcile_timeline as reconcile_timeline
from src.clients.tts_client import TTSClient
from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.reconcile_timeline import ReconcileTimelineStage
from src.services.stages.shared.speech_rate import CALIBRATION_WORDS, clear_measured_wpm_cache

_LINE = "This beat introduces the product in one short line."
_AUTHORED = 150.0


class _FakeTTS:
    """Answers the calibration probe at ``wpm``; ``None`` means it cannot answer."""

    def __init__(self, wpm: float | None) -> None:
        self.wpm = wpm
        self.calls: list[dict[str, Any]] = []

    async def synthesize(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.wpm is None:
            raise RuntimeError("tts down")
        return type("_Synth", (), {"duration_seconds": CALIBRATION_WORDS * 60.0 / self.wpm})()


@pytest.fixture(autouse=True)
def _passage_rate_is_the_prior(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the passage->narration ratio at 1.0: these tests prove the measured-rate
    PLUMBING with arithmetic written at the raw passage rate. The ratio itself is
    pinned by ``tests/services/stages/shared/test_speech_rate_narration_prior.py``."""
    from src.services.stages.shared import speech_rate

    monkeypatch.setattr(speech_rate, "PASSAGE_TO_NARRATION_RATIO", 1.0)


@pytest.fixture
def resolutions(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every ``_resolve_timing_config`` call without changing its behaviour."""
    calls: list[dict[str, Any]] = []
    real = reconcile_timeline._resolve_timing_config

    def _spy(ctx: RunContext, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return real(ctx, **kwargs)

    monkeypatch.setattr(reconcile_timeline, "_resolve_timing_config", _spy)
    return calls


@pytest.fixture(autouse=True)
def _clean_cache() -> Any:
    clear_measured_wpm_cache()
    yield
    clear_measured_wpm_cache()


def _ctx(natural: float | None = _AUTHORED) -> RunContext:
    wpm: dict[str, Any] = {"min": 130, "max": 170}
    if natural is not None:
        wpm["natural"] = natural
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "direction": {"timing": {"mode": "auto", "wpm": wpm}},
            "deliverable": {"target_duration_seconds": 8.0},
            "voice": {"voice_id": "af_heart"},
        },
        prior_run_outputs={
            "demo_video_gen": {
                "lines": [
                    {"id": "clip-1", "text": _LINE, "seconds": 3.0, "clip": None},
                    {"id": "clip-2", "text": _LINE, "seconds": 3.0, "clip": None},
                ],
                "bdd_job_id": "",
                "features_generated": 0,
                "package_url": "",
            },
            "demo_video_record": {"clips": []},
        },
    )


async def _notes(wpm: float | None, natural: float | None = _AUTHORED) -> list[str]:
    fake = _FakeTTS(wpm)
    stage = ReconcileTimelineStage(tts_client=cast("TTSClient", fake))
    out = await stage.handle(stage.input_model(), _ctx(natural))
    return [n for n in out.notes if "AUTHORED" in n]


@pytest.mark.asyncio
async def test_a_materially_different_measurement_is_reported() -> None:
    """192 wpm measured against an authored 150 — 28% apart. The run must say so."""
    notes = await _notes(192.0)

    assert len(notes) == 1
    note = notes[0]
    assert "150.0 wpm" in note  # the authored rate, which governed
    assert "192.0 wpm" in note  # the measurement it overrode
    assert "OVERRODE" in note


@pytest.mark.asyncio
async def test_the_authored_rate_still_governs_the_budget(
    resolutions: list[dict[str, Any]],
) -> None:
    """⛔ Precedence is UNCHANGED. The note is observability, not a new lever.

    Asserted on the CALLS to ``_resolve_timing_config``, like the measured-sentinel
    suite: the solver water-fills every elastic beat to the target, so the stamped
    ``seconds`` is a poor witness to which rate was budgeted at. A single call with no
    ``default_natural`` is the fact under test — the measurement never entered the
    config at all.
    """
    fake = _FakeTTS(192.0)
    stage = ReconcileTimelineStage(tts_client=cast("TTSClient", fake))

    await stage.handle(stage.input_model(), _ctx())

    assert resolutions == [{}]
    # The probe DID run (that is what makes the note possible) — it just did not win.
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_a_measurement_that_agrees_produces_no_note() -> None:
    """A note on every authored run is a note nobody reads. 153 wpm is 2% off the
    authored 150 — inside the threshold, and too small to move a beat out of the
    conformance band on its own."""
    assert await _notes(153.0) == []


@pytest.mark.asyncio
async def test_a_probe_that_cannot_answer_produces_no_note() -> None:
    """ "We have nothing to compare against" is not evidence of a bad authored rate."""
    assert await _notes(None) == []


@pytest.mark.asyncio
async def test_a_run_with_no_authored_rate_takes_the_measurement_and_says_nothing(
    resolutions: list[dict[str, Any]],
) -> None:
    """The control: without an authored constant there is no override to report, and
    the measured rate IS re-resolved into the config — the opposite branch, proving the
    assertion above is about precedence and not about the probe merely being skipped."""
    fake = _FakeTTS(192.0)
    stage = ReconcileTimelineStage(tts_client=cast("TTSClient", fake))

    out = await stage.handle(stage.input_model(), _ctx(natural=None))

    assert [n for n in out.notes if "AUTHORED" in n] == []
    assert resolutions == [{}, {"default_natural": pytest.approx(192.0)}]
