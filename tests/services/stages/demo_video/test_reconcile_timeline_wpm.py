"""The MEASURED speaking rate replaces the hardcoded 150 wpm word budget.

``DEFAULT_WPM_NATURAL = 150`` models how fast the voice talks; the live kokoro voice
measures ~94, so every word budget derived from 150 is ~60% too generous and the film
overruns. ``ReconcileTimelineStage`` now measures the rate once per voice and uses it
as the DEFAULT natural rate.

Contracts pinned here:

* a successful probe moves the resolved ``wpm_natural`` (and therefore the per-beat
  seconds demands) off 150;
* the band scales with it, so ``min <= ideal <= max`` still brackets the measurement;
* a probe that cannot answer leaves the stage byte-identical to main — 150.0 EXACTLY;
* an authored ``direction.timing.wpm.natural`` beats the measurement and is not even
  probed for.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pytest

from src.clients.tts_client import TTSClient
from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.reconcile_timeline import (
    ReconcileTimelineStage,
)
from src.services.stages.shared.speech_rate import clear_measured_wpm_cache
from src.services.stages.shared.time_budget import DEFAULT_WPM_BAND, DEFAULT_WPM_NATURAL

_MEASURED_WPM = 94.0

# 9 words. At 150 wpm its ideal is 3.60s; at the measured 94 it is 5.74s.
_SHORT = "This beat introduces the product in one short line."
_WORDS = len(_SHORT.split())


class _FakeTTS:
    """Answers ANY synthesis (the Phase-A calibration passage AND a Phase-B script
    sample) at the SAME underlying voice rate, word-count-proportional — so the two
    phases measure the one real voice this fixture models, exactly as the live
    tts_backend would, rather than a duration pinned to one text's word count."""

    def __init__(self, *, wpm: float = _MEASURED_WPM, error: Exception | None = None):
        self.wpm = wpm
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def synthesize(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        words = len(str(kwargs.get("text", "")).split())
        return type("_Synth", (), {"duration_seconds": words * 60.0 / self.wpm})()


def _client(fake: _FakeTTS) -> TTSClient:
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


def _ctx(timing: dict[str, Any]) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "direction": {"timing": timing},
            # 8s: inside the two beats' fillable band at 150 wpm ([6.35s, 8.31s]), so
            # the stage runs its normal course. At the measured 94 the same words need
            # more room than that and the timeline is flagged infeasible — a NOTE, not
            # a failure, and irrelevant to what these tests assert (the per-beat floor).
            "deliverable": {"target_duration_seconds": 8.0},
            "voice": {"voice_id": "af_heart"},
        },
        prior_run_outputs={
            "demo_video_gen": {
                "lines": [
                    {"id": "clip-1", "text": _SHORT, "seconds": 3.0, "clip": None},
                    {"id": "clip-2", "text": _SHORT, "seconds": 3.0, "clip": None},
                ],
                "bdd_job_id": "",
                "features_generated": 0,
                "package_url": "",
            },
            "demo_video_record": {"clips": []},
        },
    )


async def _min_seconds_of_first_beat(stage: ReconcileTimelineStage, ctx: RunContext) -> float:
    out = await stage.handle(stage.input_model(), ctx)
    return float(out.lines[0].min_seconds or 0.0)


# `collect_demands` computes min = words / wpm_max * 60, so the FLOOR is set by the
# band's top. These are the two floors the stage must land on.
_FLOOR_AT_150 = _WORDS / DEFAULT_WPM_BAND[1] * 60.0
_FLOOR_AT_MEASURED = _WORDS / (DEFAULT_WPM_BAND[1] * _MEASURED_WPM / DEFAULT_WPM_NATURAL) * 60.0


# ── a successful probe moves the budget ──────────────────────────────────────


@pytest.mark.asyncio
async def test_measured_rate_replaces_the_hardcoded_default() -> None:
    fake = _FakeTTS()
    stage = ReconcileTimelineStage(tts_client=_client(fake))

    floor = await _min_seconds_of_first_beat(stage, _ctx({"mode": "auto"}))

    assert floor == pytest.approx(_FLOOR_AT_MEASURED, rel=1e-6)
    # The slower the voice, the MORE seconds the same words need.
    assert floor > _FLOOR_AT_150
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_probe_failure_falls_back_to_150_exactly() -> None:
    fake = _FakeTTS(error=RuntimeError("tts down"))
    stage = ReconcileTimelineStage(tts_client=_client(fake))

    floor = await _min_seconds_of_first_beat(stage, _ctx({"mode": "auto"}))

    assert floor == pytest.approx(_FLOOR_AT_150, rel=1e-9)


@pytest.mark.asyncio
async def test_authored_natural_beats_the_measurement() -> None:
    """An author who states a rate means it: the authored value budgets the timeline and
    the measurement never enters the config.

    ⚠️ THE PROBE IS NO LONGER SKIPPED, and this test used to assert that it was. Winning
    invisibly is how a stale authored constant kept budgeting every beat at a rate the
    voice does not speak at, for as long as nobody looked — the narration-slows-down
    shape. The stage now measures anyway and reports the override on ``notes`` when the
    two materially disagree (see
    ``test_reconcile_timeline_authored_rate_override.py``). PRECEDENCE is unchanged,
    which is what the floor below pins; the cost is bounded because
    ``measure_natural_wpm`` memoises per (provider, voice) for the life of the process.
    """
    fake = _FakeTTS()
    stage = ReconcileTimelineStage(tts_client=_client(fake))
    ctx = _ctx({"mode": "auto", "wpm": {"min": 130, "natural": 150, "max": 170}})

    floor = await _min_seconds_of_first_beat(stage, ctx)

    assert floor == pytest.approx(_FLOOR_AT_150, rel=1e-9)
    # Measured, then OVERRIDDEN — not measured-and-used, and not skipped.
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_a_skipping_run_never_probes() -> None:
    from src.services.stages.base import StageSkippedError

    fake = _FakeTTS()
    stage = ReconcileTimelineStage(tts_client=_client(fake))

    with pytest.raises(StageSkippedError):
        await stage.handle(stage.input_model(), _ctx({"mode": "off"}))

    assert fake.calls == []


# ── the pure resolution helpers ──────────────────────────────────────────────
