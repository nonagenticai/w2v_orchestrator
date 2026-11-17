"""The blank hard-fail gate must know how many frames its verdict rests on.

Analyzer PR #122 started emitting ``frames_planned``/``frames_analyzed`` on
``QualityMetricsResponse``, documented explicitly as "the orchestrator's blank
hard-fail gate reads this body". The orchestrator's client schema
(``src/clients/schemas/video_analyzer.py``) omitted both, and pydantic's
``extra="ignore"`` dropped them silently — so the producer's fix was INERT and
``mean_sharpness`` remained a number with no denominator.

Two customer-visible consequences, both drawn from n=1 and both reading as
confident:

  * a clip with ONE decodable sharp frame passed the gate IDENTICALLY to a
    fully clean clip, and
  * a single DARK frame HARD-FAILED a paid render.

These tests assert BEHAVIOUR, not parsing — parsing was never the failure. Each
pair holds ``mean_sharpness`` constant and varies ONLY the sample size, so the
only thing that can change the verdict is the fix.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.clients.schemas.video_analyzer import (
    VideoAnalyzerJobResult,
    VideoAnalyzerQualityMetrics,
)
from src.config import PipelineSettings
from src.services.stages.demo_video.record import DemoVideoRecordStage

# Well above the 3.0 degrade band and the 1.0 hard-fail floor: a real UI.
_SHARP = 42.0
# Below the effective hard-fail threshold (min(1.0, 3.0) == 1.0): a blank render.
_BLANK = 0.0


def _quality(*, mean_sharpness: float, **extra: Any) -> dict[str, Any]:
    return {
        "mean_sharpness": mean_sharpness,
        "mean_brightness": 100.0,
        "motion_score": 0.5,
        "dominant_colors": ["#101010"],
        "overall_score": 0.8,
        **extra,
    }


def _analyzer(quality: dict[str, Any]) -> AsyncMock:
    """An analyzer whose own criteria PASS, carrying the given quality body.

    Validated through the real response model, so a drift on either side of the
    contract fails here rather than in production.
    """
    analyzer = AsyncMock()
    analyzer.start_analysis.return_value = "va-job"
    analyzer.poll_analysis.return_value = {"verification_passed": True}
    analyzer.get_result.return_value = VideoAnalyzerJobResult.model_validate(
        {
            "job_id": "va-job",
            "status": "completed",
            "stage": "artifact_gen",
            "progress": 1.0,
            "quality_metrics": quality,
        }
    )
    return analyzer


async def _qa(quality: dict[str, Any]):
    stage = DemoVideoRecordStage(client=AsyncMock(), analyzer=_analyzer(quality))
    return await stage._qa_clip(
        clip_key="clip-1",
        video_url="https://example.com/clip-1.webm",
        settings=PipelineSettings(),
    )


# ---------------------------------------------------------------------------
# The contract the consumer was discarding
# ---------------------------------------------------------------------------


def test_absent_sample_size_is_none_not_a_measured_zero() -> None:
    """``None``, not 0 — a consumer must tell UNMEASURED from a measured zero."""
    metrics = VideoAnalyzerQualityMetrics.model_validate(_quality(mean_sharpness=_SHARP))
    assert metrics.frames_planned is None
    assert metrics.frames_analyzed is None


def test_a_pre_122_analyzer_does_not_stand_the_blank_gate_down() -> None:
    """DELIBERATE BOUNDARY, pinned so it cannot be "tidied" into a fleet-wide outage.

    Absent fields mean an analyzer without #122, i.e. an UNKNOWN denominator — not a
    known-bad one. Treating that as insufficient would silently disable the blank
    hard-fail for every clip such an analyzer served, removing the protection against
    shipping a 100% black deliverable. That pre-existing gap is left exactly as it
    was; this change closes only the verdict we can SEE is unsupported.
    """
    metrics = VideoAnalyzerQualityMetrics.model_validate(_quality(mean_sharpness=_SHARP))
    assert metrics.sample_is_known_insufficient() is False


@pytest.mark.parametrize(
    ("planned", "analyzed", "known_insufficient"),
    [(10, 10, False)],
)
def test_sufficiency_mirrors_the_analyzers_own_rule(
    planned: int, analyzed: int, known_insufficient: bool
) -> None:
    """``frames_analyzed < min(MIN_FRAMES_FOR_QUALITY, frames_planned)``."""
    metrics = VideoAnalyzerQualityMetrics.model_validate(
        _quality(mean_sharpness=_SHARP, frames_planned=planned, frames_analyzed=analyzed)
    )
    assert metrics.sample_is_known_insufficient() is known_insufficient


# ---------------------------------------------------------------------------
# BEHAVIOUR: partial must be distinguishable from complete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_sharp_frame_no_longer_passes_as_a_clean_clip() -> None:
    """Defect half 1. Same ``mean_sharpness``; only the sample size differs."""
    complete = await _qa(_quality(mean_sharpness=_SHARP, frames_planned=10, frames_analyzed=10))
    partial = await _qa(_quality(mean_sharpness=_SHARP, frames_planned=10, frames_analyzed=1))

    # A complete analysis reaches a verdict: measured, and the clip is fine.
    assert complete.blank_gate_ran is True
    assert complete.blank is False
    assert complete.failure is None

    # A partial one does NOT — it is routed to the loud unmeasured path instead of
    # silently reading as "measured and fine".
    assert partial.blank_gate_ran is False, (
        "a verdict drawn from 1 of 10 frames still reads as a measured clean clip"
    )
    assert partial.blank is False
    assert complete.blank_gate_ran != partial.blank_gate_ran


@pytest.mark.asyncio
async def test_one_dark_frame_no_longer_hard_fails_a_paid_render() -> None:
    """Defect half 2. A blank verdict from n=1 destroyed paid work."""
    complete = await _qa(_quality(mean_sharpness=_BLANK, frames_planned=10, frames_analyzed=10))
    partial = await _qa(_quality(mean_sharpness=_BLANK, frames_planned=10, frames_analyzed=1))

    # Genuinely blank, fully measured ⇒ still a hard fail. The gate must keep working.
    assert complete.blank is True
    assert complete.blank_gate_ran is True
    assert complete.failure is not None

    # One dark frame out of ten is not evidence the clip is blank.
    assert partial.blank is False, (
        "a single dark frame still hard-fails a paid render on n=1 evidence"
    )
    assert partial.blank_gate_ran is False
    assert partial.failure is None
    assert complete.blank != partial.blank


@pytest.mark.asyncio
async def test_a_short_clip_is_not_punished_for_being_short() -> None:
    """``min(5, frames_planned)`` — a 3-frame clip that analyzed all 3 IS measured."""
    qa = await _qa(_quality(mean_sharpness=_BLANK, frames_planned=3, frames_analyzed=3))
    assert qa.blank_gate_ran is True
    assert qa.blank is True


@pytest.mark.asyncio
async def test_the_motion_curve_survives_an_insufficient_sample() -> None:
    """The curve feeds the cold-open, not a hard-fail; a short curve is usable."""
    stage = DemoVideoRecordStage(
        client=AsyncMock(),
        analyzer=_analyzer(
            _quality(
                mean_sharpness=_SHARP,
                frames_planned=10,
                frames_analyzed=1,
                motion_timeline=[{"t": 0.5, "motion": 0.25}],
            )
        ),
    )
    timeline, sharpness = await stage._fetch_quality("va-job", "clip-1")
    assert sharpness is None, "an insufficient sample must not yield a sharpness verdict"
    assert [s.t for s in timeline] == [0.5]
