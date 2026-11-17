"""The beat grid must survive the analyzer's real wire types.

Regression cover for the defect that kept `action_reel` from EVER producing a
beat-cut reel: `VideoAnalyzerAudioGrid.meter` was declared `str`, but
video_analyzer_service sends an int (`src/api/schemas.py:449`, "Beats per bar
used for the downbeat inference"). Pydantic rejected the whole grid, and
`music_select` catches ValidationError and degrades, so the only symptom was
`beats: []` plus a note — a silent fallback to even spacing.

`_LIVE_GRID` below is a verbatim capture of a real 200 response from
`POST /api/analyze/audio` in-cluster (bed `corporate_uplift`), so this test
fails if the wire contract moves again.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.clients.schemas.video_analyzer import VideoAnalyzerAudioGrid

#: Verbatim live capture — shortened lists, untouched types.
_LIVE_GRID: dict[str, object] = {
    "duration_s": 58.0,
    "bpm": 132.512,
    "bpm_confidence": 0.9408,
    "beats": [14.407982, 14.860771, 15.30195, 15.766349],
    "downbeats": [15.30195, 17.089887, 18.889433, 20.66576],
    "onsets": [0.08127, 0.301859, 0.754649],
    "meter": 4,
    "grid_source": "librosa.beat_track+onset_phase",
}


def test_the_live_analyzer_response_parses() -> None:
    """The exact body the analyzer returns must validate — this is the bug."""
    grid = VideoAnalyzerAudioGrid.model_validate(_LIVE_GRID)

    assert grid.meter == "4"
    assert grid.bpm == pytest.approx(132.512)
    assert grid.grid_source == "librosa.beat_track+onset_phase"
    # The payload that actually matters downstream survived intact.
    assert len(grid.beats) == 4
    assert len(grid.downbeats) == 4


@pytest.mark.parametrize(
    ("wire", "expected"),
    [(4, "4")],
)
def test_meter_accepts_whatever_the_analyzer_calls_a_meter(wire: object, expected: str) -> None:
    """Coerced to a string, never raising: a new meter encoding must not be able
    to hard-fail a running pipeline.

    Built through ``model_validate`` rather than the constructor because the
    field is statically ``str`` — feeding it the wire's ``int`` is precisely the
    point of the test, and only the validation path models what the client does
    with a real response body.
    """
    grid = VideoAnalyzerAudioGrid.model_validate({"duration_s": 1.0, "meter": wire})

    assert grid.meter == expected


def test_meter_defaults_to_empty_when_absent() -> None:
    assert VideoAnalyzerAudioGrid(duration_s=1.0).meter == ""


def test_duration_is_still_required() -> None:
    """The coercion must not have loosened the one genuinely required field."""
    with pytest.raises(ValidationError):
        VideoAnalyzerAudioGrid.model_validate({"meter": 4})
