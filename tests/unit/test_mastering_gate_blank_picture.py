"""`_blank_picture` must not treat one flat frame (luma_spread=0.0) as a blank film."""

from __future__ import annotations

from src.services.stages.shared.mastering_gate import (
    mastering_gate_degradations,
    mastering_gate_notes,
)

# Live run 01a08f66-7e66-712d-a80e-74ad92f62991: a fine master with one flat frame.
_LIVE_FINE_MASTER: dict[str, object] = {
    "gate_pass": False,
    "rerun_stages": ["colorspace", "deband", "duplicate_frames"],
    "dead_frame_ratio": 0.009825,
    "luma_spread": 0.0,
}


def test_single_flat_frame_is_not_a_blank_picture() -> None:
    assert mastering_gate_notes(_LIVE_FINE_MASTER) == []
    assert mastering_gate_degradations(_LIVE_FINE_MASTER) == []


def test_compiler_dead_frame_verdict_is_blank() -> None:
    report = {"gate_pass": False, "rerun_stages": ["dead_frame"], "luma_spread": 0.0}
    notes = mastering_gate_notes(report)
    assert len(notes) == 1 and "blank/dead" in notes[0]
    assert len(mastering_gate_degradations(report)) == 1


def test_dead_frame_ratio_above_cutoff_is_blank_and_cites_ratio() -> None:
    report = {
        "gate_pass": False,
        "rerun_stages": ["deband"],
        "dead_frame_ratio": 0.5,
        "luma_spread": 0.0,
    }
    notes = mastering_gate_notes(report)
    assert len(notes) == 1
    assert "dead_frame_ratio=0.5" in notes[0]
    assert "luma_spread=0.0 reported, not used" in notes[0]
    degs = mastering_gate_degradations(report)
    assert len(degs) == 1
    assert "dead_frame_ratio=0.5" in str(degs[0])
