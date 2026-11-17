"""Tests for the deterministic retention / pacing analysis (ORCH-0.3).

Pure — no DB, no client, no ctx. Mirrors ``test_loop_engine.py``: everything runs
against fixed in-memory beat lists, and identical inputs must yield byte-identical
outputs.
"""

from __future__ import annotations

from itertools import pairwise

from src.services.stages.shared.retention import (
    analyze_retention,
    beat_budget,
    pattern_interrupt_gaps,
    predicted_watch_curve,
    resolve_retention_weights,
    retention_risk,
    time_to_first_footage,
)

# A fixed beat list engineered to trip every 31s-cliff flag:
#   - first footage-pinned beat starts at 5+6+5 = 16s  (> 8s ttff budget)
#   - a leading dead-beat run of 16s                    (> 12s dead-beat bound)
#   - a 16s gap from the video start to the first cut-in (> 15s interrupt bound)
# Total duration = 29s.
_BEATS = [
    {"id": "clip-1", "text": "intro hook", "seconds": 5.0, "clip": None},
    {"id": "clip-2", "text": "talking", "seconds": 6.0, "clip": None},
    {"id": "clip-3", "text": "more talking", "seconds": 5.0, "clip": None},
    {"id": "clip-4", "text": "the demo", "seconds": 8.0, "clip": "clip-4"},
    {"id": "clip-5", "text": "outro", "seconds": 5.0, "clip": None},
]


def test_time_to_first_footage_with_and_without_clip() -> None:
    assert time_to_first_footage(_BEATS) == 16.0

    # First beat is footage-pinned → footage appears immediately.
    front = [{"id": "clip-1", "seconds": 3.0, "clip": "clip-1"}, *_BEATS]
    assert time_to_first_footage(front) == 0.0

    # No beat is footage-pinned → returns the full duration (footage never plays).
    talking_only = [{"id": b["id"], "seconds": b["seconds"], "clip": None} for b in _BEATS]
    assert time_to_first_footage(talking_only) == 29.0


def test_watch_curve_is_deterministic_and_ordered() -> None:
    curve = predicted_watch_curve(_BEATS)
    assert curve == predicted_watch_curve(_BEATS)  # deterministic
    assert len(curve) == len(_BEATS)  # one point per beat
    assert curve[0] == 1.0  # offset 0 → full retention
    # Monotonic non-increasing: retention only decays over the cut.
    assert all(later <= earlier for earlier, later in pairwise(curve))


def test_flags_on_fixed_beat_list_are_deterministic() -> None:
    report = analyze_retention(_BEATS, target_seconds=25.0)
    # Re-running is byte-identical.
    assert report == analyze_retention(_BEATS, target_seconds=25.0)

    joined = " || ".join(report.insights)
    assert "first footage at 16.0s" in joined  # late-first-footage flag
    assert "dead-beat run of 16.0s" in joined  # dead-beat-run flag
    assert "no pattern interrupt for 16.0s" in joined  # interrupt-gap flag
    # 29s of beats against a 25s target is over budget.
    assert "beats sum to 29.0s vs the 25.0s target (+4.0s)" in joined

    # Structured numbers behind the strings.
    assert report.time_to_first_footage == 16.0
    assert report.total_seconds == 29.0
    assert report.over_budget is True
    assert len(report.risk_flags) == 2
    assert report.interrupt_gaps == [{"start_offset": 0.0, "gap_seconds": 16.0}]


def test_no_footage_flags_full_talking_head() -> None:
    talking_only = [{"id": b["id"], "seconds": b["seconds"], "clip": None} for b in _BEATS]
    flags = retention_risk(talking_only)
    assert any("no footage-pinned beat" in f for f in flags)


def test_beat_budget_over_and_under() -> None:
    over = beat_budget(_BEATS, 25.0)
    assert over.total_seconds == 29.0
    assert over.over_budget is True
    assert over.under_budget is False
    assert over.delta_seconds == 4.0

    under = beat_budget(_BEATS, 40.0)
    assert under.over_budget is False
    assert under.under_budget is True


def test_empty_beats_and_absent_weights_yield_empty_results() -> None:
    assert predicted_watch_curve([]) == []
    assert time_to_first_footage([]) == 0.0
    assert retention_risk([]) == []
    assert pattern_interrupt_gaps([]) == []

    # Absent weights (None) AND empty beats → no insights, whatever the target.
    report = analyze_retention([], weights=None, target_seconds=30.0)
    assert report.insights == []
    assert report.risk_flags == []
    assert report.interrupt_gaps == []
    assert report.watch_curve == []
    assert report.over_budget is False


def test_spine_ids_preserved_and_input_untouched() -> None:
    ids_before = [b["id"] for b in _BEATS]
    report = analyze_retention(_BEATS, target_seconds=25.0)
    # The analysis never mutates or reorders the caller's beat list.
    assert (
        [b["id"] for b in _BEATS]
        == ids_before
        == [
            "clip-1",
            "clip-2",
            "clip-3",
            "clip-4",
            "clip-5",
        ]
    )
    # One curve point per beat, in the same spine order (no reordering).
    assert len(report.watch_curve) == len(_BEATS)


def test_author_override_widens_the_footage_budget() -> None:
    # Default max_time_to_first_footage_seconds is 8.0 → 16s footage is flagged.
    default_flags = retention_risk(_BEATS)
    assert any("first footage at 16.0s" in f for f in default_flags)

    # An author who raises the budget past 16s suppresses that flag (override wins
    # its provenance rung). The dead-beat run (untouched knob) still flags.
    weights = resolve_retention_weights({"max_time_to_first_footage_seconds": 20.0})
    relaxed = retention_risk(_BEATS, weights)
    assert not any("first footage" in f for f in relaxed)
    assert any("dead-beat run" in f for f in relaxed)
