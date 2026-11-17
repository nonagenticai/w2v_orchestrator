"""mastering_gate — an ABSENT mastering report is UNMEASURED, never a pass.

The gate used to `return []` for a missing/malformed report, which is byte-identical
to the return of a gate that ran and PASSED. A compiler that shipped no `mastering`
block (an older build, or its own post-mux verify pass failed) therefore produced a
run that reported a *mastered* deliverable whose mastering was never checked — this
codebase's #1 defect class, an absent measurement reported as a passing one.
"""

from __future__ import annotations

import pytest

from src.services.stages.shared.mastering_gate import mastering_gate_notes


@pytest.mark.parametrize(
    "absent",
    [None],
    ids=["none"],
)
def test_absent_mastering_report_is_flagged_as_unmeasured(absent: object) -> None:
    notes = mastering_gate_notes(absent)
    assert len(notes) == 1, notes
    note = notes[0]
    # Must NAME the gap: the gate did not run, so the master is unmeasured.
    assert "UNMEASURED" in note
    assert "never ran" in note


def test_unmeasured_note_is_distinct_from_both_pass_and_fail() -> None:
    """Three outcomes must be three distinct observable values, not two."""
    unmeasured = mastering_gate_notes(None)
    passed = mastering_gate_notes({"rerun_stages": [], "luma_spread": 210.0, "gate_pass": True})
    # A genuinely blank master: dead_frame_ratio over the compiler's 0.10 cutoff.
    # (luma_spread=0.0 alone is NOT blank — one fade-to-black frame zeroes it.)
    failed = mastering_gate_notes(
        {
            "rerun_stages": ["deband"],
            "gate_pass": False,
            "dead_frame_ratio": 0.25,
            "luma_spread": 0.0,
        }
    )
    assert len(failed) == 1
    assert passed == []
    assert unmeasured != passed
    assert unmeasured != failed
    assert len({tuple(unmeasured), tuple(passed), tuple(failed)}) == 3
    # The unmeasured note must not masquerade as a measured failure.
    assert "gate FAILED" not in unmeasured[0]


def test_unmeasured_report_never_raises() -> None:
    """Severity stays DEGRADE: a note, never an exception (a paid, playable
    deliverable is not destroyed over an absent report)."""
    assert isinstance(mastering_gate_notes(object()), list)
