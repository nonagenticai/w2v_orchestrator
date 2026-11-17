"""4.5 — the degrade tier reads the compiler's ``enforced_stages`` when it publishes one.

⚠️ ``dead_frame`` is deliberately NOT added to ``_STRUCTURAL_STAGES``. It is ALREADY
armed in this tier via ``_blank_picture``; adding it would double-report one master and
the second report would carry ``av_coverage``'s sentence, which is false of a dead
frame. Both properties are pinned below so a future edit cannot quietly reintroduce it.
"""

from __future__ import annotations

from src.services.stages.shared.mastering_gate import mastering_gate_degradations


def _details(report: dict) -> str:
    entries = mastering_gate_degradations(report)
    return entries[0]["detail"] if entries else ""


_DEAD = {"gate_pass": False, "rerun_stages": ["dead_frame"], "dead_frame_ratio": 0.44}


def test_dead_frame_already_degrades_without_being_in_the_structural_set() -> None:
    """The tier is armed for dead_frame TODAY — via _blank_picture, not the set."""
    entries = mastering_gate_degradations(_DEAD)

    assert len(entries) == 1
    assert "BLANK/dead" in entries[0]["detail"]
    assert "dead_frame_ratio=0.44" in entries[0]["detail"]


def test_dead_frame_is_reported_exactly_once_and_never_as_av_coverage() -> None:
    """⛔ THE REGRESSION THIS GUARDS — measured output of the naive edit was:

    "... — dead_frame: the delivered picture does not cover the delivered narration;
     blank_picture: the delivered master's picture is BLANK/dead (...)"

    Two reasons for one defect, the first of them untrue.
    """
    detail = _details(_DEAD)

    assert detail.count("BLANK/dead") == 1
    assert "does not cover the delivered narration" not in detail
    assert "dead_frame:" not in detail


def test_av_coverage_keeps_its_own_wording_and_metrics() -> None:
    detail = _details(
        {
            "gate_pass": False,
            "rerun_stages": ["av_coverage"],
            "av_gap_seconds": 8.8,
            "video_stream_seconds": 30.0,
            "audio_stream_seconds": 38.8,
        }
    )

    assert "av_coverage: the delivered picture does not cover the delivered narration" in detail
    assert "av_gap_seconds=8.8" in detail


def test_a_tunable_only_failure_still_stays_quiet() -> None:
    """The anti-noise guard is untouched: deband alone records nothing."""
    assert (
        mastering_gate_degradations(
            {"gate_pass": False, "rerun_stages": ["deband"], "banding_scores": [1.0]}
        )
        == []
    )


def test_the_compilers_enforced_stages_list_is_preferred_over_the_mirror() -> None:
    """A stage the compiler declares enforced degrades, even though the mirror lacks it."""
    detail = _details(
        {
            "gate_pass": False,
            "rerun_stages": ["colorspace"],
            "enforced_stages": ["av_coverage", "colorspace"],
        }
    )

    assert "colorspace:" in detail
    # It must NOT inherit av_coverage's sentence — that claim was never observed.
    assert "does not cover the delivered narration" not in detail


def test_a_stage_the_compiler_does_not_enforce_stays_advisory() -> None:
    assert (
        mastering_gate_degradations(
            {
                "gate_pass": False,
                "rerun_stages": ["colorspace"],
                "enforced_stages": ["av_coverage"],
            }
        )
        == []
    )


def test_absent_enforced_stages_falls_back_to_the_mirror_never_to_nothing() -> None:
    """⛔ Every master built before that key ships carries no ``enforced_stages``.

    Reading absence as "the compiler enforces nothing" would silently disarm this tier
    across the entire existing fleet. A malformed or EMPTY list takes the same fallback:
    ``[]`` is far likelier to be a serialization accident than a real "nothing counts".
    """
    for bad in ({}, {"enforced_stages": None}, {"enforced_stages": []}, {"enforced_stages": "av"}):
        report = {"gate_pass": False, "rerun_stages": ["av_coverage"], "av_gap_seconds": 8.8, **bad}
        assert "av_coverage:" in _details(report), bad

    # ...and the fallback must still stay quiet on a tunable-only failure.
    for bad in ({}, {"enforced_stages": []}):
        report = {"gate_pass": False, "rerun_stages": ["deband"], **bad}
        assert mastering_gate_degradations(report) == [], bad
