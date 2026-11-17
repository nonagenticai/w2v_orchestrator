"""4.3 — unmeasured loudness is INSUFFICIENT EVIDENCE, not a silent pass.

⛔ THE DEFECT. ``_loudness_ok`` returned ``True`` both when there was no loudness
block at all and when ``integrated_lufs`` was non-numeric, reasoning "no measured
value ⇒ no loudness objection". Sound for a gate, wrong for a report: it made a
master nobody listened to indistinguishable from a correctly-mastered one, so the
ABSENCE of the measurement read downstream as evidence of quality.

The three-way distinction is the whole point, so all three states are asserted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from src.services.stages.shared.gestalt_metrics import (
    LOUDNESS_FAIL,
    LOUDNESS_INSUFFICIENT_EVIDENCE,
    LOUDNESS_PASS,
    gestalt_summary,
    loudness_verdict,
)

_WINDOW = (-15.0, -13.0)


def _verdict(
    loudness: Mapping[str, object] | None,
    mastering: Mapping[str, object] | None = None,
) -> str:
    return cast(
        "str",
        gestalt_summary(
            frames=[],
            frame_analyses=[],
            loudness=loudness,
            mastering=mastering,
        )["loudness_verdict"],
    )


def test_measured_good_is_pass() -> None:
    assert loudness_verdict({"integrated_lufs": -14.0}, _WINDOW) == LOUDNESS_PASS
    assert _verdict({"integrated_lufs": -14.0}) == LOUDNESS_PASS


def test_measured_bad_is_fail() -> None:
    assert loudness_verdict({"integrated_lufs": -6.0}, _WINDOW) == LOUDNESS_FAIL
    assert loudness_verdict({"integrated_lufs": -30.0}, _WINDOW) == LOUDNESS_FAIL
    assert _verdict({"integrated_lufs": -6.0}) == LOUDNESS_FAIL


def test_unmeasured_is_insufficient_evidence_and_not_a_pass() -> None:
    """The three-way distinction: it must equal neither of the other two states."""
    for payload in (None, {}, {"integrated_lufs": None}, {"integrated_lufs": "loud"}):
        got = loudness_verdict(payload, _WINDOW)
        assert got == LOUDNESS_INSUFFICIENT_EVIDENCE, payload
        assert got != LOUDNESS_PASS
        assert got != LOUDNESS_FAIL

    assert _verdict(None) == LOUDNESS_INSUFFICIENT_EVIDENCE
    assert _verdict({"integrated_lufs": None}) == LOUDNESS_INSUFFICIENT_EVIDENCE


def test_a_bool_lufs_is_not_a_measurement() -> None:
    assert loudness_verdict({"integrated_lufs": True}, _WINDOW) == LOUDNESS_INSUFFICIENT_EVIDENCE


def test_the_compiler_can_declare_a_number_unmeasured() -> None:
    """``mastering.unmeasured`` overrides a present number, in every shape it may take.

    ⛔ THE NAME ON THE WIRE IS ``integrated_lufs``, NOT ``loudness``. The compiler builds
    this list in ``video_compiler_service`` ``mastering.py`` from the KEYWORD ARGUMENT
    names of ``_unmeasured_metrics(...)`` — ``integrated_lufs``, ``dead_frame_ratio``,
    ``cfr_ok``, ``av_gap_seconds`` — and never emits the string "loudness" at all. The
    first cut of ``_loudness_measured`` tested for "loudness", so against a REAL master
    it could never return False and the whole ``insufficient_evidence`` path was dead in
    production; its test passed only because the test hand-fed it a name the compiler
    does not produce. Every case below is a shape the compiler can actually emit.
    """
    good = {"integrated_lufs": -14.0}
    assert _verdict(good) == LOUDNESS_PASS  # baseline: same payload passes

    for unmeasured in (
        ["integrated_lufs"],
        # The real shape, verbatim: the compiler emits a SORTED list of every metric it
        # could not measure, loudness among several others.
        ["cfr_ok", "dead_frame_ratio", "integrated_lufs", "true_peak_dbtp"],
        ("integrated_lufs", "colorspace_ok"),
        {"integrated_lufs": True},
        True,
    ):
        assert _verdict(good, {"unmeasured": unmeasured}) == LOUDNESS_INSUFFICIENT_EVIDENCE

    # The legacy stage name is still TOLERATED (cheap insurance if that channel is ever
    # renamed) — but it is an alias, never the thing under test.
    assert _verdict(good, {"unmeasured": ["loudness"]}) == LOUDNESS_INSUFFICIENT_EVIDENCE


def test_absent_enforced_channel_falls_back_to_todays_behaviour() -> None:
    """⛔ Absence must NEVER read as 'unmeasured' — every pre-deploy master lacks it."""
    good = {"integrated_lufs": -14.0}
    for mastering in (None, {}, {"unmeasured": None}, {"duplicate_frame_ratio": 0.3}):
        assert _verdict(good, mastering) == LOUDNESS_PASS, mastering

    # A report that says loudness WAS measured is likewise a pass — including a report
    # that names OTHER unmeasured metrics, which is the common live shape.
    assert _verdict(good, {"unmeasured": ["colorspace_ok"]}) == LOUDNESS_PASS
    assert _verdict(good, {"unmeasured": ["cfr_ok", "dead_frame_ratio"]}) == LOUDNESS_PASS
    assert _verdict(good, {"unmeasured": {"cfr_ok": True}}) == LOUDNESS_PASS
    assert _verdict(good, {"unmeasured": False}) == LOUDNESS_PASS
