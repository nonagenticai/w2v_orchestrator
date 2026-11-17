"""Unit tests for the deterministic time-budget reconciler (A22 / C13).

No DB, no HTTP, no LLM — these are pure functions.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.services.stages.shared.provenance import VALID_SOURCES
from src.services.stages.shared.time_budget import (
    DEFAULT_FREEZE_PAD_CEILING,
    DEFAULT_TOLERANCE_PCT,
    DEFAULT_WPM_NATURAL,
    VO_RATE_BAND,
    VO_RATE_DRAG_FLOOR,
    VO_RATE_NATURAL_FLOOR,
    Allotment,
    TimeDemand,
    budget_drift,
    build_timeline_manifest,
    collect_demands,
    content_drift,
    drift_report,
    freeze_pad_ceiling,
    infeasible_beats,
    is_underfull,
    reconcile_timeline,
    timeline_manifest,
    tolerance_seconds,
    underspoken_hold,
    underspoken_uncovered,
    vo_rate_for,
)


def _words(n: int) -> str:
    return " ".join(["word"] * n)


def _elastic(beat_id: str, lo: float, ideal: float, hi: float, words: int = 0) -> TimeDemand:
    return TimeDemand(
        id=beat_id,
        kind="vo" if words else "card",
        min=lo,
        ideal=ideal,
        max=hi,
        elasticity=1.0,
        pinned=False,
        word_count=words,
    )


def _pinned(beat_id: str, seconds: float, words: int = 0, kind: str = "footage") -> TimeDemand:
    return TimeDemand(
        id=beat_id,
        kind=kind,
        min=seconds,
        ideal=seconds,
        max=seconds,
        elasticity=0.0,
        pinned=True,
        word_count=words,
    )


# --- collect_demands -------------------------------------------------------


def test_footage_beat_is_pinned_to_its_measured_length() -> None:
    demands = collect_demands(
        [{"id": "clip-1", "clip": "@clip-1", "text": _words(30)}],
        clip_seconds={"clip-1": 7.5},
    )
    demand = demands[0]
    assert demand.kind == "footage"
    assert demand.pinned is True
    assert demand.elasticity == 0.0
    assert (demand.min, demand.ideal, demand.max) == (7.5, 7.5, 7.5)
    assert demand.word_count == 30  # narration still declared → gets a vo_rate


def test_locked_line_sizes_its_beat_from_its_words_not_the_clip() -> None:
    # merqi showcase spike 01a0fc55: three author-`locked` lines over 2 recordings.
    # Pinned to the measured clip, the beat's seconds came from the RECORDING and the
    # underspoken pass padded the verbatim words to fill it. A locked line owns its
    # beat: the vo band of its own words, ideal = the authored seconds where sayable,
    # and the clip is frozen / trimmed to that — never the text to the clip.
    demands = collect_demands(
        [
            {"id": "l1", "clip": "clip-1", "text": _words(20), "seconds": 8.0, "locked": True},
            {"id": "l2", "clip": "clip-2", "text": _words(20), "locked": True},
            # Authored seconds the verbatim words cannot be said in → band edge, not text.
            {"id": "l3", "clip": "clip-2", "text": _words(20), "seconds": 1.0, "locked": True},
        ],
        clip_seconds={"clip-1": 30.0, "clip-2": 30.0},
    )
    by_id = {d.id: d for d in demands}
    for demand in demands:
        assert demand.pinned is False
        assert demand.kind == "vo"
        assert demand.min == pytest.approx(20 / 170 * 60)
        assert demand.max == pytest.approx(20 / 130 * 60)
        assert demand.max < 30.0  # nowhere near the 30s recording
    assert by_id["l1"].ideal == pytest.approx(8.0)
    assert by_id["l2"].ideal == pytest.approx(20 / DEFAULT_WPM_NATURAL * 60)
    assert by_id["l3"].ideal == pytest.approx(by_id["l3"].min)


def test_unlocked_footage_line_stays_pinned_to_its_clip() -> None:
    demands = collect_demands(
        [{"id": "l1", "clip": "clip-1", "text": _words(20), "locked": False}],
        clip_seconds={"clip-1": 30.0},
    )
    assert demands[0].pinned is True
    assert (demands[0].min, demands[0].max) == (30.0, 30.0)


def test_clip_without_measured_length_is_never_pinned_to_a_guess() -> None:
    # A beat carrying a `clip` tag but with NO measured footage length must fall
    # back to its vo demand — we never pin to a guess.
    demands = collect_demands(
        [{"id": "clip-1", "clip": "clip-1", "text": _words(150), "seconds": 4.0}],
        clip_seconds={},
    )
    demand = demands[0]
    assert demand.pinned is False
    assert demand.kind == "vo"
    assert demand.elasticity == 1.0
    assert demand.min < demand.ideal < demand.max


def test_vo_demand_derives_from_the_wpm_band() -> None:
    demands = collect_demands([{"id": "b1", "text": _words(150)}], clip_seconds={})
    demand = demands[0]
    assert demand.ideal == pytest.approx(60.0)  # 150 words at 150 wpm
    assert demand.min == pytest.approx(150 / 170 * 60)
    assert demand.max == pytest.approx(150 / 130 * 60)
    assert demand.min < demand.ideal < demand.max


def test_card_beat_uses_authored_seconds_then_a_hold_ceiling() -> None:
    demands = collect_demands([{"id": "card-1", "seconds": 3.0}], clip_seconds={})
    demand = demands[0]
    assert demand.kind == "card"
    assert demand.pinned is False
    assert demand.ideal == pytest.approx(3.0)
    assert demand.max == pytest.approx(6.0)


def test_card_floor_seconds_feeds_the_read_time_floor() -> None:
    demands = collect_demands([{"id": "card-1"}], clip_seconds={}, card_floor_seconds=2.5)
    assert demands[0].min == pytest.approx(2.5)


def test_time_lock_beat_is_pinned_at_its_authored_seconds() -> None:
    demands = collect_demands(
        [{"id": "b1", "text": _words(200), "seconds": 5.0, "time_lock": True}],
        clip_seconds={},
    )
    demand = demands[0]
    assert demand.pinned is True
    assert demand.elasticity == 0.0
    assert (demand.min, demand.ideal, demand.max) == (5.0, 5.0, 5.0)


def test_authored_min_max_seconds_override_the_computed_bounds() -> None:
    demands = collect_demands(
        [{"id": "b1", "text": _words(150), "min_seconds": 40.0, "max_seconds": 45.0}],
        clip_seconds={},
    )
    demand = demands[0]
    assert demand.min == pytest.approx(40.0)
    assert demand.max == pytest.approx(45.0)
    # The computed ideal (60s) is repaired back inside the authored bounds.
    assert demand.min <= demand.ideal <= demand.max


def test_demands_follow_spine_order_and_keep_ids_verbatim() -> None:
    lines = [{"id": "C0", "text": _words(10)}, {"id": "clip-2"}, {"id": "C3", "text": _words(5)}]
    assert [d.id for d in collect_demands(lines, clip_seconds={})] == ["C0", "clip-2", "C3"]


# --- reconcile_timeline ----------------------------------------------------


def test_feasible_timeline_sums_to_target_within_tolerance() -> None:
    demands = [
        _elastic("b1", 4.0, 8.0, 16.0, words=20),
        _elastic("b2", 6.0, 12.0, 24.0, words=30),
        _elastic("b3", 2.0, 5.0, 10.0),
    ]
    target = 30.0
    allotments = reconcile_timeline(demands, target=target)
    total = sum(a.seconds for a in allotments)
    assert abs(total - target) <= target * DEFAULT_TOLERANCE_PCT / 100.0


def test_pinned_footage_is_never_trimmed_and_absorbs_the_rest() -> None:
    demands = [
        _pinned("clip-1", 12.0),
        _elastic("b2", 2.0, 6.0, 20.0, words=15),
    ]
    target = 20.0
    allotments = {a.id: a for a in reconcile_timeline(demands, target=target)}
    # C0: the measured clip keeps its exact length, whatever the target demands.
    assert allotments["clip-1"].seconds == 12.0
    assert allotments["clip-1"].source == "default"
    assert allotments["b2"].seconds == pytest.approx(8.0)
    assert sum(a.seconds for a in allotments.values()) == pytest.approx(target)


# elastic beats collapse to their floor


def test_every_allotment_is_at_least_its_min_or_the_beat_is_flagged() -> None:
    demands = [
        _elastic("b1", 10.0, 12.0, 14.0, words=25),
        _elastic("b2", 10.0, 12.0, 14.0, words=25),
        _elastic("b3", 10.0, 12.0, 14.0),
    ]
    for target in (20.0, 36.0, 100.0):
        allotments = reconcile_timeline(demands, target=target)
        flagged = set(infeasible_beats(demands, target))
        for allotment in allotments:
            demand = next(d for d in demands if d.id == allotment.id)
            assert allotment.seconds >= demand.min - 1e-9 or allotment.id in flagged
            assert allotment.seconds <= demand.max + 1e-9


def test_saturated_beat_is_clamped_and_records_llm_clamped_provenance() -> None:
    # b1 is tiny-capped; the slack it cannot absorb must re-flood b2 (water-filling).
    demands = [_elastic("b1", 1.0, 2.0, 3.0), _elastic("b2", 1.0, 2.0, 40.0)]
    allotments = {a.id: a for a in reconcile_timeline(demands, target=30.0)}
    assert allotments["b1"].seconds == pytest.approx(3.0)  # capped at max
    assert allotments["b1"].source == "llm-clamped"
    assert allotments["b2"].seconds == pytest.approx(27.0)  # took the leftover
    assert allotments["b2"].source == "smart-auto"


def test_sources_are_all_valid_provenance_sources() -> None:
    demands = [_pinned("clip-1", 5.0), _elastic("b2", 1.0, 3.0, 4.0), _elastic("b3", 1.0, 2.0, 9.0)]
    for allotment in reconcile_timeline(demands, target=18.0):
        assert allotment.source in VALID_SOURCES


def test_reconcile_is_deterministic() -> None:
    demands = [_elastic("b1", 2.0, 5.0, 9.0, words=12), _elastic("b2", 3.0, 6.0, 11.0)]
    assert reconcile_timeline(demands, target=14.0) == reconcile_timeline(demands, target=14.0)


def test_infeasible_when_min_sum_exceeds_target() -> None:
    demands = [
        _elastic("b1", 10.0, 12.0, 14.0, words=40),
        _elastic("b2", 10.0, 12.0, 14.0, words=40),
    ]
    target = 12.0
    assert set(infeasible_beats(demands, target)) == {"b1", "b2"}
    allotments = reconcile_timeline(demands, target=target)
    assert all(a.infeasible for a in allotments)
    # Floors are honored and the sum honestly overruns the target.
    assert [a.seconds for a in allotments] == [10.0, 10.0]
    assert sum(a.seconds for a in allotments) > target


def test_infeasible_beats_empty_when_feasible() -> None:
    demands = [_elastic("b1", 2.0, 4.0, 8.0), _elastic("b2", 2.0, 4.0, 8.0)]
    assert infeasible_beats(demands, 10.0) == []
    assert all(not a.infeasible for a in reconcile_timeline(demands, target=10.0))


def test_infeasible_flags_the_beat_over_its_fair_share() -> None:
    demands = [_elastic("hog", 18.0, 20.0, 24.0), _elastic("b2", 4.0, 5.0, 6.0)]
    assert infeasible_beats(demands, 20.0) == ["hog"]


# --- budget_drift / is_underfull (C1) --------------------------------------


def test_under_supply_is_reported_not_faked() -> None:
    """C1 regression: Σmax < target. ``infeasible_beats`` is EMPTY (Σmin=2 <= 10), the
    solver saturates both beats at max (Σ=6 vs a target of 10 — 40% off), and the
    headline tolerance guarantee does NOT hold. It must be REPORTED, never faked."""
    demands = [_elastic("b1", 1.0, 2.0, 3.0), _elastic("b2", 1.0, 2.0, 3.0)]
    target = 10.0

    assert infeasible_beats(demands, target) == []  # the vacuous "feasible"
    allotments = reconcile_timeline(demands, target=target)

    assert sum(a.seconds for a in allotments) == pytest.approx(6.0)  # saturated at max
    assert budget_drift(allotments, target) == pytest.approx(-4.0)
    assert is_underfull(allotments, target) is True
    # The solver never invents seconds past a beat's max.
    assert all(a.seconds <= 3.0 + 1e-9 for a in allotments)


def test_budget_drift_is_zero_and_not_underfull_on_a_feasible_timeline() -> None:
    demands = [_elastic("b1", 1.0, 2.0, 9.0), _elastic("b2", 1.0, 2.0, 9.0)]
    allotments = reconcile_timeline(demands, target=10.0)
    assert budget_drift(allotments, 10.0) == pytest.approx(0.0)
    assert is_underfull(allotments, 10.0) is False


def test_budget_drift_is_positive_on_an_infeasible_overrun() -> None:
    demands = [_elastic("b1", 10.0, 12.0, 14.0), _elastic("b2", 10.0, 12.0, 14.0)]
    assert budget_drift(reconcile_timeline(demands, target=12.0), 12.0) == pytest.approx(8.0)


def test_tolerance_seconds_is_the_band_in_seconds() -> None:
    assert tolerance_seconds(100.0, 2.0) == pytest.approx(2.0)
    assert tolerance_seconds(50.0, DEFAULT_TOLERANCE_PCT) == pytest.approx(1.0)


# --- underspoken_uncovered / budget_drift counting only what is DELIVERED (1d) ----


def test_underspoken_uncovered_is_the_residual_beyond_the_capped_hold() -> None:
    """Same fixture as ``test_underspoken_hold_quantifies_the_unfilled_residual``:
    6 words / 10.0s allotment. 6 words read at 150 wpm is 2.40s of natural speech, and
    the slowest DELIVERY allowed is natural (``VO_RATE_MIN_DELIVERED``, 1.0), so the
    words occupy 2.40s ⇒ a 7.60s residual, hold capped at 1.50s, so 6.10s is covered by
    NEITHER speech nor a held frame.

    ⚠️ Was 7.00s / 5.50s at the 0.80 floor and 7.5258s / 6.0258s at the 0.97 floor. The
    residual GREW each time because the voice may no longer be dragged to absorb it.
    """
    allotment_seconds = 10.0
    words = 6
    hold = underspoken_hold(allotment_seconds, words, DEFAULT_WPM_NATURAL)
    assert hold == pytest.approx(DEFAULT_FREEZE_PAD_CEILING)
    uncovered = underspoken_uncovered(allotment_seconds, words, DEFAULT_WPM_NATURAL)
    assert uncovered == pytest.approx(6.10, abs=1e-4)
    # Derived, not pasted: the residual is the allotment minus the seconds the words
    # actually occupy at natural speed.
    assert uncovered == pytest.approx(
        allotment_seconds - (words / DEFAULT_WPM_NATURAL * 60.0) - DEFAULT_FREEZE_PAD_CEILING
    )
    # hold + uncovered reconstructs the full uncapped residual, exactly.
    generous = underspoken_hold(allotment_seconds, words, DEFAULT_WPM_NATURAL, ceiling=99.0)
    assert hold + uncovered == pytest.approx(generous)


def test_underspoken_uncovered_is_zero_when_the_hold_already_covers_the_residual() -> None:
    """A small residual that fits inside the ceiling leaves nothing uncovered."""
    # 34 words / 150 wpm = 13.6s natural, /0.97 floor = 14.0206s spoken; a 15.0s
    # allotment leaves a 0.9794s residual, under the 1.5s ceiling.
    #
    # ⚠️ The word count moved from 28 to 34 with the floor. At 0.97 the words occupy
    # nearly their natural length, so filling a 15.0s allotment to within the ceiling
    # needs MORE words than it did at 0.80 (which stretched 28 words across 14.0s).
    # 28 words here would now leave a 3.78s residual — over the ceiling, uncovered
    # 2.28s — so keeping the old fixture would have made this test assert the
    # opposite of its own name.
    residual = 15.0 - (34 / DEFAULT_WPM_NATURAL * 60.0) / VO_RATE_DRAG_FLOOR
    assert 0.0 < residual < DEFAULT_FREEZE_PAD_CEILING, residual
    uncovered = underspoken_uncovered(15.0, 34, DEFAULT_WPM_NATURAL)
    assert uncovered == pytest.approx(0.0)


def test_underspoken_uncovered_is_zero_on_a_beat_that_is_not_underspoken() -> None:
    assert underspoken_uncovered(10.0, 30, DEFAULT_WPM_NATURAL) == pytest.approx(0.0)


def test_budget_drift_subtracts_uncovered_seconds_not_just_the_capped_hold() -> None:
    """The 1d fix in one number. A pinned beat 40.0s long with only 6 words is
    massively underspoken: natural 2.4s, spoken at natural speed 2.4s, residual
    37.6s, hold capped at 1.5s ⇒ 36.1s uncovered. Before 1d, `budget_drift`
    summed the bare `a.seconds` (40.0) against a 40.0s target and reported ZERO
    drift — a master this short of its own picture never registered as
    underfull. It must now report the drift the uncovered residual represents.

    ⚠️ Was 3.0s / 37.0s / 35.5s at the 0.80 floor and 36.0258s at 0.97; see
    ``test_underspoken_uncovered_is_the_residual_beyond_the_capped_hold``.
    """
    # A pinned VO beat (author-locked), NOT footage: a footage beat's clip plays for its
    # whole allotment, so it has no uncovered seconds — see the footage test below.
    allotments = reconcile_timeline([_pinned("starved", 40.0, words=6, kind="vo")], target=40.0)
    (beat,) = allotments
    assert beat.underspoken is True
    assert beat.uncovered_seconds == pytest.approx(36.1, abs=1e-4)
    assert beat.seconds == pytest.approx(40.0)  # the allotment itself is unchanged
    assert budget_drift(allotments, 40.0) == pytest.approx(-36.1, abs=1e-4)
    assert is_underfull(allotments, 40.0) is True


def test_footage_beat_has_no_uncovered_seconds_its_clip_fills_the_allotment() -> None:
    """Round-5 P1 (2026-10-07): 3 footage beats 13.58 / 6.48 / 9.64s with 21 / 13 / 25
    words at ~150.9 wpm on a 30s order. The film delivered 29.77s, yet `budget_drift`
    subtracted 3.73s of "uncovered" narration slack from clip-1 and reported 25.97s ⇒
    `timeline_underfull` + `timeline_off_target`. A footage beat's moving picture plays
    for every allotted second; only the hold/flag describe the short narration."""
    wpm = 150.9
    demands = [
        _pinned("clip-1", 13.58, words=21),
        _pinned("clip-4", 6.48, words=13),
        _pinned("clip-5", 9.64, words=25),
    ]
    allotments = reconcile_timeline(demands, target=30.0, wpm_natural=wpm)
    by_id = {a.id: a for a in allotments}
    assert by_id["clip-1"].underspoken is True, "precondition: the narration IS short"
    assert by_id["clip-1"].hold_seconds == pytest.approx(1.5)
    assert all(a.uncovered_seconds == 0.0 for a in allotments)
    assert content_drift(allotments, 30.0) == pytest.approx(13.58 + 6.48 + 9.64 - 30.0)
    # clip-5's 25 words (9.94s natural) overrun its 9.64s clip ⇒ a 0.30s freeze pad, which
    # `budget_drift` counts as delivered (run 01a15ec2). Planned 30.00s vs the 29.77s
    # this film delivered — both inside the 2% band, which is what this test pins.
    pad = by_id["clip-5"].freeze_pad_seconds
    assert pad == pytest.approx(25 / wpm * 60.0 - 9.64)
    assert budget_drift(allotments, 30.0) == pytest.approx(13.58 + 6.48 + 9.64 + pad - 30.0)
    assert abs(budget_drift(allotments, 30.0)) <= tolerance_seconds(30.0, DEFAULT_TOLERANCE_PCT)
    assert is_underfull(allotments, 30.0) is False


def _run_01a15ec2_allotments() -> list[Allotment]:
    """Showcase run 01a15ec2-8b5f (2026-10-21), the reconcile_timeline lines verbatim:
    three footage beats, clip-4 and clip-5 granted a freeze pad for narration that runs
    past the clip. The compiler rendered them at 9.50 / 11.13 / 9.70s = 30.33s."""
    return [
        Allotment("clip-1", 9.48, 1.0, "default", False, freeze_pad_seconds=0.0),
        Allotment("clip-4", 10.36, 1.1, "default", False, freeze_pad_seconds=0.9027666666666683),
        Allotment("clip-5", 9.44, 1.077, "default", False, freeze_pad_seconds=0.21380000000000088),
    ]


def test_budget_drift_counts_the_freeze_held_footage_tail_the_compiler_renders() -> None:
    """Run 01a15ec2: the gate summed the bare clips (29.28s) and stamped
    `timeline_underfull` + `timeline_off_target` on a 30s order, while the master was
    30.333s. The pad IS delivered film: Σ(seconds + pad) = 30.40s, inside the 2% band,
    within 0.07s of the measured master."""
    allotments = _run_01a15ec2_allotments()
    delivered_master = 30.333333333333332  # compile.done duration_seconds

    assert content_drift(allotments, 30.0) == pytest.approx(29.28 - 30.0)  # the old number
    assert content_drift(allotments, 30.0) < -tolerance_seconds(30.0, DEFAULT_TOLERANCE_PCT)
    drift = budget_drift(allotments, 30.0)
    assert 30.0 + drift == pytest.approx(30.3966, abs=1e-3)
    assert abs(30.0 + drift - delivered_master) < 0.1
    assert abs(drift) <= tolerance_seconds(30.0, DEFAULT_TOLERANCE_PCT)
    assert is_underfull(allotments, 30.0) is False


def test_budget_drift_pad_and_uncovered_never_overlap() -> None:
    """Footage carries pad and never uncovered; a VO beat the reverse — so adding one and
    subtracting the other cannot double-count a second."""
    allotments = reconcile_timeline(
        [_pinned("starved", 40.0, words=6, kind="vo"), _pinned("clip", 4.0, words=20)],
        target=44.0,
    )
    by_id = {a.id: a for a in allotments}
    assert by_id["starved"].freeze_pad_seconds == 0.0 and by_id["starved"].uncovered_seconds > 0
    assert by_id["clip"].uncovered_seconds == 0.0 and by_id["clip"].freeze_pad_seconds > 0
    assert budget_drift(allotments, 44.0) == pytest.approx(
        content_drift(allotments, 44.0) + by_id["clip"].freeze_pad_seconds
    )


def test_budget_drift_is_unaffected_when_no_beat_is_underspoken() -> None:
    """Byte-identical guarantee: `uncovered_seconds` is 0.0 on every beat this
    cannot apply to, so a timeline with no underspoken beats is unchanged."""
    demands = [_elastic("b1", 1.0, 2.0, 9.0), _elastic("b2", 1.0, 2.0, 9.0)]
    allotments = reconcile_timeline(demands, target=10.0)
    assert all(a.uncovered_seconds == 0.0 for a in allotments)
    assert budget_drift(allotments, 10.0) == pytest.approx(0.0)


# --- vo_rate_for -----------------------------------------------------------


def test_vo_rate_is_one_at_the_natural_allotment() -> None:
    rate, unsayable, underspoken = vo_rate_for(60.0, 150, DEFAULT_WPM_NATURAL)
    assert rate == pytest.approx(1.0)
    assert unsayable is False
    assert underspoken is False


def test_vo_rate_speeds_up_when_the_allotment_shrinks() -> None:
    rate, unsayable, underspoken = vo_rate_for(40.0, 150, DEFAULT_WPM_NATURAL)
    assert rate == pytest.approx(1.5)
    assert unsayable is False
    assert underspoken is False


@pytest.mark.parametrize(
    ("allotment", "words"),
    [(0.5, 200)],
)
def test_vo_rate_always_lands_inside_the_band(allotment: float, words: int) -> None:
    rate, _, _ = vo_rate_for(allotment, words)
    assert VO_RATE_BAND[0] <= rate <= VO_RATE_BAND[1]


def test_unsayable_when_even_max_rate_overruns() -> None:
    # 300 words need 120s naturally; a 10s beat needs rate 12 → un-sayable.
    rate, unsayable, _ = vo_rate_for(10.0, 300)
    assert rate == VO_RATE_BAND[1]
    assert unsayable is True


def test_no_words_is_a_neutral_rate() -> None:
    # Nothing to say ⇒ genuinely neutral, whatever the allotment.
    assert vo_rate_for(10.0, 0) == (1.0, False, False)
    assert vo_rate_for(0.0, 0) == (1.0, False, False)


def test_words_with_zero_allotment_are_unsayable() -> None:
    """C2 regression: 5 words with ZERO seconds to speak them is the WORST unsayable
    case — the old early-return reported ``(1.0, False)`` and swallowed exactly the
    condition the flag exists to catch."""
    rate, unsayable, _ = vo_rate_for(0.0, 5)
    assert unsayable is True
    assert rate == VO_RATE_BAND[1]
    # Divide-by-zero safe on a negative allotment and on a degenerate wpm too.
    assert vo_rate_for(-3.0, 5) == (VO_RATE_BAND[1], True, False)
    assert vo_rate_for(10.0, 5, 0.0) == (VO_RATE_BAND[1], True, False)


def test_rate_never_drags_below_the_natural_floor() -> None:
    """The regression this floor exists for.

    Live 2026-09-11: a beat carrying 12 words was handed a ~19.4s allotment. The
    raw rate is ~0.25; clamped to the API's 0.5 it synthesized a 2.08x uniform
    stretch — phonemes AND pauses — and shipped as a "fitted" line. The floor is
    the OPERATIVE one, so the returned rate is never in drag territory again.

    ⚠️ The delivered rate is natural (1.0) — not ``VO_RATE_DRAG_FLOOR`` (0.97, which
    is only the DETECTOR threshold) and not ``VO_RATE_NATURAL_FLOOR`` (0.80). A line
    that fits its slot is never slowed; the picture holds the rest.
    """
    rate, unsayable, underspoken = vo_rate_for(19.42, 12, DEFAULT_WPM_NATURAL)
    assert rate == 1.0
    assert rate > VO_RATE_NATURAL_FLOOR, "the pathological limit is a backstop, not a target"
    assert rate > VO_RATE_BAND[0], "the API floor is a validity limit, not a rate to target"
    assert underspoken is True
    assert unsayable is False


def test_underspoken_is_the_mirror_of_unsayable() -> None:
    """Too FEW words flags, symmetrically with too MANY. Before this, only one
    direction had a name and the other was clamped away in silence."""
    # 5 words in 60s: nothing like enough script to fill the beat.
    _, unsayable, underspoken = vo_rate_for(60.0, 5, DEFAULT_WPM_NATURAL)
    assert (unsayable, underspoken) == (False, True)
    # 300 words in 10s: the long-standing direction, unchanged.
    _, unsayable, underspoken = vo_rate_for(10.0, 300, DEFAULT_WPM_NATURAL)
    assert (unsayable, underspoken) == (True, False)


def test_a_rate_of_zero_point_eight_is_flagged_and_is_not_deliberate_fitting() -> None:
    """⭐ THIS TEST PREVIOUSLY ASSERTED THE DEFECT WAS CORRECT BEHAVIOUR.

    It was named ``test_a_rate_inside_the_natural_band_is_not_flagged`` and its
    docstring read: *"The floor must not cry wolf on deliberate fitting. 20 words read
    naturally in 8.0s, slowed into a 10.0s footage clip, is rate 0.8 — in design, not a
    fault."* It asserted ``rate == 0.8`` and ``underspoken is False``.

    That premise is the owner's complaint, written down as an expectation. A 0.8
    delivery is a 25% uniform time-stretch of both phonemes and pauses; calling it
    "deliberate fitting" is what kept it off every signal for twelve waves. The owner's
    wave-13 decision reverses it outright: the film may come in SHORT, and the voice
    may not be dragged to fill a slot.

    So the same fixture must now flag, and the voice is spoken at natural 1.0. The
    beat's 2.00s surplus is the PICTURE's to hold, not the voice's to stretch into.
    """
    rate, unsayable, underspoken = vo_rate_for(10.0, 20, DEFAULT_WPM_NATURAL)
    assert rate == 1.0
    assert underspoken is True, "a 25% stretch is the complaint, not a design choice"
    assert unsayable is False


def test_the_detector_covers_the_whole_band_the_complaint_lived_in() -> None:
    """⭐ THE OWNER'S OWN NUMBERS. The run certified as a PASS delivered 0.8933 /
    0.8936 / 0.8650 — 11-13% slow, audible, and invisible to every signal because each
    one was armed at 0.80 (where the CLAMP binds) rather than at an audibility
    threshold. Every one of those rates must now raise the flag.

    This is the regression test for the blind spot itself, not for one beat: it walks
    the band and asserts the boundary is where the constant says it is, in BOTH
    directions, so a future "fix" that quietly re-lowers the floor fails here.
    """
    words = 20
    natural_seconds = words / DEFAULT_WPM_NATURAL * 60.0  # 8.00s

    # The three delivered rates from the falsely-certified run.
    for delivered in (0.8933, 0.8936, 0.8650):
        allotment = natural_seconds / delivered
        rate, unsayable, underspoken = vo_rate_for(allotment, words, DEFAULT_WPM_NATURAL)
        assert underspoken is True, f"{delivered} is 11-13% slow and MUST raise a signal"
        assert unsayable is False
        # And the voice is no longer delivered at that rate — it is spoken at natural.
        assert rate == 1.0

    # THE BOUNDARY BINDS BOTH WAYS. Just inside the floor flags; just outside does not.
    # Without this pair the assertion above would also pass a detector armed at 1.0,
    # which would re-ask on every run (see VO_RATE_DRAG_FLOOR's cost note).
    just_under = natural_seconds / (VO_RATE_DRAG_FLOOR - 0.005)
    just_over = natural_seconds / (VO_RATE_DRAG_FLOOR + 0.005)
    assert vo_rate_for(just_under, words, DEFAULT_WPM_NATURAL)[2] is True
    assert vo_rate_for(just_over, words, DEFAULT_WPM_NATURAL)[2] is False


def test_reconcile_carries_underspoken_onto_the_allotment() -> None:
    """The flag has to REACH the allotment or nothing downstream can report it —
    which is exactly how a 0.5-rate line shipped unremarked for a day."""
    allotments = reconcile_timeline([_pinned("starved", 40.0, words=6)], target=40.0)
    assert allotments[0].underspoken is True
    assert allotments[0].vo_rate == 1.0


def test_the_solver_honours_a_configured_drag_floor() -> None:
    """The floor is CONFIGURABLE (``direction.timing.vo_rate_drag_floor``) because the
    0.97 band was calibrated against ONE voice. Threaded from the solver's single
    ``natural_floor`` parameter into the flag, the hold and the uncovered residual
    together, so they cannot be taken against three different floors.

    Pinned at the OLD value to prove the parameter is real: with the floor put back to
    0.80 this fixture stops flagging. The floor moves ONLY the detector — the voice is
    spoken at natural 1.0 either way, and the hold (sized against the delivered rate)
    is identical, so a relaxed floor can never re-open a slowed delivery.
    """
    demands = [_pinned("clip-1", 10.0, words=20)]  # raw rate exactly 0.80
    strict = reconcile_timeline(demands, target=10.0)
    assert strict[0].underspoken is True
    assert strict[0].vo_rate == 1.0

    relaxed = reconcile_timeline(demands, target=10.0, natural_floor=VO_RATE_NATURAL_FLOOR)
    assert relaxed[0].underspoken is False
    assert relaxed[0].vo_rate == 1.0
    assert relaxed[0].hold_seconds == pytest.approx(strict[0].hold_seconds)
    assert strict[0].hold_seconds > 0.0


def test_reconcile_stamps_rates_only_on_speaking_beats() -> None:
    demands = [
        _elastic("vo", 4.0, 8.0, 16.0, words=20),
        _elastic("card", 2.0, 3.0, 6.0),
        _pinned("clip-1", 5.0),
    ]
    allotments = {a.id: a for a in reconcile_timeline(demands, target=16.0)}
    assert allotments["vo"].vo_rate is not None
    assert VO_RATE_BAND[0] <= (allotments["vo"].vo_rate or 0.0) <= VO_RATE_BAND[1]
    assert allotments["card"].vo_rate is None
    assert allotments["clip-1"].vo_rate is None


def test_unsayable_surfaces_on_the_allotment() -> None:
    # A pinned 2s clip carrying 200 words of narration cannot be spoken in band.
    allotments = reconcile_timeline([_pinned("clip-1", 2.0, words=200)], target=2.0)
    assert allotments[0].unsayable is True
    assert allotments[0].vo_rate == VO_RATE_BAND[1]


def test_footage_vo_rate_targets_clip_plus_ceiling() -> None:
    # A footage beat's narration breathes into clip_dur + the freeze-pad ceiling
    # rather than compressing into the bare clip. clip=10s, 40 words ⇒ natural
    # 40/150*60 = 16s. The rate targets 10 + 1.5 = 11.5s, NOT the bare 10s.
    clip_dur = 10.0
    natural = 40 / DEFAULT_WPM_NATURAL * 60.0  # 16.0s
    assert natural == pytest.approx(16.0)
    allotments = reconcile_timeline([_pinned("clip-1", clip_dur, words=40)], target=clip_dur)
    expected_rate = natural / (clip_dur + DEFAULT_FREEZE_PAD_CEILING)  # 16 / 11.5
    # Rate lands strictly inside the band (≈1.39), so the assertion is exact, not clamped.
    assert VO_RATE_BAND[0] < expected_rate < VO_RATE_BAND[1]
    assert allotments[0].vo_rate == pytest.approx(expected_rate)
    # Gentler than compressing into the bare clip (16/10 = 1.6).
    assert (allotments[0].vo_rate or 0.0) < natural / clip_dur
    # The clip budget itself is UNCHANGED — only the rate moved.
    assert allotments[0].seconds == pytest.approx(clip_dur)
    assert allotments[0].unsayable is False
    # Step B: the same capped pad rides the allotment (16s natural, capped at the ceiling)
    # so the caller can bound the renderer's freeze-pad at clip + this value.
    assert allotments[0].freeze_pad_seconds == pytest.approx(DEFAULT_FREEZE_PAD_CEILING)


def test_footage_vo_rate_noop_when_narration_fits() -> None:
    # When the natural read already fits inside the clip, the freeze-pad ceiling
    # contributes 0 ⇒ the stamped rate is identical to the bare-clip target.
    clip_dur = 10.0
    natural = 20 / DEFAULT_WPM_NATURAL * 60.0  # 8.0s ≤ clip_dur ⇒ no pad
    assert natural <= clip_dur
    assert freeze_pad_ceiling(clip_dur, natural) == pytest.approx(0.0)
    allotments = reconcile_timeline([_pinned("clip-1", clip_dur, words=20)], target=clip_dur)
    bare_rate, _, _ = vo_rate_for(clip_dur, 20, DEFAULT_WPM_NATURAL)
    assert allotments[0].vo_rate == pytest.approx(bare_rate)
    # ⚠️ NOT ``natural / clip_dur`` (0.80). The line fits, so it is spoken at natural
    # 1.0 — never slowed. The pad arithmetic this test is about is unaffected: the clip
    # still needs no pad, and the allotment is still the bare clip.
    assert natural / clip_dur < VO_RATE_DRAG_FLOOR
    assert allotments[0].vo_rate == 1.0
    assert allotments[0].seconds == pytest.approx(clip_dur)
    # Narration fits ⇒ no pad ⇒ max_seconds stays the bare clip (byte-identical).
    assert allotments[0].freeze_pad_seconds == pytest.approx(0.0)


def test_freeze_pad_seconds_is_zero_for_non_footage_beats() -> None:
    # Only footage beats freeze-hold; a vo beat (elastic, has words) and a card beat
    # (elastic, no words) never carry a pad ⇒ their max_seconds is byte-identical to demand.max.
    demands = [
        _elastic("vo-1", 3.0, 5.0, 8.0, words=12),
        _elastic("card-1", 1.0, 2.0, 4.0),
        _pinned("footage-1", 6.0),  # footage WITHOUT narration ⇒ still no pad
    ]
    allotments = {a.id: a for a in reconcile_timeline(demands, target=12.0)}
    assert allotments["vo-1"].freeze_pad_seconds == pytest.approx(0.0)
    assert allotments["card-1"].freeze_pad_seconds == pytest.approx(0.0)
    assert allotments["footage-1"].freeze_pad_seconds == pytest.approx(0.0)


# --- freeze_pad_ceiling ----------------------------------------------------


@pytest.mark.parametrize(
    ("clip_dur", "allotment", "expected"),
    [(4.0, 5.0, 1.0)],
)
def test_freeze_pad_never_exceeds_the_ceiling(
    clip_dur: float, allotment: float, expected: float
) -> None:
    pad = freeze_pad_ceiling(clip_dur, allotment)
    assert pad == pytest.approx(expected)
    assert 0.0 <= pad <= DEFAULT_FREEZE_PAD_CEILING


def test_freeze_pad_ceiling_is_configurable() -> None:
    assert freeze_pad_ceiling(1.0, 10.0, ceiling=0.25) == pytest.approx(0.25)


def test_freeze_pad_raw_gap_can_exceed_the_ceiling() -> None:
    # The freeze-pad advisory in reconcile_timeline fires on ``(raw_gap - capped) > 0``: a
    # clip that cannot fill its allotment within the ceiling is flagged, never silently
    # stretched. Here a 5s raw gap is capped at 1.5s, so 3.5s remains uncoverable.
    clip_dur, allotment = 4.0, 9.0
    capped = freeze_pad_ceiling(clip_dur, allotment)
    assert capped == pytest.approx(DEFAULT_FREEZE_PAD_CEILING)
    assert (allotment - clip_dur) - capped == pytest.approx(3.5)  # raw pad exceeds the ceiling


# --- drift_report ----------------------------------------------------------


def test_drift_report_under_tolerance() -> None:
    planned = {"b1": 10.0, "b2": 10.0}
    measured = {"b1": 10.1, "b2": 9.95}
    report = drift_report(planned, measured)
    assert report.per_beat["b1"] == pytest.approx(0.1)
    assert report.per_beat["b2"] == pytest.approx(-0.05)
    assert report.total == pytest.approx(0.05)
    assert report.over_tolerance is False


def test_drift_report_over_tolerance() -> None:
    planned = {"b1": 10.0, "b2": 10.0}
    measured = {"b1": 14.0, "b2": 10.0}  # +20% on a 2% band
    report = drift_report(planned, measured)
    assert report.total == pytest.approx(4.0)
    assert report.over_tolerance is True


def test_drift_report_ignores_unmeasured_beats_per_beat() -> None:
    report = drift_report({"b1": 5.0, "b2": 5.0}, {"b1": 5.0})
    assert set(report.per_beat) == {"b1"}
    assert report.total == pytest.approx(-5.0)


# --- timeline_manifest -----------------------------------------------------


def _manifest(target: float) -> dict[str, Any]:
    # 26 words keeps `b2` OFF the underspoken path at every `target` this fixture is
    # exercised with (20.0 and 26.0, incl. saturating at its `max`=12.0) — this file's
    # shape/hash assertions are not about the underspoken-accounting fix (1d), which has
    # its own fixture above.
    demands = [
        _pinned("clip-1", 6.0),
        _elastic("b2", 2.0, 5.0, 12.0, words=26),
        _elastic("b3", 1.0, 3.0, 8.0),
    ]
    return timeline_manifest(reconcile_timeline(demands, target=target), demands, target=target)


def test_manifest_shape_and_totals() -> None:
    manifest = _manifest(20.0)
    beats = manifest["beats"]
    assert isinstance(beats, list)
    assert [b["id"] for b in beats] == ["clip-1", "b2", "b3"]
    for beat in beats:
        assert set(beat) == {
            "id",
            "planned",
            "demand_min",
            "demand_max",
            "allotment",
            "source",
            "infeasible",
        }
        assert beat["source"] in VALID_SOURCES
    assert manifest["total"] == pytest.approx(20.0)
    assert manifest["target"] == pytest.approx(20.0)
    assert manifest["drift"] == pytest.approx(0.0)
    assert manifest["infeasible_beats"] == []


def test_manifest_input_hash_is_stable_for_identical_inputs() -> None:
    assert _manifest(20.0)["input_hash"] == _manifest(20.0)["input_hash"]


def test_manifest_input_hash_changes_when_the_target_changes() -> None:
    assert _manifest(20.0)["input_hash"] != _manifest(26.0)["input_hash"]


def test_manifest_input_hash_separates_targets_that_saturate_identically() -> None:
    """C3 regression: when the solver saturates at Σmax the resolved TOTAL is identical
    for every target, so the old (total, demand-vector) hash made target=10 and
    target=1000 collide — a resume with a raised target would serve a STALE timeline.
    The hash is content-addressed by (target, demand-vector) now."""
    demands = [_elastic("b1", 1.0, 2.0, 3.0), _elastic("b2", 1.0, 2.0, 3.0)]
    small = reconcile_timeline(demands, target=10.0)
    huge = reconcile_timeline(demands, target=1000.0)

    # Both saturate at Σmax = 6.0 — the totals really are identical.
    assert sum(a.seconds for a in small) == pytest.approx(6.0)
    assert sum(a.seconds for a in huge) == pytest.approx(6.0)

    hash_small = timeline_manifest(small, demands, target=10.0)["input_hash"]
    hash_huge = timeline_manifest(huge, demands, target=1000.0)["input_hash"]
    assert hash_small != hash_huge


def test_manifest_lists_infeasible_beats() -> None:
    demands = [_elastic("b1", 10.0, 12.0, 14.0), _elastic("b2", 10.0, 12.0, 14.0)]
    manifest = timeline_manifest(reconcile_timeline(demands, target=12.0), demands, target=12.0)
    assert set(manifest["infeasible_beats"]) == {"b1", "b2"}
    assert manifest["drift"] == pytest.approx(8.0)


# --- build_timeline_manifest (C13 loop provenance, §7.7) -------------------


def test_build_timeline_manifest_joins_beats_to_demands() -> None:
    # Per-beat join: seconds/vo_rate/unsayable come from the Allotment; min/max
    # from the matching demand by id. A card beat's vo_rate is None ⇒ JSON null.
    demands = [
        _elastic("vo", 4.0, 8.0, 16.0, words=20),
        _elastic("card", 2.0, 5.0, 10.0),
    ]
    target = 24.0
    allotments = reconcile_timeline(demands, target=target)
    by_alloc = {a.id: a for a in allotments}

    manifest = build_timeline_manifest(
        demands,
        allotments,
        target_seconds=target,
        loop_iterations=3,
        converged=True,
        pacing_score=0.87,
    )
    beats = {b["id"]: b for b in manifest["beats"]}
    assert set(beats) == {"vo", "card"}

    assert beats["vo"]["seconds"] == pytest.approx(by_alloc["vo"].seconds)
    assert beats["vo"]["min_seconds"] == pytest.approx(4.0)
    assert beats["vo"]["max_seconds"] == pytest.approx(16.0)
    assert beats["vo"]["vo_rate"] == pytest.approx(by_alloc["vo"].vo_rate)
    assert beats["vo"]["feasible"] is True
    assert beats["vo"]["unsayable"] is False
    # A card beat has no words ⇒ vo_rate preserved as null.
    assert beats["card"]["vo_rate"] is None


def test_build_timeline_manifest_marks_infeasible_beats_not_feasible() -> None:
    # Σmin (20) > target (12) ⇒ both beats infeasible ⇒ feasible flag is False,
    # and the ids surface in the run-level infeasible_beat_ids.
    demands = [
        _elastic("b1", 10.0, 12.0, 14.0, words=40),
        _elastic("b2", 10.0, 12.0, 14.0, words=40),
    ]
    target = 12.0
    assert set(infeasible_beats(demands, target)) == {"b1", "b2"}
    allotments = reconcile_timeline(demands, target=target)

    manifest = build_timeline_manifest(
        demands,
        allotments,
        target_seconds=target,
        loop_iterations=5,
        converged=False,
        pacing_score=None,
    )
    assert all(b["feasible"] is False for b in manifest["beats"])
    assert set(manifest["run"]["infeasible_beat_ids"]) == {"b1", "b2"}


def test_build_timeline_manifest_carries_run_level_telemetry() -> None:
    demands = [_elastic("b1", 2.0, 4.0, 8.0), _elastic("b2", 2.0, 4.0, 8.0)]
    target = 10.0
    allotments = reconcile_timeline(demands, target=target)

    manifest = build_timeline_manifest(
        demands,
        allotments,
        target_seconds=target,
        loop_iterations=2,
        converged=True,
        pacing_score=None,
    )
    run = manifest["run"]
    assert set(run) == {
        "target_seconds",
        "loop_iterations",
        "converged",
        "pacing_score",
        "infeasible_beat_ids",
        "drift_seconds",
        "convergence_outcome",
        "structural_relief_requested",
        "overrun_seconds",
        "protected_beat_count",
        "structural_rejected_ids",
        "structural_reallocated_ids",
        "structural_patches_applied",
        "card_seconds",
    }
    assert run["target_seconds"] == pytest.approx(10.0)
    assert run["card_seconds"] == 0.0
    assert run["loop_iterations"] == 2
    assert run["converged"] is True
    assert run["pacing_score"] is None
    assert run["infeasible_beat_ids"] == []
    assert run["drift_seconds"] == pytest.approx(budget_drift(allotments, target))
    # New C13 telemetry fields default to a converged/no-structural-relief run.
    assert run["convergence_outcome"] == "converged"
    assert run["structural_relief_requested"] is False
    assert run["overrun_seconds"] == pytest.approx(0.0)
    assert run["protected_beat_count"] == 0  # backward-compatible default when not supplied
    assert run["structural_rejected_ids"] == []
    assert run["structural_reallocated_ids"] == []
    assert run["structural_patches_applied"] == 0
    # Whole manifest is JSON-serializable (no dataclasses / custom types).
    import json

    json.dumps(manifest)


# --- end-to-end: collect → reconcile → manifest ----------------------------


def test_collect_reconcile_end_to_end_hits_the_target() -> None:
    lines = [
        {"id": "clip-1", "clip": "@clip-1", "text": _words(20)},  # measured footage
        {"id": "b2", "text": _words(60)},  # vo
        {"id": "card-1", "seconds": 3.0},  # card
        {"id": "b4", "text": _words(40), "time_lock": True, "seconds": 9.0},  # authored pin
    ]
    demands = collect_demands(lines, clip_seconds={"clip-1": 8.0})
    # Inside [Σmin, Σmax] — the band the sum-to-target guarantee holds over.
    target = 45.0
    allotments = reconcile_timeline(demands, target=target)
    by_id = {a.id: a for a in allotments}

    assert by_id["clip-1"].seconds == 8.0  # C0 — measured footage untouched
    assert by_id["b4"].seconds == 9.0  # authored time_lock survives the solver
    assert by_id["b4"].source == "default"
    total = sum(a.seconds for a in allotments)
    assert abs(total - target) <= target * DEFAULT_TOLERANCE_PCT / 100.0
    assert all(isinstance(a, Allotment) for a in allotments)


# ── An authored window is sized at what compile PLAYS, not the whole file ─────────────


@pytest.mark.parametrize(
    ("window", "expected"),
    [({"clip_out_seconds": 5.0}, 5.0)],
)
def test_authored_window_beat_is_sized_at_its_window(
    window: dict[str, Any], expected: float
) -> None:
    line = {"id": "clip-1", "text": _words(8), **window}
    (demand,) = collect_demands([line], clip_seconds={"clip-1": 20.0})
    assert demand.kind == "footage"
    assert demand.min == demand.ideal == demand.max == pytest.approx(expected)
    if window:
        assert demand.clip_seconds is None, "an authored window is never tail-trimmable"
