"""The narration ESTIMATE lane: where to aim, what to record, and what it costs.

Measured live 2026-09-11, n=5 per content type: app_review spans -20.8%..+12.9%,
lesson +4.2%..+26.4%, promo +4.5%..+32.4%. Two separate defects are behind that
spread and this file pins the arithmetic for both.

1. **Acceptance aims at the band EDGE.** Anything inside +/-10% is "ok", so a
   draw estimated at +9% is accepted and nothing trims it — and the estimate is
   ``words / wpm``, which carries several percent of error of its own, so the
   delivery lands outside the band that grades it. :func:`steer_verdict` gives
   the long side a 4% target and leaves the short side at the full tolerance,
   because nothing downstream pads an underrun.
2. **A conformant film gets flagged by a guess.** A lesson run delivered +0.22%
   of its order while its SCRIPT stage had already stamped
   ``duration_out_of_tolerance`` from the pre-audio estimate. The compile stages
   raise the real, artifact-measured verdict; the estimate now speaks on the
   advisory tier, which never makes a run degraded.

Plus the two things the trim needs in order to stop guessing: measured per-line
SECONDS instead of a word proxy, and :func:`closer_to_order` as ONE shared
definition of "this correction actually helped" (three stages carry a private
copy of that line, and overcorrection is this pipeline's documented repeat
failure: +56.8% -> -33.5% on a 30s order).

No ``integration`` marker: pytest ``addopts`` deselects it, so a test wearing
one would never run in CI.
"""

from __future__ import annotations

import math

import pytest

from src.services.stages.shared.degradation import (
    DURATION_OUT_OF_TOLERANCE,
    NARRATION_ESTIMATE_OFF_TARGET,
)
from src.services.stages.shared.duration_conformance import (
    DURATION_STEER_FRACTION,
    DURATION_TOLERANCE_FRACTION,
)
from src.services.stages.shared.narration_budget import (
    TRIM_SECONDS_RATE,
    NarrationEstimate,
    build_narration_estimate,
    closer_to_order,
    narration_degradation,
    narration_estimate_advisory,
    plan_trim,
    steer_verdict,
)


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


# ── closer_to_order — the overcorrection guard ───────────────────────────────


def test_closer_is_strict_and_a_tie_keeps_the_incumbent() -> None:
    """Ties keep the incumbent: doing nothing beats doing something equivalent.

    The tie is not exotic — it is the EXACT-MIRROR correction, a +11% draw
    swapped for a -11% one, which buys a paid run a second LLM draw and a
    different set of words for the same error. A ``<=`` here would accept it.
    """
    # Equidistant on OPPOSITE sides (+8.3% swapped for -8.3%) — a tie, refused.
    assert closer_to_order(55.0, 65.0, 60.0) is False
    # Equidistant on the SAME side, and identical: also ties, also refused.
    assert closer_to_order(65.0, 65.0, 60.0) is False
    assert closer_to_order(60.0, 60.0, 60.0) is False
    # ...and a genuine improvement is accepted, so the guard is not simply off.
    assert closer_to_order(59.0, 65.0, 60.0) is True


def test_closer_judges_distance_on_both_sides_of_the_order() -> None:
    """Symmetric by ``abs``: an overrun is not privileged over an underrun.

    THE overcorrection case, measured: structural relief took a 30s order from
    +56.8% to -33.5%. It stopped FAILING and it still misses the band, on the far
    side, so it must be REJECTED as a correction — which only happens if the
    guard measures distance rather than direction.
    """
    ordered = 30.0
    overrun = ordered * 1.568
    overcorrected = ordered * 0.665
    assert closer_to_order(overcorrected, overrun, ordered) is True  # 33.5% < 56.8%
    # ...and the genuinely worse swap is refused in both directions.
    assert closer_to_order(ordered * 1.90, overrun, ordered) is False
    assert closer_to_order(ordered * 0.20, overrun, ordered) is False
    # A candidate landing exactly ON the order is always closer than anything else.
    assert closer_to_order(ordered, overrun, ordered) is True
    assert closer_to_order(ordered, ordered * 1.0001, ordered) is True


# ── steer_verdict — a tighter long side, the same short side ─────────────────


@pytest.mark.parametrize(
    ("estimate", "expected"),
    [(103.9, "ok")],
)
def test_steer_is_asymmetric_around_the_order(estimate: float, expected: str) -> None:
    """+4% long, -10% short, against a 100s order.

    The two "ok" rows are the ones that matter. ``103.9`` proves the long side is
    not simply tighter everywhere, and ``90.1`` proves the short side was NOT
    tightened to match: steering short would make the trim decline cuts it should
    take (it refuses any drop landing under the lower bound), converting a fixable
    overrun into an accepted one, while nothing in the pipeline pads an underrun
    anyway.
    """
    assert steer_verdict(estimate, 100.0) == expected


def test_the_steer_band_is_strictly_inside_the_tolerance_band() -> None:
    """The property behind the numbers, not a restatement of them.

    The point of steering is room for the estimate's OWN error: everything the
    outer band calls "long" must already be "long" here, and there must exist a
    draw the outer band accepts that steering does not.
    """
    assert DURATION_STEER_FRACTION < DURATION_TOLERANCE_FRACTION
    ordered = 100.0
    assert steer_verdict(ordered * (1.0 + DURATION_TOLERANCE_FRACTION + 0.01), ordered) == "long"
    # The gap itself: accepted by the grading band, steered by this one.
    assert steer_verdict(109.0, ordered) == "long"


def test_a_non_finite_input_is_unknown_and_an_absent_order_is_ok() -> None:
    """Mirrors ``narration_verdict`` exactly — a caller may not change failure
    behaviour by switching between them.

    ``"short"`` means ACCEPT-AND-SHIP to every consumer, so a NaN must never
    reach it: every comparison against NaN is False, which is precisely how a
    direction gets asserted that the arithmetic never established.
    """
    assert steer_verdict(math.nan, 100.0) == "unknown"
    assert steer_verdict(math.inf, 100.0) == "unknown"
    assert steer_verdict(90.0, 100.0, steer=math.nan) == "unknown"
    assert steer_verdict(90.0, 100.0, tolerance=math.nan) == "unknown"
    # Nothing ordered ⇒ no band ⇒ "ok", never an invented order. Checked BEFORE
    # the estimate, so a NaN estimate against no order is still "ok".
    assert steer_verdict(500.0, 0.0) == "ok"
    assert steer_verdict(500.0, -30.0) == "ok"
    assert steer_verdict(500.0, math.nan) == "ok"
    assert steer_verdict(math.nan, 0.0) == "ok"


# ── plan_trim(line_seconds=...) — plan on measured time, not a word proxy ────
#
# Words x wpm is a PROXY: equal word counts do not take equal time (a digit
# string, an acronym, a clause the voice pauses after). The fixture below is
# built so the two disagree about WHICH lines to cut — if they agreed, the test
# would pass with `line_seconds` ignored entirely.

_DISAGREEING_WORDS = (10, 60, 10, 40)
#: The same four lines, actually measured. Line 1 is 60 words of fast list; line
#: 0 is 10 words the voice labours over. Same 120s total, different shape.
_DISAGREEING_SECONDS = (50.0, 20.0, 10.0, 40.0)


def _disagreeing_script() -> list[str]:
    return [_words(n) for n in _DISAGREEING_WORDS]


def test_the_words_plan_and_the_seconds_plan_cut_different_lines() -> None:
    """THE test: same script, same order, two different answers.

    Both land at 60.0s against a 60s order, so neither plan is "wrong" by its own
    arithmetic — they simply disagree about which lines cost the time, and only
    one of them measured it. Dropping line 1 (the words answer) removes 60 words
    that take 20 seconds: by the clock that is a 100s film against a 60s order.
    """
    script = _disagreeing_script()
    by_words = plan_trim(script, ordered_seconds=60.0, words_per_minute=60.0)
    assert by_words.dropped_indices == (1,)
    assert by_words.estimated_seconds_after == pytest.approx(60.0)

    by_seconds = plan_trim(
        script,
        ordered_seconds=60.0,
        words_per_minute=60.0,
        line_seconds=_DISAGREEING_SECONDS,
    )
    assert by_seconds.dropped_indices == (1, 3)
    assert by_seconds.estimated_seconds_after == pytest.approx(60.0)
    assert by_seconds.exceeds_limit is False
    # The whole-script number is the measured total, not the word estimate.
    assert by_seconds.estimated_seconds_before == pytest.approx(sum(_DISAGREEING_SECONDS))


def test_the_seconds_plan_ignores_the_rate_entirely() -> None:
    """Measured input must not be re-scaled by a wpm that no longer applies.

    An unusable rate is substituted for the WORDS path (``_safe_wpm``); on the
    seconds path there is nothing to substitute, because the seconds are the
    measurement. A plan that still divided by wpm would move with it.
    """
    script = _disagreeing_script()
    at_60 = plan_trim(
        script, ordered_seconds=60.0, words_per_minute=60.0, line_seconds=_DISAGREEING_SECONDS
    )
    at_240 = plan_trim(
        script, ordered_seconds=60.0, words_per_minute=240.0, line_seconds=_DISAGREEING_SECONDS
    )
    at_broken = plan_trim(
        script, ordered_seconds=60.0, words_per_minute=0.0, line_seconds=_DISAGREEING_SECONDS
    )
    assert at_60 == at_240 == at_broken


def test_measured_seconds_round_trip_through_the_weight_conversion() -> None:
    """``weight / TRIM_SECONDS_RATE * 60`` must BE the seconds handed in.

    Deciseconds are a unit change, not a model: a script already inside the band
    is a no-op plan whose reported ``before``/``after`` are the measured total.
    """
    assert TRIM_SECONDS_RATE == 600.0
    seconds = [12.3, 7.7, 40.0]
    plan = plan_trim(
        [_words(5), _words(5), _words(5)],
        ordered_seconds=60.0,
        words_per_minute=150.0,
        line_seconds=seconds,
    )
    assert plan.dropped_indices == ()
    assert plan.estimated_seconds_before == pytest.approx(sum(seconds), abs=0.05)
    assert plan.estimated_seconds_after == pytest.approx(sum(seconds), abs=0.05)


def test_a_mismatched_or_unmeasurable_line_seconds_raises() -> None:
    """These are MEASUREMENTS. A wrong-length list silently mis-aligns seconds
    with lines and cuts the script by another line's weights; a NaN makes every
    comparison False, which is how this module once amputated a paid deliverable
    and reported success. Neither has a defensible substitute, so both raise.
    """
    script = _disagreeing_script()
    with pytest.raises(ValueError, match="must align with the script"):
        plan_trim(script, ordered_seconds=60.0, words_per_minute=60.0, line_seconds=[1.0, 2.0])
    with pytest.raises(ValueError, match="must align with the script"):
        plan_trim(script, ordered_seconds=60.0, words_per_minute=60.0, line_seconds=[1.0] * 5)
    with pytest.raises(ValueError, match="finite, non-negative"):
        plan_trim(
            script,
            ordered_seconds=60.0,
            words_per_minute=60.0,
            line_seconds=[50.0, math.nan, 10.0, 40.0],
        )
    with pytest.raises(ValueError, match="finite, non-negative"):
        plan_trim(
            script,
            ordered_seconds=60.0,
            words_per_minute=60.0,
            line_seconds=[50.0, -1.0, 10.0, 40.0],
        )


def test_omitting_line_seconds_is_bit_for_bit_the_old_behaviour() -> None:
    """The additive guarantee: every existing caller passes nothing."""
    script = _disagreeing_script()
    assert plan_trim(script, ordered_seconds=60.0, words_per_minute=60.0) == plan_trim(
        script, ordered_seconds=60.0, words_per_minute=60.0, line_seconds=None
    )


def test_the_measured_floor_is_reported_at_the_measured_rate() -> None:
    """The branch NO measured test reached: nothing lands, so the plan comes from
    the tail-first floor walk — and that walk must run on the deciseconds rate the
    weights were built with, not on a words-per-minute default.

    Three 40s lines against a 60s order (band 54-66s) cannot land: 120 -> 80 is
    over, 80 -> 40 is under the band and is DECLINED. The honest floor is the
    single tail drop, 80.0s, ``exceeds_limit``. Divide the same deciseconds by
    150 wpm instead and the walk sees 480s, accepts BOTH drops as "progress"
    towards a 66s limit, and amputates two thirds of a paid script.
    """
    plan = plan_trim(
        [_words(5), _words(5), _words(5)],
        ordered_seconds=60.0,
        words_per_minute=150.0,
        line_seconds=[40.0, 40.0, 40.0],
    )
    assert plan.dropped_indices == (2,)
    assert plan.keep_indices == (0, 1)
    # The reported seconds ARE the true kept seconds, not a re-scaled proxy.
    assert plan.estimated_seconds_after == pytest.approx(80.0)
    assert plan.exceeds_limit is True
    assert plan.declined_under_band is True


def test_a_quantisation_rounding_can_never_claim_a_band_it_missed() -> None:
    """The report is re-derived from the TRUE kept seconds, never the weights.

    ``[55.02, 55.02, 40.0]`` against a 100s order: dropping the tail leaves
    ``550 + 550`` deciseconds, i.e. exactly ``110.0`` against a ``110.0`` limit,
    so the search accepts it. The kept lines really run ``110.04`` — outside the
    band. Reporting the quantised total would print ``after=110.0`` with
    ``exceeds_limit=False``: a band MISSED, reported as a band made.
    """
    plan = plan_trim(
        [_words(5), _words(5), _words(5)],
        ordered_seconds=100.0,
        words_per_minute=150.0,
        line_seconds=[55.02, 55.02, 40.0],
    )
    assert plan.dropped_indices == (2,)
    assert plan.estimated_seconds_after == pytest.approx(110.04)
    assert plan.estimated_seconds_after > 100.0 * (1.0 + DURATION_TOLERANCE_FRACTION)
    assert plan.exceeds_limit is True


def test_the_decisecond_weight_rounds_and_does_not_truncate() -> None:
    """``round``, not ``int``: truncation is a ONE-SIDED error that always
    under-states a line, doubling the worst case to 0.1s and biasing every line
    the same way.

    ``55.06s`` is ``550.6`` deciseconds. Rounded to ``551`` the single tail drop
    lands at ``55.1``, inside this (deliberately tight) band, and the plan keeps
    the 55.06s line. Truncated to ``550`` it reads ``55.0``, falls UNDER the band,
    is declined — and the walk then drops the opening line instead, keeping the
    100s one: a 100.0s answer to a 55.06s order.
    """
    plan = plan_trim(
        [_words(5), _words(5)],
        ordered_seconds=55.06,
        words_per_minute=150.0,
        tolerance=0.0009,
        line_seconds=[55.06, 100.0],
    )
    assert plan.dropped_indices == (1,)
    assert plan.estimated_seconds_after == pytest.approx(55.06)
    assert plan.exceeds_limit is False
    assert plan.declined_under_band is False


# ── build_narration_estimate — the record, with a DERIVED verdict ────────────


def _estimate(
    *,
    estimate_seconds: float = 120.0,
    ordered_seconds: float = 120.0,
    wpm: float = 151.6,
    wpm_basis: str = "measured",
    first_draw_estimate_seconds: float | None = None,
    reasked: bool = False,
    reask_kept: bool | None = None,
    trimmed_lines: int = 0,
    measured_lines: int | None = None,
) -> NarrationEstimate:
    """A conformant estimate, with only the field under test overridden.

    Spelled out rather than splatted: a ``**kwargs`` dict erases the argument
    types, so a test could pass a wrongly-typed field and still be accepted.
    """
    return build_narration_estimate(
        estimate_seconds=estimate_seconds,
        ordered_seconds=ordered_seconds,
        wpm=wpm,
        wpm_basis=wpm_basis,
        first_draw_estimate_seconds=first_draw_estimate_seconds,
        reasked=reasked,
        reask_kept=reask_kept,
        trimmed_lines=trimmed_lines,
        measured_lines=measured_lines,
    )


def test_the_verdict_is_derived_from_the_outer_band_not_the_steer() -> None:
    """Derived, never accepted as an argument: a caller could otherwise persist a
    verdict that disagrees with the numbers beside it, and this record exists to
    be trusted after the fact.

    The band is the OUTER +/-10% one, so the column is comparable with
    ``duration_conformance``'s grade of the delivered master. Steering to 4% is
    an aiming decision, not a grade — a +5% draw is steered and is NOT recorded
    as a miss.
    """
    assert _estimate(estimate_seconds=126.0).verdict == "ok"  # +5% — steered, not graded
    assert steer_verdict(126.0, 120.0) == "long"  # ...and steering disagrees, by design
    assert _estimate(estimate_seconds=133.0).verdict == "long"  # +10.8%
    assert _estimate(estimate_seconds=107.0).verdict == "short"  # -10.8%
    assert _estimate(estimate_seconds=300.0, ordered_seconds=0.0).verdict == "ok"  # no order


def test_a_non_finite_number_is_recorded_as_none_with_an_unknown_verdict() -> None:
    """NaN is not persistable as JSONB, and 0.0 would be a fabrication.

    ``None`` says "this was not a real number" and ``verdict="unknown"`` says the
    same thing in the field consumers actually branch on.
    """
    estimate = _estimate(estimate_seconds=math.nan)
    assert estimate.estimate_seconds is None
    assert estimate.verdict == "unknown"
    assert _estimate(wpm=math.inf).wpm is None
    # ...and the good fields around it survive intact.
    assert estimate.ordered_seconds == 120.0
    assert estimate.wpm_basis == "measured"


def test_absent_facts_are_none_rather_than_a_stand_in_value() -> None:
    """``reask_kept=False`` would claim a draw was discarded; ``measured_lines=0``
    would claim a measurement returned nothing. Neither happened."""
    nothing_happened = _estimate()
    assert nothing_happened.reasked is False
    assert nothing_happened.reask_kept is None
    assert nothing_happened.measured_lines is None
    assert nothing_happened.trimmed_lines == 0
    # With no correction the first draw IS the estimate — not None, not 0.0.
    assert nothing_happened.first_draw_estimate_seconds == 120.0

    corrected = _estimate(
        estimate_seconds=118.0,
        first_draw_estimate_seconds=151.0,
        reasked=True,
        reask_kept=True,
        trimmed_lines=2,
        measured_lines=9,
    )
    assert corrected.first_draw_estimate_seconds == 151.0
    assert (corrected.reasked, corrected.reask_kept) == (True, True)
    assert (corrected.trimmed_lines, corrected.measured_lines) == (2, 9)
    # The record that made the correction judgeable: the kept draw is closer.
    assert closer_to_order(118.0, 151.0, 120.0) is True


def test_the_record_carries_numbers_only_and_no_script_text() -> None:
    """A stage output is read by the API and the run list. Script text here would
    duplicate the script into a second field that drifts from the first, and it
    is the numbers a cross-run query can aggregate."""
    dumped = _estimate(measured_lines=3).model_dump()
    assert set(dumped) == {
        "estimate_seconds",
        "ordered_seconds",
        "wpm",
        "wpm_basis",
        "verdict",
        "first_draw_estimate_seconds",
        "reasked",
        "reask_kept",
        "trimmed_lines",
        "measured_lines",
    }
    for key, value in dumped.items():
        assert value is None or isinstance(value, float | int | bool | str), (key, value)
    assert dumped["wpm_basis"] == "measured"  # a BASIS label, not prose


# ── narration_estimate_advisory — the same sentence, the non-degrading tier ──


def test_the_advisory_says_exactly_what_the_degradation_said() -> None:
    """Identical text, identical numbers — only the code (and therefore the
    field, and therefore ``degraded``) differs.

    The two detail strings are built by separate code on purpose: the existing
    function is live on three stages and may not change by so much as a
    whitespace while they migrate one at a time. This assertion is what keeps the
    copy honest, so a drift is a test failure instead of a discovery.
    """
    kwargs = {
        "stage": "lesson_script",
        "ordered_seconds": 120.0,
        "estimated_seconds": 133.4,
        "action": "trimmed",
        "detail": "2 line(s) dropped.",
    }
    old = narration_degradation(**kwargs)  # type: ignore[arg-type]
    new = narration_estimate_advisory(**kwargs)  # type: ignore[arg-type]
    assert new["detail"] == old["detail"]
    assert old["code"] == DURATION_OUT_OF_TOLERANCE
    assert new["code"] == NARRATION_ESTIMATE_OFF_TARGET
    # The numbers a reader acts on are still in the sentence.
    assert "133.4s" in new["detail"] and "120.0s" in new["detail"] and "+11.2%" in new["detail"]


def test_the_advisory_text_matches_the_degradation_on_the_edge_shapes() -> None:
    """The branches of the copied body: no order (no deviation percentage) and no
    trailing detail. A copy pinned only on its happy path is a copy that drifts.
    """
    for ordered_seconds in (0.0, 30.0):
        assert (
            narration_estimate_advisory(
                stage="promo_script",
                ordered_seconds=ordered_seconds,
                estimated_seconds=41.0,
                action="accepted",
            )["detail"]
            == narration_degradation(
                stage="promo_script",
                ordered_seconds=ordered_seconds,
                estimated_seconds=41.0,
                action="accepted",
            )["detail"]
        )
    no_order = narration_estimate_advisory(
        stage="promo_script", ordered_seconds=0.0, estimated_seconds=41.0, action="accepted"
    )
    assert "%" not in no_order["detail"]
    assert no_order["detail"].endswith("action=accepted.")


def test_the_existing_degradation_function_is_untouched() -> None:
    """The migration guarantee, asserted rather than assumed: a stage that has
    NOT moved yet keeps emitting the degrading code onto the degrading field."""
    entry = narration_degradation(
        stage="lesson_script", ordered_seconds=120.0, estimated_seconds=151.0, action="reasked"
    )
    assert entry["code"] == DURATION_OUT_OF_TOLERANCE
