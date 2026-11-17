"""narration_budget: the arithmetic that makes a narration script fit the order.

Measured live 2026-09-11 — lesson ordered 120s delivered 95.54s, ordered 360s
delivered 399.68s, website_review ordered 60s delivered 134.2s. Nothing trimmed
and nothing extended, because the word budget was prompt prose. These tests pin
the reconciliation arithmetic: the estimate, the verdict band, and the
deterministic last-first trim with its two inviolable rules (a protected line is
never dropped; the script never becomes empty).
"""

from __future__ import annotations

import math

import pytest

from src.services.stages.shared import narration_budget
from src.services.stages.shared.degradation import (
    DURATION_OUT_OF_TOLERANCE,
    ORDERED_DURATION_INFEASIBLE,
    SPEECH_RATE_UNUSABLE,
    is_degraded,
)
from src.services.stages.shared.duration_conformance import DURATION_TOLERANCE_FRACTION
from src.services.stages.shared.narration_budget import (
    DEFAULT_WORDS_PER_MINUTE,
    WORDS_PER_MINUTE_CEILING,
    WORDS_PER_MINUTE_FLOOR,
    budget_words,
    estimate_seconds,
    narration_degradation,
    narration_verdict,
    plan_trim,
    speech_rate_degradation,
    trim_floor_degradation,
    wpm_is_usable,
)
from src.services.stages.shared.time_budget import DEFAULT_WPM_NATURAL


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


# ── estimate / budget ────────────────────────────────────────────────────────


def test_estimate_is_words_over_rate_times_sixty() -> None:
    """171 words at 171 wpm is exactly one minute."""
    assert estimate_seconds([_words(171)], words_per_minute=171.0) == 60.0
    # Split across lines changes nothing — the word count is what matters.
    split = estimate_seconds([_words(85), _words(86)], words_per_minute=171.0)
    assert split == 60.0


def test_budget_words_is_the_inverse_of_the_estimate() -> None:
    assert budget_words(60.0, words_per_minute=171.0) == 171
    assert budget_words(360.0, words_per_minute=171.0) == 1026
    # Nothing ordered affords nothing — never a negative word count.
    assert budget_words(0.0, words_per_minute=171.0) == 0
    assert budget_words(-30.0, words_per_minute=171.0) == 0


def test_an_unusable_rate_substitutes_the_default_not_a_band_edge() -> None:
    """wpm<=0/NaN/out-of-band must not divide by zero — and must not clamp to 60.

    The floor was justified as "conservative". It is the opposite: the estimate is
    ``words / wpm * 60``, so the LOW edge INFLATES it — at 60 wpm a legitimate
    190-word script estimates at 190s, and against a 60s order the trimmer cuts it
    to ~66 words and delivers ~26s. The clamp meant to be safe destroys the
    deliverable. An out-of-band rate carries no information about which direction
    the truth lies in, so the substitute is the cross-type default.
    """
    at_default = estimate_seconds([_words(120)], words_per_minute=DEFAULT_WORDS_PER_MINUTE)
    for broken in (0.0, -171.0, -0.0, math.nan):
        got = estimate_seconds([_words(120)], words_per_minute=broken)
        assert got == at_default
        assert got > 0.0
        assert not math.isnan(got)
    assert budget_words(60.0, words_per_minute=0.0) == round(DEFAULT_WORDS_PER_MINUTE)
    # Both bands edges are substituted, not clamped to: an absurdly fast rate is
    # NOT honoured as 300 either.
    assert estimate_seconds([_words(600)], words_per_minute=100_000.0) == estimate_seconds(
        [_words(600)], words_per_minute=DEFAULT_WORDS_PER_MINUTE
    )
    assert estimate_seconds([_words(600)], words_per_minute=5.0) == estimate_seconds(
        [_words(600)], words_per_minute=DEFAULT_WORDS_PER_MINUTE
    )
    # ...and the destructive shape the old clamp produced is gone: 190 words
    # against a 60s order no longer estimates at 190s.
    assert estimate_seconds([_words(190)], words_per_minute=0.0) == pytest.approx(76.0)


def test_the_substituted_rate_is_the_same_default_the_stages_pass() -> None:
    """Two constants for one quantity is how they end up disagreeing."""
    assert DEFAULT_WORDS_PER_MINUTE == DEFAULT_WPM_NATURAL


def test_wpm_is_usable_is_the_predicate_a_caller_reports_on() -> None:
    """A substitution the caller cannot detect is a silent substitution."""
    assert wpm_is_usable(171.0)
    assert wpm_is_usable(WORDS_PER_MINUTE_FLOOR)
    assert wpm_is_usable(WORDS_PER_MINUTE_CEILING)
    assert not wpm_is_usable(0.0)
    assert not wpm_is_usable(-171.0)
    assert not wpm_is_usable(math.nan)
    assert not wpm_is_usable(WORDS_PER_MINUTE_FLOOR - 0.1)
    assert not wpm_is_usable(WORDS_PER_MINUTE_CEILING + 0.1)


def test_the_speech_rate_entry_names_the_rejected_rate_and_the_substitute() -> None:
    entry = speech_rate_degradation(stage="lesson_script", words_per_minute=0.0)
    assert entry["code"] == SPEECH_RATE_UNUSABLE
    assert "0.0" in entry["detail"]
    assert "150" in entry["detail"]
    assert "lesson_script" in entry["detail"]
    assert is_degraded({"degradations": [entry]})


# ── verdict ──────────────────────────────────────────────────────────────────


def test_verdict_band_is_inclusive_at_exactly_the_tolerance() -> None:
    assert narration_verdict(100.0, 100.0) == "ok"
    assert narration_verdict(110.0, 100.0) == "ok"  # exactly +10%
    assert narration_verdict(90.0, 100.0) == "ok"  # exactly -10%
    assert narration_verdict(110.01, 100.0) == "long"
    assert narration_verdict(89.99, 100.0) == "short"


def test_verdict_reproduces_the_measured_live_deviations() -> None:
    """The four measured rows. Three breach the band; the -9.0% one does not."""
    assert narration_verdict(95.54, 120.0) == "short"  # -20.4%
    assert narration_verdict(399.68, 360.0) == "long"  # +11.0%, just outside +10%
    assert narration_verdict(382.31, 420.0) == "ok"  # -9.0% IS inside the band
    assert narration_verdict(134.2, 60.0) == "long"  # +123.7%


def test_nothing_ordered_is_ok_not_an_invented_order() -> None:
    assert narration_verdict(500.0, 0.0) == "ok"
    assert narration_verdict(500.0, -1.0) == "ok"


def test_a_custom_tolerance_moves_the_band() -> None:
    assert narration_verdict(105.0, 100.0, tolerance=0.10) == "ok"
    assert narration_verdict(105.0, 100.0, tolerance=0.01) == "long"


# ── plan_trim ────────────────────────────────────────────────────────────────


def test_within_tolerance_is_a_no_op_plan_keeping_every_line() -> None:
    texts = [_words(50), _words(50), _words(71)]  # 171 words = 60s at 171 wpm
    plan = plan_trim(texts, ordered_seconds=60.0, words_per_minute=171.0)
    assert plan.keep_indices == (0, 1, 2)
    assert plan.dropped_indices == ()
    assert plan.estimated_seconds_after == plan.estimated_seconds_before == 60.0
    assert plan.ordered_seconds == 60.0


def test_the_tolerance_headroom_is_actually_spent() -> None:
    """126s of script against a 120s order is +5% — inside the band, no cut."""
    texts = [_words(171), _words(188)]  # 359 words -> 125.96s
    plan = plan_trim(texts, ordered_seconds=120.0, words_per_minute=171.0)
    assert plan.estimated_seconds_before > 120.0
    assert plan.dropped_indices == ()


def test_a_protected_index_is_never_dropped_even_when_it_is_last() -> None:
    texts = [_words(171)] * 4
    plan = plan_trim(
        texts,
        ordered_seconds=60.0,
        words_per_minute=171.0,
        protected_indices=(3,),
    )
    assert 3 in plan.keep_indices
    assert 3 not in plan.dropped_indices
    # Protection is the ONLY thing that saves a line: with 3 pinned, the trim
    # walks past it and takes 2, 1 and finally 0 to reach the band.
    assert plan.dropped_indices == (0, 1, 2)
    assert plan.keep_indices == (3,)


def test_protection_can_block_the_trim_entirely_and_that_is_reported() -> None:
    """Every line protected ⇒ nothing droppable; the plan says so honestly."""
    texts = [_words(171)] * 4
    plan = plan_trim(
        texts,
        ordered_seconds=60.0,
        words_per_minute=171.0,
        protected_indices=(0, 1, 2, 3),
    )
    assert plan.dropped_indices == ()
    assert plan.estimated_seconds_after == plan.estimated_seconds_before == 240.0
    # ...and "honestly" means a FIELD, not prose: 240s against a 60s order is
    # still 4x long, and a caller must be able to ASK rather than re-derive it.
    assert plan.exceeds_limit is True
    assert plan.protected_count == 4


def test_a_trim_that_reaches_the_band_does_not_claim_a_floor() -> None:
    """The success case: exceeds_limit False, so no floor is ever reported."""
    plan = plan_trim([_words(171)] * 4, ordered_seconds=60.0, words_per_minute=171.0)
    assert plan.dropped_indices == (1, 2, 3)
    assert plan.estimated_seconds_after == 60.0
    assert plan.exceeds_limit is False
    assert plan.declined_under_band is False
    assert plan.protected_count == 0


def test_the_live_website_review_shape_60s_ordered_delivered_long() -> None:
    """The measured failure: ordered 60s, protected prefix, best trim still > 66s.

    ``website_review/script.py`` pins every authored shotlist line, an UNBOUNDED
    prefix. Six 30-word lines at 60 wpm = 180s; the first four are authored, so
    the trim can only reach 120s — and the live run reported that as
    "trimmed 2 line(s)", indistinguishable from one that landed.
    """
    texts = [_words(30)] * 6
    plan = plan_trim(
        texts,
        ordered_seconds=60.0,
        words_per_minute=60.0,
        protected_indices=(0, 1, 2, 3),
    )
    assert plan.dropped_indices == (4, 5), "it dropped everything it was allowed to"
    assert plan.estimated_seconds_after == 120.0
    assert plan.estimated_seconds_after > 60.0 * 1.1, "and is STILL over the limit"
    assert plan.exceeds_limit is True
    assert plan.protected_count == 4
    assert plan.declined_under_band is False, "protection is the floor here, not the band"


def test_a_drop_that_would_land_under_the_band_is_declined() -> None:
    """The overshoot this pipeline keeps re-committing, at the plan level.

    Three 40-word lines at 60 wpm against a 60s order: 120 -> 80 (still over) ->
    40, which is -33.3%. The old loop took that drop and reported a successful
    trim. It must stop at 80s and SAY the band was not reached.
    """
    plan = plan_trim([_words(40)] * 3, ordered_seconds=60.0, words_per_minute=60.0)
    assert plan.dropped_indices == (2,), "the second drop was declined, not taken"
    assert plan.keep_indices == (0, 1)
    assert plan.estimated_seconds_after == 80.0
    assert plan.declined_under_band is True
    assert plan.exceeds_limit is True


def test_the_under_band_guard_holds_for_unequal_line_lengths() -> None:
    """Uniform lines land exactly on a band edge and cannot express the defect.

    50/30/40 words at 60 wpm against a 60s order (band 54-66s): 120 -> 80 -> 50.
    50s is -16.7%, under the band, so the walk stops at 80s.
    """
    plan = plan_trim(
        [_words(50), _words(30), _words(40)], ordered_seconds=60.0, words_per_minute=60.0
    )
    assert plan.dropped_indices == (2,)
    assert plan.estimated_seconds_after == 80.0
    assert plan.declined_under_band is True


def test_a_declined_drop_is_skipped_and_the_walk_takes_the_next_legal_one() -> None:
    """The guard DECLINES a line; it does not abandon the trim.

    A declined drop is skipped exactly as a protected index is. Stopping instead
    forfeited plans that were legal on BOTH edges and then reported the order as
    unreachable — a degradation the arithmetic manufactured.
    """
    # 16/20/46 words at 60 wpm = 82s against a 60s order (band 54-66s). The tail
    # line is 46s, so dropping it lands at 36s — under the band, declined. Line 1
    # lands at 62s, squarely in band, and is the plan.
    texts = [_words(16), _words(20), _words(46)]
    plan = plan_trim(texts, ordered_seconds=60.0, words_per_minute=60.0)
    assert plan.dropped_indices == (1,), "skipped the declined tail, took the legal drop"
    assert plan.keep_indices == (0, 2)
    assert plan.estimated_seconds_after == pytest.approx(62.0)
    assert plan.declined_under_band is True, "the refusal is still on the record"
    assert plan.exceeds_limit is False, "and the band WAS reached"
    # Not vacuous: the declined drop really was the one the walk reached first.
    assert estimate_seconds(texts[:2], words_per_minute=60.0) == pytest.approx(36.0)


def test_the_guard_never_drops_more_lines_than_an_unguarded_walk_would() -> None:
    """The safety property: subtraction only, never an extra deletion.

    Continuing past a declined line cannot over-trim, because every accepted drop
    is still individually gated on the lower bound and the walk still stops the
    moment the estimate is at or under the limit.
    """

    def _unguarded_drop_count(texts: list[str], *, ordered: float, wpm: float) -> int:
        """Last-first until the limit is met — the trimmer with no band guard."""
        kept = list(range(len(texts)))
        count = 0
        for index in reversed(range(len(texts))):
            if estimate_seconds([texts[i] for i in kept], words_per_minute=wpm) <= ordered * 1.1:
                break
            if len(kept) <= 1:
                break
            kept.remove(index)
            count += 1
        return count

    cases = [
        [_words(16), _words(20), _words(46)],
        [_words(40), _words(40), _words(125)],
        [_words(40)] * 3,
        [_words(50), _words(30), _words(40)],
        [_words(10), _words(90), _words(5)],
        [_words(100), _words(45), _words(30)],
    ]
    for texts in cases:
        plan = plan_trim(texts, ordered_seconds=60.0, words_per_minute=60.0)
        assert len(plan.dropped_indices) <= _unguarded_drop_count(texts, ordered=60.0, wpm=60.0), (
            f"the guard added a deletion for {[len(t.split()) for t in texts]}"
        )


def test_the_declined_tail_does_not_forfeit_an_in_band_plan() -> None:
    """The defect the ``break`` caused, on the audit's exact input.

    40/40/125 words at 150 wpm is 16s/16s/50s = 82.0s against a 60s order (band
    54-66s). The walk reaches the 50s tail first and declines it (32.0s is under
    the band) — and the ``break`` then returned NOTHING dropped at 82.0s with
    ``exceeds_limit``, so the caller announced "the ordered duration is not
    reachable by trimming this script". Dropping the MIDDLE line alone lands on
    66.0s, inside the band on both edges. The order was always reachable.
    """
    texts = [_words(40), _words(40), _words(125)]
    plan = plan_trim(texts, ordered_seconds=60.0, words_per_minute=150.0)
    assert plan.estimated_seconds_before == pytest.approx(82.0)
    assert plan.dropped_indices == (1,)
    assert plan.keep_indices == (0, 2)
    assert plan.estimated_seconds_after == pytest.approx(66.0)
    assert plan.exceeds_limit is False, "the band was reached — no floor to report"
    assert narration_verdict(plan.estimated_seconds_after, 60.0) == "ok"
    # The declined drop is still recorded, and it really would have overshot.
    assert plan.declined_under_band is True
    assert estimate_seconds(texts[:2], words_per_minute=150.0) == pytest.approx(32.0)


def test_a_genuinely_unreachable_order_still_reports_the_floor() -> None:
    """The fix must not disable the signal — only stop it firing on reachable orders.

    10/90/5 words at 60 wpm = 105.0s against a 60s order (band 54-66s). Every
    subset that keeps the 90s line is over the band and every subset that drops
    it is under, so no plan can land: the walk takes the 5s tail, DECLINES the
    90s line, takes the 10s opener, and ends at 90.0s.
    """
    texts = [_words(10), _words(90), _words(5)]
    plan = plan_trim(texts, ordered_seconds=60.0, words_per_minute=60.0)
    assert plan.dropped_indices == (0, 2)
    assert plan.estimated_seconds_after == pytest.approx(90.0)
    assert plan.exceeds_limit is True
    assert plan.declined_under_band is True
    entry = trim_floor_degradation(stage="app_review_script", plan=plan)
    assert entry["code"] == ORDERED_DURATION_INFEASIBLE
    assert "90.0s" in entry["detail"]
    assert "UNDER the band" in entry["detail"]
    assert is_degraded({"degradations": [entry]})


def test_a_drop_landing_exactly_on_the_lower_edge_is_still_taken() -> None:
    """``< lower_bound`` — the band is inclusive at both edges, as the verdict is."""
    # 120 -> 54 exactly = -10.0%, which narration_verdict calls "ok".
    plan = plan_trim([_words(54), _words(66)], ordered_seconds=60.0, words_per_minute=60.0)
    assert plan.dropped_indices == (1,)
    assert plan.estimated_seconds_after == 54.0
    assert narration_verdict(plan.estimated_seconds_after, 60.0) == "ok"
    assert plan.declined_under_band is False
    assert plan.exceeds_limit is False


def test_the_floor_degradation_names_the_numbers_that_make_it_actionable() -> None:
    """Seconds after, seconds ordered, how much was protected — and the reason."""
    plan = plan_trim(
        [_words(30)] * 6,
        ordered_seconds=60.0,
        words_per_minute=60.0,
        protected_indices=(0, 1, 2, 3),
    )
    entry = trim_floor_degradation(stage="website_review_script", plan=plan)
    assert entry["code"] == ORDERED_DURATION_INFEASIBLE
    assert "120.0" in entry["detail"], "the estimate it floored at"
    assert "60.0" in entry["detail"], "the ordered seconds"
    assert "+100.0%" in entry["detail"]
    assert "4 protected" in entry["detail"]
    assert "2 line(s) dropped" in entry["detail"]
    assert is_degraded({"degradations": [entry]})
    # The under-band refusal is a DIFFERENT reason and says so.
    declined = plan_trim([_words(40)] * 3, ordered_seconds=60.0, words_per_minute=60.0)
    detail = trim_floor_degradation(stage="lesson_script", plan=declined)["detail"]
    assert "UNDER the band" in detail
    assert "80.0" in detail


def test_it_never_returns_an_empty_script() -> None:
    """A single wildly-over-budget line cannot be dropped to nothing."""
    plan = plan_trim([_words(5000)], ordered_seconds=10.0, words_per_minute=171.0)
    assert plan.keep_indices == (0,)
    assert plan.dropped_indices == ()
    # ...nor can many lines all be dropped.
    many = plan_trim([_words(1000)] * 5, ordered_seconds=1.0, words_per_minute=171.0)
    assert len(many.keep_indices) == 1
    assert many.keep_indices == (0,)


def test_the_plan_is_deterministic_across_repeated_calls() -> None:
    """Same inputs, identical plan — no set-iteration order leaks into it."""
    texts = [_words(90), _words(150), _words(40), _words(200), _words(60)]
    plans = [
        plan_trim(
            texts,
            ordered_seconds=100.0,
            words_per_minute=171.0,
            protected_indices={2, 0},
        )
        for _ in range(20)
    ]
    assert len({(p.keep_indices, p.dropped_indices) for p in plans}) == 1
    first = plans[0]
    assert first.dropped_indices != ()
    # Both index tuples are ascending and partition the input.
    assert list(first.keep_indices) == sorted(first.keep_indices)
    assert list(first.dropped_indices) == sorted(first.dropped_indices)
    assert sorted([*first.keep_indices, *first.dropped_indices]) == list(range(len(texts)))


# ── degradation entry ────────────────────────────────────────────────────────


def test_the_degradation_uses_the_existing_code_and_reads_as_degraded() -> None:
    entry = narration_degradation(
        stage="lesson_script",
        ordered_seconds=60.0,
        estimated_seconds=134.2,
        action="reask_then_trimmed",
        detail="first estimate 200.0s.",
    )
    assert entry["code"] == DURATION_OUT_OF_TOLERANCE
    # The run-specific numbers live in detail, never in the queryable code.
    assert "134.2" in entry["detail"]
    assert "60.0" in entry["detail"]
    assert "reask_then_trimmed" in entry["detail"]
    assert "first estimate 200.0s." in entry["detail"]
    assert "lesson_script" in entry["detail"]
    # And an output carrying it is surfaced as degraded by the existing reader.
    assert is_degraded({"degradations": [entry]})


# ── the search: a greedy walk forfeits reachable orders ──────────────────────


def test_an_early_cheap_drop_must_not_forfeit_a_reachable_order() -> None:
    """THE severe defect, on the reproduced counterexample.

    35/35/20 words at 60 wpm is 35s/35s/20s = 90.0s against a 60s order (band
    54-66s). A greedy last-first walk takes the 20s tail first — 70.0s, still
    over — and then cannot take either 35s line without landing at 35.0s, under
    the band. It declined both, stopped at 70.0s with ``exceeds_limit`` and made
    its caller announce "the ordered duration is not reachable by trimming this
    script". Dropping EITHER 35-word line ALONE lands at 55.0s, inside the band
    on both edges: the order was always reachable, and an accepted drop being
    irrevocable is what hid that.
    """
    texts = [_words(35), _words(35), _words(20)]
    plan = plan_trim(texts, ordered_seconds=60.0, words_per_minute=60.0, tolerance=0.10)

    assert plan.estimated_seconds_before == 90.0
    assert plan.exceeds_limit is False, "the band WAS reachable — no floor to report"
    assert len(plan.dropped_indices) == 1, (
        "one line is enough; a greedy walk needed none it could take"
    )
    assert plan.dropped_indices == (1,), "fewest lines, and the LATEST of the equal options"
    assert plan.estimated_seconds_after == 55.0
    assert narration_verdict(plan.estimated_seconds_after, 60.0) == "ok"
    # Not vacuous: the drop the greedy walk reached FIRST really was the tail,
    # and it really does leave the script over the band.
    assert estimate_seconds([texts[0], texts[1]], words_per_minute=60.0) == 70.0


def test_the_search_prefers_the_fewest_lines_then_the_latest_ones() -> None:
    """Both halves of the tie-break, on one input where both are live.

    30/12/12/36 words at 60 wpm = 90.0s against a 60s order (band 54-66s).
    THREE plans land: drop line 0 (60.0s), drop line 3 (54.0s), or drop lines
    1+2 (66.0s). Fewest lines rules out the pair; among the two singles the
    LATEST line is cut, because dropping line 0 deletes the opening for exactly
    the same number of seconds.
    """
    texts = [_words(30), _words(12), _words(12), _words(36)]
    plan = plan_trim(texts, ordered_seconds=60.0, words_per_minute=60.0)
    assert len(plan.dropped_indices) == 1, "a two-line plan lands too, and is not preferred"
    assert plan.dropped_indices == (3,), "the latest of the equal singles"
    assert plan.estimated_seconds_after == 54.0
    assert plan.exceeds_limit is False
    # Not vacuous: all three rival plans really are inside the band.
    assert estimate_seconds(texts[1:], words_per_minute=60.0) == 60.0
    assert estimate_seconds([texts[0], texts[3]], words_per_minute=60.0) == 66.0


def test_the_plan_reports_what_the_search_actually_examined() -> None:
    """A floor claim is only as good as the candidate set behind it."""
    plan = plan_trim(
        [_words(35), _words(35), _words(20)], ordered_seconds=60.0, words_per_minute=60.0
    )
    assert plan.droppable_count == 3
    assert plan.search_exhausted is True
    # 3 lines -> achievable dropped-word totals 20/35/55/70 (all 3 would empty
    # the script and is excluded), i.e. 4 distinct candidate drop sets scored.
    assert plan.candidates_evaluated == 4


def test_the_floor_degradation_says_what_the_search_tried() -> None:
    """ "Not reachable" is a strong claim; the text must show the work."""
    plan = plan_trim(
        [_words(10), _words(90), _words(5)], ordered_seconds=60.0, words_per_minute=60.0
    )
    assert plan.exceeds_limit is True
    detail = trim_floor_degradation(stage="lesson_script", plan=plan)["detail"]
    assert f"Scored {plan.candidates_evaluated} candidate drop set(s)" in detail
    assert "3 droppable line(s)" in detail
    assert "(exhaustively)" in detail
    assert "[54.0s, 66.0s]" in detail, "the band it could not land inside"
    assert "not reachable by trimming" in detail


def test_a_bounded_search_never_claims_the_order_is_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the bound is hit the claim weakens to what was actually proved."""
    monkeypatch.setattr(narration_budget, "MAX_TRIM_SEARCH_STATES", 0)
    plan = plan_trim(
        [_words(10), _words(90), _words(5)], ordered_seconds=60.0, words_per_minute=60.0
    )
    assert plan.search_exhausted is False
    # The bound gates COMBINATIONS only: all three single-line drops were still
    # scored, which is why `candidates_evaluated` is 3 and not 0. A bound that
    # could suppress the one-line sweep is the defect this module keeps having.
    assert plan.candidates_evaluated == 3
    assert plan.exceeds_limit is True, "the greedy fallback still reports the overrun"
    detail = trim_floor_degradation(stage="lesson_script", plan=plan)["detail"]
    assert "BOUNDED search, not exhaustive" in detail
    assert "not reachable by trimming" not in detail, "it did not prove that"
    assert "not reached under a bounded search" in detail


# ── non-finite inputs ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("ordered", "tolerance"),
    [(60.0, float("nan"))],
)
def test_a_non_finite_band_changes_nothing_instead_of_amputating(
    ordered: float, tolerance: float
) -> None:
    """Every comparison against NaN is False, which used to mean "keep cutting".

    Three 100-word lines at 150 wpm is 120.0s. With a NaN tolerance the limit is
    NaN, so ``current <= limit`` and ``after < lower_bound`` were both always
    False: the walk deleted lines until ONE survived — 40.0s of a 120.0s draw —
    and returned ``exceeds_limit=False``, i.e. a silently amputated paid
    deliverable reported as a clean trim. There is no band to aim at here, so
    the only honest plan is the no-op one.
    """
    texts = [_words(100)] * 3
    plan = plan_trim(texts, ordered_seconds=ordered, words_per_minute=150.0, tolerance=tolerance)
    assert plan.keep_indices == (0, 1, 2), "nothing is dropped against a band that does not exist"
    assert plan.dropped_indices == ()
    assert plan.estimated_seconds_after == 120.0 == plan.estimated_seconds_before
    assert plan.exceeds_limit is False, "and it does not invent a verdict either"


def test_budget_words_survives_a_non_finite_order() -> None:
    """``round(nan)`` raises, on a paid run's happy path."""
    assert budget_words(float("nan"), words_per_minute=150.0) == 0
    assert budget_words(float("inf"), words_per_minute=150.0) == 0


# ── protection counts only lines that exist ──────────────────────────────────


# ── a verdict never asserts a direction it has not established ───────────────


def test_a_non_finite_estimate_is_unknown_not_short() -> None:
    """``"short"`` means ACCEPT-and-ship; NaN establishes no direction at all."""
    assert narration_verdict(float("nan"), 60.0) == "unknown"
    assert narration_verdict(float("inf"), 60.0) == "unknown"
    assert narration_verdict(60.0, 60.0, tolerance=float("nan")) == "unknown"
    # It is still not "ok", so every caller records the miss rather than
    # returning early on a number it cannot compare.
    assert narration_verdict(float("nan"), 60.0) != "ok"
    # A real estimate is unaffected.
    assert narration_verdict(59.0, 60.0) == "ok"
    assert narration_verdict(90.0, 60.0) == "long"
    assert narration_verdict(30.0, 60.0) == "short"


def test_the_default_tolerance_is_the_shared_conformance_band() -> None:
    """One band, imported — not two literals that can drift apart.

    A script trimmed to the letter of this module is graded by
    ``duration_conformance`` after delivery; two independent 0.10 literals could
    be changed apart and nothing would have failed.
    """
    assert narration_budget.DEFAULT_TOLERANCE is DURATION_TOLERANCE_FRACTION
    assert narration_verdict(66.0, 60.0) == "ok"
    assert narration_verdict(66.1, 60.0) == "long"


# ── the bound must MEASURE the search, never estimate it ─────────────────────


#: The live counterexample: a ~2.5h audiobook script whose first line is a
#: 3500-word chapter. 30 lines, 9027 words, ordered 6000s at 60 wpm — band
#: [5400s, 6600s]. Dropping ONE 3500-word line lands at 5527.0s, inside it.
_AUDIOBOOK_WORDS = [3500, 3500, 2000] + [1] * 27


def _audiobook() -> list[str]:
    return [_words(n) for n in _AUDIOBOOK_WORDS]


def test_a_tractable_script_is_not_disqualified_by_an_estimated_cost() -> None:
    """The fourth instance of this module's defect class, pinned.

    The bound was ``len(droppable) * (droppable_words + 1)`` — the cost of a
    table holding every word total from 0 upwards. The REAL table holds only
    ACHIEVABLE sums: 167 entries here against an estimate of 270,840, an
    over-count of ~1600x. The search was skipped, the greedy walk deleted 28 of
    30 lines of a paid deliverable, and the plan reported
    ``candidates_evaluated=0`` — nothing had been examined at all.

    The same script with ``[1] * 4`` instead of ``[1] * 27`` (identical
    arithmetic, smaller word total) landed correctly, which is the proof the
    ARITHMETIC was never wrong and the BOUND always was.
    """
    plan = plan_trim(_audiobook(), ordered_seconds=6000.0, words_per_minute=60.0, tolerance=0.10)
    assert plan.dropped_indices == (1,), "one 3500-word line, tail-preferred"
    assert plan.estimated_seconds_after == pytest.approx(5527.0)
    assert plan.exceeds_limit is False
    assert plan.search_exhausted is True, "167 real states is three orders under the cap"
    assert plan.candidates_evaluated > 0, "a floor claim backed by 0 candidates is a lie"
    # Not vacuous: the estimate the old bound computed really does exceed the cap,
    # so this script really was disqualified by a number nobody measured.
    droppable_words = sum(_AUDIOBOOK_WORDS)
    assert len(_AUDIOBOOK_WORDS) * (droppable_words + 1) > narration_budget.MAX_TRIM_SEARCH_STATES
    # ...and the truth it hid: 9027 - 3500 = 5527 words is inside the band.
    assert narration_verdict(5527.0, 6000.0) == "ok"


def test_the_short_tail_variant_that_always_worked_still_works() -> None:
    """``[1] * 4`` — same shape, under the old estimate, same answer. A control."""
    texts = [_words(n) for n in ([3500, 3500, 2000] + [1] * 4)]
    plan = plan_trim(texts, ordered_seconds=6000.0, words_per_minute=60.0, tolerance=0.10)
    assert plan.dropped_indices == (1,)
    assert plan.estimated_seconds_after == pytest.approx(5504.0)
    assert plan.exceeds_limit is False


def test_every_single_line_drop_is_scored_even_when_the_bound_is_blown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tier one is OUTSIDE the bound, so no cost model can hide the one-line answer.

    With the combination search bounded to zero states the multi-line table is
    never built — and the plan is still correct, because dropping a single line
    is O(n) and is checked unconditionally. This is the structural half of the
    fix: it holds for ANY value of ``MAX_TRIM_SEARCH_STATES`` and for any future
    cost model, right or wrong.
    """
    monkeypatch.setattr(narration_budget, "MAX_TRIM_SEARCH_STATES", 0)
    plan = plan_trim(_audiobook(), ordered_seconds=6000.0, words_per_minute=60.0, tolerance=0.10)
    assert plan.dropped_indices == (1,)
    assert plan.estimated_seconds_after == pytest.approx(5527.0)
    assert plan.exceeds_limit is False, "28 lines were NOT amputated"
    assert plan.search_exhausted is False, "and it does not pretend the search was whole"
    assert plan.candidates_evaluated == len(set(_AUDIOBOOK_WORDS)) == 3, (
        "every single-line drop was scored; they collapse to 3 distinct word totals"
    )


def test_the_bound_counts_states_built_not_words_present() -> None:
    """A long script with FEW achievable sums stays exhaustive.

    400 lines of exactly 50 words is 20,000 droppable words — the old estimate
    put it at 8,000,400 states. The real table has one entry per multiple of 50,
    i.e. 400, and the search is genuinely exhaustive.
    """
    plan = plan_trim(
        [_words(50)] * 400, ordered_seconds=6000.0, words_per_minute=60.0, tolerance=0.10
    )
    assert plan.search_exhausted is True
    assert plan.exceeds_limit is False
    assert plan.candidates_evaluated < 500, "achievable sums, not word totals"


# ── a drop must save time ────────────────────────────────────────────────────


def test_a_drop_that_saves_no_seconds_is_never_taken() -> None:
    """Two empty beats deleted for 0.0 seconds is a destroyed script, not a trim.

    ``["", "", <200 words>]`` against a 60s order used to return
    ``dropped_indices=(0, 1)`` with ``estimated_seconds_after=200.0`` — the
    overrun untouched and two lines gone. A zero-word line shortens nothing, so
    ``after < current`` is required, strictly.
    """
    plan = plan_trim(
        ["", "", _words(200)], ordered_seconds=60.0, words_per_minute=60.0, tolerance=0.1
    )
    assert plan.dropped_indices == (), "nothing was worth dropping"
    assert plan.keep_indices == (0, 1, 2)
    assert plan.estimated_seconds_after == pytest.approx(200.0)
    assert plan.estimated_seconds_after == plan.estimated_seconds_before
    assert plan.exceeds_limit is True, "and the overrun is still reported"
    # The only line worth anything would have landed at 0.0s, far under the band.
    assert plan.declined_under_band is True


def test_a_zero_word_line_is_never_dropped_by_the_floor_walk() -> None:
    """The same rule once the band is out of reach: the walk skips a free line too."""
    plan = plan_trim(["", _words(80), _words(40)], ordered_seconds=60.0, words_per_minute=60.0)
    # 120s -> dropping line 2 (40 words) lands at 80.0s, still over; dropping
    # line 1 lands at 40.0s, under. Line 0 is worth nothing and cannot help.
    assert 0 not in plan.dropped_indices


# ── the floor must name the reason that actually bound it ────────────────────


def test_the_floor_text_names_the_last_survivor_not_a_contradiction() -> None:
    """A single long line floors because the last survivor is never dropped.

    The text used to read "0 line(s) dropped, 0 protected, and 0 line(s) are
    protected from trimming. Scored 0 candidate drop set(s) over 1 droppable
    line(s) (exhaustively)" — it named protection as the reason while reporting
    none, and called a line droppable that the search was never allowed to drop.
    """
    plan = plan_trim([_words(500)], ordered_seconds=60.0, words_per_minute=60.0)
    assert plan.exceeds_limit is True
    assert plan.droppable_count == 0, "max_drops is 0 — nothing was ever a candidate"
    detail = trim_floor_degradation(stage="lesson_script", plan=plan)["detail"]
    assert "last survivor is never dropped" in detail
    assert "line(s) are protected from trimming" not in detail, "nothing was protected"
    assert "0 droppable line(s)" in detail


def test_the_floor_text_distinguishes_full_protection_from_a_refusal() -> None:
    """Every line pinned is a DIFFERENT reason from an under-band refusal."""
    plan = plan_trim(
        [_words(171)] * 4,
        ordered_seconds=60.0,
        words_per_minute=171.0,
        protected_indices=(0, 1, 2, 3),
    )
    detail = trim_floor_degradation(stage="website_review_script", plan=plan)["detail"]
    assert "all 4 line(s) are protected from trimming" in detail
    assert "UNDER the band" not in detail


def test_the_floor_text_never_claims_exhaustion_it_did_not_have(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preserved invariant, re-pinned against the new reason text."""
    monkeypatch.setattr(narration_budget, "MAX_TRIM_SEARCH_STATES", 0)
    plan = plan_trim(
        [_words(10), _words(90), _words(5)], ordered_seconds=60.0, words_per_minute=60.0
    )
    detail = trim_floor_degradation(stage="lesson_script", plan=plan)["detail"]
    assert "not reachable by trimming" not in detail
    assert "BOUNDED search, not exhaustive" in detail
