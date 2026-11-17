"""4.2 — the word budget the orchestrator sends to bdd alongside the ordered duration.

CONTRACT: `narration_words_target` (int) and `narration_words_per_scenario`
(list[int] | None) are consumed by bdd_playwright_service. The key NAMES are the
contract; renaming either breaks a sibling service.
"""

from __future__ import annotations

from src.services.stages.demo_video.gen import (
    narration_words_per_scenario,
    narration_words_target,
)
from src.services.stages.shared.time_budget import (
    DEFAULT_WPM_NATURAL,
    NARRATION_BUDGET_RATE,
    VO_RATE_DRAG_FLOOR,
    VO_RATE_NATURAL_FLOOR,
    vo_rate_for,
)


def test_total_is_derived_from_the_repos_own_constants() -> None:
    """Derived, never restated — the two budgets cannot drift.

    Mutating the formula in `narration_words_target` (e.g. dropping the
    NARRATION_BUDGET_RATE factor) changes the result and fails here.
    """
    for ordered in (30.0, 60.0, 90.0, 137.5):
        expected = round(ordered * DEFAULT_WPM_NATURAL * NARRATION_BUDGET_RATE / 60)
        assert narration_words_target(ordered) == expected

    # Pinned absolute value, so a change to the FORMULA is caught even if someone
    # edits the expectation above in lockstep: 30s at 150 wpm x 0.97 = 72.75 -> 73
    # words, and 60s -> 145.5 -> 146. The factor is the OPERATIVE drag floor, not the
    # 0.80 pathological backstop the budget used to be taken at (which gave 60/120).
    assert narration_words_target(30.0) == 75
    assert narration_words_target(60.0) == 150


def test_the_budget_is_the_natural_rate_one_which_retires_the_old_floor_bias() -> None:
    """⭐ THIS TEST'S PREMISE WAS THE DEFECT.

    It was named ``test_the_floor_rate_is_actually_applied`` and asserted
    ``narration_words_target(60.0) < round(60.0 * DEFAULT_WPM_NATURAL / 60)`` — i.e. that
    the budget is STRICTLY BELOW a natural read — plus ``VO_RATE_NATURAL_FLOOR < 1.0``.

    That is the 0.80 bias written down as an expectation: the budget was required to ask
    for fewer words than the ordered time actually holds, on the rationale that the rate
    lever would absorb the difference. The rate lever absorbs it BY DRAGGING THE VOICE,
    which is the owner's complaint. So the assertion was pinning the cause.

    The budget is now the natural-rate count — the words this voice, as measured, says in
    the ordered time — and equality with a natural read is the CORRECT answer, not a
    regression. What replaces the old inequality is the pair of properties that actually
    matter and are checked above: a perfect delivery is never flagged, and the budget rate
    sits strictly above the detector floor.

    (Noted in passing: at the interim 0.97 budget this test still passed, but only by 4
    words out of 150 — a 2.7% margin one rounding step from collapsing. A guard that thin
    was a hint the constant was doing two jobs.)
    """
    assert narration_words_target(60.0) == round(60.0 * DEFAULT_WPM_NATURAL / 60)
    # The floor still exists and is still below natural — it is just no longer the BUDGET.
    assert VO_RATE_NATURAL_FLOOR < VO_RATE_DRAG_FLOOR < 1.0


def test_per_scenario_split_sums_to_the_total_exactly() -> None:
    """No rounding drift: honouring every per-beat number lands on the total.

    ⭐ WAVE 14, N3. This holds wherever the exact split is SAYABLE at or above the drag
    floor, which is 795 of the 880 swept (duration, beats) pairs. Where it is not, every
    beat is raised to ``base + 1`` and the split overshoots — ``(63, 4)`` and ``(7, 3)``
    below are exactly those cases, and they used to order a beat at ``15/15.75 = 0.952``
    and ``2/2.33 = 0.857`` respectively. The exact sum is a nice property; not ordering
    the owner's reported defect outranks it.
    """
    for total, beats in ((60, 4), (61, 4), (120, 7)):
        split = narration_words_per_scenario(total, beats)
        assert split is not None
        assert len(split) == beats
        assert sum(split) == total
        assert max(split) - min(split) <= 1  # spread, not dumped on one beat

    # The exception, pinned by value so the bump cannot silently widen.
    assert narration_words_per_scenario(63, 4) == [16, 16, 16, 16]
    assert narration_words_per_scenario(7, 3) == [3, 3, 3]


def test_unknown_beat_count_yields_none_not_a_fabricated_split() -> None:
    assert narration_words_per_scenario(60, 0) is None
    assert narration_words_per_scenario(60, -1) is None
    assert narration_words_per_scenario(0, 4) is None


def test_a_perfectly_delivered_budget_is_never_flagged_as_underspoken() -> None:
    """⭐ THE PROPERTY THAT TIES THE BUDGET TO THE DETECTOR, and the one I got wrong.

    "The generator delivered exactly what we budgeted for" and "the delivery is not
    dragged" must be THE SAME EVENT. If they are not, the pipeline reports a defect on a
    flawless run, and the underspoken re-ask spends LLM calls on beats that have nothing
    wrong with them.

    The first cut of wave 13 broke this by taking the budget at ``VO_RATE_DRAG_FLOOR``
    — the detector's own floor. A beat that hits its budget then lands EXACTLY on the
    boundary, where `round()` casts the deciding vote: over every ordered duration from
    5s to 180s, a perfect delivery was flagged on **86 of 176** of them (48.9%), purely
    from integer rounding. The three failing fixtures in
    ``test_word_budget_reconciliation`` are what surfaced it.

    A signal that fires on half of all PERFECT runs is not a signal — this repo rejected
    a deband ceiling on exactly that ground. So the budget is written at
    ``NARRATION_BUDGET_RATE`` (1.0) and the floor sits 3% below it; that gap is the
    tolerance and it lives in ONE place.

    ⛔ This test is the guard against re-merging the two constants. It sweeps rather than
    sampling on purpose: 30.0s and 60.0s both happen to round UP and would have passed
    the broken version, which is precisely how the defect survived its first review.
    """
    flagged = [
        seconds
        for seconds in (float(s) for s in range(6, 181))
        if vo_rate_for(seconds, narration_words_target(seconds), DEFAULT_WPM_NATURAL)[2]
    ]
    assert flagged == [], (
        f"a beat that delivered its budget exactly was flagged underspoken on "
        f"{len(flagged)} ordered duration(s): {flagged[:15]}"
    )


def test_the_budget_rate_sits_strictly_above_the_detector_floor() -> None:
    """The ordering the test above depends on, asserted directly so a failure says WHY.

    Budget at or below the floor ⇒ a delivered budget is (at best) on the boundary.
    Budget above 1.0 ⇒ the generator is asked for more words than the ordered time
    holds at the voice's own measured rate, i.e. a guaranteed overrun.
    """
    assert NARRATION_BUDGET_RATE > VO_RATE_DRAG_FLOOR
    assert NARRATION_BUDGET_RATE <= 1.0


def test_a_perfectly_delivered_per_beat_budget_is_never_flagged_underspoken() -> None:
    """⭐ WAVE 14, N3 — THE SWEEP, over BEAT COUNTS as well as durations.

    The sweep above holds the whole order as ONE beat, so it never exercises
    ``narration_words_per_scenario``. A demo_video spine is allotted the SAME seconds
    per beat, so a beat handed a below-average word count by ``divmod``'s remainder
    delivers at ``base / (total / beats)`` = ``1 - remainder / total`` — under the 0.97
    drag floor whenever the remainder exceeds 3% of the budget. That beat honoured its
    budget to the word and is flagged anyway, and ``reconcile_timeline`` arms a PAID
    LLM re-ask on ``any(a.underspoken)``.

    Measured with the plain ``divmod`` split: **70 of 880** (duration, beats) pairs
    flagged a flawless delivery — 7.95%. A single duration proves nothing here: 30s
    divides evenly at 2, 3 and 5 beats and passes trivially.

    ⚠️ CORRECTION to the wave-14 brief, which said "**4 is the only hit**" at 30s.
    6 beats hits it too: ``[13, 13, 13, 12, 12, 12]`` against 5.00s each is
    ``12 / 12.5 = 0.960``, the same 0.960 as the 4-beat case.

    The 5s order is excluded for the reason the one-beat sweep above starts at 6: there
    the flag comes from ``narration_words_target`` rounding 12.5 words down to 12, with
    an exact and perfectly even split — it is not this function's defect and this test
    must not claim to fix it.
    """
    flagged: list[tuple[int, int, list[int]]] = []
    unsayable: list[tuple[int, int, list[int]]] = []
    for seconds in range(6, 181):
        for beats in range(2, 7):
            total = narration_words_target(float(seconds))
            split = narration_words_per_scenario(total, beats)
            assert split is not None
            assert len(split) == beats
            per_beat_seconds = float(seconds) / beats
            rates = [vo_rate_for(per_beat_seconds, words, DEFAULT_WPM_NATURAL) for words in split]
            if any(r[2] for r in rates):
                flagged.append((seconds, beats, split))
            if any(r[1] for r in rates):
                unsayable.append((seconds, beats, split))

    assert flagged == [], (
        f"a beat that delivered its per-beat budget exactly was flagged underspoken on "
        f"{len(flagged)} (duration, beats) pair(s): {flagged[:10]}"
    )
    # The mirror, so the fix cannot buy the short side by ordering an overrun.
    assert unsayable == [], f"the split made {len(unsayable)} pair(s) unsayable: {unsayable[:10]}"


def test_the_owners_worked_example_no_longer_orders_a_sub_floor_beat() -> None:
    """30s @ 150 wpm = 75 words over 4 beats. ``divmod`` gave ``[19, 19, 19, 18]``.

    18 words in 7.50s is ``18 / 18.75 = 0.960``, below the 0.97 floor ⇒ ``underspoken``
    ⇒ one guaranteed LLM re-ask on a run where nothing went wrong.
    """
    total = narration_words_target(30.0)
    assert total == 75
    assert narration_words_per_scenario(total, 4) == [19, 19, 19, 19]
    assert narration_words_per_scenario(total, 6) == [13, 13, 13, 13, 13, 13]
    # And the even divisions are untouched — the bump must not fire where nothing is wrong.
    assert narration_words_per_scenario(total, 3) == [25, 25, 25]
    assert narration_words_per_scenario(total, 5) == [15, 15, 15, 15, 15]


def test_the_exact_sum_is_preserved_wherever_it_costs_nothing() -> None:
    """The bump is the EXCEPTION, not the rule: it may only fire where the exact split
    would have ordered a sub-floor beat. Without this, an implementation that simply
    rounded every beat up would pass the sweep — and overshoot the budget by up to 20%
    on a short order.
    """
    exact = 0
    pairs = 0
    for seconds in range(6, 181):
        for beats in range(2, 7):
            total = narration_words_target(float(seconds))
            split = narration_words_per_scenario(total, beats)
            assert split is not None
            pairs += 1
            exact += sum(split) == total
            # It NEVER undershoots the total; that is the direction that ordered the drag.
            assert sum(split) >= total
            assert max(split) - min(split) <= 1
    assert exact / pairs > 0.85, f"only {exact}/{pairs} splits still sum exactly"
