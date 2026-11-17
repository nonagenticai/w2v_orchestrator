"""A/V quality wave 7 — reconcile AUTHORED word counts against the computed budget.

``captions.line_word_counts`` was written to "reconcile authored word counts against a
computed beat_word_budget" and then never called: its own docstring said "nothing on the
live path calls it yet". So a beat that came back short of its budget was invisible until
``reconcile_timeline`` flagged it as underspoken — by which time the footage had been
recorded and paid for.

Also pinned here: ``_apply_feasibility_floor``'s unmeasured-yield branch, which returned
the spine unchanged with only a ``log.warning`` to show for it. A run whose plan was never
structurally bounded was indistinguishable, on the stage output, from one that was.

Both tests FAIL on the pre-change code — ``authored_word_shortfalls`` and
``structural_bound_unavailable`` did not exist, and ``DemoVideoGenOutput`` carried no
field for the bound state.
"""

from __future__ import annotations

from src.services.captions import line_word_counts
from src.services.stages.demo_video.gen import (
    _WORD_BUDGET_TOLERANCE,
    DemoVideoGenOutput,
    DemoVideoLine,
    authored_word_shortfalls,
    narration_words_per_scenario,
    narration_words_target,
    structural_bound_unavailable,
)
from src.services.stages.shared.time_budget import (
    DEFAULT_WPM_NATURAL,
    NARRATION_BUDGET_RATE,
    VO_RATE_DRAG_FLOOR,
)

#: Every fixture below is DERIVED from this order, never from a number typed by hand.
#: Wave 13 recalibration: these tests were written against a budget taken at
#: ``VO_RATE_NATURAL_FLOOR`` (0.80) — 60 words for a 30s order, 30 per beat — and every
#: fixture hardcoded those. That rate WAS the defect (it budgeted for a voice already
#: dragged 20%, i.e. it ordered the owner's "narration weirdly slows down" upstream), so
#: when :data:`NARRATION_BUDGET_RATE` replaced it with 1.0 — 75 words, 38/37 per beat —
#: three of these tests went red on a CORRECT change. Hardcoding is what did that, so the
#: numbers are now computed from the constants and will survive the next rate move.
_ORDERED_SECONDS = 30.0


def _lines(*texts: str) -> list[DemoVideoLine]:
    return [
        DemoVideoLine(id=f"clip-{i + 1}", text=text, seconds=0.0) for i, text in enumerate(texts)
    ]


def _per_beat_budget(beats: int, ordered: float = _ORDERED_SECONDS) -> list[int]:
    """The live per-beat word budget for ``ordered``, straight off the production split."""
    budget = narration_words_per_scenario(
        narration_words_target(ordered, DEFAULT_WPM_NATURAL), beats
    )
    assert budget is not None
    return budget


def _words(count: int) -> str:
    return " ".join(f"w{i}" for i in range(count))


def test_line_word_counts_is_reachable_from_the_gen_spine() -> None:
    """The helper's own contract, exercised through the shape gen actually emits.

    Guards the join it dispatches on: ``line_word_counts`` looks up the
    ``demo_video_gen`` row of ``_CAPTION_SOURCES`` by stage name, so a rename on either
    side silently returns ``None`` and every shortfall check below would go quietly
    vacuous rather than fail.
    """
    lines = _lines("one two three", "four five")
    counts = line_word_counts({"demo_video_gen": {"lines": [line.model_dump() for line in lines]}})
    assert counts == {"clip-1": 3, "clip-2": 2}


def test_a_beat_short_of_its_word_budget_is_reported() -> None:
    """THE WIRING. A 30s order at 150 wpm budgets 75 words; a 6-word spine is short.

    Also pins the RATE the budget is taken at, which is the whole of wave 13: the total
    must be the natural-rate number (``NARRATION_BUDGET_RATE`` = 1.0, 75 words) and
    STRICTLY MORE than the same order priced at the drag floor (0.97, 73 words) — the
    obvious-but-wrong reuse of one constant for two jobs. It was 60 words while the budget
    sat at ``VO_RATE_NATURAL_FLOOR`` (0.80).

    FAILS BEFORE: ``authored_word_shortfalls`` did not exist — nothing on the live path
    compared an authored count to a budget.
    """
    ordered = _ORDERED_SECONDS
    total = narration_words_target(ordered, DEFAULT_WPM_NATURAL)
    assert total == round(ordered * DEFAULT_WPM_NATURAL * NARRATION_BUDGET_RATE / 60)
    assert total > round(ordered * DEFAULT_WPM_NATURAL * VO_RATE_DRAG_FLOOR / 60)

    budget = _per_beat_budget(2)
    # The split always sums to the total, so a producer honouring every per-beat number
    # lands on the order exactly.
    assert sum(budget) == total

    short = authored_word_shortfalls(
        _lines("one two three", "four five six"), ordered, DEFAULT_WPM_NATURAL
    )
    assert short == {"clip-1": (3, budget[0]), "clip-2": (3, budget[1])}


def test_a_beat_that_meets_its_budget_is_not_reported() -> None:
    """No false positives — a signal that fires on everything is not a signal.

    Each beat is given EXACTLY its own share of the split, which is 38/37 at the wave-13
    budget rather than the flat 30/30 this fixture used to hardcode against the 0.80 rate.
    The shares differ by one word, so a single on-budget string for both beats would be
    wrong for one of them by construction.

    FAILS BEFORE: the function did not exist.
    """
    budget = _per_beat_budget(2)
    on_budget = _lines(*(_words(want) for want in budget))
    assert authored_word_shortfalls(on_budget, _ORDERED_SECONDS, DEFAULT_WPM_NATURAL) == {}


def test_tolerance_absorbs_ordinary_authoring_noise() -> None:
    """A couple of words either way is where a sentence ended, not a defect.

    ``narration_words_per_scenario`` splits an integer budget evenly; a writer lands
    wherever the prose does. Two words under its own share must pass; a beat below
    ``_WORD_BUDGET_TOLERANCE`` of it must not.

    Both fixtures are now taken RELATIVE to the per-beat share and to the tolerance
    constant. They were "28 of 30 must pass, 20 of 30 must not" — absolute counts against
    the retired 0.80 budget, where 30 happened to be the share. At the wave-13 budget the
    share is 38/37, so the old 28 is 26% under it and the *passing* fixture became a
    reported shortfall. The intent never changed; only the arithmetic it was written
    against did.

    FAILS BEFORE: the function did not exist.
    """
    budget = _per_beat_budget(2)

    near = _lines(*(_words(want - 2) for want in budget))
    assert authored_word_shortfalls(near, _ORDERED_SECONDS, DEFAULT_WPM_NATURAL) == {}

    # Strictly below the tolerated fraction of each beat's own share.
    far_counts = [int(want * _WORD_BUDGET_TOLERANCE) - 1 for want in budget]
    assert all(
        count < want * _WORD_BUDGET_TOLERANCE
        for count, want in zip(far_counts, budget, strict=True)
    )
    far = _lines(*(_words(count) for count in far_counts))
    assert authored_word_shortfalls(far, _ORDERED_SECONDS, DEFAULT_WPM_NATURAL) == {
        "clip-1": (far_counts[0], budget[0]),
        "clip-2": (far_counts[1], budget[1]),
    }


def test_no_order_means_no_budget_and_therefore_no_shortfall() -> None:
    """Nothing ordered ⇒ nothing to be short of. Never invent a budget nobody set.

    FAILS BEFORE: the function did not exist.
    """
    assert authored_word_shortfalls(_lines("one"), None, DEFAULT_WPM_NATURAL) == {}
    assert authored_word_shortfalls(_lines("one"), 0.0, DEFAULT_WPM_NATURAL) == {}
    assert authored_word_shortfalls(_lines("one"), 30.0, 0.0) == {}


def test_structural_bound_unavailable_is_true_only_when_an_order_went_unbounded() -> None:
    """The state ``_apply_feasibility_floor`` used to swallow, now answerable.

    True ONLY when a duration was ordered AND no per-scenario measurement existed to
    bound the spine against — not on an unordered run (nothing to bound to) and not on a
    measured one (the bound was applied).

    FAILS BEFORE: ``structural_bound_unavailable`` did not exist, and the condition lived
    inline in a branch whose only output was a log line.
    """
    lines = _lines("one", "two")
    ordered = {"target_duration_seconds": 30.0}

    # Ordered, but nothing measured ⇒ the plan was never structurally bounded.
    assert structural_bound_unavailable(lines, ordered, None) is True
    assert structural_bound_unavailable(lines, ordered, 0.0) is True

    # Measured ⇒ the bound was applied.
    assert structural_bound_unavailable(lines, ordered, 22.0) is False
    # Nothing ordered ⇒ there is no bound to be missing.
    assert structural_bound_unavailable(lines, {}, None) is False
    # No spine ⇒ nothing to bound.
    assert structural_bound_unavailable([], ordered, None) is False


def test_the_bound_state_rides_the_stage_output() -> None:
    """It has to LEAVE the stage, or it is still just a log line.

    FAILS BEFORE: ``DemoVideoGenOutput`` declared neither field, so
    ``model_validate`` would drop them (pydantic's default ``extra="ignore"``) and the
    state stayed invisible to E2E assertions and to every later stage.
    """
    assert "structural_bound_unavailable" in DemoVideoGenOutput.model_fields
    assert "structural_bound_source" in DemoVideoGenOutput.model_fields

    out = DemoVideoGenOutput.model_validate(
        {
            "lines": [],
            "bdd_job_id": "job-1",
            "structural_bound_unavailable": True,
            "structural_bound_source": None,
        }
    )
    assert out.structural_bound_unavailable is True
    assert out.structural_bound_source is None

    # Default stays FALSE, so a healthy output is unchanged by this addition.
    assert DemoVideoGenOutput(lines=[], bdd_job_id="").structural_bound_unavailable is False
