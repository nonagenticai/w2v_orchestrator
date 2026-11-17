"""⭐ WAVE 14, N5 — ONE WORD-BUDGET ARITHMETIC ACROSS EVERY NARRATION-DRIVEN TYPE.

Wave 13 found that the word budget was being written at the DRAG FLOOR, so the
generator was asked for ~20% too few words and the voice was dragged to fill the time.
It fixed that in ``demo_video`` by taking the budget at
:data:`~src.services.stages.shared.time_budget.NARRATION_BUDGET_RATE` (1.0).

The other four narration-driven types — ``lesson``, ``promo``, ``website_review``,
``app_review`` — budget through ``compute_plan.beat_word_budget`` (and
``narration_budget.budget_words``), which named NO rate constant at all.

⚠️ CORRECTION to the wave-14 brief, which said those four "still order the drag". They
do not. An omitted factor is an implicit 1.0, which is the value wave 13 settled on, and
all six budgets already agreed to the word. What was actually missing was any link
between them: nothing would have failed if ``NARRATION_BUDGET_RATE`` moved and the other
four stayed at their implicit 1.0. These tests are that link.

No DB, no LLM, no network — pure arithmetic over the public entry points.
"""

from __future__ import annotations

import pytest

from src.services.stages.app_review.script import word_budget as app_review_word_budget
from src.services.stages.demo_video.gen import narration_words_target
from src.services.stages.lesson.script import word_budget as lesson_word_budget
from src.services.stages.promo.script import word_budget as promo_word_budget
from src.services.stages.shared import compute_plan, narration_budget
from src.services.stages.shared.narration_budget import budget_words
from src.services.stages.shared.time_budget import (
    DEFAULT_WPM_NATURAL,
    NARRATION_BUDGET_RATE,
    VO_RATE_DRAG_FLOOR,
)
from src.services.stages.website_review.script import word_budget as website_review_word_budget

_FLEET = {
    "lesson": lesson_word_budget,
    "promo": promo_word_budget,
    "website_review": website_review_word_budget,
    "app_review": app_review_word_budget,
}


@pytest.mark.parametrize("ordered", [10.0])
@pytest.mark.parametrize("wpm", [94.0])
def test_every_type_budgets_the_same_words_as_demo_video(ordered: float, wpm: float) -> None:
    """The five budgets are ONE number, swept rather than sampled.

    demo_video reaches it via ``narration_words_target`` (explicit
    ``NARRATION_BUDGET_RATE``); the other four via ``beat_word_budget``; the shared
    narration trimmer via ``budget_words``.
    """
    expected = narration_words_target(ordered, wpm)
    assert budget_words(ordered, words_per_minute=wpm) == expected
    for name, fn in _FLEET.items():
        assert fn(ordered, wpm) == expected, name


def test_the_fleet_budget_actually_reads_the_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    """⭐ THE CALL-SITE PROOF, and the reason a no-op routing is still worth shipping.

    ``NARRATION_BUDGET_RATE`` is 1.0, so naming it changes no number today and a test
    that only compared budgets would pass against an implementation that ignored it
    entirely — the "defined but never called" shape this repo keeps finding.

    So the constant is MOVED and the budgets are required to move with it. Every one of
    the five must land on the dragged count; if a wrapper still carries an implicit 1.0
    it stays at 75 and fails here.
    """
    ordered, wpm = 30.0, DEFAULT_WPM_NATURAL
    assert narration_words_target(ordered, wpm) == 75

    monkeypatch.setattr(compute_plan, "NARRATION_BUDGET_RATE", 0.80)
    dragged = round(ordered / 60.0 * wpm * 0.80)
    assert dragged == 60
    for name, fn in _FLEET.items():
        assert fn(ordered, wpm) == dragged, f"{name} does not read NARRATION_BUDGET_RATE"

    # ``narration_budget`` holds its own binding (it is imported by the trimmer, not by
    # the stage wrappers), so it is proven separately rather than assumed.
    monkeypatch.setattr(narration_budget, "NARRATION_BUDGET_RATE", 0.80)
    assert budget_words(ordered, words_per_minute=wpm) == dragged


def test_the_budget_rate_and_the_refusal_floor_are_the_fleets_two_constants() -> None:
    """The gap between them IS the tolerance, fleet-wide and in one place.

    Budget at or below the floor puts a perfect delivery on the detection boundary
    (wave 13 measured a 48.9% false-flag rate there); budget above 1.0 orders an
    overrun. This is the ordering every type now inherits.
    """
    assert VO_RATE_DRAG_FLOOR < NARRATION_BUDGET_RATE <= 1.0
