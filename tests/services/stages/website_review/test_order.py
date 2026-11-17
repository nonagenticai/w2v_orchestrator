"""website_review's ordered length: one resolver, a narration budget net of the
title cards, and a floor enforced at the door.

Prod run 01a09f82-7179-73a5-a23a-40c15d148b37: the wizard ordered 10.0s on
``deliverable.target_duration_seconds``; the script budgeted 120s-default narration
off ``brief.duration_seconds`` and the compile added two 3s title cards → 15.1s.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from src.api.schemas import ProjectCreate
from src.services.runs.run_context import RunContext
from src.services.stages.shot_assembly import MIN_NARRATED_BEAT_SECONDS
from src.services.stages.website_review.compile import WebsiteReviewCompileStage
from src.services.stages.website_review.order import (
    MIN_NARRATED_LINES,
    MINIMUM_ORDER_SECONDS,
    REQUIRED_NARRATED_PHASES,
    OrderBelowFloorError,
    fixed_non_narration_seconds,
    minimum_narration_seconds,
    minimum_order_seconds,
    narration_budget_seconds,
    physical_minimum_order_seconds,
    validate_website_review_order,
    website_review_ordered_seconds,
)
from src.services.stages.website_review.script import WebsiteReviewScriptStage

_CARDS = {"enabled": True}


def test_deliverable_order_wins_over_brief_and_default() -> None:
    assert (
        website_review_ordered_seconds(
            {"deliverable": {"target_duration_seconds": 10}, "brief": {"duration_seconds": 90}}
        )
        == 10.0
    )
    assert website_review_ordered_seconds({"brief": {"duration_seconds": 90}}) == 90.0
    assert (
        website_review_ordered_seconds({"deliverable": {"target_duration_seconds": None}}) == 120.0
    )


@pytest.mark.parametrize(
    ("cfg", "seconds"),
    [(None, 0.0)],
)
def test_fixed_seconds_mirror_the_title_card_block(cfg: Any, seconds: float) -> None:
    assert fixed_non_narration_seconds({"title_card": cfg}) == seconds


def test_prod_order_leaves_eight_seconds_of_narration() -> None:
    """Was 4.0s (10s - two fixed 3.0s cards): the film was mostly silent card time.
    Cards are capped at 20% of the order, so a 10s order carries one 2.0s intro."""
    snap = {"deliverable": {"target_duration_seconds": 10}, "title_card": _CARDS}
    assert fixed_non_narration_seconds(snap) == pytest.approx(2.0)
    assert narration_budget_seconds(snap) == pytest.approx(8.0)


@pytest.mark.parametrize(
    ("ordered", "cards", "narration"),
    [(60, 6.0, 54.0)],
)
def test_cards_never_take_more_than_a_fifth_of_the_order(
    ordered: float, cards: float, narration: float
) -> None:
    snap = {"deliverable": {"target_duration_seconds": ordered}, "title_card": _CARDS}
    assert fixed_non_narration_seconds(snap) == pytest.approx(cards)
    assert narration_budget_seconds(snap) == pytest.approx(narration)
    assert fixed_non_narration_seconds(snap) <= 0.20 * ordered


# The budget-vs-render agreement (order minus RENDERED cards == the narration budget)
# is pinned end to end — real script output into the real compile — in
# ``test_title_card_reservation.py``; the compile no longer plans the cards itself.


def _script_ctx(snapshot: dict[str, Any]) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"voice": {"descriptor": "warm"}, **snapshot},
        prior_run_outputs={
            "website_critique": {"url": "https://acme.test", "findings": []},
            "website_redesign": {"style": "modern", "visuals": []},
        },
    )


def test_script_prompt_budgets_the_deliverable_order_net_of_cards() -> None:
    ctx = _script_ctx(
        {
            "brief": {"duration_seconds": 120},
            "deliverable": {"target_duration_seconds": 10},
            "title_card": _CARDS,
        }
    )
    stage = WebsiteReviewScriptStage(client=None)
    stage._wpm = 150.0
    prompt = stage.build_prompt(ctx)
    # 10s order - one 2.0s intro card (20% cap) = 8s of narration
    assert "Write a 8-second website review" in prompt
    # 8s at 150 wpm = 20 words — not 120s/300 words, not 10s/25 words
    assert "8 seconds is 20 words" in prompt


def test_compile_grades_against_the_deliverable_order() -> None:
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"deliverable": {"target_duration_seconds": 10}, "title_card": _CARDS},
        prior_run_outputs={},
    )
    out = WebsiteReviewCompileStage(client=None).build_output(
        {"artifact_url": "http://stub/r.mp4", "duration_seconds": 15.1}, ctx
    )
    assert out.duration_conformance is not None
    assert out.duration_conformance["ordered_seconds"] == 10.0
    assert out.duration_conformance["within_tolerance"] is False


def test_a_critique_only_run_has_no_floor() -> None:
    """No film ordered, no floor: ``brief.generate_video: False`` skips the check."""
    snap: dict[str, Any] = {"deliverable": {"target_duration_seconds": 5}, "title_card": _CARDS}
    with pytest.raises(OrderBelowFloorError):
        validate_website_review_order(snap)
    validate_website_review_order({**snap, "brief": {"generate_video": False}})


# ── the floor is 10s, EVERYWHERE ────────────────────────────────────────────────
#
# #1019 derived a PHYSICAL floor — cards + the two narrated acts — and got 8.4s with
# cards on, 2.4s off. The server accepted a 10s order while the wizard refused anything
# under 30s: two doors, two numbers, and the shorter one produced films nobody wanted.
#
# Owner decision 2026-09-15: the floor is **10s everywhere**. It sits ABOVE both
# physical floors, so nothing it accepts is beyond the compiler; and it does not move
# with ``title_card``, which is the property #1019 was reaching for.


def test_the_floor_is_ten_seconds_with_title_cards_on_or_off() -> None:
    """The ONE number, and it does not move with the ``title_card`` block."""
    bare: dict[str, Any] = {}
    carded: dict[str, Any] = {"title_card": _CARDS}
    assert MINIMUM_ORDER_SECONDS == 10.0
    assert minimum_order_seconds(bare) == pytest.approx(10.0)
    assert minimum_order_seconds(carded) == pytest.approx(10.0)
    # ...unlike the physical floor it supersedes, which DOES move with the cards.
    assert physical_minimum_order_seconds(carded) - physical_minimum_order_seconds(
        bare
    ) == pytest.approx(6.0)


@pytest.mark.parametrize("cards", [None])
def test_nine_seconds_is_refused_and_ten_is_accepted(cards: Any) -> None:
    """The exact acceptance the owner asked for, on BOTH title-card branches.

    9s was accepted before this change (it clears the 8.4s cards-on physical floor and
    the 2.4s cards-off one). Both halves matter: the accept alone passes with no check
    at all, and the refusal alone passes with the floor set absurdly high.
    """
    snap: dict[str, Any] = {"title_card": cards} if cards else {}
    with pytest.raises(OrderBelowFloorError) as exc:
        validate_website_review_order({**snap, "deliverable": {"target_duration_seconds": 9}})
    assert exc.value.refusal.ordered_seconds == pytest.approx(9.0)
    assert exc.value.refusal.minimum_seconds == pytest.approx(10.0)
    # The message names BOTH numbers — an operator must not have to guess either.
    assert "9s" in exc.value.refusal.message
    assert "10s" in exc.value.refusal.message

    validate_website_review_order({**snap, "deliverable": {"target_duration_seconds": 10}})


def test_the_refusal_does_not_blame_title_cards() -> None:
    """The remedy is a longer order, full stop.

    The old message offered "or disable title_card" — advice that now changes nothing,
    since turning the cards off does not lower the floor.
    """
    with pytest.raises(OrderBelowFloorError) as exc:
        validate_website_review_order(
            {"deliverable": {"target_duration_seconds": 1}, "title_card": _CARDS}
        )
    msg = exc.value.refusal.message
    assert "disable title_card" not in msg
    assert msg.endswith("Order at least 10s.")
    assert exc.value.refusal.reason == "order_below_floor"


def test_exotic_cards_can_no_longer_outrun_the_order() -> None:
    """A 2 x 20s ``title_card`` block used to raise the floor to 42.4s and refuse a 30s
    order. Cards are now capped at 20% of the order, so the same block on a 30s order
    is 2 x 3.0s, the physical minimum is 8.4s, and the product's 10s floor governs."""
    long_cards = {"title_card": {"enabled": True, "duration_seconds": 20}}
    snap = {**long_cards, "deliverable": {"target_duration_seconds": 30}}
    assert fixed_non_narration_seconds(snap) == pytest.approx(6.0)
    assert physical_minimum_order_seconds(snap) == pytest.approx(8.4)
    assert minimum_order_seconds(snap) == pytest.approx(10.0)
    validate_website_review_order(snap)


def test_the_narrated_minimum_is_the_two_acts_the_blueprint_guarantees() -> None:
    """One "before" (critique) line and one "after" (reveal) line — the two acts
    ``script._TWO_ACT_STRUCTURE_BLOCK`` demands and ``script._protected_indices``
    refuses to let the trim drop. Not a chosen number."""
    assert MIN_NARRATED_LINES == len(REQUIRED_NARRATED_PHASES) == 2
    assert minimum_narration_seconds() == pytest.approx(
        MIN_NARRATED_LINES * MIN_NARRATED_BEAT_SECONDS
    )
    # The compiler's flash-cut floor is NOT the product floor and must stay put.
    assert pytest.approx(1.2) == MIN_NARRATED_BEAT_SECONDS


def test_project_create_check_is_not_a_pydantic_validator() -> None:
    """Construction must SUCCEED — the floor is a 400 at the route, not a 422.

    If the check ever moves back onto a ``model_validator`` this reds, because pydantic
    would raise here instead of at :meth:`check_website_review_order`.
    """
    body = {
        "type": "website_review",
        "title": "t",
        "brief": {"title_card": _CARDS},
        "deliverables": [{"name": "d", "format": "16x9", "target_duration_seconds": 9}],
    }
    model = ProjectCreate.model_validate(body)  # no ValidationError
    with pytest.raises(OrderBelowFloorError):
        model.check_website_review_order()

    ok = {**body, "deliverables": [{"name": "d", "format": "16x9", "target_duration_seconds": 10}]}
    ProjectCreate.model_validate(ok).check_website_review_order()
