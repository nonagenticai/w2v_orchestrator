"""Tests for promo_image_prompts — per-line image-prompt selection."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.services.stages.promo.image_prompts import (
    PromoImagePromptsStage,
)


def _ctx(script: dict, storyboard: dict) -> MagicMock:
    ctx = MagicMock()
    ctx.config_snapshot = {}
    ctx.prior_output.side_effect = lambda key: {
        "promo_script": script,
        "promo_storyboard": storyboard,
    }.get(key)
    return ctx


def test_ai_image_lines_picks_only_ai_image_shots() -> None:
    script = {
        "lines": [{"id": "l1", "text": "A"}, {"id": "l2", "text": "B"}, {"id": "l3", "text": "C"}]
    }
    storyboard = {
        "shots": [
            {"line_id": "l1", "visual_type": "slide"},
            {"line_id": "l2", "visual_type": "ai_image"},
            {"line_id": "l3", "visual_type": "footage"},
        ]
    }
    out = PromoImagePromptsStage._ai_image_lines(_ctx(script, storyboard))
    assert out == [{"line_id": "l2", "text": "B"}]


def test_every_image_prompt_ends_by_forbidding_text() -> None:
    """The image model sees the prompt alone; run 01a11082 shipped a still with
    garbled letters ("PLIC") although the prompt-WRITING model was told no text."""
    from src.services.stages.promo.image_prompts import NO_TEXT_CLAUSE, textless

    prompt = "A glass facade at dusk, teal reflections"
    assert textless(prompt) == f"{prompt}. {NO_TEXT_CLAUSE}"
    assert textless(f"{prompt}.") == f"{prompt}. {NO_TEXT_CLAUSE}"
    assert textless(textless(prompt)) == textless(prompt)  # exactly once
    assert "no letters" in NO_TEXT_CLAUSE and "no text" in NO_TEXT_CLAUSE.lower()


async def test_a_per_line_motion_run_sends_no_prompt_without_the_no_text_clause() -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from src.services.stages.promo.image_prompts import NO_TEXT_CLAUSE, PromoImagePromptsStage

    wm = MagicMock()
    wm.message = AsyncMock(
        return_value=SimpleNamespace(
            output={"prompts": [{"line_id": "l1", "prompt": "A sleek interface of glowing nodes"}]},
            usage=None,
        )
    )
    ctx = _ctx(
        {"lines": [{"id": "l1", "text": "Thirty capabilities."}]},
        {"shots": [{"line_id": "l1", "visual_type": "ai_image"}]},
    )
    # The clause is part of per-line motion; without the flag the prompt is sent
    # verbatim (pinned to origin/main in ``test_promo_per_line_gates``).
    ctx.config_snapshot = {"direction": {"motion": "per_line"}}
    out = await PromoImagePromptsStage(client=wm).handle(MagicMock(), ctx)
    assert [p.prompt for p in out.prompts] == [
        f"A sleek interface of glowing nodes. {NO_TEXT_CLAUSE}"
    ]
