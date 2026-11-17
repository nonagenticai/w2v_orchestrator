"""Tests for promo_visual_director — line→screen mapping for frame-synced demo."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.services.stages.promo.visual_director import (
    PromoVisualDirectorInput,
    PromoVisualDirectorStage,
)


def _ctx(script: dict, storyboard: dict, capture: dict) -> MagicMock:
    ctx = MagicMock()
    ctx.prior_output.side_effect = lambda key: {
        "promo_script": script,
        "promo_storyboard": storyboard,
        "site_capture": capture,
    }.get(key)
    return ctx


def test_footage_lines_picks_only_footage_shots() -> None:
    script = {
        "lines": [
            {"id": "l1", "text": "A?", "speaker": "a"},
            {"id": "l2", "text": "B", "speaker": "b"},
            {"id": "l3", "text": "C", "speaker": "b"},
        ]
    }
    storyboard = {
        "shots": [
            {"line_id": "l1", "visual_type": "slide"},
            {"line_id": "l2", "visual_type": "footage"},
            {"line_id": "l3", "visual_type": "ai_image"},
        ]
    }
    out = PromoVisualDirectorStage._footage_lines(_ctx(script, storyboard, {}))
    assert out == [{"line_id": "l2", "text": "B", "speaker": "b"}]


async def test_handle_maps_lines_to_screens() -> None:
    script = {
        "lines": [
            {"id": "l1", "text": "See the leaderboard.", "speaker": "b"},
            {"id": "l2", "text": "And the build sandbox.", "speaker": "b"},
        ]
    }
    storyboard = {
        "shots": [
            {"line_id": "l1", "visual_type": "footage"},
            {"line_id": "l2", "visual_type": "footage"},
        ]
    }
    capture = {
        "screens": [
            {"id": "leaderboard", "label": "Challenges leaderboard"},
            {"id": "build", "label": "Build sandbox"},
        ]
    }
    fake = MagicMock()
    fake.message = AsyncMock(
        return_value=SimpleNamespace(
            output={
                "assignments": [
                    {"line_id": "l1", "screen_id": "leaderboard"},
                    {"line_id": "l2", "screen_id": "build"},
                ]
            }
        )
    )
    stage = PromoVisualDirectorStage(client=fake)
    out = await stage.handle(PromoVisualDirectorInput(), _ctx(script, storyboard, capture))
    by_line = {a.line_id: a.screen_id for a in out.assignments}
    assert by_line == {"l1": "leaderboard", "l2": "build"}


async def test_handle_llm_failure_falls_back_to_fill() -> None:
    script = {
        "lines": [
            {"id": "l1", "text": "a", "speaker": "b"},
            {"id": "l2", "text": "b", "speaker": "b"},
        ]
    }
    storyboard = {
        "shots": [
            {"line_id": "l1", "visual_type": "footage"},
            {"line_id": "l2", "visual_type": "footage"},
        ]
    }
    capture = {"screens": [{"id": "home", "label": "Home"}]}
    fake = MagicMock()
    fake.message = AsyncMock(side_effect=RuntimeError("LLM down"))
    stage = PromoVisualDirectorStage(client=fake)
    out = await stage.handle(PromoVisualDirectorInput(), _ctx(script, storyboard, capture))
    # Soft-fail: every footage line still resolves (to the only screen).
    by_line = {a.line_id: a.screen_id for a in out.assignments}
    assert by_line == {"l1": "home", "l2": "home"}
    assert out.notes  # records the degradation


async def test_a_shotlist_binding_is_not_reported_as_a_warning() -> None:
    """``notes`` is the degraded channel: the deterministic shotlist binding wrote an
    informational note there, and every shotlist run showed it to the user as an
    "Unclassified note" warning (prod run 01a17e6f)."""
    from src.services.stages.shared.degradation import ensure_degradation_codes, is_degraded

    ctx = MagicMock()
    ctx.config_snapshot = {
        "brief": {"shotlist": [{"kind": "url", "url": "https://app.example/", "narration": "x"}]}
    }
    stage = PromoVisualDirectorStage(client=MagicMock())
    out = await stage.handle(PromoVisualDirectorInput(), ctx)
    dumped = out.model_dump(mode="json")
    assert [a.screen_id for a in out.assignments] == [a.line_id for a in out.assignments]
    assert len(out.assignments) == 1
    assert out.notes == []
    assert not ensure_degradation_codes("promo_visual_director", dumped)
    assert not is_degraded(dumped)
