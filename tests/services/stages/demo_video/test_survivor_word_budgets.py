"""The order's narration budget is re-split over the beats that SURVIVE the plan.

Live run 01a156a6 (showcase, 2026-10-20): bdd wrote 5 beats to 1/5 of a 30s order
each, the plan then dropped them to 3 (``RESIZED_DOWN`` + feasibility refit), and the
three survivors kept their 1/5-of-the-order texts against 1/3-of-the-order footage —
``timeline_underspoken`` + ``narration_rate_floor_clamped`` + ``footage_trimmed``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

from src.services.stages.demo_video.gen import DemoVideoLine, survivor_word_budgets


def _lines(ids: list[str]) -> list[DemoVideoLine]:
    return [DemoVideoLine(id=i, text="x") for i in ids]


def test_the_order_is_split_over_the_survivors_not_the_generated_count() -> None:
    survivors = survivor_word_budgets(_lines(["clip-1", "clip-4", "clip-5"]), 30.0, 150.0)
    assert survivors == {"clip-1": 25, "clip-4": 25, "clip-5": 25}
    assert sum(survivors.values()) == 75
    # The 5-beat split bdd was budgeted against — the shortfall the drop created.
    generated = survivor_word_budgets(_lines([f"clip-{i}" for i in range(1, 6)]), 30.0, 150.0)
    assert generated is not None
    assert set(generated.values()) == {15}


def test_no_order_or_no_rate_means_no_budget() -> None:
    assert survivor_word_budgets(_lines(["clip-1"]), None, 150.0) is None
    assert survivor_word_budgets(_lines(["clip-1"]), 30.0, 0.0) is None
    assert survivor_word_budgets([], 30.0, 150.0) is None


async def test_handle_publishes_the_survivor_budget_after_the_plan_drops_beats(
    monkeypatch: Any,
) -> None:
    """THROUGH ``handle``: 5 generated beats x 8s against 30s -> 3 survivors, 25 words each."""
    from src.clients.schemas.bdd_playwright import BddJobResult
    from src.services.runs.run_context import RunContext
    from src.services.stages.demo_video.gen import DemoVideoGenInput, DemoVideoGenStage

    monkeypatch.setattr(
        "src.services.stages.demo_video.gen.get_settings",
        lambda: SimpleNamespace(
            pipeline=SimpleNamespace(
                structural_relief_autoconverge_enabled=True,
                demo_timeline_underfull_fail_ratio=0.5,
            )
        ),
    )
    client = AsyncMock()
    client.start_demo_video.return_value = "bdd-survivors"
    client.wait_for_job.return_value = {}
    client.get_job_result.return_value = BddJobResult.model_validate(
        {
            "job_id": "bdd-survivors",
            "status": "completed",
            "features_generated": 1,
            "narration_script": {
                "parts": [
                    {
                        "part_id": f"clip-{i}",
                        "text": f"Beat number {i} of the walkthrough.",
                        "expected_recording_seconds": 8.0,
                    }
                    for i in range(1, 6)
                ]
            },
        }
    )
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"target_url": "https://app.example"},
            "direction": {"timing": {"wpm": {"natural": 150.0}}},
            "deliverable": {"target_duration_seconds": 30.0},
        },
    )
    out = await DemoVideoGenStage(client=client).handle(DemoVideoGenInput(), ctx)

    survivors = [line.id for line in out.lines]
    assert len(survivors) < 5, "precondition: the plan dropped beats"
    budgets = out.narration_words_per_beat
    assert budgets is not None
    assert list(budgets) == survivors
    assert sum(budgets.values()) >= 75, "the whole order's words, over the survivors"
