"""Generic film regressions from live run 01a156a4-e906 (2026-10-20).

Every shot rendered the placeholder "Visual for: <line>" text card (no stage fed
ai_image_gen), and a 12.3s master against a 30s order (-59%) shipped with
``duration_conformance = null`` and no degradation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.generic.ai_image_gen import AiImageGenStage
from src.services.stages.generic.storyboard_draft import StoryboardDraftStage
from src.services.stages.generic.video_render import VideoRenderStage
from src.services.stages.shared.degradation import (
    DURATION_NOT_ORDERED,
    DURATION_OUT_OF_TOLERANCE,
)
from src.services.stages.shared.duration_conformance import generic_ordered_seconds

_SCRIPT = {
    "lines": [
        {"id": "l1", "text": "Solar panels turn light into power.", "seconds": 3.0},
        {"id": "l2", "text": "Batteries store it for the night.", "seconds": 3.0},
    ],
    "total_seconds": 6.0,
}


def _ctx(config: dict | None = None, prior: dict | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=config or {},
        prior_run_outputs=prior or {},
    )


# --- storyboard → per-line image prompts ------------------------------------


@pytest.mark.asyncio
async def test_storyboard_emits_one_image_prompt_per_line_and_no_placeholder() -> None:
    out = await StoryboardDraftStage(client=AsyncMock()).run(
        {}, ctx=_ctx({"title": "Home Energy"}, {"script_draft": _SCRIPT})
    )
    assert [p.line_id for p in out.prompts] == ["l1", "l2"]
    assert "Solar panels turn light into power." in out.prompts[0].prompt
    assert "Home Energy" in out.prompts[0].prompt
    assert all("Visual for" not in s.description for s in out.shots)
    assert out.shots[0].description == "Solar panels turn light into power."


@pytest.mark.asyncio
async def test_ai_image_gen_resolves_storyboard_prompts_keyed_by_line() -> None:
    story = await StoryboardDraftStage(client=AsyncMock()).run(
        {}, ctx=_ctx({"title": "T"}, {"script_draft": _SCRIPT})
    )
    cfg = {"prompts_from_stage": "storyboard_draft"}
    ctx = _ctx({"ai_image_gen": cfg}, {"storyboard_draft": story.model_dump()})
    pairs = AiImageGenStage._resolve_prompts(cfg, ctx)
    assert [line_id for _, line_id in pairs] == ["l1", "l2"]


# --- video_render keys AI stills by line ------------------------------------


def _compile_fake(duration: float = 30.0) -> AsyncMock:
    fake = AsyncMock()
    fake.compile.return_value = {"artifact_url": "s3://x/y.mp4", "duration_seconds": duration}
    return fake


_SHOTS = {
    "shots": [
        {"id": "s1", "line_id": "l1", "description": "a", "seconds": 3.0},
        {"id": "s2", "line_id": "l2", "description": "b", "seconds": 3.0},
    ]
}


# --- duration conformance ---------------------------------------------------


def test_generic_ordered_seconds_ladder() -> None:
    assert generic_ordered_seconds({"brief": {"target_duration_seconds": 45}}) == 45.0
    assert generic_ordered_seconds({"deliverable": {"target_duration_seconds": 30}}) == 30.0
    assert (
        generic_ordered_seconds(
            {
                "brief": {"target_duration_seconds": 45},
                "deliverable": {"target_duration_seconds": 30},
            }
        )
        == 45.0
    )
    assert generic_ordered_seconds({}) == 60.0


@pytest.mark.asyncio
async def test_video_render_records_conformance_and_degrades_a_59pct_short_film() -> None:
    fake = _compile_fake(duration=12.33)
    out = await VideoRenderStage(client=fake).run(
        {},
        ctx=_ctx(
            {"deliverable": {"target_duration_seconds": 30}},
            {"storyboard_draft": _SHOTS},
        ),
    )
    assert out.duration_conformance is not None
    assert out.duration_conformance["ordered_seconds"] == 30.0
    assert out.duration_conformance["within_tolerance"] is False
    codes = {d["code"] for d in out.degradations}
    assert DURATION_OUT_OF_TOLERANCE in codes, "a -59% film must surface as a degradation"


@pytest.mark.asyncio
async def test_video_render_on_target_film_has_no_duration_degradation() -> None:
    fake = _compile_fake(duration=30.5)
    out = await VideoRenderStage(client=fake).run(
        {},
        ctx=_ctx(
            {"deliverable": {"target_duration_seconds": 30}},
            {"storyboard_draft": _SHOTS},
        ),
    )
    assert out.duration_conformance is not None
    assert out.duration_conformance["within_tolerance"] is True
    codes = {d["code"] for d in out.degradations}
    assert DURATION_OUT_OF_TOLERANCE not in codes
    assert DURATION_NOT_ORDERED not in codes
