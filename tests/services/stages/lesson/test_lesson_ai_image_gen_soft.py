"""A lesson's failed ``ai_image_gen`` degrades to plain slides instead of failing.

``lesson_record`` depends on ``ai_image_gen``; with the generic stage a provider
failure failed the lesson, and ``course_compile_lessons`` then failed the course.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.services.blueprints.base import stage_handler_class
from src.services.blueprints.lesson import LESSON_BLUEPRINT
from src.services.runs.run_context import RunContext
from src.services.stages.base import StageSkippedError
from src.services.stages.lesson.ai_image_gen import LessonAiImageGenStage
from src.services.stages.shared.degradation import AI_IMAGES_UNAVAILABLE

pytestmark = pytest.mark.asyncio


def _ctx(block: dict | None) -> RunContext:
    snapshot = {"ai_image_gen": block} if block is not None else {}
    return RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot=snapshot)


async def test_lesson_blueprint_wires_the_soft_image_stage() -> None:
    spec = next(s for s in LESSON_BLUEPRINT.stages if s.name == "ai_image_gen")
    assert stage_handler_class(spec) is LessonAiImageGenStage


async def test_provider_failure_completes_empty_with_degradation() -> None:
    client = AsyncMock()
    client.create_concept_image = AsyncMock(side_effect=RuntimeError("provider down"))
    stage = LessonAiImageGenStage(client=client)
    out = await stage.handle(
        stage.input_model.model_validate({}), _ctx({"prompts": ["a slide picture"]})
    )
    assert out.images == []
    assert [d["code"] for d in out.degradations] == [AI_IMAGES_UNAVAILABLE]
    assert out.notes


async def test_skip_is_still_a_skip() -> None:
    stage = LessonAiImageGenStage(client=AsyncMock())
    with pytest.raises(StageSkippedError):
        await stage.handle(stage.input_model.model_validate({}), _ctx(None))
