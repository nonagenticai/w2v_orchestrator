"""Lesson-scoped ``ai_image_gen``: a failed image pass degrades, never fails.

``lesson_record`` depends on ``ai_image_gen`` and the DAG has no soft-dependency
kind, so a failed image stage blocked ``lesson_record`` → failed the lesson →
``course_compile_lessons`` raised ``ChildRunError`` → failed the whole COURSE,
over pictures the slide renderer already treats as optional (a slide with no
image renders title + bullets). This subclass makes the dependency effectively
soft for lessons only: any failure of the generic stage completes with zero
images and a coded ``ai_images_unavailable`` degradation. The generic stage —
whose other blueprints DO rely on it failing loud — is unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.services.stages.base import StageSkippedError
from src.services.stages.generic.ai_image_gen import (
    AiImageGenInput,
    AiImageGenOutput,
    AiImageGenStage,
)
from src.services.stages.shared.degradation import AI_IMAGES_UNAVAILABLE, degradation

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)


class LessonAiImageGenStage(AiImageGenStage):
    """``ai_image_gen`` whose failure degrades the lesson to plain slides."""

    async def handle(self, payload: AiImageGenInput, ctx: RunContext) -> AiImageGenOutput:
        try:
            return await super().handle(payload, ctx)
        except StageSkippedError:
            raise
        except Exception as exc:  # noqa: BLE001 -- degrade: images are optional for a lesson
            log.warning("lesson ai_image_gen failed for run %s; plain slides: %s", ctx.run_id, exc)
            detail = (
                f"ai_image_gen failed ({type(exc).__name__}: {exc}); slides render without images"
            )
            return AiImageGenOutput(
                images=[],
                notes=[detail],
                degradations=[degradation(AI_IMAGES_UNAVAILABLE, detail)],
            )
