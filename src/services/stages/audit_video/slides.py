"""AuditSlidesStageBase — project the audit narration lines onto branded slides.

Opt-in (``brief.generate_video``) and pure-deterministic (no LLM), mirroring
``lesson_slides``: one slide per spoken ``<type>_script`` line — the first line
(the score/grade headline) becomes the cover slide, each quick-win line its own
point slide — with a clean title + bullet derived on whole-sentence boundaries.
Each slide keeps its ``line_ids`` so ``<type>_compile`` can join a narrated line
back to its slide image. Skips cleanly when the video variant is off.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from uuid import uuid7

from src.services.stages.audit_video.gate import require_audit_video
from src.services.stages.audit_video.models import (
    AuditSlide,
    AuditSlidesInput,
    AuditSlidesOutput,
)
from src.services.stages.base import StageHandler

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

_LINES_PER_SLIDE = 1
_TITLE_MAX = 72
_BULLET_MAX = 160
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _normalize(text: str) -> str:
    return " ".join(str(text).split())


def _first_sentence(text: str) -> str:
    text = _normalize(text)
    if not text:
        return ""
    return _SENTENCE_SPLIT.split(text, maxsplit=1)[0].strip()


def _clip_words(text: str, limit: int) -> str:
    """Trim to ``limit`` chars on a word boundary — never mid-word."""
    text = _normalize(text)
    if len(text) <= limit:
        return text
    head = text[:limit].rsplit(" ", 1)[0].rstrip(",;:-—")
    return head or text[:limit]


class AuditSlidesStageBase(StageHandler[AuditSlidesInput, AuditSlidesOutput]):
    """Shared audit-slides stage. Per-type subclass sets ``name`` only."""

    input_model = AuditSlidesInput
    output_model = AuditSlidesOutput

    async def handle(self, payload: AuditSlidesInput, ctx: RunContext) -> AuditSlidesOutput:
        require_audit_video(ctx)
        prefix = self.name.removesuffix("_slides")
        script = ctx.prior_output(f"{prefix}_script")
        if not script:
            raise RuntimeError(f"{prefix}_script output missing")
        lines = script["lines"]

        slides: list[AuditSlide] = []
        groups = [lines[i : i + _LINES_PER_SLIDE] for i in range(0, len(lines), _LINES_PER_SLIDE)]
        for grp in groups:
            lead = grp[0]["text"]
            title = _clip_words(_first_sentence(lead) or lead, _TITLE_MAX)
            bullets = [_clip_words(line["text"], _BULLET_MAX) for line in grp]
            slides.append(
                AuditSlide(
                    id=str(uuid7()),
                    line_ids=[line["id"] for line in grp],
                    title=title,
                    bullets=bullets,
                )
            )
        return AuditSlidesOutput(slides=slides)
