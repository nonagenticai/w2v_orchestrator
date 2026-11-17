"""audiobook_plan — split each chapter's text into TTS-safe chunks.

Read-only over ``audiobook_ingest``; produces the ordered chunk plan the narrate
stage spends TTS on. ``default_pause_after`` lets the user review (and the FE
display) the plan — how many chunks, the chunk text — before narration runs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar
from uuid import uuid7

from src.services.stages.audiobook.parsing import DEFAULT_MAX_CHUNK_CHARS, chunk_text
from src.services.stages.base import (
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    StagePreconditionError,
)

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

logger = logging.getLogger(__name__)

# Flag a chunk that lands within 5% of the chunker cap — a single sentence that
# barely fit is the likeliest thing to trip the TTS backend's 10k hard cap or
# produce an unnaturally long clip.
_NEAR_CAP_CHARS = int(DEFAULT_MAX_CHUNK_CHARS * 0.95)


class AudiobookPlanInput(StageInput):
    pass


class PlannedChunk(StageOutput):
    id: str
    text: str
    sequence: int


class PlannedChapter(StageOutput):
    id: str
    title: str
    sequence: int
    chunks: list[PlannedChunk]


class AudiobookPlanOutput(StageOutputWithNotes):
    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. Empty here means the stage
    # reported success having produced nothing, which degrades the run.
    # The chunk plan every downstream stage falls back to: `audiobook_cast`,
    # `audiobook_direct`, `audiobook_narrate`, `audiobook_qa` and `audiobook_stitch` all
    # read `chapters` (span-mode `audiobook_segment` may legitimately be empty, this
    # may not).
    produces: ClassVar[tuple[str, ...]] = ("chapters",)

    chapters: list[PlannedChapter]
    total_chunks: int


class AudiobookPlanStage(StageHandler[AudiobookPlanInput, AudiobookPlanOutput]):
    name = "audiobook_plan"
    input_model = AudiobookPlanInput
    output_model = AudiobookPlanOutput

    async def handle(
        self,
        payload: AudiobookPlanInput,
        ctx: RunContext,
    ) -> AudiobookPlanOutput:
        ingest = ctx.prior_output("audiobook_ingest")
        if not ingest or not ingest.get("chapters"):
            raise StagePreconditionError("audiobook_plan: audiobook_ingest output missing")

        chapters: list[PlannedChapter] = []
        total = 0
        for chapter in ingest["chapters"]:
            seq = int(chapter.get("sequence") or 0)
            title = str(chapter.get("title") or "Chapter")
            pieces = chunk_text(str(chapter.get("text") or ""))
            chunks = [
                PlannedChunk(id=str(uuid7()), text=piece, sequence=i)
                for i, piece in enumerate(pieces)
            ]
            total += len(chunks)
            lengths = [len(p) for p in pieces]
            max_chars = max(lengths, default=0)
            avg_chars = sum(lengths) // len(lengths) if lengths else 0
            logger.info(
                "audiobook_plan run=%s chapter seq=%d title=%r chunks=%d chars(max=%d,avg=%d)",
                ctx.run_id,
                seq,
                title,
                len(chunks),
                max_chars,
                avg_chars,
            )
            if not chunks:
                logger.warning(
                    "audiobook_plan run=%s chapter seq=%d title=%r produced 0 chunks",
                    ctx.run_id,
                    seq,
                    title,
                )
            elif max_chars >= _NEAR_CAP_CHARS:
                logger.warning(
                    "audiobook_plan run=%s chapter seq=%d has a chunk near the %d-char cap (%d)",
                    ctx.run_id,
                    seq,
                    DEFAULT_MAX_CHUNK_CHARS,
                    max_chars,
                )
            chapters.append(
                PlannedChapter(id=str(chapter["id"]), title=title, sequence=seq, chunks=chunks)
            )

        if total == 0:
            raise StagePreconditionError("audiobook_plan: manuscript produced no narratable chunks")
        logger.info(
            "audiobook_plan run=%s chapters=%d total_chunks=%d",
            ctx.run_id,
            len(chapters),
            total,
        )
        return AudiobookPlanOutput(chapters=chapters, total_chunks=total)
