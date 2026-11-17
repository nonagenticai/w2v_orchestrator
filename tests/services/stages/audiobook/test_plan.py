"""audiobook_plan stage — chapter text → ordered TTS-safe chunks."""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.audiobook.plan import AudiobookPlanStage
from src.services.stages.base import StagePreconditionError


def _ctx(ingest_output: dict) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={"audiobook_ingest": ingest_output},
    )


async def test_plan_chunks_each_chapter() -> None:
    stage = AudiobookPlanStage()
    ctx = _ctx(
        {
            "title": "T",
            "author": "",
            "chapters": [
                {"id": "ch1", "title": "One", "text": "Alpha. Beta. Gamma.", "sequence": 0},
                {"id": "ch2", "title": "Two", "text": "Delta. Epsilon.", "sequence": 1},
            ],
        }
    )
    out = await stage.run({}, ctx=ctx)
    assert [c.id for c in out.chapters] == ["ch1", "ch2"]
    assert out.total_chunks == sum(len(c.chunks) for c in out.chapters)
    assert all(chunk.id for c in out.chapters for chunk in c.chunks)
    # chunk sequences are contiguous within a chapter
    assert out.chapters[0].chunks[0].sequence == 0


async def test_plan_missing_ingest_raises() -> None:
    stage = AudiobookPlanStage()
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    with pytest.raises(StagePreconditionError, match="audiobook_ingest"):
        await stage.run({}, ctx=ctx)


async def test_plan_no_narratable_chunks_raises() -> None:
    stage = AudiobookPlanStage()
    ctx = _ctx({"chapters": [{"id": "c", "title": "Empty", "text": "   ", "sequence": 0}]})
    with pytest.raises(StagePreconditionError, match="no narratable chunks"):
        await stage.run({}, ctx=ctx)
