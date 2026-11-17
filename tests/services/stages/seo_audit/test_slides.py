from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.errors import StageSkippedError
from src.services.stages.seo_audit.slides import SeoAuditSlidesStage


def _ctx(lines: list[dict], *, generate_video: bool = True) -> RunContext:
    brief: dict = {}
    if generate_video:
        brief["generate_video"] = True
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": brief, "brand": {}},
        prior_run_outputs={"seo_audit_script": {"lines": lines, "total_seconds": 12.0}},
    )


@pytest.mark.asyncio
async def test_slides_skip_when_video_off() -> None:
    stage = SeoAuditSlidesStage()
    with pytest.raises(StageSkippedError):
        await stage.run(
            {}, ctx=_ctx([{"id": "l1", "text": "x", "seconds": 3.0}], generate_video=False)
        )


@pytest.mark.asyncio
async def test_slides_one_per_line_keyed_by_line_id() -> None:
    lines = [
        {"id": "l1", "text": "Your site scored 72, a grade C.", "seconds": 4.0},
        {
            "id": "l2",
            "text": "Add OpenGraph tags for better sharing. It boosts CTR.",
            "seconds": 5.0,
        },
    ]
    stage = SeoAuditSlidesStage()

    out = await stage.run({}, ctx=_ctx(lines))

    assert len(out.slides) == 2
    assert all(s.id for s in out.slides)
    # Each slide carries exactly its source line id so compile can join it.
    assert out.slides[0].line_ids == ["l1"]
    assert out.slides[1].line_ids == ["l2"]
    # The score-headline line becomes the cover slide title.
    assert "72" in out.slides[0].title
    # Title trims to the first sentence (whole-word), bullet carries the point.
    assert "Add OpenGraph tags" in out.slides[1].title
    assert "It boosts CTR" not in out.slides[1].title
    assert out.slides[1].bullets
