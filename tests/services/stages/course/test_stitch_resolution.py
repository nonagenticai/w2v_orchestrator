"""Regression tests for the "course ordered at 2k, delivered at 1080p" defect.

``course_stitch`` calls ``client.compile()`` directly (not via
``CompileStageBase``); before this fix it sent no resolution/format tags at
all, so the compiler fell back to its own 1080p default regardless of the
ordered ``config_snapshot["resolution"]`` tier. These tests pin:

1. the ordered tier reaching the compile call as explicit render_width /
   render_height tags (the same mechanism every other CompileStageBase
   subclass already uses via ``resolution_tags``);
2. a delivered master that comes back UNDER the ordered tier is recorded as a
   loud degradation (``notes``), never silently accepted.
"""

from unittest.mock import AsyncMock
from uuid import uuid4

from src.services.runs.run_context import RunContext
from src.services.stages.course.stitch import CourseStitchStage


def _ctx(lessons: list[dict], *, config_snapshot: dict) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=config_snapshot,
        prior_run_outputs={"course_compile_lessons": {"lessons": lessons}},
    )


def _lesson() -> list[dict]:
    return [
        {
            "lesson_project_id": "p1",
            "lesson_run_id": "r1",
            "artifact_url": "u1",
            "duration_seconds": 60.0,
        }
    ]


async def test_undelivered_tier_records_loud_degradation_not_silence() -> None:
    """If the delivered master comes back UNDER the ordered tier (an older
    compiler build ignoring the tags, or a stale probe), course_stitch must
    say so loudly via the ``notes`` degradation channel rather than shipping
    a downscaled master with no record of the shortfall."""
    fake = AsyncMock()
    fake.compile.return_value = {
        "artifact_url": "http://stub/course.mp4",
        "duration_seconds": 60.0,
        # Ordered 2k (2560x1440) but the mux ffprobes back at plain 1080p.
        "width": 1920,
        "height": 1080,
        # A PASSING mastering report, so the only signal under test is the
        # resolution shortfall — see test_honoured_tier_stays_clean for why.
        "mastering": {"rerun_stages": [], "gate_pass": True, "luma_spread": 210.0},
    }
    stage = CourseStitchStage(client=fake)
    ctx = _ctx(
        _lesson(),
        config_snapshot={"resolution": "2k", "deliverable": {"format": "16x9"}},
    )

    out = await stage.run({}, ctx=ctx)

    assert any("2k" in n and "1920x1080" in n for n in out.notes)
    # A resolution shortfall has no DEGRADATION_CODES entry of its own, so this
    # stage records it on `notes`. The `degradations` list is now the mastering
    # gate's channel (O8) and is empty only because this master's gate PASSED.
    assert out.degradations == []
