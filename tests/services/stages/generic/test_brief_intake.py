from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.generic.brief_intake import BriefIntakeStage


@pytest.mark.asyncio
async def test_brief_intake_defaults() -> None:
    stage = BriefIntakeStage()
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
    )
    out = await stage.run({}, ctx=ctx)
    assert out.title == "Untitled"
    assert out.angle == "informational"
    assert out.audience == "general"
    assert out.target_duration_seconds == 60
    assert out.voice_descriptor == "neutral"
