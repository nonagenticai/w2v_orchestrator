from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.errors import StageSkippedError
from src.services.stages.seo_audit.compile import SeoAuditCompileStage


def _ctx(*, generate_video: bool = True) -> RunContext:
    brief: dict = {}
    if generate_video:
        brief["generate_video"] = True
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": brief,
            "deliverable": {"format": "16x9", "captions_mode": "none"},
        },
        prior_run_outputs={
            "seo_audit_slides": {
                "slides": [
                    {"id": "s1", "line_ids": ["l1"], "title": "T1", "bullets": ["a"]},
                    {"id": "s2", "line_ids": ["l2"], "title": "T2", "bullets": ["b"]},
                ]
            },
            "seo_audit_narrate": {
                "clips": [
                    {
                        "id": "c1",
                        "line_id": "l1",
                        "artifact_url": "http://a/c1.wav",
                        "seconds": 4.0,
                    },
                    {
                        "id": "c2",
                        "line_id": "l2",
                        "artifact_url": "http://a/c2.wav",
                        "seconds": 5.0,
                    },
                ]
            },
            "seo_audit_record": {
                "recordings": [
                    {"id": "r1", "slide_id": "s1", "image_url": "http://img/s1.png"},
                    {"id": "r2", "slide_id": "s2", "image_url": "http://img/s2.png"},
                ]
            },
        },
    )


@pytest.mark.asyncio
async def test_compile_skips_when_video_off() -> None:
    fake = AsyncMock()
    stage = SeoAuditCompileStage(client=fake)
    with pytest.raises(StageSkippedError):
        await stage.run({}, ctx=_ctx(generate_video=False))
    fake.compile.assert_not_called()


@pytest.mark.asyncio
async def test_compile_emits_one_shot_per_line_with_image_and_audio() -> None:
    fake = AsyncMock()
    fake.compile.return_value = {
        "artifact_url": "http://stub/seo.mp4",
        "duration_seconds": 9.0,
    }
    stage = SeoAuditCompileStage(client=fake)

    out = await stage.run({}, ctx=_ctx())

    assert out.artifact_url == "http://stub/seo.mp4"
    assert out.duration_seconds == 9.0
    _, kwargs = fake.compile.call_args
    shots = kwargs["shots"]
    assert len(shots) == 2
    # Each line's shot carries its slide image + its own narration audio.
    assert shots[0]["image_url"] == "http://img/s1.png"
    assert shots[0]["audio_url"] == "http://a/c1.wav"
    assert shots[0]["seconds"] == 4.0
    assert shots[1]["image_url"] == "http://img/s2.png"
    assert shots[1]["audio_url"] == "http://a/c2.wav"
    # Built from narration, not a storyboard: no `description: null` on the wire.
    assert "description" not in shots[0]
    assert kwargs["tags"]["format"] == "16x9"
