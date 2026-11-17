from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.clients.schemas.tts import TTSSynthesis
from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.errors import StageSkippedError
from src.services.stages.seo_audit.narrate import SeoAuditNarrateStage


class _FakeStorage:
    def __init__(self) -> None:
        self.stored: list[tuple[str, str, str, int]] = []

    async def initialize(self) -> None:
        pass

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        self.stored.append((job_id, stage_name, filename, len(data)))
        return f"artifacts/{job_id}/{stage_name}/{filename}"

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        return f"https://minio.test/{path}"


@pytest.fixture
def fake_storage():
    storage = _FakeStorage()
    tts_storage.set_storage_for_tests(storage)
    yield storage
    tts_storage.set_storage_for_tests(None)


def _ctx(lines: list[dict], *, generate_video: bool = True) -> RunContext:
    brief: dict = {}
    if generate_video:
        brief["generate_video"] = True
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": brief, "voice": {"descriptor": "clear"}},
        prior_run_outputs={"seo_audit_script": {"lines": lines}},
    )


@pytest.mark.asyncio
async def test_narrate_skips_when_video_off(fake_storage: _FakeStorage) -> None:
    fake = AsyncMock()
    stage = SeoAuditNarrateStage(client=fake)
    with pytest.raises(StageSkippedError):
        await stage.run(
            {}, ctx=_ctx([{"id": "l1", "text": "x", "seconds": 3.0}], generate_video=False)
        )
    fake.synthesize.assert_not_called()


@pytest.mark.asyncio
async def test_narrate_emits_clip_per_line(fake_storage: _FakeStorage) -> None:
    fake = AsyncMock()
    fake.synthesize.side_effect = [
        TTSSynthesis(
            audio_bytes=b"\x01\x02", sample_rate=0, duration_seconds=4.0, provider="", voice=""
        ),
        TTSSynthesis(
            audio_bytes=b"\x03\x04", sample_rate=0, duration_seconds=5.0, provider="", voice=""
        ),
    ]
    stage = SeoAuditNarrateStage(client=fake)
    ctx = _ctx(
        [
            {"id": "l1", "text": "Scored 72.", "seconds": 4.0},
            {"id": "l2", "text": "Add OG tags.", "seconds": 5.0},
        ]
    )

    out = await stage.run({}, ctx=ctx)

    assert len(out.clips) == 2
    by_line = {c.line_id: c for c in out.clips}
    assert set(by_line) == {"l1", "l2"}
    assert by_line["l1"].seconds == 4.0
    assert all(c.artifact_url.startswith("https://minio.test/artifacts/") for c in out.clips)
    assert all(stage_name == "seo_audit_narrate" for _, stage_name, _, _ in fake_storage.stored)
