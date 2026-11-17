from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.errors import StageSkippedError
from src.services.stages.seo_audit.record import SeoAuditRecordStage


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


def _ctx(*, generate_video: bool = True, brand: dict | None = None) -> RunContext:
    brief: dict = {}
    if generate_video:
        brief["generate_video"] = True
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": brief,
            "deliverable": {"title": "Acme SEO Audit", "format": "16x9"},
            "brand": brand or {},
            "resolution": "1080p",
        },
        prior_run_outputs={
            "seo_audit_slides": {
                "slides": [
                    {"id": "s1", "line_ids": ["l1"], "title": "Score 72 (C)", "bullets": ["b1"]},
                    {"id": "s2", "line_ids": ["l2"], "title": "Add OG tags", "bullets": ["b2"]},
                ]
            }
        },
    )


def _slide_client() -> AsyncMock:
    fake = AsyncMock()
    fake.resolve_template_id.return_value = "tpl-feature"
    fake.start_deck.return_value = "deck-1"
    fake.wait_for_deck.return_value = {"status": "completed", "slide_count": 2}
    fake.get_slide_png.side_effect = [b"\x89PNG-0", b"\x89PNG-1"]
    return fake


@pytest.mark.asyncio
async def test_record_skips_when_video_off(fake_storage: _FakeStorage) -> None:
    fake = AsyncMock()
    stage = SeoAuditRecordStage(client=fake)
    with pytest.raises(StageSkippedError):
        await stage.run({}, ctx=_ctx(generate_video=False))
    fake.start_deck.assert_not_called()
