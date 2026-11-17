from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.clients.schemas.tts import TTSSynthesis
from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.lesson.narrate import LessonNarrateStage


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


@pytest.mark.asyncio
async def test_narrate_emits_clip_per_paragraph(fake_storage: _FakeStorage) -> None:
    fake = AsyncMock()
    fake.synthesize.side_effect = [
        TTSSynthesis.model_validate(
            {
                "audio_bytes": b"\x01\x02",
                "sample_rate": 24000,
                "duration_seconds": 30.0,
                "provider": "kokoro",
                "voice": "af_heart",
                "format": "wav",
            }
        ),
        TTSSynthesis.model_validate(
            {
                "audio_bytes": b"\x03\x04",
                "sample_rate": 24000,
                "duration_seconds": 60.0,
                "provider": "kokoro",
                "voice": "af_heart",
                "format": "wav",
            }
        ),
    ]
    stage = LessonNarrateStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"voice": {"descriptor": "calm"}},
        prior_run_outputs={
            "lesson_script": {
                "paragraphs": [
                    {"id": "p1", "text": "First.", "seconds": 30.0},
                    {"id": "p2", "text": "Second.", "seconds": 60.0},
                ],
                "total_seconds": 90.0,
                "learning_objective": "x",
            },
        },
    )

    out = await stage.run({}, ctx=ctx)

    assert len(out.clips) == 2
    assert all(c.id for c in out.clips)
    by_paragraph = {c.paragraph_id: c for c in out.clips}
    assert set(by_paragraph) == {"p1", "p2"}
    assert by_paragraph["p1"].seconds == 30.0
    assert by_paragraph["p2"].seconds == 60.0
    assert all(c.artifact_url.startswith("https://minio.test/artifacts/") for c in out.clips)
    # Both clips landed in the fake storage layer — each wav paired with a
    # duration sidecar (the idempotent-resume presence marker).
    assert all(stage_name == "lesson_narrate" for _, stage_name, _, _ in fake_storage.stored)
    wavs = [f for _, _, f, _ in fake_storage.stored if f.endswith(".wav")]
    sidecars = [f for _, _, f, _ in fake_storage.stored if f.endswith(".json")]
    assert len(wavs) == 2
    assert len(sidecars) == 2


# instructor b


@pytest.mark.asyncio
async def test_narrate_dialogue_off_ignores_speaker(fake_storage: _FakeStorage) -> None:
    """Without the dialogue flag, a stray speaker must NOT trigger casting —
    single-narrator lessons keep voice_id (here None → TTS default)."""
    fake = AsyncMock()
    fake.synthesize.return_value = TTSSynthesis(
        audio_bytes=b"\x01", sample_rate=0, duration_seconds=3.0, provider="", voice=""
    )
    stage = LessonNarrateStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"voice": {"descriptor": "calm"}},
        prior_run_outputs={
            "lesson_script": {
                "paragraphs": [{"id": "p1", "text": "Teach.", "seconds": 4.0, "speaker": "a"}]
            }
        },
    )

    await stage.run({}, ctx=ctx)

    assert fake.synthesize.call_args.kwargs["voice"] is None


@pytest.mark.asyncio
async def test_narrate_raises_without_script(fake_storage: _FakeStorage) -> None:
    fake = AsyncMock()
    stage = LessonNarrateStage(client=fake)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    with pytest.raises(RuntimeError, match="lesson_script"):
        await stage.run({}, ctx=ctx)
