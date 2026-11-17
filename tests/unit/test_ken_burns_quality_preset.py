"""Ken Burns clips are recorded at the run's quality preset, not VP's CRF-23 default."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.clients.video_production_client import VideoProductionClient
from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.shared import ken_burns
from src.services.stages.shared.ken_burns import apply_ken_burns


class _FakeStorage:
    async def initialize(self) -> None:
        pass

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        return f"artifacts/{job_id}/{stage_name}/{filename}"

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        return f"https://minio.test/{path}"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    tts_storage.set_storage_for_tests(_FakeStorage())

    async def _fake_fetch(url: str, client: Any) -> tuple[bytes, str]:
        return (b"\x89PNG\r\n", "image/png")

    monkeypatch.setattr(ken_burns, "_fetch_own_asset", _fake_fetch)
    yield
    tts_storage.set_storage_for_tests(None)


def _video_fake() -> AsyncMock:
    v = AsyncMock()
    v.record_animation.return_value = "kb-1"
    v.wait_for_content_recording.return_value = {"duration_seconds": 3.0}
    v.get_recording_video.return_value = (b"\x00mp4", "video/mp4", None)
    return v


def _ctx(snapshot: dict[str, Any]) -> RunContext:
    return RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot=snapshot)


def _still() -> dict[str, Any]:
    return {"id": "s-a", "line_id": "a", "image_url": "https://minio.test/a.png", "seconds": 3.0}


@pytest.mark.asyncio
@pytest.mark.parametrize(("resolution", "expected"), [("4k", "4k")])
async def test_record_animation_receives_run_tier_preset(resolution: str, expected: str) -> None:
    video = _video_fake()
    await apply_ken_burns(_ctx({"resolution": resolution}), [_still()], client=video)
    assert video.record_animation.await_args.kwargs["quality_preset"] == expected


@pytest.mark.asyncio
async def test_media_ids_differ_only_by_preset(monkeypatch) -> None:
    """Same image + motion params, different preset ⇒ different content address,
    so a resume never reuses a clip recorded at another (e.g. the old CRF-23) preset."""
    seen: list[tuple[Any, ...]] = []
    real = ken_burns.deterministic_media_id

    def _spy(*args: Any) -> str:
        seen.append(args)
        return real(*args)

    monkeypatch.setattr(ken_burns, "deterministic_media_id", _spy)
    presets = iter(["high_quality", "4k"])
    monkeypatch.setattr(
        ken_burns, "_resolve_recording_quality_preset", lambda ctx, cfg: next(presets)
    )
    for _ in range(2):
        await apply_ken_burns(_ctx({}), [_still()], client=_video_fake())

    assert len(seen) == 2
    diff = [i for i, (a, b) in enumerate(zip(seen[0], seen[1], strict=True)) if a != b]
    assert [(seen[0][i], seen[1][i]) for i in diff] == [("high_quality", "4k")]
    assert real(*seen[0]) != real(*seen[1])


@pytest.mark.asyncio
async def test_client_sends_quality_preset_only_when_set() -> None:
    client = VideoProductionClient()
    request = AsyncMock(return_value={"job_id": "j"})
    client._request = request  # type: ignore[method-assign]

    await client.record_animation(template_id="ken_burns")
    assert request.await_args is not None
    assert "quality_preset" not in request.await_args.kwargs["json"]

    await client.record_animation(template_id="ken_burns", quality_preset="4k")
    assert request.await_args is not None
    assert request.await_args.kwargs["json"]["quality_preset"] == "4k"
