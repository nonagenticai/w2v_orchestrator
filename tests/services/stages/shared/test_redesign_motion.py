"""Tests for the opt-in redesign motion-hero capture (Phase 4)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.services import tts_storage
from src.services.stages.shared.redesign_motion import (
    attach_motion_hero,
)
from src.services.stages.website_review.redesign import RedesignVisual


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


@pytest.fixture
def fake_storage():
    tts_storage.set_storage_for_tests(_FakeStorage())
    yield
    tts_storage.set_storage_for_tests(None)


def _visuals() -> list[RedesignVisual]:
    return [
        RedesignVisual(
            id="v0", label="original desktop", role="before", image_url="https://s/b.png"
        ),
        RedesignVisual(
            id="v1", label="redesign desktop", role="after", image_url="https://s/a.png"
        ),
    ]


def _video_fake() -> AsyncMock:
    v = AsyncMock()
    v.record_animation.return_value = "anim-1"
    v.wait_for_content_recording.return_value = {"duration_seconds": 8.0}
    v.get_recording_video.return_value = (b"\x00mp4", "video/mp4", None)
    return v


@pytest.mark.asyncio
async def test_disabled_is_noop(fake_storage) -> None:
    video = _video_fake()
    design = AsyncMock()
    visuals = _visuals()
    out = await attach_motion_hero(
        visuals,
        design_client=design,
        video_client=video,
        job_id="rd-1",
        brief={},  # no redesign_motion → off
        deliverable={"format": "16x9"},
        run_id="run-1",
        stage_name="website_redesign",
    )
    assert all(v.video_url is None for v in out)
    video.record_animation.assert_not_awaited()
    design.get_design_html.assert_not_awaited()


@pytest.mark.asyncio
async def test_enabled_attaches_clip_to_hero(fake_storage) -> None:
    video = _video_fake()
    design = AsyncMock()
    design.get_design_html.return_value = "<html><body><h1>Redesign</h1></body></html>"

    out = await attach_motion_hero(
        _visuals(),
        design_client=design,
        video_client=video,
        job_id="rd-1",
        brief={"redesign_motion": True},
        deliverable={"format": "16x9"},
        run_id="run-1",
        stage_name="website_redesign",
    )

    # The 'after' hero (index 1) gets the clip; the 'before' visual is untouched.
    assert out[0].video_url is None
    assert out[1].video_url and out[1].video_url.startswith("https://")
    assert out[1].video_duration_s == 8.0
    # Recorded from the scroll-injected HTML at 16:9.
    kwargs = video.record_animation.await_args.kwargs
    assert "window.render" in kwargs["html"]
    assert kwargs["width"] == 1920
    assert kwargs["height"] == 1080


@pytest.mark.asyncio
async def test_capture_failure_soft_falls_to_static(fake_storage) -> None:
    video = _video_fake()
    design = AsyncMock()
    design.get_design_html.side_effect = RuntimeError("html_render down")

    out = await attach_motion_hero(
        _visuals(),
        design_client=design,
        video_client=video,
        job_id="rd-1",
        brief={"redesign_motion": True},
        deliverable={"format": "16x9"},
        run_id="run-1",
        stage_name="website_redesign",
    )
    # Soft-fail: visuals unchanged, no clip attached.
    assert all(v.video_url is None for v in out)
    video.record_animation.assert_not_awaited()
