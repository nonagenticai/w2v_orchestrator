"""Tests for the motion_graphics_gen stage handler.

Mirrors test_ai_video_gen: opt-in skip contract + downstream-response gate +
the items-resolution and per-line keying that feed render's video_by_line.
The video_production client is mocked (no real browser/ffmpeg).
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.base import StageDownstreamResponseError, StageSkippedError
from src.services.stages.generic.motion_graphics_gen import MotionGraphicsGenStage


def _make_client(*, duration: float = 3.0) -> AsyncMock:
    client = AsyncMock()
    client.record_animation = AsyncMock(return_value="job-1")
    client.wait_for_content_recording = AsyncMock(
        return_value={"status": "completed", "duration_seconds": duration}
    )
    client.get_recording_video = AsyncMock(return_value=(b"\x00mp4", "video/mp4", None))
    return client


def _ctx(cfg: dict, prior: dict | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=cfg,
        prior_run_outputs=prior or {},
    )


@pytest.mark.asyncio
async def test_skips_without_config() -> None:
    stage = MotionGraphicsGenStage(client=_make_client())
    with pytest.raises(StageSkippedError):
        await stage.run({}, ctx=_ctx({}))


@pytest.mark.asyncio
async def test_skips_when_no_template_or_html() -> None:
    stage = MotionGraphicsGenStage(client=_make_client())
    with pytest.raises(StageSkippedError):
        await stage.run({}, ctx=_ctx({"motion_graphics": {"fps": 30}}))


@pytest.mark.asyncio
async def test_single_base_clip_unkeyed(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.services.stages.generic.motion_graphics_gen.store_generated_video",
        AsyncMock(return_value="https://minio.test/buckets/video/m1.mp4"),
    )
    client = _make_client()
    stage = MotionGraphicsGenStage(client=client)
    ctx = _ctx({"motion_graphics": {"template_id": "social_card", "params": {"title": "Hi"}}})

    out = await stage.run({}, ctx=ctx)

    assert len(out.clips) == 1
    clip = out.clips[0]
    assert clip.line_id is None  # unkeyed → round-robin pool in render
    assert clip.template_id == "social_card"
    assert clip.video_url is not None and clip.video_url.endswith("/m1.mp4")
    assert clip.duration_s == 3.0
    # base params forwarded to the renderer
    assert client.record_animation.await_args.kwargs["params"] == {"title": "Hi"}


@pytest.mark.asyncio
async def test_per_line_items_keyed(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.services.stages.generic.motion_graphics_gen.store_generated_video",
        AsyncMock(side_effect=lambda **kw: f"https://minio.test/v/{kw['video_id']}.mp4"),
    )
    stage = MotionGraphicsGenStage(client=_make_client())
    ctx = _ctx(
        {
            "motion_graphics": {
                "template_id": "pipeline_story",
                "items": [{"line_id": "shot-001"}, {"line_id": "shot-002"}],
            }
        }
    )

    out = await stage.run({}, ctx=ctx)

    assert [c.line_id for c in out.clips] == ["shot-001", "shot-002"]
    assert all(c.video_url for c in out.clips)


@pytest.mark.asyncio
async def test_items_from_stage(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.services.stages.generic.motion_graphics_gen.store_generated_video",
        AsyncMock(return_value="https://minio.test/v/x.mp4"),
    )
    stage = MotionGraphicsGenStage(client=_make_client())
    ctx = _ctx(
        {"motion_graphics": {"template_id": "social_card", "items_from_stage": "motion_brief"}},
        prior={"motion_brief": {"items": [{"line_id": "shot-001", "params": {"k": "v"}}]}},
    )

    out = await stage.run({}, ctx=ctx)

    assert len(out.clips) == 1
    assert out.clips[0].line_id == "shot-001"


@pytest.mark.asyncio
async def test_base_params_merged_with_item_params(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.services.stages.generic.motion_graphics_gen.store_generated_video",
        AsyncMock(return_value="https://minio.test/v/x.mp4"),
    )
    client = _make_client()
    stage = MotionGraphicsGenStage(client=client)
    ctx = _ctx(
        {
            "motion_graphics": {
                "template_id": "social_card",
                "params": {"brand_mark": "acme", "title": "base"},
                "items": [{"line_id": "shot-001", "params": {"title": "override"}}],
            }
        }
    )

    await stage.run({}, ctx=ctx)

    sent = client.record_animation.await_args.kwargs["params"]
    assert sent == {"brand_mark": "acme", "title": "override"}


@pytest.mark.asyncio
async def test_empty_video_url_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.services.stages.generic.motion_graphics_gen.store_generated_video",
        AsyncMock(return_value=""),
    )
    stage = MotionGraphicsGenStage(client=_make_client())
    ctx = _ctx({"motion_graphics": {"template_id": "social_card"}})
    with pytest.raises(StageDownstreamResponseError):
        await stage.run({}, ctx=ctx)


@pytest.mark.asyncio
async def test_reuses_already_recorded_clip_and_skips_render(monkeypatch) -> None:
    """Resume/retry idempotency: an already-recorded+stored motion clip is reused
    without re-driving video_production_service."""
    monkeypatch.setattr(
        "src.services.stages.generic.motion_graphics_gen.find_existing_media",
        AsyncMock(return_value="https://minio/existing-clip.mp4"),
    )
    client = _make_client()
    stage = MotionGraphicsGenStage(client=client)
    ctx = _ctx({"motion_graphics": {"template_id": "social_card"}})

    out = await stage.run({}, ctx=ctx)

    assert out.clips[0].video_url == "https://minio/existing-clip.mp4"
    assert out.clips[0].status == "reused"
    client.record_animation.assert_not_awaited()
    client.get_recording_video.assert_not_awaited()
