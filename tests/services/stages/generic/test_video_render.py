from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.generic.video_render import VideoRenderStage


@pytest.mark.asyncio
async def test_video_render_returns_artifact() -> None:
    fake = AsyncMock()
    fake.compile.return_value = {"artifact_url": "s3://x/y.mp4", "duration_seconds": 30.0}
    stage = VideoRenderStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={
            "storyboard_draft": {
                "shots": [
                    {"id": "s1", "line_id": "l1", "description": "opening", "seconds": 3.0},
                ]
            }
        },
    )
    out = await stage.run({}, ctx=ctx)
    assert out.artifact_url == "s3://x/y.mp4"
    assert out.duration_seconds == 30.0
    fake.compile.assert_awaited_once()


def _shots_arg(fake: AsyncMock) -> list[dict]:
    return fake.compile.call_args.kwargs["shots"]


@pytest.mark.asyncio
async def test_video_render_composites_real_media_over_ai_pool() -> None:
    fake = AsyncMock()
    fake.compile.return_value = {"artifact_url": "s3://x/y.mp4", "duration_seconds": 30.0}
    stage = VideoRenderStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={
            "storyboard_draft": {
                "shots": [
                    {"id": "s1", "line_id": "l1", "seconds": 3.0},
                    {"id": "s2", "line_id": "l2", "seconds": 3.0},
                    {"id": "s3", "line_id": "l3", "seconds": 3.0},
                ]
            },
            "ai_image_gen": {"images": [{"image_url": "https://minio/ai0.png"}]},
            # pinned (priority 1) + discovered (priority 3) are already re-published
            # to compiler-fetchable composite_urls by their producing stages.
            "source_ingest": {
                "pinned_media": [{"ref": "https://x/p", "composite_url": "https://minio/pin.png"}],
            },
            "media_discovery": {"items": [{"composite_url": "https://minio/disc.png"}]},
        },
    )
    out = await stage.run({}, ctx=ctx)
    assert out.artifact_url == "s3://x/y.mp4"
    shots = {s["line_id"]: s for s in _shots_arg(fake)}
    # Real media (pinned → discovered) keyed onto the first two lines, fit=cover.
    assert shots["l1"]["image_url"] == "https://minio/pin.png"
    assert shots["l1"]["fit"] == "cover"
    assert shots["l2"]["image_url"] == "https://minio/disc.png"
    assert shots["l2"]["fit"] == "cover"
    # Remaining line falls through to the round-robin AI pool (no cover fit).
    assert shots["l3"]["image_url"] == "https://minio/ai0.png"
    assert "fit" not in shots["l3"]


@pytest.mark.asyncio
async def test_video_render_composites_real_media_video_on_its_own_line() -> None:
    fake = AsyncMock()
    fake.compile.return_value = {"artifact_url": "s3://x/y.mp4", "duration_seconds": 30.0}
    stage = VideoRenderStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={
            "storyboard_draft": {
                "shots": [
                    {"id": "s1", "line_id": "l1", "seconds": 3.0},
                    {"id": "s2", "line_id": "l2", "seconds": 3.0},
                    {"id": "s3", "line_id": "l3", "seconds": 3.0},
                ]
            },
            "ai_image_gen": {"images": [{"image_url": "https://minio/ai0.png"}]},
            # A pinned image AND a pinned video (re-published by source_ingest).
            "source_ingest": {
                "pinned_media": [
                    {
                        "ref": "https://x/i",
                        "composite_url": "https://minio/pin.png",
                        "kind": "image",
                    },
                    {
                        "ref": "https://x/v",
                        "composite_url": "https://minio/clip.mp4",
                        "kind": "video",
                    },
                ],
            },
        },
    )
    await stage.run({}, ctx=ctx)
    shots = {s["line_id"]: s for s in _shots_arg(fake)}
    # Image takes the leading line; the video takes its OWN line (not l1) as a
    # video_url (which the compositor honours over any image), fit=cover.
    assert shots["l1"]["image_url"] == "https://minio/pin.png"
    assert shots["l2"]["video_url"] == "https://minio/clip.mp4"
    assert shots["l2"]["fit"] == "cover"
    # Remaining line falls through to the round-robin AI pool.
    assert shots["l3"]["image_url"] == "https://minio/ai0.png"
    assert "video_url" not in shots["l3"]


@pytest.mark.asyncio
async def test_video_render_composites_ai_video_on_leftover_line_curated_wins() -> None:
    fake = AsyncMock()
    fake.compile.return_value = {"artifact_url": "s3://x/y.mp4", "duration_seconds": 30.0}
    stage = VideoRenderStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={
            "storyboard_draft": {
                "shots": [
                    {"id": "s1", "line_id": "l1", "seconds": 3.0},
                    {"id": "s2", "line_id": "l2", "seconds": 3.0},
                    {"id": "s3", "line_id": "l3", "seconds": 3.0},
                ]
            },
            "ai_image_gen": {"images": [{"image_url": "https://minio/ai0.png"}]},
            # One curated image (takes l1) — curated outranks generative video.
            "source_ingest": {
                "pinned_media": [
                    {
                        "ref": "https://x/i",
                        "composite_url": "https://minio/pin.png",
                        "kind": "image",
                    }
                ],
            },
            # An AI-generated clip (already re-published, fetchable).
            "ai_video_gen": {"videos": [{"video_url": "https://minio/ai_clip.mp4"}]},
        },
    )
    await stage.run({}, ctx=ctx)
    shots = {s["line_id"]: s for s in _shots_arg(fake)}
    # Curated image keeps l1 — AI video must NOT displace it.
    assert shots["l1"]["image_url"] == "https://minio/pin.png"
    assert "video_url" not in shots["l1"]
    # AI clip is spread across the leftover lines (skip l1): one clip over the two
    # eligible lines → bucket centre l3, as a video_url, fit=cover.
    assert shots["l3"]["video_url"] == "https://minio/ai_clip.mp4"
    assert shots["l3"]["fit"] == "cover"
    # l2 falls through to the round-robin AI image pool.
    assert shots["l2"]["image_url"] == "https://minio/ai0.png"
    assert "video_url" not in shots["l2"]


@pytest.mark.asyncio
async def test_video_render_without_sources_is_ai_pool_only() -> None:
    fake = AsyncMock()
    fake.compile.return_value = {"artifact_url": "s3://x/y.mp4", "duration_seconds": 30.0}
    stage = VideoRenderStage(client=fake)
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={
            "storyboard_draft": {
                "shots": [
                    {"id": "s1", "line_id": "l1", "seconds": 3.0},
                    {"id": "s2", "line_id": "l2", "seconds": 3.0},
                ]
            },
            "ai_image_gen": {"images": [{"image_url": "https://minio/ai0.png"}]},
        },
    )
    await stage.run({}, ctx=ctx)
    # No source stages → no keyed real media, no cover fit: byte-for-byte the
    # prior round-robin behaviour.
    for shot in _shots_arg(fake):
        assert shot["image_url"] == "https://minio/ai0.png"
        assert "fit" not in shot


@pytest.mark.asyncio
async def test_video_render_raises_without_storyboard() -> None:
    stage = VideoRenderStage(client=AsyncMock())
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
    )
    with pytest.raises(RuntimeError, match="storyboard_draft"):
        await stage.run({}, ctx=ctx)


def _ctx_with_script(captions_mode: str) -> RunContext:
    # Two-line generic run with the script_draft text the burned-in caption path
    # joins on (line id == each storyboard shot's line_id).
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"deliverable": {"format": "16x9", "captions_mode": captions_mode}},
        prior_run_outputs={
            "storyboard_draft": {
                "shots": [
                    {"id": "s1", "line_id": "l1", "seconds": 3.0},
                    {"id": "s2", "line_id": "l2", "seconds": 3.0},
                ]
            },
            "script_draft": {
                "lines": [{"id": "l1", "text": "Line one."}, {"id": "l2", "text": "Line two."}]
            },
            "ai_image_gen": {"images": [{"image_url": "https://minio/ai0.png"}]},
        },
    )


@pytest.mark.asyncio
async def test_video_render_no_captions_when_mode_none() -> None:
    fake = AsyncMock()
    fake.compile.return_value = {"artifact_url": "s3://x/y.mp4", "duration_seconds": 6.0}

    await VideoRenderStage(client=fake).run({}, ctx=_ctx_with_script("none"))

    for shot in _shots_arg(fake):
        assert "caption" not in shot


def test_video_render_compile_tags_omit_crf_when_mastering_crf_authored() -> None:
    """The gate applies on the generic render path: an authored `mastering.crf`
    suppresses the tier crf tag so the compiler honours the mastering-block crf."""
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "resolution": "4k",
            "deliverable": {"format": "16x9"},
            "mastering": {"crf": 14},
        },
    )
    tags = VideoRenderStage(client=AsyncMock()).compile_tags(ctx)
    assert "crf" not in tags
    assert tags["render_width"] == "3840"
    assert tags["render_height"] == "2160"
