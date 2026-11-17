from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.lesson.record import LessonRecordStage
from src.services.stages.shared.motion_slides import DeckMotionResult


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


def _slides_ctx(
    *,
    brand: dict | None = None,
    images: list[dict] | None = None,
    motion: dict | None = None,
) -> RunContext:
    prior: dict = {
        "lesson_slides": {
            "slides": [
                {"id": "s1", "paragraph_ids": ["p1"], "title": "T1", "bullets": ["a", "b"]},
                {"id": "s2", "paragraph_ids": ["p2"], "title": "T2", "bullets": ["c"]},
            ]
        },
    }
    if images is not None:
        prior["ai_image_gen"] = {"images": images}
    config: dict = {
        "deliverable": {"title": "Widgets 101", "format": "16x9"},
        "brand": brand or {},
        # Pin base tier so dims assert 1920x1080; 2K/4K scaling is covered in
        # tests/services/stages/shared/test_resolution_tier.py.
        "resolution": "1080p",
    }
    if motion is not None:
        config["motion_graphics"] = motion
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=config,
        prior_run_outputs=prior,
    )


def _video_fake() -> AsyncMock:
    v = AsyncMock()
    v.record_animation.return_value = "anim-1"
    v.wait_for_content_recording.return_value = {"duration_seconds": 4.0}
    v.get_recording_video.return_value = (b"\x00mp4", "video/mp4", None)
    return v


def _slide_client() -> AsyncMock:
    fake = AsyncMock()
    fake.resolve_template_id.return_value = "tpl-feature"
    fake.start_deck.return_value = "deck-1"
    fake.wait_for_deck.return_value = {"status": "completed", "slide_count": 2}
    fake.get_slide_png.side_effect = [b"\x89PNG-0", b"\x89PNG-1"]
    return fake


@pytest.mark.asyncio
async def test_record_renders_one_deck_and_image_per_slide(fake_storage: _FakeStorage) -> None:
    fake = AsyncMock()
    fake.resolve_template_id.return_value = "tpl-feature"
    fake.start_deck.return_value = "deck-1"
    fake.wait_for_deck.return_value = {"status": "completed", "slide_count": 2}
    fake.get_slide_png.side_effect = [b"\x89PNG-0", b"\x89PNG-1"]
    stage = LessonRecordStage(client=fake)

    out = await stage.run({}, ctx=_slides_ctx())

    # One deck for the whole lesson, two PNG fetches (one per slide).
    fake.start_deck.assert_awaited_once()
    payload = fake.start_deck.await_args.args[0]
    assert payload["title"] == "Widgets 101"
    assert len(payload["slides"]) == 2
    assert payload["slides"][0]["template_id"] == "tpl-feature"
    assert payload["slides"][0]["content"]["title"] == "T1"
    assert payload["slides"][0]["content"]["bullets"] == ["a", "b"]
    assert fake.get_slide_png.await_count == 2

    assert len(out.recordings) == 2
    assert out.recordings[0].slide_id == "s1"
    assert out.recordings[1].slide_id == "s2"
    assert all(r.image_url.startswith("https://minio.test/artifacts/") for r in out.recordings)
    assert len(fake_storage.stored) == 2
    assert all(filename.endswith(".png") for _, _, filename, _ in fake_storage.stored)


@pytest.mark.asyncio
async def test_record_forwards_brand_and_per_slide_image(fake_storage: _FakeStorage) -> None:
    fake = AsyncMock()
    fake.resolve_template_id.return_value = "tpl-feature"
    fake.start_deck.return_value = "deck-1"
    fake.wait_for_deck.return_value = {"status": "completed", "slide_count": 2}
    fake.get_slide_png.side_effect = [b"\x89PNG-0", b"\x89PNG-1"]
    stage = LessonRecordStage(client=fake)

    ctx = _slides_ctx(
        brand={"primary_color": "#0B1F3A", "font_family": "Inter"},
        # ai_image_gen keys each image to its slide via line_id; only s1 here.
        images=[{"line_id": "s1", "image_url": "https://img.test/hero-s1.png"}],
    )
    await stage.run({}, ctx=ctx)

    payload = fake.start_deck.await_args.args[0]
    assert payload["brand"] == {"primary_color": "#0B1F3A", "font_family": "Inter"}
    # slide with an image gets image_url; the other has none (renders text-only)
    assert payload["slides"][0]["content"]["image_url"] == "https://img.test/hero-s1.png"
    assert "image_url" not in payload["slides"][1]["content"]


@pytest.mark.asyncio
async def test_record_resigns_hero_image_urls_before_deck() -> None:
    """ai_image_gen's presigned hero-image URLs are re-signed to a fresh TTL
    before slide_service fetches them — the 1h expiry-cliff fix.

    Manages its own storage (not the ``fake_storage`` fixture) because the
    re-sign behaviour hinges on the storage exposing ``resign_presigned_url``;
    the bare ``_FakeStorage`` deliberately omits it (a backend that can't
    re-sign must no-op, keeping the original URL — proven by the brand test).
    """

    class _ResigningStorage(_FakeStorage):
        async def resign_presigned_url(self, url: str) -> str:
            return f"{url}{'&' if '?' in url else '?'}X-Amz-Resigned=1"

    tts_storage.set_storage_for_tests(_ResigningStorage())
    try:
        fake = AsyncMock()
        fake.resolve_template_id.return_value = "tpl-feature"
        fake.start_deck.return_value = "deck-1"
        fake.wait_for_deck.return_value = {"status": "completed", "slide_count": 2}
        fake.get_slide_png.side_effect = [b"\x89PNG-0", b"\x89PNG-1"]
        stage = LessonRecordStage(client=fake)

        ctx = _slides_ctx(
            images=[
                {
                    "line_id": "s1",
                    "image_url": "https://minio.test/hero-s1.png?X-Amz-Expires=3600",
                }
            ],
        )
        await stage.run({}, ctx=ctx)

        payload = fake.start_deck.await_args.args[0]
        # The embedded hero image carries the FRESH signature, not the stale one.
        assert payload["slides"][0]["content"]["image_url"] == (
            "https://minio.test/hero-s1.png?X-Amz-Expires=3600&X-Amz-Resigned=1"
        )
        # A slide with no image is untouched (no image_url key, no re-sign).
        assert "image_url" not in payload["slides"][1]["content"]
    finally:
        tts_storage.set_storage_for_tests(None)


@pytest.mark.asyncio
async def test_record_tolerates_missing_image_gen(fake_storage: _FakeStorage) -> None:
    """ai_image_gen skipped/absent → slides still render (no image_url keys)."""
    fake = AsyncMock()
    fake.resolve_template_id.return_value = "tpl-feature"
    fake.start_deck.return_value = "deck-1"
    fake.wait_for_deck.return_value = {"status": "completed", "slide_count": 2}
    fake.get_slide_png.side_effect = [b"\x89PNG-0", b"\x89PNG-1"]
    stage = LessonRecordStage(client=fake)

    await stage.run({}, ctx=_slides_ctx())  # no ai_image_gen output

    payload = fake.start_deck.await_args.args[0]
    assert all("image_url" not in s["content"] for s in payload["slides"])


@pytest.mark.asyncio
async def test_record_no_motion_when_not_opted_in(fake_storage: _FakeStorage) -> None:
    """No motion_graphics block → still PNGs only; the video client is untouched."""
    video = _video_fake()
    stage = LessonRecordStage(client=_slide_client(), video_client=video)

    out = await stage.run({}, ctx=_slides_ctx())  # no motion config

    assert all(r.motion_video_url is None for r in out.recordings)
    video.record_animation.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_attaches_motion_clip_per_slide(fake_storage: _FakeStorage) -> None:
    video = _video_fake()
    stage = LessonRecordStage(client=_slide_client(), video_client=video)

    out = await stage.run({}, ctx=_slides_ctx(motion={"enabled": True}))

    # A clip rendered per slide via animated_slide (title + bullets), 16:9.
    assert video.record_animation.await_count == 2
    kwargs = video.record_animation.await_args_list[0].kwargs
    assert kwargs["template_id"] == "animated_slide"
    assert kwargs["width"] == 1920 and kwargs["height"] == 1080
    assert all(
        r.motion_video_url and r.motion_video_url.startswith("https://") for r in out.recordings
    )


@pytest.mark.asyncio
async def test_record_motion_disabled_via_slides_false(fake_storage: _FakeStorage) -> None:
    """motion_graphics.slides: false keeps slides still even with the block on."""
    video = _video_fake()
    stage = LessonRecordStage(client=_slide_client(), video_client=video)

    out = await stage.run({}, ctx=_slides_ctx(motion={"enabled": True, "slides": False}))

    assert all(r.motion_video_url is None for r in out.recordings)
    video.record_animation.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_motion_per_clip_soft_fail(fake_storage: _FakeStorage) -> None:
    """One slide's clip fails → that recording keeps the PNG (motion None); the
    other slide still gets its clip."""
    video = _video_fake()
    # First record_animation raises, second succeeds.
    video.record_animation.side_effect = [RuntimeError("vps timeout"), "anim-2"]
    stage = LessonRecordStage(client=_slide_client(), video_client=video)

    out = await stage.run({}, ctx=_slides_ctx(motion={"enabled": True}))

    assert out.recordings[0].motion_video_url is None  # soft-failed → still PNG
    assert out.recordings[1].motion_video_url  # rendered
    # Both recordings still carry their still image.
    assert all(r.image_url.startswith("https://") for r in out.recordings)
    # The drop is no longer silent: the run degrades via a typed note naming the
    # slide and the reason (soft-fail, not a phase-budget cancel).
    assert any("soft-failed" in n and "s1" in n for n in out.notes)
    assert not any("phase-budget" in n for n in out.notes)


@pytest.mark.asyncio
async def test_record_motion_budget_cancel_surfaces_note(fake_storage: _FakeStorage) -> None:
    """A slide dropped by the phase-budget cancel surfaces a diagnosable note
    (not a silent motion_video_url=None). Drives the budget-cancel branch by
    mocking render_deck_motion_clips to report slide 1 as budget-cancelled."""
    stage = LessonRecordStage(client=_slide_client(), video_client=_video_fake())

    mock_render = AsyncMock(
        return_value=DeckMotionResult(
            urls={0: "https://minio.test/clip-0.mp4"},
            budget_exceeded=True,
            failed_indices=(1,),
        )
    )
    with patch("src.services.stages.lesson.record.render_deck_motion_clips", mock_render):
        out = await stage.run({}, ctx=_slides_ctx(motion={"enabled": True}))

    # The real fix: lesson_record must pass a phase budget >= the per-clip record
    # timeout (600s) so an in-flight clip is never guillotined. Lock the value so
    # a regression to the 450s default (the original bug) fails unit CI, not only
    # the nightly e2e.
    assert mock_render.call_args.kwargs["phase_budget_s"] == 780
    assert out.recordings[0].motion_video_url == "https://minio.test/clip-0.mp4"
    assert out.recordings[1].motion_video_url is None  # budget-cancelled → still PNG
    # Exactly one note, naming slide s2 and the phase-budget reason.
    assert any("phase-budget" in n and "s2" in n for n in out.notes)
    assert not any("soft-failed" in n for n in out.notes)


@pytest.mark.asyncio
async def test_record_raises_without_slides(fake_storage: _FakeStorage) -> None:
    fake = AsyncMock()
    stage = LessonRecordStage(client=fake)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    with pytest.raises(RuntimeError, match="lesson_slides"):
        await stage.run({}, ctx=ctx)
