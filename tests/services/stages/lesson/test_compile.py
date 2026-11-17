from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.lesson.compile import (
    LessonCompileStage,
    build_lesson_sub_chapters,
)


def _ctx_with_outputs() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"deliverable": {"format": "16x9", "captions_mode": "burned_in"}},
        prior_run_outputs={
            "lesson_slides": {
                "slides": [
                    {"id": "s1", "paragraph_ids": ["p1", "p2"], "title": "T1", "bullets": ["a"]},
                ]
            },
            "lesson_narrate": {
                "clips": [
                    {
                        "id": "c1",
                        "paragraph_id": "p1",
                        "artifact_url": "http://a/c1.wav",
                        "seconds": 30.0,
                    },
                    {
                        "id": "c2",
                        "paragraph_id": "p2",
                        "artifact_url": "http://a/c2.wav",
                        "seconds": 20.0,
                    },
                ]
            },
            "lesson_record": {
                "recordings": [
                    {"id": "r1", "slide_id": "s1", "image_url": "http://img/s1.png"},
                ]
            },
        },
    )


def _ctx_with_paragraphs(n: int) -> RunContext:
    # Wider fixture for the real-media cutaway tests: n paragraphs in one slide so
    # the distribute (bucket-centre) spread across the timeline is observable
    # (the 2-paragraph _ctx_with_outputs is too small to express striding).
    pids = [f"p{i}" for i in range(1, n + 1)]
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"deliverable": {"format": "16x9", "captions_mode": "burned_in"}},
        prior_run_outputs={
            "lesson_slides": {
                "slides": [{"id": "s1", "paragraph_ids": pids, "title": "T1", "bullets": ["a"]}]
            },
            "lesson_narrate": {
                "clips": [
                    {
                        "id": f"c{i}",
                        "paragraph_id": pid,
                        "artifact_url": f"http://a/{pid}.wav",
                        "seconds": 10.0,
                    }
                    for i, pid in enumerate(pids, 1)
                ]
            },
            "lesson_record": {
                "recordings": [{"id": "r1", "slide_id": "s1", "image_url": "http://img/s1.png"}]
            },
        },
    )


@pytest.mark.asyncio
async def test_compile_emits_one_shot_per_paragraph_with_image_and_audio() -> None:
    fake = AsyncMock()
    fake.compile.return_value = {
        "artifact_url": "http://stub/lesson.mp4",
        "duration_seconds": 50.0,
    }
    stage = LessonCompileStage(client=fake)

    out = await stage.run({}, ctx=_ctx_with_outputs())

    assert out.artifact_url == "http://stub/lesson.mp4"
    assert out.duration_seconds == 50.0
    _, kwargs = fake.compile.call_args
    shots = kwargs["shots"]
    assert len(shots) == 2
    # Each paragraph's shot carries its slide image + its own narration audio.
    assert shots[0]["image_url"] == "http://img/s1.png"
    assert shots[0]["audio_url"] == "http://a/c1.wav"
    assert shots[0]["seconds"] == 30.0
    assert shots[1]["image_url"] == "http://img/s1.png"
    assert shots[1]["audio_url"] == "http://a/c2.wav"
    # Built from narration, not a storyboard: no `description: null` on the wire.
    assert "description" not in shots[0]
    assert "description" not in shots[1]
    assert kwargs["tags"]["captions_mode"] == "burned_in"
    assert kwargs["tags"]["format"] == "16x9"


@pytest.mark.asyncio
async def test_compile_paragraph_without_slide_image_has_no_image_url() -> None:
    fake = AsyncMock()
    fake.compile.return_value = {"artifact_url": "http://stub/l.mp4", "duration_seconds": 1.0}
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"deliverable": {"format": "16x9", "captions_mode": "none"}},
        prior_run_outputs={
            # p9 belongs to no slide → no keyed image.
            "lesson_slides": {"slides": [{"id": "s1", "paragraph_ids": ["p1"]}]},
            "lesson_narrate": {
                "clips": [
                    {
                        "id": "c9",
                        "paragraph_id": "p9",
                        "artifact_url": "http://a/c9.wav",
                        "seconds": 4.0,
                    }
                ]
            },
            "lesson_record": {"recordings": []},
        },
    )
    out = await LessonCompileStage(client=fake).run({}, ctx=ctx)
    assert out.artifact_url == "http://stub/l.mp4"
    _, kwargs = fake.compile.call_args
    shot = kwargs["shots"][0]
    assert "image_url" not in shot
    assert shot["audio_url"] == "http://a/c9.wav"
    assert shot["seconds"] == 4.0


def _slides(*specs: tuple[str, list[str]]) -> list[dict]:
    return [{"id": f"s{i}", "title": t, "paragraph_ids": pids} for i, (t, pids) in enumerate(specs)]


def _clips(*pairs: tuple[str, float]) -> list[dict]:
    return [{"paragraph_id": pid, "seconds": secs} for pid, secs in pairs]


def test_sub_chapters_scaled_contiguous_first_zero_last_at_total() -> None:
    slides = _slides(("A", ["p1", "p2"]), ("B", ["p3"]))
    clips = _clips(("p1", 30.0), ("p2", 10.0), ("p3", 20.0))  # intended 40 / 20, sum 60
    subs = build_lesson_sub_chapters(slides, clips, total_seconds=120.0)  # scale x2

    assert [s.title for s in subs] == ["A", "B"]
    assert [s.start_seconds for s in subs] == [0.0, 80.0]
    assert [s.duration_seconds for s in subs] == [80.0, 40.0]
    # contiguous and ending exactly at the real total.
    assert subs[0].start_seconds == 0.0
    assert subs[-1].start_seconds + subs[-1].duration_seconds == 120.0
    assert [s.sequence for s in subs] == [1, 2]


def test_sub_chapters_single_slide_is_empty() -> None:
    subs = build_lesson_sub_chapters(_slides(("Only", ["p1"])), _clips(("p1", 30.0)), 30.0)
    assert subs == []


def test_sub_chapters_drop_zero_duration_slide() -> None:
    slides = _slides(("A", ["p1"]), ("Silent", ["p9"]), ("B", ["p2"]))  # p9 has no clip
    clips = _clips(("p1", 20.0), ("p2", 20.0))
    subs = build_lesson_sub_chapters(slides, clips, total_seconds=40.0)
    assert [s.title for s in subs] == ["A", "B"]
    assert [s.start_seconds for s in subs] == [0.0, 20.0]


def test_sub_chapters_blank_title_falls_back_to_section() -> None:
    slides = _slides(("", ["p1"]), ("", ["p2"]))
    subs = build_lesson_sub_chapters(slides, _clips(("p1", 10.0), ("p2", 10.0)), 20.0)
    assert [s.title for s in subs] == ["Section 1", "Section 2"]


@pytest.mark.asyncio
async def test_compile_populates_sub_chapters_for_multi_slide_lesson() -> None:
    fake = AsyncMock()
    fake.compile.return_value = {"artifact_url": "http://stub/l.mp4", "duration_seconds": 50.0}
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"deliverable": {"format": "16x9", "captions_mode": "none"}},
        prior_run_outputs={
            "lesson_slides": {
                "slides": [
                    {"id": "s1", "paragraph_ids": ["p1"], "title": "First"},
                    {"id": "s2", "paragraph_ids": ["p2"], "title": "Second"},
                ]
            },
            "lesson_narrate": {
                "clips": [
                    {
                        "id": "c1",
                        "paragraph_id": "p1",
                        "artifact_url": "http://a/c1.wav",
                        "seconds": 30.0,
                    },
                    {
                        "id": "c2",
                        "paragraph_id": "p2",
                        "artifact_url": "http://a/c2.wav",
                        "seconds": 20.0,
                    },
                ]
            },
            "lesson_record": {
                "recordings": [
                    {"id": "r1", "slide_id": "s1", "image_url": "http://img/s1.png"},
                    {"id": "r2", "slide_id": "s2", "image_url": "http://img/s2.png"},
                ]
            },
        },
    )
    out = await LessonCompileStage(client=fake).run({}, ctx=ctx)
    assert [s.title for s in out.sub_chapters] == ["First", "Second"]
    assert out.sub_chapters[0].start_seconds == 0.0


@pytest.mark.asyncio
async def test_compile_composites_real_media_cutaway_over_slide() -> None:
    # A grounded lesson (operator-pinned / discovered media, re-published to
    # compiler-fetchable composite_urls) lays real media over the paragraph lines
    # as full-frame cutaways — winning over that line's slide PNG — SPREAD evenly
    # across the timeline (not front-loaded), while the rest keep their slides.
    # Narration audio is preserved.
    fake = AsyncMock()
    fake.compile.return_value = {"artifact_url": "http://stub/l.mp4", "duration_seconds": 50.0}
    ctx = _ctx_with_paragraphs(6)  # 6 paragraphs, 2 cutaways → bucket centres p3, p5
    ctx.prior_run_outputs["source_ingest"] = {
        "pinned_media": [{"ref": "https://x/p", "composite_url": "https://minio/pin.png"}],
    }
    ctx.prior_run_outputs["media_discovery"] = {
        "items": [{"composite_url": "https://minio/disc.png"}]
    }

    await LessonCompileStage(client=fake).run({}, ctx=ctx)

    _, kwargs = fake.compile.call_args
    shots = {s["line_id"]: s for s in kwargs["shots"]}
    # Cutaways spread across the timeline (pinned then discovered), cover-fit;
    # audio kept; the opening line is NOT a cutaway (proves distribution).
    assert shots["p3"]["image_url"] == "https://minio/pin.png"
    assert shots["p3"]["fit"] == "cover"
    assert shots["p3"]["audio_url"] == "http://a/p3.wav"
    assert shots["p5"]["image_url"] == "https://minio/disc.png"
    assert shots["p5"]["fit"] == "cover"
    assert shots["p1"]["image_url"] == "http://img/s1.png"  # slide PNG, not a cutaway
    assert "fit" not in shots["p1"]


@pytest.mark.asyncio
async def test_compile_composites_real_media_video_on_its_own_line() -> None:
    # A pinned VIDEO becomes a full-frame clip line (video_url wins over any
    # image); a pinned IMAGE takes a different line, so an image and a clip never
    # collide. Both are spread across the timeline.
    fake = AsyncMock()
    fake.compile.return_value = {"artifact_url": "http://stub/l.mp4", "duration_seconds": 50.0}
    ctx = _ctx_with_paragraphs(6)
    ctx.prior_run_outputs["source_ingest"] = {
        "pinned_media": [
            {"ref": "https://x/p", "composite_url": "https://minio/pin.png", "kind": "image"},
            {"ref": "https://x/v", "composite_url": "https://minio/clip.mp4", "kind": "video"},
        ],
    }

    await LessonCompileStage(client=fake).run({}, ctx=ctx)

    _, kwargs = fake.compile.call_args
    shots = {s["line_id"]: s for s in kwargs["shots"]}
    # Image strides to p4 (one item over 6 lines → centre); the clip strides over
    # the remaining 5 eligible lines → p3. They never share a line.
    assert shots["p4"]["image_url"] == "https://minio/pin.png"
    assert "video_url" not in shots["p4"]
    assert shots["p3"]["video_url"] == "https://minio/clip.mp4"
    assert shots["p3"]["fit"] == "cover"


@pytest.mark.asyncio
async def test_compile_composites_ai_video_on_leftover_line_curated_wins() -> None:
    # A curated image takes its strided line; an AI-generated clip (lowest tier)
    # fills a line that curated media left bare — over the slide PNG.
    fake = AsyncMock()
    fake.compile.return_value = {"artifact_url": "http://stub/l.mp4", "duration_seconds": 50.0}
    ctx = _ctx_with_paragraphs(6)
    ctx.prior_run_outputs["source_ingest"] = {
        "pinned_media": [{"ref": "https://x/p", "composite_url": "https://minio/pin.png"}],
    }
    ctx.prior_run_outputs["ai_video_gen"] = {"videos": [{"video_url": "https://minio/ai_clip.mp4"}]}

    await LessonCompileStage(client=fake).run({}, ctx=ctx)

    _, kwargs = fake.compile.call_args
    shots = {s["line_id"]: s for s in kwargs["shots"]}
    # Curated image keeps its line (p4) — AI video must not displace it.
    assert shots["p4"]["image_url"] == "https://minio/pin.png"
    assert "video_url" not in shots["p4"]
    # AI clip strides over the remaining lines (skip p4) → p3, full-frame.
    assert shots["p3"]["video_url"] == "https://minio/ai_clip.mp4"
    assert shots["p3"]["fit"] == "cover"


@pytest.mark.asyncio
async def test_compile_without_sources_is_unchanged() -> None:
    # No source_ingest / media_discovery → no overlay: every line keeps its slide
    # PNG, no cover fit, no video_url (byte-for-byte unchanged).
    fake = AsyncMock()
    fake.compile.return_value = {"artifact_url": "http://stub/l.mp4", "duration_seconds": 50.0}

    await LessonCompileStage(client=fake).run({}, ctx=_ctx_with_outputs())

    _, kwargs = fake.compile.call_args
    for shot in kwargs["shots"]:
        assert shot["image_url"] == "http://img/s1.png"
        assert "fit" not in shot
        assert "video_url" not in shot


# still PNG kept as fallback


@pytest.mark.asyncio
async def test_compile_curated_and_ai_media_beat_animated_slide() -> None:
    # Precedence: curated image cutaway and AI video both win over the animated
    # slide; the animated slide only fills the lines neither claimed.
    fake = AsyncMock()
    fake.compile.return_value = {"artifact_url": "http://stub/l.mp4", "duration_seconds": 60.0}
    ctx = _ctx_with_paragraphs(6)
    ctx.prior_run_outputs["lesson_record"]["recordings"][0]["motion_video_url"] = (
        "https://minio/s1.mp4"
    )
    ctx.prior_run_outputs["source_ingest"] = {
        "pinned_media": [{"ref": "https://x/p", "composite_url": "https://minio/pin.png"}],
    }
    ctx.prior_run_outputs["ai_video_gen"] = {"videos": [{"video_url": "https://minio/ai.mp4"}]}

    await LessonCompileStage(client=fake).run({}, ctx=ctx)

    _, kwargs = fake.compile.call_args
    shots = {s["line_id"]: s for s in kwargs["shots"]}
    # Curated image (p4) wins — no video on that line.
    assert shots["p4"]["image_url"] == "https://minio/pin.png"
    assert "video_url" not in shots["p4"]
    # AI video (p3) wins over the animated slide.
    assert shots["p3"]["video_url"] == "https://minio/ai.mp4"
    # A line claimed by neither gets the animated slide.
    animated = [pid for pid in shots if shots[pid].get("video_url") == "https://minio/s1.mp4"]
    assert animated  # at least one line plays the animated slide
    assert "p4" not in animated and "p3" not in animated


@pytest.mark.asyncio
async def test_compile_raises_without_dependencies() -> None:
    fake = AsyncMock()
    stage = LessonCompileStage(client=fake)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    with pytest.raises(RuntimeError, match=r"lesson_slides|lesson_narrate|lesson_record"):
        await stage.run({}, ctx=ctx)


def _ctx_with_script(captions_mode: str) -> RunContext:
    # _ctx_with_outputs (p1, p2) + the lesson_script paragraph text the burned-in
    # caption path joins on (paragraph id == each shot's line_id).
    ctx = _ctx_with_outputs()
    ctx.config_snapshot["deliverable"]["captions_mode"] = captions_mode
    ctx.prior_run_outputs["lesson_script"] = {
        "paragraphs": [
            {"id": "p1", "text": "Para one."},
            {"id": "p2", "text": "Para two."},
        ]
    }
    return ctx


@pytest.mark.asyncio
async def test_compile_no_captions_when_mode_none() -> None:
    fake = AsyncMock()
    fake.compile.return_value = {"artifact_url": "http://stub/l.mp4", "duration_seconds": 50.0}

    await LessonCompileStage(client=fake).run({}, ctx=_ctx_with_script("none"))

    _, kwargs = fake.compile.call_args
    for shot in kwargs["shots"]:
        assert "caption" not in shot
