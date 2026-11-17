from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.course.stitch import CourseStitchStage
from src.services.tts_storage import set_storage_for_tests


def _ctx(
    lessons: list[dict],
    *,
    config_snapshot: dict | None = None,
    outline: dict | None = None,
    thumbnail: dict | None = None,
) -> RunContext:
    prior: dict[str, dict] = {"course_compile_lessons": {"lessons": lessons}}
    if outline is not None:
        prior["course_outline"] = outline
    if thumbnail is not None:
        prior["course_thumbnail"] = thumbnail
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=config_snapshot or {},
        prior_run_outputs=prior,
    )


async def test_stitch_builds_video_url_shots_per_lesson() -> None:
    fake = AsyncMock()
    fake.compile.return_value = {
        "artifact_url": "http://stub/course.mp4",
        "duration_seconds": 120.0,
    }
    stage = CourseStitchStage(client=fake)
    ctx = _ctx(
        [
            {
                "lesson_project_id": "p1",
                "lesson_run_id": "r1",
                "artifact_url": "http://stub/l1.mp4",
                "duration_seconds": 60.0,
            },
            {
                "lesson_project_id": "p2",
                "lesson_run_id": "r2",
                "artifact_url": "http://stub/l2.mp4",
                "duration_seconds": 0.0,
            },
        ]
    )

    out = await stage.run({}, ctx=ctx)

    assert out.course_artifact_url == "http://stub/course.mp4"
    assert out.duration_seconds == 120.0
    assert out.lesson_count == 2
    _, kwargs = fake.compile.call_args
    shots = kwargs["shots"]
    assert [s["video_url"] for s in shots] == ["http://stub/l1.mp4", "http://stub/l2.mp4"]
    # zero-duration lesson floors to a positive seconds (Shot.seconds gt 0)
    assert shots[0]["seconds"] == 60.0
    assert shots[1]["seconds"] == 1.0
    assert kwargs["tags"]["stage"] == "course_stitch"


async def test_stitch_idempotency_key_is_stable_across_a_resume() -> None:
    """A resumable-timeout re-dispatch re-runs handle(); the compile submit must
    carry the SAME idempotency key both times so the compiler dedupes it instead
    of running a second paid render."""
    fake = AsyncMock()
    fake.compile.return_value = {
        "artifact_url": "http://stub/course.mp4",
        "duration_seconds": 120.0,
    }
    stage = CourseStitchStage(client=fake)
    ctx = _ctx(
        [
            {
                "lesson_project_id": "p1",
                "lesson_run_id": "r1",
                "artifact_url": "http://stub/l1.mp4",
                "duration_seconds": 60.0,
            }
        ]
    )
    # Run the handler TWICE with the SAME ctx to simulate the re-dispatch.
    await stage.run({}, ctx=ctx)
    await stage.run({}, ctx=ctx)
    keys = [c.kwargs["idempotency_key"] for c in fake.compile.call_args_list]
    assert keys[0] == keys[1] == f"{ctx.run_id}:course_stitch"


class _FakeResignStorage:
    """Re-signs every URL by tagging ``&fresh`` (mirrors the shot_presign fake)."""

    async def resign_presigned_url(self, url: str) -> str:
        return f"{url}&fresh"


async def test_stitch_resigns_lesson_urls_before_compile() -> None:
    """A multi-lesson course paused past the 1h presign TTL: the lesson video_urls
    must be re-minted before the compile POST, or video_compiler 403s the dead
    signature → compile 500s. course_stitch calls compile() directly so the shared
    CompileStageBase re-sign tail never covers it."""
    set_storage_for_tests(cast("Any", _FakeResignStorage()))
    try:
        fake = AsyncMock()
        fake.compile.return_value = {
            "artifact_url": "http://stub/course.mp4",
            "duration_seconds": 120.0,
        }
        stage = CourseStitchStage(client=fake)
        ctx = _ctx(
            [
                {
                    "lesson_project_id": "p1",
                    "lesson_run_id": "r1",
                    "artifact_url": "https://m/l1.mp4",
                    "duration_seconds": 60.0,
                },
                {
                    "lesson_project_id": "p2",
                    "lesson_run_id": "r2",
                    "artifact_url": "https://m/l2.mp4",
                    "duration_seconds": 60.0,
                },
            ]
        )
        await stage.run({}, ctx=ctx)
    finally:
        set_storage_for_tests(None)

    _, kwargs = fake.compile.call_args
    shots_urls = [s["video_url"] for s in kwargs["shots"]]
    assert shots_urls == ["https://m/l1.mp4&fresh", "https://m/l2.mp4&fresh"]


def _three_lessons() -> list[dict]:
    return [
        {
            "lesson_project_id": "p1",
            "lesson_run_id": "r1",
            "artifact_url": "u1",
            "duration_seconds": 60.0,
        },
        {
            "lesson_project_id": "p2",
            "lesson_run_id": "r2",
            "artifact_url": "u2",
            "duration_seconds": 40.0,
        },
        {
            "lesson_project_id": "p3",
            "lesson_run_id": "r3",
            "artifact_url": "u3",
            "duration_seconds": 20.0,
        },
    ]


def _compile_ok(total: float = 120.0):
    fake = AsyncMock()
    fake.compile.return_value = {
        "artifact_url": "http://stub/course.mp4",
        "duration_seconds": total,
    }
    return fake


async def test_chapters_cumulative_offsets_first_is_zero() -> None:
    stage = CourseStitchStage(client=_compile_ok(120.0))
    ctx = _ctx(
        _three_lessons(),
        config_snapshot={
            "lessons": [
                {"lesson_project_id": "p1", "title": "Intro"},
                {"lesson_project_id": "p2", "title": "Core"},
                {"lesson_project_id": "p3", "title": "Wrap"},
            ]
        },
    )

    out = await stage.run({}, ctx=ctx)

    assert [c.start_seconds for c in out.chapters] == [0.0, 60.0, 100.0]
    assert [c.title for c in out.chapters] == ["Intro", "Core", "Wrap"]
    assert [c.sequence for c in out.chapters] == [1, 2, 3]
    # artifact_url mirrors course_artifact_url for the shape-based viewers.
    assert out.artifact_url == out.course_artifact_url == "http://stub/course.mp4"


async def test_chapters_skip_zero_duration_and_keep_first_at_zero() -> None:
    stage = CourseStitchStage(client=_compile_ok(80.0))
    lessons = [
        {
            "lesson_project_id": "p0",
            "lesson_run_id": "r0",
            "artifact_url": "u0",
            "duration_seconds": 0.0,
        },
        {
            "lesson_project_id": "p1",
            "lesson_run_id": "r1",
            "artifact_url": "u1",
            "duration_seconds": 60.0,
        },
        {
            "lesson_project_id": "p2",
            "lesson_run_id": "r2",
            "artifact_url": "u2",
            "duration_seconds": 20.0,
        },
    ]
    out = await stage.run({}, ctx=_ctx(lessons))

    # The zero-duration lesson produces no marker; first real chapter starts at 0.
    assert [c.start_seconds for c in out.chapters] == [0.0, 60.0]
    assert [c.title for c in out.chapters] == ["Lesson 1", "Lesson 2"]


async def test_chapter_titles_fall_back_to_outline_then_generic() -> None:
    stage = CourseStitchStage(client=_compile_ok(120.0))
    ctx = _ctx(
        _three_lessons(),
        config_snapshot={"lessons": [{"lesson_project_id": "p1", "title": "Intro"}]},
        outline={"sections": [{"lesson_project_id": "p2", "title": "From Outline"}]},
    )

    out = await stage.run({}, ctx=ctx)

    # p1 from config_snapshot, p2 from outline sections, p3 generic fallback.
    assert [c.title for c in out.chapters] == ["Intro", "From Outline", "Lesson 3"]


async def test_chapters_clamp_overflow_past_true_total() -> None:
    # Input durations sum to 120 but the real stitched total is only 70 — the
    # markers that would start at/after 70 are dropped, not fabricated.
    stage = CourseStitchStage(client=_compile_ok(70.0))
    out = await stage.run({}, ctx=_ctx(_three_lessons()))

    starts = [c.start_seconds for c in out.chapters]
    assert starts == [0.0, 60.0]  # third would start at 100 >= 70 → dropped


async def test_thumbnail_url_none_when_stage_skipped() -> None:
    stage = CourseStitchStage(client=_compile_ok(120.0))
    out = await stage.run({}, ctx=_ctx(_three_lessons()))  # no course_thumbnail output
    assert out.thumbnail_url is None


async def test_chapters_carry_lesson_project_id_and_level_zero() -> None:
    stage = CourseStitchStage(client=_compile_ok(120.0))
    out = await stage.run({}, ctx=_ctx(_three_lessons()))
    assert [c.level for c in out.chapters] == [0, 0, 0]
    assert [c.lesson_project_id for c in out.chapters] == ["p1", "p2", "p3"]
    assert all(c.parent_sequence is None for c in out.chapters)


def _lessons_with_subs() -> list[dict]:
    return [
        {
            "lesson_project_id": "p1",
            "lesson_run_id": "r1",
            "artifact_url": "u1",
            "duration_seconds": 60.0,
            "sub_chapters": [
                {"sequence": 1, "title": "A", "start_seconds": 0.0, "duration_seconds": 30.0},
                {"sequence": 2, "title": "B", "start_seconds": 30.0, "duration_seconds": 30.0},
            ],
        },
        {
            "lesson_project_id": "p2",
            "lesson_run_id": "r2",
            "artifact_url": "u2",
            "duration_seconds": 40.0,
            "sub_chapters": [],
        },
    ]


async def test_subchapters_interleave_with_absolute_offsets() -> None:
    stage = CourseStitchStage(client=_compile_ok(100.0))
    out = await stage.run({}, ctx=_ctx(_lessons_with_subs()))

    # level-0 p1 @0, its two subs @0 / @30 (lesson_start 0 + relative), level-0 p2 @60.
    assert [c.start_seconds for c in out.chapters] == [0.0, 0.0, 30.0, 60.0]
    assert [c.level for c in out.chapters] == [0, 1, 1, 0]
    assert [c.parent_sequence for c in out.chapters] == [None, 1, 1, None]
    assert [c.sequence for c in out.chapters] == [1, 2, 3, 4]
    assert [c.title for c in out.chapters][1:3] == ["A", "B"]
    # sub-chapters inherit their lesson's project id (overrides apply by it).
    assert [c.lesson_project_id for c in out.chapters] == ["p1", "p1", "p1", "p2"]
    # lesson_count stays a lesson count, not a marker count.
    assert out.lesson_count == 2


async def test_chapters_emit_stable_override_key() -> None:
    """Each chapter carries a ``chapter_key`` an override is applied by — the bare
    ``lesson_project_id`` at level 0, ``{lesson_project_id}#{ordinal}`` at level 1
    (1-based slide ordinal within the lesson). Sub-chapters share the parent's
    ``lesson_project_id`` but get distinct keys via the ordinal."""
    stage = CourseStitchStage(client=_compile_ok(100.0))
    out = await stage.run({}, ctx=_ctx(_lessons_with_subs()))
    # rows: p1 lesson, p1 sub A, p1 sub B, p2 lesson
    assert [c.chapter_key for c in out.chapters] == ["p1", "p1#1", "p1#2", "p2"]


async def test_subchapters_rebased_onto_second_lesson() -> None:
    lessons = _lessons_with_subs()
    lessons[1]["sub_chapters"] = [
        {"sequence": 1, "title": "C", "start_seconds": 0.0, "duration_seconds": 20.0},
        {"sequence": 2, "title": "D", "start_seconds": 20.0, "duration_seconds": 20.0},
    ]
    stage = CourseStitchStage(client=_compile_ok(100.0))
    out = await stage.run({}, ctx=_ctx(lessons))
    # p2 starts at 60; its subs land at 60 and 80 on the course timeline.
    by_title = {c.title: c for c in out.chapters}
    assert by_title["C"].start_seconds == 60.0
    assert by_title["D"].start_seconds == 80.0
    assert by_title["C"].parent_sequence == by_title["D"].parent_sequence == 4


async def test_subchapters_clamped_past_true_total() -> None:
    lessons = _lessons_with_subs()
    # Real total is only 35s — the second sub (abs 30) survives, but make a sub
    # that would land at/after 35 and confirm it is dropped.
    lessons[0]["sub_chapters"] = [
        {"sequence": 1, "title": "A", "start_seconds": 0.0, "duration_seconds": 20.0},
        {"sequence": 2, "title": "B", "start_seconds": 40.0, "duration_seconds": 20.0},
    ]
    stage = CourseStitchStage(client=_compile_ok(35.0))
    out = await stage.run({}, ctx=_ctx(lessons))
    titles = [c.title for c in out.chapters]
    assert "A" in titles
    assert "B" not in titles  # abs_start 40 >= 35 → dropped


async def test_title_card_offsets_chapters_and_keeps_lesson_count(monkeypatch) -> None:
    """When an intro card is opted in, course_stitch prepends it as a shot, the
    chapter offsets shift by its duration, and lesson_count stays a lesson count."""
    from src.services.stages.course import stitch as stitch_mod

    intro_shot = {
        "id": "title-intro",
        "line_id": "__title_intro__",
        "seconds": 3.0,
        "video_url": "http://stub/intro.mp4",
    }
    monkeypatch.setattr(
        stitch_mod, "build_title_card_shots", AsyncMock(return_value=(intro_shot, None))
    )

    fake = _compile_ok(123.0)
    stage = CourseStitchStage(client=fake)
    ctx = _ctx(
        _three_lessons(),
        config_snapshot={
            "title_card": {"enabled": True},
            "lessons": [
                {"lesson_project_id": "p1", "title": "Intro"},
                {"lesson_project_id": "p2", "title": "Core"},
                {"lesson_project_id": "p3", "title": "Wrap"},
            ],
        },
    )

    out = await stage.run({}, ctx=ctx)

    # The intro card is the first shot; the three lessons follow.
    _, kwargs = fake.compile.call_args
    assert kwargs["shots"][0]["video_url"] == "http://stub/intro.mp4"
    assert len(kwargs["shots"]) == 4
    # Chapter offsets shifted by the 3s intro; lesson_count counts lessons only.
    assert [c.start_seconds for c in out.chapters] == [3.0, 63.0, 103.0]
    assert out.lesson_count == 3


async def test_stitch_raises_without_compiled_lessons() -> None:
    stage = CourseStitchStage(client=AsyncMock())
    with pytest.raises(RuntimeError, match="course_compile_lessons"):
        await stage.run(
            {}, ctx=RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
        )


async def test_stitch_raises_when_no_lessons() -> None:
    stage = CourseStitchStage(client=AsyncMock())
    with pytest.raises(RuntimeError, match="no compiled lessons"):
        await stage.run({}, ctx=_ctx([]))


async def test_stitch_threads_authored_mastering_loudness_target() -> None:
    """An authored loudness target must reach the stitch compile.

    ``brief.voice.master_lufs`` is aliased into
    ``config_snapshot["mastering"]["loudness_target"]``; course_stitch calls
    compile() directly (not via CompileStageBase), so it must build and thread
    the mastering profile itself or the course master silently renders at the
    compiler's env default.
    """
    fake = _compile_ok(120.0)
    stage = CourseStitchStage(client=fake)
    ctx = _ctx(_three_lessons(), config_snapshot={"mastering": {"loudness_target": -16.0}})

    await stage.run({}, ctx=ctx)

    mastering = fake.compile.call_args.kwargs["mastering"]
    assert mastering is not None
    # Pin the AUTHORED value: a silently-defaulted (-12.5) profile must NOT pass.
    assert mastering["loudness_target"] == -16.0


async def test_stitch_omits_mastering_when_block_absent() -> None:
    """Presence-is-opt-in: no ``mastering`` block ⇒ no mastering on the request,
    byte-identical to the pre-fix behaviour for courses that never author it."""
    fake = _compile_ok(120.0)
    stage = CourseStitchStage(client=fake)

    await stage.run({}, ctx=_ctx(_three_lessons()))

    assert fake.compile.call_args.kwargs["mastering"] is None


async def test_stitch_disables_audio_master_but_keeps_authored_target() -> None:
    """The stitch must NOT re-master already-mastered lessons.

    Every lesson was loudness-mastered to the authored target by its own
    CompileStageBase run; a second pass over the concatenation collapses loudness
    range by ~3 LU without moving integrated LUFS (measured in
    video_compiler_service test_double_master.py). So course_stitch sends
    ``tags["mastering"]="off"`` to disable the compiler-side audio master, while
    STILL sending the mastering profile — the profile carries the picture/encode
    fields and keeps the target correct if the tag is ever flipped back.
    """
    fake = _compile_ok(120.0)
    stage = CourseStitchStage(client=fake)
    ctx = _ctx(_three_lessons(), config_snapshot={"mastering": {"loudness_target": -16.0}})

    await stage.run({}, ctx=ctx)

    kwargs = fake.compile.call_args.kwargs
    # The audio master is disabled at the stitch...
    assert kwargs["tags"]["mastering"] == "off"
    # ...but #625's authored-target wiring is intact (profile still sent, still -16.0).
    assert kwargs["mastering"] is not None
    assert kwargs["mastering"]["loudness_target"] == -16.0
