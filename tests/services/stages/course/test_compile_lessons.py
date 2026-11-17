"""course_compile_lessons idempotency — a drained+resumed course run must
re-attach to the child lesson runs it already spawned, never spawn a second set
(which would duplicate every lesson's render / TTS / upload)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository
from src.database.repositories.stage_output_repository import StageOutputRepository
from src.services.runs.run_context import RunContext
from src.services.runs.sub_dag import ChildRunError
from src.services.stages.course.compile_lessons import CourseCompileLessonsStage

pytestmark = pytest.mark.asyncio

_STAGE = "course_compile_lessons"


async def _build_course(session_factory, tenant: UUID, subject: UUID):
    """Create a course project with 2 lesson children (each with a deliverable),
    a course run, and its pending course_compile_lessons stage_output row."""
    async with session_factory() as s:
        pr = ProjectRepository(s)
        course = await pr.create_with_deliverables(
            tenant_id=tenant,
            owner_subject_id=subject,
            type_="course",
            title="course",
            brief={},
            brand={},
            voice={},
            theme={},
            visibility="tenant",
            deliverables=[{"name": "c", "format": "16x9", "captions_mode": "none"}],
        )
        for i in range(2):
            await pr.create_with_deliverables(
                tenant_id=tenant,
                owner_subject_id=subject,
                type_="lesson",
                title=f"lesson-{i}",
                brief={},
                brand={},
                voice={},
                theme={},
                visibility="tenant",
                deliverables=[{"name": f"l{i}", "format": "16x9", "captions_mode": "none"}],
                parent_project_id=course.id,
            )
        await s.commit()
        course_deliv_id = course.deliverables[0].id

        rr = RunRepository(s)
        run = await rr.create(
            deliverable_id=course_deliv_id,
            blueprint_id="course_v1",
            config_snapshot={},
            pause_after_stages=[],
        )
        await StageOutputRepository(s).upsert_pending(run.id, _STAGE)
        await s.commit()
        return course.id, run.id


async def test_compile_lessons_reattaches_after_drain(db_session_factory) -> None:
    tenant, subject = uuid4(), uuid4()
    course_id, course_run_id = await _build_course(db_session_factory, tenant, subject)

    spawned: list[UUID] = []

    async def _fake_start_run(**kwargs):
        rid = uuid4()
        spawned.append(rid)
        return SimpleNamespace(id=rid)

    engine = SimpleNamespace(start_run=AsyncMock(side_effect=_fake_start_run))
    stage = CourseCompileLessonsStage(
        run_engine=engine, redis=AsyncMock(), session_factory=db_session_factory
    )
    ctx = RunContext(
        run_id=course_run_id,
        deliverable_id=uuid4(),
        config_snapshot={"project_id": str(course_id)},
        tenant_id=tenant,
        subject_id=subject,
    )

    # Attempt 1: the pod drains while awaiting children — children get spawned
    # and recorded, but the stage never completes.
    with patch(
        "src.services.stages.course.compile_lessons.await_runs",
        new=AsyncMock(side_effect=asyncio.CancelledError),
    ):
        with pytest.raises(asyncio.CancelledError):
            await stage.run({}, ctx=ctx)

    assert engine.start_run.await_count == 2, "first attempt should spawn one run per lesson"
    async with db_session_factory() as s:
        so = await StageOutputRepository(s).get(course_run_id, _STAGE)
        assert so is not None, "stage output row must exist after the first attempt"
        recorded = (so.output or {}).get("children_by_deliverable", {})
    assert len(recorded) == 2, f"spawned children must be durably recorded: {so.output}"

    # Attempt 2: the new pod resumes the course run and re-enters the stage. It
    # MUST re-attach to the 2 recorded children, not spawn a second set.
    canned = {
        rid: {"artifact_url": f"http://a/{rid}.mp4", "duration_seconds": 1.0} for rid in spawned
    }
    with (
        patch("src.services.stages.course.compile_lessons.await_runs", new=AsyncMock()),
        patch(
            "src.services.stages.course.compile_lessons.collect_outputs",
            new=AsyncMock(return_value=canned),
        ),
    ):
        out = await stage.run({}, ctx=ctx)

    assert engine.start_run.await_count == 2, "resume must NOT spawn additional child runs"
    assert {UUID(s.lesson_run_id) for s in out.lessons} == set(spawned)


async def _run_stage_collecting_overrides(db_session_factory, ctx_prior_outputs: dict) -> list:
    """Run the stage once (children spawn + complete) and return the
    config_overrides passed to each start_run call."""
    tenant, subject = uuid4(), uuid4()
    course_id, course_run_id = await _build_course(db_session_factory, tenant, subject)

    spawned: list[UUID] = []

    async def _fake_start_run(**kwargs):
        rid = uuid4()
        spawned.append(rid)
        return SimpleNamespace(id=rid)

    engine = SimpleNamespace(start_run=AsyncMock(side_effect=_fake_start_run))
    stage = CourseCompileLessonsStage(
        run_engine=engine, redis=AsyncMock(), session_factory=db_session_factory
    )
    ctx = RunContext(
        run_id=course_run_id,
        deliverable_id=uuid4(),
        config_snapshot={"project_id": str(course_id)},
        tenant_id=tenant,
        subject_id=subject,
        prior_run_outputs=ctx_prior_outputs,
    )

    async def _fake_collect(*, run_ids, **_kw):
        return {
            rid: {"artifact_url": f"http://a/{rid}.mp4", "duration_seconds": 1.0} for rid in run_ids
        }

    with (
        patch("src.services.stages.course.compile_lessons.await_runs", new=AsyncMock()),
        patch(
            "src.services.stages.course.compile_lessons.collect_outputs",
            new=AsyncMock(side_effect=_fake_collect),
        ),
    ):
        await stage.run({}, ctx=ctx)
    return [c.kwargs.get("config_overrides") for c in engine.start_run.await_args_list]


async def test_compile_lessons_passes_course_ingest_to_children(db_session_factory) -> None:
    """When the course ran source_ingest, every lesson child is spawned with the
    precomputed payload so it skips re-fetching the same sources."""
    ingest = {
        "grounding_text": "# Source: Acme\nfacts",
        "references": [],
        "images": [],
        "pinned_media": [],
        "notes": [],
        "sources_ingested": 2,
    }
    overrides = await _run_stage_collecting_overrides(db_session_factory, {"source_ingest": ingest})
    assert len(overrides) == 2
    assert all(o == {"precomputed_source_ingest": ingest} for o in overrides)


async def test_compile_lessons_no_override_when_ingest_skipped(db_session_factory) -> None:
    """A course with no sources (source_ingest skipped → {}) injects no override,
    so a standalone lesson re-run still grounds itself."""
    overrides = await _run_stage_collecting_overrides(db_session_factory, {"source_ingest": {}})
    assert overrides == [None, None]


async def test_lesson_children_inherit_the_course_runs_provider_pins(db_session_factory) -> None:
    """P3: lessons are SYSTEM-spawned children — they run on the course run's pinned
    providers, so a tenant switching its LLM mid-course cannot split one course
    across two providers."""
    tenant, subject = uuid4(), uuid4()
    course_id, course_run_id = await _build_course(db_session_factory, tenant, subject)
    pins = {
        "llm": {
            "source": "tenant",
            "kind": "anthropic",
            "connection_id": str(uuid4()),
            "version": 2,
        }
    }
    engine = SimpleNamespace(
        start_run=AsyncMock(side_effect=lambda **_: SimpleNamespace(id=uuid4()))
    )
    stage = CourseCompileLessonsStage(
        run_engine=engine, redis=AsyncMock(), session_factory=db_session_factory
    )
    ctx = RunContext(
        run_id=course_run_id,
        deliverable_id=uuid4(),
        config_snapshot={
            "project_id": str(course_id),
            "provider_slots": pins,
            # P6: the course run's pinned tenant defaults ride down to every lesson.
            "tenant_defaults": {"version": 4, "voice": {"voice_id": "bf_emma"}},
        },
        tenant_id=tenant,
        subject_id=subject,
    )
    with patch(
        "src.services.stages.course.compile_lessons.await_runs",
        new=AsyncMock(side_effect=asyncio.CancelledError),
    ):
        with pytest.raises(asyncio.CancelledError):
            await stage.run({}, ctx=ctx)

    assert engine.start_run.await_count == 2
    for call in engine.start_run.await_args_list:
        assert call.kwargs["inherit_slots"] == pins
        assert call.kwargs["inherit_defaults_from"]["tenant_defaults"] == {
            "version": 4,
            "voice": {"voice_id": "bf_emma"},
        }


async def _course_with_one_dead_lesson(db_session_factory, dead_status: str):
    """Course with lesson 0 completed and lesson 1 in ``dead_status``:
    ``failed`` | ``cancelled_by_parent`` (cascade) | ``cancelled_by_user``."""
    tenant, subject = uuid4(), uuid4()
    course_id, course_run_id = await _build_course(db_session_factory, tenant, subject)
    async with db_session_factory() as s:
        pr, rr = ProjectRepository(s), RunRepository(s)
        course = await pr.get_with_children(course_id, tenant_id=tenant)
        assert course is not None
        lesson_delivs = []
        for child in course.children:
            refreshed = await pr.get(child.id, tenant_id=tenant)
            assert refreshed is not None
            lesson_delivs.append(refreshed.deliverables[0].id)
        done, dead = [
            await rr.create(
                deliverable_id=d,
                blueprint_id="lesson_v1",
                config_snapshot={},
                pause_after_stages=[],
            )
            for d in lesson_delivs
        ]
        await rr.mark_completed(done.id)
        if dead_status == "failed":
            await rr.mark_failed(dead.id, "lesson_record failed")
        else:
            await rr.mark_cancelled(dead.id)
            if dead_status == "cancelled_by_parent":
                assert await rr.mark_cancelled_by_parent(dead.id, course_run_id)
        await StageOutputRepository(s).record_progress(
            course_run_id,
            _STAGE,
            {
                "children_by_deliverable": {
                    str(lesson_delivs[0]): str(done.id),
                    str(lesson_delivs[1]): str(dead.id),
                }
            },
        )
        await s.commit()
    return tenant, subject, course_id, course_run_id, lesson_delivs, done, dead


@pytest.mark.parametrize("dead_status", ["failed"])
async def test_compile_lessons_respawns_a_dead_recorded_child(
    db_session_factory, dead_status: str
) -> None:
    """A retry of the stage must re-spawn a FAILED (or cascade-CANCELLED) recorded
    lesson and keep re-attaching the completed one. Re-attaching a dead child made
    await_runs raise ChildRunError on every retry — one failed lesson made the course
    unrecoverable."""
    (
        tenant,
        subject,
        course_id,
        course_run_id,
        lesson_delivs,
        done,
        _dead,
    ) = await _course_with_one_dead_lesson(db_session_factory, dead_status)

    fresh = uuid4()
    engine = SimpleNamespace(start_run=AsyncMock(return_value=SimpleNamespace(id=fresh)))
    stage = CourseCompileLessonsStage(
        run_engine=engine, redis=AsyncMock(), session_factory=db_session_factory
    )
    ctx = RunContext(
        run_id=course_run_id,
        deliverable_id=uuid4(),
        config_snapshot={"project_id": str(course_id)},
        tenant_id=tenant,
        subject_id=subject,
    )

    async def _fake_collect(*, run_ids, **_kw):
        return {
            rid: {"artifact_url": f"http://a/{rid}.mp4", "duration_seconds": 1.0} for rid in run_ids
        }

    awaited = AsyncMock()
    with (
        patch("src.services.stages.course.compile_lessons.await_runs", new=awaited),
        patch(
            "src.services.stages.course.compile_lessons.collect_outputs",
            new=AsyncMock(side_effect=_fake_collect),
        ),
    ):
        out = await stage.run({}, ctx=ctx)

    assert engine.start_run.await_count == 1
    assert engine.start_run.await_args.kwargs["deliverable_id"] == lesson_delivs[1]
    assert awaited.await_args is not None
    assert set(awaited.await_args.kwargs["run_ids"]) == {done.id, fresh}
    assert {UUID(x.lesson_run_id) for x in out.lessons} == {done.id, fresh}
    async with db_session_factory() as s:
        so = await StageOutputRepository(s).get(course_run_id, _STAGE)
        assert so is not None
        recorded = (so.output or {}).get("children_by_deliverable", {})
    assert recorded.get(str(lesson_delivs[1])) == str(fresh)


async def test_compile_lessons_keeps_a_user_cancelled_lesson_terminal(
    db_session_factory,
) -> None:
    """A lesson the USER cancelled is a decision, not a fault: a course retry must
    NOT revive (and re-buy) it. The stage fails naming the lesson; nothing spawns."""
    (
        tenant,
        subject,
        course_id,
        course_run_id,
        _delivs,
        _done,
        dead,
    ) = await _course_with_one_dead_lesson(db_session_factory, "cancelled_by_user")
    engine = SimpleNamespace(start_run=AsyncMock())
    stage = CourseCompileLessonsStage(
        run_engine=engine, redis=AsyncMock(), session_factory=db_session_factory
    )
    ctx = RunContext(
        run_id=course_run_id,
        deliverable_id=uuid4(),
        config_snapshot={"project_id": str(course_id)},
        tenant_id=tenant,
        subject_id=subject,
    )
    awaited = AsyncMock()
    with (
        patch("src.services.stages.course.compile_lessons.await_runs", new=awaited),
        pytest.raises(ChildRunError) as exc,
    ):
        await stage.run({}, ctx=ctx)

    assert engine.start_run.await_count == 0
    assert awaited.await_count == 0
    assert exc.value.run_id == dead.id and exc.value.status == "cancelled"
    assert "lesson-1" in str(exc.value) and "cancelled by the user" in str(exc.value)


async def test_compile_lessons_scales_the_child_wait_by_its_lesson_count(
    db_session_factory,
) -> None:
    """The fan-out wait is sized for N lessons on a shared GPU, not one child."""
    tenant, subject = uuid4(), uuid4()
    course_id, course_run_id = await _build_course(db_session_factory, tenant, subject)

    async def _fake_start_run(**kwargs):
        return SimpleNamespace(id=uuid4())

    engine = SimpleNamespace(start_run=AsyncMock(side_effect=_fake_start_run))
    stage = CourseCompileLessonsStage(
        run_engine=engine, redis=AsyncMock(), session_factory=db_session_factory
    )
    ctx = RunContext(
        run_id=course_run_id,
        deliverable_id=uuid4(),
        config_snapshot={"project_id": str(course_id)},
        tenant_id=tenant,
        subject_id=subject,
    )
    awaited = AsyncMock(side_effect=asyncio.CancelledError)
    with patch("src.services.stages.course.compile_lessons.await_runs", new=awaited):
        with pytest.raises(asyncio.CancelledError):
            await stage.run({}, ctx=ctx)
    assert awaited.await_count == 1
    call = awaited.await_args
    assert call is not None
    kwargs = call.kwargs
    assert kwargs["child_count"] == 2 == len(kwargs["run_ids"])
    assert kwargs["parent_run_id"] == course_run_id
