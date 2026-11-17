"""course_compile_lessons — spawn one child run per lesson project; await; collect."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import Field
from redis.asyncio import Redis

from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository, cancelled_by_parent
from src.database.repositories.stage_output_repository import StageOutputRepository
from src.database.session import bind_session_tenant
from src.services.providers.resolver import PROVIDER_SLOTS_KEY
from src.services.runs.sub_dag import ChildRunError, await_runs, collect_outputs
from src.services.stages.base import ArtifactUrl, StageHandler, StageInput, StageOutput

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

SessionFactory = Callable[[], AbstractAsyncContextManager[Any]]


async def _dead_recorded_children(
    repo: RunRepository, recorded: dict[str, str]
) -> tuple[dict[str, str], dict[str, str]]:
    """Split recorded children that can never deliver into ``(respawn, user_cancelled)``.

    * ``respawn`` — ``{deliverable_key: status}`` for a child that FAILED, or that
      was CANCELLED by the parent's cascade (``failure_detail.code ==
      cancelled_by_parent``, stamped by both sub-DAG cascades). Re-spawned fresh.
    * ``user_cancelled`` — ``{deliverable_key: child_run_id}`` for a child cancelled
      by anything else, i.e. a USER. That is a decision, not a fault: reviving it on
      a course retry would silently re-buy the lesson they stopped. It stays
      terminal and the stage fails naming it.

    An unreadable row is NOT treated as dead: re-spawning on a transient read gap
    would duplicate a paid lesson render, and ``await_runs`` already names a
    ``missing`` child in its timeout.
    """
    respawn: dict[str, str] = {}
    user_cancelled: dict[str, str] = {}
    for key, raw_id in recorded.items():
        run = await repo.get(UUID(str(raw_id)))
        if run is None:
            continue
        status = str(run.status)
        if status == "failed" or cancelled_by_parent(run):
            respawn[key] = status
        elif status == "cancelled":
            user_cancelled[key] = str(raw_id)
    return respawn, user_cancelled


def _refuse_user_cancelled(user_cancelled: dict[str, str], lesson_title: dict[str, str]) -> None:
    """Fail the stage, naming the lesson, when a recorded child was USER-cancelled."""
    if not user_cancelled:
        return
    key, child_id = next(iter(user_cancelled.items()))
    raise ChildRunError(
        UUID(child_id),
        "cancelled",
        f"lesson {lesson_title.get(key, key)!r} was cancelled by the user; "
        "re-run that lesson (or remove it from the course) and retry the course",
    )


class CourseCompileLessonsInput(StageInput):
    pass


class LessonOutputSummary(StageOutput):
    lesson_project_id: str
    lesson_run_id: str
    artifact_url: ArtifactUrl
    duration_seconds: float
    # Per-slide mid-lesson markers (lesson-relative), forwarded verbatim from
    # lesson_compile so course_stitch can re-base them onto the course timeline.
    sub_chapters: list[dict] = Field(default_factory=list)


class CourseCompileLessonsOutput(StageOutput):
    lessons: list[LessonOutputSummary]


class CourseCompileLessonsStage(
    StageHandler[CourseCompileLessonsInput, CourseCompileLessonsOutput]
):
    """Bound at construction to engine + redis + session_factory via the
    blueprint's deps-aware handler_factory (see src/services/blueprints/course.py)."""

    name = "course_compile_lessons"
    input_model = CourseCompileLessonsInput
    output_model = CourseCompileLessonsOutput

    def __init__(
        self,
        *,
        run_engine,  # noqa: ANN001  # why: RunEngine import would cycle through dispatcher → stage handler
        redis: Redis,
        session_factory: SessionFactory,
    ) -> None:
        self._engine = run_engine
        self._redis = redis
        self._sessions = session_factory

    async def handle(
        self,
        payload: CourseCompileLessonsInput,
        ctx: RunContext,
    ) -> CourseCompileLessonsOutput:
        course_project_id = ctx.config_snapshot.get("project_id")
        if not course_project_id:
            raise RuntimeError("course config_snapshot missing project_id")
        if ctx.tenant_id is None or ctx.subject_id is None:
            raise RuntimeError("course config_snapshot missing tenant/subject")
        course_uuid = UUID(course_project_id)

        async with self._sessions() as session:
            await bind_session_tenant(session, ctx.tenant_id)
            proj_repo = ProjectRepository(session)
            so_repo = StageOutputRepository(session)
            course = await proj_repo.get_with_children(course_uuid, tenant_id=ctx.tenant_id)
            if course is None:
                raise RuntimeError("course project not found in tenant scope")
            children = list(course.children or [])
            if not children:
                raise RuntimeError("course has no lesson children")
            child_deliverables: list[tuple[UUID, UUID]] = []
            lesson_title: dict[str, str] = {}
            for child in children:
                refresh = await proj_repo.get(child.id, tenant_id=ctx.tenant_id)
                if refresh is None or not refresh.deliverables:
                    raise RuntimeError(f"lesson {child.id} has no deliverables")
                child_deliverables.append((child.id, refresh.deliverables[0].id))
                lesson_title[str(refresh.deliverables[0].id)] = str(refresh.title or child.id)
            # Children we already spawned on a prior attempt of this stage,
            # keyed by lesson deliverable_id. The course run may have been
            # drained/resumed mid-stage; re-attaching to these instead of
            # re-spawning keeps the stage idempotent (no duplicate lesson
            # renders / TTS / uploads). See record_progress.
            existing_so = await so_repo.get(ctx.run_id, self.name)
            recorded: dict[str, str] = (
                dict((existing_so.output or {}).get("children_by_deliverable", {}))
                if existing_so
                else {}
            )
            # Only a child that can still deliver is re-attached. A recorded child
            # that FAILED (or was cancelled by the parent's cascade) is dead: re-
            # attaching it made await_runs raise ChildRunError on every retry of
            # this stage, so a single failed lesson made the course unrecoverable.
            dead_children, user_cancelled = await _dead_recorded_children(
                RunRepository(session), recorded
            )
        # Checked BEFORE any spawn: nothing is bought for a course that cannot
        # complete without the lesson the user stopped.
        _refuse_user_cancelled(user_cancelled, lesson_title)

        # Course-level ingest cache: the course already ran source_ingest once
        # (course_outline depends on it). Hand that result to each lesson child
        # so its own source_ingest short-circuits instead of re-fetching the same
        # brief.sources N times. Only inject when the course actually
        # ingested something; a skipped source_ingest yields {} → no override, so
        # a standalone lesson re-run still grounds itself from its own sources.
        course_ingest = ctx.prior_output("source_ingest") or {}
        ingest_override: dict[str, Any] | None = (
            {"precomputed_source_ingest": course_ingest}
            if course_ingest.get("sources_ingested")
            else None
        )

        run_ids: list[UUID] = []
        proj_id_for_run: dict[UUID, UUID] = {}
        for project_id, deliverable_id in child_deliverables:
            key = str(deliverable_id)
            recorded_id = recorded.get(key)
            if recorded_id is not None and key not in dead_children:
                child_run_id = UUID(recorded_id)
            else:
                if recorded_id is not None:
                    # Fresh spawn, not ``parent_run_id=<dead child>``: the stage-
                    # preserving restart plan (``services/runs/restart_plan.py``) is
                    # applied only by the regenerate ROUTE, never by ``start_run``
                    # itself, so a ``parent_run_id`` here would buy nothing a fresh
                    # lesson run does not already do.
                    log.info(
                        "course %s: lesson deliverable %s child %s is %s; re-spawning",
                        ctx.run_id,
                        key,
                        recorded_id,
                        dead_children[key],
                    )
                child_run = await self._engine.start_run(
                    deliverable_id=deliverable_id,
                    tenant_id=ctx.tenant_id,
                    subject_id=ctx.subject_id,
                    parent_run_id=None,
                    pause_after_stages=[],
                    config_overrides=ingest_override,
                    # System-spawned: the lessons run on the COURSE run's pinned
                    # providers, not whatever the tenant configured since.
                    inherit_slots=ctx.config_snapshot.get(PROVIDER_SLOTS_KEY),
                    # ...and the COURSE run's pinned tenant defaults (P6).
                    inherit_defaults_from=ctx.config_snapshot,
                )
                child_run_id = child_run.id
                recorded[key] = str(child_run_id)
                # Record AFTER each spawn so a drain BETWEEN spawns still
                # re-attaches this child rather than orphaning + re-spawning it.
                async with self._sessions() as session:
                    await bind_session_tenant(session, ctx.tenant_id)
                    await StageOutputRepository(session).record_progress(
                        ctx.run_id,
                        self.name,
                        {"children_by_deliverable": recorded},
                    )
                    await session.commit()
            run_ids.append(child_run_id)
            proj_id_for_run[child_run_id] = project_id

        await await_runs(
            run_ids=run_ids,
            session_factory=self._sessions,
            redis=self._redis,
            tenant_id=ctx.tenant_id,
            # Bounds the child wait by THIS run's remaining budget instead of a
            # fixed 7200s that sat above every successful run ever recorded.
            parent_run_id=ctx.run_id,
            # The lessons were all spawned at once onto the SAME GPU pool, so
            # they largely render in series: the wait cap scales with the count.
            child_count=len(run_ids),
            fail_on_any=True,
        )
        outputs = await collect_outputs(
            run_ids=run_ids,
            stage_name="lesson_compile",
            session_factory=self._sessions,
            tenant_id=ctx.tenant_id,
        )
        return CourseCompileLessonsOutput(
            lessons=[
                LessonOutputSummary(
                    lesson_project_id=str(proj_id_for_run[rid]),
                    lesson_run_id=str(rid),
                    artifact_url=str(outputs[rid]["artifact_url"]),
                    duration_seconds=float(outputs[rid]["duration_seconds"]),
                    sub_chapters=list(outputs[rid].get("sub_chapters") or []),
                )
                for rid in run_ids
            ],
        )
