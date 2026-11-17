"""In-process full-DAG dispatch integration test — the `course` archetype.

Sixth and last archetype sibling of `test_in_process_lesson.py` (and the
generic/promo/website_review/app_review tests): drives a real course run
through the in-process dispatcher to a terminal status, every downstream client
mocked at its stage-module import site (fakeredis + CI Postgres, no live
service). See `_dispatch_harness.py` for why these run in default per-PR CI.

A course is a *sub-DAG dispatcher*: `course_compile_lessons` spawns one child
lesson run per lesson project and awaits them, so a course run also executes N
full lesson DAGs underneath it. This test folds in the ENTIRE child-lesson mock
surface (`_lesson_downstream_mocks()`) and adds the course-own clients on top.

The course spine (non-opt-in stages) is `course_outline → course_compile_lessons
→ course_stitch → growth_bundle`, plus `course_thumbnail` (parallel off
course_outline, best-effort). `ai_image_gen` IS in the course DAG but is seeded
only for `type=="lesson"` runs, so for a course it skips (recorded "skipped",
NOT "completed") — as do the opt-in leaves `source_ingest` / `media_discovery`.
So it is absent from `_completed_stage_names` and the spine assertion uses `==`.
`course_thumbnail` silently SKIPS on any generation failure (the skip trap), so
its ACG client must return a real `AcgConceptImageJob` with an absolute
`image_url` or the `==` assertion drops it and fails.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from tests.integration._dispatch_harness import (
    _compile_mock,
    _completed_stage_names,
    _create_project,
    _dispatch_and_wait_with_run_engine,
    _wm_client,
)
from tests.integration.test_in_process_lesson import (
    _LESSON_BRIEF,
    _LESSON_SPINE,
    _acg_image_mock,
    _lesson_downstream_mocks,
)

# ---------------------------------------------------------------------------
# Archetype-specific downstream mocks + brief
# ---------------------------------------------------------------------------

# The exact set of stages a plain course run marks "completed". `ai_image_gen`
# is in the DAG but is seeded only for lesson runs, so it skips clean for a
# course; the opt-in leaves (source_ingest / media_discovery) skip clean too.
# Skip-clean leaves are recorded "skipped" (never "completed"), so `==` (not
# `<=`) is correct — a wrong patch target that no-ops a spine stage would drop
# it from this set and fail the assertion.
_COURSE_SPINE = {
    "course_outline",
    "course_compile_lessons",
    "course_stitch",
    "growth_bundle",
    "course_thumbnail",
}

_COURSE_BRIEF = {
    "audience": "beginners",
    "outcome": "master recursion",
    "call_to_action": "Enroll now.",
    "target_duration_seconds": 30,
}


def _course_downstream_mocks() -> ExitStack:
    """Every downstream client a plain course run touches, patched at its
    stage-module import site.

    The course fans out to N full child lesson runs, so this FIRST folds in the
    entire child-lesson mock surface (`_lesson_downstream_mocks()`), then adds
    the course-own clients:

    - course_outline: WMClient → the `sections` + `intro_summary` object the
      outline stage reads (outline.py:114/118 read `data["sections"]` items'
      `title`/`objective` and `data["intro_summary"]`).
    - course_stitch: VideoCompileClient.compile → {artifact_url, duration_seconds}.
    - course_thumbnail: AIContentGenerationClient (the shared lesson ACG helper —
      returns a real AcgConceptImageJob with an ABSOLUTE image_url; a dict or a
      relative/empty url makes course_thumbnail SKIP, dropping it from the ==
      spine) + store_generated_image (inert for absolute urls, sealed anyway).
    - growth_bundle: WMClient → one `{text, cta_url}` block per channel key
      (handler._parse_snippets, requiring >=1 channel with copy).
    """
    stack = ExitStack()
    stack.enter_context(_lesson_downstream_mocks())
    p = lambda target, **kw: stack.enter_context(patch(target, **kw))  # noqa: E731

    p(
        "src.services.stages.course.outline.WMClient",
        return_value=_wm_client(
            {
                "sections": [{"title": "Intro", "objective": "Understand the basics"}],
                "intro_summary": "A short course on recursion.",
            }
        ),
    )
    p("src.services.stages.course.stitch.VideoCompileClient", return_value=_compile_mock())
    p(
        "src.services.stages.course.thumbnail.AIContentGenerationClient",
        return_value=_acg_image_mock(),
    )
    p(
        "src.services.stages.course.thumbnail.store_generated_image",
        new=AsyncMock(return_value="http://a/c.png"),
    )
    p(
        "src.services.stages.growth_bundle.handler.WMClient",
        return_value=_wm_client({"linkedin": {"text": "Check it out", "cta_url": None}}),
    )
    return stack


async def _create_course_with_lessons(
    db_session_factory,
    *,
    tenant: UUID,
    subject: UUID,
    n_lessons: int = 2,
) -> tuple[UUID, UUID]:
    """Create a course project + N lesson children under it, returning the
    course's (project_id, deliverable_id). The lesson children are what
    course_compile_lessons spawns child runs for; they are never dispatched
    directly."""
    course_id, course_deliverable_id = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="course",
        brief=_COURSE_BRIEF,
    )
    for _ in range(n_lessons):
        await _create_project(
            db_session_factory,
            tenant=tenant,
            subject=subject,
            type_="lesson",
            brief=_LESSON_BRIEF,
            parent_project_id=course_id,
        )
    return course_id, course_deliverable_id


# ---------------------------------------------------------------------------
# Happy path — full DAG (incl. child sub-DAGs) to completed
# ---------------------------------------------------------------------------


async def test_course_dag_runs_to_completed(db_session_factory, redis_client) -> None:
    """A plain course run executes its whole DAG to `completed`, spawns + awaits
    one full child lesson DAG per lesson project, the terminal state rolls up to
    the owning project, and the anti-false-green DAG-spine assertion holds (the
    completed-stage set is EXACTLY the non-opt-in course spine)."""
    from src.database.repositories.project_repository import ProjectRepository
    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository

    tenant, subject = uuid4(), uuid4()
    course_id, course_deliverable_id = await _create_course_with_lessons(
        db_session_factory, tenant=tenant, subject=subject, n_lessons=2
    )

    with _course_downstream_mocks():
        run_id, final = await _dispatch_and_wait_with_run_engine(
            db_session_factory,
            redis_client,
            deliverable_id=course_deliverable_id,
            tenant=tenant,
            subject=subject,
        )

    async with db_session_factory() as session:
        final = await RunRepository(session).get(run_id)
        proj_after = await ProjectRepository(session).get(course_id, tenant_id=tenant)
    assert final is not None
    assert final.status == "completed", final.error
    assert proj_after is not None
    assert proj_after.status == "completed"
    assert proj_after.finished_at is not None

    # Anti-false-green: prove the course DAG actually executed every spine stage.
    completed = await _completed_stage_names(db_session_factory, run_id)
    assert completed == _COURSE_SPINE, f"unexpected completed set: {completed ^ _COURSE_SPINE}"

    # Child sub-DAG proof: course_compile_lessons must have spawned N children,
    # each of which is itself a `completed` lesson run whose completed-stage set
    # is EXACTLY the lesson spine — i.e. the whole child DAG really ran too.
    async with db_session_factory() as session:
        so = await StageOutputRepository(session).get(run_id, "course_compile_lessons")
    assert so is not None
    assert so.output is not None
    lessons = so.output["lessons"]
    assert len(lessons) == 2, f"expected 2 child lessons, got {len(lessons)}"
    for lesson in lessons:
        child_id = UUID(lesson["lesson_run_id"])
        async with db_session_factory() as session:
            child = await RunRepository(session).get(child_id)
        assert child is not None
        assert child.status == "completed", child.error
        child_completed = await _completed_stage_names(db_session_factory, child_id)
        assert child_completed == _LESSON_SPINE, (
            f"child {child_id} spine mismatch: {child_completed ^ _LESSON_SPINE}"
        )

    # The execution lease is released once the run finishes.
    assert await redis_client.get(f"runexec:{run_id}") is None


# ---------------------------------------------------------------------------
# Negative path — a child-stage failure fails the whole course run
# ---------------------------------------------------------------------------


async def test_course_dag_fails_when_child_stage_errors(db_session_factory, redis_client) -> None:
    """If a child lesson's first WM stage (lesson_script) keeps raising, that
    child fails; `course_compile_lessons` awaits with `fail_on_any=True`, so the
    course run must reach `failed` with the error recorded — not hang, and not
    silently complete.

    course_outline uses the DISTINCT `course.outline.WMClient`, so the outline
    still succeeds and the children still spawn; only the children's
    `lesson_script` (primitives.script_stage.WMClient) boom. We keep the full
    `_course_downstream_mocks()` active (so children can even spawn) and layer
    the boom patch INSIDE it — the inner patch of the SAME target is applied last,
    so it wins over the lesson helper's benign script_stage.WMClient."""
    from src.database.repositories.run_repository import RunRepository

    tenant, subject = uuid4(), uuid4()
    _course_id, course_deliverable_id = await _create_course_with_lessons(
        db_session_factory, tenant=tenant, subject=subject, n_lessons=2
    )

    boom = AsyncMock()
    boom.message.side_effect = ValueError("synthetic child failure")
    with (
        _course_downstream_mocks(),
        patch("src.services.stages.primitives.script_stage.WMClient", return_value=boom),
    ):
        run_id, final = await _dispatch_and_wait_with_run_engine(
            db_session_factory,
            redis_client,
            deliverable_id=course_deliverable_id,
            tenant=tenant,
            subject=subject,
        )

    async with db_session_factory() as session:
        final = await RunRepository(session).get(run_id)
    assert final is not None
    assert final.status == "failed", f"a child failure must fail the course, got {final.status!r}"
    assert final.error, "the run must record the child error, not swallow it"
    # A failed run also releases its execution lease for a later retry.
    assert await redis_client.get(f"runexec:{run_id}") is None
