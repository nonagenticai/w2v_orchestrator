"""Sub-DAG orphaned-child cancellation — the PAID-render leak fix.

`course_compile_lessons` spawns one independent lesson RUN per lesson project via
`RunEngine.start_run(..., parent_run_id=None)` and tracks them ONLY in its stage
output's `children_by_deliverable` map (they carry no `parent_run_id` FK). So when
the parent course run terminalizes `failed` — or a user cancels it — the FK-based
descendant cascade (`RunRepository.list_active_children` /
`_sweep_active_descendants`) cannot see them, and each child keeps spending on its
own full lesson DAG until it finishes.

These drive the fix end to end against a real DB (+ fakeredis): the dispatcher's
failure-path `_cancel_subdag_children` and the route's `_sweep_deliverable_children`,
plus the shared `subdag_child_run_ids` collector. Same dependency class as the
other `tests/integration/*` in-process tests, so they run in default per-PR CI.
"""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

from src.database.repositories.run_repository import cancelled_by_parent
from tests.integration._dispatch_harness import _create_project


async def _spawn_pending_children(
    db_session_factory,
    *,
    tenant: UUID,
    subject: UUID,
    course_id: UUID,
    n: int = 2,
) -> list[UUID]:
    """Create N lesson deliverables under a course and spawn one NON-terminal
    child run per lesson (dispatch is a no-op, so each stays `pending`). Returns
    the child run ids — the PAID renders the fix must reap."""
    from src.services.blueprints.registry import get_registry
    from src.services.run_engine import RunEngine
    from tests.integration.test_in_process_lesson import _LESSON_BRIEF

    engine = RunEngine(
        session_factory=db_session_factory,
        registry=get_registry(),
        dispatch=lambda rid: asyncio.sleep(0),
    )
    child_ids: list[UUID] = []
    for _ in range(n):
        _pid, lesson_deliverable = await _create_project(
            db_session_factory,
            tenant=tenant,
            subject=subject,
            type_="lesson",
            brief=_LESSON_BRIEF,
            parent_project_id=course_id,
        )
        child = await engine.start_run(
            deliverable_id=lesson_deliverable,
            tenant_id=tenant,
            subject_id=subject,
            parent_run_id=None,
            pause_after_stages=[],
        )
        child_ids.append(child.id)
    return child_ids


async def _record_children_on_parent(
    db_session_factory, *, tenant: UUID, parent_run_id: UUID, children: list[UUID]
) -> None:
    """Write the `children_by_deliverable` bookkeeping onto the parent's
    `course_compile_lessons` stage output exactly as the stage does at spawn."""
    from src.database.repositories.stage_output_repository import StageOutputRepository
    from src.database.session import bind_session_tenant

    async with db_session_factory() as session:
        await bind_session_tenant(session, tenant)
        recorded = {f"deliverable-{i}": str(cid) for i, cid in enumerate(children)}
        await StageOutputRepository(session).record_progress(
            parent_run_id, "course_compile_lessons", {"children_by_deliverable": recorded}
        )
        await session.commit()


async def _make_course_parent(
    db_session_factory, *, tenant: UUID, subject: UUID
) -> tuple[UUID, UUID]:
    """Create a course project + a NON-terminal course parent run. Returns
    (course_project_id, parent_run_id)."""
    from src.services.blueprints.registry import get_registry
    from src.services.run_engine import RunEngine

    course_id, course_deliverable_id = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="course",
        brief={
            "audience": "beginners",
            "outcome": "master recursion",
            "call_to_action": "Enroll now.",
            "target_duration_seconds": 30,
        },
    )
    engine = RunEngine(
        session_factory=db_session_factory,
        registry=get_registry(),
        dispatch=lambda rid: asyncio.sleep(0),
    )
    parent = await engine.start_run(
        deliverable_id=course_deliverable_id,
        tenant_id=tenant,
        subject_id=subject,
        parent_run_id=None,
        pause_after_stages=[],
    )
    return course_id, parent.id


# ---------------------------------------------------------------------------
# Dispatcher failure-path: _cancel_subdag_children
# ---------------------------------------------------------------------------


async def test_failed_parent_cancels_orphaned_course_children(
    db_session_factory, redis_client
) -> None:
    """The money-leak fix: a course parent that terminalizes must cancel its
    in-flight PAID child renders (parent_run_id=None → invisible to the FK
    cascade). Both children must end `cancelled`, and the method reports it acted
    on exactly 2."""
    from src.database.repositories.run_repository import RunRepository
    from src.database.session import bind_run_tenant
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher

    tenant, subject = uuid4(), uuid4()
    course_id, parent_id = await _make_course_parent(
        db_session_factory, tenant=tenant, subject=subject
    )
    children = await _spawn_pending_children(
        db_session_factory, tenant=tenant, subject=subject, course_id=course_id, n=2
    )
    await _record_children_on_parent(
        db_session_factory, tenant=tenant, parent_run_id=parent_id, children=children
    )

    dispatcher = RunDispatcher(
        session_factory=db_session_factory, registry=get_registry(), redis=redis_client
    )
    async with db_session_factory() as session:
        await bind_run_tenant(session, parent_id)
        acted = await dispatcher._cancel_subdag_children(session, parent_id)
        await session.commit()

    assert acted == 2, acted
    async with db_session_factory() as session:
        for cid in children:
            child = await RunRepository(session).get(cid)
            assert child is not None
            assert child.status == "cancelled", (cid, child.status)
            # Provenance: a course retry may re-spawn a CASCADE-cancelled lesson.
            assert cancelled_by_parent(child), child.failure_detail


async def test_cancel_subdag_children_skips_terminal_child(
    db_session_factory, redis_client
) -> None:
    """An already-terminal child is not re-cancelled (idempotent): only the still
    non-terminal child is acted on."""
    from src.database.repositories.run_repository import RunRepository
    from src.database.session import bind_run_tenant
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher

    tenant, subject = uuid4(), uuid4()
    course_id, parent_id = await _make_course_parent(
        db_session_factory, tenant=tenant, subject=subject
    )
    children = await _spawn_pending_children(
        db_session_factory, tenant=tenant, subject=subject, course_id=course_id, n=2
    )
    # Complete one child up front — it must be left alone by the sweep.
    async with db_session_factory() as session:
        await bind_run_tenant(session, children[0])
        await RunRepository(session).mark_completed(children[0])
        await session.commit()
    await _record_children_on_parent(
        db_session_factory, tenant=tenant, parent_run_id=parent_id, children=children
    )

    dispatcher = RunDispatcher(
        session_factory=db_session_factory, registry=get_registry(), redis=redis_client
    )
    async with db_session_factory() as session:
        await bind_run_tenant(session, parent_id)
        acted = await dispatcher._cancel_subdag_children(session, parent_id)
        await session.commit()

    assert acted == 1, acted
    async with db_session_factory() as session:
        c0 = await RunRepository(session).get(children[0])
        c1 = await RunRepository(session).get(children[1])
    assert c0 is not None and c0.status == "completed", c0.status if c0 else None
    assert c1 is not None and c1.status == "cancelled", c1.status if c1 else None


async def test_cancel_subdag_children_noop_on_non_course_run(
    db_session_factory, redis_client
) -> None:
    """A run with no `course_compile_lessons` stage output (any non-course run)
    yields 0 and never raises."""
    from src.database.session import bind_run_tenant
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher
    from src.services.run_engine import RunEngine

    tenant, subject = uuid4(), uuid4()
    _pid, deliverable_id = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="generic",
        brief={"angle": "growth", "audience": "saas", "target_duration_seconds": 30},
    )
    engine = RunEngine(
        session_factory=db_session_factory,
        registry=get_registry(),
        dispatch=lambda rid: asyncio.sleep(0),
    )
    run = await engine.start_run(
        deliverable_id=deliverable_id,
        tenant_id=tenant,
        subject_id=subject,
        parent_run_id=None,
        pause_after_stages=[],
    )
    dispatcher = RunDispatcher(
        session_factory=db_session_factory, registry=get_registry(), redis=redis_client
    )
    async with db_session_factory() as session:
        await bind_run_tenant(session, run.id)
        acted = await dispatcher._cancel_subdag_children(session, run.id)
    assert acted == 0


# ---------------------------------------------------------------------------
# Route user-cancel path: _sweep_deliverable_children
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shared collector: subdag_child_run_ids
# ---------------------------------------------------------------------------


async def test_subdag_child_run_ids_reads_map(db_session_factory) -> None:
    """Returns the child run ids from `children_by_deliverable`."""
    from src.database.session import bind_run_tenant
    from src.services.runs.sub_dag import subdag_child_run_ids

    tenant, subject = uuid4(), uuid4()
    course_id, parent_id = await _make_course_parent(
        db_session_factory, tenant=tenant, subject=subject
    )
    children = await _spawn_pending_children(
        db_session_factory, tenant=tenant, subject=subject, course_id=course_id, n=2
    )
    await _record_children_on_parent(
        db_session_factory, tenant=tenant, parent_run_id=parent_id, children=children
    )
    async with db_session_factory() as session:
        await bind_run_tenant(session, parent_id)
        ids = await subdag_child_run_ids(session, parent_id)
    assert set(ids) == set(children)


async def test_subdag_child_run_ids_empty_without_children(db_session_factory) -> None:
    """A course parent whose compile stage recorded nothing yet → [] (no crash)."""
    from src.database.session import bind_run_tenant
    from src.services.runs.sub_dag import subdag_child_run_ids

    tenant, subject = uuid4(), uuid4()
    _course_id, parent_id = await _make_course_parent(
        db_session_factory, tenant=tenant, subject=subject
    )
    async with db_session_factory() as session:
        await bind_run_tenant(session, parent_id)
        ids = await subdag_child_run_ids(session, parent_id)
    assert ids == []


# ---------------------------------------------------------------------------
# Compile-cancel propagation: the render must stop too, not just the DAG
# ---------------------------------------------------------------------------
#
# Cancelling a run stopped the ORCHESTRATOR only — the compile it had already
# submitted kept rendering on video_compiler_service and was billed in full.
# `cancel_compiles_for_run` is the missing edge; these prove it is actually WIRED
# into both cascade sites. Mutation-check: delete either `await
# cancel_compiles_for_run(...)` call and the matching test below fails.


async def test_subdag_child_cancel_also_cancels_the_childs_compile(
    db_session_factory, redis_client, monkeypatch
) -> None:
    """A course fans out one PAID render per lesson. Cancelling the child run must also
    reach the compiler for that child — one call per child acted on, keyed by the CHILD's
    run_id (the parent's would reap nothing: the compile was submitted under the child)."""
    from src.database.session import bind_run_tenant
    from src.services import run_dispatcher as rd
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher

    tenant, subject = uuid4(), uuid4()
    course_id, parent_id = await _make_course_parent(
        db_session_factory, tenant=tenant, subject=subject
    )
    children = await _spawn_pending_children(
        db_session_factory, tenant=tenant, subject=subject, course_id=course_id, n=2
    )
    await _record_children_on_parent(
        db_session_factory, tenant=tenant, parent_run_id=parent_id, children=children
    )

    propagated: list[UUID] = []

    async def _spy(run_id) -> int:
        propagated.append(run_id)
        return 1

    monkeypatch.setattr(rd, "cancel_compiles_for_run", _spy)

    dispatcher = RunDispatcher(
        session_factory=db_session_factory, registry=get_registry(), redis=redis_client
    )
    async with db_session_factory() as session:
        await bind_run_tenant(session, parent_id)
        await dispatcher._cancel_subdag_children(session, parent_id)
        await session.commit()

    assert set(propagated) == set(children), propagated


async def test_subdag_child_cancel_survives_a_dead_compiler(
    db_session_factory, redis_client, monkeypatch
) -> None:
    """A compile-cancel that blows up must not abort the parent's terminalization, nor
    skip the remaining children. The compiler being down cannot be allowed to wedge a
    course run mid-cascade."""
    from src.database.repositories.run_repository import RunRepository
    from src.database.session import bind_run_tenant
    from src.services import run_dispatcher as rd
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher

    tenant, subject = uuid4(), uuid4()
    course_id, parent_id = await _make_course_parent(
        db_session_factory, tenant=tenant, subject=subject
    )
    children = await _spawn_pending_children(
        db_session_factory, tenant=tenant, subject=subject, course_id=course_id, n=2
    )
    await _record_children_on_parent(
        db_session_factory, tenant=tenant, parent_run_id=parent_id, children=children
    )

    async def _explodes(run_id) -> int:
        raise RuntimeError("compiler unreachable")

    monkeypatch.setattr(rd, "cancel_compiles_for_run", _explodes)

    dispatcher = RunDispatcher(
        session_factory=db_session_factory, registry=get_registry(), redis=redis_client
    )
    async with db_session_factory() as session:
        await bind_run_tenant(session, parent_id)
        acted = await dispatcher._cancel_subdag_children(session, parent_id)
        await session.commit()

    assert acted == 2, acted
    async with db_session_factory() as session:
        for cid in children:
            child = await RunRepository(session).get(cid)
            assert child is not None
            assert child.status == "cancelled", (cid, child.status)


async def test_cancel_one_propagates_to_the_compiler_on_both_branches(
    db_session_factory, redis_client, monkeypatch
) -> None:
    """`_cancel_one` is the route's per-run cancel. Whether the run's task is owned by
    THIS pod (dispatcher.cancel → True) or not (the mark-cancelled fallback), the compile
    is in flight either way — so the propagation must fire on BOTH branches."""
    from src.api.routes.runs import lifecycle
    from src.database.repositories.run_repository import RunRepository
    from src.database.session import bind_run_tenant
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher

    tenant, subject = uuid4(), uuid4()
    _course_id, run_id = await _make_course_parent(
        db_session_factory, tenant=tenant, subject=subject
    )

    propagated: list[UUID] = []

    async def _spy(rid) -> int:
        propagated.append(rid)
        return 0

    monkeypatch.setattr(lifecycle, "cancel_compiles_for_run", _spy)

    # A REAL dispatcher whose task-ownership answer is forced, rather than a stand-in:
    # `_cancel_one` branches on `dispatcher.cancel`, and the two branches must both
    # propagate.
    dispatcher = RunDispatcher(
        session_factory=db_session_factory, registry=get_registry(), redis=redis_client
    )

    for owns in (True, False):
        propagated.clear()

        async def _cancel(_run_id, _owns=owns) -> bool:
            return _owns

        monkeypatch.setattr(dispatcher, "cancel", _cancel)
        async with db_session_factory() as session:
            await bind_run_tenant(session, run_id)
            run = await RunRepository(session).get(run_id)
            assert run is not None
            await lifecycle._cancel_one(session, dispatcher, run)
        assert propagated == [run_id], (owns, propagated)
