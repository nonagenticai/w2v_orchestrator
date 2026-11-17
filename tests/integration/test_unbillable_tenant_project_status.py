"""A run REFUSED for an unbillable tenant must still release its project.

The dispatch-time tenant guard (``src/auth/run_tenant.py::billable_tenant_id``)
turns a legacy nil-uuid ``config_snapshot`` into a terminal ``failed`` run
instead of unattributable spend. That is only half a terminal transition. Every
other terminal write in ``run_dispatcher`` pairs ``mark_*`` with
``_recompute_project`` before the commit, because ``Project.status`` is a stored
read model that nothing else re-derives: ``ProjectRepository.recompute_status``
has exactly two callers estate-wide (the dispatcher, and
``api/routes/runs/_shared.py`` on the HTTP lifecycle routes). A refused run is
terminal, so no later transition will ever come — without the recompute the
owning project, and its course parent, stay ``producing`` **permanently**.

That is strictly worse than the billing defect the guard fixes: it fires on
precisely the legacy nil-uuid rows the guard targets, so shipping the guard
alone would convert a silent mis-invoice into a stuck project on the same rows.

This test asserts the STORED STATUSES either side of the dispatch — not that
``_recompute_project`` was called. Deleting the recompute leaves both rows at
``producing`` and this goes red; a mock-call assertion would still pass against
a recompute that derived the wrong value.

Real Postgres, no live containers (same rationale as
``tests/integration/test_drain_stranded_runs.py``): it runs in default per-PR CI.
"""

from __future__ import annotations

import asyncio
import json
from uuid import UUID, uuid7

from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository
from tests.integration._dispatch_harness import _create_project

#: The pre-0.67 ``auth.default_tenant_id`` fallback. A valid UUID, which is why
#: it type-checked its way onto the wire; ``billable_tenant_id`` refuses it.
NIL_TENANT = "00000000-0000-0000-0000-000000000000"


async def _seed_course_with_producing_lesson_run(
    db_session_factory,
    *,
    tenant: UUID,
    subject: UUID,
    snapshot_tenant: str | None,
) -> tuple[UUID, UUID, UUID]:
    """Course ⟶ lesson ⟶ deliverable ⟶ one pending run, both projects
    recomputed to ``producing``. Returns (course_id, lesson_id, run_id).

    ``snapshot_tenant`` is what lands in ``config_snapshot["tenant_id"]`` — the
    only thing under test. The PROJECT rows carry a real tenant either way,
    which is the legacy shape: the row was written while the nil uuid was the
    configured fallback for the snapshot, not for the project.
    """
    course_id, _course_deliverable = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="course",
        brief={"topic": "unbillable-tenant-course"},
    )
    lesson_id, deliverable_id = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="lesson",
        brief={"topic": "unbillable-tenant-lesson"},
        parent_project_id=course_id,
    )

    snapshot: dict = {"subject_id": str(subject), "license_id": "lic-1"}
    if snapshot_tenant is not None:
        snapshot["tenant_id"] = snapshot_tenant

    async with db_session_factory() as session:
        run = await RunRepository(session).create(
            deliverable_id=deliverable_id,
            blueprint_id="generic_v1",
            config_snapshot=snapshot,
            pause_after_stages=[],
        )
        repo = ProjectRepository(session)
        # A `pending` run is an ACTIVE unit, so both derive `producing` — the
        # state a stuck project would never leave.
        lesson_status = await repo.recompute_status(lesson_id)
        course_status = await repo.recompute_status(course_id)
        await session.commit()

    assert lesson_status == "producing", lesson_status
    assert course_status == "producing", course_status
    return course_id, lesson_id, run.id


async def _dispatch_to_terminal(db_session_factory, redis_client, run_id: UUID, tenant: UUID):
    """Drive ONE run through a real ``RunDispatcher`` and poll it to terminal."""
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher

    dispatcher = RunDispatcher(
        session_factory=db_session_factory,
        registry=get_registry(),
        redis=redis_client,
    )
    try:
        await dispatcher.dispatch(run_id)
        last = None
        for _ in range(50):
            await asyncio.sleep(0.05)
            async with db_session_factory() as session:
                last = await RunRepository(session).get(run_id)
            if last is not None and last.status in ("completed", "failed", "cancelled", "paused"):
                break
    finally:
        await dispatcher.shutdown()
    return last


async def _statuses(db_session_factory, *, tenant: UUID, course_id: UUID, lesson_id: UUID):
    async with db_session_factory() as session:
        repo = ProjectRepository(session)
        course = await repo.get(course_id, tenant_id=tenant)
        lesson = await repo.get(lesson_id, tenant_id=tenant)
    assert course is not None and lesson is not None
    return course.status, lesson.status


async def test_nil_tenant_refusal_moves_the_lesson_and_its_course_off_producing(
    db_session_factory, redis_client
) -> None:
    """The nil-uuid legacy row: refused at dispatch, and the read model follows.

    Both stored statuses must leave ``producing``. ``_derive_status`` maps "a
    unit failed and none are active" to ``failed``, and the course rolls its
    lesson child up as one such unit, so both land on ``failed``.
    """
    tenant, subject = uuid7(), uuid7()
    course_id, lesson_id, run_id = await _seed_course_with_producing_lesson_run(
        db_session_factory, tenant=tenant, subject=subject, snapshot_tenant=NIL_TENANT
    )

    final = await _dispatch_to_terminal(db_session_factory, redis_client, run_id, tenant)

    assert final is not None
    assert final.status == "failed", f"the refusal must terminalise the run, got {final.status!r}"
    assert json.loads(final.error)["reason"] == "unbillable_tenant"

    course_status, lesson_status = await _statuses(
        db_session_factory, tenant=tenant, course_id=course_id, lesson_id=lesson_id
    )
    assert lesson_status != "producing", (
        "the refused run is TERMINAL — nothing will ever transition this project "
        "again, so leaving it 'producing' strands it permanently"
    )
    assert lesson_status == "failed", lesson_status
    assert course_status != "producing", (
        "the course rolls its lesson child up as one unit; a stuck lesson pins the "
        "whole course at 'producing' too"
    )
    assert course_status == "failed", course_status


async def test_absent_tenant_refusal_also_releases_the_project(
    db_session_factory, redis_client
) -> None:
    """The quieter refusal lane (no ``tenant_id`` key at all) takes the same
    terminal path, so it must do the same recompute. Pinned separately because
    a fix applied to only one refusal branch would still leave this one stuck.
    """
    tenant, subject = uuid7(), uuid7()
    course_id, lesson_id, run_id = await _seed_course_with_producing_lesson_run(
        db_session_factory, tenant=tenant, subject=subject, snapshot_tenant=None
    )

    final = await _dispatch_to_terminal(db_session_factory, redis_client, run_id, tenant)

    assert final is not None
    assert final.status == "failed"
    assert json.loads(final.error)["reason"] == "unbillable_tenant"

    course_status, lesson_status = await _statuses(
        db_session_factory, tenant=tenant, course_id=course_id, lesson_id=lesson_id
    )
    assert (lesson_status, course_status) == ("failed", "failed")
