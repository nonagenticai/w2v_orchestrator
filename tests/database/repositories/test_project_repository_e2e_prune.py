"""``ProjectRepository.archive_stale_for_subject`` — the e2e retention sweep's only write.

Every assertion here is about something the sweep must NOT do. The sweep runs
unattended against a tenant that also holds the owner's ~95 real projects under
a different subject, so the interesting property is not "it archives" — it is
that subject, tenant, status and age each independently exclude a row, and that
a truncated batch truncates on the safe side (children before parents).
"""

from __future__ import annotations

import datetime as dt
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Project
from src.database.repositories.project_repository import ProjectRepository

pytestmark = pytest.mark.asyncio

_HOUR = dt.timedelta(hours=1)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


async def _mk(
    repo: ProjectRepository,
    *,
    tenant_id: UUID,
    subject_id: UUID,
    type_: str = "promo",
    title: str = "p",
    parent_project_id: UUID | None = None,
) -> Project:
    return await repo.create_with_deliverables(
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        type_=type_,
        title=title,
        brief={},
        brand={},
        voice={},
        theme={},
        visibility="tenant",
        parent_project_id=parent_project_id,
        deliverables=[
            {
                "name": "d0",
                "format": "16x9",
                "captions_mode": "none",
                "target_duration_seconds": 60,
            }
        ],
    )


async def _set_state(
    session: AsyncSession,
    project_id: UUID,
    *,
    status: str,
    age_hours: float,
) -> None:
    """Stamp status + ``updated_at`` directly.

    ``Project.updated_at`` carries ``onupdate=func.now()``, so a plain ORM write
    would drag every row back to "now" and make the age window untestable. An
    explicit value in ``.values()`` overrides the onupdate default, which is the
    only way to place a row on the far side of the cutoff.
    """
    await session.execute(
        update(Project)
        .where(Project.id == project_id)
        .values(status=status, updated_at=_now() - age_hours * _HOUR)
    )
    await session.flush()


async def _status_of(session: AsyncSession, project_id: UUID) -> str:
    return (
        await session.execute(select(Project.status).where(Project.id == project_id))
    ).scalar_one()


async def test_archives_a_stale_terminal_project_of_the_configured_subject(
    db_session: AsyncSession,
) -> None:
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    proj = await _mk(repo, tenant_id=tenant_id, subject_id=subject_id)
    await _set_state(db_session, proj.id, status="completed", age_hours=24)

    archived = await repo.archive_stale_for_subject(
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        older_than=_now() - 6 * _HOUR,
        limit=50,
    )

    assert archived == [proj.id]
    assert await _status_of(db_session, proj.id) == "archived"
    archived_at = (
        await db_session.execute(select(Project.archived_at).where(Project.id == proj.id))
    ).scalar_one()
    assert archived_at is not None


async def test_leaves_another_subjects_project_alone(db_session: AsyncSession) -> None:
    """THE load-bearing one: the owner's real projects share this tenant.

    Both rows are identical except for ``owner_subject_id`` — so this can only
    pass because the subject predicate is present, not because some other clause
    happened to exclude the row.
    """
    repo = ProjectRepository(db_session)
    tenant_id = uuid4()
    e2e_subject, owner_subject = uuid4(), uuid4()
    mine = await _mk(repo, tenant_id=tenant_id, subject_id=e2e_subject, title="e2e")
    theirs = await _mk(repo, tenant_id=tenant_id, subject_id=owner_subject, title="real")
    await _set_state(db_session, mine.id, status="completed", age_hours=24)
    await _set_state(db_session, theirs.id, status="completed", age_hours=24)

    archived = await repo.archive_stale_for_subject(
        tenant_id=tenant_id,
        owner_subject_id=e2e_subject,
        older_than=_now() - 6 * _HOUR,
        limit=50,
    )

    assert archived == [mine.id]
    assert await _status_of(db_session, theirs.id) == "completed"


async def test_leaves_another_tenants_project_alone(db_session: AsyncSession) -> None:
    """Same subject id, different tenant — the tenant predicate must still exclude it."""
    repo = ProjectRepository(db_session)
    e2e_tenant, other_tenant = uuid4(), uuid4()
    subject_id = uuid4()
    mine = await _mk(repo, tenant_id=e2e_tenant, subject_id=subject_id)
    theirs = await _mk(repo, tenant_id=other_tenant, subject_id=subject_id)
    await _set_state(db_session, mine.id, status="completed", age_hours=24)
    await _set_state(db_session, theirs.id, status="completed", age_hours=24)

    archived = await repo.archive_stale_for_subject(
        tenant_id=e2e_tenant,
        owner_subject_id=subject_id,
        older_than=_now() - 6 * _HOUR,
        limit=50,
    )

    assert archived == [mine.id]
    assert await _status_of(db_session, theirs.id) == "completed"


@pytest.mark.parametrize("live_status", ["drafting"])
async def test_leaves_a_non_terminal_projects_run_alone(
    db_session: AsyncSession, live_status: str
) -> None:
    """A project whose run is still working is never eligible, at any age.

    The row is aged well past the cutoff and owned by the swept subject in the
    swept tenant — status is the only thing keeping it, which is the point.
    """
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    live = await _mk(repo, tenant_id=tenant_id, subject_id=subject_id, title="live")
    done = await _mk(repo, tenant_id=tenant_id, subject_id=subject_id, title="done")
    await _set_state(db_session, live.id, status=live_status, age_hours=240)
    await _set_state(db_session, done.id, status="completed", age_hours=240)

    archived = await repo.archive_stale_for_subject(
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        older_than=_now() - 6 * _HOUR,
        limit=50,
    )

    assert archived == [done.id]
    assert await _status_of(db_session, live.id) == live_status


async def test_leaves_a_project_inside_the_age_window_alone(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    fresh = await _mk(repo, tenant_id=tenant_id, subject_id=subject_id, title="fresh")
    stale = await _mk(repo, tenant_id=tenant_id, subject_id=subject_id, title="stale")
    await _set_state(db_session, fresh.id, status="completed", age_hours=1)
    await _set_state(db_session, stale.id, status="completed", age_hours=24)

    archived = await repo.archive_stale_for_subject(
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        older_than=_now() - 6 * _HOUR,
        limit=50,
    )

    assert archived == [stale.id]
    assert await _status_of(db_session, fresh.id) == "completed"


async def test_archives_lesson_children_before_their_course_parent(
    db_session: AsyncSession,
) -> None:
    """Children-before-parents, asserted on the RETURNED ORDER.

    ``archive_stale_for_subject`` returns ids in the order it wrote them, so the
    ordering is observable rather than inferred. ``trg_lesson_parent_is_course``
    (migration 0015) is ``BEFORE INSERT OR UPDATE`` and the parent FK is
    ``ON DELETE SET NULL``: a hard delete of the course would raise on the
    surviving lesson's cascade. This sweep soft-archives, so it cannot trip that
    — the ordering is what keeps the invariant true for the next reuse.
    """
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    course = await _mk(
        repo, tenant_id=tenant_id, subject_id=subject_id, type_="course", title="course"
    )
    await db_session.flush()
    lesson_a = await _mk(
        repo,
        tenant_id=tenant_id,
        subject_id=subject_id,
        type_="lesson",
        title="lesson-a",
        parent_project_id=course.id,
    )
    lesson_b = await _mk(
        repo,
        tenant_id=tenant_id,
        subject_id=subject_id,
        type_="lesson",
        title="lesson-b",
        parent_project_id=course.id,
    )
    for pid in (course.id, lesson_a.id, lesson_b.id):
        await _set_state(db_session, pid, status="completed", age_hours=24)

    archived = await repo.archive_stale_for_subject(
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        older_than=_now() - 6 * _HOUR,
        limit=50,
    )

    assert set(archived) == {course.id, lesson_a.id, lesson_b.id}
    assert archived.index(course.id) > archived.index(lesson_a.id)
    assert archived.index(course.id) > archived.index(lesson_b.id)
    # And the parent really is last, not merely after one child.
    assert archived[-1] == course.id


async def test_a_capped_batch_truncates_after_the_children_not_the_parent(
    db_session: AsyncSession,
) -> None:
    """The cap must strand a listed COURSE, never orphaned lessons.

    With ``limit=2`` against a course + two lessons, children-first means the two
    lessons go and the course stays listed — coherent. The other order would
    leave two lessons in the listing under a parent that is gone, which is the
    clutter the sweep exists to remove.
    """
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    course = await _mk(
        repo, tenant_id=tenant_id, subject_id=subject_id, type_="course", title="course"
    )
    await db_session.flush()
    lessons = [
        await _mk(
            repo,
            tenant_id=tenant_id,
            subject_id=subject_id,
            type_="lesson",
            title=f"lesson-{i}",
            parent_project_id=course.id,
        )
        for i in range(2)
    ]
    for pid in (course.id, *[lesson.id for lesson in lessons]):
        await _set_state(db_session, pid, status="completed", age_hours=24)

    archived = await repo.archive_stale_for_subject(
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        older_than=_now() - 6 * _HOUR,
        limit=2,
    )

    assert len(archived) == 2
    assert set(archived) == {lesson.id for lesson in lessons}
    assert await _status_of(db_session, course.id) == "completed"


async def test_honours_the_limit_and_a_resweep_is_a_no_op(db_session: AsyncSession) -> None:
    """The cap bounds one pass, and already-archived rows are not re-swept.

    ``archived`` is excluded by the same status clause that selects the
    candidates, so the second call returns nothing rather than restamping
    ``archived_at`` on rows it already moved.
    """
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    made = [
        await _mk(repo, tenant_id=tenant_id, subject_id=subject_id, title=f"p{i}") for i in range(5)
    ]
    for proj in made:
        await _set_state(db_session, proj.id, status="completed", age_hours=24)

    first = await repo.archive_stale_for_subject(
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        older_than=_now() - 6 * _HOUR,
        limit=3,
    )
    assert len(first) == 3

    second = await repo.archive_stale_for_subject(
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        older_than=_now() - 6 * _HOUR,
        limit=3,
    )
    assert len(second) == 2
    assert set(second).isdisjoint(first)

    third = await repo.archive_stale_for_subject(
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        older_than=_now() - 6 * _HOUR,
        limit=3,
    )
    assert third == []


async def test_skips_storyboard_branch_rows(db_session: AsyncSession) -> None:
    """A branch is a hidden child no listing shows — not this sweep's clutter."""
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    normal = await _mk(repo, tenant_id=tenant_id, subject_id=subject_id, title="normal")
    branch = await _mk(repo, tenant_id=tenant_id, subject_id=subject_id, title="branch")
    await db_session.execute(update(Project).where(Project.id == branch.id).values(is_branch=True))
    await _set_state(db_session, normal.id, status="completed", age_hours=24)
    await _set_state(db_session, branch.id, status="completed", age_hours=24)

    archived = await repo.archive_stale_for_subject(
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        older_than=_now() - 6 * _HOUR,
        limit=50,
    )

    assert archived == [normal.id]
    assert await _status_of(db_session, branch.id) == "completed"


async def test_a_project_that_goes_live_between_select_and_update_is_not_archived(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The anti-race, enforced at WRITE time — not just in the candidate query.

    The SELECT and the UPDATE are separate statements, so a re-run can flip a
    project to ``producing`` in between. Archiving it then is PERMANENT damage,
    not a transient blip: ``recompute_status`` returns early on ``archived``
    (``project_repository.py``, the ``if proj.status == "archived"`` guard), so
    the project never climbs back out and the run's result is stranded.

    The race is simulated by monkeypatching the repository's own SELECT to flip
    the row to ``producing`` after choosing it — i.e. the candidate list is
    exactly what an un-raced sweep would produce, and only the write-time guard
    can save it.
    """
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    racer = await _mk(repo, tenant_id=tenant_id, subject_id=subject_id, title="racer")
    calm = await _mk(repo, tenant_id=tenant_id, subject_id=subject_id, title="calm")
    await _set_state(db_session, racer.id, status="completed", age_hours=24)
    await _set_state(db_session, calm.id, status="completed", age_hours=24)

    original_execute = db_session.execute
    flipped = False

    async def racing_execute(statement: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal flipped
        result = await original_execute(statement, *args, **kwargs)
        if not flipped and "SELECT" in str(statement).upper():
            flipped = True
            # A run starts on `racer` after the sweep picked it up.
            await original_execute(
                update(Project).where(Project.id == racer.id).values(status="producing")
            )
        return result

    monkeypatch.setattr(db_session, "execute", racing_execute)
    archived = await repo.archive_stale_for_subject(
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        older_than=_now() - 6 * _HOUR,
        limit=50,
    )
    monkeypatch.undo()

    # The live project survived...
    assert await _status_of(db_session, racer.id) == "producing"
    # ...and is not reported as archived, because the UPDATE matched nothing.
    assert racer.id not in archived
    # The uncontended sibling still went through — the guard is narrow, not a
    # blanket refusal to archive anything once a race is detected.
    assert calm.id in archived
    assert await _status_of(db_session, calm.id) == "archived"


async def test_foreign_owned_ids_reports_exactly_the_rows_outside_the_subject(
    db_session: AsyncSession,
) -> None:
    """The read half of the pre-commit post-condition."""
    repo = ProjectRepository(db_session)
    tenant_id = uuid4()
    e2e_subject, owner_subject = uuid4(), uuid4()
    mine = await _mk(repo, tenant_id=tenant_id, subject_id=e2e_subject, title="mine")
    theirs = await _mk(repo, tenant_id=tenant_id, subject_id=owner_subject, title="theirs")
    await db_session.flush()

    assert await repo.foreign_owned_ids([mine.id], owner_subject_id=e2e_subject) == []
    assert await repo.foreign_owned_ids([], owner_subject_id=e2e_subject) == []
    assert list(
        await repo.foreign_owned_ids([mine.id, theirs.id], owner_subject_id=e2e_subject)
    ) == [theirs.id]


async def test_the_update_re_asserts_the_subject_not_only_the_tenant(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write-time symmetry: the subject predicate is re-asserted on the UPDATE.

    The tenant and the status are both re-asserted at write time; leaving the
    SUBJECT out — the predicate carrying the entire safety argument — is the same
    asymmetry that allowed the mid-run archive race. Nothing in ``src/`` rewrites
    ``owner_subject_id`` today, so the race is simulated the same way the status
    race is: flip the column after the candidate SELECT has chosen the row.
    """
    repo = ProjectRepository(db_session)
    tenant_id, subject_id, other_subject = uuid4(), uuid4(), uuid4()
    racer = await _mk(repo, tenant_id=tenant_id, subject_id=subject_id, title="racer")
    await _set_state(db_session, racer.id, status="completed", age_hours=24)

    original_execute = db_session.execute
    flipped = False

    async def racing_execute(statement: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal flipped
        result = await original_execute(statement, *args, **kwargs)
        if not flipped and "SELECT" in str(statement).upper():
            flipped = True
            await original_execute(
                update(Project).where(Project.id == racer.id).values(owner_subject_id=other_subject)
            )
        return result

    monkeypatch.setattr(db_session, "execute", racing_execute)
    archived = await repo.archive_stale_for_subject(
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        older_than=_now() - 6 * _HOUR,
        limit=50,
    )
    monkeypatch.undo()

    assert archived == []
    assert await _status_of(db_session, racer.id) == "completed"


async def test_foreign_owned_ids_is_not_null_blind(db_session: AsyncSession) -> None:
    """A NULL owner must be REPORTED as foreign, not silently dropped.

    Plain SQL ``!=`` evaluates to UNKNOWN against NULL, so a NULL-owned row would
    fall out of the result and be reported as in-scope — the one row least
    provably in scope reading as clean. ``owner_subject_id`` is NOT NULL, so the
    NULL is injected by dropping the constraint for this transaction; the point
    is that the comparison itself is total, independent of the schema holding.
    """
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    orphan = await _mk(repo, tenant_id=tenant_id, subject_id=subject_id, title="orphan")
    await db_session.flush()
    await db_session.execute(
        text("ALTER TABLE projects ALTER COLUMN owner_subject_id DROP NOT NULL")
    )
    await db_session.execute(
        update(Project).where(Project.id == orphan.id).values(owner_subject_id=None)
    )
    await db_session.flush()

    foreign = await repo.foreign_owned_ids([orphan.id], owner_subject_id=subject_id)

    assert list(foreign) == [orphan.id]
