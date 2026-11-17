from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Project, StageOutput
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository

pytestmark = pytest.mark.asyncio


async def _mk_project(
    repo: ProjectRepository,
    *,
    tenant_id: UUID,
    subject_id: UUID,
    type_: str = "promo",
    title: str = "p",
    n_deliv: int = 1,
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
                "name": f"d{i}",
                "format": "16x9",
                "captions_mode": "none",
                "target_duration_seconds": 60,
            }
            for i in range(n_deliv)
        ],
    )


async def _add_run(session: AsyncSession, deliverable_id: UUID, *, mark: str | None = None) -> UUID:
    rr = RunRepository(session)
    run = await rr.create(
        deliverable_id=deliverable_id,
        blueprint_id="promo.v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    if mark == "started":
        await rr.mark_started(run.id)
    elif mark == "completed":
        await rr.mark_completed(run.id)
    elif mark == "failed":
        await rr.mark_failed(run.id, "boom")
    elif mark in ("checkpoint", "manual"):
        await rr.mark_paused(run.id, reason=mark)
    return run.id


async def _add_stage_output(
    session: AsyncSession,
    run_id: UUID,
    stage_name: str,
    output: dict[str, object],
    *,
    status: str = "completed",
) -> None:
    session.add(
        StageOutput(id=uuid4(), run_id=run_id, stage_name=stage_name, status=status, output=output)
    )


async def _status_finished(session: AsyncSession, project_id: UUID) -> tuple[str, object]:
    row = (
        await session.execute(
            select(Project.status, Project.finished_at).where(Project.id == project_id)
        )
    ).one()
    return row.status, row.finished_at


async def test_create_with_deliverables_atomic(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    tenant_id = uuid4()
    subject_id = uuid4()
    proj = await repo.create_with_deliverables(
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        type_="promo",
        title="My promo",
        brief={"angle": "growth"},
        brand={},
        voice={},
        theme={},
        visibility="tenant",
        deliverables=[
            {
                "name": "16x9 hero",
                "format": "16x9",
                "captions_mode": "none",
                "target_duration_seconds": 60,
            },
            {
                "name": "9x16 vert",
                "format": "9x16",
                "captions_mode": "burned_in",
                "target_duration_seconds": 30,
            },
        ],
    )
    await db_session.commit()
    assert proj.id is not None
    fresh = await repo.get(proj.id, tenant_id=tenant_id)
    assert fresh is not None
    assert len(fresh.deliverables) == 2
    assert {d.format for d in fresh.deliverables} == {"16x9", "9x16"}


async def test_lesson_without_course_parent_raises(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    tenant_id = uuid4()
    subject_id = uuid4()
    with pytest.raises(Exception) as exc:
        await repo.create_with_deliverables(
            tenant_id=tenant_id,
            owner_subject_id=subject_id,
            type_="lesson",
            title="orphan lesson",
            brief={},
            brand={},
            voice={},
            theme={},
            visibility="tenant",
            parent_project_id=None,
            deliverables=[
                {
                    "name": "x",
                    "format": "16x9",
                    "captions_mode": "none",
                    "target_duration_seconds": 60,
                }
            ],
        )
        await db_session.commit()
    assert "parent_project_id" in str(exc.value)


async def test_soft_delete_excludes_from_default_list(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    tenant_id = uuid4()
    subject_id = uuid4()
    proj = await repo.create_with_deliverables(
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        type_="generic",
        title="x",
        brief={},
        brand={},
        voice={},
        theme={},
        visibility="tenant",
        deliverables=[
            {"name": "x", "format": "16x9", "captions_mode": "none", "target_duration_seconds": 60}
        ],
    )
    await db_session.commit()
    await repo.soft_delete(proj.id, tenant_id=tenant_id)
    await db_session.commit()
    listed = await repo.list(tenant_id=tenant_id, include_archived=False, limit=50, offset=0)
    assert proj.id not in [p.id for p in listed.items]
    listed_all = await repo.list(tenant_id=tenant_id, include_archived=True, limit=50, offset=0)
    assert proj.id in [p.id for p in listed_all.items]


async def test_cross_tenant_get_returns_none(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    t1 = uuid4()
    t2 = uuid4()
    s = uuid4()
    proj = await repo.create_with_deliverables(
        tenant_id=t1,
        owner_subject_id=s,
        type_="generic",
        title="x",
        brief={},
        brand={},
        voice={},
        theme={},
        visibility="tenant",
        deliverables=[
            {"name": "x", "format": "16x9", "captions_mode": "none", "target_duration_seconds": 60}
        ],
    )
    await db_session.commit()
    fetched = await repo.get(proj.id, tenant_id=t2)
    assert fetched is None


# ─── recompute_status ────────────────────────────────────────────────────


async def test_recompute_status_drafting_when_no_runs(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    proj = await _mk_project(repo, tenant_id=tenant_id, subject_id=subject_id)
    await db_session.commit()
    assert await repo.recompute_status(proj.id) == "drafting"


async def test_recompute_status_producing_when_run_active(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    proj = await _mk_project(repo, tenant_id=tenant_id, subject_id=subject_id)
    await db_session.commit()
    fresh = await repo.get(proj.id, tenant_id=tenant_id)
    assert fresh is not None
    await _add_run(db_session, fresh.deliverables[0].id, mark="started")
    await db_session.commit()
    assert await repo.recompute_status(proj.id) == "producing"
    status, _ = await _status_finished(db_session, proj.id)
    assert status == "producing"


async def test_recompute_status_completed_sets_finished_at(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    proj = await _mk_project(repo, tenant_id=tenant_id, subject_id=subject_id, n_deliv=2)
    await db_session.commit()
    fresh = await repo.get(proj.id, tenant_id=tenant_id)
    assert fresh is not None
    for d in fresh.deliverables:
        await _add_run(db_session, d.id, mark="completed")
    await db_session.commit()
    assert await repo.recompute_status(proj.id) == "completed"
    status, finished_at = await _status_finished(db_session, proj.id)
    assert status == "completed"
    assert finished_at is not None


async def test_recompute_status_producing_when_one_deliverable_unran(
    db_session: AsyncSession,
) -> None:
    # Two deliverables, only one completed → not all done, none active → producing.
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    proj = await _mk_project(repo, tenant_id=tenant_id, subject_id=subject_id, n_deliv=2)
    await db_session.commit()
    fresh = await repo.get(proj.id, tenant_id=tenant_id)
    assert fresh is not None
    await _add_run(db_session, fresh.deliverables[0].id, mark="completed")
    await db_session.commit()
    assert await repo.recompute_status(proj.id) == "producing"


async def test_recompute_status_failed_when_run_failed_none_active(
    db_session: AsyncSession,
) -> None:
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    proj = await _mk_project(repo, tenant_id=tenant_id, subject_id=subject_id)
    await db_session.commit()
    fresh = await repo.get(proj.id, tenant_id=tenant_id)
    assert fresh is not None
    await _add_run(db_session, fresh.deliverables[0].id, mark="failed")
    await db_session.commit()
    assert await repo.recompute_status(proj.id) == "failed"


async def test_recompute_status_idempotent(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    proj = await _mk_project(repo, tenant_id=tenant_id, subject_id=subject_id)
    await db_session.commit()
    fresh = await repo.get(proj.id, tenant_id=tenant_id)
    assert fresh is not None
    await _add_run(db_session, fresh.deliverables[0].id, mark="completed")
    await db_session.commit()
    assert await repo.recompute_status(proj.id) == "completed"
    _, first_finished = await _status_finished(db_session, proj.id)
    assert await repo.recompute_status(proj.id) == "completed"
    _, second_finished = await _status_finished(db_session, proj.id)
    assert first_finished == second_finished  # not bumped on the no-op re-run


async def test_recompute_status_latest_failed_run_clears_finished_at(
    db_session: AsyncSession,
) -> None:
    # Complete (sets finished_at), then a newer failed run flips to failed and
    # clears finished_at.
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    proj = await _mk_project(repo, tenant_id=tenant_id, subject_id=subject_id)
    await db_session.commit()
    fresh = await repo.get(proj.id, tenant_id=tenant_id)
    assert fresh is not None
    deliv_id = fresh.deliverables[0].id
    await _add_run(db_session, deliv_id, mark="completed")
    await db_session.commit()
    assert await repo.recompute_status(proj.id) == "completed"
    await _add_run(db_session, deliv_id, mark="failed")
    await db_session.commit()
    assert await repo.recompute_status(proj.id) == "failed"
    status, finished_at = await _status_finished(db_session, proj.id)
    assert status == "failed"
    assert finished_at is None


async def test_recompute_status_leaves_archived_untouched(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    proj = await _mk_project(repo, tenant_id=tenant_id, subject_id=subject_id)
    await db_session.commit()
    await repo.soft_delete(proj.id, tenant_id=tenant_id)
    await db_session.commit()
    assert await repo.recompute_status(proj.id) == "archived"
    status, _ = await _status_finished(db_session, proj.id)
    assert status == "archived"


# ─── parent / course rollup ──────────────────────────────────────────────


async def _mk_course_with_lessons(
    repo: ProjectRepository, db_session: AsyncSession, *, tenant_id: UUID, subject_id: UUID, n: int
) -> tuple[Project, list[Project]]:
    course = await _mk_project(
        repo, tenant_id=tenant_id, subject_id=subject_id, type_="course", title="course"
    )
    await db_session.commit()
    lessons = []
    for i in range(n):
        lesson = await _mk_project(
            repo,
            tenant_id=tenant_id,
            subject_id=subject_id,
            type_="lesson",
            title=f"lesson {i}",
            parent_project_id=course.id,
        )
        lessons.append(lesson)
    await db_session.commit()
    return course, lessons


async def test_recompute_parent_completed_when_own_run_and_children_done(
    db_session: AsyncSession,
) -> None:
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    course, lessons = await _mk_course_with_lessons(
        repo, db_session, tenant_id=tenant_id, subject_id=subject_id, n=2
    )
    # Complete the course's own deliverable run + every lesson's run.
    fresh_course = await repo.get(course.id, tenant_id=tenant_id)
    assert fresh_course is not None
    await _add_run(db_session, fresh_course.deliverables[0].id, mark="completed")
    for lesson in lessons:
        fresh = await repo.get(lesson.id, tenant_id=tenant_id)
        assert fresh is not None
        await _add_run(db_session, fresh.deliverables[0].id, mark="completed")
    await db_session.commit()
    for lesson in lessons:
        await repo.recompute_status(lesson.id)
    assert await repo.recompute_status(course.id) == "completed"


async def test_recompute_parent_failed_when_a_lesson_failed(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    course, lessons = await _mk_course_with_lessons(
        repo, db_session, tenant_id=tenant_id, subject_id=subject_id, n=2
    )
    fresh_course = await repo.get(course.id, tenant_id=tenant_id)
    assert fresh_course is not None
    await _add_run(db_session, fresh_course.deliverables[0].id, mark="completed")
    fresh0 = await repo.get(lessons[0].id, tenant_id=tenant_id)
    fresh1 = await repo.get(lessons[1].id, tenant_id=tenant_id)
    assert fresh0 is not None and fresh1 is not None
    await _add_run(db_session, fresh0.deliverables[0].id, mark="completed")
    await _add_run(db_session, fresh1.deliverables[0].id, mark="failed")
    await db_session.commit()
    for lesson in lessons:
        await repo.recompute_status(lesson.id)
    # A failed lesson surfaces on the course even though its own run completed.
    assert await repo.recompute_status(course.id) == "failed"


async def test_recompute_parent_producing_when_lesson_active(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    course, lessons = await _mk_course_with_lessons(
        repo, db_session, tenant_id=tenant_id, subject_id=subject_id, n=2
    )
    fresh = await repo.get(lessons[0].id, tenant_id=tenant_id)
    assert fresh is not None
    await _add_run(db_session, fresh.deliverables[0].id, mark="started")
    await db_session.commit()
    await repo.recompute_status(lessons[0].id)
    assert await repo.recompute_status(course.id) == "producing"


async def test_recompute_archived_child_excluded_from_rollup(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    course, lessons = await _mk_course_with_lessons(
        repo, db_session, tenant_id=tenant_id, subject_id=subject_id, n=2
    )
    # Course own run completed; one lesson completed, the other archived.
    fresh_course = await repo.get(course.id, tenant_id=tenant_id)
    assert fresh_course is not None
    await _add_run(db_session, fresh_course.deliverables[0].id, mark="completed")
    fresh0 = await repo.get(lessons[0].id, tenant_id=tenant_id)
    assert fresh0 is not None
    await _add_run(db_session, fresh0.deliverables[0].id, mark="completed")
    await db_session.commit()
    await repo.recompute_status(lessons[0].id)
    await repo.soft_delete(lessons[1].id, tenant_id=tenant_id)
    await db_session.commit()
    # Archived lesson is excluded → course is completed (own run + 1 live lesson).
    assert await repo.recompute_status(course.id) == "completed"


# ─── list: sorting + metadata ────────────────────────────────────────────


async def test_list_sort_by_title_asc_desc(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    for title in ("charlie", "alpha", "bravo"):
        await _mk_project(repo, tenant_id=tenant_id, subject_id=subject_id, title=title)
    await db_session.commit()
    asc = await repo.list(tenant_id=tenant_id, sort_by="title", order="asc", limit=50, offset=0)
    assert [p.title for p in asc.items] == ["alpha", "bravo", "charlie"]
    desc = await repo.list(tenant_id=tenant_id, sort_by="title", order="desc", limit=50, offset=0)
    assert [p.title for p in desc.items] == ["charlie", "bravo", "alpha"]


async def test_list_filters_still_work_with_sort(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    await _mk_project(repo, tenant_id=tenant_id, subject_id=subject_id, type_="promo", title="a")
    await _mk_project(repo, tenant_id=tenant_id, subject_id=subject_id, type_="generic", title="b")
    await db_session.commit()
    res = await repo.list(
        tenant_id=tenant_id, type_="promo", sort_by="title", order="asc", limit=50, offset=0
    )
    assert [p.type for p in res.items] == ["promo"]


async def test_list_sort_finished_at_nulls_last(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    done = await _mk_project(repo, tenant_id=tenant_id, subject_id=subject_id, title="done")
    await _mk_project(repo, tenant_id=tenant_id, subject_id=subject_id, title="unfinished")
    await db_session.commit()
    fresh = await repo.get(done.id, tenant_id=tenant_id)
    assert fresh is not None
    await _add_run(db_session, fresh.deliverables[0].id, mark="completed")
    await db_session.commit()
    await repo.recompute_status(done.id)
    await db_session.commit()
    res = await repo.list(
        tenant_id=tenant_id, sort_by="finished_at", order="asc", limit=50, offset=0
    )
    # finished project first; the NULL (unfinished) one sorts last even ascending.
    assert res.items[0].id == done.id
    assert res.items[-1].title == "unfinished"


async def test_list_deliverable_count_and_last_run_status(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    proj = await _mk_project(repo, tenant_id=tenant_id, subject_id=subject_id, n_deliv=3)
    other = await _mk_project(repo, tenant_id=tenant_id, subject_id=subject_id, n_deliv=1)
    await db_session.commit()
    fresh = await repo.get(proj.id, tenant_id=tenant_id)
    assert fresh is not None
    await _add_run(db_session, fresh.deliverables[0].id, mark="completed")
    await db_session.commit()
    res = await repo.list(tenant_id=tenant_id, limit=50, offset=0)
    assert res.deliverable_counts[proj.id] == 3
    assert res.deliverable_counts[other.id] == 1
    assert res.last_run_statuses[proj.id] == "completed"
    assert res.last_run_statuses.get(other.id) is None  # no runs
    assert res.last_run_pause_reasons.get(proj.id) is None  # not paused


@pytest.mark.parametrize("reason", ["checkpoint"])
async def test_list_reports_why_the_latest_run_is_paused(db_session: AsyncSession, reason) -> None:
    """A card says "Needs review" only for a checkpoint pause, so the list must carry
    the latest run's pause_reason (0102) beside its status."""
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    proj = await _mk_project(repo, tenant_id=tenant_id, subject_id=subject_id, n_deliv=1)
    await db_session.commit()
    fresh = await repo.get(proj.id, tenant_id=tenant_id)
    assert fresh is not None
    await _add_run(db_session, fresh.deliverables[0].id, mark=reason)
    await db_session.commit()
    res = await repo.list(tenant_id=tenant_id, limit=50, offset=0)
    assert res.last_run_statuses[proj.id] == "paused"
    assert res.last_run_pause_reasons[proj.id] == reason


async def test_list_masks_a_stale_pause_reason_on_a_finished_run(db_session: AsyncSession) -> None:
    """A cancel leaves ``runs.pause_reason`` stamped; the list reports it only while
    the latest run is actually paused."""
    from sqlalchemy import update

    from src.database.models import Run

    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    proj = await _mk_project(repo, tenant_id=tenant_id, subject_id=subject_id, n_deliv=1)
    await db_session.commit()
    fresh = await repo.get(proj.id, tenant_id=tenant_id)
    assert fresh is not None
    run_id = await _add_run(db_session, fresh.deliverables[0].id, mark="checkpoint")
    await db_session.execute(update(Run).where(Run.id == run_id).values(status="cancelled"))
    await db_session.commit()
    res = await repo.list(tenant_id=tenant_id, limit=50, offset=0)
    assert res.last_run_statuses[proj.id] == "cancelled"
    assert res.last_run_pause_reasons[proj.id] is None


async def test_list_thumbnail_prefers_override_from_latest_completed_run(
    db_session: AsyncSession,
) -> None:
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    proj = await _mk_project(repo, tenant_id=tenant_id, subject_id=subject_id)
    no_thumb = await _mk_project(repo, tenant_id=tenant_id, subject_id=subject_id)
    await db_session.commit()
    fresh = await repo.get(proj.id, tenant_id=tenant_id)
    assert fresh is not None
    run_id = await _add_run(db_session, fresh.deliverables[0].id, mark="completed")
    # Two completed stage outputs in the run: an auto cover and a stage that also
    # carries an operator override → the override must win.
    await _add_stage_output(
        db_session,
        run_id,
        "course_thumbnail",
        {"thumbnail_url": "http://minio.test/b/auto.png"},
    )
    await _add_stage_output(
        db_session,
        run_id,
        "course_stitch",
        {
            "thumbnail_url": "http://minio.test/b/auto2.png",
            "thumbnail_override_url": "http://minio.test/b/override.png",
        },
    )
    # A project whose only completed run produced NO thumbnail must be absent.
    fresh_nt = await repo.get(no_thumb.id, tenant_id=tenant_id)
    assert fresh_nt is not None
    nt_run = await _add_run(db_session, fresh_nt.deliverables[0].id, mark="completed")
    await _add_stage_output(db_session, nt_run, "audiobook_stitch", {"duration_seconds": 12})
    await db_session.commit()

    res = await repo.list(tenant_id=tenant_id, limit=50, offset=0)
    assert proj.id in res.thumbnails
    got_run, got_url = res.thumbnails[proj.id]
    assert got_run == run_id
    assert got_url == "http://minio.test/b/override.png"
    assert no_thumb.id not in res.thumbnails


async def test_list_thumbnail_skips_incomplete_runs(db_session: AsyncSession) -> None:
    repo = ProjectRepository(db_session)
    tenant_id, subject_id = uuid4(), uuid4()
    proj = await _mk_project(repo, tenant_id=tenant_id, subject_id=subject_id)
    await db_session.commit()
    fresh = await repo.get(proj.id, tenant_id=tenant_id)
    assert fresh is not None
    # A running run with a thumbnail must NOT contribute a poster.
    run_id = await _add_run(db_session, fresh.deliverables[0].id, mark="started")
    await _add_stage_output(
        db_session,
        run_id,
        "promo_render",
        {"thumbnail_url": "http://minio.test/b/cover.png"},
        status="running",
    )
    await db_session.commit()
    res = await repo.list(tenant_id=tenant_id, limit=50, offset=0)
    assert proj.id not in res.thumbnails
