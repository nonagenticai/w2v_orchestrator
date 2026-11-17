"""``restore_snapshot`` must map a raced unique violation to 409, not 500.

Restore is the sharpest write in the storyboard: it rewrites scenes + panels +
assets wholesale, and ``StoryboardSnapshotService.restore`` runs
``SET CONSTRAINTS ALL DEFERRED`` (storyboard_snapshot_service.py) so the rewrite
can hold a TRANSIENT duplicate on the DEFERRABLE
``uq_storyboard_panels_project_shot`` mid-write. Consequence: on this path the
unique check does NOT run at flush — it runs at ``COMMIT``.

The route's only ``except`` was ``SnapshotNotFoundError``, and the ``commit()``
sat AFTER the ``try``. So a writer that landed a colliding ``shot_key`` between
the caller's read and the restore produced an ``IntegrityError`` from a line no
handler covered: a 500, with the session left in a failed transaction and never
rolled back. ``register_storyboard_exception_handlers`` maps only
``ProjectArchivedError`` and ``StoryboardVersionConflictError``, so nothing
downstream caught it either.

⚠️ These call the handler DIRECTLY, for the reason
``tests/api/routes/test_deferred_constraint_maps_to_409.py`` and
``tests/api/routes/test_write_durability.py`` both give: driving it through the
FastAPI test client proves nothing about commit placement, because the client
runs the dependency teardown — which commits — before handing back the response,
so the commit under test is not the one that decides the status code.

The failure here is real, not simulated. The restore that runs is the REAL
service (it really defers, really rewrites the board); only the concurrent
writer is injected, via a subclass that writes the colliding row after
``super().restore()`` returns and before the route commits — exactly where a
racing sibling's row would land.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from src.api.routes import storyboard as storyboard_routes
from src.api.routes.storyboard import create_panel, create_snapshot, restore_snapshot
from src.api.schemas import StoryboardPanelCreate
from src.api.schemas.storyboard import RestoreSnapshotRequest, SnapshotCreateRequest
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.storyboard_repository import (
    StoryboardPanelRepository,
    StoryboardRepository,
)
from src.services.storyboard.storyboard_snapshot_service import StoryboardSnapshotService

pytestmark = pytest.mark.asyncio

_SHOT_KEY = "1A"


async def _seed_project(session, tenant_id: UUID, subject_id: UUID, title: str) -> UUID:
    """A committed project to hang storyboard rows off. Not the thing under test."""
    proj = await ProjectRepository(session).create_with_deliverables(
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        type_="generic",
        title=title,
        brief={},
        brand={},
        voice={},
        theme={},
        visibility="tenant",
        parent_project_id=None,
        deliverables=[{"name": "d1", "format": "16x9"}],
    )
    await session.commit()
    return proj.id


async def _board_map(session, project_id: UUID) -> dict[UUID, int]:
    """The whole-board ``{row id: lock_version}`` CAS map restore REQUIRES.

    Read straight from the three live tables rather than through the list routes,
    because this file never goes through the HTTP client. A map that does not
    match the board is a 409 from ``assert_board_at`` — which would short-circuit
    the restore BEFORE the commit under test and make these tests vacuous.
    """
    merged: dict[UUID, int] = {}
    for table in ("storyboard_scenes", "storyboard_panels", "storyboard_assets"):
        rows = await session.execute(
            text(f"select id, lock_version from {table} where project_id = :p"),  # noqa: S608
            {"p": str(project_id)},
        )
        merged.update({UUID(str(r.id)): r.lock_version for r in rows})
    return merged


async def _panels_on_shot_key(session, project_id: UUID, shot_key: str) -> int:
    return (
        await session.execute(
            text("select count(*) from storyboard_panels where project_id = :p and shot_key = :k"),
            {"p": str(project_id), "k": shot_key},
        )
    ).scalar_one()


async def _write_duplicate_shot_key(session, project_id: UUID, tenant_id: UUID) -> None:
    """Mint a SECOND panel on an already-used shot_key, through the real repository.

    Under the deferral the real restore already opened, this INSERT succeeds and the
    unique violation is held back to COMMIT — which is exactly the state a racing
    concurrent write leaves the restore transaction in.
    """
    storyboard = await StoryboardRepository(session).ensure_for_project(
        project_id, tenant_id=tenant_id
    )
    assert storyboard is not None
    await StoryboardPanelRepository(session).create(
        project_id=project_id,
        storyboard_id=storyboard.id,
        position=None,
        shot_key=_SHOT_KEY,
    )


def _racing_service(project_id: UUID, tenant_id: UUID) -> type[StoryboardSnapshotService]:
    """The REAL restore, plus a concurrent writer's colliding row landing after it."""

    class _RacingRestore(StoryboardSnapshotService):
        async def restore(self, *args: Any, **kwargs: Any) -> dict[str, UUID]:
            result = await super().restore(*args, **kwargs)
            await _write_duplicate_shot_key(self.session, project_id, tenant_id)
            return result

    return _RacingRestore


async def _snapshot_of(session, project_id: UUID, tenant_id: UUID, subject_id: UUID) -> UUID:
    snap = await create_snapshot(
        project_id=project_id,
        body=SnapshotCreateRequest.model_validate({"note": "v1"}),
        session=session,
        tenant_id=tenant_id,
        subject_id=subject_id,
    )
    return snap.id


async def test_restore_really_defers_the_unique_check(db_session_factory) -> None:
    """Premise check. Without this the test below could pass for the wrong reason.

    If the real ``restore`` did not actually defer ``uq_storyboard_panels_project_shot``,
    the injected duplicate would raise at FLUSH — inside the try either way — and the
    409 assertion would be exercising the flush-time path while claiming the
    commit-time one. Here the duplicate INSERT must SUCCEED and the transaction must
    still be alive afterwards, proving the check was genuinely postponed to COMMIT.
    """
    tenant_id, subject_id = uuid4(), uuid4()
    async with db_session_factory() as session:
        project_id = await _seed_project(session, tenant_id, subject_id, "restore-defer-premise")
        await create_panel(
            project_id=project_id,
            body=StoryboardPanelCreate.model_validate({"shot_key": _SHOT_KEY}),
            session=session,
            tenant_id=tenant_id,
        )
        snapshot_id = await _snapshot_of(session, project_id, tenant_id, subject_id)

        await StoryboardSnapshotService(session).restore(
            snapshot_id,
            tenant_id=tenant_id,
            actor_subject=subject_id,
            expected_lock_versions=await _board_map(session, project_id),
        )
        await _write_duplicate_shot_key(session, project_id, tenant_id)

        # Not raised yet, and the transaction is still usable: the check was deferred.
        assert await _panels_on_shot_key(session, project_id, _SHOT_KEY) == 2, (
            "the duplicate did not survive the flush — restore is not deferring the "
            "constraint, so this file is not exercising the commit-time path"
        )
        await session.rollback()


async def test_restore_raced_by_a_concurrent_write_is_409_not_500(
    db_session_factory, monkeypatch
) -> None:
    """The defect. A raced unique violation must be a retryable 409, and the session
    must be left usable — rolled back, not poisoned by the failed commit."""
    tenant_id, subject_id = uuid4(), uuid4()
    async with db_session_factory() as session:
        project_id = await _seed_project(session, tenant_id, subject_id, "restore-raced-409")
        await create_panel(
            project_id=project_id,
            body=StoryboardPanelCreate.model_validate({"shot_key": _SHOT_KEY}),
            session=session,
            tenant_id=tenant_id,
        )
        snapshot_id = await _snapshot_of(session, project_id, tenant_id, subject_id)
        board = await _board_map(session, project_id)

        monkeypatch.setattr(
            storyboard_routes,
            "StoryboardSnapshotService",
            _racing_service(project_id, tenant_id),
        )

        with pytest.raises(HTTPException) as excinfo:
            await restore_snapshot(
                snapshot_id=snapshot_id,
                session=session,
                tenant_id=tenant_id,
                subject_id=subject_id,
                body=RestoreSnapshotRequest.model_validate({"expected_lock_versions": board}),
            )

        assert excinfo.value.status_code == 409, (
            "restore's only except was SnapshotNotFoundError and the commit sat "
            "outside the try, so a raced shot_key collision surfaced at COMMIT and "
            "escaped as an uncaught 500"
        )

        # The handler must roll back BEFORE raising. A session left in the failed
        # transaction cannot answer this at all — asyncpg refuses every further
        # statement until the transaction is rolled back.
        assert (await session.execute(text("select 1"))).scalar_one() == 1, (
            "the session was left dirty: the 409 was raised without rolling back"
        )

        # And nothing partial landed: the whole raced restore is gone, not half of it.
        assert await _panels_on_shot_key(session, project_id, _SHOT_KEY) == 1


async def test_restore_unknown_snapshot_is_still_404(db_session_factory) -> None:
    """Control: adding the IntegrityError arm must not shadow the 404 arm."""
    tenant_id, subject_id = uuid4(), uuid4()
    async with db_session_factory() as session:
        await _seed_project(session, tenant_id, subject_id, "restore-unknown-404")

        with pytest.raises(HTTPException) as excinfo:
            await restore_snapshot(
                snapshot_id=uuid4(),
                session=session,
                tenant_id=tenant_id,
                subject_id=subject_id,
                body=RestoreSnapshotRequest.model_validate({"expected_lock_versions": {}}),
            )

        assert excinfo.value.status_code == 404
        await session.rollback()


async def test_uncontested_restore_still_commits_durably(db_session_factory) -> None:
    """Control for the fix itself.

    Moving a commit INTO a ``try`` can silently stop it happening. This asserts the
    ordinary restore still lands durably — read through an INDEPENDENT session, per
    ``test_write_durability.py``, because a flush would satisfy a same-session read.
    """
    tenant_id, subject_id = uuid4(), uuid4()
    async with db_session_factory() as session:
        project_id = await _seed_project(session, tenant_id, subject_id, "restore-durable")
        panel = await create_panel(
            project_id=project_id,
            body=StoryboardPanelCreate.model_validate({"shot_key": _SHOT_KEY, "vo_text": "orig"}),
            session=session,
            tenant_id=tenant_id,
        )
        snapshot_id = await _snapshot_of(session, project_id, tenant_id, subject_id)

        await session.execute(
            text("update storyboard_panels set vo_text = 'changed' where id = :i"),
            {"i": str(panel.id)},
        )
        await session.commit()

        result = await restore_snapshot(
            snapshot_id=snapshot_id,
            session=session,
            tenant_id=tenant_id,
            subject_id=subject_id,
            body=RestoreSnapshotRequest.model_validate(
                {"expected_lock_versions": await _board_map(session, project_id)}
            ),
        )
        assert result.restored_snapshot_id == snapshot_id

    async with db_session_factory() as probe:
        vo = (
            await probe.execute(
                text("select vo_text from storyboard_panels where project_id = :p"),
                {"p": str(project_id)},
            )
        ).scalar_one()

    assert vo == "orig", (
        "the restore is not durable at response-build time — moving the commit into "
        "the try stopped it committing at all"
    )
