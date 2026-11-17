"""Test helper: build the whole-board optimistic-concurrency map.

The destructive storyboard writes (``StoryboardSnapshotService.restore``, the two
``reorder`` repo methods) now REQUIRE the caller to state the board state it is writing
over — one flat ``{row id: lock_version}`` map across scenes, panels and assets. Tests
that only care about the write itself (not the concurrency contract) build a CURRENT map
through these helpers; the tests that pin the contract live in
``tests/api/routes/test_storyboard_destructive_lock.py`` and construct STALE maps by hand.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import (
    StoryboardAsset,
    StoryboardPanel,
    StoryboardScene,
    StoryboardSnapshot,
)


async def board_lock_versions(session: AsyncSession, project_id: UUID) -> dict[UUID, int]:
    """``{row id: lock_version}`` for every scene, panel and asset of ``project_id``."""
    live: dict[UUID, int] = {}
    for model in (StoryboardScene, StoryboardPanel, StoryboardAsset):
        rows = (
            await session.execute(
                select(model.id, model.lock_version).where(model.project_id == project_id)
            )
        ).all()
        live.update({row[0]: row[1] for row in rows})
    return live


async def board_lock_versions_for_snapshot(
    session: AsyncSession, snapshot_id: UUID
) -> dict[UUID, int]:
    """Same map, resolved through the snapshot's own ``project_id`` — the restore call
    sites only have the snapshot id in hand."""
    project_id = (
        await session.execute(
            select(StoryboardSnapshot.project_id).where(StoryboardSnapshot.id == snapshot_id)
        )
    ).scalar_one()
    return await board_lock_versions(session, project_id)
