"""DB-backed tests for the Phase-0 storyboard seed in ``build_config_snapshot``.

``run_setup.build_config_snapshot`` folds a project's PERSISTED storyboard panels
into ``config_snapshot["brief"]["storyboard"]`` so the authored short-circuits
render FROM the approved board. Persisted panels WIN over any inline
``brief["storyboard"]``; a project with no persisted panels leaves the brief's
storyboard key untouched (legacy byte-identical).

Uses a real ``db_session`` + seeded panels (via the storyboard repos), mirroring
how ``test_run_setup_direction.py`` builds config but exercising the real DB path
instead of stubbing the repo.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from src.database.models import Project
from src.database.repositories.storyboard_repository import (
    StoryboardPanelRepository,
    StoryboardRepository,
)
from src.services.runs.run_setup import build_config_snapshot

pytestmark = pytest.mark.asyncio


async def _seed_project(db_session, *, tenant_id: UUID, brief: dict | None = None) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=uuid4(),
        visibility="tenant",
        type="generic",
        title="storyboard-seed",
        brief=brief or {},
        brand={},
        voice={},
        theme={},
    )
    db_session.add(project)
    await db_session.commit()
    return project


async def _seed_panels(db_session, project_id: UUID, tenant_id: UUID, panels: list[dict]) -> None:
    storyboard = await StoryboardRepository(db_session).ensure_for_project(
        project_id, tenant_id=tenant_id
    )
    assert storyboard is not None
    repo = StoryboardPanelRepository(db_session)
    for panel in panels:
        await repo.create(
            project_id=project_id,
            storyboard_id=storyboard.id,
            position=panel.pop("position", None),
            shot_key=panel.pop("shot_key", None),
            **panel,
        )
    await db_session.commit()


def _proj(project: Project, brief: dict) -> SimpleNamespace:
    """A pure config-source stub (avoids lazy-loading an expired ORM row) keyed on
    the seeded project's real id so the repo lookups resolve its persisted panels."""
    return SimpleNamespace(
        id=project.id,
        title="storyboard-seed",
        type="generic",
        brief=brief,
        brand={},
        voice={},
        theme={},
    )


def _deliv() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        name="d",
        format="16x9",
        captions_mode="none",
        target_duration_seconds=None,
    )


async def _build(db_session, proj: SimpleNamespace, tenant_id: UUID) -> dict:
    return await build_config_snapshot(
        db_session,
        proj=proj,
        deliv=_deliv(),
        tenant_id=tenant_id,
        subject_id=uuid4(),
        config_overrides=None,
    )


async def test_persisted_panels_folded_into_brief_storyboard(db_session) -> None:
    tenant_id = uuid4()
    project = await _seed_project(db_session, tenant_id=tenant_id)
    await _seed_panels(
        db_session,
        project.id,
        tenant_id,
        [
            {"shot_key": "clip-1", "vo_text": "first", "duration_seconds": 2.0},
            {"shot_key": "clip-2", "vo_text": "second", "shot_number": "2A"},
        ],
    )

    snap = await _build(db_session, _proj(project, brief={}), tenant_id)

    board = snap["brief"]["storyboard"]
    assert [p["shot_key"] for p in board] == ["clip-1", "clip-2"]
    assert [p["vo_text"] for p in board] == ["first", "second"]
    assert board[0]["duration_seconds"] == 2.0
    assert board[1]["shot_number"] == "2A"


async def test_persisted_wins_over_inline_brief_storyboard(db_session) -> None:
    tenant_id = uuid4()
    inline = [{"shot_key": "clip-99", "vo_text": "inline only"}]
    project = await _seed_project(db_session, tenant_id=tenant_id, brief={"storyboard": inline})
    await _seed_panels(
        db_session,
        project.id,
        tenant_id,
        [{"shot_key": "clip-1", "vo_text": "persisted"}],
    )

    snap = await _build(db_session, _proj(project, brief={"storyboard": inline}), tenant_id)

    board = snap["brief"]["storyboard"]
    # Persisted panels overwrite the inline board entirely.
    assert [p["shot_key"] for p in board] == ["clip-1"]
    assert board[0]["vo_text"] == "persisted"


async def test_inline_brief_storyboard_preserved_without_persisted(db_session) -> None:
    tenant_id = uuid4()
    inline = [{"shot_key": "clip-7", "vo_text": "inline"}]
    project = await _seed_project(db_session, tenant_id=tenant_id, brief={"storyboard": inline})

    # No persisted panels → the inline board is copied through unchanged.
    snap = await _build(db_session, _proj(project, brief={"storyboard": inline}), tenant_id)

    assert snap["brief"]["storyboard"] == inline


async def test_no_storyboard_leaves_brief_key_absent(db_session) -> None:
    tenant_id = uuid4()
    project = await _seed_project(db_session, tenant_id=tenant_id)

    snap = await _build(db_session, _proj(project, brief={}), tenant_id)

    # Legacy byte-identical: neither persisted nor inline → no storyboard key.
    assert "storyboard" not in snap["brief"]


async def test_mixed_status_seeds_only_approved(db_session) -> None:
    tenant_id = uuid4()
    project = await _seed_project(db_session, tenant_id=tenant_id)
    await _seed_panels(
        db_session,
        project.id,
        tenant_id,
        [
            {"shot_key": "clip-1", "vo_text": "approved", "status": "approved"},
            {"shot_key": "clip-2", "vo_text": "draft"},  # server_default 'draft'
        ],
    )

    snap = await _build(db_session, _proj(project, brief={}), tenant_id)

    board = snap["brief"]["storyboard"]
    # Any approved panel → approved-only seed; the draft clip-2 is excluded.
    assert [p["shot_key"] for p in board] == ["clip-1"]
    assert board[0]["vo_text"] == "approved"


async def test_zero_approved_seeds_all(db_session) -> None:
    tenant_id = uuid4()
    project = await _seed_project(db_session, tenant_id=tenant_id)
    await _seed_panels(
        db_session,
        project.id,
        tenant_id,
        [
            {"shot_key": "clip-1", "vo_text": "first"},
            {"shot_key": "clip-2", "vo_text": "second"},
        ],
    )

    snap = await _build(db_session, _proj(project, brief={}), tenant_id)

    board = snap["brief"]["storyboard"]
    # No approved panel → `approved or rows` falls back to every panel.
    assert [p["shot_key"] for p in board] == ["clip-1", "clip-2"]


async def test_all_approved_seeds_all(db_session) -> None:
    tenant_id = uuid4()
    project = await _seed_project(db_session, tenant_id=tenant_id)
    await _seed_panels(
        db_session,
        project.id,
        tenant_id,
        [
            {"shot_key": "clip-1", "vo_text": "first", "status": "approved"},
            {"shot_key": "clip-2", "vo_text": "second", "status": "approved"},
        ],
    )

    snap = await _build(db_session, _proj(project, brief={}), tenant_id)

    board = snap["brief"]["storyboard"]
    assert [p["shot_key"] for p in board] == ["clip-1", "clip-2"]
