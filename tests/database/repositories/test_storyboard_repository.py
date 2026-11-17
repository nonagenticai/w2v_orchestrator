"""Repository tests for the ``storyboards`` grouping row (backend hardening).

Covers the two invariants WS-B1 hardens in
``src/database/repositories/storyboard_repository.py`` +
``src/database/models.py``:

* ``StoryboardRepository.ensure_for_project`` is idempotent — repeated calls on
  one project create EXACTLY ONE ``storyboards`` row and return the same id.
* The ``uq_storyboards_project`` UNIQUE constraint exists on
  ``storyboards.project_id`` (the DB-level guarantee behind idempotency).

Uses the same real-postgres ``db_session`` fixture the sibling repo tests use.
``ensure_for_project`` scope-checks ``project_id -> projects.tenant_id`` and
returns ``None`` for an unknown project, so a ``projects`` row is seeded first
(mirrors ``_seed_project`` in ``tests/api/routes/test_storyboard_routes.py``).
"""

from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Project, Storyboard
from src.database.repositories.storyboard_repository import StoryboardRepository

pytestmark = pytest.mark.asyncio


async def _seed_project(session: AsyncSession, *, tenant_id: UUID) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=uuid4(),
        visibility="tenant",
        type="generic",
        title="storyboard-repo",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    session.add(project)
    await session.commit()
    return project


async def test_ensure_for_project_is_idempotent(db_session: AsyncSession) -> None:
    """Two sequential calls return the same storyboard id and leave one row."""
    tenant_id = uuid4()
    project = await _seed_project(db_session, tenant_id=tenant_id)
    repo = StoryboardRepository(db_session)

    first = await repo.ensure_for_project(project.id, tenant_id=tenant_id)
    second = await repo.ensure_for_project(project.id, tenant_id=tenant_id)

    assert first is not None and second is not None
    assert first.id == second.id

    count = (
        await db_session.execute(
            select(func.count()).select_from(Storyboard).where(Storyboard.project_id == project.id)
        )
    ).scalar_one()
    assert count == 1

    # TODO(concurrency): a truly-concurrent asyncio.gather over two separate
    # sessions would exercise the unique-constraint race path, but that depends
    # on WS-B1's ON CONFLICT / IntegrityError handling in ensure_for_project and
    # would be non-deterministic against a create_all schema. The sequential
    # idempotency above + the constraint-presence assertion below cover the
    # frozen contract deterministically.


async def test_ensure_for_project_unknown_project_returns_none(db_session: AsyncSession) -> None:
    """Scope gate: an unknown/foreign project yields no storyboard row."""
    repo = StoryboardRepository(db_session)
    result = await repo.ensure_for_project(uuid4(), tenant_id=uuid4())
    assert result is None


async def test_uq_storyboards_project_constraint_exists(db_session: AsyncSession) -> None:
    """The UNIQUE constraint backing idempotency is present on the schema.

    create_all builds the schema from the model, so this asserts the
    ``UniqueConstraint('project_id', name='uq_storyboards_project')`` on the
    ``Storyboard`` model actually materialized in the DB (covers the migration
    intent without running ``alembic upgrade``).
    """
    row = (
        await db_session.execute(
            text(
                "SELECT constraint_type FROM information_schema.table_constraints "
                "WHERE table_name = 'storyboards' "
                "AND constraint_name = 'uq_storyboards_project'"
            )
        )
    ).first()
    assert row is not None, "uq_storyboards_project constraint is missing"
    assert row.constraint_type == "UNIQUE"
