"""RLS gate for the storyboard version-snapshot table (``storyboard_snapshots``).

The snapshot table is project-scoped and reaches its tenant via the
``project_id -> projects.tenant_id`` FK chain, so it carries the SAME join-based
``tenant_isolation`` policy every other storyboard table does — ENABLEd, not
FORCEd (owner stays inert this increment). It is registered SEPARATELY
(``RLS_JOIN_PREDICATES_SNAPSHOTS`` + ``register_late_join_snapshots_rls_ddl``) from
the 0046/0049 sets so historical migrations keep iterating only their own tables;
these tests prove the ``create_all`` apply path lands the policy AND that the
repo/service project-join isolates tenants (the isolation the owner-bypass posture
otherwise hides). Introspection is scoped to ``current_schema()`` because CI runs
each job in its own schema inside a SHARED postgres. Mirrors
``test_storyboard_events_rls.py``.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from src.database.models import Project
from src.database.repositories.storyboard_repository import StoryboardSnapshotRepository
from src.database.rls import POLICY_NAME, RLS_JOIN_PREDICATES_SNAPSHOTS
from src.services.storyboard.storyboard_snapshot_service import (
    SnapshotNotFoundError,
    StoryboardSnapshotService,
)

_TABLE = "storyboard_snapshots"


async def _rls_flags(db_session, table: str):
    return (
        await db_session.execute(
            text(
                "SELECT c.relrowsecurity, c.relforcerowsecurity "
                "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE c.relname = :t AND n.nspname = current_schema()"
            ),
            {"t": table},
        )
    ).one()


async def _policy_names(db_session, table: str) -> list[str]:
    return [
        r.policyname
        for r in (
            await db_session.execute(
                text(
                    "SELECT policyname FROM pg_policies "
                    "WHERE tablename = :t AND schemaname = current_schema()"
                ),
                {"t": table},
            )
        ).all()
    ]


@pytest.mark.asyncio
async def test_snapshots_table_has_rls_enabled_not_forced(db_session) -> None:
    row = await _rls_flags(db_session, _TABLE)
    assert row.relrowsecurity is True, f"{_TABLE} must have ROW LEVEL SECURITY enabled"
    assert row.relforcerowsecurity is False, (
        f"{_TABLE} must NOT FORCE RLS (owner stays inert this increment)"
    )


@pytest.mark.asyncio
async def test_snapshots_table_has_exactly_one_tenant_isolation_policy(db_session) -> None:
    names = await _policy_names(db_session, _TABLE)
    assert names == [POLICY_NAME], (
        f"expected exactly one {POLICY_NAME} policy on {_TABLE}, got {names}"
    )


def test_snapshots_rls_scope_is_locked() -> None:
    # Lock the separate late-snapshots set so the table can't silently ship without
    # its own RLS decision (parallel to test_storyboard_events_rls' scope lock).
    assert set(RLS_JOIN_PREDICATES_SNAPSHOTS) == {"storyboard_snapshots"}


async def _seed_project(db_session, *, tenant_id: UUID) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=uuid4(),
        visibility="tenant",
        type="generic",
        title="snapshot-rls",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    db_session.add(project)
    await db_session.commit()
    return project


@pytest.mark.asyncio
async def test_cross_tenant_cannot_reach_snapshots(db_session) -> None:
    """Tenant B cannot list, get, or restore tenant A's snapshot."""
    tenant_a = uuid4()
    tenant_b = uuid4()
    project = await _seed_project(db_session, tenant_id=tenant_a)

    svc = StoryboardSnapshotService(db_session)
    snap = await svc.create_snapshot(
        project.id, tenant_id=tenant_a, kind="manual", note="a", actor_subject=uuid4()
    )
    await db_session.commit()

    repo = StoryboardSnapshotRepository(db_session)
    # list: tenant B sees nothing for tenant A's project
    assert await repo.list_for_project(project.id, tenant_id=tenant_b) == []
    # get: tenant B cannot fetch tenant A's snapshot by id
    assert await repo.get(snap.id, tenant_id=tenant_b) is None
    # restore: tenant B is 404 (SnapshotNotFoundError), leaving state untouched
    with pytest.raises(SnapshotNotFoundError):
        await svc.restore(
            snap.id, tenant_id=tenant_b, actor_subject=uuid4(), expected_lock_versions={}
        )

    # tenant A still sees exactly its one snapshot (no auto snapshot leaked in).
    a_snaps = await repo.list_for_project(project.id, tenant_id=tenant_a)
    assert [s.id for s in a_snaps] == [snap.id]
