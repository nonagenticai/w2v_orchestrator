"""RLS gate for the branch link table (``storyboard_branches``, migration 0052).

The branch row carries no ``tenant_id`` of its own — its tenant is one hop away
(``root_project_id -> projects.tenant_id``), so its ``tenant_isolation`` policy is a
join predicate, registered SEPARATELY (``RLS_JOIN_PREDICATES_BRANCHES`` +
``register_late_join_branches_rls_ddl``) so the historical 0046/0049/0050/0051
migrations keep iterating only their own tables.

These tests prove the ``create_all`` (test/dev) path lands the policy AND — under a
restricted, non-owner, ``NOBYPASSRLS`` role, the posture prod actually runs
(``w2v_orch_app`` + ``TENANT_RLS_MODE=enforce``) — that the predicate genuinely
ISOLATES: tenant B can neither SELECT tenant A's branch rows nor INSERT a branch onto
tenant A's project. That INSERT is the attack the table has to make impossible: a
forged branch row on someone else's root is a merge path into their storyboard.

Introspection is scoped to ``current_schema()`` because CI runs each job in its own
schema inside a SHARED postgres. Mirrors ``test_storyboard_panel_assets_rls.py``.
"""

from __future__ import annotations

from uuid import UUID, uuid7

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from src.database.models import Project
from src.database.repositories.storyboard_repository import (
    StoryboardPanelRepository,
    StoryboardRepository,
)
from src.database.rls import POLICY_NAME, RLS_JOIN_PREDICATES_BRANCHES
from src.services.storyboard.branch_service import StoryboardBranchService
from tests.conftest import require_createrole

_TABLE = "storyboard_branches"


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
async def test_branches_table_has_rls_enabled_not_forced(db_session) -> None:
    row = await _rls_flags(db_session, _TABLE)
    assert row.relrowsecurity is True, f"{_TABLE} must have ROW LEVEL SECURITY enabled"
    assert row.relforcerowsecurity is False, (
        f"{_TABLE} must NOT FORCE RLS (owner stays inert — FORCE breaks the "
        "SECURITY DEFINER cross-tenant resolvers under a non-superuser owner)"
    )


@pytest.mark.asyncio
async def test_branches_table_has_exactly_one_tenant_isolation_policy(db_session) -> None:
    names = await _policy_names(db_session, _TABLE)
    assert names == [POLICY_NAME], (
        f"expected exactly one {POLICY_NAME} policy on {_TABLE}, got {names}"
    )


def test_branches_rls_scope_is_locked() -> None:
    # Lock the separate branches set so the link table can't silently ship without its
    # own RLS decision, and so nobody folds it back into a map a HISTORICAL migration
    # iterates (which would re-run that migration's DDL over a table it predates).
    assert set(RLS_JOIN_PREDICATES_BRANCHES) == {"storyboard_branches"}


async def _seed_branch(session, tenant_id: UUID) -> dict[str, UUID]:
    """One tenant's project → storyboard → panel → fork, plus a SPARE project.

    The spare exists for the forged-INSERT probe: ``child_project_id`` is UNIQUE, so
    re-using the real child would risk tripping the unique constraint instead of the
    policy — and then the test would pass for the wrong reason.
    """
    project = Project(
        id=uuid7(),
        tenant_id=tenant_id,
        owner_subject_id=uuid7(),
        visibility="tenant",
        type="generic",
        title="branch-rls",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    spare = Project(
        id=uuid7(),
        tenant_id=tenant_id,
        owner_subject_id=project.owner_subject_id,
        visibility="tenant",
        type="generic",
        title="branch-rls-spare",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    session.add_all([project, spare])
    await session.flush()

    storyboard = await StoryboardRepository(session).ensure_for_project(
        project.id, tenant_id=tenant_id
    )
    assert storyboard is not None
    await StoryboardPanelRepository(session).create(
        project_id=project.id,
        storyboard_id=storyboard.id,
        position=0,
        shot_key="clip-1",
        vo_text="one",
    )
    branch, child = await StoryboardBranchService(session).create_branch(
        project.id, tenant_id=tenant_id, name="rls", actor_subject=uuid7()
    )
    ids = {
        "project": project.id,
        "spare": spare.id,
        "branch": branch.id,
        "child": child.id,
        "base_snapshot": branch.base_snapshot_id,
    }
    await session.commit()
    return ids


@pytest.mark.asyncio
async def test_branches_rls_isolates_tenants_under_restricted_role(db_session_factory) -> None:
    """The join predicate must actually ISOLATE, not merely exist.

    ``ENABLE`` (not ``FORCE``) means the table owner — which every other test connects
    as — bypasses the policy, so isolation is invisible unless it is exercised under a
    non-owner ``NOBYPASSRLS`` role: precisely the prod runtime posture (``w2v_orch_app``).
    ``SET LOCAL ROLE`` + a transaction-local GUC, rolled back, so the owner-bypass the
    rest of the suite relies on is untouched.
    """
    role = f"br_rls_probe_{uuid7().hex[:12]}"

    async with db_session_factory() as owner:
        schema = (await owner.execute(text("SELECT current_schema()"))).scalar_one()
        try:
            await owner.execute(text(f'CREATE ROLE "{role}" NOLOGIN NOSUPERUSER NOBYPASSRLS'))
        except ProgrammingError as exc:  # insufficient_privilege → role lacks CREATEROLE
            await owner.rollback()
            require_createrole(exc)  # fail under CI, skip for a locally-restricted role
        await owner.execute(text(f'GRANT "{role}" TO CURRENT_USER'))
        await owner.execute(text(f'GRANT USAGE ON SCHEMA "{schema}" TO "{role}"'))
        await owner.execute(
            text(
                f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA "{schema}" TO "{role}"'
            )
        )
        await owner.commit()

    try:
        tenant_a, tenant_b = uuid7(), uuid7()
        async with db_session_factory() as owner:
            ids_a = await _seed_branch(owner, tenant_a)
        async with db_session_factory() as owner:
            ids_b = await _seed_branch(owner, tenant_b)

        # ── 1. GUC=A under the restricted role → ONLY tenant A's branch row ────
        async with db_session_factory() as s:
            await s.execute(text(f'SET LOCAL ROLE "{role}"'))
            await s.execute(
                text("SELECT set_config('app.current_tenant', :t, true)"), {"t": str(tenant_a)}
            )
            rows = (await s.execute(text(f"SELECT id FROM {_TABLE}"))).scalars()  # noqa: S608
            assert list(rows) == [ids_a["branch"]], "tenant B's branch must be invisible to A"
            await s.rollback()

        # ── 2. GUC=B → tenant A's branch is unreachable (the SELECT gate) ──────
        async with db_session_factory() as s:
            await s.execute(text(f'SET LOCAL ROLE "{role}"'))
            await s.execute(
                text("SELECT set_config('app.current_tenant', :t, true)"), {"t": str(tenant_b)}
            )
            count = (
                await s.execute(
                    text(f"SELECT count(*) FROM {_TABLE} WHERE id = :b"),  # noqa: S608
                    {"b": str(ids_a["branch"])},
                )
            ).scalar_one()
            assert count == 0, "tenant B must not SELECT tenant A's branch row"
            await s.rollback()

        # ── 3. No GUC → fail-closed (0 rows, not all rows) ─────────────────────
        async with db_session_factory() as s:
            await s.execute(text(f'SET LOCAL ROLE "{role}"'))
            count = (
                await s.execute(text(f"SELECT count(*) FROM {_TABLE}"))  # noqa: S608
            ).scalar_one()
            assert count == 0, "an unset GUC must be fail-closed on the branch table"
            await s.rollback()

        # ── 4. WITH CHECK rejects forging a branch onto tenant A's project ─────
        # The attack the table must make impossible: a branch row on somebody else's
        # root is a merge path into their storyboard. Every FK is satisfiable (all rows
        # were seeded by the owner) and ``child_project_id`` is a FRESH project, so the
        # UNIQUE constraint is not what stands in the way — only the policy is.
        async with db_session_factory() as s:
            await s.execute(text(f'SET LOCAL ROLE "{role}"'))
            await s.execute(
                text("SELECT set_config('app.current_tenant', :t, true)"), {"t": str(tenant_b)}
            )
            with pytest.raises(ProgrammingError) as caught:
                await s.execute(
                    text(
                        f"INSERT INTO {_TABLE} "  # noqa: S608
                        "(id, root_project_id, child_project_id, name, base_snapshot_id, status) "
                        "VALUES (:id, :root, :child, 'forged', :snap, 'open')"
                    ),
                    {
                        "id": str(uuid7()),
                        "root": str(ids_a["project"]),
                        "child": str(ids_a["spare"]),
                        "snap": str(ids_a["base_snapshot"]),
                    },
                )
            assert "row-level security" in str(caught.value.orig).lower()
            await s.rollback()

        # Sanity: as the owner, both tenants' rows are intact and nothing was smuggled in.
        async with db_session_factory() as owner:
            total = (
                await owner.execute(text(f"SELECT count(*) FROM {_TABLE}"))  # noqa: S608
            ).scalar_one()
            assert total == 2, "one branch per tenant, and no forged third"
            assert ids_a["branch"] != ids_b["branch"]
    finally:
        async with db_session_factory() as owner:
            for stmt in (
                f'REVOKE "{role}" FROM CURRENT_USER',
                f'DROP OWNED BY "{role}"',
                f'DROP ROLE IF EXISTS "{role}"',
            ):
                try:
                    await owner.execute(text(stmt))
                    await owner.commit()
                except ProgrammingError:
                    await owner.rollback()
