"""RLS gate for the panel→asset join table (``storyboard_panel_assets``, 0051).

The join row carries neither ``tenant_id`` NOR ``project_id`` — it is the first
storyboard table whose tenant is **2 hops** away (``panel_id ->
storyboard_panels.project_id -> projects.tenant_id``), so its ``tenant_isolation``
policy is a deeper ``EXISTS`` climb than every sibling's. It is registered SEPARATELY
(``RLS_JOIN_PREDICATES_PANEL_ASSETS`` + ``register_late_join_panel_assets_rls_ddl``)
from the 0046/0049/0050 sets so those historical migrations keep iterating only their
own tables; these tests prove the ``create_all`` (test/dev) apply path lands the
policy AND — under a restricted, non-owner, no-``BYPASSRLS`` role, the posture prod
actually runs (``w2v_orch_app`` + ``TENANT_RLS_MODE=enforce``) — that the predicate
genuinely ISOLATES: tenant B can neither SELECT nor INSERT tenant A's join rows.

Introspection is scoped to ``current_schema()`` because CI runs each job in its own
schema inside a SHARED postgres — a bare ``tablename`` match against the cluster-wide
``pg_policies`` would hit every schema's copy. Mirrors ``test_storyboard_events_rls.py``
(metadata gates) + ``test_rls_isolation.py`` (the restricted-role proof).
"""

from __future__ import annotations

from uuid import UUID, uuid7

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from src.database.models import Project
from src.database.repositories.storyboard_repository import (
    StoryboardAssetRepository,
    StoryboardPanelRepository,
    StoryboardRepository,
)
from src.database.rls import POLICY_NAME, RLS_JOIN_PREDICATES_PANEL_ASSETS
from tests.conftest import require_createrole

_TABLE = "storyboard_panel_assets"


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
async def test_panel_assets_table_has_rls_enabled_not_forced(db_session) -> None:
    row = await _rls_flags(db_session, _TABLE)
    assert row.relrowsecurity is True, f"{_TABLE} must have ROW LEVEL SECURITY enabled"
    assert row.relforcerowsecurity is False, (
        f"{_TABLE} must NOT FORCE RLS (owner stays inert this increment)"
    )


@pytest.mark.asyncio
async def test_panel_assets_table_has_exactly_one_tenant_isolation_policy(db_session) -> None:
    names = await _policy_names(db_session, _TABLE)
    assert names == [POLICY_NAME], (
        f"expected exactly one {POLICY_NAME} policy on {_TABLE}, got {names}"
    )


def test_panel_assets_rls_scope_is_locked() -> None:
    # Lock the separate panel-assets set so the join table can't silently ship without
    # its own RLS decision (parallel to test_storyboard_events_rls' scope lock), and so
    # nobody folds it back into a map a HISTORICAL migration iterates.
    assert set(RLS_JOIN_PREDICATES_PANEL_ASSETS) == {"storyboard_panel_assets"}


async def _seed_panel_with_asset(session, tenant_id: UUID) -> dict[str, UUID]:
    """One tenant's project → storyboard → panel → asset → ONE join row."""
    project = Project(
        id=uuid7(),
        tenant_id=tenant_id,
        owner_subject_id=uuid7(),
        visibility="tenant",
        type="generic",
        title="panel-assets-rls",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    session.add(project)
    await session.flush()

    storyboard = await StoryboardRepository(session).ensure_for_project(
        project.id, tenant_id=tenant_id
    )
    assert storyboard is not None
    asset = await StoryboardAssetRepository(session).create(
        project_id=project.id, position=0, kind="character", name="Hero"
    )
    panel = await StoryboardPanelRepository(session).create(
        project_id=project.id,
        storyboard_id=storyboard.id,
        position=0,
        shot_key="clip-1",
        vo_text="one",
        characters=[asset.id],
    )
    ids = {"project": project.id, "panel": panel.id, "asset": asset.id}
    await session.commit()
    return ids


@pytest.mark.asyncio
async def test_panel_assets_rls_isolates_tenants_under_restricted_role(db_session_factory) -> None:
    """The 2-hop predicate must actually ISOLATE, not merely exist.

    ``ENABLE`` (not ``FORCE``) means the table owner — which every other test connects
    as — bypasses the policy, so isolation is invisible unless it is exercised under a
    non-owner ``NOBYPASSRLS`` role. That is precisely the prod runtime posture
    (``w2v_orch_app``). ``SET LOCAL ROLE`` + a transaction-local GUC, rolled back, so
    the owner-bypass the rest of the suite relies on is untouched.
    """
    role = f"pa_rls_probe_{uuid7().hex[:12]}"

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
            ids_a = await _seed_panel_with_asset(owner, tenant_a)
            ids_b = await _seed_panel_with_asset(owner, tenant_b)

        # ── 1. GUC=A under the restricted role → ONLY tenant A's join row ──────
        async with db_session_factory() as s:
            await s.execute(text(f'SET LOCAL ROLE "{role}"'))
            await s.execute(
                text("SELECT set_config('app.current_tenant', :t, true)"), {"t": str(tenant_a)}
            )
            rows = (
                await s.execute(text(f"SELECT panel_id FROM {_TABLE}"))  # noqa: S608
            ).scalars()
            assert list(rows) == [ids_a["panel"]], "tenant B's join row must be invisible to A"
            await s.rollback()

        # ── 2. GUC=B → tenant A's row is unreachable (the SELECT gate) ─────────
        async with db_session_factory() as s:
            await s.execute(text(f'SET LOCAL ROLE "{role}"'))
            await s.execute(
                text("SELECT set_config('app.current_tenant', :t, true)"), {"t": str(tenant_b)}
            )
            count = (
                await s.execute(
                    text(f"SELECT count(*) FROM {_TABLE} WHERE panel_id = :p"),  # noqa: S608
                    {"p": str(ids_a["panel"])},
                )
            ).scalar_one()
            assert count == 0, "tenant B must not SELECT tenant A's join rows"
            await s.rollback()

        # ── 3. No GUC → fail-closed (0 rows, not all rows) ────────────────────
        async with db_session_factory() as s:
            await s.execute(text(f'SET LOCAL ROLE "{role}"'))
            count = (
                await s.execute(text(f"SELECT count(*) FROM {_TABLE}"))  # noqa: S608
            ).scalar_one()
            assert count == 0, "an unset GUC must be fail-closed on the join table"
            await s.rollback()

        # ── 4. WITH CHECK rejects INSERTing a row onto tenant A's panel ───────
        # This is the attack the join table exists to make impossible: attaching an
        # asset to somebody else's panel. Both parents exist (seeded by the owner), so
        # the FK is satisfiable — only the policy stands in the way.
        async with db_session_factory() as s:
            await s.execute(text(f'SET LOCAL ROLE "{role}"'))
            await s.execute(
                text("SELECT set_config('app.current_tenant', :t, true)"), {"t": str(tenant_b)}
            )
            with pytest.raises(ProgrammingError) as caught:
                await s.execute(
                    text(
                        f"INSERT INTO {_TABLE} (id, panel_id, asset_id, role, position) "  # noqa: S608
                        "VALUES (:id, :panel, :asset, 'character', 0)"
                    ),
                    {
                        "id": str(uuid7()),
                        "panel": str(ids_a["panel"]),
                        "asset": str(ids_a["asset"]),
                    },
                )
            assert "row-level security" in str(caught.value.orig).lower()
            await s.rollback()

        # Sanity: as the owner, both tenants' rows are intact (nothing was written).
        async with db_session_factory() as owner:
            total = (
                await owner.execute(text(f"SELECT count(*) FROM {_TABLE}"))  # noqa: S608
            ).scalar_one()
            assert total == 2, "one join row per tenant, and no smuggled third"
            assert ids_b["panel"] != ids_a["panel"]
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
