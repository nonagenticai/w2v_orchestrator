"""Phase D RLS — PROVABLE tenant isolation (the spec §8 backstop test).

The anti-drift test (``test_rls_policies.py``) proves the policies *exist*. This
one proves they *isolate*. The shipped policies are ``ENABLE`` (not ``FORCE``) and
the app's own DB user owns the tables → the owner bypasses them, so every other
test sees all rows. That is exactly why isolation has never been exercised.

Postgres RLS ``ENABLE`` binds every role that is NOT the table owner and lacks
``BYPASSRLS``/``SUPERUSER`` — which is precisely the posture the runtime role will
have once Phase D increment 3 lands the restricted role. ``FORCE`` only adds
*owner*-subjection, so we don't need it here: we ``SET LOCAL ROLE`` to a throwaway
non-owner, ``NOBYPASSRLS`` role and prove the already-shipped predicates enforce.
This validates the #266 (``projects`` direct) and #270 (run-tree / ``verbatim_items``
join-``EXISTS``) policies BEFORE they ever gate prod.

Single connection, no new engine/password/PgBouncer: ``SET LOCAL ROLE`` +
``set_config('app.current_tenant', …, is_local=true)`` inside a transaction, rolled
back so role/GUC reset cleanly and the owner-bypass other tests rely on is intact.

Requires ``CREATEROLE`` on the test DB user (the CI ``postgres:18`` ``ci_test``
superuser has it, as does a local ``postgres`` superuser). Under CI this MUST
run — ``require_createrole`` fails loudly if the privilege is ever missing so a
future infra change can't silently drop tenant-isolation coverage; locally it
skips for a developer connected as the restricted runtime role.
"""

from __future__ import annotations

from uuid import UUID, uuid7

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from src.database.models import (
    Annotation,
    Deliverable,
    Project,
    Run,
    RunUsage,
    StageOutput,
    VerbatimItem,
)
from tests.conftest import require_createrole

# (table, the column that proves which tenant a row belongs to via its tree).
# Every one must return ONLY the active tenant's row under the GUC.
_RLS_TABLES = (
    "projects",
    "deliverables",
    "verbatim_items",
    "runs",
    "stage_outputs",
    "run_usage",
    "annotations",
)


async def _seed_tenant(session, tenant_id: UUID) -> dict[str, UUID]:
    """Insert one full run-tree (project→deliverable→run→stage_output→
    run_usage→annotation, + verbatim_item) for ``tenant_id``. Returns the ids."""
    subject = uuid7()
    project = Project(
        id=uuid7(),
        tenant_id=tenant_id,
        owner_subject_id=subject,
        visibility="private",
        type="promo",
        title=f"proj-{tenant_id}",
    )
    deliverable = Deliverable(id=uuid7(), project_id=project.id, name="d1", format="16x9")
    run = Run(id=uuid7(), deliverable_id=deliverable.id, blueprint_id="promo_v1")
    stage = StageOutput(id=uuid7(), run_id=run.id, stage_name="analyze", status="completed")
    usage = RunUsage(id=uuid7(), run_id=run.id, stage_name="analyze")
    annotation = Annotation(id=uuid7(), stage_output_id=stage.id, text="note")
    verbatim = VerbatimItem(id=uuid7(), project_id=project.id, text="verbatim", position=0)
    session.add_all([project, deliverable, run, stage, usage, annotation, verbatim])
    await session.commit()
    return {
        "project": project.id,
        "deliverable": deliverable.id,
        "run": run.id,
        "stage_output": stage.id,
        "run_usage": usage.id,
        "annotation": annotation.id,
        "verbatim_item": verbatim.id,
    }


async def _count(session, table: str) -> int:
    # Static identifier from the hard-coded _RLS_TABLES list — no user input.
    return (await session.execute(text(f"SELECT count(*) FROM {table}"))).scalar_one()  # noqa: S608


@pytest.mark.asyncio
async def test_rls_policies_isolate_tenants_under_restricted_role(db_session_factory) -> None:
    role = f"rls_probe_{uuid7().hex[:12]}"

    # ── setup: throwaway non-owner, NOBYPASSRLS role + grants (as the owner) ──
    async with db_session_factory() as owner:
        schema = (await owner.execute(text("SELECT current_schema()"))).scalar_one()
        try:
            await owner.execute(text(f'CREATE ROLE "{role}" NOLOGIN NOSUPERUSER NOBYPASSRLS'))
        except ProgrammingError as exc:  # insufficient_privilege → role lacks CREATEROLE
            await owner.rollback()
            require_createrole(exc)  # fail under CI, skip for a locally-restricted role
        # Membership so a non-superuser owner can SET ROLE into it; grants so the
        # role can actually read/write the tables once RLS lets a row through.
        await owner.execute(text(f'GRANT "{role}" TO CURRENT_USER'))
        await owner.execute(text(f'GRANT USAGE ON SCHEMA "{schema}" TO "{role}"'))
        await owner.execute(
            text(
                f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA "{schema}" TO "{role}"'
            )
        )
        await owner.commit()

    try:
        # Seed two tenants' full trees as the owner (RLS ENABLE doesn't bind the owner).
        tenant_a, tenant_b = uuid7(), uuid7()
        async with db_session_factory() as owner:
            ids_a = await _seed_tenant(owner, tenant_a)
            ids_b = await _seed_tenant(owner, tenant_b)

        # ── 1. GUC=A under the restricted role → ONLY tenant A's rows everywhere ──
        async with db_session_factory() as s:
            await s.execute(text(f'SET LOCAL ROLE "{role}"'))
            await s.execute(
                text("SELECT set_config('app.current_tenant', :t, true)"), {"t": str(tenant_a)}
            )
            for table in _RLS_TABLES:
                assert await _count(s, table) == 1, f"{table}: expected only tenant A's row"
            visible = (await s.execute(text("SELECT id FROM projects"))).scalar_one()
            assert visible == ids_a["project"], "the one visible project must be tenant A's"
            await s.rollback()

        # ── 2. No GUC under the restricted role → fail-closed (0 rows everywhere) ──
        async with db_session_factory() as s:
            await s.execute(text(f'SET LOCAL ROLE "{role}"'))
            for table in _RLS_TABLES:
                assert await _count(s, table) == 0, f"{table}: unset GUC must be fail-closed"
            await s.rollback()

        # ── 3. WITH CHECK rejects writing another tenant's row ──
        async with db_session_factory() as s:
            await s.execute(text(f'SET LOCAL ROLE "{role}"'))
            await s.execute(
                text("SELECT set_config('app.current_tenant', :t, true)"), {"t": str(tenant_a)}
            )
            with pytest.raises(ProgrammingError) as caught:
                await s.execute(
                    text(
                        "INSERT INTO projects "
                        "(id, tenant_id, owner_subject_id, visibility, type, title, "
                        " brief, brand, voice, theme, status) "
                        "VALUES (:id, :t, :subj, 'private', 'promo', 'evil', "
                        " '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, 'drafting')"
                    ),
                    {"id": str(uuid7()), "t": str(tenant_b), "subj": str(uuid7())},
                )
            assert "row-level security" in str(caught.value.orig).lower()
            await s.rollback()

        # ── 4. Cross-tenant UPDATE / DELETE under GUC=A touch 0 of tenant B's rows ──
        async with db_session_factory() as s:
            await s.execute(text(f'SET LOCAL ROLE "{role}"'))
            await s.execute(
                text("SELECT set_config('app.current_tenant', :t, true)"), {"t": str(tenant_a)}
            )
            upd = await s.execute(
                text("UPDATE projects SET title = 'hijacked' WHERE id = :id"),
                {"id": str(ids_b["project"])},
            )
            assert upd.rowcount == 0, "must not be able to UPDATE another tenant's project"
            dele = await s.execute(
                text("DELETE FROM runs WHERE id = :id"), {"id": str(ids_b["run"])}
            )
            assert dele.rowcount == 0, "must not be able to DELETE another tenant's run"
            await s.rollback()

        # Sanity: tenant B's data is intact when viewed as the owner.
        async with db_session_factory() as owner:
            assert await _count(owner, "projects") == 2
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


@pytest.mark.asyncio
async def test_admin_rollup_is_cross_tenant_under_restricted_role(db_session_factory) -> None:
    """The platform tenants list is the ONE genuinely cross-tenant admin read, so
    it goes through the ``admin_tenant_rollup()`` SECURITY DEFINER resolver
    (Phase D increment 4). Prove that under the restricted no-``BYPASSRLS`` role:
      * a direct ``GROUP BY projects.tenant_id`` with no GUC is fail-closed (0), but
      * ``admin_tenant_rollup()`` still returns EVERY tenant's rollup (the resolver
        bypasses RLS for this trusted super-admin read), while
      * a per-tenant read (GUC bound) is scoped to exactly that tenant —
    which is the exact split ``admin_repository`` relies on under enforce."""
    role = f"rls_admin_probe_{uuid7().hex[:12]}"

    async with db_session_factory() as owner:
        schema = (await owner.execute(text("SELECT current_schema()"))).scalar_one()
        try:
            await owner.execute(text(f'CREATE ROLE "{role}" NOLOGIN NOSUPERUSER NOBYPASSRLS'))
        except ProgrammingError as exc:
            await owner.rollback()
            require_createrole(exc)  # fail under CI, skip for a locally-restricted role
        await owner.execute(text(f'GRANT "{role}" TO CURRENT_USER'))
        await owner.execute(text(f'GRANT USAGE ON SCHEMA "{schema}" TO "{role}"'))
        await owner.execute(text(f'GRANT SELECT ON ALL TABLES IN SCHEMA "{schema}" TO "{role}"'))
        # EXECUTE is REVOKEd from PUBLIC on the resolver; the cutover grants it to
        # the runtime role — mirror that here so the probe role can call it.
        await owner.execute(text(f'GRANT EXECUTE ON FUNCTION admin_tenant_rollup() TO "{role}"'))
        await owner.commit()

    try:
        tenant_a, tenant_b = uuid7(), uuid7()
        async with db_session_factory() as owner:
            await _seed_tenant(owner, tenant_a)
            await _seed_tenant(owner, tenant_b)

        async with db_session_factory() as s:
            await s.execute(text(f'SET LOCAL ROLE "{role}"'))
            # No GUC: a direct per-tenant aggregate is fail-closed (RLS hides all).
            direct = (
                await s.execute(text("SELECT count(DISTINCT tenant_id) FROM projects"))
            ).scalar_one()
            assert direct == 0, "direct cross-tenant read must be fail-closed without a GUC"
            # But the resolver returns BOTH tenants' rollups, each with project_count 1.
            rows = (
                await s.execute(
                    text(
                        "SELECT tenant_id, project_count, run_count "
                        "FROM admin_tenant_rollup() ORDER BY tenant_id"
                    )
                )
            ).all()
            seen = {r.tenant_id: (r.project_count, r.run_count) for r in rows}
            assert set(seen) == {tenant_a, tenant_b}, "rollup must see every tenant"
            assert seen[tenant_a] == (1, 1) and seen[tenant_b] == (1, 1)
            await s.rollback()

        # A per-tenant read (GUC=A) under the role is scoped to exactly tenant A —
        # the path the per-tenant admin aggregates (tenant_overview/usage) take.
        async with db_session_factory() as s:
            await s.execute(text(f'SET LOCAL ROLE "{role}"'))
            await s.execute(
                text("SELECT set_config('app.current_tenant', :t, true)"), {"t": str(tenant_a)}
            )
            scoped = (await s.execute(text("SELECT count(*) FROM projects"))).scalar_one()
            assert scoped == 1, "per-tenant read must be scoped to the bound tenant"
            await s.rollback()
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
