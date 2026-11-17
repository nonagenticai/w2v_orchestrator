"""Phase D RLS — the background-session bypass resolvers + GUC bind helper.

``orphan_runs()`` / ``run_tenant()`` are the SECURITY DEFINER functions the
reconciler and dispatcher use to bootstrap under enforce (see
``src/database/rls.py``). These tests run as the table owner (no role needed →
they run in CI too) and prove the functions return the right cross-tenant data,
plus that ``bind_session_tenant`` honours ``TENANT_RLS_MODE``.
"""

from __future__ import annotations

from uuid import UUID, uuid7

import pytest
from sqlalchemy import text

from src.config import clear_settings_cache
from src.database.models import Deliverable, Project, Run
from src.database.rls import video_source_functions_statements
from src.database.session import bind_session_tenant


async def _seed_run(session, tenant_id: UUID, status: str) -> UUID:
    project = Project(
        id=uuid7(),
        tenant_id=tenant_id,
        owner_subject_id=uuid7(),
        visibility="private",
        type="promo",
        title="p",
    )
    deliverable = Deliverable(id=uuid7(), project_id=project.id, name="d", format="16x9")
    run = Run(id=uuid7(), deliverable_id=deliverable.id, blueprint_id="promo_v1", status=status)
    session.add_all([project, deliverable, run])
    await session.commit()
    return run.id


@pytest.mark.asyncio
async def test_orphan_runs_returns_pending_running_across_tenants(db_session) -> None:
    tenant_a, tenant_b = uuid7(), uuid7()
    running_a = await _seed_run(db_session, tenant_a, "running")
    pending_b = await _seed_run(db_session, tenant_b, "pending")
    await _seed_run(db_session, tenant_a, "completed")  # must NOT appear

    rows = (
        await db_session.execute(text("SELECT run_id, tenant_id, status FROM orphan_runs()"))
    ).all()
    by_run = {r.run_id: r for r in rows}

    assert set(by_run) == {running_a, pending_b}, "only pending/running runs, both tenants"
    assert by_run[running_a].tenant_id == tenant_a
    assert by_run[pending_b].tenant_id == tenant_b


@pytest.mark.asyncio
async def test_run_tenant_resolves_owning_tenant(db_session) -> None:
    tenant = uuid7()
    run_id = await _seed_run(db_session, tenant, "pending")

    resolved = (
        await db_session.execute(text("SELECT run_tenant(:rid)"), {"rid": str(run_id)})
    ).scalar_one()
    assert resolved == tenant

    missing = (
        await db_session.execute(text("SELECT run_tenant(:rid)"), {"rid": str(uuid7())})
    ).scalar_one_or_none()
    assert missing is None


@pytest.mark.asyncio
async def test_bind_session_tenant_noop_when_mode_off(db_session, monkeypatch) -> None:
    monkeypatch.setenv("TENANT_RLS_MODE", "off")
    clear_settings_cache()
    await bind_session_tenant(db_session, uuid7())
    guc = (
        await db_session.execute(text("SELECT current_setting('app.current_tenant', true)"))
    ).scalar_one_or_none()
    # Never set → NULL (or '' if a prior txn on this pooled conn touched it); the
    # point is the helper itself did NOT bind a tenant when mode is off.
    assert not guc


@pytest.mark.asyncio
async def test_bind_session_tenant_sets_guc_when_enforce(db_session, monkeypatch) -> None:
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    clear_settings_cache()
    tenant = uuid7()
    await bind_session_tenant(db_session, tenant)
    guc = (
        await db_session.execute(text("SELECT current_setting('app.current_tenant', true)"))
    ).scalar_one()
    assert guc == str(tenant)
    clear_settings_cache()


# --- the preview-proxy sweeper's resolver ------------------------------------
#
# `pending_video_sources()` is the third SECURITY DEFINER bypass in this file's
# family, and it exists for the same reason as `orphan_runs()`: a background
# sweep has no tenant bound, so under enforce it sees zero rows and the worker
# silently does nothing forever — the stranded-state class this project keeps
# rediscovering.
#
# `prosecdef` is asserted from `pg_proc`, NOT inferred from the fact that a query
# returned rows. These tests run as the table OWNER, who bypasses RLS anyway, so
# an owner-run "it returned rows" assertion would pass just as happily against a
# SECURITY INVOKER function that is broken in production.
#
# ⚠️ TWO LEVELS, AND BOTH ARE NEEDED — measured, not assumed. Mutating
# `_FN_PREAMBLE` to SECURITY INVOKER, and separately deleting the REVOKE, left
# the two `pg_proc` tests below GREEN: this suite's schema is not rebuilt from
# source between runs, so those queries read a schema created earlier and cannot
# see source drift at all. On their own they are vacuous against exactly the
# regression they appear to guard.
#
# So the DDL text is asserted directly as well. `test_video_source_function_ddl_*`
# is what actually goes red when someone edits the preamble or drops the REVOKE;
# the `pg_proc` pair is what goes red when the DDL is right but never applied to
# a live schema. Neither subsumes the other.


def test_video_source_function_ddl_declares_security_definer() -> None:
    """MUTANT: `_FN_PREAMBLE` -> SECURITY INVOKER.

    The source-level gate. Without SECURITY DEFINER the sweeper sees zero rows
    under enforce and every preview stays `pending` forever — indistinguishable,
    from outside, from the missing-worker bug this feature exists to fix.
    """
    ddl = " ".join(video_source_functions_statements())
    assert "CREATE OR REPLACE FUNCTION pending_video_sources" in ddl
    assert "SECURITY DEFINER" in ddl
    assert "SECURITY INVOKER" not in ddl


def test_video_source_function_ddl_revokes_public_execute() -> None:
    """MUTANT: delete the REVOKE statement.

    A cross-tenant resolver executable by PUBLIC hands every one of the ~60 login
    roles a read of every tenant's source object keys.
    """
    statements = video_source_functions_statements()
    assert any(
        "REVOKE EXECUTE ON FUNCTION pending_video_sources(integer) FROM PUBLIC" in s
        for s in statements
    ), statements
    # …and the REVOKE must come AFTER the CREATE, or it revokes nothing.
    create_at = next(i for i, s in enumerate(statements) if s.startswith("CREATE OR REPLACE"))
    revoke_at = next(i for i, s in enumerate(statements) if s.startswith("REVOKE EXECUTE"))
    assert create_at < revoke_at, statements


@pytest.mark.asyncio
async def test_pending_video_sources_is_security_definer(db_session) -> None:
    """MUTANT: drop SECURITY DEFINER from the function body.

    Without it the sweeper sees nothing under enforce and every preview stays
    `pending` — indistinguishable, from the outside, from the missing-worker bug
    this whole feature exists to fix.
    """
    row = (
        await db_session.execute(
            text(
                "SELECT p.prosecdef FROM pg_proc p "
                "JOIN pg_namespace n ON n.oid = p.pronamespace "
                "WHERE p.proname = 'pending_video_sources'"
            )
        )
    ).first()
    assert row is not None, "pending_video_sources() is not installed on this schema"
    assert row[0] is True, "pending_video_sources() is not SECURITY DEFINER"


@pytest.mark.asyncio
async def test_pending_video_sources_is_not_executable_by_public(db_session) -> None:
    """MUTANT: drop the REVOKE. A cross-tenant resolver granted to PUBLIC hands
    every login role a read of every tenant's source keys."""
    granted = (
        await db_session.execute(
            text(
                "SELECT has_function_privilege('public', p.oid, 'EXECUTE') "
                "FROM pg_proc p WHERE p.proname = 'pending_video_sources'"
            )
        )
    ).scalar()
    assert granted is False, "pending_video_sources() is EXECUTE-able by PUBLIC"
