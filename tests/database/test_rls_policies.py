"""Phase D RLS anti-drift — the tenant-isolation drift gate.

Asserts the SAFE increments are exactly that: every tenant-scoped table has RLS
ENABLEd + a ``tenant_isolation`` policy, but is NOT FORCEd (so the owner still
bypasses → inert today), and the ``TENANT_RLS_MODE`` default is ``off``. Covered:
``projects`` (direct ``tenant_id``, migration 0027) and the run-tree +
``verbatim_items`` tables (join-based ``EXISTS`` policies, migration 0028).

The test introspects the schema produced by ``Base.metadata.create_all`` (the
test harness), which applies the same RLS as the migrations via the shared
``src.database.rls`` ``after_create`` event — so this also proves the two apply
paths agree. Later increments extend this gate to FORCE + the no-BYPASSRLS role.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from src.config import Settings
from src.database.rls import POLICY_NAME, RLS_JOIN_TABLES, RLS_POLICY_TABLES

# Introspection is scoped to ``current_schema()``: CI runs each job in its own
# schema inside a SHARED postgres, so a bare ``relname='projects'`` against the
# cluster-wide ``pg_class`` matches every schema's copy (MultipleResultsFound).

_ALL_RLS_TABLES = RLS_POLICY_TABLES + RLS_JOIN_TABLES


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
@pytest.mark.parametrize("table", _ALL_RLS_TABLES)
async def test_table_has_rls_enabled_not_forced(db_session, table: str) -> None:
    row = await _rls_flags(db_session, table)
    assert row.relrowsecurity is True, f"{table} must have ROW LEVEL SECURITY enabled"
    assert row.relforcerowsecurity is False, (
        f"{table} must NOT FORCE RLS (owner stays inert this increment)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("table", _ALL_RLS_TABLES)
async def test_table_has_tenant_isolation_policy(db_session, table: str) -> None:
    names = await _policy_names(db_session, table)
    assert POLICY_NAME in names, f"expected a {POLICY_NAME} policy on {table}, got {names}"


def test_rls_table_scope_is_locked() -> None:
    # Direct-tenant table(s) carry a simple equality policy; the join tables reach
    # tenant via an FK chain to projects. Lock both sets so a new tenant-scoped
    # table can't silently ship without an RLS decision.
    assert RLS_POLICY_TABLES == ("projects",)
    assert set(RLS_JOIN_TABLES) == {
        "deliverables",
        "verbatim_items",
        "runs",
        "stage_outputs",
        "run_usage",
        "annotations",
    }


def test_tenant_rls_mode_defaults_off() -> None:
    assert Settings().tenant_rls_mode == "off"
