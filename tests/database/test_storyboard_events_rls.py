"""RLS gate for the storyboard approval trail (``storyboard_panel_events``).

The append-only trail is project-scoped and reaches its tenant via the
``project_id -> projects.tenant_id`` FK chain, so it carries the SAME join-based
``tenant_isolation`` policy every other storyboard table does — ENABLEd, not
FORCEd (owner stays inert this increment). It is registered SEPARATELY
(``RLS_JOIN_PREDICATES_LATE_EVENTS`` + ``register_late_join_events_rls_ddl``) from
the 0046 storyboard set so historical migration 0046 keeps iterating only its own
tables; this test proves the ``create_all`` (test/dev) apply path lands the policy.

Introspection is scoped to ``current_schema()`` because CI runs each job in its
own schema inside a SHARED postgres — a bare ``tablename`` match against the
cluster-wide ``pg_policies`` would hit every schema's copy. Mirrors
``test_rls_policies.py``.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from src.database.rls import POLICY_NAME, RLS_JOIN_PREDICATES_LATE_EVENTS

_TABLE = "storyboard_panel_events"


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
async def test_events_table_has_rls_enabled_not_forced(db_session) -> None:
    row = await _rls_flags(db_session, _TABLE)
    assert row.relrowsecurity is True, f"{_TABLE} must have ROW LEVEL SECURITY enabled"
    assert row.relforcerowsecurity is False, (
        f"{_TABLE} must NOT FORCE RLS (owner stays inert this increment)"
    )


@pytest.mark.asyncio
async def test_events_table_has_exactly_one_tenant_isolation_policy(db_session) -> None:
    names = await _policy_names(db_session, _TABLE)
    assert names == [POLICY_NAME], (
        f"expected exactly one {POLICY_NAME} policy on {_TABLE}, got {names}"
    )


def test_events_rls_scope_is_locked() -> None:
    # Lock the separate late-events set so the trail can't silently ship without
    # its own RLS decision (parallel to test_rls_policies' scope lock).
    assert set(RLS_JOIN_PREDICATES_LATE_EVENTS) == {"storyboard_panel_events"}
