"""RLS tenant-isolation on projects — Phase D backstop (safe first increment)

ENABLE (not FORCE) ROW LEVEL SECURITY + a ``tenant_isolation`` policy on
``projects`` — the only table that carries ``tenant_id`` directly. Because the
orchestrator's DB user owns the table and we do NOT FORCE, the policy is INERT
for the app today (the owner bypasses non-forced RLS): no behaviour change. It
only takes effect for a future no-``BYPASSRLS`` role + FORCE (a later Phase D
increment), at which point ``SET LOCAL app.current_tenant`` (gated by
``TENANT_RLS_MODE``, default ``off``) scopes every query.

The DDL is a FROZEN inline snapshot of ``src.database.rls`` output so this migration and the
SQLAlchemy ``after_create`` event (which gives the same RLS to a ``create_all``-ed
test/dev schema) can never drift. Fully reversible.

Deferred to later increments: the run-tree + ``verbatim_items`` tables (no direct
``tenant_id`` → join/denormalized policies), ``FORCE``, and the restricted role.

Revision ID: 0027
Revises: 0026
Create Date: 2026-06-15
"""

from alembic import op


revision = "0027"
down_revision = "0026"
branch_labels = None
depends_on = None

# FROZEN DDL — inlined snapshot of ``src.database.rls.downgrade_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_DOWNGRADE_STATEMENTS: tuple[str, ...] = (
    "DROP POLICY IF EXISTS tenant_isolation ON projects",
    "ALTER TABLE projects DISABLE ROW LEVEL SECURITY",
)


# FROZEN DDL — inlined snapshot of ``src.database.rls.upgrade_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_UPGRADE_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE projects ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON projects USING (tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid) WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)",
)


def upgrade() -> None:
    for stmt in _UPGRADE_STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in _DOWNGRADE_STATEMENTS:
        op.execute(stmt)
