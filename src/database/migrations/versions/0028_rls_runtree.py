"""RLS tenant-isolation on the run-tree + verbatim_items — Phase D (increment 2)

ENABLE (not FORCE) ROW LEVEL SECURITY + a ``tenant_isolation`` policy on the
tables that reach their tenant only via an FK chain to ``projects``:
``deliverables``, ``verbatim_items`` (1-hop), ``runs`` (2-hop), ``stage_outputs``
/ ``run_usage`` (3-hop), ``annotations`` (4-hop). Each policy is a join-based
``EXISTS`` that climbs to ``projects.tenant_id`` (see ``src.database.rls``).

Like 0027 (``projects``), this is the SAFE additive part: because the
orchestrator's DB user owns the tables and we do NOT FORCE, the policies are
INERT for the app today (the owner bypasses non-forced RLS) — no behaviour
change, no schema/backfill/write-path change. They only take effect for a future
no-``BYPASSRLS`` role + FORCE (a later Phase D increment), at which point
``SET LOCAL app.current_tenant`` (gated by ``TENANT_RLS_MODE``, default ``off``)
scopes every query.

The DDL is a FROZEN inline snapshot of ``src.database.rls`` output so this migration and the SQLAlchemy
``after_create`` event (which gives the same RLS to a ``create_all``-ed test/dev
schema) can never drift. Fully reversible.

Deferred to later increments: ``FORCE``, the no-``BYPASSRLS`` role, the
background-session GUC, and denormalizing ``tenant_id`` onto these tables (only
if the deep ``EXISTS`` ever needs the perf at FORCE time).

Revision ID: 0028
Revises: 0027
Create Date: 2026-06-16
"""

from alembic import op


revision = "0028"
down_revision = "0027"
branch_labels = None
depends_on = None

# FROZEN DDL — inlined snapshot of ``src.database.rls.join_downgrade_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_JOIN_DOWNGRADE_STATEMENTS: tuple[str, ...] = (
    "DROP POLICY IF EXISTS tenant_isolation ON deliverables",
    "ALTER TABLE deliverables DISABLE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_isolation ON verbatim_items",
    "ALTER TABLE verbatim_items DISABLE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_isolation ON runs",
    "ALTER TABLE runs DISABLE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_isolation ON stage_outputs",
    "ALTER TABLE stage_outputs DISABLE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_isolation ON run_usage",
    "ALTER TABLE run_usage DISABLE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_isolation ON annotations",
    "ALTER TABLE annotations DISABLE ROW LEVEL SECURITY",
)


# FROZEN DDL — inlined snapshot of ``src.database.rls.join_upgrade_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_JOIN_UPGRADE_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE deliverables ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON deliverables USING (EXISTS (SELECT 1 FROM projects p WHERE p.id = deliverables.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM projects p WHERE p.id = deliverables.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
    "ALTER TABLE verbatim_items ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON verbatim_items USING (EXISTS (SELECT 1 FROM projects p WHERE p.id = verbatim_items.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM projects p WHERE p.id = verbatim_items.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
    "ALTER TABLE runs ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON runs USING (EXISTS (SELECT 1 FROM deliverables d JOIN projects p ON p.id = d.project_id WHERE d.id = runs.deliverable_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM deliverables d JOIN projects p ON p.id = d.project_id WHERE d.id = runs.deliverable_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
    "ALTER TABLE stage_outputs ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON stage_outputs USING (EXISTS (SELECT 1 FROM runs r JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id WHERE r.id = stage_outputs.run_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM runs r JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id WHERE r.id = stage_outputs.run_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
    "ALTER TABLE run_usage ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON run_usage USING (EXISTS (SELECT 1 FROM runs r JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id WHERE r.id = run_usage.run_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM runs r JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id WHERE r.id = run_usage.run_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
    "ALTER TABLE annotations ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON annotations USING (EXISTS (SELECT 1 FROM stage_outputs so JOIN runs r ON r.id = so.run_id JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id WHERE so.id = annotations.stage_output_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM stage_outputs so JOIN runs r ON r.id = so.run_id JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id WHERE so.id = annotations.stage_output_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
)


def upgrade() -> None:
    for stmt in _JOIN_UPGRADE_STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in _JOIN_DOWNGRADE_STATEMENTS:
        op.execute(stmt)
