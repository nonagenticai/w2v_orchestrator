"""RLS fail-closed hardening — NULLIF the tenant GUC — Phase D (increment 3a)

0027/0028 created the ``tenant_isolation`` policies with the predicate
``… = current_setting('app.current_tenant', true)::uuid``. That is NOT fail-closed
on a pooled connection: a custom placeholder GUC bound with ``is_local=true``
reverts to the **empty string** (not NULL) once its transaction ends, so the next
un-bound transaction on the same PgBouncer-pooled connection reports ``''`` and
``''::uuid`` raises — a 500 instead of "0 rows". Surfaced by the new
provable-isolation test (``tests/database/test_rls_isolation.py``).

This migration re-CREATEs every policy with ``NULLIF(current_setting(…), '')::uuid``
so ``''`` folds back to NULL → predicate false → 0 rows. Semantics are otherwise
identical (a real UUID GUC behaves exactly as before). ENABLE/FORCE state is
untouched; still inert for the owner today. DDL frozen inline from ``src.database.rls``
(the upgrade) so the migration and the ``after_create`` test-schema path can't
drift; the downgrade inlines the legacy predicate to truly revert.

Revision ID: 0029
Revises: 0028
Create Date: 2026-06-16
"""

from alembic import op


revision = "0029"
down_revision = "0028"
branch_labels = None
depends_on = None

# FROZEN upgrade DDL — inlined snapshot of ``src.database.rls.rebuild_policies_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_REBUILD_POLICIES_STATEMENTS: tuple[str, ...] = (
    "DROP POLICY IF EXISTS tenant_isolation ON projects",
    "CREATE POLICY tenant_isolation ON projects USING (tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid) WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)",
    "DROP POLICY IF EXISTS tenant_isolation ON deliverables",
    "CREATE POLICY tenant_isolation ON deliverables USING (EXISTS (SELECT 1 FROM projects p WHERE p.id = deliverables.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM projects p WHERE p.id = deliverables.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
    "DROP POLICY IF EXISTS tenant_isolation ON verbatim_items",
    "CREATE POLICY tenant_isolation ON verbatim_items USING (EXISTS (SELECT 1 FROM projects p WHERE p.id = verbatim_items.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM projects p WHERE p.id = verbatim_items.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
    "DROP POLICY IF EXISTS tenant_isolation ON runs",
    "CREATE POLICY tenant_isolation ON runs USING (EXISTS (SELECT 1 FROM deliverables d JOIN projects p ON p.id = d.project_id WHERE d.id = runs.deliverable_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM deliverables d JOIN projects p ON p.id = d.project_id WHERE d.id = runs.deliverable_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
    "DROP POLICY IF EXISTS tenant_isolation ON stage_outputs",
    "CREATE POLICY tenant_isolation ON stage_outputs USING (EXISTS (SELECT 1 FROM runs r JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id WHERE r.id = stage_outputs.run_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM runs r JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id WHERE r.id = stage_outputs.run_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
    "DROP POLICY IF EXISTS tenant_isolation ON run_usage",
    "CREATE POLICY tenant_isolation ON run_usage USING (EXISTS (SELECT 1 FROM runs r JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id WHERE r.id = run_usage.run_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM runs r JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id WHERE r.id = run_usage.run_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
    "DROP POLICY IF EXISTS tenant_isolation ON annotations",
    "CREATE POLICY tenant_isolation ON annotations USING (EXISTS (SELECT 1 FROM stage_outputs so JOIN runs r ON r.id = so.run_id JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id WHERE so.id = annotations.stage_output_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM stage_outputs so JOIN runs r ON r.id = so.run_id JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id WHERE so.id = annotations.stage_output_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
)


# FROZEN downgrade DDL — the pre-NULLIF (legacy GUC) policies this revision
# replaced, as rendered from the 2026-10-05 table set. Same guard as above.
_LEGACY_POLICY_STATEMENTS: tuple[str, ...] = (
    "DROP POLICY IF EXISTS tenant_isolation ON projects",
    "CREATE POLICY tenant_isolation ON projects USING (tenant_id = current_setting('app.current_tenant', true)::uuid) WITH CHECK (tenant_id = current_setting('app.current_tenant', true)::uuid)",
    "DROP POLICY IF EXISTS tenant_isolation ON deliverables",
    "CREATE POLICY tenant_isolation ON deliverables USING (EXISTS (SELECT 1 FROM projects p WHERE p.id = deliverables.project_id AND p.tenant_id = current_setting('app.current_tenant', true)::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM projects p WHERE p.id = deliverables.project_id AND p.tenant_id = current_setting('app.current_tenant', true)::uuid))",
    "DROP POLICY IF EXISTS tenant_isolation ON verbatim_items",
    "CREATE POLICY tenant_isolation ON verbatim_items USING (EXISTS (SELECT 1 FROM projects p WHERE p.id = verbatim_items.project_id AND p.tenant_id = current_setting('app.current_tenant', true)::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM projects p WHERE p.id = verbatim_items.project_id AND p.tenant_id = current_setting('app.current_tenant', true)::uuid))",
    "DROP POLICY IF EXISTS tenant_isolation ON runs",
    "CREATE POLICY tenant_isolation ON runs USING (EXISTS (SELECT 1 FROM deliverables d JOIN projects p ON p.id = d.project_id WHERE d.id = runs.deliverable_id AND p.tenant_id = current_setting('app.current_tenant', true)::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM deliverables d JOIN projects p ON p.id = d.project_id WHERE d.id = runs.deliverable_id AND p.tenant_id = current_setting('app.current_tenant', true)::uuid))",
    "DROP POLICY IF EXISTS tenant_isolation ON stage_outputs",
    "CREATE POLICY tenant_isolation ON stage_outputs USING (EXISTS (SELECT 1 FROM runs r JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id WHERE r.id = stage_outputs.run_id AND p.tenant_id = current_setting('app.current_tenant', true)::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM runs r JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id WHERE r.id = stage_outputs.run_id AND p.tenant_id = current_setting('app.current_tenant', true)::uuid))",
    "DROP POLICY IF EXISTS tenant_isolation ON run_usage",
    "CREATE POLICY tenant_isolation ON run_usage USING (EXISTS (SELECT 1 FROM runs r JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id WHERE r.id = run_usage.run_id AND p.tenant_id = current_setting('app.current_tenant', true)::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM runs r JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id WHERE r.id = run_usage.run_id AND p.tenant_id = current_setting('app.current_tenant', true)::uuid))",
    "DROP POLICY IF EXISTS tenant_isolation ON annotations",
    "CREATE POLICY tenant_isolation ON annotations USING (EXISTS (SELECT 1 FROM stage_outputs so JOIN runs r ON r.id = so.run_id JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id WHERE so.id = annotations.stage_output_id AND p.tenant_id = current_setting('app.current_tenant', true)::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM stage_outputs so JOIN runs r ON r.id = so.run_id JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id WHERE so.id = annotations.stage_output_id AND p.tenant_id = current_setting('app.current_tenant', true)::uuid))",
)


def upgrade() -> None:
    for stmt in _REBUILD_POLICIES_STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in _LEGACY_POLICY_STATEMENTS:
        op.execute(stmt)
