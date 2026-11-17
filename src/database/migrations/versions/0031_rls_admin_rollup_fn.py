"""RLS admin cross-tenant rollup resolver — Phase D (increment 4)

Adds one ``SECURITY DEFINER`` function, ``admin_tenant_rollup()``, the platform
tenants list needs once the runtime app connects as the restricted
no-``BYPASSRLS`` role under enforce. The admin/platform surface is excluded from
the per-request ``apply_tenant_rls`` gate and reads RLS-protected tables
(projects/runs/run_usage) for super-admins; every admin read but the tenants
list operates on a KNOWN tenant_id and binds that tenant's GUC instead (see
``admin_repository`` + ``bind_session_tenant``). The tenants list is genuinely
cross-tenant, so it goes through this resolver — the same pattern as
``run_tenant``/``orphan_runs`` (migration 0030).

Owned by the migration user (the postgres superuser in prod → bypasses FORCE; the
table owner in CI/tests → bypasses ENABLE-not-FORCE) so the body sees every
tenant. ``EXECUTE`` is revoked from PUBLIC; the cutover grants it to
``w2v_orch_app`` out-of-band (gitops ``phase-d-rls-stage-a.sql``). CI-safe: a
non-superuser owner (``ci_test``) can create a function it owns, and no
role/FORCE DDL is involved. Inert until the runtime role switch — the data-plane
policies are UNCHANGED (no "bypass GUC" disjunct), so provable isolation holds.

DDL is a FROZEN SNAPSHOT of the function as it existed at this revision. It was
originally single-sourced from ``src.database.rls.admin_functions_statements``,
but that generator later evolved to reference ``projects.is_branch`` (added in
``0052``) to exclude storyboard branch projects from per-tenant counts. Replaying
this migration from scratch through the live generator would then emit
``WHERE p.is_branch = false`` against a ``projects`` table that has no such column
yet. Migrations must reproduce the historical schema, so the body is inlined here;
the branch-excluding body is applied forward in ``0054`` (and to ``create_all`` via
the live generator). See also ``0035`` (admin run views) which still single-sources
its own — unchanged — generator.

Revision ID: 0031
Revises: 0030
Create Date: 2026-06-16
"""

from alembic import op

revision = "0031"
down_revision = "0030"
branch_labels = None
depends_on = None

# Frozen 2026-07-18: exact statements ``admin_functions_statements()`` returned at
# this revision (pre-``is_branch`` filter). Do NOT re-import the live generator.
_ADMIN_TENANT_ROLLUP_FN = (
    "CREATE OR REPLACE FUNCTION admin_tenant_rollup() "
    "RETURNS TABLE (tenant_id uuid, project_count bigint, last_activity timestamptz, "
    "run_count bigint, cost_usd numeric) "
    "LANGUAGE sql STABLE SECURITY DEFINER SET search_path FROM CURRENT AS $$ "
    "SELECT pa.tenant_id, pa.project_count, pa.last_activity, "
    "COALESCE(ua.run_count, 0) AS run_count, COALESCE(ua.cost_usd, 0) AS cost_usd "
    "FROM (SELECT p.tenant_id, count(p.id) AS project_count, "
    "max(COALESCE(p.updated_at, p.created_at)) AS last_activity "
    "FROM projects p GROUP BY p.tenant_id) pa "
    "LEFT JOIN (SELECT p.tenant_id, count(DISTINCT r.id) AS run_count, "
    "COALESCE(sum(ru.cost_usd), 0) AS cost_usd FROM runs r "
    "JOIN deliverables d ON d.id = r.deliverable_id "
    "JOIN projects p ON p.id = d.project_id "
    "LEFT JOIN run_usage ru ON ru.run_id = r.id GROUP BY p.tenant_id) ua "
    "ON ua.tenant_id = pa.tenant_id $$"
)
_ADMIN_ROLLUP_STATEMENTS = (
    _ADMIN_TENANT_ROLLUP_FN,
    "REVOKE EXECUTE ON FUNCTION admin_tenant_rollup() FROM PUBLIC",
)


def upgrade() -> None:
    for stmt in _ADMIN_ROLLUP_STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS admin_tenant_rollup()")
