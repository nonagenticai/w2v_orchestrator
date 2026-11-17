"""admin_tenant_rollup — exclude storyboard branch projects from per-tenant counts

The cross-tenant platform tenants list (``AdminRepository.list_tenants``) reads
its per-tenant ``project_count`` from the ``admin_tenant_rollup()`` SECURITY
DEFINER resolver (migration 0031). Its inner project-count subquery counted ALL
``projects`` rows, including hidden storyboard BRANCH rows
(``projects.is_branch = true``, added in 0052). Branches are internal fork
workspaces — every user-facing listing filters them (``project_repository``),
and the admin surface must too, or the dashboard inflates. The fixed body now
carries ``WHERE p.is_branch = false`` in that subquery.

DDL is a FROZEN SNAPSHOT of the function as it exists at this revision. It was
originally single-sourced from ``src.database.rls.admin_functions_statements``,
but re-importing the live generator here is a replay hazard: a from-scratch
``alembic upgrade head`` would emit TODAY's generator SQL against a HISTORICAL
schema, and any future evolution of that generator (as happened between 0031 and
this revision) would silently rewrite what this migration applies. Migration 0031
was frozen ONE REVISION EARLIER precisely to kill this pattern; the body is
therefore inlined here too. The branch-exclusion (``WHERE p.is_branch = false``)
is preserved in the frozen literal — that IS the behavioral fix this migration
delivers. The ``after_create`` test-schema path (``register_admin_functions_ddl``)
still tracks the live generator, so ``create_all``-ed schemas stay current.

Downgrade is a no-op: pre-prod is forward-only and the prior (unfiltered) body
is not reconstructed. The function stays defined with the corrected body rather
than being dropped, so ``list_tenants`` keeps working either direction.

Revision ID: 0054
Revises: 0053
Create Date: 2026-07-18
"""

from alembic import op

revision = "0054"
down_revision = "0053"
branch_labels = None
depends_on = None

# Frozen 2026-07-19: exact statements ``admin_functions_statements()`` returned at
# this revision (WITH the ``is_branch`` filter — the behavioral fix). Do NOT
# re-import the live generator; see 0031's banner for the replay-hazard rationale.
_ADMIN_TENANT_ROLLUP_FN = (
    "CREATE OR REPLACE FUNCTION admin_tenant_rollup() "
    "RETURNS TABLE (tenant_id uuid, project_count bigint, last_activity timestamptz, "
    "run_count bigint, cost_usd numeric) "
    "LANGUAGE sql STABLE SECURITY DEFINER SET search_path FROM CURRENT AS $$ "
    "SELECT pa.tenant_id, pa.project_count, pa.last_activity, "
    "COALESCE(ua.run_count, 0) AS run_count, COALESCE(ua.cost_usd, 0) AS cost_usd "
    "FROM (SELECT p.tenant_id, count(p.id) AS project_count, "
    "max(COALESCE(p.updated_at, p.created_at)) AS last_activity "
    "FROM projects p WHERE p.is_branch = false GROUP BY p.tenant_id) pa "
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
    # Forward-only: the function is left defined with the corrected body.
    pass
