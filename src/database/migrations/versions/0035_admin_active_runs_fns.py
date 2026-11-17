"""Admin cross-tenant ACTIVE/RECENT run-view resolvers — operator run visibility

Adds two ``SECURITY DEFINER`` functions, ``admin_active_runs()`` and
``admin_recent_runs(integer)``, that back the platform operator pane ("what is
running right now, across all tenants"). Like ``admin_tenant_rollup()`` (0031)
this is a genuinely cross-tenant super-admin read that cannot bind a single
tenant's GUC, so it goes through SECURITY DEFINER — same owner/REVOKE/static-SQL
guarantees, data-plane policies UNCHANGED (no "bypass GUC" disjunct), so provable
isolation holds. ``EXECUTE`` revoked from PUBLIC; the cutover grants it to
``w2v_orch_app`` out-of-band alongside the other admin resolvers.

DDL is now FROZEN in-file as of revision 0035 to keep from-scratch replay
deterministic: re-importing the live ``src.database.rls`` generator here is a
replay hazard — a from-scratch ``alembic upgrade head`` would emit TODAY's
generator SQL against a HISTORICAL schema, and any future evolution of that
generator would silently rewrite what this migration applies (exactly the
pattern migrations 0031 and 0054 were frozen to eliminate). The live generator +
the ``after_create`` test-schema path (``register_admin_functions_ddl``) are
UNCHANGED and keep applying the current DDL to fresh ``create_all``-ed test
schemas; only this historical migration is pinned to the frozen snapshot below.

Revision ID: 0035
Revises: 0034
Create Date: 2026-06-29
"""

from alembic import op

revision = "0035"
down_revision = "0034"
branch_labels = None
depends_on = None

# Frozen 2026-07-19: exact statements ``admin_run_view_statements()`` returned at
# this revision. Do NOT re-import the live generator — this is the frozen DDL as of
# this revision; see 0031's banner for the replay-hazard rationale.
_ADMIN_ACTIVE_RUNS_FN = (
    "CREATE OR REPLACE FUNCTION admin_active_runs() "
    "RETURNS TABLE (run_id uuid, tenant_id uuid, project_id uuid, project_title text, "
    "deliverable_id uuid, blueprint_id text, status text, current_stage text, "
    "started_at timestamptz, created_at timestamptz, updated_at timestamptz) "
    "LANGUAGE sql STABLE SECURITY DEFINER SET search_path FROM CURRENT AS $$ "
    "SELECT r.id, p.tenant_id, p.id, p.title, r.deliverable_id, r.blueprint_id, "
    "r.status, r.current_stage, r.started_at, r.created_at, r.updated_at FROM "
    "runs r JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id "
    "WHERE r.status IN ('pending', 'running') $$"
)
_ADMIN_RECENT_RUNS_FN = (
    "CREATE OR REPLACE FUNCTION admin_recent_runs(p_limit integer) "
    "RETURNS TABLE (run_id uuid, tenant_id uuid, project_id uuid, project_title text, "
    "deliverable_id uuid, blueprint_id text, status text, current_stage text, "
    "started_at timestamptz, created_at timestamptz, updated_at timestamptz) "
    "LANGUAGE sql STABLE SECURITY DEFINER SET search_path FROM CURRENT AS $$ "
    "SELECT r.id, p.tenant_id, p.id, p.title, r.deliverable_id, r.blueprint_id, "
    "r.status, r.current_stage, r.started_at, r.created_at, r.updated_at FROM "
    "runs r JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id "
    "WHERE r.status IN ('completed', 'failed', 'cancelled') "
    "ORDER BY r.updated_at DESC LIMIT p_limit $$"
)
_ADMIN_RUN_VIEW_STATEMENTS = (
    _ADMIN_ACTIVE_RUNS_FN,
    "REVOKE EXECUTE ON FUNCTION admin_active_runs() FROM PUBLIC",
    _ADMIN_RECENT_RUNS_FN,
    "REVOKE EXECUTE ON FUNCTION admin_recent_runs(integer) FROM PUBLIC",
)
_DROP_ADMIN_RUN_VIEW_STATEMENTS = (
    "DROP FUNCTION IF EXISTS admin_active_runs()",
    "DROP FUNCTION IF EXISTS admin_recent_runs(integer)",
)


def upgrade() -> None:
    for stmt in _ADMIN_RUN_VIEW_STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in _DROP_ADMIN_RUN_VIEW_STATEMENTS:
        op.execute(stmt)
