"""RLS background-session bypass resolvers — Phase D (increment 3b)

Adds two ``SECURITY DEFINER`` functions the trusted background sessions need once
the runtime app connects as the restricted no-``BYPASSRLS`` role under enforce:

  * ``run_tenant(run_id) -> uuid`` — the dispatcher bootstrap: it holds only a
    run_id, and reading the run row is itself RLS-gated, so it resolves the run's
    tenant via this bypass, binds ``app.current_tenant``, then proceeds.
  * ``orphan_runs() -> (run_id, tenant_id, status, updated_at)`` — the reconciler's
    cross-tenant orphan scan (a plain ``SELECT FROM runs`` would be RLS-scoped to a
    single tenant under enforce).

Both are owned by the migration user (the postgres superuser in prod → bypasses
FORCE; the table owner in CI/tests → bypasses ENABLE-not-FORCE) so the bodies see
every tenant. ``EXECUTE`` is revoked from PUBLIC; PR 3 grants it to ``w2v_orch_app``
out-of-band. CI-safe: a non-superuser owner (``ci_test``) can create functions it
owns, and no role/FORCE DDL is involved. Inert until the runtime role switch.

FROZEN 2026-07-21: the DDL below is an exact snapshot of what
``run_functions_statements()`` / ``drop_run_functions_statements()`` returned at this
revision. Do NOT re-import the live generator. A migration must reproduce the
HISTORICAL schema; replaying it through the live generator would emit whatever the
function bodies say TODAY, so a later edit to those bodies in ``src.database.rls``
would silently rewrite this historical migration's replayed output — and a
migration-based prod DB (which ran 0030 exactly once and never re-runs
``create_all``) would then diverge from a fresh replay. That is the exact drift
class that forced the ``0031`` / ``0035`` / ``0054`` freezes. The LIVE schema still
tracks the current bodies via ``register_run_functions_ddl``'s ``after_create``
path; the guard test ``tests/database/test_frozen_migration_ddl.py`` asserts this
snapshot still equals the live generator, so an intentional body change is
reconciled here — loudly, at PR time — instead of silently drifting.

Revision ID: 0030
Revises: 0029
Create Date: 2026-06-16
"""

from alembic import op

revision = "0030"
down_revision = "0029"
branch_labels = None
depends_on = None

# Frozen 2026-07-21: exact statements ``run_functions_statements()`` returned at this
# revision. Do NOT re-import the live generator (see the module docstring).
_RUN_TENANT_FN = (
    "CREATE OR REPLACE FUNCTION run_tenant(p_run_id uuid) RETURNS uuid "
    "LANGUAGE sql STABLE SECURITY DEFINER SET search_path FROM CURRENT AS $$ "
    "SELECT p.tenant_id FROM runs r JOIN deliverables d ON d.id = r.deliverable_id "
    "JOIN projects p ON p.id = d.project_id WHERE r.id = p_run_id $$"
)
_ORPHAN_RUNS_FN = (
    "CREATE OR REPLACE FUNCTION orphan_runs() RETURNS TABLE "
    "(run_id uuid, tenant_id uuid, status text, updated_at timestamptz) "
    "LANGUAGE sql STABLE SECURITY DEFINER SET search_path FROM CURRENT AS $$ "
    "SELECT r.id, p.tenant_id, r.status, r.updated_at FROM runs r "
    "JOIN deliverables d ON d.id = r.deliverable_id "
    "JOIN projects p ON p.id = d.project_id WHERE r.status IN ('pending', 'running') $$"
)
_RUN_FUNCTIONS_STATEMENTS = (
    _RUN_TENANT_FN,
    "REVOKE EXECUTE ON FUNCTION run_tenant(uuid) FROM PUBLIC",
    _ORPHAN_RUNS_FN,
    "REVOKE EXECUTE ON FUNCTION orphan_runs() FROM PUBLIC",
)
_DROP_RUN_FUNCTIONS_STATEMENTS = (
    "DROP FUNCTION IF EXISTS orphan_runs()",
    "DROP FUNCTION IF EXISTS run_tenant(uuid)",
)


def upgrade() -> None:
    for stmt in _RUN_FUNCTIONS_STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in _DROP_RUN_FUNCTIONS_STATEMENTS:
        op.execute(stmt)
