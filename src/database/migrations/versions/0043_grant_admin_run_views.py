"""Grant EXECUTE on the admin run-view resolvers to the runtime role

Migration 0035 created ``admin_active_runs()`` / ``admin_recent_runs(integer)``
with ``EXECUTE`` REVOKEd from PUBLIC and the grant to the restricted runtime role
``w2v_orch_app`` deferred out-of-band (gitops ``phase-d-rls-stage-a.sql``, "PR 3"),
matching the other SECURITY DEFINER resolvers. That out-of-band grant was never
applied in prod, so under ``TENANT_RLS_MODE=enforce`` the runtime role hit
``permission denied for function admin_active_runs`` — a raw 500 on
``GET /api/admin/runs?status=running`` (the super-admin operator run pane).

This codifies the grant forward-only. The migration runs as the privileged owner
``w2v_service`` (the function owner, per the initContainer/alembic connection), so
it can grant. The DDL is role-GUARDED (single-sourced from ``src.database.rls``):
``w2v_orch_app`` exists only in prod, not in CI/test DBs — the ``DO`` block no-ops
when the role is absent, keeping create_all / ephemeral-postgres green.

Revision ID: 0043
Revises: 0042
Create Date: 2026-07-06
"""

from alembic import op


revision = "0043"
down_revision = "0042"
branch_labels = None
depends_on = None

# FROZEN DDL — inlined snapshot of ``src.database.rls.grant_admin_run_views_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_GRANT_ADMIN_RUN_VIEWS_STATEMENTS: tuple[str, ...] = (
    "DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_orch_app') THEN GRANT EXECUTE ON FUNCTION admin_active_runs() TO w2v_orch_app; GRANT EXECUTE ON FUNCTION admin_recent_runs(integer) TO w2v_orch_app; END IF; END $$",
)


# FROZEN DDL — inlined snapshot of ``src.database.rls.revoke_admin_run_views_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_REVOKE_ADMIN_RUN_VIEWS_STATEMENTS: tuple[str, ...] = (
    "DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_orch_app') THEN REVOKE EXECUTE ON FUNCTION admin_active_runs() FROM w2v_orch_app; REVOKE EXECUTE ON FUNCTION admin_recent_runs(integer) FROM w2v_orch_app; END IF; END $$",
)


def upgrade() -> None:
    for stmt in _GRANT_ADMIN_RUN_VIEWS_STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in _REVOKE_ADMIN_RUN_VIEWS_STATEMENTS:
        op.execute(stmt)
