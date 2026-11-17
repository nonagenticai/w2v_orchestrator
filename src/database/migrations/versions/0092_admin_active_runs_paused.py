"""admin_active_runs() includes PAUSED runs

The super-admin Active Runs pane (``GET /api/admin/runs?status=running``) read
``admin_active_runs()``, which 0035 defined as ``status IN ('pending', 'running')``.
A paused run — waiting on a human review gate, the kind that stalls — appeared in
neither the Running nor the Recent view (UI review 2026-10-17, M9). This widens the
filter to ``('pending', 'running', 'paused')``.

``CREATE OR REPLACE`` with an IDENTICAL signature and return table keeps the owner and
every privilege (the REVOKE from PUBLIC of 0035 and the ``w2v_orch_app`` grant of
0043), so no re-grant is needed. 0035 is NOT edited — history stays as it ran.

Revision ID: 0092_admin_active_runs_paused
Revises: 0091_runs_failure_detail
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# ``version_num`` is varchar(32); this id is 29 characters.
revision = "0092_admin_active_runs_paused"
down_revision = "0091_runs_failure_detail"
branch_labels = None
depends_on = None

_RETURNS = (
    "RETURNS TABLE (run_id uuid, tenant_id uuid, project_id uuid, project_title text, "
    "deliverable_id uuid, blueprint_id text, status text, current_stage text, "
    "started_at timestamptz, created_at timestamptz, updated_at timestamptz) "
    "LANGUAGE sql STABLE SECURITY DEFINER SET search_path FROM CURRENT AS $$ "
    "SELECT r.id, p.tenant_id, p.id, p.title, r.deliverable_id, r.blueprint_id, "
    "r.status, r.current_stage, r.started_at, r.created_at, r.updated_at FROM "
    "runs r JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id "
)

# FROZEN DDL (2026-10-18) — equals ``src.database.rls.admin_run_view_statements()[0]``
# at this revision. Do NOT re-import the live generator (see 0031's banner).
_UPGRADE = (
    "CREATE OR REPLACE FUNCTION admin_active_runs() "
    + _RETURNS
    + "WHERE r.status IN ('pending', 'running', 'paused') $$"
)
# Exactly 0035's definition.
_DOWNGRADE = (
    "CREATE OR REPLACE FUNCTION admin_active_runs() "
    + _RETURNS
    + "WHERE r.status IN ('pending', 'running') $$"
)


def _has_later_shape() -> bool:
    """True when ``admin_active_runs()`` already returns 0102's ``paused_at`` column.

    Only a re-stamped DB (alembic_version moved back below this revision while the
    schema stayed at head) reaches here with that shape. ``CREATE OR REPLACE`` cannot
    change a function's result type, so replaying this revision would abort; 0102
    re-asserts the definition later in the same upgrade, so skipping is correct.
    A DB really at 0091 never has the column, so it always runs the original DDL.

    Scoped to the schema this migration writes into (``current_schema()``, the
    first entry of the search_path the unqualified CREATE below resolves to): a
    same-named function in ANY other schema says nothing about the one this
    revision would replace, and must not make it skip.
    """
    return bool(
        op.get_bind()
        .execute(
            sa.text(
                "SELECT 1 FROM pg_proc WHERE proname = 'admin_active_runs' "
                "AND pronamespace = current_schema()::regnamespace "
                "AND pg_get_function_result(oid) LIKE '%paused_at%'"
            )
        )
        .first()
    )


def upgrade() -> None:
    if _has_later_shape():
        return
    op.execute(_UPGRADE)


#: A DROP discards the function's privileges; 0035 revoked PUBLIC and 0043 granted
#: the runtime role, so a downgrade that has to drop must put both back.
_DOWNGRADE_PRIVILEGES = (
    "REVOKE EXECUTE ON FUNCTION admin_active_runs() FROM PUBLIC",
    "DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_orch_app') "
    "THEN GRANT EXECUTE ON FUNCTION admin_active_runs() TO w2v_orch_app; END IF; END $$",
)


def downgrade() -> None:
    if _has_later_shape():
        # The same re-stamped DB ``upgrade()`` guards against, in the other
        # direction: the function still carries 0102's wider result type, and
        # ``CREATE OR REPLACE`` cannot change a return type, so replaying this
        # revision's downgrade would abort ("cannot change return type of existing
        # function") with no way forward. Drop first, then recreate 0035's shape.
        op.execute("DROP FUNCTION admin_active_runs()")
        op.execute(_DOWNGRADE)
        for stmt in _DOWNGRADE_PRIVILEGES:
            op.execute(stmt)
        return
    op.execute(_DOWNGRADE)
