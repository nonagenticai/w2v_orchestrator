"""Paused runs say WHY and SINCE WHEN they wait

Run 01a1770d (promo_v1) paused for storyboard review at 2026-10-26 07:15:21Z and was
cancelled 29h later as "stuck running". Nothing was executing; nobody knew it was
waiting. This revision adds what every surface needs to say so:

``runs.pause_reason`` — ``'checkpoint'`` (a review gate tripped: a person must approve
the stage output), ``'manual'`` (someone pressed Pause; ``POST /runs/{id}/pause`` has
existed since May) or ``'unknown'``. NULL whenever the run is not paused. Only a
checkpoint pause is "needs review"; the other two are just paused.

Backfill — rows ALREADY paused when this runs carry no reason, and they were NOT all
checkpoint pauses. The evidence a checkpoint pause leaves is exact: the dispatcher
re-points ``current_stage`` at the gate, that stage is still listed in
``pause_after_stages`` (resume is what strips a consumed gate) and its stage row is
``completed``. A paused row matching all three is ``checkpoint``; anything else is
``unknown`` — never guessed into "needs review".

The admin resolvers ``admin_active_runs()`` / ``admin_recent_runs()`` also return
``paused_at`` and ``pause_reason`` — the operator pane used to age a paused run from
``started_at``, which is how a 29h review wait read as a 29h-old running run.
Changing a ``RETURNS TABLE`` needs DROP + CREATE (``CREATE OR REPLACE`` refuses a
new result type), and a DROP discards privileges, so the 0035 ``REVOKE ... FROM
PUBLIC`` and the role-guarded 0043 grant to the runtime role ``w2v_orch_app`` are
re-applied in the same transaction. An old pod still selecting the eleven named
columns keeps working against the widened function.

Who runs this: in prod the migration container connects with the privileged
``DATABASE_URL`` role ``w2v_service`` (measured 2026-10-27: it owns ``runs`` and
every SECURITY DEFINER resolver; ``proacl`` on each is
``{w2v_service=X/w2v_service,w2v_orch_app=X/w2v_service}``). The pods run as
``w2v_orch_app`` via ``RUNTIME_DATABASE_URL`` (gitops
``services/w2v/PHASE-D-RLS-CUTOVER.md``). A function created here is therefore owned
by ``w2v_service``, and ``w2v_orch_app`` can execute it ONLY because of the
explicit GRANT below — the 0043 incident was exactly that grant missing (a raw 500
on ``GET /api/admin/runs`` under enforce).

Revision ID: 0102_admin_run_views_paused_at
Revises: 0101_signup_llm_byo_required
"""

from __future__ import annotations

from alembic import op

# ``version_num`` is varchar(32); this id is 30 characters.
revision = "0102_admin_run_views_paused_at"
down_revision = "0101_signup_llm_byo_required"
branch_labels = None
depends_on = None

_PREAMBLE = "LANGUAGE sql STABLE SECURITY DEFINER SET search_path FROM CURRENT AS $$ "
_JOIN = (
    "runs r JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id"
)
_COLS_0092 = (
    "run_id uuid, tenant_id uuid, project_id uuid, project_title text, "
    "deliverable_id uuid, blueprint_id text, status text, current_stage text, "
    "started_at timestamptz, created_at timestamptz, updated_at timestamptz"
)
# Static SQL from hard-coded names only; no input ever reaches it (as in rls.py).
_SELECT_0092 = (
    "SELECT r.id, p.tenant_id, p.id, p.title, r.deliverable_id, r.blueprint_id, "  # noqa: S608
    f"r.status, r.current_stage, r.started_at, r.created_at, r.updated_at FROM {_JOIN}"
)
_COLS = f"{_COLS_0092}, paused_at timestamptz, pause_reason text"
_SELECT = (
    "SELECT r.id, p.tenant_id, p.id, p.title, r.deliverable_id, r.blueprint_id, "  # noqa: S608
    "r.status, r.current_stage, r.started_at, r.created_at, r.updated_at, r.paused_at, "
    f"r.pause_reason FROM {_JOIN}"
)

# Classify rows that were already paused (see the module docstring).
_BACKFILL = (
    "UPDATE runs r SET pause_reason = CASE WHEN r.current_stage = ANY(r.pause_after_stages) "
    "AND EXISTS (SELECT 1 FROM stage_outputs so WHERE so.run_id = r.id "
    "AND so.stage_name = r.current_stage AND so.status = 'completed') "
    "THEN 'checkpoint' ELSE 'unknown' END "
    "WHERE r.status = 'paused' AND r.pause_reason IS NULL"
)

_DROP_VIEWS = [
    "DROP FUNCTION IF EXISTS admin_active_runs()",
    "DROP FUNCTION IF EXISTS admin_recent_runs(integer)",
]


def _guarded_grant(fns: list[str]) -> str:
    grants = " ".join(f"GRANT EXECUTE ON FUNCTION {fn} TO w2v_orch_app;" for fn in fns)
    return (  # static: fn names are the literal signatures below, never input
        "DO $$ BEGIN "  # noqa: S608
        "IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_orch_app') THEN "
        f"{grants} END IF; END $$"
    )


def _create_views(cols: str, select: str) -> list[str]:
    """FROZEN DDL — the two admin resolvers, plus their privileges."""
    return [
        "CREATE FUNCTION admin_active_runs() "
        f"RETURNS TABLE ({cols}) {_PREAMBLE}{select} "
        "WHERE r.status IN ('pending', 'running', 'paused') $$",
        "REVOKE EXECUTE ON FUNCTION admin_active_runs() FROM PUBLIC",
        "CREATE FUNCTION admin_recent_runs(p_limit integer) "
        f"RETURNS TABLE ({cols}) {_PREAMBLE}{select} "
        "WHERE r.status IN ('completed', 'failed', 'cancelled') "
        "ORDER BY r.updated_at DESC LIMIT p_limit $$",
        "REVOKE EXECUTE ON FUNCTION admin_recent_runs(integer) FROM PUBLIC",
        _guarded_grant(["admin_active_runs()", "admin_recent_runs(integer)"]),
    ]


def upgrade() -> None:
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS pause_reason TEXT")
    op.execute("ALTER TABLE runs DROP CONSTRAINT IF EXISTS ck_runs_pause_reason")
    op.execute(
        "ALTER TABLE runs ADD CONSTRAINT ck_runs_pause_reason "
        "CHECK (pause_reason IS NULL OR pause_reason IN ('checkpoint', 'manual', 'unknown'))"
    )
    op.execute(_BACKFILL)
    for stmt in [*_DROP_VIEWS, *_create_views(_COLS, _SELECT)]:
        op.execute(stmt)


def downgrade() -> None:
    # Exactly 0092's resolvers (0035's recent-runs body is unchanged by 0092).
    for stmt in [*_DROP_VIEWS, *_create_views(_COLS_0092, _SELECT_0092)]:
        op.execute(stmt)
    op.execute("ALTER TABLE runs DROP CONSTRAINT IF EXISTS ck_runs_pause_reason")
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS pause_reason")
