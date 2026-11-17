"""stage_step_results — the STEP LEDGER: durable per-sub-step results

A stage is one handler call. ``stage_outputs`` gets its output only when the whole
handler returns, so a stage that dies half-way — pod roll, OOM, node loss — is
re-dispatched by the reconciler and re-entered FROM THE TOP with no partial state.
Every paid call it had already finished (LLM, TTS, render, scrape) is made again.
Every merge rolls the ``replicas: 1`` orchestrator, so this is the ordinary case,
not the rare one: one measured roll discarded 10 minutes of paid rendering and
re-executed the stage for 18.5.

This table is the store that makes a sub-step survivable, and nothing else: one row
per finished sub-step, keyed ``(run_id, stage_name, step_key)``. The helper that
reads/writes it is ``src/services/stages/shared/step_ledger.py``; no stage uses it
yet in this revision.

Shape notes (the ``StageStepResult`` model in ``src/database/models.py`` mirrors this):

* ``input_hash`` is the reuse condition. A row is reused only if the caller's hash
  equals it; otherwise the sub-step runs again and the row is REPLACED (upsert on the
  unique key) — the table holds the latest result per key, never a history.
* ``result`` is JSONB NOT NULL. Only successful sub-steps are recorded, so there is
  no status column: the row's existence is the fact.
* The unique constraint's index leads with ``run_id``, which also serves the FK
  cascade and the policy's correlated lookup — no separate ``run_id`` index.
* ``ON DELETE CASCADE`` from ``runs``: a ledger row is meaningless without its run.
* Not named "checkpoint": that word is taken by the user pause gates
  (``runs.pause_after_stages``).

RLS: ENABLE (not FORCE) + a join-based ``tenant_isolation`` policy — the same 3-hop
climb ``stage_outputs`` uses, ``run_id -> runs -> deliverables -> projects.tenant_id``.
The table carries NO ``tenant_id``. The DDL below is FROZEN literal text (see the
constants); the ``create_all`` path gets the same statements from
``src.database.rls.register_stage_step_results_rls_ddl``, and
``tests/database/test_migration_0071_stage_step_results.py`` pins the two together.

Runtime-role table privileges ride ``ALTER DEFAULT PRIVILEGES`` (gitops), like every
other table — no GRANT here (identical to 0046/0060/0062).

Purely additive: a new table, no backfill, no change to any existing table — an
OLDER image runs fine against the migrated DB (it never touches the table).

Revision ID: 0071_stage_step_results
Revises: 0070_runs_seeded_pause_stages
Create Date: 2026-10-06
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# ``version_num`` is varchar(32); this id is 23 characters.
revision = "0071_stage_step_results"
down_revision = "0070_runs_seeded_pause_stages"
branch_labels = None
depends_on = None

# FROZEN DDL — inlined snapshot of
# ``src.database.rls.stage_step_results_upgrade_statements()`` as rendered at freeze
# time (2026-10-06). Do NOT re-import the live generator: a later edit to rls.py
# would silently rewrite what this historical revision emits on a fresh bootstrap.
# Guarded by tests/database/test_migration_0071_stage_step_results.py and the
# repo-wide tests/database/test_frozen_migration_ddl.py import ban.
_STAGE_STEP_RESULTS_UPGRADE_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE stage_step_results ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON stage_step_results USING (EXISTS (SELECT 1 FROM runs r JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id WHERE r.id = stage_step_results.run_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM runs r JOIN deliverables d ON d.id = r.deliverable_id JOIN projects p ON p.id = d.project_id WHERE r.id = stage_step_results.run_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
)

# FROZEN DDL — snapshot of ``src.database.rls.stage_step_results_downgrade_statements()``
# (same freeze, same rule).
_STAGE_STEP_RESULTS_DOWNGRADE_STATEMENTS: tuple[str, ...] = (
    "DROP POLICY IF EXISTS tenant_isolation ON stage_step_results",
    "ALTER TABLE stage_step_results DISABLE ROW LEVEL SECURITY",
)


def upgrade() -> None:
    op.create_table(
        "stage_step_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        # The RLS climb key. CASCADE: a sub-step result is meaningless without its run.
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage_name", sa.Text(), nullable=False),
        sa.Column("step_key", sa.Text(), nullable=False),
        # Digest of everything the result depends on; a mismatch means "run it again".
        sa.Column("input_hash", sa.Text(), nullable=False),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "run_id", "stage_name", "step_key", name="uq_stage_step_results_run_stage_step"
        ),
    )

    # RLS: ENABLE-not-FORCE + the 3-hop run-join tenant_isolation policy (frozen).
    for stmt in _STAGE_STEP_RESULTS_UPGRADE_STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    # Reverse of upgrade(): policy first (it is ON the table), then the table, which
    # takes the unique constraint with it.
    for stmt in _STAGE_STEP_RESULTS_DOWNGRADE_STATEMENTS:
        op.execute(stmt)

    op.drop_table("stage_step_results")
