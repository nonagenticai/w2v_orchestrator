"""``stage_outputs.edited_at`` + ``stage_outputs.config_reads`` (edit-and-restart)

A Restart of a dead run (cancelled / failed) — or a regenerate of a completed one
after an edit — reuses every completed stage the change does not affect and
re-runs only the affected stages and their dependents
(``src.services.runs.restart_plan``). Two facts per stage row make that decision
possible; neither existed:

* ``edited_at`` — when a user last edited this stage's output in place (element
  patch / restamp, course-stitch chapter or thumbnail override). A stage edited
  AFTER it completed keeps its edited output and its DESCENDANTS re-run.
* ``config_reads`` — the top-level run-snapshot keys the stage actually read while
  it ran, recorded by the executor (``src.services.runs.config_reads``). A changed
  project/run input re-runs exactly the stages that read it. ``["*"]`` means the
  stage read the snapshot wholesale (any change dirties it).

Both NULLABLE with no default and no backfill: a pre-0102 row has an UNKNOWN read
set, which the planner treats as "any input change re-runs this stage" — the
pre-0102 behaviour. Purely additive; the 0063 write-once trigger watches only
``output``, so neither column is guarded by it. Frozen literal SQL.

Revision ID: 0103_stage_output_edit_reads
Revises: 0102_admin_run_views_paused_at
Create Date: 2026-10-27
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 28 characters.
revision = "0103_stage_output_edit_reads"
down_revision = "0102_admin_run_views_paused_at"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE stage_outputs ADD COLUMN IF NOT EXISTS edited_at timestamptz")
    op.execute("ALTER TABLE stage_outputs ADD COLUMN IF NOT EXISTS config_reads jsonb")


def downgrade() -> None:
    op.execute("ALTER TABLE stage_outputs DROP COLUMN IF EXISTS config_reads")
    op.execute("ALTER TABLE stage_outputs DROP COLUMN IF EXISTS edited_at")
