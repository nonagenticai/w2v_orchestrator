"""stage_outputs: per-execution runtime + a durable drain counter

``completed_at - started_at`` is NOT a stage's runtime and never was.
``mark_running`` stamps ``started_at`` once, through a ``COALESCE``, and
``mark_pending`` deliberately preserves it (both documented at length in
``StageOutputRepository``) — so the span measures FIRST DISPATCH → FINAL
COMPLETION, including burnt executions, re-dispatch gaps and pod downtime. That
is the right semantics for "when did this stage begin", and the wrong number to
compare against a stage ceiling. Two separate audits in one day used it as a
runtime and "proved" budget incoherences that do not exist.

``attempts == 0`` did not rescue it either. Both reset paths call
``mark_pending``: the resumable-timeout path bumps ``attempts`` first, the
cancellation path (pod drain / pause / cancel) bumps nothing. Orchestrator pods
roll constantly, so a drained stage silently resumed with ``attempts == 0`` and
an untouched ``started_at`` — indistinguishable from a stage that ran once,
cleanly, for the whole span.

Two nullable columns, both purely additive:

``last_execution_started_at`` — re-stamped UNCONDITIONALLY on every
``mark_running``, so ``completed_at - last_execution_started_at`` is a
per-execution runtime that CAN be compared with a ceiling, while ``started_at``
keeps its first-dispatch meaning untouched.

``interruptions`` — bumped only on the drain/cancel path, the durable twin of
``attempts``. Recorded rather than inferred, because inference is exactly what
failed here.

NULLABLE with NO server default, and NO backfill, on purpose. A default of 0
would be applied to every existing row and would assert that those stages were
never drained — fabricating history the database does not have. NULL reads as
"not recorded"; 0 reads as "recorded, never drained". Old rows keep their
ambiguous semantics and are honestly marked as ambiguous.

Compatible with the write-once trigger from 0063 by construction, not by
exemption: ``trg_stage_outputs_write_once`` fires only
``WHEN (OLD.status IN (terminal...) AND NEW.output IS DISTINCT FROM OLD.output)``.
Neither column is ``output``, and both writers are additionally guarded to a
non-terminal row, so the trigger never sees these writes and no
``w2v.allow_stage_output_edit`` escape hatch is involved.

Two ``ADD COLUMN``s with no default and no NOT NULL: metadata-only in Postgres,
no table rewrite, brief ACCESS EXCLUSIVE only. An OLDER image booted against the
migrated DB is unaffected — it simply never writes the new columns.

Revision ID: 0067_stage_execution_runtime
Revises: 0066_runs_failure_class
Create Date: 2026-09-03
"""

import sqlalchemy as sa
from alembic import op

revision = "0067_stage_execution_runtime"
down_revision = "0066_runs_failure_class"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "stage_outputs",
        sa.Column("interruptions", sa.Integer(), nullable=True),
    )
    op.add_column(
        "stage_outputs",
        sa.Column("last_execution_started_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("stage_outputs", "last_execution_started_at")
    op.drop_column("stage_outputs", "interruptions")
