"""Backfill: a TERMINATED run may not own a 'pending'/'running' stage row

Measured on the primary 2026-09-08, db ``w2v_orchestrator``: 1066
``stage_outputs`` rows sat ``pending`` under a run that is already terminal —
939 under 89 ``cancelled`` runs, 127 under 42 ``failed`` runs, oldest
2026-06-06. ``pending`` means "this stage will run"; for a stage whose run is
cancelled/failed that is false and permanently false, so every query for
outstanding work, every capacity/queue metric and every operator reading the UI
was told a stage is waiting when nothing will ever pick it up.

The forward fix is in ``RunRepository.mark_failed`` / ``mark_cancelled``, which
now sweep inside the winning compare-and-set. This migration is the one-shot
reconciliation of the rows that accumulated before it.

**``paused`` is deliberately EXCLUDED.** A paused run is NOT terminal
(``services/runs/sub_dag.TerminalRunStatus`` is ``{'completed','failed',
'cancelled'}``) and is resumable via ``RunRepository.mark_resumed`` /
``POST /api/runs/{id}/resume``, so the 286 pending rows under 27 paused runs are
telling the truth and must not be swept. Sweeping them would destroy work a
resume is entitled to execute.

**Batched and re-runnable.** The UPDATE takes an id slice at a time
(``_BATCH_SIZE``) rather than one table-wide statement, so it never holds a long
write transaction against a live table. It is idempotent by construction: a
swept row is no longer ``pending``/``running``, so it can never match again — a
second application (or a manual replay of ``_SWEEP_SQL`` by an operator) sweeps
0 rows.

**No RLS games.** RLS on the run-tree is ENABLEd but not FORCEd
(``src/database/rls.py``) and alembic connects as the privileged owner, so the
migration sees every tenant's rows without any GUC. It must never
``SET row_security = off`` — through pgbouncer that leaks onto the next
transaction on the pooled connection. It reads each row's tenant scope from the
PERSISTED ``runs`` join, never from an ambient ContextVar, and makes no outbound
call, so there is no identity to inherit or to misattribute across the many
tenants it walks.

**Safe against the write-once trigger.** ``trg_stage_outputs_write_once``
(migration 0063) fires only when ``OLD.status`` is already terminal AND
``output`` changes. This touches only ``pending``/``running`` rows and never
writes ``output``, so the trigger's WHEN clause is false and it is not invoked.

The reason prose is a FROZEN literal copy of
``StageOutputRepository.stranded_stage_reason`` per this repo's migration-freeze
convention (a replay must emit what it emitted the day it shipped, not whatever
the live helper says later);
``tests/database/repositories/test_terminal_run_leaves_no_pending_stage.py``
asserts the copy has not drifted.

Revision ID: 0068_sweep_stranded_stages
Revises: 0067_stage_execution_runtime
Create Date: 2026-09-08
"""

from alembic import op
from sqlalchemy import text

revision = "0068_sweep_stranded_stages"
down_revision = "0067_stage_execution_runtime"
branch_labels = None
depends_on = None

#: Rows per statement. Small enough that each UPDATE is a short transaction on a
#: live table, large enough that the measured 1066-row population clears in a
#: handful of passes.
BATCH_SIZE = 500

#: Hard cap on passes so a pathological state can never spin forever. At
#: ``BATCH_SIZE`` rows a pass this covers 1,000,000 rows — three orders of
#: magnitude above the measured population.
MAX_PASSES = 2000

#: FROZEN copies of ``stage_output_repository.stranded_stage_reason(...)``.
CANCELLED_REASON = "run terminated (cancelled) before this stage ran; it never ran and never will"
FAILED_REASON = "run terminated (failed) before this stage ran; it never ran and never will"

#: One batch. ``FOR UPDATE ... SKIP LOCKED`` keeps a concurrently-executing
#: orchestrator (which writes these very rows) from blocking the backfill and
#: vice-versa: a row another transaction holds is simply left for a later pass.
SWEEP_SQL = text(
    """
    WITH victim AS (
        SELECT s.id AS id, r.status AS run_status
        FROM stage_outputs s
        JOIN runs r ON r.id = s.run_id
        WHERE s.status IN ('pending', 'running')
          AND r.status IN ('cancelled', 'failed')
        ORDER BY s.id
        LIMIT :batch_size
        FOR UPDATE OF s SKIP LOCKED
    )
    UPDATE stage_outputs s
    SET status = 'cancelled',
        error = CASE v.run_status
            WHEN 'cancelled' THEN :cancelled_reason
            ELSE :failed_reason
        END,
        completed_at = now()
    FROM victim v
    WHERE s.id = v.id
    """
)


def upgrade() -> None:
    connection = op.get_bind()
    swept = 0
    for _ in range(MAX_PASSES):
        result = connection.execute(
            SWEEP_SQL,
            {
                "batch_size": BATCH_SIZE,
                "cancelled_reason": CANCELLED_REASON,
                "failed_reason": FAILED_REASON,
            },
        )
        rowcount = int(result.rowcount or 0)
        if rowcount == 0:
            break
        swept += rowcount
    print(f"0068: swept {swept} stranded stage row(s) under terminated runs")


def downgrade() -> None:
    """Deliberately a no-op.

    A swept row carries no record of whether it read ``pending`` or ``running``
    before the sweep, and restoring it would reinstate exactly the lie this
    migration exists to remove.
    """
