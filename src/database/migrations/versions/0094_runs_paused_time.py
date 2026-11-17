"""runs.paused_at + runs.paused_seconds — paused time does not count against the run deadline

The run deadline was ``started_at + run_deadline_seconds`` (wall clock), so a
course paused for human review longer than the budget (14400s) was already
"expired" the moment it was resumed and died on its first resumable timeout.

``paused_at`` is stamped by ``RunRepository.mark_paused`` (COALESCE: a re-pause
of an already-paused row keeps the original stamp); ``mark_resumed`` adds
``now() - paused_at`` to ``paused_seconds`` and clears ``paused_at`` in the same
compare-and-set. Every deadline computation is then
``started_at + run_deadline_seconds + paused_seconds``.

New columns on an existing RLS-enabled table: ``runs``' tenant policies are
row-level and column-agnostic, so no policy change is needed. ``paused_seconds``
is NOT NULL DEFAULT 0, so every existing row reads as "never paused" — which
is what the old deadline assumed.

Drift-tolerant (``IF NOT EXISTS`` / ``IF EXISTS``), like 0091.

Revision ID: 0094_runs_paused_time
Revises: 0093_run_usage_call_id_unique
"""

from __future__ import annotations

from alembic import op

# ``version_num`` is varchar(32); this id is 21 characters.
revision = "0094_runs_paused_time"
down_revision = "0093_run_usage_call_id_unique"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS paused_at TIMESTAMPTZ")
    op.execute(
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS paused_seconds INTEGER NOT NULL DEFAULT 0"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS paused_seconds")
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS paused_at")
