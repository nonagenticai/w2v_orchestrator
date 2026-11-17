"""stage_outputs.status += 'cancelled' — terminal state for stranded stage rows

When a run hard-fails, the DAG executor cancels only the failing gather batch;
independent sibling / never-scheduled downstream stages are otherwise left
'pending'/'running' under a terminal 'failed' run that is never resumed — they
strand forever and pollute the stage_outputs group-by (and surface as ghost
'pending' stages in the UI/SSE). The run-failure site now sweeps them to a
terminal 'cancelled' (the vocabulary the ``runs`` table already uses). This
widens the ``ck_stage_outputs_status`` CHECK to admit that value.

Widening a CHECK is non-destructive and forward-compatible: every existing row
already satisfies the smaller set, so the ADD CONSTRAINT validates instantly (no
table rewrite), and an OLDER image that never writes 'cancelled' boots and runs
fine against the migrated DB.

Revision ID: 0059
Revises: 0058
Create Date: 2026-07-28
"""

from alembic import op

revision = "0059"
down_revision = "0058"
branch_labels = None
depends_on = None

_CK = "ck_stage_outputs_status"
_OLD = "status IN ('pending','running','completed','failed','skipped')"
_NEW = "status IN ('pending','running','completed','failed','skipped','cancelled')"


def upgrade() -> None:
    op.drop_constraint(_CK, "stage_outputs", type_="check")
    op.create_check_constraint(_CK, "stage_outputs", _NEW)


def downgrade() -> None:
    # Forward-only in practice: a downgrade fails if any 'cancelled' rows exist.
    # Provided for symmetry — relabel/sweep those rows before downgrading.
    op.drop_constraint(_CK, "stage_outputs", type_="check")
    op.create_check_constraint(_CK, "stage_outputs", _OLD)
