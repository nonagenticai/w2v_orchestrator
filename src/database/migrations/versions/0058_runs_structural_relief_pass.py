"""runs.structural_relief_pass — spawn-once guard for demo_video auto-convergence (PR-C)

The demo_video timing-relief loop can, on a footage-pinned OVER-BUDGET timeline that
provably cannot converge in place, spawn a bounded child regenerate run from
``demo_video_gen`` with fewer beats (a shorter spine) so it auto-converges. This column
records that child's 1-based relief pass number; it is NULL for every normal run.

* Nullable + NO server default → a metadata-only ``ADD COLUMN`` (no table rewrite), safe
  on the live ``runs`` table. NON-DESTRUCTIVE and forward-compatible: an OLDER image that
  never heard of the column boots and writes fine against the migrated DB.
* ``uq_runs_parent_relief_pass`` — a UNIQUE PARTIAL index on ``(parent_run_id,
  structural_relief_pass) WHERE structural_relief_pass IS NOT NULL``. This is the ultimate
  idempotency guard: the orchestrator is replicas:2 and has two completion paths (the
  inline DAG-complete gate + the reconciler backstop), so a second attempt to spawn the
  same (parent, pass) child fails the INSERT with an IntegrityError the caller swallows —
  never a double paid render. Normal runs (NULL pass) are excluded, so the index never
  constrains them.

Revision ID: 0058
Revises: 0057
Create Date: 2026-07-24
"""

import sqlalchemy as sa
from alembic import op

revision = "0058"
down_revision = "0057"
branch_labels = None
depends_on = None

_INDEX_NAME = "uq_runs_parent_relief_pass"


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column("structural_relief_pass", sa.Integer(), nullable=True),
    )
    op.create_index(
        _INDEX_NAME,
        "runs",
        ["parent_run_id", "structural_relief_pass"],
        unique=True,
        postgresql_where=sa.text("structural_relief_pass IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(_INDEX_NAME, table_name="runs")
    op.drop_column("runs", "structural_relief_pass")
