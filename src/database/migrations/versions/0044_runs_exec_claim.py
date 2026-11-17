"""add runs.exec_pod / runs.exec_claimed_at (durable execution claim)

Durability: the second guard behind the Redis execution lease (``RunLease``).
``RunLease.acquire`` deliberately falls open (returns None) on a Redis error so a
transient blip can't wedge every run — but under a full Redis outage that lets
all 2-5 replicas execute the SAME run at once, double-running its non-idempotent
submit stages (render/TTS/upload). These two nullable columns back an atomic
conditional claim (``UPDATE ... WHERE exec_claimed_at IS NULL OR stale``) so
exactly one pod proceeds on the fail-open path. NULL on the normal Redis-lease
path — no behaviour change there.

Revision ID: 0044
Revises: 0043
Create Date: 2026-07-07
"""

import sqlalchemy as sa
from alembic import op

revision = "0044"
down_revision = "0043"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("exec_pod", sa.Text(), nullable=True))
    op.add_column(
        "runs",
        sa.Column("exec_claimed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("runs", "exec_claimed_at")
    op.drop_column("runs", "exec_pod")
