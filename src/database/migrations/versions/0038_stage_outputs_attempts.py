"""add stage_outputs.attempts (resume-on-timeout bound)

Durability P1: a genuinely slow LLM/render stage that times out is now
re-dispatched and resumed instead of terminal-failing the whole run. The resume
is bounded per-stage by this counter (incremented on each timeout) so a stage
that is ALWAYS slow can't loop forever — over ``PIPELINE_MAX_STAGE_ATTEMPTS`` it
terminal-fails. The column is durable (survives the pod death that motivates the
resume) rather than Redis-held.

Revision ID: 0038
Revises: 0037
Create Date: 2026-07-03
"""

import sqlalchemy as sa
from alembic import op

revision = "0038"
down_revision = "0037"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "stage_outputs",
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("stage_outputs", "attempts")
