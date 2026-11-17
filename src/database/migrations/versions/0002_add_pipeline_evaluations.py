"""Add pipeline_evaluations table.

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-08
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "pipeline_evaluations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "job_id",
            sa.String(36),
            sa.ForeignKey("pipeline_jobs.job_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage_name", sa.String(50), nullable=False),
        sa.Column("overall_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("grade", sa.String(2), nullable=False, server_default="'F'"),
        sa.Column("pass_fail", sa.String(10), nullable=False, server_default="'fail'"),
        sa.Column("metrics_json", postgresql.JSONB(), nullable=True),
        sa.Column("insights", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_pipeline_evaluations_job_id",
        "pipeline_evaluations",
        ["job_id"],
    )
    op.create_index(
        "idx_eval_job_stage",
        "pipeline_evaluations",
        ["job_id", "stage_name"],
    )


def downgrade() -> None:
    op.drop_index("idx_eval_job_stage", table_name="pipeline_evaluations")
    op.drop_index("ix_pipeline_evaluations_job_id", table_name="pipeline_evaluations")
    op.drop_table("pipeline_evaluations")
