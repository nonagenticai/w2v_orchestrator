"""Initial schema: pipeline_jobs, stage_executions, and pipeline_usage tables.

Revision ID: 0001
Revises:
Create Date: 2026-02-23
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "pipeline_jobs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(36), nullable=False),
        sa.Column("tenant_id", sa.String(255), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("url", sa.String(2048), nullable=False),
        sa.Column("config_json", postgresql.JSONB(), nullable=True),
        sa.Column("current_stage", sa.String(50), nullable=True),
        sa.Column("progress_pct", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("stages_json", postgresql.JSONB(), nullable=True),
        sa.Column("result_json", postgresql.JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_pipeline_jobs_job_id", "pipeline_jobs", ["job_id"], unique=True)
    op.create_index("ix_pipeline_jobs_tenant_id", "pipeline_jobs", ["tenant_id"])

    op.create_table(
        "stage_executions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "job_id",
            sa.String(36),
            sa.ForeignKey("pipeline_jobs.job_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage_name", sa.String(50), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("service_name", sa.String(100), nullable=True),
        sa.Column("service_job_id", sa.String(36), nullable=True),
        sa.Column("artifact_uris", postgresql.JSONB(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_stage_executions_job_id", "stage_executions", ["job_id"])

    op.create_table(
        "pipeline_usage",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.String(255), nullable=False),
        sa.Column("date", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("pipeline_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("stages_executed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_count", sa.Integer(), nullable=False, server_default="0"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_pipeline_usage_tenant_id", "pipeline_usage", ["tenant_id"])


def downgrade() -> None:
    op.drop_index("ix_pipeline_usage_tenant_id", table_name="pipeline_usage")
    op.drop_table("pipeline_usage")
    op.drop_index("ix_stage_executions_job_id", table_name="stage_executions")
    op.drop_table("stage_executions")
    op.drop_index("ix_pipeline_jobs_tenant_id", table_name="pipeline_jobs")
    op.drop_index("ix_pipeline_jobs_job_id", table_name="pipeline_jobs")
    op.drop_table("pipeline_jobs")
