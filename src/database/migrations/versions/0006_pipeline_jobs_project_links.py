"""Add project_id + deliverable_id FK columns to pipeline_jobs.

Both columns are nullable so existing project-less jobs (every row prior
to the video-content-studio rollout) remain valid. Deliverables that own
a pipeline_job link back via ``deliverables.pipeline_job_id``; this side
of the relationship is the back-pointer that lets stage handlers /
notification consumers know which Project + Deliverable a job belongs to
without joining through ``deliverables``.

Revision ID: 0006
Revises: 0005
Create Date: 2026-05-07
"""

import sqlalchemy as sa

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "pipeline_jobs",
        sa.Column(
            "project_id",
            sa.String(length=36),
            sa.ForeignKey("projects.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "pipeline_jobs",
        sa.Column(
            "deliverable_id",
            sa.String(length=36),
            sa.ForeignKey("deliverables.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_pipeline_jobs_project_id", "pipeline_jobs", ["project_id"])
    op.create_index("ix_pipeline_jobs_deliverable_id", "pipeline_jobs", ["deliverable_id"])


def downgrade() -> None:
    op.drop_index("ix_pipeline_jobs_deliverable_id", table_name="pipeline_jobs")
    op.drop_index("ix_pipeline_jobs_project_id", table_name="pipeline_jobs")
    op.drop_column("pipeline_jobs", "deliverable_id")
    op.drop_column("pipeline_jobs", "project_id")
