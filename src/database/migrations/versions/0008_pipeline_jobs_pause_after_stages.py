"""Add pause_after_stages text[] column to pipeline_jobs.

The column carries the set of stage names at which the executor must
pause for human review before proceeding. Empty / NULL means "never
pause" (canonical autopilot behavior). Initial value is set at run-start
from ``CreatePipelineRequest.pause_after_stages``; mutable mid-run via
``PATCH /api/pipeline/{job_id}/checkpoints``.

Postgres-native ARRAY of TEXT — psycopg3 round-trips it as a Python list.
No CHECK constraint on element values: the route layer validates each
name against the job's enabled stages (a CHECK here would force a list
literal which can't reference another column).

Revision ID: 0008
Revises: 0007
Create Date: 2026-05-07
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "pipeline_jobs",
        sa.Column(
            "pause_after_stages",
            ARRAY(sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("pipeline_jobs", "pause_after_stages")
