"""Fix timestamp columns to be timezone-aware and add missing indexes.

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-13

Background:
  Migration 0001 declared timestamp columns with ``sa.DateTime()`` instead of
  ``sa.DateTime(timezone=True)``, so alembic-bootstrapped databases (prod) ended
  up with ``timestamp without time zone`` columns even though the ORM models
  declare ``DateTime(timezone=True)``. This migration converts those columns to
  ``timestamptz``. The ``USING ... AT TIME ZONE 'UTC'`` cast is correct because
  the orchestrator writes all datetimes as UTC and the container runs in UTC.

  This migration also creates ``idx_stage_job_name`` which was declared on the
  ORM model but never added by a migration.

  Safe to run on prod (converts naive columns to tz) and on a fresh DB bootstrap
  going through 0001 -> 0002 -> 0003 (re-applies idempotent TYPE casts).
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # pipeline_jobs
    op.execute(
        "ALTER TABLE pipeline_jobs "
        "ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC'"
    )
    op.execute(
        "ALTER TABLE pipeline_jobs "
        "ALTER COLUMN updated_at TYPE TIMESTAMPTZ USING updated_at AT TIME ZONE 'UTC'"
    )

    # stage_executions
    op.execute(
        "ALTER TABLE stage_executions "
        "ALTER COLUMN started_at TYPE TIMESTAMPTZ USING started_at AT TIME ZONE 'UTC'"
    )
    op.execute(
        "ALTER TABLE stage_executions "
        "ALTER COLUMN completed_at TYPE TIMESTAMPTZ USING completed_at AT TIME ZONE 'UTC'"
    )

    # pipeline_usage
    op.execute(
        "ALTER TABLE pipeline_usage "
        "ALTER COLUMN date TYPE TIMESTAMPTZ USING date AT TIME ZONE 'UTC'"
    )

    # Missing composite index declared on the ORM model but never created.
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_stage_job_name ON stage_executions (job_id, stage_name)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_stage_job_name")
    op.execute(
        "ALTER TABLE pipeline_usage ALTER COLUMN date TYPE TIMESTAMP USING date AT TIME ZONE 'UTC'"
    )
    op.execute(
        "ALTER TABLE stage_executions "
        "ALTER COLUMN completed_at TYPE TIMESTAMP USING completed_at AT TIME ZONE 'UTC'"
    )
    op.execute(
        "ALTER TABLE stage_executions "
        "ALTER COLUMN started_at TYPE TIMESTAMP USING started_at AT TIME ZONE 'UTC'"
    )
    op.execute(
        "ALTER TABLE pipeline_jobs "
        "ALTER COLUMN updated_at TYPE TIMESTAMP USING updated_at AT TIME ZONE 'UTC'"
    )
    op.execute(
        "ALTER TABLE pipeline_jobs "
        "ALTER COLUMN created_at TYPE TIMESTAMP USING created_at AT TIME ZONE 'UTC'"
    )
