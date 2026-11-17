"""Per-user-within-tenant visibility on owned tables.

Adds (tenant_id, subject_id, visibility) to PipelineJob, StageExecution and
PipelineEvaluationRecord and backfills:

    tenant_id  := existing tenant_id (PipelineJob) or parent job's tenant_id
                  (StageExecution, PipelineEvaluationRecord)
    subject_id := tenant_id (legacy "tenant-only caller" semantics)
    visibility := 'tenant' (every tenant member sees + may mutate, bit-for-bit
                  match for pre-Plan-2 behavior)

PipelineUsage is intentionally untouched: it is a tenant-level aggregated
counter table for billing and rate-limiting, with no per-user concept.

Cross-user mutation returns 404 (don't leak existence — matches the
secrets / BOLA pattern documented in the canonical standard).

Plan-3 expansion of the per-user-within-tenant isolation rollout aligned to
/Users/mini1/.claude/plans/per-user-isolation-canonical-standard-2026-04-27.md.

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-27
"""

import sqlalchemy as sa

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

# Tables that gain the canonical (tenant_id, subject_id, visibility) triple.
# PipelineJob already has tenant_id; StageExecution + PipelineEvaluationRecord
# inherit it via FK to pipeline_jobs.job_id and need it added explicitly so
# scope() can filter without joining.
TABLES = (
    "pipeline_jobs",
    "stage_executions",
    "pipeline_evaluations",
)

# Per-table column-presence map: pipeline_jobs already has tenant_id; the
# child tables get it added in this migration and backfilled from their
# parent job.
HAS_TENANT_ID = {
    "pipeline_jobs": True,
    "stage_executions": False,
    "pipeline_evaluations": False,
}


def upgrade() -> None:
    for table in TABLES:
        # 1. Add tenant_id (where missing), subject_id, visibility as nullable
        if not HAS_TENANT_ID[table]:
            op.add_column(
                table,
                sa.Column("tenant_id", sa.String(length=255), nullable=True),
            )
        op.add_column(
            table,
            sa.Column("subject_id", sa.String(length=255), nullable=True),
        )
        op.add_column(
            table,
            sa.Column(
                "visibility",
                sa.String(length=16),
                server_default="tenant",
                nullable=True,
            ),
        )

        # 2. Backfill values
        if not HAS_TENANT_ID[table]:
            # Inherit tenant_id from the parent pipeline_jobs row via job_id FK.
            op.execute(
                f"UPDATE {table} t SET tenant_id = pj.tenant_id "  # nosec B608  # noqa: S608 — table list is hard-coded
                f"FROM pipeline_jobs pj WHERE t.job_id = pj.job_id "
                f"AND t.tenant_id IS NULL"
            )
        op.execute(
            f"UPDATE {table} SET "  # nosec B608  # noqa: S608 — table list is hard-coded
            f"subject_id = COALESCE(subject_id, tenant_id), "
            f"visibility = COALESCE(visibility, 'tenant') "
            f"WHERE subject_id IS NULL OR visibility IS NULL"
        )

        # 3. Tighten to NOT NULL now that every row has values
        op.alter_column(table, "tenant_id", nullable=False)
        op.alter_column(table, "subject_id", nullable=False)
        op.alter_column(table, "visibility", nullable=False)

        # 4. Indexes for the new (tenant_id, subject_id) and
        #    (tenant_id, visibility) read paths plus a plain tenant_id index
        #    on the child tables (parent already has one from migration 0001).
        if not HAS_TENANT_ID[table]:
            op.create_index(f"ix_{table}_tenant_id", table, ["tenant_id"])
        op.create_index(f"ix_{table}_subject_id", table, ["subject_id"])
        op.create_index(f"ix_{table}_tenant_subject", table, ["tenant_id", "subject_id"])
        op.create_index(f"ix_{table}_tenant_visibility", table, ["tenant_id", "visibility"])

        # 5. CHECK constraint on visibility values
        op.create_check_constraint(
            f"ck_{table}_visibility",
            table,
            "visibility IN ('tenant', 'user')",
        )


def downgrade() -> None:
    # Reverse order: drop check + indexes + columns. visibility='user' rows
    # lose their privacy boundary on downgrade — there is no equivalent in
    # the pre-Plan-3 schema.
    for table in TABLES:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS ck_{table}_visibility")
        op.execute(f"DROP INDEX IF EXISTS ix_{table}_tenant_visibility")
        op.execute(f"DROP INDEX IF EXISTS ix_{table}_tenant_subject")
        op.execute(f"DROP INDEX IF EXISTS ix_{table}_subject_id")
        if not HAS_TENANT_ID[table]:
            op.execute(f"DROP INDEX IF EXISTS ix_{table}_tenant_id")

        op.drop_column(table, "visibility")
        op.drop_column(table, "subject_id")
        if not HAS_TENANT_ID[table]:
            op.drop_column(table, "tenant_id")
