"""Studio v3 reset — drop every legacy table; create new project/deliverable/run schema.

This migration is a one-way door. There is no downgrade. Operator MUST run
`pg_dump $DATABASE_URL > pre-0015.sql` immediately before applying in any
environment that holds data worth keeping. See spec §8.10.

Revision ID: 0015
Revises: 0014
Create Date: 2026-05-08
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


LEGACY_TABLES = [
    "pipeline_evaluations",
    "pipeline_usage",
    "stage_executions",
    "stage_drafts",
    "annotations",
    "verbatim_items",
    "pipeline_jobs",
    "deliverables",
    "projects",
]


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing = set(insp.get_table_names())
    # CASCADE because legacy FKs link these tables in a tangled order.
    for tbl in LEGACY_TABLES:
        if tbl in existing:
            op.execute(f'DROP TABLE IF EXISTS "{tbl}" CASCADE')

    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')

    op.create_table(
        "projects",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=True), nullable=False),
        sa.Column("owner_subject_id", UUID(as_uuid=True), nullable=False),
        sa.Column("visibility", sa.Text(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("brief", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("brand", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("voice", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("theme", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "parent_project_id",
            UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("position", sa.Integer(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="drafting"),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "visibility IN ('private','tenant','public')",
            name="ck_projects_visibility",
        ),
        sa.CheckConstraint(
            "type IN ('promo','course','app_review','generic','lesson')",
            name="ck_projects_type",
        ),
        sa.CheckConstraint(
            "status IN ('drafting','producing','completed','archived')",
            name="ck_projects_status",
        ),
    )
    op.create_index("idx_projects_tenant", "projects", ["tenant_id", "status"])
    op.create_index(
        "idx_projects_parent",
        "projects",
        ["parent_project_id", "position"],
        postgresql_where=sa.text("parent_project_id IS NOT NULL"),
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION enforce_lesson_parent_is_course()
        RETURNS trigger AS $$
        DECLARE parent_type TEXT;
        BEGIN
          IF NEW.type = 'lesson' THEN
            IF NEW.parent_project_id IS NULL THEN
              RAISE EXCEPTION 'lesson projects must have a parent_project_id';
            END IF;
            SELECT type INTO parent_type FROM projects WHERE id = NEW.parent_project_id;
            IF parent_type IS DISTINCT FROM 'course' THEN
              RAISE EXCEPTION 'lesson projects must have parent of type course (got %)', parent_type;
            END IF;
          END IF;
          RETURN NEW;
        END $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_lesson_parent_is_course
          BEFORE INSERT OR UPDATE ON projects
          FOR EACH ROW EXECUTE FUNCTION enforce_lesson_parent_is_course();
        """
    )

    op.create_table(
        "deliverables",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "project_id",
            UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("format", sa.Text(), nullable=False),
        sa.Column("captions_mode", sa.Text(), nullable=False, server_default="none"),
        sa.Column("target_duration_seconds", sa.Integer(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="idle"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "format IN ('16x9','9x16','1x1','4x5')",
            name="ck_deliverables_format",
        ),
        sa.CheckConstraint(
            "captions_mode IN ('none','burned_in','overlay')",
            name="ck_deliverables_captions",
        ),
        sa.CheckConstraint(
            "status IN ('idle','running','awaiting_review','completed','failed')",
            name="ck_deliverables_status",
        ),
    )
    op.create_index("idx_deliverables_project", "deliverables", ["project_id"])

    op.create_table(
        "runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "deliverable_id",
            UUID(as_uuid=True),
            sa.ForeignKey("deliverables.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "parent_run_id",
            UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("blueprint_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("current_stage", sa.Text(), nullable=True),
        sa.Column(
            "config_snapshot",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "pause_after_stages",
            ARRAY(sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('pending','running','paused','completed','failed','cancelled')",
            name="ck_runs_status",
        ),
    )
    op.create_index(
        "idx_runs_deliverable",
        "runs",
        ["deliverable_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "idx_runs_parent",
        "runs",
        ["parent_run_id"],
        postgresql_where=sa.text("parent_run_id IS NOT NULL"),
    )
    op.create_index(
        "idx_runs_active",
        "runs",
        ["status"],
        postgresql_where=sa.text("status IN ('pending','running','paused')"),
    )

    op.create_table(
        "stage_outputs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage_name", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("output", JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("run_id", "stage_name", name="uq_stage_outputs_run_stage"),
        sa.CheckConstraint(
            "status IN ('pending','running','completed','failed','skipped')",
            name="ck_stage_outputs_status",
        ),
    )
    op.create_index("idx_stage_outputs_run", "stage_outputs", ["run_id"])

    op.create_table(
        "annotations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "stage_output_id",
            UUID(as_uuid=True),
            sa.ForeignKey("stage_outputs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("element_id", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="open"),
        sa.Column(
            "applied_in_run_id",
            UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('open','applied','dismissed')",
            name="ck_annotations_status",
        ),
    )
    op.create_index(
        "idx_annotations_stage_output",
        "annotations",
        ["stage_output_id", "status"],
    )
    op.create_index(
        "idx_annotations_open",
        "annotations",
        ["stage_output_id"],
        postgresql_where=sa.text("status = 'open'"),
    )

    op.create_table(
        "run_usage",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage_name", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("provider", sa.Text(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Numeric(12, 6), nullable=False, server_default="0"),
        sa.Column("metadata", JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("idx_run_usage_run", "run_usage", ["run_id"])
    op.create_index("idx_run_usage_stage", "run_usage", ["run_id", "stage_name"])

    op.create_table(
        "verbatim_items",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "project_id",
            UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_verbatim_project",
        "verbatim_items",
        ["project_id", "position"],
    )


def downgrade() -> None:
    raise NotImplementedError(
        "Migration 0015 is a one-way door (Studio v3 reset). "
        "Restore from `pre-0015.sql` pg_dump if rollback is required."
    )
