"""Project + Deliverable substrate for the video content studio.

Introduces two new entities above ``pipeline_jobs``:

- ``projects``: groups everything that shares a creative brief
  (content_kind ∈ {promo, app_review, training_course, generic}; brand /
  voice / theme as JSONB placeholders until the asset-library plan ships).

- ``deliverables``: each render variant of a Project (e.g. "60s 16x9",
  "30s 9x16"). One Deliverable owns one ``pipeline_jobs`` row.

Both tables carry the canonical (tenant_id, subject_id, visibility) triple
introduced in migration 0004 so per-user-within-tenant isolation works for
Project listings the same way it does for jobs.

A ``curriculum_id`` UUID column is added to ``projects`` now (nullable, no
FK constraint); the FK to ``curricula(id)`` is added in the phase-7
migration that creates the curriculum tables.

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-07
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── projects ──────────────────────────────────────────────────────
    op.create_table(
        "projects",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.String(length=255), nullable=False),
        sa.Column("subject_id", sa.String(length=255), nullable=False),
        sa.Column(
            "visibility",
            sa.String(length=16),
            nullable=False,
            server_default="tenant",
        ),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column(
            "content_kind",
            sa.String(length=32),
            nullable=False,
            server_default="generic",
        ),
        sa.Column("brief", JSONB(), nullable=True),
        sa.Column(
            "brand_config",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "voice_config",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "theme_config",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("curriculum_id", sa.String(length=36), nullable=True),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default="drafting",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "content_kind IN ('promo', 'app_review', 'training_course', 'generic',"
            " 'curriculum_design')",
            name="ck_projects_content_kind",
        ),
        sa.CheckConstraint(
            "status IN ('drafting', 'producing', 'completed', 'archived')",
            name="ck_projects_status",
        ),
        sa.CheckConstraint(
            "visibility IN ('tenant', 'user')",
            name="ck_projects_visibility",
        ),
    )
    op.create_index("ix_projects_tenant_id", "projects", ["tenant_id"])
    op.create_index("ix_projects_subject_id", "projects", ["subject_id"])
    op.create_index("ix_projects_tenant_subject", "projects", ["tenant_id", "subject_id"])
    op.create_index("ix_projects_tenant_visibility", "projects", ["tenant_id", "visibility"])
    op.create_index("ix_projects_curriculum_id", "projects", ["curriculum_id"])

    # ── deliverables ──────────────────────────────────────────────────
    op.create_table(
        "deliverables",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "project_id",
            sa.String(length=36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("format", sa.String(length=8), nullable=False),
        sa.Column("duration_target_seconds", sa.Integer(), nullable=True),
        sa.Column(
            "captions_mode",
            sa.String(length=8),
            nullable=False,
            server_default="none",
        ),
        sa.Column("pipeline_job_id", sa.String(length=36), nullable=True),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "format IN ('16x9', '9x16', '1x1', '4x5')",
            name="ck_deliverables_format",
        ),
        sa.CheckConstraint(
            "captions_mode IN ('none', 'burned', 'overlay')",
            name="ck_deliverables_captions_mode",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'awaiting_review', 'completed', 'failed')",
            name="ck_deliverables_status",
        ),
    )
    op.create_index("ix_deliverables_project_id", "deliverables", ["project_id"])
    op.create_index("ix_deliverables_pipeline_job_id", "deliverables", ["pipeline_job_id"])


def downgrade() -> None:
    op.drop_index("ix_deliverables_pipeline_job_id", table_name="deliverables")
    op.drop_index("ix_deliverables_project_id", table_name="deliverables")
    op.drop_table("deliverables")

    op.drop_index("ix_projects_curriculum_id", table_name="projects")
    op.drop_index("ix_projects_tenant_visibility", table_name="projects")
    op.drop_index("ix_projects_tenant_subject", table_name="projects")
    op.drop_index("ix_projects_subject_id", table_name="projects")
    op.drop_index("ix_projects_tenant_id", table_name="projects")
    op.drop_table("projects")
