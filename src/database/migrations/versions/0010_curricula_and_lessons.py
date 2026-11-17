"""Curriculum + curriculum_lessons substrate for the video content studio.

A Curriculum groups training_course Projects in strict order with shared
brand / voice / theme knobs. The ``curriculum_lessons`` table is the
ordered syllabus: each row at ``position`` 1..N points at one Project
(content_kind=training_course, curriculum_id set) plus an
``accepted_summary_json`` column that mirrors the lesson's
``lesson_summarize`` tail-stage output for the next lesson to consume
as ``prior_lesson_summary`` context.

Strict ordering is enforced server-side: starting lesson N requires
lesson N-1 to be ``completed``. The ``(curriculum_id, position)`` unique
constraint pins position uniqueness; the route layer's check covers
"start lesson 3 while lesson 2 is in_progress".

Phase-7 also tightens the ``projects.curriculum_id`` column (created
nullable + indexed in migration 0005) by adding the FK constraint to
``curricula(id) ON DELETE SET NULL``. The column itself is left
nullable: not every Project belongs to a curriculum.

Revision ID: 0010
Revises: 0009
Create Date: 2026-05-07
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── curricula ─────────────────────────────────────────────────────
    op.create_table(
        "curricula",
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
        sa.Column("audience", JSONB(), nullable=True),
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
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default="designing",
        ),
        sa.Column("design_pipeline_job_id", sa.String(length=36), nullable=True),
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
            "status IN ('designing', 'authoring', 'completed')",
            name="ck_curricula_status",
        ),
        sa.CheckConstraint(
            "visibility IN ('tenant', 'user')",
            name="ck_curricula_visibility",
        ),
    )
    op.create_index("ix_curricula_tenant_id", "curricula", ["tenant_id"])
    op.create_index("ix_curricula_subject_id", "curricula", ["subject_id"])
    op.create_index(
        "ix_curricula_tenant_subject",
        "curricula",
        ["tenant_id", "subject_id"],
    )
    op.create_index(
        "ix_curricula_tenant_visibility",
        "curricula",
        ["tenant_id", "visibility"],
    )

    # ── curriculum_lessons ────────────────────────────────────────────
    op.create_table(
        "curriculum_lessons",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "curriculum_id",
            sa.String(length=36),
            sa.ForeignKey("curricula.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("learning_objective", sa.Text(), nullable=True),
        sa.Column("estimated_duration_minutes", sa.Integer(), nullable=True),
        sa.Column(
            "prereqs",
            PG_ARRAY(sa.Integer()),
            nullable=True,
        ),
        sa.Column(
            "project_id",
            sa.String(length=36),
            sa.ForeignKey("projects.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("accepted_summary_json", JSONB(), nullable=True),
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
            "status IN ('pending', 'in_progress', 'completed')",
            name="ck_curriculum_lessons_status",
        ),
        sa.CheckConstraint("position >= 1", name="ck_curriculum_lessons_position_positive"),
        sa.UniqueConstraint(
            "curriculum_id",
            "position",
            name="uq_curriculum_lessons_position",
        ),
    )
    op.create_index(
        "ix_curriculum_lessons_curriculum_id",
        "curriculum_lessons",
        ["curriculum_id"],
    )
    op.create_index(
        "ix_curriculum_lessons_project_id",
        "curriculum_lessons",
        ["project_id"],
    )

    # ── tighten projects.curriculum_id with the FK ──────────────────
    op.create_foreign_key(
        "fk_projects_curriculum_id",
        "projects",
        "curricula",
        ["curriculum_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_projects_curriculum_id", "projects", type_="foreignkey")

    op.drop_index("ix_curriculum_lessons_project_id", table_name="curriculum_lessons")
    op.drop_index(
        "ix_curriculum_lessons_curriculum_id",
        table_name="curriculum_lessons",
    )
    op.drop_table("curriculum_lessons")

    op.drop_index("ix_curricula_tenant_visibility", table_name="curricula")
    op.drop_index("ix_curricula_tenant_subject", table_name="curricula")
    op.drop_index("ix_curricula_subject_id", table_name="curricula")
    op.drop_index("ix_curricula_tenant_id", table_name="curricula")
    op.drop_table("curricula")
