"""Drop curricula tables; hoist lessons into the projects tree.

Replaces the separate ``curricula`` + ``curriculum_lessons`` substrate
with the recursive projects tree introduced in 0011: a course is a
Project of ``project_type='training_course'`` whose lessons are child
Projects (``project_type='lesson'``, ``parent_project_id`` set,
``position`` assigned). Curriculum-level metadata (audience) lands in
the parent's ``brief`` JSONB.

Data migration (pre-prod, best-effort):
  Step A: Create a stub project per curricula row, project_type
          'training_course', brief={'audience': curricula.audience}.
  Step B: For each curriculum_lessons row whose project_id IS NOT NULL,
          UPDATE the linked project to point at the stub via
          parent_project_id, copy position + accepted_summary_json,
          and flip project_type to 'lesson'.
  Step C: Drop the curricula FK on projects, then drop both tables.

The ``projects.curriculum_id`` column is intentionally LEFT in place as
an orphaned (no-FK) String(36) column — callers still reference the
field, and dropping it is deferred to a follow-up migration once those
references are scrubbed.

Revision ID: 0012
Revises: 0011
Create Date: 2026-05-07
"""

from __future__ import annotations

import json
import uuid

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


# Curriculum.status -> Project.status. The mapping is best-effort —
# Curriculum has 3 lifecycle states ('designing', 'authoring',
# 'completed'); Project has 4 ('drafting', 'producing', 'completed',
# 'archived'). Designing maps to drafting, authoring to producing.
_STATUS_MAP = {
    "designing": "drafting",
    "authoring": "producing",
    "completed": "completed",
}


def _new_id() -> str:
    """Generate a 36-char project id (uuid4 rather than uuid7 for portability)."""
    return str(uuid.uuid4())


def upgrade() -> None:
    bind = op.get_bind()

    # ── Step A: per-curriculum stub project ──────────────────────────
    curricula_rows = (
        bind.execute(
            sa.text(
                "SELECT id, tenant_id, subject_id, visibility, title, audience,"
                "       brand_config, voice_config, theme_config, status,"
                "       created_at, updated_at"
                "  FROM curricula"
            )
        )
        .mappings()
        .all()
    )

    cur_to_stub: dict[str, str] = {}
    for cur in curricula_rows:
        stub_id = _new_id()
        cur_to_stub[cur["id"]] = stub_id
        brief = {"audience": cur["audience"]} if cur["audience"] is not None else {}
        bind.execute(
            sa.text(
                "INSERT INTO projects ("
                "  id, tenant_id, subject_id, visibility, title, project_type,"
                "  brief, brand_config, voice_config, theme_config, status,"
                "  created_at, updated_at"
                ") VALUES ("
                "  :id, :tenant_id, :subject_id, :visibility, :title, 'training_course',"
                "  CAST(:brief AS JSONB), CAST(:brand_config AS JSONB),"
                "  CAST(:voice_config AS JSONB), CAST(:theme_config AS JSONB), :status,"
                "  :created_at, :updated_at"
                ")"
            ),
            {
                "id": stub_id,
                "tenant_id": cur["tenant_id"],
                "subject_id": cur["subject_id"],
                "visibility": cur["visibility"],
                "title": cur["title"],
                "brief": json.dumps(brief),
                "brand_config": json.dumps(cur["brand_config"] or {}),
                "voice_config": json.dumps(cur["voice_config"] or {}),
                "theme_config": json.dumps(cur["theme_config"] or {}),
                "status": _STATUS_MAP.get(cur["status"], "drafting"),
                "created_at": cur["created_at"],
                "updated_at": cur["updated_at"],
            },
        )

    # ── Step B: hoist each curriculum_lesson into its project ───────
    lesson_rows = (
        bind.execute(
            sa.text(
                "SELECT curriculum_id, position, project_id, accepted_summary_json"
                "  FROM curriculum_lessons"
                " WHERE project_id IS NOT NULL"
            )
        )
        .mappings()
        .all()
    )

    for lesson in lesson_rows:
        stub_id = cur_to_stub.get(lesson["curriculum_id"])
        if stub_id is None:
            continue
        bind.execute(
            sa.text(
                "UPDATE projects"
                "   SET parent_project_id = :parent_project_id,"
                "       position = :position,"
                "       lesson_summary_json = CAST(:lesson_summary_json AS JSONB),"
                "       project_type = 'lesson'"
                " WHERE id = :project_id"
            ),
            {
                "parent_project_id": stub_id,
                "position": lesson["position"],
                "lesson_summary_json": (
                    json.dumps(lesson["accepted_summary_json"])
                    if lesson["accepted_summary_json"] is not None
                    else None
                ),
                "project_id": lesson["project_id"],
            },
        )

    # ── Step C: drop the FK + indexes + tables ──────────────────────
    op.drop_constraint("fk_projects_curriculum_id", "projects", type_="foreignkey")

    op.drop_index("ix_curriculum_lessons_project_id", table_name="curriculum_lessons")
    op.drop_index("ix_curriculum_lessons_curriculum_id", table_name="curriculum_lessons")
    op.drop_table("curriculum_lessons")

    op.drop_index("ix_curricula_tenant_visibility", table_name="curricula")
    op.drop_index("ix_curricula_tenant_subject", table_name="curricula")
    op.drop_index("ix_curricula_subject_id", table_name="curricula")
    op.drop_index("ix_curricula_tenant_id", table_name="curricula")
    op.drop_table("curricula")


def downgrade() -> None:
    """One-way data migration. Tables are recreated empty for rollback compat."""
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
    op.create_index("ix_curricula_tenant_subject", "curricula", ["tenant_id", "subject_id"])
    op.create_index(
        "ix_curricula_tenant_visibility",
        "curricula",
        ["tenant_id", "visibility"],
    )

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
        sa.Column("prereqs", PG_ARRAY(sa.Integer()), nullable=True),
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
        sa.CheckConstraint(
            "position >= 1",
            name="ck_curriculum_lessons_position_positive",
        ),
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

    # Re-create the FK on projects.curriculum_id.
    op.create_foreign_key(
        "fk_projects_curriculum_id",
        "projects",
        "curricula",
        ["curriculum_id"],
        ["id"],
        ondelete="SET NULL",
    )
