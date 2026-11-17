"""project_type rename + recursion fields

Renames ``projects.content_kind`` to ``projects.project_type``, allows the
``'lesson'`` value, and adds the recursion fields (``parent_project_id``,
``position``, ``lesson_summary_json``) that let a Project own child Projects.
A training-course Project owns its lessons as children; a lesson is itself a
full Project that runs the ``training_course`` DAG. Curriculum-level metadata
(audience, syllabus) lives in the parent Project's ``brief`` JSONB.

Note: ``content_kind`` / ``project_type`` is a plain ``String(32)`` with a
CHECK constraint, NOT a postgres ENUM type — the rename is column-level only.
The check is rebuilt with ``'lesson'`` added; ``'curriculum_design'`` is
preserved for backwards-compat with rows produced by vast-teapot phase 7.

Revision ID: 0011
Revises: 0010
Create Date: 2026-05-07
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Drop the old content_kind CHECK; column is plain String(32), not an enum.
    op.drop_constraint("ck_projects_content_kind", "projects", type_="check")

    # ── Rename column content_kind -> project_type. Same type, same data.
    op.alter_column("projects", "content_kind", new_column_name="project_type")

    # ── Rebuild the CHECK with 'lesson' added.
    op.create_check_constraint(
        "ck_projects_project_type",
        "projects",
        "project_type IN ('promo', 'app_review', 'training_course', 'generic',"
        " 'curriculum_design', 'lesson')",
    )

    # ── Recursion fields.
    op.add_column(
        "projects",
        sa.Column("parent_project_id", sa.String(length=36), nullable=True),
    )
    op.add_column("projects", sa.Column("position", sa.Integer(), nullable=True))
    op.add_column("projects", sa.Column("lesson_summary_json", JSONB(), nullable=True))

    # Self-referential FK with ON DELETE CASCADE.
    op.create_foreign_key(
        "fk_projects_parent_project_id",
        "projects",
        "projects",
        ["parent_project_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # Lookup index for child listing.
    op.create_index(
        "ix_projects_parent",
        "projects",
        ["parent_project_id"],
        postgresql_where=sa.text("parent_project_id IS NOT NULL"),
    )

    # Partial unique index: position uniqueness within a parent.
    op.create_index(
        "ix_projects_parent_position",
        "projects",
        ["parent_project_id", "position"],
        unique=True,
        postgresql_where=sa.text("parent_project_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_projects_parent_position", table_name="projects")
    op.drop_index("ix_projects_parent", table_name="projects")
    op.drop_constraint("fk_projects_parent_project_id", "projects", type_="foreignkey")
    op.drop_column("projects", "lesson_summary_json")
    op.drop_column("projects", "position")
    op.drop_column("projects", "parent_project_id")

    op.drop_constraint("ck_projects_project_type", "projects", type_="check")
    op.alter_column("projects", "project_type", new_column_name="content_kind")
    op.create_check_constraint(
        "ck_projects_content_kind",
        "projects",
        "content_kind IN ('promo', 'app_review', 'training_course', 'generic',"
        " 'curriculum_design')",
    )
