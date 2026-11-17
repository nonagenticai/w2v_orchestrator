"""Drop the orphaned ``projects.curriculum_id`` column.

Migration 0012 dropped the curricula + curriculum_lessons tables but
intentionally left ``projects.curriculum_id`` in place because
application-side callers (route, schemas, repository) still referenced
it. Those references are now scrubbed; the column is dead weight.

Revision ID: 0014
Revises: 0013
Create Date: 2026-05-08
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {c["name"] for c in insp.get_columns("projects")}
    if "curriculum_id" not in cols:
        return
    indexes = {ix["name"] for ix in insp.get_indexes("projects")}
    if "ix_projects_curriculum_id" in indexes:
        op.drop_index("ix_projects_curriculum_id", table_name="projects")
    op.drop_column("projects", "curriculum_id")


def downgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("curriculum_id", sa.String(length=36), nullable=True),
    )
    op.create_index(
        "ix_projects_curriculum_id",
        "projects",
        ["curriculum_id"],
        unique=False,
    )
