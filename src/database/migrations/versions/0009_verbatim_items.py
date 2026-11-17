"""Verbatim items substrate for the video content studio.

Each row is one piece of literal text that must appear verbatim in a
pipeline's output (a slide caption, a TTS-narrated phrase, a stat line).
Items are scoped at one of two levels:

- **Project-scoped** (``project_id`` set, ``job_id`` NULL) — applies to
  every Deliverable / pipeline_job whose ``pipeline_jobs.project_id``
  matches. Inheritable across reruns of the same brief.
- **Job-scoped** (``job_id`` set, ``project_id`` NULL) — applies only
  to the named pipeline_job; used for ad-hoc submissions.

The CHECK constraint enforces exactly-one-of: a row may not be
unscoped (orphan) or doubly-scoped (ambiguous inheritance).

Status is a free-text column constrained by CHECK to ``pending``,
``placed``, or ``rejected`` to keep the four states without introducing
a postgres ENUM type — same convention as ``stage_drafts.source``.

Indices:
- ``(tenant_id, project_id)`` and ``(tenant_id, job_id)`` cover the
  two list endpoints.
- ``(tenant_id, status)`` covers the orchestrator's per-stage scan
  for pending items.

Revision ID: 0009
Revises: 0008
Create Date: 2026-05-07
"""

import sqlalchemy as sa

from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "verbatim_items",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.String(length=255), nullable=False),
        sa.Column("subject_id", sa.String(length=255), nullable=False),
        sa.Column(
            "visibility",
            sa.String(length=16),
            nullable=False,
            server_default="tenant",
        ),
        sa.Column(
            "project_id",
            sa.String(length=36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "job_id",
            sa.String(length=36),
            sa.ForeignKey("pipeline_jobs.job_id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("hint", sa.Text(), nullable=True),
        sa.Column(
            "scope",
            sa.String(length=16),
            nullable=False,
            server_default="pipeline",
        ),
        sa.Column("scope_target", sa.String(length=64), nullable=True),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("placed_in_stage", sa.String(length=50), nullable=True),
        sa.Column("placed_in_element_id", sa.String(length=64), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "(project_id IS NOT NULL AND job_id IS NULL) "
            "OR (project_id IS NULL AND job_id IS NOT NULL)",
            name="ck_verbatim_items_scope_xor",
        ),
        sa.CheckConstraint(
            "kind IN ('slide', 'tts')",
            name="ck_verbatim_items_kind",
        ),
        sa.CheckConstraint(
            "scope IN ('pipeline', 'chapter', 'scene', 'shot', 'slide')",
            name="ck_verbatim_items_scope",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'placed', 'rejected')",
            name="ck_verbatim_items_status",
        ),
        sa.CheckConstraint(
            "visibility IN ('tenant', 'user')",
            name="ck_verbatim_items_visibility",
        ),
    )
    op.create_index(
        "ix_verbatim_items_tenant_project",
        "verbatim_items",
        ["tenant_id", "project_id"],
    )
    op.create_index(
        "ix_verbatim_items_tenant_job",
        "verbatim_items",
        ["tenant_id", "job_id"],
    )
    op.create_index(
        "ix_verbatim_items_tenant_status",
        "verbatim_items",
        ["tenant_id", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_verbatim_items_tenant_status", table_name="verbatim_items")
    op.drop_index("ix_verbatim_items_tenant_job", table_name="verbatim_items")
    op.drop_index("ix_verbatim_items_tenant_project", table_name="verbatim_items")
    op.drop_table("verbatim_items")
