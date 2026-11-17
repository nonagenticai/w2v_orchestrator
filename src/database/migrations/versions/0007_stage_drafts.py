"""Stage drafts substrate for the video content studio.

Each row is one snapshot of a stage's output. Drafts are the eventual
backing store for ``PipelineContext.results`` and the substrate for the
phase-3 pause-and-steer mechanics (whole-stage edits, regenerate-with-
feedback, element-level CRUD). Phase 2 ships only the table + repository
+ GET endpoints; nothing writes drafts on stage completion yet — drafts
appear lazily by backfill from ``stage_executions.artifact_uris`` when
read.

Key shape decisions:

- ``draft_id`` is a String(36) UUID (uuid7) to match every other primary
  key in this codebase.
- ``job_id`` is the **public** ``pipeline_jobs.job_id`` (also String(36)),
  not the integer surrogate. Mirrors ``stage_executions``.
- ``source`` is a free-text column constrained by CHECK to keep the four
  source kinds (``llm_initial``, ``llm_regen``, ``llm_element_regen``,
  ``user_edit``) without introducing a postgres ENUM type.
- ``is_accepted`` is enforced exactly-one-per-(job_id, stage_name) via a
  partial unique index — postgres lets multiple rows be ``false`` and
  exactly one ``true``.
- Index ``(job_id, stage_name, created_at DESC)`` covers the GET /drafts
  list query (newest first) and the retention prune scan.

Revision ID: 0007
Revises: 0006
Create Date: 2026-05-07
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "stage_drafts",
        sa.Column("draft_id", sa.String(length=36), primary_key=True),
        sa.Column(
            "job_id",
            sa.String(length=36),
            sa.ForeignKey("pipeline_jobs.job_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage_name", sa.String(length=50), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column(
            "parent_draft_id",
            sa.String(length=36),
            sa.ForeignKey("stage_drafts.draft_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("feedback_note", sa.Text(), nullable=True),
        sa.Column("target_element_id", sa.String(length=64), nullable=True),
        sa.Column("content_json", JSONB(), nullable=False),
        sa.Column(
            "is_accepted",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "source IN ('llm_initial', 'llm_regen', 'llm_element_regen', 'user_edit')",
            name="ck_stage_drafts_source",
        ),
    )
    op.create_index(
        "ix_stage_drafts_job_stage_created",
        "stage_drafts",
        ["job_id", "stage_name", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_stage_drafts_parent",
        "stage_drafts",
        ["parent_draft_id"],
    )
    op.create_index(
        "uq_stage_drafts_accepted",
        "stage_drafts",
        ["job_id", "stage_name"],
        unique=True,
        postgresql_where=sa.text("is_accepted"),
    )


def downgrade() -> None:
    op.drop_index("uq_stage_drafts_accepted", table_name="stage_drafts")
    op.drop_index("ix_stage_drafts_parent", table_name="stage_drafts")
    op.drop_index("ix_stage_drafts_job_stage_created", table_name="stage_drafts")
    op.drop_table("stage_drafts")
