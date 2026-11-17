"""annotations table — element- or stage-pinned, lifecycle: open|applied|dismissed

Persistent annotation substrate that replaces the one-shot ``feedback_note``
parameter on the regenerate endpoints. Each row pins to a stage (and
optionally a single element within that stage's output) and queues up
``open`` until the operator selects N annotation_ids on a regenerate
call. The regen flow flips the selected rows to ``applied`` and stamps
``applied_in_draft_id`` for traceability.

Scope rules:
- ``target_element_id`` NULL ⇒ the annotation is pinned to the whole stage.
- ``target_element_id`` non-NULL ⇒ the annotation is pinned to a single
  element id within ``stage_executions.artifact_uris`` for that stage.

Status is a free-text column constrained by CHECK to {open, applied,
dismissed} — same convention as ``stage_drafts.source`` /
``verbatim_items.status`` (no postgres ENUM type).

Indices:
- ``(job_id, stage_name)`` covers list-by-stage.
- partial ``(job_id, stage_name) WHERE status = 'open'`` covers the
  hot-path used by the regenerate endpoint to validate that selected
  ids are still open.

Revision ID: 0013
Revises: 0012
Create Date: 2026-05-08
"""

import sqlalchemy as sa

from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "annotations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "job_id",
            sa.String(length=36),
            sa.ForeignKey("pipeline_jobs.job_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage_name", sa.String(length=50), nullable=False),
        sa.Column("target_element_id", sa.String(length=64), nullable=True),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default="open",
        ),
        sa.Column(
            "applied_in_draft_id",
            sa.String(length=36),
            sa.ForeignKey("stage_drafts.draft_id", ondelete="SET NULL"),
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
            "status IN ('open', 'applied', 'dismissed')",
            name="ck_annotations_status",
        ),
    )
    op.create_index(
        "ix_annotations_job_stage",
        "annotations",
        ["job_id", "stage_name"],
    )
    op.create_index(
        "ix_annotations_open",
        "annotations",
        ["job_id", "stage_name"],
        postgresql_where=sa.text("status = 'open'"),
    )


def downgrade() -> None:
    op.drop_index("ix_annotations_open", table_name="annotations")
    op.drop_index("ix_annotations_job_stage", table_name="annotations")
    op.drop_table("annotations")
