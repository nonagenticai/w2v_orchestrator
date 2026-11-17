"""``runs.review_notice_dismissed_at`` — the in-app "needs your review" notice

Run 01a1770d (promo_v1) waited 29h at its storyboard review checkpoint because nobody
knew it was waiting. Studio now shows the run's owner an in-app notice for every run
paused at a review CHECKPOINT (bell + banner, see ``GET /api/review-inbox``). The
notice is DERIVED from the run row — status, ``pause_reason`` (0102), ``paused_at`` (0094) —
so it cannot drift from the truth; the only state it needs is whether the owner
dismissed it:

* ``review_notice_dismissed_at`` — set by ``POST /api/review-inbox/{run_id}/dismiss``.
  A dismissed notice comes BACK once the run has been waiting 24h (the reminder).
  Resume clears it, so the run's next checkpoint is a fresh notice.

Revision ID: 0104_review_notice_dismissed
Revises: 0103_stage_output_edit_reads
"""

from __future__ import annotations

from alembic import op

# ``version_num`` is varchar(32); this id is 28 characters.
revision = "0104_review_notice_dismissed"
down_revision = "0103_stage_output_edit_reads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS review_notice_dismissed_at TIMESTAMPTZ")


def downgrade() -> None:
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS review_notice_dismissed_at")
