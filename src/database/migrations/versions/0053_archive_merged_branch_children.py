"""archive the child projects of already-merged storyboard branches (data-only backfill)

Forward-only data hygiene, NO schema change. Before this migration ``merge`` closed the
branch (``storyboard_branches.status='merged'``) but never archived the branch's CHILD
project, so a merged branch left a second, live+editable copy of the merged work behind
that ``abandon`` could never reach (``abandon`` refuses an already-merged branch). The
code fix (``merge_service.merge`` now archives the child in the same transaction) covers
every NEW merge; this backfills the ones that merged before the fix.

Idempotent: it only touches child projects that are still ``is_branch = true`` and not
already ``'archived'``. Prod currently has 0 branches, so this affects 0 rows today — it
exists for forward hygiene and any branches merged before the next deploy.

Revision ID: 0053
Revises: 0052
Create Date: 2026-07-18
"""

import sqlalchemy as sa
from alembic import op

revision = "0053"
down_revision = "0052"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            UPDATE projects SET status='archived', archived_at=now()
            WHERE is_branch = true AND status <> 'archived'
              AND id IN (SELECT child_project_id FROM storyboard_branches WHERE status='merged')
            """
        )
    )


def downgrade() -> None:
    # No-op by design. This is a data-only backfill; each affected project's PRIOR status
    # (it could have been any non-archived value) is unrecoverable once overwritten, and
    # pre-prod is forward-only, so there is nothing faithful to restore. Leaving the rows
    # archived is the correct, lossless downgrade.
    pass
