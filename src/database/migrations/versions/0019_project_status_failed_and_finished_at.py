"""project status 'failed' + finished_at timestamp

Projects never transitioned past 'drafting' because nothing aggregated run
lifecycle up to the parent project. The dispatcher now recomputes project
status on every run transition (drafting → producing → completed/failed).
Two schema changes back that:

* ``projects.status`` gains a ``failed`` value (a deliverable's latest run
  failed/cancelled and nothing is still active). The CHECK constraint is
  drop+recreated, same idempotent shape as 0018.
* ``projects.finished_at`` (nullable timestamptz) records when the project
  reached 'completed' (all deliverables done). Sortable; cleared if the
  project ever leaves 'completed'.

Schema-only. Existing rows are reconciled out-of-band by
``scripts/backfill_project_status.py`` (run once post-deploy) — embedding the
per-deliverable-latest-run aggregation in a migration would duplicate the
repository logic.

Revision ID: 0019
Revises: 0018
Create Date: 2026-06-03
"""

import sqlalchemy as sa
from alembic import op

revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None

_STATUS_OLD = "status IN ('drafting','producing','completed','archived')"
_STATUS_NEW = "status IN ('drafting','producing','completed','failed','archived')"


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.drop_constraint("ck_projects_status", "projects", type_="check")
    op.create_check_constraint("ck_projects_status", "projects", _STATUS_NEW)


def downgrade() -> None:
    # Any rows already at the new 'failed' value would violate the restored
    # old CHECK — remap them to 'producing' first.
    op.execute("UPDATE projects SET status = 'producing' WHERE status = 'failed'")
    op.drop_constraint("ck_projects_status", "projects", type_="check")
    op.create_check_constraint("ck_projects_status", "projects", _STATUS_OLD)
    op.drop_column("projects", "finished_at")
