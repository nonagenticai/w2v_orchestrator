"""mark runs whose cost has been reconciled from billing

The billing reconciler mirrors a terminal run's priced cost into ``run_usage``.
A genuinely zero-cost run writes no ``run_usage`` rows, so the old "has no rows"
work-list re-queried billing for it on every /usage and /admin load. This adds
``runs.usage_reconciled_at`` so a reconciled run drops out of the sweep whether
or not it had cost. Existing runs that already have usage rows are backfilled as
reconciled so the change does not trigger a one-off re-bill of historical runs.

Revision ID: 0034
Revises: 0033
Create Date: 2026-06-28
"""

import sqlalchemy as sa
from alembic import op

revision = "0034"
down_revision = "0033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column("usage_reconciled_at", sa.DateTime(timezone=True), nullable=True),
    )
    # Runs that already carry usage rows are, by definition, reconciled — mark
    # them so the sweep does not re-bill them once the new predicate goes live.
    op.execute(
        "UPDATE runs SET usage_reconciled_at = now() "
        "WHERE id IN (SELECT DISTINCT run_id FROM run_usage)"
    )


def downgrade() -> None:
    op.drop_column("runs", "usage_reconciled_at")
