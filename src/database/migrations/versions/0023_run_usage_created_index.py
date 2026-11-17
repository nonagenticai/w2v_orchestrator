"""index run_usage.created_at for the admin usage time-series

Phase B, 8th increment. The admin usage time-series buckets ``run_usage`` by
``date_trunc(created_at)`` and optionally filters to the last N days via a
``created_at >= now() - interval`` predicate. There was no supporting index
(only ``run_id`` / ``run_id+stage_name``), so the range scan + grouping read
the whole table. A plain ascending btree on ``created_at`` serves both the
range predicate and the group-by; DESC buys nothing here (no newest-first
LIMIT, unlike the audit log). ``tenant_id`` lives on ``projects`` not
``run_usage``, so a composite is not possible — a standalone index is correct.

Revision ID: 0023
Revises: 0022
Create Date: 2026-06-15
"""

from alembic import op

revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("idx_run_usage_created", "run_usage", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_run_usage_created", table_name="run_usage")
