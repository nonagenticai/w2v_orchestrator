"""Drop tenant_connections.runtime_sync — connections are no longer mirrored to WM

Owner decision 2026-10-20: w2v keeps its OWN provider store and stops mirroring
tenant provider registrations into worker_management (the satellites use the
platform model; tenant BYO is honoured only by the orchestrator pipeline). The
mirror's status column (0095) has no writer or reader left, so it goes.

Downgrade re-adds the column empty (the stamps it held are not recoverable, and
nothing would write them again).

Drift-tolerant (``IF EXISTS`` / ``IF NOT EXISTS``), like 0095.

Revision ID: 0096_drop_conn_runtime_sync
Revises: 0095_tenant_conn_runtime_sync
"""

from __future__ import annotations

from alembic import op

# ``version_num`` is varchar(32); this id is 27 characters.
revision = "0096_drop_conn_runtime_sync"
down_revision = "0095_tenant_conn_runtime_sync"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE tenant_connections DROP COLUMN IF EXISTS runtime_sync")


def downgrade() -> None:
    op.execute("ALTER TABLE tenant_connections ADD COLUMN IF NOT EXISTS runtime_sync JSONB")
