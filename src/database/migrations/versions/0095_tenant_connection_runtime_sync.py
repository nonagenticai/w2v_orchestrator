"""tenant_connections.runtime_sync — the worker_management mirror status of a connection

On PUT of an ``llm`` / ``vlm`` / ``tts`` / ``asr`` / ``image`` connection the
orchestrator mirrors it into worker_management's tenant-config so the satellite
services use it too (``src/services/providers/wm_sync.py``). The mirror is
best-effort; its outcome ``{"status", "detail", "synced_at"}`` is stored here and
shown by ``GET /api/tenant/connections``. NULL = never mirrored (rows written
before this migration).

On ``tenant_connections`` and not on the version row: versions are immutable
configuration that runs pin, while the sync status is a mutable lifecycle stamp
like ``last_test_*``. New column on an RLS-enabled table: the tenant policy is
row-level and column-agnostic, so no policy change is needed.

Drift-tolerant (``IF NOT EXISTS`` / ``IF EXISTS``), like 0094.

Revision ID: 0095_tenant_conn_runtime_sync
Revises: 0094_runs_paused_time
"""

from __future__ import annotations

from alembic import op

# ``version_num`` is varchar(32); this id is 29 characters.
revision = "0095_tenant_conn_runtime_sync"
down_revision = "0094_runs_paused_time"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE tenant_connections ADD COLUMN IF NOT EXISTS runtime_sync JSONB")


def downgrade() -> None:
    op.execute("ALTER TABLE tenant_connections DROP COLUMN IF EXISTS runtime_sync")
