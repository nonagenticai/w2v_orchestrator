"""run_usage: unique (run_id, metadata->>'call_id') for service-reported usage

Services that call the platform vLLM directly (w2v_common ``LLM_BACKEND=vllm``)
report each call to ``POST /api/runs/{run_id}/usage``; the reporter's ``call_id``
lands in ``run_usage.metadata.call_id``. This partial unique index makes a retried
report idempotent (``INSERT … ON CONFLICT DO NOTHING``). Rows without a ``call_id``
(billing-mirrored, in-process) are outside the predicate and unaffected.

Drift-tolerant (``IF NOT EXISTS`` / ``IF EXISTS``), like 0083/0087: a schema that
``create_all`` already built from the models carries the index.

Revision ID: 0093_run_usage_call_id_unique
Revises: 0092_admin_active_runs_paused
"""

from __future__ import annotations

from alembic import op

# ``version_num`` is varchar(32); this id is 29 characters.
revision = "0093_run_usage_call_id_unique"
down_revision = "0092_admin_active_runs_paused"
branch_labels = None
depends_on = None

_INDEX = "uq_run_usage_run_call_id"


def upgrade() -> None:
    op.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS {_INDEX} ON run_usage (run_id, (metadata ->> 'call_id')) "
        "WHERE (metadata ->> 'call_id') IS NOT NULL"
    )


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {_INDEX}")
