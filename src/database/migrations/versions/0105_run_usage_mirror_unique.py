"""run_usage: one billing-mirrored row per (run, stage, model, provider)

The billing mirror (``RunUsageRepository.replace_for_runs``) is delete-then-insert.
Concurrent reads of one finished run interleaved those statements and each kept
its own copy of the rows: three simultaneous run-detail GETs settled a $0.03 run
at $0.09 / $0.09 / $0.06 and froze it there. The application now serialises the
mirror per run (an advisory lock, ``RunUsageRepository.try_lock_mirror``). This
partial unique index makes a duplicate mirror row impossible regardless.

The mirror now stamps ``metadata.source = 'billing_mirror'`` on every row it
writes, and the index covers exactly those rows. It deliberately does NOT cover
the source-less rows mirrored before this revision: no writer can add to them
any more (the next mirror of a run replaces them with stamped rows), and keying
on "source IS NULL" would also catch any unknown legacy writer that never set a
source. Read-only prod measurement 2026-10-28: 0 duplicated groups across 403
mirrored rows / 97 runs, so nothing needs repairing first.

Numbering: chained behind 0104 (#1165), which follows 0103 (#1160) and
0102 (#1162).

Drift-tolerant (``IF NOT EXISTS`` / ``IF EXISTS``): a schema ``create_all`` built
from the models already carries the index.

Revision ID: 0105_run_usage_mirror_unique
Revises: 0104_review_notice_dismissed
"""

from __future__ import annotations

from alembic import op

# ``version_num`` is varchar(32); this id is 28 characters.
revision = "0105_run_usage_mirror_unique"
down_revision = "0104_review_notice_dismissed"
branch_labels = None
depends_on = None

_INDEX = "uq_run_usage_mirrored_group"


def upgrade() -> None:
    op.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS {_INDEX} ON run_usage "
        "(run_id, stage_name, COALESCE(model, ''), COALESCE(provider, '')) "
        "WHERE (metadata ->> 'source') = 'billing_mirror'"
    )


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {_INDEX}")
