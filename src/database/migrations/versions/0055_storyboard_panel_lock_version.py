"""storyboard_panels.lock_version — optimistic-concurrency token

Additive-only: adds ``storyboard_panels.lock_version`` (``integer NOT NULL
DEFAULT 0``), the opt-in optimistic-concurrency token that closes PATCH-vs-PATCH
lost updates on a panel.

* Constant server-default → a metadata-only ``ADD COLUMN`` in pg11+ (no table
  rewrite; the default is materialised lazily via pg's fast-default machinery),
  so this is safe to run against a live, large ``storyboard_panels``.
* NON-DESTRUCTIVE, which is the deploy-ordering point: an OLDER image that has
  never heard of ``lock_version`` still boots and writes against the migrated DB
  — the column simply defaults to 0 and is ignored — so the migration can land
  ahead of the code that reads/bumps it without a lock-step image pin.

It is a plain column the repository bumps by hand (``lock_version = lock_version
+ 1``), deliberately NOT a SQLAlchemy ``version_id_col``: version-id tracking
would force a compare-and-set on EVERY panel write and clash with the existing
merge delete-race ``StaleDataError`` guard. The CAS is opt-in per request via
``expected_lock_version``; absent it, writes stay last-write-wins.

Revision ID: 0055
Revises: 0054
Create Date: 2026-07-19
"""

import sqlalchemy as sa
from alembic import op

revision = "0055"
down_revision = "0054"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "storyboard_panels",
        sa.Column("lock_version", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )


def downgrade() -> None:
    op.drop_column("storyboard_panels", "lock_version")
