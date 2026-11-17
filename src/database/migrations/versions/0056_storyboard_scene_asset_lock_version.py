"""storyboard_scenes/assets.lock_version — optimistic-concurrency token

Additive-only: adds ``lock_version`` (``integer NOT NULL DEFAULT 0``) to
``storyboard_scenes`` AND ``storyboard_assets`` — the opt-in optimistic-concurrency
token that closes PATCH-vs-PATCH lost updates on a scene / asset, mirroring
``storyboard_panels.lock_version`` (migration 0055).

* Constant server-default → a metadata-only ``ADD COLUMN`` in pg11+ (no table
  rewrite; the default is materialised lazily via pg's fast-default machinery),
  so this is safe to run against a live, large table.
* NON-DESTRUCTIVE, which is the deploy-ordering point: an OLDER image that has
  never heard of ``lock_version`` still boots and writes against the migrated DB
  — the column simply defaults to 0 and is ignored — so the migration can land
  ahead of the code that reads/bumps it without a lock-step image pin.

Both are plain columns the repository bumps by hand (``lock_version = lock_version
+ 1``), deliberately NOT SQLAlchemy ``version_id_col``: version-id tracking would
force a compare-and-set on EVERY write and clash with the merge delete-race
``StaleDataError`` guard. The CAS is opt-in per request via ``expected_lock_version``;
absent it, writes stay last-write-wins.

Revision ID: 0056
Revises: 0055
Create Date: 2026-07-20
"""

import sqlalchemy as sa
from alembic import op

revision = "0056"
down_revision = "0055"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "storyboard_scenes",
        sa.Column("lock_version", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "storyboard_assets",
        sa.Column("lock_version", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )


def downgrade() -> None:
    op.drop_column("storyboard_assets", "lock_version")
    op.drop_column("storyboard_scenes", "lock_version")
