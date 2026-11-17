"""storyboard_branches.parent_base_snapshot_id — the SHARED ancestor for sibling merges

Adds the one snapshot two SIBLING branches (A and B, both forked off the same parent) can
three-way merge against symmetrically. Each branch already pins its OWN ``base_snapshot_id``
(the base for merging THAT branch into its parent, which has moved on independently); two
siblings therefore have two DIFFERENT per-branch bases and no single common ancestor. This
column is that common ancestor: the parent's state before ANY sibling diverged.

* Nullable FK → ``storyboard_snapshots`` (the parent's ``branch_base`` snapshot), ``ON
  DELETE SET NULL`` — it references a DIFFERENT branch's base, so a vanished snapshot must
  null out (→ sibling merge simply unavailable for that lineage) rather than reap this
  branch row the way ``base_snapshot_id``'s CASCADE safely can.
* Nullable + NO server default → a metadata-only ``ADD COLUMN`` (no table rewrite), safe on
  a live table. NON-DESTRUCTIVE and forward-compatible: an OLDER image that never heard of
  the column boots and writes fine against the migrated DB (it is ignored), so this can land
  ahead of the code that reads it without a lock-step image pin.

Backfill policy (the "which base?" ambiguity, resolved): for every root, set ALL its
branches' ``parent_base_snapshot_id`` to the ``base_snapshot_id`` of that root's EARLIEST
branch — the closest stored approximation of the state before any sibling diverged. A
single-branch root gets its own base (it is the earliest). Going forward, ``create_branch``
maintains this: the first fork of a root establishes the shared base, later forks reuse it.

Revision ID: 0057
Revises: 0056
Create Date: 2026-07-23
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0057"
down_revision = "0056"
branch_labels = None
depends_on = None

_FK_NAME = "fk_storyboard_branches_parent_base"


def upgrade() -> None:
    op.add_column(
        "storyboard_branches",
        sa.Column("parent_base_snapshot_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        _FK_NAME,
        "storyboard_branches",
        "storyboard_snapshots",
        ["parent_base_snapshot_id"],
        ["id"],
        ondelete="SET NULL",
    )
    # Backfill: every branch of a root shares that root's EARLIEST branch's base as the
    # common ancestor. DISTINCT ON picks the earliest by (created_at, id) per root.
    op.execute(
        """
        UPDATE storyboard_branches AS b
           SET parent_base_snapshot_id = earliest.base_snapshot_id
          FROM (
                SELECT DISTINCT ON (root_project_id)
                       root_project_id,
                       base_snapshot_id
                  FROM storyboard_branches
                 ORDER BY root_project_id, created_at ASC, id ASC
               ) AS earliest
         WHERE b.root_project_id = earliest.root_project_id
        """
    )


def downgrade() -> None:
    op.drop_constraint(_FK_NAME, "storyboard_branches", type_="foreignkey")
    op.drop_column("storyboard_branches", "parent_base_snapshot_id")
