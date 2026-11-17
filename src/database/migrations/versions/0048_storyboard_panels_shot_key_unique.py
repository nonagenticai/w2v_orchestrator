"""storyboard_panels — (project_id, shot_key) UNIQUE (was a plain index)

``shot_key`` holds the sacred per-project ``clip-<n>`` spine join key
(= the Pydantic ``StoryboardPanelBase.id``). It is NOT-NULL and MUST be
unique per project so the spine can be joined on ``(project_id, shot_key)``
without ambiguity. Migration 0046 shipped it as a plain
``idx_storyboard_panels_project_shot`` index, which does NOT enforce
uniqueness. This converts that plain index into a UNIQUE constraint
(``uq_storyboard_panels_project_shot``) — the same index→constraint swap
0047 applied to ``storyboards.project_id``.

Existing duplicates (from before the constraint) are RE-KEYED, not deleted.
Panels are user/AI-authored content with no child rows hanging off them, so
deletion would be destructive with no upside — it would silently drop the
duplicate's authored data. Re-keying (append ``-dup-<id8>`` to the losing
rows' ``shot_key``, keeping the earliest per ``(project_id, shot_key)`` by
``position, created_at, id``) preserves every row and surfaces the conflict
for a human to reconcile rather than hiding it.

NOTE: downgrade restores the plain index but does NOT un-rekey the
re-keyed rows (mirrors 0047's one-way dedup) — the ``-dup-`` suffixes stay.

Revision ID: 0048
Revises: 0047
Create Date: 2026-07-12
"""

from alembic import op

revision = "0048"
down_revision = "0047"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Dedup by re-keying losers: keep the earliest row per
    #    (project_id, shot_key), append -dup-<id8> to every later collision so
    #    the row survives and the conflict is visible. Must run BEFORE the
    #    UNIQUE constraint is added.
    op.execute(
        """
        UPDATE storyboard_panels p
        SET shot_key = p.shot_key || '-dup-' || left(p.id::text, 8)
        FROM (
            SELECT id, row_number() OVER (
                PARTITION BY project_id, shot_key ORDER BY position, created_at, id
            ) AS rn
            FROM storyboard_panels
        ) r
        WHERE p.id = r.id AND r.rn > 1
        """
    )

    # 2. Swap the plain index for a UNIQUE constraint (which also indexes).
    op.drop_index("idx_storyboard_panels_project_shot", table_name="storyboard_panels")
    op.create_unique_constraint(
        "uq_storyboard_panels_project_shot",
        "storyboard_panels",
        ["project_id", "shot_key"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_storyboard_panels_project_shot", "storyboard_panels", type_="unique")
    op.create_index(
        "idx_storyboard_panels_project_shot",
        "storyboard_panels",
        ["project_id", "shot_key"],
    )
