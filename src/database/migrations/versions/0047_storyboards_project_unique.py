"""storyboards — one storyboard per project (UNIQUE constraint)

Phase 0 exposes no storyboard CRUD; ``StoryboardRepository.ensure_for_project``
get-or-creates the single project-level ``storyboards`` row that panels/scenes
hang off. That path is now an idempotent ``INSERT ... ON CONFLICT (project_id)
DO NOTHING`` so concurrent writers can't race two rows into existence — which
requires a UNIQUE constraint on ``project_id`` (ON CONFLICT needs a matching
unique/exclusion constraint, not a plain index).

Existing duplicates (from the pre-constraint SELECT-then-INSERT race) are
deduped FIRST: the earliest row per project is kept, child
``storyboard_panels`` / ``storyboard_scenes`` are re-pointed to it, and the
losing ``storyboards`` rows are deleted. Then the plain
``idx_storyboards_project`` index is dropped (the unique constraint supplies the
index) and ``uq_storyboards_project`` is created.

Revision ID: 0047
Revises: 0046
Create Date: 2026-07-12
"""

from alembic import op

revision = "0047"
down_revision = "0046"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Dedup: keep the earliest storyboard per project, re-point children,
    #    delete the losers. Must run BEFORE the UNIQUE constraint is added.
    op.execute(
        """
        WITH ranked AS (
            SELECT
                id,
                project_id,
                first_value(id) OVER (
                    PARTITION BY project_id ORDER BY created_at, id
                ) AS keep_id
            FROM storyboards
        )
        UPDATE storyboard_panels p
        SET storyboard_id = r.keep_id
        FROM ranked r
        WHERE p.storyboard_id = r.id
          AND r.id <> r.keep_id
        """
    )
    op.execute(
        """
        WITH ranked AS (
            SELECT
                id,
                project_id,
                first_value(id) OVER (
                    PARTITION BY project_id ORDER BY created_at, id
                ) AS keep_id
            FROM storyboards
        )
        UPDATE storyboard_scenes s
        SET storyboard_id = r.keep_id
        FROM ranked r
        WHERE s.storyboard_id = r.id
          AND r.id <> r.keep_id
        """
    )
    op.execute(
        """
        DELETE FROM storyboards
        WHERE id IN (
            SELECT id FROM (
                SELECT
                    id,
                    row_number() OVER (
                        PARTITION BY project_id ORDER BY created_at, id
                    ) AS rn
                FROM storyboards
            ) ranked
            WHERE ranked.rn > 1
        )
        """
    )

    # 2. Swap the plain index for a UNIQUE constraint (which also indexes).
    op.drop_index("idx_storyboards_project", table_name="storyboards")
    op.create_unique_constraint("uq_storyboards_project", "storyboards", ["project_id"])


def downgrade() -> None:
    op.drop_constraint("uq_storyboards_project", "storyboards", type_="unique")
    op.create_index("idx_storyboards_project", "storyboards", ["project_id"])
