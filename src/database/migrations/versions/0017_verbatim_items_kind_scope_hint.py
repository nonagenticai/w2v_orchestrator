"""verbatim_items: restore kind / hint / scope / scope_target columns.

The Studio v3 reset (0015) collapsed verbatim_items to a 6-column shape
(id, project_id, text, position, created_at, updated_at) and the SPA
contract was never updated; the result is that the verbatim CRUD UI has
been unusable since v3 (the list endpoint returns a bare list, the SPA
expects ``{items, total}`` and crashes; the SPA split by ``kind`` finds
no column to slot new items into). The frontend's ``VerbatimTwoTrack``
component splits items by ``kind`` ('slide' | 'tts'), so the column is
required for the UI to work at all.

This migration adds back the four columns the SPA actually needs. The
lifecycle fields from the legacy 0009 shape (status, placed_in_stage,
placed_in_element_id, retry_count) are surfaced as response-side defaults
in ``VerbatimItemResponse`` rather than stored — no consumer writes them
in the v3 codebase, so persisting them would be cargo.

Existing rows (there are essentially none — the create flow has been
broken since v3) are backfilled to kind='tts' / scope='pipeline'.

Revision ID: 0017
Revises: 0016
Create Date: 2026-05-31
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "verbatim_items",
        sa.Column("kind", sa.Text(), nullable=False, server_default="tts"),
    )
    op.add_column("verbatim_items", sa.Column("hint", sa.Text(), nullable=True))
    op.add_column(
        "verbatim_items",
        sa.Column("scope", sa.Text(), nullable=False, server_default="pipeline"),
    )
    op.add_column("verbatim_items", sa.Column("scope_target", sa.Text(), nullable=True))
    op.create_check_constraint(
        "ck_verbatim_items_kind",
        "verbatim_items",
        "kind IN ('slide', 'tts')",
    )
    op.create_check_constraint(
        "ck_verbatim_items_scope",
        "verbatim_items",
        "scope IN ('pipeline', 'chapter', 'scene', 'shot', 'slide')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_verbatim_items_scope", "verbatim_items", type_="check")
    op.drop_constraint("ck_verbatim_items_kind", "verbatim_items", type_="check")
    op.drop_column("verbatim_items", "scope_target")
    op.drop_column("verbatim_items", "scope")
    op.drop_column("verbatim_items", "hint")
    op.drop_column("verbatim_items", "kind")
