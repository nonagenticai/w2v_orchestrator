"""runs.seeded_pause_after_stages — provenance of blueprint-seeded checkpoints

``runs.pause_after_stages`` mixes two kinds of entry that are byte-identical:
gates the BLUEPRINT seeded at creation (``default_pause_after``) and checkpoints
a USER chose (explicit create list, ``PATCH /runs/{id}/checkpoints``, the
CheckpointPicker). ``resume_run`` needs to tell them apart: a seeded gate that
the blueprint has since REMOVED must stop re-pausing (BACKLOG #6), while a user
pick must always be honoured — a missed user pause lets paid stages run that the
user wanted to review.

This column records the blueprint-seeded subset at creation. Resume drops a
remaining pause only if it is in this column AND no longer a blueprint gate.

Safe on a live table:

* **Nullable, no server default, no backfill.** Catalogue-only ``ADD COLUMN`` on
  postgres 11+ — no rewrite, no long lock.
* **NULL is load-bearing.** Pre-existing rows read NULL = "provenance unknown",
  and resume keeps their list unchanged (the old behaviour; their remedy stays
  ``PATCH /runs/{id}/checkpoints []``). Backfilling a guess would let resume
  drop a checkpoint a user may have chosen.
* **Older image, migrated DB: fine.** The old image never writes it -> NULL.

Revision ID: 0070_runs_seeded_pause_stages
Revises: 0069_redact_brief_login_secrets
Create Date: 2026-10-05
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY

# ``version_num`` is varchar(32); this id is 29 characters.
revision = "0070_runs_seeded_pause_stages"
down_revision = "0069_redact_brief_login_secrets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Same type as ``runs.pause_after_stages`` (text[]), but nullable with no
    # default: NULL means "provenance unknown", distinct from '{}' ("nothing was
    # seeded").
    op.add_column(
        "runs",
        sa.Column("seeded_pause_after_stages", ARRAY(sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("runs", "seeded_pause_after_stages")
