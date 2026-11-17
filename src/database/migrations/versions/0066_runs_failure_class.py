"""runs.failure_class — persist WHOSE fault a failed run was

``runs`` records ``status`` plus a free-text ``error`` and nothing else, so the
question every failure investigation actually asks — "did we break this, or did
the customer?" — has no column to read. Measured 2026-09-01: bucketing the 72
failed runs inside retention needed an ad-hoc regex over ``runs.error`` and left
36 of the 72 unclassifiable. This column is the answer, written by
``RunRepository.mark_failed`` in the same statement as ``status='failed'``.

Vocabulary and the reasoning behind each member live in
``src/services/failure_class.py``. It is a DIFFERENT AXIS from
``services/stages/upstream.classify`` (permanent vs transient = retryability):
our own 500 and a customer's 400 are both permanent, and neither of those facts
says whose fault it was.

Safe on a live table:

* **Nullable, no server default, no backfill.** ``ADD COLUMN ... NULL`` with no
  default is a catalogue-only change in postgres 11+ — no table rewrite, no long
  lock, regardless of how many rows ``runs`` holds.
* **NULL is load-bearing, not laziness.** Every pre-existing row reads NULL =
  "never classified", which is a genuinely different statement from
  ``'unknown'`` = "classified, and the evidence did not attribute". Backfilling
  those 72 rows to ``'unknown'`` would destroy that distinction, and
  backfilling them to a *guess* is the exact failure mode the taxonomy exists to
  prevent.
* **Older image, migrated DB: fine.** A pod running the previous image never
  writes the column, so every row it fails reads NULL.

Both CHECKs validate instantly: no existing row has a non-NULL
``failure_class``, so neither predicate can be violated at ADD time.

Revision ID: 0066_runs_failure_class
Revises: 0065_preview_lease
Create Date: 2026-09-01
"""

import sqlalchemy as sa
from alembic import op

# ``version_num`` is varchar(32) — a longer id applies the DDL and then fails the
# stamp, leaving the DB migrated but unmarked. This one is 23 characters.
revision = "0066_runs_failure_class"
down_revision = "0065_preview_lease"
branch_labels = None
depends_on = None

_CK_VOCAB = "ck_runs_failure_class"
_CK_ONLY_ON_FAILED = "ck_runs_failure_class_only_on_failed"

# ⚠️ Byte-for-byte identical to the predicates in ``Run.__table_args__``
# (``src/database/models.py``). The models-vs-migrations CHECK parity guard in
# ``tests/contracts/`` compares the postgres-normalized predicates of every
# ``ck_*``, so a retyped variant fails there rather than in prod. Copy, don't
# retype — and widen BOTH sides plus the ``FailureClass`` Literal in the same PR.
_VOCAB = (
    "failure_class IS NULL OR failure_class IN ('our_defect','customer_input','external','unknown')"
)
_ONLY_ON_FAILED = "failure_class IS NULL OR status = 'failed'"


def upgrade() -> None:
    op.add_column("runs", sa.Column("failure_class", sa.Text(), nullable=True))
    op.create_check_constraint(_CK_VOCAB, "runs", _VOCAB)
    op.create_check_constraint(_CK_ONLY_ON_FAILED, "runs", _ONLY_ON_FAILED)


def downgrade() -> None:
    # Dropping the column drops its constraints with it, but they are dropped
    # explicitly first so the reverse order mirrors ``upgrade`` exactly and a
    # partially-applied upgrade still reverses cleanly.
    op.drop_constraint(_CK_ONLY_ON_FAILED, "runs", type_="check")
    op.drop_constraint(_CK_VOCAB, "runs", type_="check")
    op.drop_column("runs", "failure_class")
