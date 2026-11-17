"""runs.failure_detail — the machine-readable half of a failed run's ``error`` (#1106)

``failure_class`` says WHOSE fault a failure was; it cannot say WHAT the customer
must do. A ``byo_required`` tenant whose stage needed a platform GPU modality
fails ``customer_input`` with a fixed sentence, and the frontend needs a stable
code to route the user to the Providers page instead of parsing prose:

    {"code": "byo_provider_required", "modality": "image"|"tts"|"asr",
     "message": "Configure your own <image|TTS|ASR> provider to continue"}

Nullable JSONB, no default: NULL for every existing row and for every failure
whose exception carries no structured detail. Written only by
``RunRepository.mark_failed``, in the same statement as ``status='failed'``.

Revision ID: 0091_runs_failure_detail
Revises: 0090_api_keys_rls
"""

from __future__ import annotations

from alembic import op

# ``version_num`` is varchar(32); this id is 24 characters.
revision = "0091_runs_failure_detail"
down_revision = "0090_api_keys_rls"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drift-tolerant: a re-run (or a DB that already carries the column) is a no-op.
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS failure_detail JSONB")


def downgrade() -> None:
    # Drift-tolerant both ways: a DB that never got the column downgrades cleanly.
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS failure_detail")
