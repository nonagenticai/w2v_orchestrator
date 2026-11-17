"""``signup_approvals.llm_access`` accepts ``byo_required`` (uniform provider access)

Owner decision 2026-10-22: every onboarding path (admin provision, early-access
approve, invite) may give a new tenant ANY of the three licences ``byo_mode``
postures, CommonPractice-style — ``byo_required`` included. The row records the
approver's choice, so its CHECK widens from ``('local_allowed','platform_only')``
to all three.

Downgrade restores 0099's narrower CHECK ``NOT VALID`` (rows already recorded as
``byo_required`` are kept; the posture itself lives in licences). Frozen literal
SQL; no application code.

Revision ID: 0101_signup_llm_byo_required
Revises: 0100_signup_removal
Create Date: 2026-10-22
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 28 characters.
revision = "0101_signup_llm_byo_required"
down_revision = "0100_signup_removal"
branch_labels = None
depends_on = None

_DROP = "ALTER TABLE signup_approvals DROP CONSTRAINT IF EXISTS ck_signup_approvals_llm_access"


def upgrade() -> None:
    op.execute(_DROP)
    op.execute(
        """
        ALTER TABLE signup_approvals ADD CONSTRAINT ck_signup_approvals_llm_access
            CHECK (llm_access IS NULL
                   OR llm_access IN ('local_allowed','byo_required','platform_only'))
        """
    )


def downgrade() -> None:
    op.execute(_DROP)
    op.execute(
        """
        ALTER TABLE signup_approvals ADD CONSTRAINT ck_signup_approvals_llm_access
            CHECK (llm_access IS NULL OR llm_access IN ('local_allowed','platform_only'))
            NOT VALID
        """
    )
