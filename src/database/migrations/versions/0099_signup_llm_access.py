"""``signup_approvals.llm_access`` — the approver's per-person LLM access choice

CommonPractice lets the approver choose the plan on approve and on the invite
form; w2v's equivalent is the new tenant's LLM access posture. The super_admin
approve / invite bodies carry an optional ``llmAccess`` (``local_allowed`` |
``platform_only``; ``byo_required`` is never offered on an onboarding path);
the Telegram approve takes the deployment default. The DECISION records the
chosen value on the row, and the ``provision`` step passes it to
``license_new_tenant(llm_access=…)``.

NULL only on a pending / rejected row (nothing chosen yet). Rows already
approved before this column existed are backfilled from what their
``provision`` step recorded (``steps.provision.detail.llm_access.mode``), so the
detail panel shows the posture they were actually given.

Downgrade drops the column (the posture itself lives in licences; the row only
remembers the choice). Frozen literal SQL; no application code.

Revision ID: 0099_signup_llm_access
Revises: 0098_signup_approvals
Create Date: 2026-10-20
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 22 characters.
revision = "0099_signup_llm_access"
down_revision = "0098_signup_approvals"
branch_labels = None
depends_on = None

_UPGRADE: tuple[str, ...] = (
    "ALTER TABLE signup_approvals ADD COLUMN IF NOT EXISTS llm_access text NULL",
    """
    UPDATE signup_approvals
       SET llm_access = steps->'provision'->'detail'->'llm_access'->>'mode'
     WHERE llm_access IS NULL
       AND steps->'provision'->'detail'->'llm_access'->>'mode'
           IN ('local_allowed','platform_only')
    """,
    # Idempotent (a create_all-built schema already carries the CHECK).
    "ALTER TABLE signup_approvals DROP CONSTRAINT IF EXISTS ck_signup_approvals_llm_access",
    """
    ALTER TABLE signup_approvals ADD CONSTRAINT ck_signup_approvals_llm_access
        CHECK (llm_access IS NULL OR llm_access IN ('local_allowed','platform_only'))
    """,
)


def upgrade() -> None:
    for statement in _UPGRADE:
        op.execute(statement)


def downgrade() -> None:
    op.execute(
        "ALTER TABLE signup_approvals DROP CONSTRAINT IF EXISTS ck_signup_approvals_llm_access"
    )
    op.execute("ALTER TABLE signup_approvals DROP COLUMN IF EXISTS llm_access")
