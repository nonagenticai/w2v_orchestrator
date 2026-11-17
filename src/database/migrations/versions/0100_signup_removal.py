"""``signup_approvals.removal`` — early-access offboarding (remove an APPROVED signup)

George 2026-10-20: "Build proper offboarding first". A super_admin can remove
an approved early-access workspace end to end (``src/services/signup/removal.py``):
the row moves to ``status='removed'`` and ``removal`` records the resumable
cleanup (``{state, attempts, steps, blockers, last_error, ...}``).

* ``status`` CHECK gains ``'removed'``. ``ck_signup_approvals_approved_tenant``
  is unchanged (it only constrains ``approved``).
* ``ck_signup_approvals_removal``: a removed row always carries its ``removal``.
* ``uq_signup_approvals_open_email`` is swapped: a removed row keeps the address
  RESERVED until its cleanup is finished (``done`` / ``resolved_manually``), so
  a re-signup can never race a half-removed workspace.
* a partial index on ``(removal->>'state')`` for the tick's due scan / the
  needs-attention list.
* ``ck_admin_audit_action`` is widened to EXACTLY 0097's set plus
  ``signup.remove`` (the removal's audit trail); downgrade restores 0097's CHECK
  ``NOT VALID`` (rows written under the new action are kept).

Downgrade REFUSES while any row has ``status='removed'`` (the removal record
has no regeneration path and the old CHECK would reject it). Frozen literal SQL;
no application code.

Revision ID: 0100_signup_removal
Revises: 0099_signup_llm_access
Create Date: 2026-10-21
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 19 characters.
revision = "0100_signup_removal"
down_revision = "0099_signup_llm_access"
branch_labels = None
depends_on = None

_OPEN_EMAIL_V2 = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_signup_approvals_open_email "
    "ON signup_approvals (lower(email)) WHERE status IN ('pending','approved') "
    "OR (status = 'removed' AND (removal->>'state') NOT IN ('done','resolved_manually'))"
)
_OPEN_EMAIL_V1 = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_signup_approvals_open_email "
    "ON signup_approvals (lower(email)) WHERE status IN ('pending','approved')"
)

# 0097's ``_NEW_ACTIONS`` verbatim (the downgrade target) and that set plus
# ``signup.remove``.
_OLD_ACTIONS = (
    "action IN ('tenant.update','scope.switch','scope.reset',"
    "'member.grant_role','member.revoke_role',"
    "'member.invite','member.remove','member.set_role','invite.revoke',"
    "'apikey.create','apikey.revoke','tenant.create',"
    "'connection.set','connection.clear_secret','connection.delete',"
    "'connection.test','connection.test_failed',"
    "'youtube.oauth_client.set','youtube.oauth_client.clear',"
    "'youtube.platform_channel.set','youtube.tenant_policy.set',"
    "'defaults.update','music.upload','music.update','music.delete',"
    "'offboarding.request','offboarding.export','offboarding.delete',"
    "'offboarding.reject','connection.bulk_set','tenant.partner_bind',"
    "'tenant.licence_token_set','run.cancel')"
)
_NEW_ACTIONS = (
    "action IN ('tenant.update','scope.switch','scope.reset',"
    "'member.grant_role','member.revoke_role',"
    "'member.invite','member.remove','member.set_role','invite.revoke',"
    "'apikey.create','apikey.revoke','tenant.create',"
    "'connection.set','connection.clear_secret','connection.delete',"
    "'connection.test','connection.test_failed',"
    "'youtube.oauth_client.set','youtube.oauth_client.clear',"
    "'youtube.platform_channel.set','youtube.tenant_policy.set',"
    "'defaults.update','music.upload','music.update','music.delete',"
    "'offboarding.request','offboarding.export','offboarding.delete',"
    "'offboarding.reject','connection.bulk_set','tenant.partner_bind',"
    "'tenant.licence_token_set','run.cancel','signup.remove')"
)
_DROP_AUDIT_CHECK = "ALTER TABLE admin_audit_log DROP CONSTRAINT IF EXISTS ck_admin_audit_action"

_UPGRADE: tuple[str, ...] = (
    "ALTER TABLE signup_approvals ADD COLUMN IF NOT EXISTS removal jsonb NULL",
    "ALTER TABLE signup_approvals DROP CONSTRAINT IF EXISTS ck_signup_approvals_status",
    """
    ALTER TABLE signup_approvals ADD CONSTRAINT ck_signup_approvals_status
        CHECK (status IN ('pending','approved','rejected','removed'))
    """,
    "ALTER TABLE signup_approvals DROP CONSTRAINT IF EXISTS ck_signup_approvals_removal",
    """
    ALTER TABLE signup_approvals ADD CONSTRAINT ck_signup_approvals_removal
        CHECK (status <> 'removed' OR removal IS NOT NULL)
    """,
    "DROP INDEX IF EXISTS uq_signup_approvals_open_email",
    _OPEN_EMAIL_V2,
    "CREATE INDEX IF NOT EXISTS idx_signup_approvals_removal_state "
    "ON signup_approvals ((removal->>'state')) WHERE removal IS NOT NULL",
    _DROP_AUDIT_CHECK,
    f"ALTER TABLE admin_audit_log ADD CONSTRAINT ck_admin_audit_action CHECK ({_NEW_ACTIONS})",
)

#: Refuse to drop removal records: they have no regeneration path.
_DOWNGRADE_GUARD = """
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM signup_approvals WHERE status = 'removed') THEN
        RAISE EXCEPTION
            'refusing to downgrade 0100_signup_removal: signup_approvals holds removed rows '
            '(their removal record cannot be regenerated)';
    END IF;
END $$
"""


def upgrade() -> None:
    for statement in _UPGRADE:
        op.execute(statement)


def downgrade() -> None:
    op.execute(_DOWNGRADE_GUARD)
    op.execute("DROP INDEX IF EXISTS idx_signup_approvals_removal_state")
    op.execute("DROP INDEX IF EXISTS uq_signup_approvals_open_email")
    op.execute(_OPEN_EMAIL_V1)
    op.execute("ALTER TABLE signup_approvals DROP CONSTRAINT IF EXISTS ck_signup_approvals_removal")
    op.execute("ALTER TABLE signup_approvals DROP CONSTRAINT IF EXISTS ck_signup_approvals_status")
    op.execute(
        "ALTER TABLE signup_approvals ADD CONSTRAINT ck_signup_approvals_status "
        "CHECK (status IN ('pending','approved','rejected'))"
    )
    op.execute("ALTER TABLE signup_approvals DROP COLUMN IF EXISTS removal")
    op.execute(_DROP_AUDIT_CHECK)
    op.execute(
        "ALTER TABLE admin_audit_log ADD CONSTRAINT ck_admin_audit_action "
        f"CHECK ({_OLD_ACTIONS}) NOT VALID"
    )
