"""admin_audit_log: P6 audit actions (defaults, music, offboarding)

Widens ``ck_admin_audit_action`` (last widened in 0080, the YouTube config actions) with the WHOLE P6 vocabulary
in one step, so the later P6 commits (music library, offboarding) need no further
CHECK migration:

    defaults.update,
    music.upload, music.delete,
    offboarding.request, offboarding.export, offboarding.delete

Drift-tolerant: ``DROP CONSTRAINT IF EXISTS`` then re-add. The new set is a superset of
0080's, so every existing row validates. The downgrade re-adds 0080's set ``NOT
VALID``: it must not delete audit rows written under the new vocabulary, and a
validating CHECK would refuse them.

Revision ID: 0082_p6_audit_actions
Revises: 0081_tenant_defaults
Create Date: 2026-10-09
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 21 characters.
revision = "0082_p6_audit_actions"
down_revision = "0081_tenant_defaults"
branch_labels = None
depends_on = None

_OLD_ACTIONS = (
    "action IN ('tenant.update','scope.switch','scope.reset',"
    "'member.grant_role','member.revoke_role',"
    "'member.invite','member.remove','member.set_role','invite.revoke',"
    "'apikey.create','apikey.revoke','tenant.create',"
    "'connection.set','connection.clear_secret','connection.delete',"
    "'connection.test','connection.test_failed',"
    "'youtube.oauth_client.set','youtube.oauth_client.clear',"
    "'youtube.platform_channel.set','youtube.tenant_policy.set')"
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
    "'defaults.update','music.upload','music.delete',"
    "'offboarding.request','offboarding.export','offboarding.delete')"
)


def upgrade() -> None:
    op.execute("ALTER TABLE admin_audit_log DROP CONSTRAINT IF EXISTS ck_admin_audit_action")
    op.execute(
        f"ALTER TABLE admin_audit_log ADD CONSTRAINT ck_admin_audit_action CHECK ({_NEW_ACTIONS})"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE admin_audit_log DROP CONSTRAINT IF EXISTS ck_admin_audit_action")
    op.execute(
        "ALTER TABLE admin_audit_log ADD CONSTRAINT ck_admin_audit_action "
        f"CHECK ({_OLD_ACTIONS}) NOT VALID"
    )
