"""admin_audit_log: YouTube config audit actions

Widens ``ck_admin_audit_action`` (last widened in 0078) so the per-tenant YouTube
settings proxy and the operator platform-channel routes audit into the same
durable log as every other admin / tenant-admin mutation:

    youtube.oauth_client.set, youtube.oauth_client.clear,
    youtube.platform_channel.set, youtube.tenant_policy.set

Drift-tolerant: ``DROP CONSTRAINT IF EXISTS`` then re-add. The new set is a
superset of 0078's, so every existing row validates. The downgrade re-adds 0078's
set ``NOT VALID`` — it must not delete audit rows written under the new
vocabulary, and a validating CHECK would refuse them.

Revision ID: 0080_youtube_audit_actions
Revises: 0079_provider_secret_gc_fn
Create Date: 2026-10-09
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 26 characters.
revision = "0080_youtube_audit_actions"
down_revision = "0079_provider_secret_gc_fn"
branch_labels = None
depends_on = None

_OLD_ACTIONS = (
    "action IN ('tenant.update','scope.switch','scope.reset',"
    "'member.grant_role','member.revoke_role',"
    "'member.invite','member.remove','member.set_role','invite.revoke',"
    "'apikey.create','apikey.revoke','tenant.create',"
    "'connection.set','connection.clear_secret','connection.delete',"
    "'connection.test','connection.test_failed')"
)
_NEW_ACTIONS = (
    "action IN ('tenant.update','scope.switch','scope.reset',"
    "'member.grant_role','member.revoke_role',"
    "'member.invite','member.remove','member.set_role','invite.revoke',"
    "'apikey.create','apikey.revoke','tenant.create',"
    "'connection.set','connection.clear_secret','connection.delete',"
    "'connection.test','connection.test_failed',"
    "'youtube.oauth_client.set','youtube.oauth_client.clear',"
    "'youtube.platform_channel.set','youtube.tenant_policy.set')"
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
