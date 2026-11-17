"""admin_audit_log: provider-connection audit actions (P3, WP1)

Widens ``ck_admin_audit_action`` (last widened in 0075) with the P3 provider-slot
vocabulary so connection changes audit into the same durable log:

    connection.set, connection.clear_secret, connection.delete,
    connection.test, connection.test_failed

Drift-tolerant: ``DROP CONSTRAINT IF EXISTS`` then re-add. The new set is a
superset of 0075's, so every existing row validates. The downgrade re-adds 0075's
set ``NOT VALID`` — it must not delete audit rows written under the new
vocabulary, and a validating CHECK would refuse them.

Revision ID: 0078_connection_audit_actions
Revises: 0077_tenant_connections
Create Date: 2026-10-09
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 29 characters.
revision = "0078_connection_audit_actions"
down_revision = "0077_tenant_connections"
branch_labels = None
depends_on = None

_OLD_ACTIONS = (
    "action IN ('tenant.update','scope.switch','scope.reset',"
    "'member.grant_role','member.revoke_role',"
    "'member.invite','member.remove','member.set_role','invite.revoke',"
    "'apikey.create','apikey.revoke','tenant.create')"
)
_NEW_ACTIONS = (
    "action IN ('tenant.update','scope.switch','scope.reset',"
    "'member.grant_role','member.revoke_role',"
    "'member.invite','member.remove','member.set_role','invite.revoke',"
    "'apikey.create','apikey.revoke','tenant.create',"
    "'connection.set','connection.clear_secret','connection.delete',"
    "'connection.test','connection.test_failed')"
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
