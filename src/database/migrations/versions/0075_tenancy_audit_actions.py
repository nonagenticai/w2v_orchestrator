"""admin_audit_log: tenancy audit actions (P2, WP1)

Widens ``ck_admin_audit_action`` (0022, widened in 0025) with the P2 tenancy
vocabulary so member / invite / API-key / tenant-creation actions audit into the
same durable log:

    member.invite, member.remove, member.set_role, invite.revoke,
    apikey.create, apikey.revoke, tenant.create

Drift-tolerant: ``DROP CONSTRAINT IF EXISTS`` then re-add. The new set is a
superset of the old, so every existing row validates. The downgrade re-adds the
old set ``NOT VALID`` — it must not delete audit rows written under the new
vocabulary, and a validating CHECK would refuse them.

Revision ID: 0075_tenancy_audit_actions
Revises: 0074_tenant_api_keys
Create Date: 2026-10-08
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 26 characters.
revision = "0075_tenancy_audit_actions"
down_revision = "0074_tenant_api_keys"
branch_labels = None
depends_on = None

_OLD_ACTIONS = (
    "action IN ('tenant.update','scope.switch','scope.reset',"
    "'member.grant_role','member.revoke_role')"
)
_NEW_ACTIONS = (
    "action IN ('tenant.update','scope.switch','scope.reset',"
    "'member.grant_role','member.revoke_role',"
    "'member.invite','member.remove','member.set_role','invite.revoke',"
    "'apikey.create','apikey.revoke','tenant.create')"
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
