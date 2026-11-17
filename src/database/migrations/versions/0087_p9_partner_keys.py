"""Partner-minted tenant API keys + the tenant.partner_bind audit action (P9)

* ``tenant_api_keys.partner_licence_id uuid NULL`` — the partner licence that minted
  the key through ``POST /api/tenants/{tenant_id}/api-keys`` (NULL = a tenant_admin
  minted it in the UI). A partner may revoke only the keys it minted; the tenant's
  key list shows the key as partner-created.
* ``uq_tenant_api_keys_partner_live_name`` — one LIVE partner key per (tenant, name):
  the partner mint is idempotent by name (a second POST is 409), and two concurrent
  mints cannot both land. Partial on ``partner_licence_id IS NOT NULL`` so existing
  tenant_admin keys (which may share names today) are untouched.
* ``ck_admin_audit_action`` is widened to EXACTLY 0086's set plus
  ``tenant.partner_bind`` (super_admin ``PUT /api/admin/tenants/{id}/partner``).

Drift-tolerant: ``ADD COLUMN IF NOT EXISTS``, ``CREATE … IF NOT EXISTS``,
``DROP CONSTRAINT IF EXISTS`` before the re-add. Downgrade drops the index + column
(the key rows stay; they become tenant_admin-looking keys) and restores 0086's audit
CHECK ``NOT VALID`` (rows written under the new action are kept, new ones refused —
the 0082/0085/0086 posture).

Revision ID: 0087_p9_partner_keys
Revises: 0086_apikey_scopes_bulk_set
Create Date: 2026-10-09
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 20 characters.
revision = "0087_p9_partner_keys"
down_revision = "0086_apikey_scopes_bulk_set"
branch_labels = None
depends_on = None

# FROZEN literals, never imported.
_PARTNER_NAME_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_tenant_api_keys_partner_live_name "
    "ON tenant_api_keys (tenant_id, name) "
    "WHERE revoked_at IS NULL AND partner_licence_id IS NOT NULL"
)

# 0086's ``_NEW_ACTIONS`` verbatim (the downgrade target) and that set plus
# ``tenant.partner_bind``.
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
    "'offboarding.reject','connection.bulk_set')"
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
    "'offboarding.reject','connection.bulk_set','tenant.partner_bind')"
)
_DROP_AUDIT_CHECK = "ALTER TABLE admin_audit_log DROP CONSTRAINT IF EXISTS ck_admin_audit_action"


def upgrade() -> None:
    op.execute("ALTER TABLE tenant_api_keys ADD COLUMN IF NOT EXISTS partner_licence_id uuid NULL")
    op.execute(_PARTNER_NAME_INDEX)
    op.execute(_DROP_AUDIT_CHECK)
    op.execute(
        f"ALTER TABLE admin_audit_log ADD CONSTRAINT ck_admin_audit_action CHECK ({_NEW_ACTIONS})"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_tenant_api_keys_partner_live_name")
    op.execute("ALTER TABLE tenant_api_keys DROP COLUMN IF EXISTS partner_licence_id")
    op.execute(_DROP_AUDIT_CHECK)
    op.execute(
        "ALTER TABLE admin_audit_log ADD CONSTRAINT ck_admin_audit_action "
        f"CHECK ({_OLD_ACTIONS}) NOT VALID"
    )
