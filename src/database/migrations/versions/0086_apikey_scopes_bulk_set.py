"""tenant_api_keys.scopes + the connection.bulk_set audit action (P8)

* ``tenant_api_keys.scopes text[] NOT NULL DEFAULT '{}'`` — grants a key holds beyond
  the ``member`` role every key acts as. The vocabulary is closed by
  ``ck_tenant_api_keys_scopes`` (today only ``connections:write``: the bulk
  ``PUT /api/tenant/connections`` a machine caller such as merqi uses). Every existing
  key gets ``'{}'`` — no key silently gains a grant.
* ``ck_admin_audit_action`` is widened to EXACTLY 0085's set plus
  ``connection.bulk_set``.

Drift-tolerant: ``ADD COLUMN IF NOT EXISTS``, ``DROP CONSTRAINT IF EXISTS`` before
every re-add. Downgrade drops the column (scopes are a grant; losing them fails
closed) and restores 0085's audit CHECK ``NOT VALID`` (rows written under the new
action are kept, new ones refused — the 0082/0085 posture).

Revision ID: 0086_apikey_scopes_bulk_set
Revises: 0085_tenant_offboarding
Create Date: 2026-10-09
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 27 characters.
revision = "0086_apikey_scopes_bulk_set"
down_revision = "0085_tenant_offboarding"
branch_labels = None
depends_on = None

# FROZEN literals, never imported.
_SCOPES_CHECK = "scopes <@ ARRAY['connections:write']::text[]"

# 0085's ``_NEW_ACTIONS`` verbatim (the downgrade target) and that set plus
# ``connection.bulk_set``.
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
    "'offboarding.reject')"
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
    "'offboarding.reject','connection.bulk_set')"
)
_DROP_AUDIT_CHECK = "ALTER TABLE admin_audit_log DROP CONSTRAINT IF EXISTS ck_admin_audit_action"
_DROP_SCOPES_CHECK = (
    "ALTER TABLE tenant_api_keys DROP CONSTRAINT IF EXISTS ck_tenant_api_keys_scopes"
)


def upgrade() -> None:
    op.execute(
        "ALTER TABLE tenant_api_keys "
        "ADD COLUMN IF NOT EXISTS scopes text[] NOT NULL DEFAULT '{}'::text[]"
    )
    op.execute(_DROP_SCOPES_CHECK)
    op.execute(
        f"ALTER TABLE tenant_api_keys ADD CONSTRAINT ck_tenant_api_keys_scopes "
        f"CHECK ({_SCOPES_CHECK})"
    )
    op.execute(_DROP_AUDIT_CHECK)
    op.execute(
        f"ALTER TABLE admin_audit_log ADD CONSTRAINT ck_admin_audit_action CHECK ({_NEW_ACTIONS})"
    )


def downgrade() -> None:
    op.execute(_DROP_SCOPES_CHECK)
    op.execute("ALTER TABLE tenant_api_keys DROP COLUMN IF EXISTS scopes")
    op.execute(_DROP_AUDIT_CHECK)
    op.execute(
        "ALTER TABLE admin_audit_log ADD CONSTRAINT ck_admin_audit_action "
        f"CHECK ({_OLD_ACTIONS}) NOT VALID"
    )
