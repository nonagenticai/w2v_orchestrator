"""Tenant licence token (sealed) + licence-change lease + its audit action

Licence verification by INTROSPECTION (the commonpractice model): the tenant's own
w2v licence token is stored, sealed, and entitlement is ``POST /license/introspect``
of it — no ``/internal/*`` licences call, which an off-cluster ingress refuses.

``tenants`` gains six nullable columns:

    licence_token_ciphertext  bytea        ``v1|kid|nonce|ct+tag`` AES-256-GCM, AAD
                                           ``("tenant-licence", tenant_id)``
    licence_token_kid         text         the keyring entry that sealed it
    licence_token_set_at      timestamptz  when it was stored
    licence_token_expires_at  timestamptz  the TOKEN's own expiry (introspection
                                           ``token_expires_at``; NULL = perpetual)
    licence_op_id             uuid         per-tenant licence-change lease holder
    licence_op_until          timestamptz  lease expiry

``ck_tenants_licence_token_complete``: the three token columns are all set or all
NULL. ``ck_admin_audit_action`` is widened to EXACTLY 0087's set plus
``tenant.licence_token_set`` (``PUT /api/tenants/{id}/licence-token``).

Drift-tolerant: ``ADD COLUMN IF NOT EXISTS``; each CHECK is dropped-if-present and
re-added; a hand-written half token triple (impossible through the app) is cleared
to NULL before the CHECK lands. Downgrade drops the columns (the sealed tokens are
lost — tenants fall back to the ``/licenses/active`` lookup) and restores 0087's
audit CHECK ``NOT VALID`` (rows written under the new action are kept). Frozen
literal SQL; no application code.

Revision ID: 0089_tenant_licence_token
Revises: 0088_run_usage_connection
Create Date: 2026-10-13
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 25 characters.
revision = "0089_tenant_licence_token"
down_revision = "0088_run_usage_connection"
branch_labels = None
depends_on = None

_COLUMNS = (
    "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS licence_token_ciphertext bytea",
    "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS licence_token_kid text",
    "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS licence_token_set_at timestamptz",
    "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS licence_token_expires_at timestamptz",
    "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS licence_op_id uuid",
    "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS licence_op_until timestamptz",
)

_TRIPLE_CHECK = (
    "((licence_token_ciphertext IS NULL) = (licence_token_kid IS NULL)) "
    "AND ((licence_token_ciphertext IS NULL) = (licence_token_set_at IS NULL))"
)

_CONSTRAINTS = (
    "ALTER TABLE tenants DROP CONSTRAINT IF EXISTS ck_tenants_licence_token_complete",
    # Drift: a half triple would fail the CHECK — clear it (the tenant falls back).
    "UPDATE tenants SET licence_token_ciphertext = NULL, licence_token_kid = NULL, "
    "licence_token_set_at = NULL, licence_token_expires_at = NULL "
    "WHERE NOT (((licence_token_ciphertext IS NULL) = (licence_token_kid IS NULL)) "
    "AND ((licence_token_ciphertext IS NULL) = (licence_token_set_at IS NULL)))",
    "ALTER TABLE tenants ADD CONSTRAINT ck_tenants_licence_token_complete "
    "CHECK (((licence_token_ciphertext IS NULL) = (licence_token_kid IS NULL)) "
    "AND ((licence_token_ciphertext IS NULL) = (licence_token_set_at IS NULL)))",
)

# 0087's ``_NEW_ACTIONS`` verbatim (the downgrade target) and that set plus
# ``tenant.licence_token_set``.
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
    "'offboarding.reject','connection.bulk_set','tenant.partner_bind')"
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
    "'tenant.licence_token_set')"
)
_DROP_AUDIT_CHECK = "ALTER TABLE admin_audit_log DROP CONSTRAINT IF EXISTS ck_admin_audit_action"

_DOWNGRADE = (
    "ALTER TABLE tenants DROP CONSTRAINT IF EXISTS ck_tenants_licence_token_complete",
    "ALTER TABLE tenants DROP COLUMN IF EXISTS licence_op_until",
    "ALTER TABLE tenants DROP COLUMN IF EXISTS licence_op_id",
    "ALTER TABLE tenants DROP COLUMN IF EXISTS licence_token_expires_at",
    "ALTER TABLE tenants DROP COLUMN IF EXISTS licence_token_set_at",
    "ALTER TABLE tenants DROP COLUMN IF EXISTS licence_token_kid",
    "ALTER TABLE tenants DROP COLUMN IF EXISTS licence_token_ciphertext",
)


def upgrade() -> None:
    for statement in (*_COLUMNS, *_CONSTRAINTS):
        op.execute(statement)
    op.execute(_DROP_AUDIT_CHECK)
    op.execute(
        f"ALTER TABLE admin_audit_log ADD CONSTRAINT ck_admin_audit_action CHECK ({_NEW_ACTIONS})"
    )


def downgrade() -> None:
    for statement in _DOWNGRADE:
        op.execute(statement)
    op.execute(_DROP_AUDIT_CHECK)
    op.execute(
        "ALTER TABLE admin_audit_log ADD CONSTRAINT ck_admin_audit_action "
        f"CHECK ({_OLD_ACTIONS}) NOT VALID"
    )
