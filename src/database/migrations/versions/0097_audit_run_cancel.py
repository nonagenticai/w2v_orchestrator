"""Audit action ``run.cancel`` (super_admin cross-tenant run cancel)

``POST /api/admin/runs/{run_id}/cancel`` lets a super_admin stop any tenant's run
(owner decision 2026-10-20) and records it in ``admin_audit_log``.
``ck_admin_audit_action`` is widened to EXACTLY 0089's set plus ``run.cancel``.

Drift-tolerant: the CHECK is dropped-if-present and re-added. Downgrade restores
0089's CHECK ``NOT VALID`` (rows written under the new action are kept). Frozen
literal SQL; no application code.

Revision ID: 0097_audit_run_cancel
Revises: 0096_drop_conn_runtime_sync
Create Date: 2026-10-20
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 21 characters.
revision = "0097_audit_run_cancel"
down_revision = "0096_drop_conn_runtime_sync"
branch_labels = None
depends_on = None

# 0089's ``_NEW_ACTIONS`` verbatim (the downgrade target) and that set plus
# ``run.cancel``.
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
    "'tenant.licence_token_set')"
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
    "'tenant.licence_token_set','run.cancel')"
)
_DROP_AUDIT_CHECK = "ALTER TABLE admin_audit_log DROP CONSTRAINT IF EXISTS ck_admin_audit_action"


def upgrade() -> None:
    op.execute(_DROP_AUDIT_CHECK)
    op.execute(
        f"ALTER TABLE admin_audit_log ADD CONSTRAINT ck_admin_audit_action CHECK ({_NEW_ACTIONS})"
    )


def downgrade() -> None:
    op.execute(_DROP_AUDIT_CHECK)
    op.execute(
        "ALTER TABLE admin_audit_log ADD CONSTRAINT ck_admin_audit_action "
        f"CHECK ({_OLD_ACTIONS}) NOT VALID"
    )
