"""provisioning audit table + member role-mgmt audit actions

Phase C, increment C-2 (super-admin provisioning + member role management).

* New ``provisioning_audit`` table (adopts fr's ``provisioning_audit``): a
  ``pending`` row is committed BEFORE the Keycloak/licenses side effects of a
  provision, then updated to ``success``/``partial`` — so a crash mid-flight
  always leaves an inspectable trail. ``result`` holds created ids only (never
  the one-time secrets, which are returned to the caller and never stored).
* Extend ``admin_audit_log``'s ``action`` CHECK with ``member.grant_role`` /
  ``member.revoke_role`` so tenant-admin role grants/revokes audit into the same
  durable log (migration 0022 created the original 3-action constraint).

Revision ID: 0025
Revises: 0024
Create Date: 2026-06-15
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision = "0025"
down_revision = "0024"
branch_labels = None
depends_on = None

_OLD_AUDIT_ACTIONS = "action IN ('tenant.update','scope.switch','scope.reset')"
_NEW_AUDIT_ACTIONS = (
    "action IN ('tenant.update','scope.switch','scope.reset',"
    "'member.grant_role','member.revoke_role')"
)


def upgrade() -> None:
    op.create_table(
        "provisioning_audit",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("actor_subject_id", UUID(as_uuid=True), nullable=True),
        sa.Column("action", sa.Text(), nullable=False, server_default="provision"),
        sa.Column("target_email", sa.Text(), nullable=False),
        sa.Column("target_subject_id", UUID(as_uuid=True), nullable=True),
        sa.Column("tenant_id", UUID(as_uuid=True), nullable=True),
        sa.Column("payload", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("result", JSONB(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_provisioning_audit"),
        sa.CheckConstraint(
            "status IN ('pending','success','partial')",
            name="ck_provisioning_audit_status",
        ),
    )
    op.create_index("idx_provisioning_audit_created", "provisioning_audit", ["created_at"])

    # Widen the admin_audit_log action vocabulary for member role-mgmt.
    op.drop_constraint("ck_admin_audit_action", "admin_audit_log", type_="check")
    op.create_check_constraint("ck_admin_audit_action", "admin_audit_log", _NEW_AUDIT_ACTIONS)


def downgrade() -> None:
    op.drop_constraint("ck_admin_audit_action", "admin_audit_log", type_="check")
    op.create_check_constraint("ck_admin_audit_action", "admin_audit_log", _OLD_AUDIT_ACTIONS)
    op.drop_index("idx_provisioning_audit_created", table_name="provisioning_audit")
    op.drop_table("provisioning_audit")
