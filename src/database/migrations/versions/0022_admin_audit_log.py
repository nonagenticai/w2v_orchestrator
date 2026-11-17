"""persistent admin audit log (deliberate admin actions)

Phase B, 4th increment. Until now deliberate admin actions left no durable
trail: the scope-switch was a log line only (``w2v_orchestrator.audit``) and the
tenant rename/suspend PATCH (0021/#247) recorded nothing. This table persists
those actions so a super-admin can review them.

* One row per *deliberate* admin action — NOT one per request. A super-admin
  acting as another tenant generates many requests but only one ``scope.switch``
  row (recorded on the UI switch via POST /api/admin/scope-switch). The
  per-request inline log line stays for raw-API switches.
* ``action`` is a small controlled vocabulary (CHECK); ``detail`` (JSONB) holds
  action-specific payload (before/after for a tenant update, from-tenant for a
  switch).
* No backfill — there is no historical structured data to import.

Revision ID: 0022
Revises: 0021
Create Date: 2026-06-14
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "admin_audit_log",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("actor_subject_id", UUID(as_uuid=True), nullable=True),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("target_tenant_id", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "detail",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.CheckConstraint(
            "action IN ('tenant.update','scope.switch','scope.reset')",
            name="ck_admin_audit_action",
        ),
    )
    op.create_index("idx_admin_audit_created", "admin_audit_log", [sa.text("created_at DESC")])
    op.create_index("idx_admin_audit_tenant", "admin_audit_log", ["target_tenant_id"])
    op.create_index("idx_admin_audit_actor", "admin_audit_log", ["actor_subject_id"])


def downgrade() -> None:
    op.drop_table("admin_audit_log")
