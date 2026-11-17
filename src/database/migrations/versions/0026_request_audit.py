"""request_audit table — middleware-wide persistent request audit

The step between Phase C and Phase D RLS (owner-locked order
C → middleware-wide audit → Phase D).

New high-volume ``request_audit`` table: one row per *mutation*
(POST/PUT/PATCH/DELETE), *auth failure* (401/403), or *cross-tenant access* (a
super-admin acting into a foreign tenant via ``X-Active-Tenant-Id``, recorded
even on a GET). Successful same-tenant GETs are NOT recorded, bounding volume.
Written **fail-safe off the hot path** by ``AuditMiddleware`` (a failed insert
never breaks the request) — so this is a SEPARATE table from ``admin_audit_log``
(migration 0022), which is one-row-per-deliberate-action with a controlled-vocab
``action`` written atomically inside the acting route. ``event`` is the trigger
class; ``cross_tenant`` is an orthogonal boolean dimension. No backfill — there
is no historical per-request data to import.

Revision ID: 0026
Revises: 0025
Create Date: 2026-06-15
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision = "0026"
down_revision = "0025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "request_audit",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("actor_subject_id", UUID(as_uuid=True), nullable=True),
        sa.Column("claim_tenant_id", UUID(as_uuid=True), nullable=True),
        sa.Column("effective_tenant_id", UUID(as_uuid=True), nullable=True),
        sa.Column("cross_tenant", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("event", sa.Text(), nullable=False),
        sa.Column("method", sa.Text(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("resource", sa.Text(), nullable=True),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("ip", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("credential_fingerprint", sa.Text(), nullable=True),
        sa.Column("request_id", sa.Text(), nullable=True),
        sa.Column("detail", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.PrimaryKeyConstraint("id", name="pk_request_audit"),
        sa.CheckConstraint(
            "event IN ('mutation','auth_failure','read')",
            name="ck_request_audit_event",
        ),
    )
    # Composite (created_at, id) for stable newest-first pagination across rows
    # sharing a created_at instant (id is uuid7, time-ordered).
    op.create_index("idx_request_audit_created", "request_audit", ["created_at", "id"])
    op.create_index(
        "idx_request_audit_tenant_created",
        "request_audit",
        ["effective_tenant_id", "created_at"],
    )
    op.create_index("idx_request_audit_actor", "request_audit", ["actor_subject_id"])
    op.create_index("idx_request_audit_event", "request_audit", ["event"])


def downgrade() -> None:
    op.drop_index("idx_request_audit_event", table_name="request_audit")
    op.drop_index("idx_request_audit_actor", table_name="request_audit")
    op.drop_index("idx_request_audit_tenant_created", table_name="request_audit")
    op.drop_index("idx_request_audit_created", table_name="request_audit")
    op.drop_table("request_audit")
