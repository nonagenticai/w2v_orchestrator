"""tenant members roster (who has signed in, per tenant)

Phase C, increment C-1. Adopts fr's ``user_roles`` concept (fr keeps a DB-backed
roster keyed on the Keycloak subject; see ``fr/backend/src/db/schema.ts``) so the
tenant Members screen reads a real roster instead of the misleading "observed
authors from projects.owner_subject_id" hack flagged in Phase B increment 6.

* One row per ``(subject_id, tenant_id)`` — w2v is single-tenant-per-identity, but
  the composite PK keeps a super-admin acting INTO multiple tenants correct.
* ``roles`` (JSONB) snapshots the realm roles observed at last login — the honest
  "what this member can do". Role MANAGEMENT (grant/revoke) is a later increment
  (needs the Keycloak-Admin client to write realm role-mappings).
* No FK to ``tenants`` — ``tenant_id`` is claim-derived (w2v is downstream of
  licenses.test / Keycloak), same stance as the tenants registry (0021).
* No backfill — the roster grows from real ``/api/auth/me`` logins; backfilling
  from ``projects.owner_subject_id`` would reintroduce the observed-authors lie.

Revision ID: 0024
Revises: 0023
Create Date: 2026-06-15
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tenant_members",
        sa.Column("subject_id", UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.Text(), nullable=True),
        sa.Column("display_name", sa.Text(), nullable=True),
        sa.Column(
            "roles",
            JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("subject_id", "tenant_id", name="pk_tenant_members"),
    )
    op.create_index("idx_tenant_members_tenant", "tenant_members", ["tenant_id"])


def downgrade() -> None:
    op.drop_table("tenant_members")
