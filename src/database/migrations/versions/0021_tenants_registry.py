"""canonical tenants registry (name + status annotation layer)

w2v is downstream of licenses.test / Keycloak for ``tenant_id`` — it never
mints the id, the claim arrives on the JWT/license token. So this table is a
**naming + status annotation layer keyed on the claim-derived tenant_id**, not a
create-from-scratch tenant authority (that distinguishes it from commonpractice,
which IS the provisioning authority and generates its own ``tenants.id``).

* ``id`` is the tenant_id itself (the claim), NOT generated — same Postgres
  ``uuid`` type as ``projects.tenant_id`` (recreated as UUID in 0015) so the two
  join cleanly.
* ``name`` is an optional human label (super-admin set; UUID shown until named).
* ``status`` ∈ {active, suspended}; suspended blocks a super-admin scope-switch
  into the tenant (no license-revocation machinery this increment).

Backfill from every distinct ``projects.tenant_id`` so no live tenant is ever
invisible in the admin list or unswitchable — this is what makes the fail-closed
scope-switch existence check safe (every project-bearing tenant already has a
row). Pure ``INSERT … SELECT DISTINCT`` (no app logic) → lives in the migration.

Revision ID: 0021
Revises: 0020
Create Date: 2026-06-14
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="active"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("status IN ('active','suspended')", name="ck_tenants_status"),
    )
    # Backfill: every tenant that has ever created a project gets a row, so the
    # scope-switch existence check can be fail-closed without locking anyone out.
    op.execute(
        "INSERT INTO tenants (id) "
        "SELECT DISTINCT tenant_id FROM projects "
        "ON CONFLICT (id) DO NOTHING"
    )


def downgrade() -> None:
    op.drop_table("tenants")
