"""source_ingest_cache — persistent cross-run per-source ingest cache

Memoizes each ``brief.sources[]`` ingest (KAG scrape/parse + republished media)
across runs so many concurrent course runs over the same source don't each
re-fetch it. Keyed by a content/identity ``fingerprint`` (PK; tenant folded in →
a second tenant is a different row). Single-flight: a ``pending`` row claims the
ingest so concurrent identical requests coalesce; the winner flips it to
``ready`` with the cached ``payload``. ``expires_at`` gives lazy TTL expiry.

RLS: ENABLE (not FORCE) + a direct-tenant ``tenant_isolation`` policy, sourced
from ``src.database.rls`` (same helpers 0027 uses) so this migration and the
``create_all`` ``after_create`` path (``register_late_rls_ddl``) can't drift.
Inert for the table owner today; effective for the ``w2v_orch_app`` runtime role
under enforce. NB the table is intentionally NOT added to ``RLS_POLICY_TABLES``
(that tuple is iterated by historical migrations 0027/0029 — see the comment in
rls.py); the ENABLE+POLICY is emitted here directly instead.

Runtime-role table privileges ride ``ALTER DEFAULT PRIVILEGES`` (gitops
phase-d-rls-stage-a.sql), like every other table — no GRANT here.

Revision ID: 0045
Revises: 0044
Create Date: 2026-07-08
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "0045"
down_revision = "0044"
branch_labels = None
depends_on = None

_TABLE = "source_ingest_cache"

# FROZEN RLS DDL — inlined output of ``src.database.rls.{enable,policy,drop_policy,
# disable}_sql(_TABLE)`` at freeze time (2026-10-05). Do NOT re-import the live
# helpers: see tests/database/test_frozen_migration_ddl.py.
_ENABLE_SQL = "ALTER TABLE source_ingest_cache ENABLE ROW LEVEL SECURITY"
_POLICY_SQL = "CREATE POLICY tenant_isolation ON source_ingest_cache USING (tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid) WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)"
_DROP_POLICY_SQL = "DROP POLICY IF EXISTS tenant_isolation ON source_ingest_cache"
_DISABLE_SQL = "ALTER TABLE source_ingest_cache DISABLE ROW LEVEL SECURITY"


def upgrade() -> None:
    op.create_table(
        _TABLE,
        sa.Column("fingerprint", sa.Text(), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("ref", sa.Text(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("claimed_by", sa.Text(), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint("state IN ('pending','ready')", name="ck_source_ingest_cache_state"),
    )
    op.create_index("ix_source_ingest_cache_tenant", _TABLE, ["tenant_id"])
    op.create_index("ix_source_ingest_cache_expires_at", _TABLE, ["expires_at"])
    # RLS (ENABLE-not-FORCE + direct-tenant policy) — same helpers as 0027.
    op.execute(_ENABLE_SQL)
    op.execute(_POLICY_SQL)


def downgrade() -> None:
    op.execute(_DROP_POLICY_SQL)
    op.execute(_DISABLE_SQL)
    op.drop_index("ix_source_ingest_cache_expires_at", table_name=_TABLE)
    op.drop_index("ix_source_ingest_cache_tenant", table_name=_TABLE)
    op.drop_table(_TABLE)
