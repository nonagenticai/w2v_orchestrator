"""request_audit cross-tenant partial index

Adds ``idx_request_audit_cross_tenant`` — a PARTIAL index on the small subset of
rows where ``cross_tenant`` is true — to serve the super-admin "Cross-tenant
only" filter (``GET /api/admin/request-audit?cross_tenant=true`` →
``RequestAuditRepository.list_events``). Cross-tenant rows are a tiny minority
(legitimate super-admin scope-switches), so a plain
``WHERE cross_tenant ORDER BY created_at DESC LIMIT 50`` would walk many recent
rows to fill a page; the partial index stays small and serves it directly. The
index key is ``(created_at, id)`` to match the newest-first ``created_at DESC,
id DESC`` ordering.

Single-sourced with the model (``RequestAudit.__table_args__``) so the prod
migration and the ``create_all`` test schema can't drift. Plain ``CREATE INDEX``
(NOT ``CONCURRENTLY`` — alembic runs inside a transaction; the table is modest
and the migration runs at deploy). ``request_audit`` carries no RLS, so this is
RLS-irrelevant. CI-safe: no role/FORCE DDL.

Revision ID: 0032
Revises: 0031
Create Date: 2026-06-16
"""

from alembic import op

revision = "0032"
down_revision = "0031"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_request_audit_cross_tenant "
        "ON request_audit (created_at, id) WHERE cross_tenant"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_request_audit_cross_tenant")
