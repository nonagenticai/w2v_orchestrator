"""tenant_is_occupied() — partner tenant creation's cross-tenant occupancy check

``POST /api/tenants`` lets an allow-listed partner choose the ``tenant_id``. It
may claim an id (or adopt a ``created_via='backfill'`` registry row) only when
that tenant is NOT in use — otherwise a partner could take over a licensed
tenant that never got a registry row and invite its own ``tenant_admin`` into it.
"In use" = any ``projects`` / ``tenant_members`` / ``tenant_invites`` /
``tenant_api_keys`` row (``source_ingest_cache`` is a cache and the audit logs
are records of attempts — neither counts).

The partner's request is bound to the PARTNER's tenant (or none), so under
``TENANT_RLS_MODE=enforce`` a plain ``SELECT … FROM projects WHERE tenant_id = X``
is scoped to the wrong tenant and answers "no projects" — the takeover itself.
Same bootstrap wall as ``run_tenant``, same answer: a read-only ``SECURITY
DEFINER`` boolean owned by the migration user. It returns one bit, never a row.
EXECUTE revoked from PUBLIC and granted to ``w2v_orch_app`` under a role guard
(that role exists only in prod; an unconditional GRANT errors on a CI DB).

``search_path`` is pinned with ``pg_temp`` LAST — ``pg_catalog, <schema>,
pg_temp`` where ``<schema>`` is ``current_schema()`` at upgrade time (``public``
in prod) — so no caller can shadow a table through its temp schema.

Purely additive: one new function, no table touched. An OLDER image boots fine
against the migrated DB — it never calls the function.

⚠ The DDL below is a FROZEN literal copy of
``src.database.rls.tenant_occupancy_functions_statements()`` /
``drop_tenant_occupancy_functions_statements()`` (this repo's freeze
convention — see 0064's banner). A change to the resolver is a NEW forward
migration, not an edit here.

Revision ID: 0076_tenant_is_occupied_fn
Revises: 0075_tenancy_audit_actions
Create Date: 2026-10-09
"""

from alembic import op

# ``version_num`` is varchar(32); this id is 26 characters.
revision = "0076_tenant_is_occupied_fn"
down_revision = "0075_tenancy_audit_actions"
branch_labels = None
depends_on = None

# Frozen as of this revision. Do NOT re-import the live generator.
_TENANT_IS_OCCUPIED_FN = (
    "DO $do$ BEGIN EXECUTE "
    "'CREATE OR REPLACE FUNCTION tenant_is_occupied(p_tenant uuid) RETURNS boolean "
    "LANGUAGE sql STABLE SECURITY DEFINER "
    "SET search_path = pg_catalog, ' || quote_ident(current_schema()) || ', pg_temp "
    "AS $fn$ "
    "SELECT EXISTS (SELECT 1 FROM projects WHERE tenant_id = p_tenant) "
    "OR EXISTS (SELECT 1 FROM tenant_members WHERE tenant_id = p_tenant) "
    "OR EXISTS (SELECT 1 FROM tenant_invites WHERE tenant_id = p_tenant) "
    "OR EXISTS (SELECT 1 FROM tenant_api_keys WHERE tenant_id = p_tenant)"
    " $fn$'; END $do$"
)
_GRANT_TENANT_IS_OCCUPIED = (
    "DO $$ BEGIN "
    "IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_orch_app') THEN "
    "GRANT EXECUTE ON FUNCTION tenant_is_occupied(uuid) TO w2v_orch_app; "
    "END IF; END $$"
)
_UPGRADE_STATEMENTS: tuple[str, ...] = (
    _TENANT_IS_OCCUPIED_FN,
    "REVOKE EXECUTE ON FUNCTION tenant_is_occupied(uuid) FROM PUBLIC",
    _GRANT_TENANT_IS_OCCUPIED,
)
_DOWNGRADE_STATEMENTS: tuple[str, ...] = ("DROP FUNCTION IF EXISTS tenant_is_occupied(uuid)",)


def upgrade() -> None:
    for stmt in _UPGRADE_STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in _DOWNGRADE_STATEMENTS:
        op.execute(stmt)
