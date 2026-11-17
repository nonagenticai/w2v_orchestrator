"""tenant_api_keys: row-level security + pre-tenant key resolvers

``tenant_api_keys`` shipped in 0074 WITHOUT RLS because a presented
``Bearer w2v_sk_…`` is resolved before any tenant is known. Every sibling tenant
table (``tenant_connections`` 0077, ``tenant_defaults`` 0081,
``tenant_sound_assets`` 0084, …) is isolated by the ``tenant_isolation`` policy on
``app.current_tenant``; this revision brings the key table in line.

* RLS: ENABLE (not FORCE — exactly like every sibling: the owner is the
  non-superuser migration role and FORCE would break the SECURITY DEFINER
  resolvers; the prod runtime role ``w2v_orch_app`` is a non-owner, so ENABLE
  binds it) + ``tenant_isolation`` on ``tenant_id``.
* ``resolve_api_key(prefix text, key_hash bytea)`` — SECURITY DEFINER, the ONE
  pre-tenant read: the key row matching the public prefix AND the SHA-256 of the
  full presented key, returning only ``(key_id, tenant_id, scopes, revoked_at,
  expires_at)`` (never the hash). Same mechanism as ``run_tenant`` (0030) and
  ``tenant_is_occupied`` (0076).
* ``api_key_prefix_taken(prefix text)`` — SECURITY DEFINER boolean so minting can
  check the GLOBAL prefix uniqueness that a tenant-scoped SELECT cannot see.
* EXECUTE revoked from PUBLIC, granted to ``w2v_orch_app`` under the role guard
  (the role exists only in prod).

The DDL is a FROZEN literal of ``src.database.rls.tenant_api_keys_upgrade_statements()``
and ``api_key_functions_statements()`` (pinned by
``tests/database/test_migration_0090_api_keys_rls.py``). Drift-tolerant:
``DROP POLICY IF EXISTS`` before ``CREATE POLICY``, ``CREATE OR REPLACE FUNCTION``,
ENABLE is idempotent. Downgrade drops the functions and the policy and DISABLEs RLS.

Revision ID: 0090_api_keys_rls
Revises: 0089_tenant_licence_token
Create Date: 2026-10-14
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 17 characters.
revision = "0090_api_keys_rls"
down_revision = "0089_tenant_licence_token"
branch_labels = None
depends_on = None

# FROZEN DDL: inlined snapshot of ``src.database.rls.tenant_api_keys_upgrade_statements()``
# as rendered at freeze time (2026-10-14). Do NOT re-import the live generator.
_RLS_UPGRADE_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE tenant_api_keys ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON tenant_api_keys USING (tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid) WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)",
)

# FROZEN DDL: inlined snapshot of ``src.database.rls.api_key_functions_statements()``.
_FUNCTION_STATEMENTS: tuple[str, ...] = (
    "DO $do$ BEGIN EXECUTE 'CREATE OR REPLACE FUNCTION resolve_api_key(p_prefix text, p_key_hash bytea) RETURNS TABLE (key_id uuid, tenant_id uuid, scopes text[], revoked_at timestamptz, expires_at timestamptz) LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, ' || quote_ident(current_schema()) || ', pg_temp AS $fn$ SELECT k.id, k.tenant_id, k.scopes, k.revoked_at, k.expires_at FROM tenant_api_keys k WHERE k.prefix = p_prefix AND k.key_hash = p_key_hash $fn$'; END $do$",
    "REVOKE EXECUTE ON FUNCTION resolve_api_key(text, bytea) FROM PUBLIC",
    "DO $do$ BEGIN EXECUTE 'CREATE OR REPLACE FUNCTION api_key_prefix_taken(p_prefix text) RETURNS boolean LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, ' || quote_ident(current_schema()) || ', pg_temp AS $fn$ SELECT EXISTS (SELECT 1 FROM tenant_api_keys WHERE prefix = p_prefix) $fn$'; END $do$",
    "REVOKE EXECUTE ON FUNCTION api_key_prefix_taken(text) FROM PUBLIC",
    "DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_orch_app') THEN GRANT EXECUTE ON FUNCTION resolve_api_key(text, bytea) TO w2v_orch_app; GRANT EXECUTE ON FUNCTION api_key_prefix_taken(text) TO w2v_orch_app; END IF; END $$",
)

# FROZEN DDL: inlined snapshot of ``src.database.rls.drop_api_key_functions_statements()``.
_DROP_FUNCTION_STATEMENTS: tuple[str, ...] = (
    "DROP FUNCTION IF EXISTS api_key_prefix_taken(text)",
    "DROP FUNCTION IF EXISTS resolve_api_key(text, bytea)",
)

_DROP_POLICY = "DROP POLICY IF EXISTS tenant_isolation ON tenant_api_keys"
_DISABLE_RLS = "ALTER TABLE tenant_api_keys DISABLE ROW LEVEL SECURITY"


def upgrade() -> None:
    op.execute(_DROP_POLICY)
    for statement in _RLS_UPGRADE_STATEMENTS:
        op.execute(statement)
    for statement in _FUNCTION_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DROP_FUNCTION_STATEMENTS:
        op.execute(statement)
    op.execute(_DROP_POLICY)
    op.execute(_DISABLE_RLS)
