"""tenant_connections + tenant_connection_versions: P3 provider slots (WP1)

A tenant may bind a provider slot (``llm`` first; the slot registry lives in
``src/services/providers/registry.py``) to its own connection. Two tables:

``tenant_connections`` — one row per (tenant, slot) binding.

* ``current_version`` names the live row in ``tenant_connection_versions``.
* ``status`` ∈ {active, deleted}; soft delete stamps ``deleted_at`` (CHECK keeps
  the two in lock-step). At most ONE active connection per (tenant, slot) — a
  PARTIAL unique index, so a deleted binding never blocks re-creating the slot.
* ``last_test_at`` / ``last_test_ok`` / ``last_test_error`` — the latest live test.
* FK to ``tenants`` ON DELETE CASCADE.

``tenant_connection_versions`` — IMMUTABLE config history, PK
``(connection_id, version)``. A run pins ``(connection_id, version)`` in its
config snapshot, so a resumed run keeps the provider it started with. The only
columns ever written after insert are the lifecycle stamps (``superseded_at``,
``purged_at``) and the secret pair (re-wrapped under a newer key, or NULLed by the
GC purge). The secret is AES-256-GCM (``src/services/providers/crypto.py``,
``v1|kid|nonce|ct+tag``, AAD = tenant|connection|version); ``key_id`` names the
keyring entry that sealed it (the CHECK keeps ciphertext and key id together) and
``secret_fingerprint`` is the 12-hex public fingerprint the API shows, kept after
a purge so the UI can still say which key a historical version used.

RLS: ENABLE (not FORCE) + ``tenant_isolation`` on both — direct ``tenant_id`` on
the connection, a one-hop EXISTS to the connection for the version (the version
carries no ``tenant_id``, so it can never disagree with its connection). The
policy text is a FROZEN literal of
``src.database.rls.tenant_connections_upgrade_statements()`` (pinned by
``tests/database/test_migration_0077_tenant_connections.py``); the ``create_all``
path gets it from ``register_tenant_connections_rls_ddl``. Runtime-role table
privileges ride ``ALTER DEFAULT PRIVILEGES`` (gitops), as for every table — no
GRANT here.

Drift-tolerant: ``IF NOT EXISTS`` on every object, ``DROP POLICY IF EXISTS``
before each ``CREATE POLICY`` (CREATE POLICY has no IF NOT EXISTS), so the
revision re-applies cleanly over a DB that already holds the tables (migrated but
unstamped). Purely additive: an OLDER image never touches these tables.

Revision ID: 0077_tenant_connections
Revises: 0076_tenant_is_occupied_fn
Create Date: 2026-10-09
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 23 characters.
revision = "0077_tenant_connections"
down_revision = "0076_tenant_is_occupied_fn"
branch_labels = None
depends_on = None

_TABLES: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS tenant_connections (
        id uuid NOT NULL,
        tenant_id uuid NOT NULL,
        slot text NOT NULL,
        current_version integer NOT NULL,
        status text NOT NULL DEFAULT 'active',
        last_test_at timestamptz,
        last_test_ok boolean,
        last_test_error text,
        created_by uuid,
        updated_by uuid,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        deleted_at timestamptz,
        CONSTRAINT tenant_connections_pkey PRIMARY KEY (id),
        CONSTRAINT tenant_connections_tenant_id_fkey FOREIGN KEY (tenant_id)
            REFERENCES tenants (id) ON DELETE CASCADE,
        CONSTRAINT ck_tenant_connections_status CHECK (status IN ('active','deleted')),
        CONSTRAINT ck_tenant_connections_deleted_at
            CHECK ((status = 'deleted') = (deleted_at IS NOT NULL)),
        CONSTRAINT ck_tenant_connections_current_version CHECK (current_version >= 1),
        CONSTRAINT ck_tenant_connections_slot CHECK (slot ~ '^[a-z][a-z0-9_]*(\\.[a-z0-9_]+)*$')
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_tenant_connections_tenant_slot_active "
    "ON tenant_connections (tenant_id, slot) WHERE status = 'active'",
    """
    CREATE TABLE IF NOT EXISTS tenant_connection_versions (
        connection_id uuid NOT NULL,
        version integer NOT NULL,
        kind text NOT NULL,
        base_url text,
        model text,
        options jsonb NOT NULL DEFAULT '{}'::jsonb,
        secret_ciphertext bytea,
        key_id text,
        secret_fingerprint text,
        created_by uuid,
        created_at timestamptz NOT NULL DEFAULT now(),
        superseded_at timestamptz,
        purged_at timestamptz,
        CONSTRAINT tenant_connection_versions_pkey PRIMARY KEY (connection_id, version),
        CONSTRAINT tenant_connection_versions_connection_id_fkey FOREIGN KEY (connection_id)
            REFERENCES tenant_connections (id) ON DELETE CASCADE,
        CONSTRAINT ck_tenant_connection_versions_version CHECK (version >= 1),
        CONSTRAINT ck_tenant_connection_versions_secret_key
            CHECK ((secret_ciphertext IS NULL) = (key_id IS NULL)),
        CONSTRAINT ck_tenant_connection_versions_purged
            CHECK (purged_at IS NULL OR secret_ciphertext IS NULL),
        CONSTRAINT ck_tenant_connection_versions_fingerprint
            CHECK (secret_fingerprint IS NULL OR secret_fingerprint ~ '^[0-9a-f]{12}$')
    )
    """,
    # The GC scan: superseded versions still holding a secret.
    "CREATE INDEX IF NOT EXISTS idx_tenant_connection_versions_purgeable "
    "ON tenant_connection_versions (superseded_at) "
    "WHERE superseded_at IS NOT NULL AND secret_ciphertext IS NOT NULL",
)

# FROZEN DDL — inlined snapshot of
# ``src.database.rls.tenant_connections_upgrade_statements()`` as rendered at freeze
# time (2026-10-09). Do NOT re-import the live generator: a later edit to rls.py
# would silently rewrite what this historical revision emits on a fresh bootstrap.
_RLS_UPGRADE_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE tenant_connections ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON tenant_connections USING (tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid) WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)",
    "ALTER TABLE tenant_connection_versions ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON tenant_connection_versions USING (EXISTS (SELECT 1 FROM tenant_connections c WHERE c.id = tenant_connection_versions.connection_id AND c.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM tenant_connections c WHERE c.id = tenant_connection_versions.connection_id AND c.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
)

# Drop the policy before (re)creating it so a re-apply over existing tables works.
_DROP_POLICIES: tuple[str, ...] = (
    "DROP POLICY IF EXISTS tenant_isolation ON tenant_connections",
    "DROP POLICY IF EXISTS tenant_isolation ON tenant_connection_versions",
)

# Child first; each policy goes with its table.
_DOWNGRADE: tuple[str, ...] = (
    "DROP TABLE IF EXISTS tenant_connection_versions",
    "DROP TABLE IF EXISTS tenant_connections",
)


def upgrade() -> None:
    for statement in _TABLES:
        op.execute(statement)
    for statement in _DROP_POLICIES:
        op.execute(statement)
    for statement in _RLS_UPGRADE_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWNGRADE:
        op.execute(statement)
