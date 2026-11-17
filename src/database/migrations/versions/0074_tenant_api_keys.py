"""tenant_api_keys: tenant-scoped API keys (P2, WP1)

Key format ``w2v_sk_<prefix>_<secret>`` (``src/auth/api_key_format.py``): an
8-char lowercase base32 ``prefix`` (public, UNIQUE, the lookup key) and a 32-byte
urlsafe secret. Only ``key_hash`` = SHA-256 of the FULL key is stored; the secret
is shown once at creation. Verification looks the row up by prefix and compares
hashes with ``hmac.compare_digest``.

* ``revoked_at`` / ``expires_at`` nullable — a key is usable iff neither is in
  the past.
* FK to ``tenants`` ON DELETE CASCADE.

No RLS: the request path resolves a presented key to its tenant BEFORE any tenant
is known, i.e. cross-tenant by design. Drift-tolerant (``IF NOT EXISTS``); DDL is
inline literal text.

Revision ID: 0074_tenant_api_keys
Revises: 0073_tenant_invites
Create Date: 2026-10-08
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 20 characters.
revision = "0074_tenant_api_keys"
down_revision = "0073_tenant_invites"
branch_labels = None
depends_on = None

_UPGRADE: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS tenant_api_keys (
        id uuid NOT NULL,
        tenant_id uuid NOT NULL,
        name text NOT NULL,
        prefix text NOT NULL,
        key_hash bytea NOT NULL,
        created_by uuid,
        created_at timestamptz NOT NULL DEFAULT now(),
        last_used_at timestamptz,
        expires_at timestamptz,
        revoked_at timestamptz,
        CONSTRAINT tenant_api_keys_pkey PRIMARY KEY (id),
        CONSTRAINT tenant_api_keys_tenant_id_fkey FOREIGN KEY (tenant_id)
            REFERENCES tenants (id) ON DELETE CASCADE,
        CONSTRAINT uq_tenant_api_keys_prefix UNIQUE (prefix),
        CONSTRAINT ck_tenant_api_keys_prefix CHECK (prefix ~ '^[a-z2-7]{8}$'),
        CONSTRAINT ck_tenant_api_keys_key_hash_len CHECK (octet_length(key_hash) = 32)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tenant_api_keys_tenant ON tenant_api_keys (tenant_id)",
)

_DOWNGRADE: tuple[str, ...] = ("DROP TABLE IF EXISTS tenant_api_keys",)


def upgrade() -> None:
    for statement in _UPGRADE:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWNGRADE:
        op.execute(statement)
