"""tenant_defaults: P6 per-tenant production defaults (ORCH-1)

One row per tenant (``tenant_id`` is the PK, FK ``tenants`` ON DELETE CASCADE). Every
setting column is NULLABLE and NULL means "use the deployment default", so a row with
every field NULL behaves exactly like no row at all:

* ``voice`` jsonb ``{voice_id, speed}``
* ``brand`` jsonb ``{primary_color, secondary_color, font_family}``
* ``captions`` jsonb ``{mode none|burned_in|overlay, karaoke}``
* ``voice_direction`` jsonb ``{tone}``
* ``retention_mode`` text, one of auto | critique_only | off (the audience-retention
  planner's ``direction.retention.mode``, NOT data retention)
* ``model_tier`` jsonb ``{timing_director}``

``version`` (>= 1) is the optimistic-concurrency counter ``PUT
/api/tenant/defaults`` compares against ``expected_version``. Each jsonb column has a
CHECK that it is an object (or NULL), so a scalar can never land there.

A run reads the row ONCE, at creation (``build_config_snapshot``), and folds the
values into its config snapshot, so a running run keeps the defaults it started with.

RLS: ENABLE (not FORCE) + ``tenant_isolation`` on ``tenant_id``. The policy text is a
FROZEN literal of ``src.database.rls.tenant_defaults_upgrade_statements()`` (pinned by
``tests/database/test_migration_0081_tenant_defaults.py``); the ``create_all`` path
gets it from ``register_tenant_defaults_rls_ddl``. Runtime-role table privileges ride
``ALTER DEFAULT PRIVILEGES`` (gitops), as for every table, so there is no GRANT here.

Drift-tolerant: ``CREATE TABLE IF NOT EXISTS`` and ``DROP POLICY IF EXISTS`` before the
``CREATE POLICY``, so the revision re-applies over a DB that already holds the table
(migrated but unstamped). Purely additive: an OLDER image never touches this table.

Revision ID: 0081_tenant_defaults
Revises: 0080_youtube_audit_actions
Create Date: 2026-10-09
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 20 characters.
revision = "0081_tenant_defaults"
down_revision = "0080_youtube_audit_actions"
branch_labels = None
depends_on = None

_TABLE = """
    CREATE TABLE IF NOT EXISTS tenant_defaults (
        tenant_id uuid NOT NULL,
        voice jsonb,
        brand jsonb,
        captions jsonb,
        voice_direction jsonb,
        retention_mode text,
        model_tier jsonb,
        version integer NOT NULL DEFAULT 1,
        updated_by uuid,
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT tenant_defaults_pkey PRIMARY KEY (tenant_id),
        CONSTRAINT tenant_defaults_tenant_id_fkey FOREIGN KEY (tenant_id)
            REFERENCES tenants (id) ON DELETE CASCADE,
        CONSTRAINT ck_tenant_defaults_version CHECK (version >= 1),
        CONSTRAINT ck_tenant_defaults_retention_mode
            CHECK (retention_mode IS NULL OR retention_mode IN ('auto','critique_only','off')),
        CONSTRAINT ck_tenant_defaults_voice
            CHECK (voice IS NULL OR jsonb_typeof(voice) = 'object'),
        CONSTRAINT ck_tenant_defaults_brand
            CHECK (brand IS NULL OR jsonb_typeof(brand) = 'object'),
        CONSTRAINT ck_tenant_defaults_captions
            CHECK (captions IS NULL OR jsonb_typeof(captions) = 'object'),
        CONSTRAINT ck_tenant_defaults_voice_direction
            CHECK (voice_direction IS NULL OR jsonb_typeof(voice_direction) = 'object'),
        CONSTRAINT ck_tenant_defaults_model_tier
            CHECK (model_tier IS NULL OR jsonb_typeof(model_tier) = 'object')
    )
"""

# FROZEN DDL: inlined snapshot of
# ``src.database.rls.tenant_defaults_upgrade_statements()`` as rendered at freeze time
# (2026-10-09). Do NOT re-import the live generator: a later edit to rls.py would
# silently rewrite what this historical revision emits on a fresh bootstrap.
_RLS_UPGRADE_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE tenant_defaults ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON tenant_defaults USING (tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid) WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)",
)

_DROP_POLICY = "DROP POLICY IF EXISTS tenant_isolation ON tenant_defaults"


def upgrade() -> None:
    op.execute(_TABLE)
    op.execute(_DROP_POLICY)
    for statement in _RLS_UPGRADE_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS tenant_defaults")
