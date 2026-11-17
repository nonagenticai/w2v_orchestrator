"""tenant_sound_assets: P6 per-tenant music library (ORCH-4)

One row per sound a tenant_admin uploaded. This table is the source of truth for the
tenant's library; ``sound_library/tenants/{t}/manifest.json`` is DERIVED from it
(``src.services.tenant_sound_library.regenerate_tenant_manifest``) and never edited
by hand.

* ``sound_id`` — the catalog id runs reference. Always ``tenant_<slug>`` (CHECK), so a
  tenant id can never collide with, or shadow, a built-in or shared catalog id, and the
  asset-rights ledger can tell a tenant upload from the bundled library by id alone.
  Unique per tenant among LIVE rows (partial unique index on ``deleted_at IS NULL``),
  so a deleted id can be reused.
* ``object_key`` — CHECKed to sit under ``sound_library/tenants/<tenant_id>/``: the DB
  itself refuses a row that points into another tenant's tree.
* ``rights_attested_by`` — the subject that ticked the rights attestation on upload.
  Tenant uploads are TENANT-DECLARED rights, not verified royalty-free.
* ``deleted_at`` — soft delete. A deleted row whose object is still needed by an
  active run stays as a "retired" manifest entry (resolvable by a run that already
  pinned it, never offered for a new pick) until no run of the tenant is active; then
  the object is deleted and the row removed.
* ``pinned_categories`` — the categories a track HELD while a run of the tenant was
  active, kept when a PATCH re-categorises it. A run pins its config for its whole
  lifetime, so a run that picked the track as ``music`` must keep resolving it as
  music after an admin re-files it as ``ambient``. Cleared by the first manifest
  regeneration that finds no active run (the same rule as purging retired rows).
  CHECKed to the category vocabulary. Added by a separate ``ADD COLUMN IF NOT
  EXISTS`` so a table built before the column existed is completed, not skipped.

RLS: ENABLE (not FORCE) + ``tenant_isolation`` on ``tenant_id``. The policy text is a
FROZEN literal of ``src.database.rls.tenant_sound_assets_upgrade_statements()``
(pinned by ``tests/database/test_migration_0084_tenant_sound_assets.py``). Runtime-role
privileges ride ``ALTER DEFAULT PRIVILEGES`` (gitops), so there is no GRANT here.

Audit vocabulary: ``music.upload`` / ``music.delete`` already exist (0082), but a
library edit (PATCH description/category) needs ``music.update``, which 0082 lacks.
This revision widens ``ck_admin_audit_action`` to EXACTLY 0082's set plus
``music.update`` (``DROP CONSTRAINT IF EXISTS`` + re-add, a superset, so every
existing row validates). The downgrade restores 0082's set ``NOT VALID`` — the same
posture as 0082's own downgrade: audit rows written under the new action are kept,
new ones refused.

Drift-tolerant: ``CREATE TABLE IF NOT EXISTS``, ``CREATE INDEX IF NOT EXISTS`` and
``DROP POLICY IF EXISTS`` before ``CREATE POLICY``. Purely additive.

Revision ID: 0084_tenant_sound_assets
Revises: 0083_run_usage_attribution
Create Date: 2026-10-09
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 24 characters.
revision = "0084_tenant_sound_assets"
down_revision = "0083_run_usage_attribution"
branch_labels = None
depends_on = None

_TABLE = """
    CREATE TABLE IF NOT EXISTS tenant_sound_assets (
        id uuid NOT NULL,
        tenant_id uuid NOT NULL,
        sound_id text NOT NULL,
        category text NOT NULL,
        description text NOT NULL DEFAULT '',
        object_key text NOT NULL,
        duration_s double precision NOT NULL,
        bytes bigint NOT NULL,
        content_type text NOT NULL,
        rights_attested_by uuid NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        deleted_at timestamptz,
        CONSTRAINT tenant_sound_assets_pkey PRIMARY KEY (id),
        CONSTRAINT tenant_sound_assets_tenant_id_fkey FOREIGN KEY (tenant_id)
            REFERENCES tenants (id) ON DELETE CASCADE,
        CONSTRAINT uq_tenant_sound_assets_object_key UNIQUE (object_key),
        CONSTRAINT ck_tenant_sound_assets_category
            CHECK (category IN ('music','ambient','accent','transition')),
        CONSTRAINT ck_tenant_sound_assets_sound_id
            CHECK (sound_id ~ '^tenant_[a-z0-9_]{1,56}$'),
        CONSTRAINT ck_tenant_sound_assets_object_key
            CHECK (starts_with(object_key, 'sound_library/tenants/' || tenant_id::text || '/')),
        CONSTRAINT ck_tenant_sound_assets_duration CHECK (duration_s > 0),
        CONSTRAINT ck_tenant_sound_assets_bytes CHECK (bytes > 0)
    )
"""

_PINNED_CATEGORIES = """
    ALTER TABLE tenant_sound_assets ADD COLUMN IF NOT EXISTS pinned_categories text[]
        NOT NULL DEFAULT '{}'::text[]
        CONSTRAINT ck_tenant_sound_assets_pinned_categories
        CHECK (pinned_categories <@ ARRAY['music','ambient','accent','transition']::text[])
"""

_INDEXES: tuple[str, ...] = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_tenant_sound_assets_live_sound_id "
    "ON tenant_sound_assets (tenant_id, sound_id) WHERE deleted_at IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_tenant_sound_assets_tenant ON tenant_sound_assets (tenant_id)",
)

# FROZEN DDL: inlined snapshot of
# ``src.database.rls.tenant_sound_assets_upgrade_statements()`` as rendered at freeze
# time (2026-10-09). Do NOT re-import the live generator.
_RLS_UPGRADE_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE tenant_sound_assets ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON tenant_sound_assets USING (tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid) WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)",
)

_DROP_POLICY = "DROP POLICY IF EXISTS tenant_isolation ON tenant_sound_assets"

# FROZEN: 0082's ``_NEW_ACTIONS`` verbatim (the downgrade target) and that set plus
# ``music.update``. Literals, never imported, so a later edit to 0082 cannot move them.
_OLD_ACTIONS = (
    "action IN ('tenant.update','scope.switch','scope.reset',"
    "'member.grant_role','member.revoke_role',"
    "'member.invite','member.remove','member.set_role','invite.revoke',"
    "'apikey.create','apikey.revoke','tenant.create',"
    "'connection.set','connection.clear_secret','connection.delete',"
    "'connection.test','connection.test_failed',"
    "'youtube.oauth_client.set','youtube.oauth_client.clear',"
    "'youtube.platform_channel.set','youtube.tenant_policy.set',"
    "'defaults.update','music.upload','music.delete',"
    "'offboarding.request','offboarding.export','offboarding.delete')"
)
_NEW_ACTIONS = (
    "action IN ('tenant.update','scope.switch','scope.reset',"
    "'member.grant_role','member.revoke_role',"
    "'member.invite','member.remove','member.set_role','invite.revoke',"
    "'apikey.create','apikey.revoke','tenant.create',"
    "'connection.set','connection.clear_secret','connection.delete',"
    "'connection.test','connection.test_failed',"
    "'youtube.oauth_client.set','youtube.oauth_client.clear',"
    "'youtube.platform_channel.set','youtube.tenant_policy.set',"
    "'defaults.update','music.upload','music.update','music.delete',"
    "'offboarding.request','offboarding.export','offboarding.delete')"
)
_DROP_AUDIT_CHECK = "ALTER TABLE admin_audit_log DROP CONSTRAINT IF EXISTS ck_admin_audit_action"


def upgrade() -> None:
    op.execute(_TABLE)
    op.execute(_PINNED_CATEGORIES)
    for statement in _INDEXES:
        op.execute(statement)
    op.execute(_DROP_POLICY)
    for statement in _RLS_UPGRADE_STATEMENTS:
        op.execute(statement)
    op.execute(_DROP_AUDIT_CHECK)
    op.execute(
        f"ALTER TABLE admin_audit_log ADD CONSTRAINT ck_admin_audit_action CHECK ({_NEW_ACTIONS})"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS tenant_sound_assets")
    op.execute(_DROP_AUDIT_CHECK)
    op.execute(
        "ALTER TABLE admin_audit_log ADD CONSTRAINT ck_admin_audit_action "
        f"CHECK ({_OLD_ACTIONS}) NOT VALID"
    )
