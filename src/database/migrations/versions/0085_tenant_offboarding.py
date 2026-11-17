"""tenant_offboarding: P6 tenant export + delete jobs (ORCH-6)

One row per offboarding job — a tenant_admin REQUEST (``status='requested'``) or a
super_admin-initiated export/delete. The super_admin executes; a tenant_admin can only
request (``src/api/routes/tenancy/offboarding.py``).

* ``kind`` export | delete; ``status`` requested | running | done | failed | rejected.
* ``artifact_key`` — the export archive's object key, CHECKed to sit under
  ``exports/tenants/<tenant_id>/``. Only the KEY is stored; a download URL is presigned
  on every read and never persisted.
* ``progress`` — the resumable-delete ledger (rows-deleted phase, object prefixes still
  pending / already purged). A crashed or failed job resumes from it.
* ``result`` — the final report (row counts, purged prefixes, what was retained, and
  the downstream services whose data was NOT deleted).
* At most one OPEN (requested|running) job per ``(tenant_id, kind)`` — partial unique
  index — so two operators cannot run the same offboarding concurrently.
* FK to ``tenants`` WITHOUT cascade: a deleted tenant keeps its registry row as a
  tombstone (``status='deleted'``) and its job rows as the proof of deletion.
* NO RLS: the super_admin queue reads across tenants; the tenant_admin route filters
  on its own authorized tenant explicitly (the ``tenant_invites`` posture).

``tenants.status`` gains ``deleted`` (the tombstone). ``ck_admin_audit_action`` is
widened to EXACTLY 0084's set plus ``offboarding.reject`` (``DROP CONSTRAINT IF
EXISTS`` + re-add, a superset). Downgrades restore the old CHECKs ``NOT VALID`` (the
0082/0084 posture: rows written under the new values are kept, new ones refused).

Drift-tolerant: ``CREATE TABLE IF NOT EXISTS`` / ``CREATE INDEX IF NOT EXISTS``.

Revision ID: 0085_tenant_offboarding
Revises: 0084_tenant_sound_assets
Create Date: 2026-10-09
"""

from alembic import op

# ``alembic_version.version_num`` is varchar(32); this id is 23 characters.
revision = "0085_tenant_offboarding"
down_revision = "0084_tenant_sound_assets"
branch_labels = None
depends_on = None

_TABLE = """
    CREATE TABLE IF NOT EXISTS tenant_offboarding (
        id uuid NOT NULL,
        tenant_id uuid NOT NULL,
        kind text NOT NULL,
        status text NOT NULL DEFAULT 'requested',
        requested_by uuid,
        decided_by uuid,
        reason text,
        force boolean NOT NULL DEFAULT false,
        artifact_key text,
        error text,
        progress jsonb NOT NULL DEFAULT '{}'::jsonb,
        result jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        started_at timestamptz,
        finished_at timestamptz,
        CONSTRAINT tenant_offboarding_pkey PRIMARY KEY (id),
        CONSTRAINT tenant_offboarding_tenant_id_fkey FOREIGN KEY (tenant_id)
            REFERENCES tenants (id),
        CONSTRAINT ck_tenant_offboarding_kind CHECK (kind IN ('export','delete')),
        CONSTRAINT ck_tenant_offboarding_status
            CHECK (status IN ('requested','running','done','failed','rejected')),
        CONSTRAINT ck_tenant_offboarding_artifact_key
            CHECK (artifact_key IS NULL
                   OR starts_with(artifact_key, 'exports/tenants/' || tenant_id::text || '/'))
    )
"""

_INDEXES: tuple[str, ...] = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_tenant_offboarding_open "
    "ON tenant_offboarding (tenant_id, kind) WHERE status IN ('requested','running')",
    "CREATE INDEX IF NOT EXISTS idx_tenant_offboarding_tenant "
    "ON tenant_offboarding (tenant_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_tenant_offboarding_status "
    "ON tenant_offboarding (status, created_at)",
)

# FROZEN literals, never imported.
_OLD_TENANT_STATUS = "status IN ('active','suspended','licence_lapsed')"
_NEW_TENANT_STATUS = "status IN ('active','suspended','licence_lapsed','deleted')"
_DROP_TENANT_CHECK = "ALTER TABLE tenants DROP CONSTRAINT IF EXISTS ck_tenants_status"

# 0084's ``_NEW_ACTIONS`` verbatim (the downgrade target) and that set plus
# ``offboarding.reject``.
_OLD_ACTIONS = (
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
    "'offboarding.request','offboarding.export','offboarding.delete',"
    "'offboarding.reject')"
)
_DROP_AUDIT_CHECK = "ALTER TABLE admin_audit_log DROP CONSTRAINT IF EXISTS ck_admin_audit_action"


def upgrade() -> None:
    op.execute(_TABLE)
    for statement in _INDEXES:
        op.execute(statement)
    op.execute(_DROP_TENANT_CHECK)
    op.execute(f"ALTER TABLE tenants ADD CONSTRAINT ck_tenants_status CHECK ({_NEW_TENANT_STATUS})")
    op.execute(_DROP_AUDIT_CHECK)
    op.execute(
        f"ALTER TABLE admin_audit_log ADD CONSTRAINT ck_admin_audit_action CHECK ({_NEW_ACTIONS})"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS tenant_offboarding")
    op.execute(_DROP_TENANT_CHECK)
    op.execute(
        "ALTER TABLE tenants ADD CONSTRAINT ck_tenants_status "
        f"CHECK ({_OLD_TENANT_STATUS}) NOT VALID"
    )
    op.execute(_DROP_AUDIT_CHECK)
    op.execute(
        "ALTER TABLE admin_audit_log ADD CONSTRAINT ck_admin_audit_action "
        f"CHECK ({_OLD_ACTIONS}) NOT VALID"
    )
