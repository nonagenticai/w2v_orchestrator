"""purge_unpinned_provider_secrets() — superseded provider-secret GC (P3 WP3)

A tenant connection version's ``secret_ciphertext`` must stay readable while any
NON-TERMINAL run (pending / running / paused …) pins that version in
``runs.config_snapshot->'provider_slots'``; once superseded and unpinned it is
purged (the row stays for audit). The sweep is cross-tenant and the pod runs as
the restricted ``w2v_orch_app`` role under enforce, so — like ``orphan_runs`` /
``tenant_is_occupied`` — it is a ``SECURITY DEFINER`` function owned by the
migration user. It reads the pins and purges in ONE statement (no read-then-purge
window), raises instead of purging when a live run's pins are malformed, and only touches
versions superseded before the caller's cutoff (now - grace), so a just-superseded
version pinned by a not-yet-committed run survives. The cutoff is clamped server-
side to at most ``now() - 1h`` and ``p_limit`` to ``[0, 10000]``, and a run that
went terminal within the last hour still counts as live (the structural-relief
child inherits its parent's pins AFTER the parent is marked completed).

EXECUTE revoked from PUBLIC, granted to ``w2v_orch_app`` under a role guard.
``search_path`` pinned ``pg_catalog, <schema>, pg_temp``. Purely additive: an
older image never calls it.

⚠ FROZEN literal copy of ``src.database.rls.provider_secret_gc_statements()`` /
``drop_provider_secret_gc_statements()``; pinned by
``tests/database/test_frozen_migration_ddl.py``. A change is a NEW migration.

Revision ID: 0079_provider_secret_gc_fn
Revises: 0078_connection_audit_actions
Create Date: 2026-10-09
"""

from alembic import op

# ``version_num`` is varchar(32); this id is 26 characters.
revision = "0079_provider_secret_gc_fn"
down_revision = "0078_connection_audit_actions"
branch_labels = None
depends_on = None

# Frozen as of this revision. Do NOT re-import the live generator.
_PROVIDER_GC_FN = (
    "DO $do$ BEGIN EXECUTE $q$CREATE OR REPLACE FUNCTION "
    "purge_unpinned_provider_secrets(p_superseded_before timestamptz, p_limit "
    "integer) RETURNS integer LANGUAGE plpgsql VOLATILE SECURITY DEFINER SET "
    "search_path = pg_catalog, $q$ || quote_ident(current_schema()) || $q$, pg_temp "
    "AS $fn$ DECLARE n integer; v_cutoff timestamptz := LEAST(p_superseded_before, "
    "now() - interval '1 hour'); v_limit integer := GREATEST(LEAST(p_limit, 10000), "
    "0); BEGIN IF EXISTS (SELECT 1 FROM runs r WHERE (r.status NOT IN ('completed', "
    "'failed', 'cancelled') OR COALESCE(r.completed_at, r.updated_at) > now() - "
    "interval '1 hour') AND r.config_snapshot->'provider_slots' IS NOT NULL AND "
    "jsonb_typeof(r.config_snapshot->'provider_slots') NOT IN ('object', 'null')) "
    "THEN RAISE EXCEPTION 'malformed provider_slots on a live run; refusing to "
    "purge'; END IF; WITH pinned AS (SELECT (e.value->>'connection_id')::uuid AS "
    "connection_id, (e.value->>'version')::integer AS version FROM runs r CROSS JOIN "
    "LATERAL jsonb_each(CASE WHEN jsonb_typeof(r.config_snapshot->'provider_slots') ="
    " 'object' THEN r.config_snapshot->'provider_slots' ELSE '{}'::jsonb END) e WHERE"
    " (r.status NOT IN ('completed', 'failed', 'cancelled') OR "
    "COALESCE(r.completed_at, r.updated_at) > now() - interval '1 hour') AND "
    "e.value->>'source' = 'tenant'), victims AS (SELECT v.connection_id, v.version "
    "FROM tenant_connection_versions v WHERE v.superseded_at < v_cutoff AND "
    "v.secret_ciphertext IS NOT NULL AND v.purged_at IS NULL AND NOT EXISTS (SELECT 1"
    " FROM pinned p WHERE p.connection_id = v.connection_id AND p.version = "
    "v.version) ORDER BY v.superseded_at LIMIT v_limit) UPDATE "
    "tenant_connection_versions t SET secret_ciphertext = NULL, key_id = NULL, "
    "purged_at = now() FROM victims WHERE t.connection_id = victims.connection_id AND"
    " t.version = victims.version; GET DIAGNOSTICS n = ROW_COUNT; RETURN n; END "
    "$fn$$q$; END $do$"
)
_REVOKE_PROVIDER_GC = (
    "REVOKE EXECUTE ON FUNCTION purge_unpinned_provider_secrets(timestamptz, integer) FROM PUBLIC"
)
_GRANT_PROVIDER_GC = (
    "DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_orch_app') "
    "THEN GRANT EXECUTE ON FUNCTION purge_unpinned_provider_secrets(timestamptz, "
    "integer) TO w2v_orch_app; END IF; END $$"
)
_UPGRADE_STATEMENTS: tuple[str, ...] = (
    _PROVIDER_GC_FN,
    _REVOKE_PROVIDER_GC,
    _GRANT_PROVIDER_GC,
)
_DOWNGRADE_STATEMENTS: tuple[str, ...] = (
    "DROP FUNCTION IF EXISTS purge_unpinned_provider_secrets(timestamptz, integer)",
)


def upgrade() -> None:
    for stmt in _UPGRADE_STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in _DOWNGRADE_STATEMENTS:
        op.execute(stmt)
