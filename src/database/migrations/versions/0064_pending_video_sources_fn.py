"""pending_video_sources() — the preview-proxy sweeper's cross-tenant resolver

``video_sources`` rows commit with ``preview_status='pending'`` and stay that way
until something produces the browser-playable proxy. That producer is a
pod-lifetime background loop (``src.lifecycle.preview_proxy``) which sweeps ALL
tenants on a session with no tenant bound — so under ``TENANT_RLS_MODE=enforce``
a plain ``SELECT FROM video_sources`` is scoped to an unset tenant and returns
zero rows forever.

Same bootstrap wall as the orphan reconciler, same established answer: one
``SECURITY DEFINER`` resolver owned by the migration user (postgres superuser in
prod → bypasses FORCE; the table owner in CI/tests → bypasses ENABLE-not-FORCE),
returning each pending row plus the tenant the worker must bind BEFORE it writes.
The data-plane policy on ``video_sources`` is UNCHANGED — no "bypass GUC"
disjunct is added anywhere — so every UPDATE the worker issues still runs under
that tenant's policy and provable isolation stays intact. Read-only; EXECUTE
revoked from PUBLIC and re-granted to ``w2v_orch_app`` under a role guard (that
role exists only in prod; an unconditional GRANT errors on a CI DB).

Purely additive: one new function, no table touched, no backfill. An OLDER image
boots fine against the migrated DB — it simply never calls the function.

⚠ The DDL below is a FROZEN literal copy of
``src.database.rls.video_source_functions_statements()`` /
``drop_video_source_functions_statements()``, per this repo's freeze convention
(see ``tests/database/test_frozen_migration_ddl.py`` and 0030/0031/0035's
banners): replaying a migration through a LIVE generator would emit whatever the
generator says today, retroactively rewriting this migration's replayed output.
If the resolver must change, that is a NEW forward migration, not an edit here.

Revision ID: 0064_pending_video_src
Revises: 0063_stage_output_write_once
Create Date: 2026-08-21
"""

from alembic import op

# ``version_num`` is varchar(32) — a longer id applies the DDL and then fails the
# stamp, leaving the DB migrated but unmarked. This one is 21 characters.
revision = "0064_pending_video_src"
down_revision = "0063_stage_output_write_once"
branch_labels = None
depends_on = None

# Frozen as of this revision. Do NOT re-import the live generator.
_PENDING_VIDEO_SOURCES_FN = (
    "CREATE OR REPLACE FUNCTION pending_video_sources(p_limit integer) "
    "RETURNS TABLE (source_id uuid, tenant_id uuid, project_id uuid, "
    "object_key text, attempts integer) "
    "LANGUAGE sql STABLE SECURITY DEFINER SET search_path FROM CURRENT AS $$ "
    "SELECT vs.id, p.tenant_id, vs.project_id, vs.object_key, "
    "CASE WHEN jsonb_typeof(vs.probe->'preview_worker'->'attempts') = 'number' "
    "THEN (vs.probe->'preview_worker'->>'attempts')::int ELSE 0 END "
    "FROM video_sources vs JOIN projects p ON p.id = vs.project_id "
    "WHERE vs.preview_status = 'pending' ORDER BY vs.created_at LIMIT p_limit $$"
)
_GRANT_PENDING_VIDEO_SOURCES = (
    "DO $$ BEGIN "
    "IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_orch_app') THEN "
    "GRANT EXECUTE ON FUNCTION pending_video_sources(integer) TO w2v_orch_app; "
    "END IF; END $$"
)
_UPGRADE_STATEMENTS: tuple[str, ...] = (
    _PENDING_VIDEO_SOURCES_FN,
    "REVOKE EXECUTE ON FUNCTION pending_video_sources(integer) FROM PUBLIC",
    _GRANT_PENDING_VIDEO_SOURCES,
)
_DOWNGRADE_STATEMENTS: tuple[str, ...] = ("DROP FUNCTION IF EXISTS pending_video_sources(integer)",)


def upgrade() -> None:
    for stmt in _UPGRADE_STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in _DOWNGRADE_STATEMENTS:
        op.execute(stmt)
