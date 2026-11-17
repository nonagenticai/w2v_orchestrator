"""video_sources.preview_leased_until — a DURABLE claim for the preview-proxy worker

``_work_locked_source`` computed ``attempts = source.attempts + 1`` as a PYTHON
LOCAL and only wrote it after ``_build_proxy`` returned or raised — i.e. after up
to ``preview_proxy_timeout_seconds`` of ffmpeg. The row's only claim was
``pg_try_advisory_xact_lock``, which is transaction-scoped and dies with the
connection. So when the POD died mid-transcode (container OOMKill of PID 1, node
eviction/drain SIGKILL, node reboot — worker2 rebooted 2026-08-22) the increment
was never written: the lock evaporated, the row stayed ``preview_status='pending'``
with its counter unchanged, and the resolver re-offered it every tick FOREVER. The
``max_attempts`` cap lived only on the caught-exception path in Python, so it could
never engage for that class — one pathological clip could burn a transcode slot for
the lifetime of the deployment.

The obvious fix — commit the counter bump before transcoding — is not available on
its own: committing RELEASES the xact lock, so a second pod could duplicate the
work. Hence a lease. The worker bumps the counter AND stamps
``preview_leased_until = now() + timeout + margin`` in ONE conditional statement
and commits it before ffmpeg starts; this resolver then skips rows leased into the
future. A dead pod needs no reaper: its lease simply EXPIRES and the row is
re-offered with the attempt already durably counted, so the cap converges.

Additive and backward-compatible: the column is NULLable with no backfill (NULL =
unleased = exactly today's behaviour), and an OLDER image against the migrated DB
still works — it never writes the column, so every row reads as unleased.

⚠ The DDL below is a FROZEN literal copy of
``src.database.rls.video_source_functions_statements()`` as of this revision, per
this repo's freeze convention (see ``tests/database/test_frozen_migration_ddl.py``
and 0030/0031/0035/0064's banners). If the resolver must change again, that is a
NEW forward migration, not an edit here. ``downgrade()`` restores 0064's body
verbatim so a down/up cycle is a true round-trip.

Revision ID: 0065_preview_lease
Revises: 0064_pending_video_src
Create Date: 2026-08-22
"""

import sqlalchemy as sa
from alembic import op

# ``version_num`` is varchar(32) — a longer id applies the DDL and then fails the
# stamp, leaving the DB migrated but unmarked. This one is 18 characters.
revision = "0065_preview_lease"
down_revision = "0064_pending_video_src"
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
    "WHERE vs.preview_status = 'pending' "
    "AND (vs.preview_leased_until IS NULL OR vs.preview_leased_until < now()) "
    "ORDER BY vs.created_at LIMIT p_limit $$"
)
# 0064's body, verbatim — what ``downgrade()`` must put back. Kept as its own
# literal rather than reached for by importing 0064: a migration that reads
# another migration's module is exactly the un-freezing this convention forbids.
_PENDING_VIDEO_SOURCES_FN_0064 = (
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
# CREATE OR REPLACE does not carry privileges across, and the role guard is the
# same one 0064 used: ``w2v_orch_app`` exists only in prod, so an unconditional
# GRANT errors on a CI/test DB where migrations run as the owner.
_GRANT_PENDING_VIDEO_SOURCES = (
    "DO $$ BEGIN "
    "IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_orch_app') THEN "
    "GRANT EXECUTE ON FUNCTION pending_video_sources(integer) TO w2v_orch_app; "
    "END IF; END $$"
)
_REVOKE_PENDING_VIDEO_SOURCES = (
    "REVOKE EXECUTE ON FUNCTION pending_video_sources(integer) FROM PUBLIC"
)


def upgrade() -> None:
    # Column FIRST: the replacement function body references it, and a `sql`
    # function body is parsed at CREATE time.
    op.add_column(
        "video_sources",
        sa.Column("preview_leased_until", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(_PENDING_VIDEO_SOURCES_FN)
    op.execute(_REVOKE_PENDING_VIDEO_SOURCES)
    op.execute(_GRANT_PENDING_VIDEO_SOURCES)


def downgrade() -> None:
    # Function first, for the mirror-image reason: dropping the column out from
    # under a body that still names it would fail.
    op.execute(_PENDING_VIDEO_SOURCES_FN_0064)
    op.execute(_REVOKE_PENDING_VIDEO_SOURCES)
    op.execute(_GRANT_PENDING_VIDEO_SOURCES)
    op.drop_column("video_sources", "preview_leased_until")
