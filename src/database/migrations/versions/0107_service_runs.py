"""service_runs: the "test this GPU service" bench history (service lab)

One row per bench call. The service lab performs a REAL provider call for one slot
(``image`` / ``tts`` / ``asr`` / ``llm`` / ``vlm`` / ``ocr`` / ``video``) and returns
the produced media inline; this table is the history the bench lists afterwards.

* ``slot`` — the BENCH vocabulary (``video``), not the provider registry's
  ``video_gen``: the UI, the API and the invoker all speak one word for one thing.
* ``status`` — ``succeeded`` / ``failed`` (CHECK). A failed call still keeps its row,
  ``error`` carrying ``{"code", "message"}``, so the bench can show what went wrong.
* ``request`` / ``provider`` — jsonb parameter and provenance blobs (provider =
  ``{"source","kind","model"}``). The request NEVER holds uploaded bytes.
* ``media_object_key`` — CHECKed to sit under
  ``media-cache/tenants/<tenant_id>/service-lab/``: the DB itself refuses a row that
  points into another tenant's tree (same rule as ``tenant_sound_assets``).
* ``media_content_type`` — ``image/png`` / ``audio/wav`` / ``video/mp4`` / … so the
  read route can answer with the right header without sniffing.

Media bytes are NOT in the database. They live in the object store under the
``media-cache/`` prefix, whose bucket lifecycle rule (``_ensure_media_cache_lifecycle``)
already expires them after ``STORAGE_MEDIA_CACHE_TTL_DAYS`` (30 by default). Rows are
pruned on the SAME horizon by ``ServiceRunRepository.delete_older_than``, called
opportunistically by the invoker — there is deliberately no new background loop, and
therefore no new lifecycle job to keep alive.

RLS: ENABLE (not FORCE) + ``tenant_isolation`` on ``tenant_id``. The policy text is a
FROZEN literal of ``src.database.rls.service_runs_upgrade_statements()``; the
``create_all`` ↔ alembic parity is what pins it (``tests/database/test_schema_parity.py``
and the round-trips in ``test_migration_0077_*`` / ``_0081_*``). Runtime-role privileges ride
``ALTER DEFAULT PRIVILEGES`` (gitops), so there is no GRANT here.

Revision ID: 0107_service_runs
Revises: 0106_backfill_login_signs_into
Create Date: 2026-11-01
"""

from __future__ import annotations

from alembic import op

revision = "0107_service_runs"
down_revision = "0106_backfill_login_signs_into"
branch_labels = None
depends_on = None

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS service_runs (
        id uuid PRIMARY KEY,
        tenant_id uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
        subject_id uuid NULL,
        slot text NOT NULL,
        status text NOT NULL CONSTRAINT ck_service_runs_status
            CHECK (status IN ('succeeded','failed')),
        output_kind text NULL,
        request jsonb NOT NULL DEFAULT '{}'::jsonb,
        text_output text NULL,
        media_object_key text NULL,
        media_content_type text NULL,
        duration_ms integer NULL,
        provider jsonb NOT NULL DEFAULT '{}'::jsonb,
        error jsonb NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT ck_service_runs_media_key
            CHECK (
                media_object_key IS NULL
                OR starts_with(
                    media_object_key,
                    'media-cache/tenants/' || tenant_id::text || '/service-lab/'
                )
            )
    )
"""

#: One index serves every read the bench makes: "this tenant's history for this slot,
#: newest first" (the list route's only query shape).
_INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_service_runs_tenant_slot_created "
    "ON service_runs (tenant_id, slot, created_at DESC)",
)

# FROZEN DDL: inlined snapshot of
# ``src.database.rls.service_runs_upgrade_statements()`` as rendered at freeze
# time (2026-11-01). Do NOT re-import the live generator.
_RLS_UPGRADE_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE service_runs ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON service_runs USING (tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid) WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)",
)

_DROP_POLICY = "DROP POLICY IF EXISTS tenant_isolation ON service_runs"


def upgrade() -> None:
    op.execute(_CREATE_TABLE)
    for statement in _INDEXES:
        op.execute(statement)
    # Drift-tolerant, like 0084: a ``create_all``-built schema (tests, dev) already
    # carries the policy and the table, so the migration must be able to apply on
    # top of it rather than fail on "already exists".
    op.execute(_DROP_POLICY)
    for statement in _RLS_UPGRADE_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute(_DROP_POLICY)
    op.execute("ALTER TABLE service_runs DISABLE ROW LEVEL SECURITY")
    for statement in reversed(_INDEXES):
        name = statement.split("INDEX IF NOT EXISTS ", 1)[1].split(" ", 1)[0]
        op.execute(f"DROP INDEX IF EXISTS {name}")
    op.drop_table("service_runs")
