"""video_sources — persist an uploaded raw video (the ``action_reel`` input side)

``action_reel`` starts with the user handing us raw footage (4K drone clips), which
an LLM later cuts to music. Before any of that can exist, the upload itself must be
a durable, project-scoped row — this migration adds that table and nothing else.

Shape notes (the model in ``src/database/models.py`` is the mirror of this):

* ``object_key`` is the MinIO key of the UNTOUCHED original and is UNIQUE: the key
  is upload-addressed, so two rows pointing at one object would let deleting either
  row orphan or double-reap the same object.
* The three derived keys (``preview_key`` proxy / ``poster_key`` still /
  ``sprite_key`` filmstrip) are NULLABLE because they are produced asynchronously —
  the upload commits immediately with ``preview_status='pending'`` so the UI shows
  the file the moment it lands, and a worker later fills them and flips the status
  to ``'ready'`` (or ``'failed'`` + ``error``). ``preview_status`` is ``Text`` +
  ``CheckConstraint`` rather than a native pg ENUM — the same convention every
  storyboard table follows (0046/0051), so widening the vocabulary later is a
  drop/create CHECK rather than a type migration.
* ``bytes`` is ``BigInteger``: a 4K original passes the 2 GiB int4 ceiling routinely.
* ``probe`` is raw ffprobe JSONB (duration/fps/codec/width/height), NOT NULL with a
  ``'{}'::jsonb`` default so a pending row is an empty object and never a NULL to
  guard on. Not promoted to columns — ffprobe's shape varies per container and the
  cut planner feeds the whole blob to the LLM.
* RLS: ENABLE (not FORCE) + a join-based ``tenant_isolation`` policy — a **1-hop**
  climb ``project_id -> projects.tenant_id``. The table deliberately carries NO
  ``tenant_id``, so a source can never disagree with the tenant of the project that
  owns it. Sourced from ``src.database.rls``
  (``late_join_video_sources_*_statements``) so this migration and the ``create_all``
  ``after_create`` path can't drift, and emitted here directly — every earlier late
  map belongs to 0046/0049/0050/0051/0052, which iterate them at run time *before*
  this table exists.

Runtime-role table privileges ride ``ALTER DEFAULT PRIVILEGES`` (gitops), like every
other table — no GRANT here (identical to 0046/0049/0050/0051).

Purely additive: a new table with no backfill and no change to any existing table, so
an OLDER image boots and runs fine against the migrated DB.

Revision ID: 0060_video_sources
Revises: 0059
Create Date: 2026-08-09
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


# ``version_num`` is varchar(32) — a longer id applies the DDL and then fails the
# stamp, leaving the DB migrated but unmarked. This one is 18 characters.
revision = "0060_video_sources"
down_revision = "0059"
branch_labels = None
depends_on = None

# FROZEN DDL — inlined snapshot of ``src.database.rls.late_join_video_sources_downgrade_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_LATE_JOIN_VIDEO_SOURCES_DOWNGRADE_STATEMENTS: tuple[str, ...] = (
    "DROP POLICY IF EXISTS tenant_isolation ON video_sources",
    "ALTER TABLE video_sources DISABLE ROW LEVEL SECURITY",
)


# FROZEN DDL — inlined snapshot of ``src.database.rls.late_join_video_sources_upgrade_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_LATE_JOIN_VIDEO_SOURCES_UPGRADE_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE video_sources ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON video_sources USING (EXISTS (SELECT 1 FROM projects p WHERE p.id = video_sources.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM projects p WHERE p.id = video_sources.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
)


def upgrade() -> None:
    op.create_table(
        "video_sources",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        # The RLS climb key. CASCADE: an uploaded source is meaningless without its
        # project, and deleting a project must not leave a dangling row behind.
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # Display only — never used to build a storage key or a filesystem path.
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("object_key", sa.Text(), nullable=False),
        sa.Column("content_type", sa.Text(), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column(
            "preview_status",
            sa.Text(),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("preview_key", sa.Text(), nullable=True),
        sa.Column("poster_key", sa.Text(), nullable=True),
        sa.Column("sprite_key", sa.Text(), nullable=True),
        sa.Column(
            "probe",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "preview_status IN ('pending','ready','failed')",
            name="ck_video_sources_preview_status",
        ),
        sa.UniqueConstraint("object_key", name="uq_video_sources_object_key"),
    )
    # Every read is "the sources of THIS project" (the picker, the cut planner).
    op.create_index("idx_video_sources_project", "video_sources", ["project_id"])

    # RLS: ENABLE-not-FORCE + the 1-hop join-based tenant_isolation policy (sourced
    # frozen from rls.py; drift is caught by tests/database/test_frozen_migration_ddl.py).
    for stmt in _LATE_JOIN_VIDEO_SOURCES_UPGRADE_STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    # Exact reverse of upgrade(): policy first (it is ON the table), then the index,
    # then the table itself — which takes the CHECK + UNIQUE constraints with it.
    for stmt in _LATE_JOIN_VIDEO_SOURCES_DOWNGRADE_STATEMENTS:
        op.execute(stmt)

    op.drop_index("idx_video_sources_project", table_name="video_sources")
    op.drop_table("video_sources")
