"""video_source_uploads — durable state for a CHUNKED (resumable) video upload

Cloudflare returns 413 for any request body over 100 MB, so a 134 MB drone clip
cannot reach us as one PUT. The client slices the file into ``part_size`` parts,
PUTs them one at a time, and the server composes the final object into the
pre-allocated ``video_sources`` key. This migration adds the session table that
makes that survivable and nothing else.

⚠ **Why a TABLE and not pod memory / pod disk.** The orchestrator runs
``replicas: 2`` behind an HPA (min 2 / max 5). Part *n* and part *n+1* therefore
land on DIFFERENT pods routinely — there is no session affinity in front of it.
Anything the receiving pod remembers locally is invisible to the pod that serves
the next part, so a local running-byte-total would reset to zero on every hop and
the cumulative cap would never fire. The ledger lives here and is COMMITTED per
part PUT; that commit is the load-bearing bit, not the write.

Shape notes (the ``VideoSourceUpload`` model in ``src/database/models.py`` is the
mirror of this):

* ``id`` IS the ``upload_id`` handed to the client and echoed on every part PUT.
* ``source_id`` is PRE-ALLOCATED at create, before a byte arrives, so the final
  object key is fixed up-front and every part is written under a key derived from
  an id that already exists; completion inserts the ``video_sources`` row with
  THAT id. It is deliberately a bare uuid and **not** an FK — the row it names does
  not exist yet, so an FK would reject every open session.
* ``parts`` is the ledger: ``{"1": 33554432, ...}``, part number → bytes stored.
  JSONB keys are strings because JSON object keys cannot be integers. NOT NULL with
  a ``'{}'::jsonb`` default so a fresh session is an empty object, never a NULL to
  guard on. Re-PUTting a part overwrites its key (idempotent), and the running total
  the 512 MiB cap is checked against is the SUM of this map — never
  ``bytes_declared``, which is only the client's claim and is checked once at create.
* ``bytes_declared`` is ``BigInteger`` for the same reason ``video_sources.bytes``
  is: a 4K original passes the 2 GiB int4 ceiling routinely. ``part_size`` /
  ``part_count`` stay int4 — only the total can overflow.
* ``status`` is ``Text`` + ``CheckConstraint`` rather than a native pg ENUM, the same
  convention 0046/0051/0060 follow, so widening the vocabulary later is a drop/create
  CHECK rather than a type migration. ``aborted`` is a real terminal state and not
  merely an ``open`` row left to rot: an explicit abort must not wait 24 h for the
  sweeper.
* The ``(status, created_at)`` index serves the sweeper's only query — "open sessions
  older than 24 h". ``status`` leads so the scan starts inside the small ``open``
  slice instead of walking every completed session ever recorded.
* RLS: ENABLE (not FORCE) + a join-based ``tenant_isolation`` policy — a **1-hop**
  climb ``project_id -> projects.tenant_id``. The table deliberately carries NO
  ``tenant_id``, so a session can never disagree with the tenant of the project that
  owns it, and it deliberately does NOT climb via ``source_id`` (that row does not
  exist while the upload is open, so such a policy would hide exactly the rows it
  must protect). Sourced from ``src.database.rls``
  (``late_join_video_source_uploads_*_statements``) so this migration and the
  ``create_all`` ``after_create`` path can't drift, and emitted here directly — every
  earlier late map belongs to 0046/0049/0050/0051/0052/0060, which iterate them at run
  time *before* this table exists. Under enforce, an unbound session reads ZERO rows
  from this table; the user sees that as "my upload vanished mid-transfer", not as a
  permission error, so do not "fix" a missing-row report by relaxing the policy.

Runtime-role table privileges ride ``ALTER DEFAULT PRIVILEGES`` (gitops), like every
other table — no GRANT here (identical to 0046/0049/0050/0051/0060).

Purely additive: a new table with no backfill and no change to any existing table, so
an OLDER image boots and runs fine against the migrated DB (it simply has no chunked
upload route).

Revision ID: 0062_video_source_uploads
Revises: 0061_action_reel
Create Date: 2026-08-15
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


# ``version_num`` is varchar(32) — a longer id applies the DDL and then fails the
# stamp, leaving the DB migrated but unmarked. This one is 25 characters.
revision = "0062_video_source_uploads"
down_revision = "0061_action_reel"
branch_labels = None
depends_on = None

# FROZEN DDL — inlined snapshot of ``src.database.rls.late_join_video_source_uploads_downgrade_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_LATE_JOIN_VIDEO_SOURCE_UPLOADS_DOWNGRADE_STATEMENTS: tuple[str, ...] = (
    "DROP POLICY IF EXISTS tenant_isolation ON video_source_uploads",
    "ALTER TABLE video_source_uploads DISABLE ROW LEVEL SECURITY",
)


# FROZEN DDL — inlined snapshot of ``src.database.rls.late_join_video_source_uploads_upgrade_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_LATE_JOIN_VIDEO_SOURCE_UPLOADS_UPGRADE_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE video_source_uploads ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON video_source_uploads USING (EXISTS (SELECT 1 FROM projects p WHERE p.id = video_source_uploads.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM projects p WHERE p.id = video_source_uploads.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
)


def upgrade() -> None:
    op.create_table(
        "video_source_uploads",
        # The ``upload_id`` the client echoes on every part PUT.
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        # The RLS climb key. CASCADE: a half-finished upload is meaningless without
        # its project, and deleting a project must not leave a dangling session.
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # Pre-allocated id of the video_sources row this upload will become. NOT an
        # FK — that row is only inserted at completion.
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Display only — never used to build a storage key or a filesystem path.
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("content_type", sa.Text(), nullable=False),
        # Sanitized extension, carried separately so the object key never has to be
        # re-derived from the untrusted filename at completion time.
        sa.Column("ext", sa.Text(), nullable=False),
        # The client's CLAIM about the total size — checked once at create against the
        # per-upload ceiling, never trusted afterwards.
        sa.Column("bytes_declared", sa.BigInteger(), nullable=False),
        sa.Column("part_size", sa.Integer(), nullable=False),
        sa.Column("part_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="open"),
        # part number (string key) -> bytes stored. Rewritten and COMMITTED per part
        # PUT — this is what survives the pod hop under multi-replica.
        sa.Column(
            "parts",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "status IN ('open','completed','aborted')",
            name="ck_video_source_uploads_status",
        ),
    )
    # The 24 h garbage sweeper's only query: open sessions older than X.
    op.create_index(
        "idx_video_source_uploads_status_created",
        "video_source_uploads",
        ["status", "created_at"],
    )

    # RLS: ENABLE-not-FORCE + the 1-hop join-based tenant_isolation policy (sourced
    # frozen from rls.py; drift is caught by tests/database/test_frozen_migration_ddl.py).
    for stmt in _LATE_JOIN_VIDEO_SOURCE_UPLOADS_UPGRADE_STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    # Exact reverse of upgrade(): policy first (it is ON the table), then the index,
    # then the table itself — which takes the CHECK constraint with it.
    for stmt in _LATE_JOIN_VIDEO_SOURCE_UPLOADS_DOWNGRADE_STATEMENTS:
        op.execute(stmt)

    op.drop_index("idx_video_source_uploads_status_created", table_name="video_source_uploads")
    op.drop_table("video_source_uploads")
