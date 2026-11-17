"""storyboard_panel_events — per-panel approval trail (Boords pattern)

Adds the append-only ``storyboard_panel_events`` history table plus the
current-state columns (``approved_at`` / ``approved_by``) on ``storyboard_panels``.
Mirrors 0046's structure: project-scoped, ``Text`` + ``CheckConstraint`` enum,
``timestamptz`` server defaults, JSONB ``detail`` blob, and RLS ENABLE-not-FORCE +
a join-based ``tenant_isolation`` policy climbing ``project_id ->
projects.tenant_id`` — sourced from ``src.database.rls``
(``RLS_JOIN_PREDICATES_LATE_EVENTS``) so this migration and the ``create_all``
``after_create`` path (``register_late_join_events_rls_ddl``) can't drift.

The table is registered in ``RLS_JOIN_PREDICATES_LATE_EVENTS`` — NOT
``RLS_JOIN_PREDICATES_LATE`` — because historical migration 0046 iterates the
latter at run time and runs before this one (same trap as storyboard /
source_ingest_cache); the ENABLE+POLICY is emitted here directly instead.

Runtime-role table privileges ride ``ALTER DEFAULT PRIVILEGES`` (gitops), like
every other table — no GRANT here.

Revision ID: 0049
Revises: 0048
Create Date: 2026-07-12
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "0049"
down_revision = "0048"
branch_labels = None
depends_on = None

# FROZEN DDL — inlined snapshot of ``src.database.rls.late_join_events_downgrade_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_LATE_JOIN_EVENTS_DOWNGRADE_STATEMENTS: tuple[str, ...] = (
    "DROP POLICY IF EXISTS tenant_isolation ON storyboard_panel_events",
    "ALTER TABLE storyboard_panel_events DISABLE ROW LEVEL SECURITY",
)


# FROZEN DDL — inlined snapshot of ``src.database.rls.late_join_events_upgrade_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_LATE_JOIN_EVENTS_UPGRADE_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE storyboard_panel_events ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON storyboard_panel_events USING (EXISTS (SELECT 1 FROM projects p WHERE p.id = storyboard_panel_events.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM projects p WHERE p.id = storyboard_panel_events.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
)


def _ts(col: str) -> sa.Column:
    return sa.Column(
        col, sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
    )


def upgrade() -> None:
    # Current approval snapshot on the panel row (append-only history lives in
    # storyboard_panel_events). Nullable — a draft panel has neither set.
    op.add_column(
        "storyboard_panels",
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "storyboard_panels",
        sa.Column("approved_by", sa.Text(), nullable=True),
    )

    op.create_table(
        "storyboard_panel_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "panel_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("storyboard_panels.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("actor_subject_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column(
            "detail",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        _ts("created_at"),
        sa.CheckConstraint(
            "event_type IN ('approved','unapproved')",
            name="ck_storyboard_panel_events_type",
        ),
    )
    op.create_index(
        "idx_storyboard_panel_events_panel",
        "storyboard_panel_events",
        ["panel_id", "created_at"],
    )
    op.create_index(
        "idx_storyboard_panel_events_project",
        "storyboard_panel_events",
        ["project_id", "created_at"],
    )

    # RLS: ENABLE-not-FORCE + join-based tenant_isolation policy (sourced from
    # rls.py, frozen; drift caught by tests/database/test_frozen_migration_ddl.py).
    for stmt in _LATE_JOIN_EVENTS_UPGRADE_STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in _LATE_JOIN_EVENTS_DOWNGRADE_STATEMENTS:
        op.execute(stmt)
    op.drop_index("idx_storyboard_panel_events_project", table_name="storyboard_panel_events")
    op.drop_index("idx_storyboard_panel_events_panel", table_name="storyboard_panel_events")
    op.drop_table("storyboard_panel_events")
    op.drop_column("storyboard_panels", "approved_by")
    op.drop_column("storyboard_panels", "approved_at")
