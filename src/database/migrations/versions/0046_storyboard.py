"""storyboard — Storyboard/Scene/Panel/Asset tables (Phase 0 backbone)

Four project-scoped tables mirroring the shipped ``verbatim_items`` pattern
(model + repository + routes + RLS join predicate). ``storyboard_panels`` is the
spine: surrogate uuid ``id`` PK, ``shot_key`` carries the sacred ``clip-<n>`` join
key, ``position``-ordered per project. Enum-shaped columns are ``Text`` +
``CheckConstraint`` (no native pg ENUM — same convention as verbatim/tenants).

RLS: ENABLE (not FORCE) + a join-based ``tenant_isolation`` policy climbing
``project_id → projects.tenant_id`` (1-hop, identical shape to verbatim_items),
frozen inline from ``src.database.rls`` so this migration and the ``create_all``
``after_create`` path (``register_late_join_rls_ddl``) can't drift. The tables are
registered in ``RLS_JOIN_PREDICATES_LATE`` — NOT ``RLS_JOIN_PREDICATES`` — because
historical migration 0028 iterates the latter at run time and runs before this one
(same trap as source_ingest_cache / ``RLS_POLICY_TABLES_LATE``); the ENABLE+POLICY
is emitted here directly instead.

Runtime-role table privileges ride ``ALTER DEFAULT PRIVILEGES`` (gitops), like
every other table — no GRANT here.

Revision ID: 0046
Revises: 0045
Create Date: 2026-07-12
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "0046"
down_revision = "0045"
branch_labels = None
depends_on = None

# FROZEN DDL — inlined snapshot of ``src.database.rls.late_join_downgrade_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_LATE_JOIN_DOWNGRADE_STATEMENTS: tuple[str, ...] = (
    "DROP POLICY IF EXISTS tenant_isolation ON storyboards",
    "ALTER TABLE storyboards DISABLE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_isolation ON storyboard_scenes",
    "ALTER TABLE storyboard_scenes DISABLE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_isolation ON storyboard_panels",
    "ALTER TABLE storyboard_panels DISABLE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_isolation ON storyboard_assets",
    "ALTER TABLE storyboard_assets DISABLE ROW LEVEL SECURITY",
)


# FROZEN DDL — inlined snapshot of ``src.database.rls.late_join_upgrade_statements()`` as rendered
# at freeze time (2026-10-05). Do NOT re-import the live generator: a later edit
# to rls.py would silently rewrite what this historical revision emits on a
# fresh bootstrap. Guarded by tests/database/test_frozen_migration_ddl.py.
_LATE_JOIN_UPGRADE_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE storyboards ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON storyboards USING (EXISTS (SELECT 1 FROM projects p WHERE p.id = storyboards.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM projects p WHERE p.id = storyboards.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
    "ALTER TABLE storyboard_scenes ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON storyboard_scenes USING (EXISTS (SELECT 1 FROM projects p WHERE p.id = storyboard_scenes.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM projects p WHERE p.id = storyboard_scenes.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
    "ALTER TABLE storyboard_panels ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON storyboard_panels USING (EXISTS (SELECT 1 FROM projects p WHERE p.id = storyboard_panels.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM projects p WHERE p.id = storyboard_panels.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
    "ALTER TABLE storyboard_assets ENABLE ROW LEVEL SECURITY",
    "CREATE POLICY tenant_isolation ON storyboard_assets USING (EXISTS (SELECT 1 FROM projects p WHERE p.id = storyboard_assets.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)) WITH CHECK (EXISTS (SELECT 1 FROM projects p WHERE p.id = storyboard_assets.project_id AND p.tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid))",
)


def _ts(col: str) -> sa.Column:
    return sa.Column(
        col, sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
    )


def upgrade() -> None:
    op.create_table(
        "storyboards",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "deliverable_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("deliverables.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="draft"),
        sa.Column("aspect_ratio", sa.Text(), nullable=True),
        sa.Column("art_style", sa.Text(), nullable=True),
        _ts("created_at"),
        _ts("updated_at"),
        sa.CheckConstraint("status IN ('draft','approved')", name="ck_storyboards_status"),
    )
    op.create_index("idx_storyboards_project", "storyboards", ["project_id"])

    op.create_table(
        "storyboard_scenes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "storyboard_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("storyboards.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("int_ext", sa.Text(), nullable=True),
        sa.Column("location", sa.Text(), nullable=True),
        sa.Column("time_of_day", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        _ts("created_at"),
        _ts("updated_at"),
    )
    op.create_index(
        "idx_storyboard_scenes_project", "storyboard_scenes", ["project_id", "position"]
    )
    op.create_index("idx_storyboard_scenes_storyboard", "storyboard_scenes", ["storyboard_id"])

    op.create_table(
        "storyboard_panels",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "storyboard_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("storyboards.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "scene_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("storyboard_scenes.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("shot_key", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("shot_number", sa.Text(), nullable=True),
        sa.Column("panel_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.Text(), nullable=False, server_default="draft"),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("action_description", sa.Text(), nullable=True),
        sa.Column("dialogue", sa.Text(), nullable=True),
        sa.Column("vo_text", sa.Text(), nullable=True),
        sa.Column("sfx_cue", sa.Text(), nullable=True),
        sa.Column("music_cue", sa.Text(), nullable=True),
        sa.Column("image_ref", sa.Text(), nullable=True),
        sa.Column("visual_type", sa.Text(), nullable=True),
        sa.Column("shot_size", sa.Text(), nullable=True),
        sa.Column("composition_notes", sa.Text(), nullable=True),
        sa.Column(
            "characters",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("location_asset_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "props",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("lighting_mood", sa.Text(), nullable=True),
        sa.Column("art_style", sa.Text(), nullable=True),
        sa.Column("aspect_ratio", sa.Text(), nullable=True),
        sa.Column("angle", sa.Text(), nullable=True),
        sa.Column("movement", sa.Text(), nullable=True),
        sa.Column("movement_path", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("lens", sa.Text(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("transition_out", sa.Text(), nullable=True),
        sa.Column(
            "media_ref",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column(
            "gen",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        _ts("created_at"),
        _ts("updated_at"),
        sa.CheckConstraint("status IN ('draft','approved')", name="ck_storyboard_panels_status"),
        sa.CheckConstraint(
            "shot_size IS NULL OR shot_size IN "
            "('EWS','WS','LS','MLS','MS','MCU','CU','ECU',"
            "'establishing','insert','OTS','POV','two_shot')",
            name="ck_storyboard_panels_shot_size",
        ),
        sa.CheckConstraint(
            "angle IS NULL OR angle IN ('eye_level','high','low','dutch','overhead','worm')",
            name="ck_storyboard_panels_angle",
        ),
        sa.CheckConstraint(
            "movement IS NULL OR movement IN "
            "('static','pan','tilt','dolly','zoom','crane','handheld','push_in','pull_out')",
            name="ck_storyboard_panels_movement",
        ),
        sa.CheckConstraint(
            "transition_out IS NULL OR transition_out IN "
            "('cut','dissolve','wipe','fade_in','fade_out','match_cut')",
            name="ck_storyboard_panels_transition_out",
        ),
        sa.CheckConstraint(
            "visual_type IS NULL OR visual_type IN "
            "('footage','slide','title_card','ai_image','upload')",
            name="ck_storyboard_panels_visual_type",
        ),
    )
    op.create_index(
        "idx_storyboard_panels_project", "storyboard_panels", ["project_id", "position"]
    )
    op.create_index(
        "idx_storyboard_panels_project_shot",
        "storyboard_panels",
        ["project_id", "shot_key"],
    )
    op.create_index("idx_storyboard_panels_scene", "storyboard_panels", ["scene_id"])
    op.create_index("idx_storyboard_panels_storyboard", "storyboard_panels", ["storyboard_id"])

    op.create_table(
        "storyboard_assets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("canonical_descriptor", sa.Text(), nullable=True),
        sa.Column(
            "reference_images",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("style_token", sa.Text(), nullable=True),
        sa.Column("lora_path", sa.Text(), nullable=True),
        sa.Column(
            "attributes",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        _ts("created_at"),
        _ts("updated_at"),
        sa.CheckConstraint(
            "kind IN ('character','location','prop')", name="ck_storyboard_assets_kind"
        ),
    )
    op.create_index("idx_storyboard_assets_project", "storyboard_assets", ["project_id", "kind"])
    op.create_index(
        "idx_storyboard_assets_project_name", "storyboard_assets", ["project_id", "name"]
    )

    # RLS: ENABLE-not-FORCE + join-based tenant_isolation policy on all 4 tables
    # (FROZEN snapshot of rls.py output; drift vs create_all is caught by tests/database/test_frozen_migration_ddl.py).
    for stmt in _LATE_JOIN_UPGRADE_STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in _LATE_JOIN_DOWNGRADE_STATEMENTS:
        op.execute(stmt)
    op.drop_index("idx_storyboard_assets_project_name", table_name="storyboard_assets")
    op.drop_index("idx_storyboard_assets_project", table_name="storyboard_assets")
    op.drop_table("storyboard_assets")
    op.drop_index("idx_storyboard_panels_storyboard", table_name="storyboard_panels")
    op.drop_index("idx_storyboard_panels_scene", table_name="storyboard_panels")
    op.drop_index("idx_storyboard_panels_project_shot", table_name="storyboard_panels")
    op.drop_index("idx_storyboard_panels_project", table_name="storyboard_panels")
    op.drop_table("storyboard_panels")
    op.drop_index("idx_storyboard_scenes_storyboard", table_name="storyboard_scenes")
    op.drop_index("idx_storyboard_scenes_project", table_name="storyboard_scenes")
    op.drop_table("storyboard_scenes")
    op.drop_index("idx_storyboards_project", table_name="storyboards")
    op.drop_table("storyboards")
