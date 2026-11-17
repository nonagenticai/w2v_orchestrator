"""allow the 'action_reel' project type

action_reel (blueprint ``action_reel_v1``) turns raw camera/drone footage the
operator uploaded into ``video_sources`` (migration 0060) into a beat-cut reel:
an LLM picks the segments worth keeping, they are cut to a music bed's beat grid
with per-clip time-warp, and the canonical compile path renders them under
animated graphics.

``projects.type`` is a plain Text column gated by the ``ck_projects_type`` CHECK
constraint, so adding the new type means rebuilding it — kept in lockstep with
the Pydantic ``ProjectType`` Literal (``api/schemas/_shared.py``) and the model's
own ``CheckConstraint`` (``database/models.py``), which is what a fresh
``create_all`` schema gets.

Purely additive: WIDENING a CHECK can never reject a row that already exists, so
an OLDER image boots and runs fine against the migrated DB (it simply refuses to
CREATE an action_reel project, which it has no blueprint for anyway).

⚠ The revision id is ``0061_action_reel`` (15 chars) and not something more
descriptive: ``alembic_version.version_num`` is ``varchar(32)``, and an id past
that applies the DDL and THEN fails the stamp — leaving the DB migrated but
unmarked. Local gates miss it; CI catches it.

Revision ID: 0061_action_reel
Revises: 0060_video_sources
Create Date: 2026-08-09
"""

from alembic import op

revision = "0061_action_reel"
down_revision = "0060_video_sources"
branch_labels = None
depends_on = None

_TYPE_OLD = (
    "type IN ('promo','course','app_review','generic','lesson','website_review',"
    "'audiobook','web_design','web_design_v2','demo_video','showcase',"
    "'growth_bundle','seo_audit','cro_audit','marketing_scorecard')"
)
_TYPE_NEW = (
    "type IN ('promo','course','app_review','generic','lesson','website_review',"
    "'audiobook','web_design','web_design_v2','demo_video','showcase',"
    "'growth_bundle','seo_audit','cro_audit','marketing_scorecard','action_reel')"
)


def upgrade() -> None:
    op.drop_constraint("ck_projects_type", "projects", type_="check")
    op.create_check_constraint("ck_projects_type", "projects", _TYPE_NEW)


def downgrade() -> None:
    op.drop_constraint("ck_projects_type", "projects", type_="check")
    op.create_check_constraint("ck_projects_type", "projects", _TYPE_OLD)
