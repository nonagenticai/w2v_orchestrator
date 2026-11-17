"""allow the 'growth_bundle' project type

growth_bundle (blueprint ``growth_bundle_v1``) is the standalone outbound
channel-copy pack: a single ``growth_bundle`` LLM stage that emits LinkedIn /
Twitter / email / landing-page snippets straight from the brief, with no video
pipeline. ``projects.type`` is gated by the ``ck_projects_type`` CHECK
constraint, so adding the new type means rebuilding it.

Revision ID: 0039
Revises: 0038
Create Date: 2026-07-04
"""

from alembic import op

revision = "0039"
down_revision = "0038"
branch_labels = None
depends_on = None

_TYPE_OLD = (
    "type IN ('promo','course','app_review','generic','lesson','website_review',"
    "'audiobook','web_design','web_design_v2','demo_video','showcase')"
)
_TYPE_NEW = (
    "type IN ('promo','course','app_review','generic','lesson','website_review',"
    "'audiobook','web_design','web_design_v2','demo_video','showcase',"
    "'growth_bundle')"
)


def upgrade() -> None:
    op.drop_constraint("ck_projects_type", "projects", type_="check")
    op.create_check_constraint("ck_projects_type", "projects", _TYPE_NEW)


def downgrade() -> None:
    op.drop_constraint("ck_projects_type", "projects", type_="check")
    op.create_check_constraint("ck_projects_type", "projects", _TYPE_OLD)
