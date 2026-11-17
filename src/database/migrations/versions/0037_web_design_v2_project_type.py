"""allow the 'web_design_v2' project type

web_design_v2 (blueprint ``web_design_v2``) is the computed-assembly, interactive,
verified web-design pipeline (brand_extract → design → verify → assemble). It drives
html_render_service in ``mode="assembly"``. ``projects.type`` is gated by the
``ck_projects_type`` CHECK constraint, so adding the new type means rebuilding it.

Revision ID: 0037
Revises: 0036
Create Date: 2026-07-01
"""

from alembic import op

revision = "0037"
down_revision = "0036"
branch_labels = None
depends_on = None

_TYPE_OLD = (
    "type IN ('promo','course','app_review','generic','lesson','website_review',"
    "'audiobook','web_design','demo_video','showcase')"
)
_TYPE_NEW = (
    "type IN ('promo','course','app_review','generic','lesson','website_review',"
    "'audiobook','web_design','web_design_v2','demo_video','showcase')"
)


def upgrade() -> None:
    op.drop_constraint("ck_projects_type", "projects", type_="check")
    op.create_check_constraint("ck_projects_type", "projects", _TYPE_NEW)


def downgrade() -> None:
    op.drop_constraint("ck_projects_type", "projects", type_="check")
    op.create_check_constraint("ck_projects_type", "projects", _TYPE_OLD)
