"""allow the 'demo_video' project type

The demo_video pipeline turns a website/app URL into a narrated, real-interaction
(type/drag/click) demo video: bdd_playwright_service generates a Playwright-BDD
suite, frontend_test_runner records one clip per scenario, TTS narrates each
scenario, and the canonical compile path muxes them into an MP4. ``projects.type``
is a plain Text column gated by a CHECK constraint, so adding the new type means
rebuilding ``ck_projects_type``.

Revision ID: 0033
Revises: 0032
Create Date: 2026-06-17
"""

from alembic import op

revision = "0033"
down_revision = "0032"
branch_labels = None
depends_on = None

_TYPE_OLD = (
    "type IN ('promo','course','app_review','generic','lesson','website_review',"
    "'audiobook','web_design')"
)
_TYPE_NEW = (
    "type IN ('promo','course','app_review','generic','lesson','website_review',"
    "'audiobook','web_design','demo_video')"
)


def upgrade() -> None:
    op.drop_constraint("ck_projects_type", "projects", type_="check")
    op.create_check_constraint("ck_projects_type", "projects", _TYPE_NEW)


def downgrade() -> None:
    op.drop_constraint("ck_projects_type", "projects", type_="check")
    op.create_check_constraint("ck_projects_type", "projects", _TYPE_OLD)
