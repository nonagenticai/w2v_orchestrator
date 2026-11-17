"""allow the 'web_design' project type

The web_design pipeline generates a brand-new HTML page from a text brief
(html_render_service greenfield design) and surfaces HTML + screenshot + PDF.
``projects.type`` is a plain Text column gated by a CHECK constraint, so adding
the new type means rebuilding ``ck_projects_type``.

Revision ID: 0020
Revises: 0019
Create Date: 2026-06-14
"""

from alembic import op

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None

_TYPE_OLD = (
    "type IN ('promo','course','app_review','generic','lesson','website_review','audiobook')"
)
_TYPE_NEW = (
    "type IN ('promo','course','app_review','generic','lesson','website_review',"
    "'audiobook','web_design')"
)


def upgrade() -> None:
    op.drop_constraint("ck_projects_type", "projects", type_="check")
    op.create_check_constraint("ck_projects_type", "projects", _TYPE_NEW)


def downgrade() -> None:
    op.drop_constraint("ck_projects_type", "projects", type_="check")
    op.create_check_constraint("ck_projects_type", "projects", _TYPE_OLD)
