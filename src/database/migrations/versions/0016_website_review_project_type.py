"""allow the 'website_review' project type

W2 adds a website-review video project type whose run drives the
analysis_service (critique → redesign) into the standard
script → record → compile A/V pipeline. ``projects.type`` is a plain
``Text`` column gated by a CHECK constraint, so the only schema change
is rebuilding ``ck_projects_type`` with the new value.

Revision ID: 0016
Revises: 0015
Create Date: 2026-05-23
"""

from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None

_OLD = "type IN ('promo','course','app_review','generic','lesson')"
_NEW = "type IN ('promo','course','app_review','generic','lesson','website_review')"


def upgrade() -> None:
    op.drop_constraint("ck_projects_type", "projects", type_="check")
    op.create_check_constraint("ck_projects_type", "projects", _NEW)


def downgrade() -> None:
    op.drop_constraint("ck_projects_type", "projects", type_="check")
    op.create_check_constraint("ck_projects_type", "projects", _OLD)
