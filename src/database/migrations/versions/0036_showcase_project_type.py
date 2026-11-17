"""allow the 'showcase' project type

The showcase pipeline (blueprint ``showcase_v1``, PR #331) extends demo_video into
a branded product showcase: real form-recording footage (FTR ``package_url``) +
multi-voice dialogue narration + an interleaved branded card deck + soft-CC,
publishable to YouTube. ``api.schemas.ProjectType`` gained ``"showcase"`` but the
``projects.ck_projects_type`` CHECK constraint was not rebuilt, so creating a
showcase project failed with a CheckViolation (500). This adds the value.

Revision ID: 0036
Revises: 0035
Create Date: 2026-06-30
"""

from alembic import op

revision = "0036"
down_revision = "0035"
branch_labels = None
depends_on = None

_TYPE_OLD = (
    "type IN ('promo','course','app_review','generic','lesson','website_review',"
    "'audiobook','web_design','demo_video')"
)
_TYPE_NEW = (
    "type IN ('promo','course','app_review','generic','lesson','website_review',"
    "'audiobook','web_design','demo_video','showcase')"
)


def upgrade() -> None:
    op.drop_constraint("ck_projects_type", "projects", type_="check")
    op.create_check_constraint("ck_projects_type", "projects", _TYPE_NEW)


def downgrade() -> None:
    op.drop_constraint("ck_projects_type", "projects", type_="check")
    op.create_check_constraint("ck_projects_type", "projects", _TYPE_OLD)
