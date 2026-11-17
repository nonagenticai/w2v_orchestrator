"""allow the 'marketing_scorecard' project type

marketing_scorecard (blueprint ``marketing_scorecard_v1``) is a light product
type: a single ``marketing_scorecard_scan`` stage calls analysis_service's
synchronous ``POST /api/marketing/scorecard`` and emits the six-dimension
weighted marketing composite (content / conversion / seo / competitive / brand /
growth), an A-F grade, and quick-win / strategic buckets — no video pipeline.
``projects.type`` is gated by the ``ck_projects_type`` CHECK constraint, so
adding the new type means rebuilding it (kept in lockstep with the Pydantic
``ProjectType`` Literal + the model's ``CheckConstraint``).

Revision ID: 0042
Revises: 0041
Create Date: 2026-07-05
"""

from alembic import op

revision = "0042"
down_revision = "0041"
branch_labels = None
depends_on = None

_TYPE_OLD = (
    "type IN ('promo','course','app_review','generic','lesson','website_review',"
    "'audiobook','web_design','web_design_v2','demo_video','showcase',"
    "'growth_bundle','seo_audit','cro_audit')"
)
_TYPE_NEW = (
    "type IN ('promo','course','app_review','generic','lesson','website_review',"
    "'audiobook','web_design','web_design_v2','demo_video','showcase',"
    "'growth_bundle','seo_audit','cro_audit','marketing_scorecard')"
)


def upgrade() -> None:
    op.drop_constraint("ck_projects_type", "projects", type_="check")
    op.create_check_constraint("ck_projects_type", "projects", _TYPE_NEW)


def downgrade() -> None:
    op.drop_constraint("ck_projects_type", "projects", type_="check")
    op.create_check_constraint("ck_projects_type", "projects", _TYPE_OLD)
