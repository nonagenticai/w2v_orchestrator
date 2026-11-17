"""allow the 'seo_audit' project type

seo_audit (blueprint ``seo_audit_v1``) is the lightest product type: a single
``seo_audit_scan`` stage calls analysis_service's synchronous
``POST /api/seo/audit`` and emits the SEO Health Score, per-group breakdown, and
prioritized findings — no video pipeline. ``projects.type`` is gated by the
``ck_projects_type`` CHECK constraint, so adding the new type means rebuilding it
(kept in lockstep with the Pydantic ``ProjectType`` Literal + the model's
``CheckConstraint``).

Revision ID: 0040
Revises: 0039
Create Date: 2026-07-05
"""

from alembic import op

revision = "0040"
down_revision = "0039"
branch_labels = None
depends_on = None

_TYPE_OLD = (
    "type IN ('promo','course','app_review','generic','lesson','website_review',"
    "'audiobook','web_design','web_design_v2','demo_video','showcase',"
    "'growth_bundle')"
)
_TYPE_NEW = (
    "type IN ('promo','course','app_review','generic','lesson','website_review',"
    "'audiobook','web_design','web_design_v2','demo_video','showcase',"
    "'growth_bundle','seo_audit')"
)


def upgrade() -> None:
    op.drop_constraint("ck_projects_type", "projects", type_="check")
    op.create_check_constraint("ck_projects_type", "projects", _TYPE_NEW)


def downgrade() -> None:
    op.drop_constraint("ck_projects_type", "projects", type_="check")
    op.create_check_constraint("ck_projects_type", "projects", _TYPE_OLD)
