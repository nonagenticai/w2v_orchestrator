"""allow the 'audiobook' project type and 'audio' deliverable format

The audiobook pipeline ingests an uploaded manuscript, narrates it chunk by
chunk through the TTS service, and stitches per-chapter + combined audio
artifacts. It is an audio-only deliverable, so two plain ``Text`` columns gated
by CHECK constraints need rebuilding: ``projects.type`` (add ``audiobook``) and
``deliverables.format`` (add ``audio`` — the first non-aspect-ratio format).

Revision ID: 0018
Revises: 0017
Create Date: 2026-05-31
"""

from alembic import op

revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None

_TYPE_OLD = "type IN ('promo','course','app_review','generic','lesson','website_review')"
_TYPE_NEW = (
    "type IN ('promo','course','app_review','generic','lesson','website_review','audiobook')"
)

_FORMAT_OLD = "format IN ('16x9','9x16','1x1','4x5')"
_FORMAT_NEW = "format IN ('16x9','9x16','1x1','4x5','audio')"


def upgrade() -> None:
    op.drop_constraint("ck_projects_type", "projects", type_="check")
    op.create_check_constraint("ck_projects_type", "projects", _TYPE_NEW)
    op.drop_constraint("ck_deliverables_format", "deliverables", type_="check")
    op.create_check_constraint("ck_deliverables_format", "deliverables", _FORMAT_NEW)


def downgrade() -> None:
    op.drop_constraint("ck_deliverables_format", "deliverables", type_="check")
    op.create_check_constraint("ck_deliverables_format", "deliverables", _FORMAT_OLD)
    op.drop_constraint("ck_projects_type", "projects", type_="check")
    op.create_check_constraint("ck_projects_type", "projects", _TYPE_OLD)
