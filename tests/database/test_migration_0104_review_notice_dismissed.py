"""0104 ``runs.review_notice_dismissed_at`` — upgrade / downgrade / re-apply on a real DB."""

from __future__ import annotations

from alembic import command
from sqlalchemy import Engine, text

import tests.database.test_migration_0102_admin_run_views_paused_at as m0102

probe = m0102.probe  # same throwaway DB fixture

_REV = "0104_review_notice_dismissed"
_PRE = "0103_stage_output_edit_reads"


def _has_column(engine: Engine) -> bool:
    with engine.connect() as conn:
        return (
            conn.execute(
                text(
                    "SELECT 1 FROM information_schema.columns WHERE table_name = 'runs' "
                    "AND column_name = 'review_notice_dismissed_at'"
                )
            ).first()
            is not None
        )


def test_upgrade_downgrade_reapply(probe) -> None:
    cfg, engine = probe
    command.upgrade(cfg, "head")
    assert _has_column(engine)
    command.downgrade(cfg, _PRE)
    assert not _has_column(engine)
    command.upgrade(cfg, "head")
    assert _has_column(engine)
