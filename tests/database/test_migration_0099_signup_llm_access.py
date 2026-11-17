"""Migration 0099 (``signup_approvals.llm_access``, the approver's LLM access choice).

1. DB-free: the id fits varchar(32) and chains onto 0098; no application imports.
2. A real throwaway DATABASE: upgrade 0098 → 0099 backfills an already-approved
   row from what its provision step recorded; the CHECK refuses ``byo_required``;
   downgrade drops the column; upgrade again.
"""

from __future__ import annotations

import json
from uuid import UUID

from alembic import command
from sqlalchemy import text

from tests.database.test_migration_0084_tenant_sound_assets import (  # noqa: F401 — fixture
    _VERSIONS,
    _load,
    _rejects,
    probe,
)

_REV = "0099_signup_llm_access"
_PRE = "0098_signup_approvals"


def test_backfill_check_and_downgrade(probe) -> None:  # noqa: F811
    cfg, engine = probe
    command.upgrade(cfg, _PRE)
    steps = {"provision": {"status": "ok", "detail": {"llm_access": {"mode": "platform_only"}}}}
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO signup_approvals (id, email, name, source, status, tenant_slug, "
                "tenant_id, decided_at, steps) VALUES (:id, 'a@x.test', 'A', 'web', 'approved', "
                "'acme', :tid, now(), CAST(:steps AS jsonb))"
            ),
            {"id": UUID(int=1), "tid": UUID(int=0xA), "steps": json.dumps(steps)},
        )
        conn.execute(
            text(
                "INSERT INTO signup_approvals (id, email, name, source) "
                "VALUES (:id, 'b@x.test', 'B', 'web')"
            ),
            {"id": UUID(int=2)},
        )
    command.upgrade(cfg, _REV)
    with engine.connect() as conn:
        got = dict(conn.execute(text("SELECT id, llm_access FROM signup_approvals")).all())
    assert got == {UUID(int=1): "platform_only", UUID(int=2): None}
    upd = "UPDATE signup_approvals SET llm_access = :v WHERE id = :id"
    assert _rejects(engine, upd, {"v": "byo_required", "id": UUID(int=2)})
    assert not _rejects(engine, upd, {"v": "local_allowed", "id": UUID(int=2)})
    command.downgrade(cfg, _PRE)
    with engine.connect() as conn:
        cols = conn.execute(
            text(
                "SELECT count(*) FROM information_schema.columns "
                "WHERE table_name = 'signup_approvals' AND column_name = 'llm_access'"
            )
        ).scalar()
    assert cols == 0
    command.upgrade(cfg, _REV)
