"""Migration 0069 blanks plaintext login secrets already stored in briefs and snapshots.

Rows are planted with RAW SQL — the ORM listener would seal them, and the rows this
migration exists for were written before that listener existed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import text

from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository

pytestmark = pytest.mark.asyncio

_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/database/migrations/versions/0069_redact_brief_login_secrets.py"
)


def _migration():
    spec = importlib.util.spec_from_file_location("m_0069", _PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Fake credentials (not real secrets).
_KC = "kc-pw"  # pragma: allowlist secret
_TW = "tw-pw"  # pragma: allowlist secret
_INFO = "info-pw"  # pragma: allowlist secret
_COOKIE = "cookie-session-value"  # pragma: allowlist secret

_LEAKY = {
    "target_url": "https://marketing.example",
    "auth_config": [
        {"login_url": "https://marketing.example", "username": "kc", "password": _KC},
        {"login_url": "https://twenty.example", "username": "tw", "password": _TW},
    ],
    "auth_info": {"login_url": "https://a.example", "username": "u", "password": _INFO},
    "sources": [
        {
            "ref": "https://c.example",
            "auth_info": {
                "type": "cookies",
                "cookies": [{"name": "s", "value": _COOKIE}],
                "cookie_header": f"s={_COOKIE}",
            },
        }
    ],
}


async def test_blank_rows_removes_every_plaintext_secret_and_is_rerunnable(
    db_session_factory,
) -> None:
    async with db_session_factory() as s:
        proj = await ProjectRepository(s).create_with_deliverables(
            tenant_id=uuid4(),
            owner_subject_id=uuid4(),
            type_="showcase",
            title="leaky",
            brief={},
            brand={},
            voice={},
            theme={},
            visibility="tenant",
            deliverables=[{"name": "d", "format": "16x9"}],
        )
        run = await RunRepository(s).create(
            deliverable_id=proj.deliverables[0].id,
            blueprint_id="showcase_v1",
            config_snapshot={},
            pause_after_stages=[],
        )
        clean = await ProjectRepository(s).create_with_deliverables(
            tenant_id=uuid4(),
            owner_subject_id=uuid4(),
            type_="generic",
            title="clean",
            brief={"angle": "password-free"},
            brand={},
            voice={},
            theme={},
            visibility="tenant",
            deliverables=[{"name": "d", "format": "16x9"}],
        )
        await s.execute(
            text("UPDATE projects SET brief = CAST(:b AS jsonb) WHERE id = :id"),
            {"b": json.dumps(_LEAKY), "id": proj.id},
        )
        await s.execute(
            text("UPDATE runs SET config_snapshot = CAST(:b AS jsonb) WHERE id = :id"),
            {"b": json.dumps({"brief": _LEAKY}), "id": run.id},
        )
        await s.commit()
        proj_id, run_id, clean_id = proj.id, run.id, clean.id

    migration = _migration()
    async with db_session_factory() as s:
        conn = await s.connection()
        counts = await conn.run_sync(migration.blank_rows)
        again = await conn.run_sync(migration.blank_rows)
        await s.commit()

    assert counts == {"projects": 1, "runs": 1}
    assert again == {"projects": 0, "runs": 0}

    async with db_session_factory() as s:
        brief = (
            await s.execute(text("SELECT brief FROM projects WHERE id = :id"), {"id": proj_id})
        ).scalar_one()
        snap = (
            await s.execute(text("SELECT config_snapshot FROM runs WHERE id = :id"), {"id": run_id})
        ).scalar_one()
        untouched = (
            await s.execute(text("SELECT brief FROM projects WHERE id = :id"), {"id": clean_id})
        ).scalar_one()
    for doc in (brief, snap["brief"]):
        dumped = json.dumps(doc)
        assert _KC not in dumped and _TW not in dumped and _INFO not in dumped
        assert _COOKIE not in dumped  # a cookie list is a credential too
        assert doc["auth_config"][1]["username"] == "tw"  # only the secret is blanked
    assert untouched == {"angle": "password-free"}
