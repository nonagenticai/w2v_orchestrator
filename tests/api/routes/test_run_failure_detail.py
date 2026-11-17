"""#1106 — ``runs.failure_detail`` lands in postgres and reaches ``GET /api/runs/{id}``.

The frontend routes a byo_required tenant to the Providers page by branching on
``failure_detail.code`` / ``.modality`` — never by parsing ``error``. This proves the
whole path the frontend reads: ``mark_failed`` → the column → ``RunDetail``.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from src.database.repositories.run_repository import RunRepository
from src.models.failure_class import attribute_exception, failure_detail_for
from src.services.providers.errors import SlotNotConfiguredError

pytestmark = pytest.mark.asyncio


async def _deliverable(client) -> str:
    r = await client.post(
        "/api/projects",
        json={
            "type": "generic",
            "title": "byo refusal",
            "brief": {"angle": "growth", "target_duration_seconds": 30},
            "voice": {"descriptor": "crisp"},
            "deliverables": [
                {
                    "name": "d",
                    "format": "16x9",
                    "captions_mode": "none",
                    "target_duration_seconds": 30,
                }
            ],
        },
    )
    assert r.status_code == 201, r.text
    detail = await client.get(f"/api/projects/{r.json()['id']}")
    return detail.json()["deliverables"][0]["id"]


async def _failed_run(client, db_session_factory, exc: BaseException) -> str:
    deliv = await _deliverable(client)
    async with db_session_factory() as s:
        repo = RunRepository(s)
        run = await repo.create(
            deliverable_id=UUID(deliv),
            blueprint_id="generic_v1",
            config_snapshot={},
            pause_after_stages=[],
        )
        await repo.mark_started(run.id)
        assert await repo.mark_failed(
            run.id,
            f"{type(exc).__name__}: {exc}",
            failure_class=attribute_exception(exc),
            failure_detail=failure_detail_for(exc),
        )
        await s.commit()
        return str(run.id)


async def test_byo_refusal_reaches_the_run_detail_response(client, db_session_factory) -> None:
    run_id = await _failed_run(client, db_session_factory, SlotNotConfiguredError("tts"))

    r = await client.get(f"/api/runs/{run_id}")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "failed"
    assert body["failure_class"] == "customer_input"
    assert body["failure_detail"] == {
        "code": "byo_provider_required",
        "modality": "tts",
        # Additive (UI review 2026-10-17): ``scope`` is set only for ``upstream_auth``.
        "scope": None,
        # Additive (#1163): ``site`` / ``reason`` are set only for ``site_login_failed``.
        "site": None,
        "reason": None,
        "message": "Configure your own TTS provider to continue",
    }
    assert body["error"] == ("SlotNotConfiguredError: Configure your own TTS provider to continue")


async def test_a_failure_without_detail_serializes_null(client, db_session_factory) -> None:
    run_id = await _failed_run(client, db_session_factory, RuntimeError("boom"))

    body = (await client.get(f"/api/runs/{run_id}")).json()

    assert body["failure_detail"] is None
