"""A paused run must say WHEN it started waiting for review, on every surface.

Prod case: run 01a1770d-676f-760d-926b-d8a2f129992a (promo_v1) paused for storyboard
review at 2026-10-26 07:15:21Z and nobody approved it. Every surface could only age it
from ``started_at``, so for 29 hours it read as a run that had been EXECUTING for 29
hours, and it was cancelled on 2026-10-27 as "stuck". ``runs.paused_at`` (0094) knew
the truth the whole time; nothing put it on the wire.

Covered here: the run detail, the run history list and the live ``/status`` poll.
The operator pane (``/api/admin/runs``) is covered in ``test_admin_runs.py``.
"""

from __future__ import annotations

import datetime
from uuid import UUID

import pytest
from sqlalchemy import update

from src.database.models import Run
from src.database.repositories.run_repository import RunRepository

pytestmark = pytest.mark.asyncio

_PAUSED_AT = datetime.datetime(2026, 10, 26, 7, 15, 21, tzinfo=datetime.UTC)


async def _run(
    client, db_session_factory, *, status: str, paused_at, pause_reason: str | None = None
) -> tuple[str, str]:
    """A DB-direct run (never POSTed, so no engine races the assertions)."""
    r = await client.post(
        "/api/projects",
        json={
            "type": "generic",
            "title": "paused visibility",
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
    project_id = r.json()["id"]
    deliv = (await client.get(f"/api/projects/{project_id}")).json()["deliverables"][0]["id"]
    async with db_session_factory() as s:
        run = await RunRepository(s).create(
            deliverable_id=UUID(deliv),
            blueprint_id="generic_v1",
            config_snapshot={},
            pause_after_stages=[],
        )
        await s.commit()
        await s.execute(
            update(Run)
            .where(Run.id == run.id)
            .values(
                status=status,
                current_stage="script_draft",
                paused_at=paused_at,
                pause_reason=pause_reason,
            )
        )
        await s.commit()
    return project_id, str(run.id)


async def _surfaces(client, project_id: str, rid: str) -> dict[str, object]:
    detail = await client.get(f"/api/runs/{rid}")
    assert detail.status_code == 200, detail.text
    listing = await client.get(f"/api/projects/{project_id}/runs")
    assert listing.status_code == 200, listing.text
    lite = await client.get(f"/api/runs/{rid}/status")
    assert lite.status_code == 200, lite.text
    row = next(r for r in listing.json() if r["id"] == rid)
    return {
        "detail": detail.json().get("paused_at"),
        "list": row.get("paused_at"),
        "status": lite.json().get("paused_at"),
        "detail_reason": detail.json().get("pause_reason"),
        "list_reason": row.get("pause_reason"),
        "status_reason": lite.json().get("pause_reason"),
    }


def _parse(value: object) -> datetime.datetime:
    assert isinstance(value, str), value
    return datetime.datetime.fromisoformat(value)


async def test_paused_run_carries_paused_at_on_detail_list_and_status(
    client, db_session_factory
) -> None:
    """REGRESSION (run 01a1770d): a run parked at a review gate exposes when it
    started waiting, on all three surfaces the frontend renders a run from."""
    project_id, rid = await _run(
        client, db_session_factory, status="paused", paused_at=_PAUSED_AT, pause_reason="checkpoint"
    )
    seen = await _surfaces(client, project_id, rid)
    for surface in ("detail", "list", "status"):
        value = seen[surface]
        assert value is not None, f"{surface} hid when the paused run started waiting"
        assert _parse(value) == _PAUSED_AT, f"{surface} reported the wrong pause start: {value}"
        assert seen[f"{surface}_reason"] == "checkpoint", f"{surface} hid WHY the run is paused"


async def test_manual_pause_reports_manual_not_review(client, db_session_factory) -> None:
    """A run someone paused is not waiting for a review; the wire must say which."""
    project_id, rid = await _run(
        client, db_session_factory, status="paused", paused_at=_PAUSED_AT, pause_reason="manual"
    )
    seen = await _surfaces(client, project_id, rid)
    assert {seen["detail_reason"], seen["list_reason"], seen["status_reason"]} == {"manual"}


@pytest.mark.parametrize("status", ["cancelled"])
async def test_terminal_run_never_reports_a_stale_pause(client, db_session_factory, status) -> None:
    """A cancel/fail does not clear ``runs.paused_at`` (01a1770d still has it after its
    cancel). A finished run is not waiting for anyone, so the wire must say None."""
    # Stale stamps on both columns: neither may leak onto a run that is not paused.
    project_id, rid = await _run(
        client, db_session_factory, status=status, paused_at=_PAUSED_AT, pause_reason="checkpoint"
    )
    assert set((await _surfaces(client, project_id, rid)).values()) == {None}


async def test_running_run_has_no_paused_at(client, db_session_factory) -> None:
    project_id, rid = await _run(client, db_session_factory, status="running", paused_at=None)
    assert set((await _surfaces(client, project_id, rid)).values()) == {None}
