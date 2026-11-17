"""`degraded` must survive into the run LIST and the live-monitor poll.

A run that soft-failed downstream work (a completed stage carrying ``notes``)
was only ever flagged on ``RunDetail``. Every other surface — the project's run
history list and the ``/status`` poll the monitor renders from — reported a
clean "Completed" for a run that shipped a broken deliverable. Prod case:
run 01a00b81-2576-71a5-becc-ffc4a843e6db shipped an 11.4s video against a 60s
target with a failed mastering gate and the list said "Completed".

The cost guard at the bottom is the load-bearing one: the lite/list paths must
compute the flag with a SQL predicate, NEVER by loading stage output blobs —
polling the heavy payload is what froze the CI browser and hung the E2E in the
first place.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid7

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

from src.database.models import StageOutput
from src.database.repositories.run_repository import RunRepository
from src.database.repositories.stage_output_repository import StageOutputRepository

pytestmark = pytest.mark.asyncio


async def _make_project(client) -> tuple[str, str]:
    r = await client.post(
        "/api/projects",
        json={
            "type": "generic",
            "title": "degraded visibility",
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
    detail = await client.get(f"/api/projects/{project_id}")
    return project_id, detail.json()["deliverables"][0]["id"]


async def _completed_run(client, db_session_factory, *, stages: list[dict]) -> tuple[str, str]:
    """A DB-direct completed run under ``client``'s tenant with the given stage rows.

    DB-direct (never POST) so no engine execution races the assertions.
    Returns (project_id, run_id_str).
    """
    from sqlalchemy import update

    from src.database.models import Run

    project_id, deliv = await _make_project(client)
    async with db_session_factory() as s:
        run = await RunRepository(s).create(
            deliverable_id=UUID(deliv),
            blueprint_id="generic_v1",
            config_snapshot={},
            pause_after_stages=[],
        )
        await s.commit()
        run_uuid = run.id
        await s.execute(update(Run).where(Run.id == run_uuid).values(status="completed"))
        for spec in stages:
            s.add(
                StageOutput(
                    id=uuid7(),
                    run_id=run_uuid,
                    stage_name=spec["stage_name"],
                    status=spec.get("status", "completed"),
                    output=spec.get("output"),
                )
            )
        await s.commit()
    return project_id, str(run_uuid)


_NOTED = {"stage_name": "media_discovery", "output": {"notes": ["source unavailable"]}}
_CLEAN = {"stage_name": "script_draft", "output": {"script": "ok", "notes": []}}


async def test_degraded_run_is_flagged_in_list_and_status_lite(client, db_session_factory) -> None:
    """REGRESSION: a completed stage with notes must degrade the LIST row and the
    live poll, not just the detail view."""
    project_id, rid = await _completed_run(client, db_session_factory, stages=[_NOTED, _CLEAN])

    listing = await client.get(f"/api/projects/{project_id}/runs")
    assert listing.status_code == 200, listing.text
    row = next(r for r in listing.json() if r["id"] == rid)
    assert row["degraded"] is True, "the run history list showed a broken run as clean"

    lite = await client.get(f"/api/runs/{rid}/status")
    assert lite.status_code == 200, lite.text
    assert lite.json()["degraded"] is True, "the live-monitor poll showed a broken run as clean"


async def test_clean_run_is_not_flagged_in_list_or_status_lite(client, db_session_factory) -> None:
    project_id, rid = await _completed_run(client, db_session_factory, stages=[_CLEAN])

    listing = await client.get(f"/api/projects/{project_id}/runs")
    row = next(r for r in listing.json() if r["id"] == rid)
    assert row["degraded"] is False

    lite = await client.get(f"/api/runs/{rid}/status")
    assert lite.json()["degraded"] is False


@pytest.mark.parametrize("stages", [[_NOTED, _CLEAN]])
async def test_detail_list_and_lite_agree(client, db_session_factory, stages) -> None:
    """The three surfaces derive the flag from different queries (Python over the
    ORM rows vs. a SQL predicate); they must never disagree about the same run."""
    project_id, rid = await _completed_run(client, db_session_factory, stages=stages)

    detail = (await client.get(f"/api/runs/{rid}")).json()["degraded"]
    listing = next(
        r for r in (await client.get(f"/api/projects/{project_id}/runs")).json() if r["id"] == rid
    )["degraded"]
    lite = (await client.get(f"/api/runs/{rid}/status")).json()["degraded"]

    assert detail is listing is lite, f"divergent degraded: {detail=} {listing=} {lite=}"
    assert detail is (stages is not None and _NOTED in stages)


async def test_list_and_lite_never_load_stage_output_blobs(client, db_session_factory) -> None:
    """COST GUARD. ``degraded`` must stay a SQL aggregate.

    ``RunStatusLite`` exists to be cheap — it omits every stage ``output`` blob by
    design, and refetching the heavy payload per poll froze the CI browser and hung
    the post-promote E2E. This test fails the moment someone computes the flag by
    loading stage outputs: ``list_for_run`` (the whole-ORM-row read, blobs
    included) must not be reached from either path, exactly one stage_outputs
    query may run per request, and nothing resembling an output payload may reach
    the wire.
    """
    project_id, rid = await _completed_run(client, db_session_factory, stages=[_NOTED, _CLEAN])

    statements: list[str] = []

    def _record(conn, cursor, statement, parameters, context, executemany) -> None:
        if "stage_outputs" in statement.lower():
            statements.append(statement)

    event.listen(Engine, "before_cursor_execute", _record)
    blob_load = AsyncMock(side_effect=AssertionError("list_for_run loads stage output blobs"))
    try:
        with patch.object(StageOutputRepository, "list_for_run", blob_load):
            lite = await client.get(f"/api/runs/{rid}/status")
            lite_queries = len(statements)
            listing = await client.get(f"/api/projects/{project_id}/runs")
    finally:
        event.remove(Engine, "before_cursor_execute", _record)

    assert lite.status_code == 200, lite.text
    assert listing.status_code == 200, listing.text
    blob_load.assert_not_called()

    assert lite_queries == 1, f"lite poll issued {lite_queries} stage_outputs queries: {statements}"
    list_queries = len(statements) - lite_queries
    assert list_queries == 1, (
        f"the run list issued {list_queries} stage_outputs queries — one aggregate for the "
        f"whole page, or it is an N+1: {statements[lite_queries:]}"
    )

    # No blob on the wire, and none read off it either: the SELECT list may
    # reference `output` only inside the degraded predicate, never as a column.
    assert "output" not in lite.json()
    assert all("output" not in stage for stage in lite.json()["stage_outputs"])
    assert all("output" not in row for row in listing.json())
