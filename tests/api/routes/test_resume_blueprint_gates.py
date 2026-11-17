"""BACKLOG #6: ``resume_run`` must consult the run's blueprint's CURRENT gates.

A run's ``pause_after_stages`` is seeded from the blueprint at creation. When a
review gate is later removed from the blueprint, runs created before that keep
the stale gate in their row and used to re-pause on it after every resume
(manual remedy was ``PATCH /runs/{id}/checkpoints []``). Resume now intersects
drops a remaining pause only if the blueprint SEEDED it
(``runs.seeded_pause_after_stages``) AND no longer declares it a gate; user
picks and legacy (NULL-provenance) rows are kept unchanged.

The runs are created DB-direct and the dispatcher's ``dispatch`` is stubbed, so
no executor races the assertions (see ``test_runs._make_run_row``).
"""

from __future__ import annotations

from uuid import UUID

import pytest
from sqlalchemy import select, update

from src.database.models import Run
from src.database.repositories.run_repository import RunRepository
from src.services.blueprints.registry import get_registry

pytestmark = pytest.mark.asyncio


async def _make_deliverable(client) -> str:
    r = await client.post(
        "/api/projects",
        json={
            "type": "generic",
            "title": "resume gates",
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


async def _make_paused_run(
    client,
    db_session_factory,
    *,
    blueprint_id: str,
    pauses: list[str],
    seeded: list[str] | None,
) -> tuple[str, UUID]:
    deliv = await _make_deliverable(client)
    async with db_session_factory() as s:
        run = await RunRepository(s).create(
            deliverable_id=UUID(deliv),
            blueprint_id=blueprint_id,
            config_snapshot={},
            pause_after_stages=pauses,
            seeded_pause_after_stages=seeded,
        )
        await s.commit()
        run_uuid = run.id
        await s.execute(update(Run).where(Run.id == run_uuid).values(status="paused"))
        await s.commit()
    return str(run_uuid), run_uuid


def _stub_dispatch(app) -> list[UUID]:
    """Record dispatches instead of executing them.

    Two dispatch paths exist: resume calls ``app.state.dispatcher.dispatch`` at
    request time, but run creation and regenerate go through ``RunEngine``, which
    captured ``dispatcher.dispatch`` as a bound method when the test app was built.
    Stubbing only the dispatcher left POST /runs and /regenerate starting a REAL
    run whose ``script_draft`` stage called worker_management -- from an in-cluster
    CI runner, that is PRODUCTION worker_management.
    """
    calls: list[UUID] = []

    async def _dispatch(run_id) -> None:
        calls.append(run_id)

    app.state.dispatcher.dispatch = _dispatch
    app.state.run_engine._dispatch = _dispatch
    return calls


async def _row_pauses(db_session_factory, run_uuid: UUID) -> list[str]:
    async with db_session_factory() as s:
        row = (await s.execute(select(Run).where(Run.id == run_uuid))).scalar_one()
    return list(row.pause_after_stages or [])


def _generic_gates() -> set[str]:
    gates = {
        s.name for s in get_registry().resolve_by_id("generic_v1").stages if s.default_pause_after
    }
    # Preconditions -- if the blueprint changes, fail loudly rather than vacuously.
    assert "storyboard_draft" in gates
    assert "narrate" not in gates, "narrate stands in for a REMOVED gate / non-gate stage"
    return gates


async def _resume(client, rid: str) -> None:
    resp = await client.post(f"/api/runs/{rid}/resume")
    assert resp.status_code == 204, resp.text


async def test_resume_drops_seeded_gate_the_blueprint_no_longer_declares(
    client, app, db_session_factory
) -> None:
    _generic_gates()
    # Seeded with storyboard_draft (still a gate) and narrate (a gate since removed).
    rid, run_uuid = await _make_paused_run(
        client,
        db_session_factory,
        blueprint_id="generic_v1",
        pauses=["storyboard_draft", "narrate"],
        seeded=["storyboard_draft", "narrate"],
    )
    calls = _stub_dispatch(app)
    await _resume(client, rid)
    assert calls == [run_uuid]
    pauses = await _row_pauses(db_session_factory, run_uuid)
    assert pauses == ["storyboard_draft"], (
        f"stale seeded gate dropped, live gate kept; got {pauses}"
    )


async def test_resume_keeps_user_picked_non_gate_checkpoint(
    client, app, db_session_factory
) -> None:
    _generic_gates()
    # narrate is NOT a gate but the USER chose it (not in the seeded column).
    rid, run_uuid = await _make_paused_run(
        client,
        db_session_factory,
        blueprint_id="generic_v1",
        pauses=["storyboard_draft", "narrate"],
        seeded=["storyboard_draft"],
    )
    _stub_dispatch(app)
    await _resume(client, rid)
    assert await _row_pauses(db_session_factory, run_uuid) == ["storyboard_draft", "narrate"]


async def test_resume_legacy_null_provenance_keeps_row_unchanged(
    client, app, db_session_factory
) -> None:
    _generic_gates()
    rid, run_uuid = await _make_paused_run(
        client, db_session_factory, blueprint_id="generic_v1", pauses=["narrate"], seeded=None
    )
    _stub_dispatch(app)
    await _resume(client, rid)
    assert await _row_pauses(db_session_factory, run_uuid) == ["narrate"]


async def test_resume_keeps_row_when_blueprint_unknown(client, app, db_session_factory) -> None:
    """A retired blueprint_id fails SAFE: the row's gates are kept unchanged."""
    rid, run_uuid = await _make_paused_run(
        client,
        db_session_factory,
        blueprint_id="retired_bp_v0",
        pauses=["narrate"],
        seeded=["narrate"],
    )
    _stub_dispatch(app)
    await _resume(client, rid)
    assert await _row_pauses(db_session_factory, run_uuid) == ["narrate"]


async def _seeded_column(db_session_factory, rid: str) -> list[str] | None:
    async with db_session_factory() as s:
        row = (await s.execute(select(Run).where(Run.id == UUID(rid)))).scalar_one()
    return row.seeded_pause_after_stages


async def test_creation_records_blueprint_seed_but_not_explicit_list(
    client, app, db_session_factory
) -> None:
    gates = _generic_gates()
    deliv = await _make_deliverable(client)
    # Only the creation record is asserted on -- never execute the runs.
    calls = _stub_dispatch(app)

    seeded_run = await client.post(f"/api/deliverables/{deliv}/runs", json={})
    assert seeded_run.status_code == 201, seeded_run.text
    assert set(await _seeded_column(db_session_factory, seeded_run.json()["id"]) or []) == gates

    explicit = await client.post(
        f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": ["storyboard_draft"]}
    )
    assert explicit.status_code == 201, explicit.text
    assert await _seeded_column(db_session_factory, explicit.json()["id"]) == [], (
        "an explicit (user) checkpoint list must not be recorded as blueprint-seeded"
    )
    assert [str(c) for c in calls] == [seeded_run.json()["id"], explicit.json()["id"]]


async def test_seeded_gate_resent_via_patch_becomes_user_owned(
    client, app, db_session_factory
) -> None:
    """A seeded gate the user explicitly re-sends via PATCH is user-owned: when the
    blueprint later stops declaring it, resume must still keep it."""
    _generic_gates()
    # narrate was seeded (a since-removed gate); the user then re-sends it.
    rid, run_uuid = await _make_paused_run(
        client,
        db_session_factory,
        blueprint_id="generic_v1",
        pauses=["storyboard_draft", "narrate"],
        seeded=["storyboard_draft", "narrate"],
    )
    patch = await client.patch(
        f"/api/runs/{rid}/checkpoints", json={"pause_after_stages": ["narrate"]}
    )
    assert patch.status_code == 200, patch.text
    assert await _seeded_column(db_session_factory, rid) == ["storyboard_draft"]

    _stub_dispatch(app)
    await _resume(client, rid)
    assert await _row_pauses(db_session_factory, run_uuid) == ["narrate"]


async def test_patch_keeps_legacy_null_provenance_null(client, db_session_factory) -> None:
    rid, _ = await _make_paused_run(
        client, db_session_factory, blueprint_id="generic_v1", pauses=["narrate"], seeded=None
    )
    patch = await client.patch(
        f"/api/runs/{rid}/checkpoints", json={"pause_after_stages": ["narrate"]}
    )
    assert patch.status_code == 200, patch.text
    assert await _seeded_column(db_session_factory, rid) is None


async def test_regenerate_child_inherits_provenance_not_the_whole_list(
    client, app, db_session_factory
) -> None:
    """A regenerate child's seeded set is the parent's seeded ∩ what it inherits —
    NOT its whole inherited list. Otherwise the parent's USER-picked checkpoint is
    recorded as blueprint-seeded on the child and a later blueprint change drops it
    on resume."""
    _generic_gates()
    # storyboard_draft: a blueprint-seeded gate. narrate: a USER pick (not seeded).
    rid, _ = await _make_paused_run(
        client,
        db_session_factory,
        blueprint_id="generic_v1",
        pauses=["storyboard_draft", "narrate"],
        seeded=["storyboard_draft"],
    )
    calls = _stub_dispatch(app)
    resp = await client.post(f"/api/runs/{rid}/regenerate", json={})
    assert resp.status_code == 201, resp.text
    child_id = resp.json()["id"]
    assert [str(c) for c in calls] == [child_id], "the child must be dispatched via the stub"

    async with db_session_factory() as s:
        child = (await s.execute(select(Run).where(Run.id == UUID(child_id)))).scalar_one()
    # Precondition: the child carries BOTH checkpoints over, so the seeded set
    # is the only thing that tells them apart.
    assert set(child.pause_after_stages or []) >= {"storyboard_draft", "narrate"}
    assert child.seeded_pause_after_stages == ["storyboard_draft"], (
        "the parent's user-picked 'narrate' must not become blueprint-seeded on the child; "
        f"got {child.seeded_pause_after_stages}"
    )
