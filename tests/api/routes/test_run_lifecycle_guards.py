"""Lifecycle/edit guards on the runs surface.

Every test here reproduces a way the API could spend money twice, destroy a
delivered artifact, or silently drop a human's edit — all of which answered 2xx
before the guard existed.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from src.database.repositories.run_repository import RunRepository
from tests.api.routes.test_runs import (
    _make_project,
    _seed_completed_spine,
    _seed_run_with_full_dag,
    _seed_stage_with_elements,
    _stop_run,
)

pytestmark = pytest.mark.asyncio


# ─── #2 regenerate spawns PAID children — needs a parent gate + a dedupe ───


async def test_regenerate_409s_on_a_running_parent(client, db_session) -> None:
    """Every sibling spawn route 409s on an active parent; regenerate did not,
    so a POST while the parent was still executing created and DISPATCHED a
    second full paid run alongside it."""
    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(db_session, deliv, "storyboard_draft", [{"id": "s1"}])
    await RunRepository(db_session).mark_started(UUID(rid))
    await db_session.commit()

    r = await client.post(f"/api/runs/{rid}/regenerate", json={})
    assert r.status_code == 409, r.text


async def test_regenerate_409s_on_a_pending_parent(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(db_session, deliv, "storyboard_draft", [{"id": "s1"}])
    r = await client.post(f"/api/runs/{rid}/regenerate", json={})
    assert r.status_code == 409, r.text


async def test_regenerate_double_post_spawns_only_one_child(client, db_session) -> None:
    """A double-submit used to create TWO paid child runs off one parent —
    nothing deduped, and ``run_engine.start_run`` re-checks nothing."""
    _, deliv = await _make_project(client)
    rid = await _seed_completed_spine(db_session, deliv, [{"id": "s1"}, {"id": "s2"}])

    first = await client.post(f"/api/runs/{rid}/regenerate", json={})
    assert first.status_code == 201, first.text
    second = await client.post(f"/api/runs/{rid}/regenerate", json={})
    assert second.status_code == 409, second.text

    children = await RunRepository(db_session).list_active_children(UUID(rid))
    assert len(children) == 1, f"a double-POST spawned {len(children)} paid child runs"


# ─── #4 restamping a completed run destroys the delivered output ───────────


async def test_restamp_409s_on_a_completed_run(client, db_session) -> None:
    """A restamp resets every downstream stage — the terminal compile stage
    included — to pending/output=NULL while ``runs.status`` stays ``completed``.
    Nothing re-runs a terminal run, so on a delivered video that is an
    irrecoverable delete of the delivered artifact."""
    _, deliv = await _make_project(client)
    rid = await _seed_run_with_full_dag(
        db_session, deliv, [{"id": "L1", "text": "old", "seconds": 5.0}]
    )
    await RunRepository(db_session).mark_completed(UUID(rid))
    await db_session.commit()

    r = await client.post(
        f"/api/runs/{rid}/stages/script_draft/elements/L1/restamp",
        json={"value": {"text": "new", "seconds": 7.0}},
    )
    assert r.status_code == 409, r.text

    # ...and the delivered output is still there.
    detail = await client.get(f"/api/runs/{rid}")
    by_stage = {s["stage_name"]: s for s in detail.json()["stage_outputs"]}
    assert by_stage["video_render"]["status"] == "completed"
    assert by_stage["video_render"]["output"] is not None


# ─── #7 a patch to a non-terminal stage is clobbered by the executor ───────


async def test_patch_element_409s_on_a_running_run(client, db_session) -> None:
    """``restamp`` had this 409; ``patch`` did not. On a running run a
    ``pending``/``running`` stage row (a resumable-timeout reset PRESERVES
    ``output``) does not fire the 0063 write-once trigger, and the executor's
    ``mark_completed`` then overwrites the edit — 200, edit gone."""
    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session, deliv, "script_draft", [{"id": "L1", "text": "old"}]
    )
    await RunRepository(db_session).mark_started(UUID(rid))
    await db_session.commit()

    r = await client.patch(
        f"/api/runs/{rid}/stages/script_draft/elements/L1",
        json={"patch": {"text": "human edit"}},
    )
    assert r.status_code == 409, r.text


# ─── #6 concurrent Studio edits silently lose one ─────────────────────────


async def test_concurrent_element_patches_do_not_lose_an_edit(
    client, db_session, monkeypatch
) -> None:
    """Two PATCHes of two DIFFERENT elements of the same stage row.

    The route reads the whole ``output`` JSONB, mutates it in Python and writes
    it back. Without a row lock both requests read the same base document and
    the second write discards the first — 200 on both, one edit silently gone.

    The interleaving is forced, not raced: request A is held at the write step
    while request B runs end-to-end. Unfixed, B commits during that window and A
    then overwrites it. Fixed, B blocks on A's ``SELECT ... FOR UPDATE`` and
    re-reads A's merged document once A commits.
    """
    import src.api.routes.runs.stages as stages_mod

    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session,
        deliv,
        "script_draft",
        [{"id": "L1", "text": "old-1"}, {"id": "L2", "text": "old-2"}],
    )
    await _stop_run(db_session, rid)

    original = stages_mod._allow_terminal_output_edit
    gate = asyncio.Event()
    held = {"once": False}

    async def _gated(session) -> None:
        if not held["once"]:
            held["once"] = True
            await gate.wait()
        await original(session)

    monkeypatch.setattr(stages_mod, "_allow_terminal_output_edit", _gated)

    task_a = asyncio.create_task(
        client.patch(
            f"/api/runs/{rid}/stages/script_draft/elements/L1",
            json={"patch": {"text": "A"}},
        )
    )
    await asyncio.sleep(0.3)  # A has read the row and is parked at the gate.
    task_b = asyncio.create_task(
        client.patch(
            f"/api/runs/{rid}/stages/script_draft/elements/L2",
            json={"patch": {"text": "B"}},
        )
    )
    await asyncio.sleep(0.6)  # Unguarded, B reads-writes-commits inside here.
    gate.set()
    resp_a, resp_b = await asyncio.gather(task_a, task_b)
    assert resp_a.status_code == 200, resp_a.text
    assert resp_b.status_code == 200, resp_b.text

    out = await client.get(f"/api/runs/{rid}/stages/script_draft/output")
    lines = {line["id"]: line["text"] for line in out.json()["output"]["lines"]}
    assert lines == {"L1": "A", "L2": "B"}, f"one concurrent edit was lost: {lines}"


# ─── #8 an approved pause gate must not be re-armed ───────────────────────


async def test_checkpoints_cannot_rearm_an_already_consumed_gate(client, db_session) -> None:
    """``pause_after_stages`` is the durable APPROVAL LEDGER that ``resume``
    drains. An unconditional write here races that CAS: land after it and the
    approved gate is re-armed, so the run re-pauses the instant it resumes."""
    _, deliv = await _make_project(client)
    # Only script_draft has a row, and it is terminal — the gate the reviewer
    # already approved past. storyboard_draft has not run yet.
    rid = await _seed_stage_with_elements(
        db_session, deliv, "script_draft", [{"id": "L1", "text": "x"}]
    )
    r = await client.patch(
        f"/api/runs/{rid}/checkpoints",
        json={"pause_after_stages": ["script_draft", "storyboard_draft"]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["pause_after_stages"] == ["storyboard_draft"], r.text

    # ...and the ROW, not just the response body.
    run = await RunRepository(db_session).get(UUID(rid))
    assert run is not None
    await db_session.refresh(run)
    assert run.pause_after_stages == ["storyboard_draft"]


async def test_checkpoints_409_on_a_terminal_run(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(db_session, deliv, "script_draft", [{"id": "L1"}])
    await RunRepository(db_session).mark_cancelled(UUID(rid))
    await db_session.commit()
    r = await client.patch(
        f"/api/runs/{rid}/checkpoints", json={"pause_after_stages": ["script_draft"]}
    )
    assert r.status_code == 409, r.text


# ─── #5 cancel reports success without cancelling ─────────────────────────


async def test_cancel_still_records_when_the_run_task_handler_raises(
    app, client, db_session
) -> None:
    """Production call site for the dispatcher-cancel fallback.

    ``dispatcher.cancel`` returned True on task EXISTENCE and let anything that
    was not a ``CancelledError`` escape. A DB blip inside the unshielded handler
    therefore 500'd this route: ``mark_cancelled`` never committed, and neither
    ``_sweep_active_descendants`` nor ``_sweep_deliverable_children`` ran — so
    paid course-lesson children kept rendering while the user was told the
    cancel had failed.
    """
    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(db_session, deliv, "script_draft", [{"id": "L1"}])
    await RunRepository(db_session).mark_started(UUID(rid))
    await db_session.commit()

    async def _handler_blows_up() -> None:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            raise RuntimeError("connection reset in the CancelledError handler") from None

    task = asyncio.create_task(_handler_blows_up())
    await asyncio.sleep(0)
    app.state.dispatcher._tasks[UUID(rid)] = task
    try:
        r = await client.post(f"/api/runs/{rid}/cancel")
    finally:
        app.state.dispatcher._tasks.pop(UUID(rid), None)
    assert r.status_code == 204, r.text

    db_session.expire_all()
    run = await RunRepository(db_session).get(UUID(rid))
    assert run is not None
    assert run.status == "cancelled", "the cancel was reported but never recorded"
