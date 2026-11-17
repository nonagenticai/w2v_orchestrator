"""Studio v3 — /api/runs route tests (T13)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from prometheus_client import REGISTRY
from w2v_common.clients import WMResponse

from src.clients.schemas.tts import TTSSynthesis

pytestmark = pytest.mark.asyncio


# Auto-mock both downstream clients for the entire module.
# Without this the dispatcher would call the real services and time out in CI.
@pytest.fixture(autouse=True)
def _mock_stage_clients():
    fake_lr = AsyncMock()
    fake_lr.message.return_value = WMResponse(
        output={"lines": [{"text": "hi", "seconds": 5.0}]},
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )
    fake_render = AsyncMock()
    fake_render.compile.return_value = {
        "artifact_url": "http://a/x.mp4",
        "duration_seconds": 30.0,
    }
    # The narrate stage builds a real TTSClient and calls tts.test, which
    # 401s on a dev box (no license token). Mock the client + the storage write
    # so runs reach a terminal state fast and deterministically.
    fake_tts = AsyncMock()
    fake_tts.synthesize.return_value = TTSSynthesis.model_validate(
        {
            "audio_bytes": b"\x00",
            "format": "wav",
            "duration_seconds": 5.0,
            "sample_rate": 24000,
            "provider": "test",
            "voice": "",
        }
    )
    with (
        patch(
            "src.services.stages.primitives.script_stage.WMClient",
            return_value=fake_lr,
        ),
        patch(
            "src.services.stages.primitives.compile_stage.VideoCompileClient",
            return_value=fake_render,
        ),
        patch(
            "src.services.stages.primitives.narration.TTSClient",
            return_value=fake_tts,
        ),
        patch(
            "src.services.stages.primitives.narration.store_tts_clip",
            new=AsyncMock(return_value="http://a/clip.wav"),
        ),
    ):
        yield


async def _make_project(client) -> tuple[str, str]:
    r = await client.post(
        "/api/projects",
        json={
            "type": "generic",
            "title": "for run",
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


async def test_post_deliverable_runs_creates_pending(client) -> None:
    pid, deliv = await _make_project(client)
    r = await client.post(f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": []})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "pending"
    assert body["blueprint_id"] == "generic_v1"
    assert body["project_id"] == pid
    assert body["deliverable_id"] == deliv


async def test_post_deliverable_runs_honours_camelcase_pause_after_stages(client) -> None:
    """``RunStart`` had no camelCase alias, so a camelCase autopilot request

    ``{"pauseAfterStages": []}`` was silently DROPPED to the ``None`` default —
    which re-seeds the blueprint's review-gate pauses instead of the caller's
    explicit autopilot. Same hole, same fix, as ``RegenerateRequest.from_stage``
    (#1027)."""
    _, deliv = await _make_project(client)
    r = await client.post(
        f"/api/deliverables/{deliv}/runs",
        json={"pauseAfterStages": [], "configOverrides": {"x": 1}},
    )
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    cp = await client.get(f"/api/runs/{rid}/checkpoints")
    assert cp.status_code == 200, cp.text
    assert cp.json()["pause_after_stages"] == []


async def test_get_run_detail(client) -> None:
    pid, deliv = await _make_project(client)
    r = await client.post(f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": []})
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    detail = await client.get(f"/api/runs/{rid}")
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["id"] == rid
    assert body["project_id"] == pid
    assert isinstance(body["stage_outputs"], list)
    assert body["usage_total"]["input_tokens"] == 0


async def test_get_run_detail_503_when_db_work_times_out(client, monkeypatch) -> None:
    """Regression for issue #949 (defect B): a dead/unreachable Postgres must
    surface as a FAST, clearly-labelled 503 rather than hanging until the
    caller's own client-side timeout (measured live at 120262.6ms during the
    2026-09-04 worker4 reboot). Proves the actual `asyncio.wait_for` wiring —
    not a mocked bypass — by shrinking the real timeout constant and making the
    real DB-work coroutine outlive it, then asserting BOTH the status/body AND
    that this test itself did not take anywhere near the old 120s."""
    import time as time_module

    from src.api.routes.runs import read as read_module

    _pid, deliv = await _make_project(client)
    r = await client.post(f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": []})
    assert r.status_code == 201, r.text
    rid = r.json()["id"]

    monkeypatch.setattr(read_module, "_RUN_DETAIL_DB_TIMEOUT_SECONDS", 0.05)

    async def _hung_get_run_detail(*_args, **_kwargs):
        # Stands in for a handler blocked on a dead backend's connect/login
        # retry loop (the psycopg `server_login_retry` case from #949) — the
        # coroutine simply never finishes inside the (shrunk) budget.
        await asyncio.sleep(5.0)
        raise AssertionError("should have been cancelled by asyncio.wait_for")

    monkeypatch.setattr(read_module, "_get_run_detail", _hung_get_run_detail)

    started = time_module.perf_counter()
    detail = await client.get(f"/api/runs/{rid}")
    elapsed = time_module.perf_counter() - started

    assert detail.status_code == 503, detail.text
    assert "database did not respond in time" in detail.json()["detail"]
    assert elapsed < 2.0, f"handler took {elapsed:.2f}s — the timeout did not bound it"


async def test_get_run_detail_503_on_db_operational_error(client, monkeypatch) -> None:
    """A DB-level failure that raises immediately (rather than hanging) must
    ALSO map to 503, not an opaque 500 — the other half of #949's fix."""
    from sqlalchemy.exc import OperationalError

    from src.api.routes.runs import read as read_module

    _pid, deliv = await _make_project(client)
    r = await client.post(f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": []})
    assert r.status_code == 201, r.text
    rid = r.json()["id"]

    async def _raise_operational_error(*_args, **_kwargs):
        raise OperationalError("SELECT 1", {}, Exception("server login has been failing"))

    monkeypatch.setattr(read_module, "_get_run_detail", _raise_operational_error)

    detail = await client.get(f"/api/runs/{rid}")
    assert detail.status_code == 503, detail.text
    assert "database error" in detail.json()["detail"]


_PHASES = ("ownership", "stage_query", "other_queries", "response_build")


def _phase_counts() -> dict[str, float]:
    """Current observation count per phase of the run-detail histogram.

    prometheus_client registers on a process-global REGISTRY, so these counters
    carry whatever every earlier test in this process already observed. Callers
    MUST diff two snapshots — never assume a zero baseline. A phase whose child
    has not been labelled yet is absent from the registry (``None``) → 0.
    """
    name = "w2v_orchestrator_run_detail_phase_seconds_count"
    return {p: REGISTRY.get_sample_value(name, {"phase": p}) or 0.0 for p in _PHASES}


async def test_get_run_detail_observes_each_phase_once(client) -> None:
    """GET /api/runs/{id} records every phase of its own body exactly once.

    The phases must partition the handler for the 524-investigation subtraction
    (middleware total - sum of phases = post-handler response_model + encode cost) to
    mean anything: a phase observed twice, or not at all, silently corrupts that
    residual.
    """
    _, deliv = await _make_project(client)
    r = await client.post(f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": []})
    assert r.status_code == 201, r.text
    rid = r.json()["id"]

    before = _phase_counts()
    before_stages = REGISTRY.get_sample_value("w2v_orchestrator_run_detail_stages_count", {}) or 0.0

    detail = await client.get(f"/api/runs/{rid}")
    assert detail.status_code == 200, detail.text

    after = _phase_counts()
    for phase in _PHASES:
        assert after[phase] - before[phase] == 1, (
            f"phase {phase!r} observed {after[phase] - before[phase]} times, expected 1"
        )

    after_stages = REGISTRY.get_sample_value("w2v_orchestrator_run_detail_stages_count", {}) or 0.0
    assert after_stages - before_stages == 1


async def test_cancel_run(client) -> None:
    _, deliv = await _make_project(client)
    r = await client.post(f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": []})
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    cancel = await client.post(f"/api/runs/{rid}/cancel")
    assert cancel.status_code == 204
    detail = await client.get(f"/api/runs/{rid}")
    # Cancel may land before/after dispatcher starts; all terminal + in-flight states valid.
    assert detail.json()["status"] in ("cancelled", "completed", "running", "pending", "failed")


async def test_ancestors_empty_chain(client) -> None:
    """First run of a deliverable has no parent_run_id → empty ancestors."""
    _, deliv = await _make_project(client)
    r = await client.post(f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": []})
    rid = r.json()["id"]
    res = await client.get(f"/api/runs/{rid}/ancestors")
    assert res.status_code == 200, res.text
    assert res.json() == []


async def test_ancestors_multi_level_chain(client) -> None:
    """Walk a 3-level parent chain; oldest-first ordering. Each parent must
    have stopped (paused/completed/failed/cancelled) before the engine will
    chain a child off it."""
    _, deliv = await _make_project(client)

    async def _new_and_wait_terminal(parent_id: str | None = None) -> str:
        payload: dict = {"pause_after_stages": []}
        if parent_id is not None:
            payload["parent_run_id"] = parent_id
        resp = await client.post(f"/api/deliverables/{deliv}/runs", json=payload)
        assert resp.status_code == 201, resp.text
        rid = resp.json()["id"]
        for _ in range(50):
            await asyncio.sleep(0.1)
            d = await client.get(f"/api/runs/{rid}")
            if d.json()["status"] in ("paused", "completed", "failed"):
                return rid
        raise AssertionError(f"run {rid} never reached terminal state")

    rid1 = await _new_and_wait_terminal()
    rid2 = await _new_and_wait_terminal(rid1)
    rid3 = await _new_and_wait_terminal(rid2)

    res = await client.get(f"/api/runs/{rid3}/ancestors")
    assert res.status_code == 200, res.text
    rows = res.json()
    ids = [row["id"] for row in rows]
    assert ids == [rid1, rid2], "expected oldest-first ordering (root → immediate parent)"
    assert all(row["project_id"] for row in rows), "ancestors must carry project_id"


async def test_ancestors_cross_tenant_returns_404(client, other_tenant_client) -> None:
    """Tenant B cannot read tenant A's run ancestors."""
    _, deliv = await _make_project(client)
    r = await client.post(f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": []})
    rid = r.json()["id"]
    bleed = await other_tenant_client.get(f"/api/runs/{rid}/ancestors")
    assert bleed.status_code == 404


async def test_pause_endpoint_commits_paused_and_cancels_dispatcher(
    client, app, db_session_factory
) -> None:
    """POST /runs/{id}/pause must persist status=paused AND cancel the in-flight task.

    Regression: previously the handler only called ``mark_paused`` without
    committing and without touching the dispatcher — the row never flipped
    in the DB and the run kept executing past the pause point.
    """
    from uuid import UUID

    from sqlalchemy import select

    from src.database.models import Run
    from src.database.repositories.run_repository import RunRepository

    _, deliv = await _make_project(client)
    create = await client.post(f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": []})
    assert create.status_code == 201
    rid = create.json()["id"]

    # Patch the dispatcher's ``cancel`` to a deterministic no-op so we can
    # assert it was awaited without racing the real DAG executor.
    cancel_calls: list[str] = []

    async def _capture_cancel(run_id) -> bool:
        cancel_calls.append(str(run_id))
        return True

    app.state.dispatcher.cancel = _capture_cancel

    # Force the row into a pause-able status. The real dispatcher might
    # already have moved it to running/completed; we replace that with
    # ``running`` so the endpoint's 409 guard accepts our request.
    run_uuid = UUID(rid)
    async with db_session_factory() as session:
        await RunRepository(session).mark_started(run_uuid)
        await session.commit()

    resp = await client.post(f"/api/runs/{rid}/pause")
    assert resp.status_code == 204, resp.text
    assert cancel_calls == [rid], (
        f"pause endpoint must call dispatcher.cancel exactly once; saw {cancel_calls}"
    )

    # Status must have been persisted (committed) to the DB.
    async with db_session_factory() as session:
        row = (await session.execute(select(Run).where(Run.id == run_uuid))).scalar_one_or_none()
    assert row is not None
    assert row.status == "paused", f"expected paused; got {row.status!r}"
    # A run someone PAUSED is not waiting for a review: it must say so, on the row
    # and on the wire, or every surface calls it "needs review" (0102).
    assert row.pause_reason == "manual"
    lite = (await client.get(f"/api/runs/{rid}/status")).json()
    assert lite["status"] == "paused" and lite["pause_reason"] == "manual"


async def test_pause_resume(client) -> None:
    _, deliv = await _make_project(client)
    r = await client.post(
        f"/api/deliverables/{deliv}/runs",
        json={"pause_after_stages": ["script_draft"]},
    )
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    for _ in range(50):
        await asyncio.sleep(0.2)
        detail = await client.get(f"/api/runs/{rid}")
        if detail.json()["status"] == "paused":
            break
    resume = await client.post(f"/api/runs/{rid}/resume")
    assert resume.status_code == 204
    # A1: resuming approves the consumed checkpoint — it must be stripped from
    # pause_after_stages so a later reconciler re-dispatch (drain mid-downstream)
    # does not re-pause at an already-approved gate.
    cp = await client.get(f"/api/runs/{rid}/checkpoints")
    assert cp.status_code == 200, cp.text
    assert "script_draft" not in cp.json()["pause_after_stages"], (
        f"resume must clear the consumed checkpoint; got {cp.json()['pause_after_stages']}"
    )


async def _make_run_row(client, db_session_factory, *, status: str, updated_age_seconds: int = 0):
    """Create a run DIRECTLY in the DB (no POST → no engine execution race) under
    a project owned by ``client``'s tenant, forced to ``status`` with an optional
    backdated ``updated_at``. Returns (rid_str, run_uuid).

    The conftest RunEngine captures ``dispatcher.dispatch`` as a bound method at
    app-build time, so reassigning ``app.state.dispatcher`` cannot neutralize a
    real run started via POST — only DB-direct creation is deterministic.
    """
    import datetime
    from uuid import UUID

    from sqlalchemy import update

    from src.database.models import Run
    from src.database.repositories.run_repository import RunRepository

    _, deliv = await _make_project(client)
    async with db_session_factory() as s:
        run = await RunRepository(s).create(
            deliverable_id=UUID(deliv),
            blueprint_id="generic_v1",
            config_snapshot={},
            pause_after_stages=[],
        )
        await s.commit()
        run_uuid = run.id
        values: dict = {"status": status}
        if updated_age_seconds:
            values["updated_at"] = datetime.datetime.now(datetime.UTC) - datetime.timedelta(
                seconds=updated_age_seconds
            )
        await s.execute(update(Run).where(Run.id == run_uuid).values(**values))
        await s.commit()
    return str(run_uuid), run_uuid


async def test_control_routes_503_when_dispatcher_down(client, app, db_session_factory) -> None:
    """cancel/pause raise a clean 503 (not an opaque 500 AttributeError) when the
    dispatcher never started (Redis down at startup). cancel still records the
    cancelled row so the run is not wedged `running`; pause does not half-mutate."""
    from sqlalchemy import select

    from src.database.models import Run

    rid, run_uuid = await _make_run_row(client, db_session_factory, status="running")

    # Degraded mode: Redis down at startup → asgi sets dispatcher = None.
    app.state.dispatcher = None

    pause = await client.post(f"/api/runs/{rid}/pause")
    assert pause.status_code == 503, pause.text
    async with db_session_factory() as s:
        row = (await s.execute(select(Run).where(Run.id == run_uuid))).scalar_one()
    assert row.status != "paused", "pause must not half-mutate when dispatcher is down"

    cancel = await client.post(f"/api/runs/{rid}/cancel")
    assert cancel.status_code == 503, cancel.text
    async with db_session_factory() as s:
        row = (await s.execute(select(Run).where(Run.id == run_uuid))).scalar_one()
    assert row.status == "cancelled", f"cancel must record the row even degraded; got {row.status}"


async def test_resume_503_when_dispatcher_down(client, app, db_session_factory) -> None:
    """resume raises 503 and leaves the run `paused` (never marks it resumed with
    no dispatcher to actually re-dispatch — that would wedge it)."""
    from sqlalchemy import select

    from src.database.models import Run

    rid, run_uuid = await _make_run_row(client, db_session_factory, status="paused")

    app.state.dispatcher = None
    resume = await client.post(f"/api/runs/{rid}/resume")
    assert resume.status_code == 503, resume.text
    async with db_session_factory() as s:
        row = (await s.execute(select(Run).where(Run.id == run_uuid))).scalar_one()
    assert row.status == "paused", "resume must leave it paused when dispatcher down"


# --- cancel cascade (structural-relief child runs) -------------------------
#
# A cant_converge parent spawns a bounded CHILD render (records parent_run_id).
# Cancelling the parent must cascade to every ACTIVE descendant so a child (or
# grandchild) render can't outlive the cancel as an orphaned PAID compile.


async def _seed_run(session, deliverable_uuid, *, status: str, parent_run_id=None):
    """Create a run DIRECTLY in the DB under ``deliverable_uuid``, forced to
    ``status`` and optionally linked to ``parent_run_id``. Returns the run id."""
    from sqlalchemy import update

    from src.database.models import Run
    from src.database.repositories.run_repository import RunRepository

    run = await RunRepository(session).create(
        deliverable_id=deliverable_uuid,
        blueprint_id="generic_v1",
        config_snapshot={},
        pause_after_stages=[],
        parent_run_id=parent_run_id,
    )
    await session.execute(update(Run).where(Run.id == run.id).values(status=status))
    return run.id


def _force_dispatcher_fallback(app) -> None:
    """Make ``dispatcher.cancel`` report 'no local task' so the cancel route
    takes the deterministic mark_cancelled fallback for every run it touches,
    and stub ``broadcast_interrupt`` to a harmless no-op."""

    async def _cancel(run_id) -> bool:
        return False

    app.state.dispatcher.cancel = _cancel
    app.state.dispatcher.broadcast_interrupt = AsyncMock()


async def _statuses(db_session_factory, ids) -> dict:
    from sqlalchemy import select

    from src.database.models import Run

    async with db_session_factory() as s:
        rows = (await s.execute(select(Run).where(Run.id.in_(ids)))).scalars().all()
    return {r.id: r.status for r in rows}


async def test_cancel_cascades_to_running_child(client, app, db_session_factory) -> None:
    """Parent running + one running child → BOTH cancelled."""
    from uuid import UUID

    _, deliv = await _make_project(client)
    deliv_uuid = UUID(deliv)
    async with db_session_factory() as s:
        parent = await _seed_run(s, deliv_uuid, status="running")
        child = await _seed_run(s, deliv_uuid, status="running", parent_run_id=parent)
        await s.commit()

    _force_dispatcher_fallback(app)
    resp = await client.post(f"/api/runs/{parent}/cancel")
    assert resp.status_code == 204, resp.text

    statuses = await _statuses(db_session_factory, [parent, child])
    assert statuses[parent] == "cancelled"
    assert statuses[child] == "cancelled"


async def test_cancel_cascades_from_completed_parent(client, app, db_session_factory) -> None:
    """Completed parent + running child → child cancelled, parent untouched.

    The key regression: a cant_converge parent spawns its bounded relief child
    only AFTER it COMPLETES, so cancelling a `completed` (terminal) parent must
    STILL interrupt the child that is still burning compile time. The old code
    early-returned on a terminal parent and never reached the cascade."""
    from uuid import UUID

    _, deliv = await _make_project(client)
    deliv_uuid = UUID(deliv)
    async with db_session_factory() as s:
        parent = await _seed_run(s, deliv_uuid, status="completed")
        child = await _seed_run(s, deliv_uuid, status="running", parent_run_id=parent)
        await s.commit()

    _force_dispatcher_fallback(app)
    resp = await client.post(f"/api/runs/{parent}/cancel")
    assert resp.status_code == 204, resp.text

    statuses = await _statuses(db_session_factory, [parent, child])
    assert statuses[parent] == "completed", "terminal parent must stay completed"
    assert statuses[child] == "cancelled", "running child of a completed parent must cancel"


async def test_cancel_cascades_through_grandchild_chain(client, app, db_session_factory) -> None:
    """Parent → child → grandchild all active → all cancelled."""
    from uuid import UUID

    _, deliv = await _make_project(client)
    deliv_uuid = UUID(deliv)
    async with db_session_factory() as s:
        parent = await _seed_run(s, deliv_uuid, status="running")
        child = await _seed_run(s, deliv_uuid, status="running", parent_run_id=parent)
        grandchild = await _seed_run(s, deliv_uuid, status="running", parent_run_id=child)
        await s.commit()

    _force_dispatcher_fallback(app)
    resp = await client.post(f"/api/runs/{parent}/cancel")
    assert resp.status_code == 204, resp.text

    statuses = await _statuses(db_session_factory, [parent, child, grandchild])
    assert all(v == "cancelled" for v in statuses.values()), statuses


async def test_cancel_leaves_completed_child_untouched(client, app, db_session_factory) -> None:
    """Only ACTIVE children cascade: a completed child stays completed."""
    from uuid import UUID

    _, deliv = await _make_project(client)
    deliv_uuid = UUID(deliv)
    async with db_session_factory() as s:
        parent = await _seed_run(s, deliv_uuid, status="running")
        done_child = await _seed_run(s, deliv_uuid, status="completed", parent_run_id=parent)
        await s.commit()

    _force_dispatcher_fallback(app)
    resp = await client.post(f"/api/runs/{parent}/cancel")
    assert resp.status_code == 204, resp.text

    statuses = await _statuses(db_session_factory, [parent, done_child])
    assert statuses[parent] == "cancelled"
    assert statuses[done_child] == "completed", "a terminal child must not be re-cancelled"


async def test_cancel_no_children_still_cancels_parent(client, app, db_session_factory) -> None:
    """Parent with no children → behaviour identical to before (parent cancelled)."""
    from uuid import UUID

    _, deliv = await _make_project(client)
    deliv_uuid = UUID(deliv)
    async with db_session_factory() as s:
        parent = await _seed_run(s, deliv_uuid, status="running")
        await s.commit()

    _force_dispatcher_fallback(app)
    resp = await client.post(f"/api/runs/{parent}/cancel")
    assert resp.status_code == 204, resp.text

    statuses = await _statuses(db_session_factory, [parent])
    assert statuses[parent] == "cancelled"


class _CountingBilling:
    """Billing stub returning no cost groups; records each costs_by_correlation call."""

    def __init__(self) -> None:
        self.calls: list[list] = []

    async def costs_by_correlation(self, tenant_id, correlation_ids, *, group_by=None):
        self.calls.append(list(correlation_ids))
        return {"groups": []}


async def test_get_run_zero_cost_fresh_stays_live(client, app, db_session_factory) -> None:
    """A just-terminated zero-cost run stays live: the cost read-through re-queries
    billing on each load (so late-arriving async cost is picked up) and does NOT
    mark the run reconciled while inside the settle window."""
    from sqlalchemy import select

    from src.api.routes._common import get_billing_client
    from src.database.models import Run

    billing = _CountingBilling()
    app.dependency_overrides[get_billing_client] = lambda: billing
    rid, run_uuid = await _make_run_row(
        client, db_session_factory, status="completed", updated_age_seconds=0
    )

    assert (await client.get(f"/api/runs/{rid}")).status_code == 200
    assert (await client.get(f"/api/runs/{rid}")).status_code == 200
    assert len(billing.calls) == 2, "fresh terminal run must stay live (re-query each load)"
    async with db_session_factory() as s:
        marked = (
            await s.execute(select(Run.usage_reconciled_at).where(Run.id == run_uuid))
        ).scalar_one()
    assert marked is None, "fresh run must not be marked reconciled"
    app.dependency_overrides.pop(get_billing_client, None)


async def test_get_run_zero_cost_settled_marks_and_stops_requerying(
    client, app, db_session_factory
) -> None:
    """A settled (past-freshness) zero-cost run is marked reconciled on first load
    and never re-queries billing again — the hot-path amplification fix."""
    from sqlalchemy import select

    from src.api.routes._common import get_billing_client
    from src.database.models import Run

    billing = _CountingBilling()
    app.dependency_overrides[get_billing_client] = lambda: billing
    rid, run_uuid = await _make_run_row(
        client, db_session_factory, status="completed", updated_age_seconds=120
    )

    first = await client.get(f"/api/runs/{rid}")
    assert first.status_code == 200
    assert first.json()["usage_total"]["cost_settled"] is True
    assert len(billing.calls) == 1, "settled run queries billing once"
    async with db_session_factory() as s:
        marked = (
            await s.execute(select(Run.usage_reconciled_at).where(Run.id == run_uuid))
        ).scalar_one()
    assert marked is not None, "settled run must be marked reconciled"

    assert (await client.get(f"/api/runs/{rid}")).status_code == 200
    assert len(billing.calls) == 1, "marked run must NOT re-query billing on subsequent loads"
    app.dependency_overrides.pop(get_billing_client, None)


async def test_get_run_cancelled_run_mirrors_cost(client, app, db_session_factory) -> None:
    """A cancelled run's pre-cancel spend is read through from billing like a
    completed/failed run's (it used to be skipped, so it showed $0 forever)."""
    from src.api.routes._common import get_billing_client

    billing = _CountingBilling()
    app.dependency_overrides[get_billing_client] = lambda: billing
    rid, _ = await _make_run_row(
        client, db_session_factory, status="cancelled", updated_age_seconds=120
    )
    assert (await client.get(f"/api/runs/{rid}")).status_code == 200
    assert len(billing.calls) == 1, "a cancelled run's cost must be read from billing"
    app.dependency_overrides.pop(get_billing_client, None)


async def test_get_run_live_run_does_not_read_billing(client, app, db_session_factory) -> None:
    """A running run's cost is not final — the read never mirrors it."""
    from src.api.routes._common import get_billing_client

    billing = _CountingBilling()
    app.dependency_overrides[get_billing_client] = lambda: billing
    rid, _ = await _make_run_row(client, db_session_factory, status="running")
    r = await client.get(f"/api/runs/{rid}")
    assert r.status_code == 200
    assert r.json()["usage_total"]["cost_settled"] is False
    assert billing.calls == []
    app.dependency_overrides.pop(get_billing_client, None)


class _ScriptedBilling:
    """Billing stub answering each costs_by_correlation call with the next scripted
    list of (stage, cost_usd) groups for the requested run(s)."""

    def __init__(self, *answers: list[tuple[str, str]]) -> None:
        self.answers = list(answers)
        self.calls = 0

    async def costs_by_correlation(self, tenant_id, correlation_ids, *, group_by=None):
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return {
            "groups": [
                {
                    "key": {
                        "correlation_id": str(cid),
                        "stage": stage,
                        "model": "m",
                        "provider": "p",
                    },
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cost_usd": cost,
                }
                for cid in correlation_ids
                for stage, cost in answer
            ]
        }


async def _backdate(db_session_factory, run_uuid, seconds: int) -> None:
    import datetime

    from sqlalchemy import update

    from src.database.models import Run

    async with db_session_factory() as s:
        await s.execute(
            update(Run)
            .where(Run.id == run_uuid)
            .values(
                updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=seconds)
            )
        )
        await s.commit()


async def test_get_run_partial_first_mirror_converges_and_then_settles(
    client, app, db_session_factory
) -> None:
    """The first read after a run ends can catch only PART of billing's async cost
    events (likeliest on a mid-stage cancel). It used to be final: a mirrored row
    blocked every later re-read and the sweep skipped the run, so the cost stayed
    partial forever. Now each read REPLACES the mirror until the run is past the
    settle window, where it is marked settled and stops re-reading."""
    from src.api.routes._common import get_billing_client

    billing = _ScriptedBilling(
        [("promo_script", "0.01")],
        [("promo_script", "0.01"), ("promo_review", "0.02")],
        [("promo_script", "0.01"), ("promo_review", "0.02"), ("growth_bundle", "0.03")],
    )
    app.dependency_overrides[get_billing_client] = lambda: billing
    rid, run_uuid = await _make_run_row(client, db_session_factory, status="cancelled")

    first = (await client.get(f"/api/runs/{rid}")).json()["usage_total"]
    assert first["cost_usd"] == 0.01
    assert first["cost_settled"] is False, "inside the settle window the cost is not final"

    second = (await client.get(f"/api/runs/{rid}")).json()["usage_total"]
    assert second["cost_usd"] == 0.03, "a partial first mirror must not freeze the cost"
    assert second["cost_settled"] is False

    await _backdate(db_session_factory, run_uuid, 4000)
    third = (await client.get(f"/api/runs/{rid}")).json()["usage_total"]
    assert third["cost_usd"] == 0.06
    assert third["cost_settled"] is True
    assert len(third["by_stage"]) == 3, "the mirror is replaced, never appended"

    calls = billing.calls
    fourth = (await client.get(f"/api/runs/{rid}")).json()["usage_total"]
    assert billing.calls == calls, "a settled run is a pure local read"
    assert fourth["cost_usd"] == 0.06 and fourth["cost_settled"] is True
    app.dependency_overrides.pop(get_billing_client, None)


async def test_concurrent_reads_of_a_finished_run_settle_at_the_true_cost(
    client, app, db_session_factory
) -> None:
    """The auditor's probe: three simultaneous run-detail GETs on a finished run
    past its settle window, billing answering in 0.3s, true cost $0.03. The mirror
    is delete-then-insert, and unserialised the three interleaved and settled the
    run at $0.09 / $0.09 / $0.06 — frozen, because settled runs are never re-read.
    Now one read wins the try-lock and settles $0.03; a contended read is skipped
    (never waits) and never writes, so the stored rows total exactly $0.03."""
    import asyncio

    from sqlalchemy import func, select

    from src.api.routes._common import get_billing_client
    from src.database.models import RunUsage

    class _SlowBilling(_ScriptedBilling):
        async def costs_by_correlation(self, tenant_id, correlation_ids, *, group_by=None):
            await asyncio.sleep(0.3)
            return await super().costs_by_correlation(tenant_id, correlation_ids, group_by=group_by)

    billing = _SlowBilling([("a", "0.01"), ("b", "0.01"), ("c", "0.01")])
    app.dependency_overrides[get_billing_client] = lambda: billing
    try:
        rid, run_uuid = await _make_run_row(
            client, db_session_factory, status="completed", updated_age_seconds=120
        )
        responses = await asyncio.gather(*(client.get(f"/api/runs/{rid}") for _ in range(3)))
        totals = [r.json()["usage_total"] for r in responses]
        assert all(r.status_code == 200 for r in responses)
        # A read that loses the try-lock still read billing but writes nothing,
        # so it reports the stored cost (0 or the settled 0.03) as unsettled.
        assert all(t["cost_usd"] in (0, 0.03) for t in totals), totals
        assert any(t["cost_settled"] for t in totals), totals
        async with db_session_factory() as s:
            stored = (
                await s.execute(
                    select(func.count(), func.sum(RunUsage.cost_usd)).where(
                        RunUsage.run_id == run_uuid
                    )
                )
            ).one()
        assert stored[0] == 3 and float(stored[1]) == 0.03
        assert billing.calls <= 3
    finally:
        app.dependency_overrides.pop(get_billing_client, None)


async def test_a_cancelled_run_stays_unsettled_while_its_render_may_still_bill(
    client, app, db_session_factory
) -> None:
    """Cancel reaches the video compiler only best-effort; a render it missed keeps
    billing until the compiler's 3600s job ceiling. Settling a cancelled run 30s
    after the cancel froze that spend out. It now settles only past the ceiling."""
    from src.api.routes._common import get_billing_client

    billing = _ScriptedBilling([("promo_script", "0.01")], [("promo_render", "0.50")])
    app.dependency_overrides[get_billing_client] = lambda: billing
    try:
        rid, run_uuid = await _make_run_row(
            client, db_session_factory, status="cancelled", updated_age_seconds=600
        )
        early = (await client.get(f"/api/runs/{rid}")).json()["usage_total"]
        assert early["cost_settled"] is False, "10 min after cancel a render may still bill"
        await _backdate(db_session_factory, run_uuid, 3600 + 301)
        late = (await client.get(f"/api/runs/{rid}")).json()["usage_total"]
        assert late["cost_settled"] is True
        assert late["cost_usd"] == 0.5, "the render billed after the cancel is not frozen out"
    finally:
        app.dependency_overrides.pop(get_billing_client, None)


async def test_get_run_cost_not_settled_while_billing_is_down(
    client, app, db_session_factory
) -> None:
    """A billing outage writes and marks nothing, so the cost is reported as not
    settled (the viewer keeps saying "calculating"), never as a final $0."""
    from src.api.routes._common import get_billing_client
    from src.clients.billing_client import BillingCostUnavailableError

    class _DownBilling:
        async def costs_by_correlation(self, *a, **k):
            raise BillingCostUnavailableError("down")

    app.dependency_overrides[get_billing_client] = lambda: _DownBilling()
    rid, _ = await _make_run_row(
        client, db_session_factory, status="completed", updated_age_seconds=120
    )
    usage = (await client.get(f"/api/runs/{rid}")).json()["usage_total"]
    assert usage["cost_settled"] is False
    app.dependency_overrides.pop(get_billing_client, None)


class _HangingBilling:
    """Billing stub that never answers inside any sane request budget."""

    def __init__(self) -> None:
        self.calls = 0

    async def costs_by_correlation(self, tenant_id, correlation_ids, *, group_by=None):
        self.calls += 1
        await asyncio.sleep(60)
        return {"groups": []}


async def test_get_run_slow_billing_degrades_instead_of_503(
    client, app, db_session_factory, monkeypatch
) -> None:
    """A slow billing must not eat the run-detail DB budget: the read-through is
    bounded on its own, so the run still loads (200), just without the cost, and
    the run is NOT marked reconciled (the next load/sweep retries)."""
    from sqlalchemy import select

    from src.api.routes._common import get_billing_client
    from src.api.routes.runs import read as read_mod
    from src.database.models import Run

    # Shrink both budgets so the test is fast but keeps their ORDER: the billing
    # bound sits well inside the outer DB deadline, exactly like 3s inside 8s.
    monkeypatch.setattr(read_mod, "_RUN_DETAIL_DB_TIMEOUT_SECONDS", 1.5)
    monkeypatch.setattr(read_mod, "_RUN_DETAIL_BILLING_TIMEOUT_SECONDS", 0.1, raising=False)
    billing = _HangingBilling()
    app.dependency_overrides[get_billing_client] = lambda: billing
    try:
        rid, run_uuid = await _make_run_row(
            client, db_session_factory, status="completed", updated_age_seconds=120
        )
        resp = await client.get(f"/api/runs/{rid}")
        assert resp.status_code == 200, resp.text
        assert billing.calls == 1
        async with db_session_factory() as s:
            marked = (
                await s.execute(select(Run.usage_reconciled_at).where(Run.id == run_uuid))
            ).scalar_one()
        assert marked is None, "a timed-out read-through must not mark the run reconciled"
    finally:
        app.dependency_overrides.pop(get_billing_client, None)


async def test_get_run_with_only_byo_rows_still_reads_through_from_billing(
    client, app, db_session_factory
) -> None:
    """In-process tenant-BYO LLM rows are written locally during the run and never
    come from billing. A run that has ONLY those rows has not been mirrored yet, so
    the viewer must still read through — otherwise its other billed cost (TTS, image
    gen, …) shows $0 until the sweep."""
    from uuid import uuid7

    from src.api.routes._common import get_billing_client
    from src.database.models import RunUsage
    from src.database.repositories.run_usage_repository import IN_PROCESS_USAGE_SOURCE

    billing = _CountingBilling()
    app.dependency_overrides[get_billing_client] = lambda: billing
    try:
        rid, run_uuid = await _make_run_row(
            client, db_session_factory, status="completed", updated_age_seconds=0
        )
        async with db_session_factory() as s:
            s.add(
                RunUsage(
                    id=uuid7(),
                    run_id=run_uuid,
                    stage_name="promo_review",
                    model="claude-v1",
                    provider="anthropic",
                    input_tokens=21,
                    output_tokens=9,
                    metadata_={"source": IN_PROCESS_USAGE_SOURCE, "funding": "tenant_byo"},
                )
            )
            await s.commit()

        resp = await client.get(f"/api/runs/{rid}")
        assert resp.status_code == 200, resp.text
        assert len(billing.calls) == 1, "BYO-only rows must not suppress the billing read-through"

        # A billing-mirrored row does NOT mean the cost is final: a first mirror can
        # be partial, so an unsettled run keeps reading through (and replacing only
        # the billing-owned rows — the local BYO row survives every replace)...
        async with db_session_factory() as s:
            s.add(
                RunUsage(
                    id=uuid7(),
                    run_id=run_uuid,
                    stage_name="narrate",
                    input_tokens=0,
                    output_tokens=0,
                    metadata_={"source": "billing"},
                )
            )
            await s.commit()
        again = await client.get(f"/api/runs/{rid}")
        assert again.status_code == 200
        assert len(billing.calls) == 2, "an unsettled run re-reads billing"
        stages = {u["stage_name"] for u in again.json()["usage_total"]["by_stage"]}
        assert "promo_review" in stages, "the local BYO row must survive the mirror replace"

        # ...until it is settled past the settle window; then it is a local read.
        await _backdate(db_session_factory, run_uuid, 120)
        assert (await client.get(f"/api/runs/{rid}")).status_code == 200
        assert (await client.get(f"/api/runs/{rid}")).status_code == 200
        assert len(billing.calls) == 3, "a settled run never re-reads billing"
    finally:
        app.dependency_overrides.pop(get_billing_client, None)


@pytest.mark.parametrize("age", [0])
async def test_get_run_billing_credential_missing_is_unavailable_not_zero_not_500(
    client, app, db_session_factory, monkeypatch, age
) -> None:
    """CELL_AUTH_MODE=enforce, BILLING_BASE_URL set, no DOWNSTREAM_LICENSE_TOKEN: the
    REAL BillingServiceClient raises DownstreamCredentialMissingError. Reading the
    run must still 200, and its cost must be explicitly unavailable — never $0,
    never a 500 — and the run must stay unreconciled (retried once fixed)."""
    from sqlalchemy import select

    import src.clients.billing_client as billing_mod
    from src.api.routes._common import get_billing_client
    from src.clients.billing_client import BillingServiceClient
    from src.database.models import Run

    monkeypatch.setattr(billing_mod, "optional_out_of_cell_license_token", lambda: None)
    app.dependency_overrides[get_billing_client] = lambda: BillingServiceClient(
        base_url="http://billing.test"
    )
    try:
        rid, run_uuid = await _make_run_row(
            client, db_session_factory, status="completed", updated_age_seconds=age
        )
        resp = await client.get(f"/api/runs/{rid}")
        assert resp.status_code == 200, resp.text
        usage = resp.json()["usage_total"]
        assert usage["cost_status"] == "unavailable"
        assert usage["cost_unavailable_reason"] == "billing_credential_missing"
        assert usage["cost_usd"] is None, "unknown cost must not be served as a number"
        async with db_session_factory() as s:
            marked = (
                await s.execute(select(Run.usage_reconciled_at).where(Run.id == run_uuid))
            ).scalar_one()
        assert marked is None
    finally:
        app.dependency_overrides.pop(get_billing_client, None)


async def test_get_run_cost_status_ok_when_billing_answers(client, app, db_session_factory) -> None:
    from src.api.routes._common import get_billing_client

    app.dependency_overrides[get_billing_client] = lambda: _CountingBilling()
    try:
        rid, _ = await _make_run_row(client, db_session_factory, status="completed")
        usage = (await client.get(f"/api/runs/{rid}")).json()["usage_total"]
        assert usage["cost_status"] == "ok"
        assert usage["cost_unavailable_reason"] is None
        assert usage["cost_usd"] == 0
    finally:
        app.dependency_overrides.pop(get_billing_client, None)


async def test_get_run_usage_rows_name_the_serving_connection(
    client, app, db_session_factory
) -> None:
    """P7 / R3: a tenant-slot (in-process) row carries the connection id + pinned
    version through to run detail; a platform (billing-mirrored) row carries null."""
    from w2v_common.clients import WMResponse

    from src.api.routes._common import get_billing_client
    from src.services.llm.in_process import LLMResult
    from src.services.llm.metering import usage_row
    from tests.services.llm._fakes import CONN_ID, tenant_slot

    class _PlatformBilling:
        async def costs_by_correlation(self, tenant_id, correlation_ids, *, group_by=None):
            return {
                "groups": [
                    {
                        "key": {
                            "correlation_id": str(correlation_ids[0]),
                            "stage": "narrate",
                            "model": "platform-model",
                            "provider": "platform",
                        },
                        "cost_usd": "0.25",
                        "input_tokens": 10,
                        "output_tokens": 5,
                    }
                ]
            }

    app.dependency_overrides[get_billing_client] = _PlatformBilling
    try:
        rid, run_uuid = await _make_run_row(
            client, db_session_factory, status="completed", updated_age_seconds=120
        )
        result = LLMResult(
            response=WMResponse(
                output="x",
                output_type="string",
                output_schema_used=None,
                metadata=None,
                usage_metrics=None,
            ),
            input_tokens=3,
            output_tokens=2,
            model="tenant-model-7",
            provider_kind="openai-compatible",
        )
        async with db_session_factory() as s:
            s.add(
                usage_row(
                    run_id=run_uuid,
                    stage_name="promo_review",
                    worker_name="promo-review",
                    resolved=tenant_slot(version=7),
                    result=result,
                )
            )
            await s.commit()

        resp = await client.get(f"/api/runs/{rid}")
        assert resp.status_code == 200, resp.text
        rows = {r["stage_name"]: r for r in resp.json()["usage_total"]["by_stage"]}
        assert set(rows) == {"promo_review", "narrate"}
        assert rows["promo_review"]["connection_id"] == str(CONN_ID)
        assert rows["promo_review"]["connection_version"] == 7
        assert rows["narrate"]["connection_id"] is None
        assert rows["narrate"]["connection_version"] is None
    finally:
        app.dependency_overrides.pop(get_billing_client, None)


async def test_get_run_surfaces_degraded_stage(client, db_session_factory) -> None:
    """A completed stage carrying output.notes is surfaced as degraded, and the
    run is flagged degraded (completed-with-warnings) — not silently green."""
    from uuid import UUID, uuid7

    from src.database.models import StageOutput

    rid, run_uuid = await _make_run_row(client, db_session_factory, status="completed")
    async with db_session_factory() as s:
        s.add(
            StageOutput(
                id=uuid7(),
                run_id=run_uuid,
                stage_name="media_discovery",
                status="completed",
                output={"notes": ["source unavailable: example.com"]},
            )
        )
        s.add(
            StageOutput(
                id=uuid7(),
                run_id=run_uuid,
                stage_name="script_draft",
                status="completed",
                output={"script": "ok"},
            )
        )
        await s.commit()

    r = await client.get(f"/api/runs/{rid}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["degraded"] is True, "run with a notes-carrying stage must be degraded"
    by_stage = {so["stage_name"]: so for so in body["stage_outputs"]}
    assert by_stage["media_discovery"]["degraded"] is True
    assert by_stage["script_draft"]["degraded"] is False
    assert UUID(body["id"]) == run_uuid


async def test_get_run_clean_when_no_notes(client, db_session_factory) -> None:
    """A completed run whose stages carry no notes is NOT flagged degraded."""
    from uuid import uuid7

    from src.database.models import StageOutput

    rid, run_uuid = await _make_run_row(client, db_session_factory, status="completed")
    async with db_session_factory() as s:
        s.add(
            StageOutput(
                id=uuid7(),
                run_id=run_uuid,
                stage_name="script_draft",
                status="completed",
                output={"script": "ok", "notes": []},
            )
        )
        await s.commit()

    r = await client.get(f"/api/runs/{rid}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["degraded"] is False
    assert body["stage_outputs"][0]["degraded"] is False


async def test_get_checkpoints(client) -> None:
    _, deliv = await _make_project(client)
    r = await client.post(
        f"/api/deliverables/{deliv}/runs",
        json={"pause_after_stages": ["script_draft", "storyboard_draft"]},
    )
    rid = r.json()["id"]
    cp = await client.get(f"/api/runs/{rid}/checkpoints")
    assert cp.status_code == 200, cp.text
    body = cp.json()
    assert body["run_id"] == rid
    assert body["pause_after_stages"] == ["script_draft", "storyboard_draft"]


async def _seed_stage_row(db_session_factory, run_uuid, stage_name: str, status: str) -> None:
    """Put ONE stage_output row on ``run_uuid`` in a known state."""
    from uuid import uuid7

    from src.database.models import StageOutput

    async with db_session_factory() as s:
        s.add(
            StageOutput(
                id=uuid7(),
                run_id=run_uuid,
                stage_name=stage_name,
                status=status,
                output={},
            )
        )
        await s.commit()


async def test_patch_checkpoints_arms_a_gate_whose_stage_is_still_open(
    client, db_session_factory
) -> None:
    """PATCH stores a gate on a stage that has NOT run yet, and it round-trips.

    The run is created DIRECTLY in the DB, not via ``POST /deliverables/{id}/runs``.
    That POST ends in ``RunEngine.start_run`` -> ``self._dispatch(run.id)``, which is
    fire-and-forget: the module's ``_mock_stage_clients`` fixture makes the mocked
    stages complete in tens of milliseconds, so by the time the PATCH landed
    ``script_draft`` was often ALREADY terminal and the route's consumed-gate strip
    (correctly) removed it — ``assert [] == ['script_draft']``. That raced on wall
    clock alone: it reddened main at d6e4f6ac while passing on the identical tree in
    PR #906, and reproduces on demand with a 0.2s sleep before the PATCH. DB-direct
    creation is the deterministic construction ``_make_run_row`` already documents.
    """
    rid, run_uuid = await _make_run_row(client, db_session_factory, status="running")
    # script_draft exists but has NOT reached a terminal state, so the gate is live.
    await _seed_stage_row(db_session_factory, run_uuid, "script_draft", "running")

    patch_resp = await client.patch(
        f"/api/runs/{rid}/checkpoints",
        json={"pause_after_stages": ["script_draft"]},
    )
    assert patch_resp.status_code == 200, patch_resp.text
    assert patch_resp.json()["pause_after_stages"] == ["script_draft"]
    # Round-trip via GET — the gate is DURABLE, not just echoed back.
    cp = await client.get(f"/api/runs/{rid}/checkpoints")
    assert cp.json()["pause_after_stages"] == ["script_draft"]
    # And durable in the ROW the dispatcher reads, not only in the API projection.
    async with db_session_factory() as s:
        from src.database.models import Run

        row = await s.get(Run, run_uuid)
        assert row is not None
        assert list(row.pause_after_stages or []) == ["script_draft"]


@pytest.mark.parametrize("consumed_status", ["completed"])
async def test_patch_checkpoints_cannot_rearm_an_already_consumed_gate(
    client, db_session_factory, consumed_status: str
) -> None:
    """A gate on a stage that is ALREADY terminal is stripped; live gates survive.

    This is the property the racing version of the test was hitting by accident and
    asserting the opposite of. ``resume_run`` uses ``pause_after_stages`` as a
    durable APPROVAL LEDGER; re-arming a gate the reviewer already approved past
    makes the run re-pause the instant it resumes. Both terminal statuses count —
    ``skipped`` is terminal too (``TERMINAL_STAGE_STATUSES``), so a deterministically
    skipped stage must not be re-armable either.
    """
    rid, run_uuid = await _make_run_row(client, db_session_factory, status="running")
    await _seed_stage_row(db_session_factory, run_uuid, "script_draft", consumed_status)
    await _seed_stage_row(db_session_factory, run_uuid, "storyboard_draft", "pending")

    patch_resp = await client.patch(
        f"/api/runs/{rid}/checkpoints",
        json={"pause_after_stages": ["script_draft", "storyboard_draft"]},
    )
    assert patch_resp.status_code == 200, patch_resp.text
    # The consumed gate is dropped; the still-open one is kept — not "all or nothing".
    assert patch_resp.json()["pause_after_stages"] == ["storyboard_draft"]
    cp = await client.get(f"/api/runs/{rid}/checkpoints")
    assert cp.json()["pause_after_stages"] == ["storyboard_draft"]
    async with db_session_factory() as s:
        from src.database.models import Run

        row = await s.get(Run, run_uuid)
        assert row is not None
        assert list(row.pause_after_stages or []) == ["storyboard_draft"], (
            "the stripped gate must not be persisted — resume would re-pause on it"
        )


async def test_checkpoints_404_on_unknown_run(client) -> None:
    bogus = "00000000-0000-0000-0000-000000000000"
    r = await client.get(f"/api/runs/{bogus}/checkpoints")
    assert r.status_code == 404


async def test_ancestors_survives_parent_cycle(client, db_session) -> None:
    """Regression: parent_run_id is not constrained acyclic at write time, so a
    cycle would make the ancestor walk loop one DB query per hop forever. The
    visited-set guard breaks the cycle and returns a bounded chain."""
    from uuid import UUID

    from sqlalchemy import update

    from src.database.models import Run
    from src.database.repositories.run_repository import RunRepository

    _, deliv = await _make_project(client)
    rr = RunRepository(db_session)
    run_a = await rr.create(
        deliverable_id=UUID(deliv),
        blueprint_id="generic_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    run_b = await rr.create(
        deliverable_id=UUID(deliv),
        blueprint_id="generic_v1",
        config_snapshot={},
        pause_after_stages=[],
        parent_run_id=run_a.id,
    )
    # Close the loop: A's parent points back at B → cycle A→B→A.
    await db_session.execute(update(Run).where(Run.id == run_a.id).values(parent_run_id=run_b.id))
    await db_session.commit()

    res = await client.get(f"/api/runs/{run_b.id}/ancestors")
    assert res.status_code == 200, res.text
    ids = [r["id"] for r in res.json()]
    # Walk from B visits A, then sees B already-visited and stops — bounded.
    assert str(run_a.id) in ids
    assert len(ids) <= 2


async def test_regenerate_404_on_unknown_run(client) -> None:
    bogus = "00000000-0000-0000-0000-000000000000"
    r = await client.post(f"/api/runs/{bogus}/regenerate", json={})
    assert r.status_code == 404


async def test_regenerate_resolves_subject_id_from_contextvar(app, db_session) -> None:
    """Regression (HIGH): regenerate must read subject_id from the identity
    ContextVar, not request.state.

    Real traffic goes through w2v_common.AuthMiddleware (a BaseHTTPMiddleware)
    which sets request.state.tenant_id but puts subject_id ONLY in the identity
    ContextVar. The old inline ``getattr(request.state, "subject_id")`` therefore
    401'd on every production caller while passing under the test conftest's
    _SeedAuthState (which seeds request.state). This test mirrors prod auth and
    asserts a 201, not a 401.
    """
    from uuid import UUID, uuid4

    from httpx import ASGITransport, AsyncClient
    from starlette.middleware.base import BaseHTTPMiddleware
    from w2v_common.auth.identity_context import (
        clear_identity_context,
        set_identity_context,
    )

    from src.database.repositories.run_repository import RunRepository

    tenant_id = str(uuid4())
    subject_id = str(uuid4())

    class _ProdAuth(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            # Mirror AuthMiddleware: tenant on request.state, subject_id ONLY in
            # the ContextVar (set before call_next so it propagates downstream).
            request.state.tenant_id = tenant_id
            set_identity_context(tenant_id=tenant_id, license_id="", subject_id=subject_id)
            try:
                return await call_next(request)
            finally:
                clear_identity_context()

    app.add_middleware(_ProdAuth)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        r = await client.post(
            "/api/projects",
            json={
                "type": "generic",
                "title": "regen-auth",
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
        deliv_id = detail.json()["deliverables"][0]["id"]

        # regenerate requires a stopped (paused/completed/failed/cancelled) parent — seed one directly.
        run_repo = RunRepository(db_session)
        parent = await run_repo.create(
            deliverable_id=UUID(deliv_id),
            blueprint_id="generic_v1",
            config_snapshot={},
            pause_after_stages=[],
        )
        await run_repo.mark_completed(parent.id)
        await db_session.commit()

        resp = await client.post(f"/api/runs/{parent.id}/regenerate", json={})
        assert resp.status_code != 401, f"subject_id not resolved from ContextVar: {resp.text}"
        assert resp.status_code == 201, resp.text
        assert resp.json()["parent_run_id"] == str(parent.id)


# ─── Reorder clips (studio timeline) ───────────────────────────────────────


async def test_reorder_clips_404_unknown_run(client) -> None:
    bogus = "00000000-0000-0000-0000-000000000000"
    r = await client.post(
        f"/api/runs/{bogus}/reorder-clips",
        json={"stage_name": "storyboard_draft", "ordered_element_ids": ["s1"]},
    )
    assert r.status_code == 404


async def test_reorder_clips_422_when_stage_not_a_spine(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session, deliv, "storyboard_draft", [{"id": "s1"}, {"id": "s2"}]
    )
    # video_render is the compile stage, not a clip spine → 422.
    r = await client.post(
        f"/api/runs/{rid}/reorder-clips",
        json={"stage_name": "video_render", "ordered_element_ids": ["s1", "s2"]},
    )
    assert r.status_code == 422


async def test_reorder_clips_creates_reordered_child(client, db_session) -> None:
    from uuid import UUID as _UUID

    from src.database.repositories.run_repository import RunRepository

    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session,
        deliv,
        "storyboard_draft",
        [{"id": "s1", "seconds": 3}, {"id": "s2", "seconds": 4}, {"id": "s3", "seconds": 5}],
    )
    # A reorder re-stitches a finished video, so the parent must be terminal.
    await RunRepository(db_session).mark_completed(_UUID(rid))
    await db_session.commit()
    r = await client.post(
        f"/api/runs/{rid}/reorder-clips",
        json={"stage_name": "storyboard_draft", "ordered_element_ids": ["s3", "s1", "s2"]},
    )
    assert r.status_code == 201, r.text
    assert r.json()["parent_run_id"] == rid

    # 400 on a non-permutation.
    bad = await client.post(
        f"/api/runs/{rid}/reorder-clips",
        json={"stage_name": "storyboard_draft", "ordered_element_ids": ["s1"]},
    )
    assert bad.status_code == 400


# ─── Apply per-clip edits (studio) ─────────────────────────────────────────


async def test_apply_clip_edits_404_unknown_run(client) -> None:
    bogus = "00000000-0000-0000-0000-000000000000"
    r = await client.post(
        f"/api/runs/{bogus}/apply-edits",
        json={"stage_name": "storyboard_draft", "edits": {"s1": {"clip_lead_seconds": 1.0}}},
    )
    assert r.status_code == 404


async def test_apply_clip_edits_422_when_stage_not_a_spine(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session, deliv, "storyboard_draft", [{"id": "s1"}, {"id": "s2"}]
    )
    # video_render is the compile stage, not a clip spine → 422.
    r = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={"stage_name": "video_render", "edits": {"s1": {"clip_lead_seconds": 1.0}}},
    )
    assert r.status_code == 422


async def test_apply_clip_edits_creates_edited_child(client, db_session) -> None:
    from uuid import UUID as _UUID

    from src.database.repositories.run_repository import RunRepository

    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session,
        deliv,
        "storyboard_draft",
        [{"id": "s1", "seconds": 3}, {"id": "s2", "seconds": 4}, {"id": "s3", "seconds": 5}],
    )
    # An edit re-stitches a finished video, so the parent must be terminal.
    await RunRepository(db_session).mark_completed(_UUID(rid))
    await db_session.commit()
    r = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={
            "stage_name": "storyboard_draft",
            "edits": {"s2": {"clip_lead_seconds": 1.5, "clip_anchor": "start"}},
        },
    )
    assert r.status_code == 201, r.text
    assert r.json()["parent_run_id"] == rid

    # 400 when no edited id exists on the current timeline.
    bad = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={"stage_name": "storyboard_draft", "edits": {"sX": {"clip_lead_seconds": 1.0}}},
    )
    assert bad.status_code == 400, bad.text


async def test_apply_clip_edits_422_on_negative_lead(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(db_session, deliv, "storyboard_draft", [{"id": "s1"}])
    # clip_lead_seconds has ge=0; a negative value is a 422 body-validation error.
    r = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={"stage_name": "storyboard_draft", "edits": {"s1": {"clip_lead_seconds": -1.0}}},
    )
    assert r.status_code == 422


async def test_apply_clip_edits_accepts_caption_and_fit(client, db_session) -> None:
    from uuid import UUID as _UUID

    from src.database.repositories.run_repository import RunRepository

    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session, deliv, "storyboard_draft", [{"id": "s1", "line_id": "s1"}]
    )
    await RunRepository(db_session).mark_completed(_UUID(rid))
    await db_session.commit()
    r = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={
            "stage_name": "storyboard_draft",
            "edits": {"s1": {"caption": "burned-in text", "fit": "cover"}},
        },
    )
    assert r.status_code == 201, r.text
    assert r.json()["parent_run_id"] == rid


async def test_apply_clip_edits_422_on_foreign_voiceover_url(client, db_session) -> None:
    # A ClipEdit.audio_url is the only client-settable asset URL, and the
    # compile-time re-sign pass re-mints ANY object in a bucket we own with our
    # own MinIO creds. An audio_url under ANOTHER tenant's voiceover prefix must
    # be rejected — otherwise it is a cross-tenant read primitive.
    from uuid import UUID as _UUID
    from uuid import uuid4

    from src.database.repositories.run_repository import RunRepository

    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(db_session, deliv, "storyboard_draft", [{"id": "s1"}])
    # Complete the run so we pass the pending-run gate and actually reach the
    # ownership guard (the point of this test).
    await RunRepository(db_session).mark_completed(_UUID(rid))
    await db_session.commit()
    foreign = uuid4()  # some other tenant's namespace
    url = (
        f"https://minio.test/artifacts/voiceovers/{foreign}"
        f"/clip_voiceover_override/take.mp3?X-Amz-Signature=deadbeef"
    )
    r = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={"stage_name": "storyboard_draft", "edits": {"s1": {"audio_url": url}}},
    )
    assert r.status_code == 422, r.text
    assert "audio_url" in r.text


async def test_apply_clip_edits_accepts_own_voiceover_url(client, db_session) -> None:
    from uuid import UUID as _UUID

    from src.database.repositories.run_repository import RunRepository

    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session, deliv, "storyboard_draft", [{"id": "s1", "line_id": "s1"}]
    )
    await RunRepository(db_session).mark_completed(_UUID(rid))
    await db_session.commit()
    # This tenant's own voiceover prefix passes the ownership guard.
    url = (
        f"https://minio.test/artifacts/voiceovers/{client.tenant_id}"
        f"/clip_voiceover_override/take.mp3?X-Amz-Signature=deadbeef"
    )
    r = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={
            "stage_name": "storyboard_draft",
            "edits": {"s1": {"audio_url": url, "audio_seconds": 12.4}},
        },
    )
    assert r.status_code == 201, r.text


async def test_apply_clip_edits_422_on_bad_fit(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(db_session, deliv, "storyboard_draft", [{"id": "s1"}])
    # fit is a Literal[contain|cover|scroll]; anything else is a 422 body error.
    r = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={"stage_name": "storyboard_draft", "edits": {"s1": {"fit": "zoom"}}},
    )
    assert r.status_code == 422


async def test_apply_clip_edits_409_on_running_parent(client, db_session) -> None:
    from uuid import UUID as _UUID

    from src.database.repositories.run_repository import RunRepository

    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session, deliv, "storyboard_draft", [{"id": "s1"}, {"id": "s2"}]
    )
    await RunRepository(db_session).mark_started(_UUID(rid))
    await db_session.commit()
    r = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={"stage_name": "storyboard_draft", "edits": {"s1": {"clip_lead_seconds": 1.0}}},
    )
    assert r.status_code == 409, r.text


async def test_apply_clip_edits_422_when_nothing_to_apply(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(db_session, deliv, "storyboard_draft", [{"id": "s1"}])
    # No clip edits and no music change → the model validator rejects it (422).
    r = await client.post(
        f"/api/runs/{rid}/apply-edits", json={"stage_name": "storyboard_draft", "edits": {}}
    )
    assert r.status_code == 422


async def test_apply_clip_edits_music_only_sets_child_music(client, db_session) -> None:
    from uuid import UUID as _UUID

    from sqlalchemy import select

    from src.database.models import Run
    from src.database.repositories.run_repository import RunRepository

    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(db_session, deliv, "storyboard_draft", [{"id": "s1"}])
    await RunRepository(db_session).mark_completed(_UUID(rid))
    await db_session.commit()
    # A music-only change (no clip edits) re-stitches and sets the bed on the child.
    r = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={
            "stage_name": "storyboard_draft",
            "edits": {},
            "music": {"track": "corporate_uplift", "gain_db": -14.0},
        },
    )
    assert r.status_code == 201, r.text
    child = (
        await db_session.execute(select(Run).where(Run.id == _UUID(r.json()["id"])))
    ).scalar_one()
    sound = child.config_snapshot["sound_effects"]
    assert sound["music"] == "corporate_uplift"
    assert sound["enabled"] is True
    assert sound["music_gain_db"] == -14.0


async def _make_website_review_project(client) -> str:
    r = await client.post(
        "/api/projects",
        json={
            "type": "website_review",
            "title": "review",
            "brief": {"source_url": "https://example.com"},
            "deliverables": [{"name": "d", "format": "16x9", "target_duration_seconds": 60}],
        },
    )
    assert r.status_code == 201, r.text
    detail = await client.get(f"/api/projects/{r.json()['id']}")
    return detail.json()["deliverables"][0]["id"]


async def _seed_completed_website_review(
    db_session, deliv_id: str, config_snapshot: dict | None = None
) -> str:
    from uuid import UUID as _UUID

    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository

    run_repo = RunRepository(db_session)
    run = await run_repo.create(
        deliverable_id=_UUID(deliv_id),
        blueprint_id="website_review_v1",
        config_snapshot=config_snapshot or {},
        pause_after_stages=[],
    )
    so_repo = StageOutputRepository(db_session)
    await so_repo.upsert_pending(run.id, "website_review_script")
    await db_session.commit()
    await so_repo.mark_completed(
        run.id,
        "website_review_script",
        {"lines": [{"id": "L1", "text": "a", "seconds": 3.0}, {"id": "L2", "text": "b"}]},
    )
    await run_repo.mark_completed(run.id)
    await db_session.commit()
    return str(run.id)


async def test_apply_edits_on_website_review_script_music_bed(client, db_session) -> None:
    """Prod 01a09f82: the Studio's music edit on a website_review run 422'd
    ``stage 'website_review_script' does not carry a clip timeline``."""
    from uuid import UUID as _UUID

    from sqlalchemy import select

    from src.database.models import Run

    deliv = await _make_website_review_project(client)
    rid = await _seed_completed_website_review(db_session, deliv)
    music = {"gain_db": -16.0, "fade_in_s": 1.0, "fade_out_s": 2.0, "duck_db": -6.0}
    r = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={
            "stage_name": "website_review_script",
            "edits": {},
            "music": {"track": "jazz_lounge", **music},
        },
    )
    assert r.status_code == 201, r.text
    child = (
        await db_session.execute(select(Run).where(Run.id == _UUID(r.json()["id"])))
    ).scalar_one()
    sound = child.config_snapshot["sound_effects"]
    assert sound["music"] == "jazz_lounge"
    assert sound["duck_db"] == -6.0


async def test_apply_edits_music_track_null_persists_explicit_off(client, db_session) -> None:
    """Studio "No music" sends ``music: {track: null}``. The child must carry an
    explicit ``sound_effects.music = False`` — a DROPPED key means "auto" and the
    audio bed re-picks a track, so the child shipped music anyway."""
    from uuid import UUID as _UUID

    from sqlalchemy import select

    from src.database.models import Run

    deliv = await _make_website_review_project(client)
    rid = await _seed_completed_website_review(db_session, deliv)
    r = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={"stage_name": "website_review_script", "edits": {}, "music": {"track": None}},
    )
    assert r.status_code == 201, r.text
    child = (
        await db_session.execute(select(Run).where(Run.id == _UUID(r.json()["id"])))
    ).scalar_one()
    assert child.config_snapshot["sound_effects"]["music"] is False


async def _child_sound(client, db_session, rid: str, body: dict) -> dict:
    from uuid import UUID as _UUID

    from sqlalchemy import select

    from src.database.models import Run

    r = await client.post(f"/api/runs/{rid}/apply-edits", json=body)
    assert r.status_code == 201, r.text
    child = (
        await db_session.execute(select(Run).where(Run.id == _UUID(r.json()["id"])))
    ).scalar_one()
    return dict(child.config_snapshot.get("sound_effects") or {})


_CLIP_ONLY_EDIT = {
    "stage_name": "website_review_script",
    "edits": {"L1": {"clip_lead_seconds": 0.5}},
}


async def test_a_clip_only_edit_keeps_the_parents_studio_music(client, db_session) -> None:
    """Studio contract: an omitted ``music`` on apply-edits carries the PARENT's
    resolved bed. The child rebuilds config from the brief, so without the carry a
    caption/trim edit after a music choice silently reverted the soundtrack."""
    deliv = await _make_website_review_project(client)
    parent_sound = {
        "enabled": True,
        "music": "jazz_lounge",
        "music_gain_db": -16.0,
        "fade_in_s": 1.0,
        "fade_out_s": 2.0,
        "duck_db": -6.0,
    }
    rid = await _seed_completed_website_review(db_session, deliv, {"sound_effects": parent_sound})
    assert await _child_sound(client, db_session, rid, _CLIP_ONLY_EDIT) == parent_sound


async def test_a_clip_only_edit_keeps_the_parents_music_off(client, db_session) -> None:
    deliv = await _make_website_review_project(client)
    rid = await _seed_completed_website_review(
        db_session, deliv, {"sound_effects": {"enabled": True, "music": False}}
    )
    sound = await _child_sound(client, db_session, rid, _CLIP_ONLY_EDIT)
    assert sound["music"] is False


async def test_an_explicit_music_change_wins_over_the_parents_bed(client, db_session) -> None:
    deliv = await _make_website_review_project(client)
    rid = await _seed_completed_website_review(
        db_session,
        deliv,
        {"sound_effects": {"enabled": True, "music": "jazz_lounge", "duck_db": -6.0}},
    )
    sound = await _child_sound(
        client, db_session, rid, {**_CLIP_ONLY_EDIT, "music": {"track": "corporate_uplift"}}
    )
    assert sound["music"] == "corporate_uplift"
    assert sound["duck_db"] == -6.0  # an omitted knob keeps the parent's value
    off = await _child_sound(client, db_session, rid, {**_CLIP_ONLY_EDIT, "music": {"track": None}})
    assert off["music"] is False


async def test_apply_edits_rejects_a_music_track_label_instead_of_its_id(
    client, db_session
) -> None:
    """The FE sent the track's LABEL; downstream it silently resolved to no bed."""
    deliv = await _make_website_review_project(client)
    rid = await _seed_completed_website_review(db_session, deliv)
    r = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={
            "stage_name": "website_review_script",
            "edits": {},
            "music": {"track": "smooth lounge jazz background bed", "duck_db": -6.0},
        },
    )
    assert r.status_code == 422, r.text
    detail = r.json().get("detail") or r.text
    assert "unknown music track" in str(detail)
    assert "jazz_lounge" in str(detail)


# ─── Regenerate a single clip (studio) ─────────────────────────────────────


async def test_regenerate_clip_404_unknown_run(client) -> None:
    bogus = "00000000-0000-0000-0000-000000000000"
    r = await client.post(
        f"/api/runs/{bogus}/regenerate-clip",
        json={"stage_name": "storyboard_draft", "element_id": "s1"},
    )
    assert r.status_code == 404


async def test_regenerate_clip_422_when_stage_not_a_spine(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session, deliv, "storyboard_draft", [{"id": "s1"}, {"id": "s2"}]
    )
    r = await client.post(
        f"/api/runs/{rid}/regenerate-clip",
        json={"stage_name": "video_render", "element_id": "s1"},
    )
    assert r.status_code == 422


async def test_regenerate_clip_422_when_pipeline_cannot_target_single_clip(
    client, db_session
) -> None:
    from uuid import UUID as _UUID

    from src.database.repositories.run_repository import RunRepository

    _, deliv = await _make_project(client)
    # generic pipeline: ai_image_gen uses static round-robin prompts (no
    # prompts_from_stage), so a single clip cannot be regenerated surgically → 422.
    rid = await _seed_stage_with_elements(
        db_session,
        deliv,
        "storyboard_draft",
        [{"id": "s1", "line_id": "L1"}, {"id": "s2", "line_id": "L2"}],
    )
    await RunRepository(db_session).mark_completed(_UUID(rid))
    await db_session.commit()
    r = await client.post(
        f"/api/runs/{rid}/regenerate-clip",
        json={"stage_name": "storyboard_draft", "element_id": "s1"},
    )
    assert r.status_code == 422, r.text


# ─── Reorder / regen-clip door: parent must be a completed run ─────────────


async def _seed_completed_spine(db_session, deliv, elements):
    """Seed a storyboard spine and mark the RUN completed (a finished timeline)."""
    from uuid import UUID as _UUID

    from src.database.repositories.run_repository import RunRepository

    rid = await _seed_stage_with_elements(db_session, deliv, "storyboard_draft", elements)
    await RunRepository(db_session).mark_completed(_UUID(rid))
    await db_session.commit()
    return rid


async def test_reorder_clips_409_on_running_parent(client, db_session) -> None:
    from uuid import UUID as _UUID

    from src.database.repositories.run_repository import RunRepository

    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session, deliv, "storyboard_draft", [{"id": "s1"}, {"id": "s2"}]
    )
    await RunRepository(db_session).mark_started(_UUID(rid))
    await db_session.commit()
    r = await client.post(
        f"/api/runs/{rid}/reorder-clips",
        json={"stage_name": "storyboard_draft", "ordered_element_ids": ["s2", "s1"]},
    )
    assert r.status_code == 409, r.text


@pytest.mark.parametrize("bad_status", ["paused"])
async def test_reorder_clips_422_when_parent_not_completed(client, db_session, bad_status) -> None:
    from uuid import UUID as _UUID

    from src.database.repositories.run_repository import RunRepository

    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session, deliv, "storyboard_draft", [{"id": "s1"}, {"id": "s2"}]
    )
    run_repo = RunRepository(db_session)
    if bad_status == "paused":
        await run_repo.mark_paused(_UUID(rid), reason="checkpoint")
    else:
        await run_repo.mark_failed(_UUID(rid), "boom")
    await db_session.commit()
    r = await client.post(
        f"/api/runs/{rid}/reorder-clips",
        json={"stage_name": "storyboard_draft", "ordered_element_ids": ["s2", "s1"]},
    )
    # A reorder re-stitches a finished timeline; a paused/failed run has none.
    assert r.status_code == 422, r.text


async def test_reorder_clips_cross_tenant_returns_404(
    client, other_tenant_client, db_session
) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_completed_spine(db_session, deliv, [{"id": "s1"}, {"id": "s2"}])
    bleed = await other_tenant_client.post(
        f"/api/runs/{rid}/reorder-clips",
        json={"stage_name": "storyboard_draft", "ordered_element_ids": ["s2", "s1"]},
    )
    assert bleed.status_code == 404


async def test_regenerate_clip_422_when_parent_not_completed(client, db_session) -> None:
    from uuid import UUID as _UUID

    from src.database.repositories.run_repository import RunRepository

    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session, deliv, "storyboard_draft", [{"id": "s1", "line_id": "L1"}]
    )
    await RunRepository(db_session).mark_paused(_UUID(rid), reason="checkpoint")
    await db_session.commit()
    r = await client.post(
        f"/api/runs/{rid}/regenerate-clip",
        json={"stage_name": "storyboard_draft", "element_id": "s1"},
    )
    assert r.status_code == 422, r.text


async def _seed_completed_promo_regen_clip(db_session, client) -> str:
    """Seed a completed promo parent whose ``ai_image_gen`` uses per-line prompts
    and whose storyboard clip ``s1`` carries open feedback — the shape a surgical
    single-clip regenerate accepts. Bound to ``client``'s tenant so the route's
    ownership check passes. Returns the parent run id (str)."""
    from uuid import UUID as _UUID
    from uuid import uuid4

    from src.database.models import Annotation, Deliverable, Project, Run, StageOutput

    tenant_id = _UUID(client.tenant_id)
    subject_id = _UUID(client.subject_id)
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="promo",
        title="regen-clip-http",
        brief={"angle": "growth", "target_duration_seconds": 30},
        brand={},
        voice={"descriptor": "crisp"},
        theme={},
    )
    deliverable = Deliverable(
        id=uuid4(),
        project_id=project.id,
        name="d",
        format="16x9",
        captions_mode="none",
        target_duration_seconds=30,
    )
    parent = Run(
        id=uuid4(),
        deliverable_id=deliverable.id,
        blueprint_id="promo_v1",
        status="completed",
        config_snapshot={"ai_image_gen": {"prompts_from_stage": "promo_image_prompts"}},
        pause_after_stages=[],
    )
    storyboard = StageOutput(
        id=uuid4(),
        run_id=parent.id,
        stage_name="promo_storyboard",
        status="completed",
        output={
            "shots": [
                {"id": "s1", "line_id": "L1", "seconds": 3, "description": "one"},
                {"id": "s2", "line_id": "L2", "seconds": 4, "description": "two"},
            ]
        },
    )
    note = Annotation(
        id=uuid4(),
        stage_output_id=storyboard.id,
        element_id="s1",
        text="make it brighter",
        status="open",
    )
    db_session.add_all([project, deliverable, parent, storyboard, note])
    await db_session.commit()
    return str(parent.id)


async def test_regenerate_clip_cross_tenant_returns_404(
    client, other_tenant_client, db_session
) -> None:
    rid = await _seed_completed_promo_regen_clip(db_session, client)
    bleed = await other_tenant_client.post(
        f"/api/runs/{rid}/regenerate-clip",
        json={"stage_name": "promo_storyboard", "element_id": "s1"},
    )
    assert bleed.status_code == 404


async def test_regenerate_clip_409_on_running_parent(client, db_session) -> None:
    from uuid import UUID as _UUID

    from src.database.repositories.run_repository import RunRepository

    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session, deliv, "storyboard_draft", [{"id": "s1", "line_id": "L1"}]
    )
    await RunRepository(db_session).mark_started(_UUID(rid))
    await db_session.commit()
    r = await client.post(
        f"/api/runs/{rid}/regenerate-clip",
        json={"stage_name": "storyboard_draft", "element_id": "s1"},
    )
    assert r.status_code == 409, r.text


async def test_regenerate_clip_creates_child(client, db_session) -> None:
    rid = await _seed_completed_promo_regen_clip(db_session, client)
    r = await client.post(
        f"/api/runs/{rid}/regenerate-clip",
        json={"stage_name": "promo_storyboard", "element_id": "s1"},
    )
    assert r.status_code == 201, r.text
    assert r.json()["parent_run_id"] == rid


# ─── Regenerate: from_stage ('rerun this stage') targeting ─────────────────


async def test_regenerate_from_stage_creates_child(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_completed_spine(db_session, deliv, [{"id": "s1"}, {"id": "s2"}])
    r = await client.post(
        f"/api/runs/{rid}/regenerate",
        json={"from_stage": "storyboard_draft"},
    )
    assert r.status_code == 201, r.text
    assert r.json()["parent_run_id"] == rid


async def test_regenerate_from_stage_unknown_stage_422(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_completed_spine(db_session, deliv, [{"id": "s1"}])
    r = await client.post(
        f"/api/runs/{rid}/regenerate",
        json={"from_stage": "no_such_stage"},
    )
    assert r.status_code == 422, r.text


# ─── Element PATCH (Phase A.2) ─────────────────────────────────────────────


async def _stop_run(db_session, run_id) -> None:
    """Park a seeded run at a review gate (``paused``).

    The seeders leave a freshly created run ``pending`` — a run the dispatcher is
    about to own — and in-place output edits (element patch / restamp) are refused
    on pending and running runs. An edit test needs a run nobody is executing.
    """
    from uuid import UUID as _UUID

    from src.database.repositories.run_repository import RunRepository

    assert await RunRepository(db_session).mark_paused(_UUID(str(run_id)), reason="manual")
    await db_session.commit()


async def _seed_stage_with_elements(db_session, deliv_id: str, stage: str, elements: list[dict]):
    """Insert a StageOutput row with the given list of elements addressable
    by id (path key matches `_ELEMENT_PATHS[stage]`'s first entry)."""
    from uuid import UUID as _UUID

    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository
    from src.services.runs.elements import _ELEMENT_PATHS

    run_repo = RunRepository(db_session)
    run = await run_repo.create(
        deliverable_id=_UUID(deliv_id),
        blueprint_id="generic_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    so_repo = StageOutputRepository(db_session)
    await so_repo.upsert_pending(run.id, stage)
    await db_session.commit()
    path_key = _ELEMENT_PATHS[stage][0]
    await so_repo.mark_completed(run.id, stage, {path_key: elements})
    await db_session.commit()
    return str(run.id)


async def test_patch_element_shallow_merge(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session,
        deliv,
        "script_draft",
        [
            {"id": "L1", "text": "old", "seconds": 5.0},
            {"id": "L2", "text": "keep", "seconds": 3.0},
        ],
    )
    await _stop_run(db_session, rid)
    r = await client.patch(
        f"/api/runs/{rid}/stages/script_draft/elements/L1",
        json={"patch": {"text": "new", "extra": "added"}},
    )
    assert r.status_code == 200, r.text
    output = r.json()["output"]
    lines = output["lines"]
    assert lines[0] == {"id": "L1", "text": "new", "seconds": 5.0, "extra": "added"}
    assert lines[1] == {"id": "L2", "text": "keep", "seconds": 3.0}


async def test_patch_element_id_is_preserved(client, db_session) -> None:
    """Even if patch tries to overwrite `id`, it stays put."""
    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session, deliv, "script_draft", [{"id": "L1", "text": "x"}]
    )
    await _stop_run(db_session, rid)
    r = await client.patch(
        f"/api/runs/{rid}/stages/script_draft/elements/L1",
        json={"patch": {"id": "EVIL", "text": "y"}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["output"]["lines"][0]["id"] == "L1"


async def test_patch_element_unknown_id_returns_404(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session, deliv, "script_draft", [{"id": "L1", "text": "x"}]
    )
    await _stop_run(db_session, rid)
    r = await client.patch(
        f"/api/runs/{rid}/stages/script_draft/elements/NOPE",
        json={"patch": {"text": "y"}},
    )
    assert r.status_code == 404


async def test_patch_element_unsupported_stage_returns_422(client) -> None:
    _, deliv = await _make_project(client)
    r = await client.post(f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": []})
    rid = r.json()["id"]
    bad = await client.patch(
        f"/api/runs/{rid}/stages/not_a_real_stage/elements/L1",
        json={"patch": {"text": "y"}},
    )
    assert bad.status_code == 422


async def test_patch_element_cross_tenant_returns_404(
    client, other_tenant_client, db_session
) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session, deliv, "script_draft", [{"id": "L1", "text": "x"}]
    )
    bleed = await other_tenant_client.patch(
        f"/api/runs/{rid}/stages/script_draft/elements/L1",
        json={"patch": {"text": "y"}},
    )
    assert bleed.status_code == 404


# ─── Element RESTAMP (Phase A.3) ─────────────────────────────────────────────


async def _seed_run_with_full_dag(db_session, deliv_id: str, script_lines: list[dict]):
    """Seed a generic_v1 run with all 4 stages completed.

    brief_intake (no elements) → script_draft (lines) → storyboard_draft (shots)
    → video_render (no elements). Returns run_id (str).
    """
    from uuid import UUID as _UUID

    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository

    run_repo = RunRepository(db_session)
    run = await run_repo.create(
        deliverable_id=_UUID(deliv_id),
        blueprint_id="generic_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    so_repo = StageOutputRepository(db_session)
    for stage in ("brief_intake", "script_draft", "storyboard_draft", "video_render"):
        await so_repo.upsert_pending(run.id, stage)
    await db_session.commit()
    await so_repo.mark_completed(run.id, "brief_intake", {"summary": "ok"})
    await so_repo.mark_completed(run.id, "script_draft", {"lines": script_lines})
    await so_repo.mark_completed(
        run.id, "storyboard_draft", {"shots": [{"id": "S1", "description": "old"}]}
    )
    await so_repo.mark_completed(run.id, "video_render", {"url": "https://example/old.mp4"})
    await db_session.commit()
    return str(run.id)


async def test_restamp_element_replaces_and_marks_downstream_pending(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_run_with_full_dag(
        db_session,
        deliv,
        [
            {"id": "L1", "text": "old", "seconds": 5.0, "extra": "drop-me"},
            {"id": "L2", "text": "keep", "seconds": 3.0},
        ],
    )
    await _stop_run(db_session, rid)
    r = await client.post(
        f"/api/runs/{rid}/stages/script_draft/elements/L1/restamp",
        json={"value": {"text": "new", "seconds": 7.0}},
    )
    assert r.status_code == 200, r.text
    output = r.json()["output"]
    lines = output["lines"]
    # L1 fully replaced (no `extra` key) but id preserved.
    assert lines[0] == {"id": "L1", "text": "new", "seconds": 7.0}
    assert lines[1] == {"id": "L2", "text": "keep", "seconds": 3.0}
    # The restamped stage itself stays completed.
    assert r.json()["status"] == "completed"
    # Downstream stages are now pending with no output.
    detail = await client.get(f"/api/runs/{rid}")
    assert detail.status_code == 200
    by_stage = {s["stage_name"]: s for s in detail.json()["stage_outputs"]}
    assert by_stage["brief_intake"]["status"] == "completed"  # upstream untouched
    assert by_stage["storyboard_draft"]["status"] == "pending"
    assert by_stage["storyboard_draft"]["output"] is None
    assert by_stage["video_render"]["status"] == "pending"
    assert by_stage["video_render"]["output"] is None


async def test_restamp_element_id_is_preserved(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_run_with_full_dag(
        db_session, deliv, [{"id": "L1", "text": "x", "seconds": 2.0}]
    )
    await _stop_run(db_session, rid)
    r = await client.post(
        f"/api/runs/{rid}/stages/script_draft/elements/L1/restamp",
        json={"value": {"id": "EVIL", "text": "y", "seconds": 4.0}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["output"]["lines"][0]["id"] == "L1"


async def test_restamp_element_unknown_id_returns_404(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_run_with_full_dag(
        db_session, deliv, [{"id": "L1", "text": "x", "seconds": 2.0}]
    )
    await _stop_run(db_session, rid)
    r = await client.post(
        f"/api/runs/{rid}/stages/script_draft/elements/NOPE/restamp",
        json={"value": {"text": "y"}},
    )
    assert r.status_code == 404


async def test_restamp_element_unsupported_stage_returns_422(client) -> None:
    _, deliv = await _make_project(client)
    r = await client.post(f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": []})
    rid = r.json()["id"]
    bad = await client.post(
        f"/api/runs/{rid}/stages/not_a_real_stage/elements/L1/restamp",
        json={"value": {"text": "y"}},
    )
    assert bad.status_code == 422


async def test_restamp_element_cross_tenant_returns_404(
    client, other_tenant_client, db_session
) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_run_with_full_dag(
        db_session, deliv, [{"id": "L1", "text": "x", "seconds": 2.0}]
    )
    bleed = await other_tenant_client.post(
        f"/api/runs/{rid}/stages/script_draft/elements/L1/restamp",
        json={"value": {"text": "y"}},
    )
    assert bleed.status_code == 404


async def test_restamp_element_running_run_returns_409(client, db_session) -> None:
    from sqlalchemy import update as _update

    from src.database.models import Run as _Run

    _, deliv = await _make_project(client)
    rid = await _seed_run_with_full_dag(
        db_session, deliv, [{"id": "L1", "text": "x", "seconds": 2.0}]
    )
    await db_session.execute(_update(_Run).where(_Run.id == rid).values(status="running"))
    await db_session.commit()
    r = await client.post(
        f"/api/runs/{rid}/stages/script_draft/elements/L1/restamp",
        json={"value": {"text": "y"}},
    )
    assert r.status_code == 409


# ─── Course chapter title + thumbnail overrides ──────────────────────────────


def _stitch_output() -> dict:
    return {
        "course_artifact_url": "http://minio.test:8389/w2v-courses/c.mp4",
        "artifact_url": "http://minio.test:8389/w2v-courses/c.mp4",
        "duration_seconds": 120.0,
        "lesson_count": 2,
        "thumbnail_url": "http://minio.test:8389/w2v-orchestrator/auto.png",
        "chapters": [
            {
                "sequence": 1,
                "title": "Intro",
                "start_seconds": 0.0,
                "duration_seconds": 60.0,
                "level": 0,
                "parent_sequence": None,
                "lesson_project_id": "p1",
                "chapter_key": "p1",
            },
            {
                "sequence": 2,
                "title": "Sub",
                "start_seconds": 10.0,
                "duration_seconds": 50.0,
                "level": 1,
                "parent_sequence": 1,
                "lesson_project_id": "p1",
                "chapter_key": "p1#1",
            },
            {
                "sequence": 3,
                "title": "Core",
                "start_seconds": 60.0,
                "duration_seconds": 60.0,
                "level": 0,
                "parent_sequence": None,
                "lesson_project_id": "p2",
                "chapter_key": "p2",
            },
        ],
    }


async def _seed_course_stitch(db_session, deliv_id: str, output: dict | None = None) -> str:
    from uuid import UUID as _UUID

    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository

    run_repo = RunRepository(db_session)
    run = await run_repo.create(
        deliverable_id=_UUID(deliv_id),
        blueprint_id="course_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    so_repo = StageOutputRepository(db_session)
    await so_repo.upsert_pending(run.id, "course_stitch")
    await db_session.commit()
    await so_repo.mark_completed(run.id, "course_stitch", output or _stitch_output())
    await db_session.commit()
    return str(run.id)


async def test_patch_chapters_overrides_title_and_stores_map(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_course_stitch(db_session, deliv)
    r = await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/chapters",
        json={"overrides": {"p2": "Deep Dive"}},
    )
    assert r.status_code == 200, r.text
    output = r.json()["output"]
    assert output["chapter_title_overrides"] == {"p2": "Deep Dive"}
    by_seq = {c["sequence"]: c for c in output["chapters"]}
    assert by_seq[3]["title"] == "Deep Dive"  # level-0 p2 rewritten
    assert by_seq[1]["title"] == "Intro"  # untouched (not in this override)
    assert by_seq[2]["title"] == "Sub"  # level-1 untouched (not in this override)


async def test_patch_chapters_trims_and_merges_across_calls(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_course_stitch(db_session, deliv)
    await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/chapters",
        json={"overrides": {"p1": "  Welcome  "}},
    )
    r = await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/chapters",
        json={"overrides": {"p2": "Core II"}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["output"]["chapter_title_overrides"] == {"p1": "Welcome", "p2": "Core II"}


async def test_patch_chapters_overrides_slide_subchapter_title(client, db_session) -> None:
    """A level-1 sub-chapter is editable by its ``{lesson_project_id}#{ordinal}``
    chapter_key; the edit is denormalized into the level-1 row + stored, while the
    parent lesson (same lesson_project_id) is left untouched."""
    _, deliv = await _make_project(client)
    rid = await _seed_course_stitch(db_session, deliv)
    r = await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/chapters",
        json={"overrides": {"p1#1": "Renamed Slide"}},
    )
    assert r.status_code == 200, r.text
    output = r.json()["output"]
    assert output["chapter_title_overrides"] == {"p1#1": "Renamed Slide"}
    by_seq = {c["sequence"]: c for c in output["chapters"]}
    assert by_seq[2]["title"] == "Renamed Slide"  # level-1 slide rewritten
    assert by_seq[1]["title"] == "Intro"  # parent lesson (same pid) untouched


async def test_patch_chapters_unknown_slide_key_404(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_course_stitch(db_session, deliv)
    r = await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/chapters",
        json={"overrides": {"p1#9": "X"}},  # no 9th slide in lesson p1
    )
    assert r.status_code == 404, r.text


async def test_patch_chapters_legacy_rows_without_chapter_key(client, db_session) -> None:
    """Runs stitched before ``chapter_key`` existed: level-0 stays editable via
    its lesson_project_id fallback; the keyless level-1 row is not addressable."""
    legacy = _stitch_output()
    for c in legacy["chapters"]:
        c.pop("chapter_key", None)
    _, deliv = await _make_project(client)
    rid = await _seed_course_stitch(db_session, deliv, output=legacy)
    ok = await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/chapters",
        json={"overrides": {"p1": "Welcome"}},
    )
    assert ok.status_code == 200, ok.text
    by_seq = {c["sequence"]: c for c in ok.json()["output"]["chapters"]}
    assert by_seq[1]["title"] == "Welcome"
    # the legacy level-1 row has no stable key → 404, not silently applied
    bad = await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/chapters",
        json={"overrides": {"p1#1": "Nope"}},
    )
    assert bad.status_code == 404, bad.text


async def test_patch_chapters_hides_subchapter(client, db_session) -> None:
    """Hiding a level-1 sub-chapter persists its key in ``hidden_chapter_keys`` and
    denormalizes ``hidden: True`` onto that row (lesson rows stay un-hidden)."""
    _, deliv = await _make_project(client)
    rid = await _seed_course_stitch(db_session, deliv)
    r = await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/chapters",
        json={"overrides": {}, "hidden": ["p1#1"]},
    )
    assert r.status_code == 200, r.text
    output = r.json()["output"]
    assert output["hidden_chapter_keys"] == ["p1#1"]
    by_seq = {c["sequence"]: c for c in output["chapters"]}
    assert by_seq[2]["hidden"] is True  # level-1 sub now hidden
    assert by_seq[1]["hidden"] is False  # lesson p1 backbone
    assert by_seq[3]["hidden"] is False  # lesson p2 backbone


async def test_patch_chapters_unhide_clears(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_course_stitch(db_session, deliv)
    await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/chapters",
        json={"overrides": {}, "hidden": ["p1#1"]},
    )
    r = await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/chapters",
        json={"overrides": {}, "hidden": []},
    )
    assert r.status_code == 200, r.text
    output = r.json()["output"]
    assert output["hidden_chapter_keys"] == []
    by_seq = {c["sequence"]: c for c in output["chapters"]}
    assert by_seq[2]["hidden"] is False


async def test_patch_chapters_title_only_leaves_hidden_untouched(client, db_session) -> None:
    """A patch that omits ``hidden`` (None) must not wipe an existing hide set."""
    _, deliv = await _make_project(client)
    rid = await _seed_course_stitch(db_session, deliv)
    await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/chapters",
        json={"overrides": {}, "hidden": ["p1#1"]},
    )
    r = await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/chapters",
        json={"overrides": {"p2": "Deep Dive"}},  # no hidden key → leave untouched
    )
    assert r.status_code == 200, r.text
    output = r.json()["output"]
    assert output["hidden_chapter_keys"] == ["p1#1"]
    by_seq = {c["sequence"]: c for c in output["chapters"]}
    assert by_seq[2]["hidden"] is True
    assert by_seq[3]["title"] == "Deep Dive"


async def test_patch_chapters_hide_lesson_key_404(client, db_session) -> None:
    """A level-0 lesson marker is the backbone and is never hideable."""
    _, deliv = await _make_project(client)
    rid = await _seed_course_stitch(db_session, deliv)
    r = await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/chapters",
        json={"overrides": {}, "hidden": ["p1"]},
    )
    assert r.status_code == 404, r.text


async def test_patch_chapters_hide_unknown_key_404(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_course_stitch(db_session, deliv)
    r = await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/chapters",
        json={"overrides": {}, "hidden": ["p1#9"]},  # no 9th slide
    )
    assert r.status_code == 404, r.text


async def test_patch_chapters_unknown_pid_404(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_course_stitch(db_session, deliv)
    r = await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/chapters",
        json={"overrides": {"nope": "X"}},
    )
    assert r.status_code == 404, r.text


async def test_patch_chapters_blank_title_422(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_course_stitch(db_session, deliv)
    r = await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/chapters",
        json={"overrides": {"p1": "   "}},
    )
    assert r.status_code == 422, r.text


async def test_patch_chapters_missing_stage_404(client) -> None:
    _, deliv = await _make_project(client)
    r = await client.post(f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": []})
    rid = r.json()["id"]
    bad = await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/chapters",
        json={"overrides": {"p1": "X"}},
    )
    assert bad.status_code == 404, bad.text


async def test_patch_chapters_cross_tenant_404(client, other_tenant_client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_course_stitch(db_session, deliv)
    bleed = await other_tenant_client.patch(
        f"/api/runs/{rid}/stages/course_stitch/chapters",
        json={"overrides": {"p1": "X"}},
    )
    assert bleed.status_code == 404, bleed.text


async def test_patch_thumbnail_set_then_clear_is_lossless(client, db_session) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_course_stitch(db_session, deliv)
    # The real flow: POST /uploads/thumbnail stages a tenant-scoped object OUTSIDE
    # any run, then this PATCH adopts it. A URL that is neither this run's own
    # artifact nor this tenant's own upload is refused (see the 422 tests below),
    # so the override must carry the upload key shape the uploader actually mints.
    picked = _tenant_upload_url(client.tenant_id, "picked.png")
    r = await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/thumbnail",
        json={"thumbnail_url": picked},
    )
    assert r.status_code == 200, r.text
    out = r.json()["output"]
    # override stored (proxied on the way out); the auto cover key is left intact.
    assert "thumbnail_override_url" in out
    assert "thumbnail_url" in out
    # clear → override removed, auto cover still present (lossless revert).
    r2 = await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/thumbnail",
        json={"thumbnail_url": None},
    )
    assert r2.status_code == 200, r2.text
    out2 = r2.json()["output"]
    assert "thumbnail_override_url" not in out2
    assert "thumbnail_url" in out2


@pytest.mark.parametrize(
    "url",
    ["javascript:alert(1)"],
)
async def test_patch_thumbnail_rejects_non_artifact_url_422(client, db_session, url) -> None:
    _, deliv = await _make_project(client)
    rid = await _seed_course_stitch(db_session, deliv)
    r = await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/thumbnail",
        json={"thumbnail_url": url},
    )
    assert r.status_code == 422, r.text


# ─── Write-path artifact-ref ownership (the proxy's read gate is circular
#     without it) ───────────────────────────────────────────────────────────────
#
# The artifact proxy authorises a download by asking "does this run's own
# persisted stage output reference that object?". Every route below lets a caller
# put a value INTO that output, so without a write-side check the caller answers
# the proxy's question for itself: own one throwaway run, patch the victim's
# audiobook URL into your own output, and `_run_artifact_refs` returns it.
#
# `tests/services/test_artifact_ref_guard.py` covers the rule's shape; these
# drive the actual exploit through the real routes and assert the DB never took
# the value.

_VICTIM_RUN = "00000000-0000-0000-0000-0000000000ff"
_VICTIM_URL = (
    f"http://minio.test:8389/w2v-audiobooks/artifacts/{_VICTIM_RUN}/audiobook_stitch/book.mp3"
)


def _tenant_upload_url(tenant_id: str, filename: str) -> str:
    """The URL shape `POST /uploads/thumbnail` mints: tenant-scoped, own bucket."""
    return (
        f"http://minio.test:8389/w2v-orchestrator/artifacts/thumbnails/"
        f"{tenant_id}/course_thumbnail_override/{filename}"
    )


async def test_patch_element_cannot_inject_a_foreign_runs_artifact(client, db_session) -> None:
    """THE exploit, end to end: it must 422 and leave the output untouched.

    Mutation killed: removing the `_reject_unowned_refs` call from
    `patch_stage_element` — the ref is persisted, and from that moment the
    artifact proxy treats the victim's audiobook as this run's own property.
    """
    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session, deliv, "script_draft", [{"id": "L1", "text": "x"}]
    )
    await _stop_run(db_session, rid)
    r = await client.patch(
        f"/api/runs/{rid}/stages/script_draft/elements/L1",
        json={"patch": {"artifact_url": _VICTIM_URL}},
    )
    assert r.status_code == 422, r.text
    assert "unowned artifact reference" in r.json()["detail"]

    # The write must not have half-landed: read the output back through the API.
    after = await client.get(f"/api/runs/{rid}/stages/script_draft/output")
    assert after.status_code == 200
    assert _VICTIM_RUN not in after.text


async def test_patch_element_cannot_bury_a_foreign_artifact_in_a_nested_value(
    client, db_session
) -> None:
    """`patch` is a free-form dict — the ref can sit at any depth.

    Mutation killed: checking only top-level values instead of walking the payload
    with the same collector the proxy's ref set is built from.
    """
    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session, deliv, "script_draft", [{"id": "L1", "text": "x"}]
    )
    await _stop_run(db_session, rid)
    r = await client.patch(
        f"/api/runs/{rid}/stages/script_draft/elements/L1",
        json={"patch": {"meta": {"clips": [{"variants": [{"url": _VICTIM_URL}]}]}}},
    )
    assert r.status_code == 422, r.text


async def test_restamp_element_cannot_inject_a_foreign_runs_artifact(client, db_session) -> None:
    """Restamp replaces the element outright — the easier injection point.

    Mutation killed: guarding only `patch_stage_element` and forgetting restamp,
    which is exactly how a write-path fix ends up half-applied.
    """
    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session, deliv, "script_draft", [{"id": "L1", "text": "x"}]
    )
    await _stop_run(db_session, rid)
    r = await client.post(
        f"/api/runs/{rid}/stages/script_draft/elements/L1/restamp",
        json={"value": {"text": "y", "artifact_url": _VICTIM_URL}},
    )
    assert r.status_code == 422, r.text


async def test_patch_thumbnail_cannot_adopt_a_foreign_runs_artifact(client, db_session) -> None:
    """Artifact SHAPE is not ownership; `_is_presignable_artifact_url` only ever
    answered the shape question, and this URL passes it.

    Mutation killed: relying on `_is_presignable_artifact_url` alone (a 200 here)
    — the override lands in the stitch output and is re-signed by the YouTube
    passthrough on top of being proxy-servable.
    """
    _, deliv = await _make_project(client)
    rid = await _seed_course_stitch(db_session, deliv)
    r = await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/thumbnail",
        json={"thumbnail_url": _VICTIM_URL},
    )
    assert r.status_code == 422, r.text
    assert "unowned artifact reference" in r.json()["detail"]


async def test_patch_thumbnail_cannot_adopt_another_tenants_upload(client, db_session) -> None:
    """The upload arm is scoped to the CALLER's tenant, matched as a segment.

    Mutation killed: dropping the tenant-slot comparison in `is_own_tenant_upload`
    (or making it a `startswith`) — one tenant adopts another's staged upload.
    """
    from uuid import uuid4 as _uuid4

    _, deliv = await _make_project(client)
    rid = await _seed_course_stitch(db_session, deliv)
    r = await client.patch(
        f"/api/runs/{rid}/stages/course_stitch/thumbnail",
        json={"thumbnail_url": _tenant_upload_url(str(_uuid4()), "someone-elses.png")},
    )
    assert r.status_code == 422, r.text


async def test_patch_element_may_repoint_at_the_runs_own_artifact(client, db_session) -> None:
    """The guard must not break legitimate editing.

    A patch naming an artifact the run ALREADY carries is the normal case (that is
    what element editing is for) and must still 200. Mutation killed: a guard that
    rejects every minio ref outright, which would pass every 422 test above while
    breaking the product.
    """
    _, deliv = await _make_project(client)
    owned = "http://minio.test:8389/w2v-orchestrator/artifacts/mine/narrate/clip.wav"
    rid = await _seed_stage_with_elements(
        db_session, deliv, "script_draft", [{"id": "L1", "artifact_url": owned}]
    )
    await _stop_run(db_session, rid)
    r = await client.patch(
        f"/api/runs/{rid}/stages/script_draft/elements/L1",
        json={"patch": {"artifact_url": owned, "text": "new caption"}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["output"]["lines"][0]["text"] == "new caption"


async def test_patch_element_with_no_artifact_refs_is_unaffected(client, db_session) -> None:
    """Ordinary text edits — the overwhelming majority — are not policed at all."""
    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(
        db_session, deliv, "script_draft", [{"id": "L1", "text": "old"}]
    )
    await _stop_run(db_session, rid)
    r = await client.patch(
        f"/api/runs/{rid}/stages/script_draft/elements/L1",
        json={"patch": {"text": "new", "external": "https://cdn.example.com/i.png"}},
    )
    assert r.status_code == 200, r.text


# ─── Soft-CC caption download (GET /api/runs/{id}/captions.srt) ───────────────


async def _seed_captioned_run(db_session, deliv_id: str, stages: dict[str, dict]) -> str:
    """Seed a run with several completed stage outputs (for the captions route)."""
    from uuid import UUID as _UUID

    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository

    run_repo = RunRepository(db_session)
    run = await run_repo.create(
        deliverable_id=_UUID(deliv_id),
        blueprint_id="promo_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    so_repo = StageOutputRepository(db_session)
    for stage, output in stages.items():
        await so_repo.upsert_pending(run.id, stage)
        await so_repo.mark_completed(run.id, stage, output)
    await db_session.commit()
    return str(run.id)


async def test_get_run_captions_returns_srt(client, db_session) -> None:
    """A captionable run returns its soft-CC track as a downloadable SRT."""
    _, deliv = await _make_project(client)
    rid = await _seed_captioned_run(
        db_session,
        deliv,
        {
            "promo_script": {"lines": [{"id": "l1", "text": "Hello captions"}]},
            "promo_narrate": {"clips": [{"line_id": "l1", "seconds": 2.0}]},
        },
    )

    r = await client.get(f"/api/runs/{rid}/captions.srt")

    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/x-subrip")
    # Named after the project ("for run"), not a bare "captions.srt": this header
    # overrides the frontend link's `download` attribute.
    assert r.headers["content-disposition"] == 'attachment; filename="for-run-captions.srt"'
    body = r.text
    assert "-->" in body
    assert "Hello captions" in body


async def test_get_run_captions_filename_keeps_a_non_ascii_title(client, db_session) -> None:
    """A non-ASCII project title gets an ASCII ``filename`` plus an RFC 5987
    ``filename*`` that keeps it intact — and nothing from the title can break out
    of the header (quotes, semicolons, CR/LF are folded away)."""
    r = await client.post(
        "/api/projects",
        json={
            "type": "generic",
            "title": 'Café "Launch"; Q4\r\nX-Evil: 1',
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
    deliv = detail.json()["deliverables"][0]["id"]
    rid = await _seed_captioned_run(
        db_session,
        deliv,
        {
            "promo_script": {"lines": [{"id": "l1", "text": "Hello"}]},
            "promo_narrate": {"clips": [{"line_id": "l1", "seconds": 2.0}]},
        },
    )
    resp = await client.get(f"/api/runs/{rid}/captions.srt")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-disposition"] == (
        'attachment; filename="cafe-launch-q4-x-evil-1-captions.srt"; '
        "filename*=UTF-8''caf%C3%A9-launch-q4-x-evil-1-captions.srt"
    )
    assert "x-evil" not in {k.lower() for k in resp.headers}


async def test_get_run_captions_404_when_no_captions(client, db_session) -> None:
    """A run with no captionable narration (audiobook) → 404."""
    _, deliv = await _make_project(client)
    rid = await _seed_captioned_run(
        db_session,
        deliv,
        {"audiobook_stitch": {"artifact_url": "http://minio.test:8389/w2v-audiobooks/a.mp3"}},
    )

    r = await client.get(f"/api/runs/{rid}/captions.srt")
    assert r.status_code == 404


async def test_get_run_captions_cross_tenant_returns_404(
    client, other_tenant_client, db_session
) -> None:
    """Tenant B cannot download tenant A's run captions."""
    _, deliv = await _make_project(client)
    rid = await _seed_captioned_run(
        db_session,
        deliv,
        {
            "promo_script": {"lines": [{"id": "l1", "text": "secret"}]},
            "promo_narrate": {"clips": [{"line_id": "l1", "seconds": 2.0}]},
        },
    )

    bleed = await other_tenant_client.get(f"/api/runs/{rid}/captions.srt")
    assert bleed.status_code == 404


def _roles_seeded_runs_app(db_session_factory, *, tenant_id: str, roles: list[str]):
    """A runs-router app whose middleware seeds `request.state.roles`.

    The conftest `client` fixture seeds no roles, so a super-admin path needs its
    own app (same idiom as tests/api/routes/test_scope_switch.py). Only
    `session_factory` is needed — the captions route is a pure read.
    """
    from uuid import uuid4

    from fastapi import FastAPI
    from starlette.middleware.base import BaseHTTPMiddleware

    from src.api.routes import runs_router

    test_app = FastAPI(title="Test W2V Orchestrator (roles-seeded)")
    test_app.state.session_factory = db_session_factory
    test_app.include_router(runs_router)
    subject_id = str(uuid4())

    class _SeedRoles(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.tenant_id = tenant_id
            request.state.roles = roles
            request.state.subject_id = subject_id
            return await call_next(request)

    test_app.add_middleware(_SeedRoles)
    return test_app


async def test_captions_active_tenant_param_super_vs_non_super(
    client, db_session, db_session_factory
) -> None:
    """`<a href download>` cannot send X-Active-Tenant-Id — the query param must work.

    ADOPTION proof for GET /api/runs/{id}/captions.srt: the same URL is driven
    twice against the same seeded run, differing ONLY in the caller's roles.
    super_admin → 200 with tenant A's caption text; tenant_admin → 404. A route
    still on the header-only `TenantIdDep` would 404 in both arms, and a
    mis-seeded fixture would 200 in both — the delta is the proof.
    """
    from uuid import uuid4

    from httpx import ASGITransport, AsyncClient

    _, deliv = await _make_project(client)
    rid = await _seed_captioned_run(
        db_session,
        deliv,
        {
            "promo_script": {"lines": [{"id": "l1", "text": "Hello captions"}]},
            "promo_narrate": {"clips": [{"line_id": "l1", "seconds": 2.0}]},
        },
    )
    admin_claim = str(uuid4())
    url = f"/api/runs/{rid}/captions.srt?active_tenant={client.tenant_id}"

    super_app = _roles_seeded_runs_app(
        db_session_factory, tenant_id=admin_claim, roles=["super_admin"]
    )
    async with AsyncClient(
        transport=ASGITransport(app=super_app), base_url="http://testserver"
    ) as c:
        allowed = await c.get(url)

    plain_app = _roles_seeded_runs_app(
        db_session_factory, tenant_id=admin_claim, roles=["tenant_admin"]
    )
    async with AsyncClient(
        transport=ASGITransport(app=plain_app), base_url="http://testserver"
    ) as c:
        denied = await c.get(url)

    assert allowed.status_code == 200, allowed.text
    assert "Hello captions" in allowed.text  # real body, not just a status
    assert allowed.headers["content-type"].startswith("application/x-subrip")

    assert denied.status_code == 404  # param ignored for non-super → no escape
    assert denied.json()["detail"] == "run not found"


# ─── Lightweight live-status poll (GET /api/runs/{id}/status) ─────────────────


async def test_get_run_status_returns_stage_statuses_without_output(client, db_session) -> None:
    """The live-monitor poll endpoint returns run + per-stage lifecycle state
    and DROPS every heavy field (stage ``output`` blob / ``config_snapshot`` /
    ``usage_total``) that ``GET /{id}`` ships. Polling the heavy detail every
    few seconds froze the browser on long runs — this is the trimmed shape."""
    _, deliv = await _make_project(client)
    rid = await _seed_run_with_full_dag(
        db_session,
        deliv,
        [{"id": "L1", "text": "x", "seconds": 2.0}],
    )

    r = await client.get(f"/api/runs/{rid}/status")
    assert r.status_code == 200, r.text
    # no-store is REQUIRED: this poll observes a running→completed transition; a
    # cached pre-completion response would freeze the poller on stale `running`
    # (the e2e website-review 20-min hang, 2026-07-03). Regression guard.
    assert r.headers.get("cache-control") == "no-store", r.headers
    body = r.json()

    assert body["id"] == rid
    assert body["status"] in ("pending", "running", "completed", "failed", "cancelled", "paused")

    # Per-stage lifecycle state is present, keyed the same way the reconciler reads it.
    by_stage = {s["stage_name"]: s for s in body["stage_outputs"]}
    assert by_stage["script_draft"]["status"] == "completed"
    assert by_stage["video_render"]["status"] == "completed"
    for s in body["stage_outputs"]:
        # `attempts` is a scalar int, not one of the payload-growing fields this
        # guard exists to keep out — it is the only signal that distinguishes a
        # slow stage from a stage on its second go at the same work.
        assert set(s.keys()) == {
            "stage_name",
            "status",
            "started_at",
            "completed_at",
            "attempts",
        }, f"status poll must not ship heavy stage fields; saw {sorted(s.keys())}"

    # The heavy, payload-growing fields RunDetail carries must be absent here.
    assert "stage_outputs" in body
    assert "config_snapshot" not in body
    assert "usage_total" not in body
    assert "output" not in body


async def test_get_run_status_404_on_unknown_run(client) -> None:
    bogus = "00000000-0000-0000-0000-000000000000"
    r = await client.get(f"/api/runs/{bogus}/status")
    assert r.status_code == 404


async def test_get_run_status_cross_tenant_returns_404(
    client, other_tenant_client, db_session
) -> None:
    """Tenant B cannot read tenant A's run status."""
    _, deliv = await _make_project(client)
    rid = await _seed_run_with_full_dag(
        db_session, deliv, [{"id": "L1", "text": "x", "seconds": 2.0}]
    )
    bleed = await other_tenant_client.get(f"/api/runs/{rid}/status")
    assert bleed.status_code == 404


# ─── audit-report PDF (GET /api/runs/{run_id}/report/pdf) ─────────────────────

_SEO_SCAN_OUTPUT = {
    "url": "https://acme.test",
    "overall_score": 93.0,
    "grade": "A",
    "groups": [{"group": "on_page", "score": 90.0, "weight": 0.2, "count": 1}],
    "quick_wins": [{"title": "Add OpenGraph tags", "severity": "medium"}],
}


async def _seed_scan_stage(db_session_factory, run_uuid, stage_name, output) -> None:
    from uuid import uuid7

    from src.database.models import StageOutput

    async with db_session_factory() as s:
        s.add(
            StageOutput(
                id=uuid7(),
                run_id=run_uuid,
                stage_name=stage_name,
                status="completed",
                output=output,
            )
        )
        await s.commit()


async def test_report_pdf_renders_from_seo_scan(client, db_session_factory) -> None:
    rid, run_uuid = await _make_run_row(client, db_session_factory, status="completed")
    await _seed_scan_stage(db_session_factory, run_uuid, "seo_audit_scan", _SEO_SCAN_OUTPUT)

    fake_design = AsyncMock()
    fake_design.render_pdf.return_value = b"%PDF-1.4 fake"
    with patch("src.api.routes.runs.read.DesignClient", return_value=fake_design):
        r = await client.get(f"/api/runs/{rid}/report/pdf")

    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/pdf"
    assert r.content == b"%PDF-1.4 fake"
    # Filename names the audit type (dispatched on the matched stage) and the host.
    assert r.headers["content-disposition"] == 'attachment; filename="seo_audit-acme.test.pdf"'
    # The scan output was turned into the SEO template and handed to the renderer.
    html_arg = fake_design.render_pdf.call_args.args[0]
    assert "<h1>Search &amp; AI discoverability audit</h1>" in html_arg
    assert 'data-section="serp-preview"' in html_arg
    assert "Add OpenGraph tags" in html_arg
    # The per-request client is closed, not leaked (one httpx pool per download).
    fake_design.close.assert_awaited_once()


async def test_report_pdf_renders_from_cro_scan(client, db_session_factory) -> None:
    rid, run_uuid = await _make_run_row(client, db_session_factory, status="completed")
    cro = {
        "url": "https://acme.test",
        "overall_score": 62.0,
        "grade": "C",
        "subsections": [{"section": "hero_clarity", "score": 80.0, "weight": 0.2, "count": 1}],
    }
    await _seed_scan_stage(db_session_factory, run_uuid, "cro_audit_scan", cro)

    fake_design = AsyncMock()
    fake_design.render_pdf.return_value = b"%PDF-1.4 cro"
    with patch("src.api.routes.runs.read.DesignClient", return_value=fake_design):
        r = await client.get(f"/api/runs/{rid}/report/pdf")

    assert r.status_code == 200, r.text
    assert r.headers["content-disposition"] == 'attachment; filename="cro_audit-acme.test.pdf"'
    html_arg = fake_design.render_pdf.call_args.args[0]
    assert "<h1>Conversion audit</h1>" in html_arg
    assert '<svg data-chart="funnel"' in html_arg


async def test_report_pdf_renders_from_marketing_scan(client, db_session_factory) -> None:
    rid, run_uuid = await _make_run_row(client, db_session_factory, status="completed")
    marketing = {
        "url": "https://acme.test",
        "overall_score": 74.0,
        "grade": "B",
        "dimensions": [{"dimension": "growth", "score": 68.0, "weight": 0.2, "count": 2}],
        # ``dimension`` is required on the persisted MarketingFindingRef; the
        # scorecard lists only own-dimension (brand/competitive/growth) findings.
        "quick_wins": [
            {
                "check_id": "mkt-growth-no-analytics",
                "dimension": "growth",
                "title": "Add analytics tracking",
                "severity": "high",
            }
        ],
    }
    await _seed_scan_stage(db_session_factory, run_uuid, "marketing_scorecard_scan", marketing)

    fake_design = AsyncMock()
    fake_design.render_pdf.return_value = b"%PDF-1.4 mkt"
    with patch("src.api.routes.runs.read.DesignClient", return_value=fake_design):
        r = await client.get(f"/api/runs/{rid}/report/pdf")

    assert r.status_code == 200, r.text
    assert (
        r.headers["content-disposition"]
        == 'attachment; filename="marketing_scorecard-acme.test.pdf"'
    )
    html_arg = fake_design.render_pdf.call_args.args[0]
    assert "<h1>Go-to-market scorecard</h1>" in html_arg
    assert '<svg data-chart="radar"' in html_arg
    assert "growth" in html_arg
    assert "Add analytics tracking" in html_arg


async def test_audit_scan_stages_cover_every_audit_family() -> None:
    # The PDF whitelist is derived from AUDIT_FAMILIES so a new scorecard type can't
    # drift out of the download surface.
    from src.api.routes.runs import _AUDIT_SCAN_STAGES
    from src.services.blueprints.audit_family import AUDIT_FAMILIES

    assert set(_AUDIT_SCAN_STAGES) == {f"{f.project_type}_scan" for f in AUDIT_FAMILIES}
    assert "marketing_scorecard_scan" in _AUDIT_SCAN_STAGES


async def test_report_pdf_404_when_no_audit_scan(client, db_session_factory) -> None:
    # A non-audit run (no seo/cro scan stage) has no scorecard → 404.
    rid, _ = await _make_run_row(client, db_session_factory, status="completed")
    r = await client.get(f"/api/runs/{rid}/report/pdf")
    assert r.status_code == 404


async def test_report_pdf_cross_tenant_returns_404(
    client, other_tenant_client, db_session_factory
) -> None:
    rid, run_uuid = await _make_run_row(client, db_session_factory, status="completed")
    await _seed_scan_stage(db_session_factory, run_uuid, "seo_audit_scan", _SEO_SCAN_OUTPUT)
    bleed = await other_tenant_client.get(f"/api/runs/{rid}/report/pdf")
    assert bleed.status_code == 404


# ─── Sweep-to-cancelled stage rows must stay READABLE ──────────────────────


async def test_get_run_returns_200_when_stage_rows_were_swept_to_cancelled(
    client, db_session_factory
) -> None:
    """Regression (prod 500): a hard-failed run sweeps its leftover non-terminal
    stage rows to ``'cancelled'`` (``StageOutputRepository.bulk_cancel_non_terminal``,
    admitted by migration 0059's widened ``ck_stage_outputs_status``). The API's
    ``StageStatus`` Literal never learned that value, so ``StageOutputResponse``
    raised a pydantic ValidationError inside ``_stage_response`` and the run detail
    endpoint 500'd — the run became permanently unreadable in the Studio."""
    from uuid import uuid7

    from src.database.models import StageOutput
    from src.database.repositories.stage_output_repository import StageOutputRepository

    rid, run_uuid = await _make_run_row(client, db_session_factory, status="failed")
    async with db_session_factory() as s:
        s.add(
            StageOutput(
                id=uuid7(),
                run_id=run_uuid,
                stage_name="script_draft",
                status="completed",
                output={"script": "ok"},
            )
        )
        s.add(
            StageOutput(
                id=uuid7(),
                run_id=run_uuid,
                stage_name="storyboard_draft",
                status="pending",
                output=None,
            )
        )
        s.add(
            StageOutput(
                id=uuid7(),
                run_id=run_uuid,
                stage_name="video_render",
                status="running",
                output=None,
            )
        )
        await s.commit()

    # The real sweep path the run-failure site takes.
    async with db_session_factory() as s:
        swept = await StageOutputRepository(s).bulk_cancel_non_terminal(run_uuid)
        await s.commit()
    assert swept == 2, f"expected the two non-terminal rows to be swept; got {swept}"

    detail = await client.get(f"/api/runs/{rid}")
    assert detail.status_code == 200, detail.text
    by_stage = {so["stage_name"]: so for so in detail.json()["stage_outputs"]}
    assert by_stage["storyboard_draft"]["status"] == "cancelled"
    assert by_stage["video_render"]["status"] == "cancelled"
    assert by_stage["script_draft"]["status"] == "completed"

    # The lightweight poll projection shares the same StageStatus alias.
    lite = await client.get(f"/api/runs/{rid}/status")
    assert lite.status_code == 200, lite.text
    lite_by_stage = {so["stage_name"]: so["status"] for so in lite.json()["stage_outputs"]}
    assert lite_by_stage["storyboard_draft"] == "cancelled"


# ─── Clip spine resolution (studio reorder) ────────────────────────────────
#
# Live run 01a08f79-ab79-7359-9656-3238c07e19ff (demo_video_v1) 422'd on reorder
# because the frontend guessed the spine from output SHAPE: four of its stages
# carry an identical ``lines[]`` of the same clip ids and only ``demo_video_gen``
# is a legal reorder target. The backend now PUBLISHES the answer on run detail.


async def _make_demo_video_project(client) -> tuple[str, str]:
    """A REAL demo_video project. The blueprint the reorder machinery resolves is
    derived from the deliverable's project type, not from ``run.blueprint_id``, so
    a demo_video run hung off a ``generic`` deliverable cannot exercise the route."""
    r = await client.post(
        "/api/projects",
        json={
            "type": "demo_video",
            "title": "for clip spine",
            "brief": {"target_url": "https://app.example.com", "focus_areas": ["onboarding"]},
            "deliverables": [
                {
                    "name": "16x9",
                    "format": "16x9",
                    "captions_mode": "none",
                    "target_duration_seconds": 60,
                }
            ],
        },
    )
    assert r.status_code == 201, r.text
    project_id = r.json()["id"]
    detail = await client.get(f"/api/projects/{project_id}")
    return project_id, detail.json()["deliverables"][0]["id"]


async def _seed_demo_video_run(db_session, deliv_id: str) -> str:
    """A completed demo_video_v1 run carrying the live look-alike shape:
    ``demo_video_gen`` AND ``script_doctor`` both completed with the same
    ``lines[]``. Only the former is a legal reorder spine."""
    from uuid import UUID as _UUID

    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository

    run_repo = RunRepository(db_session)
    run = await run_repo.create(
        deliverable_id=_UUID(deliv_id),
        blueprint_id="demo_video_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    lines = [{"id": f"clip-{n}", "text": f"t{n}"} for n in (1, 3, 4, 5)]
    so_repo = StageOutputRepository(db_session)
    for stage in ("demo_video_gen", "script_doctor"):
        await so_repo.upsert_pending(run.id, stage)
    await db_session.commit()
    for stage in ("demo_video_gen", "script_doctor"):
        await so_repo.mark_completed(run.id, stage, {"lines": [dict(x) for x in lines]})
    # A reorder re-stitches a finished video, so the parent must be terminal.
    await run_repo.mark_completed(run.id)
    await db_session.commit()
    return str(run.id)


async def test_run_detail_publishes_clip_spine_stage(client, db_session) -> None:
    """The wire name is snake_case ``clip_spine_stage`` (these schemas do NOT
    camelise) and it must name the ONE legal spine, not either look-alike."""
    _, deliv = await _make_demo_video_project(client)
    rid = await _seed_demo_video_run(db_session, deliv)

    r = await client.get(f"/api/runs/{rid}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "clip_spine_stage" in body, sorted(body)
    assert body["clip_spine_stage"] == "demo_video_gen", body["clip_spine_stage"]

    # The look-alike really is present in the same payload — i.e. a client that
    # shape-sniffed ``stage_outputs`` genuinely could have picked the wrong one.
    with_lines = {
        so["stage_name"] for so in body["stage_outputs"] if (so.get("output") or {}).get("lines")
    }
    assert with_lines == {"demo_video_gen", "script_doctor"}, with_lines


async def test_run_detail_clip_spine_stage_is_none_before_the_spine_completes(
    client, db_session
) -> None:
    """No completed spine ⇒ ``null``, not a guess at a look-alike."""
    from uuid import UUID as _UUID

    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository

    _, deliv = await _make_demo_video_project(client)
    run = await RunRepository(db_session).create(
        deliverable_id=_UUID(deliv),
        blueprint_id="demo_video_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    so_repo = StageOutputRepository(db_session)
    await so_repo.upsert_pending(run.id, "script_doctor")
    await db_session.commit()
    # Only the NON-spine look-alike is completed.
    await so_repo.mark_completed(run.id, "script_doctor", {"lines": [{"id": "clip-1"}]})
    await db_session.commit()

    r = await client.get(f"/api/runs/{run.id}")
    assert r.status_code == 200, r.text
    assert r.json()["clip_spine_stage"] is None, r.json()["clip_spine_stage"]


async def test_reorder_clips_422_on_lookalike_stage_script_doctor(client, db_session) -> None:
    """``script_doctor`` carries an identical ``lines[]`` but is NOT a spine —
    the guard that produced the live 422 is correct and must not regress."""
    _, deliv = await _make_demo_video_project(client)
    rid = await _seed_demo_video_run(db_session, deliv)

    r = await client.post(
        f"/api/runs/{rid}/reorder-clips",
        json={
            "stage_name": "script_doctor",
            "ordered_element_ids": ["clip-5", "clip-1", "clip-3", "clip-4"],
        },
    )
    assert r.status_code == 422, r.text


async def test_reorder_clips_201_on_the_published_spine_stage(client, db_session) -> None:
    """The stage run detail PUBLISHES is accepted by the reorder route — the
    round-trip the studio actually performs."""
    _, deliv = await _make_demo_video_project(client)
    rid = await _seed_demo_video_run(db_session, deliv)

    detail = await client.get(f"/api/runs/{rid}")
    spine = detail.json()["clip_spine_stage"]
    assert spine == "demo_video_gen"

    r = await client.post(
        f"/api/runs/{rid}/reorder-clips",
        json={
            "stage_name": spine,
            "ordered_element_ids": ["clip-5", "clip-1", "clip-3", "clip-4"],
        },
    )
    assert r.status_code == 201, r.text
    assert r.json()["parent_run_id"] == rid
