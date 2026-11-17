"""The paid-spawn decision must be AUDITABLE from the record, not only from a log line.

``maybe_spawn_structural_relief_run`` bills an EXTRA PAID RENDER per spawn and
``PIPELINE_STRUCTURAL_RELIEF_AUTOCONVERGE_ENABLED`` is ``true`` in production gitops while
the code default is ``False`` — so the spend is live. Before this record existed the
decision left no persisted trace of WHY it spawned or declined: a prior session escalated a
misspend concern to the owner on a premise it then had to refute, purely because the
question was unmeasurable. These tests pin that the trace exists, that it names the refusal
that actually fired, and — the load-bearing one — that the recorder can NEVER change the
decision it records.

Storage shape is a ``stage_outputs`` pseudo-stage row (``structural_relief_decision``) on
the evaluated run, written through the two EXISTING repository methods. It is deliberately
NOT ``runs.config_snapshot``: ``order_fingerprint`` hashes every non-run-scoped snapshot key
and ``lifecycle`` compares that hash against the parent's persisted snapshot to decide
whether a regenerate may resume, so an audit key there would silently make the run
un-resumable. See the ``_DECISION_STAGE`` docstring.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest

from src.services.runs.structural_relief import (
    _DECISION_STAGE,
    maybe_spawn_structural_relief_run,
)


class _FakeRepo:
    """Captures what would be persisted, in place of ``StageOutputRepository``."""

    written: ClassVar[list[tuple[Any, str, dict[str, Any]]]] = []
    seeded: ClassVar[list[tuple[Any, str]]] = []

    def __init__(self, session: Any) -> None:
        self.session = session

    async def upsert_pending(self, run_id: Any, stage_name: str) -> Any:
        type(self).seeded.append((run_id, stage_name))
        return SimpleNamespace(id=uuid7())

    async def mark_completed(self, run_id: Any, stage_name: str, output: dict[str, Any]) -> bool:
        type(self).written.append((run_id, stage_name, output))
        return True


@pytest.fixture(autouse=True)
def _recorder(monkeypatch: Any) -> type[_FakeRepo]:
    """Stub the persistence seam: a no-op tenant bind + a capturing repository."""
    _FakeRepo.written = []
    _FakeRepo.seeded = []
    monkeypatch.setattr("src.services.runs.structural_relief.bind_run_tenant", AsyncMock())
    monkeypatch.setattr("src.services.runs.structural_relief.StageOutputRepository", _FakeRepo)
    return _FakeRepo


@pytest.fixture(autouse=True)
def _settings(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "src.services.runs.structural_relief.get_settings",
        lambda: SimpleNamespace(
            pipeline=SimpleNamespace(
                structural_relief_autoconverge_enabled=True,
                structural_relief_max_passes=2,
                demo_timeline_underfull_fail_ratio=0.0,
            )
        ),
    )


def _sessions(*, fail: bool = False) -> Callable[[], AbstractAsyncContextManager[Any]]:
    """A stand-in for ``RunEngine._sessions``, typed to the SAME shape the real engine
    declares (``Callable[[], AbstractAsyncContextManager[AsyncSession]]``).

    The inner function is an async GENERATOR (it ``yield``s), which is what
    ``asynccontextmanager`` requires; annotating it ``AsyncIterator`` rather than ``Any``
    is what makes a checker able to say so. Were it a coroutine instead (a ``return``, or
    no ``yield`` at all), ``async with`` would not hand the fake session to the code under
    test and every assertion below would pass having observed nothing — so the annotation
    is load-bearing for the non-vacuity of this whole module, not decoration.
    """

    @contextlib.asynccontextmanager
    async def factory() -> AsyncIterator[Any]:
        if fail:
            raise RuntimeError("database is unreachable")
        yield SimpleNamespace(commit=AsyncMock())

    return factory


def _engine(*, child: Any = None, sessions_fail: bool = False, no_sessions: bool = False) -> Any:
    engine = SimpleNamespace(start_run=AsyncMock(return_value=child))
    if not no_sessions:
        engine._sessions = _sessions(fail=sessions_fail)
    return engine


def _rows(
    *,
    outcome: str = "cant_converge",
    overrun: float = 13.53,
    beats: int = 3,
    target: float = 20.0,
    beat_min_seconds: list[float] | None = None,
) -> list[Any]:
    beat_rows: list[dict[str, Any]] = [{"id": f"clip-{i}"} for i in range(beats)]
    if beat_min_seconds is not None:
        for row, mins in zip(beat_rows, beat_min_seconds, strict=True):
            row["min_seconds"] = mins
    manifest = {
        "run": {
            "convergence_outcome": outcome,
            "structural_relief_requested": True,
            "overrun_seconds": overrun,
            "target_seconds": target,
            "infeasible_beat_ids": ["clip-0", "clip-1"],
        },
        "beats": beat_rows,
    }
    return [
        SimpleNamespace(
            stage_name="reconcile_timeline",
            output={
                "timeline_manifest": manifest,
                "lines": [{"id": f"clip-{i}", "text": f"Beat {i}."} for i in range(beats)],
            },
        )
    ]


def _snapshot(*, pass_number: int = 0, prior: float | None = None, identity: bool = True) -> Any:
    snap: dict[str, Any] = {}
    if identity:
        snap["tenant_id"] = str(uuid7())
        snap["subject_id"] = str(uuid7())
    relief: dict[str, Any] = {"pass": pass_number}
    if prior is not None:
        relief["prior_overrun"] = prior
    snap["structural_relief"] = relief
    # The relief opt-in the spawner requires (``direction.timing.relief.mode == "auto"``).
    snap["direction"] = {"timing": {"mode": "auto", "relief": {"mode": "auto"}}}
    return snap


async def _run(engine: Any, snapshot: Any, rows: list[Any], run_id: Any = None) -> Any:
    return await maybe_spawn_structural_relief_run(
        run_engine=engine,
        parent_run_id=run_id or uuid7(),
        deliverable_id=uuid7(),
        config_snapshot=snapshot,
        stage_rows=rows,
    )


def _record() -> dict[str, Any]:
    assert len(_FakeRepo.written) == 1, f"expected exactly one record, got {_FakeRepo.written}"
    _run_id, stage_name, output = _FakeRepo.written[0]
    assert stage_name == _DECISION_STAGE
    # Nested under ONE key so no top-level key of any stage vocabulary is shadowed —
    # ``prior_run_outputs`` carries this row's output to a blueprint-agnostic scanner.
    assert set(output) == {_DECISION_STAGE}
    return output[_DECISION_STAGE]


@pytest.mark.asyncio
async def test_spawn_is_recorded_with_the_child_it_paid_for() -> None:
    """A spawn is the money event: the record must name the child, the pass, and the
    lever that was moved, or the spend cannot be joined to the render it bought."""
    child = SimpleNamespace(id=uuid7())
    parent = uuid7()
    engine = _engine(child=child)
    assert await _run(engine, _snapshot(), _rows(), run_id=parent) == child.id

    rec = _record()
    assert rec["outcome"] == "spawned"
    assert rec["refusal"] is None
    # Identity: joinable to the evaluated run AND to the child render it authorised.
    assert rec["spawned_child_run_id"] == str(child.id)
    assert rec["spawned_pass"] == 1
    assert rec["identity"]["evaluated_run_id"] == str(parent)
    assert _FakeRepo.seeded == [(parent, _DECISION_STAGE)]
    # The inputs the decision actually turned on.
    assert rec["inputs"]["convergence_outcome"] == "cant_converge"
    assert rec["inputs"]["overrun_seconds"] == pytest.approx(13.53)
    assert rec["inputs"]["pass_number"] == 0
    assert rec["inputs"]["max_passes"] == 2
    assert rec["inputs"]["beat_count"] == 3
    assert rec["inputs"]["infeasible_beat_ids"] == ["clip-0", "clip-1"]
    # The terms of the spend.
    assert 1 <= rec["inputs"]["spawned_target_beats"] < 3
    assert rec["inputs"]["shape"]["verdict"] == "SPAWN"
    # Which measure SIZED the child is persisted with the spend. No delivery was
    # measured here, so the cut is sized off the footage plan, and on that basis the
    # sizing figures ARE the footage overrun and the measured per-beat yield.
    shape_rec = rec["inputs"]["shape"]
    assert shape_rec["sizing_basis"] == "footage"
    assert shape_rec["sizing_overrun_seconds"] == pytest.approx(13.53)
    assert shape_rec["sizing_seconds_per_beat"] == pytest.approx(
        shape_rec["measured_seconds_per_beat"]
    )
    assert rec["schema_version"] == 2


@pytest.mark.asyncio
async def test_cap_refusal_names_the_verdict_that_fired() -> None:
    """Three policy verdicts answer three different audit questions. Collapsing them to
    "declined" would make "budget exhausted" indistinguishable from "unfixable"."""
    engine = _engine()
    assert await _run(engine, _snapshot(pass_number=2, prior=20.0), _rows()) is None
    engine.start_run.assert_not_awaited()

    rec = _record()
    assert rec["outcome"] == "declined"
    assert rec["refusal"] == "STOP_CAP"
    assert rec["spawned_child_run_id"] is None
    assert rec["inputs"]["pass_number"] == 2
    # A run refused by the POLICY never computes a shape — recording zeros there would
    # assert arithmetic that was never performed.
    assert "shape" not in rec["inputs"]


@pytest.mark.asyncio
async def test_flat_overrun_refusal_records_the_prior_it_compared_against() -> None:
    engine = _engine()
    assert await _run(engine, _snapshot(pass_number=1, prior=13.5), _rows()) is None
    rec = _record()
    assert rec["refusal"] == "STOP_CANT_CONVERGE"
    assert rec["inputs"]["prior_overrun_seconds"] == pytest.approx(13.5)
    assert rec["inputs"]["overrun_seconds"] == pytest.approx(13.53)


@pytest.mark.asyncio
async def test_shape_refusal_records_both_the_gate_and_the_proved_sentence() -> None:
    """The unconvergeable-floor gate is the one that refuses to bill a render that
    provably cannot succeed. An auditor groups a population by the VERDICT and explains a
    single row by the SENTENCE, so both are persisted."""
    engine = _engine()
    rows = _rows(overrun=70.0, target=20.0, beat_min_seconds=[30.0, 30.0, 30.0])
    assert await _run(engine, _snapshot(), rows) is None
    engine.start_run.assert_not_awaited()

    rec = _record()
    assert rec["outcome"] == "declined"
    assert rec["refusal"] == "STOP_UNCONVERGEABLE"
    assert rec["inputs"]["stop_reason"]
    assert rec["inputs"]["shape"]["verdict"] == "STOP_UNCONVERGEABLE"
    assert rec["inputs"]["shape"]["seconds_floor"] == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_missing_identity_refusal_is_still_recordable() -> None:
    """The refusal CAUSED by an absent snapshot tenant must not be the one refusal that
    cannot be audited. It is recordable only because ``bind_run_tenant`` reads the tenant
    off the run row rather than off the snapshot."""
    engine = _engine()
    assert await _run(engine, _snapshot(identity=False), _rows()) is None
    engine.start_run.assert_not_awaited()

    rec = _record()
    assert rec["refusal"] == "missing_tenant_or_subject"
    assert rec["identity"]["tenant_id"] is None
    assert rec["identity"]["subject_id"] is None


@pytest.mark.asyncio
async def test_non_finite_inputs_refusal_records_figures_as_repr() -> None:
    """``inf``/``nan`` is exactly what this branch refuses on, and JSONB cannot hold it —
    serialising raw would raise and lose the record to its own cause."""
    engine = _engine()
    rows = _rows()
    rows[0].output["timeline_manifest"]["run"]["overrun_seconds"] = float("inf")
    assert await _run(engine, _snapshot(), rows) is None

    rec = _record()
    assert rec["refusal"] == "non_finite_inputs"
    assert rec["inputs"]["overrun_seconds_repr"] == "inf"
    assert rec["inputs"]["manifest_run_is_mapping"] is True


@pytest.mark.asyncio
async def test_a_failing_recorder_never_suppresses_the_paid_spawn() -> None:
    """⛔ THE LOAD-BEARING CLAIM. The hook body is wrapped in a blanket
    ``except Exception`` that turns any error into "no spawn". If the audit write raised
    into that wrapper it would SUPPRESS a render the decision had already approved — the
    recorder would become a silent decision-maker, and an unreachable database would
    change what the pipeline delivers. Recording stays strictly downstream of deciding.
    """
    child = SimpleNamespace(id=uuid7())
    engine = _engine(child=child, sessions_fail=True)
    assert await _run(engine, _snapshot(), _rows()) == child.id
    engine.start_run.assert_awaited_once()
    assert _FakeRepo.written == []


@pytest.mark.asyncio
async def test_an_engine_without_a_session_factory_still_spawns() -> None:
    # The existing unit suite stubs the engine with ``start_run`` alone; the recorder must
    # degrade to a no-op there rather than turning every sibling test red.
    child = SimpleNamespace(id=uuid7())
    engine = _engine(child=child, no_sessions=True)
    assert await _run(engine, _snapshot(), _rows()) == child.id
    assert _FakeRepo.written == []


@pytest.mark.asyncio
async def test_a_non_relief_run_writes_no_record() -> None:
    """Every completed run in the fleet reaches this hook. Recording the "no
    reconcile_timeline manifest" exit would put an audit row on all of them — the record
    starts only once the run is a genuine relief candidate."""
    engine = _engine()
    rows = [SimpleNamespace(stage_name="demo_video_gen", output={"lines": []})]
    assert await _run(engine, _snapshot(), rows) is None
    assert _FakeRepo.written == []


@pytest.mark.asyncio
async def test_converged_run_writes_no_spawn_and_records_the_reason() -> None:
    engine = _engine()
    assert await _run(engine, _snapshot(), _rows(outcome="converged")) is None
    assert _record()["refusal"] == "STOP_CONVERGED"


@pytest.mark.asyncio
async def test_a_relief_off_run_writes_no_record_and_no_spawn() -> None:
    """A relief-OFF run can carry a manifest (footage out-points only). It is refused
    (STOP_RELIEF_OFF, logged) BEFORE it becomes a relief candidate, so — like a run with no
    manifest — it writes no audit row and never reaches the engine."""
    engine = _engine()
    snap = _snapshot()
    snap["direction"] = {"timing": {"mode": "auto"}}
    assert await _run(engine, snap, _rows()) is None
    engine.start_run.assert_not_awaited()
    assert _FakeRepo.written == []


@pytest.mark.asyncio
async def test_relief_child_inherits_the_parents_provider_pins() -> None:
    """A system-spawned re-render runs on the PARENT's pinned providers (P3), not on
    whatever the tenant configured since the parent started."""
    pins = {
        "llm": {
            "source": "tenant",
            "kind": "anthropic",
            "connection_id": str(uuid7()),
            "version": 3,
        }
    }
    snapshot = _snapshot()
    snapshot["provider_slots"] = pins
    engine = _engine(child=SimpleNamespace(id=uuid7()))
    await _run(engine, snapshot, _rows())
    engine.start_run.assert_awaited_once()
    assert engine.start_run.await_args.kwargs["inherit_slots"] == pins
