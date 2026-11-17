"""Parallel-group scheduler tests for DAGExecutor."""

import asyncio
from uuid import uuid4

import pytest
from w2v_common.stages import StageTimeoutError

from src.services.blueprints.base import DAGBlueprint, StageSpec
from src.services.runs.dag_executor import (
    DAGExecutor,
    PersistFn,
    StageResumableTimeoutError,
)
from src.services.runs.run_context import RunContext
from src.services.stages.base import StageHandler, StageInput, StageOutput


class _In(StageInput):
    pass


class _Out(StageOutput):
    name: str = ""


def _slow_stage(name: str, sleep_s: float) -> type[StageHandler]:
    async def _handle(self, payload, ctx):
        await asyncio.sleep(sleep_s)
        return _Out(name=name)

    return type(
        f"_S_{name}",
        (StageHandler,),
        {
            "name": name,
            "input_model": _In,
            "output_model": _Out,
            "handle": _handle,
        },
    )


_RootStage = _slow_stage("root", 0.0)
_AStage = _slow_stage("a", 0.4)
_BStage = _slow_stage("b", 0.4)


def _make_root(deps: dict) -> StageHandler:
    return _RootStage()


def _make_a(deps: dict) -> StageHandler:
    return _AStage()


def _make_b(deps: dict) -> StageHandler:
    return _BStage()


@pytest.mark.asyncio
async def test_parallel_group_runs_concurrently() -> None:
    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(name="root", handler_factory=_make_root),
            StageSpec(
                name="a",
                depends_on=("root",),
                parallel_group="g",
                handler_factory=_make_a,
            ),
            StageSpec(
                name="b",
                depends_on=("root",),
                parallel_group="g",
                handler_factory=_make_b,
            ),
        ],
    )
    timing: list[tuple[str, float]] = []
    loop_t0 = asyncio.get_event_loop().time()

    async def emit(event, **kw):
        if event == "stage_completed":
            timing.append((kw["stage_name"], asyncio.get_event_loop().time() - loop_t0))

    async def persist(stage_name, status, output=None, error=None):
        return None

    executor = DAGExecutor(bp, emit=emit, persist=persist)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    await executor.run(ctx, pause_after_stages=set())

    a_t = next(t for n, t in timing if n == "a")
    b_t = next(t for n, t in timing if n == "b")
    # If serial they'd be ~0.4s apart; parallel should land within ~150ms.
    assert abs(a_t - b_t) < 0.2, f"a={a_t}, b={b_t} (expected concurrent)"


class _BoomError(RuntimeError):
    pass


def _failing_stage(name: str, delay_s: float) -> type[StageHandler]:
    async def _handle(self, payload, ctx):
        await asyncio.sleep(delay_s)
        raise _BoomError(name)

    return type(
        f"_Fail_{name}",
        (StageHandler,),
        {"name": name, "input_model": _In, "output_model": _Out, "handle": _handle},
    )


def _recording_stage(
    name: str, delay_s: float, completed: list[str], cancelled: list[str]
) -> type[StageHandler]:
    async def _handle(self, payload, ctx):
        try:
            await asyncio.sleep(delay_s)
        except asyncio.CancelledError:
            cancelled.append(name)
            raise
        completed.append(name)  # only reached if NOT cancelled
        return _Out(name=name)

    return type(
        f"_Rec_{name}",
        (StageHandler,),
        {"name": name, "input_model": _In, "output_model": _Out, "handle": _handle},
    )


@pytest.mark.asyncio
async def test_parallel_sibling_cancelled_on_failure() -> None:
    """When one parallel sibling fails, the other in-flight sibling is cancelled
    and never completes — no detached non-idempotent stage runs on after the run
    has already failed (duplicate artifacts / late stage_completed SSE)."""
    completed: list[str] = []
    cancelled: list[str] = []
    persisted: list[tuple[str, str]] = []
    emitted: list[tuple[str, str]] = []

    fail_cls = _failing_stage("a", 0.05)
    slow_cls = _recording_stage("b", 1.0, completed, cancelled)

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(name="root", handler_factory=lambda deps: _RootStage()),
            StageSpec(
                name="a",
                depends_on=("root",),
                parallel_group="g",
                handler_factory=lambda deps: fail_cls(),
            ),
            StageSpec(
                name="b",
                depends_on=("root",),
                parallel_group="g",
                handler_factory=lambda deps: slow_cls(),
            ),
        ],
    )

    async def emit(event, **kw):
        if "stage_name" in kw:
            emitted.append((event, kw["stage_name"]))

    async def persist(stage_name, status, output=None, error=None):
        persisted.append((stage_name, status))

    executor = DAGExecutor(bp, emit=emit, persist=persist)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    with pytest.raises(_BoomError):
        await executor.run(ctx, pause_after_stages=set())

    # Let the loop settle so any *detached* sibling would have completed by now.
    await asyncio.sleep(1.2)

    assert "b" in cancelled, "slow sibling should have been cancelled on a's failure"
    assert "b" not in completed, "cancelled sibling must NOT run to completion"
    assert ("b", "completed") not in persisted, "no post-terminal completed persist for b"
    assert ("stage_completed", "b") not in emitted, "no late stage_completed SSE for b"


def _timeout_stage(name: str, delay_s: float) -> type[StageHandler]:
    async def _handle(self, payload, ctx):
        await asyncio.sleep(delay_s)
        raise StageTimeoutError(f"stage {name!r} timed out")

    return type(
        f"_Timeout_{name}",
        (StageHandler,),
        {"name": name, "input_model": _In, "output_model": _Out, "handle": _handle},
    )


@pytest.mark.asyncio
async def test_parallel_sibling_drained_not_cancelled_on_resumable_timeout() -> None:
    """When one parallel sibling hits a RESUMABLE timeout, the other in-flight
    sibling is NOT cancelled — it drains to completion so its (possibly
    non-idempotent) side effect runs exactly once and its 'completed' state is
    preserved on resume. Only the timed-out stage is reset to 'pending'."""
    completed: list[str] = []
    cancelled: list[str] = []
    persisted: list[tuple[str, str]] = []

    timeout_cls = _timeout_stage("a", 0.02)
    slow_cls = _recording_stage("b", 0.2, completed, cancelled)

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(name="root", handler_factory=lambda deps: _RootStage()),
            StageSpec(
                name="a",
                depends_on=("root",),
                parallel_group="g",
                handler_factory=lambda deps: timeout_cls(),
            ),
            StageSpec(
                name="b",
                depends_on=("root",),
                parallel_group="g",
                handler_factory=lambda deps: slow_cls(),
            ),
        ],
    )

    async def emit(event, **kw):
        return None

    async def persist(stage_name, status, output=None, error=None):
        persisted.append((stage_name, status))

    async def _bump(name):
        return 1

    executor = DAGExecutor(
        bp,
        emit=emit,
        persist=persist,
        bump_attempts=_bump,
        max_stage_attempts=3,
    )
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    with pytest.raises(StageResumableTimeoutError):
        await executor.run(ctx, pause_after_stages=set())

    # Let the loop settle so the drained sibling reaches its natural terminal state.
    await asyncio.sleep(0.4)

    assert "b" not in cancelled, "sibling must NOT be cancelled on a resumable timeout"
    assert "b" in completed, "sibling should drain to completion (side effect runs once)"
    assert ("b", "completed") in persisted, "sibling's completed state must be persisted"
    assert ("a", "pending") in persisted, "only the timed-out stage is reset to pending"


@pytest.mark.asyncio
async def test_handler_deps_passed_through() -> None:
    captured: list[dict] = []

    class _Capture(StageHandler[_In, _Out]):
        name = "cap"
        input_model = _In
        output_model = _Out

        def __init__(self, mark: str) -> None:
            self.mark = mark

        async def handle(self, payload, ctx):
            return _Out(name=self.mark)

    def _factory(deps: dict) -> _Capture:
        captured.append(deps)
        return _Capture(mark=str(deps.get("token", "")))

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[StageSpec(name="cap", handler_factory=_factory)],
    )

    async def emit(event, **kw):
        return None

    async def persist(stage_name, status, output=None, error=None):
        return None

    executor = DAGExecutor(bp, emit=emit, persist=persist, handler_deps={"token": "abc"})
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    outputs = await executor.run(ctx, pause_after_stages=set())

    assert captured == [{"token": "abc"}]
    assert outputs["cap"]["name"] == "abc"


def _cancel_aware_stage(name: str, delay_s: float, cancelled: list[str]) -> type[StageHandler]:
    async def _handle(self, payload, ctx):
        try:
            await asyncio.sleep(delay_s)
        except asyncio.CancelledError:
            cancelled.append(name)
            raise
        return _Out(name=name)

    return type(
        f"_CA_{name}",
        (StageHandler,),
        {"name": name, "input_model": _In, "output_model": _Out, "handle": _handle},
    )


def _drain_probe() -> tuple[list[tuple[str, str]], asyncio.Event, PersistFn]:
    """A persist fn whose 'pending' reset for stage 'b' takes REAL time.

    That models the DB write in ``_run_one``'s CancelledError branch. If a second
    cancel aborts the drain, the write never lands and the row stays 'running'.
    """
    persisted: list[tuple[str, str]] = []
    reset_started = asyncio.Event()

    async def persist(stage_name, status, output=None, error=None):
        if stage_name == "b" and status == "pending":
            reset_started.set()
            await asyncio.sleep(0.2)
        persisted.append((stage_name, status))

    return persisted, reset_started, persist


@pytest.mark.asyncio
async def test_second_cancel_during_failure_drain_still_resets_sibling() -> None:
    """A cancel of the RUN task delivered while the failure drain is in flight must
    not strand the cancelled sibling half-reset.

    Stage a fails -> b is cancelled -> b starts persisting 'pending' -> an
    independent cancel (pod SIGTERM / user cancel / lease loss) hits run(). Before
    the fix that cancel aborted the drain AND re-cancelled b mid-write, so b's row
    stayed 'running' forever and was later swept to 'cancelled' — the exact
    operator-blind state W1-1 exists to prevent.
    """
    cancelled: list[str] = []
    persisted, reset_started, persist = _drain_probe()

    fail_cls = _failing_stage("a", 0.02)
    slow_cls = _cancel_aware_stage("b", 5.0, cancelled)

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(name="root", handler_factory=lambda deps: _RootStage()),
            StageSpec(name="a", depends_on=("root",), handler_factory=lambda deps: fail_cls()),
            StageSpec(name="b", depends_on=("root",), handler_factory=lambda deps: slow_cls()),
        ],
    )

    async def emit(event, **kw):
        return None

    executor = DAGExecutor(bp, emit=emit, persist=persist)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    run_task = asyncio.create_task(executor.run(ctx, pause_after_stages=set()))
    await asyncio.wait_for(reset_started.wait(), timeout=5.0)
    run_task.cancel()  # the SECOND, independent cancel

    with pytest.raises(asyncio.CancelledError):
        await run_task

    # Asserted IMMEDIATELY: a detached b would land this write a moment later, so
    # sleeping here would make the test pass on the broken code.
    assert "b" in cancelled
    assert ("b", "pending") in persisted, (
        "sibling's 'running' -> 'pending' reset was abandoned by the second cancel"
    )


@pytest.mark.asyncio
async def test_second_cancel_during_timeout_drain_still_resets_sibling() -> None:
    """Same guarantee on the resumable-timeout drain, which is MORE exposed: it
    deliberately does not cancel, so it can sit in the drain for the slowest
    sibling's whole stage ceiling with a cancel arriving at any point.

    On absorbing that cancel the batch is cancelled ONCE so the sibling runs its
    own reset, and the CancelledError — not the resumable timeout — is what the
    dispatcher sees, so a cancelled run is not left 'running' for the reconciler.
    """
    cancelled: list[str] = []
    persisted, reset_started, persist = _drain_probe()

    timeout_cls = _timeout_stage("a", 0.02)
    slow_cls = _cancel_aware_stage("b", 5.0, cancelled)

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(name="root", handler_factory=lambda deps: _RootStage()),
            StageSpec(name="a", depends_on=("root",), handler_factory=lambda deps: timeout_cls()),
            StageSpec(name="b", depends_on=("root",), handler_factory=lambda deps: slow_cls()),
        ],
    )

    async def emit(event, **kw):
        return None

    async def _bump(name):
        return 1

    executor = DAGExecutor(
        bp, emit=emit, persist=persist, bump_attempts=_bump, max_stage_attempts=3
    )
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    run_task = asyncio.create_task(executor.run(ctx, pause_after_stages=set()))
    # a resets itself to 'pending' and raises, then run() enters the no-cancel drain.
    await asyncio.sleep(0.15)
    run_task.cancel()  # e.g. the user cancels the run
    # Wait until b is mid-reset, then land the SECOND cancel on top (pod SIGTERM
    # after a user cancel is the documented pair). A bare gather re-cancels its
    # children on every cancel(), which kills b's DB write half-way.
    await asyncio.wait_for(reset_started.wait(), timeout=5.0)
    run_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run_task

    assert "b" in cancelled
    assert ("b", "pending") in persisted, (
        "sibling's 'running' -> 'pending' reset was abandoned by the cancel"
    )
