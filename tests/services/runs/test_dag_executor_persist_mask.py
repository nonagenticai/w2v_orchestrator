"""Regression: a persist()/emit() failure in DAGExecutor's `except Exception`
chokepoint (dag_executor.py) must NOT replace the real stage exception that
triggered it. See the class docstring in dag_executor.py's failure handler.
"""

from uuid import uuid4

import pytest

from src.services.blueprints.base import DAGBlueprint, StageSpec
from src.services.runs.dag_executor import DAGExecutor
from src.services.runs.run_context import RunContext
from src.services.stages.base import StageHandler, StageInput, StageOutput


class _NoopIn(StageInput):
    pass


class _NoopOut(StageOutput):
    value: int = 0


class _RealStageError(RuntimeError):
    """The genuine stage failure we want callers to see."""


async def _handle_fails(self, payload, ctx):
    raise _RealStageError("boom: the actual bug")


_FailStage = type(
    "_FailStage",
    (StageHandler,),
    {
        "name": "a",
        "input_model": _NoopIn,
        "output_model": _NoopOut,
        "handle": _handle_fails,
    },
)


@pytest.mark.asyncio
async def test_persist_failure_does_not_mask_the_real_stage_exception() -> None:
    """A persist() failure while recording the terminal 'failed' status must not
    replace the original stage exception on the way back up to the dispatcher.

    Mutation check: reverting dag_executor.py's inner try/except around
    `persist`/`emit` in the `except Exception` handler back to unguarded calls
    turns this red — `excinfo.value` becomes the injected ConnectionError
    instead of `_RealStageError`.
    """
    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[StageSpec(name="a", handler_factory=lambda deps: _FailStage())],
    )

    async def emit(event, **kw):
        return None

    async def persist(stage_name, status, output=None, error=None):
        if status == "failed":
            raise ConnectionError("db pool exhausted while persisting failure")

    executor = DAGExecutor(bp, emit=emit, persist=persist)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    with pytest.raises(_RealStageError):
        await executor.run(ctx, pause_after_stages=set())


class _AlwaysTimesOut(StageHandler):
    name = "a"
    input_model = _NoopIn
    output_model = _NoopOut

    async def handle(self, payload, ctx):
        raise TimeoutError("stage never returned")


@pytest.mark.asyncio
async def test_persist_failure_does_not_mask_the_exhausted_resume_timeout() -> None:
    """Same hazard as above, but in the resume-EXHAUSTED branch of the
    ``_RESUMABLE_TIMEOUT_EXC`` handler (dag_executor.py, before the bare
    ``raise`` that follows the terminal 'failed' persist/emit pair). With
    ``max_stage_attempts=0`` and ``bump_attempts`` returning 1, the very first
    timeout is already over the cap, so the exhausted path runs immediately.

    Mutation check: reverting the try/except added around that persist/emit
    pair back to unguarded calls turns this red — `excinfo.value` becomes the
    injected ConnectionError instead of the original `TimeoutError`.
    """
    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[StageSpec(name="a", handler_factory=lambda deps: _AlwaysTimesOut())],
    )

    async def emit(event, **kw):
        return None

    async def persist(stage_name, status, output=None, error=None):
        if status == "failed":
            raise ConnectionError("db pool exhausted while persisting failure")

    async def bump_attempts(stage_name):
        return 1

    executor = DAGExecutor(
        bp,
        emit=emit,
        persist=persist,
        bump_attempts=bump_attempts,
        max_stage_attempts=0,
    )
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    with pytest.raises(TimeoutError):
        await executor.run(ctx, pause_after_stages=set())
