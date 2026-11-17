"""Every log line a run's stages emit must carry THAT run's id.

Measured defect: grepping the orchestrator pod log for acceptance run
``01a0ab23-…`` returned 92 lines, all HTTP access lines, zero stage events — so
nobody could tell whether ``demo_video_compile``'s ERROR guard had fired.
``set_correlation_id`` existed and had no caller; stage lines carried
``correlation_id=""`` (reconciler-resumed runs) or the creating request's id.

The assertions read the ``correlation_id`` field of lines rendered by the
PRODUCTION ``CorrelatedJSONFormatter`` — the exact field the live pod emits — not
the ContextVar, so a bind the formatter does not surface would fail here.

Two runs execute concurrently, each spawning stage tasks the way
``DAGExecutor`` does (``asyncio.create_task`` + ``gather``), and a barrier forces
every stage of BOTH runs to be mid-flight before any of them logs. A single-run
test cannot see a bind that bleeds from one run onto another.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.observability.logging import CorrelatedJSONFormatter, correlation_id_var

pytestmark = pytest.mark.asyncio

_STAGE_LOGGER = "src.services.stages.test_correlation_probe"
_CALLER_LOGGER = "src.api.routes.test_correlation_probe"
_STAGES_PER_RUN = 3


class _FakeRedis:
    async def publish(self, channel: str, message: str) -> int:
        return 1


class _Capture(logging.Handler):
    """Renders through the production JSON formatter and keeps the parsed line."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.setFormatter(CorrelatedJSONFormatter())
        self.lines: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(json.loads(self.format(record)))


def _run_obj() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        blueprint_id="bp-1",
        deliverable_id=uuid4(),
        config_snapshot={
            "tenant_id": str(uuid4()),
            "subject_id": str(uuid4()),
            "license_id": "lic-1",
            "tenant_source": "jwt-claim",
        },
        pause_after_stages=[],
        started_at=None,
    )


async def _dispatch_concurrently(n_runs: int, capture: _Capture) -> list[UUID]:
    from src.services.run_dispatcher import RunDispatcher

    runs = [_run_obj() for _ in range(n_runs)]
    by_id = {run.id: run for run in runs}
    stage_log = logging.getLogger(_STAGE_LOGGER)
    caller_log = logging.getLogger(_CALLER_LOGGER)

    # Every stage of every run must be started before ANY stage logs, so the
    # runs genuinely overlap: a serial "A finishes, then B starts" would pass a
    # leaky bind.
    barrier_count = n_runs * _STAGES_PER_RUN
    arrived = 0
    all_arrived = asyncio.Event()

    session = AsyncMock()
    session.commit = AsyncMock()
    exec_result = MagicMock()
    exec_result.first = MagicMock(return_value=SimpleNamespace(id=None, parent_project_id=None))
    session.execute = AsyncMock(return_value=exec_result)

    run_repo = AsyncMock()
    run_repo.get = AsyncMock(side_effect=lambda run_id: by_id.get(run_id))
    run_repo.mark_failed = AsyncMock(return_value=True)
    for name in ("mark_started", "mark_cancelled", "mark_completed", "mark_paused"):
        setattr(run_repo, name, AsyncMock())
    stage_repo = AsyncMock()
    stage_repo.list_for_run = AsyncMock(return_value=[])
    blueprint = MagicMock()
    blueprint.topological_order.return_value = []
    registry = MagicMock()
    registry.resolve_by_id.return_value = blueprint

    async def _stage(ctx, index: int) -> None:
        nonlocal arrived
        arrived += 1
        if arrived >= barrier_count:
            all_arrived.set()
        await all_arrived.wait()
        stage_log.error("stage %d of run %s", index, ctx.run_id)
        await asyncio.sleep(0)
        stage_log.info("stage %d of run %s done", index, ctx.run_id)

    async def _run_dag(ctx, **_kwargs) -> None:
        # The shape DAGExecutor uses for a parallel group (dag_executor.py:636).
        tasks = [asyncio.create_task(_stage(ctx, i)) for i in range(_STAGES_PER_RUN)]
        await asyncio.gather(*tasks)

    executor = MagicMock()
    executor.run = AsyncMock(side_effect=_run_dag)

    @asynccontextmanager
    async def _factory():
        yield session

    with (
        patch("src.services.run_dispatcher.RunRepository", return_value=run_repo),
        patch("src.services.run_dispatcher.StageOutputRepository", return_value=stage_repo),
        patch("src.services.run_dispatcher.ProjectRepository", return_value=AsyncMock()),
        patch("src.services.run_dispatcher.DAGExecutor", return_value=executor),
    ):
        dispatcher = RunDispatcher(
            session_factory=_factory,
            registry=registry,
            redis=_FakeRedis(),  # ty: ignore[invalid-argument-type]
        )
        for run in runs:
            await dispatcher.dispatch(run.id)
            # The dispatching context (an HTTP handler / the reconciler loop)
            # keeps logging after each dispatch. It must not inherit a run id.
            caller_log.warning("dispatched a run")
        tasks = [t for t in (dispatcher._tasks.get(run.id) for run in runs) if t is not None]
        assert len(tasks) == n_runs
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
    caller_log.warning("all runs finished")

    stage_lines = [line for line in capture.lines if line["logger"] == _STAGE_LOGGER]
    assert len(stage_lines) == n_runs * _STAGES_PER_RUN * 2, (
        f"stubbed stages did not all log: {len(stage_lines)} lines"
    )
    return [run.id for run in runs]


@pytest.fixture
def capture():
    handler = _Capture()
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    token = correlation_id_var.set("")
    try:
        yield handler
    finally:
        correlation_id_var.reset(token)
        root.removeHandler(handler)
        root.setLevel(old_level)


async def test_each_concurrent_runs_stage_lines_carry_its_own_run_id(capture) -> None:
    run_ids = await _dispatch_concurrently(2, capture)

    stage_lines = [line for line in capture.lines if line["logger"] == _STAGE_LOGGER]
    for line in stage_lines:
        # The message names the run the stage belongs to (read off the
        # RunContext the dispatcher built), so a leaked id shows as a mismatch.
        owner = next(str(rid) for rid in run_ids if str(rid) in line["message"])
        assert line["correlation_id"] == owner, f"stage line attributed to the wrong run: {line}"
    for rid in run_ids:
        mine = [line for line in stage_lines if line["correlation_id"] == str(rid)]
        assert len(mine) == _STAGES_PER_RUN * 2, f"run {rid} lost stage lines: {len(mine)}"


async def test_the_dispatching_context_never_inherits_a_run_id(capture) -> None:
    run_ids = await _dispatch_concurrently(2, capture)

    caller_lines = [line for line in capture.lines if line["logger"] == _CALLER_LOGGER]
    assert len(caller_lines) == 3
    leaked = {str(rid) for rid in run_ids} & {line["correlation_id"] for line in caller_lines}
    assert not leaked, f"a run id leaked onto the dispatcher's caller: {caller_lines}"
    assert correlation_id_var.get() == ""
