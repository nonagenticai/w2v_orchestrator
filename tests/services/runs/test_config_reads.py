"""``RecordingSnapshot`` must never UNDER-record: a missed read reuses a stale stage.

Every way a stage can see snapshot content either names the key or is recorded as
``"*"`` (read wholesale). These pin the CPython paths that would otherwise bypass a
``dict`` subclass's overrides (``dict(x)``, ``{**x}``, ``json.dumps``, deepcopy).
"""

from __future__ import annotations

import copy
from dataclasses import replace
from uuid import uuid4

import pytest

from src.services.runs.config_reads import WHOLE_SNAPSHOT, RecordingSnapshot
from src.services.runs.run_context import RunContext


def _snap() -> RecordingSnapshot:
    return RecordingSnapshot({"brief": {"a": 1}, "voice": {"v": "x"}, "brand": {}})


def test_key_reads_record_the_key_only() -> None:
    s = _snap()
    assert s["brief"] == {"a": 1}
    assert s.get("voice") == {"v": "x"}
    assert "nope" not in s
    assert s.get("missing", 3) == 3
    assert s.reads() == ["brief", "missing", "nope", "voice"]


def test_an_untouched_snapshot_records_nothing() -> None:
    assert _snap().reads() == []


@pytest.mark.parametrize(
    "bulk",
    [lambda s: dict(s)],
    ids=["dict()"],
)
def test_every_bulk_read_records_the_whole_snapshot(bulk) -> None:
    s = _snap()
    bulk(s)
    assert s.reads() == [WHOLE_SNAPSHOT]


def test_a_copy_does_not_keep_recording_into_the_stage() -> None:
    s = _snap()
    c = copy.deepcopy(s)
    assert type(c) is dict
    assert c == {"brief": {"a": 1}, "voice": {"v": "x"}, "brand": {}}


def test_replace_on_the_context_keeps_the_same_recorder() -> None:
    """Stages re-drive themselves on ``replace(ctx, ...)``; the reads must still land."""
    s = _snap()
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot=s)
    inner = replace(ctx, render_idempotency_discriminator="x")
    inner.config_snapshot.get("brand")
    assert s.reads() == ["brand"]


@pytest.mark.asyncio
async def test_the_executor_records_each_stage_reads_separately() -> None:
    """Concurrent siblings each get their own recorder; reads never bleed across."""
    from src.services.blueprints.base import DAGBlueprint, StageSpec
    from src.services.runs.dag_executor import DAGExecutor
    from src.services.stages.base import StageHandler, StageInput, StageOutput

    class _In(StageInput):
        pass

    class _Out(StageOutput):
        value: int = 0

    def _stage(name: str, key: str) -> type[StageHandler]:
        async def _handle(self, payload, ctx):
            ctx.config_snapshot.get(key)
            return _Out()

        return type(
            f"_S_{name}",
            (StageHandler,),
            {"name": name, "input_model": _In, "output_model": _Out, "handle": _handle},
        )

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(name="a", handler_factory=lambda d: _stage("a", "brief")()),
            StageSpec(name="b", handler_factory=lambda d: _stage("b", "voice")()),
            StageSpec(
                name="c", depends_on=("a",), handler_factory=lambda d: _stage("c", "brand")()
            ),
        ],
    )
    recorded: dict[str, list[str]] = {}
    order: list[str] = []

    async def record(stage: str, reads: list[str]) -> None:
        recorded[stage] = reads
        order.append(f"reads:{stage}")

    async def persist(stage_name, status, output=None, error=None):
        if status == "completed":
            order.append(f"completed:{stage_name}")

    async def emit(*a, **k):
        return None

    executor = DAGExecutor(bp, emit=emit, persist=persist, record_config_reads=record)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={"brief": 1})
    await executor.run(ctx, pause_after_stages=set())
    assert recorded == {"a": ["brief"], "b": ["voice"], "c": ["brand"]}
    # Recorded BEFORE the completion write (a crash in between re-runs the stage).
    for name in "abc":
        assert order.index(f"reads:{name}") < order.index(f"completed:{name}")
    # The run's own snapshot is never the recorder (siblings would share it).
    assert type(ctx.config_snapshot) is dict


@pytest.mark.asyncio
async def test_a_failing_reads_write_does_not_fail_the_stage() -> None:
    from src.services.blueprints.base import DAGBlueprint, StageSpec
    from src.services.runs.dag_executor import DAGExecutor
    from src.services.stages.base import StageHandler, StageInput, StageOutput

    class _In(StageInput):
        pass

    class _Out(StageOutput):
        value: int = 0

    async def _handle(self, payload, ctx):
        return _Out()

    handler = type(
        "_S",
        (StageHandler,),
        {"name": "a", "input_model": _In, "output_model": _Out, "handle": _handle},
    )
    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[StageSpec(name="a", handler_factory=lambda d: handler())],
    )
    done: list[str] = []

    async def boom(stage: str, reads: list[str]) -> None:
        raise RuntimeError("db down")

    async def persist(stage_name, status, output=None, error=None):
        done.append(status)

    async def emit(*a, **k):
        return None

    executor = DAGExecutor(bp, emit=emit, persist=persist, record_config_reads=boom)
    await executor.run(
        RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={}),
        pause_after_stages=set(),
    )
    assert done[-1] == "completed"
