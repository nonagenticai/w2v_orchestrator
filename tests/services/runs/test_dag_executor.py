import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from w2v_common.stages import StageTimeoutError

from src.services.blueprints.base import DAGBlueprint, StageSpec
from src.services.runs.dag_executor import DAGExecutor, StageResumableTimeoutError
from src.services.runs.run_context import RunContext
from src.services.stages.base import (
    StageHandler,
    StageInput,
    StageOutput,
    StagePreconditionError,
    StageSkippedError,
)


class _NoopIn(StageInput):
    pass


class _NoopOut(StageOutput):
    value: int


def _make_stage(name: str, value: int) -> type[StageHandler]:
    _v = value  # capture for closure

    async def _handle(self, payload, ctx):
        return _NoopOut(value=_v)

    return type(
        f"_Stage_{name}",
        (StageHandler,),
        {
            "name": name,
            "input_model": _NoopIn,
            "output_model": _NoopOut,
            "handle": _handle,
        },
    )


@pytest.mark.asyncio
async def test_dag_executor_runs_in_topo_order_and_collects_outputs() -> None:
    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(name="a", handler_factory=lambda deps: _make_stage("a", 1)()),
            StageSpec(
                name="b",
                depends_on=("a",),
                handler_factory=lambda deps: _make_stage("b", 2)(),
            ),
        ],
    )
    captured: list[tuple[str, str | None]] = []

    async def emit(event, **kw):
        captured.append((event, kw.get("stage_name")))

    async def persist(stage_name, status, output=None, error=None):
        captured.append((f"persist.{status}", stage_name))

    executor = DAGExecutor(bp, emit=emit, persist=persist)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    outputs = await executor.run(ctx, pause_after_stages=set())
    assert "a" in outputs and "b" in outputs
    completed = [s for ev, s in captured if ev == "stage_completed"]
    assert completed == ["a", "b"]


@pytest.mark.asyncio
async def test_dag_executor_sets_cost_correlation_per_stage() -> None:
    """Each stage runs with correlation_id=run_id + stage=spec.name in context."""
    from w2v_common.auth import (
        get_correlation_id_from_context,
        get_stage_from_context,
    )

    seen: dict[str, tuple[str, str]] = {}

    def _capturing_stage(name: str) -> type[StageHandler]:
        async def _handle(self, payload, ctx):
            seen[name] = (get_correlation_id_from_context(), get_stage_from_context())
            return _NoopOut(value=1)

        return type(
            f"_Cap_{name}",
            (StageHandler,),
            {
                "name": name,
                "input_model": _NoopIn,
                "output_model": _NoopOut,
                "handle": _handle,
            },
        )

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(name="a", handler_factory=lambda deps: _capturing_stage("a")()),
            StageSpec(
                name="b",
                depends_on=("a",),
                handler_factory=lambda deps: _capturing_stage("b")(),
            ),
        ],
    )

    async def emit(event, **kw):
        pass

    async def persist(*a, **kw):
        pass

    executor = DAGExecutor(bp, emit=emit, persist=persist)
    run_id = uuid4()
    ctx = RunContext(run_id=run_id, deliverable_id=uuid4(), config_snapshot={})
    await executor.run(ctx, pause_after_stages=set())

    assert seen["a"] == (str(run_id), "a")
    assert seen["b"] == (str(run_id), "b")


@pytest.mark.asyncio
async def test_stage_outbound_llm_call_carries_run_id_as_w2v_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stage's outbound LLM call carries ``X-W2V-Job-Id = run_id``; unbound after the run.

    Drives the REAL ``w2v_common`` ``WMClient.message`` request path on the ir
    backend, whose pooled client installs ``identity_event_hook`` — only the
    transport is mocked (the conftest guard blocks real LLM sockets). Two
    co-eligible stages run as parallel ``create_task`` siblings, so this also
    proves the bind reaches stage tasks spawned by the executor, not just the run
    task.
    """
    import httpx
    from w2v_common.auth import (
        W2V_JOB_ID_HEADER,
        clear_identity_context,
        get_w2v_job_id_from_context,
        set_identity_context,
    )
    from w2v_common.tenant import TenantSource

    from tests.support.platform_llm import common_ir_client, use_ir_backend

    use_ir_backend(monkeypatch)
    seen: list[httpx.Request] = []

    def _calling_stage(name: str) -> type[StageHandler]:
        async def _handle(self, payload, ctx):
            wm = common_ir_client(seen)
            try:
                await wm.message(name, "hello")
            finally:
                await wm.aclose()
            return _NoopOut(value=1)

        return type(
            f"_Wm_{name}",
            (StageHandler,),
            {
                "name": name,
                "input_model": _NoopIn,
                "output_model": _NoopOut,
                "handle": _handle,
            },
        )

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(name="a", handler_factory=lambda deps: _calling_stage("a")()),
            StageSpec(name="b", handler_factory=lambda deps: _calling_stage("b")()),
        ],
    )

    async def emit(event, **kw):
        pass

    async def persist(*a, **kw):
        pass

    executor = DAGExecutor(bp, emit=emit, persist=persist)
    run_id = uuid4()
    ctx = RunContext(run_id=run_id, deliverable_id=uuid4(), config_snapshot={})
    # The dispatcher binds the run's tenant before the executor runs; mirror it
    # (an outbound call with no tenant is refused before it is ever sent).
    set_identity_context(
        tenant_id="dddddddd-dddd-dddd-dddd-dddddddddddd",
        license_id="lic",
        subject_id="sub",
        tenant_source=TenantSource.LICENSE_TOKEN,
    )
    try:
        assert get_w2v_job_id_from_context() == "", "precondition: no job bound"
        await executor.run(ctx, pause_after_stages=set())
        after = get_w2v_job_id_from_context()
    finally:
        clear_identity_context()

    sent = {r.headers["X-Worker-Name"]: r.headers.get(W2V_JOB_ID_HEADER) for r in seen}
    assert len(seen) == 2
    assert sent == {"a": str(run_id), "b": str(run_id)}
    # Cleared in ``finally``: the job must not outlive the DAG on this task.
    assert after == ""


@pytest.mark.asyncio
async def test_dag_executor_pauses_after_marked_stage() -> None:
    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(name="a", handler_factory=lambda deps: _make_stage("a", 1)()),
            StageSpec(
                name="b",
                depends_on=("a",),
                handler_factory=lambda deps: _make_stage("b", 2)(),
            ),
        ],
    )

    async def emit(event, **kw):
        pass

    async def persist(*a, **kw):
        pass

    executor = DAGExecutor(bp, emit=emit, persist=persist)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    outputs = await executor.run(ctx, pause_after_stages={"a"})
    assert "a" in outputs and "b" not in outputs


@pytest.mark.asyncio
async def test_dag_executor_records_failed_stage() -> None:
    class _BoomIn(StageInput):
        pass

    class _BoomOut(StageOutput):
        pass

    class _Boom(StageHandler[_BoomIn, _BoomOut]):
        name = "boom"
        input_model = _BoomIn
        output_model = _BoomOut

        async def handle(self, payload: _BoomIn, ctx: object) -> _BoomOut:
            raise RuntimeError("kaboom")

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[StageSpec(name="boom", handler_factory=lambda deps: _Boom())],
    )
    seen: list[tuple[str, str | object]] = []

    async def emit(event, **kw):
        seen.append((event, kw.get("error", "")))

    async def persist(stage_name, status, output=None, error=None):
        seen.append((f"persist.{status}", error or ""))

    executor = DAGExecutor(bp, emit=emit, persist=persist)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    with pytest.raises(RuntimeError, match="kaboom"):
        await executor.run(ctx, pause_after_stages=set())
    assert any(ev == "stage_failed" for ev, _ in seen)
    assert any(ev == "persist.failed" for ev, _ in seen)


@pytest.mark.asyncio
async def test_dag_executor_resets_stage_to_pending_on_cancellation() -> None:
    """Regression: a stage cancelled mid-flight must not be left 'running'.

    CancelledError is a BaseException, so it bypasses the `except Exception`
    failure handler — without a dedicated branch the stage row persisted
    "running" at start is never moved off "running", and the run shows a stage
    stuck running forever (the drain/pause/cancel symptom). The executor resets
    it to "pending" so a resumed run re-executes it cleanly.
    """
    started = asyncio.Event()
    release = asyncio.Event()

    class _BlockIn(StageInput):
        pass

    class _BlockOut(StageOutput):
        pass

    class _Block(StageHandler[_BlockIn, _BlockOut]):
        name = "block"
        input_model = _BlockIn
        output_model = _BlockOut

        async def handle(self, payload: _BlockIn, ctx: object) -> _BlockOut:
            started.set()
            await release.wait()  # blocks until the task is cancelled
            return _BlockOut()

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[StageSpec(name="block", handler_factory=lambda deps: _Block())],
    )
    persisted: list[tuple[str, str]] = []

    async def emit(event, **kw):
        pass

    async def persist(stage_name, status, output=None, error=None):
        persisted.append((stage_name, status))

    executor = DAGExecutor(bp, emit=emit, persist=persist)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    task = asyncio.create_task(executor.run(ctx, pause_after_stages=set()))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    block_statuses = [s for (n, s) in persisted if n == "block"]
    assert "running" in block_statuses
    assert block_statuses[-1] == "pending", persisted
    assert "failed" not in block_statuses  # cancellation is not a failure


@pytest.mark.asyncio
async def test_failing_stage_is_attempted_exactly_once() -> None:
    """The executor must NOT retry a failing handler.

    Retry was removed from the stage chain: handlers own their own
    call-boundary retries (where idempotency is known), and an outer blind
    retry re-ran non-idempotent compile/render handlers that deliberately set
    ``retries=0`` — causing duplicate renders. This pins that a handler whose
    ``handle`` raises is invoked exactly once and the stage is marked failed
    once, with the underlying error surfaced (no CircuitBreakerOpen wrapper).
    """

    class _BoomIn(StageInput):
        pass

    class _BoomOut(StageOutput):
        pass

    calls = {"n": 0}

    class _Boom(StageHandler[_BoomIn, _BoomOut]):
        name = "boom"
        input_model = _BoomIn
        output_model = _BoomOut

        async def handle(self, payload: _BoomIn, ctx: object) -> _BoomOut:
            calls["n"] += 1
            raise RuntimeError("boom")

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[StageSpec(name="boom", handler_factory=lambda deps: _Boom())],
    )
    errors: list[str] = []

    async def emit(event, **kw):
        pass

    async def persist(stage_name, status, output=None, error=None):
        if status == "failed":
            errors.append(error or "")

    executor = DAGExecutor(bp, emit=emit, persist=persist)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={}, tenant_id=uuid4())
    with contextlib.suppress(Exception):
        await executor.run(ctx, pause_after_stages=set())

    assert calls["n"] == 1, "handler must be invoked exactly once (no outer retry)"
    assert errors and all("RuntimeError" in e for e in errors), errors


@pytest.mark.asyncio
async def test_stage_skipped_marks_stage_skipped_and_dag_continues() -> None:
    """Regression: an opt-in stage that raises StageSkippedError must be 'skipped'
    (not 'failed', not 'completed'), and the DAG must keep running so dependents
    still execute and the run can reach completion. A naive 'skip = drop from
    completed-set' hangs the run forever; this asserts the skip is recorded AND
    its dependent still runs.
    """

    class _SkipIn(StageInput):
        pass

    class _SkipOut(StageOutput):
        pass

    class _Skip(StageHandler[_SkipIn, _SkipOut]):
        name = "optional"
        input_model = _SkipIn
        output_model = _SkipOut

        async def handle(self, payload: _SkipIn, ctx: object) -> _SkipOut:
            raise StageSkippedError("optional: not requested", missing=["optional"])

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(name="optional", handler_factory=lambda deps: _Skip()),
            StageSpec(
                name="after",
                depends_on=("optional",),
                handler_factory=lambda deps: _make_stage("after", 7)(),
            ),
        ],
    )
    persisted: list[tuple[str, str]] = []
    emitted: list[tuple[str, str | None]] = []

    async def emit(event, **kw):
        emitted.append((event, kw.get("stage_name")))

    async def persist(stage_name, status, output=None, error=None):
        persisted.append((stage_name, status))

    executor = DAGExecutor(bp, emit=emit, persist=persist)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    outputs = await executor.run(ctx, pause_after_stages=set())

    # The optional stage was skipped (not failed) and still appears in outputs
    # (as {}) so its dependent stayed eligible and ran.
    assert ("optional", "skipped") in persisted
    assert ("optional", "failed") not in persisted
    assert outputs.get("optional") == {}
    assert "after" in outputs  # dependent ran → no deadlock, no hang
    assert ("stage_skipped", "optional") in emitted


@pytest.mark.asyncio
async def test_skipped_stage_seeded_from_prior_outputs_is_not_rerun_on_resume() -> None:
    """Resume durability: a stage that finished as 'skipped' in a prior attempt is
    terminal-SUCCESS (run_finalizer.TERMINAL_STAGE_STATUSES = {'completed','skipped'})
    but is persisted with NO output. The dispatcher seeds it into
    ``prior_run_outputs`` with an empty dict ``{}`` so the executor treats it as
    already-completed and does NOT re-invoke its handler — while its dependent still
    runs. Regression: seeding only 'completed'+output rows dropped skips from the
    completed-set and needlessly RE-RAN them (re-issuing side-effect-free but wasteful
    LLM/network work) on every resume.
    """
    calls = {"optional": 0}

    class _OptIn(StageInput):
        pass

    class _OptOut(StageOutput):
        pass

    class _Opt(StageHandler[_OptIn, _OptOut]):
        name = "optional"
        input_model = _OptIn
        output_model = _OptOut

        async def handle(self, payload: _OptIn, ctx: object) -> _OptOut:
            calls["optional"] += 1  # must NOT fire on resume
            return _OptOut()

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(name="optional", handler_factory=lambda deps: _Opt()),
            StageSpec(
                name="after",
                depends_on=("optional",),
                handler_factory=lambda deps: _make_stage("after", 7)(),
            ),
        ],
    )
    persisted: list[tuple[str, str]] = []

    async def emit(event, **kw):
        pass

    async def persist(stage_name, status, output=None, error=None):
        persisted.append((stage_name, status))

    executor = DAGExecutor(bp, emit=emit, persist=persist)
    # Mirror the dispatcher's resume seed: a skipped stage carries {} output.
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={"optional": {}},
    )
    outputs = await executor.run(ctx, pause_after_stages=set())

    assert calls["optional"] == 0, "skipped stage must NOT be re-invoked on resume"
    assert ("optional", "running") not in persisted  # never re-dispatched
    assert outputs.get("optional") == {}  # empty seed preserved for dependents
    assert "after" in outputs  # dependent still ran off the {} seed


@pytest.mark.asyncio
async def test_skipped_pause_stage_does_not_pause_dag() -> None:
    """Regression (critique-only website_review wedge): a stage that is BOTH a
    pause_after gate AND gets skipped must NOT halt the DAG.

    A skipped stage produced nothing to review, so pausing there is meaningless —
    and it wedges the run: the executor used to flip `paused` for any processed
    pause-stage, but the finalizer's pause check counts only *completed* stages
    (`pause_set & completed`). The two disagreed → the run was abandoned in
    `running` and the orphan reconciler re-dispatched it forever. This pins that
    a skipped pause-stage lets the DAG drain to its dependent instead.
    """

    class _SkipIn(StageInput):
        pass

    class _SkipOut(StageOutput):
        pass

    class _Skip(StageHandler[_SkipIn, _SkipOut]):
        name = "gated"
        input_model = _SkipIn
        output_model = _SkipOut

        async def handle(self, payload: _SkipIn, ctx: object) -> _SkipOut:
            raise StageSkippedError("gated: not requested", missing=["gated"])

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(name="gated", handler_factory=lambda deps: _Skip()),
            StageSpec(
                name="after",
                depends_on=("gated",),
                handler_factory=lambda deps: _make_stage("after", 7)(),
            ),
        ],
    )
    persisted: list[tuple[str, str]] = []

    async def emit(event, **kw):
        pass

    async def persist(stage_name, status, output=None, error=None):
        persisted.append((stage_name, status))

    executor = DAGExecutor(bp, emit=emit, persist=persist)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    # 'gated' is a pause-after stage but it skips → must not pause; 'after' runs.
    outputs = await executor.run(ctx, pause_after_stages={"gated"})

    assert ("gated", "skipped") in persisted
    assert "after" in outputs, "skipped pause-stage must not halt the DAG"


@pytest.mark.asyncio
async def test_completed_pause_stage_still_pauses_dag() -> None:
    """Counterpart to the skip case: a pause_after stage that actually *completes*
    must still halt the DAG (so the human-review gate is preserved). Only the
    skip case is exempt.
    """
    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(name="a", handler_factory=lambda deps: _make_stage("a", 1)()),
            StageSpec(
                name="b",
                depends_on=("a",),
                handler_factory=lambda deps: _make_stage("b", 2)(),
            ),
        ],
    )

    async def emit(event, **kw):
        pass

    async def persist(*a, **kw):
        pass

    executor = DAGExecutor(bp, emit=emit, persist=persist)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    outputs = await executor.run(ctx, pause_after_stages={"a"})
    assert "a" in outputs and "b" not in outputs


@pytest.mark.asyncio
async def test_precondition_error_marks_stage_failed_not_skipped() -> None:
    """A StagePreconditionError (a REQUIRED input is missing, e.g. 'no shots to
    compile') must fail the stage — NOT be silently skipped. This guards the
    boundary between StageSkippedError (opt-out → skipped) and StagePreconditionError
    (degenerate input → failed); conflating them would let an empty compile
    'complete' green, the exact lie this work removes.
    """

    class _PreIn(StageInput):
        pass

    class _PreOut(StageOutput):
        pass

    class _Pre(StageHandler[_PreIn, _PreOut]):
        name = "compile"
        input_model = _PreIn
        output_model = _PreOut

        async def handle(self, payload: _PreIn, ctx: object) -> _PreOut:
            raise StagePreconditionError("compile: no shots to compile")

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[StageSpec(name="compile", handler_factory=lambda deps: _Pre())],
    )
    persisted: list[tuple[str, str]] = []

    async def emit(event, **kw):
        pass

    async def persist(stage_name, status, output=None, error=None):
        persisted.append((stage_name, status))

    executor = DAGExecutor(bp, emit=emit, persist=persist)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    with pytest.raises(StagePreconditionError):
        await executor.run(ctx, pause_after_stages=set())
    assert ("compile", "failed") in persisted
    assert ("compile", "skipped") not in persisted


# ── resume-on-timeout (durability P1) ────────────────────────────────────


def _raising_stage(name: str, exc: BaseException) -> type[StageHandler]:
    async def _handle(self, payload, ctx):
        raise exc

    return type(
        f"_Raise_{name}",
        (StageHandler,),
        {"name": name, "input_model": _NoopIn, "output_model": _NoopOut, "handle": _handle},
    )


def _timeout_bp(exc: BaseException) -> DAGBlueprint:
    # 'a' completes, 'b' raises the given (timeout) exception.
    return DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(name="a", handler_factory=lambda deps: _make_stage("a", 1)()),
            StageSpec(
                name="b",
                depends_on=("a",),
                handler_factory=lambda deps: _raising_stage("b", exc)(),
            ),
        ],
    )


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(TimeoutError("outer ceiling"), id="outer-TimeoutError"),
        pytest.param(
            __import__("w2v_common.stages", fromlist=["StageTimeoutError"]).StageTimeoutError(
                "inner all-attempts"
            ),
            id="inner-StageTimeoutError",
        ),
    ],
)
@pytest.mark.asyncio
async def test_timeout_under_cap_resumes_and_preserves_completed(exc) -> None:
    from src.services.runs.dag_executor import StageResumableTimeoutError

    persisted: list[tuple[str, str]] = []

    async def emit(event, **kw):
        pass

    async def persist(stage_name, status, output=None, error=None):
        persisted.append((stage_name, status))

    bumps: list[str] = []

    async def bump_attempts(stage_name):
        bumps.append(stage_name)
        return len(bumps)  # 1 on first timeout

    executor = DAGExecutor(
        _timeout_bp(exc),
        emit=emit,
        persist=persist,
        bump_attempts=bump_attempts,
        max_stage_attempts=2,
    )
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    with pytest.raises(StageResumableTimeoutError) as ei:
        await executor.run(ctx, pause_after_stages=set())

    assert ei.value.stage_name == "b"
    assert bumps == ["b"]  # counter bumped once
    # 'a' completed (preserved on resume); 'b' reset to pending, NOT failed.
    assert ("a", "completed") in persisted
    assert ("b", "pending") in persisted
    assert ("b", "failed") not in persisted


@pytest.mark.asyncio
async def test_the_cap_is_a_resume_count_so_n_permits_n_plus_1_executions() -> None:
    """`max_stage_attempts=N` allows N+1 EXECUTIONS of the stage, not N.

    The comparison is `attempts <= max_stage_attempts` on the POST-bump counter, so
    with a cap of 2: bump→1 resumes, bump→2 resumes, bump→3 fails. Three runs of the
    stage body, not two.

    That is what the field's *description* says ("how many times a stage may be
    RESUMED") and NOT what its *name* says. The gap is expensive: sizing a wall-clock
    budget from the name gives `2x ceiling`; the truth is `3x ceiling`. For an LLM
    stage inheriting `LLM_STAGE_BASE_CEILING_S = 1950` that is 3900s vs the real
    5850s — and the e2e harness deadline for the same blueprint is 3600s.

    Pinned here at the BOUNDARY, which neither neighbouring test covers: the existing
    pair exercises `attempts=1` (resume) and a hardcoded `attempts=3` (fail), so the
    `attempts == cap` edge — the one that decides whether N means N or N+1 — was
    unasserted. If someone changes `<=` to `<`, this is the test that says so.
    """
    from src.services.runs.dag_executor import StageResumableTimeoutError

    persisted: list[tuple[str, str]] = []

    async def emit(event, **kw):
        return None

    async def persist(stage_name, status, output=None, error=None):
        persisted.append((stage_name, status))

    async def bump_attempts(stage_name):
        return 2  # exactly AT the cap

    executor = DAGExecutor(
        _timeout_bp(TimeoutError("slow")),
        emit=emit,
        persist=persist,
        bump_attempts=bump_attempts,
        max_stage_attempts=2,
    )
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    # AT the cap the stage is still RESUMED — this is execution #3 being authorised.
    with pytest.raises(StageResumableTimeoutError):
        await executor.run(ctx, pause_after_stages=set())
    assert ("b", "pending") in persisted
    assert ("b", "failed") not in persisted


@pytest.mark.asyncio
async def test_timeout_over_cap_terminal_fails() -> None:
    persisted: list[tuple[str, str]] = []
    emitted: list[str] = []

    async def emit(event, **kw):
        emitted.append(event)

    async def persist(stage_name, status, output=None, error=None):
        persisted.append((stage_name, status))

    async def bump_attempts(stage_name):
        return 3  # already over the cap of 2

    executor = DAGExecutor(
        _timeout_bp(TimeoutError("slow")),
        emit=emit,
        persist=persist,
        bump_attempts=bump_attempts,
        max_stage_attempts=2,
    )
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    with pytest.raises(TimeoutError):  # original timeout re-raised, NOT resumable
        await executor.run(ctx, pause_after_stages=set())
    assert ("b", "failed") in persisted
    assert ("b", "pending") not in persisted
    assert "stage_failed" in emitted


@pytest.mark.asyncio
async def test_timeout_without_bump_wired_terminal_fails() -> None:
    """A bare executor (no resume wiring) keeps the old semantics: a timeout
    terminal-fails immediately."""
    persisted: list[tuple[str, str]] = []

    async def emit(event, **kw):
        pass

    async def persist(stage_name, status, output=None, error=None):
        persisted.append((stage_name, status))

    executor = DAGExecutor(_timeout_bp(TimeoutError("slow")), emit=emit, persist=persist)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    with pytest.raises(TimeoutError):
        await executor.run(ctx, pause_after_stages=set())
    assert ("b", "failed") in persisted


def _make_sleeper(name: str, seconds: float) -> type[StageHandler]:
    _s = seconds

    async def _handle(self, payload, ctx):
        await asyncio.sleep(_s)
        return _NoopOut(value=1)

    return type(
        f"_Sleeper_{name}",
        (StageHandler,),
        {"name": name, "input_model": _NoopIn, "output_model": _NoopOut, "handle": _handle},
    )


@pytest.mark.asyncio
async def test_none_timeout_inherits_injected_base_ceiling() -> None:
    """A stage whose ``default_timeout_seconds`` is None inherits the executor's
    injected base ceiling — raising the ceiling raises the effective per-stage timeout.

    Proves the GitOps knob (PipelineSettings.llm_stage_base_ceiling_s, injected by the
    dispatcher) actually reaches inherit-stages, i.e. it is NOT inert like the old
    dead stage_timeouts dict."""
    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        # No default_timeout_seconds => None => inherit the injected ceiling.
        stages=[StageSpec(name="a", handler_factory=lambda deps: _make_sleeper("a", 2.0)())],
    )

    async def emit(event, **kw):
        pass

    async def persist(stage_name, status, output=None, error=None):
        pass

    # Injected ceiling below the stage's work -> the outer TimeoutMiddleware fires.
    tight = DAGExecutor(bp, emit=emit, persist=persist, stage_base_ceiling_seconds=1)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    with pytest.raises(TimeoutError):
        await tight.run(ctx, pause_after_stages=set())

    # Injected ceiling above the work -> the same stage completes.
    loose = DAGExecutor(bp, emit=emit, persist=persist, stage_base_ceiling_seconds=5)
    ctx2 = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    await loose.run(ctx2, pause_after_stages=set())


@pytest.mark.asyncio
async def test_non_timeout_exception_still_terminal_even_with_resume_wired() -> None:
    """StageRetriesExhaustedError / bad-output errors are deterministic failures,
    NOT slowness — they must terminal-fail immediately, never resume."""
    from w2v_common.stages import StageRetriesExhaustedError

    persisted: list[tuple[str, str]] = []
    bumped = False

    async def emit(event, **kw):
        pass

    async def persist(stage_name, status, output=None, error=None):
        persisted.append((stage_name, status))

    async def bump_attempts(stage_name):
        nonlocal bumped
        bumped = True
        return 1

    exc = StageRetriesExhaustedError(
        stage_name="b", attempts=2, last_error=ValueError("bad output")
    )
    executor = DAGExecutor(
        _timeout_bp(exc),
        emit=emit,
        persist=persist,
        bump_attempts=bump_attempts,
        max_stage_attempts=2,
    )
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    with pytest.raises(StageRetriesExhaustedError):
        await executor.run(ctx, pause_after_stages=set())
    assert ("b", "failed") in persisted
    assert not bumped  # resume path never entered


@pytest.mark.asyncio
async def test_cross_group_coeligible_stages_run_concurrently() -> None:
    """Every co-eligible stage runs in ONE round, concurrently — even across
    different ``parallel_group`` labels.

    Two independent stages carrying *different* ``parallel_group`` values used to
    be serialized one group per round (the executor dispatched only the largest
    group each round). They must now run together: ``depends_on`` is the sole
    ordering constraint. Proven with a rendezvous — each stage blocks until the
    other has started, so a serialized executor would deadlock (and time out).
    """
    x_started = asyncio.Event()
    y_started = asyncio.Event()

    def _rendezvous_stage(name: str, mine: asyncio.Event, other: asyncio.Event):
        async def _handle(self, payload, ctx):
            mine.set()
            # Serialized execution never sets ``other`` (it hasn't run yet) → the
            # wait_for times out and the run fails. Concurrent execution completes.
            await asyncio.wait_for(other.wait(), timeout=3.0)
            return _NoopOut(value=1)

        return type(
            f"_Rzv_{name}",
            (StageHandler,),
            {
                "name": name,
                "input_model": _NoopIn,
                "output_model": _NoopOut,
                "handle": _handle,
            },
        )

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(
                name="x",
                parallel_group="g1",
                handler_factory=lambda deps: _rendezvous_stage("x", x_started, y_started)(),
            ),
            StageSpec(
                name="y",
                parallel_group="g2",
                handler_factory=lambda deps: _rendezvous_stage("y", y_started, x_started)(),
            ),
        ],
    )

    async def emit(event, **kw):
        pass

    async def persist(*a, **kw):
        pass

    executor = DAGExecutor(bp, emit=emit, persist=persist)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    outputs = await asyncio.wait_for(executor.run(ctx, pause_after_stages=set()), timeout=6.0)
    assert "x" in outputs and "y" in outputs


@pytest.mark.asyncio
async def test_sibling_stages_cancelled_and_drained_when_one_fails() -> None:
    """When one stage in a co-eligible batch fails, its still-running siblings are
    cancelled and drained before the failure propagates — nothing (a non-idempotent
    render/TTS/upload) is left running detached after the run is already failing.
    """
    boom_ran = asyncio.Event()
    slow_cancelled = asyncio.Event()

    def _boom_stage():
        async def _handle(self, payload, ctx):
            boom_ran.set()
            # Let the slow sibling get going first, then fail the batch.
            await asyncio.sleep(0.02)
            raise RuntimeError("kaboom")

        return type(
            "_Boom2",
            (StageHandler,),
            {"name": "boom", "input_model": _NoopIn, "output_model": _NoopOut, "handle": _handle},
        )

    def _slow_stage():
        async def _handle(self, payload, ctx):
            try:
                await asyncio.sleep(30)  # far longer than the boom sibling
            except asyncio.CancelledError:
                slow_cancelled.set()
                raise
            return _NoopOut(value=1)

        return type(
            "_Slow2",
            (StageHandler,),
            {"name": "slow", "input_model": _NoopIn, "output_model": _NoopOut, "handle": _handle},
        )

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(name="boom", handler_factory=lambda deps: _boom_stage()()),
            StageSpec(name="slow", handler_factory=lambda deps: _slow_stage()()),
        ],
    )

    async def emit(event, **kw):
        pass

    async def persist(*a, **kw):
        pass

    executor = DAGExecutor(bp, emit=emit, persist=persist)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    with pytest.raises(RuntimeError, match="kaboom"):
        await asyncio.wait_for(executor.run(ctx, pause_after_stages=set()), timeout=6.0)
    # The slow sibling was cancelled (not left running) before the error surfaced.
    assert slow_cancelled.is_set()


# --- run-level deadline -------------------------------------------------------
#
# `max_stage_attempts` is per-STAGE, so nothing bounded the RUN. Measured over 30
# days of production runs: five burned more than two hours each, all already lost,
# the worst 9.7 HOURS ("all 8 attempts timed out"), against a longest-ever
# SUCCESSFUL run of 1.44h.


def _deadline_executor(
    blueprint: DAGBlueprint,
    *,
    run_deadline_at: datetime | None,
    attempts_seen: list[int],
) -> tuple[DAGExecutor, list[tuple[str, str, str | None]]]:
    persisted: list[tuple[str, str, str | None]] = []

    async def _persist(stage: str, status: str, *, output=None, error=None) -> None:
        persisted.append((stage, status, error))

    async def _emit(event: str, **_kw: object) -> None:
        return None

    async def _bump(_stage: str) -> int:
        attempts_seen.append(len(attempts_seen) + 1)
        return len(attempts_seen)

    return (
        DAGExecutor(
            blueprint,
            emit=_emit,
            persist=_persist,
            bump_attempts=_bump,
            max_stage_attempts=5,
            run_deadline_at=run_deadline_at,
        ),
        persisted,
    )


def _ctx() -> RunContext:
    return RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})


def _timing_out_blueprint() -> DAGBlueprint:
    async def _handle(self, payload, ctx):
        raise StageTimeoutError("stage 'slow': all 2 attempts timed out")

    slow = type(
        "_Slow",
        (StageHandler,),
        {
            "name": "slow",
            "input_model": _NoopIn,
            "output_model": _NoopOut,
            "handle": _handle,
        },
    )
    return DAGBlueprint(
        id="deadline_probe",
        project_type="generic",
        stages=[StageSpec(name="slow", handler_factory=lambda deps: slow())],
    )


@pytest.mark.asyncio
async def test_a_run_inside_its_deadline_still_resumes() -> None:
    """The control. The deadline must not turn every timeout terminal."""
    attempts: list[int] = []
    executor, persisted = _deadline_executor(
        _timing_out_blueprint(),
        run_deadline_at=datetime.now(UTC) + timedelta(hours=1),
        attempts_seen=attempts,
    )
    with pytest.raises(StageResumableTimeoutError):
        await executor.run(_ctx(), pause_after_stages=set())
    assert ("slow", "pending", None) in persisted, "an in-budget timeout must reset to pending"


@pytest.mark.asyncio
async def test_a_run_past_its_deadline_is_not_resumed() -> None:
    attempts: list[int] = []
    executor, persisted = _deadline_executor(
        _timing_out_blueprint(),
        run_deadline_at=datetime.now(UTC) - timedelta(seconds=30),
        attempts_seen=attempts,
    )
    with pytest.raises(StageTimeoutError):
        await executor.run(_ctx(), pause_after_stages=set())

    failures = [p for p in persisted if p[1] == "failed"]
    assert failures, "an over-deadline timeout must terminal-fail, not resume"
    error = failures[0][2] or ""
    assert "RunDeadlineExceeded" in error, error
    assert "all 2 attempts timed out" in error, "the causing timeout must survive in the message"
    assert not [p for p in persisted if p[1] == "pending"], "it must not also reset to pending"


@pytest.mark.asyncio
async def test_no_deadline_is_byte_identical_to_before() -> None:
    attempts: list[int] = []
    executor, persisted = _deadline_executor(
        _timing_out_blueprint(), run_deadline_at=None, attempts_seen=attempts
    )
    with pytest.raises(StageResumableTimeoutError):
        await executor.run(_ctx(), pause_after_stages=set())
    assert ("slow", "pending", None) in persisted


@pytest.mark.asyncio
async def test_cancellation_records_an_interruption_but_a_timeout_does_not() -> None:
    """The executor is the ONLY place that still knows which reset happened.

    Both branches end in ``persist(name, 'pending')``, so by the time the write
    reaches ``StageOutputRepository.mark_pending`` the two are indistinguishable.
    The distinction has to be recorded here, at the ``except``, or it is lost:
    that loss is what let a drained stage resume with ``attempts == 0`` and an
    untouched ``started_at``, i.e. looking like a clean single execution.
    """
    started = asyncio.Event()
    release = asyncio.Event()

    class _BlockIn(StageInput):
        pass

    class _BlockOut(StageOutput):
        pass

    class _Block(StageHandler[_BlockIn, _BlockOut]):
        name = "block"
        input_model = _BlockIn
        output_model = _BlockOut

        async def handle(self, payload: _BlockIn, ctx: object) -> _BlockOut:
            started.set()
            await release.wait()
            return _BlockOut()

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[StageSpec(name="block", handler_factory=lambda deps: _Block())],
    )
    interrupted: list[str] = []
    bumped: list[str] = []

    async def emit(event, **kw):
        pass

    async def persist(stage_name, status, output=None, error=None):
        pass

    async def note_interrupted(stage_name):
        interrupted.append(stage_name)
        return len(interrupted)

    async def bump_attempts(stage_name):
        bumped.append(stage_name)
        return len(bumped)

    executor = DAGExecutor(
        bp,
        emit=emit,
        persist=persist,
        bump_attempts=bump_attempts,
        note_interrupted=note_interrupted,
        max_stage_attempts=3,
    )
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    task = asyncio.create_task(executor.run(ctx, pause_after_stages=set()))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert interrupted == ["block"], (
        "a drain went unrecorded, so the resumed row is indistinguishable from a "
        "stage that was never interrupted"
    )
    assert bumped == [], (
        "a drain was recorded as a TIMEOUT retry — the two paths are still conflated, "
        "just under a different counter"
    )


@pytest.mark.asyncio
async def test_a_resumable_timeout_is_not_recorded_as_an_interruption() -> None:
    """The other direction of the same invariant.

    A genuine timeout must bump ``attempts`` and leave ``interruptions`` alone. If
    both counters moved on both paths they would carry the same information as one
    counter, and "was this stage drained, or was it slow?" would still be
    unanswerable from the row.
    """

    class _SlowIn(StageInput):
        pass

    class _SlowOut(StageOutput):
        pass

    class _Slow(StageHandler[_SlowIn, _SlowOut]):
        name = "slow"
        input_model = _SlowIn
        output_model = _SlowOut

        async def handle(self, payload: _SlowIn, ctx: object) -> _SlowOut:
            await asyncio.sleep(30)
            return _SlowOut()

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(
                name="slow",
                handler_factory=lambda deps: _Slow(),
                default_timeout_seconds=1,
            )
        ],
    )
    interrupted: list[str] = []
    bumped: list[str] = []

    async def emit(event, **kw):
        pass

    async def persist(stage_name, status, output=None, error=None):
        pass

    async def note_interrupted(stage_name):
        interrupted.append(stage_name)
        return len(interrupted)

    async def bump_attempts(stage_name):
        bumped.append(stage_name)
        return len(bumped)

    executor = DAGExecutor(
        bp,
        emit=emit,
        persist=persist,
        bump_attempts=bump_attempts,
        note_interrupted=note_interrupted,
        max_stage_attempts=3,
    )
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    # Under the cap, so the executor asks for a resume rather than terminal-failing.
    with pytest.raises(StageResumableTimeoutError):
        await executor.run(ctx, pause_after_stages=set())

    assert bumped == ["slow"], f"the timeout did not bump the resume counter: {bumped}"
    assert interrupted == [], (
        "a TIMEOUT was recorded as a drain/cancel interruption, so the two "
        "mechanisms are conflated in the opposite direction"
    )
