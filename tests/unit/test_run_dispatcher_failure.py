"""Negative-path coverage for ``RunDispatcher._run``.

The happy path is exercised by the integration smoke in
``tests/services/test_run_dispatcher.py``. These unit tests pin the two
error transitions that the integration test never reaches:

* executor.run raises → ``RunRepository.mark_failed`` + ``run_failed`` event.
* dispatcher.shutdown() during an in-flight task → task cancelled,
  ``mark_cancelled`` called, ``run_cancelled`` event published.

Without these, a future refactor of the ``try/except`` shape could
silently regress error handling and would only be caught when something
actually fails in production.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.models.failure_class import OUR_DEFECT, UNKNOWN

pytestmark = pytest.mark.asyncio


class _FakeRedis:
    """Minimal async redis stand-in capturing every ``publish`` call."""

    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    async def publish(self, channel: str, message: str) -> int:
        self.published.append((channel, json.loads(message)))
        return 1


def _make_run_obj() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        blueprint_id="bp-1",
        deliverable_id=uuid4(),
        config_snapshot={"tenant_id": str(uuid4()), "subject_id": str(uuid4())},
        pause_after_stages=[],
        # Real column; the dispatcher reads it to anchor the run deadline.
        started_at=None,
    )


def _session_factory(session: AsyncMock):
    @asynccontextmanager
    async def _factory():
        yield session

    return _factory


async def _drive_dispatcher(
    executor_side_effect,
    *,
    cancel_before_complete: bool = False,
    user_cancel: bool = False,
    mark_cancelled_return: object = None,
    mark_failed_return: object = True,
    topo_order: list[str] | None = None,
    stage_rows: list | None = None,
    project_id: object = None,
    snapshot_extra: dict | None = None,
) -> tuple[AsyncMock, AsyncMock, _FakeRedis]:
    """Construct the dispatcher with stub deps and drive a single run.

    Returns the captured ``RunRepository`` mock, the ``StageOutputRepository``
    mock, and the FakeRedis so each test can assert on the failure /
    cancellation surface (including the stranded-stage-row sweep).

    ``mark_failed_return`` is the write-once transition bool the stubbed
    ``mark_failed`` reports (default ``True`` = this pod won the transition;
    pass ``False`` to simulate a lost failure race where another replica already
    committed the row ``failed`` — the sweep must then be skipped).

    ``cancel_before_complete`` drives the DRAIN path (``dispatcher.shutdown()``
    → ``_draining`` True → run left ``running``). ``user_cancel`` instead drives
    the explicit-user-cancel path (``dispatcher.cancel()`` → CancelledError with
    ``_draining`` False → the ``mark_cancelled`` else-branch). ``run_obj`` has no
    ``status`` attr, so the handler's ``status == "paused"`` intent read is False
    and control reaches the cancel branch. ``mark_cancelled_return`` overrides the
    transition-bool that stubbed ``mark_cancelled`` reports (``None`` leaves the
    historical truthy ``AsyncMock()``; pass ``False`` to simulate a lost cancel
    race where the row was already committed ``cancelled`` on another replica)."""
    from src.services.run_dispatcher import RunDispatcher

    run_obj = _make_run_obj()
    if snapshot_extra:
        run_obj.config_snapshot.update(snapshot_extra)
    session = AsyncMock()
    session.commit = AsyncMock()

    # Pin the parent-project lookup so the dispatcher's
    # ``await session.execute(...).first()`` returns a deterministic row.
    # These failure tests don't care about parent-channel publishes, so we
    # leave both ids unset (NULL parent → child_status mirror is a no-op).
    # ``project_id`` sets the run's own (child) project for the w2v-project
    # attribution tests; the parent stays NULL either way.
    proj_row = SimpleNamespace(id=project_id, parent_project_id=None)
    exec_result = MagicMock()
    exec_result.first = MagicMock(return_value=proj_row)
    session.execute = AsyncMock(return_value=exec_result)

    run_repo_instance = AsyncMock()
    run_repo_instance.get = AsyncMock(return_value=run_obj)
    run_repo_instance.mark_started = AsyncMock()
    run_repo_instance.mark_failed = AsyncMock(return_value=mark_failed_return)
    run_repo_instance.mark_cancelled = AsyncMock()
    if mark_cancelled_return is not None:
        run_repo_instance.mark_cancelled = AsyncMock(return_value=mark_cancelled_return)
    run_repo_instance.mark_completed = AsyncMock()
    run_repo_instance.mark_paused = AsyncMock()
    run_repo_instance.mark_current_stage = AsyncMock()

    stage_repo_instance = AsyncMock()
    stage_repo_instance.list_for_run = AsyncMock(return_value=stage_rows or [])

    registry = MagicMock()
    blueprint = MagicMock()
    blueprint.topological_order.return_value = topo_order or []
    registry.resolve_by_id.return_value = blueprint

    redis = _FakeRedis()

    executor_instance = MagicMock()
    executor_instance.run = AsyncMock(side_effect=executor_side_effect)

    with (
        patch("src.services.run_dispatcher.RunRepository", return_value=run_repo_instance),
        patch(
            "src.services.run_dispatcher.StageOutputRepository",
            return_value=stage_repo_instance,
        ),
        patch("src.services.run_dispatcher.DAGExecutor", return_value=executor_instance),
    ):
        dispatcher = RunDispatcher(
            session_factory=_session_factory(session),
            registry=registry,
            redis=redis,  # ty: ignore[invalid-argument-type]
        )
        await dispatcher.dispatch(run_obj.id)
        if cancel_before_complete:
            # Give the dispatch task a tick to enter the executor.run await.
            await asyncio.sleep(0)
            await dispatcher.shutdown()
        elif user_cancel:
            # Explicit user cancel (no drain): cancel the in-flight task so the
            # CancelledError handler takes the else-branch (mark_cancelled).
            await asyncio.sleep(0)
            await dispatcher.cancel(run_obj.id)
        else:
            # Wait for the dispatch task to finish naturally.
            task = dispatcher._tasks.get(run_obj.id)
            if task is not None:
                await asyncio.wait_for(task, timeout=2.0)

    return run_repo_instance, stage_repo_instance, redis


class TestExecutorFailure:
    async def test_executor_run_raises_marks_run_failed_and_publishes_event(self) -> None:
        run_repo, _stage_repo, redis = await _drive_dispatcher(
            executor_side_effect=RuntimeError("upstream blew up")
        )

        run_repo.mark_failed.assert_awaited_once()
        # mark_failed is called with (run_id, "RuntimeError: upstream blew up")
        _run_id_arg, error_arg = run_repo.mark_failed.call_args.args
        assert "RuntimeError" in error_arg
        assert "upstream blew up" in error_arg

        events = [msg["event"] for _channel, msg in redis.published]
        assert "run_started" in events
        assert "run_failed" in events

    async def test_run_completed_not_published_on_executor_failure(self) -> None:
        _run_repo, _stage_repo, redis = await _drive_dispatcher(
            executor_side_effect=RuntimeError("boom")
        )
        events = [msg["event"] for _channel, msg in redis.published]
        assert "run_completed" not in events

    async def test_failure_sweeps_stranded_stage_rows_on_transition(self) -> None:
        """When this pod wins the run→failed transition, the dispatcher sweeps the
        run's leftover non-terminal stage rows to 'cancelled' so independent
        sibling / never-scheduled stages don't strand 'pending' forever."""
        run_repo, stage_repo, _redis = await _drive_dispatcher(
            executor_side_effect=RuntimeError("boom"), mark_failed_return=True
        )

        run_repo.mark_failed.assert_awaited_once()
        stage_repo.bulk_cancel_non_terminal.assert_awaited_once()
        # Swept the SAME run that was failed.
        (swept_run_id,) = stage_repo.bulk_cancel_non_terminal.call_args.args
        (failed_run_id, _err) = run_repo.mark_failed.call_args.args
        assert swept_run_id == failed_run_id

    async def test_dag_failure_records_the_attributed_failure_class(self) -> None:
        """The DAG-execution failure route must pass the ATTRIBUTED class, not a
        constant.

        This is the only run-failure site holding the exception object, so it is
        the only one that can attribute anything. Driven with an
        ``AttributeError`` — a Python programming error, i.e. unambiguously ours —
        so the assertion distinguishes real attribution from the ``unknown``
        default that a site passing nothing would land on.
        """
        run_repo, _stage_repo, _redis = await _drive_dispatcher(
            executor_side_effect=AttributeError("'NoneType' has no attribute 'clips'")
        )

        run_repo.mark_failed.assert_awaited_once()
        assert run_repo.mark_failed.call_args.kwargs.get("failure_class") == OUR_DEFECT, (
            "the DAG-failure route must call attribute_exception(exc); passing no "
            "failure_class (or a hardcoded one) leaves every run this route fails "
            "recorded as 'unknown' and the column carries no information."
        )

    async def test_a_byo_refusal_records_customer_input_and_its_failure_detail(self) -> None:
        """#1106: a byo_required tenant's refusal must reach ``runs.failure_detail``
        (the code + modality the frontend branches on), not just the prose."""
        from src.models.failure_class import CUSTOMER_INPUT
        from src.services.providers.errors import SlotNotConfiguredError

        run_repo, _stage_repo, _redis = await _drive_dispatcher(
            executor_side_effect=SlotNotConfiguredError("asr")
        )

        run_repo.mark_failed.assert_awaited_once()
        kwargs = run_repo.mark_failed.call_args.kwargs
        assert kwargs.get("failure_class") == CUSTOMER_INPUT
        assert kwargs.get("failure_detail") == {
            "code": "byo_provider_required",
            "modality": "asr",
            "slots": "asr",
            "message": "Configure your own ASR provider to continue",
        }

    async def test_a_failure_without_structured_detail_records_none(self) -> None:
        run_repo, _stage_repo, _redis = await _drive_dispatcher(
            executor_side_effect=RuntimeError("boom")
        )
        assert run_repo.mark_failed.call_args.kwargs.get("failure_detail") is None

    async def test_an_unattributable_dag_failure_records_unknown(self) -> None:
        """The converse, and the more important half: no guessing.

        A bare ``RuntimeError`` carries no structural signal. It must reach the
        DB as ``unknown`` — never forced into ``our_defect`` just because it
        surfaced inside our process. Without this assertion the test above is
        satisfied by hardcoding ``failure_class=OUR_DEFECT`` at the call site.
        """
        run_repo, _stage_repo, _redis = await _drive_dispatcher(
            executor_side_effect=RuntimeError("boom")
        )

        run_repo.mark_failed.assert_awaited_once()
        assert run_repo.mark_failed.call_args.kwargs.get("failure_class") == UNKNOWN

    async def test_post_drain_stage_failure_does_not_guess_a_class(self) -> None:
        """The post-drain route has only prose, so it must attribute nothing.

        It is reached with no exception object — the executor returned normally
        and all this site has is a failed stage row's free-text ``error``.
        Re-deriving a class from prose is exactly the practice this column
        replaces, so the route must fall through to ``mark_failed``'s 'unknown'
        default rather than pattern-matching the string.
        """
        rows = [SimpleNamespace(stage_name="analyze", status="failed", error="boom")]
        run_repo, _stage_repo, _redis = await _drive_dispatcher(
            executor_side_effect=None, topo_order=["analyze"], stage_rows=rows
        )

        run_repo.mark_failed.assert_awaited_once()
        assert run_repo.mark_failed.call_args.kwargs.get("failure_class") is None, (
            "the post-drain route must pass NO failure_class and inherit the "
            "'unknown' default; anything else is a guess made from prose."
        )

    async def test_failure_does_not_sweep_when_transition_lost(self) -> None:
        """Two-replica failure race: another pod already committed the run
        'failed' (mark_failed → False). This pod must NOT sweep — the winner
        already terminalized, and a stray sweep could clobber stage rows a
        concurrent resume is legitimately touching."""
        run_repo, stage_repo, _redis = await _drive_dispatcher(
            executor_side_effect=RuntimeError("boom"), mark_failed_return=False
        )

        run_repo.mark_failed.assert_awaited_once()
        stage_repo.bulk_cancel_non_terminal.assert_not_awaited()


class TestDispatcherShutdown:
    async def test_shutdown_drain_leaves_run_running_for_resume(self) -> None:
        """Shutdown drain MUST NOT mark in-flight runs ``cancelled`` —
        Phase 1b flipped this so a drained run stays ``running`` and the
        next pod's startup reconciler resumes it. ``run_started`` is
        still published at dispatch start; ``run_cancelled`` is NOT,
        and ``mark_cancelled`` is never awaited."""

        async def block_forever(*_args, **_kwargs):
            await asyncio.sleep(60)

        run_repo, _stage_repo, redis = await _drive_dispatcher(
            executor_side_effect=block_forever, cancel_before_complete=True
        )

        run_repo.mark_cancelled.assert_not_awaited()
        events = [msg["event"] for _channel, msg in redis.published]
        assert "run_started" in events
        assert "run_cancelled" not in events


class TestUserCancel:
    """The explicit-user-cancel CancelledError branch now gates its
    ``run_cancelled`` SSE (and the implicit runs_total{cancelled} increment) on
    ``mark_cancelled``'s write-once bool — the same shape #523 gave the
    completed/failed gates — so a two-replica cancel race publishes/counts
    exactly once."""

    async def test_user_cancel_transition_publishes_run_cancelled_once(self) -> None:
        """First cancel wins: mark_cancelled transitions the row (True) →
        exactly one ``run_cancelled`` published."""

        async def block_forever(*_args, **_kwargs):
            await asyncio.sleep(60)

        run_repo, _stage_repo, redis = await _drive_dispatcher(
            executor_side_effect=block_forever,
            user_cancel=True,
            mark_cancelled_return=True,
        )

        run_repo.mark_cancelled.assert_awaited_once()
        events = [msg["event"] for _channel, msg in redis.published]
        assert events.count("run_cancelled") == 1

    async def test_no_publish_when_cancel_lost_the_race(self) -> None:
        """Two-replica cancel race: the row was already committed ``cancelled``
        on another replica, so this pod's ``mark_cancelled`` transitions 0 rows
        (returns False). The dispatcher must NOT publish a second
        ``run_cancelled`` nor re-increment runs_total{cancelled}."""

        async def block_forever(*_args, **_kwargs):
            await asyncio.sleep(60)

        run_repo, _stage_repo, redis = await _drive_dispatcher(
            executor_side_effect=block_forever,
            user_cancel=True,
            mark_cancelled_return=False,
        )

        run_repo.mark_cancelled.assert_awaited_once()
        events = [msg["event"] for _channel, msg in redis.published]
        assert "run_cancelled" not in events


class TestPostDrainStageFailure:
    """The executor returns NORMALLY (no raise) but a stage row is 'failed' — the
    resumable-timeout-drain transient-failure case. ``decide_run_terminal_state``
    now returns FAILED (previously CONTINUE → wedged run until the 2h backstop).
    The dispatcher must fail the run loud, sweep leftovers, and publish
    ``run_failed`` — the same terminal shape as the except-Exception path, gated
    on the write-once ``mark_failed`` transition."""

    @staticmethod
    async def _returns_normally(*_args, **_kwargs):
        return {}

    async def test_post_drain_failed_stage_fails_run_and_sweeps(self) -> None:
        good = SimpleNamespace(stage_name="a", status="completed", error=None, output=None)
        failed = SimpleNamespace(
            stage_name="b", status="failed", error="ValueError: transient blip", output=None
        )
        run_repo, stage_repo, redis = await _drive_dispatcher(
            executor_side_effect=self._returns_normally,
            topo_order=["a", "b"],
            stage_rows=[good, failed],
            mark_failed_return=True,
        )

        run_repo.mark_failed.assert_awaited_once()
        (_run_id_arg, reason) = run_repo.mark_failed.call_args.args
        # The reason surfaces the offending stage + its stored error (a JSON blob).
        assert "stage_failed_post_drain" in reason
        assert '"b"' in reason
        assert "transient blip" in reason
        # Leftover non-terminal rows are swept on the winning transition.
        stage_repo.bulk_cancel_non_terminal.assert_awaited_once()
        (swept_run_id,) = stage_repo.bulk_cancel_non_terminal.call_args.args
        (failed_run_id, _err) = run_repo.mark_failed.call_args.args
        assert swept_run_id == failed_run_id
        # Terminal failure is published; the run is NOT completed.
        events = [msg["event"] for _channel, msg in redis.published]
        assert "run_failed" in events
        assert "run_completed" not in events
        run_repo.mark_completed.assert_not_awaited()

    async def test_post_drain_cancelled_row_also_fails_run(self) -> None:
        # 'cancelled' is terminal-bad too (it only appears via the failure sweep).
        good = SimpleNamespace(stage_name="a", status="completed", error=None, output=None)
        cancelled = SimpleNamespace(stage_name="b", status="cancelled", error=None, output=None)
        run_repo, stage_repo, _redis = await _drive_dispatcher(
            executor_side_effect=self._returns_normally,
            topo_order=["a", "b"],
            stage_rows=[good, cancelled],
            mark_failed_return=True,
        )
        run_repo.mark_failed.assert_awaited_once()
        stage_repo.bulk_cancel_non_terminal.assert_awaited_once()

    async def test_post_drain_failed_lost_race_no_sweep_no_publish(self) -> None:
        """Two-replica race: another pod already committed the run 'failed'
        (mark_failed → False). This pod must NOT sweep nor publish a second
        ``run_failed`` — mirrors the except-Exception lost-race guard."""
        good = SimpleNamespace(stage_name="a", status="completed", error=None, output=None)
        failed = SimpleNamespace(stage_name="b", status="failed", error="boom", output=None)
        run_repo, stage_repo, redis = await _drive_dispatcher(
            executor_side_effect=self._returns_normally,
            topo_order=["a", "b"],
            stage_rows=[good, failed],
            mark_failed_return=False,
        )
        run_repo.mark_failed.assert_awaited_once()
        stage_repo.bulk_cancel_non_terminal.assert_not_awaited()
        events = [msg["event"] for _channel, msg in redis.published]
        assert "run_failed" not in events


class TestReapTerminalizedRun:
    """``reap_terminalized_run`` is the cleanup half of a terminal transition, for
    the ONE writer that flips the run row itself: the reconciler's stale/wedged
    fail (``lifecycle.reconciler._resolve_orphan_run``). It must do all four
    things the dispatcher's own failure gates do."""

    async def _reap(self, *, config_snapshot: dict | None = None, compile_cancel=None):
        from src.services.run_dispatcher import RunDispatcher

        run_id = uuid4()
        child_id = uuid4()

        session = AsyncMock()
        proj_row = SimpleNamespace(id=uuid4(), parent_project_id=uuid4())
        exec_result = MagicMock()
        exec_result.first = MagicMock(return_value=proj_row)
        session.execute = AsyncMock(return_value=exec_result)

        stage_repo = AsyncMock()
        stage_repo.bulk_cancel_non_terminal = AsyncMock(return_value=2)

        # The reaper reads the RUN row for its snapshot (tenant + provenance);
        # the child read is the sub-DAG cancel's. Keyed by id so the two cannot
        # be confused. Default snapshot is a vouched, billable tenant.
        snapshot = (
            config_snapshot
            if config_snapshot is not None
            else {"tenant_id": str(uuid4()), "tenant_source": "jwt-claim"}
        )
        rows = {
            run_id: SimpleNamespace(id=run_id, status="failed", config_snapshot=snapshot),
            child_id: SimpleNamespace(id=child_id, status="running"),
        }
        run_repo = AsyncMock()
        run_repo.get = AsyncMock(side_effect=lambda rid: rows.get(rid))
        run_repo.mark_cancelled = AsyncMock()
        if compile_cancel is None:
            compile_cancel = AsyncMock()

        project_repo = AsyncMock()

        with (
            patch("src.services.run_dispatcher.StageOutputRepository", return_value=stage_repo),
            patch("src.services.run_dispatcher.RunRepository", return_value=run_repo),
            patch("src.services.run_dispatcher.ProjectRepository", return_value=project_repo),
            patch(
                "src.services.run_dispatcher.subdag_child_run_ids",
                AsyncMock(return_value=[child_id]),
            ),
            patch("src.services.run_dispatcher.cancel_compiles_for_run", compile_cancel),
        ):
            dispatcher = RunDispatcher(
                session_factory=_session_factory(session),
                registry=MagicMock(),
                redis=_FakeRedis(),  # ty: ignore[invalid-argument-type]
            )
            await dispatcher.reap_terminalized_run(session, run_id)
        return run_id, child_id, stage_repo, run_repo, project_repo, compile_cancel, proj_row

    async def test_sweeps_the_stranded_stage_rows(self) -> None:
        run_id, _c, stage_repo, _rr, _pr, _cc, _proj = await self._reap()
        stage_repo.bulk_cancel_non_terminal.assert_awaited_once()
        (swept,) = stage_repo.bulk_cancel_non_terminal.call_args.args
        assert swept == run_id

    async def test_reaps_the_paid_subdag_children(self) -> None:
        """Course lessons carry ``parent_run_id=None`` — nothing else can see them,
        so each keeps spending on its own DAG until it finishes."""
        _r, child_id, _sr, run_repo, _pr, _cc, _proj = await self._reap()
        run_repo.mark_cancelled.assert_awaited_once_with(child_id)

    async def test_cancels_the_runs_own_in_flight_compile(self) -> None:
        """Stopping the orchestrator's task does not stop the render already
        submitted to the compile service — it keeps rendering and being billed."""
        run_id, child_id, _sr, _rr, _pr, compile_cancel, _proj = await self._reap()
        cancelled = {c.args[0] for c in compile_cancel.call_args_list}
        assert run_id in cancelled, "the run's OWN compile was never cancelled"
        assert child_id in cancelled, "each child's compile must be cancelled too"

    async def test_recomputes_the_owning_project_and_its_parent(self) -> None:
        """Otherwise the project (and its course parent) sits at 'producing'
        forever with no run left to move it."""
        _r, _c, _sr, _rr, project_repo, _cc, proj_row = await self._reap()
        recomputed = [c.args[0] for c in project_repo.recompute_status.call_args_list]
        assert recomputed == [proj_row.id, proj_row.parent_project_id]

    async def test_compile_cancel_runs_as_the_runs_tenant(self) -> None:
        """The reconciler task has no inbound request, so nothing bound identity —
        and ``VideoCompileClient`` bakes an SA Bearer, so ``_apply_outbound_tenant``
        would take the ``service_credentialed`` early return and cancel as the SA
        tenant. Once video_compile is DELEGATING (w2v_common#190) every job is
        filed under the CUSTOMER tenant, an SA-scoped cancel matches none, and the
        orphaned render runs to completion billing the customer. The cancel must
        see the run's snapshot tenant, vouched, at the moment it is issued."""
        from uuid import UUID

        from w2v_common.auth.identity_context import (
            get_tenant_id_from_context,
            get_tenant_source_from_context,
            is_tenant_vouched,
        )
        from w2v_common.tenant import TenantSource

        tenant = uuid4()
        seen: list[tuple[UUID | None, object, bool]] = []

        async def _capture(_rid):
            seen.append(
                (
                    get_tenant_id_from_context(),
                    get_tenant_source_from_context(),
                    is_tenant_vouched(),
                )
            )
            return 0

        run_id, child_id, _sr, _rr, _pr, compile_cancel, _proj = await self._reap(
            config_snapshot={
                "tenant_id": str(tenant),
                "tenant_source": TenantSource.JWT_CLAIM.value,
                "license_id": "lic-1",
                "subject_id": str(uuid4()),
            },
            compile_cancel=AsyncMock(side_effect=_capture),
        )
        cancelled = [c.args[0] for c in compile_cancel.call_args_list]
        assert run_id in cancelled and child_id in cancelled
        # Every compile cancel — the run's own AND each child's — carried T, vouched.
        assert seen and all(t == tenant for t, _s, _v in seen), seen
        assert all(s is TenantSource.JWT_CLAIM and v for _t, s, v in seen), seen
        # And the long-lived reconciler task does not carry T into the next row.
        assert get_tenant_id_from_context() is None

    async def test_no_snapshot_tenant_skips_the_compile_cancel(self, caplog) -> None:
        """A legacy row with no billable tenant gets NO invented one: the run's own
        compile cancel is skipped (an SA-scoped cancel cannot reach a customer-filed
        job anyway) and the skip is logged. The rest of the reap still happens."""
        from w2v_common.auth.identity_context import get_tenant_id_from_context

        with caplog.at_level("ERROR", logger="src.services.run_dispatcher"):
            run_id, _c, stage_repo, _rr, project_repo, compile_cancel, _proj = await self._reap(
                config_snapshot={"tenant_id": "", "tenant_source": "jwt-claim"}
            )
        cancelled = [c.args[0] for c in compile_cancel.call_args_list]
        assert run_id not in cancelled, "must not cancel undelegated as the SA tenant"
        assert any(
            "could not bind identity" in r.getMessage()
            and "skipping the compile cancel" in r.getMessage()
            for r in caplog.records
        ), [r.getMessage() for r in caplog.records]
        stage_repo.bulk_cancel_non_terminal.assert_awaited_once()
        assert project_repo.recompute_status.await_count == 2
        assert get_tenant_id_from_context() is None
