"""Unit coverage for ``RunDispatcher.finalize_if_dag_complete``.

This is the reconciler backstop for the e2e-post-promote promo hang: a run
whose DAG is fully terminal ('skipped' counts) but whose row is still
``running`` because the owning dispatch task was interrupted after the last
stage persisted terminal but before the completion gate committed
``mark_completed``. The method reuses the dispatcher's own completion
primitives (``decide_run_terminal_state`` → ``verify_terminal_artifact`` →
``mark_completed``/``mark_failed``) and runs regardless of any still-present
lease, since a DAG-complete run has no pending work.

Pinned here:
* fully-terminal DAG + ``running`` row → ``mark_completed`` + ``run_completed``, returns ``'completed'``.
* not-yet-terminal DAG → no-op, returns ``None`` (left to the normal reconciler flow).
* row already terminal (inline gate won the race) → no-op, returns ``None``.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid7

import pytest
from w2v_common.tenant import TenantSource

pytestmark = pytest.mark.asyncio


class _FakeRedis:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    async def publish(self, channel: str, message: str) -> int:
        self.published.append((channel, json.loads(message)))
        return 1


def _session_factory(session: AsyncMock):
    @asynccontextmanager
    async def _factory():
        yield session

    return _factory


async def _finalize(
    *,
    run_status: str,
    stage_rows: list,
    topo_order: list[str],
    mark_completed_return: object = None,
) -> tuple[str | None, AsyncMock, _FakeRedis]:
    """Drive ``finalize_if_dag_complete`` with stubbed deps; return
    (result, run_repo_mock, redis).

    ``mark_completed_return`` sets the transition-bool the stubbed
    ``mark_completed`` reports. ``None`` (default) leaves it a plain
    ``AsyncMock()`` (a truthy MagicMock), preserving the historical
    "publish fires" behaviour for the existing tests. Pass ``False`` to
    simulate a lost finalize↔inline-gate race (row already terminal)."""
    from src.services.run_dispatcher import RunDispatcher

    run_obj = SimpleNamespace(
        id=uuid7(),
        blueprint_id="bp-1",
        deliverable_id=uuid7(),
        status=run_status,
        pause_after_stages=[],
        # A real ``Run`` row always carries one (NOT NULL). The backstop now reads
        # it unconditionally — it binds the run's identity before the terminal-
        # artifact probe, not just before the relief spawn — so the stub has to
        # carry the shape a row does.
        config_snapshot={
            "tenant_id": str(uuid7()),
            "license_id": str(uuid7()),
            "subject_id": str(uuid7()),
            "tenant_source": TenantSource.LICENSE_TOKEN.value,
        },
    )

    session = AsyncMock()
    session.commit = AsyncMock()
    # The project lookup (`select(Project.id, ...).first()`) — NULL ids so the
    # child-status mirror + recompute are no-ops.
    proj_row = SimpleNamespace(id=None, parent_project_id=None)
    exec_result = MagicMock()
    exec_result.first = MagicMock(return_value=proj_row)
    session.execute = AsyncMock(return_value=exec_result)

    run_repo_instance = AsyncMock()
    run_repo_instance.get = AsyncMock(return_value=run_obj)
    if mark_completed_return is None:
        run_repo_instance.mark_completed = AsyncMock()
    else:
        run_repo_instance.mark_completed = AsyncMock(return_value=mark_completed_return)
    run_repo_instance.mark_failed = AsyncMock()

    stage_repo_instance = AsyncMock()
    stage_repo_instance.list_for_run = AsyncMock(return_value=stage_rows)

    blueprint = MagicMock()
    blueprint.topological_order.return_value = topo_order
    # No terminal-artifact stage → verify_terminal_artifact passes (returns None).
    blueprint.terminal_artifact_stage = None
    registry = MagicMock()
    registry.resolve_by_id.return_value = blueprint

    redis = _FakeRedis()

    with (
        patch("src.services.run_dispatcher.RunRepository", return_value=run_repo_instance),
        patch(
            "src.services.run_dispatcher.StageOutputRepository",
            return_value=stage_repo_instance,
        ),
    ):
        dispatcher = RunDispatcher(
            session_factory=_session_factory(session),
            registry=registry,
            redis=redis,  # ty: ignore[invalid-argument-type]
        )
        result = await dispatcher.finalize_if_dag_complete(run_obj.id)

    return result, run_repo_instance, redis


async def test_finalizes_stranded_dag_complete_run() -> None:
    """Every stage terminal + row still ``running`` → mark_completed +
    run_completed, returns ``'completed'``. 'skipped' counts as terminal."""
    stage_rows = [
        SimpleNamespace(stage_name="s1", status="completed", output={}),
        SimpleNamespace(stage_name="s2", status="skipped", output={}),
    ]
    result, run_repo, redis = await _finalize(
        run_status="running", stage_rows=stage_rows, topo_order=["s1", "s2"]
    )

    assert result == "completed"
    run_repo.mark_completed.assert_awaited_once()
    run_repo.mark_failed.assert_not_awaited()
    events = [msg["event"] for _channel, msg in redis.published]
    assert "run_completed" in events


async def test_noop_when_dag_not_terminal() -> None:
    """A run with a still-pending stage is not DAG-complete → no finalize,
    returns ``None`` (the normal reconciler flow handles it)."""
    stage_rows = [
        SimpleNamespace(stage_name="s1", status="completed", output={}),
        SimpleNamespace(stage_name="s2", status="running", output={}),
    ]
    result, run_repo, redis = await _finalize(
        run_status="running", stage_rows=stage_rows, topo_order=["s1", "s2"]
    )

    assert result is None
    run_repo.mark_completed.assert_not_awaited()
    run_repo.mark_failed.assert_not_awaited()
    assert redis.published == []


async def test_no_publish_when_mark_completed_lost_the_race() -> None:
    """Finalize↔inline-gate race: the DAG is terminal and the row still reads
    ``running`` when finalize probes it, but ``mark_completed`` transitions 0
    rows (the inline gate committed the terminal write in between) → it returns
    ``False``. The dispatcher must NOT publish a second ``run_completed`` (nor
    re-increment ``runs_total``), even though it still returns ``'completed'``."""
    stage_rows = [
        SimpleNamespace(stage_name="s1", status="completed", output={}),
        SimpleNamespace(stage_name="s2", status="skipped", output={}),
    ]
    result, run_repo, redis = await _finalize(
        run_status="running",
        stage_rows=stage_rows,
        topo_order=["s1", "s2"],
        mark_completed_return=False,
    )

    assert result == "completed"
    run_repo.mark_completed.assert_awaited_once()
    events = [msg["event"] for _channel, msg in redis.published]
    assert "run_completed" not in events


async def test_noop_when_a_blueprint_stage_has_no_row() -> None:
    """A run whose blueprint gained ``s2`` after creation has NO row for it. The
    backstop must still return ``None``: a row-less stage has demonstrably not
    run, so counting it as done would complete the run without executing it.
    Reconciliation is the dispatcher's job (it seeds the row ``pending``, which
    is not in TERMINAL_STAGE_STATUSES, so this verdict is unchanged either way).

    This pins the terminal check AGAINST a future 'fix' that loosens the
    set-equality to make row-less stages disappear — that would flip this to
    'completed' and silently ship an unrun stage."""
    stage_rows = [SimpleNamespace(stage_name="s1", status="completed", output={})]
    result, run_repo, redis = await _finalize(
        run_status="running", stage_rows=stage_rows, topo_order=["s1", "s2"]
    )

    assert result is None
    run_repo.mark_completed.assert_not_awaited()
    run_repo.mark_failed.assert_not_awaited()
    assert redis.published == []


async def test_noop_when_row_already_terminal() -> None:
    """If the inline completion gate already won the race (row is
    ``completed``), the backstop must not touch it — returns ``None``."""
    stage_rows = [SimpleNamespace(stage_name="s1", status="completed", output={})]
    result, run_repo, _redis = await _finalize(
        run_status="completed", stage_rows=stage_rows, topo_order=["s1"]
    )

    assert result is None
    run_repo.mark_completed.assert_not_awaited()


async def test_finalize_binds_identity_before_relief_spawn() -> None:
    """FIX B: the reconciler backstop must bind the parent's three-piece identity from
    its ``config_snapshot`` BEFORE ``maybe_spawn_structural_relief_run``. The backstop
    runs in the reconciler task with NO inbound request, so the identity ContextVars are
    empty; without this bind the spawned child's ``start_run`` builds a snapshot with an
    EMPTY ``license_id`` and every paid S2S call (bdd/record/compile/KAG) goes out with no
    X-License-Id → license-authz 402/403. Mirrors the inline gate's bind at dispatch start.
    """
    from src.services.run_dispatcher import RunDispatcher

    license_id = str(uuid7())
    tenant_id = str(uuid7())
    subject_id = str(uuid7())
    config_snapshot = {
        "license_id": license_id,
        "tenant_id": tenant_id,
        "subject_id": subject_id,
        # FOURTH piece since w2v-common 0.67.0. Without it every metered hop the
        # spawned child makes raises UnvouchedTenantError — see
        # tests/unit/test_tenant_provenance.py, which proves that consequence.
        "tenant_source": TenantSource.LICENSE_TOKEN.value,
    }
    run_obj = SimpleNamespace(
        id=uuid7(),
        blueprint_id="bp-1",
        deliverable_id=uuid7(),
        status="running",
        pause_after_stages=[],
        config_snapshot=config_snapshot,
    )

    session = AsyncMock()
    session.commit = AsyncMock()
    proj_row = SimpleNamespace(id=None, parent_project_id=None)
    exec_result = MagicMock()
    exec_result.first = MagicMock(return_value=proj_row)
    session.execute = AsyncMock(return_value=exec_result)

    run_repo_instance = AsyncMock()
    run_repo_instance.get = AsyncMock(return_value=run_obj)
    run_repo_instance.mark_completed = AsyncMock(return_value=True)
    run_repo_instance.mark_failed = AsyncMock()

    stage_rows = [
        SimpleNamespace(stage_name="s1", status="completed", output={}),
        SimpleNamespace(stage_name="s2", status="skipped", output={}),
    ]
    stage_repo_instance = AsyncMock()
    stage_repo_instance.list_for_run = AsyncMock(return_value=stage_rows)

    blueprint = MagicMock()
    blueprint.topological_order.return_value = ["s1", "s2"]
    blueprint.terminal_artifact_stage = None
    registry = MagicMock()
    registry.resolve_by_id.return_value = blueprint

    redis = _FakeRedis()
    spawn_mock = AsyncMock()
    identity_mock = MagicMock()

    with (
        patch("src.services.run_dispatcher.RunRepository", return_value=run_repo_instance),
        patch(
            "src.services.run_dispatcher.StageOutputRepository",
            return_value=stage_repo_instance,
        ),
        patch("src.services.run_dispatcher.maybe_spawn_structural_relief_run", spawn_mock),
        patch("src.services.run_dispatcher.set_identity_context", identity_mock),
    ):
        dispatcher = RunDispatcher(
            session_factory=_session_factory(session),
            registry=registry,
            redis=redis,  # ty: ignore[invalid-argument-type]
        )
        dispatcher.bind_run_engine(MagicMock())
        result = await dispatcher.finalize_if_dag_complete(run_obj.id)

    assert result == "completed"
    # Identity bound with the parent's NON-EMPTY license_id (the whole point of FIX B).
    # ``tenant_id`` is now the PARSED uuid, not the raw snapshot string: it goes
    # through ``billable_tenant_id`` so an unbillable parent (nil uuid / absent /
    # non-uuid) skips the spawn instead of delegating a sentinel — see
    # tests/unit/test_run_dispatcher_tenant.py. ``UUID(str) == str`` is False, so
    # this assertion has to name the uuid.
    identity_mock.assert_called_once_with(
        tenant_id=UUID(tenant_id),
        license_id=license_id,
        subject_id=subject_id,
        tenant_source=TenantSource.LICENSE_TOKEN,
    )
    # ...and the relief spawn actually fired on this path, carrying the same snapshot.
    spawn_mock.assert_awaited_once()
    call = spawn_mock.await_args
    assert call is not None
    assert call.kwargs["parent_run_id"] == run_obj.id
    assert call.kwargs["config_snapshot"] == config_snapshot


async def _finalize_capturing_probe_identity(
    config_snapshot: dict,
) -> tuple[str | None, list[tuple[UUID | None, TenantSource | None, str]]]:
    """Drive ``finalize_if_dag_complete`` with a REAL identity context and a
    stubbed ``verify_terminal_artifact`` that records what the probe would see.

    Returns (result, [(tenant, tenant_source, license_id) observed at probe time]).
    Nothing here is patched at ``set_identity_context`` — the point is the actual
    ContextVar state the outbound hook reads, not the call arguments.
    """
    from w2v_common.auth.identity_context import (
        clear_identity_context,
        get_license_id_from_context,
        get_tenant_id_from_context,
        get_tenant_source_from_context,
    )

    from src.services.run_dispatcher import RunDispatcher

    run_obj = SimpleNamespace(
        id=uuid7(),
        blueprint_id="bp-1",
        deliverable_id=uuid7(),
        status="running",
        pause_after_stages=[],
        config_snapshot=config_snapshot,
    )

    session = AsyncMock()
    session.commit = AsyncMock()
    proj_row = SimpleNamespace(id=None, parent_project_id=None)
    exec_result = MagicMock()
    exec_result.first = MagicMock(return_value=proj_row)
    session.execute = AsyncMock(return_value=exec_result)

    run_repo_instance = AsyncMock()
    run_repo_instance.get = AsyncMock(return_value=run_obj)
    run_repo_instance.mark_completed = AsyncMock(return_value=True)
    run_repo_instance.mark_failed = AsyncMock()

    stage_rows = [SimpleNamespace(stage_name="s1", status="completed", output={})]
    stage_repo_instance = AsyncMock()
    stage_repo_instance.list_for_run = AsyncMock(return_value=stage_rows)

    blueprint = MagicMock()
    blueprint.topological_order.return_value = ["s1"]
    registry = MagicMock()
    registry.resolve_by_id.return_value = blueprint

    seen: list[tuple[UUID | None, TenantSource | None, str]] = []

    async def _probe(*_args, **_kwargs):
        seen.append(
            (
                get_tenant_id_from_context(),
                get_tenant_source_from_context(),
                get_license_id_from_context(),
            )
        )
        # ``None`` is verify_terminal_artifact's "the deliverable is there" answer,
        # not a bare fall-off-the-end — say it explicitly.
        return None  # noqa: RET501

    clear_identity_context()
    try:
        with (
            patch("src.services.run_dispatcher.RunRepository", return_value=run_repo_instance),
            patch(
                "src.services.run_dispatcher.StageOutputRepository",
                return_value=stage_repo_instance,
            ),
            patch(
                "src.services.run_dispatcher.verify_terminal_artifact",
                AsyncMock(side_effect=_probe),
            ),
        ):
            dispatcher = RunDispatcher(
                session_factory=_session_factory(session),
                registry=registry,
                redis=_FakeRedis(),  # ty: ignore[invalid-argument-type]
            )
            result = await dispatcher.finalize_if_dag_complete(run_obj.id)
    finally:
        clear_identity_context()
    return result, seen


async def test_finalize_binds_identity_before_terminal_artifact_probe() -> None:
    """The backstop's terminal-artifact probe must be DELEGATED like the inline gate's.

    ``verify_terminal_artifact``'s ``design_job`` probe is an outbound call to
    html_render, which w2v-common 0.73.0 added to the metered/delegating
    destination set. ``DesignClient`` bakes ``X-Tenant-Id: Bearer <SA token>``, so
    ``_apply_outbound_tenant`` takes the ``service_credentialed`` branch — and with
    NO tenant on the ContextVar it returns early and sends no
    ``X-Effective-Tenant-Id``. html_render then filters by the SA tenant, cannot see
    a redesign_job created under the RUN's tenant, and 404s;
    ``design_job_exists`` maps 404 → ``False``, and this method turns ``False`` into
    ``mark_failed``/OUR_DEFECT. A run whose design EXISTS is marked failed.

    ``bind_run_tenant`` at the top of the method is only the DB GUC — it never
    touches the identity ContextVars. So the run's identity must be bound from the
    snapshot BEFORE the probe, exactly as the dispatch path binds it at run-task
    start.
    """
    tenant_id = str(uuid7())
    license_id = str(uuid7())
    subject_id = str(uuid7())
    result, seen = await _finalize_capturing_probe_identity(
        {
            "tenant_id": tenant_id,
            "license_id": license_id,
            "subject_id": subject_id,
            "tenant_source": TenantSource.LICENSE_TOKEN.value,
        }
    )

    assert result == "completed"
    assert seen, "verify_terminal_artifact was never called"
    probe_tenant, probe_source, probe_license = seen[0]
    # The RUN's tenant, not None — this is what becomes X-Effective-Tenant-Id.
    assert probe_tenant == UUID(tenant_id)
    # ...and the FOURTH piece: without a vouched provenance the delegated hop
    # refuses rather than delegating, so binding the tenant alone is not enough.
    assert probe_source is TenantSource.LICENSE_TOKEN
    assert probe_license == license_id


async def test_finalize_probe_identity_degrades_when_provenance_absent() -> None:
    """A legacy run with no recorded ``tenant_source`` must NOT make the backstop raise.

    ``tenant_source_from_snapshot`` returns ``None`` for a pre-0.67 row, which is the
    UNVOUCHED end of the ladder: the delegated hop raises ``UnvouchedTenantError``
    *inside* ``design_job_exists``, whose bare ``except`` maps it to ``None`` —
    INDETERMINATE, never a verdict. What must not happen is the exception escaping
    into ``finalize_if_dag_complete`` and turning a currently-successful finalize
    into a crash, so the bind is the only thing that happens here and the run still
    completes.
    """
    tenant_id = str(uuid7())
    result, seen = await _finalize_capturing_probe_identity(
        {"tenant_id": tenant_id, "license_id": "", "subject_id": ""}
    )

    assert result == "completed"
    probe_tenant, probe_source, _ = seen[0]
    assert probe_tenant == UUID(tenant_id)
    assert probe_source is None


async def test_finalize_probe_identity_unbound_when_tenant_unbillable() -> None:
    """An unbillable parent tenant binds NOTHING and still finalizes.

    ``billable_tenant_id`` raises on the nil uuid / absent / unparseable. The
    dispatch path fails such a run outright (``_fail_unbillable``), so this shape
    is only reachable for a legacy row — and the honest answer is an unattributed,
    therefore INDETERMINATE, probe: never an exception out of the backstop, and
    never a stale tenant left bound from the previous run this long-lived
    reconciler task finalized.
    """
    result, seen = await _finalize_capturing_probe_identity(
        {"tenant_id": str(UUID(int=0)), "tenant_source": TenantSource.LICENSE_TOKEN.value}
    )

    assert result == "completed"
    probe_tenant, probe_source, _ = seen[0]
    assert probe_tenant is None
    assert probe_source is None


async def _finalize_then_read_identity(
    *, artifact_problem: str | None
) -> tuple[str | None, UUID | None]:
    """Drive ``finalize_if_dag_complete`` to completion with a REAL identity
    context (nothing patched at ``set_identity_context``/``clear_identity_context``)
    and return (result, tenant_id_on_the_ContextVar_AFTER_the_call_returns).

    ``artifact_problem`` selects which terminal branch fires: ``None`` →
    ``mark_completed`` → ``'completed'``; any string → ``mark_failed`` →
    ``'failed'``. This is this long-lived reconciler task's own next call's
    view of the world — if the identity bound for THIS run is still on the
    ContextVar here, the NEXT run finalized by this task would silently
    inherit it.
    """
    from w2v_common.auth.identity_context import (
        clear_identity_context,
        get_tenant_id_from_context,
    )

    from src.services.run_dispatcher import RunDispatcher

    tenant_id = str(uuid7())
    config_snapshot = {
        "tenant_id": tenant_id,
        "license_id": str(uuid7()),
        "subject_id": str(uuid7()),
        "tenant_source": TenantSource.LICENSE_TOKEN.value,
    }
    run_obj = SimpleNamespace(
        id=uuid7(),
        blueprint_id="bp-1",
        deliverable_id=uuid7(),
        status="running",
        pause_after_stages=[],
        config_snapshot=config_snapshot,
    )

    session = AsyncMock()
    session.commit = AsyncMock()
    proj_row = SimpleNamespace(id=None, parent_project_id=None)
    exec_result = MagicMock()
    exec_result.first = MagicMock(return_value=proj_row)
    session.execute = AsyncMock(return_value=exec_result)

    run_repo_instance = AsyncMock()
    run_repo_instance.get = AsyncMock(return_value=run_obj)
    run_repo_instance.mark_completed = AsyncMock(return_value=True)
    run_repo_instance.mark_failed = AsyncMock(return_value=True)

    stage_rows = [SimpleNamespace(stage_name="s1", status="completed", output={})]
    stage_repo_instance = AsyncMock()
    stage_repo_instance.list_for_run = AsyncMock(return_value=stage_rows)

    blueprint = MagicMock()
    blueprint.topological_order.return_value = ["s1"]
    registry = MagicMock()
    registry.resolve_by_id.return_value = blueprint

    clear_identity_context()
    try:
        with (
            patch("src.services.run_dispatcher.RunRepository", return_value=run_repo_instance),
            patch(
                "src.services.run_dispatcher.StageOutputRepository",
                return_value=stage_repo_instance,
            ),
            patch(
                "src.services.run_dispatcher.verify_terminal_artifact",
                AsyncMock(return_value=artifact_problem),
            ),
        ):
            dispatcher = RunDispatcher(
                session_factory=_session_factory(session),
                registry=registry,
                redis=_FakeRedis(),  # ty: ignore[invalid-argument-type]
            )
            result = await dispatcher.finalize_if_dag_complete(run_obj.id)
        # Read the ContextVar OUTSIDE the patched block, right after the call
        # returns, to mirror what the NEXT reconciler-task call would see.
        tenant_after = get_tenant_id_from_context()
    finally:
        clear_identity_context()
    return result, tenant_after


async def test_finalize_clears_identity_after_completed_return() -> None:
    """BLOCKING regression: ``mark_completed``'s ``return "completed"`` must not
    leave this run's tenant bound on the ContextVar for the next run this
    long-lived reconciler task finalizes to inherit."""
    result, tenant_after = await _finalize_then_read_identity(artifact_problem=None)

    assert result == "completed"
    assert tenant_after is None


async def test_finalize_clears_identity_after_failed_return() -> None:
    """BLOCKING regression: ``mark_failed``'s ``return "failed"`` must not leave
    this run's tenant bound on the ContextVar for the next run this long-lived
    reconciler task finalizes to inherit."""
    result, tenant_after = await _finalize_then_read_identity(
        artifact_problem="deliverable missing"
    )

    assert result == "failed"
    assert tenant_after is None
