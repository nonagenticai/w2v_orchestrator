"""The tenant that reaches the WIRE on a background run's LLM calls.

``WMClient.message()`` takes no tenant argument — by design; the repo's own
style rule is "identity flows via ContextVars on S2S calls, never thread
``tenant_id`` through call chains manually". So the ONLY thing that decides how
``growth_bundle`` / ``promo_*`` / every other stage's spend is attributed is the
tenant ``RunDispatcher`` binds once per run task, which
``w2v_common.auth.identity_context.identity_event_hook`` then stamps as
``X-Effective-Tenant-Id`` on the metered worker_management hop.

These tests therefore assert the OUTBOUND HEADERS, not a call signature. A test
that asserted ``mock_wm.message.call_args`` would pass on a call that never left
the process with a tenant on it, which is precisely the defect.

``_probe_outbound_headers`` is deliberately the REAL hook against a REAL
``httpx.Request`` aimed at the REAL discovery URL for worker_management —
``is_metered_destination`` matches on the destination host, so a fake URL would
take the plain-backend branch and the delegation header would never appear at
all. The stand-in is only for the *transport*; every decision under test is
made by production code.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import httpx
import pytest
from w2v_common.auth.identity_context import identity_event_hook
from w2v_common.discovery import service_url

pytestmark = pytest.mark.asyncio

#: The all-zeros tenant. Registered in ``licenses.tenants`` as ``default-anon``
#: and a perfectly valid UUID, which is exactly why it passed every type check
#: while mis-attributing spend. It was the pre-0.67 ``auth.default_tenant_id``
#: fallback, so legacy ``config_snapshot`` rows still hold it.
NIL_TENANT = "00000000-0000-0000-0000-000000000000"

#: What ``WMClient._get_client`` bakes into every outbound request: the w2v
#: SERVICE-account license token, in ``X-Tenant-Id``. Its presence is what makes
#: an unbound tenant fail SILENTLY instead of raising — the hook treats a
#: service-credentialed request as legitimately acting for nobody.
_SA_HEADER = {"X-Tenant-Id": "Bearer w2v-service-account-token"}


class _FakeRedis:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    async def publish(self, channel: str, message: str) -> int:
        self.published.append((channel, json.loads(message)))
        return 1


async def _probe_outbound_headers() -> dict[str, str]:
    """Fire the production identity hook at worker_management and return the wire.

    Stands in for one ``WMClient.message()`` from inside a stage handler: same
    baked service-account header, same destination, same hook. Runs in whatever
    execution context the caller is in, which is the point — the dispatcher's
    ContextVar binding is task-local.
    """
    request = httpx.Request(
        "POST",
        f"{service_url('worker_management')}/workers/growth-bundle/message",
        headers=_SA_HEADER,
    )
    await identity_event_hook(request)
    return dict(request.headers)


def _run_obj(snapshot: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        blueprint_id="bp-1",
        deliverable_id=uuid4(),
        config_snapshot=snapshot,
        pause_after_stages=[],
        started_at=None,
    )


async def _dispatch_all(
    *snapshots: dict,
    child_project_id: UUID | None = None,
    parent_project_id: UUID | None = None,
) -> tuple[AsyncMock, _FakeRedis, dict[UUID, list[dict[str, str]]]]:
    """Drive N runs through ONE dispatcher concurrently.

    Returns ``(run_repo, redis, outbound-headers-per-run-id)``. Runs are keyed by
    ``ctx.run_id`` — read off the ``RunContext`` the dispatcher itself built — so
    a cross-attribution bug shows up as the WRONG tenant against a run id rather
    than as a missing entry.

    The stubbed executor is the only stand-in: instead of running a DAG it fires
    ``_probe_outbound_headers`` once, exactly where a stage handler's WM call
    would sit. A missing/empty entry therefore means "no LLM call was ever
    attempted", which is the assertion the refusal tests need.

    One ``patch`` block covers all of them on purpose: patching the same module
    globals from two overlapping ``with`` blocks restores the first while the
    second is still live, which is a harness artefact and not a finding.

    ``child_project_id`` / ``parent_project_id`` pin what the dispatcher's
    project lookup resolves to. They default to ``None`` (a top-level project),
    which keeps ``_recompute_project`` and ``_publish_child_status`` no-ops for
    the tests that only care about the wire. Passing real ids turns the parent
    mirror ON, so the ``child_status_changed`` half of a terminal transition
    becomes assertable; ``ProjectRepository`` is stubbed in that case because
    ``recompute_status`` issues real aggregation SQL that a mocked session
    cannot answer (same stub, same reason, as
    ``tests/unit/test_run_dispatcher_child_status.py``). The recompute's real
    effect on stored project status is proven against Postgres instead, in
    ``tests/integration/test_unbillable_tenant_project_status.py``.
    """
    from src.services.run_dispatcher import RunDispatcher

    runs = [_run_obj(snap) for snap in snapshots]
    by_id = {run.id: run for run in runs}
    outbound: dict[UUID, list[dict[str, str]]] = {run.id: [] for run in runs}

    session = AsyncMock()
    session.commit = AsyncMock()
    exec_result = MagicMock()
    exec_result.first = MagicMock(
        return_value=SimpleNamespace(id=child_project_id, parent_project_id=parent_project_id)
    )
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

    redis = _FakeRedis()

    executor = MagicMock()

    async def _run_dag(ctx, **_kwargs):
        # Yield first so concurrent runs genuinely interleave inside their
        # probes — a serial "A finishes, then B starts" would pass the
        # cross-attribution test without proving task-locality.
        await asyncio.sleep(0)
        outbound[ctx.run_id].append(await _probe_outbound_headers())

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
            redis=redis,  # ty: ignore[invalid-argument-type]
        )
        for run in runs:
            await dispatcher.dispatch(run.id)
        tasks = [t for t in (dispatcher._tasks.get(run.id) for run in runs) if t is not None]
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)

    return run_repo, redis, outbound


async def _dispatch(snapshot: dict) -> tuple[AsyncMock, _FakeRedis, list[dict[str, str]]]:
    """Single-run convenience wrapper over :func:`_dispatch_all`."""
    run_repo, redis, outbound = await _dispatch_all(snapshot)
    (headers,) = outbound.values()
    return run_repo, redis, headers


def _snapshot(tenant: str | None, *, source: str = "jwt-claim") -> dict:
    snap: dict = {"subject_id": str(uuid4()), "license_id": "lic-1", "tenant_source": source}
    if tenant is not None:
        snap["tenant_id"] = tenant
    return snap


class TestRealTenantReachesTheWire:
    async def test_run_tenant_is_delegated_on_the_worker_management_hop(self) -> None:
        """The run's own tenant — not the service account's — is what WM bills."""
        tenant = str(uuid4())
        _run_repo, _redis, outbound = await _dispatch(_snapshot(tenant))

        assert len(outbound) == 1, "the stubbed stage never reached its WM call"
        headers = outbound[0]
        assert headers["x-effective-tenant-id"] == tenant
        # The service-account credential still authenticates the hop; the
        # customer tenant rides in the side-channel. If these two ever collapse
        # into one header, WM introspects the SA tenant and bills it.
        assert headers["x-tenant-id"] == _SA_HEADER["X-Tenant-Id"]

    async def test_two_concurrent_runs_do_not_cross_attribute(self) -> None:
        """ContextVars are task-local: a shared WMClient must not leak a tenant.

        One run finishing before the other starts would pass this trivially, so
        both are dispatched together and each is asserted against ITS OWN id.
        """
        first, second = str(uuid4()), str(uuid4())
        _run_repo, _redis, outbound = await _dispatch_all(_snapshot(first), _snapshot(second))
        seen = sorted(headers[0]["x-effective-tenant-id"] for headers in outbound.values())
        assert seen == sorted([first, second]), (
            "both runs must bill their OWN tenant; one binding leaking into the "
            "other shows up here as a duplicate"
        )


class TestSentinelIsRefusedNotSubstituted:
    """The nil uuid must never reach the wire, and must never be swapped for
    something else either — a run with no billable tenant does not run at all."""

    async def test_nil_tenant_never_reaches_the_wire(self) -> None:
        _run_repo, _redis, outbound = await _dispatch(_snapshot(NIL_TENANT))

        # The load-bearing assertion is the ABSENCE. Asserting only
        # ``!= NIL_TENANT`` would pass on a substituted placeholder, which is
        # the same defect wearing a different value.
        assert outbound == [], f"an unbillable run reached its LLM call; headers: {outbound}"

    async def test_nil_tenant_fails_the_run_naming_the_reason(self) -> None:
        run_repo, redis, _outbound = await _dispatch(_snapshot(NIL_TENANT))

        _run_id, reason = run_repo.mark_failed.call_args.args
        assert json.loads(reason)["reason"] == "unbillable_tenant"
        # An operator has to be able to tell WHICH lane produced it.
        assert "nil uuid" in json.loads(reason)["error"]

        events = [msg["event"] for _channel, msg in redis.published]
        assert "run_failed" in events
        assert "run_completed" not in events

    async def test_absent_tenant_is_refused_rather_than_billed_to_the_service_account(
        self,
    ) -> None:
        """The quiet half. With no tenant bound, the hook takes its
        service-credentialed branch and the call goes out attributed to the w2v
        SA tenant — no exception, no sentinel, just a wrong invoice. So the
        refusal has to happen HERE, before any call is made."""
        run_repo, _redis, outbound = await _dispatch(_snapshot(None))

        assert outbound == []
        _run_id, reason = run_repo.mark_failed.call_args.args
        assert json.loads(reason)["reason"] == "unbillable_tenant"

    async def test_unparseable_tenant_is_refused(self) -> None:
        run_repo, _redis, outbound = await _dispatch(_snapshot("tenant-alpha"))

        assert outbound == []
        _run_id, reason = run_repo.mark_failed.call_args.args
        assert json.loads(reason)["reason"] == "unbillable_tenant"


class TestTheRefusalIsAWholeTerminalTransition:
    """A refused run is TERMINAL, so it owes the same bookkeeping as every other
    terminal transition in ``run_dispatcher``: recompute the owning project (and
    its course parent) and mirror the outcome onto the parent's SSE channel.

    ``recompute_status`` has two callers estate-wide, and a refused run is never
    transitioned again — so a refusal that skips them pins the project at
    ``producing`` forever, on exactly the legacy nil-uuid rows this guard exists
    for. The STORED-STATUS half is proven against real Postgres in
    ``tests/integration/test_unbillable_tenant_project_status.py``; this pins the
    SSE mirror, which that test cannot see.
    """

    async def test_refusal_mirrors_failed_onto_the_parent_project_channel(self) -> None:
        parent, child = uuid4(), uuid4()
        _run_repo, redis, outbound = await _dispatch_all(
            _snapshot(NIL_TENANT), child_project_id=child, parent_project_id=parent
        )

        assert outbound[next(iter(outbound))] == []
        mirrored = [
            msg
            for channel, msg in redis.published
            if channel == f"projects:{parent}" and msg.get("event") == "child_status_changed"
        ]
        assert mirrored == [
            {
                "event": "child_status_changed",
                "project_id": str(child),
                "status": "failed",
            }
        ], (
            "a refused run must mirror 'failed' onto the parent project's channel like "
            f"every other terminal transition does; published: {redis.published}"
        )


async def _finalize_relief(snapshot: dict) -> tuple[AsyncMock, MagicMock, AsyncMock, str | None]:
    """Drive ``finalize_if_dag_complete`` over a DAG-complete run with a
    ``run_engine`` bound, so the reconciler's structural-relief spawn is reached.

    Returns ``(spawn_mock, identity_mock, run_repo, result)``. ``snapshot`` is the
    parent run's ``config_snapshot`` — the ONLY input that decides whether the
    relief spawn is refused.
    """
    from src.services.run_dispatcher import RunDispatcher

    run_obj = SimpleNamespace(
        id=uuid4(),
        blueprint_id="bp-1",
        deliverable_id=uuid4(),
        status="running",
        pause_after_stages=[],
        config_snapshot=snapshot,
    )

    session = AsyncMock()
    session.commit = AsyncMock()
    exec_result = MagicMock()
    exec_result.first = MagicMock(return_value=SimpleNamespace(id=None, parent_project_id=None))
    session.execute = AsyncMock(return_value=exec_result)

    run_repo = AsyncMock()
    run_repo.get = AsyncMock(return_value=run_obj)
    run_repo.mark_completed = AsyncMock(return_value=True)
    run_repo.mark_failed = AsyncMock()

    stage_repo = AsyncMock()
    stage_repo.list_for_run = AsyncMock(
        return_value=[
            SimpleNamespace(stage_name="s1", status="completed", output={}),
            SimpleNamespace(stage_name="s2", status="skipped", output={}),
        ]
    )

    blueprint = MagicMock()
    blueprint.topological_order.return_value = ["s1", "s2"]
    blueprint.terminal_artifact_stage = None
    registry = MagicMock()
    registry.resolve_by_id.return_value = blueprint

    spawn_mock = AsyncMock()
    identity_mock = MagicMock()

    @asynccontextmanager
    async def _factory():
        yield session

    with (
        patch("src.services.run_dispatcher.RunRepository", return_value=run_repo),
        patch("src.services.run_dispatcher.StageOutputRepository", return_value=stage_repo),
        patch("src.services.run_dispatcher.maybe_spawn_structural_relief_run", spawn_mock),
        patch("src.services.run_dispatcher.set_identity_context", identity_mock),
    ):
        dispatcher = RunDispatcher(
            session_factory=_factory,
            registry=registry,
            redis=_FakeRedis(),  # ty: ignore[invalid-argument-type]
        )
        dispatcher.bind_run_engine(MagicMock())
        result = await dispatcher.finalize_if_dag_complete(run_obj.id)

    return spawn_mock, identity_mock, run_repo, result


class TestTheReconcilerReliefSpawnTakesTheSameRefusal:
    """The dispatch path is not the only ``set_identity_context`` site.

    ``finalize_if_dag_complete`` — the reconciler backstop for a DAG-complete run
    stranded in ``running`` — re-binds the parent's identity from its snapshot
    and spawns the bounded structural-relief re-render. That spawn inherits the
    SAME snapshot, so an unbillable parent would produce a child whose every LLM
    call is delegated as the nil uuid (or, absent, billed to the w2v service
    account). The guard there was written correct but had NO test: mutating
    ``if _relief_tenant is not None:`` to ``if True:`` survived every other test
    in this file, because none of them reach this method with a bad snapshot.
    """

    async def test_relief_spawn_is_skipped_when_the_parent_snapshot_is_unbillable(
        self,
    ) -> None:
        spawn_mock, identity_mock, run_repo, result = await _finalize_relief(
            _snapshot(NIL_TENANT, source="license-token")
        )

        # The parent still finalizes — the refusal costs only the DISCRETIONARY
        # re-render, never the completed run's own terminal transition.
        assert result == "completed"
        run_repo.mark_completed.assert_awaited_once()
        run_repo.mark_failed.assert_not_awaited()

        spawn_mock.assert_not_awaited()
        # And nothing was bound either: reaching set_identity_context with an
        # unbillable tenant is the step that makes the spawn spend it.
        identity_mock.assert_not_called()

    async def test_relief_spawn_still_fires_for_a_billable_parent(self) -> None:
        """Non-vacuity twin: the skip above must be caused by the REFUSAL, not by
        a harness that never reaches the spawn at all."""
        tenant = str(uuid4())
        spawn_mock, identity_mock, _run_repo, result = await _finalize_relief(
            _snapshot(tenant, source="license-token")
        )

        assert result == "completed"
        spawn_mock.assert_awaited_once()
        identity_mock.assert_called_once()
        assert identity_mock.call_args.kwargs["tenant_id"] == UUID(tenant)


class TestTheGuardIsLoadBearing:
    """Non-vacuity anchor for the class above.

    **The day this class anticipated has arrived.** It used to bind the nil
    tenant and assert ``identity_event_hook`` delegated it anyway — proving the
    library did not refuse, so the dispatcher's guard was the only thing
    standing between a nil tenant and an unattributable invoice. w2v-common
    0.68.0 made ``parse_tenant_id`` refuse the nil uuid, so
    ``set_identity_context`` now raises and that binding is impossible: the old
    assertion is not merely red, its PREMISE is gone.

    It is retired deliberately rather than reshaped into one more nil-rejection
    assertion, and the two facts it existed to establish are pinned separately
    below, because the guard is still load-bearing — for a different reason:

    * ``billable_tenant_id`` refuses at DISPATCH, before
      ``set_identity_context`` is ever called (``identity_mock.assert_not_called()``
      in the class above). Removing the guard would not silently delegate a nil
      tenant any more — it would RAISE inside the spawn. The guard is what turns
      that into a graceful skip where the parent run still completes and only
      the discretionary re-render is dropped. That behaviour is the guard's, not
      the library's, and the refusal tests above still exercise it.
    * The hook itself still DELEGATES rather than judges: it stamps whatever
      tenant it is handed, verbatim. That is why binding an unbillable tenant
      would be a billing event, and it stays true for every value the library
      now lets through.
    """

    async def test_the_hook_delegates_whatever_tenant_it_is_given(self) -> None:
        """The hook judges nothing — it stamps the bound tenant verbatim.

        Exercised with a REAL tenant, because the nil uuid can no longer be
        bound. This is the half of the original premise that survives: the hook
        is a delegator, so whatever reaches it becomes the billing record.
        """
        from w2v_common.auth.identity_context import (
            clear_identity_context,
            set_identity_context,
        )
        from w2v_common.tenant import TenantSource

        tenant = str(uuid4())
        try:
            set_identity_context(
                tenant_id=tenant,
                license_id="lic-1",
                subject_id=str(uuid4()),
                tenant_source=TenantSource.JWT_CLAIM,
            )
            headers = await _probe_outbound_headers()
        finally:
            clear_identity_context()

        assert headers["x-effective-tenant-id"] == tenant

    @pytest.mark.parametrize(
        "spelling",
        [
            NIL_TENANT,
            "urn:uuid:" + "0" * 32,
            "0" * 32,
            "{00000000-0000-0000-0000-000000000000}",
            "urn:uuid:00000000-0000-0000-0000-000000000000",
            "  " + NIL_TENANT + "  ",
            UUID(int=0),
        ],
        ids=["hyphenated", "uppercase", "unhyphenated", "braced", "urn", "padded", "uuid-object"],
    )
    async def test_the_nil_tenant_can_no_longer_be_bound_at_all(self, spelling: UUID | str) -> None:
        """The other half: the library now refuses, so the hook never sees nil.

        This replaces the old delegation assertion as the anchor. It fails the
        day the library stops refusing — at which point the nil tenant could
        reach the hook again and be delegated verbatim, and the delegation pin
        above would need to cover it once more.
        """
        from w2v_common.auth.identity_context import set_identity_context
        from w2v_common.tenant import InvalidTenantIdError, TenantSource

        with pytest.raises(InvalidTenantIdError, match=r"(?i)nil uuid"):
            set_identity_context(
                tenant_id=spelling,
                license_id="lic-1",
                subject_id=str(uuid4()),
                tenant_source=TenantSource.JWT_CLAIM,
            )


class TestBillableTenantId:
    """Unit-level contract of the parser the dispatcher gates on."""

    def test_returns_the_parsed_uuid_for_a_real_tenant(self) -> None:
        from src.auth.run_tenant import billable_tenant_id

        tenant = uuid4()
        assert billable_tenant_id(str(tenant)) == tenant

    @pytest.mark.parametrize(
        "spelling",
        [
            NIL_TENANT,
            "0" * 32,
            "{00000000-0000-0000-0000-000000000000}",
            "urn:uuid:00000000-0000-0000-0000-000000000000",
            UUID(int=0),
        ],
        ids=["hyphenated", "unhyphenated", "braced", "urn", "uuid-object"],
    )
    def test_every_spelling_of_the_nil_uuid_is_refused(self, spelling: object) -> None:
        """``uuid.UUID`` normalises all of these to the same value, so a guard
        written against the hyphenated STRING would miss four of the five."""
        from w2v_common.tenant import InvalidTenantIdError

        from src.auth.run_tenant import billable_tenant_id

        with pytest.raises(InvalidTenantIdError, match="nil uuid"):
            billable_tenant_id(spelling)

    @pytest.mark.parametrize("absent", [None, "", "   "])
    def test_absent_tenant_raises_rather_than_defaulting(self, absent: object) -> None:
        from w2v_common.tenant import MissingTenantError

        from src.auth.run_tenant import billable_tenant_id

        with pytest.raises(MissingTenantError):
            billable_tenant_id(absent)

    def test_refusal_message_does_not_echo_the_value(self) -> None:
        """An ``X-Tenant-Id`` field can carry a ``Bearer <license token>``; the
        refusal must name the lane, never the payload."""
        from w2v_common.tenant import InvalidTenantIdError

        from src.auth.run_tenant import billable_tenant_id

        with pytest.raises(InvalidTenantIdError) as excinfo:
            billable_tenant_id("Bearer sk-live-not-a-real-token")
        assert "sk-live" not in str(excinfo.value)
