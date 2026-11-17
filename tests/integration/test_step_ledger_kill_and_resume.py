"""Kill a pod mid-stage, let the reconciler resume the run, count what was paid twice.

The end-to-end proof for the step ledger, on the production posture:

* the REAL ``RunDispatcher`` executes the run (so ``ctx.step_ledger`` is the one it
  wires, over the real ``stage_step_results`` table);
* every dispatcher/ledger session runs as a throwaway **non-owner NOBYPASSRLS**
  role with ``TENANT_RLS_MODE=enforce`` — the ``w2v_orch_app`` posture. The tenant
  GUC is bound only by the app's own ``bind_session_tenant`` / ``bind_run_tenant``;
  nothing here sets it, and nothing touches ``row_security``. Every other test in
  the suite connects as the table OWNER, which bypasses RLS entirely;
* the death is a ``BaseException`` raised out of the paid call. No handler code
  runs for it (the executor catches ``Exception`` and ``CancelledError`` only), so
  the stage row is left ``running`` and no counter moves — case (a) of the ledger's
  "Key scope" table, which is what a SIGKILL / OOM / node loss leaves behind;
* the resume is the REAL orphan reconciler on a second ``RunDispatcher`` ("the new
  pod"), which re-enters the stage from the top under the SAME ``run_id``.

The stage is the generic ``script_draft`` exactly as the blueprint builds it: its
first draw comes back under the element floor and the primitive re-asks with the
corrective ``RETRY`` block — two paid model draws, two ledger rows.

One thing the simulation gets for free that a real death does not: the dying
``_run`` still executes its ``finally`` and releases the run lease, so the
reconciler can resume immediately instead of waiting out the lease TTL.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from w2v_common.auth.identity_context import clear_identity_context

from src.config import clear_settings_cache
from src.database.models import StageStepResult
from src.database.repositories.run_repository import RunRepository
from src.database.repositories.stage_output_repository import StageOutputRepository
from src.database.rls_events import install_rls_tenant_rebind
from src.services.stages.shared.step_ledger import SessionStepStore, StepLedger
from tests.conftest import TEST_DATABASE_URL, require_createrole
from tests.integration._dispatch_harness import _create_project
from tests.support.reentry import Outbound, SimulatedPodDeath

pytestmark = pytest.mark.asyncio

_WM = "wm.message"
_STAGE = "script_draft"
_BRIEF = {"angle": "growth", "audience": "saas", "target_duration_seconds": 30}
#: ON the 30s order at the harness's stubbed 120 wpm (60 words, 20 per line), so
#: script_draft's length re-ask does not fire and ``_DRAWS`` stays the retry pair.
_GOOD_LINES = [
    {"text": "Open on the problem " + " ".join(["today"] * 16), "seconds": 10.0},
    {"text": "Show the fix " + " ".join(["clearly"] * 17), "seconds": 10.0},
    {"text": "Close on the ask " + " ".join(["now"] * 16), "seconds": 10.0},
]
#: One uninterrupted execution of the stage: the under-floor draw + the re-ask.
_DRAWS = 2

SessionFactory = async_sessionmaker[AsyncSession]


class _HardPodDeath(BaseException):
    """The pod is gone. A ``BaseException`` so NO handler in the stack runs for it.

    ``tests.support.reentry.SimulatedPodDeath`` is an ``Exception`` because that
    harness calls the stage directly. Through the real executor an ``Exception``
    is a stage FAILURE (``mark_failed`` → terminal run), which is a different
    case: the retry of a failed stage is a new run with an empty ledger.
    """


@pytest.fixture(autouse=True)
def _passage_rate_is_the_prior(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the passage->narration ratio at 1.0 so ``_GOOD_LINES`` stays ON the order.

    ``_GOOD_LINES`` is sized at the harness's raw 120 wpm; at the production ratio
    (0.80) the Phase-A ask becomes 48 words, the 60-word draw is >10% over, and the
    length re-ask adds a third paid draw — a different count than this ledger test
    proves. The ratio itself is pinned by
    ``tests/services/stages/shared/test_speech_rate_narration_prior.py``.
    """
    from src.services.stages.shared import speech_rate

    monkeypatch.setattr(speech_rate, "PASSAGE_TO_NARRATION_RATIO", 1.0)


@pytest.fixture(autouse=True)
def _enforce_rls(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The prod RLS posture for the app's own binders; restored afterwards."""
    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    clear_settings_cache()
    clear_identity_context()
    install_rls_tenant_rebind()  # idempotent — asgi installs it at startup in prod
    yield
    clear_settings_cache()
    clear_identity_context()


@pytest.fixture
async def runtime_sessions(db_session_factory: SessionFactory) -> AsyncIterator[SessionFactory]:
    """A sessionmaker whose EVERY connection is a non-owner ``NOBYPASSRLS`` role.

    ``SET ROLE`` runs once per new connection (pool ``connect`` event) on a
    dedicated engine, so the role holds across commits and rollbacks alike and
    cannot leak into the owner engine's pool. Row visibility on these sessions
    depends on ``app.current_tenant`` and on nothing else.
    """
    role = f"ledger_runtime_{uuid4().hex[:12]}"
    async with db_session_factory() as owner:
        schema = (await owner.execute(text("SELECT current_schema()"))).scalar_one()
        try:
            await owner.execute(text(f'CREATE ROLE "{role}" NOLOGIN NOSUPERUSER NOBYPASSRLS'))
        except ProgrammingError as exc:  # insufficient_privilege → no CREATEROLE
            await owner.rollback()
            require_createrole(exc)  # fail under CI, skip for a locally-restricted role
        for grant in (
            f'GRANT "{role}" TO CURRENT_USER',
            f'GRANT USAGE ON SCHEMA "{schema}" TO "{role}"',
            f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA "{schema}" TO "{role}"',
            f'GRANT USAGE ON ALL SEQUENCES IN SCHEMA "{schema}" TO "{role}"',
            # The SECURITY DEFINER resolvers (run_tenant, orphan_runs, …) are not
            # PUBLIC-executable; prod grants them to the runtime role by name.
            f'GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA "{schema}" TO "{role}"',
        ):
            await owner.execute(text(grant))
        await owner.commit()

    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    @event.listens_for(engine.sync_engine, "connect")
    def _become_runtime_role(dbapi_connection: Any, _record: object) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute(f'SET ROLE "{role}"')
        cursor.close()
        dbapi_connection.commit()

    try:
        yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    finally:
        await engine.dispose()
        async with db_session_factory() as owner:
            for stmt in (
                f'REVOKE ALL ON ALL TABLES IN SCHEMA "{schema}" FROM "{role}"',
                f'REVOKE ALL ON ALL SEQUENCES IN SCHEMA "{schema}" FROM "{role}"',
                f'REVOKE ALL ON ALL FUNCTIONS IN SCHEMA "{schema}" FROM "{role}"',
                f'REVOKE ALL ON SCHEMA "{schema}" FROM "{role}"',
                f'REVOKE "{role}" FROM CURRENT_USER',
                f'DROP OWNED BY "{role}"',
                f'DROP ROLE IF EXISTS "{role}"',
            ):
                try:
                    await owner.execute(text(stmt))
                    await owner.commit()
                except ProgrammingError:
                    await owner.rollback()


def _wm(out: Outbound) -> SimpleNamespace:
    """The model: an EMPTY first answer, a real script once it is re-asked.

    A pure function of the prompt, so the two pods and the reference run are
    answered identically. The armed :class:`Outbound` turns the call that "kills
    the pod" into a :class:`_HardPodDeath`.
    """

    def _answer(*, message: str, **_: object) -> SimpleNamespace:
        lines = _GOOD_LINES if "REJECTED" in message else []
        return SimpleNamespace(output={"lines": lines}, usage_metrics={"total_tokens": 7})

    counted = out.fake(_WM, _answer)

    async def message(**kwargs: object) -> object:
        try:
            return await counted(**kwargs)
        except SimulatedPodDeath as exc:
            raise _HardPodDeath(str(exc)) from exc

    return SimpleNamespace(message=message)


def _dispatcher(sessions: SessionFactory, redis_client: Any) -> Any:
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher

    return RunDispatcher(session_factory=sessions, registry=get_registry(), redis=redis_client)


async def _start_run(
    owner_sessions: SessionFactory, runtime: SessionFactory, *, tenant: UUID, subject: UUID
) -> UUID:
    """A generic run that pauses for review right after ``script_draft``."""
    from src.services.blueprints.registry import get_registry
    from src.services.run_engine import RunEngine

    _project, deliverable = await _create_project(
        owner_sessions, tenant=tenant, subject=subject, type_="generic", brief=_BRIEF
    )
    engine = RunEngine(
        session_factory=runtime, registry=get_registry(), dispatch=lambda rid: asyncio.sleep(0)
    )
    run = await engine.start_run(
        deliverable_id=deliverable,
        tenant_id=tenant,
        subject_id=subject,
        parent_run_id=None,
        pause_after_stages=[_STAGE],
    )
    return run.id


async def _execute_on_a_pod(dispatcher: Any, run_id: UUID) -> BaseException | None:
    """Run ``_run`` in its own task (as ``dispatch`` does); return what killed it."""
    task = asyncio.create_task(dispatcher._run(run_id))
    await asyncio.wait({task})
    return task.exception()


async def _run_status(sessions: SessionFactory, run_id: UUID) -> str:
    async with sessions() as session:
        run = await RunRepository(session).get(run_id)
    return run.status if run is not None else "missing"


async def _wait_for_status(sessions: SessionFactory, run_id: UUID, status: str) -> str:
    """Poll until the run reaches ``status`` or any terminal one; return what was seen."""
    seen = "?"
    for _ in range(150):
        seen = await _run_status(sessions, run_id)
        if seen in (status, "failed", "completed", "cancelled"):
            break
        await asyncio.sleep(0.1)
    return seen


async def _stage_row(sessions: SessionFactory, run_id: UUID) -> Any:
    async with sessions() as session:
        row = await StageOutputRepository(session).get(run_id, _STAGE)
    assert row is not None
    return row


async def _ledger_rows(sessions: SessionFactory, run_id: UUID) -> list[StageStepResult]:
    """Ledger rows of the stage, read as the table OWNER (bypasses RLS): ground truth."""
    from sqlalchemy import select

    async with sessions() as session:
        rows = await session.execute(
            select(StageStepResult)
            .where(StageStepResult.run_id == run_id, StageStepResult.stage_name == _STAGE)
            .order_by(StageStepResult.created_at, StageStepResult.step_key)
        )
        return list(rows.scalars())


def _element_ids(row: StageStepResult) -> list[str]:
    """The element ids a stored script draw minted (``[]`` for an empty draw)."""
    envelope = cast("dict[str, dict[str, list[str]]]", row.result)
    return envelope["value"]["element_ids"]


def _script(output: dict[str, Any]) -> list[tuple[str, float]]:
    return [(line["text"], line["seconds"]) for line in output["lines"]]


async def _uninterrupted_reference(
    owner_sessions: SessionFactory,
    runtime: SessionFactory,
    redis_client: Any,
    *,
    tenant: UUID,
    subject: UUID,
) -> tuple[dict[str, Any], int]:
    """The same stage on the same posture with no death: its output and its bill."""
    out = Outbound()
    run_id = await _start_run(owner_sessions, runtime, tenant=tenant, subject=subject)
    pod = _dispatcher(runtime, redis_client)
    with patch("src.services.stages.primitives.script_stage.WMClient", return_value=_wm(out)):
        died = await _execute_on_a_pod(pod, run_id)
    await pod.shutdown()
    assert died is None
    assert await _wait_for_status(owner_sessions, run_id, "paused") == "paused"
    row = await _stage_row(owner_sessions, run_id)
    assert row.status == "completed"
    return dict(row.output), out.calls[_WM]


async def _resume_on_a_new_pod(
    runtime: SessionFactory, owner_sessions: SessionFactory, redis_client: Any, run_id: UUID
) -> None:
    """What the next pod's startup does: the real orphan reconciler, a fresh dispatcher."""
    from src.lifecycle.reconciler import _reconcile_orphan_runs

    pod = _dispatcher(runtime, redis_client)
    try:
        await _reconcile_orphan_runs(runtime, pod, max_age_seconds=7200, redis=redis_client)
        assert await _wait_for_status(owner_sessions, run_id, "paused") == "paused", (
            "the reconciler did not resume the orphaned run to its review pause"
        )
    finally:
        await pod.shutdown()


async def test_pod_death_mid_stage_resumes_without_repaying_finished_draws(
    db_session_factory: SessionFactory, runtime_sessions: SessionFactory, redis_client: Any
) -> None:
    tenant, subject = uuid4(), uuid4()
    reference_output, reference_calls = await _uninterrupted_reference(
        db_session_factory, runtime_sessions, redis_client, tenant=tenant, subject=subject
    )
    assert reference_calls == _DRAWS
    assert _script(reference_output) == [(x["text"], x["seconds"]) for x in _GOOD_LINES]

    out = Outbound()
    run_id = await _start_run(db_session_factory, runtime_sessions, tenant=tenant, subject=subject)
    with patch("src.services.stages.primitives.script_stage.WMClient", return_value=_wm(out)):
        # ── pod 1: the first draw finishes and is recorded; the re-ask kills the pod.
        out.arm(die_after=1)
        pod_1 = _dispatcher(runtime_sessions, redis_client)
        died = await _execute_on_a_pod(pod_1, run_id)
        out.disarm()
        await pod_1.shutdown()

        assert isinstance(died, _HardPodDeath), f"pod 1 did not die mid-stage: {died!r}"
        assert out.calls == {_WM: 1}
        orphan = await _stage_row(db_session_factory, run_id)
        # Left exactly as a dead pod leaves it: no handler ran, no counter moved.
        assert orphan.status == "running"
        assert (orphan.attempts, orphan.interruptions) == (0, 0)
        assert await _run_status(db_session_factory, run_id) == "running"
        finished = await _ledger_rows(db_session_factory, run_id)
        assert len(finished) == 1, "the finished draw must be durable before the pod died"

        # ── pod 2: the reconciler finds the orphan and re-enters the stage.
        await _resume_on_a_new_pod(runtime_sessions, db_session_factory, redis_client, run_id)

    # Paid for exactly what ONE uninterrupted execution pays for — not 1 + 2.
    assert out.calls == {_WM: reference_calls}
    resumed = await _stage_row(db_session_factory, run_id)
    assert resumed.status == "completed"
    assert (resumed.attempts, resumed.interruptions) == (0, 0)
    # …and delivered the same script (element ids are minted per run, so compare
    # everything else; id stability across the death is the next test).
    assert _script(dict(resumed.output)) == _script(reference_output)
    assert resumed.output["total_seconds"] == reference_output["total_seconds"]
    assert {k for k in resumed.output if k != "lines"} == {
        k for k in reference_output if k != "lines"
    }
    rows = await _ledger_rows(db_session_factory, run_id)
    assert len(rows) == _DRAWS
    assert rows[0].id == finished[0].id, "the dead pod's row was replaced, not reused"
    assert rows[0].updated_at == finished[0].updated_at

    await _assert_only_the_owning_tenant_can_read(
        runtime_sessions, db_session_factory, rows, tenant
    )


async def test_pod_death_after_the_last_draw_resumes_with_zero_calls_and_the_same_ids(
    db_session_factory: SessionFactory, runtime_sessions: SessionFactory, redis_client: Any
) -> None:
    """Every sub-step finished, the stage output never landed: nothing is re-paid.

    The pod dies writing the stage's ``completed`` row. The re-entry replays BOTH
    draws from the table and hands downstream the element ids the dead pod minted —
    through a real JSONB round trip, under RLS.
    """
    tenant, subject = uuid4(), uuid4()
    out = Outbound()
    run_id = await _start_run(db_session_factory, runtime_sessions, tenant=tenant, subject=subject)

    real_mark_completed = StageOutputRepository.mark_completed
    armed = True

    async def _die_persisting(self: StageOutputRepository, rid: UUID, stage: str, output: dict):
        nonlocal armed
        if armed and stage == _STAGE:
            armed = False
            raise _HardPodDeath("simulated pod death before the stage output was written")
        return await real_mark_completed(self, rid, stage, output)

    with (
        patch("src.services.stages.primitives.script_stage.WMClient", return_value=_wm(out)),
        patch.object(StageOutputRepository, "mark_completed", _die_persisting),
    ):
        pod_1 = _dispatcher(runtime_sessions, redis_client)
        died = await _execute_on_a_pod(pod_1, run_id)
        await pod_1.shutdown()

        assert isinstance(died, _HardPodDeath), f"pod 1 did not die: {died!r}"
        assert out.calls == {_WM: _DRAWS}
        orphan = await _stage_row(db_session_factory, run_id)
        assert orphan.status == "running"
        assert orphan.output is None
        before = await _ledger_rows(db_session_factory, run_id)
        assert len(before) == _DRAWS
        minted = [ids for row in before if (ids := _element_ids(row))]
        assert len(minted) == 1
        assert len(minted[0]) == len(_GOOD_LINES)

        await _resume_on_a_new_pod(runtime_sessions, db_session_factory, redis_client, run_id)

    assert out.calls == {_WM: _DRAWS}, "the re-entry re-sent a model draw that had finished"
    resumed = await _stage_row(db_session_factory, run_id)
    assert resumed.status == "completed"
    assert _script(dict(resumed.output)) == [(x["text"], x["seconds"]) for x in _GOOD_LINES]
    assert [line["id"] for line in resumed.output["lines"]] == minted[0]
    after = await _ledger_rows(db_session_factory, run_id)
    assert [(r.id, r.input_hash, r.updated_at) for r in after] == [
        (r.id, r.input_hash, r.updated_at) for r in before
    ]


async def _assert_only_the_owning_tenant_can_read(
    runtime: SessionFactory,
    owner_sessions: SessionFactory,
    rows: list[StageStepResult],
    tenant: UUID,
) -> None:
    """The same store class the dispatcher wires, on the runtime role, per tenant."""
    run_id = rows[0].run_id
    stranger = uuid4()

    async def _visible(as_tenant: UUID | None) -> int:
        async with runtime() as session:
            from src.database.session import bind_session_tenant

            await bind_session_tenant(session, as_tenant)
            return (
                await session.execute(
                    text("SELECT count(*) FROM stage_step_results WHERE run_id = :r"),
                    {"r": run_id},
                )
            ).scalar_one()

    # Positive control first: a probe that cannot see the owner's rows either
    # would make every "invisible" below vacuous.
    own_store = SessionStepStore(runtime, tenant)
    for row in rows:
        mine = await own_store.get(run_id, _STAGE, row.step_key)
        assert mine is not None
        assert mine.input_hash == row.input_hash
    assert await _visible(tenant) == len(rows)

    foreign_store = SessionStepStore(runtime, stranger)
    unbound_store = SessionStepStore(runtime, None)
    for row in rows:
        assert await foreign_store.get(run_id, _STAGE, row.step_key) is None
        assert await unbound_store.get(run_id, _STAGE, row.step_key) is None
    assert await _visible(stranger) == 0
    assert await _visible(None) == 0

    # A foreign ledger on the same key misses (it runs its own step), and its
    # write is refused by the policy — absorbed by the ledger, never an overwrite.
    calls = 0

    async def _step() -> str:
        nonlocal calls
        calls += 1
        return "written by another tenant"

    target = rows[0]
    foreign: Callable[[], Any] = lambda: StepLedger(foreign_store, run_id, _STAGE).once(  # noqa: E731
        target.step_key, target.input_hash, _step
    )
    assert await foreign() == "written by another tenant"
    assert calls == 1
    untouched = await _ledger_rows(owner_sessions, run_id)
    assert [(r.id, r.input_hash, r.result, r.updated_at) for r in untouched] == [
        (r.id, r.input_hash, r.result, r.updated_at) for r in rows
    ]
