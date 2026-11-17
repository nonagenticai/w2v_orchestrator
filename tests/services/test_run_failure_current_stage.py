"""A failed run must name the stage that FAILED in ``current_stage``.

Prod run 01a0a054 (website_review_v1): ``website_review_record`` failed with
``StageRetriesExhaustedError`` while its concurrent siblings
``website_review_visual_director`` / ``website_review_slides`` started a few ms
LATER and completed. ``current_stage`` is written at stage START, so the failed
run reported ``current_stage='website_review_visual_director'`` — a stage that
succeeded — next to an ``error`` naming ``website_review_record``.

This drives the REAL dispatcher + DAG executor + Postgres with exactly that shape:
two co-eligible stages, the earlier-started one fails, the later-started one
completes first.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from src.services.stages.base import StageHandler, StageInput, StageOutput

pytestmark = pytest.mark.asyncio

_FAILS = "aa_parallel_fails"  # sorts first ⇒ its task is created (and starts) first
_COMPLETES = "zz_parallel_completes"  # starts after, finishes before the failure


class _In(StageInput):
    """No inputs."""


class _Out(StageOutput):
    ok: bool = True


_completes_done = asyncio.Event()
_observed_before_failure: dict[str, str | None] = {}


class _CompletesStage(StageHandler[_In, _Out]):
    name = _COMPLETES
    input_model = _In
    output_model = _Out

    async def handle(self, payload: _In, ctx: object) -> _Out:
        _completes_done.set()
        return _Out()


class _FailsStage(StageHandler[_In, _Out]):
    name = _FAILS
    input_model = _In
    output_model = _Out

    session_factory = None  # injected by the test

    async def handle(self, payload: _In, ctx: object) -> _Out:
        from src.database.repositories.run_repository import RunRepository
        from src.database.repositories.stage_output_repository import StageOutputRepository

        await asyncio.wait_for(_completes_done.wait(), timeout=10)
        # Wait until the sibling's START write is what the run row shows — the
        # exact pre-failure state of the prod run. Recorded so the test can prove
        # it is not vacuous (the defect's precondition actually held).
        run_id = ctx.run_id  # ty: ignore[unresolved-attribute]
        factory = type(self).session_factory
        assert factory is not None
        for _ in range(100):
            async with factory() as s:
                r = await RunRepository(s).get(run_id)
            _observed_before_failure["current_stage"] = r.current_stage if r else None
            if r is not None and r.current_stage == _COMPLETES:
                break
            await asyncio.sleep(0.02)

        # …and then wait for the sibling to have actually FINISHED. Both signals
        # above fire at stage START — `current_stage` is written when the stage
        # begins, and `_completes_done` is set at the top of the sibling's
        # handler — so on those alone this test only established that the sibling
        # STARTED first. A failure raised here makes the executor cancel the rest
        # of the batch (dag_executor `_drain_batch(cancel_first=True)`), and
        # `mark_failed` then sweeps rows still 'running' to 'cancelled': the
        # sibling that SUCCEEDED ends up reading 'cancelled'. That is a race in
        # the test, not in the product — a sibling genuinely still running when
        # the run fails is supposed to be cancelled. Measured 2026-10-29: one
        # full-suite run at load 20+ failed here and the same test passed 3/3 in
        # isolation, which is what a lost race looks like.
        for _ in range(500):
            async with factory() as s:
                rows = await StageOutputRepository(s).list_for_run(run_id)
            if any(r.stage_name == _COMPLETES and r.status == "completed" for r in rows):
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError(
                "the completing sibling never reached 'completed' before the failure; "
                "this test can no longer create the prod shape it asserts on"
            )
        raise ValueError("record: tts /synthesize returned 503")


async def test_failed_run_current_stage_names_the_failing_parallel_stage(
    db_session_factory, redis_client
) -> None:
    from src.database.repositories.project_repository import ProjectRepository
    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository
    from src.services.blueprints.base import DAGBlueprint, StageSpec, handler
    from src.services.blueprints.generic import GENERIC_BLUEPRINT
    from src.services.blueprints.registry import BlueprintRegistry, get_registry
    from src.services.run_dispatcher import RunDispatcher
    from src.services.run_engine import RunEngine

    _completes_done.clear()
    _observed_before_failure.clear()
    _FailsStage.session_factory = db_session_factory

    tenant, subject = uuid4(), uuid4()
    async with db_session_factory() as session:
        proj = await ProjectRepository(session).create_with_deliverables(
            tenant_id=tenant,
            owner_subject_id=subject,
            type_="generic",
            title="x",
            brief={"angle": "growth", "audience": "saas", "target_duration_seconds": 30},
            brand={},
            voice={"descriptor": "crisp"},
            theme={},
            visibility="tenant",
            deliverables=[
                {
                    "name": "d",
                    "format": "16x9",
                    "captions_mode": "none",
                    "target_duration_seconds": 30,
                }
            ],
        )
        await session.commit()
        deliverable_id = proj.deliverables[0].id

    engine = RunEngine(
        session_factory=db_session_factory,
        registry=get_registry(),
        dispatch=lambda rid: asyncio.sleep(0),
    )
    run = await engine.start_run(
        deliverable_id=deliverable_id,
        tenant_id=tenant,
        subject_id=subject,
        parent_run_id=None,
        pause_after_stages=[],
    )
    # Every shipped stage already done ⇒ the only remaining wave is the two
    # co-eligible stages below (no downstream client is touched).
    async with db_session_factory() as session:
        sor = StageOutputRepository(session)
        for name in GENERIC_BLUEPRINT.topological_order():
            await sor.mark_completed(run.id, name, {"ok": True})
        await session.commit()

    tail = GENERIC_BLUEPRINT.topological_order()[-1]
    widened = DAGBlueprint(
        id=GENERIC_BLUEPRINT.id,
        project_type=GENERIC_BLUEPRINT.project_type,
        terminal_artifact_stage=GENERIC_BLUEPRINT.terminal_artifact_stage,
        stages=[
            *GENERIC_BLUEPRINT.stages,
            StageSpec(
                name=_FAILS,
                depends_on=(tail,),
                handler_factory=handler(_FailsStage),
                default_timeout_seconds=30,
            ),
            StageSpec(
                name=_COMPLETES,
                depends_on=(tail,),
                handler_factory=handler(_CompletesStage),
                default_timeout_seconds=30,
            ),
        ],
    )
    registry = BlueprintRegistry()
    registry.register(widened)

    dispatcher = RunDispatcher(
        session_factory=db_session_factory, registry=registry, redis=redis_client
    )
    await dispatcher._run(run.id)

    async with db_session_factory() as session:
        after = await RunRepository(session).get(run.id)
        stages = {
            s.stage_name: s.status
            for s in await StageOutputRepository(session).list_for_run(run.id)
        }

    # Non-vacuity: the defect's precondition held — right before the failure the
    # run row named the COMPLETED sibling.
    assert _observed_before_failure.get("current_stage") == _COMPLETES, _observed_before_failure
    assert stages[_COMPLETES] == "completed"
    assert stages[_FAILS] == "failed"

    assert after is not None
    assert after.status == "failed"
    assert after.error is not None and "503" in after.error
    assert after.current_stage == _FAILS, (
        "a failed run must name the stage that failed, not the last sibling to "
        f"start; got current_stage={after.current_stage!r} while {_FAILS!r} failed"
    )
    assert stages[after.current_stage] == "failed"
