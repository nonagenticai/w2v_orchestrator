import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import delete

from src.clients.schemas.tts import TTSSynthesis
from src.database.models import StageOutput as StageOutputRow
from src.services.stages.base import StageHandler, StageInput, StageOutput

# Needs the DB + redis_client fixtures, both of which CI provides (ephemeral
# postgres container + fakeredis). Deliberately NOT `@pytest.mark.integration`:
# in this repo that marker is deselected by `addopts` and re-selected NOWHERE,
# so marking a test with it means the test runs on no machine but a developer's.
pytestmark = pytest.mark.asyncio

#: Name of the stage the blueprint "gains" after a run was already created — see
#: test_dispatcher_seeds_rows_for_a_stage_the_blueprint_gained_later.
_LATE_STAGE = "late_added_audit"


class _LateAddedInput(StageInput):
    """No inputs — the stage exists only to be a trailing DAG node."""


class _LateAddedOutput(StageOutput):
    ok: bool = True


class _LateAddedStage(StageHandler[_LateAddedInput, _LateAddedOutput]):
    """A deterministic, network-free stage appended to a blueprint mid-flight.

    Stands in for any stage added to a shipped blueprint (a new critique, audit,
    or QA node). It always completes, so the ONLY thing that can keep the run
    from reaching a terminal state is a missing ``stage_output`` row.
    """

    name = _LATE_STAGE
    input_model = _LateAddedInput
    output_model = _LateAddedOutput

    async def handle(self, payload: _LateAddedInput, ctx: object) -> _LateAddedOutput:
        return _LateAddedOutput()


async def test_dispatcher_runs_generic_blueprint(db_session_factory, redis_client) -> None:
    """Full integration smoke — generic DAG with mocked downstream clients."""
    from src.database.repositories.project_repository import ProjectRepository
    from src.database.repositories.run_repository import RunRepository
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher
    from src.services.run_engine import RunEngine

    tenant = uuid4()
    subject = uuid4()

    # Create project + deliverable using a dedicated session
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
        project_id = proj.id

    from w2v_common.clients import WMResponse

    fake_lr = AsyncMock()
    fake_lr.message.return_value = WMResponse(
        output={"lines": [{"text": "hi", "seconds": 5.0}]},
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )
    fake_render = AsyncMock()
    fake_render.compile.return_value = {
        "artifact_url": "http://a/x.mp4",
        "duration_seconds": 30.0,
    }
    # The narrate stage builds a real TTSClient and calls tts.test, which
    # 401s on a dev box (no license token). Mock the client + the storage write
    # so the run completes instead of failing on TTS.
    fake_tts = AsyncMock()
    fake_tts.synthesize.return_value = TTSSynthesis.model_validate(
        {
            "audio_bytes": b"\x00",
            "format": "wav",
            "duration_seconds": 5.0,
            "sample_rate": 24000,
            "provider": "test",
            "voice": "",
        }
    )

    # generic seeds ai_image_gen by default (per-shot stills), and script_draft
    # probes the voice's measured wpm through TTS. Both are metered calls; this
    # run is started outside a request, so its snapshot carries no tenant
    # provenance and a REAL client refuses to delegate (UnvouchedTenantError) —
    # exactly as every other metered client here is mocked rather than called.
    from src.clients.schemas.ai_content_generation import AcgConceptImageJob
    from src.services.stages.shared.speech_rate import ScriptSpeechRate, SpeechRate

    fake_acg = AsyncMock()
    fake_acg.create_concept_image.return_value = "img-job-1"
    fake_acg.wait_for_concept_image.return_value = AcgConceptImageJob(
        job_id="img-job-1",
        status="completed",
        asset_id="asset-1",
        image_url="http://a/img.png",
        provider="test",
    )
    rate = SpeechRate(wpm=150.0, measured=True)

    with (
        patch(
            "src.services.stages.generic.ai_image_gen.AIContentGenerationClient",
            return_value=fake_acg,
        ),
        patch(
            "src.services.stages.generic.ai_image_gen.find_existing_media",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "src.services.stages.generic.script_draft.resolve_effective_wpm",
            new=AsyncMock(return_value=ScriptSpeechRate(rate=rate, prior=rate)),
        ),
        patch(
            "src.services.stages.primitives.script_stage.WMClient",
            return_value=fake_lr,
        ),
        patch(
            "src.services.stages.primitives.compile_stage.VideoCompileClient",
            return_value=fake_render,
        ),
        patch(
            "src.services.stages.primitives.narration.TTSClient",
            return_value=fake_tts,
        ),
        patch(
            "src.services.stages.primitives.narration.store_tts_clip",
            new=AsyncMock(return_value="http://a/clip.wav"),
        ),
    ):
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
        dispatcher = RunDispatcher(
            session_factory=db_session_factory,
            registry=get_registry(),
            redis=redis_client,
        )
        await dispatcher.dispatch(run.id)
        for _ in range(50):
            await asyncio.sleep(0.2)
            async with db_session_factory() as poll_session:
                r = await RunRepository(poll_session).get(run.id)
            if r is not None and r.status in ("completed", "failed"):
                break
        await dispatcher.shutdown()

    async with db_session_factory() as final_session:
        final = await RunRepository(final_session).get(run.id)
        # The dispatcher rolls the run's terminal state up to the owning
        # project: a completed run → project 'completed' with finished_at set.
        proj_after = await ProjectRepository(final_session).get(project_id, tenant_id=tenant)
    assert final is not None
    assert final.status == "completed", final.error
    assert proj_after is not None
    assert proj_after.status == "completed"
    assert proj_after.finished_at is not None
    # The execution lease is released once the run finishes, so a later
    # reconcile (or rerun) can re-acquire it.
    assert await redis_client.get(f"runexec:{run.id}") is None


async def test_run_skips_when_lease_held_by_another_instance(
    db_session_factory, redis_client
) -> None:
    """A run whose execution lease is already held by another instance must be
    skipped without touching its row — this is what stops the reconciler from
    double-executing a run still live on a draining pod."""
    from src.database.repositories.project_repository import ProjectRepository
    from src.database.repositories.run_repository import RunRepository
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher
    from src.services.run_engine import RunEngine

    tenant = uuid4()
    subject = uuid4()
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

    # Another live instance already owns the run's lease.
    await redis_client.set(f"runexec:{run.id}", b"other-instance", ex=60)

    dispatcher = RunDispatcher(
        session_factory=db_session_factory,
        registry=get_registry(),
        redis=redis_client,
    )
    # No downstream clients are patched on purpose: if _run did NOT short-circuit
    # it would try to execute the DAG and fail loudly.
    await dispatcher._run(run.id)

    async with db_session_factory() as session:
        after = await RunRepository(session).get(run.id)
    assert after is not None
    assert after.status == "pending"  # never advanced to "running"
    # The foreign lease was not clobbered.
    assert await redis_client.get(f"runexec:{run.id}") == b"other-instance"


async def test_dispatcher_seeds_rows_for_a_stage_the_blueprint_gained_later(
    db_session_factory, redis_client
) -> None:
    """THE headline case: a blueprint that GAINS a stage after a run was created.

    ``stage_output`` rows are written exactly once, at run creation. Ship a new
    trailing stage and every pre-existing run has no row for it — and every
    writer (``mark_running``/``mark_completed``/``mark_skipped``) is a bare
    ``UPDATE``, so each one silently touches ZERO rows. The stage then never
    appears in ``terminal_stage_names``, ``decide_run_terminal_state``'s
    set-equality against ``blueprint.topological_order()`` can never hold, and
    the dispatcher returns CONTINUE — leaving the run ``running`` FOREVER.

    WITHOUT ``reconcile_blueprint_stage_rows`` this test fails: the executor
    would run ``late_added_audit`` and persist 'completed' into nothing, the
    final gate would see 12 terminal rows against 13 blueprint stages, and the
    run would still read ``running`` at the assertion below (the CONTINUE
    invariant-breach branch). It is a real hang, not a slow test — the run never
    transitions again without operator intervention.
    """
    from src.database.repositories.project_repository import ProjectRepository
    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository
    from src.services.blueprints.base import DAGBlueprint, StageSpec, handler
    from src.services.blueprints.generic import GENERIC_BLUEPRINT
    from src.services.blueprints.registry import BlueprintRegistry, get_registry
    from src.services.run_dispatcher import RunDispatcher
    from src.services.run_engine import RunEngine

    tenant = uuid4()
    subject = uuid4()
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

    # The run is created against the SHIPPED generic blueprint, so it gets rows
    # for exactly those stages and nothing more.
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

    # Pre-complete every original stage so the DAG has only the newly-added
    # trailing stage left to execute — no downstream clients are patched, and
    # none are needed: `_LateAddedStage` is network-free.
    async with db_session_factory() as session:
        sor = StageOutputRepository(session)
        for name in GENERIC_BLUEPRINT.topological_order():
            await sor.mark_completed(run.id, name, {"ok": True})
        await session.commit()

    # NOW the blueprint gains a stage that depends on the previously-terminal
    # one. Same `id`, so `resolve_by_id(run.blueprint_id)` still resolves — this
    # is exactly what a deploy of a widened blueprint looks like to an old run.
    # A private registry instance (rather than monkeypatching the process-wide
    # singleton) keeps the mutation scoped to this test.
    extended = DAGBlueprint(
        id=GENERIC_BLUEPRINT.id,
        project_type=GENERIC_BLUEPRINT.project_type,
        terminal_artifact_stage=GENERIC_BLUEPRINT.terminal_artifact_stage,
        stages=[
            *GENERIC_BLUEPRINT.stages,
            StageSpec(
                name=_LATE_STAGE,
                # Hangs off the previously-terminal stage, so it is the new tail.
                depends_on=("video_render",),
                handler_factory=handler(_LateAddedStage),
                default_timeout_seconds=30,
            ),
        ],
    )
    extended_registry = BlueprintRegistry()
    extended_registry.register(extended)

    dispatcher = RunDispatcher(
        session_factory=db_session_factory,
        registry=extended_registry,
        redis=redis_client,
    )
    await dispatcher._run(run.id)

    async with db_session_factory() as session:
        after = await RunRepository(session).get(run.id)
        stages = {
            s.stage_name: s.status
            for s in await StageOutputRepository(session).list_for_run(run.id)
        }
    assert after is not None
    assert after.status == "completed", (
        "a run whose blueprint gained a stage must still reach a terminal state; "
        f"got {after.status!r} (error={after.error!r}) — this is the forever-'running' hang"
    )
    # The reconciler created the row, and the executor then actually wrote to it.
    assert _LATE_STAGE in stages, "the gained stage must have been seeded a row"
    assert stages[_LATE_STAGE] == "completed", (
        f"the gained stage's mark_completed must have hit a real row; got {stages[_LATE_STAGE]!r}"
    )
    assert await redis_client.get(f"runexec:{run.id}") is None


async def test_dispatcher_seeds_a_missing_row_mid_dag(db_session_factory, redis_client) -> None:
    """Same defect, reached from the other direction: a row-less stage in the
    MIDDLE of the DAG (here forced by deleting it, which is indistinguishable
    from the row a widened blueprint never had).

    ``media_discovery`` is opt-in and skips-clean without ``brief.media_discovery``,
    so the executor persists 'skipped' — which counts as terminal. But
    ``mark_skipped`` is a bare ``UPDATE`` too: WITHOUT the reconcile it writes
    zero rows, ``media_discovery`` stays absent from ``list_for_run``, the
    completion set-equality never holds, and the run sits ``running`` forever
    exactly as in the headline case above. The two assertions at the end are
    what would fail: no row would exist for the stage at all.
    """
    from src.database.repositories.project_repository import ProjectRepository
    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository
    from src.services.blueprints.generic import GENERIC_BLUEPRINT
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher
    from src.services.run_engine import RunEngine

    missing_stage = "media_discovery"

    tenant = uuid4()
    subject = uuid4()
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

    # Complete every OTHER stage, then delete the one row — leaving a DAG whose
    # single remaining stage has no row to write to.
    async with db_session_factory() as session:
        sor = StageOutputRepository(session)
        for name in GENERIC_BLUEPRINT.topological_order():
            if name != missing_stage:
                await sor.mark_completed(run.id, name, {"ok": True})
        await session.execute(
            delete(StageOutputRow).where(
                StageOutputRow.run_id == run.id,
                StageOutputRow.stage_name == missing_stage,
            )
        )
        await session.commit()

    async with db_session_factory() as session:
        before = {s.stage_name for s in await StageOutputRepository(session).list_for_run(run.id)}
    assert missing_stage not in before, "precondition: the row really is gone"

    dispatcher = RunDispatcher(
        session_factory=db_session_factory,
        registry=get_registry(),
        redis=redis_client,
    )
    await dispatcher._run(run.id)

    async with db_session_factory() as session:
        after = await RunRepository(session).get(run.id)
        stages = {
            s.stage_name: s.status
            for s in await StageOutputRepository(session).list_for_run(run.id)
        }
    assert after is not None
    assert after.status == "completed", (
        f"a row-less mid-DAG stage must not wedge the run; got {after.status!r} "
        f"(error={after.error!r})"
    )
    assert missing_stage in stages, "the reconciler must have re-seeded the deleted row"
    assert stages[missing_stage] != "pending", (
        f"the re-seeded row must have been written by the executor; got {stages[missing_stage]!r}"
    )
