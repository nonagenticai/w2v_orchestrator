"""In-process full-DAG dispatch integration tests.

These drive a real run through the in-process dispatcher
(`RunDispatcher.dispatch` -> `asyncio.create_task` on the test's own event
loop) to a terminal status, with every downstream service client mocked at its
stage-module import site. They validate pipeline WIRING end-to-end — the thing
that, until now, was only ever exercised by the 1.5h post-promote live-cluster
e2e suite (`.github/workflows/e2e-post-promote.yaml`), never in per-PR CI.

Why these are NOT `@pytest.mark.integration` (and therefore DO run in default
per-PR CI): that marker means "hits live Docker containers on localhost". These
tests hit only Postgres (already required by the whole default suite via the
`db_session_factory` fixture) and *fakeredis* (`redis_client` is an in-memory
`fakeredis.aioredis.FakeRedis`, see tests/conftest.py) — no live redis, no live
downstream service. They are the same dependency class as `tests/database/*`
and `tests/api/routes/*`, which already run in CI. The full-DAG generic run
executes in ~1.2s.
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.clients.schemas.tts import TTSSynthesis
from tests.integration._dispatch_harness import (
    _compile_mock,
    _create_project,
    _dispatch_and_wait,
    _slide_service_mock,
    _tts_mock,
    _wm_client,
)

# ---------------------------------------------------------------------------
# Archetype-specific downstream mocks + briefs
# ---------------------------------------------------------------------------


def _generic_downstream_mocks() -> ExitStack:
    """The complete set of downstream client mocks a plain `generic` run needs.

    - script_stage.WMClient       -> LLM script draft
    - compile_stage.VideoCompileClient -> video_production compile
    - narration.TTSClient + narration.store_tts_clip -> TTS synth + storage write

    - ai_image_gen.AIContentGenerationClient (+ store_generated_image) -> the
      per-line stills ai_image_gen renders by DEFAULT on every generic run
      (run_setup seeds ``prompts_from_stage = "storyboard_draft"``)

    All other stages (storyboard_draft is pure; ai_video_gen /
    motion_graphics_gen / source_ingest / media_discovery are opt-in skip-clean
    leaves) require no mock on a config-less run.
    """
    from src.clients.schemas.ai_content_generation import AcgConceptImageJob

    fake_acg = AsyncMock()
    fake_acg.create_concept_image.return_value = "img-job-1"
    fake_acg.wait_for_concept_image.return_value = AcgConceptImageJob(
        job_id="img-job-1",
        status="completed",
        asset_id="asset-1",
        image_url="http://a/img.png",
        provider="test",
    )
    fake_render = AsyncMock()
    fake_render.compile.return_value = {
        "artifact_url": "http://a/x.mp4",
        "duration_seconds": 30.0,
    }
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

    stack = ExitStack()
    stack.enter_context(
        patch(
            "src.services.stages.primitives.script_stage.WMClient",
            return_value=_wm_client({"lines": [{"text": "hi", "seconds": 5.0}]}),
        )
    )
    stack.enter_context(
        patch(
            "src.services.stages.primitives.compile_stage.VideoCompileClient",
            return_value=fake_render,
        )
    )
    stack.enter_context(
        patch(
            "src.services.stages.primitives.narration.TTSClient",
            return_value=fake_tts,
        )
    )
    stack.enter_context(
        patch(
            "src.services.stages.primitives.narration.store_tts_clip",
            new=AsyncMock(return_value="http://a/clip.wav"),
        )
    )
    # The clip-reuse probe runs BEFORE the (mocked) store and is otherwise
    # unmocked — it would init the real StorageService (a local /app mkdir).
    # Seal it: None == "no cached clip, synthesize fresh" (the first-run path).
    stack.enter_context(
        patch(
            "src.services.stages.primitives.narration.find_existing_clip",
            new=AsyncMock(return_value=None),
        )
    )
    # ai_image_gen: ACG submit→poll + its storage probes (absolute url ⇒ the
    # re-publish path is inert; patched to keep the import site sealed).
    stack.enter_context(
        patch(
            "src.services.stages.generic.ai_image_gen.AIContentGenerationClient",
            return_value=fake_acg,
        )
    )
    stack.enter_context(
        patch(
            "src.services.stages.generic.ai_image_gen.find_existing_media",
            new=AsyncMock(return_value=None),
        )
    )
    stack.enter_context(
        patch(
            "src.services.stages.generic.ai_image_gen.store_generated_image",
            new=AsyncMock(return_value="http://a/img.png"),
        )
    )
    return stack


_GENERIC_BRIEF = {"angle": "growth", "audience": "saas", "target_duration_seconds": 30}


def _promo_downstream_mocks() -> ExitStack:
    """Downstream client mocks for a plain `promo` run (no source_url / no AI
    image/video / no sound-effects config). Every opt-in visual/footage stage
    skips clean; the mocked clients are the WM/TTS/compile/slide surface."""
    stack = ExitStack()
    p = lambda target, **kw: stack.enter_context(patch(target, **kw))  # noqa: E731

    p(
        "src.services.stages.promo.analyze.WMClient",
        return_value=_wm_client(
            {"pain_points": ["p"], "hook_candidates": ["h"], "tone_hints": ["neutral"]}
        ),
    )
    p(
        "src.services.stages.primitives.script_stage.WMClient",
        return_value=_wm_client({"lines": [{"text": "hi", "seconds": 5.0, "speaker": "a"}]}),
    )
    p(
        "src.services.stages.promo.image_prompts.WMClient",
        return_value=_wm_client({"prompts": []}),
    )
    p("src.services.stages.promo.slides.SlideServiceClient", return_value=_slide_service_mock())
    p("src.services.stages.primitives.narration.TTSClient", return_value=_tts_mock())
    p(
        "src.services.stages.primitives.narration.store_tts_clip",
        new=AsyncMock(return_value="http://a/clip.wav"),
    )
    # Seal the pre-store clip-reuse probe (real StorageService init otherwise).
    p(
        "src.services.stages.primitives.narration.find_existing_clip",
        new=AsyncMock(return_value=None),
    )
    # Slide PNG persistence hits the real StorageService too (local_path defaults
    # to /app/data, unwritable on a CI runner). Seal it like the TTS clip probe.
    p(
        "src.services.stages.shared.slide_deck.find_existing_media",
        new=AsyncMock(return_value=None),
    )
    p(
        "src.services.stages.shared.slide_deck.store_slide_image",
        new=AsyncMock(return_value="http://a/slide.png"),
    )
    p(
        "src.services.stages.primitives.compile_stage.VideoCompileClient",
        return_value=_compile_mock(),
    )
    p(
        "src.services.stages.promo.review.WMClient",
        return_value=_wm_client(
            {
                "items": [{"criterion": "clarity", "score": 4, "rationale": "ok"}],
                "summary": "s",
            }
        ),
    )
    p(
        "src.services.stages.growth_bundle.handler.WMClient",
        return_value=_wm_client({"linkedin": {"text": "t", "cta_url": None}}),
    )
    return stack


_PROMO_BRIEF = {
    "value_proposition": "x",
    "call_to_action": "y",
    "audience": "z",
    "target_duration_seconds": 30,
}


async def test_promo_dag_runs_to_completed(db_session_factory, redis_client) -> None:
    """A plain promo run executes its whole DAG to `completed` with the WM /
    TTS / slide / compile surface mocked and all opt-in visual stages skipped."""
    from src.database.repositories.project_repository import ProjectRepository
    from src.database.repositories.run_repository import RunRepository

    tenant, subject = uuid4(), uuid4()
    project_id, deliverable_id = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="promo",
        brief=_PROMO_BRIEF,
    )

    with _promo_downstream_mocks():
        run_id, final = await _dispatch_and_wait(
            db_session_factory,
            redis_client,
            deliverable_id=deliverable_id,
            tenant=tenant,
            subject=subject,
        )

    async with db_session_factory() as session:
        final = await RunRepository(session).get(run_id)
        proj_after = await ProjectRepository(session).get(project_id, tenant_id=tenant)
    assert final is not None
    assert final.status == "completed", final.error
    assert proj_after is not None
    assert proj_after.status == "completed"


# ---------------------------------------------------------------------------
# Happy path — full DAG to completed
# ---------------------------------------------------------------------------


async def test_generic_dag_runs_to_completed(db_session_factory, redis_client) -> None:
    """A config-less generic run executes its whole DAG to `completed`, and the
    terminal state rolls up to the owning project."""
    from src.database.repositories.project_repository import ProjectRepository
    from src.database.repositories.run_repository import RunRepository

    tenant, subject = uuid4(), uuid4()
    project_id, deliverable_id = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="generic",
        brief=_GENERIC_BRIEF,
    )

    with _generic_downstream_mocks():
        run_id, final = await _dispatch_and_wait(
            db_session_factory,
            redis_client,
            deliverable_id=deliverable_id,
            tenant=tenant,
            subject=subject,
        )

    async with db_session_factory() as session:
        final = await RunRepository(session).get(run_id)
        proj_after = await ProjectRepository(session).get(project_id, tenant_id=tenant)
    assert final is not None
    assert final.status == "completed", final.error
    assert proj_after is not None
    assert proj_after.status == "completed"
    assert proj_after.finished_at is not None
    # The execution lease is released once the run finishes.
    assert await redis_client.get(f"runexec:{run_id}") is None


# ---------------------------------------------------------------------------
# Negative path — a downstream failure fails the run (and surfaces the error)
# ---------------------------------------------------------------------------


async def test_generic_dag_fails_when_downstream_errors(db_session_factory, redis_client) -> None:
    """If a downstream stage client raises a non-retriable error, the run must
    reach `failed` with the error recorded — not hang, and not silently complete.

    Regression guard for the class of bug where a status poll or DAG-complete
    path masks a real downstream failure (see HANDOFF 2026-07-14)."""
    from src.database.repositories.run_repository import RunRepository

    tenant, subject = uuid4(), uuid4()
    _project_id, deliverable_id = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="generic",
        brief=_GENERIC_BRIEF,
    )

    boom = AsyncMock()
    boom.message.side_effect = ValueError("synthetic downstream failure")
    with patch(
        "src.services.stages.primitives.script_stage.WMClient",
        return_value=boom,
    ):
        run_id, final = await _dispatch_and_wait(
            db_session_factory,
            redis_client,
            deliverable_id=deliverable_id,
            tenant=tenant,
            subject=subject,
        )

    async with db_session_factory() as session:
        final = await RunRepository(session).get(run_id)
    assert final is not None
    assert final.status == "failed", f"a downstream failure must fail the run, got {final.status!r}"
    assert final.error, "the run must record the downstream error, not swallow it"
    # A failed run also releases its execution lease for a later retry.
    assert await redis_client.get(f"runexec:{run_id}") is None


# ---------------------------------------------------------------------------
# Reconciler / drain-orphan scenarios (relocated from the integration-marked
# tests/services/test_run_dispatcher.py so they gain per-PR CI signal)
# ---------------------------------------------------------------------------


async def test_run_skips_when_lease_held_by_another_instance(
    db_session_factory, redis_client
) -> None:
    """A run whose execution lease is already held by another instance must be
    skipped without touching its row — this is what stops the reconciler from
    double-executing a run still live on a draining pod."""
    from src.database.repositories.run_repository import RunRepository
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher
    from src.services.run_engine import RunEngine

    tenant, subject = uuid4(), uuid4()
    _project_id, deliverable_id = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="generic",
        brief=_GENERIC_BRIEF,
    )

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
    assert await redis_client.get(f"runexec:{run.id}") == b"other-instance"


async def test_dispatcher_repauses_raced_orphan_instead_of_skipping_review(
    db_session_factory, redis_client
) -> None:
    """A drain AFTER a checkpoint committed but BEFORE mark_paused leaves a
    leaseless 'running' orphan: the pause-stage is 'completed', downstream is
    'pending'. The reconciler re-dispatches it; without the dispatch-entry guard
    the executor would seed the completed checkpoint, never re-fire the pause,
    drain downstream, and DAG_COMPLETE would silently complete the run WITHOUT
    the human review. The guard must re-pause it instead."""
    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher
    from src.services.run_engine import RunEngine

    tenant, subject = uuid4(), uuid4()
    _project_id, deliverable_id = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="generic",
        brief=_GENERIC_BRIEF,
    )

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
        pause_after_stages=["script_draft"],
    )

    # Reproduce the race: checkpoint + upstreams 'completed', run row still
    # 'running' (mark_paused never committed), downstream 'pending', no lease.
    async with db_session_factory() as session:
        sor = StageOutputRepository(session)
        for name in ("brief_intake", "source_ingest", "script_draft"):
            await sor.mark_completed(run.id, name, {"ok": True})
        await RunRepository(session).mark_started(run.id)  # status -> running
        await session.commit()

    dispatcher = RunDispatcher(
        session_factory=db_session_factory,
        registry=get_registry(),
        redis=redis_client,
    )
    # No downstream clients patched ON PURPOSE: if the guard did not fire, the
    # executor would run storyboard_draft/narrate against real clients and the
    # run would end 'failed' or 'completed' — never 'paused'.
    await dispatcher._run(run.id)

    async with db_session_factory() as session:
        after = await RunRepository(session).get(run.id)
        stages = {
            s.stage_name: s.status
            for s in await StageOutputRepository(session).list_for_run(run.id)
        }
    assert after is not None
    assert after.status == "paused", (
        f"raced orphan must re-pause for review, not run past it; got {after.status!r}"
    )
    assert stages["storyboard_draft"] == "pending", "downstream must not have executed"
    assert await redis_client.get(f"runexec:{run.id}") is None


async def test_pause_repoints_current_stage_off_a_skipped_first_stage(
    db_session_factory, redis_client
) -> None:
    """When a run pauses, ``current_stage`` must name the checkpoint that tripped
    the gate — never a stage that SKIPPED.

    ``current_stage`` is otherwise only written at stage START, and a wave runs
    its independent stages concurrently, so the last writer can be a no-dep leaf
    that then raised ``StageSkippedError``. Measured live on a demo_video run:
    checkpoint ``demo_video_gen`` completed and the run paused, but
    ``current_stage`` was still ``source_ingest`` — SKIPPED, never executed. The
    FE builds the pause banner text AND the stage-editor href from this field, so
    the reviewer was sent to an empty editor for a stage that never ran.
    """
    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher
    from src.services.run_engine import RunEngine

    tenant, subject = uuid4(), uuid4()
    _project_id, deliverable_id = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="generic",
        brief=_GENERIC_BRIEF,
    )

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
        pause_after_stages=["script_draft"],
    )

    # The defect's exact shape: an early stage SKIPPED but left behind as
    # ``current_stage`` (it was the last to enter 'running'), the checkpoint
    # COMPLETED, the run row still 'running' so the dispatch-entry pause gate
    # fires.
    async with db_session_factory() as session:
        sor = StageOutputRepository(session)
        rr = RunRepository(session)
        await sor.mark_completed(run.id, "brief_intake", {"ok": True})
        await sor.mark_skipped(run.id, "source_ingest", "no source url")
        await sor.mark_completed(run.id, "script_draft", {"ok": True})
        await rr.mark_started(run.id)
        await rr.mark_current_stage(run.id, "source_ingest")
        await session.commit()

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
    assert after.status == "paused"
    assert after.current_stage == "script_draft", (
        "the pause must name the checkpoint the human is asked to review; got "
        f"{after.current_stage!r}"
    )
    assert stages[after.current_stage] == "completed", (
        "a skipped/never-run stage must never be offered as the review target"
    )


# ---------------------------------------------------------------------------
# Blueprint-stage reconciliation at dispatch entry
# ---------------------------------------------------------------------------


async def test_dispatch_seeds_stage_rows_a_blueprint_gained_after_run_creation(
    db_session_factory, redis_client
) -> None:
    """`reconcile_blueprint_stage_rows` must seed, against a REAL database, the
    `stage_outputs` rows a pre-existing run is missing — and must not touch the
    rows it already has.

    The production wedge: rows are inserted exactly once, at run creation. A
    stage ADDED to a blueprint afterwards has no row on any older run, and every
    writer (`mark_running`/`mark_completed`/`mark_skipped`) is a bare `UPDATE`
    that silently touches zero rows. The stage never becomes terminal, so
    `decide_run_terminal_state`'s `declared <= terminal` gate can never close and
    the run sits `running` forever.

    The pre-existing unit coverage (`tests/unit/test_reconcile_blueprint_stage_rows.py`)
    is fully stubbed — it proves the function's arithmetic, never that the INSERT
    reaches Postgres through the dispatcher's tenant-bound session and commit.
    This drives the real `RunDispatcher.dispatch` entry path instead.

    Shape: we reproduce the raced-orphan precondition (checkpoint completed, row
    still `running`) so the crash-safe pause gate returns immediately after
    reconcile. Nothing downstream executes, so a seeded row that is still
    `pending` at the end proves reconcile inserted it — not that some stage ran
    and left it that way. That is also why no downstream clients are patched.
    """
    from sqlalchemy import delete, select

    from src.database.models import StageOutput
    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher
    from src.services.run_engine import RunEngine

    tenant, subject = uuid4(), uuid4()
    _project_id, deliverable_id = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="generic",
        brief=_GENERIC_BRIEF,
    )

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
        pause_after_stages=["brief_intake"],
    )

    declared = set(get_registry().resolve_by_id(run.blueprint_id).topological_order())
    kept_output = {"seeded": True, "note": "must survive reconcile untouched"}

    async with db_session_factory() as session:
        # `brief_intake` is the row the run DID have when it was created: complete
        # it with a real output, then delete every other row. What remains is
        # exactly a run whose blueprint gained the rest of its stages afterwards.
        await StageOutputRepository(session).mark_completed(run.id, "brief_intake", kept_output)
        await session.execute(
            delete(StageOutput).where(
                StageOutput.run_id == run.id, StageOutput.stage_name != "brief_intake"
            )
        )
        await RunRepository(session).mark_started(run.id)  # status -> running
        await session.commit()
        kept_before = (
            await session.execute(
                select(StageOutput).where(
                    StageOutput.run_id == run.id, StageOutput.stage_name == "brief_intake"
                )
            )
        ).scalar_one()
        kept_row_id, kept_completed_at = kept_before.id, kept_before.completed_at

    assert declared - {"brief_intake"}, "the generic blueprint must declare >1 stage for this test"

    dispatcher = RunDispatcher(
        session_factory=db_session_factory,
        registry=get_registry(),
        redis=redis_client,
    )
    await dispatcher._run(run.id)

    async with db_session_factory() as session:
        rows = {r.stage_name: r for r in await StageOutputRepository(session).list_for_run(run.id)}

    # 1. Every stage the blueprint declares now has a row. Without reconcile the
    #    deleted stages would still be absent and the completion gate could never
    #    close — this assertion is what fails if the call is dropped from dispatch.
    assert declared <= set(rows), f"unseeded blueprint stages: {sorted(declared - set(rows))}"

    # 2. The seeded rows are `pending` — reconcile creates work to be done, it
    #    does not fabricate completion. (Nothing ran: the pause gate returned.)
    seeded = {name: rows[name].status for name in declared - {"brief_intake"}}
    assert set(seeded.values()) == {"pending"}, seeded

    # 3. Reconcile is INSERT-ONLY. The pre-existing row must be byte-for-byte the
    #    row it was: same PK (not deleted-and-recreated), same status, same
    #    `output`, same `completed_at`. This is the assertion that stops a future
    #    "cleanup" from deleting an orphan row and destroying a completed stage's
    #    output — which `prior_run_outputs` and `copy_from_parent` both read.
    kept_after = rows["brief_intake"]
    assert kept_after.id == kept_row_id
    assert kept_after.status == "completed"
    assert kept_after.output == kept_output
    assert kept_after.completed_at == kept_completed_at


# ---------------------------------------------------------------------------
# Failure path — a hard-failed run leaves no stranded 'pending'/'running' rows
# ---------------------------------------------------------------------------


def _wm_raises() -> AsyncMock:
    """A fake WMClient whose .message always raises — injects a hard stage failure."""
    fake = AsyncMock()
    fake.message.side_effect = RuntimeError("injected WM failure")
    return fake


async def test_failed_run_sweeps_stranded_stage_rows_to_cancelled(
    db_session_factory, redis_client
) -> None:
    """End-to-end: a generic run whose script stage hard-fails ends `failed` with
    ZERO stage rows left `pending`/`running`.

    The DAG executor cancels only the failing gather batch; every downstream stage
    that was never scheduled would otherwise strand `pending` under the terminal
    `failed` run (never resumed) and pollute the stage_outputs group-by. The
    run-failure site now sweeps them to the terminal `cancelled` state. This drives
    the REAL executor -> dispatcher failure handler -> DB, not just the SQL/mocks.
    """
    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher
    from src.services.run_engine import RunEngine

    tenant, subject = uuid4(), uuid4()
    _project_id, deliverable_id = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="generic",
        brief=_GENERIC_BRIEF,
    )
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
    stack = _generic_downstream_mocks()
    # Override the script-draft WM call so the stage hard-fails mid-DAG, stranding
    # every downstream stage `pending` (never scheduled) under the failed run.
    stack.enter_context(
        patch(
            "src.services.stages.primitives.script_stage.WMClient",
            return_value=_wm_raises(),
        )
    )
    with stack:
        await dispatcher._run(run.id)

    async with db_session_factory() as session:
        final = await RunRepository(session).get(run.id)
        rows = {
            r.stage_name: r.status
            for r in await StageOutputRepository(session).list_for_run(run.id)
        }

    assert final is not None
    assert final.status == "failed", final.error
    # The injected stage is recorded failed.
    assert "failed" in rows.values(), rows
    # NOTHING is left non-terminal — every stranded sibling/downstream row was swept.
    stranded = {n: s for n, s in rows.items() if s in ("pending", "running")}
    assert stranded == {}, f"stranded non-terminal rows under a failed run: {stranded}"
    # And the sweep actually fired (downstream stages -> cancelled), not merely that
    # the DAG happened to terminalize everything on its own.
    assert "cancelled" in rows.values(), rows


# ---------------------------------------------------------------------------
# Failure path — a stuck 'failed' stage row FAILS the run instead of wedging it
# ---------------------------------------------------------------------------


async def test_stuck_failed_stage_row_fails_run_not_wedged(
    db_session_factory, redis_client
) -> None:
    """A stage row left `failed` while the rest of the DAG drives to terminal must
    FAIL the run — not leave it CONTINUE/`running` for the ~2h max-age backstop.

    This is the resumable-timeout-drain transient-failure case, reproduced end to
    end: a `failed` `script_draft` row is seeded, then the run is dispatched with
    the normal generic mocks so `script_draft` RE-RUNS and succeeds *in memory*.
    Its success write (`mark_completed`) is a guarded no-op on a terminal row, so
    the DB row stays `failed` even though the executor returns normally with the
    whole DAG complete in memory. Before the finalizer's FAIL direction the
    completion gate returned CONTINUE here and the run wedged `running` forever;
    now it must terminalize `failed`. Drives the REAL executor -> dispatcher ->
    finalizer -> DB path.
    """
    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher
    from src.services.run_engine import RunEngine

    tenant, subject = uuid4(), uuid4()
    _project_id, deliverable_id = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="generic",
        brief=_GENERIC_BRIEF,
    )
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

    # Seed a stuck-terminal `failed` row for a spine stage while leaving the run
    # `pending` — so the normal claim+reconcile dispatch path still fires (marking
    # it `running` instead would skip reconcile and strand opt-in rows). The
    # guarded `mark_failed` (pending -> failed) is the write-once terminal write,
    # so when `script_draft` RE-RUNS under dispatch and succeeds in memory, its
    # `mark_completed` is a no-op and the DB row stays `failed`.
    async with db_session_factory() as session:
        seeded = await StageOutputRepository(session).mark_failed(
            run.id, "script_draft", "ValueError: transient drain blip"
        )
        await session.commit()
    assert seeded, "seed precondition: script_draft stage row must exist and be pending"

    dispatcher = RunDispatcher(
        session_factory=db_session_factory,
        registry=get_registry(),
        redis=redis_client,
    )
    with _generic_downstream_mocks():
        await dispatcher._run(run.id)

    async with db_session_factory() as session:
        final = await RunRepository(session).get(run.id)
        rows = {
            r.stage_name: r.status
            for r in await StageOutputRepository(session).list_for_run(run.id)
        }

    assert final is not None
    # The run terminalizes `failed`, NOT `running` (the pre-fix wedge).
    assert final.status == "failed", f"stuck failed row must fail the run, got {final.status!r}"
    # The stuck row is still `failed` (its in-memory re-run success was a no-op),
    # and that is what drove the FAIL verdict + the recorded reason.
    assert rows.get("script_draft") == "failed", rows
    assert final.error and "stage_failed_post_drain" in final.error, final.error
    # No row is left non-terminal (the fail path swept any leftovers).
    stranded = {n: s for n, s in rows.items() if s in ("pending", "running")}
    assert stranded == {}, f"stranded non-terminal rows under a failed run: {stranded}"
    assert await redis_client.get(f"runexec:{run.id}") is None


@pytest.mark.parametrize(
    ("terminalize", "expected"),
    [("mark_cancelled", "cancelled")],
)
async def test_dispatch_abandons_a_run_that_left_the_startable_set(
    db_session_factory, redis_client, terminalize, expected
) -> None:
    """The reconciler selects an orphan, the user cancels (or pauses) it, and only
    THEN does the dispatch reach ``mark_started``. That write used to be a bare
    UPDATE: it flipped the row back to `running` and the run billed to completion
    — or, on a paused row, ran straight past the human-review gate.

    No downstream clients are patched ON PURPOSE: if the guard did not fire, the
    executor would run the DAG against real clients.
    """
    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher
    from src.services.run_engine import RunEngine

    tenant, subject = uuid4(), uuid4()
    _project_id, deliverable_id = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="generic",
        brief=_GENERIC_BRIEF,
    )
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

    async with db_session_factory() as session:
        kwargs = {"reason": "manual"} if terminalize == "mark_paused" else {}
        assert await getattr(RunRepository(session), terminalize)(run.id, **kwargs) is True
        await session.commit()

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
    assert after.status == expected, f"dispatch resurrected a {expected!r} run as {after.status!r}"
    # NOT-EXECUTED looks different per terminalization. A `cancelled` run is
    # TERMINAL, so `RunRepository.mark_cancelled` sweeps its never-run stage rows
    # to `cancelled` — leaving them `pending` would claim they are still going to
    # run. A `paused` run is RESUMABLE, so its rows stay `pending`, which is true.
    # Either way `running`/`completed` would mean the DAG executed anyway.
    not_executed = {"cancelled"} if expected == "cancelled" else {"pending"}
    assert set(stages.values()) <= not_executed, f"the DAG executed anyway: {stages}"
