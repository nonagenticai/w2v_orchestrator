"""In-process full-DAG dispatch integration test — the `lesson` archetype.

Sibling of `test_in_process_dispatch.py` (generic/promo): drives a real lesson
run through the in-process dispatcher to a terminal status, every downstream
client mocked at its stage-module import site (fakeredis + CI Postgres, no live
service). See that module's header for why these run in default per-PR CI.

The lesson spine (non-opt-in stages) is
`lesson_script → lesson_slides → (lesson_narrate ‖ ai_image_gen) → lesson_record
→ lesson_compile`. `ai_image_gen` is DEFAULT-ENABLED for lessons
(`run_setup.build_config_snapshot` seeds `ai_image_gen.prompts_from_stage =
"lesson_slides"`), so it is on the spine — the ACG submit→poll client is the #1
mock-shape risk here. The opt-in leaves `source_ingest` / `media_discovery` /
`ai_video_gen` skip clean (recorded status "skipped", NOT "completed"), so they
are absent from `_completed_stage_names` and the spine assertion uses `==`.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from tests.integration._dispatch_harness import (
    _compile_mock,
    _completed_stage_names,
    _create_project,
    _dispatch_and_wait,
    _slide_service_mock,
    _tts_mock,
    _wm_client,
)

# ---------------------------------------------------------------------------
# Archetype-specific downstream mocks + brief
# ---------------------------------------------------------------------------

# The exact set of stages a plain lesson run marks "completed". Opt-in leaves
# (source_ingest / media_discovery / ai_video_gen) are "skipped", never
# "completed", so `==` (not `<=`) is correct — a wrong patch target that
# no-ops a spine stage would drop it from this set and fail the assertion.
_LESSON_SPINE = {
    "lesson_script",
    "lesson_slides",
    "lesson_narrate",
    "ai_image_gen",
    "lesson_record",
    "lesson_compile",
}


def _acg_image_mock() -> AsyncMock:
    """Fake AIContentGenerationClient for `ai_image_gen` (submit→poll→result).

    ai_image_gen.py: `create_concept_image` → job_id, `wait_for_concept_image`
    → an `AcgConceptImageJob` (NOT a dict — `_to_generated_image` reads
    `result.asset_id`/`.image_url`/`.download_url`/`.provider`/`.status` by
    ATTRIBUTE). Returning an ABSOLUTE http url makes `_republish_relative` pass
    the image through WITHOUT the download/re-store path (that fires only for
    ACG-relative "/..." paths), and clears the no-usable-url guard."""
    from src.clients.schemas.ai_content_generation import AcgConceptImageJob

    fake = AsyncMock()
    fake.create_concept_image.return_value = "img-job-1"
    fake.wait_for_concept_image.return_value = AcgConceptImageJob(
        job_id="img-job-1",
        status="completed",
        asset_id="asset-1",
        image_url="http://a/img.png",
        provider="test",
    )
    return fake


def _lesson_downstream_mocks() -> ExitStack:
    """Every downstream client a plain lesson run touches, patched at its
    stage-module import site. No source_url / no ai_video_gen / no motion, so
    every opt-in leaf skips clean and needs no mock."""
    stack = ExitStack()
    p = lambda target, **kw: stack.enter_context(patch(target, **kw))  # noqa: E731

    # lesson_script (LLMScriptStageBase): WM returns the `paragraphs` element
    # list — script_stage.py reads data[element_key] (line 163) where
    # element_key="paragraphs"; make_element reads raw["text"]/raw["seconds"]
    # (lesson/script.py:175-179). NOT the generic "lines" shape.
    p(
        "src.services.stages.primitives.script_stage.WMClient",
        return_value=_wm_client({"paragraphs": [{"text": "hi", "seconds": 5.0, "speaker": "b"}]}),
    )
    # lesson_narrate (NarrationStageBase): TTS synth + storage write.
    # narration.py: TTSClient.synthesize (line 65) + store_tts_clip (line 67).
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
    # ai_image_gen: ACG submit→poll client + the re-publish storage write.
    # store_generated_image (ai_image_gen.py:186) is only reached for relative
    # urls — inert here (absolute url), patched to keep the import site sealed.
    p(
        "src.services.stages.generic.ai_image_gen.AIContentGenerationClient",
        return_value=_acg_image_mock(),
    )
    p(
        "src.services.stages.generic.ai_image_gen.store_generated_image",
        new=AsyncMock(return_value="http://a/img.png"),
    )
    # lesson_record: SlideServiceClient (resolve_template_id → start_deck →
    # wait_for_deck → get_slide_png, lesson/record.py:87 + shared/slide_deck.py:
    # 153-161) + VideoProductionClient (only used by the opt-in motion path,
    # inert without a `motion_graphics` block — constructed at __init__ so it is
    # sealed regardless).
    p("src.services.stages.lesson.record.SlideServiceClient", return_value=_slide_service_mock())
    p("src.services.stages.lesson.record.VideoProductionClient", return_value=AsyncMock())
    # render_deck_slide_images (shared/slide_deck.py:163) persists each rendered
    # PNG via store_slide_image → a real local-FS write (unlike generic/promo,
    # which never render a deck to images). Seal it like the other storage
    # writes so the test stays hermetic; it returns the stored slide URL.
    p(
        "src.services.stages.shared.slide_deck.store_slide_image",
        new=AsyncMock(return_value="http://a/slide.png"),
    )
    # lesson_compile (CompileStageBase): VideoCompileClient.compile →
    # {artifact_url, duration_seconds} (compile_stage.py:325, read by
    # lesson/compile.py:275/284) + TranscriptionWorkerClient (word-timed
    # captions only, inert for captions_mode="none" — sealed anyway).
    p(
        "src.services.stages.primitives.compile_stage.VideoCompileClient",
        return_value=_compile_mock(),
    )
    p(
        "src.services.stages.primitives.compile_stage.TranscriptionWorkerClient",
        return_value=AsyncMock(),
    )
    return stack


_LESSON_BRIEF = {
    "learning_objective": "Understand recursion",
    "duration_minutes": 2,
    "target_duration_seconds": 30,
}

# Minimal parent course — only needed to satisfy the lesson's parent_project_id
# invariant; this project is never dispatched.
_COURSE_PARENT_BRIEF = {"topic": "Recursion", "target_duration_seconds": 30}


# ---------------------------------------------------------------------------
# Happy path — full DAG to completed
# ---------------------------------------------------------------------------


async def test_lesson_dag_runs_to_completed(db_session_factory, redis_client) -> None:
    """A plain lesson run executes its whole DAG to `completed`, the terminal
    state rolls up to the owning project, and the anti-false-green DAG-spine
    assertion holds (the completed-stage set is EXACTLY the non-opt-in spine)."""
    from src.database.repositories.project_repository import ProjectRepository
    from src.database.repositories.run_repository import RunRepository

    tenant, subject = uuid4(), uuid4()
    # A lesson must nest under a course parent (ProjectRepository invariant); the
    # course itself is never run — it only satisfies the parent_project_id FK.
    course_id, _ = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="course",
        brief=_COURSE_PARENT_BRIEF,
    )
    project_id, deliverable_id = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="lesson",
        brief=_LESSON_BRIEF,
        parent_project_id=course_id,
    )

    with _lesson_downstream_mocks():
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
    # Anti-false-green: prove the DAG actually executed every spine stage. Skip-
    # clean leaves are recorded "skipped" (never "completed"), so `==` is exact.
    completed = await _completed_stage_names(db_session_factory, run_id)
    assert completed == _LESSON_SPINE, f"unexpected completed set: {completed ^ _LESSON_SPINE}"
    # The execution lease is released once the run finishes.
    assert await redis_client.get(f"runexec:{run_id}") is None


# ---------------------------------------------------------------------------
# Negative path — a downstream failure fails the run (and surfaces the error)
# ---------------------------------------------------------------------------


async def test_lesson_dag_fails_when_downstream_errors(db_session_factory, redis_client) -> None:
    """If the first downstream stage client (lesson_script's WM worker) keeps
    raising, the run must reach `failed` with the error recorded — not hang, and
    not silently complete. (ValueError is retryable per llm_budget, so the WM
    stage retries and then exhausts before the run fails.) Mirrors
    `test_generic_dag_fails_when_downstream_errors`."""
    from src.database.repositories.run_repository import RunRepository

    tenant, subject = uuid4(), uuid4()
    course_id, _ = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="course",
        brief=_COURSE_PARENT_BRIEF,
    )
    _project_id, deliverable_id = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="lesson",
        brief=_LESSON_BRIEF,
        parent_project_id=course_id,
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
