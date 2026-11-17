"""In-process full-DAG dispatch integration test for the `app_review` archetype.

Sibling of `test_in_process_dispatch.py` (generic/promo) — see that module's
header for WHY these are NOT `@pytest.mark.integration` (they run in default
per-PR CI: postgres + fakeredis only, no live redis / downstream service).

`app_review` is DORMANT (parked 2026-06-03, hidden from the frontend) but still
REGISTERED and fully driveable through the dispatcher, so it is worth the same
anti-false-green DAG coverage as the live archetypes. Its DAG is:

    app_inspect → app_critique → app_review_script → app_review_record
                → app_review_compile → growth_bundle

Every stage here has a downstream client; none is an opt-in skip-clean leaf, so
the completed-stage spine is the *entire* stage list (asserted `==` below). The
one client unique to this archetype is `AppInspectorClient` (a submit→poll
client): the stage only ever calls its `inspect_app(url)` convenience wrapper
(which internally is `start_inspection`→`wait_for_inspection`, i.e.
`submit_job`→`poll_job`), so patching the class at the stage's import site and
stubbing `inspect_app` short-circuits the whole submit/poll dance at that seam.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from src.clients.schemas.app_inspector import (
    InspectionResult,
    NavigationEdge,
    ScreenSummary,
)
from tests.integration._dispatch_harness import (
    _compile_mock,
    _completed_stage_names,
    _create_project,
    _dispatch_and_wait,
    _tts_mock,
    _wm_client,
)

# ---------------------------------------------------------------------------
# Archetype-specific downstream mocks + brief
# ---------------------------------------------------------------------------

# The non-opt-in "spine": every stage in the app_review blueprint. None of them
# skip-clean on a config-less run, so a correct wiring completes ALL of them.
# A wrong patch target silently no-ops → the stage fails/pauses → the set breaks,
# which is exactly the false-green this asserts against.
_APP_REVIEW_SPINE = {
    "app_inspect",
    "app_critique",
    "app_review_script",
    "app_review_record",
    "app_review_compile",
    "growth_bundle",
}


def _app_inspector_mock() -> AsyncMock:
    """A fake AppInspectorClient. The app_inspect stage calls only
    `inspect_app(url)`, which on the real client is submit→poll
    (`start_inspection`→`wait_for_inspection`); patching the class replaces that
    whole path, so we only stub the wrapper. It returns a typed `InspectionResult`
    matching the server's `/api/inspect/{job_id}/result` (`InspectionResultResponse`),
    and the stage's extractor derives purpose/screen/accessibility/navigation/
    screenshot facts off that model. The first screen carries a URL-shaped
    `screenshot_path` so the extractor emits a screenshot fact + `screenshot_url`
    for the downstream `app_review_record`/`app_review_compile` visual path."""
    fake = AsyncMock()
    fake.inspect_app.return_value = InspectionResult(
        job_id="insp-1",
        status="completed",
        platform="android",
        app_path="com.acme.app",
        estimated_purpose="A productivity app",
        screens_discovered=2,
        overall_accessibility_score=90,
        screens=[
            ScreenSummary(
                screen_id="home",
                screen_type="home",
                headings=["Welcome", "Today"],
                text_content="Your tasks for today at a glance.",
                accessibility_score=92,
                screenshot_path="http://a/shot.png",
            ),
            ScreenSummary(
                screen_id="settings",
                screen_type="settings",
                headings=["Settings", "Account"],
                text_content="Manage your account and notifications.",
                accessibility_score=88,
            ),
        ],
        navigation_graph=[NavigationEdge(from_screen="home", to_screen="settings")],
    )
    return fake


def _app_review_downstream_mocks() -> ExitStack:
    """Downstream client mocks for a plain `app_review` run.

    Every stage has a client and none skip-clean, so the full WM / inspector /
    TTS / compile surface is mocked here.
    """
    stack = ExitStack()
    p = lambda target, **kw: stack.enter_context(patch(target, **kw))  # noqa: E731

    # app_inspect: submit→poll AppInspectorClient, patched at its OWN stage
    # import site (inspect.py:20). Stubs inspect_app → typed inspection facts.
    p(
        "src.services.stages.app_review.inspect.AppInspectorClient",
        return_value=_app_inspector_mock(),
    )
    # app_critique: WMClient.message → pros/cons/recommendations/overall_take
    # (critique.py:104-118 reads exactly these keys off resp.output).
    p(
        "src.services.stages.app_review.critique.WMClient",
        return_value=_wm_client(
            {
                "pros": ["fast onboarding"],
                "cons": ["sparse docs"],
                "recommendations": ["add a tour"],
                "overall_take": "solid",
            }
        ),
    )
    # app_review_script: LLMScriptStageBase → the shared script primitive's
    # WMClient (script.py subclasses primitives.script_stage). Element schema
    # requires text+seconds (script.py:36-45); make_element reads text/seconds.
    p(
        "src.services.stages.primitives.script_stage.WMClient",
        return_value=_wm_client({"lines": [{"text": "hi", "seconds": 5.0}]}),
    )
    # app_review_record: ⚠ app_review imports TTSClient at its OWN module site
    # (record.py:15 `from src.clients.tts_client import TTSClient`; instantiated
    # record.py:55-56), NOT via primitives.narration — so patch THIS path, not
    # `primitives.narration.TTSClient`. The client's `synthesize` is what
    # `synthesize_and_store` calls.
    p("src.services.stages.app_review.record.TTSClient", return_value=_tts_mock())
    # ...but the clip PERSISTENCE (`store_tts_clip`) is called from inside
    # `synthesize_and_store`, which lives in primitives.narration (narration.py:67)
    # — so it is patched at the narration module site regardless of the caller.
    p(
        "src.services.stages.primitives.narration.store_tts_clip",
        new=AsyncMock(return_value="http://a/clip.wav"),
    )
    # Seal the pre-store clip-reuse probe (real StorageService init otherwise).
    p(
        "src.services.stages.primitives.narration.find_existing_clip",
        new=AsyncMock(return_value=None),
    )
    # app_review_compile: CompileStageBase → VideoCompileClient.compile
    # (compile_stage.py:325). TranscriptionWorkerClient is NOT needed: the
    # word-timed-captions path is gated on captions_mode=="burned_in" (the test
    # deliverable is "none") AND AppReviewCompileStage does not override
    # caption_clip_refs, so no ASR client is constructed.
    p(
        "src.services.stages.primitives.compile_stage.VideoCompileClient",
        return_value=_compile_mock(),
    )
    # growth_bundle: WMClient.message → one `{text, cta_url}` block per channel key
    # (handler._parse_snippets). Same return shape as promo.
    p(
        "src.services.stages.growth_bundle.handler.WMClient",
        return_value=_wm_client({"linkedin": {"text": "t", "cta_url": None}}),
    )
    return stack


# app_inspect hard-requires `brief.target_url` (inspect.py:111-113). The remaining
# keys are optional prompt inputs read by critique/script/growth_bundle.
_APP_REVIEW_BRIEF = {
    "target_url": "https://acme.example/app",
    "review_lens": "onboarding",
    "audience": "founders",
    "duration_seconds": 30,
    "call_to_action": "Try Acme",
}


async def test_app_review_dag_runs_to_completed(db_session_factory, redis_client) -> None:
    """A plain app_review run executes its whole DAG to `completed` with the
    inspector / WM / TTS / compile surface mocked, and the terminal state rolls
    up to the owning project. The completed-stage set must equal the full
    non-opt-in spine — the anti-false-green guard."""
    from src.database.repositories.project_repository import ProjectRepository
    from src.database.repositories.run_repository import RunRepository

    tenant, subject = uuid4(), uuid4()
    project_id, deliverable_id = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="app_review",
        brief=_APP_REVIEW_BRIEF,
    )

    with _app_review_downstream_mocks():
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

    # Anti-false-green: prove the DAG actually executed every non-opt-in stage,
    # not that a short-circuit reached a green terminal. `==` because none of
    # app_review's stages skip-clean on a config-less run.
    completed = await _completed_stage_names(db_session_factory, run_id)
    assert completed == _APP_REVIEW_SPINE, (
        f"completed spine mismatch: missing {_APP_REVIEW_SPINE - completed}, "
        f"unexpected {completed - _APP_REVIEW_SPINE}"
    )
