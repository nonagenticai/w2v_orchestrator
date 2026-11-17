"""In-process full-DAG dispatch integration test for the `website_review` archetype.

Mirrors `test_in_process_dispatch.py` (generic/promo): a real run is driven
through the in-process dispatcher to a terminal status with every downstream
client mocked at its stage-module import site (fakeredis + CI Postgres, no live
service). See that module's header for why these are NOT `@pytest.mark.integration`
and therefore DO run in default per-PR CI.

The website_review DAG has two SUBMIT→POLL clients that the generic/promo paths
lack — CritiqueClient (critique.py) and DesignClient (redesign.py) — so the mocks
below must both terminate the poll AND return the exact payload keys each stage
reads off the result, or the run false-passes / KeyErrors. Each non-obvious mock
carries an inline comment naming the stage:line that consumes it.
"""

from __future__ import annotations

import io
import zipfile
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from src.clients.schemas.brand_extractor import BrandExtraction
from src.clients.schemas.critique import CritiqueResult
from src.clients.schemas.design import DesignJobResult, DesignJobStatus, DesignVisualAsset
from tests.integration._dispatch_harness import (
    _compile_mock,
    _completed_stage_names,
    _create_project,
    _dispatch_and_wait,
    _slide_service_mock,
    _tts_mock,
    _wm_client,
)


def _critique_mock() -> AsyncMock:
    """Fake CritiqueClient: submit → job_id, poll → the critique result.

    ``wait_for_critique`` must return the keys website_critique.handle reads:
    ``findings`` (each: finding_id/category/severity/title/description/
    recommendation — critique.py:110-120), ``url`` (:123), ``overall_score``
    (:124), ``summary`` (:125), ``category_scores`` (:126). A wrong/empty shape
    silently drops the findings that steer the redesign + script.
    """
    fake = AsyncMock()
    fake.start_critique.return_value = "critique-job-1"  # critique.py:83/92 (submit → job_id)
    fake.wait_for_critique.return_value = CritiqueResult.model_validate(
        {
            "job_id": "critique-job-1",
            "status": "completed",
            "url": "https://example.test",
            "overall_score": 62.0,
            "summary": "Cluttered hero, weak contrast.",
            "category_scores": {"accessibility": 55.0, "clarity": 60.0},
            "findings": [  # critique.py:119 iterates result.findings
                {
                    "finding_id": "f1",
                    "category": "accessibility",
                    "severity": "high",
                    "title": "Low contrast CTA",
                    "description": "The primary CTA fails WCAG AA contrast.",
                    "recommendation": "Darken the button text.",
                }
            ],
        }
    )
    return fake


def _design_mock() -> AsyncMock:
    """Fake DesignClient: submit → job_id, poll → status, result → the redesign.

    - ``wait_for_design`` → ``DesignJobStatus``; the stage reads
      ``accessibility_score`` off it.
    - ``get_design_result`` MUST return >=1 fetchable http(s) visual or
      website_redesign RAISES StageError ("no fetchable visuals").
      ``_assets_to_visuals`` reads visual_assets[*].{asset_id, title|kind, role,
      url}; the stage also reads ``url``/``style``/``redesigned_html_path``.
    """
    fake = AsyncMock()
    fake.start_design.return_value = "design-job-1"  # redesign.py (submit → job_id)
    fake.wait_for_design.return_value = DesignJobStatus(
        job_id="design-job-1", status="completed", accessibility_score=88.0
    )
    # website_review_script's grounding inventory reads the rendered HTML.
    fake.get_design_html.return_value = "<html><body><main><h1>Redesign</h1></main></body></html>"
    fake.get_design_result.return_value = DesignJobResult(
        job_id="design-job-1",
        url="https://example.test",
        scope="redesign",
        style="modern",
        redesigned_html_path="/artifacts/redesign.html",
        visual_assets=[
            DesignVisualAsset(
                asset_id="a1",
                title="Redesigned homepage",
                kind="screenshot",
                # _bucket → "after" pool (compile.py:78); fetchable → visuals non-empty
                role="after",
                url="https://example.test/redesign-after.png",
                source="design",
            )
        ],
    )
    return fake


def _site_capture_recorder_mock() -> AsyncMock:
    """Fake VideoProductionClient for site_capture's per-screen pass.

    A website_review brief names its site ``target_url``; site_capture now captures
    it with the default ``live-top`` / ``live-bottom`` screens, so the recorder
    answers with a zip of one PNG per capture id.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("live-top.png", b"\x89PNG top")
        zf.writestr("live-bottom.png", b"\x89PNG bottom")
    fake = AsyncMock()
    fake.capture_screens.return_value = "rec-job-1"
    fake.wait_for_content_recording.return_value = None
    fake.get_recording_video.return_value = (buf.getvalue(), "application/zip", "")
    return fake


def _brand_extractor_mock() -> AsyncMock:
    """Fake BrandExtractorClient so website_redesign's best-effort
    ``resolve_brand_context(url)`` (redesign.py:169 → brand_context.py:91) does NOT
    make a real network call. An empty dict maps to (None, None) — the exact
    fall-back the stage already tolerates — keeping the test hermetic + deterministic.
    """
    fake = AsyncMock()
    # Empty extraction → server defaults → _map_brand → (None, None), the exact fall-back.
    fake.extract_brand.return_value = BrandExtraction()
    return fake


def _website_review_downstream_mocks() -> ExitStack:
    """Every downstream client a plain website_review run touches.

    Spine (all mocked here): critique → redesign → script → record →
    visual_director → slides → compile → growth_bundle, plus site_capture, which
    captures the brief's ``target_url`` on a website_review. The two remaining
    opt-in leaves (source_ingest / media_discovery) skip clean on a brief with
    only ``target_url`` (no sources / media_discovery), so they need no mock. VideoProductionClient (redesign + slides) is patched defensively —
    the motion-hero (brief.redesign_motion off) and slide-deck (no slide shots)
    paths that would call it never fire on this brief.
    """
    stack = ExitStack()
    p = lambda target, **kw: stack.enter_context(patch(target, **kw))  # noqa: E731

    # --- SUBMIT→POLL pair #1: analysis_service critique (website_critique).
    p("src.services.stages.website_review.critique.CritiqueClient", return_value=_critique_mock())
    # --- SUBMIT→POLL pair #2: html_render redesign (website_redesign).
    p("src.services.stages.website_review.redesign.DesignClient", return_value=_design_mock())
    p(
        "src.services.stages.website_review.redesign.VideoProductionClient",
        return_value=AsyncMock(),  # motion-hero only (brief.redesign_motion off) — never called
    )
    # website_review_script reads the redesign HTML for its grounding inventory.
    p(
        "src.services.stages.website_review.script.DesignClient",
        return_value=_design_mock(),
    )
    # --- site_capture: per-screen capture of the brief's target_url.
    p(
        "src.services.stages.generic.site_capture.VideoProductionClient",
        return_value=_site_capture_recorder_mock(),
    )
    p(
        "src.services.stages.generic.site_capture.store_capture_image",
        new=AsyncMock(side_effect=lambda **kw: f"http://a/{kw['image_id']}.png"),
    )
    # website_redesign's best-effort brand extraction — keep it off the network.
    p(
        "src.services.stages.brand_context.BrandExtractorClient",
        return_value=_brand_extractor_mock(),
    )
    # --- website_review_script: LLM draft via the shared script primitive. Two
    # lines with a redesign reveal so _assign_phases splits into a before→after arc
    # (script.py:308 _find_redesign_pivot matches "here is ... new redesign").
    p(
        "src.services.stages.primitives.script_stage.WMClient",
        return_value=_wm_client(
            {
                "lines": [
                    {
                        "text": "This homepage is cluttered and the CTA is hard to read.",
                        "seconds": 5.0,
                        "phase": "before",
                        "speaker": "a",
                    },
                    {
                        "text": "Here is the new redesign that fixes the contrast.",
                        "seconds": 5.0,
                        "phase": "after",
                        "speaker": "b",
                    },
                ]
            }
        ),
    )
    # --- website_review_visual_director: WMClient constructed in __init__. On this
    # brief the redesign carries no `redesign_section:*` assets, so the stage
    # short-circuits BEFORE the LLM call (visual_director.py:122) and completes with
    # an empty mapping — this patch only stops a real WMClient being built.
    p(
        "src.services.stages.website_review.visual_director.WMClient",
        return_value=_wm_client({"assignments": []}),
    )
    # --- website_review_slide_plan: no `slide`-kind shots in the brief, so this
    # stage's LLM authors the BEFORE-act deck from the critique + script lines
    # (slide_plan.py). Mocked at the stage's own import site like every other
    # WMClient call on this spine. Real script line ids are `uuid7()` (assigned
    # at run time — script_stage.py:248), unknowable here, so an empty
    # `{"slides": []}` reply is the only id-independent way to mock this: it is
    # a structurally valid LLM reply that `_slides_from_reply` reads as "named no
    # usable slide" (slide_plan.py:261), which deterministically exercises the
    # documented `critique_fallback` path (slide_plan.py:326-337) — the same
    # deck-authoring path the real CI failure went through when the (then
    # unmocked) WMClient call raised. `website_review_slides` below renders that
    # fallback deck, which is the actual defect this test now covers: it is no
    # longer a dead code path (see module docstring).
    p(
        "src.services.stages.website_review.slide_plan.WMClient",
        return_value=_wm_client({"slides": []}),
    )
    # --- website_review_slides: renders the deck `website_review_slide_plan`
    # authored above. SlideServiceClient is faked; the still-image write goes
    # through `slide_deck.store_slide_image` / `find_existing_media`, which are
    # patched at their shared import site (same idiom as promo/lesson in
    # `test_in_process_dispatch.py`) so this hermetic test never touches real
    # object storage.
    p(
        "src.services.stages.website_review.slides.SlideServiceClient",
        return_value=_slide_service_mock(),
    )
    p(
        "src.services.stages.website_review.slides.VideoProductionClient",
        return_value=AsyncMock(),  # slide-deck motion path — motion_graphics off by default
    )
    p(
        "src.services.stages.shared.slide_deck.find_existing_media",
        new=AsyncMock(return_value=None),
    )
    p(
        "src.services.stages.shared.slide_deck.store_slide_image",
        new=AsyncMock(return_value="http://a/slide.png"),
    )
    # --- website_review_record: TTS every script line via the narration primitive.
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
    # --- website_review_compile: video-compile via the compile primitive.
    p(
        "src.services.stages.primitives.compile_stage.VideoCompileClient",
        return_value=_compile_mock(),
    )
    p(
        "src.services.stages.primitives.compile_stage.TranscriptionWorkerClient",
        return_value=AsyncMock(),  # word-timed captions only (captions_mode="none") — never built
    )
    # --- growth_bundle: outbound-copy LLM (same return shape as the promo test).
    p(
        "src.services.stages.growth_bundle.handler.WMClient",
        return_value=_wm_client({"linkedin": {"text": "t", "cta_url": None}}),
    )
    return stack


# Only `target_url` — deliberately no `sources` (source_ingest) or
# `media_discovery` (media_discovery), so those two opt-in leaves skip clean.
# site_capture captures `target_url` on a website_review (no `source_url` needed).
_WEBSITE_REVIEW_BRIEF = {
    "target_url": "https://example.test",
    "duration_seconds": 30,
    "audience": "saas founders",
}


# The non-opt-in "spine": the stages a plain website_review run MUST complete.
# The two opt-in leaves (source_ingest / media_discovery) reach terminal status
# "skipped", not "completed", so they are excluded here — an `==` assertion
# therefore proves the spine ran AND that no leaf silently ran/failed.
_WEBSITE_REVIEW_SPINE = {
    "site_capture",
    "website_critique",
    "website_redesign",
    "website_review_script",
    "website_review_record",
    "website_review_visual_director",
    "website_review_slide_plan",
    "website_review_slides",
    "website_review_compile",
    "growth_bundle",
}


async def test_website_review_dag_runs_to_completed(db_session_factory, redis_client) -> None:
    """A plain website_review run executes its whole DAG to `completed` with the
    critique / redesign / WM / TTS / slide / compile / site-capture surface mocked
    and the two opt-in ingest leaves skipping clean.

    The DAG-spine assertion is the anti-false-green guard: a wrong patch target
    silently no-ops and a bare `status == "completed"` can still pass if the DAG
    short-circuited, so we assert the exact set of stages that actually `completed`.
    """
    from src.database.repositories.project_repository import ProjectRepository
    from src.database.repositories.run_repository import RunRepository

    tenant, subject = uuid4(), uuid4()
    project_id, deliverable_id = await _create_project(
        db_session_factory,
        tenant=tenant,
        subject=subject,
        type_="website_review",
        brief=_WEBSITE_REVIEW_BRIEF,
    )

    with _website_review_downstream_mocks():
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

    completed = await _completed_stage_names(db_session_factory, run_id)
    assert completed == _WEBSITE_REVIEW_SPINE, (
        "website_review spine mismatch — a mispatched client no-opped or an opt-in "
        f"leaf ran instead of skipping: {completed ^ _WEBSITE_REVIEW_SPINE}"
    )
