"""web_design — generate a brand-new HTML page from a text brief.

Drives html_render_service's design engine (the same engine behind website_review's
redesign) in greenfield mode: ``scope="landing"`` with no source URL designs a page
FROM the brief; an optional ``brief.target_url`` switches to the refine path so an
existing site can seed the look. Surfaces the generated HTML + a representative
screenshot; the run-view offers HTML / screenshot / PDF downloads (the PDF is minted
on demand by html_render at ``/api/designs/jobs/{id}/pdf``).

The output deliberately mirrors ``WebsiteRedesignOutput``'s field names
(``redesign_job_id`` / ``redesigned_html_path`` / ``visuals``) so the frontend's
polymorphic run-view renders it with the existing redesign viewer + download wiring.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from w2v_common.stages import StageTimeoutError, stage_call

from src.clients.design_client import DesignClient
from src.clients.video_production_client import VideoProductionClient
from src.services.stages.base import StageError, StageHandler, StageInput, StageOutput
from src.services.stages.brand_context import resolve_brand_context
from src.services.stages.llm_budget import LLM_RETRY_ON
from src.services.stages.shared.design_job_stash import (
    SessionFactory,
    peek_status,
    reattachable_job_id,
    stash_job_id,
)
from src.services.stages.shared.design_queue import await_design_start
from src.services.stages.shared.motion_slides import quality_tier
from src.services.stages.shared.redesign_motion import attach_motion_hero
from src.services.stages.upstream import guarded
from src.services.stages.website_review.redesign import RedesignVisual, WebsiteRedesignStage
from src.services.tenant_defaults import authored_brand, brief_brand, fill_tenant_brand

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

_START_TIMEOUT_SECONDS = 60.0
_POLL_TIMEOUT_SECONDS = 590.0
_WAIT_RETRIES = 3
_RESULT_TIMEOUT_SECONDS = 60.0


class WebDesignInput(StageInput):
    pass


class WebDesignOutput(StageOutput):
    # Field names mirror WebsiteRedesignOutput so the frontend run-view renders this
    # with the existing redesign viewer (visuals + Download HTML/PDF keyed on the id).
    redesign_job_id: str
    url: str = ""
    style: str
    redesigned_html_path: str | None = None
    # The orchestrator redesign proxy path (mounted in prod + dev), which forwards to
    # html_render — the only SERVED ref for the generated HTML. `redesigned_html_path`
    # is an internal MinIO key no consumer can fetch, so without this the brand-artifact
    # critique never reads the page and `critique_score` stays NULL. Same shape as
    # web_design_v2 (`web_design_v2/design.py`).
    live_html_url: str
    accessibility_score: float | None = None
    visuals: list[RedesignVisual] = []  # noqa: RUF012 — pydantic copies field defaults per-instance


class WebDesignStage(StageHandler[WebDesignInput, WebDesignOutput]):
    name = "web_design"
    input_model = WebDesignInput
    output_model = WebDesignOutput

    def __init__(
        self,
        client: DesignClient | None = None,
        video_client: VideoProductionClient | None = None,
        *,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self._client = client or DesignClient()
        self._video_client = video_client or VideoProductionClient()
        # Optional on purpose: without it the stage still runs end-to-end, it
        # just loses in-flight-job re-attachment across a resume (i.e. the old
        # orphaning behaviour). The blueprint passes the dispatcher's factory.
        self._sessions = session_factory

    async def handle(self, payload: WebDesignInput, ctx: RunContext) -> WebDesignOutput:
        brief = ctx.config_snapshot.get("brief") or {}
        description = str(brief.get("description") or brief.get("angle") or "").strip()
        if not description:
            raise StageError("web_design requires brief.description (what to design)")

        style = str(brief.get("redesign_style") or brief.get("style") or "modern")
        page_type = str(brief.get("page_type") or "landing")
        audience = str(brief.get("audience") or "general")
        ref_url = str(brief.get("target_url") or "").strip()
        # A reference URL seeds the existing refine path; without one it's greenfield.
        scope = "full_page" if ref_url else "landing"

        # A resumed attempt re-attaches to the job it already submitted. Only a
        # first attempt (or one whose stashed job is dead) builds a payload and
        # submits — which also skips the reference-URL brand scrape on resume.
        job_id = await reattachable_job_id(self._client, self._sessions, ctx, self.name)
        if not job_id:
            user_context: dict = {"description": description, "site_type": page_type}
            if audience and audience != "general":
                user_context["target_audience"] = [audience]
            goals = brief.get("goals")
            if isinstance(goals, list) and goals:
                user_context["goals"] = goals

            # Best-effort brand seeding (palette/fonts) — from the project brand block, or
            # extracted from the reference URL when one was given. Failures are non-fatal.
            # Precedence: project/brief brand > the site's extracted brand > the
            # tenant's default brand (P6, merged under ``snapshot.brand`` at run
            # creation — read the AUTHORED part, add the tenant's back as a gap-filler).
            brand = authored_brand(ctx.config_snapshot) or brief_brand(brief) or None
            site_analysis = None
            if ref_url:
                brand_ctx, site_analysis = await resolve_brand_context(ref_url)
                brand = brand or brand_ctx
            brand = fill_tenant_brand(brand, ctx.config_snapshot)

            job_id = await stage_call(
                lambda: guarded(
                    lambda: self._client.start_design(
                        url=ref_url,
                        scope=scope,
                        style=style,
                        generate_screenshots=True,
                        # The run-view shows one representative still; one viewport keeps
                        # the Chromium work lean (compile/preview only use desktop).
                        screenshot_viewports=["desktop"],
                        # Greenfield has no original page → don't capture an original or a
                        # before/after comparison (only meaningful when seeding from a URL).
                        capture_original=bool(ref_url),
                        generate_comparison=bool(ref_url),
                        page_type=page_type,
                        user_context=user_context,
                        brand=brand,
                        site_analysis=site_analysis,
                    )
                ),
                timeout_seconds=_START_TIMEOUT_SECONDS,
                retries=0,  # non-idempotent submit
                stage_name="web_design.start",
            )
            # Durably BEFORE the wait: a poll-deadline expiry resets this stage to
            # 'pending' and re-enters handle() from the top, and without the stash
            # that re-entry submits a second job while this one keeps rendering.
            await stash_job_id(self._sessions, ctx, self.name, str(job_id))
        # retry_on=() keeps _WAIT_RETRIES a poll-*timeout* budget only (stage_call's
        # TimeoutError branch ignores retry_on): a still-running job keeps waiting,
        # a server-marked-failed job propagates the service's error text immediately
        # rather than re-polling the terminal job until retries are "exhausted".
        await await_design_start(self._client, str(job_id), "design")
        try:
            status = await stage_call(
                lambda: guarded(
                    lambda: self._client.wait_for_design(job_id, timeout=_POLL_TIMEOUT_SECONDS)
                ),
                timeout_seconds=_POLL_TIMEOUT_SECONDS + 10.0,
                retries=_WAIT_RETRIES,
                # Preserved deliberately: a poll-TIMEOUT budget, not an error budget.
                # guarded() re-raises RuntimeError (poll_job's terminal signal) untouched.
                retry_on=(),
                stage_name="web_design.wait",
            )
        except (StageTimeoutError, TimeoutError) as exc:
            # poll_job's own message says only "timed out after 590.0s", which cannot
            # distinguish "sat QUEUED behind html_render's single worker" from "the
            # render genuinely takes longer than the budget" — the exact question
            # nobody could answer from the logs. Name the last-seen status so the
            # next reader can tell whether the fix is capacity or a bigger budget.
            # Stays a resumable-timeout type (dag_executor._RESUMABLE_TIMEOUT_EXC),
            # so the stage still resumes — and now re-attaches to THIS job.
            last = await peek_status(self._client, str(job_id), self.name)
            seen = last.status if last is not None else "unknown (status probe failed)"
            raise StageTimeoutError(
                f"web_design job {job_id} not finished after "
                f"{1 + _WAIT_RETRIES} x {_POLL_TIMEOUT_SECONDS:.0f}s polls; "
                f"last-seen job status: {seen}. The job is still running server-side "
                "and a resume re-attaches to it rather than resubmitting."
            ) from exc
        result = await stage_call(
            lambda: guarded(lambda: self._client.get_design_result(job_id)),
            timeout_seconds=_RESULT_TIMEOUT_SECONDS,
            retries=1,
            # Was the default retry_on=(Exception,): a 401 from an expired service
            # licence was retried as a blip, then buried under "retries exhausted".
            retry_on=LLM_RETRY_ON,
            stage_name="web_design.result",
        )

        # Reuse the redesign visual collector (visual_assets → screenshots fallback).
        visuals = WebsiteRedesignStage._collect_visuals(result)
        if not result.redesigned_html_path and not visuals:
            raise StageError(
                f"web_design produced no HTML or visuals (job {job_id}); "
                "the generation returned nothing usable."
            )
        # Opt-in animated hero (brief.redesign_motion): scroll-through clip of the
        # generated page on the hero visual. Soft-fail keeps the static visuals.
        visuals = await attach_motion_hero(
            visuals,
            design_client=self._client,
            video_client=self._video_client,
            job_id=str(job_id),
            brief=brief,
            deliverable=ctx.config_snapshot.get("deliverable") or {},
            run_id=str(ctx.run_id),
            stage_name=self.name,
            tier=quality_tier(ctx.config_snapshot),
        )
        return WebDesignOutput(
            redesign_job_id=str(job_id),
            url=result.url or ref_url or "",
            style=result.style or style,
            redesigned_html_path=result.redesigned_html_path,
            # `job_id` is the html_render DESIGN job id (from start_design / the stash),
            # which is what the proxy route resolves — NOT ctx.run_id/deliverable_id.
            live_html_url=f"/api/redesign/jobs/{job_id}/html",
            accessibility_score=status.accessibility_score,
            visuals=visuals,
        )
