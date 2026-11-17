"""website_redesign — analysis_service redesign driven by the critique findings.

Maps the critique findings into a redesign job (``critique_findings``), polls to
completion, then pulls the full result for the before/after screenshots used as
video visuals. Only fetchable (http[s]) image URLs are kept — relative artifact
paths fall back to narrated text cards in the compositor. ``redesign_job_id`` is
surfaced so the frontend can offer the downloadable redesign HTML (W1 routes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid7

from w2v_common.stages import stage_call

from src.clients.design_client import DesignClient
from src.clients.schemas.design import DesignJobResult
from src.clients.video_production_client import VideoProductionClient
from src.services.stages.base import StageError, StageHandler, StageInput, StageOutput
from src.services.stages.brand_context import resolve_brand_context
from src.services.stages.shared.design_job_stash import (
    SessionFactory,
    reattachable_job_id,
    stash_job_id,
)
from src.services.stages.shared.design_queue import await_design_start
from src.services.stages.shared.motion_slides import quality_tier
from src.services.stages.shared.redesign_motion import attach_motion_hero
from src.services.stages.shared.video_gate import require_redesign

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

_START_TIMEOUT_SECONDS = 60.0
# poll_job hard-rejects a deadline > 600s (BaseServiceClient._clamp_timeout), so
# one poll waits up to ~590s; stage_call retries it _WAIT_RETRIES times for a
# ~(1 + _WAIT_RETRIES) * 590s ceiling. The redesign runs an LLM + screenshot
# render (slower than the critique scan), and the variant fan-out scenario runs
# *two* full LLM redesigns of a tall real page — so the ceiling is raised by
# adding poll attempts (the per-poll value can't exceed the 600s clamp).
_POLL_TIMEOUT_SECONDS = 590.0
# ⚠️ DELIBERATELY NOT RAISED. 8 attempts (1 + 7 retries) x 590s = 4720s is the
# ceiling of ONE attempt, and the measured 2-variant redesigns (5180 / 5580 /
# 5885 / 7102 / 7370s, html_render_service.redesign_jobs, 2026-09-02) ALL exceed
# it. The old "~40-50min" premise here was an estimate, never a measurement.
# The fix is that the attempts now COMPOSE — a resumed stage re-attaches to the
# job it already submitted (shared/design_job_stash) instead of orphaning it and
# starting a fresh one — not a bigger number here. Raising it is blocked three
# independent ways anyway: `stage_call` hard-caps `retries` at 10 (w2v_common
# stages/execution._MAX_RETRIES, so 13 raises ValueError at runtime); its backoff
# is exponential and UNBOUNDED (2**retries - 1 s, i.e. 8191s at 13 retries); and
# the outer stage Timeout in blueprints/website_review.py would make the raise
# inert regardless.
_WAIT_RETRIES = 7
# Mirror of html_render's REDESIGN_MAX_JOB_LIFETIME_SECONDS (gitops-production
# clusters/strato/services/w2v/configmaps.yaml, html-render-service-config; being
# raised 5400 -> 9000 in gitops PR #3319, which merges FIRST). Declared here so the
# budget invariants are DERIVED assertions in
# tests/services/stages/website_review/test_redesign.py rather than two numbers in
# two repos that drift apart silently. If you change one, change both.
_SERVICE_JOB_LIFETIME_SECONDS = 9000.0
# Longest 2-variant redesign observed to complete successfully (the 2026-09-02 set
# above). The stage's TOTAL wait — across the resume ladder, which now re-attaches
# — must cover this with headroom; a SINGLE attempt deliberately does not.
_MEASURED_MAX_JOB_SECONDS = 7370.0
_RESULT_TIMEOUT_SECONDS = 60.0
_MAX_FINDINGS = 20


def _is_fetchable(url: object) -> bool:
    return isinstance(url, str) and (url.startswith("http://") or url.startswith("https://"))


class WebsiteRedesignInput(StageInput):
    pass


class RedesignVisual(StageOutput):
    id: str
    label: str
    # Free-form role from analysis_service: the structured visual_assets path
    # passes through granular roles (original_desktop / redesign_mobile /
    # comparison_tablet), the screenshot-fallback path uses literal
    # "before"/"after". website_review_compile classifies these into the two-act
    # before/after buckets (see compile._bucket).
    role: str
    image_url: str
    # Opt-in (brief.redesign_motion): a deterministic scroll-through clip of the
    # redesign, attached to the hero visual. The run-view plays this when present
    # and falls back to image_url otherwise (the still is always produced).
    video_url: str | None = None
    video_duration_s: float | None = None


class WebsiteRedesignOutput(StageOutput):
    redesign_job_id: str
    url: str
    style: str
    redesigned_html_path: str | None = None
    accessibility_score: float | None = None
    visuals: list[RedesignVisual] = []  # noqa: RUF012 — pydantic copies field defaults per-instance


class WebsiteRedesignStage(StageHandler[WebsiteRedesignInput, WebsiteRedesignOutput]):
    name = "website_redesign"
    input_model = WebsiteRedesignInput
    output_model = WebsiteRedesignOutput

    def __init__(
        self,
        client: DesignClient | None = None,
        video_client: VideoProductionClient | None = None,
        *,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self._client = client or DesignClient()
        self._video_client = video_client or VideoProductionClient()
        # Optional on purpose: without it the stage still runs end-to-end, it just
        # loses in-flight-job re-attachment across a resume (i.e. the old orphaning
        # behaviour). The blueprint passes the dispatcher's session factory.
        self._sessions = session_factory

    @staticmethod
    def _assets_to_visuals(result: DesignJobResult) -> list[RedesignVisual]:
        # The structured visual_assets are explicitly "for downstream consumers"
        # and carry absolute URLs — the preferred source.
        return [
            RedesignVisual(
                id=asset.asset_id or str(uuid7()),
                label=asset.title or asset.kind or "redesign",
                role=asset.role or "after",
                image_url=asset.url,
            )
            for asset in (result.visual_assets or [])
            if _is_fetchable(asset.url)
        ]

    @staticmethod
    def _screenshots_to_visuals(result: DesignJobResult) -> list[RedesignVisual]:
        # Fallback when no structured assets: before/after screenshot maps and
        # per-variant screenshots.
        maps: list[tuple[str, str, dict[str, str]]] = [
            ("before", "original", result.original_screenshots or {}),
            ("after", "redesign", result.screenshots or {}),
        ]
        for variant in result.variants or []:
            maps.append(
                (
                    "after",
                    variant.style or "variant",
                    variant.screenshot_paths or {},
                )
            )
        visuals: list[RedesignVisual] = []
        for role, prefix, shots in maps:
            for label, url in shots.items():
                if _is_fetchable(url):
                    visuals.append(
                        RedesignVisual(
                            id=str(uuid7()),
                            label=f"{prefix} {label}",
                            role=role,
                            image_url=url,
                        )
                    )
        return visuals

    @classmethod
    def _collect_visuals(cls, result: DesignJobResult) -> list[RedesignVisual]:
        return cls._assets_to_visuals(result) or cls._screenshots_to_visuals(result)

    async def _submit(
        self,
        url: str,
        style: str,
        styles: list[str] | None,
        critique: dict,
    ) -> str:
        """Build the redesign payload and submit ONE non-idempotent design job.

        Split out of ``handle`` so a resumed attempt that re-attaches skips the whole
        thing — including the brand scrape, which is pure waste on a job that is
        already rendering.
        """
        critique_findings = [
            {
                "severity": str(f.get("severity") or "medium"),
                "title": str(f.get("title") or ""),
                "description": str(f.get("description") or ""),
                "recommendation": f.get("recommendation") or None,
                "category": f.get("category") or None,
            }
            for f in (critique.get("findings") or [])[:_MAX_FINDINGS]
        ]

        # Best-effort: feed the source site's real palette/fonts/logo so the redesign
        # refines the existing brand instead of defaulting to a generic blue. Any
        # failure returns (None, None) and the redesign proceeds as before.
        brand, site_analysis = await resolve_brand_context(url)

        return await stage_call(
            lambda: self._client.start_design(
                url=url,
                scope="redesign",
                critique_findings=critique_findings or None,
                style=style,
                styles=styles,
                generate_screenshots=True,
                # Render a single desktop viewport only. The 16:9 after act uses one
                # representative shot per variant, and website_review_compile already
                # discards tablet/mobile (compile._prefer_desktop keeps the widest
                # shot). Rendering all three viewports per variant + per original on a
                # tall page tripled the Chromium work for captures that were thrown
                # away — the dominant cost behind website_redesign.wait timing out on
                # the variant fan-out scenario. (Promo's hero path already does this.)
                screenshot_viewports=["desktop"],
                # Also capture a discrete still per landmark section so the after
                # act can show each section settled, synced to the line that
                # discusses it (website_review_visual_director), instead of one
                # tall full-page scroll. Surfaced as redesign_section:<name> assets.
                section_screenshots=True,
                brand=brand,
                site_analysis=site_analysis,
            ),
            timeout_seconds=_START_TIMEOUT_SECONDS,
            retries=0,  # non-idempotent submit: a retry would start a duplicate design job
            stage_name="website_redesign.start",
        )

    async def handle(
        self,
        payload: WebsiteRedesignInput,
        ctx: RunContext,
    ) -> WebsiteRedesignOutput:
        require_redesign(ctx)  # runs when generate_redesign OR generate_video; else skip
        # Critique is OPTIONAL: a redesign-only run (generate_critique=false) has no
        # findings, so the redesign proceeds from the target URL alone. When the
        # critique did run, its findings steer the redesign (critique_findings below).
        critique = ctx.prior_output("website_critique") or {}
        brief = ctx.config_snapshot.get("brief") or {}
        url = str(critique.get("url") or brief.get("target_url") or "")
        if not url:
            raise RuntimeError("no url available for redesign (brief.target_url required)")
        style = str(brief.get("redesign_style") or "modern")
        raw_styles = brief.get("redesign_styles") or None
        styles = [str(x) for x in raw_styles] if isinstance(raw_styles, list) else None

        # A resumed attempt RE-ATTACHES to the job it already submitted. Only a first
        # attempt — or one whose stashed job the service has since marked dead —
        # builds a payload and submits, which also skips the brand scrape on resume.
        # Without this, a poll-budget expiry ORPHANED a healthy render: html_render
        # kept rendering job 01a060f2-eb9d-724d-a49b-f76294cf447d to a successful
        # completion ~25 min AFTER the orchestrator had abandoned it and submitted a
        # second one, discarding 103 minutes of finished (and paid-for) work.
        job_id = await reattachable_job_id(self._client, self._sessions, ctx, self.name)
        if job_id is None:
            job_id = await self._submit(url, style, styles, critique)
            # Durably BEFORE the wait — and before the claim-queue wait, which can
            # itself blow the outer stage budget. `record_progress` writes only a
            # non-terminal row, so the write-once trigger never fires; `mark_pending`
            # (the resumable-timeout reset) leaves `output` intact, so the id is there
            # on the next attempt.
            await stash_job_id(self._sessions, ctx, self.name, str(job_id))

        # retry_on=() makes _WAIT_RETRIES a *poll-timeout* budget only. A poll that
        # times out still retries (the design job is still running — that is the
        # deliberate ~(1 + _WAIT_RETRIES) * 590s headroom above), because stage_call's
        # TimeoutError branch fires regardless of retry_on. But a job the service has
        # already marked `failed` — which poll_job raises as a RuntimeError carrying
        # the service's own error text ("Model token limit exceeded", …) — now fails
        # the stage on attempt 1 instead of re-polling the same corpse 3 more times
        # and burying the real cause under StageRetriesExhaustedError.
        await await_design_start(self._client, job_id, "website_redesign")
        status = await stage_call(
            lambda: self._client.wait_for_design(job_id, timeout=_POLL_TIMEOUT_SECONDS),
            timeout_seconds=_POLL_TIMEOUT_SECONDS + 10.0,
            retries=_WAIT_RETRIES,
            retry_on=(),
            stage_name="website_redesign.wait",
        )
        result = await stage_call(
            lambda: self._client.get_design_result(job_id),
            timeout_seconds=_RESULT_TIMEOUT_SECONDS,
            retries=1,
            stage_name="website_redesign.result",
        )

        visuals = self._collect_visuals(result)
        # Fail loud rather than emit a completed-but-empty redesign. A redesign
        # whose whole point is the before/after imagery, that comes back with no
        # fetchable visual, means generation produced nothing usable (e.g. the
        # html_render redesign worker returned prose/empty, or every variant
        # failed). Returning here would hand a videoless redesign to the
        # record/compile stages and silently ship a bad deliverable; surfacing it
        # as a stage failure keeps `completed` honest and makes the cause visible.
        if not visuals:
            raise StageError(
                f"website_redesign produced no fetchable visuals "
                f"(job {job_id}, styles={styles or [style]}); "
                "the redesign generation returned nothing usable."
            )

        # Opt-in animated hero (brief.redesign_motion): attach a deterministic
        # scroll-through clip of the redesign to the hero visual. Soft-fail keeps
        # the static visuals; no-op when the flag is off.
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

        return WebsiteRedesignOutput(
            redesign_job_id=str(job_id),
            url=result.url or url,
            style=result.style or style,
            redesigned_html_path=result.redesigned_html_path,
            accessibility_score=status.accessibility_score,
            visuals=visuals,
        )
