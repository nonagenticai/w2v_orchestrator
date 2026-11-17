"""promo_html — designed HTML segments for the promo, via html_render_service.

Calls the standalone web-design service (``html_render_service`` ``/api/designs``)
with ``scope="hero"``: it generates a self-contained, brand-styled hero treatment
of the source page and rasterises it to PNGs — the "augment the UI with designed
assets" the promo wants. ``scope="hero"`` is the *fast* path (one section, no
full-page chunking, no variants), so the promo no longer waits 13-20 min on a
full-page redesign — a single poll suffices.

We start one hero design of the brief's ``source_url``, poll once to completion,
and keep the fetchable (http[s]) "after" screenshots as segment images (the
service returns absolute artifact URLs). promo_render routes the storyboard's
``html`` lines onto these (cycling), and falls back to the per-line slide when
none are available.

Opt-in + soft-fail, mirroring ``site_capture`` / ``ai_image_gen``: no
``source_url`` ⇒ clean skip; any design error ⇒ zero segments + a recorded
note, never a failed run (an enhancement, not a precondition).

**Re-entry.** ``POST /api/designs`` has no idempotency key, so a pod death
mid-stage used to submit a SECOND paid design job on re-entry while the first
kept rendering. Two step-ledger rows per segment close that (the promo has one
segment, ``hero``):

* ``design:start:<segment>`` — the submitted job id, written BEFORE the wait, so
  a re-entry polls the job it already has;
* ``design:result:<segment>`` — the finished segments, so a re-entry after the
  result is in makes no design call at all.

The sibling design stages do the first half with ``design_job_stash``; that
writes the stage's whole progress dict, which is why this uses the ledger.
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, NotRequired, TypedDict
from uuid import uuid7

from w2v_common.stages import stage_call

from src.clients.design_client import DesignClient
from src.clients.schemas.design import DesignJobResult
from src.services.stages.base import (
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithDegradations,
    StageSkippedError,
)
from src.services.stages.brand_context import resolve_brand_context
from src.services.stages.shared.degradation import PROMO_HTML_DESIGN_FAILED, degradation
from src.services.stages.shared.design_queue import await_design_start
from src.services.stages.shared.step_ledger import hash_inputs

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

_START_TIMEOUT_SECONDS = 60.0
# poll_job hard-rejects deadlines > 600s, so one poll waits ~590s. A hero design
# (single-section LLM generation + one screenshot) finishes well inside one poll,
# so — unlike the old full-page redesign — no re-poll patience is needed.
_POLL_TIMEOUT_SECONDS = 590.0
# No retries: hero is fast. If it somehow doesn't finish within one poll, soft-fail
# gracefully (0 segments → slide fallback) rather than burning the stage budget.
_POLL_RETRIES = 0
_RESULT_TIMEOUT_SECONDS = 60.0

# The promo designs exactly one segment. The key is still per-segment so a
# fan-out adds keys instead of renaming this one.
_HERO_SEGMENT = "hero"


class _HeroRequest(TypedDict):
    """The caller-determined part of the hero design request (submitted AND hashed)."""

    url: str
    scope: str
    style: str
    generate_screenshots: bool
    capture_original: bool
    generate_comparison: bool
    screenshot_viewports: list[str]
    full_page: bool
    html_content: NotRequired[str]


def _is_fetchable(url: object) -> bool:
    return isinstance(url, str) and url.startswith(("http://", "https://"))


class PromoHtmlInput(StageInput):
    pass


class PromoHtmlSegment(StageOutput):
    id: str
    image_url: str
    label: str


class PromoHtmlOutput(StageOutputWithDegradations):
    segments: list[PromoHtmlSegment]
    notes: list[str]
    # First-class signal (not inferred from list length) of whether the render
    # gets designed HTML segments or falls back to slides for ``html`` lines.
    used_html: bool = False


class PromoHtmlStage(StageHandler[PromoHtmlInput, PromoHtmlOutput]):
    name = "promo_html"
    input_model = PromoHtmlInput
    output_model = PromoHtmlOutput

    def __init__(self, client: DesignClient | None = None) -> None:
        self._client = client or DesignClient()

    @staticmethod
    def _collect_after_images(result: DesignJobResult) -> list[PromoHtmlSegment]:
        """Fetchable designed-HTML ("after") screenshots, asset URLs first."""
        segments: list[PromoHtmlSegment] = []
        for asset in result.visual_assets or []:
            if (asset.role or "after") == "after" and _is_fetchable(asset.url):
                segments.append(
                    PromoHtmlSegment(
                        id=asset.asset_id or str(uuid7()),
                        image_url=asset.url,
                        label=asset.title or asset.kind or "redesign",
                    )
                )
        if segments:
            return segments
        # Fallback: the redesigned-HTML screenshot map (viewport -> url).
        for label, url in (result.screenshots or {}).items():
            if _is_fetchable(url):
                segments.append(
                    PromoHtmlSegment(id=str(uuid7()), image_url=url, label=f"redesign {label}")
                )
        return segments

    async def handle(
        self,
        payload: PromoHtmlInput,
        ctx: RunContext,
    ) -> PromoHtmlOutput:
        brief = ctx.config_snapshot.get("brief") or {}
        source_url = (brief.get("source_url") or "").strip()
        if not source_url:
            raise StageSkippedError(
                "promo_html: no brief.source_url (opt-in stage not requested)",
                missing=["brief.source_url"],
            )

        style = str(brief.get("redesign_style") or "modern")
        # Lean render shape: the promo consumes a single desktop hero image,
        # so skip the live-original capture, the before/after comparison, the
        # tablet/mobile viewports, and full-page capture. Rendering all of
        # those for a heavy live SPA is what OOM-killed html-render-service
        # (6 full-page Chromium captures + a PIL diff at a 3Gi limit).
        # ONE dict is both what is submitted and what is hashed, so the two
        # cannot drift apart.
        request: _HeroRequest = {
            "url": source_url,
            "scope": "hero",
            "style": style,
            "generate_screenshots": True,
            "capture_original": False,
            "generate_comparison": False,
            "screenshot_viewports": ["desktop"],
            "full_page": False,
        }
        capture = ctx.prior_output("site_capture") or {}
        captured_html = str(capture.get("html_content") or "")
        if captured_html:
            request["html_content"] = captured_html
        elif brief.get("auth_info"):
            # Do not launch another anonymous session against an authenticated app.
            supplied = str(
                brief.get("value_proposition") or brief.get("instructions") or ""
            ).strip()
            if not supplied:
                raise StageSkippedError(
                    "promo_html: authenticated page content is unavailable",
                    missing=["site_capture.html_content"],
                )
            request["html_content"] = "<main><p>" + escape(supplied[:12000]) + "</p></main>"
        ledger = ctx.step_ledger(self.name)

        async def _start() -> str:
            # Best-effort brand extraction so the hero treatment keeps the source
            # site's real palette/fonts instead of a generic blue. Resolved HERE,
            # inside the submit step, so a re-entry that already has a job does
            # not extract again — and deliberately NOT part of the step's hash:
            # it is derived from ``source_url`` by a live scrape, so hashing it
            # would let one wobble in the extraction resubmit a paid job.
            if request.get("html_content"):
                brand = ctx.config_snapshot.get("brand") or {}
                site_analysis = None
            else:
                brand, site_analysis = await resolve_brand_context(source_url)
            return str(
                await self._client.start_design(**request, brand=brand, site_analysis=site_analysis)
            )

        notes: list[str] = []
        try:
            # Submit ONCE per request. The job id is stored the moment
            # start_design returns — before any wait — so a pod death during the
            # queue wait or the poll re-enters onto the same job.
            job_id = await ledger.once(
                f"design:start:{_HERO_SEGMENT}", hash_inputs(request), _start
            )

            async def _finish() -> list[PromoHtmlSegment]:
                # Single poll: a hero design finishes well inside one ~590s poll —
                # but only once a worker has it. Queue time is waited out first.
                await await_design_start(self._client, job_id, "promo_html")
                await stage_call(
                    lambda: self._client.wait_for_design(job_id, timeout=_POLL_TIMEOUT_SECONDS),
                    timeout_seconds=_POLL_TIMEOUT_SECONDS + 10.0,
                    retries=_POLL_RETRIES,
                    stage_name="promo_html.wait",
                )
                result = await self._client.get_design_result(job_id)
                return self._collect_after_images(result)

            # Keyed on the job id: a resubmitted job never inherits the previous
            # job's segments. A poll/result failure raises out of ``_finish`` and
            # stores nothing, so the soft-fail below is unchanged.
            segments = await ledger.once(
                f"design:result:{_HERO_SEGMENT}",
                hash_inputs(job_id),
                _finish,
                dump=lambda segs: [seg.model_dump(mode="json") for seg in segs],
                load=lambda raw: [
                    PromoHtmlSegment.model_validate(item)
                    for item in (raw if isinstance(raw, list) else [])
                ],
            )
            if not segments:
                notes.append("design produced no fetchable 'after' screenshots")
        except Exception as exc:  # noqa: BLE001 -- soft-fail: enhancement, never abort the run
            notes.append(f"promo_html design failed: {exc}")
            segments = []

        return PromoHtmlOutput(
            segments=segments,
            notes=notes,
            used_html=bool(segments),
            degradations=[degradation(PROMO_HTML_DESIGN_FAILED, note) for note in notes],
        )
