"""web_design_v2.design — computed-assembly interactive web design.

Drives html_render_service in ``mode="assembly"``: the service selects vetted
catalog blocks, fills content, themes from the brand, injects any ``embeds`` (the
three.js seam), and runs its verification lane. A design that fails verification
is marked failed by the service, so ``wait_for_design`` (poll_job) raises and this
stage fails — the run cannot complete with a broken design.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from w2v_common.stages import stage_call

from src.clients.design_client import DesignClient
from src.services.stages.base import StageError, StageHandler, StageInput, StageOutput
from src.services.stages.llm_budget import LLM_RETRY_ON
from src.services.stages.shared.design_job_stash import (
    SessionFactory,
    reattachable_job_id,
    stash_job_id,
)
from src.services.stages.shared.design_queue import await_design_start
from src.services.stages.upstream import guarded
from src.services.stages.website_review.redesign import RedesignVisual, WebsiteRedesignStage

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

_START_TIMEOUT_SECONDS = 60.0
_POLL_TIMEOUT_SECONDS = 590.0
_WAIT_RETRIES = 3
_RESULT_TIMEOUT_SECONDS = 60.0


# Structured brief fields that carry the business's own name, in precedence order.
# ``brand`` may be authored as a bare string (growth_bundle reads it that way) or
# as a block with ``name``.
_BRIEF_NAME_FIELDS = ("brand_name", "business_name", "company_name")


def _text(value: object) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def brief_brand_name(snapshot: Mapping[str, object]) -> str | None:
    """The business name the operator wrote in a structured brief field, else ``None``.

    NOT the project title: a title is the operator's label for the PROJECT
    ("Bakery landing page", "Q3 redesign"), and sent as the brand it was printed in
    the header as if it were the business. With ``None``, html_render's
    ``canonical_brand`` adopts ONE invented name and uses it everywhere — the
    defect the title fallback was added for (run 01a156a6: header "Hearth & Crumb",
    footer "The Daily Crumb") was two names, not an invented one."""
    brief = snapshot.get("brief")
    if isinstance(brief, Mapping):
        brand = brief.get("brand")
        if (name := _text(brand.get("name") if isinstance(brand, Mapping) else brand)) is not None:
            return name
        for field in _BRIEF_NAME_FIELDS:
            if (name := _text(brief.get(field))) is not None:
                return name
    return None


def with_brief_brand_name(
    brand: Mapping[str, object] | None, snapshot: Mapping[str, object]
) -> dict[str, object] | None:
    """``brand`` with a ``name`` filled from the brief when it carries none.

    brand_extract only yields a name from a ``target_url`` site (or an authored
    brand block). Without one, html_render got no name and its copy INVENTED one
    — run 01a156a6: header "Hearth & Crumb", footer "The Daily Crumb". An
    extracted/authored name always wins; this is a gap-filler only."""
    if brand is not None and _text(brand.get("name")) is not None:
        return dict(brand)
    name = brief_brand_name(snapshot)
    if name is None:
        return dict(brand) if brand is not None else None
    return {**(brand or {}), "name": name}


class WebDesignV2Input(StageInput):
    pass


class WebDesignV2Output(StageOutput):
    redesign_job_id: str
    style: str = "assembly"
    redesigned_html_path: str | None = None
    # The orchestrator redesign proxy path (mounted in prod + dev), which forwards
    # to html_render — a live, interactive, embeddable preview URL.
    live_html_url: str
    interactivity: str = "static"
    accessibility_score: float | None = None
    verification_status: str | None = None
    verification_report: dict | None = None
    visuals: list[RedesignVisual] = []  # noqa: RUF012 — pydantic copies field defaults per-instance


class WebDesignV2Stage(StageHandler[WebDesignV2Input, WebDesignV2Output]):
    name = "design"
    input_model = WebDesignV2Input
    output_model = WebDesignV2Output

    def __init__(
        self,
        client: DesignClient | None = None,
        *,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self._client = client or DesignClient()
        # Optional on purpose: without it the stage still runs end-to-end, it just
        # loses in-flight-job re-attachment across a resume (the old orphaning
        # behaviour). The blueprint passes the dispatcher's session factory.
        self._sessions = session_factory

    async def handle(self, payload: WebDesignV2Input, ctx: RunContext) -> WebDesignV2Output:
        brief = ctx.config_snapshot.get("brief") or {}
        description = str(brief.get("description") or brief.get("angle") or "").strip()
        if not description:
            raise StageError("web_design_v2 requires brief.description (what to design)")

        page_type = str(brief.get("page_type") or "landing")
        audience = str(brief.get("audience") or "general")
        interactivity = "interactive" if brief.get("interactive") else "static"
        if brief.get("interactivity") in ("static", "interactive"):
            interactivity = str(brief["interactivity"])
        component_set = brief.get("component_set")
        # "No embeds" must serialize as an empty list, NOT null: html_render's
        # RedesignRequest.embeds is `list[EmbedSpec]` with default_factory=list and
        # no `| None`, so an explicit JSON null fails validation (422 list_type)
        # while an empty list (== absent) is accepted. Sending None here broke every
        # embed-less web_design_v2 run at the design stage.
        raw_embeds = brief.get("embeds")
        embeds = raw_embeds if isinstance(raw_embeds, list) else []

        brand = with_brief_brand_name(
            (ctx.prior_output("brand_extract") or {}).get("brand"), ctx.config_snapshot
        )

        user_context: dict = {"description": description, "site_type": page_type}
        if audience and audience != "general":
            user_context["target_audience"] = [audience]
        if isinstance(brief.get("goals"), list) and brief["goals"]:
            user_context["goals"] = brief["goals"]

        # A resumed attempt RE-ATTACHES to the job it already submitted rather than
        # orphaning a healthy render and paying for a second one; same mechanism and
        # same reasoning as website_redesign (stages/shared/design_job_stash.py).
        job_id = await reattachable_job_id(self._client, self._sessions, ctx, self.name)
        if job_id is None:
            job_id = await stage_call(
                lambda: guarded(
                    lambda: self._client.start_design(
                        url="",
                        scope="landing",
                        mode="assembly",
                        interactivity=interactivity,
                        component_set=component_set,
                        embeds=embeds,
                        page_type=page_type,
                        user_context=user_context,
                        brand=brand,
                        generate_screenshots=True,
                        screenshot_viewports=["desktop"],
                        capture_original=False,
                        generate_comparison=False,
                    )
                ),
                timeout_seconds=_START_TIMEOUT_SECONDS,
                retries=0,  # non-idempotent submit
                stage_name="design.start",
            )
            # Durably BEFORE the wait: `record_progress` writes a non-terminal row
            # (the write-once trigger never fires) and `mark_pending` leaves `output`
            # intact, so the id survives into the next attempt.
            await stash_job_id(self._sessions, ctx, self.name, str(job_id))
        # A design that fails verification is marked failed by the service →
        # poll_job raises → this stage fails → the run does not complete.
        # retry_on=() keeps _WAIT_RETRIES a poll-*timeout* budget only (stage_call's
        # TimeoutError branch ignores retry_on): a still-running job keeps waiting,
        # a server-marked-failed job propagates the service's error text immediately
        # rather than re-polling the terminal job until retries are "exhausted".
        await await_design_start(self._client, str(job_id), "design")
        status = await stage_call(
            lambda: guarded(
                lambda: self._client.wait_for_design(job_id, timeout=_POLL_TIMEOUT_SECONDS)
            ),
            timeout_seconds=_POLL_TIMEOUT_SECONDS + 10.0,
            retries=_WAIT_RETRIES,
            # retry_on=() is PRESERVED deliberately (see the comment above): this
            # is a poll-timeout budget, not an error budget. guarded() changes
            # nothing here — it re-raises RuntimeError (poll_job's terminal-job
            # signal) untouched, so a server-marked-failed job still propagates on
            # attempt 1 instead of being re-polled until retries are "exhausted".
            retry_on=(),
            stage_name="design.wait",
        )
        result = await stage_call(
            lambda: guarded(lambda: self._client.get_design_result(job_id)),
            timeout_seconds=_RESULT_TIMEOUT_SECONDS,
            retries=1,
            # Was the default retry_on=(Exception,), which retried a 401 from an
            # expired service licence as though it were a blip and then buried it
            # under "retries exhausted". guarded() now raises PermanentUpstreamError
            # for 401 — absent from this allow-list, so it fails fast carrying the
            # upstream's own message.
            retry_on=LLM_RETRY_ON,
            stage_name="design.result",
        )

        visuals = WebsiteRedesignStage._collect_visuals(result)
        if not result.redesigned_html_path and not visuals:
            raise StageError(
                f"web_design_v2 produced no HTML or visuals (job {job_id}); nothing usable."
            )
        return WebDesignV2Output(
            redesign_job_id=str(job_id),
            style=result.style or "assembly",
            redesigned_html_path=result.redesigned_html_path,
            live_html_url=f"/api/redesign/jobs/{job_id}/html",
            interactivity=interactivity,
            accessibility_score=status.accessibility_score,
            verification_status=result.verification_status,
            verification_report=result.verification_report,
            visuals=visuals,
        )
