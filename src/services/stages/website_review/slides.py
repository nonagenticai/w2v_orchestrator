"""website_review_slides — render the BEFORE-act slide shots.

Two producers feed this stage, in priority order:

1. The authored-shotlist path (A2): the operator places ``slide`` shots in
   ``brief.shotlist`` and this renders that explicit content, keyed to the shot's
   ``line_id``. The operator's words always win.
2. ``website_review_slide_plan``: for the fully-AI path, that stage authors a card
   per BEFORE-act narration line from the critique findings, keyed to the SCRIPT
   line id. Before it existed this stage was inert BY CONSTRUCTION — nothing in
   the blueprint ever authored a ``slide`` shot, so every run found zero, returned
   in ~0.3s and emitted ``slides_not_authored``, and compile fell back to
   scroll-capture stills.

Either way ``website_review_compile`` binds each rendered slide to its line 1:1,
and a slide never outranks a live site-capture screen — it fills the lines that
would otherwise fall through to the full-page scroll pool.

URL shots get nothing here — they bind to their captured screen in compile. So
unlike ``promo_slides`` (which renders a universal slide fallback for EVERY line),
this stage only renders slide content, and returns an empty deck when neither
producer supplied any (a critique-only run, or a run with no before-act lines).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from src.clients.slide_service_client import SlideServiceClient
from src.clients.video_production_client import VideoProductionClient
from src.models.failure_class import UNKNOWN, FailureClass
from src.services.stages.base import (
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithDegradations,
)
from src.services.stages.shared.degradation import SLIDES_NOT_AUTHORED, advisory
from src.services.stages.shared.placeholder_text import slide_anchor
from src.services.stages.shared.shotlist import ShotSpec, normalize_shotlist, normalize_slide
from src.services.stages.shared.slide_deck import (
    _BRAND_DEFAULTS,
    _CTA_TEMPLATE_NAME,
    _FEATURE_TEMPLATE_NAME,
    _TITLE_TEMPLATE_NAME,
    _shotlist_deck_slide,
)
from src.services.stages.shared.slide_stage import DeckEntry, render_deck_slide_stage
from src.services.stages.shared.video_gate import require_video

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


def planned_slide_shots(ctx: RunContext) -> list[ShotSpec]:
    """The ``website_review_slide_plan`` deck, as ``ShotSpec``s this stage can render.

    Built directly rather than through ``normalize_shotlist``: that function assigns
    ``line_id`` POSITIONALLY (``shot-000``, ``shot-001``, …), which is the join key
    of the authored shotlist. A planned card is keyed to the SCRIPT line it plays
    under, so running it through the positional normalizer would silently re-key the
    whole deck and compile would bind nothing.

    ``seconds=0.0`` is deliberate and load-bearing: a planned slide is a VISUAL and
    must never influence narration length. Nothing on this path reads it —
    ``_shotlist_deck_slide`` projects ``kind``/``slide``/``narration`` only — so the
    zero is a statement of the contract, not a value anyone consumes.
    """
    plan = ctx.prior_output("website_review_slide_plan") or {}
    shots: list[ShotSpec] = []
    for i, entry in enumerate(plan.get("slides") or []):
        if not isinstance(entry, dict):
            continue
        line_id = str(entry.get("line_id") or "").strip()
        slide = normalize_slide(entry.get("slide"))
        # A card with no content is the hollow deck this must never render: the
        # canonical shape keeps only non-empty fields, so a bare {"template": ...}
        # means every authored field was blank.
        if not line_id or len(slide) <= 1:
            continue
        shots.append(
            ShotSpec(
                index=i,
                line_id=line_id,
                kind="slide",
                narration=str(entry.get("narration") or ""),
                seconds=0.0,
                slide=slide,
            )
        )
    return shots


class WebsiteReviewSlidesInput(StageInput):
    pass


class WebsiteReviewSlide(StageOutput):
    id: str
    line_id: str
    image_url: str
    # Set only when motion graphics are enabled: a branded animated clip of this
    # authored slide. website_review_compile supersedes the still ``image_url``
    # with this clip on the slide's line; the PNG stays the degrade path.
    motion_video_url: str | None = None


class WebsiteReviewSlidesOutput(StageOutputWithDegradations):
    # NO `produces` on `slides`, deliberately. An empty deck is a LEGITIMATE
    # operating mode here — a legacy run or a url-only shotlist authors no
    # `slide` shots, and those lines bind to their captured screen in compile.
    # Declaring `produces` would mark every one of those correct runs `degraded`,
    # and `produces_required` would fail them outright. The empty case is made
    # tellable apart from a render outage STRUCTURALLY instead, via the two flags
    # below — the `cards_stubbed` / `cards_render_failed` shape from
    # `action_reel/graphics_design.py`.
    slides: list[WebsiteReviewSlide] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    #: Same meaning as `ShowcaseSlidesOutput.slides_stubbed`: the brief AUTHORED lines
    #: that should have carried slides and none arrived. website_review has no such
    #: path — an empty deck here (no `slide` shots, no planned cards) is recorded by the
    #: `slides_not_authored` ADVISORY with `degraded: false`, so this stays False.
    #: Setting it there too gave the field two meanings across stages and made one
    #: output say both "not a degradation" and "stubbed". Kept on the model for shape
    #: parity with showcase; `(False, False)` still differs from an outage `(False, True)`.
    slides_stubbed: bool = False
    #: True ONLY when authored slide shots EXISTED and their deck render failed.
    #: Never infer this from `slides == []` — that is also the stub reading above.
    #: The lines that should have carried a rendered slide will render as bare
    #: text cards, and `website_review_compile` still emits a valid artifact, so
    #: this flag is the only machine-readable evidence the outage happened.
    slides_render_failed: bool = False
    #: True when the deck came from ``website_review_slide_plan`` (the fully-AI
    #: path) rather than from operator-authored ``brief.shotlist`` slide shots.
    #: Recorded so "which producer wrote this deck" is answerable from the run
    #: record without re-deriving it from the brief.
    slides_planned: bool = False
    #: The FAULT half of the same outage, carried through from the exception the
    #: soft-fail boundary swallowed (`shared/slide_stage.py` stamps
    #: `attribute_exception(exc)` on its result). Copied onto the output because
    #: `dag_executor._enforce_failure_flags` reads only the DUMPED mapping — the
    #: exception object is long gone by then, so this field is the only way the
    #: attribution reaches `StageReportedFailureError` and, from there, the
    #: persisted `runs.failure_class`. `unknown` on every healthy render.
    failure_class: FailureClass = UNKNOWN
    # ...and this is what makes them ACTIONABLE rather than merely recorded.
    # Until this line, `slides_render_failed` was set here and read NOWHERE in
    # `src/`: the stage still recorded `completed`, `website_review_compile`
    # built its slide map by comprehension over `[]` with no check, and the run
    # shipped and was billed as a success with every authored slide missing.
    # Keyed on the FAILURE flag, never on `produces`/`slides == []` — the
    # url-only/legacy shotlist above is legitimately empty and must keep
    # completing.
    fails_when_flagged: ClassVar[tuple[str, ...]] = ("slides_render_failed",)
    notes: list[str] = []  # noqa: RUF012 — pydantic copies field defaults per-instance


class WebsiteReviewSlidesStage(StageHandler[WebsiteReviewSlidesInput, WebsiteReviewSlidesOutput]):
    name = "website_review_slides"
    input_model = WebsiteReviewSlidesInput
    output_model = WebsiteReviewSlidesOutput

    def __init__(
        self,
        client: SlideServiceClient | None = None,
        video_client: VideoProductionClient | None = None,
    ) -> None:
        self._client = client or SlideServiceClient()
        self._video_client = video_client or VideoProductionClient()

    async def handle(
        self,
        payload: WebsiteReviewSlidesInput,
        ctx: RunContext,
    ) -> WebsiteReviewSlidesOutput:
        require_video(ctx)  # video-only: authored before-act slides have no critique-only role
        brief = ctx.config_snapshot.get("brief") or {}
        authored_shots = [s for s in normalize_shotlist(brief) if s.kind == "slide"]
        # The operator's own shotlist wins; the planned deck only fills the
        # fully-AI path, where there has never been a slide shot to render.
        planned = [] if authored_shots else planned_slide_shots(ctx)
        slide_shots = authored_shots or planned
        if not slide_shots:
            # No before-act slides from either producer (a critique-only run, a
            # url-only shotlist, or a plan stage that found no lines) —
            # nothing to render; compile binds those lines elsewhere. This
            # correct-and-empty deck is never read as the render outage below
            # (identical `slides == []`) because `slides_render_failed` stays False
            # and the advisory records why it is empty.
            #
            # ADVISORY, not degradation, and it is no longer the normal path: a
            # plain website_review now gets its deck from
            # `website_review_slide_plan`, so this advisory fires only when the
            # deck genuinely does not exist. It stays an ADVISORY because a
            # `slides_stubbed`
            # degradation here marked EVERY such run `degraded` (prod run
            # 01a0a054), which is a signal that fires on everything and therefore
            # on nothing. Nothing is missing from the video. The empty deck stays
            # answerable from the run record via the advisory; the real outage
            # sets `slides_render_failed` and HARD-FAILS via `fails_when_flagged`.
            return WebsiteReviewSlidesOutput(
                slides=[],
                advisories=[
                    advisory(
                        SLIDES_NOT_AUTHORED,
                        "no `slide` shots in the shotlist and no cards from "
                        "website_review_slide_plan, so the deck is empty and "
                        "slide_service was never called",
                    )
                ],
            )

        deliverable = ctx.config_snapshot.get("deliverable") or {}
        brand = ctx.config_snapshot.get("brand") or {}
        # Never a hardcoded house brand: an unbranded run renders no eyebrow.
        anchor = slide_anchor(ctx.config_snapshot)
        deck_brand = {**_BRAND_DEFAULTS, **brand}  # per-project brand wins

        async def _build_entries() -> list[DeckEntry]:
            title_id = await self._client.resolve_template_id(_TITLE_TEMPLATE_NAME)
            cta_id = await self._client.resolve_template_id(_CTA_TEMPLATE_NAME)
            needs_feature = any(s.slide.get("template") == "bullets" for s in slide_shots)
            feature_id = (
                await self._client.resolve_template_id(_FEATURE_TEMPLATE_NAME)
                if needs_feature
                else ""
            )
            return [
                (
                    shot.line_id,
                    _shotlist_deck_slide(
                        shot,
                        shot.narration,
                        anchor,
                        title_id=title_id,
                        cta_id=cta_id,
                        feature_id=feature_id,
                    ),
                )
                for shot in slide_shots
            ]

        rendered = await render_deck_slide_stage(
            client=self._client,
            video_client=self._video_client,
            build_entries=_build_entries,
            anchor=anchor,
            deck_brand=deck_brand,
            config_snapshot=ctx.config_snapshot,
            deliverable=deliverable,
            run_id=str(ctx.run_id),
            stage_name=self.name,
            media_id_prefix="website_review_slide",
            make_slide=lambda slide_id, line_id, image_url: WebsiteReviewSlide(
                id=slide_id, line_id=line_id, image_url=image_url
            ),
            # website_review_slides stage timeout is 600s; keep the motion phase
            # under it with headroom for the stills deck, so an oversized deck
            # degrades to stills gracefully instead of hard-failing the stage.
            phase_budget_s=450.0,
        )
        return WebsiteReviewSlidesOutput(
            slides=rendered.slides,
            slides_planned=bool(planned),
            slides_render_failed=rendered.render_failed,
            failure_class=rendered.failure_class,
            notes=rendered.notes,
        )
