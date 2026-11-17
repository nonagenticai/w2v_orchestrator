"""showcase_slides — render the authored branded card beats of a showcase video.

The showcase blueprint interleaves branded cards (an intro, section dividers, a
value-prop, an outro) between the real form-recording footage. Those cards are
authored as ``brief.lines`` entries carrying a ``card`` block; this stage renders
each one to a still PNG via slide_service and (opt-in) a branded animated clip via
video_production, keyed by the line's join id so ``demo_video_compile`` binds it
1:1. Footage lines (no ``card``) are bound to their recorded clip in compile and
produce nothing here — so an all-footage demo_video run yields an empty deck and
is untouched.

The card content reuses the same deterministic deck projection promo /
website_review use (``_shotlist_deck_slide`` + ``render_deck_motion_clips``), so a
showcase card is on the same seeded slide_service templates and the same
``animated_title_card`` / ``animated_slide`` motion templates as every other
branded slide in the fleet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from src.clients.slide_service_client import SlideServiceClient
from src.clients.video_production_client import VideoProductionClient
from src.models.failure_class import UNKNOWN, FailureClass
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithDegradations,
)
from src.services.stages.demo_video.line_ids import authored_line_id
from src.services.stages.shared.degradation import (
    SLIDES_NOT_AUTHORED,
    SLIDES_STUBBED,
    advisory,
    degradation,
)
from src.services.stages.shared.design_system import apply_design_tokens, tokens_for_snapshot
from src.services.stages.shared.shotlist import ShotSpec
from src.services.stages.shared.slide_deck import (
    _BRAND_DEFAULTS,
    _CTA_TEMPLATE_NAME,
    _FEATURE_TEMPLATE_NAME,
    _TITLE_TEMPLATE_NAME,
    _shotlist_deck_slide,
)
from src.services.stages.shared.slide_stage import DeckEntry, render_deck_slide_stage

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


class ShowcaseSlidesInput(StageInput):
    pass


class ShowcaseSlide(StageOutput):
    id: str
    line_id: str
    image_url: str
    # Set only when motion graphics are enabled: a branded animated clip of this
    # card. demo_video_compile supersedes the still ``image_url`` with this clip on
    # the card's line; the PNG stays the degrade path.
    motion_video_url: str | None = None


class ShowcaseSlidesOutput(StageOutputWithDegradations):
    # NO `produces` on `slides`, deliberately, and this is the site where getting
    # that wrong is most expensive: an ALL-FOOTAGE demo_video run authors no
    # `card` lines and yields an empty deck BY CONSTRUCTION (see `_card_shots`
    # and the early return in `handle`). Declaring `produces` would mark every
    # such run `degraded`; `produces_required` would redden the whole showcase
    # E2E. The empty case is made tellable apart from a render outage
    # STRUCTURALLY instead, via the two flags below — the `cards_stubbed` /
    # `cards_render_failed` shape from `action_reel/graphics_design.py`.
    slides: list[ShowcaseSlide] = []  # noqa: RUF012 — pydantic copies field defaults per-instance
    #: True when the empty deck is a STUB the author should have filled: the brief
    #: AUTHORED lines but none carries a `card` (an all-footage run, or a non-showcase
    #: reuse of this stage), so slide_service was never asked for anything. Always
    #: paired with the `slides_stubbed` DEGRADATION.
    #:
    #: ⛔ NOT set for a generated-mode showcase (no authored lines at all). That deck
    #: has no card source by construction and records the `slides_not_authored`
    #: ADVISORY with `degraded: false`; setting this flag too made one output say
    #: both "not a degradation" and "stubbed" (run 01a0afab). The advisory is the
    #: record there, and `(False, False)` still differs from a render outage
    #: `(False, True)`.
    slides_stubbed: bool = False
    #: True ONLY when authored card lines EXISTED and their deck render failed.
    #: Never infer this from `slides == []` — that is also the stub reading above.
    #: The interleaved branded beats then silently vanish from the timeline while
    #: `demo_video_compile` still emits a valid artifact, so this flag is the only
    #: machine-readable evidence the outage happened.
    slides_render_failed: bool = False
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
    # `src/`: the stage still recorded `completed`, `demo_video_compile` built
    # its slide map by comprehension over `[]` with no check, and the run shipped
    # and was billed as a success with every branded card missing. Keyed on the
    # FAILURE flag, never on `produces`/`slides == []` — the all-footage run
    # above is legitimately empty and must keep completing.
    fails_when_flagged: ClassVar[tuple[str, ...]] = ("slides_render_failed",)
    notes: list[str] = []  # noqa: RUF012 — pydantic copies field defaults per-instance


def _card_shots(brief: dict) -> list[tuple[str, ShotSpec]]:
    """Authored card lines → (line_id, slide ShotSpec), in brief order.

    A line is a card when it carries a non-empty ``card`` block; the line id is the
    same one demo_video_gen assigns, so narration / image / shot all join on it.
    """
    out: list[tuple[str, ShotSpec]] = []
    for idx, raw in enumerate(brief.get("lines") or []):
        if not isinstance(raw, dict):
            continue
        card = raw.get("card")
        if not isinstance(card, dict) or not card:
            continue
        line_id = authored_line_id(raw, idx)
        narration = str(raw.get("text") or "").strip()
        slide = {**card, "template": str(card.get("template") or "title").strip().lower()}
        out.append(
            (
                line_id,
                ShotSpec(
                    index=idx,
                    line_id=line_id,
                    kind="slide",
                    narration=narration,
                    seconds=0.0,
                    slide=slide,
                ),
            )
        )
    return out


class ShowcaseSlidesStage(StageHandler[ShowcaseSlidesInput, ShowcaseSlidesOutput]):
    name = "showcase_slides"
    input_model = ShowcaseSlidesInput
    output_model = ShowcaseSlidesOutput
    # Renders cards through slide_service + video_production only: no LLM, no TTS.
    provider_slots: ClassVar[tuple[SlotUse, ...]] = ()

    def __init__(
        self,
        client: SlideServiceClient | None = None,
        video_client: VideoProductionClient | None = None,
    ) -> None:
        self._client = client or SlideServiceClient()
        self._video_client = video_client or VideoProductionClient()

    async def handle(self, payload: ShowcaseSlidesInput, ctx: RunContext) -> ShowcaseSlidesOutput:
        brief = ctx.config_snapshot.get("brief") or {}
        card_shots = _card_shots(brief)
        if not card_shots:
            # Empty deck, never readable as the render outage below (which produces the
            # identical `slides == []`, sets `slides_render_failed` and HARD-FAILS via
            # `fails_when_flagged`): `slides_render_failed` stays False here, and the
            # record says WHY it is empty — an advisory for generated mode, the
            # `slides_stubbed` flag + degradation only when the brief authored lines.
            #
            # ⚠️ GENERATED-MODE SHOWCASES HAVE NO CARD SOURCE. Cards come ONLY from
            # `brief.lines[*].card`; nothing in `src/` ever writes a `card` block, and
            # `ShowcaseBrief._authored_coherence` (api/schemas/projects.py) accepts
            # `brief.lines` only with a `package_url`. So a generated-mode showcase
            # (no authored lines) reaches this branch EVERY time, with slide_service
            # healthy and simply never asked. That is a known PRODUCT gap (what goes on
            # generated intro/outro cards is undecided), not a runtime defect — so it
            # is an ADVISORY here: as a `slides_stubbed` degradation it marked every
            # generated showcase `degraded` (prod run 01a0ab23), a signal that fires on
            # everything and therefore on nothing. Same tiering and reason as
            # `website_review/slides.py` (prod run 01a0a054).
            #
            # Where the brief DID author lines, cards were the author's to write and
            # none arrived — that stays the `slides_stubbed` degradation.
            if not brief.get("lines"):
                # `slides_stubbed` stays False here: it is the authored-lines flag, and
                # this branch is an advisory, not a degradation (see the field).
                return ShowcaseSlidesOutput(
                    slides=[],
                    advisories=[
                        advisory(
                            SLIDES_NOT_AUTHORED,
                            "generated-mode showcase: the brief authors no lines, so "
                            "there is no `card` source and slide_service was never called",
                        )
                    ],
                )
            return ShowcaseSlidesOutput(
                slides=[],
                slides_stubbed=True,
                degradations=[
                    degradation(
                        SLIDES_STUBBED,
                        "the brief authored lines but no `card` lines, so the deck is "
                        "empty and slide_service was never called",
                    )
                ],
            )

        deliverable = ctx.config_snapshot.get("deliverable") or {}
        brand = ctx.config_snapshot.get("brand") or {}
        anchor = str(deliverable.get("title") or brand.get("name") or "Showcase")
        # Per-project brand wins over the defaults; a present ``design`` block then
        # overlays the resolved C23 palette onto the deck brand. STRICT identity
        # when no design block (``apply_design_tokens(_, None) is _``) so the deck
        # brand — and every downstream media id — is byte-identical to today.
        deck_brand = apply_design_tokens(
            {**_BRAND_DEFAULTS, **brand}, ctx.config_snapshot.get("design")
        )
        # ...and the FULL token bundle (palette + modular type scale + baseline grid
        # + font/icon set) goes to slide_service as ``tokens``, which its templates
        # consume as CSS custom properties (today's values are the fallbacks). The
        # brand channel above stays: it still carries the 4 legacy colour/font keys
        # the pptx path and the motion templates read. ``None`` without a ``design``
        # block ⇒ no ``tokens`` key ⇒ byte-identical deck payload.
        deck_tokens = tokens_for_snapshot(ctx.config_snapshot)

        async def _build_entries() -> list[DeckEntry]:
            title_id = await self._client.resolve_template_id(_TITLE_TEMPLATE_NAME)
            cta_id = await self._client.resolve_template_id(_CTA_TEMPLATE_NAME)
            needs_feature = any(shot.slide.get("template") == "bullets" for _, shot in card_shots)
            feature_id = (
                await self._client.resolve_template_id(_FEATURE_TEMPLATE_NAME)
                if needs_feature
                else ""
            )
            return [
                (
                    line_id,
                    _shotlist_deck_slide(
                        shot,
                        shot.narration,
                        anchor,
                        title_id=title_id,
                        cta_id=cta_id,
                        feature_id=feature_id,
                    ),
                )
                for line_id, shot in card_shots
            ]

        # Each card is SHOWN for its full narration length, so the motion clip is
        # rendered that long (see slide_stage._motion_durations). narrate runs before
        # this stage (dep), so its per-line durations are available; only consumed
        # when motion graphics are enabled.
        narrate = ctx.prior_output("demo_video_narrate") or {}
        secs_by_line = {
            str(c["line_id"]): float(c.get("seconds") or 0.0)
            for c in narrate.get("clips", [])
            if c.get("line_id")
        }

        rendered = await render_deck_slide_stage(
            client=self._client,
            video_client=self._video_client,
            build_entries=_build_entries,
            anchor=anchor,
            deck_brand=deck_brand,
            tokens=deck_tokens,
            config_snapshot=ctx.config_snapshot,
            deliverable=deliverable,
            run_id=str(ctx.run_id),
            stage_name=self.name,
            media_id_prefix="showcase_slide",
            make_slide=lambda slide_id, line_id, image_url: ShowcaseSlide(
                id=slide_id, line_id=line_id, image_url=image_url
            ),
            # Generous budget so all cards finish (12s clips → ~70-90s render each;
            # 10 cards at concurrency 2 ≈ 400s). Off the critical path — FTR record
            # (~900s) still dominates, so this doesn't lengthen the run.
            phase_budget_s=1100.0,
            narration_seconds_by_line=secs_by_line,
        )
        return ShowcaseSlidesOutput(
            slides=rendered.slides,
            slides_render_failed=rendered.render_failed,
            failure_class=rendered.failure_class,
            notes=rendered.notes,
        )
