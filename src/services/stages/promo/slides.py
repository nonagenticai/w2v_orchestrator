"""promo_slides — render a designed slide per script line via slide_service.

Mirrors ``lesson_record`` (one deck → per-slide PNG → re-published to the run's
artifact store) but keys each rendered slide to its ``line_id`` so promo_render
can join it 1:1. Every script line gets a slide: it is the on-screen visual for
lines the storyboard planned as ``slide``, and the **universal fallback** for
every other line — so a footage/html/ai-image asset that fails to materialise
never leaves a blank shot.

The slide content is a deterministic projection of the narration line (no extra
LLM): a short headline + the line as a supporting point, on the brand template.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

from src.clients.slide_service_client import SlideServiceClient
from src.clients.video_production_client import VideoProductionClient
from src.models.failure_class import UNKNOWN, FailureClass
from src.services.stages.base import StageHandler, StageInput, StageOutput, StageOutputWithNotes
from src.services.stages.errors import StagePreconditionError
from src.services.stages.shared.motion_slides import (
    motion_brand_params,
    motion_for_deck_slide,
)
from src.services.stages.shared.placeholder_text import slide_anchor
from src.services.stages.shared.shotlist import normalize_shotlist
from src.services.stages.shared.slide_deck import (
    _BRAND_DEFAULTS,
    _CTA_TEMPLATE_NAME,
    _FEATURE_TEMPLATE_NAME,
    _TITLE_TEMPLATE_NAME,
    _shotlist_deck_slide,
    _slide_content,
    display_headline,
)
from src.services.stages.shared.slide_stage import DeckEntry, render_deck_slide_stage

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

# Motion-slide rendering is shared with website_review_slides; these aliases keep
# the historical promo import paths (and the existing unit tests) working.
_motion_brand_params = motion_brand_params
_motion_for_deck_slide = motion_for_deck_slide


# A URL token: an http(s) URL, or a bare domain with a plausible TLD + optional
# path. Used to lift a URL out of the closing line into the CTA's ``cta_url``.
_URL_RE = re.compile(
    r"(?:https?://\S+|\bwww\.\S+|\b[\w-]+(?:\.[\w-]+)*\.(?:ai|io|com|net|org|no|app|dev|co)\b[^\s]*)",
    re.IGNORECASE,
)


def _cta_slide_content(text: str, anchor: str) -> dict[str, object]:
    """Closing CTA card content.

    If the line carries a URL, lift it into ``cta_url`` (the CTA layout renders
    it with ``overflow-wrap: anywhere`` so it wraps cleanly at the box edge,
    never mid-word — the exact defect of the old PIL preview) and use the
    remaining prose as the title. Otherwise the line is the CTA title.
    """
    text = " ".join(text.split())
    match = _URL_RE.search(text)
    if match:
        url = match.group(0)
        rest = (text[: match.start()] + text[match.end() :]).strip(" .,:;-")
        return {"eyebrow": anchor, "title": display_headline(rest, anchor), "cta_url": url}
    return {"eyebrow": anchor, "title": display_headline(text, anchor)}


class PromoSlidesInput(StageInput):
    pass


class PromoSlide(StageOutput):
    id: str
    line_id: str
    image_url: str
    # Set only when motion graphics are enabled: a branded animated clip of this
    # slide. The render supersedes the still ``image_url`` with this clip on lines
    # that would otherwise show the still; the PNG stays the degrade path.
    motion_video_url: str | None = None


class PromoSlidesOutput(StageOutputWithNotes):
    # The slide deck is the UNIVERSAL visual fallback: `promo_render` seeds
    # `image_by_line` from it, so every line with no captured screen / AI image
    # falls back to its slide. With zero slides those lines drop to a bare text
    # card. The only benign empty is "the script had no lines", which is already
    # a `produces` violation on `promo_script`.
    produces: ClassVar[tuple[str, ...]] = ("slides",)
    # REQUIRED, unlike its website_review / showcase siblings, because unlike them
    # this stage has NO legitimate empty mode: it renders one slide per script
    # line unconditionally, and the only path to an empty deck was an empty
    # `promo_script.lines` — which now fails above with a precondition naming that
    # stage, so it never reaches the gate. Every remaining way to arrive here with
    # zero slides is the soft-fail boundary in `render_deck_slide_stage` having
    # swallowed a slide_service outage.
    #
    # Degrade-only was not enough. A note sets the run's `degraded` flag but
    # leaves the terminal status `completed`, and `scripts/e2e_validate.py` never
    # reads that flag — so with `produces_required=False` a total deck outage was
    # invisible end to end: every line renders as a bare text card while
    # `promo_render` still emits a perfectly valid `artifact_url`, and no scenario
    # disagrees. This is the same shape as the `site_capture` failure that
    # completed a run holding zero screens.
    produces_required: ClassVar[bool] = True

    # Belt AND braces, deliberately, even though `produces_required` above
    # already reddens the only way this stage reaches an empty deck today. The
    # two gates answer different questions and only the second one survives a
    # refactor: `produces_required` fires on `slides == []`, which is true only
    # while the soft-fail boundary happens to leave `slides` empty on failure. A
    # partial deck (some cards rendered, the render then failed) would satisfy
    # `produces` and ship silently — the same defect its showcase /
    # website_review siblings had in full. Declaring the flag here also keeps all
    # three `render_deck_slide_stage` consumers uniformly guarded, which is what
    # `test_slide_stage_consumers_are_gated.py` enforces so a fourth cannot be
    # added without one.
    fails_when_flagged: ClassVar[tuple[str, ...]] = ("slides_render_failed",)

    slides: list[PromoSlide]
    #: True ONLY when a deck render was attempted and failed (never a stub — this
    #: stage has no legitimate empty mode; see `produces_required` above).
    slides_render_failed: bool = False
    #: The FAULT half of the same outage, carried through from the exception the
    #: soft-fail boundary swallowed (`shared/slide_stage.py` stamps
    #: `attribute_exception(exc)` on its result). Copied onto the output because
    #: `dag_executor._enforce_failure_flags` reads only the DUMPED mapping — the
    #: exception object is long gone by then, so this field is the only way the
    #: attribution reaches `StageReportedFailureError` and, from there, the
    #: persisted `runs.failure_class`. `unknown` on every healthy render.
    failure_class: FailureClass = UNKNOWN
    notes: list[str] = []  # noqa: RUF012 — pydantic copies field defaults per-instance


class PromoSlidesStage(StageHandler[PromoSlidesInput, PromoSlidesOutput]):
    name = "promo_slides"
    input_model = PromoSlidesInput
    output_model = PromoSlidesOutput

    def __init__(
        self,
        client: SlideServiceClient | None = None,
        video_client: VideoProductionClient | None = None,
    ) -> None:
        self._client = client or SlideServiceClient()
        self._video_client = video_client or VideoProductionClient()

    async def handle(
        self,
        payload: PromoSlidesInput,
        ctx: RunContext,
    ) -> PromoSlidesOutput:
        script = ctx.prior_output("promo_script")
        if not script:
            raise RuntimeError("promo_script output missing")
        lines = script["lines"]
        if not lines:
            # Fail HERE, naming promo_script, rather than returning an empty deck
            # and letting the `produces_required` gate fail this stage: the deck
            # is unbuildable because its spine is missing, and an operator sent to
            # promo_slides would go looking for a slide_service outage that never
            # happened. `promo_script.produces = ("lines",)` degrades rather than
            # fails, so this path is reachable.
            raise StagePreconditionError(
                "promo_script produced no lines — there is no script to render a slide deck from",
                missing=["promo_script.lines"],
            )

        deliverable = ctx.config_snapshot.get("deliverable") or {}
        brand = ctx.config_snapshot.get("brand") or {}
        brief = ctx.config_snapshot.get("brief") or {}
        # Never a hardcoded house brand: an unbranded run renders no eyebrow.
        anchor = slide_anchor(ctx.config_snapshot)
        shots = normalize_shotlist(brief)
        deck_brand = {**_BRAND_DEFAULTS, **brand}  # per-project brand wins

        async def _build_entries() -> list[DeckEntry]:
            title_id = await self._client.resolve_template_id(_TITLE_TEMPLATE_NAME)
            cta_id = await self._client.resolve_template_id(_CTA_TEMPLATE_NAME)
            if shots:
                # Authored shotlist: explicit slide content per slide-shot; URL
                # shots keep the title card as their capture-fail fallback. Every
                # line still gets one deck entry (universal-fallback contract).
                needs_feature = any(
                    s.kind == "slide" and s.slide.get("template") == "bullets" for s in shots
                )
                feature_id = (
                    await self._client.resolve_template_id(_FEATURE_TEMPLATE_NAME)
                    if needs_feature
                    else ""
                )
                shot_by_line = {s.line_id: s for s in shots}
                return [
                    (
                        str(line["id"]),
                        _shotlist_deck_slide(
                            shot_by_line.get(str(line["id"])),
                            str(line["text"]),
                            anchor,
                            title_id=title_id,
                            cta_id=cta_id,
                            feature_id=feature_id,
                        ),
                    )
                    for line in lines
                ]
            # Closing line → CTA layout (url-safe); all others → title card.
            # A single-line promo is not a CTA, so only route the last line of
            # a multi-line deck.
            last_index = len(lines) - 1
            entries: list[DeckEntry] = []
            for index, line in enumerate(lines):
                text = str(line["text"])
                if index == last_index and len(lines) > 1:
                    entries.append(
                        (
                            str(line["id"]),
                            {
                                "template_id": cta_id,
                                "role": "cta",
                                "content": _cta_slide_content(text, anchor),
                            },
                        )
                    )
                else:
                    entries.append(
                        (
                            str(line["id"]),
                            {
                                "template_id": title_id,
                                "role": "content",
                                "content": _slide_content(text, anchor),
                            },
                        )
                    )
            return entries

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
            media_id_prefix="promo_slide",
            make_slide=lambda slide_id, line_id, image_url: PromoSlide(
                id=slide_id, line_id=line_id, image_url=image_url
            ),
            # promo_slides stage timeout is 900s; keep the motion phase under it
            # with headroom for the stills deck, so an oversized deck degrades to
            # stills gracefully instead of hard-failing the stage.
            phase_budget_s=720.0,
            # promo_render never lays the animated variant (its text builds in
            # from an empty frame under narration that starts at the cut — see
            # ``PromoRenderStage.build_shots``), so it is not recorded either: on
            # run 01a10fcf seven such clips were paid for, dropped, and held the
            # recorder slots the Ken Burns stills then starved for.
            attach_motion=False,
        )
        return PromoSlidesOutput(
            slides=rendered.slides,
            slides_render_failed=rendered.render_failed,
            failure_class=rendered.failure_class,
            notes=rendered.notes,
        )
