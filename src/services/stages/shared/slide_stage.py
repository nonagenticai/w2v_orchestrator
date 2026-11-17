"""Config-driven deck-slide stage skeleton.

``promo_slides`` / ``website_review_slides`` / ``showcase_slides`` are the same
stage with different content sources: each builds a list of ``(line_id, deck
slide)`` entries, renders them to still PNGs via slide_service, and (opt-in)
renders a branded motion clip per slide via video_production. Only the entry
construction, the slide model, the media-id prefix and the motion budget differ.

This module owns that identical skeleton — deck-render + per-line model build +
soft-fail + motion attach — so each stage supplies ONLY its differing config via
:func:`render_deck_slide_stage`. The deterministic deck-entry projection stays in
``slide_deck``; the still/motion rendering primitives stay in ``slide_deck`` /
``motion_slides``. This is a pure refactor: the produced deck payloads, media
ids, motion clips and output values are unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NamedTuple

from src.models.failure_class import UNKNOWN, FailureClass, attribute_exception
from src.services.stages.base import StageOutput
from src.services.stages.shared.cadence import build_cadence_spec
from src.services.stages.shared.motion_slides import (
    SLIDE_CAPTURE_SCALE,
    quality_tier,
    render_deck_motion_clips,
    token_fp_for_snapshot,
)
from src.services.stages.shared.slide_deck import render_deck_slide_images
from src.services.tts_storage import deterministic_media_id

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.clients.slide_service_client import SlideServiceClient
    from src.clients.video_production_client import VideoProductionClient

log = logging.getLogger(__name__)

# A deck entry: the join id for the line + the deterministic slide_service deck
# dict (template_id / role / content) the shared projection produced for it.
DeckEntry = tuple[str, dict[str, object]]


class DeckRenderResult[SlideT: StageOutput](NamedTuple):
    """What :func:`render_deck_slide_stage` produced, and whether it failed.

    ``slides == []`` from this helper is ALWAYS a defect — every caller guards the
    "nothing to render" case with its own early return and only reaches here with
    a non-empty content source. But the soft-fail boundary below turns a render
    outage into exactly the same shape a legitimately empty deck would have
    (``[]`` plus a note), and note prose is unfilterable, so the two readings of
    ``slides == []`` are not tellable apart by a consumer looking at ``slides``.

    ``render_failed`` is the machine-readable half. It is modelled on
    ``action_reel/graphics_design.py``'s ``cards_render_failed`` / ``cards_stubbed``
    pair for the same reason: a consumer that wants to retry, alert on, or count
    render outages reads the flag and never pattern-matches the note. The caller
    carries it onto its own output as ``slides_render_failed``, and records why its
    own early return is empty (a ``slides_not_authored`` advisory, or
    ``slides_stubbed`` when authored lines carried no cards) — so the two ways a deck
    ends up empty stay distinguishable in the persisted stage output.

    ``failure_class`` is the FAULT half of the same answer, and it exists because
    the soft-fail boundary below is where the original exception DIES. Everything
    downstream of it sees only ``render_failed`` plus prose: the exception object
    never reaches ``run_dispatcher``'s persistence site, so
    ``attribute_exception`` — which reads the ``failure_class`` that
    ``upstream.guarded`` stamped where the HTTP status actually was — has nothing
    left to read and the run records ``unknown``. A slide_service 503 is
    ``external`` and was knowable; recording it as ``unknown`` is a fact thrown
    away here, not a fact we never had.

    Attribution is computed at the boundary and carried as the CLASS, not as the
    exception: the consumer is a caller's pydantic ``StageOutput`` field, which a
    ``BaseException`` cannot be persisted into. It defaults to ``unknown`` — the
    value a consumer that ignores this field would have recorded anyway — so it
    is additive at every existing construction site, and it asserts nothing on
    the success path or on a motion-only failure (see below: a motion failure is
    not a deck failure, and attributing the deck off it would be exactly the
    wrong-attribution ``models/failure_class.py`` calls worse than none).
    """

    slides: list[SlideT]
    notes: list[str]
    render_failed: bool
    failure_class: FailureClass = UNKNOWN


def _cadence_master_fps(config_snapshot: dict[str, Any]) -> int | None:
    """Resolve the run's master cadence fps for motion clips (opt-in).

    Presence-is-opt-in: an absent ``cadence`` block yields no spec ⇒ ``None`` ⇒
    :func:`render_deck_motion_clips` keeps its default fps (byte-identical)."""
    spec = build_cadence_spec(config_snapshot)
    return spec.target_fps if spec else None


def _motion_durations(
    line_ids: list[str], narration_seconds_by_line: dict[str, float]
) -> dict[int, int]:
    """Per-card motion clip length (ms) from each card's narration seconds.

    Each card is on screen for its full narration, so the motion clip is rendered
    that long (+ a 1s tail) to keep its animated background drifting the whole
    time. Clamped to [4.5s, 12s]: 12s renders fast enough that every card still
    gets a clip, and a longer card freeze-holds its settled last frame. Cards with
    no known duration are absent (fall back to the template's default length).
    """
    return {
        i: int(min(max(narration_seconds_by_line.get(lid, 0.0) + 1.0, 4.5), 12.0) * 1000)
        for i, lid in enumerate(line_ids)
        if narration_seconds_by_line.get(lid, 0.0) > 0.0
    }


async def render_deck_slide_stage[SlideT: StageOutput](
    *,
    client: SlideServiceClient,
    video_client: VideoProductionClient,
    build_entries: Callable[[], Awaitable[list[DeckEntry]]],
    anchor: str,
    deck_brand: dict[str, object],
    config_snapshot: dict[str, Any],
    deliverable: dict[str, Any],
    run_id: str,
    stage_name: str,
    media_id_prefix: str,
    make_slide: Callable[[str, str, str], SlideT],
    phase_budget_s: float,
    narration_seconds_by_line: dict[str, float] | None = None,
    tokens: dict[str, str] | None = None,
    attach_motion: bool = True,
) -> DeckRenderResult[SlideT]:
    """Render a deck to per-line slide models, then (opt-in) attach motion clips.

    ``build_entries`` returns the index-aligned ``(line_id, deck slide)`` list for
    this stage; it runs inside the soft-fail boundary so a slide_service template
    resolution hiccup degrades the run (empty deck) instead of aborting it — the
    same contract all three slide stages had. ``make_slide(id, line_id, image_url)``
    mints the stage's own slide model.

    ``tokens`` (``design_system.tokens_for_snapshot``) is the run's resolved C23
    design-token bundle, forwarded to slide_service on the deck submit. ``None``
    (the default, and the value for any run with no ``design`` block) omits the key
    entirely — a stage that does not pass it is byte-identical to today.

    Returns a :class:`DeckRenderResult`. Callers MUST propagate its
    ``render_failed`` onto their own output rather than re-deriving it from
    ``not slides`` — see that class for why the two are not the same question.
    """
    slides: list[SlideT] = []
    notes: list[str] = []
    line_ids: list[str] = []
    deck_slides: list[dict[str, object]] = []
    # Soft-fail: slides degrade the video (compile falls the line back), they never
    # fail the whole run — mirrors site_capture / ai_image_gen soft-fail semantics.
    try:
        entries = await build_entries()
        if not entries:
            # Every caller reaches this helper only with a non-empty content
            # source, so an empty entry list is a projection defect, not a mode.
            # It is folded into `render_failed` rather than being returned as a
            # bare `[]` because from the caller's side the two are the same
            # thing: the deck it asked for does not exist.
            raise ValueError("deck projection produced no entries from a non-empty content source")
        line_ids = [line_id for line_id, _ in entries]
        deck_slides = [slide for _, slide in entries]
        deck_payload = {
            "title": anchor,
            "brand": deck_brand,
            "slides": deck_slides,
            # Slide PNGs at the run's resolution tier so the stills match the motion
            # clips (2K/4K) instead of being a 1080p upscale at compile.
            "render_scale": SLIDE_CAPTURE_SCALE,
        }
        # C23 design/motion fingerprint, computed once from the run snapshot and
        # threaded into both the still-image and the motion-clip render paths.
        # ``None`` (no design/motion block) → conditional omission keeps every
        # media id byte-identical to today. The stable-join id below is NOT folded.
        token_fp = token_fp_for_snapshot(config_snapshot)
        image_urls = await render_deck_slide_images(
            client,
            deck_payload=deck_payload,
            run_id=run_id,
            stage_name=stage_name,
            count=len(line_ids),
            token_fp=token_fp,
            tokens=tokens,
        )
        slides = [
            make_slide(
                deterministic_media_id(media_id_prefix, line_id, index),
                line_id,
                image_urls[index],
            )
            for index, line_id in enumerate(line_ids)
        ]
    except Exception as exc:  # noqa: BLE001 -- soft-fail: degrade, don't abort the run
        log.warning("%s: deck render failed: %s", stage_name, exc)
        notes.append(f"slide render failed: {exc}")
        # `render_failed=True`, not "the caller can see slides == []". A note is
        # NOT observable: `runs/_shared.py` turns notes into a `degraded` flag,
        # the run's terminal status stays `completed`, and nothing that reads a
        # finished run looks at the flag. The structural flag is what a consumer
        # can actually filter, alert or retry on.
        #
        # This is the LAST point at which `exc` exists. `attribute_exception`
        # walks its `__cause__` chain for the class `upstream.guarded` stamped
        # where the status was in hand; it returns `unknown` for anything it
        # cannot justify, so this never manufactures an attribution.
        return DeckRenderResult(
            slides, notes, render_failed=True, failure_class=attribute_exception(exc)
        )

    # Motion graphics (opt-in): render each slide as a branded animated clip
    # alongside the still PNG. Reuses the motion_graphics gate; a per-run
    # `motion_graphics.slides: false` keeps slides still. Soft-fail to the PNG.
    # `attach_motion=False` is the STAGE's own opt-out: a product whose render
    # never lays the animated variant must not pay the recorder for it (promo).
    motion_cfg = config_snapshot.get("motion_graphics")
    if attach_motion and slides and motion_cfg and motion_cfg.get("slides", True) is not False:
        try:
            slides = await _attach_motion_clips(
                video_client,
                slides=slides,
                line_ids=line_ids,
                deck_slides=deck_slides,
                deck_brand=deck_brand,
                motion_cfg=motion_cfg,
                deliverable=deliverable,
                run_id=run_id,
                stage_name=stage_name,
                deck_theme=config_snapshot.get("theme") or {},
                tier=quality_tier(config_snapshot),
                phase_budget_s=phase_budget_s,
                narration_seconds_by_line=narration_seconds_by_line,
                master_fps=_cadence_master_fps(config_snapshot),
                config_snapshot=config_snapshot,
            )
        except Exception as exc:  # noqa: BLE001 -- soft-fail: keep the still slides
            log.warning("%s: motion render failed: %s", stage_name, exc)
            notes.append(f"motion slide render failed: {exc}")

    # A motion failure is NOT a deck failure: the still PNGs shipped and compile
    # falls the line back to them, so `render_failed` stays False here.
    return DeckRenderResult(slides, notes, render_failed=False)


async def _attach_motion_clips[SlideT: StageOutput](
    video_client: VideoProductionClient,
    *,
    slides: list[SlideT],
    line_ids: list[str],
    deck_slides: list[dict[str, object]],
    deck_brand: dict[str, object],
    motion_cfg: dict[str, Any],
    deliverable: dict[str, Any],
    run_id: str,
    stage_name: str,
    deck_theme: dict[str, object] | None,
    tier: str,
    phase_budget_s: float,
    narration_seconds_by_line: dict[str, float] | None,
    master_fps: int | None = None,
    config_snapshot: dict[str, Any] | None = None,
) -> list[SlideT]:
    """Render a branded motion clip per slide and attach ``motion_video_url``.

    Slides are index-aligned to ``deck_slides`` / ``line_ids``. Per-clip soft-fail:
    a slide whose clip fails keeps ``motion_video_url=None`` and the render falls
    back to its still PNG. When ``narration_seconds_by_line`` is given, each clip is
    rendered to its card's narration length (see :func:`_motion_durations`).
    ``config_snapshot`` carries the run's design/motion author-blocks through to
    the C23 fingerprint + motion-signature seam (inert when neither is present).
    """
    duration_ms_by_index = (
        _motion_durations(line_ids, narration_seconds_by_line)
        if narration_seconds_by_line is not None
        else None
    )
    # This helper has no notes channel (returns bare slides); read only `.urls`
    # and keep the existing per-clip soft-fail behaviour (a missing index keeps
    # its still PNG). The budget_exceeded / failed_indices diagnostics are surfaced
    # by callers that carry a notes channel (e.g. lesson_record).
    url_by_index = (
        await render_deck_motion_clips(
            video_client,
            line_ids=line_ids,
            deck_slides=deck_slides,
            deck_brand=deck_brand,
            motion_cfg=motion_cfg,
            deliverable=deliverable,
            run_id=run_id,
            stage_name=stage_name,
            deck_theme=deck_theme,
            tier=tier,
            phase_budget_s=phase_budget_s,
            duration_ms_by_index=duration_ms_by_index,
            master_fps=master_fps,
            config_snapshot=config_snapshot,
        )
    ).urls
    return [
        slide.model_copy(update={"motion_video_url": url_by_index[index]})
        if index in url_by_index
        else slide
        for index, slide in enumerate(slides)
    ]
