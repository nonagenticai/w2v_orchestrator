"""lesson_record — render the lesson slides to per-slide PNG images.

slide-service renders each slide to a PNG via HTML/CSS + headless Chromium
(``html_slide_renderer``; LibreOffice is used only for the optional PPTX/PDF
export, not this PNG path). This stage builds one deck for the whole lesson —
passing the project **brand** so slides are on-brand, and each slide's hero
**image** (the on-brand picture ``ai_image_gen`` produced for that slide) so the
slide shows a title + bullets + a relevant image instead of an empty
placeholder — then downloads each rendered slide PNG and re-publishes it to the
run's artifact store so ``lesson_compile`` can hand the image URL to the
video-compile service. The narration audio (``lesson_narrate``) is muxed onto
these stills by the compositor — that is where the real video is born.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from src.clients.slide_service_client import SlideServiceClient
from src.clients.video_production_client import VideoProductionClient
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
)
from src.services.stages.shared.motion_slides import (
    SLIDE_CAPTURE_SCALE,
    quality_tier,
    render_deck_motion_clips,
)
from src.services.stages.shared.slide_deck import render_deck_slide_images

# The SAME per-card sizing demo_video gets via ``render_deck_slide_stage`` (its only
# other caller). lesson_record does not go through that helper — it drives
# ``render_deck_motion_clips`` directly — so it reuses the sizing function itself
# rather than reimplementing the clamp/tail rule a second time.
from src.services.stages.shared.slide_stage import _motion_durations as motion_durations
from src.services.stages.shared.speech_rate import resolve_natural_wpm
from src.services.stages.shared.time_budget import DEFAULT_WPM_NATURAL
from src.services.stages.shot_presign import refresh_shot_asset_urls
from src.services.tts_storage import deterministic_media_id

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

# Template used for lesson slides: title + a bulleted body. Resolved by stable
# name to the seeded system template's id at runtime.
_SLIDE_TEMPLATE_NAME = "feature_spotlight"


def estimated_narration_seconds_by_slide(
    paragraph_ids_by_slide: Mapping[str, Sequence[str]],
    words_by_paragraph: Mapping[str, int],
    wpm: float,
) -> dict[str, float]:
    """How long each slide is on screen, estimated at the run's MEASURED speech rate.

    Pure arithmetic over already-extracted primitives: which paragraphs each slide
    covers, how many words each paragraph is, and the voice's rate. Digging those
    two mappings out of the untyped stage outputs is the CALLER's job (see
    :meth:`LessonRecordStage.handle`) — keeping this function free of raw stage
    payloads is what lets it be typed at all, and what makes it trivially testable.

    ``lesson_record`` renders the picture BEFORE the narration exists: its blueprint
    deps are ``(lesson_slides, ai_image_gen)`` and ``lesson_narrate`` is a SIBLING
    that becomes eligible in the same scheduler round (``blueprints/lesson.py`` —
    the concurrency of the two long poles is deliberate). So there are no measured
    clip lengths to read here, and ``lesson_script``'s per-paragraph ``seconds`` is
    the model's own guess ("has never been checked against anything",
    ``lesson/script.py`` ``LessonScriptOutput``). What IS countable is the narration
    WORD COUNT, and what IS measured is the voice's rate — ``speech_rate`` probes
    tts_backend for the real wpm of the run's voice. words / wpm is therefore a
    measured-rate estimate, not a hardcoded constant.

    ``max``, not ``sum``: ``lesson_compile`` emits one shot PER PARAGRAPH and lays
    the slide's single motion clip over EACH of that slide's paragraph lines
    (``lesson/compile.py``, the ``motion_by_slide`` loop). Every one of those shots
    is held for its OWN paragraph's audio, so the clip has to cover the LONGEST
    paragraph of the slide, not their total.

    Slides with no countable narration are absent from the result — that is the
    "unknown duration" signal :func:`motion_durations` already understands.
    """
    if wpm <= 0.0:
        return {}
    seconds_by_slide: dict[str, float] = {}
    for slide_id, paragraph_ids in paragraph_ids_by_slide.items():
        longest = max(
            (words_by_paragraph.get(pid, 0) / wpm * 60.0 for pid in paragraph_ids),
            default=0.0,
        )
        if longest > 0.0:
            seconds_by_slide[slide_id] = longest
    return seconds_by_slide


class LessonRecordInput(StageInput):
    pass


class LessonRecording(StageOutput):
    id: str
    slide_id: str
    image_url: str
    # Opt-in animated variant of this slide (motion_graphics enabled). None when
    # off or when this slide's clip soft-failed — lesson_compile then keeps the PNG.
    motion_video_url: str | None = None


class LessonRecordOutput(StageOutputWithNotes):
    # ``notes`` (from StageOutputWithNotes) surface a dropped motion clip so the run
    # degrades (diagnosable) instead of silently shipping a slide with
    # ``motion_video_url=None``. runs/_shared.py auto-translates non-empty notes on a
    # completed stage into a ``degraded`` run.
    recordings: list[LessonRecording]


class LessonRecordStage(StageHandler[LessonRecordInput, LessonRecordOutput]):
    name = "lesson_record"
    input_model = LessonRecordInput
    output_model = LessonRecordOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("tts"),)

    def __init__(
        self,
        client: SlideServiceClient | None = None,
        video_client: VideoProductionClient | None = None,
    ) -> None:
        self._client = client or SlideServiceClient()
        self._video_client = video_client or VideoProductionClient()

    async def handle(
        self,
        payload: LessonRecordInput,
        ctx: RunContext,
    ) -> LessonRecordOutput:
        slides_output = ctx.prior_output("lesson_slides")
        if not slides_output:
            raise RuntimeError("lesson_slides output missing")
        slides = slides_output["slides"]
        if not slides:
            return LessonRecordOutput(recordings=[])

        run_id = str(ctx.run_id)
        deliverable = ctx.config_snapshot.get("deliverable") or {}
        brand = ctx.config_snapshot.get("brand") or {}
        template_id = await self._client.resolve_template_id(_SLIDE_TEMPLATE_NAME)

        # Hero image per slide, keyed by slide id (ai_image_gen emits line_id =
        # slide id via lesson_slides' prompts). Tolerant of a skipped/empty
        # ai_image_gen: a slide with no image still renders title + bullets.
        image_gen = ctx.prior_output("ai_image_gen") or {}
        image_by_slide = {
            str(img["line_id"]): str(img["image_url"])
            for img in image_gen.get("images", [])
            if img.get("line_id") and img.get("image_url")
        }
        # Re-sign each hero image to a fresh TTL right before slide_service fetches
        # it (the consumption-side twin of the compile tail / course_stitch #207):
        # ai_image_gen's image_url is a presigned MinIO URL with a 1h TTL frozen in
        # its stage output, but slide_service downloads it to composite the slide.
        # A lesson sub-run paused/iterated >1h would otherwise hand it a dead SigV4
        # sig → broken/blank hero images. Best-effort: a failure keeps the original.
        if image_by_slide:
            slide_ids = list(image_by_slide)
            shots = [{"image_url": image_by_slide[sid]} for sid in slide_ids]
            await refresh_shot_asset_urls(shots)
            image_by_slide = {
                sid: shot["image_url"] for sid, shot in zip(slide_ids, shots, strict=True)
            }

        deck_payload = {
            "title": str(deliverable.get("title") or "Lesson"),
            "brand": dict(brand),
            "slides": [
                {
                    "template_id": template_id,
                    "role": "content",
                    "content": {
                        "title": str(slide["title"]),
                        "bullets": [str(b) for b in slide["bullets"]],
                        **(
                            {"image_url": image_by_slide[str(slide["id"])]}
                            if str(slide["id"]) in image_by_slide
                            else {}
                        ),
                    },
                }
                for slide in slides
            ],
            # Slide PNGs at the run's resolution tier so stills match the motion
            # clips instead of being a 1080p upscale at lesson_compile.
            "render_scale": SLIDE_CAPTURE_SCALE,
        }
        image_urls = await render_deck_slide_images(
            self._client,
            deck_payload=deck_payload,
            run_id=run_id,
            stage_name="lesson_record",
            count=len(slides),
        )
        recordings: list[LessonRecording] = [
            LessonRecording(
                id=deterministic_media_id("lesson_record", str(slide["id"]), index),
                slide_id=str(slide["id"]),
                image_url=image_urls[index],
            )
            for index, slide in enumerate(slides)
        ]

        # Motion graphics (opt-in): render each slide deck entry as a branded
        # animated clip alongside the still PNG, reusing the promo/website_review
        # machinery. Gated on the same motion_graphics block (a per-run
        # `motion_graphics.slides: false` keeps slides still while leaving the
        # standalone motion_graphics_gen stage on). The animated clip only
        # animates the title + bullets text; the hero image stays still. Per-clip
        # soft-fail to the PNG — lesson_compile keeps the still image when a clip
        # is absent.
        motion_cfg = ctx.config_snapshot.get("motion_graphics")
        notes: list[str] = []
        if recordings and motion_cfg and motion_cfg.get("slides", True) is not False:
            # Size each clip to how long its card is actually on screen. Without
            # this every clip fell back to the template's hardcoded
            # RENDER_DURATION_MS (feature_spotlight: 1500 + n*420 + 1200 ms ≈ 4s)
            # which has no relationship to the narration, so lesson_compile laid a
            # ~4s clip over a ~11.5s narrated shot and the compiler's av_coverage
            # guard failed the whole job (live run 2026-09-10 13:35Z: audio outran
            # picture by ~7.5s). The probe never fails a run — it returns the
            # default wpm on any error — and an unknown-length card is simply
            # absent from the map, keeping the old template default for it.
            script_output = ctx.prior_output("lesson_script")
            paragraphs = (script_output or {}).get("paragraphs") or []
            narration_seconds_by_slide: dict[str, float] = {}
            if paragraphs:
                # Extraction lives HERE, with the rest of this method's
                # untyped-stage-output handling (cf. ``image_by_slide`` above), so
                # ``estimated_narration_seconds_by_slide`` stays a pure, fully typed
                # function over primitives instead of taking raw stage payloads.
                words_by_paragraph = {
                    str(p["id"]): len(str(p.get("text") or "").split())
                    for p in paragraphs
                    if p.get("id")
                }
                paragraph_ids_by_slide = {
                    str(slide["id"]): [str(pid) for pid in (slide.get("paragraph_ids") or [])]
                    for slide in slides
                }
                # Gated on there being narration to count: a run with no countable
                # script must not pay for (or pollute the memo cache with) a probe
                # whose answer it cannot use.
                wpm = await resolve_natural_wpm(ctx, default=DEFAULT_WPM_NATURAL)
                narration_seconds_by_slide = estimated_narration_seconds_by_slide(
                    paragraph_ids_by_slide, words_by_paragraph, wpm
                )
            recordings, notes = await self._render_motion_slides(
                recordings,
                deck_payload["slides"],
                dict(brand),
                motion_cfg,
                deliverable,
                run_id,
                ctx.config_snapshot.get("theme") or {},
                tier=quality_tier(ctx.config_snapshot),
                narration_seconds_by_slide=narration_seconds_by_slide,
            )

        return LessonRecordOutput(recordings=recordings, notes=notes)

    async def _render_motion_slides(
        self,
        recordings: list[LessonRecording],
        deck_slides: list[dict[str, Any]],
        deck_brand: dict[str, object],
        motion_cfg: dict[str, Any],
        deliverable: dict[str, Any],
        run_id: str,
        deck_theme: dict[str, object] | None = None,
        tier: str = "1080p",
        narration_seconds_by_slide: dict[str, float] | None = None,
    ) -> tuple[list[LessonRecording], list[str]]:
        """Render a branded motion clip per slide and attach ``motion_video_url``.

        ``recordings`` and ``deck_slides`` are index-aligned (one per slide).
        Per-clip soft-fail: a slide whose clip fails keeps ``motion_video_url=None``
        and lesson_compile falls back to its still PNG. A failure of the whole
        batch is swallowed so the still-PNG lesson still compiles. Returns the
        (possibly motion-annotated) recordings plus a list of degradation ``notes``
        — one per slide that fell back to its still image — so the run surfaces the
        drop as ``degraded`` instead of silently shipping ``motion_video_url=None``.

        ``narration_seconds_by_slide`` sizes each clip to its card's on-screen time
        (see :func:`estimated_narration_seconds_by_slide`). ``None``/empty keeps the
        template's own default length — the pre-fix behaviour, retained only for the
        case where nothing about the narration is countable."""
        line_ids = [r.slide_id for r in recordings]
        duration_ms_by_index = (
            motion_durations(line_ids, narration_seconds_by_slide)
            if narration_seconds_by_slide
            else None
        )
        try:
            result = await render_deck_motion_clips(
                self._video_client,
                line_ids=line_ids,
                deck_slides=deck_slides,
                deck_brand=deck_brand,
                motion_cfg=motion_cfg,
                deliverable=deliverable,
                run_id=run_id,
                stage_name=self.name,
                deck_theme=deck_theme,
                tier=tier,
                # Reconcile the phase budget to this stage's timeout. Invariant:
                # phase_budget_s (780) >= _MOTION_RECORD_TIMEOUT_S (600) so an
                # in-flight clip within its own record timeout is never guillotined
                # by the phase budget, AND 780 + stills headroom < lesson_record
                # stage timeout (900s, blueprints/lesson.py). The global default
                # (450s) is left alone — website_review_slides has a 600s stage
                # timeout and a global bump would push it over.
                phase_budget_s=780,
                duration_ms_by_index=duration_ms_by_index,
            )
        except Exception as exc:
            # Returning `[]` here contradicted this method's own contract two docstrings
            # up: the per-slide path emits one note per fallback precisely so the run
            # renders `degraded` (`runs/_shared.py` derives it from non-empty notes), but
            # the WHOLE-batch failure — the worse outcome, every slide falls back — emitted
            # none, so a customer who enabled and paid for motion graphics received a deck
            # of static PNGs presented as a clean success. Soft-fail is still correct; a
            # silent soft-fail is not. One note per slide, matching the per-slide shape.
            log.error("lesson_record: motion render failed for the whole batch", exc_info=True)
            return recordings, [
                f"lesson_record: slide {rec.slide_id} kept still image; "
                f"motion render failed for the whole batch ({exc})"
                for rec in recordings
            ]
        url_by_index = result.urls
        updated = [
            rec.model_copy(update={"motion_video_url": url_by_index[index]})
            if index in url_by_index
            else rec
            for index, rec in enumerate(recordings)
        ]
        reason = "phase-budget cancelled" if result.budget_exceeded else "soft-failed"
        notes = [
            f"lesson_record: slide {rec.slide_id} kept still image; motion clip {reason}"
            for index, rec in enumerate(recordings)
            if index not in url_by_index
        ]
        return updated, notes
