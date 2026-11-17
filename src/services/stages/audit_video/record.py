"""AuditRecordStageBase — render the audit slides to per-slide PNG images.

Opt-in (``brief.generate_video``). Builds one branded deck for the whole audit
(title + bullets per slide, on the seeded ``feature_spotlight`` template) via
``slide_service``, downloads each rendered slide PNG, and re-publishes it to the
run's artifact store so ``<type>_compile`` can hand the image URL to the
video-compile service. Mirrors ``lesson_record`` minus the ai_image_gen hero
image / motion machinery (an audit deck is text slides). The narration audio is
muxed onto these stills by the compositor. Skips cleanly when the video variant
is off.

Per-type subclass sets ``name`` + ``deck_title`` (the deck-title fallback).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from src.clients.slide_service_client import SlideServiceClient
from src.services.stages.audit_video.gate import require_audit_video
from src.services.stages.audit_video.models import (
    AuditRecording,
    AuditRecordInput,
    AuditRecordOutput,
)
from src.services.stages.base import StageHandler
from src.services.stages.errors import StagePreconditionError
from src.services.stages.shared.motion_slides import SLIDE_CAPTURE_SCALE
from src.services.stages.shared.slide_deck import render_deck_slide_images
from src.services.tts_storage import deterministic_media_id

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

# Same title + bulleted-body layout lesson_record uses.
_SLIDE_TEMPLATE_NAME = "feature_spotlight"


class AuditRecordStageBase(StageHandler[AuditRecordInput, AuditRecordOutput]):
    """Shared audit-record stage. Per-type subclass sets ``name`` + ``deck_title``."""

    input_model = AuditRecordInput
    output_model = AuditRecordOutput
    #: Deck-title fallback when the deliverable carries no title.
    deck_title: ClassVar[str]

    def __init__(self, client: SlideServiceClient | None = None) -> None:
        self._client = client or SlideServiceClient()

    async def handle(self, payload: AuditRecordInput, ctx: RunContext) -> AuditRecordOutput:
        require_audit_video(ctx)
        prefix = self.name.removesuffix("_record")
        slides_output = ctx.prior_output(f"{prefix}_slides")
        if not slides_output:
            raise RuntimeError(f"{prefix}_slides output missing")
        slides = slides_output["slides"]
        if not slides:
            # Was `return AuditRecordOutput(recordings=[])` — a silent
            # short-circuit that reported `completed` while handing `<type>_compile`
            # zero images, so the audit video muxed its narration over nothing and
            # nothing downstream disagreed.
            #
            # Fail HERE naming the upstream rather than letting the
            # `produces_required` gate on AuditRecordOutput catch the empty return:
            # `AuditSlidesOutput.produces` is degrade-only, so an empty deck gets
            # this far, and the operator needs to be sent to `<prefix>_slides`
            # (whose own comment says empty is drift, not a mode) instead of
            # hunting a slide_service outage in this stage. The gate stays as the
            # backstop for every other way this stage could produce nothing.
            raise StagePreconditionError(
                f"{prefix}_slides produced no slides — there is no deck to render "
                f"into audit recordings",
                missing=[f"{prefix}_slides.slides"],
            )

        run_id = str(ctx.run_id)
        deliverable = ctx.config_snapshot.get("deliverable") or {}
        brand = ctx.config_snapshot.get("brand") or {}
        template_id = await self._client.resolve_template_id(_SLIDE_TEMPLATE_NAME)

        deck_payload = {
            "title": str(deliverable.get("title") or self.deck_title),
            "brand": dict(brand),
            "slides": [
                {
                    "template_id": template_id,
                    "role": "content",
                    "content": {
                        "title": str(slide["title"]),
                        "bullets": [str(b) for b in slide["bullets"]],
                    },
                }
                for slide in slides
            ],
            # Slide PNGs at the run's resolution tier so stills match the compile
            # canvas instead of being a 1080p upscale.
            "render_scale": SLIDE_CAPTURE_SCALE,
        }
        image_urls = await render_deck_slide_images(
            self._client,
            deck_payload=deck_payload,
            run_id=run_id,
            stage_name=self.name,
            count=len(slides),
        )
        recordings = [
            AuditRecording(
                id=deterministic_media_id(self.name, str(slide["id"]), index),
                slide_id=str(slide["id"]),
                image_url=image_urls[index],
            )
            for index, slide in enumerate(slides)
        ]
        return AuditRecordOutput(recordings=recordings)
