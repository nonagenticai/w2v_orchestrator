"""Video render — assemble narrated, image-backed shots and compile them.

Joins the storyboard (one shot per script line) with ``narrate`` audio (by
``line_id``) and ``ai_image_gen`` images (round-robin, since images carry no
line key), then calls the video-compile service for the final artifact.

When the run was grounded in real media — operator-**pinned**, document-
**extracted**, or auto-**discovered** images/clips recorded by ``source_ingest``
/ ``media_discovery`` — those (already compiler-safe, re-published) URLs are laid
over the leading shot lines as keyed visuals that win over the round-robin AI
pool, so curated/real media actually lands on screen (mirrors ``promo_render``).
Real-media **images** take the leading lines; **video** clips take their own
lines (the compositor honours ``video_url`` over ``image_url``), so an image and
a clip never collide on one line. Runs without sources are byte-for-byte
unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.services.captions import caption_text_by_line
from src.services.stages.base import ArtifactUrl, StageInput, StageOutputWithNotes
from src.services.stages.primitives.compile_stage import (
    CompileStageBase,
    captions_mode,
    playable_duration_seconds,
    resolution_tags,
)
from src.services.stages.shared.duration_conformance import (
    DurationConformance,
    duration_conformance_advisories,
    duration_conformance_degradations,
    generic_ordered_seconds,
    observe_duration_conformance,
    picture_truncation_degradations,
)
from src.services.stages.shot_assembly import assemble_shots
from src.services.visual_media import assign_generic_line_media
from src.services.word_timed_captions import ClipRef

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


class VideoRenderInput(StageInput):
    pass


class VideoRenderOutput(StageOutputWithNotes):
    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. This is a TERMINAL artifact
    # producer: `artifact_url` IS the deliverable, not a hint about it, so an
    # empty one is never an acceptable outcome. Hence `produces_required` — the
    # stage FAILS instead of degrading. Degrade was the wrong default here: it let
    # the run finish while handing the user a reference to nothing, which is the
    # exact defect class this contract exists to close.
    produces: ClassVar[tuple[str, ...]] = ("artifact_url",)
    produces_required: ClassVar[bool] = True

    artifact_url: ArtifactUrl
    duration_seconds: float
    # Delivered-vs-ORDERED duration (same contract as every sibling compile
    # stage). Missing on generic until 2026-10-20: run 01a156a4-e906 shipped 12.3s
    # of a 30s order (-59%) with this field null and no degradation recorded.
    duration_conformance: DurationConformance | None = None


class VideoRenderStage(CompileStageBase[VideoRenderInput, VideoRenderOutput]):
    name = "video_render"
    input_model = VideoRenderInput
    output_model = VideoRenderOutput

    def build_shots(self, ctx: RunContext) -> list[dict[str, Any]]:
        story = ctx.prior_output("storyboard_draft")
        if not story:
            raise RuntimeError("storyboard_draft output missing")
        narrate = ctx.prior_output("narrate") or {}
        image_gen = ctx.prior_output("ai_image_gen") or {}
        # Real media / motion graphics / AI video per line — the SAME rule
        # ai_image_gen uses to skip rendering stills for these lines.
        media = assign_generic_line_media(ctx)
        image_by_line = media.image_by_line
        video_by_line = media.video_by_line
        # Burned-in captions (opt-in via deliverable captions_mode=burned_in).
        # Keyed by script_draft line id — the same value used as each shot's
        # line_id — the same text source the soft-CC publish track uses, so the
        # two caption renderings never diverge. None for the default
        # captions_mode=none → unchanged runs stay byte-identical.
        captions_by_line = (
            caption_text_by_line({"script_draft": ctx.prior_output("script_draft") or {}})
            if captions_mode(ctx) == "burned_in"
            else None
        )
        # Per-line AI stills (storyboard_draft prompts → ai_image_gen, keyed by
        # line_id) join 1:1 under every curated/real visual; only un-keyed images
        # (static `prompts`) keep the round-robin pool. Only fetchable URLs are
        # keyed — a non-http asset would crash the compile.
        claimed = media.covered_line_ids
        all_images = image_gen.get("images", [])
        ai_image_by_line = {
            str(img["line_id"]): str(img["image_url"])
            for img in all_images
            if img.get("line_id")
            and str(img.get("image_url") or "").startswith(("http://", "https://"))
            and str(img["line_id"]) not in claimed
        }
        unkeyed_images = [img for img in all_images if not img.get("line_id")]
        return assemble_shots(
            storyboard_shots=story["shots"],
            images=unkeyed_images,
            narration_clips=narrate.get("clips", []),
            captions_by_line=captions_by_line,
            image_by_line={**ai_image_by_line, **image_by_line},
            video_by_line=video_by_line,
            fit_by_line=media.fit_by_line,
        )

    def caption_clip_refs(self, ctx: RunContext) -> dict[str, ClipRef]:
        """Per-line narration clips to word-time-caption (ORCH-4.2).

        Mirrors the EXACT text source ``build_shots`` uses for the static
        ``captions_by_line`` map (``script_draft`` via ``caption_text_by_line``)
        joined to the ``narrate`` clips, so the word-timed and static caption text
        can never drift. Inert ({}) unless the deliverable opted into
        ``captions_mode=burned_in``; the kill-switch gate lives one layer up in
        ``CompileStageBase.handle``.
        """
        if captions_mode(ctx) != "burned_in":
            return {}
        text_by_line = (
            caption_text_by_line({"script_draft": ctx.prior_output("script_draft") or {}}) or {}
        )
        narrate = ctx.prior_output("narrate") or {}
        refs: dict[str, ClipRef] = {}
        for clip in narrate.get("clips", []):
            line_id = str(clip["line_id"])
            text = text_by_line.get(line_id)
            if not text:
                continue
            refs[line_id] = ClipRef(
                line_id=line_id,
                text=text,
                narrate_stage="narrate",
                clip_filename=f"{clip['id']}.wav",
                seconds=float(clip.get("seconds") or 0.0),
            )
        return refs

    def compile_tags(self, ctx: RunContext) -> dict[str, str]:
        # video_render carries no deliverable format/captions tags (the generic
        # blueprint predates them); keep the minimal pair, plus the resolution
        # tags so a generic render still honours the run's 2K/4K tier.
        return {"run_id": str(ctx.run_id), "stage": self.name, **resolution_tags(ctx)}

    def build_output(self, result: dict[str, Any], ctx: RunContext) -> VideoRenderOutput:
        delivered = float(result["duration_seconds"])
        # Graded against the SAME ladder brief_intake budgeted the script with
        # (`generic_ordered_seconds`), on the PLAYABLE length when the compiler
        # measured it. A breach lands on the structured field AND `degradations`.
        conformance = observe_duration_conformance(
            self.name,
            generic_ordered_seconds(ctx.config_snapshot),
            delivered,
            playable_seconds=playable_duration_seconds(result),
        )
        return VideoRenderOutput(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=delivered,
            duration_conformance=conformance,
            degradations=[
                *duration_conformance_degradations(conformance),
                *picture_truncation_degradations(result.get("mastering")),
            ],
            advisories=duration_conformance_advisories(conformance),
        )
