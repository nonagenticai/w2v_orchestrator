"""lesson_compile — composite slide images + narration into one video.

Emits one shot per narrated paragraph: the still image of the slide that
paragraph belongs to, with the paragraph's narration audio muxed over it and
the shot held for the narration's length. The video-compile service stitches
the shots into a single narrated MP4.

It also derives per-slide ``sub_chapters`` (lesson-relative offsets) from the
narration clip lengths — carried up through ``course_compile_lessons`` so
``course_stitch`` can place mid-lesson markers in the stitched course timeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from src.services.captions import caption_text_by_line
from src.services.stages.base import ArtifactUrl, StageInput, StageOutput, StageOutputWithNotes
from src.services.stages.primitives.compile_stage import (
    CompileStageBase,
    captions_mode,
    playable_duration_seconds,
)
from src.services.stages.shared.duration_conformance import (
    DurationConformance,
    duration_conformance_advisories,
    duration_conformance_degradations,
    lesson_ordered_seconds,
    observe_duration_conformance,
    picture_truncation_degradations,
)
from src.services.stages.shot_assembly import assemble_shots
from src.services.visual_media import (
    assign_real_media_by_line,
    assign_real_media_videos_by_line,
    collect_ai_video_urls,
    collect_real_media_urls,
    collect_real_media_videos,
)
from src.services.word_timed_captions import ClipRef

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


class LessonCompileInput(StageInput):
    pass


class LessonSubChapter(StageOutput):
    """One mid-lesson marker (one per slide), in playback order, with offsets
    relative to the start of the lesson (the first emitted sub-chapter is at
    0.0). ``course_stitch`` re-bases these onto the stitched course timeline."""

    sequence: int
    title: str
    start_seconds: float
    duration_seconds: float


class LessonCompileOutput(StageOutputWithNotes):
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
    sub_chapters: list[LessonSubChapter] = Field(default_factory=list)
    # Delivered-vs-ORDERED duration, recorded so a deviation is visible in the
    # run record: lesson masters measured 216% of their ordered length on
    # 2026-09-10 and nothing anywhere noticed, because nothing compared the two.
    # Observation ONLY — it never fails or degrades the stage, and it is
    # deliberately NOT a `notes` entry (a note flips the whole run to
    # `degraded`). `None` means no usable duration was ordered, which is NOT the
    # same fact as "on target".
    duration_conformance: DurationConformance | None = None


def build_lesson_sub_chapters(
    slides: list[dict[str, Any]],
    narrate_clips: list[dict[str, Any]],
    total_seconds: float,
) -> list[LessonSubChapter]:
    """Derive per-slide sub-chapters from narration clip lengths, scaled to fit.

    A slide groups several paragraphs; its intended length is the sum of its
    paragraphs' narration ``seconds``. We accumulate intended slide offsets in
    playback order, then scale them to the lesson's actual ``total_seconds``
    (the compositor stretches each shot to its narration audio, so the summed
    intended durations track the real total within rounding). Sub-chapter starts
    are the scaled offsets — contiguous, with the last ending exactly at
    ``total_seconds`` — and the first is pinned to 0.0.

    Slides with no narrated duration are dropped. A lesson with fewer than two
    positive-duration slides yields no sub-chapters: a single slide spans the
    whole lesson and is already represented by the lesson-level marker.
    """
    seconds_by_paragraph: dict[str, float] = {}
    for clip in narrate_clips:
        pid = clip.get("paragraph_id")
        if pid is not None:
            seconds_by_paragraph[str(pid)] = float(clip.get("seconds") or 0.0)

    intended_starts: list[float] = []
    titles: list[str] = []
    cursor = 0.0
    for slide in slides:
        dur = sum(
            seconds_by_paragraph.get(str(pid), 0.0) for pid in (slide.get("paragraph_ids") or [])
        )
        if dur <= 0.0:
            continue
        intended_starts.append(cursor)
        titles.append(str(slide.get("title") or "").strip())
        cursor += dur

    sum_intended = cursor
    if len(intended_starts) < 2 or sum_intended <= 0.0 or total_seconds <= 0.0:
        return []

    scale = total_seconds / sum_intended
    subs: list[LessonSubChapter] = []
    for i, intended_start in enumerate(intended_starts):
        start = 0.0 if i == 0 else intended_start * scale
        end = total_seconds if i + 1 == len(intended_starts) else intended_starts[i + 1] * scale
        dur = end - start
        if dur <= 0.0:
            continue
        subs.append(
            LessonSubChapter(
                sequence=len(subs) + 1,
                title=titles[i] or f"Section {len(subs) + 1}",
                start_seconds=round(start, 3),
                duration_seconds=round(dur, 3),
            )
        )
    return subs


class LessonCompileStage(CompileStageBase[LessonCompileInput, LessonCompileOutput]):
    name = "lesson_compile"
    input_model = LessonCompileInput
    output_model = LessonCompileOutput
    no_shots_message = "no narrated shots to compile"

    def build_shots(self, ctx: RunContext) -> list[dict[str, Any]]:
        slides = ctx.prior_output("lesson_slides")
        narrate = ctx.prior_output("lesson_narrate")
        record = ctx.prior_output("lesson_record")
        if not (slides and narrate and record):
            raise RuntimeError("lesson_slides + lesson_narrate + lesson_record required")

        image_by_slide = {rec["slide_id"]: rec["image_url"] for rec in record["recordings"]}
        slide_by_paragraph = {
            pid: slide["id"] for slide in slides["slides"] for pid in slide["paragraph_ids"]
        }
        # The paragraph→slide→image two-hop expressed as a keyed image join.
        image_by_line: dict[str, str] = {}
        for clip in narrate["clips"]:
            pid = clip["paragraph_id"]
            slide_id = slide_by_paragraph.get(pid)
            image_url = image_by_slide.get(slide_id) if slide_id else None
            if image_url:
                image_by_line[pid] = image_url

        # Spine = narrated paragraphs; each clip is its own audio (joined by the
        # paragraph id used as line_id). No storyboard, so no `description`.
        storyboard_shots = [
            {"id": c["id"], "line_id": c["paragraph_id"], "seconds": float(c["seconds"] or 0.0)}
            for c in narrate["clips"]
        ]

        # When the lesson (or its parent course, propagated per-child by the
        # wizard) was grounded in real media — operator-**pinned**, document-
        # **extracted**, or auto-**discovered** images/clips recorded by
        # ``source_ingest`` / ``media_discovery`` — lay those (already
        # compiler-safe, re-published) URLs over the paragraph lines as full-frame
        # CUTAWAYS: a curated image/clip wins over that line's slide PNG (the
        # narration audio is muxed per-line and is preserved). Cutaways are SPREAD
        # EVENLY across the lesson (``distribute=True``) so real imagery recurs
        # through the timeline instead of clustering on the opening paragraphs;
        # video clips take their own lines (the compositor honours ``video_url``
        # over ``image_url``), so an image and a clip never collide. Runs without
        # sources → empty overlays → byte-for-byte unchanged.
        real_urls = collect_real_media_urls(
            source_ingest=ctx.prior_output("source_ingest"),
            media_discovery=ctx.prior_output("media_discovery"),
        )
        real_videos = collect_real_media_videos(
            source_ingest=ctx.prior_output("source_ingest"),
            media_discovery=ctx.prior_output("media_discovery"),
        )
        img_overlay, img_fit = assign_real_media_by_line(
            storyboard_shots, real_urls, distribute=True
        )
        vid_overlay, vid_fit = assign_real_media_videos_by_line(
            storyboard_shots, real_videos, skip_line_ids=img_overlay.keys(), distribute=True
        )
        image_by_line.update(img_overlay)  # real media wins over the slide PNG

        # AI-generated clips (ai_video_gen, opt-in) are the lowest-priority
        # cutaway: curated real media wins, so they take only the lines no
        # curated image/clip claimed (a generative clip still beats the slide
        # PNG via video_url), again spread across the timeline. Empty when
        # ai_video_gen skipped → unchanged.
        ai_video_urls = collect_ai_video_urls(ctx.prior_output("ai_video_gen"))
        ai_vid_overlay, ai_vid_fit = assign_real_media_videos_by_line(
            storyboard_shots,
            ai_video_urls,
            skip_line_ids=set(img_overlay) | set(vid_overlay),
            distribute=True,
        )
        vid_overlay.update(ai_vid_overlay)

        # Animated slide (opt-in, lesson_record motion_video_url): the motion clip
        # of a paragraph's slide plays as a video over that line, beating the still
        # slide PNG. It sits BELOW curated media and AI video in precedence
        # (footage > curated > AI video > animated slide > still PNG), so it only
        # fills paragraph lines no curated image cutaway / curated-or-AI video has
        # claimed. Empty when motion off / all clips soft-failed → unchanged.
        motion_by_slide = {
            rec["slide_id"]: rec["motion_video_url"]
            for rec in record["recordings"]
            if rec.get("motion_video_url")
        }
        if motion_by_slide:
            claimed = set(vid_overlay) | set(img_overlay)
            for clip in narrate["clips"]:
                pid = clip["paragraph_id"]
                if pid in claimed:
                    continue
                slide_id = slide_by_paragraph.get(pid)
                motion_url = motion_by_slide.get(slide_id) if slide_id else None
                if motion_url:
                    vid_overlay[pid] = motion_url

        # Burned-in captions (opt-in via deliverable captions_mode=burned_in).
        # Keyed by paragraph id — the same value used as each shot's line_id —
        # from lesson_script, the same text source the soft-CC publish track
        # uses, so the two caption renderings never diverge. None (no captions)
        # for the default captions_mode=none, so unchanged runs stay byte-identical.
        captions_by_line = (
            caption_text_by_line({"lesson_script": ctx.prior_output("lesson_script") or {}})
            if captions_mode(ctx) == "burned_in"
            else None
        )

        return assemble_shots(
            storyboard_shots=storyboard_shots,
            images=[],
            narration_clips=[
                {
                    "line_id": c["paragraph_id"],
                    "artifact_url": c["artifact_url"],
                    "seconds": c["seconds"],
                }
                for c in narrate["clips"]
            ],
            captions_by_line=captions_by_line,
            image_by_line=image_by_line,
            video_by_line=vid_overlay,
            fit_by_line={**img_fit, **vid_fit, **ai_vid_fit},
        )

    def caption_clip_refs(self, ctx: RunContext) -> dict[str, ClipRef]:
        """Per-line narration clips to word-time-caption (ORCH-4.2).

        Mirrors the EXACT text source ``build_shots`` uses for the static
        ``captions_by_line`` map (``lesson_script`` via ``caption_text_by_line``,
        keyed by paragraph id) joined to the ``lesson_narrate`` clips, so the
        word-timed and static caption text can never drift. Inert ({}) unless the
        deliverable opted into ``captions_mode=burned_in``; the kill-switch gate
        lives one layer up in ``CompileStageBase.handle``.
        """
        if captions_mode(ctx) != "burned_in":
            return {}
        text_by_line = (
            caption_text_by_line({"lesson_script": ctx.prior_output("lesson_script") or {}}) or {}
        )
        narrate = ctx.prior_output("lesson_narrate") or {}
        refs: dict[str, ClipRef] = {}
        for clip in narrate.get("clips", []):
            line_id = str(clip["paragraph_id"])
            text = text_by_line.get(line_id)
            if not text:
                continue
            refs[line_id] = ClipRef(
                line_id=line_id,
                text=text,
                narrate_stage="lesson_narrate",
                clip_filename=f"{clip['id']}.wav",
                seconds=float(clip.get("seconds") or 0.0),
            )
        return refs

    def build_output(self, result: dict[str, Any], ctx: RunContext) -> LessonCompileOutput:
        total_seconds = float(result["duration_seconds"])
        slides = ctx.prior_output("lesson_slides") or {}
        narrate = ctx.prior_output("lesson_narrate") or {}
        sub_chapters = build_lesson_sub_chapters(
            slides.get("slides") or [],
            narrate.get("clips") or [],
            total_seconds,
        )
        # Ordered length resolves exactly as `lesson/script.py:173` does
        # (`brief.duration_minutes`, default 6) converted to seconds — the same
        # number the script was written against, so the comparison is
        # like-for-like. A breach lands on BOTH the structured
        # `duration_conformance` field (the numbers) and the `degradations`
        # channel (the queryable verdict, which is what makes the run report
        # `degraded` instead of clean).
        conformance = observe_duration_conformance(
            self.name,
            lesson_ordered_seconds(ctx.config_snapshot),
            total_seconds,
            playable_seconds=playable_duration_seconds(result),
        )
        return LessonCompileOutput(
            artifact_url=str(result["artifact_url"]),
            duration_seconds=total_seconds,
            sub_chapters=sub_chapters,
            duration_conformance=conformance,
            degradations=[
                *duration_conformance_degradations(conformance),
                # A SECOND, independent fact about the same master. The verdict
                # above grades the PLAYABLE length (max of the two streams)
                # against the order; it cannot express a picture that stopped
                # early, because on such a master that max IS the audio. The
                # picture-vs-audio gap therefore gets its own code — one number
                # carrying both facts is the regression this replaces.
                *picture_truncation_degradations(result.get("mastering")),
            ],
            # The unmeasured fact rides the NON-degrading tier: a length nobody
            # probed is an unverified verdict, not a bad artifact.
            advisories=duration_conformance_advisories(conformance),
        )
