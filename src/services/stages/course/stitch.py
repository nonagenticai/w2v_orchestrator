"""course_stitch — concatenate per-lesson MP4s into one course-length video.

``course_compile_lessons`` produces a discrete narrated MP4 per lesson; this
stage feeds those back through the compositor as ``video_url`` shots so the
deliverable also has a single stitched artifact. The compositor re-encodes
each clip to the uniform segment format and preserves its embedded audio, so
the stitched output keeps every lesson's narration.

The per-lesson list stays available (via ``course_compile_lessons`` output);
this stage adds the stitched ``course_artifact_url`` plus ``chapters`` (a
per-lesson marker — level 0 — offsets accumulated from each lesson's duration,
optionally followed by mid-lesson sub-chapters — level 1 — one per slide, at
absolute course offsets carried up from ``lesson_compile``) and an optional
``thumbnail_url`` (the AI cover from ``course_thumbnail``, best-effort — None
when that stage was skipped).

``artifact_url`` mirrors ``course_artifact_url`` so the generic, shape-based
video result viewer + the YouTube ``_find_video_url`` resolver (both match
``artifact_url``/``video_url``) resolve a course's final video.

Chapter offsets are derived from per-lesson ``duration_seconds`` (the same
values fed as shot floors), accumulated; they are approximate by design — the
compositor preserves each clip's natural audio length, so cumulative offsets
track it closely, and we clamp/skip rather than fabricate markers past the true
stitched total.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from src.clients.video_compile import VideoCompileClient
from src.services.compile_budget import COMPILE_WAIT_CEILING_SECONDS
from src.services.stages.base import (
    SLOT_USE_OPTIONAL,
    ArtifactUrl,
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    StagePreconditionError,
    validate_compile_result,
)
from src.services.stages.primitives.compile_stage import (
    _compile_with_backstop,
    output_dimension_fields,
    resolution_tags,
)
from src.services.stages.shared.mastering_gate import (
    mastering_gate_advisories,
    mastering_gate_degradations,
    mastering_gate_notes,
)
from src.services.stages.shared.mastering_plan import build_mastering_profile
from src.services.stages.shared.motion_slides import format_dims, quality_tier
from src.services.stages.shared.title_cards import build_title_card_shots, splice_title_cards
from src.services.stages.shot_presign import refresh_shot_asset_urls
from src.services.stages.upstream import guarded

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

# The compiler's own liveness-bounded worst case, shared with every
# CompileStageBase compile (see compile_budget.py). The old 1800s (+120s
# stage_call) abandoned a stitch the compiler still considered healthy — the same
# class of abandonment #1039 fixed for website_review. stage_call cannot carry it
# (it hard-caps timeout_seconds at 3600s), so the wait uses the same
# `_compile_with_backstop` as CompileStageBase.
_COMPILE_TIMEOUT_SECONDS = COMPILE_WAIT_CEILING_SECONDS


class CourseStitchInput(StageInput):
    pass


class CourseChapter(StageOutput):
    """One chapter marker, in playback order. ``start_seconds`` is the absolute
    offset of the marker in the stitched course video (course timeline) — this
    holds for sub-chapters too, so every consumer (YouTube composer, player
    seek/active-index) stays offset-uniform and never adds a parent offset.

    ``level`` is ``0`` for a per-lesson marker (top level) and ``1`` for a
    mid-lesson sub-chapter (one per slide). ``parent_sequence`` is the ``sequence``
    of the owning lesson marker for a level-1 row (``None`` at level 0).
    ``lesson_project_id`` is the parent lesson's stable id.

    ``chapter_key`` is the stable key a title override is applied by — the bare
    ``lesson_project_id`` at level 0, and ``f"{lesson_project_id}#{ordinal}"``
    (1-based slide ordinal within the lesson) at level 1. A level-1 row shares its
    parent's ``lesson_project_id``, so the ordinal suffix is what makes each slide
    individually addressable; the ordinal (not the slide's fresh ``uuid7``) keeps
    the key stable across re-stitches, matching ``lesson_project_id``'s contract."""

    sequence: int
    title: str
    start_seconds: float
    duration_seconds: float
    level: int = 0
    parent_sequence: int | None = None
    lesson_project_id: str | None = None
    chapter_key: str | None = None


class CourseStitchOutput(StageOutputWithNotes):
    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. This is a TERMINAL artifact
    # producer: `artifact_url` IS the deliverable, not a hint about it, so an
    # empty one is never an acceptable outcome. Hence `produces_required` — the
    # stage FAILS instead of degrading. Degrade was the wrong default here: it let
    # the run finish while handing the user a reference to nothing, which is the
    # exact defect class this contract exists to close.
    produces: ClassVar[tuple[str, ...]] = ("artifact_url",)
    produces_required: ClassVar[bool] = True

    course_artifact_url: ArtifactUrl
    # Mirror of course_artifact_url so the shape-based FE video viewer +
    # youtube _find_video_url (which match artifact_url/video_url) resolve a
    # course's final video.
    artifact_url: ArtifactUrl
    duration_seconds: float
    lesson_count: int
    chapters: list[CourseChapter]
    # AI cover from course_thumbnail (best-effort; None when that stage skipped).
    thumbnail_url: str | None = None


def _build_chapters(
    lessons: list[dict],
    *,
    total_seconds: float,
    titles_by_pid: dict[str, str],
    section_titles_by_pid: dict[str, str],
    title_overrides: dict[str, str] | None = None,
    start_offset: float = 0.0,
) -> list[CourseChapter]:
    """Accumulate per-lesson markers (level 0) plus mid-lesson sub-chapters
    (level 1, one per slide) in playback order, on a single monotonic
    ``sequence``.

    Level-0 offsets come from per-lesson ``duration_seconds``; lessons with a
    non-positive duration are skipped (no reliable placement), and once the
    cumulative offset reaches the true stitched ``total_seconds`` we stop rather
    than emit a marker past the end of the video (input durations can drift from
    the re-encoded total). The first emitted chapter is always at 0.0.

    Sub-chapters are read from each lesson's ``sub_chapters`` (lesson-relative
    offsets derived in ``lesson_compile``); each is placed at the absolute course
    offset ``lesson_start + sub.start_seconds`` and clamped the same way (skipped
    when it lands at/after the true total). Non-positive-duration sub-chapters
    are dropped.

    ``title_overrides`` (by ``lesson_project_id``) wins over the
    config/outline/``Lesson N`` fallback for level-0 titles. Production passes
    none at build time — operator edits are applied post-hoc via the chapters
    PATCH route — but threading it here keeps the title precedence in one place.

    ``start_offset`` shifts every marker forward by an opt-in intro title card's
    duration (the lesson content no longer starts at 0.0 when a card is prepended).
    """
    overrides = title_overrides or {}
    chapters: list[CourseChapter] = []
    cumulative = start_offset
    lesson_no = 0
    for lesson in lessons:
        dur = float(lesson.get("duration_seconds") or 0.0)
        if dur <= 0.0:
            continue
        start = cumulative
        cumulative += dur
        if chapters and total_seconds > 0.0 and start >= total_seconds:
            break
        lesson_no += 1
        pid = str(lesson.get("lesson_project_id") or "")
        title = (
            overrides.get(pid)
            or titles_by_pid.get(pid)
            or section_titles_by_pid.get(pid)
            or f"Lesson {lesson_no}"
        )
        lesson_seq = len(chapters) + 1
        chapters.append(
            CourseChapter(
                sequence=lesson_seq,
                title=title,
                start_seconds=round(start, 3),
                duration_seconds=round(dur, 3),
                level=0,
                parent_sequence=None,
                lesson_project_id=pid or None,
                chapter_key=pid or None,
            )
        )
        sub_no = 0
        for sub in lesson.get("sub_chapters") or []:
            sub_dur = float(sub.get("duration_seconds") or 0.0)
            if sub_dur <= 0.0:
                continue
            abs_start = start + float(sub.get("start_seconds") or 0.0)
            if total_seconds > 0.0 and abs_start >= total_seconds:
                continue
            sub_no += 1
            sub_title = str(sub.get("title") or "").strip() or f"Section {sub_no}"
            chapters.append(
                CourseChapter(
                    sequence=len(chapters) + 1,
                    title=sub_title,
                    start_seconds=round(abs_start, 3),
                    duration_seconds=round(sub_dur, 3),
                    level=1,
                    parent_sequence=lesson_seq,
                    lesson_project_id=pid or None,
                    chapter_key=f"{pid}#{sub_no}" if pid else None,
                )
            )
    return chapters


class CourseStitchStage(StageHandler[CourseStitchInput, CourseStitchOutput]):
    name = "course_stitch"
    input_model = CourseStitchInput
    output_model = CourseStitchOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("asr", when=SLOT_USE_OPTIONAL),)

    def __init__(self, client: VideoCompileClient | None = None) -> None:
        # compile() is async submit→poll; the render budget is the poll-loop
        # wall-clock (_COMPILE_TIMEOUT_SECONDS, passed below), not the 600s
        # per-request read cap. See compile_stage.py for the full rationale.
        self._client = client or VideoCompileClient()

    async def handle(
        self,
        payload: CourseStitchInput,
        ctx: RunContext,
    ) -> CourseStitchOutput:
        compiled = ctx.prior_output("course_compile_lessons")
        if not compiled:
            raise RuntimeError("course_compile_lessons output missing")
        lessons = compiled.get("lessons") or []
        if not lessons:
            raise StagePreconditionError("course_stitch: no compiled lessons to stitch")
        shots = [
            {
                "id": str(lesson["lesson_run_id"]),
                # Shot.seconds must be > 0; clips with their own audio play at
                # their natural length regardless, so this is only a floor.
                "seconds": float(lesson.get("duration_seconds") or 0.0) or 1.0,
                "video_url": str(lesson["artifact_url"]),
            }
            for lesson in lessons
        ]
        # Opt-in animated intro/outro title card. course_stitch builds shots and
        # calls compile() directly (not via CompileStageBase), so the title-card
        # injection is wired here too; splice before the re-sign so the freshly
        # stored card URL gets the same TTL refresh as the lesson shots.
        # Pass the degrade sink: a card that fails to render is soft-dropped, and
        # without this the ONLY record of why is a log line that dies at the next
        # pod roll. That is what made the 2026-09-03 title-card loss (-60.000s,
        # two 30s cards) undiagnosable for three sessions.
        title_card_notes: list[str] = []
        # ordered_seconds=None: a course has no single ordered length to cap the
        # cards against, so its block renders exactly as configured.
        intro, outro = await build_title_card_shots(
            ctx, ordered_seconds=None, notes=title_card_notes
        )
        shots = splice_title_cards(shots, intro, outro)
        # Lesson artifact_urls were presigned by course_compile_lessons with a 1h
        # TTL; a multi-lesson course paused/iterated past that hour hands
        # video_compiler a dead signature → its fetch 403s → compile 500s. Re-mint
        # a fresh TTL right before the POST (the helper's _ASSET_URL_KEYS includes
        # "video_url"). course_stitch calls compile() directly, so the shared
        # CompileStageBase re-sign tail (#200) never covers it.
        await refresh_shot_asset_urls(shots)
        # Deterministic mastering PROFILE (ORCH-0.5), same presence-is-opt-in
        # contract as CompileStageBase: an absent `mastering` block ⇒ None ⇒ no
        # `mastering` key on the request. It carries the authored loudness target
        # (brief.voice.master_lufs → config_snapshot["mastering"]["loudness_target"],
        # PR #608) plus the picture/encode fields (colorspace, crf, bitrate…).
        #
        # But the AUDIO master itself is DISABLED at the stitch, via
        # `tags["mastering"] = "off"`. Every lesson was already loudness-mastered to
        # this same target when its own run went through CompileStageBase, so the
        # concatenation is already level-consistent. `master_audio` has no
        # idempotency guard (its only skip is "no audio stream"), so a second pass
        # would re-run the acompressor → alimiter → loudnorm chain over already-
        # mastered audio: loudnorm is near-idempotent so integrated LUFS stays put,
        # but the compressor is not — it collapses loudness RANGE by ~3 LU and pumps.
        # Measured in video_compiler_service tests/render_harness/test_double_master.py.
        # With the tag the bed-less `_finalize_audio` returns the `-c copy` concat
        # unchanged (faststart is still applied by concat_shots). The profile is kept
        # on the request so the target stays correct if the tag is ever flipped back.
        #
        # DELIBERATELY absent here (not an oversight):
        #   • `finish` — each lesson already ran the finish/grade chain through
        #     CompileStageBase; re-sending it at the stitch would re-grade
        #     already-graded picture.
        #   • `music` / `ambient` — the stitch has no bed of its own; every lesson's
        #     bed is already baked into its MP4.
        #   • `captions` — there are no caption cues at stitch level; lesson captions
        #     are already burned in / muxed per lesson.
        mastering_profile = build_mastering_profile(ctx.config_snapshot)
        mastering = mastering_profile.as_mastering_payload() if mastering_profile else None
        # Ordered resolution tier for the WHOLE run lives on
        # `config_snapshot["resolution"]` (one of 1080p|2k|4k; set in
        # `run_setup.py:598` from the per-run override or the project brief,
        # normalized by `quality_tier` so an unknown/missing value resolves to
        # the default tier rather than silently landing on 1080p). Every other
        # CompileStageBase subclass forwards it via `resolution_tags()`
        # (render_width/render_height/crf) on `compile_tags()`
        # (`primitives/compile_stage.py:398-410`); course_stitch calls
        # `compile()` directly (not via CompileStageBase) and, until now, sent
        # NO resolution/format field at all — so the compiler's `_FORMAT_DIMS`
        # tier lookup never fired and it fell back to `settings.render_width/
        # render_height` (1080p default), silently downgrading a 2k/4k course
        # order. Fold the same tags in here so the terminal stitched canvas
        # matches the per-lesson canvases it concatenates.
        tier = quality_tier(ctx.config_snapshot)
        deliv = ctx.config_snapshot.get("deliverable") or {}
        expected_width, expected_height = format_dims(deliv.get("format"), tier)
        result = await _compile_with_backstop(
            guarded(
                lambda: self._client.compile(
                    shots=shots,
                    tags={
                        "run_id": str(ctx.run_id),
                        "stage": "course_stitch",
                        "format": str(deliv.get("format", "16x9")),
                        **resolution_tags(ctx),
                        # Master ONCE per lesson, never again at the stitch — see above.
                        "mastering": "off",
                    },
                    mastering=mastering,
                    timeout=_COMPILE_TIMEOUT_SECONDS,
                    # Stable across a resumable-timeout re-dispatch (one stitch per
                    # run; regenerate mints a fresh run_id) so the compiler dedupes a
                    # re-submit instead of running a second paid render.
                    idempotency_key=f"{ctx.run_id}:course_stitch",
                )
            ),
            stage_name="course_stitch",
        )
        validate_compile_result(result, stage="course_stitch")
        total_seconds = float(result["duration_seconds"])

        # Loud degradation, not a silent downscale: the compiler ffprobes the
        # delivered mux and reports its ACTUAL dims (`output_dimension_fields`);
        # if that comes back short of what the ordered tier demands (an older
        # compiler build ignoring the new tags, or a probe returning a stale
        # canvas), say so in `notes` rather than shipping a 1080p master that
        # LOOKS like it satisfied a 2k/4k order. `notes` is the same soft-
        # degradation channel `_compile_duration_qa_notes` uses for the sibling
        # "declared vs actual" gap on every other CompileStageBase subclass — no
        # existing `DEGRADATION_CODES` entry names a resolution shortfall, and
        # `shared/degradation.py` is owned by another workstream in this
        # worktree, so this stays on the channel this stage already carries
        # (`CourseStitchOutput` is a `StageOutputWithNotes`).
        resolution_notes: list[str] = []
        actual_dims = output_dimension_fields(result)
        actual_width = actual_dims.get("width")
        actual_height = actual_dims.get("height")
        if (
            actual_width
            and actual_height
            and (actual_width < expected_width or actual_height < expected_height)
        ):
            resolution_notes.append(
                f"course_stitch: ordered resolution tier {tier!r} "
                f"({expected_width}x{expected_height}) but the delivered master is "
                f"{actual_width}x{actual_height} — degrading rather than shipping a "
                "silently downscaled course"
            )

        # Lesson titles: config_snapshot["lessons"] is authoritative (set at config
        # time, same UUID-string form as compile_lessons); course_outline sections
        # are an LLM-ordered secondary lookup (keyed, never positional).
        titles_by_pid = {
            str(item.get("lesson_project_id")): str(item.get("title"))
            for item in (ctx.config_snapshot.get("lessons") or [])
            if item.get("lesson_project_id") and item.get("title")
        }
        outline = ctx.prior_output("course_outline") or {}
        section_titles_by_pid = {
            str(sec.get("lesson_project_id")): str(sec.get("title"))
            for sec in (outline.get("sections") or [])
            if sec.get("lesson_project_id") and sec.get("title")
        }
        chapters = _build_chapters(
            lessons,
            total_seconds=total_seconds,
            titles_by_pid=titles_by_pid,
            section_titles_by_pid=section_titles_by_pid,
            # An intro card (when opted in) shifts the lesson timeline forward.
            start_offset=float(intro["seconds"]) if intro else 0.0,
        )

        # Best-effort AI cover (course_thumbnail runs in parallel off
        # course_outline and is done well before this stage; None when skipped).
        thumb = ctx.prior_output("course_thumbnail") or {}
        thumbnail_url = thumb.get("thumbnail_url") if isinstance(thumb, dict) else None

        # ── THE COMPILER'S MASTERING GATE, read here because nothing else does ──
        # `course` was the one genuine A/V blind spot in the fleet. Every other
        # product type gets this for free from `CompileStageBase.handle`, which reads
        # `mastering_gate_notes` + `mastering_gate_degradations` off the compile
        # response (`primitives/compile_stage.py`, the `qa_notes` /
        # `stage_degradations` block). `course_stitch` calls `client.compile()`
        # DIRECTLY, so no base-class fix can ever reach it: loudness, true peak,
        # colour, CFR, dead-frame and the whole gate verdict were measured by the
        # compiler and then DISCARDED for every course deliverable.
        #
        # The call shape MIRRORS the base deliberately — same two helpers, same
        # argument (`result.get("mastering")`), notes appended after this stage's own
        # notes, degradations appended, both on the identical `result` the base would
        # have read. If the two ever drift, course silently grades differently from
        # everything else, which is the defect this closes.
        #
        # ⚠️ Safe despite `tags["mastering"] = "off"` above: that tag disables the
        # compiler's AUDIO MASTER pass, not its verification. `post_mux_verify` runs
        # unconditionally in `video_compiler_service/src/api/routes/compile.py` after
        # the mux, so a stitch still comes back with a full report — checked on the
        # compiler's `origin/main`. It is NOT the case that the gate would stamp every
        # course run UNMEASURED.
        mastering_report = result.get("mastering")
        artifact_url = str(result["artifact_url"])
        return CourseStitchOutput(
            course_artifact_url=artifact_url,
            artifact_url=artifact_url,
            duration_seconds=total_seconds,
            # Count lessons, not shots — shots may now include spliced title cards.
            lesson_count=len(lessons),
            chapters=chapters,
            thumbnail_url=thumbnail_url or None,
            # Persist any title-card drop reason (and a resolution shortfall,
            # if the delivered master came back under the ordered tier) so it
            # survives the pod.
            notes=[*title_card_notes, *resolution_notes, *mastering_gate_notes(mastering_report)],
            # The MACHINE-READABLE half, same as the base: a structurally failed or
            # UNMEASURED mastering gate as a `MASTERING_GATE_FAILED` entry. A tunable
            # ceiling alone records nothing (see `mastering_gate_degradations`).
            degradations=mastering_gate_degradations(mastering_report),
            # The compiler's measured-but-ungated tier, on the non-degrading field.
            advisories=mastering_gate_advisories(mastering_report),
        )
