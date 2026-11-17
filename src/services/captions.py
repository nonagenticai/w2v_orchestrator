"""Soft closed-caption (SRT/VTT) minting from narration timing.

Every promo / lesson / website_review / generic run already produces, per spoken line,
both the *text* (its script/paragraph) and the *duration* (its narration clip's
``seconds``). This module joins the two into an ordered caption track — one cue
per spoken line, timed by accumulating clip durations along the playback order
of the script — and serialises it as SRT or WebVTT.

These are pure functions (no I/O): the YouTube publish path calls
:func:`caption_cues_for_run` on the run's completed stage outputs, serialises
with :func:`build_srt`, stores the bytes via ``store_captions_track``, and
hands the URL to ``youtube_upload_service`` for ``captions().insert``.

YouTube's auto-CC is timing-accurate because the durations come from the actual
synthesised audio, not a word-count estimate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.services.stages.shared.captions_only import reading_seconds


@dataclass(frozen=True)
class CaptionCue:
    """One caption cue: ``[start, end)`` seconds of spoken ``text``."""

    start: float
    end: float
    text: str


def _timecode(seconds: float, *, sep: str) -> str:
    """Format ``seconds`` as ``HH:MM:SS<sep>mmm`` (``sep=","`` SRT, ``"."`` VTT)."""
    total_ms = max(round(seconds * 1000), 0)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def srt_timecode(seconds: float) -> str:
    """SRT timecode: ``HH:MM:SS,mmm``."""
    return _timecode(seconds, sep=",")


def vtt_timecode(seconds: float) -> str:
    """WebVTT timecode: ``HH:MM:SS.mmm``."""
    return _timecode(seconds, sep=".")


def build_srt(cues: Sequence[CaptionCue]) -> str:
    """Serialise cues as an SRT document (1-based, blank-line separated)."""
    blocks: list[str] = []
    for cue in cues:
        text = cue.text.strip()
        if not text:
            continue
        # SRT cue numbers must be sequential with no gaps — index by emitted count.
        idx = len(blocks) + 1
        blocks.append(f"{idx}\n{srt_timecode(cue.start)} --> {srt_timecode(cue.end)}\n{text}\n")
    return "\n".join(blocks)


def build_vtt(cues: Sequence[CaptionCue]) -> str:
    """Serialise cues as a WebVTT document."""
    blocks: list[str] = ["WEBVTT\n"]
    for cue in cues:
        text = cue.text.strip()
        if not text:
            continue
        blocks.append(f"{vtt_timecode(cue.start)} --> {vtt_timecode(cue.end)}\n{text}\n")
    return "\n".join(blocks)


def build_caption_cues(
    lines: Sequence[dict[str, Any]],
    clips: Sequence[dict[str, Any]],
    *,
    line_id_key: str = "id",
    text_key: str = "text",
    clip_join_key: str = "line_id",
) -> list[CaptionCue]:
    """Join spoken ``lines`` (text) with narration ``clips`` (timing) into cues.

    Cues follow the *script order* of ``lines`` (the playback order); each line's
    duration is its matching clip's ``seconds`` (joined ``line[line_id_key] ==
    clip[clip_join_key]``). A line with no matching clip is skipped without
    advancing the clock (it has no audio); a matching clip with non-positive
    duration advances the clock but emits no cue; a line whose text is blank
    advances the clock but emits no cue. The result is contiguous and aligned to
    the real audio timeline.
    """
    seconds_by_id: dict[str, float] = {}
    for clip in clips:
        cid = clip.get(clip_join_key)
        if cid is None:
            continue
        try:
            seconds_by_id[str(cid)] = float(clip.get("seconds") or 0.0)
        except TypeError, ValueError:
            seconds_by_id[str(cid)] = 0.0

    cues: list[CaptionCue] = []
    cursor = 0.0
    for line in lines:
        lid = line.get(line_id_key)
        if lid is None or str(lid) not in seconds_by_id:
            continue  # no audio for this line — don't advance the clock
        dur = seconds_by_id[str(lid)]
        start = cursor
        cursor += max(dur, 0.0)
        text = str(line.get(text_key) or "").strip()
        if text and dur > 0.0:
            cues.append(CaptionCue(start=start, end=cursor, text=text))
    return cues


def build_course_caption_cues(
    lesson_tracks: Sequence[tuple[float, Sequence[dict[str, Any]], Sequence[dict[str, Any]]]],
) -> list[CaptionCue]:
    """Concatenate per-lesson caption cues onto the stitched-course timeline.

    A course video has no single script+narrate pair — it is stitched from child
    lesson videos, so each lesson's caption text+timing lives in that child run's
    ``lesson_script`` joined with ``lesson_narrate``. Each ``track`` is
    ``(chapter_start_seconds, lesson_script paragraphs, lesson_narrate clips)``:
    the cues are built *lesson-relative* via :func:`build_caption_cues` (the same
    paragraph join the ``lesson`` type uses), then shifted by the lesson's
    ``chapter_start_seconds`` — the authoritative absolute offset ``course_stitch``
    assigns each lesson in the stitched video. Tracks are emitted in the given
    order; a track whose clips don't join its paragraphs contributes nothing and
    never disturbs a later lesson's timing.
    """
    cues: list[CaptionCue] = []
    for start_offset, paragraphs, clips in lesson_tracks:
        for cue in build_caption_cues(
            paragraphs,
            clips,
            line_id_key="id",
            text_key="text",
            clip_join_key="paragraph_id",
        ):
            cues.append(
                CaptionCue(
                    start=cue.start + start_offset,
                    end=cue.end + start_offset,
                    text=cue.text,
                )
            )
    return cues


# Per-type sources: (script stage, lines key, line-id key, text key,
#                    narrate stage, clips key, clip-join key). Order is priority.
_CAPTION_SOURCES: tuple[tuple[str, str, str, str, str, str, str], ...] = (
    ("promo_script", "lines", "id", "text", "promo_narrate", "clips", "line_id"),
    (
        "lesson_script",
        "paragraphs",
        "id",
        "text",
        "lesson_narrate",
        "clips",
        "paragraph_id",
    ),
    (
        "website_review_script",
        "lines",
        "id",
        "text",
        "website_review_record",
        "clips",
        "line_id",
    ),
    # generic deliverable: script_draft `lines` (id/text) joined to narrate `clips` (line_id/seconds)
    ("script_draft", "lines", "id", "text", "narrate", "clips", "line_id"),
    # demo_video/showcase (ORCH-1.2): script_doctor re-emits the demo_video_gen shape
    # with only line TEXT doctored (same ids/order). Listed BEFORE demo_video_gen so
    # the doctored text wins the soft-CC + burned-in tracks; falls through to
    # demo_video_gen when the doctor skipped (its output is absent).
    ("script_doctor", "lines", "id", "text", "demo_video_narrate", "clips", "line_id"),
    # demo_video: demo_video_gen exposes per-scenario `lines` (id=clip-<n>/text),
    # joined to demo_video_narrate `clips` (line_id/seconds) — soft-CC + burned-in.
    ("demo_video_gen", "lines", "id", "text", "demo_video_narrate", "clips", "line_id"),
    # audit family (seo_audit / cro_audit). Both narrated-video sub-chains reuse
    # audit_video's shared models, so the shapes are already the standard ones:
    # AuditScriptOutput.lines (id/text) joined to AuditNarrateOutput.clips
    # (line_id/seconds). They were simply never listed here, so every narrated audit
    # produced a video whose "Download captions (.srt)" button 404'd — and the
    # burned-in path (`caption_text_by_line`, same table) had nothing to burn either.
    ("seo_audit_script", "lines", "id", "text", "seo_audit_narrate", "clips", "line_id"),
    ("cro_audit_script", "lines", "id", "text", "cro_audit_narrate", "clips", "line_id"),
)


def caption_cues_for_run(outputs: dict[str, Any]) -> list[CaptionCue] | None:
    """Build caption cues from a run's completed stage outputs, by deliverable type.

    ``outputs`` maps ``stage_name -> stage output dict``. Dispatches on which
    script+narrate stage pair is present (promo / lesson / website_review /
    generic). Returns ``None`` when no recognised pair exists (course /
    audiobook) or when the pair yields no cues — callers publish without captions.
    """
    for (
        script_stage,
        lines_key,
        line_id_key,
        text_key,
        clip_stage,
        clips_key,
        clip_join_key,
    ) in _CAPTION_SOURCES:
        script = outputs.get(script_stage)
        narrate = outputs.get(clip_stage)
        if not isinstance(script, dict) or not isinstance(narrate, dict):
            continue
        lines = script.get(lines_key) or []
        clips = narrate.get(clips_key) or []
        if not lines or not clips:
            continue
        cues = build_caption_cues(
            lines,
            clips,
            line_id_key=line_id_key,
            text_key=text_key,
            clip_join_key=clip_join_key,
        )
        if cues:
            return cues
    return captions_only_cues(outputs)


def _span(entry: object) -> tuple[str, float, float] | None:
    """``(line_id, start, end)`` from one compiler ``shot_timeline`` entry, or None."""
    if not isinstance(entry, dict) or entry.get("line_id") is None:
        return None
    try:
        start, end = float(entry["start"]), float(entry["end"])
    except KeyError, TypeError, ValueError:
        return None
    return (str(entry["line_id"]), start, end) if end > start else None


def _timeline_cues(timeline: list[object], text_by_line: dict[str, str]) -> list[CaptionCue]:
    """One cue per captioned line, at its real span on the compiled film."""
    cues = [
        CaptionCue(start=span[1], end=span[2], text=text)
        for span in filter(None, map(_span, timeline))
        if (text := text_by_line.get(span[0], "").strip())
    ]
    return sorted(cues, key=lambda c: c.start)


def _accumulated_cues(lines: list[object], text_by_line: dict[str, str]) -> list[CaptionCue]:
    """Script-order cues timed by each line's ``seconds`` (reading time when unauthored)."""
    cues: list[CaptionCue] = []
    cursor = 0.0
    for line in lines:
        if not isinstance(line, dict):
            continue
        text = text_by_line.get(str(line.get("id")), "").strip()
        try:
            seconds = float(line.get("seconds") or 0.0)
        except TypeError, ValueError:
            seconds = 0.0
        if seconds <= 0 and text:
            seconds = reading_seconds(text)
        if text and seconds > 0:
            cues.append(CaptionCue(start=cursor, end=cursor + seconds, text=text))
        cursor += max(seconds, 0.0)
    return cues


def captions_only_cues(outputs: Mapping[str, object]) -> list[CaptionCue] | None:
    """Caption cues for a captions-only (silent) demo_video run, which has no narration.

    Its ``demo_video_narrate`` deliberately emits NO clips, so the narration join in
    :func:`caption_cues_for_run` finds nothing and the SRT sidecar would be empty.
    Each line's caption text is timed by the compiler's authoritative
    ``demo_video_compile.shot_timeline`` (real film spans, title cards included);
    without one, by accumulating each script line's ``seconds`` (or its reading
    time when unauthored -- the same rule the compile used to size the shot).
    ``None`` for any run that is not captions-only.
    """
    narrate = outputs.get("demo_video_narrate")
    if not isinstance(narrate, dict) or narrate.get("clips"):
        return None
    text_by_line = caption_text_by_line(dict(outputs))
    if not text_by_line:
        return None
    compile_out = outputs.get("demo_video_compile")
    timeline = compile_out.get("shot_timeline") if isinstance(compile_out, dict) else None
    if isinstance(timeline, list) and timeline:
        return _timeline_cues(timeline, text_by_line) or None
    for key in ("script_doctor", "demo_video_gen"):
        script = outputs.get(key)
        if isinstance(script, dict):
            lines = script.get("lines")
            return _accumulated_cues(lines if isinstance(lines, list) else [], text_by_line) or None
    return None


def caption_text_by_line(outputs: dict[str, Any]) -> dict[str, str] | None:
    """The per-line spoken text keyed by the storyboard line id, by deliverable type.

    The burned-in compile path (``captions_mode=burned_in``) needs the
    ``{line_id: text}`` map ``assemble_shots`` stamps onto each shot's ``caption``
    field — timing comes from the narration clip already on the shot, so unlike
    :func:`caption_cues_for_run` no clip join is needed. Dispatches on the same
    per-type ``(script stage, lines key, line-id key, text key)`` columns of
    :data:`_CAPTION_SOURCES`, so the burned-in caption text can never drift from
    the soft-CC track minted at publish. Returns ``None`` when no recognised
    script stage is present (course / audiobook) — the caller burns no captions.
    """
    for script_stage, lines_key, line_id_key, text_key, *_ in _CAPTION_SOURCES:
        script = outputs.get(script_stage)
        if not isinstance(script, dict):
            continue
        lines = script.get(lines_key) or []
        mapping = {
            str(line[line_id_key]): str(line.get(text_key) or "")
            for line in lines
            if isinstance(line, dict) and line.get(line_id_key) is not None
        }
        if mapping:
            return mapping
    return None


def line_word_counts(outputs: dict[str, Any]) -> dict[str, int] | None:
    """The per-line spoken-word count keyed by the storyboard line id, by type.

    A pure companion to :func:`caption_text_by_line`: it walks the same per-type
    ``(script stage, lines key, line-id key, text key)`` columns of
    :data:`_CAPTION_SOURCES` and returns ``{line_id: word_count}`` — the whitespace
    word count of each line's spoken text. Returns ``None`` when no recognised script
    stage is present (course / audiobook).

    ⚠️ THIS IS NOW ON THE LIVE PATH. This docstring read "No behavior change — nothing
    on the live path calls it yet" from the day it was written; the caller it was
    written for is
    :func:`~src.services.stages.demo_video.gen.authored_word_shortfalls`, which
    reconciles each beat's authored count against the budget
    :func:`~src.services.stages.demo_video.gen.narration_words_per_scenario` computes.
    It is reused rather than reimplemented there so the count a beat is GRADED on is
    byte-identical to the count the caption tracks are built from — two word counts for
    one line is a drift waiting to happen.
    """
    for script_stage, lines_key, line_id_key, text_key, *_ in _CAPTION_SOURCES:
        script = outputs.get(script_stage)
        if not isinstance(script, dict):
            continue
        lines = script.get(lines_key) or []
        mapping = {
            str(line[line_id_key]): len(str(line.get(text_key) or "").split())
            for line in lines
            if isinstance(line, dict) and line.get(line_id_key) is not None
        }
        if mapping:
            return mapping
    return None
