"""Captions-only (silent) mode: ``brief.captions_only`` — captions, no voice.

A showcase that is read, not heard (a muted autoplay feed, a locale with no approved
voice yet) skips TTS entirely: ``demo_video_narrate`` emits NO audio clips, every shot
takes its authored duration (or a reading-time estimate when it has none), and the
compile ships each line's text as a caption -- a SOFT overlay track (mov_text + SRT
sidecar) unless the deliverable explicitly asks for ``captions_mode=burned_in``. This module is the one place the flag
and the reading-time rule live, so the narrate stage, the compile stage and the
``captions_mode`` gate cannot disagree about what "silent" means.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext
    from src.services.word_timed_captions import ShotCaptionsPayload

#: Comfortable on-screen reading pace, words per second (~180 wpm).
READING_WORDS_PER_SECOND = 3.0
#: A caption is never up for less than this, however short (a flash cut is unreadable)...
MIN_READING_SECONDS = 2.0
#: ...nor held longer than this on the estimate alone (a long line is the author's to time).
MAX_READING_SECONDS = 12.0


def captions_only(ctx: RunContext) -> bool:
    """True when the brief asked for a silent, captions-only render (strict ``True``)."""
    brief = ctx.config_snapshot.get("brief") or {}
    return brief.get("captions_only") is True


def reading_seconds(text: str) -> float:
    """How long ``text`` needs on screen to be read: ~3 words/s, clamped to [2s, 12s]."""
    words = len(text.split())
    return min(max(words / READING_WORDS_PER_SECOND, MIN_READING_SECONDS), MAX_READING_SECONDS)


def static_caption_cues(
    shots: Sequence[Mapping[str, Any]], text_by_line: Mapping[str, str]
) -> list[ShotCaptionsPayload] | None:
    """One shot-spanning cue per captioned shot, for the soft overlay track.

    A captions-only film has no narration clip to word-time, so each shot whose
    ``line_id`` has caption text gets a single cue over its whole duration
    (shot-local ``[0, seconds)``). A shot that already carries an author's
    per-shot ``caption`` is left alone (that override burns, by explicit choice),
    as is a shot with no positive duration. ``None`` when nothing qualifies, so
    the compile stays byte-identical.
    """
    entries: list[ShotCaptionsPayload] = []
    for idx, shot in enumerate(shots):
        if shot.get("caption"):
            continue
        text = (text_by_line.get(str(shot.get("line_id"))) or "").strip()
        seconds = float(shot.get("seconds") or 0.0)
        if text and seconds > 0:
            entries.append(
                {
                    "shot_index": idx,
                    "cues": [{"text": text, "start": 0.0, "end": seconds, "words": []}],
                }
            )
    return entries or None
