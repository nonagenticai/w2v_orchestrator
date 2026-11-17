"""Deterministic, LLM-free NLE exports (CMX3600 EDL + FCPXML) for a storyboard.

Pure functions only — no DB, no LLM, no I/O. The route layer
(``api/routes/storyboard.py``) fetches the panels + scenes via the repositories
and hands them here; this module lays each panel out as one cut on a rough
timeline so a shot list can be imported into an editor (Resolve / Premiere /
Final Cut) as a starting sequence.

Timing: a panel's ``duration_seconds`` (the ONLY timing field on the model) is
its clip length. When it is ``None`` we fall back to :data:`DEFAULT_CLIP_SECONDS`
— a placeholder length so the shot still lands on the timeline; a reviewer
should treat those clips as un-timed. Source timecode always runs
``00:00:00:00 .. duration``; record timecode accumulates across panels so the
events form a contiguous cut list. Frame math is done in whole frames (integers)
so the record TC never drifts.

Ordering matches ``csv_export``: panels sort by
``(scene.position, panel.position, panel.created_at)`` so a reader walks the
board scene-by-scene; panels with no scene (or a dangling ``scene_id``) sort
LAST under an ``"Unassigned"`` label.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID
from xml.etree.ElementTree import Element, SubElement, indent, tostring

from src.database.models import StoryboardPanel, StoryboardScene
from src.services.storyboard.csv_export import scene_label

# 24fps is the storyboard default cutting rate; a reviewer may want 25/30.
DEFAULT_FPS = 24
# Placeholder clip length (seconds) for a panel whose duration_seconds is None
# — ASSUMPTION worth sanity-checking: un-timed shots get a nominal 4s on the
# rough timeline so they are visible and monotonic, not zero-length.
DEFAULT_CLIP_SECONDS = 4.0


def _one_line(value: str | None) -> str:
    """Collapse a value to a single line (strip newlines / runs of whitespace).

    ``None``/empty → ``""``. Used to keep EDL comments and FCPXML notes on one
    line — an embedded newline would corrupt the CMX3600 line grammar.
    """
    if not value:
        return ""
    return " ".join(str(value).split())


def _seconds_to_frames(seconds: float, fps: int) -> int:
    """Round a duration in seconds to whole frames (never below 1, never negative)."""
    return max(round(max(seconds, 0.0) * fps), 1)


def _frames_to_timecode(frames: int, fps: int) -> str:
    """Format an absolute frame count as an SMPTE ``HH:MM:SS:FF`` timecode."""
    frames = max(frames, 0)
    total_secs, frame = divmod(frames, fps)
    hours, rem = divmod(total_secs, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frame:02d}"


def smpte_timecode(seconds: float, fps: int = DEFAULT_FPS) -> str:
    """SMPTE frame timecode ``HH:MM:SS:FF``.

    ``frames = round((seconds - whole_seconds) * fps)``, carrying into the
    seconds field on overflow (e.g. ``0.99999s`` at 24fps rounds to 24 frames →
    carries to the next whole second).
    """
    total = max(seconds, 0.0)
    whole = int(total)
    frames = round((total - whole) * fps)
    return _frames_to_timecode(whole * fps + frames, fps)


def _ordered_panels(
    panels: list[StoryboardPanel], scenes: list[StoryboardScene]
) -> list[tuple[StoryboardPanel, str]]:
    """Panels in export order paired with their resolved scene label.

    Mirrors ``csv_export.build_csv``: sort by ``(scene.position, panel.position,
    panel.created_at)`` with a dangling/None ``scene_id`` sorting last as
    ``"Unassigned"``.
    """
    scene_meta: dict[UUID, tuple[int, str]] = {
        scene.id: (scene.position, scene_label(scene)) for scene in scenes
    }

    def sort_key(panel: StoryboardPanel) -> tuple[float, int, Any]:
        meta = scene_meta.get(panel.scene_id) if panel.scene_id is not None else None
        scene_pos = float(meta[0]) if meta is not None else float("inf")
        return (scene_pos, panel.position, panel.created_at)

    ordered: list[tuple[StoryboardPanel, str]] = []
    for panel in sorted(panels, key=sort_key):
        meta = scene_meta.get(panel.scene_id) if panel.scene_id is not None else None
        label = meta[1] if meta is not None else scene_label(None)
        ordered.append((panel, label))
    return ordered


def _panel_duration_frames(panel: StoryboardPanel, fps: int) -> int:
    """Clip length in whole frames — ``DEFAULT_CLIP_SECONDS`` when un-timed."""
    seconds = panel.duration_seconds if panel.duration_seconds is not None else DEFAULT_CLIP_SECONDS
    return _seconds_to_frames(seconds, fps)


def build_edl(
    panels: list[StoryboardPanel],
    scenes: list[StoryboardScene],
    *,
    fps: int = DEFAULT_FPS,
    title: str = "STORYBOARD",
) -> str:
    """Render the storyboard as a CMX3600 EDL (one cut event per panel).

    Header is a ``TITLE:`` line + ``FCM: NON-DROP FRAME``. Each event is numbered
    ``001``+, reel ``AX``, channel ``V``, transition ``C`` (cut); source TC runs
    ``00:00:00:00 .. duration`` and record TC accumulates across panels. A
    ``* FROM CLIP NAME:`` comment carries the ``shot_key`` + scene label, and a
    second ``*`` comment carries the (single-line-sanitised) action description,
    falling back to dialogue.
    """
    lines: list[str] = [f"TITLE: {_one_line(title)}", "FCM: NON-DROP FRAME"]
    record_frames = 0
    for index, (panel, label) in enumerate(_ordered_panels(panels, scenes), start=1):
        dur_frames = _panel_duration_frames(panel, fps)
        src_in = _frames_to_timecode(0, fps)
        src_out = _frames_to_timecode(dur_frames, fps)
        rec_in = _frames_to_timecode(record_frames, fps)
        rec_out = _frames_to_timecode(record_frames + dur_frames, fps)
        record_frames += dur_frames

        lines.append(f"{index:03d}  AX       V     C        {src_in} {src_out} {rec_in} {rec_out}")
        clip_name = _one_line(panel.shot_key) or f"shot-{index}"
        lines.append(f"* FROM CLIP NAME: {clip_name} ({_one_line(label)})")
        comment = _one_line(panel.action_description) or _one_line(panel.dialogue)
        if comment:
            lines.append(f"* {comment}")
    return "\n".join(lines) + "\n"


def build_fcpxml(
    panels: list[StoryboardPanel],
    scenes: list[StoryboardScene],
    *,
    fps: int = DEFAULT_FPS,
    title: str = "STORYBOARD",
) -> str:
    """Render the storyboard as a minimal, well-formed FCPXML (version 1.9).

    Built with stdlib ``xml.etree.ElementTree`` (no lxml). One ``<gap>`` per
    panel sits on the sequence spine — ``name`` from ``shot_key``, ``offset`` and
    ``duration`` as rational ``{frames}/{fps}s`` times (offsets monotonic,
    durations positive), each carrying a ``<note>`` with the action/dialogue
    text. This favours a valid, parseable tree over perfect NLE fidelity.
    """
    ordered = _ordered_panels(panels, scenes)
    total_frames = sum(_panel_duration_frames(panel, fps) for panel, _ in ordered)

    fcpxml = Element("fcpxml", version="1.9")
    resources = SubElement(fcpxml, "resources")
    SubElement(
        resources,
        "format",
        id="r1",
        name=f"FFVideoFormat1080p{fps}",
        frameDuration=f"1/{fps}s",
        width="1920",
        height="1080",
    )
    library = SubElement(fcpxml, "library")
    event = SubElement(library, "event", name=title)
    project = SubElement(event, "project", name=title)
    sequence = SubElement(
        project,
        "sequence",
        format="r1",
        duration=f"{total_frames}/{fps}s",
        tcStart="0s",
        tcFormat="NDF",
    )
    spine = SubElement(sequence, "spine")

    offset_frames = 0
    for index, (panel, _label) in enumerate(ordered, start=1):
        dur_frames = _panel_duration_frames(panel, fps)
        gap = SubElement(
            spine,
            "gap",
            name=_one_line(panel.shot_key) or f"shot-{index}",
            offset=f"{offset_frames}/{fps}s",
            duration=f"{dur_frames}/{fps}s",
            start="0s",
        )
        note_text = _one_line(panel.action_description) or _one_line(panel.dialogue)
        if note_text:
            SubElement(gap, "note").text = note_text
        offset_frames += dur_frames

    indent(fcpxml)
    body = tostring(fcpxml, encoding="unicode")
    # Final Cut Pro requires the DTD declaration immediately after the XML
    # declaration; it rejects an otherwise well-formed tree that lacks it.
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n{body}\n'
