"""Deterministic, LLM-free shot-list CSV export for a project's storyboard.

Pure functions only — no DB, no LLM, no I/O beyond an in-memory
``io.StringIO``. The route layer (``api/routes/storyboard.py``) fetches the
panels + scenes via the repositories and hands them here; this module maps
each panel to one RFC-4180 CSV row (stdlib ``csv.writer`` handles quoting of
embedded commas/quotes/newlines).

Ordering: panels sort by ``(scene.position, panel.position, panel.created_at)``
so a reader walks the board scene-by-scene; panels with no scene (or a dangling
``scene_id``) sort LAST under an ``"Unassigned"`` label.
"""

from __future__ import annotations

import csv
import datetime
import io
from collections.abc import Sequence
from typing import cast
from uuid import UUID

from src.database.models import StoryboardPanel, StoryboardScene

CSV_COLUMNS: list[str] = [
    "scene",
    "shot_key",
    "position",
    "shot_size",
    "angle",
    "movement",
    "lens",
    "duration_seconds",
    "action_description",
    "dialogue",
    "vo_text",
    "sfx_cue",
    "music_cue",
    "characters",
    "props",
    "art_style",
    "aspect_ratio",
    "transition_out",
    "status",
]


def scene_label(scene: StoryboardScene | None) -> str:
    """Human-readable scene label for the ``scene`` column.

    ``None`` → ``"Unassigned"``. Otherwise prefer ``name``; fall back to a
    slug-line ``"INT. LOCATION"`` (gracefully dropping whichever of
    ``int_ext``/``location`` is missing); finally ``"Scene N"`` (1-based).
    """
    if scene is None:
        return "Unassigned"
    if scene.name:
        return scene.name
    left = (scene.int_ext or "").strip()
    right = (scene.location or "").strip()
    if left and right:
        return f"{left}. {right}"
    if left or right:
        return left or right
    return f"Scene {scene.position + 1}"


def _render_list(value: Sequence[object] | None) -> str:
    """Flatten a JSONB list column to a ``"; "``-joined cell.

    ``None``/empty → ``""``. A dict item prefers its ``name`` then ``ref``
    key; anything else is ``str()``-ified.
    """
    if not value:
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, dict):
            d = cast("dict[str, object]", item)
            parts.append(str(d.get("name") or d.get("ref") or d))
        else:
            parts.append(str(item))
    return "; ".join(parts)


def panel_to_row(panel: StoryboardPanel, label: str) -> list[str]:
    """Map one panel to a list of cells aligned to ``CSV_COLUMNS`` (``None`` → ``""``)."""

    def cell(value: object) -> str:
        return "" if value is None else str(value)

    values: dict[str, str] = {
        "scene": label,
        "shot_key": cell(panel.shot_key),
        "position": cell(panel.position),
        "shot_size": cell(panel.shot_size),
        "angle": cell(panel.angle),
        "movement": cell(panel.movement),
        "lens": cell(panel.lens),
        "duration_seconds": cell(panel.duration_seconds),
        "action_description": cell(panel.action_description),
        "dialogue": cell(panel.dialogue),
        "vo_text": cell(panel.vo_text),
        "sfx_cue": cell(panel.sfx_cue),
        "music_cue": cell(panel.music_cue),
        "characters": _render_list(panel.characters),
        "props": _render_list(panel.props),
        "art_style": cell(panel.art_style),
        "aspect_ratio": cell(panel.aspect_ratio),
        "transition_out": cell(panel.transition_out),
        "status": cell(panel.status),
    }
    return [values[col] for col in CSV_COLUMNS]


def build_csv(panels: list[StoryboardPanel], scenes: list[StoryboardScene]) -> str:
    """Render the full shot-list CSV (header row always written, even when empty)."""
    scene_meta: dict[UUID, tuple[int, str]] = {
        scene.id: (scene.position, scene_label(scene)) for scene in scenes
    }

    def sort_key(panel: StoryboardPanel) -> tuple[float, int, datetime.datetime]:
        meta = scene_meta.get(panel.scene_id) if panel.scene_id is not None else None
        scene_pos = float(meta[0]) if meta is not None else float("inf")
        return (scene_pos, panel.position, panel.created_at)

    ordered = sorted(panels, key=sort_key)

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for panel in ordered:
        meta = scene_meta.get(panel.scene_id) if panel.scene_id is not None else None
        label = meta[1] if meta is not None else scene_label(None)
        writer.writerow(panel_to_row(panel, label))
    return buf.getvalue()
