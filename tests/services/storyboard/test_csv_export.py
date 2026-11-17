"""Pure unit tests for the LLM-free storyboard CSV export.

Exercises ``src/services/storyboard/csv_export.py`` with lightweight
``SimpleNamespace`` stand-ins for the ORM rows (the functions only read
attributes, never touch the DB). Covers: header order, field-by-field
mapping, list flattening, RFC-4180 quoting round-trip, scene labelling +
fallbacks, ordering across scenes, and unassigned-sorts-last.
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

from src.database.models import StoryboardPanel, StoryboardScene
from src.services.storyboard.csv_export import (
    CSV_COLUMNS,
    build_csv,
    panel_to_row,
    scene_label,
)

_PANEL_FIELDS = (
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
    "scene_id",
    "created_at",
)


def _panel(**overrides: Any) -> StoryboardPanel:
    base: dict[str, Any] = dict.fromkeys(_PANEL_FIELDS)
    base["position"] = 0
    base["created_at"] = datetime(2026, 1, 1, tzinfo=UTC)
    base["characters"] = []
    base["props"] = []
    base.update(overrides)
    # Read-only attribute stand-in — csv_export only reads fields, never the DB.
    return cast(StoryboardPanel, SimpleNamespace(**base))


def _scene(
    *,
    scene_id: UUID | None = None,
    position: int = 0,
    name: str | None = None,
    int_ext: str | None = None,
    location: str | None = None,
) -> StoryboardScene:
    return cast(
        StoryboardScene,
        SimpleNamespace(
            id=scene_id if scene_id is not None else uuid4(),
            position=position,
            name=name,
            int_ext=int_ext,
            location=location,
        ),
    )


def test_header_order_matches_columns() -> None:
    rows = list(csv.reader(io.StringIO(build_csv([], []))))
    assert rows == [CSV_COLUMNS]


def test_empty_board_is_header_only() -> None:
    assert build_csv([], []).splitlines() == [",".join(CSV_COLUMNS)]


def test_full_field_mapping() -> None:
    scene = _scene(name="Opening", position=0)
    panel = _panel(
        shot_key="clip-1",
        position=0,
        shot_size="MS",
        angle="low",
        movement="pan",
        lens="35mm",
        duration_seconds=3.5,
        action_description="hero enters",
        dialogue="hi",
        vo_text="narration",
        sfx_cue="whoosh",
        music_cue="swell",
        characters=["Hero", "Villain"],
        props=["sword"],
        art_style="noir",
        aspect_ratio="16:9",
        transition_out="cut",
        status="approved",
        scene_id=scene.id,
    )
    row = panel_to_row(panel, scene_label(scene))
    mapped = dict(zip(CSV_COLUMNS, row, strict=True))
    assert mapped == {
        "scene": "Opening",
        "shot_key": "clip-1",
        "position": "0",
        "shot_size": "MS",
        "angle": "low",
        "movement": "pan",
        "lens": "35mm",
        "duration_seconds": "3.5",
        "action_description": "hero enters",
        "dialogue": "hi",
        "vo_text": "narration",
        "sfx_cue": "whoosh",
        "music_cue": "swell",
        "characters": "Hero; Villain",
        "props": "sword",
        "art_style": "noir",
        "aspect_ratio": "16:9",
        "transition_out": "cut",
        "status": "approved",
    }


def test_none_fields_render_empty() -> None:
    row = panel_to_row(_panel(shot_key="clip-1"), "Unassigned")
    mapped = dict(zip(CSV_COLUMNS, row, strict=True))
    assert mapped["dialogue"] == ""
    assert mapped["duration_seconds"] == ""
    assert mapped["shot_size"] == ""
    assert mapped["characters"] == ""
    assert mapped["props"] == ""


def test_rfc4180_quoting_round_trip() -> None:
    nasty = 'she said, "go!"\nthen left'
    panel = _panel(shot_key="clip-1", dialogue=nasty, scene_id=None)
    csv_text = build_csv([panel], [])
    parsed = list(csv.reader(io.StringIO(csv_text)))
    assert parsed[0] == CSV_COLUMNS
    row = dict(zip(CSV_COLUMNS, parsed[1], strict=True))
    assert row["dialogue"] == nasty


def test_scene_label_fallbacks() -> None:
    assert scene_label(None) == "Unassigned"
    assert scene_label(_scene(name="Named")) == "Named"
    assert scene_label(_scene(int_ext="INT", location="LOCATION")) == "INT. LOCATION"
    assert scene_label(_scene(int_ext="INT")) == "INT"
    assert scene_label(_scene(location="Rooftop")) == "Rooftop"
    assert scene_label(_scene(position=4)) == "Scene 5"


def test_null_scene_is_unassigned_and_sorts_last() -> None:
    scene = _scene(name="S1", position=0)
    assigned = _panel(shot_key="clip-1", position=0, scene_id=scene.id)
    orphan = _panel(shot_key="clip-2", position=0, scene_id=None)
    rows = list(csv.reader(io.StringIO(build_csv([orphan, assigned], [scene]))))
    # header + 2 rows; assigned first, unassigned last
    assert [r[CSV_COLUMNS.index("scene")] for r in rows[1:]] == ["S1", "Unassigned"]
    assert [r[CSV_COLUMNS.index("shot_key")] for r in rows[1:]] == ["clip-1", "clip-2"]


def test_ordering_across_two_scenes_plus_unassigned() -> None:
    s_late = _scene(name="Late", position=1)
    s_early = _scene(name="Early", position=0)
    # Deliberately shuffled input; expect early-scene panels, then late, then orphan.
    p_late = _panel(shot_key="clip-late", position=0, scene_id=s_late.id)
    p_early_b = _panel(shot_key="clip-eb", position=1, scene_id=s_early.id)
    p_early_a = _panel(shot_key="clip-ea", position=0, scene_id=s_early.id)
    p_orphan = _panel(shot_key="clip-orphan", position=0, scene_id=None)
    csv_text = build_csv([p_late, p_orphan, p_early_b, p_early_a], [s_early, s_late])
    rows = list(csv.reader(io.StringIO(csv_text)))
    assert [r[CSV_COLUMNS.index("shot_key")] for r in rows[1:]] == [
        "clip-ea",
        "clip-eb",
        "clip-late",
        "clip-orphan",
    ]
    assert [r[CSV_COLUMNS.index("scene")] for r in rows[1:]] == [
        "Early",
        "Early",
        "Late",
        "Unassigned",
    ]


def test_dangling_scene_id_treated_as_unassigned() -> None:
    # scene_id references a scene NOT in the scenes list → Unassigned, sorts last.
    panel = _panel(shot_key="clip-1", scene_id=uuid4())
    rows = list(csv.reader(io.StringIO(build_csv([panel], []))))
    assert rows[1][CSV_COLUMNS.index("scene")] == "Unassigned"
