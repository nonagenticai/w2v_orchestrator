"""Pure unit tests for the LLM-free storyboard EDL + FCPXML export.

Exercises ``src/services/storyboard/edl_export.py`` with lightweight
``SimpleNamespace`` stand-ins for the ORM rows (the functions only read
attributes, never touch the DB). Covers: SMPTE frame-timecode conversion
(incl. frame carry-over), EDL event numbering + record-TC accumulation +
unassigned-last ordering + newline sanitisation, and a parseable, monotonic
FCPXML tree.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4
from xml.etree.ElementTree import fromstring

from src.database.models import StoryboardPanel, StoryboardScene
from src.services.storyboard.edl_export import (
    DEFAULT_CLIP_SECONDS,
    DEFAULT_FPS,
    build_edl,
    build_fcpxml,
    smpte_timecode,
)

_PANEL_FIELDS = (
    "shot_key",
    "position",
    "shot_size",
    "movement",
    "duration_seconds",
    "action_description",
    "dialogue",
    "vo_text",
    "sfx_cue",
    "music_cue",
    "transition_out",
    "status",
    "scene_id",
    "created_at",
)


def _panel(**overrides: Any) -> StoryboardPanel:
    base: dict[str, Any] = dict.fromkeys(_PANEL_FIELDS)
    base["position"] = 0
    base["created_at"] = datetime(2026, 1, 1, tzinfo=UTC)
    base.update(overrides)
    # Read-only attribute stand-in — edl_export only reads fields, never the DB.
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


# ─── smpte_timecode ──────────────────────────────────────────────────────────
def test_smpte_timecode_zero() -> None:
    assert smpte_timecode(0.0) == "00:00:00:00"


def test_smpte_timecode_whole_and_fractional_frames() -> None:
    # 1.5s @ 24fps → 1 second + 12 frames
    assert smpte_timecode(1.5, fps=24) == "00:00:01:12"


def test_smpte_timecode_minutes_and_hours() -> None:
    assert smpte_timecode(3661.0, fps=24) == "01:01:01:00"


def test_smpte_timecode_frame_carry_on_overflow() -> None:
    # 0.99999s @ 24fps rounds to 24 frames → carries into the seconds field.
    assert smpte_timecode(0.99999, fps=24) == "00:00:01:00"


def test_smpte_timecode_negative_clamps_to_zero() -> None:
    assert smpte_timecode(-5.0) == "00:00:00:00"


# ─── build_edl ───────────────────────────────────────────────────────────────
def test_edl_header_and_empty_board() -> None:
    edl = build_edl([], [])
    lines = edl.splitlines()
    assert lines == ["TITLE: STORYBOARD", "FCM: NON-DROP FRAME"]


def test_edl_custom_title() -> None:
    edl = build_edl([], [], title="MY BOARD")
    assert edl.splitlines()[0] == "TITLE: MY BOARD"


def test_edl_event_numbering_and_record_tc_accumulation() -> None:
    scene = _scene(name="S1", position=0)
    p1 = _panel(shot_key="clip-1", position=0, duration_seconds=4.0, scene_id=scene.id)
    p2 = _panel(shot_key="clip-2", position=1, duration_seconds=2.0, scene_id=scene.id)
    edl = build_edl([p1, p2], [scene])
    lines = edl.splitlines()
    events = [ln for ln in lines if ln[:3].isdigit()]
    assert len(events) == 2
    # Event 001: source 0..4s, record 0..4s @ 24fps.
    assert events[0].startswith("001  AX       V     C        ")
    assert events[0].endswith("00:00:00:00 00:00:04:00 00:00:00:00 00:00:04:00")
    # Event 002: source 0..2s; record continues 4s..6s.
    assert events[1].startswith("002  AX       V     C        ")
    assert events[1].endswith("00:00:00:00 00:00:02:00 00:00:04:00 00:00:06:00")


def test_edl_none_duration_uses_default_clip_seconds() -> None:
    panel = _panel(shot_key="clip-1", duration_seconds=None)
    edl = build_edl([panel], [])
    src_out = smpte_timecode(DEFAULT_CLIP_SECONDS, fps=DEFAULT_FPS)
    event = next(ln for ln in edl.splitlines() if ln.startswith("001"))
    assert f"00:00:00:00 {src_out} 00:00:00:00 {src_out}" in event


def test_edl_clip_name_and_comment_lines() -> None:
    scene = _scene(name="Opening", position=0)
    panel = _panel(
        shot_key="clip-7",
        action_description="hero enters",
        scene_id=scene.id,
    )
    lines = build_edl([panel], [scene]).splitlines()
    assert "* FROM CLIP NAME: clip-7 (Opening)" in lines
    assert "* hero enters" in lines


def test_edl_comment_falls_back_to_dialogue() -> None:
    panel = _panel(shot_key="clip-1", action_description=None, dialogue="stop right there")
    assert "* stop right there" in build_edl([panel], []).splitlines()


def test_edl_newline_sanitization() -> None:
    panel = _panel(shot_key="clip-1", action_description="line one\nline two\r\nline three")
    lines = build_edl([panel], []).splitlines()
    # The multiline description is collapsed to a single comment line.
    assert "* line one line two line three" in lines


def test_edl_unassigned_sorts_last() -> None:
    scene = _scene(name="S1", position=0)
    assigned = _panel(shot_key="clip-1", position=0, scene_id=scene.id)
    orphan = _panel(shot_key="clip-2", position=0, scene_id=None)
    lines = build_edl([orphan, assigned], [scene]).splitlines()
    clip_names = [ln for ln in lines if ln.startswith("* FROM CLIP NAME:")]
    assert clip_names == [
        "* FROM CLIP NAME: clip-1 (S1)",
        "* FROM CLIP NAME: clip-2 (Unassigned)",
    ]


def test_edl_dangling_scene_id_is_unassigned() -> None:
    panel = _panel(shot_key="clip-1", scene_id=uuid4())
    lines = build_edl([panel], []).splitlines()
    assert "* FROM CLIP NAME: clip-1 (Unassigned)" in lines


# ─── build_fcpxml ────────────────────────────────────────────────────────────
def test_fcpxml_parses_with_declaration() -> None:
    xml_text = build_fcpxml([], [])
    assert xml_text.startswith("<?xml")
    # Final Cut Pro requires the DTD line right after the XML declaration or it
    # rejects the import ("not a valid FCPXML document"), even when well-formed.
    assert "<!DOCTYPE fcpxml>" in xml_text.split("\n", 2)[1]
    root = fromstring(xml_text)  # noqa: S314 — trusted, self-generated
    assert root.tag == "fcpxml"
    assert root.get("version") == "1.9"


def test_fcpxml_one_spine_child_per_panel() -> None:
    scene = _scene(name="S1", position=0)
    panels = [
        _panel(shot_key="clip-1", position=0, duration_seconds=4.0, scene_id=scene.id),
        _panel(shot_key="clip-2", position=1, duration_seconds=2.0, scene_id=scene.id),
        _panel(shot_key="clip-3", position=2, scene_id=scene.id),
    ]
    root = fromstring(build_fcpxml(panels, [scene]))  # noqa: S314
    spine = root.find("./library/event/project/sequence/spine")
    assert spine is not None
    gaps = list(spine)
    assert len(gaps) == len(panels)
    assert [g.get("name") for g in gaps] == ["clip-1", "clip-2", "clip-3"]


def test_fcpxml_offsets_are_monotonic() -> None:
    scene = _scene(name="S1", position=0)
    panels = [
        _panel(shot_key="clip-1", position=0, duration_seconds=4.0, scene_id=scene.id),
        _panel(shot_key="clip-2", position=1, duration_seconds=2.0, scene_id=scene.id),
        _panel(shot_key="clip-3", position=2, duration_seconds=3.0, scene_id=scene.id),
    ]
    root = fromstring(build_fcpxml(panels, [scene]))  # noqa: S314
    spine = root.find("./library/event/project/sequence/spine")
    assert spine is not None
    offsets = [int(g.get("offset", "0/1s").split("/")[0]) for g in spine]
    durations = [int(g.get("duration", "0/1s").split("/")[0]) for g in spine]
    assert offsets == sorted(offsets)
    # Each offset equals the running sum of prior durations; all durations positive.
    assert all(d > 0 for d in durations)
    assert offsets == [0, durations[0], durations[0] + durations[1]]


def test_fcpxml_note_carries_action_text() -> None:
    panel = _panel(shot_key="clip-1", action_description="hero\nenters")
    root = fromstring(build_fcpxml([panel], []))  # noqa: S314
    gap = root.find("./library/event/project/sequence/spine/gap")
    assert gap is not None
    note = gap.find("note")
    assert note is not None
    assert note.text == "hero enters"
