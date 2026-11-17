"""Pure unit tests for the LLM-free storyboard PDF board export.

Exercises ``src/services/storyboard/pdf_export.py`` with the same
``SimpleNamespace`` ORM stand-ins ``test_csv_export.py`` uses (the builder only
reads attributes, never touches the DB). Covers: a real PDF magic header, the
empty board, scene ordering (shared with the CSV/EDL exports), the placeholder
path for panels with no image, an undecodable image degrading to the placeholder
rather than raising, and text that would otherwise break reportlab's mini-HTML
paragraph parser.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

from src.database.models import StoryboardPanel, StoryboardScene
from src.services.storyboard.pdf_export import build_pdf, panel_order
from tests.support.tiny_image import tiny_png as _png

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
    "media_ref",
    "image_ref",
)


def _panel(**overrides: Any) -> StoryboardPanel:
    base: dict[str, Any] = dict.fromkeys(_PANEL_FIELDS)
    base["id"] = uuid4()
    base["position"] = 0
    base["status"] = "draft"
    base["created_at"] = datetime(2026, 1, 1, tzinfo=UTC)
    base["characters"] = []
    base["props"] = []
    base["media_ref"] = {}
    base.update(overrides)
    return cast(StoryboardPanel, SimpleNamespace(**base))


def _scene(
    *,
    scene_id: UUID | None = None,
    position: int = 0,
    name: str | None = None,
) -> StoryboardScene:
    return cast(
        StoryboardScene,
        SimpleNamespace(
            id=scene_id if scene_id is not None else uuid4(),
            position=position,
            name=name,
            int_ext=None,
            location=None,
        ),
    )


def test_empty_board_still_renders_a_pdf() -> None:
    data = build_pdf([], [], project_name="Empty")
    assert data.startswith(b"%PDF")
    assert len(data) > 500


def test_board_with_scenes_renders_non_empty_pdf() -> None:
    scene_a = _scene(position=0, name="Opening")
    scene_b = _scene(position=1, name="Reveal")
    panels = [
        _panel(
            shot_key="clip-1", position=0, scene_id=scene_a.id, action_description="hero enters"
        ),
        _panel(shot_key="clip-2", position=1, scene_id=scene_a.id, dialogue="hello"),
        _panel(shot_key="clip-3", position=0, scene_id=scene_b.id, duration_seconds=2.5),
    ]
    data = build_pdf(panels, [scene_a, scene_b], project_name="Board")
    assert data.startswith(b"%PDF")
    assert data.rstrip().endswith(b"%%EOF")


def test_panel_order_follows_scene_order_then_position() -> None:
    scene_a = _scene(position=1, name="Second")
    scene_b = _scene(position=0, name="First")
    panels = [
        _panel(shot_key="a2", position=1, scene_id=scene_a.id),
        _panel(shot_key="orphan", position=0, scene_id=None),
        _panel(shot_key="b1", position=0, scene_id=scene_b.id),
        _panel(shot_key="a1", position=0, scene_id=scene_a.id),
    ]
    assert panel_order(panels, [scene_a, scene_b]) == ["b1", "a1", "a2", "orphan"]
    # …and the same board renders.
    assert build_pdf(panels, [scene_a, scene_b]).startswith(b"%PDF")


def test_panel_without_image_renders_placeholder_path() -> None:
    panel = _panel(shot_key="clip-1")
    with_placeholder = build_pdf([panel], [], images={})
    with_frame = build_pdf([panel], [], images={panel.id: _png()})
    assert with_placeholder.startswith(b"%PDF")
    assert with_frame.startswith(b"%PDF")
    # Reportlab emits an image XObject only when a raster was actually embedded,
    # so this distinguishes the placeholder branch from a silent no-op (it fails
    # if _frame_flowable regresses to always/never returning the placeholder).
    assert b"/Subtype /Image" in with_frame
    assert b"/Subtype /Image" not in with_placeholder
    assert len(with_frame) > len(with_placeholder)


def test_undecodable_image_degrades_to_placeholder_instead_of_raising() -> None:
    panel = _panel(shot_key="clip-1")
    broken = build_pdf([panel], [], images={panel.id: b"<html>404 not found</html>"})
    empty_bytes = build_pdf([panel], [], images={panel.id: b""})
    reference = build_pdf([panel], [], images={})
    assert broken.startswith(b"%PDF")
    assert empty_bytes.startswith(b"%PDF")
    # Same page content as the no-image board (only the trailer's dates differ in
    # length by 0), so compare structurally: neither embeds an image XObject.
    assert b"/Subtype /Image" not in broken
    assert b"/Subtype /Image" not in reference


def test_markup_in_panel_text_is_escaped_not_parsed() -> None:
    panel = _panel(
        shot_key="clip-<1>",
        action_description="cut to <b>black</b> & hold",
        dialogue="a < b & c > d",
        characters=["<Hero>"],
    )
    data = build_pdf([panel], [], project_name="R&D <draft>")
    assert data.startswith(b"%PDF")


def test_images_for_unknown_panel_ids_are_ignored() -> None:
    panel = _panel(shot_key="clip-1")
    data = build_pdf([panel], [], images={uuid4(): _png()})
    assert data.startswith(b"%PDF")
    assert b"/Subtype /Image" not in data


def test_long_board_paginates_without_error() -> None:
    scene = _scene(position=0, name="Long")
    panels = [
        _panel(
            shot_key=f"clip-{i}",
            position=i,
            scene_id=scene.id,
            action_description="lorem ipsum dolor sit amet " * 4,
        )
        for i in range(40)
    ]
    data = build_pdf(panels, [scene], project_name="Long board")
    assert data.startswith(b"%PDF")
    # Reportlab writes one ``/Type /Page`` object per page; 40 cards cannot fit on
    # a single landscape-A4 sheet, so a 1-page result would mean cards were silently
    # dropped rather than flowed.
    assert data.count(b"/Type /Page\n") > 1
    assert build_pdf(panels[:3], [scene]).count(b"/Type /Page\n") == 1
