"""Blueprint-agnostic footage-plan resolution (ORCH-3.3, shared by demo + promo).

The plan is INTENT only: a zoom-OFF co-removes its focus+cursor tracks and emits
NO callout, a ``static`` reframe drops only the cursor, a ``split`` resolves a
second clip's URL (dropping a dangling/self ref), and a callout lands ONLY at the
bbox captured in the focus track — never at a coordinate the LLM invented. With no
plan the input maps ride through UNCHANGED (the byte-identical/absent path).
"""

from __future__ import annotations

from src.services.stages.shared.footage_plans import resolve_footage_plans

_BBOX = {"x": 10, "y": 20, "w": 100, "h": 40}
_FOCUS: list[dict[str, object]] = [{"t": 0.5, "bbox": _BBOX, "action": "click"}]
_CURSOR: list[dict[str, object]] = [{"t": 0.0, "x": 5, "y": 5}, {"t": 0.5, "x": 60, "y": 40}]


def test_empty_plan_returns_inputs_unchanged() -> None:
    focus = {"clip-1": _FOCUS}
    cursor = {"clip-1": _CURSOR}
    focus_map, cursor_map, callouts, splits = resolve_footage_plans(
        line_ids=["clip-1"],
        plan_by_line={},
        focus_track_by_line=focus,
        cursor_track_by_line=cursor,
        video_by_line={},
    )
    # Identity: same objects handed back, no derived maps.
    assert focus_map is focus
    assert cursor_map is cursor
    assert callouts is None
    assert splits is None


def test_zoom_off_pops_focus_and_cursor_and_emits_no_callout() -> None:
    focus_map, cursor_map, callouts, splits = resolve_footage_plans(
        line_ids=["clip-1", "clip-2"],
        plan_by_line={
            "clip-1": {"zoom": "off", "callout": {"type": "badge", "label": "x"}},
            "clip-2": {"zoom": "auto"},
        },
        focus_track_by_line={"clip-1": _FOCUS, "clip-2": _FOCUS},
        cursor_track_by_line={"clip-1": _CURSOR, "clip-2": _CURSOR},
        video_by_line={},
    )
    assert focus_map is not None and "clip-1" not in focus_map
    assert cursor_map is not None and "clip-1" not in cursor_map
    # clip-2 untouched.
    assert focus_map["clip-2"] == _FOCUS
    assert cursor_map["clip-2"] == _CURSOR
    # zoom-off suppresses the callout entirely.
    assert callouts is None
    assert splits is None


def test_split_survives_zoom_off_on_same_line() -> None:
    # Ordering guard: a before/after split has no click ⇒ _validate forces zoom=off,
    # so the split MUST resolve BEFORE the zoom-off gate co-removes the tracks + skips.
    focus_map, cursor_map, callouts, splits = resolve_footage_plans(
        line_ids=["clip-1"],
        plan_by_line={
            "clip-1": {
                "zoom": "off",
                "split": {"layout": "v", "second_clip": "clip-2"},
                "callout": {"type": "badge", "label": "dropped"},
            }
        },
        focus_track_by_line={"clip-1": _FOCUS},
        cursor_track_by_line={"clip-1": _CURSOR},
        video_by_line={"clip-2": "http://v/2.mp4"},
    )
    # zoom-off still co-removes focus+cursor and suppresses the callout...
    assert focus_map is None and cursor_map is None
    assert callouts is None
    # ...but the split resolved (it ran before the zoom-off continue).
    assert splits == {"clip-1": {"layout": "v", "second_source_url": "http://v/2.mp4"}}


def test_static_reframe_drops_cursor_keeps_focus() -> None:
    focus_map, cursor_map, callouts, splits = resolve_footage_plans(
        line_ids=["clip-1"],
        plan_by_line={"clip-1": {"zoom": "auto", "reframe": "static"}},
        focus_track_by_line={"clip-1": _FOCUS},
        cursor_track_by_line={"clip-1": _CURSOR},
        video_by_line={},
    )
    assert focus_map is not None and focus_map["clip-1"] == _FOCUS  # zoom target kept
    assert cursor_map is None  # only cursor dropped ⇒ empty ⇒ None
    assert callouts is None
    assert splits is None


def test_callout_uses_captured_bbox() -> None:
    _, _, callouts, _ = resolve_footage_plans(
        line_ids=["clip-1"],
        plan_by_line={"clip-1": {"zoom": "auto", "callout": {"type": "badge", "label": "Click"}}},
        focus_track_by_line={"clip-1": _FOCUS},
        cursor_track_by_line=None,
        video_by_line={},
    )
    assert callouts == {"clip-1": {"type": "badge", "label": "Click", "at": 0.5, "bbox": _BBOX}}


def test_callout_dropped_when_no_captured_track() -> None:
    # No focus track for the line ⇒ no bbox ⇒ callout dropped (never a guess).
    _, _, callouts, _ = resolve_footage_plans(
        line_ids=["clip-1"],
        plan_by_line={"clip-1": {"zoom": "auto", "callout": {"type": "label", "label": "Nope"}}},
        focus_track_by_line=None,
        cursor_track_by_line=None,
        video_by_line={},
    )
    assert callouts is None


def test_callout_skips_a_placeholder_keyframe_for_the_interacted_one() -> None:
    # merqi spike 01a0fc55: the FIRST keyframe was a zero-filled placeholder, so the
    # callout (and zoom) landed on the empty top-left corner. Only an interacted,
    # visible element is a target.
    placeholder: dict[str, object] = {
        "t": 0.1,
        "bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
        "action": "click",
    }
    _, _, callouts, _ = resolve_footage_plans(
        line_ids=["clip-1"],
        plan_by_line={"clip-1": {"zoom": "auto", "callout": {"type": "badge", "label": "Go"}}},
        focus_track_by_line={"clip-1": [placeholder, *_FOCUS]},
        cursor_track_by_line=None,
        video_by_line={},
    )
    assert callouts == {"clip-1": {"type": "badge", "label": "Go", "at": 0.5, "bbox": _BBOX}}


def test_split_resolves_dangling_and_self_ref_dropped() -> None:
    # Valid second ref present in video_by_line ⇒ resolved.
    _, _, _, splits = resolve_footage_plans(
        line_ids=["clip-1"],
        plan_by_line={
            "clip-1": {"zoom": "auto", "split": {"layout": "h", "second_clip": "clip-2"}}
        },
        focus_track_by_line=None,
        cursor_track_by_line=None,
        video_by_line={"clip-2": "http://v/2.mp4"},
    )
    assert splits == {"clip-1": {"layout": "h", "second_source_url": "http://v/2.mp4"}}

    # Dangling ref (no such footage line) ⇒ dropped.
    _, _, _, dangling = resolve_footage_plans(
        line_ids=["clip-1"],
        plan_by_line={"clip-1": {"zoom": "auto", "split": {"layout": "h", "second_clip": "ghost"}}},
        focus_track_by_line=None,
        cursor_track_by_line=None,
        video_by_line={"clip-2": "http://v/2.mp4"},
    )
    assert dangling is None

    # Self-ref (second == line_id) ⇒ dropped even though the URL resolves.
    _, _, _, self_ref = resolve_footage_plans(
        line_ids=["clip-1"],
        plan_by_line={
            "clip-1": {"zoom": "auto", "split": {"layout": "h", "second_clip": "clip-1"}}
        },
        focus_track_by_line=None,
        cursor_track_by_line=None,
        video_by_line={"clip-1": "http://v/1.mp4"},
    )
    assert self_ref is None
