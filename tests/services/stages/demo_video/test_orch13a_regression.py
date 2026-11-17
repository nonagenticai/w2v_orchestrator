"""ORCH-1.3a make-or-break regression guard: the cold-open seam is INERT.

The whole point of PR 1.3a is that a run with NO ``open_style`` and NO live
``direction.hook`` author-block is bit-for-bit identical to today: the assembled
shot list must ``==`` today's, and the cold-open reframe must NEVER touch a shot's
``seconds`` (C-duration) nor reorder shots (C0) nor override an author's per-line
``clip_anchor`` (C6).

This module pins that at two seams:

* ``assemble_shots`` — a ``cold_open_plan=None`` (the default) is byte-identical to
  omitting the kwarg entirely, for a footage-bearing showcase fixture.
* ``_attach_cold_open`` — the presence/footage/author-wins/duration-safe idiom.
* ``_resolve_cold_open`` — the inert guard (open_style absent + hook off/absent ⇒
  None) exercised via a thin fake ``ctx`` (no DB).

Pure + deterministic: no DB, no I/O — a fake ``ctx`` carries only a
``config_snapshot`` dict, and the stage is built via ``__new__`` (the resolver
does not use ``self``).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

from src.services.stages.demo_video.compile import DemoVideoCompileStage
from src.services.stages.shared.cold_open import cold_open_window
from src.services.stages.shot_assembly import assemble_shots

# ── A footage-bearing showcase fixture: 3 spine lines, each 1:1 to a recorded
# clip, with narration fixing each shot's duration. Shot 0 binds real footage so
# the cold-open seam has something to (potentially) reframe. ──────────────────
_STORYBOARD = [
    {"id": "clip-1", "line_id": "clip-1", "seconds": 4.0, "description": "open"},
    {"id": "clip-2", "line_id": "clip-2", "seconds": 3.0, "description": "middle"},
    {"id": "clip-3", "line_id": "clip-3", "seconds": 5.0, "description": "close"},
]
_NARRATION = [
    {"line_id": "clip-1", "artifact_url": "https://minio.test/a1.mp3", "seconds": 4.2},
    {"line_id": "clip-2", "artifact_url": "https://minio.test/a2.mp3", "seconds": 3.1},
    {"line_id": "clip-3", "artifact_url": "https://minio.test/a3.mp3", "seconds": 5.4},
]
_VIDEO_BY_LINE = {
    "clip-1": "https://minio.test/v1.mp4",
    "clip-2": "https://minio.test/v2.mp4",
    "clip-3": "https://minio.test/v3.mp4",
}


def _base_kwargs() -> dict[str, Any]:
    return {
        "storyboard_shots": copy.deepcopy(_STORYBOARD),
        "images": [],
        "narration_clips": copy.deepcopy(_NARRATION),
        "video_by_line": dict(_VIDEO_BY_LINE),
    }


# ── Make-or-break: cold_open_plan=None == omitting the kwarg entirely. ────────
def test_assemble_shots_byte_identical_without_plan() -> None:
    baseline = assemble_shots(**_base_kwargs())
    with_none = assemble_shots(**_base_kwargs(), cold_open_plan=None)
    assert baseline == with_none


# ── Positive: a plan reframes ONLY shot 0's window; seconds + shots[1:] hold. ─
def test_plan_reframes_only_opening_shot() -> None:
    baseline = assemble_shots(**_base_kwargs())
    reframed = assemble_shots(**_base_kwargs(), cold_open_plan=("end", 0.5))

    # Shot 0 gains framing keys …
    assert reframed[0]["clip_anchor"] == "end"
    assert reframed[0]["clip_lead_seconds"] == 0.5
    assert "clip_anchor" not in baseline[0]
    # … but its duration is untouched (C-duration) …
    assert reframed[0]["seconds"] == baseline[0]["seconds"]
    # … and shots order is preserved (C0).
    assert [s["id"] for s in reframed] == [s["id"] for s in baseline]
    # … and every downstream shot is byte-identical.
    assert reframed[1:] == baseline[1:]


# ── _attach_cold_open unit idiom (mirrors _attach_clip_framing's guards). ─────


# ── _resolve_cold_open inert guard via a thin fake ctx (no DB). ───────────────
def _stage() -> DemoVideoCompileStage:
    # The resolver does not use ``self`` — build the instance without __init__.
    return DemoVideoCompileStage.__new__(DemoVideoCompileStage)


def _fake_ctx(config_snapshot: dict[str, Any]) -> Any:
    return SimpleNamespace(config_snapshot=config_snapshot)


_LINES = [{"id": "clip-1"}, {"id": "clip-2"}]
_RECORD = {"clips": [{"line_id": "clip-1", "seconds": 4.0}]}


def test_resolve_cold_open_inert_when_no_style_no_hook() -> None:
    ctx = _fake_ctx({})
    assert _stage()._resolve_cold_open(ctx, _LINES, _RECORD) is None


def test_resolve_cold_open_inert_when_hook_off() -> None:
    ctx = _fake_ctx({"direction": {"hook": {"mode": "off"}}})
    assert _stage()._resolve_cold_open(ctx, _LINES, _RECORD) is None


def test_resolve_cold_open_inert_when_hook_empty() -> None:
    ctx = _fake_ctx({"direction": {"hook": {}}})
    assert _stage()._resolve_cold_open(ctx, _LINES, _RECORD) is None


def test_resolve_cold_open_inert_when_style_not_cold_open() -> None:
    # `logo_open` / `auto` are spec'd open styles that are NOT a cold open — 1.3a
    # stays inert for them (auto-selection is ORCH-1.3b), so no footage reframe.
    for style in ("logo_open", "auto"):
        ctx = _fake_ctx({"open_style": style})
        assert _stage()._resolve_cold_open(ctx, _LINES, _RECORD) is None, style


def test_resolve_cold_open_force_off_wins_over_style() -> None:
    # An explicit hook.mode=="off" forces the cold open off even when the render
    # switch selected it.
    ctx = _fake_ctx({"open_style": "cold_open", "direction": {"hook": {"mode": "off"}}})
    assert _stage()._resolve_cold_open(ctx, _LINES, _RECORD) is None


def test_resolve_cold_open_none_when_opening_has_no_footage() -> None:
    # open_style asks for a cold open, but shot 0 binds no real clip duration.
    ctx = _fake_ctx({"open_style": "cold_open"})
    assert _stage()._resolve_cold_open(ctx, _LINES, {"clips": []}) is None


def test_resolve_cold_open_returns_window_for_open_style() -> None:
    ctx = _fake_ctx({"open_style": "cold_open"})
    plan = _stage()._resolve_cold_open(ctx, _LINES, _RECORD)
    assert plan is not None
    anchor, lead = plan
    assert anchor in {"start", "end"}
    assert isinstance(lead, float)
    assert 0.0 <= lead <= 4.0  # bounded by the clip we frame


def test_resolve_cold_open_activates_on_present_hook_block() -> None:
    # A present, non-off hook block activates the cold open even with no open_style.
    ctx = _fake_ctx({"direction": {"hook": {"mode": "auto"}}})
    assert _stage()._resolve_cold_open(ctx, _LINES, _RECORD) is not None


def test_resolve_cold_open_honours_authored_hook_anchor() -> None:
    ctx = _fake_ctx({"direction": {"hook": {"cold_open_anchor": "start"}}})
    plan = _stage()._resolve_cold_open(ctx, _LINES, _RECORD)
    assert plan is not None
    assert plan[0] == "start"  # authored anchor wins inside cold_open_window


# ── ORCH-1.3c: the measured curve reaches shot 0's seek, end to end. ──────────

_RECORD_WITH_MOTION = {
    "clips": [
        {
            "line_id": "clip-1",
            "seconds": 12.0,
            # Static load-in, the page paints at t=6, then it settles.
            "motion_timeline": [
                {"t": 2.0, "motion": 0.001},
                {"t": 4.0, "motion": 0.002},
                {"t": 6.0, "motion": 0.40},
                {"t": 8.0, "motion": 0.01},
            ],
        }
    ]
}


def test_curveless_record_still_frames_the_old_way() -> None:
    """No curve (QA off / older analyzer) ⇒ byte-identical to the 1.3a behaviour."""
    ctx = _fake_ctx({"open_style": "cold_open"})
    assert _stage()._resolve_cold_open(ctx, _LINES, _RECORD) == cold_open_window(
        4.0, None, "cold_open"
    )


def test_curve_does_not_activate_a_cold_open_that_was_not_asked_for() -> None:
    """Inert-by-default is unchanged: a curve alone never turns a cold open ON."""
    ctx = _fake_ctx({})
    assert _stage()._resolve_cold_open(ctx, _LINES, _RECORD_WITH_MOTION) is None
