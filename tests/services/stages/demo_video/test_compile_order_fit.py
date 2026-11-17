"""demo_video/showcase films must not overshoot the order at compile.

Two leaks, both measured on origin/main a4f3969d:

* ``DemoVideoCompileStage`` inherited the base ``title_card_plan`` (``ordered_seconds=
  None``), so the 20% card cap never applied — a 6s outro landed ON TOP of a 30s order.
* reconcile_timeline's per-beat ``footage_out_point_seconds`` had no reader, so every
  footage beat played its full raw recording.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

from src.services.stages.demo_video.compile import DemoVideoCompileStage
from tests.services.stages.demo_video.test_compile_trimmed_clip_window import (
    _compiler_shown_span,
)

FRAME = 1 / 30
FILE_SECONDS = 12.0


def _stage() -> DemoVideoCompileStage:
    return DemoVideoCompileStage.__new__(DemoVideoCompileStage)


def _ctx(cfg: dict[str, Any], outputs: dict[str, Any] | None = None) -> Any:
    outs = outputs or {}
    return SimpleNamespace(config_snapshot=copy.deepcopy(cfg), prior_output=outs.get)


# --- title cards count inside the order ---------------------------------------------


def test_30s_order_with_6s_outro_is_capped_to_20_percent() -> None:
    cfg = {
        "deliverable": {"target_duration_seconds": 30},
        "title_card": {"enabled": True, "intro": True, "outro": True, "duration_seconds": 6.0},
    }
    plan = _stage().title_card_plan(_ctx(cfg))
    assert plan is not None
    assert plan.total_seconds <= 0.20 * 30 + 1e-9


def test_lone_6s_outro_on_20s_order_is_shrunk_to_the_cap() -> None:
    cfg = {
        "deliverable": {"target_duration_seconds": 20},
        "title_card": {"enabled": True, "intro": False, "outro": True, "duration_seconds": 6.0},
    }
    plan = _stage().title_card_plan(_ctx(cfg))
    assert plan is not None
    assert plan.outro and not plan.intro
    assert plan.total_seconds == pytest.approx(4.0)


def test_explicit_timing_target_wins_over_deliverable() -> None:
    cfg = {
        "direction": {"timing": {"target_seconds": 10}},
        "deliverable": {"target_duration_seconds": 300},
        "title_card": {"enabled": True, "duration_seconds": 3.0},
    }
    plan = _stage().title_card_plan(_ctx(cfg))
    # 20% of 10s = 2.0s: two 1.5s-minimum cards cannot fit ⇒ outro dropped.
    assert plan is not None
    assert plan.card_count == 1 and plan.total_seconds == pytest.approx(2.0)


def test_no_order_plans_cards_as_configured() -> None:
    cfg = {"title_card": {"enabled": True, "duration_seconds": 6.0}}
    plan = _stage().title_card_plan(_ctx(cfg))
    assert plan is not None and plan.total_seconds == pytest.approx(12.0)


# --- reconcile's per-beat footage out-point -------------------------------------------


def _build(lines: list[dict[str, Any]], clips: list[dict[str, Any]], cfg: dict | None = None):
    """Compile ``lines``; a line's ``footage_out_point_seconds`` is moved where reconcile
    really writes it — ``timeline_manifest.beats[]`` — and OFF the line. No line carries
    the key ⇒ the manifest is ``None``, as reconcile emits it when nothing was trimmed."""
    lines = copy.deepcopy(lines)
    beats = [
        {"id": str(ln["id"]), "footage_out_point_seconds": ln.pop("footage_out_point_seconds")}
        for ln in lines
        if "footage_out_point_seconds" in ln
    ]
    outputs = {
        "demo_video_gen": {
            "lines": lines,
            "timeline_manifest": {"beats": beats, "run": {}} if beats else None,
        },
        "demo_video_narrate": {
            "clips": [
                {
                    "line_id": str(ln["id"]),
                    "artifact_url": f"http://a/{ln['id']}.mp3",
                    "seconds": 3.0,
                }
                for ln in lines
            ]
        },
        "demo_video_record": {"clips": copy.deepcopy(clips)},
    }
    return {s["line_id"]: s for s in _stage().build_shots(_ctx(cfg or {}, outputs))}


def _clip(line_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "line_id": line_id,
        "video_url": f"http://v/{line_id}.mp4",
        "seconds": FILE_SECONDS,
        **extra,
    }


@pytest.mark.parametrize("anchor", [None])
def test_footage_out_point_cuts_the_clip_within_one_frame(anchor: str | None) -> None:
    cut = 7.25
    line = {"id": "clip-1", "text": "add", "seconds": 7.25, "footage_out_point_seconds": cut}
    if anchor:
        line["clip_anchor"] = anchor
    shots = _build([line], [_clip("clip-1")])
    shot = shots["clip-1"]
    assert shot["clip_out_seconds"] == pytest.approx(cut, abs=FRAME)
    _start, end = _compiler_shown_span(shot, FILE_SECONDS)
    assert end <= cut + FRAME
    lead = shot.get("clip_lead_seconds")
    if lead is not None:
        assert lead < shot["clip_out_seconds"]


def test_footage_out_point_on_cold_open_end_anchor_shows_no_tail() -> None:
    cut = 5.0
    shots = _build(
        [{"id": "clip-1", "text": "open", "seconds": 5.0, "footage_out_point_seconds": cut}],
        [_clip("clip-1")],
        cfg={"open_style": "cold_open"},
    )
    shot = shots["clip-1"]
    assert shot["clip_anchor"] == "end"
    assert shot["clip_out_seconds"] == pytest.approx(cut, abs=FRAME)
    assert _compiler_shown_span(shot, FILE_SECONDS)[1] <= cut + FRAME


def test_earlier_record_trim_wins_over_a_later_out_point() -> None:
    shots = _build(
        [{"id": "clip-1", "text": "add", "seconds": 4.0, "footage_out_point_seconds": 9.0}],
        [_clip("clip-1", seconds=6.0, recorded_seconds=FILE_SECONDS)],
    )
    assert shots["clip-1"]["clip_out_seconds"] == pytest.approx(6.0)


def test_cut_is_per_beat_when_lines_share_a_clip() -> None:
    shots = _build(
        [
            {
                "id": "q",
                "clip": "clip-1",
                "text": "q",
                "seconds": 4.0,
                "footage_out_point_seconds": 4.0,
            },
            {"id": "a", "clip": "clip-1", "text": "a", "seconds": 4.0},
        ],
        [_clip("clip-1")],
    )
    assert shots["q"]["clip_out_seconds"] == pytest.approx(4.0)
    assert shots["a"].get("clip_out_seconds") is None


@pytest.mark.parametrize("value", [None])
def test_absent_or_unusable_out_point_is_byte_identical(value: object) -> None:
    base = {"id": "clip-1", "text": "add", "seconds": 5.0, "clip_anchor": "end"}
    with_field = {**base, "footage_out_point_seconds": value}
    assert _build([with_field], [_clip("clip-1")]) == _build([base], [_clip("clip-1")])


def test_out_point_past_the_clip_end_is_a_no_op() -> None:
    base = {"id": "clip-1", "text": "add", "seconds": 5.0}
    cut = {**base, "footage_out_point_seconds": FILE_SECONDS + 1.0}
    assert _build([cut], [_clip("clip-1")]) == _build([base], [_clip("clip-1")])


def test_an_out_point_on_the_line_itself_is_not_read() -> None:
    """reconcile never writes the key onto ``lines[]``; only the manifest is the contract."""
    base = {"id": "clip-1", "text": "add", "seconds": 5.0}
    outputs = {
        "demo_video_gen": {"lines": [{**base, "footage_out_point_seconds": 3.0}]},
        "demo_video_narrate": {
            "clips": [{"line_id": "clip-1", "artifact_url": "http://a/1.mp3", "seconds": 3.0}]
        },
        "demo_video_record": {"clips": [_clip("clip-1")]},
    }
    shots = _stage().build_shots(_ctx({}, outputs))
    assert shots[0].get("clip_out_seconds") is None


# --- round-5 P4: the cut never lands before the MEASURED narration ----------------------
#
# 01a11610-6ada (2026-10-07): reconcile's floor used 25 words at the pooled script-sample
# rate (162.14 wpm) ⇒ clip-4 out-point 9.251s; narrate measured that line at 9.454s, so the
# compiled shot froze its last 0.2s on a frame real footage existed for.


def _p4_build(narration_seconds: float) -> dict[str, Any]:
    line = {"id": "clip-4", "text": "w", "seconds": 9.251, "footage_out_point_seconds": 9.251}
    outputs = {
        "demo_video_gen": {
            "lines": [{k: v for k, v in line.items() if k != "footage_out_point_seconds"}],
            "timeline_manifest": {
                "beats": [{"id": "clip-4", "footage_out_point_seconds": 9.251}],
                "run": {},
            },
        },
        "demo_video_narrate": {
            "clips": [
                {
                    "line_id": "clip-4",
                    "artifact_url": "http://a/clip-4.mp3",
                    "seconds": narration_seconds,
                }
            ]
        },
        "demo_video_record": {"clips": [_clip("clip-4", seconds=13.08)]},
    }
    return _stage().build_shots(_ctx({}, outputs))[0]


def test_p4_out_point_is_raised_to_the_measured_narration() -> None:
    shot = _p4_build(9.453625)
    assert shot["clip_out_seconds"] == pytest.approx(9.453625, abs=FRAME)
    _start, end = _compiler_shown_span(shot, 13.08)
    assert end >= 9.453625 - FRAME, "footage cut before its own narration ends"


def test_p4_out_point_past_the_narration_is_kept() -> None:
    shot = _p4_build(8.0)
    assert shot["clip_out_seconds"] == pytest.approx(9.251, abs=FRAME)


def test_narration_longer_than_the_clip_plays_the_clip_untrimmed() -> None:
    shot = _p4_build(20.0)
    assert shot.get("clip_out_seconds") in (None, pytest.approx(13.08, abs=FRAME))
