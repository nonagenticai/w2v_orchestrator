"""A GENERATED beat with no anchor starts on its peak of motion, not on page load.

Generated lines carry ``clip_anchor=None`` (``ShowcaseLine``), which the compiler reads
as ``start`` with lead 0 — the first frame of the recording, i.e. a blank page load.
Only an explicit ``clip_anchor: "auto"`` used to get the motion-peak lead. A generated
beat has no author to have chosen one, so ``build_shots`` now resolves it the same way
``"auto"`` does, falling back to the head when the curve has no distinguished peak,
and never lets ``lead + shot seconds`` run past the clip's usable out-point.

Author-set anchors (any value, or a numeric lead) and authored spines are untouched.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

from src.services.stages.demo_video.compile import DemoVideoCompileStage

# Static load-in, the page paints at t=6, then settles ⇒ lead 6.0 - 0.35 = 5.65.
_PEAK: list[dict[str, float]] = [
    {"t": 2.0, "motion": 0.001},
    {"t": 4.0, "motion": 0.002},
    {"t": 6.0, "motion": 0.40},
    {"t": 8.0, "motion": 0.01},
]
_FLAT: list[dict[str, float]] = [
    {"t": 1.0, "motion": 0.1},
    {"t": 2.0, "motion": 0.1},
    {"t": 3.0, "motion": 0.1},
]


def _build(
    lines: list[dict[str, Any]],
    clips: list[dict[str, Any]],
    cfg: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    narrate = {
        "clips": [
            {"line_id": str(ln["id"]), "artifact_url": f"http://a/{ln['id']}.mp3", "seconds": 3.0}
            for ln in lines
        ]
    }
    outputs: dict[str, Any] = {
        "demo_video_gen": {"lines": copy.deepcopy(lines), "bdd_job_id": "job-1"},
        "demo_video_narrate": narrate,
        "demo_video_record": {"clips": copy.deepcopy(clips)},
    }
    ctx: Any = SimpleNamespace(
        config_snapshot=copy.deepcopy(cfg or {}),
        prior_output=lambda stage: outputs.get(stage),
    )
    stage = DemoVideoCompileStage.__new__(DemoVideoCompileStage)
    return {s["line_id"]: s for s in stage.build_shots(ctx)}


def _clip(line_id: str, seconds: float, timeline: list[dict[str, float]] | None, **extra: Any):
    clip: dict[str, Any] = {
        "line_id": line_id,
        "video_url": f"http://v/{line_id}.mp4",
        "seconds": seconds,
        **extra,
    }
    if timeline is not None:
        clip["motion_timeline"] = copy.deepcopy(timeline)
    return clip


_OPENER = {"id": "clip-1", "text": "open", "seconds": 4.0}
_OPENER_CLIP = _clip("clip-1", 4.0, None)


def test_generated_beat_with_no_distinguished_peak_falls_back_to_start() -> None:
    shots = _build(
        [_OPENER, {"id": "clip-2", "text": "mid", "seconds": 4.0}],
        [_OPENER_CLIP, _clip("clip-2", 6.0, _FLAT)],
    )
    assert shots["clip-2"].get("clip_anchor") is None
    assert shots["clip-2"].get("clip_lead_seconds") is None


def test_peak_lead_never_runs_the_shot_past_an_untrimmed_clip_end() -> None:
    # 12s clip, 10s shot: lead 5.65 + 10 = 15.65 > 12 ⇒ clamped to 12 - 10 = 2.0.
    shots = _build(
        [_OPENER, {"id": "clip-2", "text": "mid", "seconds": 10.0}],
        [_OPENER_CLIP, _clip("clip-2", 12.0, _PEAK)],
    )
    shot = shots["clip-2"]
    assert shot["clip_anchor"] == "start"
    assert shot["clip_lead_seconds"] == pytest.approx(2.0)
    assert shot["clip_lead_seconds"] + shot["seconds"] <= 12.0 + 1e-9


def test_peak_lead_is_bounded_by_the_trimmed_window_out_point() -> None:
    # A record-trimmed clip: usable 9.56s of a 14.56s file. Peak at t=8 ⇒ raw lead 7.65;
    # a 4s shot from 7.65 would reach 11.65, into the dead tail.
    peak_late = [
        {"t": 2.0, "motion": 0.001},
        {"t": 4.0, "motion": 0.002},
        {"t": 8.0, "motion": 0.40},
        {"t": 9.0, "motion": 0.01},
    ]
    shots = _build(
        [_OPENER, {"id": "clip-2", "text": "mid", "seconds": 4.0}],
        [_OPENER_CLIP, _clip("clip-2", 9.56, peak_late, recorded_seconds=14.56)],
    )
    shot = shots["clip-2"]
    assert shot["clip_anchor"] == "start"
    assert shot["clip_out_seconds"] == pytest.approx(9.56)
    assert shot["clip_lead_seconds"] == pytest.approx(9.56 - 4.0)
    assert shot["clip_lead_seconds"] + shot["seconds"] <= shot["clip_out_seconds"] + 1e-9


def test_author_set_anchor_on_a_generated_beat_is_untouched() -> None:
    shots = _build(
        [
            _OPENER,
            {"id": "clip-2", "text": "mid", "seconds": 4.0, "clip_anchor": "end"},
            {"id": "clip-3", "text": "late", "seconds": 4.0, "clip_lead_seconds": 1.2},
        ],
        [_OPENER_CLIP, _clip("clip-2", 12.0, _PEAK), _clip("clip-3", 12.0, _PEAK)],
    )
    assert shots["clip-2"]["clip_anchor"] == "end"
    assert shots["clip-2"].get("clip_lead_seconds") is None
    assert shots["clip-3"].get("clip_anchor") is None
    assert shots["clip-3"]["clip_lead_seconds"] == 1.2


def test_authored_spine_without_anchor_is_untouched() -> None:
    lines = [_OPENER, {"id": "clip-2", "text": "mid", "seconds": 4.0}]
    brief = {"lines": [{"id": ln["id"], "text": ln["text"]} for ln in lines]}
    shots = _build(
        lines,
        [_OPENER_CLIP, _clip("clip-2", 12.0, _PEAK)],
        {"brief": brief},
    )
    assert shots["clip-2"].get("clip_anchor") is None
    assert shots["clip-2"].get("clip_lead_seconds") is None


def test_generated_opener_stays_deferred_to_an_active_cold_open() -> None:
    lines = [{"id": "clip-1", "text": "open", "seconds": 4.0}, {"id": "clip-2", "text": "m"}]
    clips = [_clip("clip-1", 12.0, _PEAK), _clip("clip-2", 4.0, None)]
    cfg = {"open_style": "cold_open"}
    absent = _build(lines, clips, cfg)["clip-1"]
    auto = _build([{**lines[0], "clip_anchor": "auto"}, lines[1]], clips, cfg)["clip-1"]
    # "auto" on the opener is deferred to cold_open; the no-anchor generated opener must
    # land on exactly the same framing (cold_open's), not a second C4 stamp.
    assert absent == auto
