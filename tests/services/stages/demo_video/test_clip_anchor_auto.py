"""C4 content-aware ``clip_anchor: "auto"`` resolution in the demo_video compile.

The layered ladder:

* absent / explicit (``start`` / ``end`` / numeric lead) ⇒ author wins, the value
  passes through byte-identically (today's behaviour);
* ``"auto"`` ⇒ resolve the motion-settled hero-frame lead off the clip's
  ``motion_timeline`` and emit ``("start", lead)``;
* an unresolvable curve (flat / too-short / no clip / zero-length) ⇒ degrade to the
  default ``(None, None)`` framing;
* the opener (idx 0) is DEFERRED to the richer, hook-aware cold_open pass whenever
  that pass is active — C4 never force-stamps a shot cold_open already owns.

Pure + deterministic: a fake ``ctx`` carries a ``config_snapshot`` + ``prior_output``
and the stage is built via ``__new__`` (``build_shots`` does not use ``self``), mirroring
``test_footage_direction_compile``.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

from src.services.stages.demo_video.compile import (
    DemoVideoCompileStage,
)

# A distinguished-peak curve: a static load-in, the page paints hard at t=6, then it
# settles — peak 0.40 stands well clear of the ~0 mean (12s clip ⇒ lead 6.0-0.35=5.65).
_PEAK_MOTION: list[dict[str, float]] = [
    {"t": 2.0, "motion": 0.001},
    {"t": 4.0, "motion": 0.002},
    {"t": 6.0, "motion": 0.40},
    {"t": 8.0, "motion": 0.01},
]


# ── _resolve_clip_anchor units ────────────────────────────────────────────────


# ── build_shots integration (fake ctx, no DB) ─────────────────────────────────
def _stage() -> DemoVideoCompileStage:
    return DemoVideoCompileStage.__new__(DemoVideoCompileStage)


def _ctx(
    *,
    lines: list[dict[str, Any]],
    clips: list[dict[str, Any]],
    config_snapshot: dict[str, Any] | None = None,
) -> Any:
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
    return SimpleNamespace(
        config_snapshot=copy.deepcopy(config_snapshot or {}),
        prior_output=lambda stage: outputs.get(stage),
    )


def _by_line(shots: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {s["line_id"]: s for s in shots}


def test_build_shots_resolves_non_opening_auto_line() -> None:
    ctx = _ctx(
        lines=[
            {"id": "clip-1", "text": "open"},
            {"id": "clip-2", "text": "mid", "clip_anchor": "auto"},
        ],
        clips=[
            {"line_id": "clip-1", "video_url": "http://v/1.mp4", "seconds": 4.0},
            {
                "line_id": "clip-2",
                "video_url": "http://v/2.mp4",
                "seconds": 12.0,
                "motion_timeline": copy.deepcopy(_PEAK_MOTION),
            },
        ],
    )
    shot = _by_line(_stage().build_shots(ctx))["clip-2"]
    assert shot["clip_anchor"] == "start"
    assert isinstance(shot["clip_lead_seconds"], float)
    assert shot["clip_lead_seconds"] == 5.65


def test_build_shots_without_auto_is_byte_identical_passthrough() -> None:
    # No "auto": explicit values copy through verbatim; an absent anchor stamps no
    # framing key at all (today's behaviour).
    ctx = _ctx(
        lines=[
            {"id": "clip-1", "text": "a"},
            {"id": "clip-2", "text": "b", "clip_anchor": "end", "clip_lead_seconds": 0.5},
        ],
        clips=[
            {"line_id": "clip-1", "video_url": "http://v/1.mp4", "seconds": 4.0},
            {"line_id": "clip-2", "video_url": "http://v/2.mp4", "seconds": 4.0},
        ],
    )
    by = _by_line(_stage().build_shots(ctx))
    assert by["clip-2"]["clip_anchor"] == "end"
    assert by["clip-2"]["clip_lead_seconds"] == 0.5
    assert "clip_anchor" not in by["clip-1"]  # absent ⇒ no key, unchanged


def test_build_shots_opener_auto_deferred_to_cold_open() -> None:
    # lines[0] is "auto" AND cold_open is active: C4 must NOT force-stamp shot 0 —
    # it stays whatever cold_open owns, which is proven by byte-identity against the
    # same run with NO authored anchor on line 0 (that line can't be C4-resolved, so
    # an identical shot 0 means the cold_open pass, not C4, produced the framing).
    clips = [
        {
            "line_id": "clip-1",
            "video_url": "http://v/1.mp4",
            "seconds": 12.0,
            "motion_timeline": copy.deepcopy(_PEAK_MOTION),
        },
        {"line_id": "clip-2", "video_url": "http://v/2.mp4", "seconds": 4.0},
    ]
    cfg = {"open_style": "cold_open"}
    auto = _by_line(
        _stage().build_shots(
            _ctx(
                lines=[
                    {"id": "clip-1", "text": "open", "clip_anchor": "auto"},
                    {"id": "clip-2", "text": "mid"},
                ],
                clips=copy.deepcopy(clips),
                config_snapshot=cfg,
            )
        )
    )["clip-1"]
    absent = _by_line(
        _stage().build_shots(
            _ctx(
                lines=[{"id": "clip-1", "text": "open"}, {"id": "clip-2", "text": "mid"}],
                clips=copy.deepcopy(clips),
                config_snapshot=cfg,
            )
        )
    )["clip-1"]
    assert auto == absent  # C4 deferred — cold_open owns shot 0 in both
    assert absent["clip_anchor"] == "start"  # …and cold_open did frame it
