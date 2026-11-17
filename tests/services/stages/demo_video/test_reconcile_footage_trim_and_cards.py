"""demo_video/showcase films overshooting the order — the reconcile half (stream D1).

Measured on origin/main a4f3969d:

* **Bed-tail run** — 30s order, 6s outro card, delivered 37.6s: the body was sized to the
  FULL order and nothing was reserved for the card.
* **P5** — footage minimums 32.36s on a 30s order, "3 of 3 beats pinned to footage": a
  footage beat's demand was ``min == clip`` and never trimmed.
* **01a1122a** — narration 32.82s, footage 32.36s, 30s order: proportional trim spent
  budget on beats whose segment was already narration-bound (+15.1%); trimming only the
  footage that runs PAST its narration lands +9.4%.

Owner rulings: cards count INSIDE the order, capped at 20% (``plan_title_cards``); footage
over the order is trimmed at the TAIL, floor ``max(narration, 60% clip, 2.0s)``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest

from src.services.llm import WMClient
from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.reconcile_timeline import (
    ReconcileTimelineStage,
)
from src.services.stages.shared.degradation import (
    DURATION_OUT_OF_TOLERANCE,
    FOOTAGE_TRIMMED,
    TIMELINE_OFF_TARGET,
    TIMELINE_UNDERFULL,
    TIMELINE_UNDERSPOKEN,
)
from src.services.stages.shared.time_budget import (
    collect_demands,
    projected_segment_seconds,
    trim_footage_tails,
)

_OUTRO_6S = {"enabled": True, "intro": False, "outro": True, "duration_seconds": 6.0}


def _words(n: int, tag: str = "w") -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def _ctx(
    *,
    lines: list[dict[str, Any]],
    clips: dict[str, float],
    order: float = 30.0,
    title_card: dict[str, Any] | None = None,
    relief: bool = False,
) -> RunContext:
    timing: dict[str, Any] = {"mode": "auto"}
    if relief:
        timing["relief"] = {"mode": "auto", "iterations": 1, "score_threshold": 4.0}
    snapshot: dict[str, Any] = {
        "direction": {"timing": timing},
        "deliverable": {"target_duration_seconds": order},
        "voice": {"voice_id": "af_heart"},
    }
    if title_card is not None:
        snapshot["title_card"] = title_card
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=snapshot,
        prior_run_outputs={
            "demo_video_gen": {
                "lines": lines,
                "bdd_job_id": "",
                "features_generated": 0,
                "package_url": "",
            },
            "demo_video_record": {
                "clips": [
                    {
                        "id": cid,
                        "line_id": cid,
                        "video_url": f"https://minio.test/{cid}.mp4",
                        "seconds": s,
                    }
                    for cid, s in clips.items()
                ]
            },
        },
    )


# ── 1. cards count inside the order ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bed_tail_narration_body_fits_inside_order_minus_outro() -> None:
    # 3 VO beats of 22 words: Σmin 23.3s (170 wpm), Σmax 30.5s (130 wpm) — content that
    # can fill EITHER 24s or the full 30s, so the solver's target alone decides the body.
    # Before the card reservation it filled the whole 30s order and the outro ran past it.
    lines = [{"id": f"vo-{i}", "text": _words(22), "seconds": None, "clip": None} for i in range(3)]
    stage = ReconcileTimelineStage()
    out = await stage.handle(stage.input_model(), _ctx(lines=lines, clips={}, title_card=_OUTRO_6S))
    body = sum(float(line.seconds) for line in out.lines)
    assert body <= 24.0 + 1e-6, f"body {body:.2f}s leaves no room for the 6s outro"


# ── 2. footage tails trimmed (P5) ────────────────────────────────────────────────────

_P5_CLIPS = {"clip-1": 12.0, "clip-2": 10.36, "clip-3": 10.0}  # 32.36s on a 30s order


@pytest.mark.asyncio
async def test_p5_footage_over_the_order_is_tail_trimmed_to_fit() -> None:
    lines = [{"id": cid, "text": _words(5), "seconds": None, "clip": cid} for cid in _P5_CLIPS]
    stage = ReconcileTimelineStage()
    out = await stage.handle(stage.input_model(), _ctx(lines=lines, clips=_P5_CLIPS))

    total = sum(float(line.seconds) for line in out.lines)
    assert total <= 30.0 * 1.02 + 1e-6  # the 2% reconcile tolerance
    # The manifest is the compile's contract — present even on a non-relief run.
    assert out.timeline_manifest is not None
    points = {b.id: b.footage_out_point_seconds for b in out.timeline_manifest.beats}
    # Largest footage-over-narration first: clip-1 (12.0s vs 2.0s of words) absorbs the
    # whole 2.36s excess; the others play untrimmed.
    assert points["clip-1"] == pytest.approx(12.0 - 2.36)
    assert points["clip-2"] is None
    assert points["clip-3"] is None
    trimmed = [d for d in out.degradations if d["code"] == FOOTAGE_TRIMMED]
    assert len(trimmed) == 1
    assert "clip-1: -2.36s" in trimmed[0]["detail"]
    # Fits ⇒ nothing reports the order as missed.
    assert not [d for d in out.degradations if d["code"] == DURATION_OUT_OF_TOLERANCE]
    # A non-relief run's synthesized manifest must not pretend relief ran.
    assert out.timeline_manifest.run.loop_iterations == 0


def test_footage_is_never_trimmed_below_its_floor() -> None:
    # 3 x 20s silent footage on a 10s order: floors are 60% of the clip (12s each).
    demands = collect_demands(
        [{"id": f"c{i}", "text": "", "clip": f"c{i}"} for i in range(3)],
        clip_seconds={f"c{i}": 20.0 for i in range(3)},
        target=10.0,
    )
    assert [d.footage_out_point_seconds for d in demands] == [12.0, 12.0, 12.0]


def test_a_short_clip_is_floored_at_two_seconds_and_never_extended() -> None:
    demands = collect_demands(
        [
            {"id": "a", "text": "", "clip": "a"},
            {"id": "b", "text": "", "clip": "b"},
        ],
        clip_seconds={"a": 3.0, "b": 1.5},
        target=1.0,
    )
    assert demands[0].footage_out_point_seconds == pytest.approx(2.0)  # max(1.8, 2.0)
    assert demands[1].footage_out_point_seconds is None  # 1.5s clip: floor 2.0 > clip


def test_time_locked_and_author_bounded_footage_is_never_trimmed() -> None:
    demands = collect_demands(
        [
            {"id": "a", "text": "", "clip": "a", "time_lock": True, "seconds": 20.0},
            {"id": "b", "text": "", "clip": "b", "max_seconds": 20.0},
        ],
        clip_seconds={"a": 20.0, "b": 20.0},
        target=5.0,
    )
    assert [d.footage_out_point_seconds for d in demands] == [None, None]


@pytest.mark.parametrize(
    "window",
    [{"clip_out_seconds": 15.0}],
)
def test_footage_with_an_authored_window_is_never_trimmed(window: dict[str, Any]) -> None:
    # The out-point is FILE time from 0; a 20s clip at a 10s order would be cut to its
    # 12.0 floor — inside a 10-15s authored window. The author's window wins.
    demands = collect_demands(
        [{"id": "a", "text": "", "clip": "a", **window}],
        clip_seconds={"a": 20.0},
        target=10.0,
    )
    assert demands[0].footage_out_point_seconds is None


def test_an_end_anchor_alone_is_still_trimmable() -> None:
    demands = collect_demands(
        [{"id": "a", "text": "", "clip": "a", "clip_anchor": "end"}],
        clip_seconds={"a": 20.0},
        target=10.0,
    )
    assert demands[0].footage_out_point_seconds == pytest.approx(12.0)


def test_no_target_means_no_trim() -> None:
    demands = collect_demands([{"id": "a", "text": "", "clip": "a"}], clip_seconds={"a": 40.0})
    assert demands[0].ideal == 40.0
    assert demands[0].footage_out_point_seconds is None


# ── 3. narration-aware: 01a1122a ─────────────────────────────────────────────────────
#
# Footage 12.0 / 10.36 / 10.0 = 32.36s; narration 20 / 32 / 30 words at 150 wpm =
# 8.0 / 12.8 / 12.0 = 32.8s; order 30s. Segments max(n, f) = 12.0 + 12.8 + 12.0 = 36.8s.
# Only clip-1's footage runs past its narration (by 4.0s) — cutting clips 2/3 buys
# nothing, their segments are narration-bound. Proportional trim would have spent cuts on
# them; narration-aware trim takes clip-1 to its narration (8.0s) and stops.

_A1122A_WORDS = {"clip-1": 20, "clip-2": 32, "clip-3": 30}


def _a1122a_lines() -> list[dict[str, Any]]:
    return [
        {"id": cid, "text": _words(n), "seconds": None, "clip": cid}
        for cid, n in _A1122A_WORDS.items()
    ]


def test_01a1122a_trims_only_footage_that_runs_past_its_narration() -> None:
    demands = collect_demands(
        _a1122a_lines(), clip_seconds=_P5_CLIPS, wpm_natural=150.0, target=30.0
    )
    points = {d.id: d.footage_out_point_seconds for d in demands}
    assert points == {"clip-1": pytest.approx(8.0), "clip-2": None, "clip-3": None}
    # Never cut below narration ⇒ the residual is narration-bound: 32.8s, not 30s.
    assert projected_segment_seconds(demands, 150.0) == pytest.approx(32.8)


def test_proportional_trim_would_have_wasted_cuts_on_narration_bound_beats() -> None:
    """The guard against regressing to proportional trim: untrimmed beats stay at the clip."""
    untrimmed = collect_demands(_a1122a_lines(), clip_seconds=_P5_CLIPS, wpm_natural=150.0)
    trimmed = trim_footage_tails(untrimmed, target=30.0, wpm_natural=150.0)
    assert [d.ideal for d in trimmed] == pytest.approx([8.0, 10.36, 10.0])


@pytest.mark.asyncio
async def test_01a1122a_residual_over_the_order_is_reported_not_hidden() -> None:
    stage = ReconcileTimelineStage()
    out = await stage.handle(stage.input_model(), _ctx(lines=_a1122a_lines(), clips=_P5_CLIPS))
    codes = [d["code"] for d in out.degradations]
    assert FOOTAGE_TRIMMED in codes
    assert DURATION_OUT_OF_TOLERANCE in codes
    assert out.timeline_manifest is not None
    points = {b.id: b.footage_out_point_seconds for b in out.timeline_manifest.beats}
    assert points["clip-1"] == pytest.approx(8.0)
    assert points["clip-2"] is None and points["clip-3"] is None


# ── 4. the underspoken re-ask is capped for the WHOLE spine ──────────────────────────


class _ExpanderWM:
    def __init__(self, reply_words: int) -> None:
        self.reply = _words(reply_words, "x")
        self.calls: list[str] = []

    async def message(self, **kwargs: Any) -> Any:
        worker = str(kwargs.get("worker_name") or "")
        self.calls.append(worker)
        if worker == "beat-expander":
            return SimpleNamespace(output={"text": self.reply}, output_type="object")
        if worker == "pacing-critic":
            return SimpleNamespace(output={"overall_score": 4}, output_type="object")
        return SimpleNamespace(output={"directives": []}, output_type="object")


def _reask_ctx() -> RunContext:
    # 3 footage beats, 8.0s clips, 8 words each (3.2s of narration) ⇒ underspoken. 30s
    # order with a 6s outro ⇒ a 24s body, which the 24s of footage fills without a trim.
    # Spine cap: floor(24 x 150 / 60 x 1.10) = 66 words. Per-beat window ceiling: 23
    # (8.0s + 1.5s pad at 150 wpm) ⇒ 3 x 23 = 69 > 66.
    clips = {f"clip-{i}": 8.0 for i in range(3)}
    lines = [{"id": cid, "text": _words(8), "seconds": None, "clip": cid} for cid in clips]
    return _ctx(lines=lines, clips=clips, title_card=_OUTRO_6S, relief=True)


@pytest.mark.asyncio
async def test_reask_cannot_push_total_words_past_the_spine_ceiling() -> None:
    # Each reply sits exactly at the per-beat window ceiling (23). Before the spine cap all
    # three were applied: 69 words on a 66-word ceiling.
    wm = _ExpanderWM(23)
    stage = ReconcileTimelineStage(client=cast("WMClient", wm))
    out = await stage.handle(stage.input_model(), _reask_ctx())

    assert "beat-expander" in wm.calls, "precondition: the re-ask ran"
    total = sum(len(str(line.text).split()) for line in out.lines)
    assert total <= 66, f"re-ask grew the spine to {total} words, past the 66-word ceiling"


@pytest.mark.asyncio
async def test_reask_inside_the_shrunk_ceiling_is_still_applied() -> None:
    # The proportional share is 22 words per beat — a 22-word reply is admitted on all three.
    wm = _ExpanderWM(22)
    stage = ReconcileTimelineStage(client=cast("WMClient", wm))
    out = await stage.handle(stage.input_model(), _reask_ctx())

    words = [len(str(line.text).split()) for line in out.lines]
    assert words == [22, 22, 22]
    assert sum(words) == 66


# ── 5. round-5 P1: a trimmed, in-band footage film is not "underfull" ────────────────
#
# 01a11600-aad1 (2026-10-07): 3 footage beats, 21 / 13 / 25 words, clips 13.92 / 10.80 /
# 9.64s on a 30s order. Trimmed to 29.70s and delivered 29.77s, yet stamped
# `timeline_underfull` "fills only 25.97s" + `timeline_off_target` beside the
# `footage_trimmed` note that says the film OVERRAN. Not staleness: `budget_drift`
# subtracted the short narration's "uncovered" slack on clip-1 as if the clip's own
# picture did not play under it.


@pytest.mark.asyncio
async def test_p1_trimmed_footage_film_in_band_is_not_reported_underfull() -> None:
    clips = {"clip-1": 13.92, "clip-4": 10.80, "clip-5": 9.64}
    words = {"clip-1": 21, "clip-4": 13, "clip-5": 25}
    lines = [{"id": c, "text": _words(words[c]), "seconds": None, "clip": c} for c in clips]
    stage = ReconcileTimelineStage()
    out = await stage.handle(stage.input_model(), _ctx(lines=lines, clips=clips))

    codes = [d["code"] for d in out.degradations]
    assert FOOTAGE_TRIMMED in codes, "precondition: the footage was tail-trimmed"
    assert TIMELINE_UNDERSPOKEN in codes, "precondition: clip-1's narration IS short"
    body = sum(float(line.seconds) for line in out.lines)
    assert body == pytest.approx(30.0, abs=0.6), "precondition: the final beats are in band"
    assert TIMELINE_UNDERFULL not in codes
    assert TIMELINE_OFF_TARGET not in codes
