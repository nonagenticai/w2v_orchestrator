"""Unit tests for ReconcileTimelineStage — deterministic, zero-LLM (no WM, no DB).

The stage is pure compute: the ``shared/time_budget.py`` solver runs FOR REAL and no
worker_management client exists to mock. Contracts pinned here:

* presence-opt-in — absent ``direction.timing`` (or ``mode == "off"``, or no resolvable
  target) skips and leaves the spine untouched;
* the happy path — a feasible timeline stamps its allotment and emits NO notes;
* C0 — footage beats keep their measured clip length exactly; ids / order / ``clip``
  never change;
* HONESTY — what the solver cannot fix is surfaced on ``notes`` (which auto-marks the
  run ``degraded``), never swallowed: an infeasible timeline (``Σmin > target``), an
  underfull one (``Σmax < target``), and an unsayable beat each get a note.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.base import StageSkippedError
from src.services.stages.demo_video import reconcile_timeline as reconcile_timeline_module
from src.services.stages.demo_video.reconcile_timeline import (
    ReconcileTimelineStage,
    _TimingConfig,
)
from src.services.stages.shared.time_budget import (
    DEFAULT_FREEZE_PAD_CEILING,
    DEFAULT_WPM_NATURAL,
    MAX_PAD_CREDIT_RATIO,
    VO_RATE_DRAG_FLOOR,
    Allotment,
    content_floor_credit,
    max_content_floor_credit,
)
from src.services.stages.upstream import PermanentUpstreamError


def _timing_config_for(*, target: float, pad_ceiling: float = 2.0) -> _TimingConfig:
    """A `_TimingConfig` carrying only the fields the underfull gate reads."""
    return _TimingConfig(
        mode="auto",
        target=target,
        tolerance_pct=2.0,
        wpm_band=(120.0, 180.0),
        wpm_natural=150.0,
        freeze_pad_ceiling_seconds=pad_ceiling,
        vo_rate_drag_floor=VO_RATE_DRAG_FLOOR,
        relief_mode="off",
        relief_iterations=0,
        relief_threshold=4.0,
    )


# ~40 words: at 170 wpm its floor alone is ~14s, so two of these blow a 5s target.
_LONG = " ".join(["word"] * 40)
_SHORT = "This beat introduces the product in one short line."

# _SHORT is 9 words ⇒ [3.18s, 4.15s]; with the 4s pinned clip the fillable band for
# the default spine is roughly [10.4s, 12.3s]. 11s sits inside it: feasible AND fillable.
_FEASIBLE_TARGET = 11.0


def _lines(text: str = _SHORT) -> list[dict[str, Any]]:
    """clip-1 = VO-only, clip-2 = footage-only (no narration), clip-3 = VO-only."""
    return [
        {"id": "clip-1", "text": text, "seconds": 3.0, "clip": None},
        {"id": "clip-2", "text": "", "seconds": 5.0, "clip": "clip-2"},
        {"id": "clip-3", "text": text, "seconds": 3.0, "clip": None},
    ]


def _clips(seconds: float = 4.0, line_id: str = "clip-2") -> list[dict[str, Any]]:
    return [
        {
            "id": line_id,
            "line_id": line_id,
            "video_url": f"https://minio.test/{line_id}.mp4",
            "seconds": seconds,
        }
    ]


def _ctx(
    *,
    timing: dict[str, Any] | None,
    lines: list[dict[str, Any]] | None = None,
    clips: list[dict[str, Any]] | None = None,
    target_seconds: float | None = None,
) -> RunContext:
    cfg: dict[str, Any] = {}
    if timing is not None:
        cfg["direction"] = {"timing": timing}
    if target_seconds is not None:
        cfg["deliverable"] = {"target_duration_seconds": target_seconds}
    prior: dict[str, Any] = {
        "demo_video_gen": {
            "lines": lines if lines is not None else _lines(),
            "bdd_job_id": "",
            "features_generated": 0,
            "package_url": "",
        },
        "demo_video_record": {"clips": clips or []},
    }
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=cfg,
        prior_run_outputs=prior,
    )


def _set_underfull_ratio(monkeypatch: pytest.MonkeyPatch, ratio: float) -> None:
    """Set the gross-underfill hard floor (0.0 = gate off) and drop the settings cache."""
    from src.config import clear_settings_cache

    monkeypatch.setenv("PIPELINE_DEMO_TIMELINE_UNDERFULL_FAIL_RATIO", str(ratio))
    # conftest's autouse `_clear_settings` drops the cache again after the test.
    clear_settings_cache()


# ── (a) presence-opt-in: absent / off / target-less blocks skip clean ─────────


@pytest.mark.asyncio
@pytest.mark.parametrize("timing", [None])
async def test_absent_or_off_block_skips(timing: dict[str, Any] | None) -> None:
    stage = ReconcileTimelineStage()
    ctx = _ctx(timing=timing, target_seconds=30.0)

    with pytest.raises(StageSkippedError):
        await stage.handle(stage.input_model(), ctx)


@pytest.mark.asyncio
async def test_no_target_skips() -> None:
    stage = ReconcileTimelineStage()
    # No direction.timing.target_seconds AND no deliverable.target_duration_seconds.
    ctx = _ctx(timing={"mode": "auto"}, target_seconds=None)

    with pytest.raises(StageSkippedError):
        await stage.handle(stage.input_model(), ctx)


@pytest.mark.asyncio
async def test_no_lines_skips() -> None:
    stage = ReconcileTimelineStage()
    ctx = _ctx(timing={"mode": "auto"}, lines=[], target_seconds=30.0)

    with pytest.raises(StageSkippedError):
        await stage.handle(stage.input_model(), ctx)


# ── (b) the happy path: a feasible timeline is stamped, with NO notes ─────────


@pytest.mark.asyncio
async def test_feasible_timeline_applies_allotments_and_emits_no_notes() -> None:
    stage = ReconcileTimelineStage()
    ctx = _ctx(timing={"mode": "auto"}, clips=_clips(), target_seconds=_FEASIBLE_TARGET)

    out = await stage.handle(stage.input_model(), ctx)

    assert [ln.id for ln in out.lines] == ["clip-1", "clip-2", "clip-3"]
    assert all(ln.seconds is not None and ln.seconds > 0 for ln in out.lines)
    assert all(ln.min_seconds is not None for ln in out.lines)
    assert all(ln.max_seconds is not None for ln in out.lines)
    total = sum(float(ln.seconds or 0.0) for ln in out.lines)
    assert total == pytest.approx(_FEASIBLE_TARGET, abs=0.05)
    # Nothing the solver could not fix ⇒ a clean (non-degraded) run.
    assert out.notes == []


# ── (c) C0: footage beats keep their MEASURED seconds; ids/order/clip intact ──


@pytest.mark.asyncio
async def test_footage_beat_keeps_measured_seconds() -> None:
    stage = ReconcileTimelineStage()
    ctx = _ctx(timing={"mode": "auto"}, clips=_clips(seconds=4.0), target_seconds=_FEASIBLE_TARGET)

    out = await stage.handle(stage.input_model(), ctx)

    # clip-2 is pinned to the ffprobe'd length of its recorded footage — bit-for-bit,
    # NOT the 5.0s the spine authored.
    assert out.lines[1].seconds == pytest.approx(4.0)
    assert [ln.id for ln in out.lines] == ["clip-1", "clip-2", "clip-3"]
    assert [ln.clip for ln in out.lines] == [None, "clip-2", None]


# ── (d) vo_rate is stamped on VO beats and absent on a footage-only beat ──────


@pytest.mark.asyncio
async def test_vo_rate_stamped_on_vo_beats_absent_on_footage() -> None:
    stage = ReconcileTimelineStage()
    ctx = _ctx(timing={"mode": "auto"}, clips=_clips(), target_seconds=_FEASIBLE_TARGET)

    out = await stage.handle(stage.input_model(), ctx)

    assert out.lines[0].vo_rate is not None  # narrated beat: narrate speaks to fit
    assert out.lines[2].vo_rate is not None
    assert out.lines[1].vo_rate is None  # footage-only beat: nothing to say


# ── (e) reconcile_only: compute the timeline, mutate nothing ──────────────────


@pytest.mark.asyncio
async def test_reconcile_only_mutates_nothing() -> None:
    stage = ReconcileTimelineStage()
    ctx = _ctx(timing={"mode": "reconcile_only"}, clips=_clips(), target_seconds=_FEASIBLE_TARGET)

    out = await stage.handle(stage.input_model(), ctx)

    assert [ln.seconds for ln in out.lines] == [3.0, 5.0, 3.0]  # authored, untouched


# ── (f) infeasible (Σmin > target) ⇒ flagged on notes, never faked ────────────


@pytest.mark.asyncio
async def test_infeasible_timeline_is_flagged_on_notes() -> None:
    # Two 40-word beats floor at ~14s each — a 5s target is unreachable by arithmetic.
    stage = ReconcileTimelineStage()
    ctx = _ctx(
        timing={"mode": "auto"},
        lines=_lines(_LONG),
        clips=_clips(),
        target_seconds=5.0,
    )

    out = await stage.handle(stage.input_model(), ctx)

    assert any("INFEASIBLE" in note for note in out.notes)  # ⇒ degraded run
    assert any("clip-1" in note for note in out.notes)
    # The spine survives: ids, order and the footage join are untouched, and no beat is
    # trimmed below its floor to fake a fit.
    assert [ln.id for ln in out.lines] == ["clip-1", "clip-2", "clip-3"]
    assert [ln.clip for ln in out.lines] == [None, "clip-2", None]
    assert sum(float(ln.seconds or 0.0) for ln in out.lines) > 5.0


# ── (g) underfull (Σmax < target) ⇒ flagged on notes (the C1 regression) ──────


@pytest.mark.asyncio
async def test_underfull_timeline_is_flagged_on_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Σmin (~10.4s) <= 30s, so infeasible_beats() is EMPTY and the timeline reads
    # "feasible" — but every beat saturates at its max and the film fills ~12s of a 30s
    # target. Silently shipping that was the C1 defect.
    # ~12s of 30s is now ALSO below the gross-underfill hard floor, so the gate is
    # disabled here to keep this test on its original subject (the NOTE). The gate's own
    # behaviour is pinned in section (g2) below.
    _set_underfull_ratio(monkeypatch, 0.0)
    stage = ReconcileTimelineStage()
    ctx = _ctx(timing={"mode": "auto"}, clips=_clips(), target_seconds=30.0)

    out = await stage.handle(stage.input_model(), ctx)

    assert any("UNDERFULL" in note for note in out.notes)  # ⇒ degraded run
    total = sum(float(ln.seconds or 0.0) for ln in out.lines)
    assert total < 30.0
    # The solver did NOT invent seconds past each beat's max.
    for line in out.lines:
        assert float(line.seconds or 0.0) <= float(line.max_seconds or 0.0) + 1e-9


# ── (g2) GROSS underfill ⇒ the run FAILS; mild underfill still only warns ─────
#
# Prod run 01a00b81 shipped an 11.4s video against a 60s target, `completed`, with the
# UNDERFULL note as its only trace. These pin the hard gate that stops that, and — just
# as importantly — that it does not over-fire on a merely short film.


@pytest.mark.asyncio
async def test_grossly_underfull_timeline_fails_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """The production case: content fills ~12s of a 60s target (0.2 of it) ⇒ RAISE."""
    _set_underfull_ratio(monkeypatch, 0.5)
    stage = ReconcileTimelineStage()
    ctx = _ctx(timing={"mode": "auto"}, clips=_clips(), target_seconds=60.0)

    with pytest.raises(RuntimeError) as excinfo:
        await stage.handle(stage.input_model(), ctx)

    message = str(excinfo.value)
    assert "GROSSLY UNDERFULL" in message
    assert "60.00s target" in message
    # The diagnostic is never lost: the soft note rides the raised message.
    assert "timeline UNDERFULL" in message
    # ...and the shortfall is ATTRIBUTABLE. One aggregate ("10.98s of 30.00s")
    # cannot distinguish "too few scenarios ordered" from "every scenario
    # recorded a fraction of what we assumed", and those need opposite fixes.
    # Five runs died on this message in 30 days with nobody able to say which.
    assert "Yield:" in message, "the failure must name where the seconds went"
    assert "pinned beat" in message
    assert "mean=" in message
    # The realised yield is quoted against the per-beat share of the duration that was
    # ORDERED. It used to be quoted against `gen._SECONDS_PER_SCENARIO` (4.0) — a
    # constant unrelated to this run's order, so the comparison could not distinguish
    # "too few beats" from "each beat too short". The constant is gone; the order is
    # the reference.
    assert "s/beat share of the" in message
    assert "ordered duration" in message
    assert "per scenario" not in message
    assert "shortest:" in message


@pytest.mark.asyncio
async def test_rewording_the_underfull_note_does_not_disarm_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate is armed by a STRUCTURED verdict, never by an English substring.

    ``_underfull_failure`` used to arm itself on ``"UNDERFULL" in note`` — a substring of
    a sentence written for a human. A copy edit to ``_underfull_note`` (the cheapest
    change anyone could make to that module) silently disabled the 50%-of-target hard
    floor, and not one test noticed: every other underfull test asserts on the note it
    itself produced, so all of them stayed green with the gate dead.

    Here the note is reworded to prose that does NOT contain the word, on the exact run
    that must fail (content fills ~12s of a 60s target). The raise must still happen, and
    the raised message must still carry the (reworded) diagnostic, proving the gate reads
    ``_underfull_finding`` and merely QUOTES the prose.

    FAILS BEFORE: the substring lookup returned ``None``, ``_underfull_failure`` returned
    early, and the stage shipped the 12s timeline as a degraded ``completed`` run.
    """
    _set_underfull_ratio(monkeypatch, 0.5)
    reworded = "reconcile_timeline: the content came up well short of the ordered length."
    monkeypatch.setattr(
        reconcile_timeline_module,
        "_underfull_note",
        lambda *_args, **_kwargs: reworded,
    )
    stage = ReconcileTimelineStage()
    ctx = _ctx(timing={"mode": "auto"}, clips=_clips(), target_seconds=60.0)

    with pytest.raises(RuntimeError) as excinfo:
        await stage.handle(stage.input_model(), ctx)

    message = str(excinfo.value)
    assert "GROSSLY UNDERFULL" in message
    # The prose is free to change; what may not change is that the gate fired.
    assert "timeline UNDERFULL" in message, "the gate quotes its OWN structured finding"


def test_pad_credit_is_capped_at_a_fraction_of_target_far_below_the_floor() -> None:
    """The invariant, stated directly: pad can never carry a run over the floor.

    With ZERO content the credit is at most ``MAX_PAD_CREDIT_RATIO`` x target, which
    must sit well under the ``demo_timeline_underfull_fail_ratio`` floor — otherwise
    padding alone would be sufficient, which is the whole defect.
    """
    target = 5.0
    floor = 0.5 * target
    assert MAX_PAD_CREDIT_RATIO < 0.5
    # An unbounded pile of pad on no content at all still cannot reach the floor.
    assert content_floor_credit(0.0, 1_000.0, target) == pytest.approx(
        MAX_PAD_CREDIT_RATIO * target
    )
    assert content_floor_credit(0.0, 1_000.0, target) < floor
    # Below the cap the pad is credited in full — the 58s/60s film is not regressed.
    assert content_floor_credit(28.0, 2.0, 60.0) == pytest.approx(30.0)


@pytest.mark.parametrize(
    ("bare", "target"),
    [(1.40, 5.0)],
)
def test_harvest_predictor_and_the_gate_never_disagree(bare: float, target: float) -> None:
    """The predictor's claim must be TRUE, for every pad the gate could ever credit.

    ``record._short_clip_notes`` says "this run cannot satisfy reconcile_timeline". The
    01a059a2 defect was that reconcile_timeline then disagreed with its own predictor —
    two components computing one quantity by two formulas. They now share
    ``content_floor_credit``; this pins the relationship that makes the claim sound:
    when the predictor fires, NO amount of freeze pad lets the gate pass.
    """
    floor = 0.5 * target
    predictor_says_impossible = max_content_floor_credit(bare, target) < floor
    # Sweep every pad the gate could see, including absurdly more than the ceiling.
    gate_passes_somewhere = any(
        content_floor_credit(bare, pad, target) >= floor
        for pad in (0.0, 0.5, 1.5, 3.0, 30.0, 1_000.0)
    )
    assert predictor_says_impossible is not gate_passes_somewhere


@pytest.mark.asyncio
async def test_mildly_underfull_timeline_only_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """~12.3s of a 13s target (0.95) is a pacing miss, not a broken deliverable."""
    _set_underfull_ratio(monkeypatch, 0.5)
    stage = ReconcileTimelineStage()
    ctx = _ctx(timing={"mode": "auto"}, clips=_clips(), target_seconds=13.0)

    out = await stage.handle(stage.input_model(), ctx)

    assert any("UNDERFULL" in note for note in out.notes)  # degraded, NOT failed
    total = sum(float(ln.seconds or 0.0) for ln in out.lines)
    assert 0.5 * 13.0 < total < 13.0


@pytest.mark.asyncio
async def test_gross_underfill_gate_is_disable_able(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kill-switch: ratio 0.0 restores the pre-fix note-only behaviour on the gross case."""
    _set_underfull_ratio(monkeypatch, 0.0)
    stage = ReconcileTimelineStage()
    ctx = _ctx(timing={"mode": "auto"}, clips=_clips(), target_seconds=60.0)

    out = await stage.handle(stage.input_model(), ctx)

    assert any("UNDERFULL" in note for note in out.notes)
    assert sum(float(ln.seconds or 0.0) for ln in out.lines) < 0.5 * 60.0


@pytest.mark.asyncio
async def test_filled_timeline_neither_warns_nor_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Content AT the target: no UNDERFULL note and no failure, gate armed."""
    _set_underfull_ratio(monkeypatch, 0.5)
    stage = ReconcileTimelineStage()
    ctx = _ctx(timing={"mode": "auto"}, clips=_clips(), target_seconds=_FEASIBLE_TARGET)

    out = await stage.handle(stage.input_model(), ctx)

    assert not any("UNDERFULL" in note for note in out.notes)
    assert out.notes == []
    assert sum(float(ln.seconds or 0.0) for ln in out.lines) == pytest.approx(
        _FEASIBLE_TARGET, abs=0.05
    )


# ── (h) unsayable beat ⇒ flagged on notes (the C2 surface) ────────────────────


@pytest.mark.asyncio
async def test_unsayable_beat_is_flagged_on_notes() -> None:
    # A 2s recorded clip carrying 40 words of narration: even at the max delivery rate
    # the audio overruns the shot.
    stage = ReconcileTimelineStage()
    ctx = _ctx(
        timing={"mode": "auto"},
        lines=[{"id": "clip-1", "text": _LONG, "seconds": None, "clip": "clip-1"}],
        clips=_clips(seconds=2.0, line_id="clip-1"),
        target_seconds=2.0,
    )

    out = await stage.handle(stage.input_model(), ctx)

    assert any("UNSAYABLE" in note for note in out.notes)  # ⇒ degraded run
    assert any("clip-1" in note for note in out.notes)
    assert out.lines[0].seconds == pytest.approx(2.0)  # C0: the clip is never trimmed


# ── (h2) underspoken beat ⇒ flagged on notes (unsayable's mirror) ─────────────


def _allotment(id_: str, *, underspoken: bool = False) -> Allotment:
    return Allotment(
        id=id_,
        seconds=10.0,
        vo_rate=0.85,
        source="smart-auto",
        infeasible=False,
        underspoken=underspoken,
    )


@pytest.mark.asyncio
async def test_underspoken_beat_is_flagged_on_stage_notes() -> None:
    """End-to-end: a beat with too few words for its allotment degrades the run.

    Mirrors ``test_unsayable_beat_is_flagged_on_notes`` in the opposite direction —
    a tiny amount of text stretched over a long footage clip needs a delivery rate
    below ``VO_RATE_NATURAL_FLOOR``, so it must land on ``notes`` (⇒ degraded run)
    instead of shipping a silently slow voiceover.
    """
    stage = ReconcileTimelineStage()
    ctx = _ctx(
        timing={"mode": "auto"},
        lines=[{"id": "clip-1", "text": "Hi.", "seconds": None, "clip": "clip-1"}],
        clips=_clips(seconds=30.0, line_id="clip-1"),
        target_seconds=30.0,
    )

    out = await stage.handle(stage.input_model(), ctx)

    assert any("UNDERSPOKEN" in note for note in out.notes)  # ⇒ degraded run
    assert any("clip-1" in note for note in out.notes)
    assert out.lines[0].seconds == pytest.approx(30.0)  # C0: the clip is never trimmed


# ── (i) notes APPEND — an upstream stage's degradation is never clobbered ─────


# ── (j) freeze-pad advisory ⇒ footage narration overruns the clip past the ceiling ──


@pytest.mark.asyncio
async def test_overpad_fires_note() -> None:
    # A 10s recorded clip carrying 40 words (~16s of narration at 150 wpm): the narration
    # overruns the clip by ~6s, far past the 1.5s freeze-pad ceiling. The delivery rate
    # (~1.6) is still in band — this is the honest freeze-pad advisory, not an unsayable.
    stage = ReconcileTimelineStage()
    ctx = _ctx(
        timing={"mode": "auto"},
        lines=[{"id": "clip-1", "text": _LONG, "seconds": None, "clip": "clip-1"}],
        clips=_clips(seconds=10.0, line_id="clip-1"),
        target_seconds=10.0,
    )

    out = await stage.handle(stage.input_model(), ctx)

    assert any("FREEZE-PAD ADVISORY" in note for note in out.notes)  # ⇒ degraded run
    assert any("clip-1" in note for note in out.notes)
    assert out.notes  # truthy ⇒ degraded run
    assert out.lines[0].seconds == pytest.approx(10.0)  # clip budget is never trimmed
    # Step B: the freeze-pad is bounded at the ceiling — max_seconds = clip + ceiling
    # (10 + 1.5) — so the renderer holds the frozen tail at most 1.5s past the clip.
    assert out.lines[0].max_seconds == pytest.approx(10.0 + DEFAULT_FREEZE_PAD_CEILING)
    assert out.lines[0].min_seconds == pytest.approx(10.0)  # floor stays at the bare clip


@pytest.mark.asyncio
async def test_overpad_footage_vo_rate_breathes_into_the_pad() -> None:
    # Same over-clip footage beat as test_overpad_fires_note: a 10s clip carrying 40 words
    # (~16s at 150 wpm). The stamped vo_rate must target clip + freeze-pad ceiling
    # (10 + 1.5 = 11.5s), i.e. 16/11.5 ≈ 1.39 — gentler than compressing into the bare
    # 10s clip (16/10 = 1.6). The advisory STILL fires (the pad cannot close the whole
    # gap) and the clip budget is untouched.
    stage = ReconcileTimelineStage()
    ctx = _ctx(
        timing={"mode": "auto"},
        lines=[{"id": "clip-1", "text": _LONG, "seconds": None, "clip": "clip-1"}],
        clips=_clips(seconds=10.0, line_id="clip-1"),
        target_seconds=10.0,
    )

    out = await stage.handle(stage.input_model(), ctx)

    natural = 40 / DEFAULT_WPM_NATURAL * 60.0  # 16.0s
    padded_rate = natural / (10.0 + DEFAULT_FREEZE_PAD_CEILING)  # 16 / 11.5 ≈ 1.39
    assert out.lines[0].vo_rate == pytest.approx(padded_rate)
    assert (out.lines[0].vo_rate or 0.0) < natural / 10.0  # gentler than the bare-clip target
    assert any("FREEZE-PAD ADVISORY" in note for note in out.notes)  # advisory unchanged
    assert out.lines[0].seconds == pytest.approx(10.0)  # clip budget untouched
    # Step B: the same 1.5s pad the vo_rate breathes into is stamped as the upper bound.
    assert out.lines[0].max_seconds == pytest.approx(10.0 + DEFAULT_FREEZE_PAD_CEILING)


@pytest.mark.asyncio
async def test_within_ceiling_is_silent() -> None:
    # The same 10s footage clip carrying short narration (~3.6s): the narration sits well
    # inside the clip, so there is no gap to pad and no advisory.
    stage = ReconcileTimelineStage()
    ctx = _ctx(
        timing={"mode": "auto"},
        lines=[{"id": "clip-1", "text": _SHORT, "seconds": None, "clip": "clip-1"}],
        clips=_clips(seconds=10.0, line_id="clip-1"),
        target_seconds=10.0,
    )

    out = await stage.handle(stage.input_model(), ctx)

    assert not any("FREEZE-PAD ADVISORY" in note for note in out.notes)
    # Narration fits inside the clip ⇒ no pad ⇒ max_seconds is the bare clip, byte-identical
    # to the pre-Step-B stamping.
    assert out.lines[0].max_seconds == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_partial_pad_within_ceiling_bounds_max_seconds_without_advisory() -> None:
    # A 10s clip carrying 27 words (~10.8s at 150 wpm): the narration overruns the clip by
    # ~0.8s, INSIDE the 1.5s ceiling. The pad absorbs the whole gap, so no advisory fires —
    # but max_seconds is still bounded at clip + the exact 0.8s pad (never trimmed, never
    # stretched to the full audio length).
    stage = ReconcileTimelineStage()
    ctx = _ctx(
        timing={"mode": "auto"},
        lines=[
            {"id": "clip-1", "text": " ".join(["word"] * 27), "seconds": None, "clip": "clip-1"}
        ],
        clips=_clips(seconds=10.0, line_id="clip-1"),
        target_seconds=10.0,
    )

    out = await stage.handle(stage.input_model(), ctx)

    natural = 27 / DEFAULT_WPM_NATURAL * 60.0  # 10.8s
    expected_pad = natural - 10.0  # 0.8s, < ceiling
    assert not any("FREEZE-PAD ADVISORY" in note for note in out.notes)  # gap fully absorbed
    assert out.lines[0].seconds == pytest.approx(10.0)  # clip untouched
    assert out.lines[0].max_seconds == pytest.approx(10.0 + expected_pad)  # bounded at the gap


@pytest.mark.asyncio
async def test_overpad_absent_when_no_timing_block() -> None:
    # Presence-opt-in: with no direction.timing the stage skips BEFORE building the config,
    # so the freeze-pad advisory is fully inert on a run that did not opt in.
    stage = ReconcileTimelineStage()
    ctx = _ctx(
        timing=None,
        lines=[{"id": "clip-1", "text": _LONG, "seconds": None, "clip": "clip-1"}],
        clips=_clips(seconds=10.0, line_id="clip-1"),
        target_seconds=10.0,
    )

    with pytest.raises(StageSkippedError):
        await stage.handle(stage.input_model(), ctx)


# ── C13 relief loop (OPT-IN, INERT-BY-DEFAULT) ────────────────────────────────
#
# ``direction.timing.relief.mode == "auto"`` on an INFEASIBLE timeline arms a bounded
# WM evaluator→optimizer loop (spec 23 §5). ``pacing-critic`` scores rhythm;
# ``timing-director`` is a DIRECTIVE PROPOSER (emits ``{op, target_ids, new_word_target}``,
# never text); ``beat-rewriter`` executes one focused per-beat shortening. All three are an
# ``AsyncMock`` keyed on ``worker_name`` + prompt; the deterministic solver runs FOR REAL.
#
# The stub is FAITHFUL, not canned: ``timing-director`` reads the over-long beats OUT of its
# prompt and proposes one trim per beat, and ``beat-rewriter`` truncates the beat's OWN text
# (embedded in the rewrite prompt) to the requested word target — so a green result means the
# loop genuinely fed the right beats through the right calls, not that a fixture pre-answered.

_RELIEF_TARGET = 8.0


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _parse_optimizer_beats(message: str) -> list[dict[str, Any]]:
    """Pull the over-long ``[{id, text, word_count}, ...]`` block out of the director prompt.

    ``_optimizer_msg`` ends with ``"Over-long beats (id + current text + word_count):\\n\\n
    <json>"`` — parsing it is what makes the stub FAITHFUL: it proposes trims for exactly the
    beats the stage handed it, keyed by the real id, instead of returning a canned list.
    """
    marker = "Over-long beats (id + current text + word_count):"
    _, _, tail = message.partition(marker)
    return list(json.loads(tail.strip()))


def _parse_rewrite_prompt(message: str) -> tuple[str, int]:
    """Pull ``(original_text, word_target)`` back out of a beat-rewriter prompt.

    ``_rewrite_msg`` is ``"...to {target} words or fewer...\\n\\nBeat ({n} words): {text}"``.
    """
    match = re.search(r"to (\d+) words or fewer", message)
    target = int(match.group(1)) if match else 0
    _, _, text = message.partition("words): ")
    return text, target


def _wm_fake(*, pacing_score: int = 5, keep_words: int = 5) -> AsyncMock:
    """Faithful WM stub for the C13 relief loop, keyed on ``worker_name`` + prompt.

    * ``pacing-critic`` → a rhythm dict scoring ``pacing_score``.
    * ``timing-director`` → one ``trim`` directive per over-long beat IN THE PROMPT
      (:func:`_parse_optimizer_beats`), each targeting that beat's own id with
      ``new_word_target = keep_words`` — the directive-proposer contract (no text).
    * ``beat-rewriter`` → the beat's OWN text (embedded in the rewrite prompt) truncated to
      the requested word target — a genuine, prompt-responsive shortening, never canned.
    """
    fake = AsyncMock()

    def _side(worker_name: str | None = None, message: str = "", **_: Any) -> WMResponse:
        if worker_name == "pacing-critic":
            return _resp({"overall_score": pacing_score})
        if worker_name == "beat-rewriter":
            text, target = _parse_rewrite_prompt(message)
            return _resp({"text": " ".join(text.split()[:target])})
        return _resp(
            {
                "directives": [
                    {
                        "op": "trim",
                        "target_ids": [beat["id"]],
                        "new_word_target": keep_words,
                        "rationale": "fit the budget",
                    }
                    for beat in _parse_optimizer_beats(message)
                ]
            }
        )

    fake.message.side_effect = _side
    return fake


def _relief_timing() -> dict[str, Any]:
    return {"mode": "auto", "relief": {"mode": "auto", "iterations": 2, "score_threshold": 4.0}}


_RELIEF_LOG = "src.services.stages.demo_video.reconcile_timeline"


# ── (k) byte-identical when the relief block is absent: no WM traffic at all ───


@pytest.mark.asyncio
async def test_relief_off_is_byte_identical_no_wm() -> None:
    # An INFEASIBLE timeline (the loop WOULD fire if armed) but NO direction.timing.relief
    # block ⇒ relief_mode "off" ⇒ the WM loop is fully inert and the output is exactly the
    # deterministic no-loop path.
    fake = _wm_fake()
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(timing={"mode": "auto"}, lines=_lines(_LONG), clips=_clips(), target_seconds=5.0)

    out = await stage.handle(stage.input_model(), ctx)

    fake.message.assert_not_called()  # zero WM traffic on an opted-out run
    # Text is untouched and the deterministic INFEASIBLE finding still fires (no manifest).
    assert [ln.text for ln in out.lines] == [ln["text"] for ln in _lines(_LONG)]
    assert any("INFEASIBLE" in note for note in out.notes)
    assert not any("timeline_manifest=" in note for note in out.notes)
    assert [ln.id for ln in out.lines] == ["clip-1", "clip-2", "clip-3"]


# ── (l) infeasible → the loop shortens the copy and re-verifies FEASIBLE ──────


@pytest.mark.asyncio
async def test_relief_loop_converges_to_feasible() -> None:
    # Two 40-word beats blow an 8s target; timing-director proposes trim directives, the
    # per-beat beat-rewriter shortens each eligible beat to 5 words, and the deterministic
    # solver re-verifies the result as feasible (pacing-critic then scores it 5).
    fake = _wm_fake()
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(
        timing=_relief_timing(), lines=_lines(_LONG), clips=_clips(), target_seconds=_RELIEF_TARGET
    )

    out = await stage.handle(stage.input_model(), ctx)

    # The loop actually ran end-to-end — propose (director) + execute (rewriter) + re-score.
    assert fake.message.called
    workers = {c.kwargs["worker_name"] for c in fake.message.call_args_list}
    assert workers == {"timing-director", "beat-rewriter", "pacing-critic"}
    # The eligible beats were shortened in place to 5 words, ids/order intact.
    assert len(out.lines[0].text.split()) == 5
    assert len(out.lines[2].text.split()) == 5
    assert out.lines[0].text != _LONG
    assert [ln.id for ln in out.lines] == ["clip-1", "clip-2", "clip-3"]
    # The re-solved timeline is FEASIBLE and the loop stamped a manifest note.
    assert not any("INFEASIBLE" in note for note in out.notes)
    assert any("timeline_manifest=" in note for note in out.notes)


# ── (l2) new_word_target is HONOURED — the rewriter shortens to the proposed count ──


@pytest.mark.asyncio
async def test_relief_honours_new_word_target() -> None:
    # The director proposes new_word_target=3; the beat-rewriter shortens each eligible beat
    # to exactly 3 words — proving the proposed target flows through to the executed rewrite.
    fake = _wm_fake(keep_words=3)
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(
        timing=_relief_timing(), lines=_lines(_LONG), clips=_clips(), target_seconds=_RELIEF_TARGET
    )

    out = await stage.handle(stage.input_model(), ctx)

    assert len(out.lines[0].text.split()) == 3
    assert len(out.lines[2].text.split()) == 3
    assert not any("INFEASIBLE" in note for note in out.notes)


# ── (m) a claim-bearing infeasible beat is NEVER rewritten (soft structural flag) ──


@pytest.mark.asyncio
async def test_relief_claim_bearing_line_not_rewritten() -> None:
    # clip-1 is the ONLY infeasible beat and it is claim-bearing (claim_status truthy) ⇒
    # ineligible to shorten. The loop must leave its text untouched, spend NO WM call, and
    # surface a soft structural-relief hint instead of faking a fit.
    lines: list[dict[str, Any]] = [
        {"id": "clip-1", "text": _LONG, "seconds": None, "clip": None, "claim_status": "softened"},
        {"id": "clip-2", "text": "", "seconds": 5.0, "clip": "clip-2"},
        {"id": "clip-3", "text": "A short line.", "seconds": None, "clip": None},
    ]
    fake = _wm_fake()
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(timing=_relief_timing(), lines=lines, clips=_clips(), target_seconds=15.0)

    out = await stage.handle(stage.input_model(), ctx)

    # The claim-bearing beat's text is byte-identical — never paraphrased.
    assert out.lines[0].text == _LONG
    # C13 phase 2: because the only over-share beat is INELIGIBLE to trim, the structural
    # pre-pass now PROBES the director once for cut/merge/reallocate relief — but a cut or
    # merge-away of a claim-bearing beat is rejected, so the beat is never mutated and no
    # in-place rewrite is ever executed (the beat-rewriter is never reached).
    workers = {c.kwargs["worker_name"] for c in fake.message.call_args_list}
    assert "beat-rewriter" not in workers
    # Honest structural relief is surfaced, not faked.
    assert any("STRUCTURAL RELIEF NEEDED" in note for note in out.notes)
    assert any("INFEASIBLE" in note for note in out.notes)


# ── (n) feasibility-first: a feasible-but-low-rhythm rewrite beats the infeasible original ──


@pytest.mark.asyncio
async def test_relief_feasibility_beats_rhythm() -> None:
    # The rewrite is feasible but the pacing-critic hates its rhythm (score 1 ⇒ 3.4). It
    # must STILL be retained over the infeasible original (capped < 2.0): feasibility wins.
    fake = _wm_fake(pacing_score=1)
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(
        timing=_relief_timing(), lines=_lines(_LONG), clips=_clips(), target_seconds=_RELIEF_TARGET
    )

    out = await stage.handle(stage.input_model(), ctx)

    # The feasible (shorter) draft is kept, not discarded back to the infeasible original.
    assert len(out.lines[0].text.split()) == 5
    assert len(out.lines[2].text.split()) == 5
    assert not any("INFEASIBLE" in note for note in out.notes)
    assert any("timeline_manifest=" in note for note in out.notes)


# ── (o) the pacing advisory NEVER raises, even at the worst rhythm score ───────


@pytest.mark.asyncio
async def test_relief_low_pacing_never_raises() -> None:
    # A rock-bottom rhythm score is advisory only: the loop returns the best draft and the
    # stage neither raises nor blocks the deliverable.
    fake = _wm_fake(pacing_score=1)
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(
        timing=_relief_timing(), lines=_lines(_LONG), clips=_clips(), target_seconds=_RELIEF_TARGET
    )

    # No exception even though the critic scored the rhythm 1/5.
    out = await stage.handle(stage.input_model(), ctx)

    assert [ln.id for ln in out.lines] == ["clip-1", "clip-2", "clip-3"]
    # The loop still recorded its telemetry (pacing_score persisted in the manifest note).
    assert any("timeline_manifest=" in note for note in out.notes)
    assert any('"pacing_score"' in note for note in out.notes)


# ── (p) MIXED eligibility: the core end-to-end relief proof ────────────────────────────
#
# One infeasible timeline that mixes both: two eligible over-long beats ARE shortened while a
# claim / footage / locked beat in the same spine is left byte-identical. The faithful stub
# proposes a trim per eligible id and truncates each beat's OWN text, so a green result means
# the loop fed the right beats through director → rewriter and merged back by id.

# Two DISTINCT 40-word beats (recognisable first word) so the per-id prefix is visibly the
# beat's OWN text, not a shared fixture constant.
_LONG_A = "Alpha " + " ".join(["aardvark"] * 39)  # 40 words, starts "Alpha"
_LONG_B = "Bravo " + " ".join(["buffalo"] * 39)  # 40 words, starts "Bravo"
_THREE_CLAIM = "Grounded factual claim."  # 3 words, claim-bearing ⇒ ineligible
_THREE_LOCKED = "Author locked this."  # 3 words, author-locked ⇒ ineligible
_KEEP_WORDS = 5  # the faithful stub truncates each eligible beat to its first 5 words
_MIXED_TARGET = 10.0


@pytest.mark.asyncio
async def test_relief_loop_mixed_eligibility_persists_manifest() -> None:
    # One infeasible timeline with BOTH eligible and ineligible over-share beats:
    #   clip-1, clip-2 — 40-word non-claim, unlocked VO (each floors ~14.1s) → ELIGIBLE
    #   clip-3         — 3-word claim-bearing VO                              → ineligible
    #   clip-4         — footage (measured 4s, no narration), pinned         → ineligible
    #   clip-5         — 3-word author-locked VO                             → ineligible
    # At target 10s (fair share 2.0s) infeasible_beats flags ONLY clip-1/clip-2 (the two
    # >fair-share beats); shortening just those two to 5 words (Σmin ≈ 9.65s ≤ 10 ≤ Σmax
    # ≈ 11.4s) reaches a feasible AND fillable timeline — so the loop converges with no
    # INFEASIBLE and no STRUCTURAL-RELIEF note, and the ineligible beats stay byte-identical.
    lines: list[dict[str, Any]] = [
        {"id": "clip-1", "text": _LONG_A, "seconds": None, "clip": None},
        {"id": "clip-2", "text": _LONG_B, "seconds": None, "clip": None},
        {
            "id": "clip-3",
            "text": _THREE_CLAIM,
            "seconds": None,
            "clip": None,
            "claim_status": "grounded",
        },
        {"id": "clip-4", "text": "", "seconds": 5.0, "clip": "clip-4"},
        {"id": "clip-5", "text": _THREE_LOCKED, "seconds": None, "clip": None, "locked": True},
    ]
    fake = _wm_fake(keep_words=_KEEP_WORDS)
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(
        timing=_relief_timing(),
        lines=lines,
        clips=_clips(seconds=4.0, line_id="clip-4"),
        target_seconds=_MIXED_TARGET,
    )

    out = await stage.handle(stage.input_model(), ctx)

    # All three workers were exercised — the loop ran end-to-end (propose + execute + re-score).
    workers = {c.kwargs["worker_name"] for c in fake.message.call_args_list}
    assert workers == {"timing-director", "beat-rewriter", "pacing-critic"}

    by_id = {ln.id: ln for ln in out.lines}
    # (a) FAITHFUL: each eligible beat was shortened to ITS OWN 5-word prefix — the director
    #     proposed a trim per id and the rewriter truncated that beat's own text (not canned).
    assert by_id["clip-1"].text == " ".join(_LONG_A.split()[:_KEEP_WORDS])
    assert by_id["clip-2"].text == " ".join(_LONG_B.split()[:_KEEP_WORDS])
    assert by_id["clip-1"].text.startswith("Alpha")
    assert by_id["clip-2"].text.startswith("Bravo")
    assert len(by_id["clip-1"].text.split()) == _KEEP_WORDS

    # (b) id-preserving, order intact (C0).
    assert [ln.id for ln in out.lines] == ["clip-1", "clip-2", "clip-3", "clip-4", "clip-5"]

    # (c) the ineligible beats are byte-identical — claim / footage / locked are protected.
    assert by_id["clip-3"].text == _THREE_CLAIM
    assert by_id["clip-4"].text == ""
    assert by_id["clip-4"].clip == "clip-4"
    assert by_id["clip-4"].seconds == pytest.approx(4.0)  # pinned to the MEASURED clip
    assert by_id["clip-5"].text == _THREE_LOCKED

    # (d) the re-solved timeline is FEASIBLE — shortening the eligible beats alone fixed it,
    #     with no dishonest structural fake.
    assert not any("INFEASIBLE" in note for note in out.notes)
    assert not any("STRUCTURAL RELIEF" in note for note in out.notes)

    # (e) the manifest + convergence telemetry (pacing_score) are persisted on the output.
    assert any("timeline_manifest=" in note for note in out.notes)
    assert any('"pacing_score"' in note for note in out.notes)


# ── (q)-(v) diagnosability: a live silent no-op is now LOGGED, not swallowed ───────────
#
# Each of these rebuts an adversarial "the loop silently does nothing" hypothesis by proving
# the added WARNING actually fires — so a live relief run that fails to shorten anything tells
# an operator EXACTLY why (empty directives / structural op / dropped id / off-contract output
# / not-shorter rewrite / no eligible beat) instead of looking like it just chose not to help.


@pytest.mark.asyncio
async def test_relief_loop_failure_is_logged_at_error_with_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # S810: the relief loop is TOLERABLE to lose — the deliverable is the deterministic
    # allotment vector, which is already computed and untouched, and the loss is surfaced on
    # notes (⇒ degraded run). But the swallow must not be silent: the exception is logged at
    # ERROR with its traceback, so a permanently-broken WM is not mistaken for a run that
    # simply had nothing to relieve.
    fake = AsyncMock()
    fake.message.side_effect = RuntimeError("worker_management down")
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(
        timing=_relief_timing(), lines=_lines(_LONG), clips=_clips(), target_seconds=_RELIEF_TARGET
    )

    with caplog.at_level(logging.ERROR, logger=_RELIEF_LOG):
        out = await stage.handle(stage.input_model(), ctx)

    # Degrade semantics preserved: no raise, deterministic timeline shipped + flagged.
    assert [ln.text for ln in out.lines] == [ln["text"] for ln in _lines(_LONG)]
    assert any("relief loop UNAVAILABLE" in note for note in out.notes)
    assert any("INFEASIBLE" in note for note in out.notes)
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert errors, "a swallowed relief failure must be logged at ERROR, not WARNING"
    assert any(r.exc_info is not None for r in errors)  # the traceback is not thrown away


@pytest.mark.asyncio
async def test_relief_empty_directives_is_warned(caplog: pytest.LogCaptureFixture) -> None:
    # timing-director proposes an EMPTY directive set — the EXACT prod symptom this refactor
    # targets. Nothing is shortened, the rewriter is never reached, the timeline stays
    # infeasible, and the empty proposal is logged instead of a silent no-op.
    fake = AsyncMock()

    def _side(worker_name: str | None = None, message: str = "", **_: Any) -> WMResponse:
        if worker_name == "pacing-critic":
            return _resp({"overall_score": 5})
        return _resp({"directives": []})

    fake.message.side_effect = _side
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(
        timing=_relief_timing(), lines=_lines(_LONG), clips=_clips(), target_seconds=_RELIEF_TARGET
    )

    with caplog.at_level(logging.WARNING, logger=_RELIEF_LOG):
        out = await stage.handle(stage.input_model(), ctx)

    assert [ln.text for ln in out.lines] == [ln["text"] for ln in _lines(_LONG)]  # untouched
    assert "EMPTY directive set" in caplog.text
    assert any("INFEASIBLE" in note for note in out.notes)
    workers = {c.kwargs["worker_name"] for c in fake.message.call_args_list}
    assert "beat-rewriter" not in workers  # never executed a rewrite


@pytest.mark.asyncio
async def test_relief_structural_op_is_rejected_and_warned(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # timing-director proposes a MERGE — a structural op not yet disposed (it would change the
    # id-set and trip the C0 guard). It is rejected with a log, no beat is rewritten, and the
    # timeline stays infeasible.
    fake = AsyncMock()

    def _side(worker_name: str | None = None, message: str = "", **_: Any) -> WMResponse:
        if worker_name == "pacing-critic":
            return _resp({"overall_score": 5})
        return _resp({"directives": [{"op": "merge", "target_ids": ["clip-1", "clip-3"]}]})

    fake.message.side_effect = _side
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(
        timing=_relief_timing(), lines=_lines(_LONG), clips=_clips(), target_seconds=_RELIEF_TARGET
    )

    with caplog.at_level(logging.WARNING, logger=_RELIEF_LOG):
        out = await stage.handle(stage.input_model(), ctx)

    assert [ln.text for ln in out.lines] == [ln["text"] for ln in _lines(_LONG)]  # untouched
    assert "structural op" in caplog.text
    assert any("INFEASIBLE" in note for note in out.notes)
    workers = {c.kwargs["worker_name"] for c in fake.message.call_args_list}
    assert "beat-rewriter" not in workers


@pytest.mark.asyncio
async def test_relief_not_shorter_rewrite_is_dropped_and_warned(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # timing-director proposes valid trims, but beat-rewriter returns the ORIGINAL text (no
    # shorter): each rewrite is DROPPED (never fabricated), the zero net reduction is logged,
    # and the timeline stays infeasible instead of faking a fit.
    fake = AsyncMock()

    def _side(worker_name: str | None = None, message: str = "", **_: Any) -> WMResponse:
        if worker_name == "pacing-critic":
            return _resp({"overall_score": 5})
        if worker_name == "beat-rewriter":
            text, _target = _parse_rewrite_prompt(message)
            return _resp({"text": text})  # identical ⇒ not shorter ⇒ dropped
        return _resp(
            {
                "directives": [
                    {"op": "trim", "target_ids": [b["id"]], "new_word_target": 5}
                    for b in _parse_optimizer_beats(message)
                ]
            }
        )

    fake.message.side_effect = _side
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(
        timing=_relief_timing(), lines=_lines(_LONG), clips=_clips(), target_seconds=_RELIEF_TARGET
    )

    with caplog.at_level(logging.WARNING, logger=_RELIEF_LOG):
        out = await stage.handle(stage.input_model(), ctx)

    assert [ln.text for ln in out.lines] == [ln["text"] for ln in _lines(_LONG)]  # untouched
    assert "not-shorter" in caplog.text or "zero net word reduction" in caplog.text
    assert any("INFEASIBLE" in note for note in out.notes)


@pytest.mark.asyncio
async def test_relief_dropped_echoed_id_is_ignored_and_warned(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # timing-director trims WRONG-CASED ids ("CLIP-1" not "clip-1"): the spine is never grown,
    # neither beat is rewritten (text byte-identical), the timeline stays INFEASIBLE, and the
    # drop is logged so the no-op is diagnosable rather than silent.
    fake = AsyncMock()

    def _side(worker_name: str | None = None, message: str = "", **_: Any) -> WMResponse:
        if worker_name == "pacing-critic":
            return _resp({"overall_score": 5})
        return _resp(
            {
                "directives": [
                    {"op": "trim", "target_ids": ["CLIP-1"], "new_word_target": 5},
                    {"op": "trim", "target_ids": ["CLIP-3"], "new_word_target": 5},
                ]
            }
        )

    fake.message.side_effect = _side
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(
        timing=_relief_timing(), lines=_lines(_LONG), clips=_clips(), target_seconds=_RELIEF_TARGET
    )

    with caplog.at_level(logging.WARNING, logger=_RELIEF_LOG):
        out = await stage.handle(stage.input_model(), ctx)

    assert [ln.text for ln in out.lines] == [ln["text"] for ln in _lines(_LONG)]  # untouched
    assert [ln.id for ln in out.lines] == ["clip-1", "clip-2", "clip-3"]
    assert "outside the eligible set" in caplog.text
    assert any("INFEASIBLE" in note for note in out.notes)  # never faked feasible
    workers = {c.kwargs["worker_name"] for c in fake.message.call_args_list}
    assert "beat-rewriter" not in workers


@pytest.mark.asyncio
async def test_relief_non_dict_optimizer_output_is_warned(caplog: pytest.LogCaptureFixture) -> None:
    # timing-director returns an OFF-CONTRACT (non-dict) output: zero directives, timeline stays
    # infeasible, and the contract breach is logged instead of scoring a silent no-op.
    fake = AsyncMock()

    def _side(worker_name: str | None = None, message: str = "", **_: Any) -> WMResponse:
        if worker_name == "pacing-critic":
            return _resp({"overall_score": 5})
        return _resp("not-a-dict")

    fake.message.side_effect = _side
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(
        timing=_relief_timing(), lines=_lines(_LONG), clips=_clips(), target_seconds=_RELIEF_TARGET
    )

    with caplog.at_level(logging.WARNING, logger=_RELIEF_LOG):
        out = await stage.handle(stage.input_model(), ctx)

    assert [ln.text for ln in out.lines] == [ln["text"] for ln in _lines(_LONG)]  # unchanged
    assert "non-dict" in caplog.text
    assert any("INFEASIBLE" in note for note in out.notes)


@pytest.mark.asyncio
async def test_relief_no_eligible_beat_is_warned(caplog: pytest.LogCaptureFixture) -> None:
    # The ONLY infeasible beat is claim-bearing ⇒ nothing eligible to shorten. The loop no-ops
    # into the soft structural flag AND logs which flagged beats were skipped — the single most
    # likely live outcome (an over-stuffed brief whose long beats are all claims) is now visible.
    lines: list[dict[str, Any]] = [
        {"id": "clip-1", "text": _LONG, "seconds": None, "clip": None, "claim_status": "softened"},
        {"id": "clip-2", "text": "", "seconds": 5.0, "clip": "clip-2"},
        {"id": "clip-3", "text": "A short line.", "seconds": None, "clip": None},
    ]
    fake = _wm_fake()
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(timing=_relief_timing(), lines=lines, clips=_clips(), target_seconds=15.0)

    with caplog.at_level(logging.WARNING, logger=_RELIEF_LOG):
        out = await stage.handle(stage.input_model(), ctx)

    assert out.lines[0].text == _LONG  # claim beat never rewritten
    assert "no beat is" in caplog.text and "eligible to shorten" in caplog.text
    assert any("STRUCTURAL RELIEF NEEDED" in note for note in out.notes)


# ── (q) SOFT gate: a PERMANENT WM/LLM relief failure degrades, never fails the run ──
#
# The relief loop makes UNGUARDED worker_management calls. A permanent upstream failure
# (402 byo-not-configured / token-limit ⇒ PermanentUpstreamError, or retries-exhausted)
# must NOT raise out of handle() and mark the run FAILED — the VALID deterministic
# allotments computed before the relief call are the safe fallback and must be persisted,
# exactly as footage_critique / final_critique / encode_master accept-current on any loop
# error.


@pytest.mark.asyncio
async def test_relief_permanent_failure_degrades_to_deterministic(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An infeasible timeline arms the loop; the relief pass then raises PermanentUpstreamError
    # (the 402/token-limit shape). handle() must catch it, keep the deterministic allotments,
    # and DEGRADE (note ⇒ degraded run) instead of hard-failing the whole render.
    stage = ReconcileTimelineStage(client=_wm_fake())
    # Make the relief loop raise the permanent-upstream error (the WM-call failure it wraps).
    monkeypatch.setattr(
        stage,
        "_relief_loop",
        AsyncMock(side_effect=PermanentUpstreamError("byo not configured (402)")),
    )
    ctx = _ctx(
        timing=_relief_timing(), lines=_lines(_LONG), clips=_clips(), target_seconds=_RELIEF_TARGET
    )

    with caplog.at_level(logging.WARNING, logger=_RELIEF_LOG):
        out = await stage.handle(stage.input_model(), ctx)  # does NOT raise

    # The deterministic allotments survived: text untouched, ids/order intact, and the
    # deterministic INFEASIBLE finding still fires (the timeline was NOT LLM-relieved).
    assert [ln.text for ln in out.lines] == [ln["text"] for ln in _lines(_LONG)]
    assert [ln.id for ln in out.lines] == ["clip-1", "clip-2", "clip-3"]
    assert out.lines[0].seconds is not None  # allotment was stamped from the solver
    assert any("INFEASIBLE" in note for note in out.notes)
    # Degraded, not failed: the soft-gate note is present and no manifest was produced.
    assert any("relief loop UNAVAILABLE" in note for note in out.notes)
    assert not any("timeline_manifest=" in note for note in out.notes)
    assert "relief loop failed" in caplog.text


# ── (u) UNSAYABLE on an UNDERFULL timeline arms the trim loop ─────────────────
#
# THE previously-broken case. Measured on run 01a021c6-01ba-70dc-a9cb-0aa194de7eb1
# (2026-08-21): 15 clips filling 51.92s of a 60s target — UNDERFULL, so
# ``infeasible_beats`` was empty — while 11 of 15 beats were UNSAYABLE (narration
# overruns its own shot even at the 2.0 rate cap). The C13 shortening machinery
# existed and was armed ONLY on infeasibility, so it never fired on exactly the runs
# that needed it. Footage beats are pinned (min==ideal==max==measured) and the
# freeze-pad is ceiling-capped on purpose, so the narration is the only free variable.

# 30 words on a 2.0s clip. The stage under test carries NO freeze-pad override, so it
# uses the module default 1.5s: the beat's window is 2.0 + 1.5 = 3.5s, whose sayable
# ceiling at 150 wpm x the 2.0 band top is 3.5 * 150 * 2.0 / 60 = 17 words. 30 > 17 ⇒
# UNSAYABLE. Computed from the fixture's own inputs, not read back from the
# implementation, so this stays an independent check rather than a tautology.
# NOTE: the precondition below must build its config with the SAME 1.5s ceiling the stage
# uses — _timing_config_for defaults to 2.0 for the other tests in this file.
_UNSAYABLE_TEXT = " ".join(["word"] * 30)
_SAYABLE_CEILING_WORDS = 17


def _footage_lines(text: str = _UNSAYABLE_TEXT, **extra: Any) -> list[dict[str, Any]]:
    """Three footage-pinned beats, each 2.0s of measured clip carrying ``text``."""
    return [
        {"id": f"clip-{i}", "text": text, "seconds": 2.0, "clip": f"clip-{i}", **extra}
        for i in (1, 2, 3)
    ]


def _footage_clips(seconds: float = 2.0) -> list[dict[str, Any]]:
    return [
        {
            "id": f"clip-{i}",
            "line_id": f"clip-{i}",
            "video_url": f"https://minio.test/clip-{i}.mp4",
            "seconds": seconds,
        }
        for i in (1, 2, 3)
    ]


# Sigma-max = 6.0s against a 7.0s target ⇒ UNDERFULL (so infeasible_beats is EMPTY),
# but only mildly so — well above the gross-underfill hard gate.
_UNSAYABLE_TARGET = 7.0


def _unsayable_cfg() -> _TimingConfig:
    """The config the STAGE resolves for these fixtures (default freeze-pad ceiling)."""
    return replace(
        _timing_config_for(target=_UNSAYABLE_TARGET),
        freeze_pad_ceiling_seconds=DEFAULT_FREEZE_PAD_CEILING,
    )


@pytest.mark.asyncio
async def test_unsayable_relief_off_stays_inert() -> None:
    # OPT-IN is unchanged: without the relief block an unsayable timeline is still
    # note-only and spends ZERO WM calls.
    fake = _wm_fake()
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(
        timing={"mode": "auto"},
        lines=_footage_lines(),
        clips=_footage_clips(),
        target_seconds=_UNSAYABLE_TARGET,
    )

    out = await stage.handle(stage.input_model(), ctx)

    fake.message.assert_not_called()
    assert [len(ln.text.split()) for ln in out.lines] == [30, 30, 30]
    assert any("UNSAYABLE" in note for note in out.notes)


# ══ THE SOLVER'S FREEZE PAD REACHES THE LINE, AND THEREFORE THE FIT PASS ══════
#
# `solve_timeline` chooses a FOOTAGE beat's `vo_rate` against
# `allotted + freeze_pad_seconds`. That pad lived only inside the solver, so
# `demo_video_narrate.fit_target_for` re-fit the synthesized line to `seconds` ALONE and
# re-synthesized it up to the whole 1.5s ceiling SLOWER than the solver intended. It
# cannot be recovered downstream from `max_seconds` (that is `demand.max + pad`, and
# `demand.max` is not on the line), so the stage stamps it.


@pytest.mark.asyncio
async def test_the_freeze_pad_is_stamped_on_the_footage_beat_it_was_granted_to() -> None:
    # One footage beat pinned to a measured 4.0s clip carrying ~40 words of narration:
    # the words need ~16s at 150 wpm, so the solver grants the pad at its ceiling.
    lines = [{"id": "clip-1", "text": _LONG, "seconds": 4.0, "clip": "clip-1"}]
    stage = ReconcileTimelineStage()
    out = await stage.handle(
        stage.input_model(),
        _ctx(
            timing={"mode": "auto", "freeze_pad_ceiling_seconds": DEFAULT_FREEZE_PAD_CEILING},
            lines=lines,
            clips=_clips(seconds=4.0, line_id="clip-1"),
            # 8.0s: above the 50% underfull floor for a 4.0s clip (which would otherwise
            # hard-fail before anything is stamped), and well under the ~16s the 40 words
            # need — so the pad is granted at its ceiling, which is the case under test.
            target_seconds=8.0,
        ),
    )

    beat = out.lines[0]
    assert beat.seconds == pytest.approx(4.0)  # the clip is never stretched
    assert beat.freeze_pad_seconds == pytest.approx(DEFAULT_FREEZE_PAD_CEILING)

    # THE JOIN, asserted end-to-end: the narrate stage reads this very line and must fit
    # to the window the rate was chosen against, not to the bare clip. Without the stamp
    # the pad is unrecoverable here and the line is re-synthesized 1.5s too slow.
    from src.services.stages.demo_video.narrate import DemoVideoNarrateStage

    fit = DemoVideoNarrateStage().fit_target_for(beat.model_dump(), _ctx(timing=None))
    assert fit == pytest.approx(4.0 + DEFAULT_FREEZE_PAD_CEILING)


@pytest.mark.asyncio
async def test_an_unpadded_beat_is_stamped_zero_so_the_fit_target_is_unchanged() -> None:
    """Control. Only a footage beat whose narration OVERRUNS its clip gets a pad; every
    other beat must be stamped 0.0, leaving the fit target byte-identical to before."""
    stage = ReconcileTimelineStage()
    out = await stage.handle(
        stage.input_model(),
        _ctx(timing={"mode": "auto"}, clips=_clips(), target_seconds=_FEASIBLE_TARGET),
    )

    by_id = {line.id: line for line in out.lines}
    # clip-1 / clip-3 are VO-only (elastic, never padded); clip-2's footage carries no
    # narration at all, so there is nothing to overrun it.
    for lid in ("clip-1", "clip-2", "clip-3"):
        assert by_id[lid].freeze_pad_seconds == pytest.approx(0.0), lid


# ── (z) every DEGRADING note carries a CODE — never the executor's unclassified_note ──
#
# `degradations` used to be optional here and the degrading notes merged without a code,
# so the DAG executor stamped `unclassified_note` ("degraded via prose notes only",
# runs/dag_executor.py `_enforce_degradation_codes`) on every degraded reconcile_timeline
# output. The run record then could not say WHICH timeline fault shipped.


def _codes_of(out: Any) -> set[str]:
    return {d["code"] for d in out.model_dump(mode="json")["degradations"]}


def _executor_would_stamp_unclassified(out: Any) -> bool:
    from src.services.stages.shared.degradation import ensure_degradation_codes

    return ensure_degradation_codes("reconcile_timeline", out.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_underfull_timeline_carries_coded_degradations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services.stages.shared.degradation import (
        TIMELINE_OFF_TARGET,
        TIMELINE_UNDERFULL,
        UNCLASSIFIED_NOTE,
    )

    _set_underfull_ratio(monkeypatch, 0.0)
    stage = ReconcileTimelineStage()
    out = await stage.handle(
        stage.input_model(), _ctx(timing={"mode": "auto"}, clips=_clips(), target_seconds=30.0)
    )

    assert any("UNDERFULL" in note for note in out.notes)
    codes = _codes_of(out)
    assert {TIMELINE_UNDERFULL, TIMELINE_OFF_TARGET} <= codes
    assert UNCLASSIFIED_NOTE not in codes
    assert not _executor_would_stamp_unclassified(out)
    # The detail is the human sentence, so the coded entry is self-explanatory.
    underfull = next(
        d for d in out.model_dump(mode="json")["degradations"] if d["code"] == TIMELINE_UNDERFULL
    )
    assert "UNDERFULL" in underfull["detail"]


@pytest.mark.asyncio
async def test_unsayable_beat_carries_a_coded_degradation() -> None:
    from src.services.stages.shared.degradation import NARRATION_UNSAYABLE

    stage = ReconcileTimelineStage()
    out = await stage.handle(
        stage.input_model(),
        _ctx(
            timing={"mode": "auto"},
            lines=[{"id": "clip-1", "text": _LONG, "seconds": None, "clip": "clip-1"}],
            clips=_clips(seconds=2.0, line_id="clip-1"),
            target_seconds=2.0,
        ),
    )

    assert NARRATION_UNSAYABLE in _codes_of(out)
    assert not _executor_would_stamp_unclassified(out)


@pytest.mark.asyncio
async def test_relief_failure_carries_a_coded_degradation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services.stages.shared.degradation import (
        ORDERED_DURATION_INFEASIBLE,
        TIMELINE_RELIEF_UNAVAILABLE,
    )

    stage = ReconcileTimelineStage(client=_wm_fake())
    monkeypatch.setattr(
        stage,
        "_relief_loop",
        AsyncMock(side_effect=PermanentUpstreamError("byo not configured (402)")),
    )
    out = await stage.handle(
        stage.input_model(),
        _ctx(
            timing=_relief_timing(),
            lines=_lines(_LONG),
            clips=_clips(),
            target_seconds=_RELIEF_TARGET,
        ),
    )

    codes = _codes_of(out)
    assert TIMELINE_RELIEF_UNAVAILABLE in codes
    assert ORDERED_DURATION_INFEASIBLE in codes
    # INFEASIBLE is coded exactly ONCE (its note's code is the existing degradation).
    dumped = out.model_dump(mode="json")["degradations"]
    assert sum(d["code"] == ORDERED_DURATION_INFEASIBLE for d in dumped) == 1
    assert not _executor_would_stamp_unclassified(out)
