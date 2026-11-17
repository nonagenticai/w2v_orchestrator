"""reconcile_timeline -> demo_video_compile: the footage out-point contract, end to end.

reconcile writes each beat's ``footage_out_point_seconds`` onto
``timeline_manifest.beats[]`` (not onto ``lines[]``); compile must cut the clip there.
Both halves were unit-tested against their OWN idea of the shape and disagreed — compile
read the key off ``lines[]``, where reconcile never puts it, so no cut ever reached the
film. This drives the REAL reconcile stage and hands its serialized output, unmodified,
to the REAL compile planner. No mock between the two.

P5-shaped: 3 footage beats, footage 12.0 + 10.36 + 10.0 = 32.36s, narration
8.0 + 12.8 + 12.02 = 32.82s, 30s order.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest

from src.services.llm import WMClient
from src.services.runs.structural_relief import maybe_spawn_structural_relief_run
from src.services.runs.structural_relief_policy import decide_structural_relief
from src.services.stages.demo_video.compile import DemoVideoCompileStage
from src.services.stages.demo_video.reconcile_timeline import ReconcileTimelineStage
from tests.services.stages.demo_video.test_compile_order_fit import _ctx as _compile_ctx
from tests.services.stages.demo_video.test_compile_trimmed_clip_window import (
    _compiler_shown_span,
)
from tests.services.stages.demo_video.test_reconcile_footage_trim_and_cards import (
    _ctx as _reconcile_ctx,
)
from tests.services.stages.demo_video.test_reconcile_footage_trim_and_cards import (
    _ExpanderWM,
    _words,
)

FRAME = 1 / 30
_CLIPS = {"clip-1": 12.0, "clip-2": 10.36, "clip-3": 10.0}
_WORDS = {"clip-1": 20, "clip-2": 32, "clip-3": 30}
_NARRATION = {"clip-1": 8.0, "clip-2": 12.8, "clip-3": 12.02}


@pytest.mark.asyncio
async def test_compile_cuts_each_clip_at_reconciles_out_point() -> None:
    assert sum(_CLIPS.values()) == pytest.approx(32.36)
    assert sum(_NARRATION.values()) == pytest.approx(32.82)
    lines = [
        {"id": cid, "text": _words(n), "seconds": None, "clip": cid} for cid, n in _WORDS.items()
    ]
    rctx = _reconcile_ctx(lines=lines, clips=_CLIPS, order=30.0)
    stage = ReconcileTimelineStage()
    reconciled = (await stage.handle(stage.input_model(), rctx)).model_dump(mode="json")

    manifest = reconciled["timeline_manifest"]
    assert manifest is not None, "a trimmed run must carry the manifest the compile reads"
    # …but that manifest is an out-points carrier, NOT a relief run's record: relief is
    # off here, so it must never classify into a paid-spawn candidate.
    assert manifest["run"]["convergence_outcome"] == "relief_off"
    assert manifest["run"]["converged"] is False
    points = {b["id"]: b["footage_out_point_seconds"] for b in manifest["beats"]}
    assert points["clip-1"] is not None and points["clip-1"] < _CLIPS["clip-1"]
    assert all("footage_out_point_seconds" not in line for line in reconciled["lines"])

    record = copy.deepcopy(rctx.prior_run_outputs["demo_video_record"])
    outputs = {
        "reconcile_timeline": reconciled,
        "demo_video_gen": rctx.prior_run_outputs["demo_video_gen"],
        "demo_video_narrate": {
            "clips": [
                {"line_id": cid, "artifact_url": f"http://a/{cid}.mp3", "seconds": s}
                for cid, s in _NARRATION.items()
            ]
        },
        "demo_video_record": record,
    }
    compile_stage = DemoVideoCompileStage.__new__(DemoVideoCompileStage)
    shots = {
        s["line_id"]: s
        for s in compile_stage.build_shots(_compile_ctx(rctx.config_snapshot, outputs))
    }

    for cid, clip_seconds in _CLIPS.items():
        _start, end = _compiler_shown_span(shots[cid], clip_seconds)
        if points[cid] is None:
            assert shots[cid].get("clip_out_seconds") is None
            assert end == pytest.approx(clip_seconds, abs=FRAME)
        else:
            assert shots[cid]["clip_out_seconds"] == pytest.approx(points[cid], abs=FRAME)
            assert end <= points[cid] + FRAME


# ── PR #1078 audit, defect 1: a relief-OFF trimmed run must never buy a PAID child ─────
#
# 6 footage beats x 10s at a 30s order, relief NOT armed. Tail-trimming cannot reach the
# order, so the out-points-only manifest used to classify ``cant_converge`` with a 6.0s
# overrun — and the spawner, which read "manifest present" as "relief ran", spawned. The
# global kill-switch is ON in prod, so nothing else stood in the way.


async def _relief_off_overrun_run() -> tuple[dict[str, Any], dict[str, Any]]:
    clips = {f"clip-{i}": 10.0 for i in range(1, 7)}
    lines = [{"id": c, "text": _words(8), "seconds": None, "clip": c} for c in clips]
    rctx = _reconcile_ctx(lines=lines, clips=clips, order=30.0)  # relief NOT armed
    stage = ReconcileTimelineStage()
    out = (await stage.handle(stage.input_model(), rctx)).model_dump(mode="json")
    return out, rctx.config_snapshot


@pytest.mark.asyncio
async def test_relief_off_trimmed_run_gets_no_spawn_from_the_policy() -> None:
    out, _snapshot = await _relief_off_overrun_run()
    run = out["timeline_manifest"]["run"]
    assert run["loop_iterations"] == 0
    decision = decide_structural_relief(
        convergence_outcome=run["convergence_outcome"],
        structural_relief_requested=run["structural_relief_requested"],
        overrun_seconds=run["overrun_seconds"],
        prior_overrun_seconds=None,
        pass_number=0,
        max_passes=2,
    )
    assert decision != "SPAWN"
    assert decision == "STOP_RELIEF_OFF"


@pytest.mark.asyncio
async def test_relief_off_trimmed_run_gets_no_spawn_from_the_spawner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, snapshot = await _relief_off_overrun_run()
    # Prod settings: kill-switch ON, underfull clamp off.
    monkeypatch.setattr(
        "src.services.runs.structural_relief.get_settings",
        lambda: SimpleNamespace(
            pipeline=SimpleNamespace(
                structural_relief_autoconverge_enabled=True,
                structural_relief_max_passes=2,
                demo_timeline_underfull_fail_ratio=0.0,
            )
        ),
    )
    start_run = AsyncMock(return_value=SimpleNamespace(id=uuid7()))
    engine: Any = SimpleNamespace(start_run=start_run)
    rows: list[Any] = [SimpleNamespace(stage_name="reconcile_timeline", output=out)]
    child = await maybe_spawn_structural_relief_run(
        run_engine=engine,
        parent_run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot={**snapshot, "tenant_id": str(uuid7()), "subject_id": str(uuid7())},
        stage_rows=rows,
    )
    assert child is None
    start_run.assert_not_awaited()


# ── PR #1078 audit, defect 2: the out-point is FILE time; authored windows are kept ─────
#
# A beat authored to show 10-15s of a 20s recording. A 10s order makes the tail-trim want
# that clip at its 60% floor, 12.0 — a DURATION from file time 0. Compile read it as the
# file-time usable end, so the authored window collapsed to 10-12s: 2s of footage in the
# beat's slot, the rest freeze-hold.


@pytest.mark.asyncio
async def test_authored_clip_window_is_honoured_not_cut_from_file_time_zero() -> None:
    clips = {"clip-1": 20.0}
    lines = [
        {
            "id": "clip-1",
            "text": _words(5),
            "seconds": None,
            "clip": "clip-1",
            "clip_anchor": "start",
            "clip_lead_seconds": 10.0,
            "clip_out_seconds": 15.0,
        }
    ]
    rctx = _reconcile_ctx(lines=lines, clips=clips, order=10.0)
    stage = ReconcileTimelineStage()
    reconciled = (await stage.handle(stage.input_model(), rctx)).model_dump(mode="json")
    manifest = reconciled["timeline_manifest"]
    if manifest is not None:
        points = {b["id"]: b["footage_out_point_seconds"] for b in manifest["beats"]}
        assert points.get("clip-1") is None, "an authored window is never tail-trimmed"

    outputs = {
        "reconcile_timeline": reconciled,
        "demo_video_gen": rctx.prior_run_outputs["demo_video_gen"],
        "demo_video_narrate": {
            "clips": [{"line_id": "clip-1", "artifact_url": "http://a/c1.mp3", "seconds": 2.0}]
        },
        "demo_video_record": copy.deepcopy(rctx.prior_run_outputs["demo_video_record"]),
    }
    compile_stage = DemoVideoCompileStage.__new__(DemoVideoCompileStage)
    shot = compile_stage.build_shots(_compile_ctx(rctx.config_snapshot, outputs))[0]
    assert shot["clip_lead_seconds"] == pytest.approx(10.0)
    assert shot["clip_out_seconds"] == pytest.approx(15.0)


def test_compile_ignores_a_stale_out_point_on_an_authored_window() -> None:
    """Defence in depth: a manifest written before the fix (or by any other writer)
    carrying a duration-from-zero cut on a windowed beat must not collapse the window."""
    line = {
        "id": "clip-1",
        "text": "w",
        "seconds": 5.0,
        "clip": "clip-1",
        "clip_anchor": "start",
        "clip_lead_seconds": 10.0,
        "clip_out_seconds": 15.0,
    }
    gen = {
        "lines": [line],
        "bdd_job_id": "",
        "features_generated": 0,
        "package_url": "",
        "timeline_manifest": {"beats": [{"id": "clip-1", "footage_out_point_seconds": 12.0}]},
    }
    outputs = {
        "demo_video_gen": gen,
        "demo_video_narrate": {
            "clips": [{"line_id": "clip-1", "artifact_url": "http://a/c1.mp3", "seconds": 5.0}]
        },
        "demo_video_record": {
            "clips": [
                {
                    "id": "clip-1",
                    "line_id": "clip-1",
                    "video_url": "http://v/c1.mp4",
                    "seconds": 20.0,
                }
            ]
        },
    }
    compile_stage = DemoVideoCompileStage.__new__(DemoVideoCompileStage)
    shot = compile_stage.build_shots(_compile_ctx({}, outputs))[0]
    assert shot["clip_lead_seconds"] == pytest.approx(10.0)
    assert shot["clip_out_seconds"] == pytest.approx(15.0)
    start, end = _compiler_shown_span(shot, 20.0)
    assert (start, end) == (pytest.approx(10.0), pytest.approx(15.0))


# ── Round-5 defect 1: a relief-ON run whose trim already fit is NOT "relief_off" ───────
#
# P1/P2/P4 (2026-10-07, all ``direction.timing.relief.mode="auto"``) were stamped
# ``convergence_outcome="relief_off"`` ⇒ ``STOP_RELIEF_OFF`` at overrun 0: the loop did not
# run because the tail-trimmed timeline already fit, and the out-points-only manifest
# hard-coded the relief-OFF label. P5-shaped footage (32.36s on a 30s order) is trimmed
# to fit, so relief has nothing to do.


@pytest.mark.asyncio
async def test_relief_auto_trimmed_fit_is_converged_not_relief_off() -> None:
    clips = {"clip-1": 12.0, "clip-2": 10.36, "clip-3": 10.0}
    lines = [{"id": c, "text": _words(5), "seconds": None, "clip": c} for c in clips]
    rctx = _reconcile_ctx(lines=lines, clips=clips, order=30.0, relief=True)
    stage = ReconcileTimelineStage(client=cast("WMClient", _ExpanderWM(5)))
    out = (await stage.handle(stage.input_model(), rctx)).model_dump(mode="json")

    manifest = out["timeline_manifest"]
    assert manifest is not None, "precondition: a beat was trimmed, so the manifest exists"
    assert any(b["footage_out_point_seconds"] is not None for b in manifest["beats"])
    run = manifest["run"]
    assert run["loop_iterations"] == 0, "precondition: the relief loop did not run"
    assert run["overrun_seconds"] == 0.0
    assert run["convergence_outcome"] == "converged"
    decision = decide_structural_relief(
        convergence_outcome=run["convergence_outcome"],
        structural_relief_requested=run["structural_relief_requested"],
        overrun_seconds=run["overrun_seconds"],
        prior_overrun_seconds=None,
        pass_number=0,
        max_passes=2,
    )
    assert decision == "STOP_CONVERGED"


@pytest.mark.asyncio
async def test_relief_off_trimmed_fit_still_reads_relief_off() -> None:
    clips = {"clip-1": 12.0, "clip-2": 10.36, "clip-3": 10.0}
    lines = [{"id": c, "text": _words(5), "seconds": None, "clip": c} for c in clips]
    rctx = _reconcile_ctx(lines=lines, clips=clips, order=30.0)  # relief NOT armed
    stage = ReconcileTimelineStage()
    out = (await stage.handle(stage.input_model(), rctx)).model_dump(mode="json")
    run = out["timeline_manifest"]["run"]
    assert run["convergence_outcome"] == "relief_off"
    assert run["converged"] is False


# ── PR #1081 audit: a relief loop that never RAN can never yield a spawn-eligible outcome ─
#
# relief=auto, 14 + 8 + 8 = 30s of footage on a 30s order with a 6s outro (a 24s body).
# The tail-trim fits the body, nothing is unsayable, so the relief loop does NOT run. The
# underspoken re-ask runs AFTER the loop and its spine cap allows 110% of the body, so
# 20-word replies push the fitting timeline 0.40s over. Classifying that final state
# (641c22f7) read ``cant_converge`` and the real spawner bought a PAID child for an overrun
# the FREE re-ask caused, on a run whose relief loop never executed.

_OUTRO_6S = {"enabled": True, "intro": False, "outro": True, "duration_seconds": 6.0}


def _prod_spawn_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Prod settings: kill-switch ON, underfull clamp off.
    monkeypatch.setattr(
        "src.services.runs.structural_relief.get_settings",
        lambda: SimpleNamespace(
            pipeline=SimpleNamespace(
                structural_relief_autoconverge_enabled=True,
                structural_relief_max_passes=2,
                demo_timeline_underfull_fail_ratio=0.0,
            )
        ),
    )


async def _spawn_from(out: dict[str, Any], snapshot: dict[str, Any]) -> AsyncMock:
    start_run = AsyncMock(return_value=SimpleNamespace(id=uuid7()))
    engine: Any = SimpleNamespace(start_run=start_run, db=None)
    rows: list[Any] = [SimpleNamespace(stage_name="reconcile_timeline", output=out)]
    await maybe_spawn_structural_relief_run(
        run_engine=engine,
        parent_run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot={**snapshot, "tenant_id": str(uuid7()), "subject_id": str(uuid7())},
        stage_rows=rows,
    )
    return start_run


async def _relief_auto_reask_overrun_run() -> tuple[dict[str, Any], dict[str, Any]]:
    clips = {"clip-1": 14.0, "clip-2": 8.0, "clip-3": 8.0}
    lines = [{"id": c, "text": _words(8), "seconds": None, "clip": c} for c in clips]
    rctx = _reconcile_ctx(lines=lines, clips=clips, order=30.0, title_card=_OUTRO_6S, relief=True)
    stage = ReconcileTimelineStage(client=cast("WMClient", _ExpanderWM(20)))
    out = (await stage.handle(stage.input_model(), rctx)).model_dump(mode="json")
    return out, rctx.config_snapshot


@pytest.mark.asyncio
async def test_relief_auto_loop_not_run_reask_overrun_never_spawns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, snapshot = await _relief_auto_reask_overrun_run()
    run = out["timeline_manifest"]["run"]
    assert run["loop_iterations"] == 0, "precondition: the relief loop did not run"
    assert run["overrun_seconds"] > 0, "precondition: the free re-ask pushed it over"
    _prod_spawn_settings(monkeypatch)
    start_run = await _spawn_from(out, snapshot)
    start_run.assert_not_awaited()
    assert run["convergence_outcome"] == "relief_not_run"
    assert run["converged"] is False


def test_relief_not_run_maps_to_a_stop_whatever_the_overrun() -> None:
    for overrun in (0.0, 0.4, 60.0):
        decision = decide_structural_relief(
            convergence_outcome="relief_not_run",
            structural_relief_requested=True,
            overrun_seconds=overrun,
            prior_overrun_seconds=None,
            pass_number=0,
            max_passes=2,
        )
        assert decision == "STOP_CANT_CONVERGE"


@pytest.mark.asyncio
async def test_relief_auto_loop_that_ran_and_cannot_converge_still_spawns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 6 footage beats x 10s at a 30s order, relief ARMED: infeasible ⇒ the loop runs, the
    # footage floor exceeds the order ⇒ cant_converge ⇒ the paid child IS the lever.
    clips = {f"clip-{i}": 10.0 for i in range(1, 7)}
    lines = [{"id": c, "text": _words(8), "seconds": None, "clip": c} for c in clips]
    rctx = _reconcile_ctx(lines=lines, clips=clips, order=30.0, relief=True)
    stage = ReconcileTimelineStage(client=cast("WMClient", _ExpanderWM(8)))
    out = (await stage.handle(stage.input_model(), rctx)).model_dump(mode="json")
    run = out["timeline_manifest"]["run"]
    assert run["loop_iterations"] >= 1, "precondition: the relief loop ran"
    assert run["convergence_outcome"] == "cant_converge"
    _prod_spawn_settings(monkeypatch)
    start_run = await _spawn_from(out, rctx.config_snapshot)
    start_run.assert_awaited_once()
