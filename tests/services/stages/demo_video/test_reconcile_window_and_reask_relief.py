"""Relief spawn-eligibility regressions for reconcile_timeline (round-6 R1).

Both are proved through the REAL spawner (``maybe_spawn_structural_relief_run``) reading
the stage's own output, with prod settings (kill-switch ON) — never a mocked decision.

1. An AUTHORED-WINDOW footage beat (compile plays only the window and never trims it)
   was pinned at its FULL file length: a 0-5s window on a 20s recording counted 20s, the
   overstated overrun classified ``cant_converge`` and a relief=auto run bought a PAID
   child for seconds that never play.
2. A relief loop that CONVERGED and was then pushed out of band by the post-loop
   underspoken re-ask (spine cap 110% of the body). The free re-ask's lengthened footage
   narration raised the footage trim floors past the target, ``_refresh_run_budget``
   reclassified ``converged`` -> ``cant_converge`` and the spawner bought a PAID child for
   an overrun the loop did not leave. Same class as #1081's loop-never-ran case.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from src.services.llm import WMClient
from src.services.runs.structural_relief_policy import decide_structural_relief
from src.services.stages.demo_video.reconcile_timeline import ReconcileTimelineStage
from tests.services.stages.demo_video.test_reconcile_compile_wiring import (
    _OUTRO_6S,
    _prod_spawn_settings,
    _spawn_from,
)
from tests.services.stages.demo_video.test_reconcile_footage_trim_and_cards import (
    _ctx,
    _ExpanderWM,
    _words,
)


def _decision(run: dict[str, Any]) -> str:
    return decide_structural_relief(
        convergence_outcome=run["convergence_outcome"],
        structural_relief_requested=run["structural_relief_requested"],
        overrun_seconds=run["overrun_seconds"],
        prior_overrun_seconds=None,
        pass_number=0,
        max_passes=2,
    )


# ── 1. authored window ⇒ sized at the window ───────────────────────────────────────────


async def _windowed_run() -> tuple[dict[str, Any], dict[str, Any]]:
    # clip-1: a 20s recording whose author plays only 0-5s. Pinned at 20s the floor is
    # 20 + 6 + 6 (the two 10s clips' 60% trim floors) = 32s > the 30s order ⇒ the relief
    # loop arms and the footage floor alone exceeds the order ⇒ cant_converge ⇒ SPAWN.
    # Sized at the 5s it plays: 5 + 10 + 10 = 25s ⇒ fits, nothing to relieve.
    clips = {"clip-1": 20.0, "clip-2": 10.0, "clip-3": 10.0}
    lines: list[dict[str, Any]] = [
        {"id": c, "text": _words(8), "seconds": None, "clip": c} for c in clips
    ]
    lines[0]["clip_out_seconds"] = 5.0
    rctx = _ctx(lines=lines, clips=clips, order=30.0, relief=True)
    stage = ReconcileTimelineStage(client=cast("WMClient", _ExpanderWM(8)))
    out = (await stage.handle(stage.input_model(), rctx)).model_dump(mode="json")
    return out, rctx.config_snapshot


@pytest.mark.asyncio
async def test_authored_window_beat_is_budgeted_at_its_window_not_the_file() -> None:
    out, _ = await _windowed_run()
    by_id = {line["id"]: line for line in out["lines"]}
    assert by_id["clip-1"]["seconds"] == pytest.approx(5.0)
    manifest = out["timeline_manifest"]
    if manifest is not None:
        assert manifest["run"]["overrun_seconds"] == 0.0


@pytest.mark.asyncio
async def test_authored_window_overrun_never_buys_a_paid_relief_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, snapshot = await _windowed_run()
    manifest = out["timeline_manifest"]
    if manifest is not None:
        assert _decision(manifest["run"]) != "SPAWN"
    _prod_spawn_settings(monkeypatch)
    start_run = await _spawn_from(out, snapshot)
    start_run.assert_not_awaited()


# ── 2. loop converged, then the free re-ask pushed it out of band ──────────────────────


class _TrimThenExpandWM:
    """The relief loop trims the VO beat ``v`` to ``trim_to`` words (⇒ converged); the
    post-loop underspoken re-ask then answers every expansion with ``expand`` words."""

    def __init__(self, *, trim_to: int, expand: int) -> None:
        self.trim_to, self.expand = trim_to, expand
        self.calls: list[str] = []

    async def message(self, **kwargs: Any) -> Any:
        worker = str(kwargs.get("worker_name") or "")
        self.calls.append(worker)
        if worker == "beat-expander":
            return SimpleNamespace(output={"text": _words(self.expand, "x")}, output_type="object")
        if worker == "pacing-critic":
            return SimpleNamespace(output={"overall_score": 5}, output_type="object")
        if worker == "beat-rewriter":
            return SimpleNamespace(output={"text": _words(self.trim_to, "t")}, output_type="object")
        if worker == "timing-director" and "trim" in json.dumps(kwargs.get("output_schema") or {}):
            directive = {"op": "trim", "target_ids": ["v"], "new_word_target": self.trim_to}
            return SimpleNamespace(output={"directives": [directive]}, output_type="object")
        return SimpleNamespace(output={"directives": []}, output_type="object")


async def _converged_then_reask_overrun(clip_1: float) -> tuple[dict[str, Any], dict[str, Any]]:
    # 30s order - 6s outro = 24s body. A 40-word VO beat makes the spine infeasible, so the
    # loop arms; the director trims it to 1 word and the re-solved timeline FITS
    # (converged). The re-ask then lengthens the three 8-word footage narrations to 20
    # words (8.0s at 150 wpm), lifting each footage trim floor to its narration: 0.35s
    # (clip-1 12s) / 0.75s (clip-1 14s) past the body.
    clips = {"clip-1": clip_1, "clip-2": 8.0, "clip-3": 8.0}
    lines: list[dict[str, Any]] = [
        {"id": c, "text": _words(8), "seconds": None, "clip": c} for c in clips
    ]
    lines.insert(1, {"id": "v", "text": _words(40, "v"), "seconds": None})
    rctx = _ctx(lines=lines, clips=clips, order=30.0, title_card=_OUTRO_6S, relief=True)
    rctx.config_snapshot["direction"]["timing"]["relief"]["iterations"] = 3
    wm = _TrimThenExpandWM(trim_to=1, expand=20)
    stage = ReconcileTimelineStage(client=cast("WMClient", wm))
    out = (await stage.handle(stage.input_model(), rctx)).model_dump(mode="json")
    assert "pacing-critic" in wm.calls, "precondition: the loop reached a FEASIBLE draft"
    assert "beat-expander" in wm.calls, "precondition: the re-ask ran"
    return out, rctx.config_snapshot


@pytest.mark.parametrize("clip_1", [12.0])
@pytest.mark.asyncio
async def test_converged_loop_pushed_over_by_the_reask_never_buys_a_paid_child(
    clip_1: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, snapshot = await _converged_then_reask_overrun(clip_1)
    run = out["timeline_manifest"]["run"]
    assert run["loop_iterations"] >= 1, "precondition: the relief loop ran"
    assert run["overrun_seconds"] > 0.0, "precondition: the re-ask pushed it out of band"
    # Out of band ⇒ never labelled converged, on EITHER field.
    assert run["convergence_outcome"] == "progress"
    assert run["converged"] is False
    assert _decision(run) == "STOP_CANT_CONVERGE"
    _prod_spawn_settings(monkeypatch)
    start_run = await _spawn_from(out, snapshot)
    start_run.assert_not_awaited()
