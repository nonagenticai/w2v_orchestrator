"""``timeline_manifest`` must never lie about whether a relief run happened.

``timeline_manifest`` is the ONLY machine-readable signal that distinguishes "this was a
C13 relief run" from "this was never a relief run" — ``structural_relief.py`` reads
``None`` as the latter. Two silent falsehoods were possible and are pinned shut here:

* the broad ``except Exception`` around the relief loop left the manifest ``None``, so a
  CRASHED relief attempt was indistinguishable from a healthy feasible run. It now emits a
  manifest whose ``convergence_outcome`` is ``"relief_failed"``;
* the ``reconcile_only`` early return DROPPED an already-computed manifest, so a
  reconcile_only relief run looked like a non-relief run downstream.

And the guard that keeps the fix honest: on a FEASIBLE run the manifest must STAY ``None``.
``None`` is the correct value there — an over-fix that starts always stamping a
``convergence_outcome`` would make every run look like a relief run, which is the same
silent falsehood with the sign flipped.

Setup (``_ctx`` / ``_lines`` / ``_clips`` / ``_wm_fake`` / ``_relief_timing``) is reused
from :mod:`tests.services.stages.demo_video.test_reconcile_timeline` — the faithful,
prompt-responsive WM stub with the deterministic solver running FOR REAL. The stage is
called DIRECTLY (no TestClient).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.services.stages.demo_video.reconcile_timeline import ReconcileTimelineStage
from src.services.stages.upstream import PermanentUpstreamError

# The demo_video suite's autouse network guard: these tests call the stage with relief
# armed, and its measured line fit (like the rate probe) must never build a real
# TTSClient in a unit test — it fails fast here and falls back, as with TTS down.
from tests.services.stages.demo_video.conftest import (
    _no_speech_rate_network,  # noqa: F401  (autouse fixture, applied by import)
)
from tests.services.stages.demo_video.test_reconcile_timeline import (
    _FEASIBLE_TARGET,
    _LONG,
    _RELIEF_TARGET,
    _clips,
    _ctx,
    _lines,
    _relief_timing,
    _wm_fake,
)


@pytest.mark.asyncio
async def test_relief_loop_failure_stamps_relief_failed_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relief loop that RAISES still emits a manifest, marked ``relief_failed``.

    ``_relief_loop`` is the symbol the ``except Exception`` wraps; making it raise the
    permanent-upstream shape (402 byo-not-configured / retries-exhausted) exercises the
    accept-current degrade path. Without the ``build_timeline_manifest`` call in that
    handler the manifest is ``None`` and this test fails — a crashed relief attempt would
    be indistinguishable from a run that never armed relief at all.
    """
    stage = ReconcileTimelineStage(client=_wm_fake())
    monkeypatch.setattr(
        stage,
        "_relief_loop",
        AsyncMock(side_effect=PermanentUpstreamError("byo not configured (402)")),
    )
    ctx = _ctx(
        timing=_relief_timing(),
        lines=_lines(_LONG),
        clips=_clips(),
        target_seconds=_RELIEF_TARGET,
    )

    out = await stage.handle(stage.input_model(), ctx)  # SOFT gate: must NOT raise

    assert out.timeline_manifest is not None, "a crashed relief run must still emit a manifest"
    run = out.timeline_manifest.run
    assert run.convergence_outcome == "relief_failed"
    # Honest numbers: no pass completed, nothing converged, no pacing score was ever taken,
    # and no structural rerun is being requested off the back of a failure.
    assert run.loop_iterations == 0
    assert run.converged is False
    assert run.pacing_score is None
    assert run.structural_relief_requested is False


@pytest.mark.asyncio
async def test_reconcile_only_carries_the_computed_manifest() -> None:
    """``reconcile_only`` mutates the SPINE, not the telemetry — the manifest rides along.

    A real relief loop runs (infeasible timeline + ``relief.mode == "auto"``) and produces a
    manifest; the ``reconcile_only`` early return used to discard it. "Mutate nothing" is
    about the authored spine, which is still asserted untouched below.
    """
    fake = _wm_fake()
    stage = ReconcileTimelineStage(client=fake)
    timing: dict[str, Any] = {**_relief_timing(), "mode": "reconcile_only"}
    ctx = _ctx(
        timing=timing,
        lines=_lines(_LONG),
        clips=_clips(),
        target_seconds=_RELIEF_TARGET,
    )

    out = await stage.handle(stage.input_model(), ctx)

    assert fake.message.called, "the relief loop must actually have run for this to prove anything"
    assert out.timeline_manifest is not None, "reconcile_only must not drop the relief manifest"
    # It is the REAL loop's manifest, not the crashed-relief placeholder.
    assert out.timeline_manifest.run.convergence_outcome != "relief_failed"
    assert out.timeline_manifest.run.loop_iterations >= 1
    # …and the spine itself was still left alone.
    assert [ln.seconds for ln in out.lines] == [3.0, 5.0, 3.0]


@pytest.mark.asyncio
async def test_feasible_run_leaves_manifest_none() -> None:
    """GUARD: a feasible, non-relief run carries ``timeline_manifest is None``.

    Relief is ARMED here (``relief.mode == "auto"``) but the timeline is already feasible,
    so the loop never fires and no WM call is made. ``None`` is the honest value —
    ``structural_relief.py`` reads it as "relief never ran". This test exists to fail on a
    future over-fix that starts stamping a manifest (and therefore a
    ``convergence_outcome``) on every run.
    """
    fake = _wm_fake()
    stage = ReconcileTimelineStage(client=fake)
    ctx = _ctx(
        timing=_relief_timing(),
        lines=_lines(),
        clips=_clips(),
        target_seconds=_FEASIBLE_TARGET,
    )

    out = await stage.handle(stage.input_model(), ctx)

    fake.message.assert_not_called()  # feasible ⇒ the relief loop is fully inert
    assert out.timeline_manifest is None
    assert out.notes == []  # nothing the solver could not fix
