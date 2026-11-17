"""FootageCritiqueStage (L4) — base loop + authoritative-out validation + soft gate.

The base stage is exercised through a stub subclass that overrides the four data/
render hooks so the tests drive the analyzer/worker/loop behaviour without a real
compile. Pure helpers (validation, time→shot bucketing, plan nudging) are tested
directly.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from w2v_common.clients import WMResponse

from src.clients.schemas.video_analyzer import (
    VideoAnalyzerJobResult,
    VideoAnalyzerQualityMetrics,
)
from src.services.runs.run_context import RunContext
from src.services.stages.base import StageSkippedError
from src.services.stages.shared import footage_critique as fc_mod
from src.services.stages.shared.footage_critique import (
    FootageCritiqueOutput,
    FootageCritiqueStage,
    ShotWindow,
)


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    )


class _StubCritiqueStage(FootageCritiqueStage):
    """Base stage with the compile-coupled hooks stubbed out."""

    name = "stub_footage_critique"

    def __init__(
        self,
        *,
        wm: AsyncMock,
        analyzer: AsyncMock,
        plans: list[dict[str, Any]] | None = None,
        windows: list[ShotWindow] | None = None,
        rerender_seq: list[tuple[str, float]] | None = None,
    ) -> None:
        super().__init__(client=wm, analyzer=analyzer, compile_client=AsyncMock())
        self._start = ("http://v/orig.mp4", 10.0)
        self._plans: list[dict[str, Any]] = (
            plans
            if plans is not None
            else [
                {"line_id": "clip-1", "zoom": "auto", "reframe": "follow-cursor"},
                {"line_id": "clip-2", "zoom": "auto", "reframe": "follow-cursor"},
            ]
        )
        self._windows: list[ShotWindow] = (
            windows
            if windows is not None
            else [
                {"shot_id": "clip-1", "t0": 0.0, "t1": 4.0},
                {"shot_id": "clip-2", "t0": 4.0, "t1": 7.0},
            ]
        )
        self._rerender_seq = list(rerender_seq or [("http://v/re-1.mp4", 11.0)])
        self.rerender_calls: list[list[Any]] = []
        # Parallel record of the 1-based pass ordinal handed to each rerender submit,
        # so tests can assert the loop-position discriminator is resume-stable.
        self.rerender_passes: list[int] = []

    def starting_artifact(self, ctx: RunContext) -> tuple[str, float]:
        return self._start

    def base_plans(self, ctx: RunContext) -> list[Any]:
        return list(self._plans)

    def shot_windows(self, ctx: RunContext) -> list[ShotWindow]:
        return list(self._windows)

    async def rerender(self, ctx: RunContext, plans: Any, pass_idx: int) -> tuple[str, float]:
        self.rerender_calls.append(list(plans))
        self.rerender_passes.append(pass_idx)
        if self._rerender_seq:
            return self._rerender_seq.pop(0)
        return ("http://v/re-last.mp4", 12.0)


def _analyzer(
    motion: list[dict[str, Any]] | None = None,
    frame_analyses: list[dict[str, Any]] | None = None,
) -> AsyncMock:
    analyzer = AsyncMock()
    analyzer.start_analysis.return_value = "job-1"
    analyzer.poll_analysis.return_value = {"status": "completed"}
    # get_result returns the validated wire model — the analyzer's real field is
    # ``frame_analyses`` (NOT ``frames``; the route strips any other key).
    analyzer.get_result.return_value = VideoAnalyzerJobResult(
        job_id="job-1",
        status="completed",
        stage="done",
        progress=1.0,
        quality_metrics=VideoAnalyzerQualityMetrics.model_validate(
            {
                "motion_timeline": motion
                if motion is not None
                else [{"t": 1.0, "motion": 0.1}, {"t": 5.0, "motion": 0.9}],
                "mean_sharpness": 50.0,
                "mean_brightness": 120.0,
                "motion_score": 0.5,
                "dominant_colors": ["#ffffff"],
                "overall_score": 3.0,
            }
        ),
        frame_analyses=frame_analyses,
    )
    return analyzer


def _scale_budget(
    monkeypatch: pytest.MonkeyPatch,
    stage: FootageCritiqueStage,
    *,
    outer: float,
    poll: float,
    headroom: float,
    floor: float,
    retries: int,
) -> None:
    """Shrink EVERY term of the batch-phase derivation so the real 950s ladder runs in
    test time, and the ladder's SHAPE is what is under test rather than the wall clock.

    All four terms, not some of them. ``_batch_phase_seconds`` is
    ``outer - analyzer_elapsed - _VALIDATION_HEADROOM_S``, and ``_score_critic_batch``
    then refuses any attempt under ``_CRITIC_FLOOR_SECONDS``; leaving either of the two
    ABSOLUTE constants at its production value while scaling ``outer`` down to ~1s makes
    the phase zero or the batch unconditionally refused, which is a test measuring the
    scaling and not the behaviour. (This helper exists because that is exactly how two
    of these tests broke when the headroom + floor were introduced.)
    """
    monkeypatch.setattr(fc_mod, "_ANALYSIS_POLL_TIMEOUT_S", poll)
    monkeypatch.setattr(fc_mod, "_VALIDATION_HEADROOM_S", headroom)
    monkeypatch.setattr(fc_mod, "_CRITIC_FLOOR_SECONDS", floor)
    stage._llm_timeout = outer
    stage._llm_retries = retries


def _ctx(*, critique: bool = True) -> RunContext:
    footage: dict[str, Any] = {"auto_zoom": True}
    if critique:
        footage["critique"] = {"enabled": True}
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"direction": {"footage": footage}},
    )


# ── Skip-clean ──────────────────────────────────────────────────────────────
async def test_skip_clean_when_critique_flag_absent() -> None:
    wm, analyzer = AsyncMock(), AsyncMock()
    stage = _StubCritiqueStage(wm=wm, analyzer=analyzer)
    with pytest.raises(StageSkippedError) as exc:
        await stage.run({}, ctx=_ctx(critique=False))
    assert exc.value.missing == ["direction.footage.critique"]
    # Byte-identical opt-out: no analyzer, no worker, no re-render.
    analyzer.start_analysis.assert_not_called()
    wm.message.assert_not_called()
    assert stage.rerender_calls == []


class _PassThroughCritiqueStage(FootageCritiqueStage):
    """Terminal-artifact blueprint variant: absent critique passes through."""

    name = "passthrough_footage_critique"
    skip_when_unrequested = False

    def starting_artifact(self, ctx: RunContext) -> tuple[str, float]:
        return ("http://x/v.mp4", 12.0)

    def base_plans(self, ctx: RunContext) -> list[Any]:
        return []

    def shot_windows(self, ctx: RunContext) -> list[ShotWindow]:
        return []

    async def rerender(self, ctx: RunContext, plans: Any, pass_idx: int) -> tuple[str, float]:
        raise AssertionError("rerender must not be reached on pass-through")


# ── Pass-through: skip_when_unrequested=False emits current artifact unchanged ──
async def test_passthrough_delivers_current_artifact_when_critique_absent() -> None:
    wm, analyzer = AsyncMock(), AsyncMock()
    stage = _PassThroughCritiqueStage(client=wm, analyzer=analyzer, compile_client=AsyncMock())
    out = await stage.run({}, ctx=_ctx(critique=False))
    assert isinstance(out, FootageCritiqueOutput)
    # Current terminal artifact delivered byte-identically — no analyzer/WM call.
    assert out.artifact_url == "http://x/v.mp4"
    assert out.duration_seconds == 12.0
    analyzer.start_analysis.assert_not_called()
    wm.message.assert_not_called()
    assert out.critique.reason == "critique_not_requested"
    assert out.critique.flagged_shots == []
    assert out.notes == []
    # No nudge on the pass-through ⇒ no persisted footage-plan override (empty ⇒ a
    # re-compile falls back to the base footage-direction plans, byte-identical).
    assert out.footage_plans == []


# ── Evaluator flags exactly the low-scoring shots + bounded ≤2 passes ────────
async def test_flags_exactly_low_scoring_shots_and_is_bounded() -> None:
    wm = AsyncMock()
    # clip-1 stays weak (framing 2) across BOTH passes; clip-2 is always fine (5).
    # The re-render doesn't improve, so the ORIGINAL is the retained best and its
    # pass-1 critique — flagging exactly clip-1 — is what the output reports.
    low = _resp(
        {
            "shots": [
                {"shot_id": "clip-1", "score": 2.0, "framing": 2.0},
                {"shot_id": "clip-2", "score": 5.0},
            ]
        }
    )
    wm.message.side_effect = [low, low]
    stage = _StubCritiqueStage(
        wm=wm,
        analyzer=_analyzer(),
        plans=[
            {"line_id": "clip-1", "zoom": "off", "reframe": "static"},
            {"line_id": "clip-2", "zoom": "auto", "reframe": "follow-cursor"},
        ],
    )
    out = await stage.run({}, ctx=_ctx())
    # Exactly clip-1 was flagged, and the re-render saw clip-1's plan nudged on.
    assert out.critique.flagged_shots == ["clip-1"]
    assert len(stage.rerender_calls) == 1
    nudged = {str(p["line_id"]): p for p in stage.rerender_calls[0]}
    assert nudged["clip-1"]["zoom"] == "auto"  # framing weak → zoom flipped off→on
    assert nudged["clip-2"]["zoom"] == "auto"  # unflagged clip-2 untouched
    # Hard bound: exactly 2 critic calls / 1 optimize, never a third pass.
    assert wm.message.call_count == 2
    assert out.critique.passes == 2
    assert out.token_usage.calls == 2
    assert out.token_usage.total_tokens == 28
    # No improvement ⇒ original artifact kept; non-convergence is a soft note.
    assert out.artifact_url == "http://v/orig.mp4"
    assert out.critique.converged is False
    assert out.notes


# ── Loop keeps the best-scoring artifact when the re-render improves ──────────
async def test_keeps_rerendered_artifact_when_it_improves() -> None:
    wm = AsyncMock()
    wm.message.side_effect = [
        _resp(
            {
                "shots": [
                    {"shot_id": "clip-1", "score": 2.0, "framing": 2.0},
                    {"shot_id": "clip-2", "score": 5.0},
                ]
            }
        ),
        # The re-rendered video scores clean ⇒ it becomes the retained best.
        _resp(
            {"shots": [{"shot_id": "clip-1", "score": 5.0}, {"shot_id": "clip-2", "score": 5.0}]}
        ),
    ]
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer())
    out = await stage.run({}, ctx=_ctx())
    assert out.artifact_url == "http://v/re-1.mp4"  # higher-scoring re-render wins
    assert out.critique.passes == 2
    assert out.critique.converged is True
    assert out.critique.flagged_shots == []  # the retained best has no weak shots


# ── No shots flagged ⇒ accept current, no re-render ──────────────────────────


# ── Analyzer down ⇒ soft-degrade to accept-current WITHOUT raising ───────────
async def test_analyzer_down_soft_degrades_without_raising() -> None:
    analyzer = AsyncMock()
    analyzer.start_analysis.side_effect = RuntimeError("video_analyzer unreachable")
    wm = AsyncMock()
    stage = _StubCritiqueStage(wm=wm, analyzer=analyzer)
    out = await stage.run({}, ctx=_ctx())  # must NOT raise
    assert out.artifact_url == "http://v/orig.mp4"  # current best accepted
    assert stage.rerender_calls == []
    wm.message.assert_not_called()  # never reached the critic worker
    assert out.critique.shots == []


# ── A genuine analyzer OUTAGE stays a CLEAN soft-accept (contrast to a bug) ───
async def test_httpx_outage_soft_degrades_cleanly() -> None:
    """A real upstream HTTP failure (httpx.HTTPError) is a genuine outage, not a bug:
    the narrowed inner catch still swallows it into a passing (never-raises) critique
    that accepts the current artifact. This is the contract the soft gate keeps — an
    analyzer being down must never fail the run or change the DELIVERABLE.

    ``converged`` is now False (was True): the critic never scored anything, so the
    verdict must not read as "reviewed and met the bar" — see
    ``test_analyzer_transport_outage_is_also_marked_unavailable``. The artifact is
    byte-identical either way; only the honesty of the score/converged/reason changed."""
    analyzer = AsyncMock()
    analyzer.start_analysis.side_effect = httpx.ConnectError("analyzer refused")
    stage = _StubCritiqueStage(wm=AsyncMock(), analyzer=analyzer)
    out = await stage.run({}, ctx=_ctx())  # must NOT raise
    assert out.artifact_url == "http://v/orig.mp4"
    # NOT the degraded "loop_error" path a real bug takes, and NOT "threshold" either
    # (that would mean "reviewed and passed") — a soft accept-current with nothing
    # measured plateaus honestly instead.
    assert out.critique.reason not in ("loop_error", "threshold")
    assert out.critique.converged is False
    assert out.critique.score == pytest.approx(fc_mod._FAILED_EVAL_SCORE)


# ── A CODE BUG in _analyze now SURFACES as degraded (was a fake-clean pass) ───
async def test_analyzer_parse_bug_surfaces_as_degraded_not_clean_pass() -> None:
    """The point of narrowing the inner catch: a programming bug (KeyError/etc.) in the
    analyze/parse path must NOT be mislabelled a clean "analyzer unavailable" pass. It
    now escapes the inner handler, rides the loop's outer belt-and-suspenders gate, and
    marks the run DEGRADED with the real error — the soft gate still never raises."""
    analyzer = AsyncMock()
    analyzer.start_analysis.return_value = "job-1"
    analyzer.poll_analysis.return_value = {"status": "completed"}
    analyzer.get_result.side_effect = KeyError("frame_analyses")  # a parse bug, not an outage
    stage = _StubCritiqueStage(wm=AsyncMock(), analyzer=analyzer)
    out = await stage.run({}, ctx=_ctx())  # soft gate holds — no raise
    assert out.artifact_url == "http://v/orig.mp4"
    # The bug is now VISIBLE: degraded run, real exception in the note — not a fake pass.
    assert out.critique.reason == "loop_error"
    assert out.notes
    assert "KeyError" in out.notes[0] or "frame_analyses" in out.notes[0]


# ── A CODE BUG in the critic-worker path also surfaces as degraded ────────────
async def test_worker_parse_bug_surfaces_as_degraded_not_clean_pass() -> None:
    """A TypeError raised building/parsing the WM exchange is a bug, not a worker outage:
    the narrowed catch at the WM call lets it escape to the degraded gate instead of
    masquerading as 'critic worker error, accepting current'."""
    wm = AsyncMock()
    wm.message.side_effect = TypeError("unexpected response shape")
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer())
    out = await stage.run({}, ctx=_ctx())  # soft gate holds — no raise
    assert out.artifact_url == "http://v/orig.mp4"
    assert out.critique.reason == "loop_error"
    assert out.notes


# ── Control-flow signals PROPAGATE through the loop, not swallowed to degraded ─
@pytest.mark.parametrize("signal_cls", [StageSkippedError])
async def test_skip_precondition_signals_propagate_through_loop(
    signal_cls: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    """StageSkippedError / StagePreconditionError are control-flow, not failures. Before
    the guard, the loop's belt-and-suspenders ``except Exception`` swallowed them into a
    degraded 'accept current' — silently discarding an intended skip/precondition. They
    must now reach the run engine."""
    stage = _StubCritiqueStage(wm=AsyncMock(), analyzer=_analyzer())

    async def _raise_signal(*args: Any, **kwargs: Any) -> Any:
        raise signal_cls("stage signalled mid-loop")

    # Force the signal from inside the guarded loop body (post opt-out checks).
    monkeypatch.setattr(fc_mod, "run_evaluator_optimizer", _raise_signal)
    with pytest.raises(signal_cls):
        await stage.run({}, ctx=_ctx())


# ── Empty plan ⇒ nothing to critique ⇒ accept current ────────────────────────
async def test_no_directed_plan_accepts_current() -> None:
    wm, analyzer = AsyncMock(), AsyncMock()
    stage = _StubCritiqueStage(wm=wm, analyzer=analyzer, plans=[])
    out = await stage.run({}, ctx=_ctx())
    analyzer.start_analysis.assert_not_called()
    assert out.artifact_url == "http://v/orig.mp4"
    assert out.critique.reason == "no_directed_footage"
    assert out.footage_plans == []  # no directed plans ⇒ nothing to persist


# ── footage_plans: the retained best draft's nudged plans are persisted (S20) ──
async def test_footage_plans_persist_the_nudged_best_draft() -> None:
    wm = AsyncMock()
    wm.message.side_effect = [
        _resp(
            {
                "shots": [
                    {"shot_id": "clip-1", "score": 2.0, "framing": 2.0},
                    {"shot_id": "clip-2", "score": 5.0},
                ]
            }
        ),
        # The re-render scores clean ⇒ its NUDGED plans become the retained best.
        _resp(
            {"shots": [{"shot_id": "clip-1", "score": 5.0}, {"shot_id": "clip-2", "score": 5.0}]}
        ),
    ]
    stage = _StubCritiqueStage(
        wm=wm,
        analyzer=_analyzer(),
        plans=[
            {"line_id": "clip-1", "zoom": "off", "reframe": "static"},
            {"line_id": "clip-2", "zoom": "auto", "reframe": "follow-cursor"},
        ],
    )
    out = await stage.run({}, ctx=_ctx())
    assert out.artifact_url == "http://v/re-1.mp4"  # improved re-render retained
    # The persisted footage_plans are the nudged rows that produced that best artifact —
    # so encode_master's re-compile reproduces THIS footage (compile prefers them).
    persisted = {str(p["line_id"]): p for p in out.footage_plans}
    assert set(persisted) == {"clip-1", "clip-2"}
    assert persisted["clip-1"]["zoom"] == "auto"  # nudged off→on, persisted
    assert persisted["clip-2"]["zoom"] == "auto"  # unflagged clip-2 carried through


# ── Batched critic calls: bounded output schema per WM call (single-vLLM fix) ──
def _many(n: int) -> tuple[list[dict[str, Any]], list[ShotWindow]]:
    """n directed shots as 1-second windows clip-1..clip-n, each with a plan."""
    plans = [
        {"line_id": f"clip-{i}", "zoom": "auto", "reframe": "follow-cursor"}
        for i in range(1, n + 1)
    ]
    windows: list[ShotWindow] = [
        {"shot_id": f"clip-{i}", "t0": float(i - 1), "t1": float(i)} for i in range(1, n + 1)
    ]
    return plans, windows


def _batch_aware_wm(score: float = 5.0) -> AsyncMock:
    """A WM mock that answers each critic call with a verdict for EXACTLY the shot_ids
    present in that call's message, and records the per-call id batches so a test can
    prove the batches were disjoint and jointly complete."""
    wm = AsyncMock()
    batches: list[list[str]] = []

    def _side_effect(*, message: str, **_kw: Any) -> WMResponse:
        ids = re.findall(r'"shot_id": "([^"]+)"', message)
        batches.append(ids)
        return _resp({"shots": [{"shot_id": sid, "score": score} for sid in ids]})

    wm.message.side_effect = _side_effect
    wm.seen_batches = batches
    return wm


def _failing_batch_wm(
    *,
    fail_shot: str,
    exc_factory: Callable[[], BaseException],
    score: float = 5.0,
) -> AsyncMock:
    """Batch-aware WM mock that ALWAYS fails the call carrying ``fail_shot``.

    Deliberately not a positional ``side_effect`` list: the critic batches now
    RETRY a transient upstream failure, so a fixed list would be consumed out of
    step with the batches and a sibling batch would silently eat another batch's
    canned answer.
    """
    wm = AsyncMock()
    attempts: list[str] = []

    def _side_effect(*, message: str, **_kw: Any) -> WMResponse:
        ids = re.findall(r'"shot_id": "([^"]+)"', message)
        attempts.append(",".join(ids))
        if fail_shot in ids:
            raise exc_factory()
        return _resp({"shots": [{"shot_id": sid, "score": score} for sid in ids]})

    wm.message.side_effect = _side_effect
    wm.seen_attempts = attempts
    return wm


def _flaky_batch_wm(
    *,
    exc_factory: Callable[[], BaseException],
    failures: int = 1,
    score: float = 5.0,
) -> AsyncMock:
    """Batch-aware WM mock whose FIRST ``failures`` calls raise, then succeed.

    Models a transient upstream: the identical request succeeds on a later draw.
    """
    wm = AsyncMock()
    state = {"failed": 0}

    def _side_effect(*, message: str, **_kw: Any) -> WMResponse:
        if state["failed"] < failures:
            state["failed"] += 1
            raise exc_factory()
        ids = re.findall(r'"shot_id": "([^"]+)"', message)
        return _resp({"shots": [{"shot_id": sid, "score": score} for sid in ids]})

    wm.message.side_effect = _side_effect
    return wm


async def test_critique_splits_shots_into_bounded_batches() -> None:
    """12 directed shots are scored in ceil(12/5)=3 disjoint WM calls (bounded output
    schema per call), and the merged critique covers every shot with its score — so
    the single small vLLM never sees an unbounded ``shots`` array in one call."""
    from src.services.stages.shared.footage_critique import _CRITIC_BATCH_SIZE

    plans, windows = _many(12)
    wm = _batch_aware_wm(score=5.0)
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer(), plans=plans, windows=windows)
    out = await stage.run({}, ctx=_ctx())

    # All 12 shots scored 5 ⇒ converges on pass 1: exactly 3 critic calls, no rerender.
    assert wm.message.call_count == 3
    assert stage.rerender_calls == []
    # Each call carried at most _CRITIC_BATCH_SIZE shots; batches disjoint + complete.
    assert [len(b) for b in wm.seen_batches] == [5, 5, 2]
    assert all(len(b) <= _CRITIC_BATCH_SIZE for b in wm.seen_batches)
    flat = [sid for b in wm.seen_batches for sid in b]
    assert len(flat) == len(set(flat)) == 12  # disjoint, no shot double-scored
    # The merged critique reconstructs a verdict per shot, scores preserved.
    scored = {r.shot_id: r.score for r in out.critique.shots}
    assert set(scored) == {f"clip-{i}" for i in range(1, 13)}
    assert set(scored.values()) == {5.0}
    # Usage is summed across the 3 calls.
    assert out.token_usage.calls == 3


async def test_critic_batches_run_concurrently_and_bounded() -> None:
    """The batches bound each call's PROMPT; they used to be awaited SERIALLY, so a
    demo with N directed shots cost ceil(N/5) sequential model calls inside one
    evaluator-optimizer pass. They now run bounded-parallel.

    Two assertions, because either one alone is satisfied by a bug: more than one
    call in flight (not serial) and never more than ``_CRITIC_BATCH_CONCURRENCY``
    (not an unbounded fan-out at one small vLLM). NO speedup is asserted — the
    per-request token rate under concurrency is unmeasured.
    """
    from src.services.stages.shared.footage_critique import _CRITIC_BATCH_CONCURRENCY

    plans, windows = _many(5 * (_CRITIC_BATCH_CONCURRENCY + 1))
    in_flight = 0
    peak = 0

    async def _slow(*, message: str, **_kw: Any) -> WMResponse:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0.01)
            ids = re.findall(r'"shot_id": "([^"]+)"', message)
            return _resp({"shots": [{"shot_id": sid, "score": 5.0} for sid in ids]})
        finally:
            in_flight -= 1

    wm = AsyncMock()
    wm.message.side_effect = _slow
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer(), plans=plans, windows=windows)
    out = await stage.run({}, ctx=_ctx())

    assert wm.message.call_count == _CRITIC_BATCH_CONCURRENCY + 1
    assert peak > 1, "the critic batches ran one after another"
    assert peak <= _CRITIC_BATCH_CONCURRENCY, f"unbounded fan-out: {peak} calls in flight"
    assert len(out.critique.shots) == len(plans)


async def test_single_batch_when_few_shots_is_one_call() -> None:
    """≤ _CRITIC_BATCH_SIZE shots ⇒ a single WM call (no needless fan-out)."""
    plans, windows = _many(3)
    wm = _batch_aware_wm(score=5.0)
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer(), plans=plans, windows=windows)
    out = await stage.run({}, ctx=_ctx())
    assert wm.message.call_count == 1
    assert {r.shot_id for r in out.critique.shots} == {"clip-1", "clip-2", "clip-3"}


async def test_batch_outage_still_soft_accepts_current() -> None:
    """A WM outage on ANY batch keeps the SOFT contract: accept the current artifact,
    no raise, no re-render.

    Behaviour change (deliberate): the loop no longer ABORTS at the failing batch and
    throws away the batches that already succeeded — it continues and scores the rest.
    All three batches are therefore attempted, and the outage is recorded rather than
    silently laundered into a clean pass."""
    plans, windows = _many(12)
    wm = _failing_batch_wm(
        fail_shot="clip-6", exc_factory=lambda: TimeoutError("batch 2 timed out")
    )
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer(), plans=plans, windows=windows)
    out = await stage.run({}, ctx=_ctx())  # must NOT raise
    assert out.artifact_url == "http://v/orig.mp4"
    assert stage.rerender_calls == []
    assert out.critique.reason != "loop_error"  # clean soft-accept, not a bug path
    assert out.critique.converged is True
    # Continued past the failing batch: all 3 batches attempted (batch 2 twice — a
    # timeout is transient and now spends the configured retry budget).
    assert wm.message.call_count == 4
    # ...and the surviving batches' verdicts were kept, not discarded.
    assert {r.shot_id for r in out.critique.shots} == {
        *(f"clip-{i}" for i in range(1, 6)),
        "clip-11",
        "clip-12",
    }
    # The transient outage still degrades QUIETLY (no raise, no failure) but is now
    # honestly recorded instead of reading as "the critic reviewed this and passed it".
    assert out.critique.critique_unavailable is True
    assert "batch 2" in out.critique.unavailable_reason


# ── The prod defect: the WM critic call must carry an explicit max_tokens ─────
async def test_critic_call_passes_explicit_max_tokens() -> None:
    """Measured live 2026-08-15: with ``max_tokens`` omitted, EVERY critic call was
    rejected 422 "Model token limit (6000) exceeded before any response was
    generated" — and the 4xx was laundered into an outage degrade, making the critic
    permanently inert while runs went green.

    ⚠️ THIS DOCSTRING USED TO SAY WM "RESERVED 6000 output tokens against the small
    self-hosted model's context window", and the assertion below was
    ``_CRITIC_MAX_TOKENS < 6000``. Both were wrong, and the wrong explanation is what
    set the constant to 1800. The model runs at ``--max-model-len=131072``; there is
    no reservation. The 422 comes from ``pydantic_ai/_agent_graph.py:1100-1113``
    AFTER generation, when ``finish_reason == 'length'`` and every part is a
    ``ThinkingPart``: the reasoning model burned the budget thinking. Measured live
    2026-08-19, one prompt, only max_tokens varying: 500 -> 422, 1760 -> 422,
    6000 -> no 422. So the budget is a FLOOR, and the assertion is inverted."""
    from src.services.stages.llm_budget import MIN_OUTPUT_TOKEN_BUDGET
    from src.services.stages.shared.footage_critique import (
        _CRITIC_ANSWER_TOKENS,
        _CRITIC_BATCH_SIZE,
        _CRITIC_MAX_TOKENS,
        _CRITIC_MAX_TOKENS_PER_SHOT,
    )

    wm = _batch_aware_wm(score=5.0)
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer())
    await stage.run({}, ctx=_ctx())

    assert wm.message.call_count == 1
    kwargs = wm.message.call_args.kwargs
    assert kwargs["max_tokens"] == _CRITIC_MAX_TOKENS
    # The ANSWER estimate stays TIED to the batch size so the two cannot drift apart
    # when one of them is tuned...
    assert _CRITIC_ANSWER_TOKENS == _CRITIC_BATCH_SIZE * _CRITIC_MAX_TOKENS_PER_SHOT
    # ...and the BUDGET clears the floor, with reasoning headroom on top of it.
    assert _CRITIC_MAX_TOKENS >= MIN_OUTPUT_TOKEN_BUDGET
    assert _CRITIC_MAX_TOKENS > _CRITIC_ANSWER_TOKENS


def _http_status_error(status: int, body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://wm/workers/footage-critic/message")
    response = httpx.Response(status, text=body, request=request)
    return httpx.HTTPStatusError(
        f"worker_management returned {status}: {body}", request=request, response=response
    )


async def test_4xx_on_middle_batch_keeps_earlier_shots_and_marks_unavailable() -> None:
    """A 422 is a CONTRACT/CONFIG fault, not an outage. It must not (a) discard the
    batches that already succeeded, nor (b) read as a clean 'critic passed it'."""
    plans, windows = _many(12)  # 3 batches of 5/5/2
    wm = AsyncMock()
    wm.message.side_effect = [
        _resp({"shots": [{"shot_id": f"clip-{i}", "score": 5.0} for i in range(1, 6)]}),
        _http_status_error(
            422,
            '{"error":{"code":"VALIDATION_ERROR","message":"Model token limit (6000) '
            'exceeded before any response was generated.","status":422}}',
        ),
        _resp({"shots": [{"shot_id": f"clip-{i}", "score": 5.0} for i in range(11, 13)]}),
    ]
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer(), plans=plans, windows=windows)
    out = await stage.run({}, ctx=_ctx())  # the advisory critic NEVER fails the run

    # (a) batch 1 + batch 3 survived the batch-2 fault — the fan-out continued.
    assert wm.message.call_count == 3
    scored = {r.shot_id for r in out.critique.shots}
    assert scored == {*(f"clip-{i}" for i in range(1, 6)), "clip-11", "clip-12"}
    # (b) the incomplete critique is VISIBLE, structurally and as a degrade note.
    assert out.critique.critique_unavailable is True
    assert "422" in out.critique.unavailable_reason
    assert "batch 2" in out.critique.unavailable_reason
    assert out.notes and "INCOMPLETE" in out.notes[0]
    # Still a deliverable: the artifact ships and the run is not failed.
    assert out.artifact_url == "http://v/orig.mp4"
    assert out.critique.reason != "loop_error"


async def test_4xx_on_every_batch_does_not_fail_the_run_but_is_never_a_clean_pass() -> None:
    """The live prod shape: every call 422s. The run must survive (advisory critic,
    orchestrator #812) — but `flagged_shots: []` must NOT be reported as a clean pass."""
    plans, windows = _many(12)
    wm = AsyncMock()
    wm.message.side_effect = _http_status_error(422, "Model token limit (6000) exceeded")
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer(), plans=plans, windows=windows)
    out = await stage.run({}, ctx=_ctx())  # must NOT raise

    assert wm.message.call_count == 3  # every batch attempted, none aborted the loop
    assert out.artifact_url == "http://v/orig.mp4"
    assert stage.rerender_calls == []
    assert out.critique.shots == []
    assert out.critique.flagged_shots == []
    # The whole point: an empty verdict from a critic that NEVER RAN is marked as such.
    assert out.critique.critique_unavailable is True
    assert "422" in out.critique.unavailable_reason
    assert out.notes and "INCOMPLETE" in out.notes[0]


async def test_5xx_batch_degrades_quietly_without_failing_the_run() -> None:
    """A 5xx is a genuine upstream outage: it degrades quietly (WARNING, no raise, the
    current artifact accepted) — the outage path must not regress into a hard error."""
    plans, windows = _many(3)  # a single batch
    wm = AsyncMock()
    wm.message.side_effect = _http_status_error(503, "upstream unavailable")
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer(), plans=plans, windows=windows)
    out = await stage.run({}, ctx=_ctx())  # must NOT raise

    assert out.artifact_url == "http://v/orig.mp4"
    assert stage.rerender_calls == []
    assert out.critique.reason not in ("loop_error", "threshold")  # clean soft-accept
    # Nothing was scored ⇒ must NOT read as "reviewed and met the bar" (was True).
    assert out.critique.converged is False
    assert out.critique.critique_unavailable is True  # honest, but not a run failure


async def test_all_batches_succeed_leaves_critique_marked_available() -> None:
    """Regression guard on the happy path: when every batch answers, the new markers
    stay OFF and no degrade note is emitted (behaviour unchanged)."""
    plans, windows = _many(12)
    wm = _batch_aware_wm(score=5.0)
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer(), plans=plans, windows=windows)
    out = await stage.run({}, ctx=_ctx())
    assert wm.message.call_count == 3
    assert out.critique.critique_unavailable is False
    assert out.critique.unavailable_reason == ""
    assert out.notes == []


# ── The token budget must be BACKED by the schema, not just by a comment ──────
def test_output_schema_bounds_the_free_text_fields_the_budget_assumes() -> None:
    """``_CRITIC_MAX_TOKENS`` is sized from ``_MAX_ISSUE_CHARS`` / ``_MAX_FIX_CHARS``, but
    those constants are applied in ``_validate`` AFTER the response is parsed — they
    constrain what we KEEP, not what the model may EMIT. The schema must therefore TELL
    the model the bound, in the one way that actually reaches it.

    NOT via ``maxLength``: vLLM guided decoding constrains grammar and type only, so the
    keyword cannot stop a long string being generated — and WM strips it before building
    the grammar while still validating the reply against the unstripped schema, with
    ``retries=0``, so one over-long string would permanently kill the stage. That is what
    ``tests/test_output_schema_bounds.py`` guards, and it is why the bound lives in
    ``description`` and is enforced on consumption instead."""
    from src.services.stages.llm_budget import MIN_OUTPUT_TOKEN_BUDGET
    from src.services.stages.shared.footage_critique import (
        _CRITIC_BATCH_SIZE,
        _CRITIC_MAX_TOKENS,
        _MAX_FIX_CHARS,
        _MAX_ISSUE_CHARS,
        _OUTPUT_SCHEMA,
    )

    schema: dict[str, Any] = cast("dict[str, Any]", _OUTPUT_SCHEMA)
    props: dict[str, Any] = schema["properties"]["shots"]["items"]["properties"]
    assert "maxLength" not in props["issue"], "unenforceable, and fatal under WM retries=0"
    assert "maxLength" not in props["suggested_fix"]
    assert str(_MAX_ISSUE_CHARS) in props["issue"]["description"]
    assert str(_MAX_FIX_CHARS) in props["suggested_fix"]["description"]
    # And the per-call budget still has room for a full batch of worst-case verdicts:
    # two 300-char fields ≈ 300 tok at a conservative 2 chars/tok, plus ids/numbers.
    assert _CRITIC_MAX_TOKENS >= _CRITIC_BATCH_SIZE * (
        (_MAX_ISSUE_CHARS + _MAX_FIX_CHARS) // 2 + 40
    )
    # ...and it must ALSO clear the floor. This line used to read
    # ``< 6000  # still far under WM's default reservation`` — the refuted
    # mechanism, and the reason the constant was 1800. See llm_budget.py.
    assert _CRITIC_MAX_TOKENS >= MIN_OUTPUT_TOKEN_BUDGET


# ── A truncated/undecodable critic response is NEVER a clean pass ─────────────
async def test_truncated_json_response_marks_unavailable_and_is_not_a_clean_pass() -> None:
    """The silent-green this PR exists to remove. When generation truncates, the WM
    client's JSON decode raises ``ValueError``/``JSONDecodeError`` — which is in NEITHER
    of ``critic_fn``'s except tuples, so it escapes to the loop's belt-and-suspenders
    ``except Exception`` → ``_accept_current(reason="loop_error")``. That path used to
    build the record with the honest-degrade markers left at their defaults, so a critique
    that never parsed reported ``critique_unavailable: False`` + ``flagged_shots: []``:
    indistinguishable from "the critic reviewed this footage and passed it".

    It must now read as "the critic did not deliver a verdict" — and STILL not fail the
    run (the critic is advisory, orchestrator #812)."""
    wm = AsyncMock()
    wm.message.side_effect = json.JSONDecodeError(
        "Unterminated string starting at", '{"shots": [{"shot_id": "clip-1", "issue": "the', 30
    )
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer())

    out = await stage.run({}, ctx=_ctx())  # advisory critic — must NOT raise

    # The deliverable still ships, unchanged, with no re-render.
    assert out.artifact_url == "http://v/orig.mp4"
    assert stage.rerender_calls == []
    assert out.critique.reason == "loop_error"
    assert out.critique.shots == []
    assert out.critique.flagged_shots == []
    # ...but it is NOT reported as a pass: the markers + the degrade note make the
    # truncation visible to a reader of stage_outputs.
    assert out.critique.critique_unavailable is True
    assert out.critique.unavailable_reason
    assert out.notes and "INCOMPLETE" in out.notes[0]


# ── A later SUCCESSFUL pass on the same artifact clears the stale marker ───────
async def test_successful_second_pass_clears_stale_unavailable_marker() -> None:
    """``records_by_url`` is overwritten on every critic pass, but the unavailable marker
    used to be only ever SET. ``optimizer_fn`` returns the SAME draft when a re-render
    fails, so pass 2 re-critiques the identical ``artifact_url``: a fully successful
    second pass refreshed the records while keeping pass 1's stale marker, and a clean
    critique degraded the run.

    Pass 1: batch 2 times out (marker set, clip-1 flagged at 2.0). The re-render then
    fails, so the loop re-critiques the SAME url. Pass 2: every batch answers ⇒ the
    marker must be gone and no degrade note emitted."""
    plans, windows = _many(12)  # 3 batches of 5 / 5 / 2

    class _RerenderFailsStage(_StubCritiqueStage):
        async def rerender(self, ctx: RunContext, plans: Any, pass_idx: int) -> tuple[str, float]:
            self.rerender_calls.append(list(plans))
            self.rerender_passes.append(pass_idx)
            # Same draft comes back out of optimizer_fn ⇒ pass 2 scores the SAME url.
            raise RuntimeError("compile job lost")

    # Batch-AWARE (not a positional list): a transient failure is now RETRIED, so a
    # fixed list would be consumed out of step and a sibling batch would eat another
    # batch's canned answer. Keyed on the batch's first shot id + how often that batch
    # has been asked.
    seen: dict[str, int] = {}

    def _side_effect(*, message: str, **_kw: Any) -> WMResponse:
        ids = re.findall(r'"shot_id": "([^"]+)"', message)
        key = ids[0]
        seen[key] = seen.get(key, 0) + 1
        if key == "clip-6" and seen[key] <= 2:
            # Pass 1: both attempts (initial + the one retry) time out.
            raise TimeoutError("critic worker timed out on batch 2")
        if key == "clip-1" and seen[key] == 1:
            return _resp(
                {
                    "shots": [
                        {"shot_id": "clip-1", "score": 2.0, "framing": 2.0},
                        *({"shot_id": f"clip-{i}", "score": 5.0} for i in range(2, 6)),
                    ]
                }
            )
        return _resp({"shots": [{"shot_id": sid, "score": 5.0} for sid in ids]})

    wm = AsyncMock()
    wm.message.side_effect = _side_effect
    stage = _RerenderFailsStage(wm=wm, analyzer=_analyzer(), plans=plans, windows=windows)

    out = await stage.run({}, ctx=_ctx())

    # Two full 3-batch passes + the one retry batch 2 spent on its transient timeout.
    assert wm.message.call_count == 7
    assert stage.rerender_passes == [1]  # one re-render attempt, and it failed
    assert out.artifact_url == "http://v/orig.mp4"  # the same url both passes
    # Pass 2 scored every shot ⇒ the pass-1 marker is STALE and must be cleared.
    assert {r.shot_id for r in out.critique.shots} == {f"clip-{i}" for i in range(1, 13)}
    assert out.critique.critique_unavailable is False
    assert out.critique.unavailable_reason == ""
    assert out.notes == []  # a clean pass must not degrade the run


# ── _validate: drop unknown shot_id, clamp score, bound strings ──────────────
def test_validate_drops_unknown_and_clamps_and_bounds() -> None:
    stage = _StubCritiqueStage(wm=AsyncMock(), analyzer=AsyncMock())
    records = stage._validate(
        [
            {"shot_id": "clip-1", "score": 9.0, "issue": "x" * 500, "suggested_fix": "y" * 500},
            {"shot_id": "clip-1", "score": 1.0},  # duplicate id → dropped
            {"shot_id": "ghost", "score": 3.0},  # unknown id → dropped
            {"score": 3.0},  # missing id → dropped
        ],
        {"clip-1"},
    )
    assert [r.shot_id for r in records] == ["clip-1"]
    assert records[0].score == 5.0  # clamped to the [1,5] band
    assert len(records[0].issue) == 300
    assert len(records[0].suggested_fix) == 300


# ── time→shot bucketing places a sample in the right window ──────────────────


# ── _nudge_plans: intent-only, flagged shots only ───────────────────────────


# ── Regression: _analyze surfaces frame_analyses (NOT the never-sent ``frames``) ──


# ── The ONE upstream policy: classify()/guarded() decide, stage_call spends ───
#
# Before this suite the critic call caught ``httpx.HTTPStatusError`` itself and
# decided alone. Consequences, both live:
#   * a TRANSIENT upstream (429/408/5xx blip, or the 422 "Exceeded maximum output
#     retries" bad DRAW) spent ZERO of the configured ``llm_retries`` and degraded
#     the critique to "unavailable" on the first failure;
#   * an analyzer 4xx returned a passing, no-flag verdict with NO unavailable
#     marker — structurally identical to "the critic reviewed it and found nothing".


async def test_transient_429_is_retried_and_the_critique_is_not_lost() -> None:
    """A 429 is in ``upstream.TRANSIENT_STATUSES``. The retry budget must be spent on
    it, and a successful second draw must produce a REAL critique — not a degraded
    "unavailable" verdict thrown away after one attempt."""
    plans, windows = _many(3)  # a single batch
    wm = _flaky_batch_wm(exc_factory=lambda: _http_status_error(429, "rate limited"))
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer(), plans=plans, windows=windows)
    out = await stage.run({}, ctx=_ctx())

    assert wm.message.call_count == 2  # retried, not given up on
    assert {r.shot_id for r in out.critique.shots} == {"clip-1", "clip-2", "clip-3"}
    assert out.critique.critique_unavailable is False
    assert out.critique.unavailable_reason == ""
    assert out.notes == []


async def test_422_maximum_output_retries_is_a_bad_draw_and_is_retried() -> None:
    """``upstream.TRANSIENT_4XX_MARKERS``: WM answers 422 for pydantic-ai's "Exceeded
    maximum output retries (3)", but that is a bad DRAW, not a bad REQUEST (measured
    103:1 success ratio in prod). The scoped escape must reach this call site."""
    plans, windows = _many(3)
    body = (
        '{"error":{"code":"VALIDATION_ERROR","message":'
        '"Exceeded maximum output retries (3)","status":422}}'
    )
    wm = _flaky_batch_wm(exc_factory=lambda: _http_status_error(422, body))
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer(), plans=plans, windows=windows)
    out = await stage.run({}, ctx=_ctx())

    assert wm.message.call_count == 2
    assert {r.shot_id for r in out.critique.shots} == {"clip-1", "clip-2", "clip-3"}
    assert out.critique.critique_unavailable is False


async def test_transient_5xx_is_retried_before_the_critique_is_called_unavailable() -> None:
    """A 5xx with no permanent marker is an infrastructure blip — the reason the retry
    budget exists at all."""
    plans, windows = _many(3)
    wm = _flaky_batch_wm(exc_factory=lambda: _http_status_error(503, "upstream unavailable"))
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer(), plans=plans, windows=windows)
    out = await stage.run({}, ctx=_ctx())

    assert wm.message.call_count == 2
    assert out.critique.critique_unavailable is False


async def test_permanent_402_is_not_retried_and_carries_the_real_sentence() -> None:
    """A 402 ``byo_llm_not_configured`` can never be fixed by repeating the request:
    exactly ONE attempt, and the user-facing sentence survives into the reason."""
    plans, windows = _many(3)
    body = (
        '{"error":{"code":"ERROR","message":{"code":"byo_llm_not_configured",'
        '"message":"Configure your LLM provider in Settings before running agents."},'
        '"status":402}}'
    )
    wm = AsyncMock()
    wm.message.side_effect = _http_status_error(402, body)
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer(), plans=plans, windows=windows)
    out = await stage.run({}, ctx=_ctx())  # advisory critic never fails the run

    assert wm.message.call_count == 1  # no retry burned on a guaranteed loss
    assert out.critique.critique_unavailable is True
    assert "permanent upstream fault" in out.critique.unavailable_reason
    assert "Configure your LLM provider" in out.critique.unavailable_reason
    assert out.notes and "INCOMPLETE" in out.notes[0]


async def test_exhausted_transient_is_reported_as_unavailable_not_as_a_clean_pass() -> None:
    """When the retries genuinely run out, the critique is INCOMPLETE — visible in the
    record and in a degrade note — and never a silent green."""
    plans, windows = _many(3)
    wm = _failing_batch_wm(
        fail_shot="clip-1",
        exc_factory=lambda: _http_status_error(429, "still rate limited"),
    )
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer(), plans=plans, windows=windows)
    out = await stage.run({}, ctx=_ctx())

    assert wm.message.call_count == 2  # llm_retries=1 ⇒ 2 attempts, then give up
    assert out.critique.shots == []
    assert out.critique.flagged_shots == []
    assert out.critique.critique_unavailable is True
    assert "transient upstream failure after 2 attempt(s)" in out.critique.unavailable_reason
    assert out.notes and "INCOMPLETE" in out.notes[0]


async def test_analyzer_4xx_is_never_reported_as_a_clean_pass() -> None:
    """The other half of the silent-green hole: the ANALYZER call. A 4xx there (its
    SSRF allow-list rejecting the artifact host is the live shape) used to return a
    passing, no-flag verdict with ``critique_unavailable: False`` — i.e. exactly what a
    critic that ran and found nothing looks like. The deliverable still ships; the
    report must say the footage was never reviewed."""
    analyzer = AsyncMock()
    analyzer.start_analysis.side_effect = _http_status_error(400, "url host not allowed")
    stage = _StubCritiqueStage(wm=AsyncMock(), analyzer=analyzer)
    out = await stage.run({}, ctx=_ctx())  # must NOT raise

    assert out.artifact_url == "http://v/orig.mp4"  # deliverable unchanged
    assert out.critique.shots == []
    assert out.critique.critique_unavailable is True
    assert "analyzer" in out.critique.unavailable_reason
    assert out.notes and "INCOMPLETE" in out.notes[0]


async def test_analyzer_transport_outage_is_also_marked_unavailable() -> None:
    """A connect error never reached the analyzer, so it says nothing about the
    footage either — same honest marker, still no run failure. ``converged`` must be
    False: an analyzer that never spoke did not "meet the bar" (it measured nothing),
    which is the exact silent-green hole ``_FAILED_EVAL_SCORE`` exists to close."""
    analyzer = AsyncMock()
    analyzer.start_analysis.side_effect = httpx.ConnectError("analyzer refused")
    stage = _StubCritiqueStage(wm=AsyncMock(), analyzer=analyzer)
    out = await stage.run({}, ctx=_ctx())

    assert out.critique.reason not in ("loop_error", "threshold")
    assert out.critique.converged is False
    assert out.critique.critique_unavailable is True
    assert out.critique.critique_unavailable is True


# ── Inner ladder vs. the outer slice (the two properties #872 lifted from qa.py) ──
#
# Both of these went RED before the ladder was converted to ``attempt_plan``: the
# per-batch ``stage_call`` was handed the SAME ``timeout_seconds`` + ``retries`` as the
# evaluator-optimizer's own ``stage_call`` that wraps ``critic_fn``, so `attempts x the
# whole outer slice` of inner ladder was authorised inside ONE outer slice of that size.
# The outer clock therefore always fired first (the inner second attempt was dead code),
# the cancel discarded every batch already scored, and the outer retry re-paid WM for
# all of them.


async def test_the_inner_critic_ladder_cannot_outlive_its_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hung worker is cut by the BATCH's own clamp, inside the outer slice.

    12 shots ⇒ 3 batches ⇒ one wave at ``_CRITIC_BATCH_CONCURRENCY``, every call hangs
    forever. The assertions are observable outcomes, not call counts:

    * the stage RETURNS inside one outer per-attempt budget — with the inner ladder
      sized at the full outer budget x (retries + 1) that was arithmetically impossible;
    * the reason is NOT ``loop_error``, i.e. the batches were cut by their own share and
      the pass came back on its own feet rather than being cancelled by the outer clock;
    * the critique is still marked UNAVAILABLE with the share in the sentence, so a
      timed-out critic can never read as "reviewed, found nothing" (#864).
    """
    plans, windows = _many(12)

    async def _hang(**_kw: Any) -> WMResponse:
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    wm = AsyncMock()
    wm.message.side_effect = _hang
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer(), plans=plans, windows=windows)
    # Scale EVERY term of the phase derivation so the real 950s/180s ladder runs in test
    # time. ``retries=1`` is the ladder that used to be 2 x the WHOLE outer slice.
    _scale_budget(monkeypatch, stage, outer=1.0, poll=0.25, headroom=0.02, floor=0.01, retries=1)

    loop = asyncio.get_running_loop()
    started = loop.time()
    out = await stage.run({}, ctx=_ctx())
    elapsed = loop.time() - started

    assert elapsed < stage._llm_timeout, (
        f"the critic pass took {elapsed:.2f}s of a {stage._llm_timeout:.2f}s outer slice — "
        "the inner ladder outlived the slice it was carved out of"
    )
    assert out.critique.reason != "loop_error", (
        "the pass was cancelled by the outer clock instead of stopping at its own share"
    )
    assert out.critique.critique_unavailable is True
    assert "share" in out.critique.unavailable_reason
    assert out.critique.shots == []
    # SOFT contract intact: the current artifact is still delivered, run not failed.
    assert out.artifact_url == "http://v/orig.mp4"


async def test_partial_verdicts_survive_an_outer_cancel_instead_of_being_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass 1 scores all 12 shots; the outer clock then kills pass 2 — the 12 verdicts
    must still be in the emitted critique.

    ``critic_fn`` publishes its record list into ``records_by_url`` BEFORE its first
    await and mutates it in place as each batch lands, and the ``loop_error`` fallback
    now reads it back. Before that, the records were only assigned AFTER the gather and
    the fallback emitted ``shots=[]``, so a cancelled pass threw away every batch already
    paid for and reported the footage as never reviewed.

    The degrade stays HONEST (#864): partial verdicts do NOT make the critique available
    — ``critique_unavailable`` is still True with the loop-error sentence.
    """
    plans, windows = _many(12)

    class _RerenderFailsStage(_StubCritiqueStage):
        async def rerender(self, ctx: RunContext, plans: Any, pass_idx: int) -> tuple[str, float]:
            self.rerender_passes.append(pass_idx)
            # Same draft back out of optimizer_fn ⇒ pass 2 re-critiques the SAME url.
            raise RuntimeError("compile job lost")

    def _score(*, message: str, **_kw: Any) -> WMResponse:
        ids = re.findall(r'"shot_id": "([^"]+)"', message)
        return _resp(
            {
                "shots": [
                    # clip-1 is weak ⇒ flagged ⇒ the optimizer runs ⇒ there IS a pass 2.
                    {"shot_id": sid, "score": 2.0 if sid == "clip-1" else 5.0}
                    for sid in ids
                ]
            }
        )

    wm = AsyncMock()
    wm.message.side_effect = _score

    analyzer = _analyzer()
    completed = analyzer.get_result.return_value
    reads = {"n": 0}

    async def _get_result(_job_id: str) -> Any:
        reads["n"] += 1
        if reads["n"] == 1:
            return completed
        # Pass 2's analyzer read never returns ⇒ the loop engine's per-attempt
        # stage_call times out and the whole pass is cancelled.
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    analyzer.get_result.side_effect = _get_result

    stage = _RerenderFailsStage(wm=wm, analyzer=analyzer, plans=plans, windows=windows)
    # >= stage_call's own 1.0s floor; the phase leaves the batches a real slice while
    # the hung pass-2 analyzer still blows the outer clock fast.
    _scale_budget(monkeypatch, stage, outer=1.2, poll=0.2, headroom=0.02, floor=0.01, retries=0)

    out = await stage.run({}, ctx=_ctx())

    # THE property: pass 1's work is still here after pass 2 was cancelled.
    assert {r.shot_id for r in out.critique.shots} == {f"clip-{i}" for i in range(1, 13)}
    assert out.critique.flagged_shots == ["clip-1"]
    # ...and it is still an HONEST degrade, not a clean pass.
    assert out.critique.reason == "loop_error"
    assert out.critique.critique_unavailable is True
    assert "critic loop error" in out.critique.unavailable_reason
    assert any("12 shot(s) were scored" in n for n in out.notes)
    # SOFT contract intact: the current artifact is still delivered, run not failed.
    assert out.artifact_url == "http://v/orig.mp4"


# ── The analyzer poll bound is not the outlier (measured vacuous critic) ─────
#
# Live run 01a021c6-01ba-70dc-a9cb-0aa194de7eb1 (2026-08-21, demo_video_v1): the
# analyzer read timed out at 180s, ZERO shots were scored, and the stage still
# reported ``completed`` after 3m09s — a critic that contributed nothing. 180 was the
# only site in the estate polling THIS analyzer for THIS job shape that low.
async def test_analyzer_poll_bound_is_not_below_the_sibling_stages() -> None:
    """The floor is the sibling stage; the ceiling is the TRANSPORT, not a budget.

    ⚠️ This guard was briefly UNSATISFIABLE — it asserted `polled >= 1100` (the value
    `final_critique` then held, derived from its StageSpec budget) AND
    `polled <= 600` (the client default). No value could pass. The contradiction was
    the honest signal: 1100 was above the clamp in `BaseServiceClient.poll_job`, which
    RAISES rather than caps, so the sibling it was being held to was itself broken.

    With both stages at the transport ceiling the invariant is expressible again, and
    both halves still bite:

    * floor — this stage polls the SAME analyzer for the SAME whole-video read as
      `demo_video/final_critique`, so it may not be handed LESS time than its sibling.
      This is a floor, not a derivation: the two size their polls independently (see
      the constant's comment), they simply both land on the ceiling today.
    * ceiling — `_clamp_timeout` rejects anything above it, checked here against the
      library rather than a literal, and generically for every stage in
      `tests/services/stages/test_poll_timeouts_within_client_ceiling.py`.
    """
    from w2v_common.clients.base import _clamp_timeout

    from src.services.stages.demo_video.final_critique import (
        _ANALYSIS_POLL_TIMEOUT_S as _FINAL_CRITIQUE_POLL,
    )

    wm = AsyncMock()
    wm.message.return_value = _resp(
        {"shots": [{"shot_id": "clip-1", "score": 5.0}, {"shot_id": "clip-2", "score": 5.0}]}
    )
    analyzer = _analyzer()
    stage = _StubCritiqueStage(wm=wm, analyzer=analyzer)

    await stage.run({}, ctx=_ctx())

    # The bound the stage ACTUALLY hands the client on the live path, not the constant.
    polled = analyzer.poll_analysis.call_args.kwargs["timeout"]
    assert polled >= _FINAL_CRITIQUE_POLL, (
        f"footage_critique polls the same analyzer for the same whole-video read as "
        f"final_critique but gives it only {polled}s vs {_FINAL_CRITIQUE_POLL}s — the "
        "short bound is what produced a 0-shot 'completed' critique in prod"
    )
    # ...and never above what the client will ACCEPT. `_clamp_timeout` raises above
    # its ceiling instead of capping, so an over-large bound here does not merely fail
    # to help — it makes every analyzer poll blow up before the first GET.
    _clamp_timeout(polled)


# ── Coupling: the raised poll bound must not starve the critic waves ─────────
def test_critic_phase_still_funds_the_batch_waves() -> None:
    """``_batch_phase_seconds`` is ``llm_budget - analyzer_elapsed - headroom``, so an
    analyzer that spends its WHOLE poll bound shrinks the batch-wave slice the most. This
    pins that even that worst case funds a full retry ladder at the largest shot count
    ``footage_direction`` plans (~25).
    """
    from math import ceil

    from src.services.stages.llm_budget import attempt_plan, llm_budget_seconds, llm_retries

    stage = _StubCritiqueStage(wm=AsyncMock(), analyzer=AsyncMock())
    phase = stage._batch_phase_seconds(fc_mod._ANALYSIS_POLL_TIMEOUT_S)
    # The by-construction identity: analyzer worst case + wave phase + the return-path
    # reserve == the outer attempt, with nothing unaccounted for.
    assert (
        fc_mod._ANALYSIS_POLL_TIMEOUT_S + phase + fc_mod._VALIDATION_HEADROOM_S
    ) == pytest.approx(llm_budget_seconds())

    max_planned_shots = 25
    batches = ceil(max_planned_shots / fc_mod._CRITIC_BATCH_SIZE)
    waves = max(1, ceil(batches / fc_mod._CRITIC_BATCH_CONCURRENCY))
    assert waves == 1, "more than one wave would divide the phase again — re-check the slice"

    per_attempt, planned_retries = attempt_plan(phase / waves, llm_retries())
    assert planned_retries == llm_retries(), "the wave slice no longer funds the retry ladder"
    assert per_attempt >= 120.0, f"a {per_attempt:.0f}s critic attempt cannot finish a WM call"
    # ...and it must clear the floor the batch itself enforces, or the worst case would
    # refuse to start every batch rather than merely run them short.
    assert per_attempt >= fc_mod._CRITIC_FLOOR_SECONDS


# ── Apportionment: the analyzer's elapsed must NOT be charged to the batch phase ──
def test_analyzer_elapsed_does_not_shrink_the_batch_phase() -> None:
    """THE bug, stated as arithmetic.

    The phase deadline used to be captured BEFORE the analyzer poll as a constant
    ``llm_budget - _ANALYSIS_POLL_TIMEOUT_S`` (950 - 600 = 350), so the analyzer's real
    elapsed time came off the BATCH PHASE as well as off its own 600s reserve —
    double-charged. Production back-solves it exactly: a logged per-attempt budget of
    23.2s implies a 47.6s wave share, i.e. ~302 of the 350 eaten by the analyzer, and
    48.7% of completed runs scored zero shots.

    Sized from the MEASURED elapsed, a fast analyzer funds a LARGER phase than a slow
    one, and the identity ``elapsed + phase + headroom == outer`` holds at every point.
    """
    from src.services.stages.llm_budget import llm_budget_seconds

    stage = _StubCritiqueStage(wm=AsyncMock(), analyzer=AsyncMock())
    outer = llm_budget_seconds()

    fast = stage._batch_phase_seconds(30.0)
    slow = stage._batch_phase_seconds(302.0)

    # 1. The identity, at both points. Nothing unaccounted for, nothing charged twice.
    assert fast + 30.0 + fc_mod._VALIDATION_HEADROOM_S == pytest.approx(outer)
    assert slow + 302.0 + fc_mod._VALIDATION_HEADROOM_S == pytest.approx(outer)

    # 2. A fast analyzer is REWARDED, not merely not-punished: the seconds it did not
    #    spend go to the waves. Under the old pre-poll constant both were 350.0.
    assert fast > slow, "the analyzer's elapsed is not being returned to the batch phase"
    assert fast - slow == pytest.approx(272.0)

    # 3. The concrete regression: the production case (a 302s analyzer read) must no
    #    longer be capped at the old pre-poll remainder of 350 - 302 = 48s.
    starved_share = outer - fc_mod._ANALYSIS_POLL_TIMEOUT_S - 302.0
    assert starved_share == pytest.approx(48.0), "the guard's own arithmetic drifted"
    assert slow > starved_share * 10


async def test_batch_phase_is_sized_after_the_analyzer_poll() -> None:
    """The apportionment on the LIVE path, not just in the helper.

    A pure-arithmetic assertion on ``_batch_phase_seconds`` cannot see the defect that
    actually shipped: the helper takes the elapsed as an argument, and the CALL SITE is
    what has to measure it. So drive the real ``critic_fn`` with an analyzer that BLOCKS
    for a known time and read the share the batch was actually handed.

    ⚠️ The TOLERANCE is the assertion here, and it is deliberately tight relative to
    ``analyzer_seconds``. Written with a loose band this test is VACUOUS: it passes
    identically whether the call site measures the elapsed or hardcodes zero, because
    the two differ by exactly ``analyzer_seconds``. Verified by mutation — the band must
    stay well under the block.

    (Moving the capture from after the poll to before it, on its own, is an EQUIVALENT
    mutant and nothing can catch it: ``now + (outer - elapsed - h)`` and
    ``entry + (outer - h)`` are the same instant. The defect was never the position —
    it was sizing from the RESERVE CONSTANT instead of the measurement, which is what
    these bounds pin.)
    """
    from src.services.stages.llm_budget import llm_budget_seconds

    analyzer_seconds = 1.0
    seen: list[float] = []
    real_score = fc_mod._score_critic_batch

    async def _spy(**kwargs: Any) -> Any:
        seen.append(float(kwargs["share_seconds"]))
        return await real_score(**kwargs)

    wm = AsyncMock()
    wm.message.return_value = _resp(
        {"shots": [{"shot_id": "clip-1", "score": 5.0}, {"shot_id": "clip-2", "score": 5.0}]}
    )
    analyzer = _analyzer()
    inner_poll = analyzer.poll_analysis

    async def _slow_poll(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(analyzer_seconds)
        return await inner_poll(*args, **kwargs)

    analyzer.poll_analysis = _slow_poll
    stage = _StubCritiqueStage(wm=wm, analyzer=analyzer)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fc_mod, "_score_critic_batch", _spy)
        await stage.run({}, ctx=_ctx())

    assert seen, "no critic batch ran — the spy never saw a share"
    share = seen[0]
    # ONE wave, so share == the whole phase. The analyzer read cost `analyzer_seconds`
    # of the outer attempt, so the phase is everything that was left of it — and the
    # band is a third of the block, so a call site that charged 0s instead of measuring
    # 1s misses it.
    expected = llm_budget_seconds() - analyzer_seconds - fc_mod._VALIDATION_HEADROOM_S
    assert share == pytest.approx(expected, abs=0.3), (
        f"share {share:.2f}s is not what the outer attempt had left after a "
        f"{analyzer_seconds}s analyzer read ({expected:.2f}s) — the call site is not "
        "sizing the phase from the MEASURED elapsed"
    )
    # ...and emphatically NOT bounded by the analyzer's RESERVE, which is the constant
    # that starved 48.7% of completed runs.
    assert share > llm_budget_seconds() - fc_mod._ANALYSIS_POLL_TIMEOUT_S, (
        f"share {share:.1f}s is still capped by the analyzer's 600s reserve — the phase "
        "is being sized from the constant again, not from what the analyzer spent"
    )


# ── The floor: refuse a doomed batch instead of burning two attempts ────────────
async def test_a_batch_below_the_critic_floor_refuses_instead_of_calling_wm() -> None:
    """A per-attempt budget under the MEASURED 34-67s critic window cannot finish, so
    the batch must not start. Production spent 14 days on 23.2s attempts and scored
    zero shots; the refusal costs no WM call and names the arithmetic.
    """
    wm = AsyncMock()
    wm.message.return_value = _resp({"shots": [{"shot_id": "clip-1", "score": 5.0}]})
    loop = asyncio.get_running_loop()

    # A slice that leaves an attempt below the floor but is NOT exhausted, so this
    # exercises the floor and not the pre-existing ``remaining <= 0`` refusal.
    share = fc_mod._CRITIC_FLOOR_SECONDS / 2.0
    shots, failure, usage = await fc_mod._score_critic_batch(
        wm=wm,
        worker_name="footage-critic",
        stage_name="stub_footage_critique",
        message="m",
        batch_number=1,
        limit=asyncio.Semaphore(5),
        deadline=loop.time() + share,
        share_seconds=share,
        retries=1,
    )

    assert shots == []
    assert usage is None
    # NOT started: the whole point is that no WM call is paid for.
    wm.message.assert_not_awaited()
    # A refusal that says WHY. A non-empty reason is what keeps the critique marked
    # unavailable rather than reading as "reviewed, found nothing".
    assert failure is not None
    assert "critic floor" in failure
    assert "not started" in failure


async def test_a_batch_above_the_critic_floor_still_calls_wm() -> None:
    """The floor's other side — it must not refuse a SATISFIABLE batch.

    Without this, the refusal above would pass just as well with the comparison
    inverted, or with the floor set absurdly high.
    """
    wm = AsyncMock()
    wm.message.return_value = _resp({"shots": [{"shot_id": "clip-1", "score": 5.0}]})
    loop = asyncio.get_running_loop()

    share = fc_mod._CRITIC_FLOOR_SECONDS * 6.0
    shots, failure, usage = await fc_mod._score_critic_batch(
        wm=wm,
        worker_name="footage-critic",
        stage_name="stub_footage_critique",
        message="m",
        batch_number=1,
        limit=asyncio.Semaphore(5),
        deadline=loop.time() + share,
        share_seconds=share,
        retries=1,
    )

    assert failure is None, f"a satisfiable batch was refused: {failure}"
    assert shots and usage is not None
    wm.message.assert_awaited()


def test_the_critic_floor_is_the_top_of_the_measured_window() -> None:
    """The floor is a MEASUREMENT, not a taste. Reasoning-off critic calls answered in
    34-67s (``llm_budget.downshift_reasoning``, #873/#883); the floor is the TOP of that
    window, so an attempt below it is unsatisfiable for the slowest observed call.

    Pinned because the failure mode of this fix is a floor nobody can justify: too low
    and it readmits the 23.2s attempts, too high and it refuses the worst-case phase the
    coupling test above proves is fine.
    """
    assert fc_mod._CRITIC_FLOOR_SECONDS == 67.0
    # It must reject what production actually ran...
    assert fc_mod._CRITIC_FLOOR_SECONDS > 23.2
    # ...and admit the worst-case wave slice, or the stage would refuse every batch
    # whenever the analyzer spends its full bound.
    stage = _StubCritiqueStage(wm=AsyncMock(), analyzer=AsyncMock())
    worst = stage._batch_phase_seconds(fc_mod._ANALYSIS_POLL_TIMEOUT_S)
    assert worst > fc_mod._CRITIC_FLOOR_SECONDS


# ── Visibility: the honest degrade must carry a CODE, not just prose ───────────
async def test_an_unavailable_critique_carries_a_coded_degradation() -> None:
    """48.7% of completed runs were invisible because this path raised prose only:
    ``degradations`` was empty, so the DAG executor stamped ``unclassified_note`` with a
    NULL error and no query could count an unreviewed picture.
    """
    from src.services.stages.shared.degradation import (
        FOOTAGE_CRITIQUE_UNAVAILABLE,
        is_degraded,
    )

    wm = AsyncMock()
    # A PERMANENT upstream failure on every batch: no retry ladder, no backoff sleeps.
    wm.message.side_effect = _http_status_error(422, "Model token limit (6000) exceeded")
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer())

    out = await stage.run({}, ctx=_ctx())

    assert out.critique.critique_unavailable is True
    codes = [d["code"] for d in out.degradations]
    assert codes == [FOOTAGE_CRITIQUE_UNAVAILABLE], (
        f"the unreviewed-footage degrade is still uncoded ({codes}) — it lands as "
        "unclassified_note with a NULL error, which is why 48.7% went unnoticed"
    )
    # The detail must name the actual reason, not a generic "quality problem".
    assert out.degradations[0]["detail"] == out.critique.unavailable_reason
    assert out.degradations[0]["detail"]
    assert is_degraded(out.model_dump(mode="json"))


@pytest.mark.parametrize(
    "error",
    ["job failed: worker lost the job"],
)
async def test_an_analyzer_job_failure_ships_and_is_coded_degraded(error: str) -> None:
    """Live run 01a16423-349f's shape on the FOOTAGE critic: the analyzer job is
    orphaned by a rollout (failed / 404). The current artifact still ships, and the
    run is degraded with the coded ``footage_critique_unavailable`` carrying the
    analyzer error — no resubmit (video_analyzer re-claims orphaned jobs itself)."""
    from src.services.stages.shared.degradation import (
        FOOTAGE_CRITIQUE_UNAVAILABLE,
        is_degraded,
    )
    from src.services.stages.upstream import PermanentUpstreamError

    analyzer = _analyzer()
    analyzer.poll_analysis.side_effect = PermanentUpstreamError(f"video_analyzer {error}")
    out = await _StubCritiqueStage(wm=AsyncMock(), analyzer=analyzer).run({}, ctx=_ctx())

    assert out.artifact_url
    assert out.critique.critique_unavailable is True
    assert [d["code"] for d in out.degradations] == [FOOTAGE_CRITIQUE_UNAVAILABLE]
    assert error.split(":")[0] in out.degradations[0]["detail"]
    assert is_degraded(out.model_dump(mode="json"))
    assert analyzer.start_analysis.await_count == 1, "never an orchestrator-side resubmit"


async def test_the_loop_error_fallback_also_carries_the_coded_degradation() -> None:
    """The SECOND emission site. ``handle`` codes the marker on the normal exit, but the
    ``_accept_current(reason="loop_error")`` fallback builds its own output — so a code
    added only to ``handle`` leaves the truncated-response path emitting prose alone.

    Found by mutation: deleting the ``degradations=`` argument from ``_accept_current``
    left the whole module GREEN, which means this branch had no assertion on it at all.
    """
    from src.services.stages.shared.degradation import FOOTAGE_CRITIQUE_UNAVAILABLE

    wm = AsyncMock()
    # A decode error escapes ``critic_fn``'s narrow catches by design ⇒ loop_error.
    wm.message.side_effect = json.JSONDecodeError(
        "Unterminated string starting at", '{"shots": [{"shot_id": "clip-1", "issue": "the', 30
    )
    out = await _StubCritiqueStage(wm=wm, analyzer=_analyzer()).run({}, ctx=_ctx())

    assert out.critique.reason == "loop_error"
    assert out.critique.critique_unavailable is True
    assert [d["code"] for d in out.degradations] == [FOOTAGE_CRITIQUE_UNAVAILABLE]
    assert out.degradations[0]["detail"] == out.critique.unavailable_reason


async def test_an_honest_no_op_carries_no_degradation_either() -> None:
    """The OTHER ``_accept_current`` callers — nothing was directed to critique, so
    nothing was un-reviewed. ``unavailable_reason`` is the discriminator, and it must
    keep these paths clean or the code fires on runs with no footage direction at all.
    """
    stage = _StubCritiqueStage(wm=AsyncMock(), analyzer=_analyzer(), plans=[], windows=[])
    out = await stage.run({}, ctx=_ctx())

    assert out.critique.reason == "no_directed_footage"
    assert out.critique.critique_unavailable is False
    assert out.degradations == []


async def test_a_reviewed_critique_carries_no_degradation() -> None:
    """The other side of the code, or it would fire on every clean run — the
    near-universal-flag failure ``degradation.py`` exists to avoid.
    """
    wm = AsyncMock()
    wm.message.return_value = _resp(
        {"shots": [{"shot_id": "clip-1", "score": 5.0}, {"shot_id": "clip-2", "score": 5.0}]}
    )
    out = await _StubCritiqueStage(wm=wm, analyzer=_analyzer()).run({}, ctx=_ctx())
    assert out.critique.critique_unavailable is False
    assert out.degradations == []
    assert out.notes == []


# ── Below the bar: a coded degradation that tells FAILED from DID-NOT-HELP ─────
# Run 01a0ab23 (showcase_v1, 30s order) shipped footage scored 2.0 against the 4.0 bar
# with a prose note only. Worse, a FAILED re-render hands back the same draft, so it
# also stops as ``plateau`` — the two causes were indistinguishable.
_LOW_TWICE = {
    "shots": [
        {"shot_id": "clip-1", "score": 2.0, "framing": 2.0},
        {"shot_id": "clip-2", "score": 5.0},
    ]
}


class _FailingRerenderStage(_StubCritiqueStage):
    async def rerender(self, ctx: RunContext, plans: Any, pass_idx: int) -> tuple[str, float]:
        self.rerender_calls.append(list(plans))
        raise RuntimeError("compile job lost")


async def test_below_bar_after_a_real_rerender_is_coded_as_rendered() -> None:
    from src.services.stages.shared.degradation import FOOTAGE_BELOW_BAR, is_degraded

    wm = AsyncMock()
    wm.message.side_effect = [_resp(_LOW_TWICE), _resp(_LOW_TWICE)]
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer())
    out = await stage.run({}, ctx=_ctx())

    assert len(stage.rerender_calls) == 1
    assert out.critique.reason == "plateau"
    assert out.critique.rerender_outcome == fc_mod.RERENDER_RENDERED
    assert [d["code"] for d in out.degradations] == [FOOTAGE_BELOW_BAR]
    detail = out.degradations[0]["detail"]
    assert "scored 2.0" in detail and "bar of 4.0" in detail
    assert "stop=plateau" in detail and "rerender=rendered" in detail
    assert is_degraded(out.model_dump(mode="json"))
    # Never another paid pass: the cap is the cost control.
    assert wm.message.call_count == 2


async def test_below_bar_after_a_failed_rerender_is_distinguishable() -> None:
    from src.services.stages.shared.degradation import FOOTAGE_BELOW_BAR

    wm = AsyncMock()
    wm.message.side_effect = [_resp(_LOW_TWICE), _resp(_LOW_TWICE)]
    stage = _FailingRerenderStage(wm=wm, analyzer=_analyzer())
    out = await stage.run({}, ctx=_ctx())  # must NOT raise

    assert len(stage.rerender_calls) == 1
    assert out.artifact_url == "http://v/orig.mp4"
    # The loop's own reason is still "plateau" — that is exactly the conflation.
    assert out.critique.reason == "plateau"
    assert out.critique.rerender_outcome == fc_mod.RERENDER_FAILED
    assert [d["code"] for d in out.degradations] == [FOOTAGE_BELOW_BAR]
    assert "rerender=failed" in out.degradations[0]["detail"]
    assert "FAILED" in out.degradations[0]["detail"]


_CUT_ONLY_LOW = {
    "shots": [
        {"shot_id": "clip-1", "score": 3.0, "boring_cut": 2.0, "framing": 4.5},
        {"shot_id": "clip-2", "score": 5.0},
    ]
}


async def test_a_cut_only_shortfall_skips_the_rerender_that_cannot_help() -> None:
    """``boring_cut``'s only lever is a hard-cut transition: it cannot change the pixels
    the critic scored, so the min-across-shots score cannot move. No second render, no
    second critic spend, and the below-bar detail says why.
    """
    from src.services.stages.shared.degradation import FOOTAGE_BELOW_BAR

    wm = AsyncMock()
    wm.message.side_effect = [_resp(_CUT_ONLY_LOW), _resp(_CUT_ONLY_LOW)]
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer())
    out = await stage.run({}, ctx=_ctx())

    assert stage.rerender_calls == []
    assert wm.message.call_count == 1, "the skipped pass must not re-pay the critic"
    assert out.artifact_url == "http://v/orig.mp4"
    assert out.critique.rerender_outcome == fc_mod.RERENDER_SKIPPED_CUT_ONLY
    assert [d["code"] for d in out.degradations] == [FOOTAGE_BELOW_BAR]
    detail = out.degradations[0]["detail"]
    assert "rerender=skipped_cut_only" in detail and "boring_cut" in detail


async def test_a_cut_shortfall_with_another_weak_dimension_still_rerenders() -> None:
    mixed = {
        "shots": [
            {"shot_id": "clip-1", "score": 3.0, "boring_cut": 2.0},
            {"shot_id": "clip-2", "score": 3.0, "framing": 2.0},
        ]
    }
    wm = AsyncMock()
    wm.message.side_effect = [_resp(mixed), _resp(mixed)]
    stage = _StubCritiqueStage(wm=wm, analyzer=_analyzer())
    out = await stage.run({}, ctx=_ctx())

    assert len(stage.rerender_calls) == 1
    assert out.critique.rerender_outcome == fc_mod.RERENDER_RENDERED


async def test_footage_that_meets_the_bar_carries_no_below_bar_code() -> None:
    wm = AsyncMock()
    wm.message.return_value = _resp(
        {"shots": [{"shot_id": "clip-1", "score": 5.0}, {"shot_id": "clip-2", "score": 4.5}]}
    )
    out = await _StubCritiqueStage(wm=wm, analyzer=_analyzer()).run({}, ctx=_ctx())

    assert out.critique.converged is True
    assert out.critique.rerender_outcome == fc_mod.RERENDER_NOT_ATTEMPTED
    assert out.degradations == []
    assert out.notes == []
