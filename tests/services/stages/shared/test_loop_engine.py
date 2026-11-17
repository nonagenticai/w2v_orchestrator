"""Tests for the shared evaluator-optimizer loop engine (aspect A10).

Fakes only — no live WM, no network. Critics/optimizers are plain async
closures; ``timeout_seconds`` + ``retries`` are pinned per call so the tests do
not depend on pipeline settings and never sleep on a real backoff.
"""

from __future__ import annotations

import pytest

from src.services.stages.errors import StageSkippedError
from src.services.stages.shared.loop_engine import (
    CRITIC_UNAVAILABLE,
    ON_LOOP_ERROR_KEEP_BEST,
    Critique,
    LoopResult,
    run_evaluator_optimizer,
)

# Pin the stage_call budget so tests are hermetic + fast (no settings, no retry).


def _scripted_critic(scores: list[float], *, rerun: list[str] | None = None):
    """A critic that returns ``scores`` in call order (last score repeats)."""
    calls = {"n": 0}

    async def critic(_draft: object) -> Critique:
        i = min(calls["n"], len(scores) - 1)
        calls["n"] += 1
        return Critique(score=scores[i], rerun_stages=list(rerun or []))

    return critic


async def test_converges_early_optimizer_never_called() -> None:
    """Initial draft already meets threshold → one pass, threshold, no optimize."""
    optimize_calls = {"n": 0}

    async def optimizer(draft: str, _c: Critique) -> str:
        optimize_calls["n"] += 1
        return draft + "!"

    result = await run_evaluator_optimizer(
        "draft-0",
        critic_fn=_scripted_critic([5.0]),
        optimizer_fn=optimizer,
        threshold=4.0,
        timeout_seconds=30.0,
        retries=0,
    )

    assert result.passes == 1
    assert result.reason == "threshold"
    assert result.converged is True
    assert result.score == 5.0
    assert result.draft == "draft-0"
    assert optimize_calls["n"] == 0
    assert result.history == [(5.0, "draft-0")]


async def test_plateau_delta_below_epsilon() -> None:
    """3.0 → 3.1 with Δ < ε stops on plateau after exactly two passes."""

    async def optimizer(draft: str, _c: Critique) -> str:
        return draft + "*"

    result = await run_evaluator_optimizer(
        "d",
        critic_fn=_scripted_critic([3.0, 3.1]),
        optimizer_fn=optimizer,
        threshold=4.0,
        epsilon=0.25,
        timeout_seconds=30.0,
        retries=0,
    )

    assert result.passes == 2
    assert result.reason == "plateau"
    assert result.converged is False
    assert result.score == 3.1  # 3.1 > 3.0, best is the pass-2 draft
    assert result.draft == "d*"


async def test_max_passes_caps_at_two() -> None:
    """Rising-but-below-threshold with Δ >= ε stops at the 2-pass budget."""

    async def optimizer(draft: str, _c: Critique) -> str:
        return draft + "+"

    result = await run_evaluator_optimizer(
        "d",
        critic_fn=_scripted_critic([2.0, 3.0]),  # Δ = 1.0 >= ε, both < threshold
        optimizer_fn=optimizer,
        threshold=4.0,
        epsilon=0.25,
        max_passes=2,
        timeout_seconds=30.0,
        retries=0,
    )

    assert result.passes == 2
    assert result.reason == "max_passes"
    assert result.converged is False


async def test_max_passes_nine_still_clamps_to_two() -> None:
    """max_passes=9 is hard-clamped to 2 — the critic is called at most twice."""
    critic = _scripted_critic([1.0, 2.0, 3.0, 4.5, 5.0])
    scored: list[float] = []

    async def counting_critic(draft: object) -> Critique:
        c = await critic(draft)
        scored.append(c.score)
        return c

    async def optimizer(draft: str, _c: Critique) -> str:
        return draft + "+"

    result = await run_evaluator_optimizer(
        "d",
        critic_fn=counting_critic,
        optimizer_fn=optimizer,
        threshold=99.0,  # never reachable, force the cap
        epsilon=0.0,  # never plateau
        max_passes=9,
        timeout_seconds=30.0,
        retries=0,
    )

    assert result.passes == 2
    assert result.reason == "max_passes"
    assert len(scored) == 2  # only two critic calls despite max_passes=9


async def test_best_draft_retained_on_regression() -> None:
    """Pass 2 regresses 4.2 → 3.1 → the pass-1 draft is returned."""

    async def optimizer(_draft: str, _c: Critique) -> str:
        return "worse-draft"

    result = await run_evaluator_optimizer(
        "good-draft",
        critic_fn=_scripted_critic([4.2, 3.1]),
        optimizer_fn=optimizer,
        threshold=5.0,  # 4.2 does not converge → a second pass runs
        epsilon=0.25,
        timeout_seconds=30.0,
        retries=0,
    )

    assert result.passes == 2
    assert result.draft == "good-draft"  # regression discarded
    assert result.score == 4.2
    assert result.reason == "plateau"  # 3.1 - 4.2 = -1.1 < ε
    # History records both passes in execution order.
    assert result.history == [(4.2, "good-draft"), (3.1, "worse-draft")]


async def test_no_critic_is_identity_noop_zero_awaits() -> None:
    """critic_fn=None → same object identity, passes=0, and no awaits run.

    The optimizer raises if ever awaited; a live critic/optimizer would also
    schedule an await. Neither fires.
    """
    sentinel = object()
    optimizer_called = {"n": 0}

    async def optimizer(_draft: object, _c: Critique) -> object:
        optimizer_called["n"] += 1
        raise AssertionError("optimizer must not run on the no-critic path")

    result: LoopResult[object] = await run_evaluator_optimizer(
        sentinel,
        critic_fn=None,
        optimizer_fn=optimizer,
        timeout_seconds=30.0,
        retries=0,
    )

    assert result.draft is sentinel  # same object identity, no copy
    assert result.passes == 0
    assert result.converged is False
    assert result.reason == "no_critic"
    assert result.score == 0.0
    assert result.rerun_stages == []
    assert result.history == []
    assert optimizer_called["n"] == 0


async def test_spine_id_guard_raises_when_optimizer_drops_id() -> None:
    """id_extractor guards C0: an optimizer that drops a spine id raises."""

    async def optimizer(_draft: frozenset[str], _c: Critique) -> frozenset[str]:
        return frozenset({"a"})  # dropped "b"

    with pytest.raises(ValueError, match="spine id-set"):
        await run_evaluator_optimizer(
            frozenset({"a", "b"}),
            critic_fn=_scripted_critic([2.0, 3.0]),
            optimizer_fn=optimizer,
            threshold=9.0,  # force a second pass so the optimizer runs
            epsilon=0.0,
            id_extractor=lambda d: frozenset(d),
            timeout_seconds=30.0,
            retries=0,
        )


# ---------------------------------------------------------------------------
# on_loop_error — an ADVISORY critic must not be able to kill a finished run.
#
# Regression cover for the measured production kills: over 218 `retention_plan`
# executions, 4 died because `retention-critic` returned nothing and the
# exception escaped the loop. The worst of them (`showcase_v1`, 2026-08-10) had
# already completed `demo_video_record` — a ~20-stage run destroyed by an
# advisory opinion the stage's own docstring says never blocks the deliverable.
# ---------------------------------------------------------------------------


class _CriticExplodedError(RuntimeError):
    """Stand-in for the upstream 500/422 the WM critic call raises."""


async def _boom_critic(_draft: object) -> Critique:
    raise _CriticExplodedError("upstream 500: token limit exceeded before any response")


async def _identity_optimizer(draft: str, _c: Critique) -> str:
    return draft


async def test_default_still_raises_on_critic_failure() -> None:
    """Negative control: the historical contract is unchanged without the opt-in."""
    with pytest.raises(_CriticExplodedError):
        await run_evaluator_optimizer(
            "draft-0",
            critic_fn=_boom_critic,
            optimizer_fn=_identity_optimizer,
            timeout_seconds=30.0,
            retries=0,
        )


async def test_keep_best_degrades_to_initial_when_first_critic_dies() -> None:
    """Pass-1 critic failure keeps the INPUT draft instead of failing the stage."""
    result = await run_evaluator_optimizer(
        "draft-0",
        critic_fn=_boom_critic,
        optimizer_fn=_identity_optimizer,
        timeout_seconds=30.0,
        retries=0,
        on_loop_error=ON_LOOP_ERROR_KEEP_BEST,
    )

    assert result.draft == "draft-0"
    assert result.reason == CRITIC_UNAVAILABLE
    assert result.converged is False
    assert result.passes == 0
    assert result.history == []


async def test_keep_best_retains_the_pass_one_improvement() -> None:
    """A pass-2 failure still ships the improvement pass 1 earned — not `initial`."""
    calls = {"n": 0}

    async def flaky_critic(_draft: str) -> Critique:
        calls["n"] += 1
        if calls["n"] == 1:
            return Critique(score=2.0, rerun_stages=["demo_video_gen"])
        raise _CriticExplodedError("second call returns nothing")

    async def optimizer(draft: str, _c: Critique) -> str:
        return draft + "-improved"

    result = await run_evaluator_optimizer(
        "draft-0",
        critic_fn=flaky_critic,
        optimizer_fn=optimizer,
        threshold=4.0,
        max_passes=2,
        timeout_seconds=30.0,
        retries=0,
        on_loop_error=ON_LOOP_ERROR_KEEP_BEST,
    )

    # Pass 1 scored `initial` at 2.0; the pass-2 re-score died. 2.0 is still the
    # best SCORED draft, so `initial` is what ships — and it ships honestly.
    assert result.draft == "draft-0"
    assert result.score == 2.0
    assert result.passes == 1
    assert result.reason == CRITIC_UNAVAILABLE
    assert result.rerun_stages == ["demo_video_gen"]


@pytest.mark.parametrize(
    "control_flow_exc",
    [StageSkippedError("opted out")],
)
async def test_keep_best_never_swallows_control_flow(
    control_flow_exc: Exception,
) -> None:
    """Skip/precondition are run-engine signals, not failures — they must escape."""

    async def signalling_critic(_draft: object) -> Critique:
        raise control_flow_exc

    with pytest.raises(type(control_flow_exc)):
        await run_evaluator_optimizer(
            "draft-0",
            critic_fn=signalling_critic,
            optimizer_fn=_identity_optimizer,
            timeout_seconds=30.0,
            retries=0,
            on_loop_error=ON_LOOP_ERROR_KEEP_BEST,
        )


async def test_keep_best_logs_at_error_with_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Degraded is never silent: ERROR + traceback, naming the stage."""
    with caplog.at_level("ERROR"):
        await run_evaluator_optimizer(
            "draft-0",
            critic_fn=_boom_critic,
            optimizer_fn=_identity_optimizer,
            timeout_seconds=30.0,
            retries=0,
            stage_name="retention_plan",
            on_loop_error=ON_LOOP_ERROR_KEEP_BEST,
        )

    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert errors, "a swallowed loop failure must still be logged at ERROR"
    assert "retention_plan" in errors[0].getMessage()
    assert errors[0].exc_info is not None, "traceback must be attached"


async def test_keep_best_c0_violation_ships_the_pre_optimizer_draft() -> None:
    """A C0-violating optimizer costs the run its enrichment, never its id-set."""

    async def id_dropping_optimizer(_draft: list[str], _c: Critique) -> list[str]:
        return ["a"]  # dropped "b" — the exact C0 violation the guard exists for

    result = await run_evaluator_optimizer(
        ["a", "b"],
        critic_fn=_scripted_critic([2.0]),  # below threshold ⇒ the loop optimizes
        optimizer_fn=id_dropping_optimizer,
        threshold=4.0,
        max_passes=2,
        id_extractor=lambda ls: frozenset(ls),
        timeout_seconds=30.0,
        retries=0,
        on_loop_error=ON_LOOP_ERROR_KEEP_BEST,
    )

    # The guard still wins: what ships is the draft from BEFORE the bad pass.
    assert result.draft == ["a", "b"]
    assert result.reason == CRITIC_UNAVAILABLE
