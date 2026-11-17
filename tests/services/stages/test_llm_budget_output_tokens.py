"""``output_token_budget`` — the reasoning-aware WM output budget.

Guards the CORRECTION to PR #851. That PR read the WM 422 ("Model token limit
(N) exceeded before any response was generated") as a pre-generation reservation
against a small context window, and therefore LOWERED ``max_tokens`` to 1760 on
``segment_select``. The raise is actually post-generation
(``pydantic_ai/_agent_graph.py:1100-1113``): ``finish_reason == 'length'`` with
every part a ``ThinkingPart``, i.e. the reasoning model spent the whole budget
thinking and never emitted the answer.

Measured live 2026-08-19, one worker, one prompt, only ``max_tokens`` varying:
500 → 422 (43.5 s); **1760 → 422 (131.5 s)**; 6000 → no 422; 10000 → no 422.

So these tests assert a FLOOR. A budget sized to the answer alone is not
conservative, it is the bug.
"""

from __future__ import annotations

import pytest

from src.config import LLM_BUDGET_SECONDS
from src.services.stages.llm_budget import (
    _CHAIN_DEADLINE_TOKENS,
    _MEASURED_TOKENS_PER_SEC,
    _REASONING_HEADROOM_TOKENS,
    _WM_SAFE_MAX_TOKENS,
    LARGE_ANSWER_TOKENS,
    MEDIUM_ANSWER_TOKENS,
    MIN_OUTPUT_TOKEN_BUDGET,
    SMALL_ANSWER_TOKENS,
    attempt_plan,
    output_token_budget,
)

# worker_management's DEFAULT_SELF_HOSTED_MAX_TOKENS. Passing LESS than this is
# strictly worse than passing nothing at all — it lowers the ceiling the model
# has to finish reasoning under. Mirrored, not imported: worker_management is a
# separate service and this repo must not depend on its internals.
_WM_DEFAULT = 6000

# The exact value PR #851 shipped, kept as a named regression witness.
_PR851_VALUE = 2 * 8 * 110


def test_the_floor_is_strictly_above_wms_own_default() -> None:
    """STRICTLY above, not >=. Equality is the refuted design.

    6000 is what a caller gets by omitting ``max_tokens`` entirely, and it is an
    OBSERVED production failure on six different workers (promo_script,
    promo_analyze, promo_image_prompts, website_review_script, growth_bundle,
    retention_plan, segment_select; newest 2026-08-19 08:28Z). A floor set TO
    WM's default is a gate that passes fleet-wide while every one of those keeps
    failing — green and inert, which is worse than no gate at all because it
    reads as coverage.
    """
    assert MIN_OUTPUT_TOKEN_BUDGET > _WM_DEFAULT, (
        f"floor {MIN_OUTPUT_TOKEN_BUDGET} is not above WM's own default "
        f"{_WM_DEFAULT}, which is itself a measured production failure"
    )


@pytest.mark.parametrize("answer_tokens", [1])
def test_every_budget_clears_the_floor(answer_tokens: int) -> None:
    assert output_token_budget(answer_tokens=answer_tokens) >= MIN_OUTPUT_TOKEN_BUDGET


def test_the_pr851_answer_size_no_longer_produces_the_pr851_budget() -> None:
    """THE regression. 1760 was measured to 422 in production."""
    budget = output_token_budget(answer_tokens=_PR851_VALUE)
    assert budget != _PR851_VALUE
    assert budget > _PR851_VALUE
    assert budget >= _WM_DEFAULT


def test_reasoning_headroom_is_added_on_top_of_the_answer() -> None:
    """The answer is not the budget — thinking is charged against it too.

    Additivity is only OBSERVABLE where neither bound binds, so the probe is
    derived from the bounds rather than hardcoded. Two ways to measure the wrong
    thing here, both of which this test has to dodge:

    - Probe above ``_CHAIN_DEADLINE_TOKENS - headroom`` and both values clamp to
      the cap; the test then measures the clamp while claiming to measure headroom.
    - Probe below ``MIN_OUTPUT_TOKEN_BUDGET - headroom`` and both clamp to the
      FLOOR. That is what the original ``lo, hi = 100, 700`` did once the floor
      moved to 8000: it went red reporting ``8000 - 8000 == 700 - 100``, which is
      the test correctly refusing to certify a property it could no longer see.
    """
    lo = MIN_OUTPUT_TOKEN_BUDGET - _REASONING_HEADROOM_TOKENS + 100
    hi = _CHAIN_DEADLINE_TOKENS - _REASONING_HEADROOM_TOKENS - 100
    assert lo < hi, (
        f"no window where headroom is observable: floor={MIN_OUTPUT_TOKEN_BUDGET}, "
        f"cap={_CHAIN_DEADLINE_TOKENS}, headroom={_REASONING_HEADROOM_TOKENS}. "
        "If this fires, the ladder has been squeezed to nothing and the fix is "
        "wall clock, not tokens."
    )
    small = output_token_budget(answer_tokens=lo)
    large = output_token_budget(answer_tokens=hi)
    assert MIN_OUTPUT_TOKEN_BUDGET < small < _CHAIN_DEADLINE_TOKENS
    assert MIN_OUTPUT_TOKEN_BUDGET < large < _CHAIN_DEADLINE_TOKENS
    assert large - small == hi - lo, "headroom must be additive, not a replacement"
    assert small > lo and large > hi


def test_the_token_ceiling_tracks_the_code_default_clock() -> None:
    """The ceiling is DERIVED from the wall clock — this is what makes that true.

    ``llm_budget.py`` tells the reader the ceiling cannot go stale because it is
    computed rather than written down, and names THIS test as the reason to
    believe it. Before this existed the claim was unbacked: the only nearby
    assertion compared ``output_token_budget(...)`` to ``_CHAIN_DEADLINE_TOKENS``,
    i.e. the module constant against itself, which holds for any value of either
    and cannot see a desync.

    Two distinct properties, both load-bearing:

    1. The ceiling equals ``clock x rate``. Three PRs in three days each pinned
       their own ceiling (1760, 7760, 6800) from an input that was stale within a
       day; this is the assertion that stops a fourth.
    2. The clock is ``src.config.LLM_BUDGET_SECONDS`` — the CODE DEFAULT, read at
       import — not ``get_settings().pipeline.llm_budget_seconds``. That
       distinction is why ``PIPELINE_LLM_BUDGET_SECONDS`` is deliberately unset in
       GitOps: setting it there would move the wall clock and leave this ceiling
       frozen. If someone ever does set it, this test still passes (it reads the
       same default) — so the guard against THAT is the comment plus the
       assertion below that the two are equal today.
    """
    assert int(LLM_BUDGET_SECONDS * _MEASURED_TOKENS_PER_SEC) == _CHAIN_DEADLINE_TOKENS

    # And the settings-resolved clock must still agree with the code default, so
    # a GitOps override lands as a test failure here rather than as a silent
    # desync between the wall clock and the token ceiling it is supposed to size.
    from src.config import get_settings

    assert get_settings().pipeline.llm_budget_seconds == LLM_BUDGET_SECONDS, (
        "PIPELINE_LLM_BUDGET_SECONDS has been overridden. The token ceiling is "
        "derived from the CODE DEFAULT at import and will not follow it — update "
        "src/config.py's LLM_BUDGET_SECONDS in the same commit, or stop overriding."
    )


def test_the_budget_is_capped_by_the_router_chain_deadline() -> None:
    """Above what the per-call wall clock can deliver, more budget buys nothing.

    The cap is DERIVED from ``LLM_BUDGET_SECONDS x _MEASURED_TOKENS_PER_SEC``, not
    from the "300 s at ~37 tok/s" this docstring used to cite — that rate was
    measured on an idle box running a model since retired, and the 300 s router
    deadline is overridden to 1200 in the cluster.
    """
    assert output_token_budget(answer_tokens=1_000_000) == _CHAIN_DEADLINE_TOKENS
    assert _CHAIN_DEADLINE_TOKENS <= _WM_SAFE_MAX_TOKENS


def test_the_budget_never_exceeds_wms_silent_clamp() -> None:
    """WM clamps to SAFE_MAX_TOKENS without telling us; never rely on that."""
    for answer in (1, 5_000, 50_000):
        assert output_token_budget(answer_tokens=answer) <= _WM_SAFE_MAX_TOKENS


@pytest.mark.parametrize("bad", [0])
def test_a_nonpositive_answer_estimate_is_a_programming_error(bad: int) -> None:
    """Fail loud rather than quietly returning the floor for a caller that
    forgot to compute its answer size."""
    with pytest.raises(ValueError, match="answer_tokens must be positive"):
        output_token_budget(answer_tokens=bad)


def test_the_named_tiers_are_ordered_and_all_land_in_the_useful_range() -> None:
    assert SMALL_ANSWER_TOKENS < MEDIUM_ANSWER_TOKENS < LARGE_ANSWER_TOKENS
    budgets = [
        output_token_budget(answer_tokens=t)
        for t in (SMALL_ANSWER_TOKENS, MEDIUM_ANSWER_TOKENS, LARGE_ANSWER_TOKENS)
    ]
    assert budgets == sorted(budgets), "a bigger answer must not buy a smaller budget"
    assert all(MIN_OUTPUT_TOKEN_BUDGET <= b <= _CHAIN_DEADLINE_TOKENS for b in budgets)


def test_the_cap_can_never_fall_below_the_floor() -> None:
    """The one invariant that must survive every future re-tuning.

    The floor and the cap are derived from DIFFERENT things — the floor from
    worker-management's own default (the budget observed to clear real
    reasoning), the cap from a measured token rate against WM's 600 s
    ``agent_run_timeout_s``. Nothing structurally stops someone lowering the cap,
    on a slower measurement, to below the floor. If that ever happened,
    ``output_token_budget`` would return the floor while the comment claimed a
    cap, and every call would be asking for more than WM can deliver — silently.

    The window is currently ~800 tokens wide, so this is not a theoretical
    margin.
    """
    assert _CHAIN_DEADLINE_TOKENS >= MIN_OUTPUT_TOKEN_BUDGET, (
        f"cap {_CHAIN_DEADLINE_TOKENS} is below floor {MIN_OUTPUT_TOKEN_BUDGET} — "
        "re-derive both together, and re-measure the token rate before doing so"
    )


def test_every_budget_lands_inside_the_measured_window() -> None:
    """No caller can produce a budget outside what the box can actually serve."""
    for answer in (1, 100, 500, 1760, 2000, 5000, 9900, 100_000):
        budget = output_token_budget(answer_tokens=answer)
        assert MIN_OUTPUT_TOKEN_BUDGET <= budget <= _CHAIN_DEADLINE_TOKENS


# ── attempt_plan: a retry ladder that fits INSIDE the slice it was given ──────
class TestAttemptPlan:
    """``stage_call``'s ``timeout_seconds`` is PER ATTEMPT, so handing it a slice
    carved out of a SHARED deadline authorises 3x that slice. ``attempt_plan`` is
    what makes the ladder fit inside the slice instead."""

    def test_the_ladder_fits_inside_the_slice(self) -> None:
        per_attempt, retries = attempt_plan(300.0, 2)
        assert retries == 2
        assert per_attempt * (retries + 1) <= 300.0

    def test_a_slice_too_small_to_split_buys_fewer_attempts_not_shorter_ones(self) -> None:
        """3 attempts of 2s cannot complete a WM round-trip; 1 attempt of 6s can.
        The retry count is what gives way, never the per-attempt budget."""
        per_attempt, retries = attempt_plan(6.0, 2)
        assert retries == 0
        assert per_attempt == 6.0

    def test_never_returns_below_stage_calls_own_minimum(self) -> None:
        """``stage_call`` raises ValueError under 1.0s — a sub-second slice must
        still produce a legal call (the caller's own ``asyncio.timeout`` bounds it)."""
        per_attempt, retries = attempt_plan(0.4, 3)
        assert retries == 0
        assert per_attempt >= 1.0
