"""Unit tests for RetentionPlanStage — WM gateway mocked (no live LLM / DB).

The ``retention-critic`` and ``hook-doctor`` WM workers are an ``AsyncMock`` whose
``message`` returns a :class:`WMResponse` keyed on ``worker_name``; the
deterministic :func:`analyze_retention` evidence pass runs for real (it is pure).
Mirrors the injection pattern of ``tests/services/stages/shared/test_script_doctor.py``.

Contracts pinned here:
* presence-opt-in — absent ``direction.retention`` (or ``mode == "off"``) skips
  before any WM traffic and leaves the spine untouched;
* C0 — a critic + optimizer that rewrite text (and hallucinate an unknown id)
  never change the spine id-set or order, only text for echoed known ids;
* critique_only — the critic runs but no rewrite is applied;
* active happy-path — a below→above-threshold critic drives the loop to converge,
  applying rewrites while ``id`` / ``clip`` stay intact.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.base import StageSkippedError
from src.services.stages.demo_video.retention_plan import (
    _CRITIC_MAX_TOKENS,
    RetentionPlanStage,
)
from src.services.stages.llm_budget import MIN_OUTPUT_TOKEN_BUDGET


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _wm_fake(
    critic_outputs: list[dict[str, Any]] | dict[str, Any],
    optimizer_lines: list[dict[str, Any]],
) -> AsyncMock:
    """AsyncMock WM: retention-critic → critique dict(s); hook-doctor → rewrites.

    ``critic_outputs`` may be a single dict (returned every critic call) or a list
    consumed one-per-call so a test can score below-then-above threshold.
    """
    fake = AsyncMock()
    critic_queue = list(critic_outputs) if isinstance(critic_outputs, list) else None
    critic_single = critic_outputs if isinstance(critic_outputs, dict) else None

    def _side(worker_name: str | None = None, **_: Any) -> WMResponse:
        if worker_name == "retention-critic":
            if critic_queue is not None:
                # Repeat the final verdict once the queue is drained.
                return _resp(critic_queue.pop(0) if len(critic_queue) > 1 else critic_queue[0])
            return _resp(critic_single)
        return _resp({"lines": optimizer_lines})

    fake.message.side_effect = _side
    return fake


def _ctx(
    *,
    retention: dict[str, Any] | None,
    prior_lines: list[dict[str, Any]] | None = None,
    target_seconds: float | None = None,
    retention_weights: dict[str, Any] | None = None,
) -> RunContext:
    cfg: dict[str, Any] = {}
    if retention is not None:
        cfg["direction"] = {"retention": retention}
    if retention_weights is not None:
        cfg["retention_weights"] = retention_weights
    if target_seconds is not None:
        cfg["deliverable"] = {"target_duration_seconds": target_seconds}
    prior = {
        "demo_video_gen": {
            "lines": prior_lines or [],
            "bdd_job_id": "",
            "features_generated": 0,
            "package_url": "",
        }
    }
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=cfg,
        prior_run_outputs=prior,
    )


def _lines() -> list[dict[str, Any]]:
    return [
        {"id": "l1", "text": "Welcome to the app.", "seconds": 3.0, "clip": None},
        {"id": "l2", "text": "Here is the dashboard.", "seconds": 4.0, "clip": "clip-2"},
        {"id": "l3", "text": "Thanks for watching.", "seconds": 3.0, "clip": "clip-3"},
    ]


# ── (a) presence-opt-in: absent block skips before any WM call ────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("retention", [None])
async def test_absent_block_skips_without_wm(retention: dict[str, Any] | None) -> None:
    fake = _wm_fake({"overall_score": 5, "directives": []}, [])
    stage = RetentionPlanStage(client=fake)
    ctx = _ctx(retention=retention, prior_lines=_lines())

    with pytest.raises(StageSkippedError):
        await stage.handle(stage.input_model(), ctx)

    fake.message.assert_not_called()


# ── (b) mode == "off" → skip ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mode_off_skips_without_wm() -> None:
    fake = _wm_fake({"overall_score": 5, "directives": []}, [])
    stage = RetentionPlanStage(client=fake)
    ctx = _ctx(retention={"mode": "off"}, prior_lines=_lines())

    with pytest.raises(StageSkippedError):
        await stage.handle(stage.input_model(), ctx)

    fake.message.assert_not_called()


# ── (b2) no prior lines → skip ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_lines_skips() -> None:
    fake = _wm_fake({"overall_score": 2, "directives": []}, [])
    stage = RetentionPlanStage(client=fake)
    ctx = _ctx(retention={"mode": "auto"}, prior_lines=[])

    with pytest.raises(StageSkippedError):
        await stage.handle(stage.input_model(), ctx)

    fake.message.assert_not_called()


# ── (c) C0: rewrite text for some ids, hallucinated id ignored, order intact ──


@pytest.mark.asyncio
async def test_c0_id_preservation_and_hallucination_dropped() -> None:
    # Critic scores below threshold and flags l1 (via directive) — then, once l1's
    # text changed, scores above threshold so the loop converges.
    critic_low = {
        "overall_score": 2,
        "directives": [{"op": "rewrite", "target_ids": ["l1"]}],
    }
    critic_high = {"overall_score": 5, "directives": []}
    fake = _wm_fake(
        [critic_low, critic_high],
        # hook-doctor rewrites l1 AND hallucinates an unknown id l999.
        [
            {"id": "l1", "text": "Stop scrolling — this changes everything."},
            {"id": "l999", "text": "HALLUCINATED"},
        ],
    )
    stage = RetentionPlanStage(client=fake)
    ctx = _ctx(retention={"mode": "auto"}, prior_lines=_lines())

    out = await stage.handle(stage.input_model(), ctx)

    # id-set + order identical to input.
    assert [ln.id for ln in out.lines] == ["l1", "l2", "l3"]
    # only l1's text changed; the hallucinated id never appears.
    assert out.lines[0].text == "Stop scrolling — this changes everything."
    assert out.lines[1].text == "Here is the dashboard."
    assert out.lines[2].text == "Thanks for watching."
    assert all(ln.id != "l999" for ln in out.lines)
    # clip / id untouched.
    assert [ln.clip for ln in out.lines] == [None, "clip-2", "clip-3"]


# ── (d) critique_only: critic runs, optimizer NOT applied, text unchanged ─────


@pytest.mark.asyncio
async def test_critique_only_applies_no_rewrite() -> None:
    fake = _wm_fake(
        {"overall_score": 2, "directives": [{"op": "rewrite", "target_ids": ["l1"]}]},
        [{"id": "l1", "text": "SHOULD NOT BE APPLIED"}],
    )
    stage = RetentionPlanStage(client=fake)
    ctx = _ctx(retention={"mode": "critique_only"}, prior_lines=_lines())

    out = await stage.handle(stage.input_model(), ctx)

    # text is byte-identical to input — advisory only.
    assert [ln.text for ln in out.lines] == [ln["text"] for ln in _lines()]
    assert [ln.id for ln in out.lines] == ["l1", "l2", "l3"]
    # the critic ran; the optimizer (hook-doctor) never did.
    workers = [c.kwargs["worker_name"] for c in fake.message.call_args_list]
    assert "retention-critic" in workers
    assert "hook-doctor" not in workers


# ── (e) active happy path: below→above threshold converges, clip/id intact ────


@pytest.mark.asyncio
async def test_active_happy_path_converges_and_applies() -> None:
    critic_low = {
        "overall_score": 2,
        "directives": [{"op": "cold_open", "target_ids": ["l1"]}],
        "per_beat": [{"id": "l1", "drop_risk": "high"}],
    }
    critic_high = {"overall_score": 5, "directives": []}
    fake = _wm_fake(
        [critic_low, critic_high],
        [{"id": "l1", "text": "A punchier hook.", "hook_role": "cold_open"}],
    )
    stage = RetentionPlanStage(client=fake)
    ctx = _ctx(
        retention={"mode": "auto", "iterations": 2, "score_threshold": 4.0},
        prior_lines=_lines(),
        target_seconds=30.0,
        retention_weights={"max_time_to_first_footage_seconds": 8.0},
    )

    out = await stage.handle(stage.input_model(), ctx)

    # loop applied the rewrite + annotation and converged.
    assert out.lines[0].text == "A punchier hook."
    assert out.lines[0].hook_role == "cold_open"
    # both workers were exercised.
    workers = {c.kwargs["worker_name"] for c in fake.message.call_args_list}
    assert workers == {"retention-critic", "hook-doctor"}
    # spine id-set + clip never mutated.
    assert [ln.id for ln in out.lines] == ["l1", "l2", "l3"]
    assert [ln.clip for ln in out.lines] == [None, "clip-2", "clip-3"]


# ── (e2) optimizer batching: many flagged lines rewrite in bounded WM calls ───


@pytest.mark.asyncio
async def test_optimizer_rewrites_flagged_lines_in_bounded_batches() -> None:
    """When the critic flags more lines than the batch size, the optimizer rewrites
    them in bounded ``hook-doctor`` calls (≤ _OPTIMIZER_BATCH_SIZE each) and merges
    every revision back. A single unbounded call is the small-vLLM timeout bug this
    split fixes; the holistic critic stays a single call (one arc score)."""
    from src.services.stages.demo_video.retention_plan import _OPTIMIZER_BATCH_SIZE

    n = _OPTIMIZER_BATCH_SIZE * 2 + 1  # 11 flagged → 3 optimizer batches (5, 5, 1)
    prior = [{"id": f"l{i}", "text": f"Beat {i}.", "seconds": 3.0, "clip": None} for i in range(n)]
    all_ids = [str(ln["id"]) for ln in prior]

    # Pass 1 critic flags every beat (below threshold); pass 2 scores above so the
    # loop converges after exactly one optimize pass.
    critic_low = {"overall_score": 1, "directives": [{"op": "rewrite", "target_ids": all_ids}]}
    critic_high = {"overall_score": 5, "directives": []}
    critic_queue = [critic_low, critic_high]
    optimizer_batch_sizes: list[int] = []

    def _side(worker_name: str | None = None, **kwargs: Any) -> WMResponse:
        if worker_name == "retention-critic":
            return _resp(critic_queue.pop(0) if len(critic_queue) > 1 else critic_queue[0])
        # hook-doctor: rewrite ONLY this batch's ids. Parse them from the beats
        # block, NOT the critic-guidance block below it (which echoes every id).
        beats_block = kwargs["message"].split("Critic guidance:")[0]
        ids = [i for i in all_ids if f'"id": "{i}"' in beats_block]
        optimizer_batch_sizes.append(len(ids))
        return _resp({"lines": [{"id": i, "text": f"HOOKED {i}"} for i in ids]})

    fake = AsyncMock()
    fake.message.side_effect = _side
    stage = RetentionPlanStage(client=fake)
    ctx = _ctx(
        retention={"mode": "auto", "iterations": 2, "score_threshold": 4.0},
        prior_lines=prior,
        target_seconds=60.0,
    )

    out = await stage.handle(stage.input_model(), ctx)

    # ceil(11 / 5) = 3 bounded optimizer calls, order-preserving, nothing dropped.
    assert optimizer_batch_sizes == [5, 5, 1]
    # Every flagged line's rewrite was merged back (all batches, not just the first).
    by_id = {ln.id: ln.text for ln in out.lines}
    assert all(by_id[i] == f"HOOKED {i}" for i in all_ids)
    assert [ln.id for ln in out.lines] == all_ids  # spine id-set + order intact


# ── (f) soft-gate: a persistently low score still returns the best draft ──────


@pytest.mark.asyncio
async def test_low_score_never_raises_soft_gate() -> None:
    # Critic never crosses threshold; the loop is bounded and returns the best draft
    # rather than raising.
    fake = _wm_fake(
        {"overall_score": 1, "directives": [{"op": "rewrite", "target_ids": ["l2"]}]},
        [{"id": "l2", "text": "Revised middle beat."}],
    )
    stage = RetentionPlanStage(client=fake)
    ctx = _ctx(retention={"mode": "auto", "iterations": 2}, prior_lines=_lines())

    out = await stage.handle(stage.input_model(), ctx)

    # bounded loop (≤2 passes) — never blocks the deliverable.
    assert [ln.id for ln in out.lines] == ["l1", "l2", "l3"]
    assert 1 <= fake.message.call_count <= 2 * 2


# ── (f) a dead retention-critic degrades — it must not kill the run ───────────
#
# The measured production kill this covers: `showcase_v1` run
# 019fece1-072f-72e9-98af-23f2b6e8e99b (2026-08-10) reached `retention_plan`
# with `demo_video_record` already COMPLETED, the critic came back
# "Model token limit (6000) exceeded before any response was generated", and
# the whole ~20-stage run went to `failed`. Three `demo_video_v1` runs died the
# same way on 2026-07-20. The stage's own docstring says a low score "NEVER
# raises / blocks the deliverable"; an ABSENT score must not either.


# ── (f2) the critic call carries an explicit output-token budget ──────────────


@pytest.mark.asyncio
async def test_critic_call_sets_explicit_max_tokens() -> None:
    """Pins an explicit, floor-clearing ``max_tokens`` on the retention-critic call.

    Without it WM applies DEFAULT_SELF_HOSTED_MAX_TOKENS=6000, which a reasoning
    model can burn entirely on thinking before emitting any JSON — the empty-reply
    failure in the dead-critic test below.

    ⚠️ THIS TEST USED TO PIN THE LITERAL 8000, on the reasoning that 8000 "stays
    inside inference-router's 300 s chain deadline". Both halves were wrong. The
    binding cap is not the router's 300 s but worker-management's
    ``agent_run_timeout_s = 600 s``, and at the ~11.5 tok/s measured on the live box
    under load, 8000 tokens is ~700 s — i.e. a budget WM would kill before it could
    be spent. Proven, not assumed: the real segment-selector worker at
    ``max_tokens=7500`` returned 500 "Agent timed out".

    So the number now comes from ``output_token_budget``, which clamps to what the
    box can actually deliver. It asserts the FLOOR and the derivation rather than a
    literal, because the literal is exactly what went stale.
    """
    fake = _wm_fake({"overall_score": 5, "directives": []}, [])
    stage = RetentionPlanStage(client=fake)
    ctx = _ctx(retention={"mode": "auto"}, prior_lines=_lines())

    await stage.handle(stage.input_model(), ctx)

    critic_calls = [
        c for c in fake.message.call_args_list if c.kwargs["worker_name"] == "retention-critic"
    ]
    assert critic_calls, "the retention-critic was never called"
    for call in critic_calls:
        assert not call.args, "WMClient.message must be called with keyword args only"
        assert call.kwargs["max_tokens"] == _CRITIC_MAX_TOKENS
        assert call.kwargs["max_tokens"] >= MIN_OUTPUT_TOKEN_BUDGET


class _CriticUpstream500Error(RuntimeError):
    """Stand-in for the WM 500/422 the retention-critic call raises."""


@pytest.mark.asyncio
async def test_dead_critic_returns_untouched_spine_instead_of_failing() -> None:
    fake = AsyncMock()

    def _side(worker_name: str | None = None, **_: Any) -> WMResponse:
        if worker_name == "retention-critic":
            raise _CriticUpstream500Error(
                "upstream 500 from .../workers/retention-critic/message: Model token "
                "limit (6000) exceeded before any response was generated."
            )
        return _resp({"lines": []})

    fake.message.side_effect = _side
    stage = RetentionPlanStage(client=fake)
    ctx = _ctx(retention={"mode": "auto"}, prior_lines=_lines())

    out = await stage.handle(stage.input_model(), ctx)

    # The spine survives verbatim — ids, order, text and clip bindings intact.
    assert [ln.id for ln in out.lines] == ["l1", "l2", "l3"]
    assert [ln.text for ln in out.lines] == [ln["text"] for ln in _lines()]
    assert [ln.clip for ln in out.lines] == [None, "clip-2", "clip-3"]
    # The critic WAS attempted — this is a degrade, not a silent opt-out.
    assert fake.message.await_count >= 1
