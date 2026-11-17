"""Shared evaluator-optimizer loop engine (aspect A10).

A self-contained substrate for the *critic → optimize → re-score* pattern: a
draft ``D`` is scored by an injected text critic, an injected optimizer produces
a revised draft, that draft is re-scored, and the loop keeps the highest-scoring
draft. The engine is WM-agnostic — callers pass closures (typically wrapping a
``WMClient.message`` call) as ``critic_fn`` / ``optimizer_fn`` — so it imports no
compile/render/video client and can never trigger a re-render. Cost is bounded:
each pass is one critic call plus (from pass 2) one optimizer call, and the pass
count is hard-clamped to ``[1, 2]``.

Nothing consumes this module yet (zero blast radius); it is the reusable
primitive the per-blueprint loop stages will drive later. History is kept
in-memory only — durable persistence (C14) is a later consumer PR.

Both injected calls run inside ``stage_call`` so they inherit the same
per-attempt timeout + retry budget as every other LLM stage
(``llm_budget_seconds`` / ``llm_retries`` when the caller does not override),
and both are wrapped in ``upstream.guarded`` under ``LLM_RETRY_ON`` so a
permanent upstream failure aborts the loop instead of consuming that budget.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from w2v_common.stages import stage_call

from src.services.stages.errors import StagePreconditionError, StageSkippedError
from src.services.stages.llm_budget import (
    LLM_RETRY_ON,
    downshift_reasoning,
    llm_budget_seconds,
    llm_retries,
)
from src.services.stages.upstream import guarded

log = logging.getLogger(__name__)

#: ``on_loop_error`` values. ``raise`` (the default) preserves the historical
#: contract: any critic/optimizer failure escapes and fails the stage. ``keep_best``
#: is the OPT-IN degrade for stages whose critic is advisory by contract.
_ON_LOOP_ERROR_RAISE = "raise"
ON_LOOP_ERROR_KEEP_BEST = "keep_best"

#: ``LoopResult.reason`` emitted by the ``keep_best`` degrade. Distinct from
#: ``"no_critic"`` (the caller never asked for a critic) and from ``"max_passes"``
#: (the critic ran and did not converge) — here the critic was ASKED and FAILED.
CRITIC_UNAVAILABLE = "critic_unavailable"


@dataclass(frozen=True)
class Critique:
    """A critic's verdict on one draft.

    ``score`` drives convergence; ``rerun_stages`` is the plain list of upstream
    stage names the critic wants re-run (the live annotations channel is
    ``StageOutput`` annotations — ORCH-1.1 only emits a plain ``list[str]`` and
    does not touch that schema); ``directives`` carries an
    optimizer-specific payload (free-form) the optimizer consumes to revise.
    """

    score: float
    rerun_stages: list[str] = field(default_factory=list)
    directives: Any = None


@dataclass(frozen=True)
class LoopResult[D]:
    """Outcome of one evaluator-optimizer run.

    ``draft`` is the retained (max-scoring) draft, ``score`` its critic score.
    ``passes`` counts scoring rounds (pass 1 = the initial draft; each optimizer
    revision adds one). ``converged`` is True only when the retained draft met
    ``threshold``. ``reason`` is why the loop stopped. ``rerun_stages`` mirrors
    the retained draft's critique. ``history`` is the per-pass ``(score, draft)``
    trail in execution order, in-memory only.
    """

    draft: D
    score: float
    passes: int
    converged: bool
    reason: str
    rerun_stages: list[str]
    history: list[tuple[float, D]]


async def run_evaluator_optimizer[D](  # noqa: C901 -- one cohesive algorithm: the
    # no-critic short-circuit, the bounded drive loop and the opt-in degrade all read
    # against each other; extracting a branch would hide the control flow, not clarify it.
    initial: D,
    *,
    critic_fn: Callable[[D], Awaitable[Critique]] | None,
    optimizer_fn: Callable[[D, Critique], Awaitable[D]],
    gate_fn: Callable[[float, float, int], bool] | None = None,
    threshold: float = 4.0,
    epsilon: float = 0.25,
    max_passes: int = 2,
    timeout_seconds: float | None = None,
    retries: int | None = None,
    retry_on: tuple[type[BaseException], ...] = LLM_RETRY_ON,
    stage_name: str = "loop_engine",
    id_extractor: Callable[[D], frozenset[str]] | None = None,
    on_loop_error: str = _ON_LOOP_ERROR_RAISE,
) -> LoopResult[D]:
    """Drive a critic → optimize → re-score loop, keeping the best draft.

    With ``critic_fn is None`` the loop is a no-op: ``initial`` is returned
    unchanged (same object identity), ``score=0.0``, ``passes=0``,
    ``converged=False``, ``reason="no_critic"`` — and **zero awaits** run, so no
    WM/network traffic is issued for a blueprint that opted out of evaluation.

    Otherwise ``initial`` is scored (pass 1). While the gate says continue, the
    retained best draft is fed to ``optimizer_fn`` and the revision is re-scored
    (each a further pass). The max-scoring draft across all passes is retained; a
    regression therefore returns the earlier, better draft.

    Convergence is decided by ``gate_fn(score, delta, passes) -> stop?``; the
    default gate stops when ``score >= threshold`` (met the bar) OR
    ``delta < epsilon`` (plateaued/regressed) OR ``passes >= max_passes`` (out of
    budget). ``max_passes`` is hard-clamped to ``[1, 2]`` regardless of the gate.
    ``reason`` is one of
    ``{"threshold", "plateau", "max_passes", "no_critic", "critic_unavailable"}``.

    ``id_extractor``, when given, guards the C0 spine invariant: after each
    optimizer step the candidate's id-set must equal the pre-optimization best's
    id-set, else a ``ValueError`` is raised — the optimizer may rewrite content
    but must never add, drop, or reorder spine ids.

    ``timeout_seconds`` / ``retries`` default to ``llm_budget_seconds()`` /
    ``llm_retries()`` (read lazily, so test/env overrides apply) when ``None``.

    ``on_loop_error`` decides what a critic/optimizer failure does. The default
    ``"raise"`` is the historical contract — the exception escapes and fails the
    stage. ``ON_LOOP_ERROR_KEEP_BEST`` is the OPT-IN degrade for stages whose critic
    is ADVISORY by their own documented contract: the best draft seen so far is
    returned with ``reason=CRITIC_UNAVAILABLE`` and ``converged=False``, logged at
    ERROR with a traceback. ``StageSkippedError`` / ``StagePreconditionError`` are
    control flow, never degraded; ``asyncio.CancelledError`` is a ``BaseException``
    and so is never caught either.

    Under ``keep_best`` the C0 spine-guard ``ValueError`` below degrades too, and that
    is the SAFE outcome rather than a hole in the guard: the returned ``best`` is the
    draft from BEFORE the offending optimizer pass, so the id-set the guard exists to
    protect is exactly what ships. The violation is still surfaced loudly (ERROR +
    traceback); what changes is that a misbehaving optimizer costs the run its
    enrichment instead of costing it the whole deliverable.

    That opt-in exists because these loops used to be able to destroy a finished run
    over an advisory opinion. Measured: over 218 ``retention_plan`` executions, 4 died
    on a ``retention-critic`` call that returned nothing (three ``demo_video_v1`` runs
    on 2026-07-20, one ``showcase_v1`` run on 2026-08-10 that had already completed
    ``demo_video_record``), each killing a ~20-stage run for a score the stage's own
    docstring says "NEVER raises / blocks the deliverable". The same 6000-token
    ceiling has since been classified upstream as non-retryable
    (``worker_management`` #169), so the retry ladder can no longer paper over it —
    which makes the degrade the only thing standing between one empty decode and a
    dead run.

    ``retry_on`` defaults to ``LLM_RETRY_ON``, and both injected calls are routed
    through ``guarded`` — so an upstream failure that no retry can fix (WM 402
    "configure your LLM provider", a prompt over the worker's token limit) aborts
    the loop on the first critic/optimizer call carrying the actionable sentence,
    instead of paying the retry ladder twice per pass and reporting "retries
    exhausted". The loop stages are the most expensive LLM callers in the run;
    they are exactly the ones that must not burn the budget on a lost cause.
    """
    if critic_fn is None:
        # No-op path: no evaluation requested. Return the SAME object identity
        # with zero awaits — a blueprint that did not opt into a critic pays
        # nothing (no WM call, no network).
        return LoopResult(
            draft=initial,
            score=0.0,
            passes=0,
            converged=False,
            reason="no_critic",
            rerun_stages=[],
            history=[],
        )

    critic = critic_fn  # narrowed non-None for the closures below
    max_passes = max(1, min(2, max_passes))
    timeout = timeout_seconds if timeout_seconds is not None else llm_budget_seconds()
    retry_count = retries if retries is not None else llm_retries()

    gate = gate_fn or (
        lambda score, delta, passes: score >= threshold or delta < epsilon or passes >= max_passes
    )

    async def _score(draft: D) -> Critique:
        return await stage_call(
            downshift_reasoning(lambda _think: guarded(lambda: critic(draft))),
            timeout_seconds=timeout,
            retries=retry_count,
            retry_on=retry_on,
            stage_name=f"{stage_name}:critic",
        )

    async def _optimize(draft: D, critique: Critique) -> D:
        return await stage_call(
            downshift_reasoning(lambda _think: guarded(lambda: optimizer_fn(draft, critique))),
            timeout_seconds=timeout,
            retries=retry_count,
            retry_on=retry_on,
            stage_name=f"{stage_name}:optimizer",
        )

    def _reason(score: float, delta: float, passes: int) -> str:
        if score >= threshold:
            return "threshold"
        if delta < epsilon:
            return "plateau"
        return "max_passes"

    # Everything the drive loop mutates, hoisted so the ``keep_best`` degrade below
    # can return the best draft SEEN SO FAR — not just ``initial``. A pass-2 critic
    # failure therefore still ships the pass-1 improvement.
    best: D = initial
    best_score: float = 0.0
    best_critique: Critique = Critique(score=0.0, rerun_stages=[], directives={})
    history: list[tuple[float, D]] = []
    passes = 0
    current_score = 0.0
    delta = float("inf")

    try:
        # Pass 1: score the initial draft. delta is +inf so a first-pass "plateau"
        # can never fire (there is no prior score to compare against).
        critique = await _score(initial)
        best_score = critique.score
        best_critique = critique
        history = [(critique.score, initial)]
        passes = 1
        current_score = critique.score

        # Hard cap at max_passes even if a custom gate would keep going.
        while passes < max_passes and not gate(current_score, delta, passes):
            candidate = await _optimize(best, best_critique)
            if id_extractor is not None and id_extractor(candidate) != id_extractor(best):
                raise ValueError(
                    f"{stage_name}: optimizer changed the spine id-set "
                    f"(C0 violation) — expected {id_extractor(best)}, "
                    f"got {id_extractor(candidate)}"
                )
            cand_critique = await _score(candidate)
            passes += 1
            delta = cand_critique.score - current_score
            current_score = cand_critique.score
            history.append((cand_critique.score, candidate))
            if cand_critique.score > best_score:
                best, best_score, best_critique = candidate, cand_critique.score, cand_critique
    except StageSkippedError, StagePreconditionError:
        # Control-flow signals are NOT failures — they must reach the run engine
        # rather than be swallowed into a degraded "keep the best draft".
        raise
    except Exception as exc:
        if on_loop_error == _ON_LOOP_ERROR_RAISE:
            raise
        # SOFT degrade (opt-in, for stages whose critic is ADVISORY): an LLM that
        # returns nothing must not destroy a 20-stage run. Keep the best draft seen
        # so far and report it honestly via ``reason`` so the run-read layer can tell
        # "the critic passed it" from "the critic never spoke". Logged at ERROR with
        # a traceback — degraded is not silent.
        log.error(
            "%s: evaluator-optimizer loop failed after %d pass(es); keeping the best "
            "draft and continuing (critic is advisory): %s",
            stage_name,
            passes,
            exc,
            exc_info=True,
        )
        return LoopResult(
            draft=best,
            score=best_score,
            passes=passes,
            converged=False,
            reason=CRITIC_UNAVAILABLE,
            rerun_stages=list(best_critique.rerun_stages),
            history=history,
        )

    reason = _reason(current_score, delta, passes)
    return LoopResult(
        draft=best,
        score=best_score,
        passes=passes,
        converged=best_score >= threshold,
        reason=reason,
        rerun_stages=list(best_critique.rerun_stages),
        history=history,
    )
