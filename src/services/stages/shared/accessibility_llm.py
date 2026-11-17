"""Accessibility Tier-2 LLM layer (A19) — advisory critic + audio description.

The deterministic Tier-1 pass (:mod:`accessibility`) already computes WCAG
contrast/backplate findings and the over-threshold silent gaps. A19 adds the
*LLM* half on top of that read-out, entirely as an ADVISORY enrichment of the
growth bundle — it never rewrites the delivered render, so an opted-in run's
video is byte-identical to an opted-out one; only the growth bundle carries the
extra advisory fields.

Two single-pass workers (no evaluator-optimizer loop — there is nothing to
re-score because A19 never changes the render):

- ``a11y-critic`` RANKS which of the pre-computed deterministic remediations
  matter most and explains them for a run-monitor operator. It NEVER sets or
  overrides a numeric pass/fail threshold — that stays authoritative code.
- ``audio-describer`` writes an audio-description sidecar: one concise line per
  silent gap, describing on-screen action, each clamped to its gap's word
  budget (:func:`clamp_description_words`) BEFORE acceptance so a verbose model
  can never overrun the gap it must fit in.

Presence-is-opt-in and never-gates: the caller only runs these when the run
carries an ``accessibility`` block with the matching opt-in, and wraps the call
so a failure degrades to empty advisory output — never discarding the stage's
successful result.

LLM calls route through ``worker_management`` (the ``WMClient`` gateway) — the
model behind each worker is config, not code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from src.services.stages.llm_budget import (
    LARGE_ANSWER_TOKENS,
    MEDIUM_ANSWER_TOKENS,
    output_token_budget,
)
from src.services.stages.shared.speech_rate import authored_natural_wpm, resolve_natural_wpm_detail

if TYPE_CHECKING:
    from src.services.llm import WMClient
    from src.services.runs.run_context import RunContext
    from src.services.stages.shared.accessibility import A11yPolicy, A11yReport

# WM worker names (regex ^[a-zA-Z0-9_-]+$). WM auto-creates them on first call;
# the provider/model + curated prompt live on the WM seed config, not here.
CRITIC_WORKER = "a11y-critic"
DESCRIBER_WORKER = "audio-describer"

# Words per minute a listener comfortably follows — the budget an audio
# description line must fit inside its silent gap. 150 wpm is the audio-book /
# narration norm the rest of the pipeline assumes; it is the byte-identical
# fallback used when no ``RunContext`` is available (a bare unit test, or a
# caller that predates the ctx thread below) — see :func:`_phase_a_wpm`.
_WORDS_PER_MINUTE = 150

_CRITIC_SYSTEM = (
    "You are an accessibility reviewer. You are given a DETERMINISTIC WCAG "
    "contrast / silent-gap report (authoritative code already decided pass/fail). "
    "You RANK which of the pre-computed remediations to prefer and explain the "
    "findings for a run-monitor operator. You NEVER set, lower, or override a "
    "numeric threshold — that is code's job. Echo every line id UNCHANGED and "
    "return ONLY the requested JSON."
)
_CRITIC_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["priority_findings"],
    "properties": {
        "summary": {"type": "string"},
        "priority_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["line_id", "severity"],
                "properties": {
                    "line_id": {"type": "string"},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "recommendation": {"type": "string"},
                },
            },
        },
    },
}

_DESCRIBER_SYSTEM = (
    "You write audio-description lines for the narration-free silent gaps you are "
    "given, describing ONLY visible on-screen action (never narrating dialogue). "
    "Each line must fit its gap's word budget. Echo the beat id each gap follows "
    "UNCHANGED. Never invent gaps or reorder beats. Return ONLY the requested JSON."
)
_DESCRIBER_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["lines"],
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["after_line_id", "text"],
                "properties": {
                    "after_line_id": {"type": "string"},
                    "text": {"type": "string"},
                },
            },
        }
    },
}


# Reasoning-model output budgets (see stages/llm_budget.py — this is a FLOOR).
#
# CRITIC: ``{summary, priority_findings[]}`` — a handful of ranked objects, each an
# echoed line_id + a severity enum + one recommendation sentence, plus a short prose
# summary. A modest list of small objects with prose fields ⇒ MEDIUM.
_CRITIC_MAX_TOKENS = output_token_budget(answer_tokens=MEDIUM_ANSWER_TOKENS)
# DESCRIBER: ``{lines[]}`` — one audio-description line per silent gap, and the gap
# count tracks the rendered timeline's length (this call is NOT batched: every gap
# goes in one prompt). A list whose length grows with the input ⇒ LARGE.
_DESCRIBER_MAX_TOKENS = output_token_budget(answer_tokens=LARGE_ANSWER_TOKENS)


def _gap_ms(gap: Mapping[str, object]) -> float:
    """The gap's duration in ms, or 0.0 when absent/non-numeric (JSON-safe)."""
    value = gap.get("gap_ms")
    return float(value) if isinstance(value, (int, float)) else 0.0


def gap_word_budget(gap_ms: float, wpm: float = _WORDS_PER_MINUTE) -> int:
    """Words an audio-description line may use to fit inside a ``gap_ms`` silence.

    ``gap_ms / 60000 * wpm`` words, rounded (not truncated — a floor silently
    under-fills the gap, leaving residual dead-air), never below 1 so even a
    just-over-threshold gap can carry a one-word description.

    ``wpm`` defaults to the hardcoded 150 for a bare caller (a unit test, or one
    with no ``RunContext``); :func:`describe_gaps` passes the run's own resolved
    rate instead — see :func:`_phase_a_wpm`.
    """
    return max(1, round(gap_ms / 60_000.0 * wpm))


async def _phase_a_wpm(ctx: RunContext | None) -> float:
    """Phase-A wpm to size a gap's word budget with — never Phase B.

    A11y audio-description describes silent gaps in EXISTING recorded video: there
    is no narration SCRIPT for that gap (that is the whole point of a silent gap),
    so :func:`~src.services.stages.shared.speech_rate.resolve_script_sample_wpm`
    (Phase B, which samples drafted lines) has nothing to sample. Phase A — the
    author's stated rate, else the calibration-passage measurement, else the
    hardcoded default — is the correct and only rung here.

    ``ctx=None`` (a caller with no run context, incl. every pre-existing unit test)
    is byte-identical to the previous hardcoded-150 behaviour: no probe, no ctx
    read, nothing new to fail.
    """
    if ctx is None:
        return float(_WORDS_PER_MINUTE)
    authored = authored_natural_wpm(ctx)
    if authored is not None:
        return authored
    rate = await resolve_natural_wpm_detail(ctx, default=float(_WORDS_PER_MINUTE))
    return rate.wpm


def clamp_description_words(text: str, budget: int) -> str:
    """Truncate ``text`` to at most ``budget`` whitespace-delimited words.

    Deterministic guard applied to every describer line BEFORE acceptance, so a
    verbose model can never overrun its gap. A text already within budget is
    returned unchanged.
    """
    words = text.split()
    if len(words) <= budget:
        return text
    return " ".join(words[:budget])


def _critic_message(report: A11yReport, policy: A11yPolicy) -> str:
    lines = [
        f"WCAG level: {policy.wcag_level} (min ratio {policy.min_ratio:.1f}:1). "
        "Rank the findings below; do NOT change any numeric threshold.",
        "Deterministic findings (authoritative — rank, do not overrule):",
        *(f"- {insight}" for insight in report.insights),
    ]
    if not report.insights:
        lines.append("- (none — the deterministic pass found no contrast/gap issues)")
    return "\n".join(lines)


async def rank_findings(
    wm: WMClient,
    report: A11yReport,
    policy: A11yPolicy,
    *,
    system_prompt: str = _CRITIC_SYSTEM,
    enable_reasoning: bool | None = None,
) -> dict[str, object]:
    """Ask ``a11y-critic`` to rank the deterministic findings (advisory).

    Returns the critic's JSON (``{summary, priority_findings}``) or an empty dict
    when the worker returns a non-object. The caller decides whether to run this
    (opt-in) and wraps it so a failure never gates.
    """
    resp = await wm.message(
        enable_reasoning=enable_reasoning,
        worker_name=CRITIC_WORKER,
        message=_critic_message(report, policy),
        system_prompt=system_prompt,
        output_schema=_CRITIC_SCHEMA,
        max_tokens=_CRITIC_MAX_TOKENS,
    )
    return resp.output if isinstance(resp.output, dict) else {}


def _describer_message(
    gaps: Sequence[Mapping[str, object]], text_by_line: dict[str, str], wpm: float
) -> str:
    parts = [
        "Write one audio-description line per silent gap below, describing the "
        "on-screen action. Echo after_line_id UNCHANGED and stay within word_budget.",
    ]
    for gap in gaps:
        after = str(gap.get("after_line_id") or "")
        budget = gap_word_budget(_gap_ms(gap), wpm)
        context = text_by_line.get(after, "")
        parts.append(
            f"- after_line_id={after} word_budget={budget} (preceding narration: {context!r})"
        )
    return "\n".join(parts)


async def describe_gaps(
    wm: WMClient,
    gaps: Sequence[Mapping[str, object]],
    text_by_line: dict[str, str],
    *,
    ctx: RunContext | None = None,
    system_prompt: str = _DESCRIBER_SYSTEM,
    enable_reasoning: bool | None = None,
) -> list[dict[str, object]]:
    """Ask ``audio-describer`` for one AD line per gap, word-clamped to each gap.

    Returns ``[{"after_line_id", "text"}, ...]`` for gaps the worker described,
    each ``text`` clamped to its gap's :func:`gap_word_budget`. Gaps the worker
    echoed for an unknown id, or returned no text for, are dropped — the sidecar
    never invents a beat. Empty gaps ⇒ no worker call by the caller.

    ``ctx``, when given, resolves the run's own Phase-A wpm (:func:`_phase_a_wpm`)
    instead of the hardcoded 150 default — the same rate an author's
    ``direction.timing.wpm.natural`` or a measured voice implies elsewhere in this
    run. ``ctx=None`` keeps the previous byte-identical behaviour.
    """
    if not gaps:
        return []
    wpm = await _phase_a_wpm(ctx)
    budget_by_line = {
        str(g.get("after_line_id") or ""): gap_word_budget(_gap_ms(g), wpm) for g in gaps
    }
    resp = await wm.message(
        enable_reasoning=enable_reasoning,
        worker_name=DESCRIBER_WORKER,
        message=_describer_message(gaps, text_by_line, wpm),
        system_prompt=system_prompt,
        output_schema=_DESCRIBER_SCHEMA,
        max_tokens=_DESCRIBER_MAX_TOKENS,
    )
    raw = resp.output.get("lines", []) if isinstance(resp.output, dict) else []
    described: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        after = str(item.get("after_line_id") or "")
        text = item.get("text")
        if after not in budget_by_line or not text:
            continue
        described.append(
            {
                "after_line_id": after,
                "text": clamp_description_words(str(text), budget_by_line[after]),
            }
        )
    return described
