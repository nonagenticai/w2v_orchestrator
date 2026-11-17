"""retention_plan — a bounded WM-LLM evaluator-optimizer for hook/retention.

An OPT-IN stage (ORCH-1.3b) that runs a critique → rewrite → re-score loop over
the ordered narration ``lines`` spine, improving the *cold open*, open loops and
pattern-interrupt cadence WITHOUT ever touching the spine's shape. It is modeled
exactly on ``shared/script_doctor.py``: a deterministic evidence pass
(:func:`analyze_retention`) hands hard numbers to an LLM ``retention-critic``,
which scores the draft (1-5) and names the weak beats; a second LLM
``hook-doctor`` then rewrites ONLY those flagged, unlocked lines and the revision
is merged back into the *original ordered draft by id* — so join ids, line order
and locked lines survive by construction, never by trust. The loop is
hard-bounded by the shared ``loop_engine`` (≤2 passes) and keeps the highest
scoring draft.

Presence-is-opt-in: with no ``direction.retention`` block (or ``mode == "off"``)
the stage raises :class:`StageSkippedError` before any WM traffic — a run that
did not opt in is byte-identical to before. ``mode == "critique_only"`` runs the
critic once for advisory scoring but applies NO rewrite (the spine is returned
unchanged). The stage re-emits the exact ``demo_video_gen`` output shape
(:class:`DemoVideoGenOutput`), so it is a drop-in the spine resolver
(``demo_video/spine.py``) prefers over the raw generated spine.

SCOPE (C0 / soft-gate): the optimizer may only rewrite line ``text`` and set the
per-line ``hook_role`` / ``pin_position`` annotation fields; it can NEVER add,
drop, reorder, or re-clip a line. The engine's ``id_extractor`` enforces the
id-set invariant on every pass, and a low score NEVER blocks the deliverable —
the current best draft is always returned.

LLM calls route through ``worker_management`` (the ``WMClient`` gateway) — never
inference-router / a provider directly; the model behind each worker is config,
not code.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from itertools import batched
from typing import TYPE_CHECKING, ClassVar, cast

from src.services.llm import WMClient
from src.services.stages.base import SlotUse, StageHandler, StageInput, StageSkippedError
from src.services.stages.batch_ledger import BatchLedger
from src.services.stages.demo_video.gen import DemoVideoGenOutput
from src.services.stages.demo_video.spine import resolve_plan_output
from src.services.stages.llm_budget import (
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
    reasoning_for_attempt,
)
from src.services.stages.shared.loop_engine import (
    ON_LOOP_ERROR_KEEP_BEST,
    Critique,
    run_evaluator_optimizer,
)
from src.services.stages.shared.retention import analyze_retention
from src.utils import clamp_score

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


def _as_list(value: object) -> list[object]:
    """Read-boundary coercion for LLM/generator output: a list only if it is one."""
    return list(value) if isinstance(value, (list, tuple)) else []


# WM worker names (regex ^[a-zA-Z0-9_-]+$). WM auto-creates them on first call;
# the provider/model lives on the WM worker config, not here.
_CRITIC_WORKER = "retention-critic"
_OPTIMIZER_WORKER = "hook-doctor"

# Per-WM-call batch size for the optimizer: the ``lines`` output array scales with
# the flagged-line count, so a wholesale rewrite otherwise times out the small
# vLLM and silently returns fewer revisions than asked (mirrors footage_critique).
# The critic is intentionally NOT batched — it emits one holistic arc score over
# the whole draft, which cannot be scored in 5-line fragments.
_OPTIMIZER_BATCH_SIZE = 5

# Per-call OUTPUT budgets. Both WM calls here go to a REASONING model whose
# thinking tokens are charged against ``max_tokens``, so these are sized as
# ANSWER estimates and ``output_token_budget`` adds the reasoning headroom and
# the router chain-deadline cap (see ``stages/llm_budget.py``).
#
# The two sizes differ because the two answers differ in shape. The critic is
# NOT batched — it echoes every line id plus per-beat flags over the WHOLE draft,
# so its answer grows with script length (2000 keeps the previous explicit
# ``max_tokens=8000`` total unchanged). The optimizer rewrites at most
# ``_OPTIMIZER_BATCH_SIZE`` lines per call, so its answer is bounded by the batch:
# 5 lines * ~120 tokens of rewritten text + JSON overhead.
_CRITIC_ANSWER_TOKENS = 2000
_CRITIC_MAX_TOKENS = output_token_budget(answer_tokens=_CRITIC_ANSWER_TOKENS)
_OPTIMIZER_ANSWER_TOKENS = _OPTIMIZER_BATCH_SIZE * 120
_OPTIMIZER_MAX_TOKENS = output_token_budget(answer_tokens=_OPTIMIZER_ANSWER_TOKENS)

_DEFAULT_ITERATIONS = 2
_DEFAULT_THRESHOLD = 4.0

# Critic ``op`` values that ask for a STRUCTURAL change (reorder / merge / inject a
# new beat) this stage cannot make under C0 — surfaced as an advisory rerun hint.
_STRUCTURAL_OPS = frozenset({"merge", "reorder", "inject_interrupt"})

# The C16 retention-critic output schema (verbatim). ``overall_score`` (1-5) drives
# convergence; ``directives`` name the weak beats + the op the critic wants.
#
# The 1-5 band is a ``description``, NOT ``minimum``/``maximum``. Guided decoding cannot
# express a numeric bound in its grammar, so a schema bound is unenforced where it would
# help and fatal where it does not: WM runs ``inference_router`` with ``retries=0``, so
# the first out-of-range integer kills the stage for good. This stage is where that bit
# us — every prod ``demo_video`` run died here on ``overall_score: 12``, almost certainly
# echoed out of the "hard numbers" evidence block ``_critic_msg`` injects below. The band
# is steered by the seeded ``retention-critic`` system prompt ("Score the draft 1-5…") and ENFORCED by ``clamp_score``
# at the read below. See ``tests/test_output_schema_bounds.py``.
_CRITIC_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["overall_score", "directives"],
    "properties": {
        "overall_score": {"type": "integer", "description": "1-5, where 5 is strongest"},
        "cold_open_strength": {"type": "integer", "description": "1-5, where 5 is strongest"},
        "first_3s_hook": {"type": "integer", "description": "1-5, where 5 is strongest"},
        "open_loops": {"type": "integer", "description": "1-5, where 5 is strongest"},
        "pattern_interrupt_cadence": {
            "type": "integer",
            "description": "1-5, where 5 is strongest",
        },
        "momentum": {"type": "integer", "description": "1-5, where 5 is strongest"},
        "chapter_pacing": {"type": "integer", "description": "1-5, where 5 is strongest"},
        "per_beat": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "drop_risk"],
                "properties": {
                    "id": {"type": "string"},
                    # ENUM, not a free string. Without it the model answers in
                    # whatever vocabulary it likes and `_is_risky` cannot know
                    # which words mean what — see that function's docstring for
                    # what the free-string version cost.
                    "drop_risk": {"type": "string", "enum": ["low", "medium", "high"]},
                    "rationale": {"type": "string"},
                },
            },
        },
        "directives": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["op", "target_ids"],
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            "rewrite",
                            "merge",
                            "reorder",
                            "inject_interrupt",
                            "open_loop",
                            "payoff",
                            "cold_open",
                        ],
                    },
                    "target_ids": {"type": "array", "items": {"type": "string"}},
                    "payload": {"type": "object"},
                },
            },
        },
    },
}

# The hook-doctor rewrite schema. Text-only (+ optional annotation fields); the
# id is echoed so the merge can map revisions back onto the ordered spine.
_OPTIMIZER_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["lines"],
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id"],
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "candidates": {"type": "array", "items": {"type": "string"}},
                    "hook_role": {"type": "string"},
                    "pin_position": {"type": "integer"},
                },
            },
        }
    },
}


def _lines_json(lines: list[dict[str, object]]) -> str:
    """Deterministic id+text projection of the spine for a prompt block."""
    return json.dumps(
        [{"id": str(line["id"]), "text": str(line.get("text", ""))} for line in lines],
        indent=2,
        ensure_ascii=False,
    )


def _spine_ids(lines: list[dict[str, object]]) -> frozenset[str]:
    """The C0 invariant key: the ordered spine's id-set (order-independent)."""
    return frozenset(str(line["id"]) for line in lines)


def _weak_lines(
    draft: list[dict[str, object]], critic_output: dict[str, object]
) -> list[dict[str, object]]:
    """The rewrite set: beats the critic flagged that the author did not lock.

    A beat is flagged when a directive targets its id OR its ``per_beat`` entry
    carries a non-trivial ``drop_risk``. Hallucinated ids (not in the spine) and
    locked lines are dropped, so the optimizer can only ever touch real, unlocked
    beats — the id-set is preserved by construction.
    """
    spine = {str(line["id"]) for line in draft}
    locked = {str(line["id"]) for line in draft if line.get("locked")}

    flagged: set[str] = set()
    for directive in _as_list(critic_output.get("directives")):
        if not isinstance(directive, dict):
            continue
        directive = cast("dict[str, object]", directive)
        for target in _as_list(directive.get("target_ids")):
            flagged.add(str(target))
    for beat in _as_list(critic_output.get("per_beat")):
        if not isinstance(beat, dict):
            continue
        beat = cast("dict[str, object]", beat)
        if beat.get("id") is None:
            continue
        raw_risk = beat.get("drop_risk")
        if _is_risky(raw_risk):
            flagged.add(str(beat["id"]))
        elif _is_unrecognised_risk(raw_risk):
            # Not silently "safe". An unreadable verdict means this beat was
            # NOT considered, and the stage would otherwise complete green
            # having flagged nothing. There is no notes surface on this stage,
            # so WARNING is the visibility.
            log.warning(
                "retention_plan: unreadable drop_risk %r on beat %s — beat NOT flagged; "
                "widen _RISKY_PREFIXES/_SAFE_PREFIXES or tighten the schema enum",
                raw_risk,
                beat["id"],
            )

    allowed = (flagged & spine) - locked
    return [line for line in draft if str(line["id"]) in allowed]


#: Token stems `_is_risky` understands as "rewrite this beat". Matched PER TOKEN,
#: never against the whole string — see `_is_risky` for why that distinction is
#: load-bearing. Kept next to the schema enum below so widening one without the
#: other is visible in a diff.
_RISKY_TOKEN_PREFIXES = (
    "high",
    "med",
    "moderat",
    "sever",
    "critical",
    "elevat",
    "significant",
    "serious",
    "extreme",
    "substantial",
)

#: Recognised low-risk stems. A value matching NEITHER set is unknown, and
#: unknown is reported rather than silently treated as safe — see `_is_risky`.
_SAFE_TOKEN_PREFIXES = ("low", "none", "minimal", "neglig", "safe", "ok", "nil", "trivial")


def _is_risky(risk: object) -> bool:
    """True when a critic ``drop_risk`` reads as a beat worth rewriting.

    ⚠️ This used to be an exact-membership test — ``risk.strip().lower() in
    {"high", "med", "medium"}`` — against a schema that declared ``drop_risk`` as
    a free ``{"type": "string"}`` with no enum. So the model was under no
    obligation to produce those three tokens, and everything else it naturally
    writes — ``"moderate"``, ``"HIGH RISK"``, ``"High — viewers drop here"``,
    ``"elevated"`` — read as NOT risky. ``flagged`` came back empty, no beat was
    rewritten, and the whole retention pass completed green having done nothing.

    Two changes, because either alone is insufficient: the schema now pins an
    ``enum`` (so structured output constrains the model), and this parser matches
    on PREFIX over a wider vocabulary (so a model that answers outside the enum,
    or an older cached generation, is still read correctly).
    """
    if isinstance(risk, bool):
        return risk
    if isinstance(risk, int | float):
        return float(risk) >= 0.5
    if isinstance(risk, str):
        return _classify(risk) is True
    return False


def _tokens(text: str) -> list[str]:
    """Lowercased alphabetic tokens. `"High - viewers drop"` -> `["high", ...]`."""
    return [t for t in re.split(r"[^a-zA-Z]+", text.lower()) if t]


def _classify(risk: str) -> bool | None:
    """``True`` risky, ``False`` safe, ``None`` unrecognised.

    ⚠️ Matched per TOKEN, not against the head of the whole string. Prefix-matching
    the string was the second version of this bug: it read ``"modest"`` as risky
    (it starts with no risky stem, but ``"mod"`` was in the list) while STILL
    missing the natural risky phrasings the fix was written for — ``"very high"``,
    ``"somewhat high"``, ``"significant"`` — because only the leading token was
    ever examined.

    Safe wins a tie: ``"moderately low"`` carries both stems and is not a beat
    worth re-paying an LLM rewrite for. Over-flagging costs a rewrite pass on
    every beat; that is a real cost, not a free default.
    """
    tokens = _tokens(risk)
    if not tokens:
        return None
    safe = any(t.startswith(_SAFE_TOKEN_PREFIXES) for t in tokens)
    risky = any(t.startswith(_RISKY_TOKEN_PREFIXES) for t in tokens)
    if safe:
        return False
    if risky:
        return True
    return None


def _is_unrecognised_risk(risk: object) -> bool:
    """True when ``risk`` is a non-empty string in neither vocabulary.

    Such a value is the failure the old exact-match hid: it silently means "not
    risky". It is surfaced as a stage note (hence a degraded run) instead of
    being guessed at in either direction — over-flagging costs a rewrite pass on
    every beat, under-flagging is what shipped.
    """
    if not isinstance(risk, str):
        return False
    return _classify(risk) is None


def _apply_by_id(
    draft: list[dict[str, object]],
    revised: list[dict[str, object]],
    allowed_ids: frozenset[str],
) -> list[dict[str, object]]:
    """Merge revised TEXT (+ hook_role / pin_position) into the ordered draft, by id.

    Iterates ``draft`` in its original order and patches ONLY the ``text`` (and,
    when echoed, the ``hook_role`` / ``pin_position`` annotation fields) of a line
    whose id the worker echoed AND that is in ``allowed_ids`` (flagged + unlocked).
    Ids the worker hallucinated (not in the spine) are ignored — never added;
    ``clip`` / ``id`` are NEVER touched; nothing is added, dropped, or reordered.
    So the spine id-set + join ids survive by construction and a locked line can
    never be rewritten. ``text`` falls back to the first ``candidates`` entry when
    the worker returned candidates instead of a single line.
    """
    patches: dict[str, dict[str, str | int]] = {}
    for item in revised:
        if not isinstance(item, dict):
            continue
        lid = item.get("id")
        if lid is None or str(lid) not in allowed_ids:
            continue
        patch: dict[str, str | int] = {}
        text = item.get("text")
        if text is None:
            candidates = item.get("candidates")
            if isinstance(candidates, list) and candidates:
                text = candidates[0]
        if text is not None:
            patch["text"] = str(text)
        if item.get("hook_role") is not None:
            patch["hook_role"] = str(item["hook_role"])
        if item.get("pin_position") is not None:
            patch["pin_position"] = int(cast("str | int", item["pin_position"]))
        if patch:
            patches[str(lid)] = patch
    return [
        {**line, **patches[str(line["id"])]} if str(line["id"]) in patches else line
        for line in draft
    ]


def _critic_msg(
    draft: list[dict[str, object]], evidence: dict[str, object], director: object
) -> str:
    """Build the critic prompt: ordered spine + deterministic retention evidence."""
    parts = [
        f"Director intent: {director if director else 'none stated'}",
        "Narration spine (ordered, by id):",
        _lines_json(draft),
        "Deterministic retention evidence (hard numbers — ground your critique in these):",
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False),
        "Score the draft and name the weak beats by id. Preserve every id; do not reorder.",
    ]
    return "\n\n".join(parts)


def _optimizer_msg(flagged: list[dict[str, object]], critic_output: dict[str, object]) -> str:
    """Build the rewrite prompt: the weak beats + critic guidance."""
    parts = [
        "Rewrite ONLY these weak beats for hook / retention. Echo each id UNCHANGED.",
        "One entry per input beat, same id — no add/drop/merge/split/reorder, no re-clip.",
        "Beats to rewrite:",
        _lines_json(flagged),
        "Critic guidance:\n"
        + json.dumps(critic_output, indent=2, sort_keys=True, ensure_ascii=False),
    ]
    return "\n\n".join(parts)


class RetentionPlanInput(StageInput):
    """No typed inputs — everything is read from ``ctx`` (prior output + config)."""


class RetentionPlanStage(StageHandler[RetentionPlanInput, DemoVideoGenOutput]):
    name = "retention_plan"
    input_model = RetentionPlanInput
    output_model = DemoVideoGenOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (
        SlotUse("llm"),
        SlotUse("tts"),
    )

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    async def handle(self, payload: RetentionPlanInput, ctx: RunContext) -> DemoVideoGenOutput:
        gen: dict[str, object] = resolve_plan_output(ctx) or {}
        lines: list[dict[str, object]] = cast("list[dict[str, object]]", _as_list(gen.get("lines")))
        if not lines:
            raise StageSkippedError("retention_plan: no spine lines to plan over")

        direction = ctx.config_snapshot.get("direction") or {}
        ret = direction.get("retention") or {}
        if not isinstance(ret, dict) or not ret or ret.get("mode") == "off":
            raise StageSkippedError("retention_plan: no direction.retention block")

        iterations = int(ret.get("iterations", _DEFAULT_ITERATIONS))
        threshold = float(ret.get("score_threshold", _DEFAULT_THRESHOLD))
        mode = ret.get("mode", "auto")  # auto | critique_only | off (off skipped above)

        # ``retention_weights`` was resolved from direction.retention by run_setup;
        # target length comes from the deliverable descriptor (may be None).
        weights = ctx.config_snapshot.get("retention_weights")
        deliverable = ctx.config_snapshot.get("deliverable") or {}
        target_seconds = deliverable.get("target_duration_seconds")
        director = direction.get("intent")

        async def critic_fn(draft: list[dict[str, object]]) -> Critique:
            # Recompute the deterministic evidence on the CURRENT draft so the critic
            # scores against fresh numbers each pass.
            report = analyze_retention(draft, weights=weights, target_seconds=target_seconds)
            evidence: dict[str, object] = {
                "risk_flags": report.risk_flags,
                "watch_curve": report.watch_curve,
                "time_to_first_footage": report.time_to_first_footage,
                "interrupt_gaps": report.interrupt_gaps,
                "over_budget": report.over_budget,
                "total_seconds": report.total_seconds,
                "target_seconds": report.target_seconds,
            }
            # ``max_tokens`` is explicit here, NOT left to WM's
            # DEFAULT_SELF_HOSTED_MAX_TOKENS=6000: the critic must echo every line id
            # plus per-beat flags, so its output grows with script length and 6000
            # truncates it (4/218 retention_plan runs died on an empty critic reply —
            # see loop_engine.py:150). Routed through ``_CRITIC_MAX_TOKENS`` so the
            # answer estimate and the reasoning headroom stay separable; the total is
            # unchanged at 8000 tok ≈ 216s at the measured 37.05 tok/s decode rate
            # (worker_management pydantic_ai_agent.py:496-518), inside
            # inference-router's 300s chain deadline.
            resp = await self._wm.message(
                enable_reasoning=reasoning_for_attempt(),
                worker_name=_CRITIC_WORKER,
                message=_critic_msg(draft, evidence, director),
                output_schema=_CRITIC_SCHEMA,
                max_tokens=_CRITIC_MAX_TOKENS,
            )
            output = resp.output if isinstance(resp.output, dict) else {}
            # Clamped, not trusted: the schema can only ASK for 1-5 (see _CRITIC_SCHEMA).
            score = clamp_score(output.get("overall_score"))
            # Advisory: a critic asking for a structural op we cannot make under C0
            # (reorder / merge / inject) is surfaced as an upstream rerun hint.
            rerun = (
                ["demo_video_gen"]
                if any(
                    isinstance(d, dict) and d.get("op") in _STRUCTURAL_OPS
                    for d in (output.get("directives") or [])
                )
                else []
            )
            return Critique(score=score, rerun_stages=rerun, directives=output)

        # One ledger per distinct flagged-batch payload, so a `stage_call` retry of
        # the optimizer resumes instead of re-paying. Lives here, outside
        # `optimizer_fn`, because the retry re-ENTERS that function.
        optimizer_ledgers: dict[
            str, BatchLedger[list[dict[str, object]], list[dict[str, object]]]
        ] = {}

        async def optimizer_fn(
            draft: list[dict[str, object]], critique: Critique
        ) -> list[dict[str, object]]:
            critic_output = critique.directives if isinstance(critique.directives, dict) else {}
            flagged = _weak_lines(draft, critic_output)
            if not flagged:
                return draft
            allowed = frozenset(str(line["id"]) for line in flagged)
            # Rewrite the flagged lines in bounded batches (≤ _OPTIMIZER_BATCH_SIZE
            # per WM call). Each flagged line is rewritten independently against the
            # same critic notes, so the flat merge is exact; ``_apply_by_id`` applies
            # the union once and only ``allowed`` ids take effect. A malformed batch
            # simply contributes nothing (as the single-call path returned no
            # revisions on a malformed response) — the other batches still apply.
            batches = [list(b) for b in batched(flagged, _OPTIMIZER_BATCH_SIZE, strict=False)]

            # `loop_engine` wraps THIS function in `stage_call(retries=...)`, so a
            # timeout re-invokes it with the SAME draft and critique. `revised` was
            # local to the call, so attempt 2 restarted at batch 0 and re-paid every
            # WM call attempt 1 had already answered — a flagged set whose batches do
            # not fit inside ONE attempt could never complete. See `stages/batch_ledger`.
            #
            # Keyed on a digest of the exact prompt payload, not on the pass number:
            # the ledger may only resume when the batches are byte-identical. A later
            # pass re-derives `flagged` from a REWRITTEN draft, so its lines differ,
            # so it gets its own ledger — reusing answers there would pin a stale
            # rewrite onto new text.
            # ⚠️ The digest MUST cover `critic_output` too, not just `flagged`.
            # The prompt is `_optimizer_msg(batch, critic_output)`, so two passes
            # can produce a byte-identical `flagged` under a DIFFERENT critique —
            # which happens exactly when pass 1's batch was malformed and
            # contributed no revision. Keying on `flagged` alone would hit the
            # same ledger entry, issue ZERO calls, and pin pass 1's empty answer
            # for the rest of the loop.
            key = hashlib.sha256(
                json.dumps(
                    {"flagged": flagged, "critique": critic_output},
                    sort_keys=True,
                    default=str,
                ).encode()
            ).hexdigest()
            ledger = optimizer_ledgers.setdefault(key, BatchLedger())

            async def _rewrite(batch: list[dict[str, object]]) -> list[dict[str, object]]:
                resp = await self._wm.message(
                    enable_reasoning=reasoning_for_attempt(),
                    worker_name=_OPTIMIZER_WORKER,
                    message=_optimizer_msg(batch, critic_output),
                    output_schema=_OPTIMIZER_SCHEMA,
                    max_tokens=_OPTIMIZER_MAX_TOKENS,
                )
                batch_lines = resp.output.get("lines", []) if isinstance(resp.output, dict) else []
                if not isinstance(batch_lines, list):
                    # Unchanged contract: a malformed batch contributes nothing and
                    # the other batches still apply (the single-call path returned no
                    # revisions on a malformed response).
                    return []
                return cast("list[dict[str, object]]", batch_lines)

            parts = await ledger.run(batches, _rewrite)
            revised = [line for part in parts for line in part]
            return _apply_by_id(draft, revised, allowed)

        # critique_only: score once (advisory) but apply NO rewrite. Forcing a single
        # pass means the engine calls the critic once and never the optimizer, so the
        # best draft is the untouched initial spine.
        max_passes = 1 if mode == "critique_only" else iterations

        result = await run_evaluator_optimizer(
            lines,
            critic_fn=critic_fn,
            optimizer_fn=optimizer_fn,
            threshold=threshold,
            id_extractor=_spine_ids,
            max_passes=max_passes,
            stage_name=self.name,
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
            # ADVISORY by contract (see the soft-gate note below): a critic that
            # returns nothing must not kill a run that has already recorded footage.
            on_loop_error=ON_LOOP_ERROR_KEEP_BEST,
        )
        # Soft-gate: accept the current best draft even on non-convergence; a low
        # score NEVER raises / blocks the deliverable.
        return DemoVideoGenOutput.model_validate({**gen, "lines": result.draft})
