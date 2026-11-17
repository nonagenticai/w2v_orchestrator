"""script_doctor — a bounded WM-LLM evaluator-optimizer over the line spine.

An OPT-IN stage (ORCH-1.2) that runs a critique → revise → re-score loop over
the ordered script ``lines`` produced by ``demo_video_gen``. Each pass a
rubric (:func:`score_script`) scores the draft and an LLM critic names the weak
lines; the optimizer then asks a second LLM to rewrite ONLY the flagged, unlocked
lines and merges the revised text back into the *original ordered draft by id* —
so join ids, line order, verbatim pins and locked lines survive by construction,
never by trust. The loop is hard-bounded by the shared ``loop_engine`` (≤2 passes)
and keeps the highest-scoring draft.

Inert-by-default: with no enabled ``script_doctor`` config block the stage raises
:class:`StageSkippedError` before any WM traffic — a run that did not opt in pays
nothing. The stage re-emits the exact ``demo_video_gen`` output shape
(:class:`DemoVideoGenOutput`), so it is a drop-in the spine resolver
(``demo_video/spine.py``) prefers over the raw generated spine.

LLM calls route through ``worker_management`` (the ``WMClient`` gateway) — never
inference-router / a provider directly; the model behind each worker is config,
not code.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

from src.services.evaluation.rubrics.script_craft import score_script
from src.services.llm import WMClient
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StagePreconditionError,
    StageSkippedError,
    assert_verbatim_present,
    items_for_stage,
    partition_for_stage,
    record_verbatim_drop,
    verbatim_prompt_block,
)
from src.services.stages.demo_video.gen import DemoVideoGenOutput
from src.services.stages.llm_budget import (
    LARGE_ANSWER_TOKENS,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
    reasoning_for_attempt,
)
from src.services.stages.primitives.script_stage import non_empty_array
from src.services.stages.shared.beat_grounding import (
    MAX_UNGROUNDED_WORDS,
    beat_source,
    ungrounded_words,
)
from src.services.stages.shared.loop_engine import Critique, run_evaluator_optimizer
from src.services.stages.shared.narration_budget import budget_words, count_words, word_band
from src.services.stages.shared.speech_rate import resolve_natural_wpm
from src.services.stages.shared.time_budget import DEFAULT_WPM_NATURAL

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

# WM worker names (regex ^[a-zA-Z0-9_-]+$). WM auto-creates them on first call;
# the provider/model lives on the WM worker config, not here.
_CRITIQUE_WORKER = "script-doctor-critique"
_REVISE_WORKER = "script-doctor-revise"

# Default composite gate + default deliverable target length (seconds).
_DEFAULT_COMPOSITE_TARGET = 80.0
_DEFAULT_TARGET_SECONDS = 300.0
_DEFAULT_MAX_PASSES = 2

_CRITIQUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["critiques", "story_notes"],
    "properties": {
        "critiques": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "line_id",
                    "violated_principle",
                    "failing_metrics",
                    "severity",
                    "suggested_rewrite",
                ],
                "properties": {
                    "line_id": {"type": "string"},
                    "violated_principle": {"type": "string"},
                    "failing_metrics": {"type": "array", "items": {"type": "string"}},
                    "severity": {"type": "string", "enum": ["low", "med", "high"]},
                    "suggested_rewrite": {"type": "string"},
                },
            },
        },
        "story_notes": {
            "type": "object",
            "properties": {
                "missing_turn": {"type": "boolean"},
                "weak_hook": {"type": "boolean"},
                "arc_summary": {"type": "string"},
            },
        },
    },
}

# The revise call fires ONLY when at least one line is flagged (``optimizer_fn``
# returns early otherwise), and an empty ``lines`` merges to the unchanged draft —
# a silent no-op revision. So ``[]`` is never a valid answer here: make the list
# GRAMMAR-non-empty (WM strips ``minItems``; see :func:`non_empty_array`).
_REVISE_ITEM_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["id", "text"],
    "properties": {
        "id": {"type": "string"},
        "text": {"type": "string"},
    },
}

_REVISE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["lines"],
    "properties": {"lines": non_empty_array(_REVISE_ITEM_SCHEMA)},
}


# Reasoning-model output budgets (see stages/llm_budget.py — a FLOOR, not a cap).
#
# CRITIQUE: ``{critiques[], story_notes}`` — one critique object per flagged line,
# each carrying a violated_principle, a metrics list AND a full suggested_rewrite of
# the narration line. Not batched: the whole draft goes in one prompt, so the list
# length tracks the script ⇒ LARGE.
_CRITIQUE_MAX_TOKENS = output_token_budget(answer_tokens=LARGE_ANSWER_TOKENS)
# REVISE: ``{lines[]}`` — the rewritten narration TEXT for every flagged line. The
# answer is prose whose length tracks the flagged-line count ⇒ LARGE.
_REVISE_MAX_TOKENS = output_token_budget(answer_tokens=LARGE_ANSWER_TOKENS)


def _resolve_target(config_snapshot: dict[str, Any]) -> float:
    """Deliverable target narration length in seconds (default 300)."""
    for source in (
        config_snapshot,
        config_snapshot.get("deliverable") or {},
        config_snapshot.get("brief") or {},
    ):
        value = source.get("target_duration_seconds")
        if value is not None:
            return float(value)
    return _DEFAULT_TARGET_SECONDS


def _flagged_lines(draft: list[dict[str, Any]], report: dict[str, Any]) -> list[dict[str, Any]]:
    """Lines the rubric flagged AND the author did not lock — the revise set."""
    flags = report.get("flags") or {}
    return [line for line in draft if flags.get(str(line["id"])) and not line.get("locked")]


def _apply_by_id(
    draft: list[dict[str, Any]],
    revised: list[dict[str, Any]],
    allowed_ids: frozenset[str],
) -> list[dict[str, Any]]:
    """Merge revised TEXT into the original ordered draft, by id.

    Iterates ``draft`` in its original order and replaces ONLY the ``text`` of a
    line whose id the worker echoed AND that is in ``allowed_ids`` (flagged +
    unlocked). Ids the worker hallucinated (not in draft) are ignored — never
    added; lines the worker dropped keep their original text. Nothing is added,
    dropped, or reordered, so the spine id-set + join ids are preserved by
    construction and a locked line can never be rewritten.
    """
    new_text: dict[str, str] = {
        str(item["id"]): str(item["text"])
        for item in revised
        if item.get("id") is not None
        and item.get("text") is not None
        and str(item["id"]) in allowed_ids
    }
    return [
        {**line, "text": new_text[str(line["id"])]}
        if str(line["id"]) in new_text and _grounded(line, new_text[str(line["id"])])
        else line
        for line in draft
    ]


def _grounded(line: Mapping[str, Any], revised: str) -> bool:
    """False when a revision talks about things the beat and its recording never show.

    Only a line that carries its recorded scenario (``scenario_name`` / ``gherkin`` —
    the bdd-generated demo_video spine) can be judged; any other line passes. A refused
    revision falls back to the UNREVISED line. Run 01a156a6 (2026-10-20) shipped
    "I am about to break the React TodoMVC demo in three specific ways" over a clip
    that adds a todo — this stage's revise output, with nothing to check it against.
    """
    source = beat_source(line)
    if source is None:
        return True
    novel = ungrounded_words(revised, str(line.get("text", "")), source)
    if len(novel) <= MAX_UNGROUNDED_WORDS:
        return True
    log.warning(
        "script_doctor: revision of line %s introduced %d word(s) absent from the line "
        "and its recording (%s) — refused as ungrounded; keeping the unrevised line.",
        line.get("id"),
        len(novel),
        ", ".join(novel[:8]),
    )
    return False


def _revise_lines_json(lines: Sequence[Mapping[str, Any]], word_budgets: Mapping[str, int]) -> str:
    """The revise prompt's line block: id + text, plus what grounds and sizes each line.

    ``recording_shows`` (:func:`beat_source`) and ``word_budget`` (gen's
    ``narration_words_per_beat``) ride ONLY when known, so a line without them renders
    exactly as :func:`_lines_json` does.
    """
    out: list[dict[str, object]] = []
    for line in lines:
        item: dict[str, object] = {"id": str(line["id"]), "text": str(line.get("text", ""))}
        source = beat_source(line)
        if source is not None:
            item["recording_shows"] = source
        budget = word_budgets.get(str(line["id"]))
        if budget is not None:
            item["word_budget"] = budget
        out.append(item)
    return json.dumps(out, indent=2, ensure_ascii=False)


def _word_budgets(gen: Mapping[str, Any]) -> dict[str, int]:
    """gen's ``narration_words_per_beat`` (positive ints only); empty when absent."""
    raw = gen.get("narration_words_per_beat")
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(k): int(v)
        for k, v in raw.items()
        if isinstance(v, int | float) and not isinstance(v, bool) and v > 0
    }


def _script_word_floor(
    config_snapshot: Mapping[str, Any], word_budgets: Mapping[str, int], wpm: float
) -> int | None:
    """The fewest words the doctored script may hold — ``None`` when no budget is known.

    The budget is gen's per-beat ``narration_words_per_beat`` summed when present
    (what the script was planned to), else the EXPLICITLY ordered duration at
    ``wpm``; :func:`_resolve_target`'s 300s fallback is not an order and states no
    floor. The floor is the low edge of :func:`narration_budget.word_band`, the band
    every script prompt states around its target.
    """
    if word_budgets:
        budget = sum(word_budgets.values())
    else:
        ordered = next(
            (
                source.get("target_duration_seconds")
                for source in (
                    config_snapshot,
                    config_snapshot.get("deliverable") or {},
                    config_snapshot.get("brief") or {},
                )
                if isinstance(source, Mapping) and source.get("target_duration_seconds")
            ),
            None,
        )
        if ordered is None:
            return None
        budget = budget_words(float(ordered), words_per_minute=wpm)
    if budget <= 0:
        return None
    return word_band(budget)[0]


def _shrinks_below_floor(
    draft: Sequence[Mapping[str, Any]], merged: Sequence[Mapping[str, Any]], floor: int | None
) -> bool:
    """True when a revision lands under ``floor`` AND is shorter than its draft.

    A draft already under the floor may still be revised as long as it does not get
    SHORTER — the guard refuses lost length, not an imperfect starting point.
    """
    if floor is None:
        return False
    before = count_words([str(line.get("text", "")) for line in draft])
    after = count_words([str(line.get("text", "")) for line in merged])
    if after >= floor or after >= before:
        return False
    log.warning(
        "script_doctor: revision cut the script from %d to %d words, under its %d-word "
        "floor — a film is exactly its narration, so it would ship short; keeping the "
        "unrevised draft.",
        before,
        after,
        floor,
    )
    return True


def _lines_json(lines: list[dict[str, Any]]) -> str:
    """Deterministic id+text projection of lines for a prompt block."""
    return json.dumps(
        [{"id": str(line["id"]), "text": str(line.get("text", ""))} for line in lines],
        indent=2,
        ensure_ascii=False,
    )


def _json_default(o: object) -> object:
    """``json.dumps`` fallback for the rubric report's pydantic values.

    ``score_script`` returns ``per_line``/``per_story`` as ``{metric: MetricResult}``,
    and ``MetricResult`` is a pydantic model the stdlib encoder cannot serialize.
    Dump any pydantic model to its dict; anything else genuinely unserializable
    still raises (never silently stringify a surprise type).
    """
    if isinstance(o, BaseModel):
        return o.model_dump()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _critique_msg(
    draft: list[dict[str, Any]],
    report: dict[str, Any],
    director: Any,
    notes: list[dict[str, Any]],
) -> str:
    """Build the critic prompt: ordered lines + rubric flags + director intent."""
    flags = report.get("flags") or {}
    per_line = report.get("per_line") or {}
    # ``per_line[lid]`` is ``{metric_name: MetricResult}`` (script_craft.score_script),
    # and ``MetricResult`` is a pydantic model — not JSON-serializable by default. Dump
    # it via ``model_dump`` so the critic prompt renders the full sub-score breakdown
    # instead of raising ``TypeError: Object of type MetricResult is not JSON
    # serializable`` (which, thrown here BEFORE the WM call, escapes the loop's
    # soft-degrade and hard-fails the stage — it took script_doctor down on first
    # end-to-end contact).
    weak = json.dumps(
        {
            lid: {"failing_metrics": flags.get(lid, []), "sub_scores": per_line.get(lid, {})}
            for lid in sorted(flags)
        },
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        default=_json_default,
    )
    parts = [
        f"Composite score: {report.get('composite')}",
        f"Director intent: {director if director else 'none stated'}",
        "Script lines (ordered, by id):",
        _lines_json(draft),
        "Rubric-flagged lines (id -> failing metrics + sub-scores):",
        weak,
    ]
    if notes:
        rendered = "\n".join(
            f"- {json.dumps(note, sort_keys=True, ensure_ascii=False)}" for note in notes
        )
        parts.append("Open reviewer annotations:\n" + rendered)
    parts.append("Critique each flagged line in place. Preserve every line id; do not reorder.")
    return "\n\n".join(parts)


def _revise_msg(
    flagged: list[dict[str, Any]],
    crit: Any,
    verbatim: list[dict[str, Any]],
    word_budgets: Mapping[str, int] | None = None,
    *,
    script_words: int | None = None,
    word_floor: int | None = None,
) -> str:
    """Build the revise prompt: the weak lines + critic guidance + verbatim pins.

    ``word_floor`` (:func:`_script_word_floor`) states the WHOLE script's minimum,
    with ``script_words`` as what it holds now: a film is exactly its narration, so a
    revise that only tightens lines ships a film shorter than the order.
    """
    budgets = word_budgets or {}
    parts = [
        "Rewrite ONLY these weak lines. Echo each id UNCHANGED; return revised text.",
        "One output line per input line, same id — no add/drop/merge/split/reorder.",
    ]
    if any(beat_source(line) is not None for line in flagged):
        parts.append(
            "A line's `recording_shows` is what its screen recording actually performs. "
            "Use ONLY what the line already says and what its recording shows — never "
            "mention a feature, action, item, outcome or claim that is not in them."
        )
    if any(str(line["id"]) in budgets for line in flagged):
        parts.append(
            "A line's `word_budget` is how many words it must hold to fill its share of "
            "the film: write each such line to about that many words, never far under it."
        )
    if word_floor is not None and script_words is not None:
        parts.append(
            f"The whole script holds {script_words} words and must keep at least "
            f"{word_floor} after your rewrite — the film is exactly as long as its "
            "narration, so a rewrite that only shortens lines makes the film too short. "
            "Tighten wording without losing length; never drop content to cut words."
        )
    parts += ["Lines to revise:", _revise_lines_json(flagged, budgets)]
    if crit is not None:
        parts.append(
            "Critic guidance:\n" + json.dumps(crit, indent=2, sort_keys=True, ensure_ascii=False)
        )
    vb = verbatim_prompt_block(verbatim)
    if vb:
        parts.append(vb.strip())
    return "\n\n".join(parts)


class ScriptDoctorInput(StageInput):
    """No typed inputs — everything is read from ``ctx`` (prior output + config)."""


class ScriptDoctorStage(StageHandler[ScriptDoctorInput, DemoVideoGenOutput]):
    name = "script_doctor"
    input_model = ScriptDoctorInput
    output_model = DemoVideoGenOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (
        SlotUse("llm"),
        SlotUse("tts"),
    )

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    async def handle(self, payload: ScriptDoctorInput, ctx: RunContext) -> DemoVideoGenOutput:
        gen: dict[str, Any] = ctx.prior_output("demo_video_gen") or {}
        lines: list[dict[str, Any]] = list(gen.get("lines") or [])

        sd = ctx.config_snapshot.get("script_doctor") or {}
        if not isinstance(sd, dict) or not sd.get("enabled"):
            raise StageSkippedError("script_doctor: no enabled script_doctor block")

        # Verbatim kinds this stage does NOT consume are narrowed away, but never
        # silently: `record_verbatim_drop` logs them and appends a soft-degrade note
        # (→ `degraded` run) on BOTH return paths below.
        _, dropped = partition_for_stage(ctx.config_snapshot.get("verbatim_items") or [], self.name)

        # SOFT stage. script_doctor is a quality *enhancement* — so a WM/upstream failure
        # (2 of 3 live prod failures
        # were WM 401/402: expired downstream license token / byo-mode misconfig) or a
        # malformed author `script_doctor` block must NOT hard-fail a paid render AFTER
        # the human gen-gate. On any such failure, accept the un-doctored draft — exactly
        # what shipped before this stage existed — mirroring final_critique's soft-degrade
        # idiom. StageSkippedError / StagePreconditionError still propagate.
        try:
            doctored = await self._doctor(lines, sd, ctx, word_budgets=_word_budgets(gen))
        except StageSkippedError, StagePreconditionError:
            raise
        except Exception as exc:  # noqa: BLE001 — see soft-degrade rationale above
            # Accept the un-doctored draft — EXCEPT never ship content missing
            # author-mandated verbatim text. script_doctor's optimizer is the ONLY
            # verbatim gate in the demo_video chain (demo_video_gen never checks it),
            # so a blind degrade could ship a render without required CTA/disclaimer
            # text. Re-assert on the raw draft: it is what shipped before this stage
            # existed, so it is a safe fallback ONLY while it still carries the
            # verbatim. If it does not, refuse to ship — `assert_verbatim_present`
            # raises StageError and the run fails loud, exactly as before this stage
            # became soft. (Empty verbatim_items ⇒ trivially passes ⇒ degrade proceeds.)
            assert_verbatim_present(
                " ".join(str(line.get("text", "")) for line in lines),
                items_for_stage(ctx.config_snapshot.get("verbatim_items") or [], self.name),
                stage=self.name,
            )
            log.warning(
                "script_doctor: soft-degrade, accepting un-doctored draft (%s): %s",
                type(exc).__name__,
                exc,
            )
            return record_verbatim_drop(
                DemoVideoGenOutput.model_validate({**gen, "lines": lines}),
                dropped,
                stage=self.name,
            )
        return record_verbatim_drop(
            DemoVideoGenOutput.model_validate({**gen, "lines": doctored}),
            dropped,
            stage=self.name,
        )

    async def _doctor(
        self,
        lines: list[dict[str, Any]],
        sd: Mapping[str, Any],
        ctx: RunContext,
        *,
        word_budgets: Mapping[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Run the evaluator-optimizer loop and return the doctored lines.

        May raise on a WM/upstream failure or a malformed author ``script_doctor``
        block; :meth:`handle` soft-degrades every such error to the un-doctored draft.
        """
        verbatim: list[dict[str, Any]] = items_for_stage(
            ctx.config_snapshot.get("verbatim_items") or [], self.name
        )
        director = (ctx.config_snapshot.get("direction") or {}).get("intent")
        notes = ctx.annotations_for("script_doctor")
        target = _resolve_target(ctx.config_snapshot)
        # THE RATE THE RUBRIC SCORES AGAINST. Resolved ONCE here, not inside
        # `critic_fn`: the loop scores up to `max_passes` drafts and the voice's
        # speed does not change between them, so a per-pass call would only buy
        # repeated cache lookups. `resolve_natural_wpm` never raises and never
        # fails a run — it returns `default` on any probe failure, which makes
        # this byte-identical to the old hardcoded reference on a bad day.
        #
        # ⚠️ This is what makes the rubric's `word_budget`/`duration_budget`
        # grade against the voice that will actually speak the lines. Without it
        # both metrics score every beat against a fixed 150 wpm while the live
        # voice delivers ~120, so an over-packed beat passes and this stage's
        # rewrite loop is told to make it LONGER.
        wpm = await resolve_natural_wpm(ctx, default=DEFAULT_WPM_NATURAL)
        floor = _script_word_floor(ctx.config_snapshot, word_budgets or {}, wpm)
        max_passes = int(sd.get("max_passes", _DEFAULT_MAX_PASSES))
        composite_target = float(
            sd.get("target_scores", {}).get("composite", _DEFAULT_COMPOSITE_TARGET)
        )

        async def critic_fn(draft: list[dict[str, Any]]) -> Critique:
            report = score_script(
                draft,
                verbatim_items=verbatim,
                target_duration_seconds=target,
                director=director,
                wpm=wpm,
            )
            resp = await self._wm.message(
                enable_reasoning=reasoning_for_attempt(),
                worker_name=_CRITIQUE_WORKER,
                message=_critique_msg(draft, report, director, notes),
                output_schema=_CRITIQUE_SCHEMA,
                max_tokens=_CRITIQUE_MAX_TOKENS,
            )
            return Critique(score=float(report["composite"]), directives=(resp.output, report))

        async def optimizer_fn(
            draft: list[dict[str, Any]], critique: Critique
        ) -> list[dict[str, Any]]:
            crit, report = critique.directives
            flagged = _flagged_lines(draft, report)
            if not flagged:
                return draft
            allowed = frozenset(str(line["id"]) for line in flagged)
            resp = await self._wm.message(
                enable_reasoning=reasoning_for_attempt(),
                worker_name=_REVISE_WORKER,
                message=_revise_msg(
                    flagged,
                    crit,
                    verbatim,
                    word_budgets,
                    script_words=count_words([str(line.get("text", "")) for line in draft]),
                    word_floor=floor,
                ),
                output_schema=_REVISE_SCHEMA,
                max_tokens=_REVISE_MAX_TOKENS,
            )
            revised = resp.output.get("lines", []) if isinstance(resp.output, dict) else []
            merged = _apply_by_id(draft, revised, allowed)
            if _shrinks_below_floor(draft, merged, floor):
                return draft
            assert_verbatim_present(
                " ".join(str(line.get("text", "")) for line in merged),
                verbatim,
                stage=self.name,
            )
            return merged

        def gate_fn(score: float, delta: float, passes: int) -> bool:
            return score >= composite_target or delta < 0.25 or passes >= max_passes

        result = await run_evaluator_optimizer(
            lines,
            critic_fn=critic_fn,
            optimizer_fn=optimizer_fn,
            gate_fn=gate_fn,
            # gate_fn drives iteration; threshold/epsilon only feed the engine's
            # reason/converged reporting — pass the composite (0-100) scale so
            # those fields are meaningful (engine defaults are a 0-5 scale).
            threshold=composite_target,
            epsilon=0.25,
            id_extractor=lambda ls: frozenset(str(line["id"]) for line in ls),
            max_passes=max_passes,
            stage_name=self.name,
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
        )
        return result.draft
