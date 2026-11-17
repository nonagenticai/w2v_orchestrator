"""reconcile_timeline — the master-timeline stage (A22 / contract C13).

The single owner of the film's per-beat second budget. It collects every timing
owner's ``TimeDemand`` (VO word-count ÷ wpm band, card read-time floors, the
ffprobe'd length of a recorded footage clip — pinned, elasticity 0), reconciles
them against the deliverable's ``target_duration_seconds`` with the deterministic
water-filling solver in ``shared/time_budget.py``, and stamps the resulting
allotment back onto the ``lines`` spine (``seconds`` + a clamped ``vo_rate`` for
narrate, plus the ``[min_seconds, max_seconds]`` bounds the compile tail honours).

**DETERMINISTIC + ZERO-LLM by default.** Unless ``direction.timing.relief.mode ==
"auto"`` AND the timeline is INFEASIBLE (``Σmin > target``), this stage never calls
worker_management. When BOTH hold, a bounded WM evaluator→optimizer loop
(``pacing-critic`` + ``timing-director``) SHORTENS over-long narration text in place
(id-preserving), the deterministic solver re-verifies feasibility, and honest
structural relief — a beat that is claim-bearing / locked / footage-pinned and so
cannot be shortened — is surfaced as a SOFT ``rerun_stages=["demo_video_gen"]`` hint,
never faked.
Every number comes from the pure solver; the stage's only other job is to be
HONEST about what the solver cannot fix. Three conditions are appended to
``notes`` (which auto-marks the run ``degraded``, see ``stages/base.py``
``StageOutputWithNotes``) instead of being silently swallowed:

* **infeasible** (``Σmin > target``) — the beats' floors already overrun the target.
  Arithmetic cannot relieve this: seconds can only be moved, not created.
* **underfull** (``Σallotment < target`` beyond tolerance) — even at every beat's
  ``max`` hold the content cannot fill the target. The solver does NOT invent
  seconds past a beat's ``max``; it reports the shortfall. A note for a mildly-short
  film — but below ``pipeline.demo_timeline_underfull_fail_ratio`` x target (default
  50%) the shortfall stops being a preference and the stage HARD-FAILS the run rather
  than ship a different deliverable from the one ordered (kill-switch: set the ratio to
  ``0.0``). The note is emitted in both cases.
* **unsayable** — a beat whose ``vo_rate`` had to clamp at the band ceiling: even at
  max speed the words overrun the beat.

C0: the stage re-emits the exact ``demo_video_gen`` shape (:class:`DemoVideoGenOutput`)
by merging patches onto the ORIGINAL ordered draft by id — it never adds, drops,
reorders, or re-clips a line.

SOFT gate: infeasibility, an ``unsayable`` VO rate, or drift outside tolerance NEVER
block the deliverable — the best timeline is always returned, flagged. The ONE exception
is gross underfill (above): a timeline filling less than
``pipeline.demo_timeline_underfull_fail_ratio`` of its target fails the run.

Presence-is-opt-in: with no ``direction.timing`` block, ``mode == "off"``, or no
resolvable target duration, the stage raises :class:`StageSkippedError` — a run that
did not opt in is byte-identical to before.

The WM relief loop (C13) is OPT-IN and INERT-BY-DEFAULT: with no
``direction.timing.relief.mode == "auto"`` block, or a feasible timeline, the code path
is byte-identical to the deterministic stage — no WM call, no manifest note, no extra
fields. When it does run it follows the spec-§5 evaluator-optimizer shape: the
``timing-director`` worker is a DIRECTIVE PROPOSER, not a text rewriter — given the
over-long beats (id + text + word_count + target) it emits ``{op, target_ids,
new_word_target}`` intent (an easy classification task even a small model handles),
never the rewritten prose. The stage DISPOSES each directive deterministically: a
``trim`` fires one focused ``beat-rewriter`` call per targeted beat that actually
shortens ``line["text"]`` to the proposed word target (the small, in-reach task), and
the solver re-solves against the shortened copy to VERIFY feasibility — a candidate that
does not actually restore feasibility is rejected (spec Acceptance #10). This
decomposition of one hard batched rewrite into (1 classification) + (N focused per-beat
rewrites) is what makes the loop tractable on the single small local model; an optional
``PIPELINE_TIMING_DIRECTOR_MODEL_TIER`` routes both calls to a more capable tier when it
is not. It NEVER uses a seconds-bound trick that would leave the solver reading a
FICTITIOUS word count off unmodified text — that lie is exactly what the deferred design
warned against: a beat bounded to 10 words' seconds while the line still holds 40 words
makes narrate speak 40 words in 10 words' time, ``vo_rate`` clamps at 2.0, the beat is
``unsayable``, and the timeline is internally consistent and a lie. Structural ops
(``merge`` / ``cut`` / ``reallocate``) change the id-set or the budget, so they are NOT
disposed inside the trim optimizer loop (``_dispose_directives`` STAYS trim-only —
removing an id mid-loop would trip the loop engine's C0 id-set guard). C13 phase 2 runs
them as a deterministic PRE-PASS (``_apply_structural_prepass``) BEFORE the trim loop:
one ``timing-director`` call proposes structural ops over the currently-infeasible spine,
the pre-pass applies them (``cut`` drops a beat, ``merge`` fuses two, ``reallocate`` pins
one), and the deterministic timeline is re-solved so the loop's ``_spine_ids`` derive
from the REDUCED spine and the C0 guard never trips. The pre-pass is armed only when at
least one infeasible beat CANNOT be trimmed (claim-bearing / locked / footage-pinned) —
structural surgery is a last resort when in-place shortening cannot reach feasibility, not
a first move. A ``cut`` / ``merge`` that would drop a protected beat is rejected-with-log
and never applied; if structural relief still cannot make the timeline honest, the beat is
flagged for an upstream ``demo_video_gen`` re-run rather than faked feasible.

**Resume determinism.** The deterministic OFF-path is bit-reproducible. The relief loop
is NOT: the ``timeline_manifest`` note it appends carries LLM-shortened copy + a pacing
score, and the content-addressed cache key (``time_budget.timeline_manifest``) hashes only
the deterministic ``(target, demand-vector)`` — so a resume that re-enters an armed +
infeasible timeline re-runs the WM workers and MAY land different text. This is an accepted
property of an LLM-in-the-loop stage (it cannot be byte-reproducible), bounded by the ≤2-pass
cap; only the OFF-path carries the byte-identical guarantee.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Final, Literal, NamedTuple, cast

from w2v_common.stages import stage_call

from src.auth.run_providers import current_run_model_tier
from src.config import get_settings
from src.services.llm import WMClient
from src.services.runs.structural_relief_policy import (
    RELIEF_NOT_RUN_OUTCOME,
    RELIEF_OFF_OUTCOME,
    decide_structural_relief,
    relief_mode_of_timing,
    relief_shape_from_manifest,
)
from src.services.stages.base import SlotUse, StageHandler, StageInput, StageSkippedError
from src.services.stages.demo_video.gen import DemoVideoGenOutput
from src.services.stages.demo_video.narrate import DemoVideoNarrateStage
from src.services.stages.demo_video.narration_fit import (
    FIRST_TAKE_SPEED,
    FIT_SYSTEM,
    FitTarget,
    fit_message,
    fit_summary,
    fit_targets,
    fitted_line,
    measured_demands,
    measured_underspoken_note,
    needs_recount,
    recount_message,
    word_miss,
)
from src.services.stages.demo_video.spine import resolve_plan_output
from src.services.stages.generic.line_fit import LineFit, LineTake, fit_lines
from src.services.stages.llm_budget import (
    LARGE_ANSWER_TOKENS,
    LLM_RETRY_ON,
    MEDIUM_ANSWER_TOKENS,
    SMALL_ANSWER_TOKENS,
    downshift_reasoning,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
    reasoning_for_attempt,
    timing_director_model_tier,
)
from src.services.stages.primitives.dialogue import voice_for_speaker
from src.services.stages.primitives.narration import synthesize_and_store_clip
from src.services.stages.shared import speech_rate as _speech_rate
from src.services.stages.shared.authored_storyboard import authored_lines
from src.services.stages.shared.beat_grounding import (
    MAX_UNGROUNDED_WORDS as _MAX_UNGROUNDED_WORDS,
)
from src.services.stages.shared.beat_grounding import (
    beat_source as _beat_source,
)
from src.services.stages.shared.beat_grounding import (
    uncovered_steps as _uncovered_steps,
)
from src.services.stages.shared.beat_grounding import (
    ungrounded_words as _ungrounded_words,
)
from src.services.stages.shared.degradation import (
    AUTHORED_SPEECH_RATE_OVERRIDE,
    DURATION_OUT_OF_TOLERANCE,
    FOOTAGE_TRIMMED,
    FREEZE_PAD_CEILING_EXCEEDED,
    NARRATION_UNSAYABLE,
    ORDERED_DURATION_INFEASIBLE,
    TIMELINE_CANT_CONVERGE,
    TIMELINE_OFF_TARGET,
    TIMELINE_RELIEF_UNAVAILABLE,
    TIMELINE_UNDERFULL,
    TIMELINE_UNDERSPOKEN,
    Degradation,
    degradation,
)
from src.services.stages.shared.duration_conformance import (
    # "finite float, or None" — the ONE coercion this repo already trusts for an ordered
    # duration read off unvalidated config. Imported rather than restated: a second
    # implementation of the same predicate is how the two lines drift apart.
    _usable as finite_or_none,
)
from src.services.stages.shared.loop_engine import Critique, run_evaluator_optimizer
from src.services.stages.shared.speakable_text import contains_url, speakable_text
from src.services.stages.shared.speech_rate import (
    CALIBRATION_PASSAGE,
    FallbackReason,
    RateBasis,
    ScriptSpeechRate,
    SpeechRate,
    authored_natural_wpm,
    resolve_cast_wpm,
    resolve_effective_wpm,
    resolve_natural_wpm_detail,
    script_sample_degradation,
    speech_rate_fallback_degradation,
    spoken_text_digests,
    spoken_texts,
)
from src.services.stages.shared.time_budget import (
    DEFAULT_FREEZE_PAD_CEILING,
    DEFAULT_TOLERANCE_PCT,
    DEFAULT_WPM_BAND,
    DEFAULT_WPM_NATURAL,
    MAX_PAD_CREDIT_RATIO,
    NARRATION_BUDGET_RATE,
    VO_RATE_BAND,
    VO_RATE_DRAG_FLOOR,
    VO_RATE_NATURAL_FLOOR,
    Allotment,
    TimeDemand,
    TimelineManifest,
    budget_drift,
    build_timeline_manifest,
    collect_demands,
    content_drift,
    content_floor_credit,
    footage_cuts,
    freeze_pad_ceiling,
    infeasible_beats,
    projected_segment_seconds,
    tolerance_seconds,
)
from src.services.stages.shared.time_budget import (
    # The module-level water-filling solver shares its name with this stage — alias it
    # so the stage body stays readable.
    reconcile_timeline as solve_timeline,
)
from src.services.stages.shared.title_cards import title_card_seconds
from src.services.stages.upstream import PermanentUpstreamError, guarded
from src.services.stages.verbatim import partition_for_stage
from src.services.stages.website_review.script import floor_wpm
from src.utils import as_float, clamp_score, opt_float

# The stage a structural-relief re-run restarts from — named in the degrade note when the
# user is the only remaining lever. Mirrors ``structural_relief._FROM_STAGE``.
_RELIEF_RERUN_STAGE = "demo_video_gen"
# The prefix of the user-facing ``cant_converge`` note — ONE definition, so the post-re-ask
# refresh (:meth:`ReconcileTimelineStage._refresh_cant_converge_note`) finds the note it
# must replace by the same string that wrote it.
_CANT_CONVERGE_NOTE_HEAD = "reconcile_timeline: STRUCTURAL RELIEF NEEDED"

if TYPE_CHECKING:
    from src.clients.tts_client import TTSClient
    from src.services.runs.run_context import RunContext


log = logging.getLogger(__name__)

# Honest convergence outcome of the relief loop's re-solve (surfaced on the typed manifest):
# feasible; still-infeasible-but-shortenable; or still-infeasible-and-every-over-budget-beat
# -protected (a bounded gen re-run is the only remaining lever — the degrade case).
_ConvergenceOutcome = Literal["converged", "cant_converge", "progress"]


def _as_list(value: object) -> list[object]:
    """Read-boundary coercion for LLM/generator output: a list only if it is one."""
    return list(value) if isinstance(value, (list, tuple)) else []


# WM worker names (regex ^[a-zA-Z0-9_-]+$). WM auto-creates them on first call; the
# provider/model behind each lives on the WM worker config, not here.
_CRITIC_WORKER = "pacing-critic"
_OPTIMIZER_WORKER = "timing-director"
_REWRITER_WORKER = "beat-rewriter"
# The lengthen twin of `beat-rewriter`. A SEPARATE worker rather than a second prompt on
# the same one so the two opposite tasks can be routed to different models in WM config
# without touching this file.
_EXPANDER_WORKER = "beat-expander"

_CRITIC_SYSTEM = (
    "You are a rhythm/pacing evaluator for short-form product video. You are given a "
    "reconciled narration timeline — each beat has a stable id, its narration text, and "
    "the deterministic per-beat second budget the solver assigned. Score 1-5 whether the "
    "timeline reads as a DESIGNED rhythm: no run of rushed beats crammed below a natural "
    "delivery rate, the VO rate sits in a natural band, and no draggy over-long hold "
    "stalls momentum. Judge STRICTLY IN PLACE by id: name how each beat felt and the "
    "second-delta that would fix it, but NEVER rewrite, add, drop, merge, or reorder a "
    "beat — you only score and advise."
)
_OPTIMIZER_SYSTEM = (
    "You are a timing director. The reconciled timeline is INFEASIBLE: some narration "
    "beats hold more words than their per-beat time budget allows. You are given each "
    "over-long beat's id, its current text, and its word_count, plus the target. Do NOT "
    "rewrite any text. Instead PROPOSE DIRECTIVES that would make the timeline fit: for "
    "each beat that must get shorter, emit "
    '{"op": "trim", "target_ids": ["<that beat\'s id>"], '
    '"new_word_target": <a whole number of words small enough to fit>, '
    '"rationale": "<why>"}. Echo ids EXACTLY as given and target ONLY the ids you '
    "were given — never invent an id and never target a beat that was not provided. "
    "Return one directive per beat that needs shortening; a beat that already fits needs "
    "no directive. You decide how much shorter each beat must be; you do not write the "
    "new words."
)
_REWRITER_SYSTEM = (
    "You rewrite ONE narration beat to be shorter while preserving its meaning. You are "
    "given the beat's current text and a maximum word count. Return ONLY the rewritten "
    "narration at or under that word count — no id, no quotes, no preamble, no "
    "explanation. Never introduce a claim, statistic, or superlative the original did "
    "not already make; say the same thing in fewer words."
)
_EXPANDER_SYSTEM = (
    "You rewrite ONE narration beat to be LONGER, because it currently has too few "
    "words for the seconds it must fill. You are given the beat's current text and a "
    "MINIMUM word count. Return ONLY the rewritten narration at or just above that word "
    "count — no id, no quotes, no preamble, no explanation. Keep every fact the original "
    "states and keep its subject the same. ⛔ NEVER introduce a claim, statistic, "
    "number, feature, or superlative that neither the original nor the recorded steps "
    "you are given contain. When you are told which recorded steps the beat does not "
    "narrate yet, lengthen it by narrating THOSE steps, in order — that is the "
    "material, not new information. Otherwise elaborate on what is already said "
    "(describing the same action more fully). Do NOT pad with filler, "
    "repetition, or restatement of the same sentence; write prose a narrator can read "
    "aloud without it sounding stretched. ⛔ NEVER write a URL, web address, domain or "
    "file path — the voice spells those out character by character. Refer to a site or "
    "page by its name instead (e.g. 'the TodoMVC demo'), even if the original text "
    "quotes its address."
)

# The pacing-critic output schema. ``overall_score`` (1-5) drives the feasible-band
# score; ``per_beat`` is advisory (how each beat felt + the delta that would fix it).
_CRITIC_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["overall_score"],
    "properties": {
        # No `minimum`/`maximum`: vLLM guided decoding constrains grammar and type
        # only, so a numeric bound never stops a bad value being generated — it only
        # fails post-hoc validation, and WM runs inference_router with retries=0, so
        # that failure permanently kills the stage. State the range for the model and
        # clamp it on read instead (see `clamp_score` at the consumption site).
        "overall_score": {"type": "integer", "description": "1-5, where 5 is strongest"},
        "per_beat": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id"],
                "properties": {
                    "id": {"type": "string"},
                    "felt": {"type": "string"},
                    "suggested_delta_seconds": {"type": "number"},
                },
            },
        },
        "directives": {"type": "array", "items": {"type": "object"}},
    },
}

# The trim-optimizer DIRECTIVE schema (spec 23 §5). The worker proposes INTENT
# (``op`` + ``target_ids`` + optional ``new_word_target``) — never the rewritten prose.
# This is the schema for the in-place TRIM loop (``_relief_loop``'s ``optimizer_fn``),
# which disposes ONLY ``trim`` (id-stable, via a per-beat ``beat-rewriter`` call). The enum
# is deliberately trim-ONLY: ``merge`` / ``cut`` / ``reallocate`` change the id-set or the
# budget and are handled by the separate deterministic structural pre-pass
# (``_apply_structural_prepass`` + ``_STRUCTURAL_SCHEMA`` below), NOT by this loop — so
# advertising them here only lets the director waste a pass on a directive the trim loop
# rejects (``_dispose_directives`` → ``unsupported``).
_OPTIMIZER_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["directives"],
    "properties": {
        "directives": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["op", "target_ids"],
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["trim"],
                    },
                    "target_ids": {"type": "array", "items": {"type": "string"}},
                    "new_word_target": {
                        "type": "integer",
                        # NB: no numeric ``minimum`` — a JSON-schema bound is unenforceable
                        # by vLLM guided decoding and turns an out-of-range value into a
                        # fatal validation outage (see tests/test_output_schema_bounds.py).
                        # The ">= 1" floor is documented here and ENFORCED at the consumer
                        # in ``_parse_trim_directives`` (a non-positive target is discarded
                        # and the default-ratio floor of 1 is used instead).
                        "description": (
                            "the whole number of words to shorten each targeted beat to; "
                            "a positive integer smaller than the beat's current word_count"
                        ),
                    },
                    "rationale": {"type": "string"},
                },
            },
        }
    },
}

# The STRUCTURAL DIRECTIVE schema (C13 phase 2). Paired with ``_STRUCTURAL_SYSTEM`` for the
# deterministic structural pre-pass (``_apply_structural_prepass``), which runs BEFORE the
# trim loop over an infeasible spine whose over-budget beats cannot be shortened in place.
# The enum is cut/merge/reallocate ONLY — ``trim`` is NOT a structural op (it is the trim
# loop's job) and ``_apply_structural_ops`` rejects any non-structural op. Sharing the
# trim-optimizer schema here previously advertised ``trim`` to this director (and, being
# first in the enum, biased guided decoding toward it), so it would periodically emit a
# ``trim`` that the pre-pass silently rejected — a paid WM pass spent for zero relief.
_STRUCTURAL_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["directives"],
    "properties": {
        "directives": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["op", "target_ids"],
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["cut", "merge", "reallocate"],
                    },
                    "target_ids": {"type": "array", "items": {"type": "string"}},
                    "seconds": {
                        "type": "number",
                        # C13 phase 2 ``reallocate``: the pinned duration in seconds for the
                        # target beat. No numeric bound (same guided-decoding rationale as
                        # ``new_word_target``); the value is CLAMPED into the beat's pre-pin
                        # ``[demand.min, demand.max]`` at the consumer in the structural
                        # pre-pass so it can never invert the solver's ``_repair`` floor.
                        "description": (
                            "for a reallocate op: the duration in seconds to pin the target "
                            "beat at (clamped into the beat's feasible band on apply)"
                        ),
                    },
                    "rationale": {"type": "string"},
                },
            },
        }
    },
}

# The per-beat ``beat-rewriter`` schema — a single shortened text. Kept trivially small
# so the local model reliably returns it; a bare-string output_type is also accepted.
_REWRITER_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["text"],
    "properties": {"text": {"type": "string"}},
}

# When ``timing-director`` proposes a ``trim`` with no (or an invalid) ``new_word_target``,
# fall back to this fraction of the beat's current word count so the rewrite still attempts
# a real reduction; the deterministic re-solve is the actual feasibility gate, so an
# imperfect target simply means the loop may need its second pass or reject the candidate.
_DEFAULT_TRIM_RATIO = 0.75

# LARGE answer: ``_CRITIC_SCHEMA``'s ``per_beat`` carries ONE row per beat of the
# draft it is scoring — a list whose length tracks the input spine (which can run to
# dozens of beats), each row with an id, a ``felt`` phrase and a delta. That is the
# "list that grows with the input" tier, even though the score itself is one integer.
_CRITIC_MAX_TOKENS = output_token_budget(answer_tokens=LARGE_ANSWER_TOKENS)

# MEDIUM answer: ``_OPTIMIZER_SCHEMA`` returns INTENT, never prose — trim directives
# (op + target_ids + a word target + a one-line rationale) over the ELIGIBLE
# infeasible beats only, which is a subset of the spine. A handful of small objects.
_OPTIMIZER_MAX_TOKENS = output_token_budget(answer_tokens=MEDIUM_ANSWER_TOKENS)

# MEDIUM answer: ``_STRUCTURAL_SCHEMA`` is the same directive shape (cut / merge /
# reallocate + target_ids + seconds + rationale) over the infeasible beats the trim
# loop cannot reach — again intent, not rewritten narration.
_STRUCTURAL_MAX_TOKENS = output_token_budget(answer_tokens=MEDIUM_ANSWER_TOKENS)

# SMALL answer: ``_REWRITER_SCHEMA`` is ``{"text": ...}`` — ONE shortened narration
# beat, called once per trimmed beat. The reply is a single sentence or two.
_REWRITER_MAX_TOKENS = output_token_budget(answer_tokens=SMALL_ANSWER_TOKENS)

# The per-beat ``beat-expander`` contract — the exact mirror of the rewriter's, and
# deliberately the SAME trivial ``{"text": ...}`` shape so the local model handles it as
# reliably. A distinct WORKER name (not a distinct schema) is what separates the two, so
# the shorten and lengthen tasks can be pointed at different models in WM config without
# a code change.
_EXPANDER_SCHEMA: dict[str, object] = _REWRITER_SCHEMA
_EXPANDER_MAX_TOKENS = output_token_budget(answer_tokens=SMALL_ANSWER_TOKENS)

#: The delivery rate the re-ask AIMS at: 1.0 is natural pace, i.e. the beat is asked for
#: the number of words its allotment holds when nobody is dragged or hurried.
#:
#: ⛔ It is deliberately NOT :data:`VO_RATE_NATURAL_FLOOR`. Asking for the floor count is
#: asking for exactly the number of words that fills the beat AT the 0.80 floor, so
#: ``vo_rate_for`` clamps the result right back to 0.80 — a beat topped up to the floor is
#: BY CONSTRUCTION a beat that gets clamped to the floor. Measured live on run 01a0a136:
#: the generator returned 14 words against a 24-word budget on every beat, the re-ask
#: topped them to 21/26/21 (the floor counts), and the stage still recorded "clip-4 needs
#: speed 0.746 … clamped to 0.80". The floor keeps its job as the ACCEPT threshold (it is
#: what :func:`~...time_budget.vo_rate_for` actually tests); it is no longer the ask.
#:
#: ⛔ It IS :data:`NARRATION_BUDGET_RATE`, imported — never a restated literal. Both name
#: the same quantity: the rate a beat's word count is sized at. The generator is ORDERED
#: ``window * wpm * NARRATION_BUDGET_RATE / 60`` words; the re-ask tops a short beat up to
#: the same arithmetic. A hardcoded ``1.0`` here would silently keep re-asking at the old
#: rate the day the budget rate moves, chasing a count the generator was never told to
#: write. Pinned by ``test_expand_ask_rate_is_the_narration_budget_rate``.
_EXPAND_ASK_RATE = NARRATION_BUDGET_RATE

#: How far past the ASKED word count a re-ask may land before the result is refused.
#: A rewrite that blows far past the ask converts an underspoken beat into an
#: ``unsayable`` one — trading the "narration drags" complaint for the "narration is
#: rushed / audio overruns the clip" complaint, which is strictly worse because the
#: compiler cannot cut audio. 1.6x leaves real headroom for a model that overshoots a word
#: target (they routinely do) while still refusing a reply that is a different length of
#: text entirely.
#:
#: It scales off the ASK, not the floor, and it has to: the ask is ``floor /
#: VO_RATE_DRAG_FLOOR`` = 1.03x the floor (it was 1.25x while the floor was 0.80), so a
#: band held at 1.6x the FLOOR would refuse a reply that landed past a target we
#: ourselves set. The headroom only grew when the floor rose. Against the ask it
#: still cannot reach ``unsayable``: ``unsayable`` starts above ``VO_RATE_BAND[1]``/``rate``
#: = 2.0x the ask's count, and 1.6 < 2.0.
_EXPAND_CEILING_RATIO = 1.6

#: How many re-asks ONE beat may cost. The first pass was a single shot, which meant a
#: model that made real but partial progress (the common case — they under-deliver on a
#: word target) left the beat short and flagged with no second chance. The loop continues
#: only while the beat is still under its ask AND the previous pass actually ADDED words,
#: so it cannot cycle: every extra call is paid for by measured progress.
_EXPAND_MAX_PASSES = 3


def _reask_word_ceiling(word_ask: int, window_seconds: float, wpm_natural: float) -> int:
    """The most words a re-ask may return: the ratio ceiling AND the beat's own window.

    ``int(word_ask * _EXPAND_CEILING_RATIO)`` alone was not a bound on the BEAT. Run
    ``01a0b007``: clip-1 (14.56s clip, 136.06 wpm) was asked for 34 words, the ratio
    allowed 54, and the model's 41 were applied — 18.08s of speech against the
    14.56s + 1.50s freeze-pad window, i.e. the stage's OWN FREEZE-PAD ADVISORY fired on
    the text its own re-ask had just written.

    ``window_seconds`` is the allotment plus ``cfg.freeze_pad_ceiling_seconds`` — exactly
    the window :func:`_overpad_note` (via :func:`freeze_pad_ceiling`) tests the
    ``word_count / wpm_natural * 60`` estimate against. So no new constant: a reply this
    admits cannot, at the solved rate, raise that advisory. ``floor`` because the
    advisory fires on any strictly positive excess.
    """
    window_words = math.floor(window_seconds * wpm_natural / 60.0 + 1e-9)
    return min(int(word_ask * _EXPAND_CEILING_RATIO), window_words)


#: The underspoken re-ask may grow the WHOLE spine to at most this share over the body's
#: word budget. Measured: the per-beat ceiling alone let 8-word lines grow to 26-34 words,
#: and in 15 of 16 overshooting runs narration alone ran >10% over the order — so the
#: spine-level cap sits exactly at the edge those runs crossed. Same 1.10 as gen's
#: reported word ceiling (``narration_s x content wpm / 60 x 1.10``).
_REASK_TOTAL_WORD_RATIO = 1.10


def _reask_total_word_cap(body_seconds: float, content_wpm: float) -> int:
    """Most words the WHOLE spine may carry after an underspoken re-ask.

    ``body_seconds`` is the order minus the capped title cards (``cfg.target``);
    ``content_wpm`` is the CONTENT-basis rate (:func:`content_wpm_for`), never a raw
    calibration-passage rate, which reads ~47% faster than narration and would let the
    re-ask write a script far longer than the order.
    """
    return math.floor(body_seconds * content_wpm / 60.0 * _REASK_TOTAL_WORD_RATIO + 1e-9)


def content_wpm_for(wpm_natural: float, basis: object) -> float:
    """The content-basis rate — website_review's :func:`floor_wpm`, as D3 uses it in gen.

    A calibration-passage rate (or an UNKNOWN basis — the conservative reading) is held
    to at most :data:`DEFAULT_WPM_NATURAL`; content / authored rates pass through.
    """
    return floor_wpm(wpm_natural, basis if isinstance(basis, str) else CALIBRATION_PASSAGE)


def _total_capped_ceilings(
    growable: Mapping[str, tuple[int, int]], fixed_words: int, total_cap: int
) -> dict[str, int]:
    """Shrink each re-ask beat's word ceiling proportionally so the SPINE fits ``total_cap``.

    ``growable`` is ``{beat_id: (has, ceiling)}`` for the beats about to be re-asked;
    ``fixed_words`` is every word on the spine that is NOT one of those beats' current
    text. The room left, ``total_cap - fixed_words - Σhas``, is shared out in proportion
    to each beat's requested growth (``ceiling - has``), floored to whole words — so
    ``fixed_words + Σceiling' <= total_cap`` holds by construction. A beat whose share
    rounds to nothing comes back at ``has`` (the caller skips it). Unchanged when the
    requested growth already fits.
    """
    has_total = sum(has for has, _ in growable.values())
    room = max(total_cap - fixed_words - has_total, 0)
    wanted = {bid: max(ceiling - has, 0) for bid, (has, ceiling) in growable.items()}
    total_wanted = sum(wanted.values())
    if total_wanted <= room:
        return {bid: ceiling for bid, (_, ceiling) in growable.items()}
    scale = room / total_wanted if total_wanted else 0.0
    return {bid: has + math.floor(wanted[bid] * scale + 1e-9) for bid, (has, _) in growable.items()}


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


def _eligible_infeasible_lines(
    draft: list[dict[str, object]],
    target: float,
    wpm_band: tuple[float, float],
    wpm_natural: float,
    clip_seconds: dict[str, float],
) -> list[dict[str, object]]:
    """The infeasibility-contributing beats that are ELIGIBLE to rewrite, in spine order.

    A beat is eligible only when the solver flags it :func:`infeasible_beats` AND it can
    be honestly SHORTENED: it must NOT be claim-bearing (a truthy ``claim_status`` means
    the line carries a grounded/softened/pinned factual claim we must not paraphrase into
    an unsupported one), NOT ``locked`` by the author, and NOT footage-pinned (a beat
    whose demand ``kind == "footage"`` is pinned to its measured clip — shortening its
    text cannot move its seconds). Everything else is structural relief the loop cannot
    make, surfaced as a soft ``rerun_stages`` hint instead.
    """
    demands = collect_demands(
        draft,
        clip_seconds=clip_seconds,
        wpm_band=wpm_band,
        wpm_natural=wpm_natural,
        target=target,
    )
    flagged = set(infeasible_beats(demands, target))
    if not flagged:
        return []
    footage_ids = {d.id for d in demands if d.kind == "footage"}
    return [
        line
        for line in draft
        if str(line["id"]) in flagged
        and not line.get("claim_status")
        and not line.get("locked")
        and not line.get("time_lock")
        and str(line["id"]) not in footage_ids
    ]


# The shortest a trim may leave a beat. Below this the "fix" is gibberish, so a beat
# whose sayable ceiling is under it is NOT trim-eligible — it needs a gen re-run, and the
# ``_unsayable_note`` already on the stage's notes says so.
_MIN_SAYABLE_WORDS = 4


def _unsayable_trim_targets(
    draft: list[dict[str, object]],
    cfg: _TimingConfig,
    clip_seconds: dict[str, float],
) -> dict[str, int]:
    """Per-beat WORD target for every ``unsayable`` beat that can honestly be shortened.

    ``unsayable`` (:func:`~...time_budget.vo_rate_for`) means the narration overruns its
    own shot even at the top of ``VO_RATE_BAND`` — the audio will run past the picture.
    It is INDEPENDENT of :func:`infeasible_beats` (``Sigma min > target``): the live
    demo_video timeline is UNDERFULL and had 11 of 15 beats unsayable, so the trim loop,
    armed only on infeasibility, never fired on exactly the runs that needed it.

    ⚠️ Footage-pinned beats are deliberately NOT excluded here, unlike
    :func:`_eligible_infeasible_lines`. That exclusion is right for infeasibility (a
    pinned beat's SECONDS cannot move, so trimming its text buys no budget) and wrong
    here: the words are the only free variable, and cutting them is precisely what makes
    the beat sayable. Holding the shot longer instead re-creates the duplicate-frame
    defect the freeze-pad ceiling exists to bound.

    The target is arithmetic, not judgement — the inverse of the same rate check that set
    the flag: ``ceiling = window * wpm_natural * VO_RATE_BAND[1] / 60``, where ``window``
    is the beat's allotment plus its capped freeze-pad. Derived from the flag's own
    constants so the two CANNOT drift. Claim-bearing / author-locked / time-locked beats
    are excluded exactly as they are for infeasibility, and a beat whose ceiling is not
    strictly below its current word count is omitted — that keeps every pass a STRICT
    shortening, so the loop cannot cycle.
    """
    demands = collect_demands(
        draft,
        clip_seconds=clip_seconds,
        wpm_band=cfg.wpm_band,
        wpm_natural=cfg.wpm_natural,
        target=cfg.target,
    )
    allotments = solve_timeline(
        demands,
        target=cfg.target,
        tolerance_pct=cfg.tolerance_pct,
        wpm_natural=cfg.wpm_natural,
        freeze_pad_ceiling_seconds=cfg.freeze_pad_ceiling_seconds,
    )
    by_id = {str(line["id"]): line for line in draft}
    targets: dict[str, int] = {}
    for allotment in allotments:
        if not allotment.unsayable:
            continue
        line = by_id.get(allotment.id)
        if line is None:
            continue
        if line.get("claim_status") or line.get("locked") or line.get("time_lock"):
            continue
        window = allotment.seconds + allotment.freeze_pad_seconds
        ceiling = int(window * cfg.wpm_natural * VO_RATE_BAND[1] / 60.0)
        words = len(str(line.get("text", "")).split())
        if ceiling < _MIN_SAYABLE_WORDS or ceiling >= words:
            continue
        targets[allotment.id] = ceiling
    return targets


def _is_protected(line: dict[str, object], footage_ids: frozenset[str]) -> bool:
    """A beat a structural ``cut`` / ``merge`` may never DROP — mirrors eligibility.

    The same predicate :func:`_eligible_infeasible_lines` uses to exclude a beat from
    in-place trimming: a truthy ``claim_status`` (carries a grounded/softened/pinned
    factual claim we must not silently delete), an author ``locked`` flag, a hard
    ``time_lock`` (collect_demands pins it ``min==ideal==max`` INDEPENDENT of word count,
    so trimming its text cannot move its seconds — same as footage), or a footage-pinned
    id (its seconds are its measured clip — dropping it discards recorded footage).
    ``cut`` on such a beat, or a ``merge`` that would drop one, is rejected.
    """
    return (
        bool(line.get("claim_status"))
        or bool(line.get("locked"))
        or bool(line.get("time_lock"))
        or str(line["id"]) in footage_ids
    )


def _word_counts(lines: list[dict[str, object]]) -> dict[str, int]:
    """Per-id current narration word count — the ``len(text.split())`` the solver reads."""
    return {str(line["id"]): len(str(line.get("text", "")).split()) for line in lines}


def _parse_trim_directives(
    directives: list[object],
    allowed_ids: frozenset[str],
    word_counts: dict[str, int],
) -> tuple[dict[str, int], list[str], list[str]]:
    """Reduce a timing-director directive list to disposable per-beat trim targets.

    Returns ``(trims, dropped_ids, unsupported_ops)``:

    * ``trims`` maps each ELIGIBLE beat id to the word target its text must be shortened
      to — the directive's ``new_word_target`` when the model supplied a usable one, else
      ``_DEFAULT_TRIM_RATIO`` of the beat's current word count (floored at 1). When several
      directives touch the same id the SMALLEST (most aggressive) target wins.
    * ``dropped_ids`` are ``trim`` targets outside ``allowed_ids`` — a hallucinated /
      case-mismatched id, or a beat that is not flagged+eligible (claim-bearing / locked /
      footage-pinned). They are NEVER disposed (the spine is never grown or a protected
      beat touched), only recorded so a live no-op is diagnosable.
    * ``unsupported_ops`` are ``merge`` / ``cut`` / ``reallocate`` (and any unknown op):
      accepted by the schema but not yet disposed (they change the id-set / budget and
      trip the C0 guard) — recorded and rejected until a follow-up relaxes the guard.

    Pure + id-stable by construction: only existing, allowed ids ever enter ``trims``.
    """
    trims: dict[str, int] = {}
    dropped: list[str] = []
    unsupported: list[str] = []
    for raw_item in directives:
        if not isinstance(raw_item, dict):
            continue
        item = cast("dict[str, object]", raw_item)
        op = str(item.get("op", "")).strip().lower()
        target_ids = [str(t) for t in _as_list(item.get("target_ids"))]
        if op != "trim":
            unsupported.append(op or "<missing>")
            continue
        raw = item.get("new_word_target")
        explicit = (
            int(raw)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool) and int(raw) >= 1
            else None
        )
        for tid in target_ids:
            if tid not in allowed_ids:
                dropped.append(tid)
                continue
            target = (
                explicit
                if explicit is not None
                else max(1, round(word_counts.get(tid, 0) * _DEFAULT_TRIM_RATIO))
            )
            trims[tid] = min(trims.get(tid, target), target)
    return trims, dropped, unsupported


def _critic_msg(
    draft: list[dict[str, object]], demands: list[TimeDemand], target: float, director: object
) -> str:
    """Build the pacing-critic prompt: reconciled spine + deterministic timing evidence."""
    evidence = {
        "target_seconds": round(float(target), 3),
        "total_min_seconds": round(sum(d.min for d in demands), 3),
        "beats": [
            {
                "id": d.id,
                "kind": d.kind,
                "words": d.word_count,
                "min": round(d.min, 3),
                "ideal": round(d.ideal, 3),
                "max": round(d.max, 3),
            }
            for d in demands
        ],
    }
    return "\n\n".join(
        [
            f"Director intent: {director if director else 'none stated'}",
            "Reconciled narration spine (ordered, by id):",
            _lines_json(draft),
            "Deterministic timing evidence (seconds — ground your rhythm read in these):",
            json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False),
            "Score the reconciled timeline's rhythm 1-5. Judge in place by id; never rewrite.",
        ]
    )


def _beats_json(lines: list[dict[str, object]]) -> str:
    """Deterministic id+text+word_count projection of the over-long beats for a prompt."""
    return json.dumps(
        [
            {
                "id": str(line["id"]),
                "text": str(line.get("text", "")),
                "word_count": len(str(line.get("text", "")).split()),
            }
            for line in lines
        ],
        indent=2,
        ensure_ascii=False,
    )


def _optimizer_msg(eligible: list[dict[str, object]], target: float, overrun: float) -> str:
    """Build the timing-director prompt: the over-long beats + the honest overrun.

    Asks for DIRECTIVES (intent), not rewritten text — the disposer executes them.
    """
    return "\n\n".join(
        [
            "The reconciled timeline is INFEASIBLE: the beats' minimum time demands "
            f"exceed the {target:.2f}s target by {overrun:.2f}s. Propose trim directives "
            "so these over-long narration beats fit their time budget.",
            "For EACH beat that must get shorter, emit one directive "
            '{"op":"trim","target_ids":["<id>"],"new_word_target":<int>,"rationale":"..."}. '
            "Target only the ids below, echoed exactly; do NOT rewrite the text yourself.",
            "Over-long beats (id + current text + word_count):",
            _beats_json(eligible),
        ]
    )


_STRUCTURAL_SYSTEM = (
    "You are a timing director performing STRUCTURAL relief. The reconciled timeline is "
    "INFEASIBLE and at least one over-long beat CANNOT be shortened in place (it is claim-"
    "bearing, author-locked, or footage-pinned). You are given each over-long beat's id, "
    "text and word_count, plus which ids are protected. Propose STRUCTURAL directives that "
    "restore feasibility: "
    '{"op":"cut","target_ids":["<id>"]} to drop a beat, '
    '{"op":"merge","target_ids":["<id-a>","<id-b>"]} to fuse two beats into one, or '
    '{"op":"reallocate","target_ids":["<id>"],"seconds":<n>} to pin a beat\'s duration. '
    "NEVER cut a protected beat and NEVER merge in a way that drops a protected beat — you "
    "may keep a protected beat by merging another beat INTO it. Echo ids EXACTLY as given "
    "and target ONLY the ids provided; you do not rewrite any narration text."
)


def _structural_msg(
    infeasible_lines: list[dict[str, object]],
    protected_ids: frozenset[str],
    target: float,
    overrun: float,
    director: object,
) -> str:
    """Build the structural pre-pass prompt: the infeasible spine + the protected set.

    Reuses the ``timing-director`` worker but pairs ``_STRUCTURAL_SYSTEM`` with the
    structural-only ``_STRUCTURAL_SCHEMA`` (cut/merge/reallocate — never ``trim``). Ends with
    the same ``"Over-long beats (id + current text + word_count):"`` block :func:`_optimizer_msg`
    uses so a directive proposer reads the beats to operate on out of the prompt; names the
    protected ids explicitly so a well-behaved model never proposes dropping one.
    """
    return "\n\n".join(
        [
            f"Director intent: {director if director else 'none stated'}",
            "The reconciled timeline is INFEASIBLE: the beats' minimum time demands exceed "
            f"the {target:.2f}s target by {overrun:.2f}s, and some over-long beats CANNOT be "
            "shortened in place. Propose STRUCTURAL directives (cut / merge / reallocate) to "
            "restore feasibility.",
            f"Protected beats (never cut, never merge-away): {sorted(protected_ids)}.",
            "Over-long beats (id + current text + word_count):",
            _beats_json(infeasible_lines),
        ]
    )


@dataclass(slots=True)
class _StructuralState:
    """Mutable accumulator threaded through the per-op structural disposers.

    ``removed`` ids are dropped from the rebuilt spine; ``text_overrides`` carry a merged
    survivor's concatenated text; ``pins`` map a reallocated id to its clamped seconds;
    ``rejected`` records every op that was refused (protected / out-of-spine / malformed).
    """

    by_id: dict[str, dict[str, object]]
    order: dict[str, int]
    spine_ids: frozenset[str]
    footage_ids: frozenset[str]
    demand_by_id: dict[str, TimeDemand]
    removed: set[str] = field(default_factory=set)
    text_overrides: dict[str, str] = field(default_factory=dict)
    pins: dict[str, float] = field(default_factory=dict)
    rejected: list[str] = field(default_factory=list)

    def text_of(self, beat_id: str) -> str:
        """The beat's current text — a prior merge override wins over the original."""
        return self.text_overrides.get(beat_id, str(self.by_id[beat_id].get("text", "")))

    def live(self, beat_id: str) -> bool:
        """True while ``beat_id`` is a real, not-yet-removed spine id."""
        return beat_id in self.spine_ids and beat_id not in self.removed

    def opener_id(self) -> str | None:
        """The earliest-order live beat — the positional cold-open the pass must never drop.

        Only the gen.py ``_apply_target_beats`` lever protects index 0 today; this in-place
        structural pass has no such guard, so ``_dispose_cut`` / ``_dispose_merge`` read this
        to keep the opening beat. Returns ``None`` when nothing is live (a fully-removed spine).
        """
        live_ids = [bid for bid in self.order if self.live(bid)]
        if not live_ids:
            return None
        return min(live_ids, key=lambda bid: self.order[bid])


def _dispose_cut(state: _StructuralState, target_ids: list[str]) -> None:
    """Drop each unprotected, live target id; reject an out-of-spine or protected id."""
    for tid in target_ids:
        if not state.live(tid):
            state.rejected.append(tid)
            log.warning(
                "reconcile_timeline structural: cut targets an out-of-spine or already-"
                "removed id %s — ignored.",
                tid,
            )
        elif _is_protected(state.by_id[tid], state.footage_ids):
            state.rejected.append(tid)
            log.warning(
                "reconcile_timeline structural: cut targets a protected beat %s (claim-"
                "bearing / locked / footage-pinned) — rejected, not applied.",
                tid,
            )
        elif tid == state.opener_id():
            # The positional opener (cold-open) is protected here — gen.py's lever guards
            # index 0, but this in-place path must not silently drop the opening beat.
            state.rejected.append(tid)
            log.warning(
                "reconcile_timeline structural: cut targets the opening beat %s — rejected, "
                "not applied.",
                tid,
            )
        else:
            state.removed.add(tid)


def _merge_survivor_rank(state: _StructuralState, beat_id: str) -> tuple[int, float, int]:
    """Quality rank of one merge candidate — HIGHER wins (:func:`max`), fully deterministic.

    A merge drops exactly one of the two beats, and the survivor keeps its own ``clip`` /
    ``clip_anchor`` / ``clip_lead_seconds`` / ``expected_text`` / measured ``seconds`` — only
    the narration text of both sides is preserved (concatenated). Choosing the survivor by
    spine POSITION therefore threw away the better beat's non-text fields whenever the later
    beat was the better one. This ranks the pair on the signals that ARE available on the
    line dict + the solver's demand; it is used ONLY when neither side is protected (a
    protected side always survives, and both-protected is rejected upstream).

    Keys, in priority order:

    1. ``has_expected_text`` — the beat carries an O1b ``expected_text`` OCR assertion, i.e.
       it is the beat whose recorded screen is VERIFIED. Dropping it discards the pair's only
       wrong-screen check (the survivor's ``expected_text`` is what the record stage asserts),
       so it outranks every other signal.
    2. ``wanted_seconds`` — the solver's ``demand.ideal`` (falling back to the beat's measured
       ``seconds`` when no demand was collected for the id): the beat the solver wants to give
       more time to carries more of the timeline's substance.
    3. ``-order`` — the spine index NEGATED so the EARLIER beat wins. This is the historic
       positional rule, demoted to a pure tiebreak: with the quality keys equal the survivor
       is exactly the beat the pre-quality code kept, so a pair carrying no distinguishing
       signal behaves identically to before. It also guarantees a total order (spine indices
       are unique), so the choice is never arbitrary.
    """
    line = state.by_id[beat_id]
    has_expected_text = 1 if str(line.get("expected_text") or "").strip() else 0
    demand = state.demand_by_id.get(beat_id)
    if demand is not None:
        wanted_seconds = float(demand.ideal)
    else:
        raw_seconds = line.get("seconds")
        wanted_seconds = (
            float(raw_seconds)
            if isinstance(raw_seconds, (int, float)) and not isinstance(raw_seconds, bool)
            else 0.0
        )
    return (has_expected_text, wanted_seconds, -state.order[beat_id])


def _dispose_merge(state: _StructuralState, target_ids: list[str]) -> None:
    """Fuse two live beats: keep the protected one (else the better-ranked one), drop the other.

    Rejected when fewer than two live ids are named, when BOTH are protected, or when the pair
    couples the positional opener with a protected beat (the merge drops exactly one beat, and it
    may drop neither the cold-open nor a protected beat). With exactly one side protected the
    protected beat survives; with NEITHER protected the survivor is the higher
    :func:`_merge_survivor_rank` beat (verified-screen assertion > wanted seconds > earlier spine
    index). The survivor's text becomes both texts joined in reading (spine) order REGARDLESS of
    which side survived; it keeps its own footage/clip fields (it is the retained line).
    """
    # Dedupe while preserving spine order: a duplicate id (e.g. a small model echoing
    # ["a","a"]) must NOT self-merge into an unintended cut — it collapses to one distinct
    # live id and is rejected below like any under-specified merge.
    live = list(dict.fromkeys(tid for tid in target_ids if state.live(tid)))
    if len(live) < 2:
        state.rejected.extend(target_ids)
        log.warning(
            "reconcile_timeline structural: merge needs two DISTINCT live in-spine ids, "
            "got %s — rejected.",
            target_ids,
        )
        return
    if len(live) > 2:
        # A merge fuses exactly two beats; any beyond the first two are silently discarded by
        # ``live[:2]``. Record them as rejected (the same channel _dispose_cut uses) and log so
        # a live no-op is diagnosable — every no-op path in this module must surface.
        discarded = live[2:]
        state.rejected.extend(discarded)
        log.warning(
            "reconcile_timeline structural: merge named %d live ids %s — only the first two "
            "are fused; extra ids %s discarded (rejected, not merged).",
            len(live),
            live,
            discarded,
        )
    ordered = sorted(live[:2], key=lambda tid: state.order[tid])
    first_id, second_id = ordered[0], ordered[1]
    prot_first = _is_protected(state.by_id[first_id], state.footage_ids)
    prot_second = _is_protected(state.by_id[second_id], state.footage_ids)
    if prot_first and prot_second:
        state.rejected.extend([first_id, second_id])
        log.warning(
            "reconcile_timeline structural: merge(%s, %s) would drop a protected beat "
            "(both sides protected) — rejected, not applied.",
            first_id,
            second_id,
        )
        return
    # Opener-survival: the positional cold-open must never be the DROPPED side of a merge —
    # only gen.py's index-0 lever guards it, not this in-place path. ``ordered`` is sorted by
    # spine order, so the opener (global min order) is ``first_id`` whenever it is in the pair.
    # A merge drops exactly ONE beat, so it can drop neither the opener nor a protected beat: if
    # the opener is paired with a protected beat, keeping the opener would drop the protected one
    # (and vice versa) — the merge is unsatisfiable, so reject it, exactly like both-protected.
    opener_id = state.opener_id()
    if opener_id == first_id:
        if prot_second:
            state.rejected.extend([first_id, second_id])
            log.warning(
                "reconcile_timeline structural: merge(%s, %s) pairs the opening beat with a "
                "protected beat — cannot drop either, rejected, not applied.",
                first_id,
                second_id,
            )
            return
        survivor = first_id
    elif prot_first:
        survivor = first_id
    elif prot_second:
        survivor = second_id
    else:
        # Neither side is protected: pick on QUALITY, not position. Position survives only as
        # the final tiebreak inside the rank, so an undifferentiated pair keeps the old result.
        survivor = max((first_id, second_id), key=lambda tid: _merge_survivor_rank(state, tid))
    other = second_id if survivor == first_id else first_id
    state.text_overrides[survivor] = f"{state.text_of(first_id)} {state.text_of(second_id)}".strip()
    state.removed.add(other)


def _dispose_reallocate(
    state: _StructuralState, target_ids: list[str], raw_seconds: object
) -> None:
    """Pin each live target at ``raw_seconds`` clamped into its pre-pin ``[min, max]``."""
    if not isinstance(raw_seconds, (int, float)) or isinstance(raw_seconds, bool):
        state.rejected.extend(target_ids)
        log.warning(
            "reconcile_timeline structural: reallocate %s carried no numeric seconds — rejected.",
            target_ids,
        )
        return
    for tid in target_ids:
        demand = state.demand_by_id.get(tid)
        if not state.live(tid) or demand is None:
            state.rejected.append(tid)
            log.warning(
                "reconcile_timeline structural: reallocate targets an out-of-spine or "
                "already-removed id %s — ignored.",
                tid,
            )
            continue
        state.pins[tid] = min(max(float(raw_seconds), demand.min), demand.max)


def _rebuild_structural_lines(
    lines: list[dict[str, object]], state: _StructuralState
) -> list[dict[str, object]]:
    """Emit the post-structural spine: drop ``removed``, apply merged text + reallocate pins."""
    new_lines: list[dict[str, object]] = []
    for line in lines:
        lid = str(line["id"])
        if lid in state.removed:
            continue
        patched = dict(line)
        if lid in state.text_overrides:
            patched["text"] = state.text_overrides[lid]
        if lid in state.pins:
            patched["time_lock"] = True
            patched["seconds"] = state.pins[lid]
        new_lines.append(patched)
    return new_lines


def _apply_structural_ops(
    lines: list[dict[str, object]],
    directives: list[object],
    footage_ids: frozenset[str],
    *,
    clip_seconds: dict[str, float],
    wpm_band: tuple[float, float],
    wpm_natural: float,
) -> tuple[list[dict[str, object]], dict[str, float], list[str]]:
    """Apply cut / merge / reallocate directives to the spine — pure, id-changing.

    Returns ``(new_lines, reallocate_pins, rejected)``. A ``cut`` / ``merge`` on a protected
    beat, an out-of-spine or already-removed id, a malformed ``reallocate``, or an unknown op
    is appended to ``rejected`` and logged — never applied. The reallocate clamp reads each
    beat's PRE-pin demand so a pin can never invert the solver's ``_repair`` floor.
    """
    state = _StructuralState(
        by_id={str(line["id"]): line for line in lines},
        order={str(line["id"]): idx for idx, line in enumerate(lines)},
        spine_ids=frozenset(str(line["id"]) for line in lines),
        footage_ids=footage_ids,
        demand_by_id={
            d.id: d
            for d in collect_demands(
                lines, clip_seconds=clip_seconds, wpm_band=wpm_band, wpm_natural=wpm_natural
            )
        },
    )
    for raw_item in directives:
        if not isinstance(raw_item, dict):
            continue
        item = cast("dict[str, object]", raw_item)
        op = str(item.get("op", "")).strip().lower()
        target_ids = [str(t) for t in _as_list(item.get("target_ids"))]
        if op == "cut":
            _dispose_cut(state, target_ids)
        elif op == "merge":
            _dispose_merge(state, target_ids)
        elif op == "reallocate":
            _dispose_reallocate(state, target_ids, item.get("seconds"))
        else:
            state.rejected.extend(target_ids or [op or "<missing>"])
            log.warning(
                "reconcile_timeline structural: unknown op %r over %s — rejected.",
                op,
                target_ids,
            )
    return _rebuild_structural_lines(lines, state), dict(state.pins), state.rejected


def _rewrite_msg(text: str, word_target: int) -> str:
    """Build the beat-rewriter prompt: one beat's text + the word ceiling to hit."""
    return (
        f"Rewrite this narration beat to {word_target} words or fewer, preserving its "
        f"meaning. Return only the rewritten text.\n\n"
        f"Beat ({len(text.split())} words): {text}"
    )


def _expand_msg(
    text: str,
    word_target: int,
    word_ceiling: int | None = None,
    source: str | None = None,
    uncovered: Sequence[str] = (),
) -> str:
    """Build the beat-expander prompt: one beat's text + the word count to reach.

    ``word_target`` is the NATURAL-RATE count (:data:`_EXPAND_ASK_RATE`), not the
    ``underspoken`` floor — see that constant for why asking for the floor guarantees a
    clamp at the floor. The current count is restated from the text passed in, so a later
    pass of the bounded loop asks against what the beat ACTUALLY has now rather than
    against the original.

    ⛔ THE ASK STATES THE MISSING WORDS AS A DELTA and names "unchanged" as a failure.
    An absolute target alone ("AT LEAST 28 words") is a count the local model does not
    track: on showcase run 01a156a6 (2026-10-20) pass 2 handed back all three 20-21-word
    beats verbatim against asks of 24-28, the loop stopped on "no progress", and the run
    shipped ``narration_rate_floor_clamped`` + ``footage_trimmed``. Replayed live on
    ``local-llm`` (3 tries per beat): the absolute prompt returned +0 words 9/9; this
    delta prompt added +4 to +11 words 9/9, and a second pass on its output added more —
    which is what lets the bounded pass loop actually climb.

    ``source`` is what the beat's RECORDING actually performs (:func:`_beat_source`: the
    scenario name + its gherkin steps). Without it the only material the model has to
    lengthen with is its own imagination, and on the same run it wrote "refactor the
    entire component tree to eliminate redundant re-renders" over a TodoMVC clip that
    adds a todo. With it the prompt names those steps as the ONLY permitted content, and
    :func:`_ungrounded_words` rejects a reply that strays anyway. ``None`` (an authored
    beat with no scenario) keeps the prompt as it was.

    ``uncovered`` (:func:`uncovered_steps`) names the recorded steps the beat does NOT
    narrate yet and asks for the missing words to come from them. Without it the model
    could only "elaborate" — on showcase run 01a158a5-e826 it padded clip-4 with "the
    list of items currently displayed" (refused as ungrounded, 2 of 3 passes) while the
    clip's "mark it completed" step was never narrated.
    """
    has = len(text.split())
    grounding = (
        f"\n\nWhat the screen recording under this beat actually shows (its scripted "
        f"steps):\n{source}\n\nUse ONLY what the beat already says and what these steps "
        f"show — the page, the items typed, the clicks and what appears. Never mention a "
        f"feature, action, item, outcome or claim that is not in them."
        if source
        else ""
    )
    most = f" and AT MOST {word_ceiling} words" if word_ceiling is not None else ""
    todo = (
        "\n\nThe recording ALSO performs these steps, which the beat does not narrate "
        "yet. Get the missing words by narrating them, in this order, in plain spoken "
        "language (say what the viewer sees happen, not the step syntax):\n"
        + "\n".join(f"- {step}" for step in uncovered)
        if source and uncovered
        else ""
    )
    how = (
        "that narrates the not-yet-narrated recorded steps listed below, keeping what "
        "it already says"
        if todo
        else "that elaborates on what it already says, keeping its meaning and adding no new facts"
    )
    missing = max(word_target - has, 1)
    noun = "word" if missing == 1 else "words"
    return (
        f"This narration beat has {has} words, too few for the seconds it must fill. "
        f"Rewrite it to AT LEAST {word_target} words{most} — that is at least {missing} "
        f"more {noun} than it has now. Returning it unchanged, or reworded at the same "
        f"length, is a failure: lengthen it by adding one more short sentence or clause "
        f"{how}. Do not write any URL, web address or file path; name the site or page "
        f"instead. Return only the full rewritten text.{grounding}{todo}\n\n"
        f"Beat ({has} words): {text}"
    )


def _order_word_budgets(gen: Mapping[str, object]) -> dict[str, int] | None:
    """``demo_video_gen``'s ``narration_words_per_beat``, validated, or ``None``."""
    raw = gen.get("narration_words_per_beat")
    if not isinstance(raw, Mapping):
        return None
    out = {
        str(k): int(v)
        for k, v in raw.items()
        if isinstance(v, int | float) and not isinstance(v, bool) and v > 0
    }
    return out or None


def _ask_with_order_share(word_ask: int, word_ceiling: int, share: int | None) -> int:
    """Raise a re-ask to the beat's share of the ORDER, inside the beat's own window.

    ``share`` is gen's ``narration_words_per_beat`` for this beat — the order's word
    budget split over the beats that SURVIVED the plan. It only ever RAISES the ask
    (the footage allotment may already ask for more), and never past ``word_ceiling``,
    so it cannot produce an ask the window then refuses.
    """
    if share is None:
        return word_ask
    return max(word_ask, min(share, word_ceiling))


def _expandable_underspoken_lines(
    draft: list[dict[str, object]],
    targets: dict[str, tuple[int, int]],
) -> dict[str, tuple[int, int]]:
    """The underspoken beats a re-ask may honestly LENGTHEN: ``{id: (word_floor, has)}``.

    The mirror of :func:`_eligible_infeasible_lines`, and it excludes a DELIBERATELY
    DIFFERENT set. Claim-bearing (``claim_status``) and author-``locked`` beats are
    excluded for the same reason they are excluded from trimming — writing INTO a beat
    that carries a grounded factual claim is exactly how an unsupported one gets
    invented, and a locked beat is the author's.

    ⚠️ But ``time_lock`` and FOOTAGE-PINNED beats are **eligible here**, where trimming
    excludes them, and that asymmetry is the whole point rather than an oversight. A
    pinned beat's seconds cannot move — which is precisely WHY it is the one that must
    gain words: its allotment is fixed at the measured clip, so the only lever left on
    the underspoken side is the word count. Excluding footage here (by copying the trim
    predicate) would have made this pass inert on demo_video, whose spine is footage
    beats almost end to end.

    A beat with no narration text at all is skipped: a footage-only / card beat has
    nothing to elaborate ON, and asking a model to write a beat from nothing is
    generation, not repair.
    """
    by_id = {str(line["id"]): line for line in draft}
    eligible: dict[str, tuple[int, int]] = {}
    for beat_id, (word_floor, has) in targets.items():
        line = by_id.get(beat_id)
        if line is None:
            continue
        if line.get("claim_status") or line.get("locked"):
            continue
        if not str(line.get("text", "")).strip():
            continue
        eligible[beat_id] = (word_floor, has)
    return eligible


def _rewritten_text(output: object) -> str:
    """Extract the shortened text from a beat-rewriter response (dict ``text`` or bare str)."""
    if isinstance(output, dict):
        return str(cast("dict[str, object]", output).get("text") or "").strip()
    if isinstance(output, str):
        return output.strip()
    return ""


@dataclass(frozen=True, slots=True)
class _TimingConfig:
    """The resolved ``direction.timing`` block — every knob the stage honours.

    ``mode`` — auto (reconcile + apply) | reconcile_only (compute, don't mutate).
    ``off`` never reaches here: it is a skip (see :func:`_resolve_timing_config`).
    """

    mode: str
    target: float
    tolerance_pct: float
    wpm_band: tuple[float, float]
    wpm_natural: float
    freeze_pad_ceiling_seconds: float
    # The slowest rate a delivered line may run at — ``VO_RATE_DRAG_FLOOR`` unless the
    # author states one. The band is calibrated against ONE voice, so a different cast
    # wants a different number; see :data:`VO_RATE_DRAG_FLOOR` for why it is 0.97.
    vo_rate_drag_floor: float
    # The nested ``direction.timing.relief`` knobs — the OPT-IN WM relief loop.
    # ``relief_mode`` absent ⇒ "off" (the loop is fully inert). "auto" arms it.
    relief_mode: str
    relief_iterations: int
    relief_threshold: float
    # ``target`` above is the NARRATION BODY: the order minus the title cards the compile
    # splices in (owner ruling — cards count INSIDE the order, capped at 20% of it).
    # ``ordered_seconds`` is the order itself; ``None`` ⇒ same as ``target`` (no cards).
    ordered_seconds: float | None = None
    card_seconds: float = 0.0


def _resolve_timing_config(
    ctx: RunContext, *, default_natural: float = DEFAULT_WPM_NATURAL
) -> _TimingConfig:
    """Read ``direction.timing`` + the target ladder, or SKIP.

    Presence-is-opt-in: with no ``direction.timing`` block or ``mode == "off"`` there
    is nothing to reconcile. Target ladder: ``direction.timing.target_seconds`` (the
    explicit override) wins, else the deliverable's authored target; with NEITHER there
    is no budget to honour, so the stage skips clean rather than invent one.

    ``default_natural`` is the natural wpm to assume when the author states none — the
    rate MEASURED off the run's voice (``shared.speech_rate.resolve_natural_wpm_detail``),
    falling back
    to :data:`DEFAULT_WPM_NATURAL` when the probe cannot answer. Pure: the caller does
    the measuring.
    """
    direction = ctx.config_snapshot.get("direction") or {}
    timing = direction.get("timing") or {}
    if not isinstance(timing, dict) or not timing or timing.get("mode") == "off":
        raise StageSkippedError("reconcile_timeline: no direction.timing block")

    deliverable = ctx.config_snapshot.get("deliverable") or {}
    # THE BOUNDARY. ``config_overrides`` is unvalidated and round-trips whole into
    # ``direction``, and ``json.loads("1e400")`` is a legal JSON number that parses to
    # ``inf`` — so ``raw_target`` here can be non-finite from an authenticated request,
    # with no DB corruption required. ``float(raw_target)`` accepted it and every
    # downstream ``<= 0.0`` guard let it past (each comparison against ``nan`` is False)
    # until ``math.ceil`` raised inside the relief arithmetic, mid-stage.
    #
    # A non-finite order is NO order, so it gets the refusal an ABSENT target already
    # gets — not a zero-second order, and above all not a default: substituting one would
    # manufacture an order nobody placed, which is the exact defect this arithmetic
    # exists to delete.
    # The ladder itself is UNCHANGED — the override still wins outright, and an
    # unreadable override is never quietly demoted to the deliverable's target (that
    # would swap one order for a different one).
    raw_target = timing.get("target_seconds")
    if raw_target is None:
        raw_target = deliverable.get("target_duration_seconds")
    if raw_target is None:
        raise StageSkippedError("reconcile_timeline: no target duration to reconcile to")
    target = finite_or_none(raw_target)
    if target is None:
        raise StageSkippedError(
            f"reconcile_timeline: target duration {raw_target!r} is not a finite number"
        )

    wpm_band, wpm_natural = _wpm_settings(timing, default_natural=default_natural)
    relief = timing.get("relief") or {}
    if not isinstance(relief, dict):
        relief = {}

    # THE SAME BOUNDARY, APPLIED TO THE TUNABLES (wave 13, O9). Everything above this
    # line guarded exactly ONE field — ``target_seconds`` — while its four siblings were
    # read with a bare ``float()``/``int()`` off the SAME unvalidated blob, reached by the
    # same authenticated ``config_overrides`` path, and ``json.loads("1e400")`` is still a
    # legal JSON number. What that bought:
    #
    #   * ``freeze_pad_ceiling_seconds=1e400`` ⇒ ``inf``. It flows straight into
    #     ``underspoken_hold(..., ceiling)``: ``min(residual, inf)`` un-caps the held frame
    #     entirely, and ``underspoken_uncovered``'s ``residual - inf`` clamps to 0.0 — so
    #     the uncovered accounting reads CLEAN on exactly the run that broke it, and every
    #     footage beat's ``vo_rate`` window becomes unbounded.
    #   * ``tolerance_pct=inf`` ⇒ every drift is in-tolerance; ``nan`` ⇒
    #     ``tolerance_seconds`` returns 0.0, silently arming ``is_underfull`` at ZERO
    #     tolerance. Both are worse than a refusal because both look like a pass.
    #   * ``relief.iterations=1e400`` ⇒ ``int(inf)`` raises ``OverflowError`` (``nan`` ⇒
    #     ``ValueError``) UNCAUGHT, mid-stage, from a config read.
    #
    # ⛔ THE DISPOSITION IS NOT THE TARGET'S. A non-finite TARGET is refused, because
    # substituting a default would manufacture an order nobody placed. A non-finite
    # TUNABLE is different in kind: the default is a real, correct default that every run
    # without a ``direction.timing`` knob already uses, so falling back to it invents
    # nothing. Refusing the run instead would let one junk tuning field cancel a valid
    # order. So: guard with the SAME predicate, dispose differently, and say why here
    # rather than in four places.
    #
    # ``finite_or_none`` is REUSED, never restated — a second implementation of the same
    # predicate is how the two lines drift apart (see its import).
    def _tunable(value: object, default: float) -> float:
        resolved = finite_or_none(value)
        return default if resolved is None else resolved

    # CARDS COUNT INSIDE THE ORDER (owner ruling). The body is sized to what the order
    # leaves after the intro/outro cards, planned against the order so the SAME 20% cap
    # website_review reserves applies (``plan_title_cards`` is the one gate — reused, not
    # restated). Bed-tail run: 30s order + 6s outro sized the body to the full 30s and
    # delivered 37.6s.
    card_seconds = title_card_seconds(ctx.config_snapshot.get("title_card"), ordered_seconds=target)

    return _TimingConfig(
        mode=str(timing.get("mode", "auto")),
        target=target - card_seconds,
        ordered_seconds=target,
        card_seconds=card_seconds,
        tolerance_pct=_tunable(timing.get("tolerance_pct"), DEFAULT_TOLERANCE_PCT),
        wpm_band=wpm_band,
        wpm_natural=wpm_natural,
        freeze_pad_ceiling_seconds=_tunable(
            timing.get("freeze_pad_ceiling_seconds"), DEFAULT_FREEZE_PAD_CEILING
        ),
        # Clamped into ``[VO_RATE_NATURAL_FLOOR, 1.0]`` as well as finiteness-guarded: a
        # configured floor BELOW the pathological limit would re-open the very drag this
        # constant closes, and one ABOVE 1.0 would order the voice permanently hurried.
        # An author may tighten the band or relax it to the hard backstop — never past
        # either end.
        vo_rate_drag_floor=min(
            max(
                _tunable(timing.get("vo_rate_drag_floor"), VO_RATE_DRAG_FLOOR),
                VO_RATE_NATURAL_FLOOR,
            ),
            1.0,
        ),
        # Absent ``relief`` block ⇒ "off": the WM loop stays fully inert.
        # Read through the SAME helper the structural-relief spawner gates on, so the
        # loop arming here and the paid-spawn opt-in can never disagree.
        relief_mode=relief_mode_of_timing(timing),
        # ``int()`` on the GUARDED float, so ``inf``/``nan``/junk can no longer raise
        # ``OverflowError``/``ValueError`` out of a config read.
        relief_iterations=int(_tunable(relief.get("iterations"), 2.0)),
        relief_threshold=_tunable(relief.get("score_threshold"), 4.0),
    )


def timeline_bounds_were_ordered(ctx: RunContext) -> bool:
    """Whether this run ORDERED a reconciled timeline, read from the run's own config.

    This is the question ``demo_video_compile`` has to ask to know whether an absent
    ``[min_seconds, max_seconds]`` window is expected or impossible, and it cannot ask
    it of ``ctx.prior_output("reconcile_timeline")``: ``prior_run_outputs`` collapses
    "row absent", "skipped" and "completed with a falsy output" into ONE falsy value
    (``run_dispatcher`` seeds skips as ``{}``), so that call cannot distinguish the
    state where bounds were never ordered from the state where they were ordered and
    went missing. Those are the two states a silent-short guard has to tell apart, and
    the incident it exists for lives in the second one.

    ``config_snapshot`` has no such collapse — it is the order itself, present on every
    context, and unchanged by anything reconcile_timeline did or failed to do.

    Deliberately delegates to :func:`_resolve_timing_config` rather than restating its
    gate: a second copy of "no timing block / no target / non-finite target" is how the
    two answers drift apart, and a compile-side guard that disagreed with the stage's
    own skip condition would fire on exactly the runs it must stay quiet on.
    """
    try:
        _resolve_timing_config(ctx)
    except StageSkippedError:
        return False
    return True


def _wpm_settings(
    timing: dict[str, object], *, default_natural: float = DEFAULT_WPM_NATURAL
) -> tuple[tuple[float, float], float]:
    """Resolve the ``(min, max)`` wpm band + the natural wpm from ``direction.timing``.

    Accepts the spec's ``{"min": 130, "natural": 150, "max": 170}`` object as well as a
    bare ``[min, max]`` pair; anything else falls back to the shared defaults.

    ``default_natural`` is the rate to use when the author did NOT state one — the
    MEASURED rate for the run's voice when :mod:`~src.services.stages.shared.speech_rate`
    could measure it, else :data:`DEFAULT_WPM_NATURAL`. An authored ``natural`` always
    wins over it: an author who states a rate means it. Kept a PURE function — the
    measurement is performed by the async ``handle`` and threaded in.
    """
    raw = timing.get("wpm")
    if isinstance(raw, dict):
        wpm = cast("dict[str, object]", raw)
        low = as_float(wpm.get("min"), DEFAULT_WPM_BAND[0])
        high = as_float(wpm.get("max"), DEFAULT_WPM_BAND[1])
        authored_natural = as_float(wpm.get("natural"), -1.0)
        if authored_natural > 0.0:
            # AUTHORED natural: the measurement is not consulted at all and the band is
            # left exactly as it resolves today — byte-identical to main. An absent or
            # unparseable value (``-1.0`` sentinel) is NOT authored and falls through
            # to the measured default.
            return (low, high), authored_natural
        return _rescale_band((low, high), default_natural), default_natural
    if isinstance(raw, list | tuple) and len(raw) == 2:
        band = (
            as_float(raw[0], DEFAULT_WPM_BAND[0]),
            as_float(raw[1], DEFAULT_WPM_BAND[1]),
        )
        return _rescale_band(band, default_natural), default_natural
    return _rescale_band(DEFAULT_WPM_BAND, default_natural), default_natural


# How far a MEASURED rate must sit from an AUTHORED one before the override is worth a
# note. Half of `DEFAULT_TOLERANCE_PCT`'s 10% band: a smaller disagreement cannot by
# itself carry a beat outside the band the deliverable is judged against, so noting it
# would degrade runs for arithmetic that changes no verdict.
_AUTHORED_RATE_NOTE_FRACTION: Final = 0.05


def _has_authored_natural_wpm(ctx: RunContext) -> bool:
    """True when ``direction.timing.wpm.natural`` states a usable rate.

    The one case where measurement is not even attempted: an author who writes a rate
    means it, and probing would cost a synthesis whose answer is discarded.

    Delegates to :func:`~...shared.speech_rate.authored_natural_wpm` — the SAME
    resolution ``demo_video_gen`` uses to size the word budget it hands bdd. The two
    used to read the config independently, which is how the producer's budget and the
    solver's budget were free to disagree.
    """
    return authored_natural_wpm(ctx) is not None


def _rescale_band(band: tuple[float, float], natural: float) -> tuple[float, float]:
    """Keep the band's RATIO to natural fixed as natural moves off 150.

    The band is not an independent quantity — it is "how much faster / slower than
    natural this voice may be pushed", written down (``(130, 170)``) against the
    hardcoded ``natural=150``. Leaving it pinned while natural falls to a measured ~94
    breaks the bracket: ``collect_demands`` computes ``min = words/wpm_max*60`` and
    ``max = words/wpm_min*60``, so with ``natural < wpm_min`` the beat's IDEAL
    (``words/natural*60``) lands ABOVE its own ``max`` and ``_repair`` clamps it — the
    measured rate would be thrown away by the very bounds meant to bracket it, and the
    ``min``/``max`` seconds demands would be nonsense besides.

    Scaling by ``natural / DEFAULT_WPM_NATURAL`` preserves the +/-13.3% bracket exactly
    as authored, so ``min <= ideal <= max`` holds by construction and the seconds
    window widens by the same factor the words budget shrinks. At ``natural == 150``
    the factor is 1.0 and the band is byte-identical to today.

    An authored band with NO authored ``natural`` is rescaled too: it was written
    against the default natural, so it means the same relative bracket. An author who
    wants absolute bounds states ``natural`` as well, which disables measurement
    entirely.
    """
    if natural <= 0.0 or DEFAULT_WPM_NATURAL <= 0.0:
        return band
    factor = natural / DEFAULT_WPM_NATURAL
    return (band[0] * factor, band[1] * factor)


def _speakable_generated_spine(
    gen: dict[str, object], lines: list[dict[str, object]], ctx: RunContext
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Rewrite URLs out of MODEL-GENERATED line text; owner-authored spines are untouched.

    Run ``01a0b007``: script_doctor's clip-5 read ``…application at "https://…".``; the
    voice spelled the URL out and every budget priced it as one word. gen's bdd line build
    already applies :func:`speakable_text`, but script_doctor, claim_finish and
    retention_plan rewrite text after it — this is the one point that sees their final
    text before it is measured, budgeted and handed to narrate.

    Authored vs generated is decided the way ``demo_video_gen`` decides it —
    :func:`authored_lines` over ``brief`` (``brief.storyboard`` / ``brief.lines``) — NOT
    by ``locked``, which defaults to False on authored lines. On an authored run nothing
    is rewritten, including a model's rewrite of an authored line.

    ``gen`` is returned with the same ``lines`` so the ``reconcile_only`` path (which
    re-emits ``gen``) also hands narrate speakable text.
    """
    if authored_lines(ctx.config_snapshot.get("brief") or {}) is not None:
        return gen, lines
    speakable = [
        {**line, "text": speakable_text(str(line["text"]))}
        if isinstance(line.get("text"), str) and contains_url(str(line["text"]))
        else line
        for line in lines
    ]
    if speakable == lines:
        return gen, lines
    return {**gen, "lines": speakable}, speakable


def _stored_rate_matches_spoken_text(gen: Mapping[str, object], texts: list[str]) -> bool:
    """True only when the stored ``resolved_wpm`` was sampled over exactly ``texts``.

    ``demo_video_gen`` samples the raw bdd DRAFT; ``script_doctor`` (and any later plan
    mutator) rewrites the lines before this stage runs. A rate is a property of the TEXT
    — run ``01a0afab``'s draft carried a quoted URL per line that kokoro spells out
    (6.41s vs 3.07s for one line, measured), so the draft sampled at 111.7 wpm for
    doctored lines delivered at ~150. An absent digest list (a pre-this-change output) is
    "unknown text", which is NOT a match.
    """
    stored = gen.get("resolved_wpm_text_digests")
    if not isinstance(stored, list) or not texts:
        return False
    sampled = {digest for digest in stored if isinstance(digest, str)}
    # A SUBSET matches: gen drops whole beats after sampling, which leaves the remaining
    # text exactly what was measured. Any rewritten line is absent from the set.
    return all(digest in sampled for digest in spoken_text_digests(texts))


def _rate_record(sample: ScriptSpeechRate, texts: list[str]) -> dict[str, object]:
    """The ``resolved_wpm*`` fields for a rate THIS stage re-sampled, digest included.

    Written over the spread ``**gen`` so the output records the rate the solver actually
    used and the text it was measured on — not gen's draft rate beside a spine it no
    longer describes.
    """
    return {
        "resolved_wpm": sample.rate.wpm,
        "resolved_wpm_measured": sample.rate.measured,
        # The Phase-B failure reason wins: a MEASURED passage prior carries no reason of
        # its own, and dropping the sample's would record a fallback as a clean rate.
        "resolved_wpm_reason": sample.reason or sample.rate.reason,
        "resolved_wpm_basis": sample.rate.basis,
        # Only a sample that LANDED belongs to this text. A failed one carries the prior
        # (its basis/reason say so) and no digests, so nothing can later reuse it as a
        # measurement of these lines, and a retry measures them again.
        "resolved_wpm_text_digests": spoken_text_digests(texts) if sample.reason is None else None,
        "speech_seconds_split": sample.speech_seconds_split,
        "pause_seconds_split": sample.pause_seconds_split,
        "approx_pieces_split": sample.approx_pieces_split,
        "lines_split": sample.lines_split,
    }


def _stored_effective_rate(gen: Mapping[str, object]) -> SpeechRate | None:
    """The rate ``demo_video_gen`` stored on its output, or ``None``.

    ⚠️ This is the rate of the text gen SAMPLED, which is not necessarily the text in
    front of the caller — check :func:`_stored_rate_matches_spoken_text` first.

    ``gen`` is ``resolve_plan_output(ctx)`` — whichever plan mutator ran last
    (``reconcile_timeline`` itself never wrote these fields before now, and every
    mutator between it and ``demo_video_gen`` re-emits the gen shape by spreading
    ``**gen``, so the fields travel through the chain untouched). ``None`` when
    ``resolved_wpm`` is absent (a pre-this-change output being replayed) — the
    caller re-measures rather than trusting a partial/missing record.
    """
    wpm = gen.get("resolved_wpm")
    if not isinstance(wpm, (int, float)) or wpm <= 0.0:
        return None
    reason = gen.get("resolved_wpm_reason")
    basis = gen.get("resolved_wpm_basis")
    return SpeechRate(
        wpm=float(wpm),
        measured=bool(gen.get("resolved_wpm_measured")),
        reason=cast(FallbackReason | None, reason) if isinstance(reason, str) else None,
        basis=cast(RateBasis, basis) if isinstance(basis, str) else CALIBRATION_PASSAGE,
    )


def _cast_segments(lines: list[dict[str, object]], ctx: RunContext) -> list[tuple[str, str | None]]:
    """``(text, voice_id)`` per NARRATED beat — the voice ``demo_video_narrate`` WILL use.

    Deliberately the SAME expression ``demo_video/narrate.py::voice_for`` evaluates —
    ``voice_for_speaker(item, voice, brief)`` on the shared ``DEFAULT_CAST``, with no
    ``default_cast`` override, because that stage passes none either. A budget resolved
    by a second, parallel implementation of the cast rule is a budget that goes wrong the
    first time somebody edits one of the two, which is exactly how the rate and the
    casting drifted apart in the first place.

    Beats with no text are dropped: they are narrated by nobody, they contribute zero
    words to the weighting, and probing their nominal voice would buy a synthesis whose
    answer is discarded. Dialogue-off runs take ``voice_for_speaker``'s own gate and come
    back as the single ``voice.voice_id`` for every beat.
    """
    voice = ctx.config_snapshot.get("voice") or {}
    brief = ctx.config_snapshot.get("brief") or {}
    segments: list[tuple[str, str | None]] = []
    for line in lines:
        text = str(line.get("text") or "")
        if not text.split():
            continue
        segments.append((text, voice_for_speaker(cast("dict", line), voice, brief)))
    return segments


def _clip_seconds(ctx: RunContext) -> dict[str, float]:
    """The REAL measured footage length per clip join-key, from ``demo_video_record``.

    ``demo_video_record`` keys each harvested clip by the same join key the compile
    step uses (``line.clip or line.id``) and carries the ffprobe'd ``seconds``; both
    the clip's ``id`` and ``line_id`` are mapped so either binding resolves. A missing
    or skipped record stage stores ``{}`` ⇒ no beat is footage-pinned and the solver
    degrades gracefully to a pure text/card budget.
    """
    record = ctx.prior_output("demo_video_record") or {}
    out: dict[str, float] = {}
    for clip in record.get("clips") or []:
        if not isinstance(clip, dict):
            continue
        seconds = float(clip.get("seconds") or 0.0)
        if seconds <= 0.0:
            continue
        for key in (clip.get("line_id"), clip.get("id")):
            if key:
                out[str(key)] = seconds
    return out


def _infeasible_note(demands: list[TimeDemand], cfg: _TimingConfig) -> str | None:
    """``Σmin > target``: the floors already overrun. Arithmetic cannot fix this."""
    flagged = infeasible_beats(demands, cfg.target)
    if not flagged:
        return None
    overrun = sum(d.min for d in demands) - cfg.target
    return (
        f"reconcile_timeline: timeline INFEASIBLE — the beats' minimum time demands "
        f"exceed the {cfg.target:.2f}s target by {overrun:.2f}s. Flagged beats: "
        f"{', '.join(flagged)}. Seconds can only be moved, never created: the narration "
        f"must be shortened (rewrite the copy) for this target to be reachable."
    )


def _infeasible_degradation(
    demands: list[TimeDemand], cfg: _TimingConfig, *, relief_ran: bool
) -> Degradation | None:
    """The SAME INFEASIBLE finding as :func:`_infeasible_note`, on the QUERYABLE channel.

    MEASURED, 2026-09-11, two runs of ``demo_video_full_chain_todomvc`` at the same
    44.00s order: this stage diagnosed the overrun correctly BOTH times — "exceeds the
    44.00s target by 38.76s" (run ee67) and "by 17.56s" (run 5bf5) — and both runs
    recorded ``reconcile_timeline.degradations = []``. ee67 delivered 68.075s, +54.72%
    of the order. The finding existed only as English inside ``notes``, so nothing
    could ask "which runs stamped a plan that already projected past the order".
    ``notes`` does flip the ``degraded`` boolean, which is why the run was not silent —
    but a prose sentence carries no identity, and ``shared/degradation.py``'s docstring
    is explicit that machine facts do not belong there.

    WHY THIS IS ALSO THE MEASURED-FOOTAGE BOUND. ``demo_video_record`` runs BEFORE this
    stage, and :func:`_clip_seconds` already pins every footage beat's demand to the
    ffprobe'd clip length, so ``sum(d.min for d in demands)`` is a projection built on
    REAL recorded seconds — no bdd ``expected_recording_seconds`` export and no parent
    ``_PARENT_MEASURED_YIELD`` is needed for it. On ee67 the recorded clips totalled
    82.76s against the 44.00s order and ``sum(min) - target`` came to exactly the 38.76s
    the note named; on 5bf5, 61.56s and 17.56s. The arithmetic that bounds the plan
    against real recording length was therefore already here and already right — what
    was missing was a channel a query can bind to, which is what this adds.

    ⛔ What this entry does NOT claim. It reports an OVERRUN AGAINST THE FIGURES THIS
    RUN HOLDS: the ordered seconds, the projected floor, their difference, and how much
    of that projection is pinned measured footage. It does not claim the order was
    unreachable, and it does not claim the delivered film will be this long — narrate,
    the freeze-pad and the compiler all sit downstream. It also does not fail the run:
    per ``shared/degradation.py``, recording a degradation is never fatal.

    Returns ``None`` on a feasible timeline, so a plan that fits emits nothing and the
    stage output is unchanged.
    """
    if not infeasible_beats(demands, cfg.target):
        return None
    projected = sum(d.min for d in demands)
    overrun = projected - cfg.target
    footage = [d for d in demands if d.kind == "footage"]
    footage_seconds = sum(d.min for d in footage)
    if footage:
        measured = (
            f"{len(footage)} of {len(demands)} beat(s) are pinned to footage "
            f"demo_video_record actually measured, totalling {footage_seconds:.2f}s"
        )
    else:
        measured = (
            f"no beat is pinned to measured footage — the {len(demands)} beat floor(s) "
            f"are text/card budgets"
        )
    action = (
        "a structural relief pass ran and the re-solved timeline still projects past the order"
        if relief_ran
        else f"no structural relief pass ran (direction.timing.relief.mode={cfg.relief_mode!r})"
    )
    return degradation(
        ORDERED_DURATION_INFEASIBLE,
        f"ordered {cfg.target:.2f}s; the reconciled beats' minimum time demands project "
        f"{projected:.2f}s, an overrun of {overrun:.2f}s past the order; {measured}; "
        f"{action}, so the plan this stage stamped still projects past the order. "
        f"Seconds can only be moved, never created: shorten the narration, record less "
        f"footage, or raise the order.",
    )


class _UnderfullFinding(NamedTuple):
    """The STRUCTURED underfull verdict — what arms the hard gate, prose aside.

    The gate below used to arm itself on ``"UNDERFULL" in note``, i.e. on an English
    substring of a sentence written for a human. Rewording the note — a copy edit, the
    cheapest change anyone could make to this file — silently disarmed the 50%-of-target
    floor and no test failed. So the condition gets an identity: :func:`_underfull_finding`
    is the single place the verdict is computed, the note is DERIVED from it, and the gate
    reads the finding rather than grepping the prose. The note stays exactly as it was for
    the human reading one run.
    """

    #: Seconds the content actually fills (``target + drift``; drift is negative here).
    filled_seconds: float
    #: How far short of the target that lands, as a positive magnitude.
    short_seconds: float
    #: The human-readable sentence — the ONLY thing that may be reworded freely.
    note: str


def _underfull_finding(
    allotments: list[Allotment], drift: float, cfg: _TimingConfig
) -> _UnderfullFinding | None:
    """``drift`` short of the target beyond tolerance, or ``None``. THE underfull condition.

    The verdict is taken on the ``drift`` the CALLER passes, because the two callers
    measure different things on purpose: :func:`_timeline_findings` passes
    :func:`budget_drift` (what the compiler DELIVERS, freeze pad included — the soft
    ``timeline_underfull`` note), :func:`_underfull_failure` passes :func:`content_drift`
    (pad excluded — the hard floor's precondition, which pad must never disarm). A
    non-finite band compares False here exactly as it does in :func:`is_underfull`.
    """
    if not drift < -tolerance_seconds(cfg.target, cfg.tolerance_pct):
        return None
    return _UnderfullFinding(
        filled_seconds=cfg.target + drift,
        short_seconds=abs(drift),
        note=(
            f"reconcile_timeline: timeline UNDERFULL — even holding every beat at its "
            f"maximum (freeze-held footage tails included) the content fills only {cfg.target + drift:.2f}s of the "
            f"{cfg.target:.2f}s target, {abs(drift):.2f}s short. The solver does not "
            f"invent seconds past a beat's max: add content, or lower the target."
        ),
    )


def _underfull_note(allotments: list[Allotment], drift: float, cfg: _TimingConfig) -> str | None:
    """The prose half of :func:`_underfull_finding`, for the ``notes`` channel."""
    finding = _underfull_finding(allotments, drift, cfg)
    return None if finding is None else finding.note


def _underfull_failure(allotments: list[Allotment], cfg: _TimingConfig) -> None:
    """HARD-FAIL a GROSSLY underfull timeline instead of shipping the wrong deliverable.

    The soft note above is the right answer for a timeline that misses its target by a
    pacing margin. It is the WRONG answer for a collapse: prod run 01a00b81 filled 2.32s
    of a 60.00s target, emitted the note, and shipped ``completed`` — an 11.4s video
    against a minute-long order is not a degraded deliverable, it is a different one, and
    the customer was billed for it. So below
    ``pipeline.demo_timeline_underfull_fail_ratio`` x target the stage raises, which the
    DAG executor persists into ``StageOutput.error`` and lands the run at ``failed``
    rather than returning a stamped spine. NOTE: there is NO refund path —
    ``grep -rn refund`` over this repo finds docstrings only, and
    ``run_usage_reconciler`` merely
    mirrors usage read-only. Failing here therefore forfeits the spend of the stages that
    already ran (``demo_video_record``'s browser capture above all); narrate and compile
    are DOWNSTREAM and are saved. That trade is why the ratio is deliberately low.

    ARMED BY :func:`_underfull_finding`, never by the prose. Until 2026-09-15 this read
    ``next(note for note in notes if "UNDERFULL" in note)`` — an English substring of a
    human-facing sentence — so a copy edit to :func:`_underfull_note` would have disarmed
    the floor entirely with every test still green. The structured finding is now the
    single definition and the note is derived from it; the prose can be reworded freely.

    Deliberately NOT a redefinition of underfull: the gate only ever fires on a timeline
    whose pad-blind :func:`content_drift` is ALREADY outside the ``tolerance_pct`` band, so
    a mildly-short film keeps its old note-only behaviour, and the note is carried into the
    raised message so the diagnostic is never lost. ``ratio <= 0`` disables the gate
    entirely (note-only, the pre-fix behaviour) — the operator kill-switch, flippable in
    GitOps without a deploy of this code. Unlike the UNSAYABLE path (a contract-named soft
    gate, see this module's docstring) gross underfill was never promised to be soft.
    """
    # PAD-BLIND precondition (`content_drift`, not `budget_drift`): since the soft note
    # counts delivered freeze pad, gating the floor on it would let a short order of a
    # few sub-second clips, each padded to the 1.50s ceiling, read "not underfull" and
    # skip the floor altogether. The pad is credited below, capped, and nowhere else.
    finding = _underfull_finding(allotments, content_drift(allotments, cfg.target), cfg)
    if finding is None or cfg.target <= 0.0:
        return
    underfull = finding.note
    # Read the setting only once the soft condition has already fired: an otherwise-fine
    # run must not depend on this knob at all.
    ratio = get_settings().pipeline.demo_timeline_underfull_fail_ratio
    if ratio <= 0.0:
        return
    # Freeze pad counts, but only as a RESIDUAL — see `content_floor_credit` for the
    # invariant, the two measured films that bracket it, and why the credit is capped
    # instead of deleted. Short version: `_apply_allotments` hands the compiler
    # ``demand.max + freeze_pad_seconds``, so a footage beat really does occupy its
    # frozen tail (28.00s bare / 58.00s held on a 60s target — a bare-sum gate would
    # have KILLED that 58-second film); but the pad ceiling is 1.50s PER BEAT, which is
    # 30% of a 5.00s order, so uncapped credit made the gate structurally unable to fail
    # a short target. Live run 01a059a2 landed `completed` on 1.40s of captured footage
    # against a 5.00s order because 1.40 + 1.50 = 2.90 scored 58%, and shipped a blank
    # master (luma_spread=0.0, duplicate_frame_ratio=0.6429, gate_pass=false).
    bare = sum(float(a.seconds) for a in allotments)
    pad = sum(float(a.freeze_pad_seconds) for a in allotments)
    filled = content_floor_credit(bare, pad, cfg.target)
    if filled >= ratio * cfg.target:
        return
    credited = filled - bare
    raise RuntimeError(
        f"reconcile_timeline: timeline GROSSLY UNDERFULL — {filled:.2f}s of the "
        f"{cfg.target:.2f}s target ({filled / cfg.target:.1%}) is below the "
        f"{ratio:.0%} floor; refusing to ship a deliverable this far short of the one "
        f"that was ordered. Captured content is {bare:.2f}s; of the {pad:.2f}s of "
        f"freeze pad only {credited:.2f}s is credited "
        f"({MAX_PAD_CREDIT_RATIO:.0%} of target max — pad is held frames, it cannot "
        f"carry a run over the floor by itself). "
        f"{underfull}{_yield_attribution(allotments, cfg.target)}"
    )


def _yield_attribution(allotments: list[Allotment], target_seconds: float) -> str:
    """Name WHERE the missing seconds were, not just how many are missing.

    The gate reported one aggregate — "10.98s of the 30.00s target" — and nothing
    else. That is unactionable: it cannot distinguish "we ordered too few
    scenarios" from "every scenario recorded a fraction of what we assumed", and
    those need opposite fixes. Five runs died on this message in 30 days without
    anyone being able to say which.

    A PINNED beat (``source == "default"``) is one whose length is a measured
    truth the solver may not grow — for footage that is the clip's own recorded
    duration (``time_budget`` sets ``lo = ideal = hi = measured``). So the mean
    pinned length IS the realised yield per recording, so it is compared against
    the per-beat share of the duration that was actually ORDERED
    (``target / len(pinned)``). It used to be compared against
    ``gen._SECONDS_PER_SCENARIO``, a 4.0s constant justified by two hand-observed
    samples and unrelated to this run's order — a comparison that could not
    distinguish "too few beats" from "each beat too short" because its reference
    was not the budget the gate applies. The constant is gone; the order is the
    reference.

    Deliberately part of the RAISED message rather than a log line: this run is
    about to fail, and the operator reads the error, not the pod's logs.
    """
    pinned = sorted(
        (a for a in allotments if a.source == "default"),
        key=lambda a: a.seconds + a.freeze_pad_seconds,
    )
    if not pinned:
        return " No pinned beats — every second here was solver-computed, so the shortfall is upstream of capture."
    total = sum(a.seconds + a.freeze_pad_seconds for a in pinned)
    mean = total / len(pinned)
    shortest = ", ".join(f"{a.id}={a.seconds + a.freeze_pad_seconds:.2f}s" for a in pinned[:3])
    ordered_share = target_seconds / len(pinned) if target_seconds > 0 else 0.0
    return (
        f" Yield: {len(pinned)} pinned beat(s), sum={total:.2f}s, "
        f"mean={mean:.2f}s/beat against the {ordered_share:.2f}s/beat share of the "
        f"{target_seconds:.2f}s ordered duration; shortest: {shortest}."
    )


def _unsayable_note(allotments: list[Allotment]) -> str | None:
    """Beats whose ``vo_rate`` clamped: even at max speed the words overrun the beat."""
    beats = [a.id for a in allotments if a.unsayable]
    if not beats:
        return None
    return (
        f"reconcile_timeline: UNSAYABLE narration — {', '.join(beats)} cannot be spoken "
        f"inside their allotment even at the maximum delivery rate; the audio will "
        f"overrun the shot. Shorten the narration text on these beats."
    )


def _underspoken_word_targets(
    allotments: list[Allotment],
    demands: list[TimeDemand],
    wpm_natural: float,
    *,
    rate: float | None = None,
) -> dict[str, tuple[int, int]]:
    """Per-beat MINIMUM word count for every ``underspoken`` beat: ``{id: (floor, has)}``.

    The mirror of :func:`_unsayable_trim_targets`. ``underspoken`` is set by
    :func:`~...time_budget.vo_rate_for` when ``raw_rate < floor``, i.e.
    ``words < window * wpm_natural * floor / 60``; inverting that same check gives the
    smallest word count that clears it: ``ceil(window * wpm_natural * floor / 60)``.
    ``window`` and ``floor`` are taken exactly as the flag took them (allotment plus its
    capped freeze-pad; ``max(VO_RATE_DRAG_FLOOR, VO_RATE_BAND[0])``) and the word
    count is the demand's own ``word_count`` — so target and flag CANNOT drift.

    ``rate`` re-aims the SAME inversion at a different delivery rate and defaults to the
    floor, i.e. to exactly the arithmetic above. :meth:`ReconcileTimelineStage._expand_underspoken`
    passes :data:`_EXPAND_ASK_RATE` (1.0, natural pace) to get the count it ASKS the
    expander for, because the floor count is by construction the count that clamps back to
    the floor. Parameterised rather than duplicated so the ask can never drift from the
    flag it is supposed to clear.

    This function itself is REPORT ONLY: it makes :func:`_underspoken_note` actionable and
    supplies the re-ask its arithmetic. It arms no relief loop, rewrites no text and
    changes nothing about what degrades — duration corrections here have a measured
    history of overshooting.
    """
    words_by_id = {d.id: d.word_count for d in demands}
    # ⛔ THE DRAG FLOOR, NOT THE NATURAL FLOOR. This inversion's whole contract is that
    # it takes `floor` "exactly as the flag took it" — and `vo_rate_for` now tests
    # against `VO_RATE_DRAG_FLOOR`. Left at `VO_RATE_NATURAL_FLOOR` this would compute
    # the count that clears a 0.80 flag for a beat flagged at 0.97, i.e. a minimum that
    # does not clear the flag it exists to clear, which is precisely the drift the
    # docstring above promises is impossible.
    effective_rate = max(VO_RATE_DRAG_FLOOR, VO_RATE_BAND[0]) if rate is None else rate
    targets: dict[str, tuple[int, int]] = {}
    for allotment in allotments:
        if not allotment.underspoken:
            continue
        has = words_by_id.get(allotment.id)
        if has is None:
            continue
        window = allotment.seconds + allotment.freeze_pad_seconds
        targets[allotment.id] = (math.ceil(window * wpm_natural * effective_rate / 60.0), has)
    return targets


def _underspoken_note(
    allotments: list[Allotment],
    demands: list[TimeDemand] | None = None,
    wpm_natural: float = DEFAULT_WPM_NATURAL,
) -> str | None:
    """Beats whose ``vo_rate`` clamped at the natural floor: too FEW words for the beat.

    The exact mirror of :func:`_unsayable_note`: there the beat has too many words to
    say inside its allotment even at the maximum delivery rate; here it has too few,
    so fitting the beat would need a rate below ``VO_RATE_NATURAL_FLOOR`` and the
    narration is instead dragged below natural pace. This is a script-length fault
    (the beat's seconds outran its word count), not a TTS malfunction.

    With ``demands`` supplied, each beat carries its arithmetic remedy from
    :func:`_underspoken_word_targets` ("clip-1 needs >= 19 words (has 6)").
    """
    beats = [a.id for a in allotments if a.underspoken]
    if not beats:
        return None
    targets = _underspoken_word_targets(allotments, demands or [], wpm_natural)
    named = [
        f"{b} needs >= {targets[b][0]} words (has {targets[b][1]})" if b in targets else b
        for b in beats
    ]
    # The residual the clamp leaves unfilled, SUMMED — the seconds this timeline is short
    # by for this reason alone, and therefore the part of a short master these beats
    # explain. Reported because it is the number the note was missing: "the voiceover
    # sounds slow" and "the film is short" are the same defect measured two ways, and
    # without the seconds a reader cannot tell whether this beat costs 0.2s or 4s.
    #
    # ⛔ `hold` alone UNDERSTATES the hole (1e): `hold_seconds` is capped at the 1.5s
    # per-beat freeze-pad ceiling (`underspoken_hold`), so a 6.00s residual reports as
    # "1.50s" with the other 4.50s invisible — covered by NEITHER a held frame (past the
    # ceiling) NOR speech (the rate lever is already exhausted). `uncovered_seconds`
    # (`underspoken_uncovered`, ORCH item 1d) is exactly that leftover; report it
    # separately from what a held frame actually covers rather than silently folding it
    # into the same number the reader would read as "handled".
    hold = sum(a.hold_seconds for a in allotments if a.underspoken)
    uncovered = sum(a.uncovered_seconds for a in allotments if a.underspoken)
    if uncovered > 0.0:
        residual_report = (
            f"Even at the floor the words run out {hold + uncovered:.2f}s before the "
            f"allotment does — {hold:.2f}s of that is held as a frozen frame, but "
            f"{uncovered:.2f}s is covered by NEITHER speech nor a held frame, so these "
            f"beats ALSO ship that much short of the timeline they were allotted."
        )
    else:
        residual_report = (
            f"Even at the floor the words run out {hold:.2f}s before the allotment does, "
            f"so these beats ALSO ship short unless that residual is held."
        )
    return (
        f"reconcile_timeline: UNDERSPOKEN narration — {', '.join(named)}; these beats have "
        f"too few words for their allotted seconds, so the delivery rate was dragged below "
        f"the natural speaking floor; the voiceover will sound artificially slow. This is a "
        f"script-length fault, not a synthesis problem — add narration text on these "
        f"beats, or shorten their allotment. {residual_report}"
    )


def _drift_note(drift: float, cfg: _TimingConfig) -> str | None:
    """The headline guarantee: ``|delivered - target| <= tolerance_pct`` — or say so.

    ``drift`` is :func:`budget_drift`: the allotments PLUS the freeze-held footage tails
    the compiler renders, so an over-long film is flagged and a film the pads bring into
    the band is not (showcase run 01a15ec2: bare 29.28s, delivered 30.33s on a 30s order).
    """
    if abs(drift) <= tolerance_seconds(cfg.target, cfg.tolerance_pct):
        return None
    return (
        f"reconcile_timeline: reconciled total {cfg.target + drift:.2f}s misses the "
        f"{cfg.target:.2f}s target by {drift:+.2f}s — outside the "
        f"{cfg.tolerance_pct:.1f}% tolerance band."
    )


def _overpad_note(demands: list[TimeDemand], cfg: _TimingConfig) -> str | None:
    """Footage beats whose narration overruns the clip past the freeze-pad ceiling.

    A footage beat is pinned to its measured clip length (``min == ideal == max``), so a
    beat that also carries narration needs ``word_count / wpm_natural * 60`` seconds to be
    spoken (the same word-budget formula ``time_budget.collect_demands`` uses for a vo
    beat). When that word-budget exceeds the clip by more than
    ``cfg.freeze_pad_ceiling_seconds``, closing the gap costs more than an honest amount of
    downstream compensation — the narration gets time-compressed (``vo_rate``) into the
    clip-plus-ceiling window, and the freeze-pad is now bounded at the ceiling (Step B:
    ``max_seconds = clip + freeze_pad``, so the frozen tail no longer stretches to the full
    audio length). Bounding the pad does NOT make the over-narration honest — the beat still
    demanded more screen time than the clip-plus-ceiling window gives it — so the finding is
    still recorded on ``notes`` (⇒ degraded run) so a human extends the B-roll or trims the
    narration. The audio is never truncated (truncating IS the freeze bug).

    ⛔ THIS IS A WORD-COUNT ESTIMATE, NOT A MEASUREMENT — AND THE NOTE MUST SAY SO.
    This stage runs BEFORE ``demo_video_narrate``; no audio exists yet. The "narration
    length" here is ``word_count / wpm_natural * 60``. Run
    ``01a0afab-24a3-7745-b8b9-9625966f31ca``: clip-4, 23 words at a resolved 111.72 wpm
    ⇒ an ESTIMATED 12.35s against a 10.28s clip ⇒ gap 2.07s, pad capped at 1.50s, 0.57s
    past the ceiling ⇒ flagged. Narrate then MEASURED the line: 9.87s at speed 0.97
    (≈9.58s at 1.0) — SHORTER than the clip. The old text ("carry narration that
    overruns their recorded clip") stated the estimate as a fact with no numbers, so it
    read as a measured overrun the run's own narrate output contradicted. Each flagged
    beat now carries its words, the wpm, the estimated seconds, the clip and the excess,
    so a reader can see the claim's basis and check it against narrate's measurement.
    The arithmetic and the tiering (a note ⇒ degraded) are unchanged.
    """
    flagged: list[str] = []
    for d in demands:
        if d.kind != "footage" or d.word_count <= 0:
            continue
        clip_dur = d.ideal  # footage beats have min == ideal == max == measured
        allotment = d.word_count / cfg.wpm_natural * 60.0
        capped = freeze_pad_ceiling(clip_dur, allotment, cfg.freeze_pad_ceiling_seconds)
        excess = (allotment - clip_dur) - capped
        if excess > 1e-9:
            flagged.append(
                f"{d.id} ({d.word_count} words ≈ {allotment:.2f}s estimated vs a "
                f"{clip_dur:.2f}s clip, {excess:.2f}s past the ceiling)"
            )
    if not flagged:
        return None
    return (
        f"reconcile_timeline: FREEZE-PAD ADVISORY (word-count ESTIMATE at "
        f"{cfg.wpm_natural:.1f} wpm, made before any narration is synthesized) — "
        f"{'; '.join(flagged)}. By that estimate the narration would overrun the recorded "
        f"clip by more than the {cfg.freeze_pad_ceiling_seconds:.2f}s freeze-pad ceiling, so "
        f"the pad is capped at the ceiling and the planned rate compresses the narration "
        f"into the clip-plus-ceiling window; the clip is never trimmed and the audio is "
        f"never cut. demo_video_narrate MEASURES each line and re-fits it — if its measured "
        f"duration fits the clip, this estimate was wrong (check the resolved wpm), not the "
        f"line. Otherwise extend the B-roll or trim the narration on these beats."
    )


def _timeline_findings(
    demands: list[TimeDemand], allotments: list[Allotment], cfg: _TimingConfig
) -> list[tuple[str, str | None]]:
    """Everything the deterministic solver could NOT fix: ``(note, code)`` pairs.

    The SINGLE place each finding is computed, so the prose note and its coded twin
    cannot drift apart. ``code`` is ``None`` only for the INFEASIBLE note, whose
    degradation (:data:`ORDERED_DURATION_INFEASIBLE`) is raised separately by
    :func:`_infeasible_degradation` with richer provenance (and on ``reconcile_only``,
    which writes no notes) — coding it here as well would report one fault twice.
    """
    drift = budget_drift(allotments, cfg.target)
    candidates: tuple[tuple[str | None, str | None], ...] = (
        (_infeasible_note(demands, cfg), None),
        (_underfull_note(allotments, drift, cfg), TIMELINE_UNDERFULL),
        (_unsayable_note(allotments), NARRATION_UNSAYABLE),
        (_underspoken_note(allotments, demands, cfg.wpm_natural), TIMELINE_UNDERSPOKEN),
        (_drift_note(drift, cfg), TIMELINE_OFF_TARGET),
        (_overpad_note(demands, cfg), FREEZE_PAD_CEILING_EXCEEDED),
        (_footage_trimmed_note(demands, cfg), FOOTAGE_TRIMMED),
        (_narration_over_order_note(demands, cfg), DURATION_OUT_OF_TOLERANCE),
    )
    return [(note, code) for note, code in candidates if note is not None]


def _footage_trimmed_note(demands: list[TimeDemand], cfg: _TimingConfig) -> str | None:
    """Footage tails ``trim_footage_tails`` cut to fit the order — per beat, in seconds."""
    cuts = footage_cuts(demands)
    if not cuts:
        return None
    per_beat = ", ".join(f"{bid}: -{cut:.2f}s" for bid, cut in cuts.items())
    measured = sum(d.clip_seconds or 0.0 for d in demands if d.id in cuts)
    return (
        f"reconcile_timeline: FOOTAGE TRIMMED — {len(cuts)} footage beat(s) ran past their "
        f"narration and the film projected past the {_order_label(cfg)}; their tails were "
        f"cut by {sum(cuts.values()):.2f}s in total, {measured:.2f}s of measured footage "
        f"-> {measured - sum(cuts.values()):.2f}s ({per_beat})."
    )


def _narration_over_order_note(demands: list[TimeDemand], cfg: _TimingConfig) -> str | None:
    """Footage trimming could not bring the projection inside the order's tolerance.

    Fires only when a footage tail WAS trimmed (the case the trim exists for) and the
    projection still overruns: the residual is narration-bound or floor-bound — footage is
    never cut below its own narration, 60% of the clip or 2.0s — so the overrun is REPORTED
    here rather than hidden. A spine whose narration alone overruns with nothing to trim is
    already reported by the INFEASIBLE / UNSAYABLE findings.
    """
    if not footage_cuts(demands):
        return None
    projected = projected_segment_seconds(demands, cfg.wpm_natural)
    over = projected - cfg.target
    if over <= tolerance_seconds(cfg.target, cfg.tolerance_pct):
        return None
    return (
        f"reconcile_timeline: the spine projects {projected:.2f}s against the "
        f"{_order_label(cfg)} ({over:+.2f}s) even after every footage tail was trimmed to "
        f"its floor (its narration, 60% of the clip, 2.0s) — narration alone overruns; "
        f"shorten the narration or raise the order."
    )


def _order_label(cfg: _TimingConfig) -> str:
    """``"30.00s order"`` or ``"24.00s body (30.00s order - 6.00s title cards)"``."""
    if cfg.card_seconds <= 0.0 or cfg.ordered_seconds is None:
        return f"{cfg.target:.2f}s order"
    return (
        f"{cfg.target:.2f}s body ({cfg.ordered_seconds:.2f}s order - "
        f"{cfg.card_seconds:.2f}s title cards)"
    )


def _timeline_notes(
    demands: list[TimeDemand], allotments: list[Allotment], cfg: _TimingConfig
) -> list[str]:
    """Everything the deterministic solver could NOT fix, in human-readable form.

    A truthy ``notes`` list marks the run ``degraded`` — so these are the stage's only
    channel for "the numbers are honest, but the timeline is not what you asked for".
    Each one has a coded twin in :func:`_timeline_degradations`.
    """
    return [note for note, _ in _timeline_findings(demands, allotments, cfg)]


def _timeline_degradations(
    demands: list[TimeDemand], allotments: list[Allotment], cfg: _TimingConfig
) -> list[Degradation]:
    """The coded twins of :func:`_timeline_notes` (INFEASIBLE excepted — see there).

    Without these the notes reached the run record only through the executor's
    ``unclassified_note`` stamp ("degraded via prose notes only"), which no query can
    tell apart from any other producer's prose.
    """
    return [
        degradation(code, note)
        for note, code in _timeline_findings(demands, allotments, cfg)
        if code is not None
    ]


def _apply_allotments(
    gen: dict[str, object],
    draft: list[dict[str, object]],
    allotments: list[Allotment],
    demands: list[TimeDemand],
    notes: list[str],
    *,
    timeline_manifest: TimelineManifest | None = None,
    degradations: list[Degradation] | None = None,
) -> DemoVideoGenOutput:
    """Stamp the reconciled timeline onto the ordered spine and re-emit the gen shape.

    Per beat: ``seconds`` becomes the solver's allotment (a footage beat's allotment is
    its pinned, measured clip length — bit-for-bit), ``vo_rate`` is the clamped delivery
    multiplier narrate must speak to (absent on a footage/card beat, where it is None),
    and ``min_seconds`` / ``max_seconds`` carry the demand bounds forward so
    ``assemble_shots`` + the compiler can clamp the narration-driven duration and cap
    the freeze-pad. Ids / order / ``clip`` are untouched.

    ``notes`` APPENDS to whatever the upstream plan already carried — a prior stage's
    degradation is never clobbered by this one. ``degradations`` appends the same way,
    and the key is written ONLY when this stage has an entry to add: with none, the
    ``{**gen}`` spread is left exactly as it was, so a feasible run's output is
    unchanged.
    """
    by_alloc = {str(a.id): a for a in allotments}
    by_demand = {str(d.id): d for d in demands}
    merged: list[dict[str, object]] = []
    for line in draft:
        lid = str(line["id"])
        allotment = by_alloc.get(lid)
        if allotment is None:
            merged.append(line)
            continue
        patch: dict[str, float] = {"seconds": float(allotment.seconds)}
        if allotment.vo_rate is not None:
            patch["vo_rate"] = float(allotment.vo_rate)
        demand = by_demand.get(lid)
        if demand is not None:
            patch["min_seconds"] = float(demand.min)
            # Step B: extend a footage beat's upper bound by its ceiling-capped
            # freeze-pad so the renderer holds the pad at the ceiling instead of
            # stretching the frozen tail to the full narration length. 0.0 for every
            # non-footage / already-fits beat ⇒ ``max_seconds == demand.max`` exactly
            # (byte-identical). ``seconds`` / ``min_seconds`` stay at the bare clip:
            # the clip is never trimmed and the audio is never cut.
            patch["max_seconds"] = float(demand.max) + float(allotment.freeze_pad_seconds)
        # The pad itself, stamped so DOWNSTREAM consumers can see the window the rate was
        # chosen against. `solve_timeline` picks a footage beat's `vo_rate` against
        # `seconds + freeze_pad_seconds`; `demo_video_narrate.fit_target_for` used to
        # re-fit the synthesized line to `seconds` ALONE and therefore re-synthesized it
        # up to the full 1.5s ceiling SLOWER than the solver intended — a second,
        # independent slow-narration lever on top of the word-budget drift. It cannot be
        # recovered from `max_seconds` (that is `demand.max + pad`, and `demand.max` is
        # not on the line), so it rides explicitly. 0.0 on every non-footage /
        # already-fits beat ⇒ the fit target is byte-identical to before for those.
        patch["freeze_pad_seconds"] = float(allotment.freeze_pad_seconds)
        # The underspoken residual. When a beat's authored words are too few for its
        # allotment, `vo_rate_for` clamps the rate at VO_RATE_NATURAL_FLOOR and the beat
        # BY CONSTRUCTION still finishes short of its allotment — the owner's "narration
        # weirdly slows down" and the short master are the same event. `hold_seconds` is
        # that unfilled remainder, and stamping it lets `shot_assembly` hold the PICTURE
        # for it instead of stretching the SPEECH further.
        #
        # This was deliberately omitted until 2026-09-13 because both halves of the
        # consumer were missing: `AuthoredLine` declared no `hold_seconds` (so pydantic's
        # `extra="ignore"` dropped the key and the stamp would have been vacuous), and
        # `shot_assembly` treated `max_seconds` as a CAP with no hold floor. Both landed
        # in this same change, so the stamp now reaches a real consumer.
        #
        # ⛔ It must NOT be folded into `freeze_pad_seconds`: `narrate.fit_target_for`
        # returns `seconds + freeze_pad_seconds` and RE-SYNTHESIZES to fill it, so routing
        # the residual there would slow the speech further — the opposite of the fix. A
        # test pins that separation.
        patch["hold_seconds"] = float(allotment.hold_seconds)
        merged.append({**line, **patch})
    prior_notes = [str(note) for note in _as_list(gen.get("notes"))]
    extra: dict[str, object] = {}
    if degradations:
        prior_degradations = [d for d in _as_list(gen.get("degradations")) if isinstance(d, dict)]
        extra["degradations"] = [*prior_degradations, *degradations]
    # The C13 relief manifest (per-beat provenance, spec 23 §7.7) now rides a TYPED
    # ``timeline_manifest`` field so it is machine-assertable end-to-end (the parallel
    # ``notes[]`` string is kept for humans). None on every non-relief run ⇒ the field is
    # absent-equivalent and the output is byte-identical to before for those runs.
    return DemoVideoGenOutput.model_validate(
        {
            **gen,
            "lines": merged,
            "notes": [*prior_notes, *notes],
            "timeline_manifest": timeline_manifest,
            **extra,
        }
    )


def _cant_converge_degradations(
    manifest: TimelineManifest | None, loop_notes: list[str]
) -> list[Degradation]:
    """:data:`TIMELINE_CANT_CONVERGE` iff the FINAL manifest says ``cant_converge``.

    Keyed on the structured outcome, never on the prose: it is the same condition that
    writes (and, via :meth:`ReconcileTimelineStage._refresh_cant_converge_note`,
    re-writes) the ``STRUCTURAL RELIEF NEEDED`` note, read off the refreshed manifest
    the spawner also reads. The note only supplies the human ``detail``.
    """
    if manifest is None or manifest.run.convergence_outcome != "cant_converge":
        return []
    detail = next(
        (note for note in loop_notes if note.startswith(_CANT_CONVERGE_NOTE_HEAD)),
        f"{_CANT_CONVERGE_NOTE_HEAD} — the timeline is "
        f"{manifest.run.overrun_seconds:.1f}s over target and cannot be shortened in place.",
    )
    return [degradation(TIMELINE_CANT_CONVERGE, detail)]


def _mark_superseded_beats(
    manifest: TimelineManifest | None,
    before: list[Allotment],
    after: list[Allotment],
) -> TimelineManifest | None:
    """Name the manifest rows the UNDERSPOKEN re-ask invalidated. Refreshes nothing.

    THE DEFECT THIS CLOSES. ``timeline_manifest`` is built inside :meth:`_relief_loop`;
    :meth:`_expand_underspoken` runs AFTERWARDS, lengthens narration and RE-SOLVES. So
    the manifest's ``beats[]`` rows are a PRE-re-ask snapshot while ``lines[].vo_rate``
    (stamped by :func:`_apply_allotments` from the POST-re-ask allotments, and the only
    rate ``demo_video_narrate.speed_for`` reads) is the final one. Measured on live run
    ``01a0a582-2531-7145-ba7d-a1295e92b232``: ``beats[].vo_rate`` said ``0.80`` — the
    ``VO_RATE_NATURAL_FLOOR`` clamp, i.e. the owner's "narration weirdly slows down" —
    on the same three beats where ``lines[]`` said ``1.0383 / 1.0 / 1.0``, in the SAME
    stage-output row, while the row's own note said the re-ask had succeeded. Which array
    a reader happened to open decided the verdict on the headline complaint.

    ⛔ This marks the ``beats[]`` rows; it does not refresh them, and that asymmetry is
    deliberate — the snapshot is the build-time record and the spine is the live one.
    The RUN-level budget is a separate matter and is no longer left stale: the caller
    follows this with :func:`_refresh_run_budget`, because ``runs/structural_relief.py``
    decides and sizes a PAID child render from ``run.convergence_outcome`` /
    ``run.overrun_seconds``, and a pre-re-ask value there describes a timeline that no
    longer exists. This function is provenance; it is not load-bearing for that decision.

    A beat is superseded when its ``seconds`` or ``vo_rate`` moved across the re-ask.
    That includes beats the re-ask did not rewrite: the re-solve water-fills the whole
    spine, so an untouched elastic beat's allotment can move too, and its manifest row is
    then equally stale. ``None``/empty in, ``None``/unchanged out — a run with no relief
    manifest, or a re-ask that changed nothing, is byte-identical to before.
    """
    if manifest is None:
        return None
    was = {a.id: (a.seconds, a.vo_rate) for a in before}
    # The re-ask only patches TEXT, so it cannot add or drop an id; an id missing from
    # either side is therefore not a rate disagreement and is left unmarked rather than
    # reported as one.
    superseded = sorted(a.id for a in after if a.id in was and was[a.id] != (a.seconds, a.vo_rate))
    if not superseded:
        return manifest
    return manifest.model_copy(
        update={
            "run": manifest.run.model_copy(
                update={"beats_superseded_by_underspoken_reask": superseded}
            )
        }
    )


def _refresh_beat_delivery(
    manifest: TimelineManifest | None, allotments: list[Allotment]
) -> TimelineManifest | None:
    """Overwrite each ``beats[]`` row's ``vo_rate`` / ``unsayable`` with the FINAL allotment.

    Run ``01a0b007``: ``beats[].vo_rate`` read 0.97 on all three beats — the pre-re-ask
    clamp — while ``lines[]`` (what narrate reads) said 1.126 / 1.0 / 0.97. The superseded
    marker named two of them, but a reader of ``beats[]`` still read a rate nobody
    delivers. These two fields describe DELIVERY and no spawn decision reads them.

    ⚠️ ``seconds`` / ``min_seconds`` / ``max_seconds`` / ``feasible`` are deliberately
    left as the relief loop built them: ``structural_relief_policy`` sizes a paid child's
    cut from those, and relief sizing is out of this change's scope.
    ``run.beats_superseded_by_underspoken_reask`` keeps naming the rows whose values
    moved, so the provenance survives the refresh.
    """
    if manifest is None:
        return None
    final = {a.id: a for a in allotments}
    beats = [
        beat.model_copy(
            update={"vo_rate": final[beat.id].vo_rate, "unsayable": final[beat.id].unsayable}
        )
        if beat.id in final
        else beat
        for beat in manifest.beats
    ]
    if beats == list(manifest.beats):
        return manifest
    return manifest.model_copy(update={"beats": beats})


def _stamp_footage_out_points(
    manifest: TimelineManifest | None,
    lines: list[dict[str, object]],
    demands: list[TimeDemand],
    allotments: list[Allotment],
    cfg: _TimingConfig,
) -> TimelineManifest | None:
    """Write each beat's FINAL ``footage_out_point_seconds`` onto the manifest.

    The compile reads the out-point off ``timeline_manifest.beats[]`` (the line model does
    not carry it), so it must be refreshed from the FINAL demands — the relief loop and the
    underspoken re-ask both move narration, and the trim is narration-aware.

    No manifest (the relief loop did not run) and nothing trimmed ⇒ ``None`` stays
    ``None``: the output is byte-identical. No manifest but a beat WAS trimmed ⇒ one is
    built from the final state (0 relief iterations), because without it the cut would be
    invisible to the compile.

    ⛔ On a relief-OFF run (``direction.timing.relief.mode != "auto"``) that out-points-only
    manifest carries ``convergence_outcome="relief_off"``, NEVER a classification.
    ``runs/structural_relief.py`` reads "a manifest is present" as "this was a relief
    run"; a classified ``cant_converge`` here made a relief-OFF run spawn a PAID relief
    child (6 footage beats x 10s at a 30s order ⇒ overrun 6.0 ⇒ SPAWN). ``relief_off``
    maps to ``STOP_RELIEF_OFF`` in ``decide_structural_relief``, and ``converged`` is
    False because no relief pass converged anything.

    ⛔ ``relief_off`` is for relief-OFF runs ONLY. A relief-ON (``"auto"``) run reaches
    this branch because the loop had nothing to do — the (tail-trimmed) timeline already
    fit and no beat was unsayable. Round-5 proof runs (P1/P2/P4, all ``mode="auto"``)
    were stamped ``relief_off`` / ``STOP_RELIEF_OFF`` at overrun 0, a label that says the
    opt-in was absent when it was not. That run gets the SAME classifier
    :func:`_refresh_run_budget` applies to a relief run's manifest
    (:func:`_classify_convergence`, over the final ``lines`` / ``demands``), so a fit
    timeline reads ``converged`` ⇒ ``STOP_CONVERGED``.

    ⛔ …but ONLY ``converged`` survives. The loop arms on an infeasible/unsayable timeline;
    the underspoken re-ask runs AFTER it with a spine cap of 110% of the body, so it can
    push a fitting timeline past the order (14+8+8s clips on a 24s body, 20-word replies
    ⇒ 0.40s over). Classifying that read ``cant_converge`` and the spawner bought a PAID
    child for an overrun the FREE re-ask caused. A loop that never ran can never yield a
    spawn-eligible outcome: out of band ⇒ ``relief_not_run`` ⇒ ``STOP_CANT_CONVERGE``.
    """
    points = {d.id: d.footage_out_point_seconds for d in demands}
    if manifest is None:
        if not any(p is not None for p in points.values()):
            return None
        # The loop did NOT run (this branch is reached only when it built no manifest), so
        # nothing here may be spawn-eligible: only ``converged`` survives classification.
        outcome: Literal["converged", "relief_off", "relief_not_run"]
        if cfg.relief_mode != "auto":
            outcome = RELIEF_OFF_OUTCOME
        elif _classify_convergence(lines, demands, cfg) == "converged":
            outcome = "converged"
        else:
            outcome = RELIEF_NOT_RUN_OUTCOME
        footage_ids = frozenset(d.id for d in demands if d.kind == "footage")
        return TimelineManifest.model_validate(
            build_timeline_manifest(
                demands,
                allotments,
                target_seconds=cfg.target,
                card_seconds=cfg.card_seconds,
                loop_iterations=0,
                converged=outcome == "converged",
                pacing_score=None,
                convergence_outcome=outcome,
                overrun_seconds=_overrun_seconds(demands, cfg.target),
                protected_beat_count=sum(1 for line in lines if _is_protected(line, footage_ids)),
            )
        )
    beats = [
        beat.model_copy(update={"footage_out_point_seconds": points.get(beat.id)})
        for beat in manifest.beats
    ]
    if beats == list(manifest.beats):
        return manifest
    return manifest.model_copy(update={"beats": beats})


def _overrun_seconds(demands: list[TimeDemand], target: float) -> float:
    """``max(0, Sigma min - target)`` — the one definition of timeline overrun.

    Lifted out of the TWO sites that each spelled this clamped arithmetic inline (the
    ``overrun_seconds`` the manifest stores, in the relief-failed branch and in the
    post-relief classification) so the post-re-ask refresh (:func:`_refresh_run_budget`)
    recomputes the SAME number from new demands instead of a third copy drifting away.

    ⚠️ Two OTHER inline occurrences of ``sum(d.min ...) - cfg.target`` remain, in
    :func:`_infeasible_note` and in the structural-ops call. They are NOT copies of this
    and are deliberately NOT folded in: they are the SIGNED residual, with no
    ``max(0.0, …)``, and each is reached only after ``infeasible_beats(...)`` has already
    returned a non-empty set — i.e. only on a branch where the overrun is known positive
    and the clamp could never bind. Both feed a MAGNITUDE into human/LLM-facing text (the
    INFEASIBLE note's "exceeds the target by Xs", and the timing-director prompt); neither
    produces the ``overrun_seconds`` this function defines. Sharing one helper between the
    recorded budget and those operands would couple two unrelated things for no gain.
    """
    return max(0.0, sum(d.min for d in demands) - target)


def _refresh_run_budget(
    manifest: TimelineManifest | None,
    demands: list[TimeDemand],
    cfg: _TimingConfig,
    *,
    lines: list[dict[str, object]],
) -> TimelineManifest | None:
    """Recompute the manifest's run-level SPAWN INPUTS from the POST-re-ask spine.

    THE DEFECT THIS CLOSES. ``timeline_manifest`` is built inside :meth:`_relief_loop`;
    :meth:`_expand_underspoken` runs AFTERWARDS, rewrites narration and re-solves, and
    :meth:`_remeasure_if_rewritten` then RE-RESOLVES THE SPEECH RATE over the rewritten
    text (#1042) and re-collects every demand under the new wpm band. So every number the
    relief loop classified is a snapshot of a timeline that no longer exists.
    ``runs/structural_relief.py`` (and ``structural_relief_policy``) read these numbers
    verbatim to decide and size a PAID child render.

    Refreshed here — every run-level field a spawn decision reads, all pure functions of
    ``lines`` + ``demands`` + ``cfg.target``:

    * ``overrun_seconds`` / ``infeasible_beat_ids`` — size the child's cut
      (``_cheapest_first_cut`` drops beats until it has removed ``overrun_seconds``).
    * ``convergence_outcome`` — its ``cant_converge`` branch is the spawn gate itself
      (``decide_structural_relief`` §2). #1031 left it pre-re-ask on the premise that the
      re-ask only LENGTHENS unprotected text, which cannot move the protected floor
      :func:`_classify_convergence` tests. #1042's re-measure broke that premise: a new
      wpm band rescales EVERY vo beat's ``min`` — claim-bearing (protected) beats
      included — in EITHER direction. A stale value can therefore both BUY a render
      (stale ``cant_converge`` on a spine whose re-measured protected floor now fits, i.e.
      in-place trimming suffices) and BLOCK one (stale ``progress`` on a spine whose
      protected floor now exceeds the target). Recomputed with the same classifier, over
      the same post-re-ask ``lines`` / ``demands`` the other two fields read.

      ⛔ EXCEPT ``relief_failed``. That outcome means the relief loop RAISED and no
      classification was ever made; it is safe by construction precisely because the
      policy maps it to a STOP. Reclassifying it here would turn a crashed relief run
      into a spawn candidate — a paid render bought on the strength of a pass that never
      completed. It is kept verbatim.

    NOT refreshed: ``drift_seconds`` (a function of ``allotments``, read by no spawn
    decision) and ``protected_beat_count`` (a count of protection FLAGS, which no re-ask
    or re-measure changes). ``beats[]`` rows stay a build-time snapshot, marked by
    :func:`_mark_superseded_beats`.

    ``None`` in, ``None`` out. A run whose demands the re-ask did not move recomputes the
    identical values (same formula, same inputs) and gets the SAME object back.
    """
    if manifest is None:
        return None
    overrun = _overrun_seconds(demands, cfg.target)
    infeasible_ids = infeasible_beats(demands, cfg.target)
    stored_outcome = manifest.run.convergence_outcome
    # ``relief_off`` likewise: an out-points-only manifest must never be reclassified
    # into a spawn candidate.
    outcome = (
        stored_outcome
        if stored_outcome in ("relief_failed", RELIEF_OFF_OUTCOME, RELIEF_NOT_RUN_OUTCOME)
        else _classify_convergence(lines, demands, cfg)
    )
    # ⛔ A loop that CONVERGED left a timeline in band; anything that put it back out is
    # post-loop and FREE (the underspoken re-ask's spine cap is 110% of the body, and its
    # longer footage narration lifts the footage trim floors). Reclassifying that as
    # ``cant_converge`` bought a PAID child for an overrun relief did not leave — the same
    # defect #1081 closed for a loop that never ran. Out of band it reads ``progress``
    # (in-place text is the lever: the re-ask's own words) ⇒ STOP_CANT_CONVERGE.
    if stored_outcome == "converged" and outcome == "cant_converge":
        outcome = "progress"
    # ``converged`` is the loop's score verdict; it must never stay True on a timeline
    # the refresh just moved OUT of ``converged``.
    left_band = stored_outcome == "converged" and outcome != "converged"
    converged = manifest.run.converged and not left_band
    if (
        overrun == manifest.run.overrun_seconds
        and infeasible_ids == manifest.run.infeasible_beat_ids
        and outcome == stored_outcome
        and converged == manifest.run.converged
    ):
        return manifest
    if outcome != stored_outcome:
        # A flip here changes whether a PAID render can be bought, and the relief loop's
        # user-facing note (built pre-re-ask) still describes the old outcome — so it must
        # be visible in the pod log, not only in the manifest.
        log.warning(
            "reconcile_timeline: post-re-ask reclassification %s -> %s "
            "(overrun %.2fs -> %.2fs) — the spawn decision reads the new outcome",
            stored_outcome,
            outcome,
            manifest.run.overrun_seconds,
            overrun,
        )
    return manifest.model_copy(
        update={
            "run": manifest.run.model_copy(
                update={
                    "overrun_seconds": overrun,
                    "infeasible_beat_ids": infeasible_ids,
                    "convergence_outcome": outcome,
                    "converged": converged,
                }
            )
        }
    )


def _classify_convergence(
    lines: list[dict[str, object]],
    demands: list[TimeDemand],
    cfg: _TimingConfig,
) -> _ConvergenceOutcome:
    """Deterministic feasibility classification of the RE-SOLVED relief timeline (spec 23 §7.7).

    ``converged`` — feasible (``Σmin <= target``, no beat over its own max).
    ``progress`` — still infeasible but ≥1 over-budget beat is trim-eligible, so a bounded
    rerun could yet converge (the loop merely ran out of passes).
    ``cant_converge`` — still infeasible AND every over-budget beat is protected
    (claim-bearing / locked / footage-pinned): no in-place op can help, so the run degrades
    honestly rather than looping — this is the case a future gen re-run would target.
    """
    if not infeasible_beats(demands, cfg.target):
        return "converged"
    # Infeasible (Σmin > target). The irreducible floor is Σ(PROTECTED mins). Cutting an
    # UNPROTECTED beat removes its FULL min — the most a single beat can shed; relief
    # text-trimming sheds only part of it (fewer words → lower min) and reallocate sheds
    # none, so cutting is the ceiling on in-place reduction, and only unprotected beats may
    # be cut. So in-place relief can reach `Σmin <= target` iff the PROTECTED floor
    # (footage-pinned / claim-bearing / author-locked beats, which no in-place op may drop)
    # itself fits the target. When that floor alone exceeds target — e.g. a
    # footage-pinned spine whose clips total more than target with only a few small VO
    # beats — cutting EVERY unprotected beat still can't fit, so a gen re-run that drops
    # protected footage beats is the only lever: `cant_converge`. This subsumes the
    # all-protected case (the floor is then the whole Σmin) and, crucially, no longer
    # mis-labels a footage-dominated MIXED spine as `progress` (which looped the in-stage
    # trimmer to no effect and starved PR-C's gen re-run). Otherwise a bounded pass can yet
    # converge by cutting unprotected beats: `progress`.
    footage_ids = frozenset(d.id for d in demands if d.kind == "footage")
    protected_ids = {str(line["id"]) for line in lines if _is_protected(line, footage_ids)}
    protected_floor = sum(d.min for d in demands if d.id in protected_ids)
    if protected_floor > cfg.target:
        return "cant_converge"
    return "progress"


#: Concurrent first-take syntheses the measured fit keeps in flight (narrate's own
#: fan-out width is of the same order; tts_backend serializes behind it anyway).
_FIT_SYNTH_CONCURRENCY: Final = 4
#: Concurrent one-line rewrite calls to worker_management (each line costs at most
#: two, sequentially; this caps how many LINES ask at once).
_FIT_REWRITE_CONCURRENCY: Final = 3


@dataclass(frozen=True, slots=True)
class _MeasuredFit:
    """The measured per-line fit's outcome: ``{line_id: LineFit}`` over the fitted lines."""

    fits: dict[str, LineFit]

    @property
    def seconds(self) -> dict[str, float]:
        return {lid: f.take.seconds for lid, f in self.fits.items()}

    def with_texts(self, lines: list[dict[str, object]]) -> list[dict[str, object]]:
        """``lines`` carrying each fitted line's shipped text (ids/order untouched)."""
        return [
            {**line, "text": self.fits[str(line["id"])].take.text}
            if str(line.get("id")) in self.fits
            else line
            for line in lines
        ]


def _measured_findings(
    findings: list[tuple[str, str | None]], fit: _MeasuredFit | None
) -> list[tuple[str, str | None]]:
    """``findings`` with the estimate-based UNDERSPOKEN one replaced by the MEASURED one.

    Unchanged without a measured fit. With one, the estimate's 0.97 drag-floor test
    would flag a beat 3% short — inside the fit's own 5% keep band — and its "needs >= N
    words" arithmetic is the pooled-rate guess the fit replaced. What is still short
    AFTER the fit is reported from the clips themselves.
    """
    if fit is None:
        return findings
    out = [(n, c) for n, c in findings if c != TIMELINE_UNDERSPOKEN]
    measured_note = measured_underspoken_note(list(fit.fits.items()))
    if measured_note is not None:
        out.append((measured_note, TIMELINE_UNDERSPOKEN))
    return out


def _overlay_measured_fit(
    out: DemoVideoGenOutput,
    fit: _MeasuredFit,
    allotments: list[Allotment],
    demands: list[TimeDemand],
) -> DemoVideoGenOutput:
    """Stamp each fitted line's measured ``vo_rate`` / ``hold_seconds`` and fit marker.

    ``_apply_allotments`` wrote the solver's ESTIMATED rate and hold; for a fitted line
    the measurement is the truth, and these are exactly what narrate reads to request
    (``speed_for``) and keep the measured clip (``fitted_seconds``).
    """
    pads = {str(a.id): float(a.freeze_pad_seconds) for a in allotments}
    footage = {str(d.id) for d in demands if d.kind == "footage"}
    lines = []
    for line in out.lines:
        lid = str(line.id)
        if lid not in fit.fits:
            lines.append(line)
            continue
        patched = fitted_line(
            line.model_dump(), fit.fits[lid], pad=pads.get(lid, 0.0), footage=lid in footage
        )
        keys = ("text", "vo_rate", "hold_seconds", "fitted_seconds", "fitted_slack_seconds")
        lines.append(line.model_copy(update={k: patched[k] for k in keys}))
    return out.model_copy(update={"lines": lines})


class ReconcileTimelineInput(StageInput):
    """No typed inputs — everything is read from ``ctx`` (prior outputs + config)."""


class ReconcileTimelineStage(StageHandler[ReconcileTimelineInput, DemoVideoGenOutput]):
    name = "reconcile_timeline"
    input_model = ReconcileTimelineInput
    output_model = DemoVideoGenOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (
        SlotUse("llm"),
        SlotUse("tts"),
    )

    def __init__(self, client: WMClient | None = None, tts_client: TTSClient | None = None) -> None:
        # Read the LLM budget lazily at construction (env/test overrides apply). The
        # WMClient is only ever CALLED behind the opt-in relief gate — an off run never
        # touches it — but constructing it here mirrors the retention_plan pattern.
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())
        # Used ONLY for the one-per-process speech-rate calibration probe (see
        # ``shared/speech_rate.py``); None ⇒ that module builds the default client
        # lazily, so constructing the stage opens no TTS connection.
        self._tts = tts_client

    @property
    def _model_tier(self) -> str | None:
        """Inference-router tier for the directive-proposal + per-beat rewrite calls.

        Read PER RUN from the run snapshot's ``model_tier.timing_director`` (pinned at
        run start, bound by the dispatcher — ``src/auth/run_providers.py``), so a
        GitOps flip mid-run does not change a paused run's model. A legacy run
        (snapshot without ``model_tier``) falls back to the env setting. None ⇒ the
        small local model (byte-identical to today)."""
        return current_run_model_tier("timing_director", default=timing_director_model_tier())

    async def _effective_natural_wpm(
        self,
        lines: list[dict[str, object]],
        ctx: RunContext,
        degradations: list[Degradation],
        gen: Mapping[str, object],
        rate_record: dict[str, object] | None = None,
    ) -> float | None:
        """The rate this spine will ACTUALLY be spoken at, blended across its cast.

        ``rate_record``, when given, receives the ``resolved_wpm*`` fields of a rate this
        call RE-SAMPLED (see the single-voice path) so ``handle`` can persist what the
        solver really used. Left empty when the stored rate was reused.

        Returns the natural wpm to re-resolve ``_TimingConfig`` against, or ``None``
        meaning "leave the config exactly as the hardcoded default resolved it" — the
        byte-identical-to-main answer for every run whose rate could not be established.

        **Single-voice and uncast runs are untouched.** When every narrated beat resolves
        to ``config_snapshot["voice"]["voice_id"]`` — which is every dialogue-off
        demo_video — this takes the single-voice path and probes exactly the one voice it
        always did, with the same branch on ``rate.measured`` and the same warning. No
        extra probe, no degradation, no change of any kind.

        **A cast is weighted by words, not averaged.** ``resolve_cast_wpm`` measures each
        DISTINCT voice once (memoised per ``(provider, voice)``, so a two-voice spine costs
        two probes regardless of beat count) and returns ``Σ(words_i / wpm_i) * 60``
        re-expressed as one rate. That single number is what the whole downstream machinery
        (``collect_demands``, ``solve_timeline``, ``_underspoken_word_targets``) consumes,
        so nothing else in this stage changes shape.

        **An unmeasurable voice is unmeasured, not "the default rate".** It contributes at
        ``DEFAULT_WPM_NATURAL`` — the number this stage would have used anyway — and
        appends a ``SPEECH_RATE_UNUSABLE`` degradation NAMING that voice, so a run budgeted
        partly blind is queryable rather than silent. ``per_voice`` is keyed by distinct
        voice, so at most one entry per voice is ever appended.

        The sentinel discipline is the same one the single-voice path documents: the
        cast branch turns on ``rate.weighted`` / ``per.assumed``, never on a comparison of
        ``wpm`` against ``DEFAULT_WPM_NATURAL`` — a voice that genuinely speaks at exactly
        150.0 is one plausible draw away, and a value comparison would discard that
        measurement as a probe failure.
        """
        segments = _cast_segments(lines, ctx)
        configured = (ctx.config_snapshot.get("voice") or {}).get("voice_id")
        distinct = {voice for _, voice in segments}
        if not distinct or distinct == {configured}:
            # SINGLE OWNER, READ BACK — BUT ONLY FOR THE SAME TEXT. `demo_video_gen`
            # stamped the effective wpm it sampled off the RAW bdd DRAFT, plus a digest of
            # exactly those texts. `narrate` speaks `item["text"]` of `resolve_plan_output`
            # (narrate.py `resolve_script` / `synth_text_for`) — the same `lines` this stage
            # was handed — and by now `script_doctor` has usually rewritten them. Reusing
            # the draft's rate for rewritten text is how run 01a0afab budgeted doctored
            # lines at 111.7 wpm (the draft's spelled-out URLs) while they were delivered
            # at ~150: clip-4 estimated 12.35s vs a 10.28s clip, a 1.5s pad, and a voice
            # dragged to the floor to fill it. So: reuse when the digest matches (the
            # budget and the solver stay bit-identical), otherwise measure THESE lines.
            texts = spoken_texts(str(line.get("text") or "") for line in lines)
            stored = _stored_effective_rate(gen)
            if stored is not None and not _stored_rate_matches_spoken_text(gen, texts):
                sample = await resolve_effective_wpm(
                    ctx, texts=texts, client=self._tts, default=DEFAULT_WPM_NATURAL
                )
                if rate_record is not None:
                    rate_record.update(_rate_record(sample, texts))
                if sample.reason is not None:
                    # The CURRENT text could not be measured. The stored draft rate is
                    # NOT a fallback — it measured different text. Budget on the prior
                    # the resolver returned (its basis says which) and record it.
                    degradations.append(script_sample_degradation(stage=self.name, sample=sample))
                    log.warning(
                        "reconcile_timeline: the stored %.1f wpm was sampled over different "
                        "text and re-sampling this spine failed (reason=%s) — budgeting at "
                        "the %.1f wpm %s rate, not the stale draft rate.",
                        stored.wpm,
                        sample.reason,
                        sample.rate.wpm,
                        sample.rate.basis,
                    )
                else:
                    log.info(
                        "reconcile_timeline: spine text differs from what demo_video_gen "
                        "sampled (%.1f wpm) — re-sampled it at %.1f wpm (%s)",
                        stored.wpm,
                        sample.rate.wpm,
                        sample.rate.basis,
                    )
                rate = sample.rate
            else:
                rate = stored
            if rate is None:
                # No stored rate — an authored-mode / pre-this-change gen output, or a
                # fixture with no `demo_video_gen` prior output at all. Phase A ONLY,
                # exactly as this stage always did before gen started storing its own
                # Phase-B measurement: falling all the way to a fresh Phase-B sample
                # here would probe every one of THIS lines' sampled entries a SECOND
                # time (gen already spent that cost), which is the "genuinely must
                # re-measure" case the docstring warns is the exception, not the norm.
                rate = await resolve_natural_wpm_detail(
                    ctx, client=self._tts, default=DEFAULT_WPM_NATURAL
                )
            if rate.measured:
                return rate.wpm
            log.warning(
                "reconcile_timeline: speech-rate probe did not answer (reason=%s) — "
                "budgeting at the %.1f wpm default.",
                rate.reason,
                rate.wpm,
            )
            return None

        cast_rate = await resolve_cast_wpm(
            ctx, segments=segments, default=DEFAULT_WPM_NATURAL, client=self._tts
        )
        for voice, per in cast_rate.per_voice.items():
            if per.assumed:
                degradations.append(
                    speech_rate_fallback_degradation(stage=self.name, rate=per, voice=voice)
                )
        if not cast_rate.weighted:
            log.warning(
                "reconcile_timeline: cast speech-rate weighting over %d voices produced no "
                "usable rate — budgeting at the %.1f wpm default.",
                len(cast_rate.per_voice),
                DEFAULT_WPM_NATURAL,
            )
            return None
        log.info(
            "reconcile_timeline cast rate %.1f wpm (%.2fs over %d beats, %d voices; the "
            "configured %r alone would have budgeted this spine at a single rate)",
            cast_rate.wpm,
            cast_rate.seconds,
            len(segments),
            len(cast_rate.per_voice),
            configured,
        )
        return cast_rate.wpm

    async def _remeasure_if_rewritten(
        self,
        texts_before: list[str],
        lines: list[dict[str, object]],
        ctx: RunContext,
        gen: dict[str, object],
        clip_seconds: dict[str, float],
        current: tuple[list[TimeDemand], list[Allotment], _TimingConfig, list[Degradation]],
        *,
        authored: bool,
    ) -> tuple[
        list[dict[str, object]],
        list[TimeDemand],
        list[Allotment],
        _TimingConfig,
        dict[str, object],
        list[Degradation],
    ]:
        """Re-resolve the rate over REWRITTEN ``lines``, then re-collect and re-solve.

        THE DEFECT THIS CLOSES. The opening rate was measured on the lines this stage was
        HANDED; the underspoken re-ask then rewrites them. Run ``01a0b007``: 136.06 wpm
        measured on script_doctor's lines, the re-ask rewrote all three, clip-5's rewrite
        was delivered at 82.3 wpm — while ``vo_rate``, freeze pads, holds, advisories and
        the persisted digests all still described the pre-re-ask text.

        No-op (``current`` returned untouched) when no line's text changed, or when the
        rate is ``authored`` — a stated rate is not a property of the text.

        The same :meth:`_effective_natural_wpm` the stage opened with — so the stored
        digests (now those of the pre-rewrite text) do not match, the resolver samples the
        FINAL text (its content cache answers a repeat for free), and ``_rate_record``
        writes that text's digests. A failed sample keeps #1041's honest shape: a
        ``speech_rate_unusable`` degradation, no digests, and the resolver's prior —
        never the rate measured on text that is no longer spoken. ``None`` resolves the
        config against the hardcoded default, exactly as the opening call does.

        Returns the degradations of THIS measurement; they replace the opening ones,
        which described text that is no longer spoken.
        """
        if authored or texts_before == [str(line.get("text") or "") for line in lines]:
            demands, allotments, cfg, degradations = current
            return lines, demands, allotments, cfg, gen, degradations
        degradations = []
        record: dict[str, object] = {}
        natural = await self._effective_natural_wpm(lines, ctx, degradations, gen, record)
        if record:
            gen = {**gen, **record}
        cfg = (
            _resolve_timing_config(ctx)
            if natural is None
            else _resolve_timing_config(ctx, default_natural=natural)
        )
        demands = collect_demands(
            lines,
            clip_seconds=clip_seconds,
            wpm_band=cfg.wpm_band,
            wpm_natural=cfg.wpm_natural,
            target=cfg.target,
        )
        allotments = solve_timeline(
            demands,
            target=cfg.target,
            tolerance_pct=cfg.tolerance_pct,
            wpm_natural=cfg.wpm_natural,
            freeze_pad_ceiling_seconds=cfg.freeze_pad_ceiling_seconds,
        )
        return lines, demands, allotments, cfg, gen, degradations

    async def _authored_rate_override_note(self, ctx: RunContext, authored: float) -> list[str]:
        """Say out loud when an AUTHORED ``wpm.natural`` is overriding a live measurement.

        The authored value still WINS — that is deliberate and unchanged (an author who
        writes a rate means it, and silently substituting a measurement would swap one
        order for a different one). What was wrong is that it won *invisibly*: a stale
        constant left in ``direction.timing.wpm.natural`` months ago would keep budgeting
        every beat at a rate the voice does not speak at, and the run carried no record
        that a better number existed. The narration-slows-down class of defect is exactly
        this shape, so the override is now on the record.

        Costs one probe on authored runs, where the old code paid none. That is
        affordable and bounded: :func:`measure_natural_wpm` memoises per
        ``(provider, voice)`` for the life of the process, so it is one synthesis per pod
        per voice, not one per run — and :func:`resolve_natural_wpm_detail` never raises,
        so a sick tts_backend costs this note, not the run.

        ⚠️ SILENT WHEN THE TWO AGREE. A note on every authored run is a note nobody
        reads; the threshold is :data:`_AUTHORED_RATE_NOTE_FRACTION`, chosen as half the
        default conformance band — a disagreement smaller than that cannot on its own
        push a beat outside the band the deliverable is judged against, so it is not
        worth degrading a run over. A probe that could not answer produces no note at
        all: "we have nothing to compare against" is not evidence of a bad authored rate.
        """
        rate = await resolve_natural_wpm_detail(ctx, client=self._tts, default=DEFAULT_WPM_NATURAL)
        if not rate.measured or authored <= 0:
            return []
        drift = abs(rate.wpm - authored) / authored
        if drift <= _AUTHORED_RATE_NOTE_FRACTION:
            return []
        return [
            f"reconcile_timeline: the AUTHORED direction.timing.wpm.natural of "
            f"{authored:.1f} wpm governs this run's word budget and OVERRODE the rate "
            f"this run's voice was measured at ({rate.wpm:.1f} wpm, {drift:.1%} apart). "
            f"Every beat was budgeted at the authored rate; if that constant is stale, "
            f"remove it and the measurement will be used."
        ]

    async def handle(self, payload: ReconcileTimelineInput, ctx: RunContext) -> DemoVideoGenOutput:
        gen: dict[str, object] = resolve_plan_output(ctx) or {}
        lines: list[dict[str, object]] = cast("list[dict[str, object]]", _as_list(gen.get("lines")))
        if not lines:
            raise StageSkippedError("reconcile_timeline: no spine lines to budget")
        # THE SPEAKABLE BOUNDARY for everything written after gen's line build
        # (script_doctor, claim_finish, retention_plan are all models): no URL reaches the
        # rate measurement, the budget or — via this stage's output — narrate. See
        # `_speakable_generated_spine`.
        gen, lines = _speakable_generated_spine(gen, lines, ctx)

        # Presence-is-opt-in: skips (no timing block / mode off / no target) raise here.
        # Resolved FIRST against the hardcoded default so a run that does NOT opt in
        # never pays for a calibration probe.
        cfg = _resolve_timing_config(ctx)
        # Then re-resolve against the rate this run's voice ACTUALLY speaks at. Skipped
        # entirely when the author stated a `wpm.natural` (it wins regardless), and a
        # probe that cannot answer leaves `cfg` untouched, i.e. byte-identical to the
        # pre-measurement behaviour.
        #
        # ⚠️ The branch is on `rate.measured`, NEVER on `rate.wpm != DEFAULT_WPM_NATURAL`.
        # The fallback constant and a genuine measurement are the same Python float, and
        # the delivered rate measured today is ~151.6 wpm against a 150.0 default — so a
        # voice that really does speak at exactly 150.0 is one plausible draw away, and a
        # value comparison would file that measurement as "the probe failed" and discard
        # it. `resolve_natural_wpm_detail` carries measured-ness explicitly; the plain
        # `resolve_natural_wpm` cannot answer the question and must not be used here.
        #
        # ⚠️ And it is resolved over the CAST, not over one voice. ``demo_video_narrate``
        # casts each line through ``voice_for_speaker`` while this budget used to be taken
        # at a single rate for ``config_snapshot["voice"]["voice_id"]`` — a voice that, on
        # a dialogue run, may never speak. Measured live: ``am_michael`` 156.60 wpm against
        # the default ``af_heart`` 175.24 wpm, a ~12% slower cast, which on the lesson stage
        # hid a +23% overrun behind a +7.6% estimate and left the corrective machinery
        # unreached. See ``_effective_natural_wpm``.
        rate_degradations: list[Degradation] = []
        rate_notes: list[str] = []
        authored_natural = authored_natural_wpm(ctx)
        if authored_natural is None:
            rate_record: dict[str, object] = {}
            natural = await self._effective_natural_wpm(
                lines, ctx, rate_degradations, gen, rate_record
            )
            if rate_record:
                # Persist the rate the solver ACTUALLY used, with the text it measured —
                # every output path below spreads `gen`.
                gen = {**gen, **rate_record}
            if natural is not None:
                cfg = _resolve_timing_config(ctx, default_natural=natural)
        else:
            rate_notes = await self._authored_rate_override_note(ctx, authored_natural)

        clip_seconds = _clip_seconds(ctx)

        # The deterministic core: collect → solve. Pure compute, zero LLM calls.
        demands = collect_demands(
            lines,
            clip_seconds=clip_seconds,
            wpm_band=cfg.wpm_band,
            wpm_natural=cfg.wpm_natural,
            target=cfg.target,
        )
        allotments = solve_timeline(
            demands,
            target=cfg.target,
            tolerance_pct=cfg.tolerance_pct,
            wpm_natural=cfg.wpm_natural,
            freeze_pad_ceiling_seconds=cfg.freeze_pad_ceiling_seconds,
        )

        # OPT-IN, INERT-BY-DEFAULT relief loop. ONLY an "auto" relief mode on an already
        # INFEASIBLE timeline enters here; every other run leaves ``loop_notes`` empty and
        # the code below is byte-identical to the deterministic stage (no WM call).
        loop_notes: list[str] = []
        loop_degradations: list[Degradation] = []
        timeline_manifest: TimelineManifest | None = None
        # ⚠️ ARMED ON EITHER FAILURE MODE. ``infeasible_beats`` alone is EMPTY on an
        # underfull timeline, which is exactly the shape the live demo_video runs have —
        # so the C13 shortening machinery existed and never fired while 11 of 15 beats
        # were unsayable. ``unsayable`` is the second, independent trigger.
        if cfg.relief_mode == "auto" and (
            infeasible_beats(demands, cfg.target) or any(a.unsayable for a in allotments)
        ):
            # SOFT gate (mirrors footage_critique / final_critique / encode_master): the relief
            # loop makes UNGUARDED worker_management calls, so a PERMANENT WM/LLM failure
            # (``PermanentUpstreamError`` from a 402 byo-not-configured / token-limit, or
            # retries-exhausted) would otherwise raise out of ``handle`` and mark the whole run
            # FAILED — discarding the VALID deterministic ``allotments`` already computed above.
            # Instead we accept-current: keep the deterministic ``lines`` / ``demands`` /
            # ``allotments`` (the tuple-unpack only lands on success, so they still hold their
            # deterministic values here), log, and degrade honestly via a note. ``except
            # Exception`` covers ``PermanentUpstreamError`` (a plain ``Exception`` subclass) and,
            # in py3.11+, deliberately does NOT catch ``asyncio.CancelledError`` (a
            # ``BaseException``), so run cancellation still propagates — exactly as the siblings do.
            try:
                (
                    lines,
                    demands,
                    allotments,
                    loop_notes,
                    timeline_manifest,
                    loop_degradations,
                ) = await self._relief_loop(lines, demands, allotments, cfg, ctx, clip_seconds)
            except Exception:
                # TOLERABLE degrade, and genuinely so: the DELIVERABLE here is the
                # deterministic allotment vector, which was already computed above and is
                # untouched by this failure — relief is pure enrichment on top of it, and
                # its absence is surfaced on ``notes`` (⇒ the run is marked degraded, not
                # silently passed). ERROR, not WARNING: a swallowed exception that costs a
                # feature is an operator-visible fault, not routine noise.
                log.error(
                    "reconcile_timeline: relief loop failed — accepting the deterministic "
                    "allotments and degrading; the WM/LLM relief pass could not run.",
                    exc_info=True,
                )
                loop_notes = [
                    "reconcile_timeline: relief loop UNAVAILABLE (WM/LLM call failed) — kept the "
                    "deterministic allotments; the timeline was not LLM-relieved this run."
                ]
                loop_degradations = [degradation(TIMELINE_RELIEF_UNAVAILABLE, loop_notes[0])]
                # …and still emit a manifest. WHY: ``timeline_manifest`` is the ONLY signal
                # that distinguishes "this was a relief run" from "this was never a relief
                # run" (structural_relief.py reads None as the latter). Leaving it None here
                # would make a CRASHED relief attempt indistinguishable from a healthy
                # feasible run — a silent falsehood. The numbers below come from the
                # pre-loop deterministic state (untouched by the failure, per the note
                # above), so every field is honest: no pass completed (0 iterations, not
                # converged, no pacing score), no structural provenance was produced, and
                # the outcome is "relief_failed" — which the relief policy maps to
                # STOP_CANT_CONVERGE, so this can never buy a paid rerun.
                failed_footage_ids = frozenset(d.id for d in demands if d.kind == "footage")
                timeline_manifest = TimelineManifest.model_validate(
                    build_timeline_manifest(
                        demands,
                        allotments,
                        target_seconds=cfg.target,
                        card_seconds=cfg.card_seconds,
                        loop_iterations=0,
                        converged=False,
                        pacing_score=None,
                        convergence_outcome="relief_failed",
                        structural_relief_requested=False,
                        overrun_seconds=_overrun_seconds(demands, cfg.target),
                        protected_beat_count=sum(
                            1 for line in lines if _is_protected(line, failed_footage_ids)
                        ),
                    )
                )

        # THE MIRROR PASS. The relief loop above only ever makes narration SHORTER, so a
        # timeline whose beats are UNDERSPOKEN — too few words for their seconds — had no
        # corrective path at all: the rate clamped at the natural floor, the delivery
        # dragged, the beat still shipped short, and `_underspoken_note` said so without
        # changing anything. This is the bounded re-ask that closes it.
        #
        # Armed on the SAME opt-in relief switch as the loop above, and for the same
        # reason: it spends WM calls, so a run that did not opt into relief must stay
        # byte-identical and pay nothing. Skipped in `reconcile_only`, where the spine is
        # discarded a few lines below — paying for rewrites nobody keeps would be pure
        # waste.
        # Captured BEFORE the re-ask: the manifest above was built from these allotments,
        # so they are the only honest baseline for deciding which of its rows the re-ask
        # goes on to invalidate (``_mark_superseded_beats``).
        allotments_before_reask = list(allotments)
        # THE MEASURED FIT, else the word-count re-ask — see `_fit_or_reask`.
        (
            lines,
            demands,
            allotments,
            cfg,
            gen,
            rate_degradations,
            expand_notes,
            measured_fit,
        ) = await self._fit_or_reask(
            lines,
            demands,
            allotments,
            cfg,
            ctx,
            gen=gen,
            clip_seconds=clip_seconds,
            rate_degradations=rate_degradations,
            authored_natural=authored_natural,
        )
        # The manifest is a PRE-re-ask snapshot from here on. Say so ON the record, in the
        # same object, so a reader can never again read its floor-clamped ``vo_rate`` as
        # the live delivery rate (which is ``lines[].vo_rate``, stamped below).
        timeline_manifest = _mark_superseded_beats(
            timeline_manifest, allotments_before_reask, allotments
        )
        # …and REFRESH the run-level spawn inputs (overrun, infeasible ids AND the
        # convergence outcome that gates the paid structural-relief spawn): the re-ask and
        # the rate re-measure both move demands, so pre-re-ask values can buy or block a
        # child render on a timeline that no longer exists. See ``_refresh_run_budget``.
        manifest_before_refresh = timeline_manifest
        timeline_manifest = _refresh_run_budget(timeline_manifest, demands, cfg, lines=lines)
        # …and the user-facing ``cant_converge`` note the relief loop wrote PRE-re-ask must
        # agree with the refreshed outcome (and overrun) the spawner will read.
        loop_notes = self._refresh_cant_converge_note(
            ctx, loop_notes, before=manifest_before_refresh, after=timeline_manifest
        )
        # …and the per-beat DELIVERY rate, so `beats[].vo_rate` agrees with `lines[]`.
        timeline_manifest = _refresh_beat_delivery(timeline_manifest, allotments)
        # …and the footage out-points the compile cuts at, from the FINAL demands. Whether
        # relief RAN is read BEFORE this: the stamp may build a manifest for a trimmed
        # non-relief run, and that must not read as "a relief pass ran".
        relief_ran = timeline_manifest is not None
        timeline_manifest = _stamp_footage_out_points(
            timeline_manifest, lines, demands, allotments, cfg
        )

        # The INFEASIBLE finding on the machine-readable channel, computed from the FINAL
        # (post-relief) demands — which carry demo_video_record's measured clip seconds.
        # ``relief_ran`` is read off the manifest because that is the ONLY field that
        # distinguishes a relief run from a non-relief one (see the manifest comment in
        # the except-branch above); it is not inferred from ``cfg.relief_mode``, which
        # says what was ARMED, not what executed.
        infeasible_degradations = [
            d
            for d in (_infeasible_degradation(demands, cfg, relief_ran=relief_ran),)
            if d is not None
        ]
        if infeasible_degradations:
            # ERROR, not WARNING: the stage is about to stamp a spine it has just
            # measured as projecting past the order. That is operator-visible, and the
            # prose twin in ``notes`` is read only by someone already looking at the run.
            log.error("reconcile_timeline: %s", infeasible_degradations[0]["detail"])

        # reconcile_only: compute the timeline for inspection but mutate nothing.
        if cfg.mode == "reconcile_only":
            out = DemoVideoGenOutput.model_validate(gen)
            # "Mutate nothing" is about the SPINE, not about telemetry: the manifest is
            # inspection output, and dropping it here made a reconcile_only relief run
            # look like a non-relief run downstream. Set only when one exists, so a
            # feasible / non-relief run still carries None (the honest "no relief" value).
            if timeline_manifest is not None:
                out.timeline_manifest = timeline_manifest
            # Same reasoning as the manifest directly above: "mutate nothing" is about
            # the SPINE. An inspection run that computed an infeasible timeline and then
            # dropped the finding would be indistinguishable from one that fit.
            out.degradations = [*out.degradations, *rate_degradations, *infeasible_degradations]
            return out

        findings = _measured_findings(_timeline_findings(demands, allotments, cfg), measured_fit)
        timeline_notes = [note for note, _ in findings]
        notes = rate_notes + timeline_notes + loop_notes + expand_notes
        # Every DEGRADING note above carries a coded twin, so the executor never has to
        # stamp ``unclassified_note`` on this stage. Informational notes — the
        # ``timeline_manifest=`` dump and the underspoken re-ask's progress report (whose
        # residual, if any, is re-flagged by ``_underspoken_note`` on the re-solved
        # allotments) — stay prose only.
        note_degradations = [
            *(degradation(AUTHORED_SPEECH_RATE_OVERRIDE, note) for note in rate_notes),
            *(degradation(code, note) for note, code in findings if code is not None),
            *loop_degradations,
            *_cant_converge_degradations(timeline_manifest, loop_notes),
        ]
        # The one HARD gate on this stage's apply path: a grossly underfull timeline is a
        # broken deliverable, not a degraded one. It computes its OWN structured verdict
        # (`_underfull_finding`) rather than grepping `notes` for the word "UNDERFULL", so
        # a copy edit to the human-facing note can no longer disarm it. Only on the mutate
        # path — a ``reconcile_only`` run stamps nothing and is inspection, so it stays
        # advisory.
        _underfull_failure(allotments, cfg)
        out = _apply_allotments(
            gen,
            lines,
            allotments,
            demands,
            notes,
            timeline_manifest=timeline_manifest,
            degradations=[*rate_degradations, *infeasible_degradations, *note_degradations],
        )
        return (
            out
            if measured_fit is None
            else _overlay_measured_fit(out, measured_fit, allotments, demands)
        )

    def _refresh_cant_converge_note(
        self,
        ctx: RunContext,
        loop_notes: list[str],
        *,
        before: TimelineManifest | None,
        after: TimelineManifest | None,
    ) -> list[str]:
        """Re-derive the ``cant_converge`` note when :func:`_refresh_run_budget` moved the run.

        :meth:`_relief_loop` writes the note from its PRE-re-ask classification. Once the
        refresh has reclassified (or re-sized the overrun), that note would tell the user
        "STRUCTURAL RELIEF NEEDED" on a spine that is now ``progress`` — or say nothing on
        one that became ``cant_converge`` — and its spawn forecast would describe a
        timeline the spawner will never read. So when the refresh changed the run block,
        the stale note is dropped and, iff the refreshed outcome is ``cant_converge``,
        re-written by the SAME :meth:`_cant_converge_note` from the refreshed manifest.

        Untouched (same list back) when the refresh changed nothing — ``_refresh_run_budget``
        returns the identical object then — so a clean run's notes are byte-identical.
        ``rerun_stages=[]`` loses nothing: the loop only ever sets ``["demo_video_gen"]``,
        which is exactly the note's ``_RELIEF_RERUN_STAGE`` fallback.
        """
        if after is None or before is None or after is before or after.run == before.run:
            return loop_notes
        kept = [note for note in loop_notes if not note.startswith(_CANT_CONVERGE_NOTE_HEAD)]
        if after.run.convergence_outcome == "cant_converge":
            kept.append(
                self._cant_converge_note(
                    ctx,
                    overrun_seconds=after.run.overrun_seconds,
                    rerun_stages=[],
                    structural_relief_requested=after.run.structural_relief_requested,
                    manifest=after.model_dump(mode="json"),
                )
            )
        return kept

    def _cant_converge_note(
        self,
        ctx: RunContext,
        *,
        overrun_seconds: float,
        rerun_stages: list[str],
        structural_relief_requested: bool,
        manifest: dict[str, object],
    ) -> str:
        """The user-facing ``cant_converge`` degrade note, honest in BOTH branches.

        The spawn is decided by FOUR gates, not one: the kill-switch,
        :func:`decide_structural_relief`, and the two shape guards in
        :func:`decide_relief_shape` (can the spine shrink at all; can any spine of that
        size fit the budget). The first cut of this note consulted only the first two and
        so LIED exactly where it hurts: on a footage-pinned spine whose floor already
        exceeds the target — the case where the user is the ONLY remaining lever — the
        spawner refuses to bill an unconvergeable render while the note said "no action
        needed". Both callers now read the same :func:`decide_relief_shape` verdict off the
        same manifest, so forecast and action cannot diverge by construction.

        The spawn branch promises an ATTEMPT conditional on completion, never a completed
        re-render: the hook only runs on the dispatcher's ``mark_completed`` path, so a run
        that emits this note and then FAILS downstream never reaches it.
        """
        head = (
            f"{_CANT_CONVERGE_NOTE_HEAD} — the timeline is "
            f"{overrun_seconds:.1f}s over target and every over-budget beat is claim-bearing "
            f"/ locked / footage-pinned, so it cannot be shortened in place"
        )
        settings = get_settings().pipeline
        relief_cfg = ctx.config_snapshot.get("structural_relief") or {}
        prior_raw = relief_cfg.get("prior_overrun") if isinstance(relief_cfg, dict) else None
        decision = (
            decide_structural_relief(
                convergence_outcome="cant_converge",
                structural_relief_requested=structural_relief_requested,
                overrun_seconds=overrun_seconds,
                prior_overrun_seconds=opt_float(prior_raw),
                pass_number=int(as_float(relief_cfg.get("pass")))
                if isinstance(relief_cfg, dict)
                else 0,
                max_passes=settings.structural_relief_max_passes,
            )
            if settings.structural_relief_autoconverge_enabled
            else "STOP_CANT_CONVERGE"
        )
        # Same floor the spawner applies, so the note cannot promise a re-render the
        # underfull guard will refuse to bill.
        shape = relief_shape_from_manifest(
            manifest, floor_ratio=settings.demo_timeline_underfull_fail_ratio
        )
        shape_allows_spawn = shape is not None and shape.verdict == "SPAWN"
        stages = ", ".join(rerun_stages) or _RELIEF_RERUN_STAGE
        if decision == "SPAWN" and shape_allows_spawn:
            # Deliberately NOT "no action needed": ``_infeasible_note`` fires on this same
            # path telling the user to rewrite the copy, and the hook is completion-gated —
            # a run that fails downstream never reaches it.
            return (
                f"{head}. If this run completes, a shorter-spine re-render will be attempted "
                f"automatically; otherwise re-run {stages} with fewer beats or raise the "
                f"target duration."
            )
        return (
            f"{head}, and no automatic re-render will follow; "
            f"re-run {stages} with fewer beats or raise the target duration."
        )

    async def _trim_unsayable_or_flag(
        self,
        draft: list[dict[str, object]],
        cfg: _TimingConfig,
        clip_seconds: dict[str, float],
    ) -> list[dict[str, object]]:
        """Optimizer fallback when no INFEASIBLE beat is trim-eligible.

        Two distinct shapes reach here and only one of them used to exist:

        * ``unsayable`` beats that CAN be shortened — the underfull-timeline case this
          loop was blind to. The word target is arithmetic (:func:`_unsayable_trim_targets`),
          so no timing-director call is spent proposing it; the beat-rewriter inside
          :meth:`_dispose_directives` still writes the shorter line and still applies only
          a strictly-shorter rewrite.
        * nothing honestly shortenable — the pre-existing soft structural flag. Logged
          (never silent) and the draft is returned unchanged.
        """
        unsayable_targets = _unsayable_trim_targets(draft, cfg, clip_seconds)
        if unsayable_targets:
            return await self._dispose_directives(
                draft,
                [
                    {"op": "trim", "target_ids": [beat_id], "new_word_target": target}
                    for beat_id, target in sorted(unsayable_targets.items())
                ],
                frozenset(unsayable_targets),
            )
        # Log which infeasible beats were skipped so a live no-op is never silent — every
        # flagged beat here is claim-bearing / locked / footage-pinned and needs an
        # upstream structural cut, not a rewrite.
        skipped = collect_demands(
            draft,
            clip_seconds=clip_seconds,
            wpm_band=cfg.wpm_band,
            wpm_natural=cfg.wpm_natural,
            target=cfg.target,
        )
        log.warning(
            "reconcile_timeline relief: timeline INFEASIBLE but no beat is eligible "
            "to shorten — flagged beats %s are all claim-bearing / locked / "
            "footage-pinned; structural relief (demo_video_gen re-run) required.",
            infeasible_beats(skipped, cfg.target),
        )
        return draft

    async def _maybe_expand_underspoken(
        self,
        lines: list[dict[str, object]],
        demands: list[TimeDemand],
        allotments: list[Allotment],
        cfg: _TimingConfig,
        clip_seconds: dict[str, float],
        *,
        content_wpm: float | None = None,
        order_word_budgets: Mapping[str, int] | None = None,
    ) -> tuple[list[dict[str, object]], list[TimeDemand], list[Allotment], list[str]]:
        """Arm-and-guard wrapper around :meth:`_expand_underspoken`.

        Armed on the SAME opt-in relief switch as the trim loop, and for the same reason:
        it spends WM calls, so a run that did not opt into relief must stay
        byte-identical and pay nothing. Skipped in ``reconcile_only``, where the caller
        discards the spine anyway — paying for rewrites nobody keeps would be pure waste.

        SOFT-DEGRADE on any failure, exactly as the trim loop: the deliverable is the
        deterministic allotment vector, already computed by the caller and untouched by
        anything that goes wrong here, so a permanent LLM outage degrades the run (an
        honest note ⇒ ``degraded``) and never fails it. ``except Exception`` deliberately
        does not catch ``CancelledError``, so run cancellation still propagates.
        """
        if cfg.relief_mode != "auto" or cfg.mode == "reconcile_only":
            return lines, demands, allotments, []
        if not any(a.underspoken for a in allotments):
            return lines, demands, allotments, []
        try:
            return await self._expand_underspoken(
                lines,
                demands,
                allotments,
                cfg,
                clip_seconds,
                content_wpm=content_wpm,
                order_word_budgets=order_word_budgets,
            )
        except Exception:
            log.error(
                "reconcile_timeline: underspoken re-ask failed — accepting the "
                "deterministic allotments and degrading; narration on the underspoken "
                "beats is still delivered at the natural-rate floor.",
                exc_info=True,
            )
            return (
                lines,
                demands,
                allotments,
                [
                    "reconcile_timeline: UNDERSPOKEN RE-ASK UNAVAILABLE (WM/LLM call "
                    "failed) — kept the deterministic allotments; the underspoken beats "
                    "were not lengthened this run."
                ],
            )

    async def _fit_or_reask(
        self,
        lines: list[dict[str, object]],
        demands: list[TimeDemand],
        allotments: list[Allotment],
        cfg: _TimingConfig,
        ctx: RunContext,
        *,
        gen: dict[str, object],
        clip_seconds: dict[str, float],
        rate_degradations: list[Degradation],
        authored_natural: float | None,
    ) -> tuple[
        list[dict[str, object]],
        list[TimeDemand],
        list[Allotment],
        _TimingConfig,
        dict[str, object],
        list[Degradation],
        list[str],
        _MeasuredFit | None,
    ]:
        """Lengthen / shorten narration to its beats: MEASURED when possible.

        THE MEASURED FIT (see :mod:`narration_fit`): every spoken line synthesized and
        fitted to its beat's window by its REAL clip length. When it lands it SUPERSEDES
        the word-count re-ask — that re-ask sized and judged words at one pooled rate,
        which is how showcase ``01a15b57-9403`` "hit" its counts and still shipped every
        beat short with the voice clamped at the drag floor. When it cannot measure (TTS
        down, any error) it returns ``None`` and the re-ask + rate re-measure run exactly
        as before.
        """
        measured_fit = await self._maybe_fit_measured(
            lines, allotments, cfg, ctx, authored=authored_natural is not None
        )
        if measured_fit is not None:
            lines = measured_fit.with_texts(lines)
            # Feed the solver the MEASUREMENT: its every estimate reads word_count/wpm,
            # so each fitted beat's word count becomes the one its measured seconds
            # imply. Footage stays pinned; elastic beats keep their bounds — so the
            # windows the lines were fitted to are the windows re-solved here.
            demands = measured_demands(demands, measured_fit.seconds, cfg.wpm_natural)
            allotments = solve_timeline(
                demands,
                target=cfg.target,
                tolerance_pct=cfg.tolerance_pct,
                wpm_natural=cfg.wpm_natural,
                freeze_pad_ceiling_seconds=cfg.freeze_pad_ceiling_seconds,
            )
            expand_notes: list[str] = []
        else:
            texts_before_reask = [str(line.get("text") or "") for line in lines]
            # The re-ask's spine cap is stated at the CONTENT-basis rate (website_review's
            # `floor_wpm`, as gen budgets words): an authored rate is the operator's word, a
            # measured one passes through, a calibration-passage (or unrecorded) one is held
            # to DEFAULT_WPM_NATURAL.
            content_wpm = content_wpm_for(
                cfg.wpm_natural,
                "authored" if authored_natural is not None else gen.get("resolved_wpm_basis"),
            )
            lines, demands, allotments, expand_notes = await self._maybe_expand_underspoken(
                lines,
                demands,
                allotments,
                cfg,
                clip_seconds,
                content_wpm=content_wpm,
                order_word_budgets=_order_word_budgets(gen),
            )
            # RE-MEASURE THE TEXT THE RE-ASK WROTE — see `_remeasure_if_rewritten`.
            (
                lines,
                demands,
                allotments,
                cfg,
                gen,
                rate_degradations,
            ) = await self._remeasure_if_rewritten(
                texts_before_reask,
                lines,
                ctx,
                gen,
                clip_seconds,
                (demands, allotments, cfg, rate_degradations),
                authored=authored_natural is not None,
            )
        return (
            lines,
            demands,
            allotments,
            cfg,
            gen,
            rate_degradations,
            expand_notes,
            measured_fit,
        )

    async def _maybe_fit_measured(
        self,
        lines: list[dict[str, object]],
        allotments: list[Allotment],
        cfg: _TimingConfig,
        ctx: RunContext,
        *,
        authored: bool,
    ) -> _MeasuredFit | None:
        """Fit every spoken line to its beat's window by MEASUREMENT — or ``None``.

        Replaces the word-count UNDERSPOKEN re-ask whenever it lands (see
        :mod:`~src.services.stages.demo_video.narration_fit` for the defect). Armed on
        the SAME switch as that re-ask (relief ``auto``, not ``reconcile_only``) so a run
        that did not opt in stays byte-identical and pays nothing, and skipped on an
        AUTHORED rate (the operator's rate outranks every measurement, as in generic).

        FAIL-OPEN: ``None`` whenever the first take of ANY line cannot be measured
        (TTS down, a backend error, a zero-length clip) or anything else raises — the
        caller then runs today's re-ask path unchanged. A failed REWRITE or re-synthesis
        never lands here: ``line_fit`` keeps the best take in hand. Cancellation
        propagates.
        """
        if cfg.relief_mode != "auto" or cfg.mode == "reconcile_only" or authored:
            return None
        try:
            return await self._fit_measured(lines, allotments, ctx)
        except Exception:
            log.warning(
                "reconcile_timeline: measured line fit failed — falling back to the "
                "word-count underspoken re-ask",
                exc_info=True,
            )
            return None

    def _fit_tts(self) -> TTSClient:
        """The injected client, else a default one built through the SAME seam the
        speech-rate probe uses (``speech_rate.TTSClient``) — so the one place a unit
        test blocks network TTS construction covers this stage's fit too."""
        if self._tts is None:
            self._tts = _speech_rate.TTSClient()
        return self._tts

    async def _fit_measured(
        self, lines: list[dict[str, object]], allotments: list[Allotment], ctx: RunContext
    ) -> _MeasuredFit | None:
        snapshot = ctx.config_snapshot
        items, _dropped = partition_for_stage(snapshot.get("verbatim_items") or [], self.name)
        targets = fit_targets(
            lines,
            allotments,
            voice=snapshot.get("voice") or {},
            brief=snapshot.get("brief") or {},
            verbatim_items=items,
        )
        if not targets:
            return None
        client = self._fit_tts()
        run_id = str(ctx.run_id)
        gate = asyncio.Semaphore(_FIT_SYNTH_CONCURRENCY)

        async def _synth(pos: int, text: str, speed: float | None) -> LineTake:
            # Stored under NARRATE's stage at the speed persisted as the line's
            # `vo_rate`, with narrate's voice: `deterministic_clip_id(text, voice, speed)`
            # is then the id narrate looks up, so it plays this exact clip.
            requested = FIRST_TAKE_SPEED if speed is None else float(speed)
            async with gate:
                clip = await synthesize_and_store_clip(
                    client,
                    text=text,
                    voice=targets[pos].voice,
                    run_id=run_id,
                    stage_name=DemoVideoNarrateStage.name,
                    speed=requested,
                )
            return LineTake(
                text=text,
                seconds=float(clip.seconds),
                speed=requested,
                pause_seconds=clip.service_pause_seconds,
            )

        # TaskGroup, not gather: one failed first take CANCELS its siblings before the
        # fallback, so no paid clips keep landing after the stage has given up on them.
        async with asyncio.TaskGroup() as group:
            first = [
                group.create_task(_synth(pos, t.text, FIRST_TAKE_SPEED))
                for pos, t in enumerate(targets)
            ]
        takes = [task.result() for task in first]
        if any(not math.isfinite(t.seconds) or t.seconds <= 0.0 for t in takes):
            log.warning("reconcile_timeline: measured line fit skipped — a zero-length take")
            return None

        wm_gate = asyncio.Semaphore(_FIT_REWRITE_CONCURRENCY)

        async def _rewrite(
            pos: int, text: str, measured: float, budget: float, want: int
        ) -> str | None:
            async with wm_gate:
                return await self._fit_rewrite(
                    targets[pos], text, measured=measured, budget=budget, want=want
                )

        fitted = await fit_lines(
            takes,
            [t.budget for t in targets],
            synth=_synth,
            rewrite=_rewrite,
            rewritable=[t.rewritable for t in targets],
        )
        result = _MeasuredFit(
            fits={t.line_id: f for t, f in zip(targets, fitted, strict=True)},
        )
        log.info(
            "reconcile_timeline: measured line fit — %s",
            fit_summary(list(result.fits.items())),
        )
        return result

    async def _fit_rewrite(
        self, target: FitTarget, text: str, *, measured: float, budget: float, want: int
    ) -> str | None:
        """Rewrite one line to ``want`` words; ``None`` when no usable reply came back.

        ONE model call, plus ONE recount re-ask (:func:`recount_message`) when the reply
        misses ``want`` by more than :data:`FIT_WORD_TOLERANCE` words — the model's word
        count, not the target, is what missed on the recorded showcase runs (see that
        constant). Of the two replies the one nearer ``want`` is returned (the first on
        a tie); a refused or FAILED re-ask keeps the first. Every reply passes the underspoken
        re-ask's refusals: a URL, or more than :data:`_MAX_UNGROUNDED_WORDS` words
        absent from the beat and its recording, is refused (``line_fit`` then keeps the
        take in hand). Worker split as before: ``beat-expander`` when the reply must
        grow, ``beat-rewriter`` when it must shrink.
        """
        lengthen = want > len(text.split())
        uncovered = _uncovered_steps(text, target.source) if lengthen and target.source else []
        first = await self._fit_reply(
            target,
            lengthen=lengthen,
            message=fit_message(
                text,
                measured=measured,
                budget=budget,
                want=want,
                source=target.source,
                uncovered=uncovered,
            ),
        )
        if first is None or not needs_recount(first, want):
            return first
        grow = want > len(first.split())
        try:
            second = await self._fit_reply(
                target,
                lengthen=grow,
                message=recount_message(
                    first,
                    want=want,
                    budget=budget,
                    source=target.source,
                    uncovered=(
                        _uncovered_steps(first, target.source) if grow and target.source else []
                    ),
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- the re-ask is optional; keep the paid first
            # `first` is already paid for and usable; a transient worker error on the
            # OPTIONAL recount must not discard it (and with it the whole line fit).
            log.warning(
                "reconcile_timeline: recount re-ask of %s failed (%s: %s); keeping first reply",
                target.line_id,
                type(exc).__name__,
                exc,
            )
            return first
        log.info(
            "reconcile_timeline: measured-fit rewrite of %s asked %d words, got %d; "
            "recount re-ask got %s",
            target.line_id,
            want,
            len(first.split()),
            "refused" if second is None else len(second.split()),
        )
        if second is not None and abs(word_miss(second, want)) < abs(word_miss(first, want)):
            return second
        return first

    async def _fit_reply(self, target: FitTarget, *, lengthen: bool, message: str) -> str | None:
        """One fit model call; the reply, or ``None`` when empty, a URL, or ungrounded."""
        resp = await self._wm.message(
            enable_reasoning=reasoning_for_attempt(),
            worker_name=_EXPANDER_WORKER if lengthen else _REWRITER_WORKER,
            message=message,
            system_prompt=FIT_SYSTEM,
            output_schema=_REWRITER_SCHEMA,
            max_tokens=_EXPANDER_MAX_TOKENS,
            model_override=self._model_tier,
        )
        new = _rewritten_text(resp.output)
        if not new or contains_url(new):
            return None
        if target.source and (
            len(_ungrounded_words(new, target.text, target.source)) > _MAX_UNGROUNDED_WORDS
        ):
            log.warning(
                "reconcile_timeline: measured-fit rewrite of %s refused as ungrounded",
                target.line_id,
            )
            return None
        return new

    async def _expand_one_beat(
        self,
        beat_id: str,
        original: str,
        *,
        word_ask: int,
        word_floor: int,
        word_ceiling: int | None = None,
        source: str | None = None,
    ) -> str | None:
        """Up to :data:`_EXPAND_MAX_PASSES` re-asks on ONE beat; the grown text or ``None``.

        The loop the single-shot version was missing. A model that under-delivers on a word
        target (the common case) used to leave the beat short with no second chance; here a
        pass that ADDED words buys the next one, and a pass that did not ends the beat.

        Every pass re-asks against the text the beat has NOW — ``_expand_msg`` restates the
        current count from the text it is handed — so the model is never told it has 6 words
        when the previous pass already took it to 18.

        Returns ``None`` when nothing usable was produced (empty, never longer than the
        original, or every pass overshot), in which case the caller keeps the original
        text and the beat stays flagged. A partial win is returned: text that grew
        but did not reach the ask is still strictly better than the original, and the
        deterministic re-solve — not this method — decides whether it cleared the flag.

        ``word_ceiling`` is :func:`_reask_word_ceiling` — the beat's own window at the
        solved rate, never looser than the ratio ceiling. ``None`` keeps the ratio alone.

        ⛔ A reply carrying a URL / raw path (:func:`contains_url`) is NEVER applied: the
        word budget prices it as one word and the voice spells it out. It spends a pass
        (so the bound on calls is unchanged) and the next pass re-asks against the text
        the beat has now; if every pass returns one, the beat keeps what it had.

        An OVERSHOOT (more words than ``ceiling``) is handled the same way: it is dropped,
        spends its pass, and the next pass re-asks against the text the beat has now. It
        used to END the beat, so a single-beat showcase got one ask instead of
        :data:`_EXPAND_MAX_PASSES` — and an overshoot on pass 1 is exactly when a second,
        tighter answer is most likely.
        """
        best = original
        best_words = len(original.split())
        ratio_ceiling = int(word_ask * _EXPAND_CEILING_RATIO)
        ceiling = ratio_ceiling if word_ceiling is None else min(ratio_ceiling, word_ceiling)
        for attempt in range(_EXPAND_MAX_PASSES):
            resp = await self._wm.message(
                enable_reasoning=reasoning_for_attempt(),
                worker_name=_EXPANDER_WORKER,
                message=_expand_msg(
                    best,
                    word_ask,
                    ceiling,
                    source,
                    _uncovered_steps(best, source) if source else (),
                ),
                system_prompt=_EXPANDER_SYSTEM,
                output_schema=_EXPANDER_SCHEMA,
                max_tokens=_EXPANDER_MAX_TOKENS,
                model_override=self._model_tier,
            )
            new_text = _rewritten_text(resp.output)
            new_words = len(new_text.split())
            if new_text and contains_url(new_text):
                log.warning(
                    "reconcile_timeline underspoken: beat-expander pass %d/%d put a URL or "
                    "raw path into beat %s — refused (the voice spells it out, the word "
                    "budget counts it as one word); the beat keeps its %d-word text.",
                    attempt + 1,
                    _EXPAND_MAX_PASSES,
                    beat_id,
                    best_words,
                )
                continue
            novel = _ungrounded_words(new_text, original, source) if source else []
            if len(novel) > _MAX_UNGROUNDED_WORDS:
                # Same handling as a URL: spend the pass, keep what the beat has, and let
                # the next pass re-ask (grounded) against it.
                log.warning(
                    "reconcile_timeline underspoken: beat-expander pass %d/%d on beat %s "
                    "introduced %d word(s) absent from the beat and its recording (%s) — "
                    "refused as ungrounded; the beat keeps its %d-word text.",
                    attempt + 1,
                    _EXPAND_MAX_PASSES,
                    beat_id,
                    len(novel),
                    ", ".join(novel[:8]),
                    best_words,
                )
                continue
            if not new_text or new_words <= best_words:
                log.warning(
                    "reconcile_timeline underspoken: beat-expander pass %d/%d returned "
                    "empty or not-longer text for beat %s (has %d words, ask %d, floor "
                    "%d) — no progress, stopping this beat.",
                    attempt + 1,
                    _EXPAND_MAX_PASSES,
                    beat_id,
                    best_words,
                    word_ask,
                    word_floor,
                )
                break
            if new_words > ceiling:
                log.warning(
                    "reconcile_timeline underspoken: beat-expander pass %d/%d OVERSHOT "
                    "beat %s — %d words against a %d-word ask (ceiling %d: the beat's "
                    "window at the solved rate, or the ratio ceiling if tighter). Applying "
                    "it would overrun the beat, so it was dropped; the beat keeps its "
                    "%d-word text and the next pass re-asks.",
                    attempt + 1,
                    _EXPAND_MAX_PASSES,
                    beat_id,
                    new_words,
                    word_ask,
                    ceiling,
                    best_words,
                )
                continue
            best, best_words = new_text, new_words
            if best_words >= word_ask:
                break
        return None if best_words <= len(original.split()) else best

    async def _expand_one_beat_isolated(
        self,
        beat_id: str,
        original: str,
        *,
        word_ask: int,
        word_floor: int,
        word_ceiling: int | None,
        source: str | None = None,
    ) -> tuple[str | None, bool]:
        """:meth:`_expand_one_beat` with PER-BEAT failure isolation: ``(grown, failed)``.

        A WM call that raised used to escape :meth:`_expand_underspoken`'s whole per-beat
        loop into ``_maybe_expand_underspoken``'s catch-all, which returned the ORIGINAL
        lines — discarding every rewrite earlier beats had already PAID for. One beat's
        outage now costs that beat alone; the caller keeps the earned patches and names
        the failed beat in its note (⇒ the run is ``degraded``). ``except Exception``
        does not catch ``CancelledError``, so cancellation still propagates.
        """
        try:
            grown = await self._expand_one_beat(
                beat_id,
                original,
                word_ask=word_ask,
                word_floor=word_floor,
                word_ceiling=word_ceiling,
                source=source,
            )
        except Exception as exc:
            log.error(
                "reconcile_timeline underspoken: beat-expander call for beat %s failed "
                "(%s) — the beat keeps its original text; other beats are unaffected.",
                beat_id,
                type(exc).__name__,
                exc_info=True,
            )
            return None, True
        return grown, False

    async def _expand_underspoken(
        self,
        draft: list[dict[str, object]],
        demands: list[TimeDemand],
        allotments: list[Allotment],
        cfg: _TimingConfig,
        clip_seconds: dict[str, float],
        *,
        content_wpm: float | None = None,
        order_word_budgets: Mapping[str, int] | None = None,
    ) -> tuple[list[dict[str, object]], list[TimeDemand], list[Allotment], list[str]]:
        """ONE bounded re-ask per UNDERSPOKEN beat, then a deterministic re-solve.

        The missing half of this stage's relief machinery. The trim loop above answers
        "too many words for the seconds"; nothing answered its mirror, so a beat with too
        FEW words had its ``vo_rate`` clamped to :data:`VO_RATE_NATURAL_FLOOR`, was
        delivered in ``natural / 0.80`` seconds — still SHORT of its allotment by
        construction — and raised a note that fixed nothing. Both live complaints fall out
        of that one clamp: the listener hears dragged speech AND the master ships under
        its order.

        The word target is NOT invented here. It is :func:`_underspoken_word_targets`,
        which inverts the very inequality :func:`vo_rate_for` used to raise the flag —
        target and flag cannot drift the way a separately-derived budget would.

        ⛔ But it is inverted at :data:`_EXPAND_ASK_RATE` (natural pace), NOT at the
        ``underspoken`` floor. The floor count is the count that fills the beat AT 0.80,
        so a beat topped up to it is by construction a beat that clamps back to 0.80 —
        measured on live run 01a0a136, where the re-ask landed 21/26/21 words (the floor
        counts) and the stage still recorded "clip-4 needs speed 0.746 … clamped to 0.80".
        The floor stays the ACCEPT threshold (it is what the re-solve below re-tests); the
        ASK is the natural-rate count.

        HARD BOUNDS, because this spends money on a paid run mid-flight:

        * **At most :data:`_EXPAND_MAX_PASSES` calls per beat**, and every pass after the
          first has to be EARNED: the loop continues only while the beat is still short of
          its ask AND the previous pass actually added words. A pass that adds nothing
          stops the beat there, so this cannot cycle and cannot be talked into spending
          more than the constant.
        * **The rewrite must actually help, or it is dropped.** It must be strictly
          LONGER than what the beat has now; and it must not overshoot past
          :data:`_EXPAND_CEILING_RATIO` of the ask, which would convert an underspoken beat
          into an ``unsayable`` one — a strictly worse trade, since the compiler can hold a
          frame but cannot cut audio. The ``underspoken`` FLOOR remains the accept
          threshold in the only sense that matters: the re-solve below re-tests it.
        * **The SPINE is capped, not only each beat.** After the re-ask the total words may
          not exceed :func:`_reask_total_word_cap` of the body (order minus capped cards)
          at the content-basis rate; per-beat ceilings shrink proportionally to fit
          (:func:`_total_capped_ceilings`). ``content_wpm`` ``None`` ⇒ the natural rate,
          capped at :data:`DEFAULT_WPM_NATURAL`.
        * **Never raises, never fabricates.** Every rejection keeps the ORIGINAL text and
          logs why, so a live no-op is diagnosable instead of silent.

        The deterministic solver then re-collects and re-solves over the lengthened copy,
        so every number downstream comes from the solver reading the REAL text — never
        from this method asserting that it improved something. That is the same
        verify-by-re-solve discipline the trim loop uses, and it is what makes a rewrite
        that did not actually help visible as a beat that is STILL flagged.
        """
        targets = _underspoken_word_targets(allotments, demands, cfg.wpm_natural)
        asks = _underspoken_word_targets(
            allotments, demands, cfg.wpm_natural, rate=_EXPAND_ASK_RATE
        )
        eligible = _expandable_underspoken_lines(draft, targets)
        if not eligible:
            log.info(
                "reconcile_timeline underspoken: %d flagged beat(s) but none is eligible "
                "to lengthen (claim-bearing / locked / no narration text) — nothing "
                "re-asked.",
                len(targets),
            )
            return draft, demands, allotments, []

        by_id = {str(line["id"]): line for line in draft}
        allotment_by_id = {a.id: a for a in allotments}
        patches: dict[str, str] = {}
        refused: list[str] = []
        errored: list[str] = []
        # Window ceilings first (no spend), so the SPINE-level cap can shrink them together.
        window_ceilings = {
            beat_id: _reask_word_ceiling(
                asks.get(beat_id, (word_floor, has))[0],
                allotment_by_id[beat_id].seconds + cfg.freeze_pad_ceiling_seconds,
                cfg.wpm_natural,
            )
            for beat_id, (word_floor, has) in eligible.items()
        }
        current_words = {
            str(line["id"]): len(str(line.get("text") or "").split()) for line in draft
        }
        total_cap = _reask_total_word_cap(
            cfg.target,
            content_wpm if content_wpm is not None else min(cfg.wpm_natural, DEFAULT_WPM_NATURAL),
        )
        capped = _total_capped_ceilings(
            {bid: (current_words.get(bid, 0), c) for bid, c in window_ceilings.items()},
            sum(w for bid, w in current_words.items() if bid not in window_ceilings),
            total_cap,
        )
        for beat_id, (word_floor, has) in sorted(eligible.items()):
            original = str(by_id[beat_id].get("text", ""))
            # The ask is the natural-rate count; fall back to the floor only if this beat
            # somehow has no ask row (it cannot, both dicts are built off the same
            # allotments — belt and braces so a KeyError can never fail a paid run).
            word_ask = asks.get(beat_id, (word_floor, has))[0]
            word_ceiling = window_ceilings[beat_id]
            word_ask = _ask_with_order_share(
                word_ask, word_ceiling, (order_word_budgets or {}).get(beat_id)
            )
            if word_ceiling < word_ask:
                # The beat's own window at the solved rate holds FEWER words than the
                # natural-rate ask (e.g. `freeze_pad_ceiling_seconds: 0` with 10.3s at
                # 136.06 wpm: ask 24, window 23). No reply can be both >= the ask and
                # inside the window, so the prompt would contradict itself; spend nothing
                # and keep the text, exactly as an unusable reply would.
                log.info(
                    "reconcile_timeline underspoken: beat %s not re-asked — its window "
                    "holds %d words at the solved rate, below the %d-word ask.",
                    beat_id,
                    word_ceiling,
                    word_ask,
                )
                refused.append(beat_id)
                continue
            # THE SPINE CAP. The proportional share of the whole-spine room; the ask is
            # pulled down to it so the prompt never asks for more than may be applied.
            if capped[beat_id] < word_ceiling:
                word_ceiling = capped[beat_id]
                if word_ceiling <= current_words.get(beat_id, 0):
                    log.info(
                        "reconcile_timeline underspoken: beat %s not re-asked — the spine "
                        "is already at its %d-word total cap (body %.2fs at the content rate "
                        "x %.2f).",
                        beat_id,
                        total_cap,
                        cfg.target,
                        _REASK_TOTAL_WORD_RATIO,
                    )
                    refused.append(beat_id)
                    continue
                word_ask = min(word_ask, word_ceiling)
            grown, failed = await self._expand_one_beat_isolated(
                beat_id,
                original,
                word_ask=word_ask,
                word_floor=word_floor,
                word_ceiling=word_ceiling,
                source=_beat_source(by_id[beat_id]),
            )
            if grown is None:
                # A FAILED call and an unusable reply both keep the original text, but
                # only the first is an outage — the note names them separately.
                (errored if failed else refused).append(beat_id)
                continue
            patches[beat_id] = grown

        errored_note = _expand_errored_note(errored)
        if not patches:
            note = _expand_nothing_usable_note(eligible, errored, errored_note)
            log.warning(note)
            return draft, demands, allotments, [note]

        expanded = [
            {**line, "text": patches[str(line["id"])]} if str(line["id"]) in patches else line
            for line in draft
        ]
        # RE-SOLVE, never assert. The allotments below are the solver's verdict on the
        # text that actually exists now — so a beat this pass failed to fix is still
        # flagged by `_timeline_notes`, and a beat it did fix silently stops being.
        new_demands = collect_demands(
            expanded,
            clip_seconds=clip_seconds,
            wpm_band=cfg.wpm_band,
            wpm_natural=cfg.wpm_natural,
            target=cfg.target,
        )
        new_allotments = solve_timeline(
            new_demands,
            target=cfg.target,
            tolerance_pct=cfg.tolerance_pct,
            wpm_natural=cfg.wpm_natural,
            freeze_pad_ceiling_seconds=cfg.freeze_pad_ceiling_seconds,
        )
        still_flagged = sorted(a.id for a in new_allotments if a.underspoken)
        note = (
            f"reconcile_timeline: UNDERSPOKEN RE-ASK lengthened {len(patches)} beat(s) "
            f"({', '.join(sorted(patches))}) toward their natural-rate word count so the "
            f"delivery rate no longer has to drag below the natural floor."
        )
        if refused:
            note += (
                f" {len(refused)} beat(s) ({', '.join(sorted(refused))}) kept their "
                f"original text — the re-ask came back empty, not-longer, or overshot."
            )
        note += errored_note
        if still_flagged:
            note += (
                f" {len(still_flagged)} beat(s) ({', '.join(still_flagged)}) are STILL "
                f"underspoken after the re-solve."
            )
        log.info(note)
        return expanded, new_demands, new_allotments, [note]

    async def _dispose_directives(
        self,
        draft: list[dict[str, object]],
        directives: list[object],
        allowed_ids: frozenset[str],
    ) -> list[dict[str, object]]:
        """Dispose a timing-director directive list onto the draft (spec 23 §5).

        Reduces the directives to per-beat trim targets (:func:`_parse_trim_directives`),
        then fires ONE focused ``beat-rewriter`` call per eligible trim target to produce
        genuinely shorter text hitting the proposed word target. Only a rewrite that is
        actually SHORTER (fewer words) than the original is applied — an empty or
        not-shorter rewrite is dropped, never fabricated. ``merge`` / ``cut`` /
        ``reallocate`` ops and out-of-set ids are rejected-with-log. The returned spine is
        id-stable by construction (only ``text`` on allowed ids is ever changed), so the
        C0 id-set guard and the join ids survive. Every no-op path logs so a live silent
        no-op is diagnosable.
        """
        if not directives:
            log.warning(
                "reconcile_timeline relief: timing-director proposed an EMPTY directive set "
                "over eligible beats %s — no beat shortened this pass (the model may be "
                "under-resourced; consider PIPELINE_TIMING_DIRECTOR_MODEL_TIER).",
                sorted(allowed_ids),
            )
            return draft
        trims, dropped, unsupported = _parse_trim_directives(
            directives, allowed_ids, _word_counts(draft)
        )
        if unsupported:
            log.warning(
                "reconcile_timeline relief: timing-director proposed %d structural op(s) %s "
                "not yet disposed (merge/cut/reallocate need the C0 id-set guard relaxed) — "
                "rejected this pass.",
                len(unsupported),
                sorted(set(unsupported)),
            )
        if dropped:
            log.warning(
                "reconcile_timeline relief: timing-director trimmed %d id(s) outside the "
                "eligible set %s — ignored (hallucinated / case-mismatched / not "
                "shortenable): %s",
                len(dropped),
                sorted(allowed_ids),
                dropped,
            )
        if not trims:
            log.warning(
                "reconcile_timeline relief: no disposable (trim) directive over eligible "
                "beats %s — no beat shortened this pass.",
                sorted(allowed_ids),
            )
            return draft
        by_id = {str(ln["id"]): ln for ln in draft}
        patches: dict[str, str] = {}
        for tid, word_target in trims.items():
            original = str(by_id[tid].get("text", ""))
            resp = await self._wm.message(
                enable_reasoning=reasoning_for_attempt(),
                worker_name=_REWRITER_WORKER,
                message=_rewrite_msg(original, word_target),
                system_prompt=_REWRITER_SYSTEM,
                output_schema=_REWRITER_SCHEMA,
                max_tokens=_REWRITER_MAX_TOKENS,
                model_override=self._model_tier,
            )
            new_text = _rewritten_text(resp.output)
            if new_text and len(new_text.split()) < len(original.split()):
                patches[tid] = new_text
            else:
                log.warning(
                    "reconcile_timeline relief: beat-rewriter returned empty or not-shorter "
                    "text for beat %s (target %d words) — kept the original.",
                    tid,
                    word_target,
                )
        if not patches:
            log.warning(
                "reconcile_timeline relief: every trim rewrite came back empty or not "
                "shorter — zero net word reduction this pass over %s.",
                sorted(trims),
            )
        return [
            {**ln, "text": patches[str(ln["id"])]} if str(ln["id"]) in patches else ln
            for ln in draft
        ]

    async def _apply_structural_prepass(
        self,
        lines: list[dict[str, object]],
        ctx: RunContext,
        cfg: _TimingConfig,
        clip_seconds: dict[str, float],
        footage_ids: frozenset[str],
    ) -> tuple[list[dict[str, object]], dict[str, float], list[str], str | None]:
        """C13 phase 2 structural relief — the deterministic PRE-PASS before the trim loop.

        Fires ONLY when at least one infeasible beat cannot be trimmed in place (claim-bearing
        / locked / footage-pinned): structural surgery is a last resort, so an infeasible
        timeline whose over-long beats are ALL trim-eligible skips this entirely and is left to
        the loop's in-place shortening (byte-identical to phase 1). When it fires it makes ONE
        ``timing-director`` call proposing ``cut`` / ``merge`` / ``reallocate`` over the
        currently-infeasible spine and applies them deterministically (:func:`_apply_structural_ops`).

        SOFT-DEGRADE: any WM error / timeout / off-contract (non-dict) response logs a warning
        and returns ``lines`` UNCHANGED — a structural failure never hard-fails the render, it
        just falls back to the trim-only loop. The WM call is routed through
        ``upstream.guarded`` + ``stage_call``, so a TRANSIENT upstream spends the configured
        ``llm_retries`` before that degrade and a PERMANENT one degrades immediately carrying
        the real upstream sentence in its note — the two are distinguishable, and neither
        raises.

        Returns ``(new_lines, reallocate_pins, rejected, soft_degrade_note)``; the caller
        re-solves the deterministic timeline over ``new_lines`` so the loop's ``_spine_ids``
        derive from the (possibly reduced / repinned) spine and the loop-engine C0 guard never
        trips. ``soft_degrade_note`` is a human-readable string on the two soft-degrade paths
        (WM error / off-contract output) so a live no-op is surfaced in the run manifest, not
        just the pod log — ``None`` on every normal path.
        """
        demands = collect_demands(
            lines,
            clip_seconds=clip_seconds,
            wpm_band=cfg.wpm_band,
            wpm_natural=cfg.wpm_natural,
            target=cfg.target,
        )
        flagged = frozenset(infeasible_beats(demands, cfg.target))
        if not flagged:
            return lines, {}, [], None
        eligible_ids = frozenset(
            str(ln["id"])
            for ln in _eligible_infeasible_lines(
                lines, cfg.target, cfg.wpm_band, cfg.wpm_natural, clip_seconds
            )
        )
        if flagged <= eligible_ids:
            # Every infeasible beat is trim-eligible ⇒ the loop's in-place shortening can reach
            # feasibility; spend no structural WM call and change nothing.
            return lines, {}, [], None

        infeasible_lines = [ln for ln in lines if str(ln["id"]) in flagged]
        protected_ids = frozenset(
            str(ln["id"]) for ln in infeasible_lines if _is_protected(ln, footage_ids)
        )
        overrun = sum(d.min for d in demands) - cfg.target
        director = (ctx.config_snapshot.get("direction") or {}).get("intent")
        # THE ONE UPSTREAM POLICY (stages/upstream.py), not a local ``except``.
        #
        # This call used to be a BARE ``self._wm.message`` under ``except Exception``.
        # That is the only WM call in this stage that runs OUTSIDE
        # ``run_evaluator_optimizer`` — which routes both of its injected closures through
        # ``guarded`` — so it was the one site where the policy did not apply, and it got
        # the worst of both halves:
        #
        #   * a genuinely TRANSIENT upstream (a 5xx blip, a 429, the 422 "Exceeded
        #     maximum output retries" bad draw) spent ZERO of the configured
        #     ``llm_retries`` and threw the structural pre-pass away on the first
        #     hiccup — on a timeline that is, by definition, already infeasible;
        #   * a PERMANENT one (402 "configure your LLM provider", a prompt over the
        #     worker's token limit) was logged as an anonymous traceback and reported
        #     with a note that names no cause.
        #
        # ``stage_call(lambda: guarded(...))`` fixes both: ``TransientUpstreamError`` is
        # in ``LLM_RETRY_ON`` and spends the budget, ``PermanentUpstreamError`` is
        # deliberately NOT (see llm_budget.py) and lands on the branch below on attempt 1.
        #
        # BOTH branches still DEGRADE — this pre-pass is a last-resort enrichment over an
        # already-honest deterministic timeline and must never hard-fail a paid render —
        # but they degrade DISTINGUISHABLY: each returns its own ``soft_degrade_note``,
        # which reaches the run manifest and makes the run ``degraded``.
        try:
            resp = await stage_call(
                downshift_reasoning(
                    lambda think: guarded(
                        lambda: self._wm.message(
                            enable_reasoning=think,
                            worker_name=_OPTIMIZER_WORKER,
                            message=_structural_msg(
                                infeasible_lines, protected_ids, cfg.target, overrun, director
                            ),
                            system_prompt=_STRUCTURAL_SYSTEM,
                            output_schema=_STRUCTURAL_SCHEMA,
                            max_tokens=_STRUCTURAL_MAX_TOKENS,
                            model_override=self._model_tier,
                        )
                    )
                ),
                timeout_seconds=self._llm_timeout,
                retries=self._llm_retries,
                retry_on=LLM_RETRY_ON,
                stage_name=f"{self.name}:structural_prepass",
            )
        except PermanentUpstreamError as exc:
            # No retry can fix this one, so none was spent. ``str(exc)`` IS the upstream
            # sentence (status + url + WM's own message) — carry it into the note rather
            # than the generic "call failed", because for the 402 case that sentence is
            # the only thing that tells the operator what to actually do.
            detail = str(exc)[:300]
            log.error(
                "reconcile_timeline structural: timing-director PERMANENT upstream fault "
                "over infeasible beats %s — %s (soft-degrading to the trim-only loop, "
                "spine unchanged, retry budget deliberately NOT spent).",
                sorted(flagged),
                detail,
            )
            return (
                lines,
                {},
                [],
                f"structural relief unavailable (timing-director permanent upstream "
                f"fault: {detail}) — fell back to the trim-only loop; spine unchanged.",
            )
        except Exception:
            # TOLERABLE degrade: the pre-pass is a LAST-RESORT enrichment over an
            # already-honest deterministic timeline, and its failure is reported to the
            # caller as ``soft_degrade_note`` (which reaches the run manifest), so nothing
            # about the deliverable is hidden. ERROR + traceback so the pod log names the
            # WM fault instead of burying it among routine warnings.
            #
            # Reached by a transient upstream that outlived its retries
            # (``StageRetriesExhaustedError``), a clean ``StageTimeoutError``, and any
            # non-upstream defect. Deliberately still broad: the note is the contract.
            log.error(
                "reconcile_timeline structural: timing-director call failed over infeasible "
                "beats %s — soft-degrading to the trim-only loop (spine unchanged).",
                sorted(flagged),
                exc_info=True,
            )
            return (
                lines,
                {},
                [],
                "structural relief unavailable (timing-director call failed) — "
                "fell back to the trim-only loop; spine unchanged.",
            )
        output = resp.output
        if not isinstance(output, dict):
            log.warning(
                "reconcile_timeline structural: timing-director returned a non-dict output "
                "(type=%s) — no structural op applied, falling back to the trim-only loop.",
                resp.output_type,
            )
            return (
                lines,
                {},
                [],
                "structural relief unavailable (timing-director returned off-contract output) — "
                "fell back to the trim-only loop; spine unchanged.",
            )
        directives = _as_list(output.get("directives"))
        new_lines, reallocate_pins, rejected = _apply_structural_ops(
            lines,
            directives,
            footage_ids,
            clip_seconds=clip_seconds,
            wpm_band=cfg.wpm_band,
            wpm_natural=cfg.wpm_natural,
        )
        return new_lines, reallocate_pins, rejected, None

    async def _relief_loop(
        self,
        lines: list[dict[str, object]],
        demands: list[TimeDemand],
        allotments: list[Allotment],
        cfg: _TimingConfig,
        ctx: RunContext,
        clip_seconds: dict[str, float],
    ) -> tuple[
        list[dict[str, object]],
        list[TimeDemand],
        list[Allotment],
        list[str],
        TimelineManifest,
        list[Degradation],
    ]:
        """The OPT-IN, INERT-BY-DEFAULT WM relief loop (spec 23 §5).

        Only the caller's ``relief_mode == "auto"`` AND already-infeasible timeline reaches
        here — an off / feasible run never calls this, so the deterministic path stays
        byte-identical. Runs the bounded ``pacing-critic`` → ``timing-director`` evaluator-
        optimizer loop, re-solves the deterministic timeline against the (possibly shortened)
        spine, and returns ``(lines, demands, allotments, loop_notes, manifest, degradations)``
        (``degradations`` = the coded twins of any degrading ``loop_notes``) — the
        manifest note plus its typed :class:`TimelineManifest` twin (the machine-assertable
        run telemetry), plus a soft structural-relief hint when an infeasible beat could not
        be shortened.
        """
        # C13 phase 2: structural PRE-PASS (cut / merge / reallocate) BEFORE the trim loop.
        # It changes the id-set / budget, so it must run outside run_evaluator_optimizer — the
        # loop's ``_spine_ids`` C0 guard is derived from the REDUCED spine below, never tripped.
        footage_ids = frozenset(d.id for d in demands if d.kind == "footage")
        beats_before_prepass = len(lines)
        (
            lines,
            reallocate_pins,
            structural_rejected,
            structural_note,
        ) = await self._apply_structural_prepass(lines, ctx, cfg, clip_seconds, footage_ids)
        # Count only re-pins on beats that SURVIVED the pre-pass: a director that emits both a
        # ``reallocate`` and a ``cut`` on the same id leaves the cut beat lingering in
        # ``reallocate_pins`` though it is gone from the spine — counting it would double-count
        # the patch (once via before-minus-after, once via the pin) and wrongly list a removed
        # id as reallocated. Filtering to the surviving ids makes the telemetry exact.
        surviving_ids = {str(ln["id"]) for ln in lines}
        structural_reallocated_ids = sorted(pid for pid in reallocate_pins if pid in surviving_ids)
        # cut/merge drop a beat from the spine (before-minus-after); reallocate re-pins one.
        structural_patches_applied = (beats_before_prepass - len(lines)) + len(
            structural_reallocated_ids
        )

        director = (ctx.config_snapshot.get("direction") or {}).get("intent")

        async def critic_fn(draft: list[dict[str, object]]) -> Critique:
            # Re-derive feasibility on the CURRENT draft each pass.
            d = collect_demands(
                draft,
                clip_seconds=clip_seconds,
                wpm_band=cfg.wpm_band,
                wpm_natural=cfg.wpm_natural,
                target=cfg.target,
            )
            flagged = infeasible_beats(d, cfg.target)
            if flagged:
                # Still infeasible: an infeasible draft must NEVER outscore a feasible one,
                # so it is capped strictly below the 3.0 feasible floor — graded by overrun
                # so the loop still prefers the least-infeasible draft. No WM rhythm call is
                # spent while the timeline is not even feasible.
                eligible_ids = {
                    str(ln["id"])
                    for ln in _eligible_infeasible_lines(
                        draft, cfg.target, cfg.wpm_band, cfg.wpm_natural, clip_seconds
                    )
                }
                rerun = (
                    ["demo_video_gen"] if any(fid not in eligible_ids for fid in flagged) else []
                )
                overrun = sum(dd.min for dd in d) - cfg.target
                score = min(2.0, max(0.0, 2.0 - overrun / max(cfg.target, 1.0)))
                return Critique(score=score, rerun_stages=rerun, directives={})
            unsayable_targets = _unsayable_trim_targets(draft, cfg, clip_seconds)
            if unsayable_targets:
                # Feasible by ``Sigma min <= target`` and STILL not shippable: >=1 beat's
                # narration overruns its own shot even at the band ceiling. Scored strictly
                # inside (2.0 infeasible cap, 3.0 feasible floor) so an unsayable draft can
                # never outrank a clean one nor undercut an infeasible one, and so it stays
                # BELOW any relief threshold — which is what actually enters the optimizer.
                # Graded by how many beats are still unsayable so the loop prefers the
                # least-unsayable draft. No WM rhythm call is spent on a draft this loop is
                # about to rewrite anyway.
                share = len(unsayable_targets) / max(len(draft), 1)
                return Critique(score=2.9 - 0.8 * share, rerun_stages=[], directives={})
            # Feasible: score its rhythm 3.0-5.0 (always above the 2.0 infeasible cap).
            resp = await self._wm.message(
                enable_reasoning=reasoning_for_attempt(),
                worker_name=_CRITIC_WORKER,
                message=_critic_msg(draft, d, cfg.target, director),
                system_prompt=_CRITIC_SYSTEM,
                output_schema=_CRITIC_SCHEMA,
                max_tokens=_CRITIC_MAX_TOKENS,
            )
            if isinstance(resp.output, dict):
                output = resp.output
            else:
                # output_schema guarantees a dict in normal operation; an off-contract
                # response would otherwise score neutral SILENTLY — log it so a live
                # regression in the worker contract is visible, not swallowed.
                log.warning(
                    "reconcile_timeline relief: pacing-critic returned a non-dict output "
                    "(type=%s) — scoring its rhythm as neutral for this pass.",
                    resp.output_type,
                )
                output = {}
            # Clamp before the affine map: the schema cannot enforce the 1-5 range
            # (guided decoding ignores numeric bounds), and an out-of-range rhythm
            # would push `score` clean out of the 3.0-5.0 feasible band this stage
            # asserts sits above the 2.0 infeasible cap — an unclamped 12 yields 7.8.
            rhythm = clamp_score(output.get("overall_score"))
            score = 3.0 + (rhythm / 5.0) * 2.0
            return Critique(score=score, rerun_stages=[], directives=output)

        async def optimizer_fn(
            draft: list[dict[str, object]], critique: Critique
        ) -> list[dict[str, object]]:
            eligible = _eligible_infeasible_lines(
                draft, cfg.target, cfg.wpm_band, cfg.wpm_natural, clip_seconds
            )
            if not eligible:
                return await self._trim_unsayable_or_flag(draft, cfg, clip_seconds)
            d = collect_demands(
                draft,
                clip_seconds=clip_seconds,
                wpm_band=cfg.wpm_band,
                wpm_natural=cfg.wpm_natural,
                target=cfg.target,
            )
            overrun = sum(dd.min for dd in d) - cfg.target
            allowed = frozenset(str(ln["id"]) for ln in eligible)
            resp = await self._wm.message(
                enable_reasoning=reasoning_for_attempt(),
                worker_name=_OPTIMIZER_WORKER,
                message=_optimizer_msg(eligible, cfg.target, overrun),
                system_prompt=_OPTIMIZER_SYSTEM,
                output_schema=_OPTIMIZER_SCHEMA,
                max_tokens=_OPTIMIZER_MAX_TOKENS,
                model_override=self._model_tier,
            )
            if isinstance(resp.output, dict):
                directives = _as_list(resp.output.get("directives"))
            else:
                # A non-dict optimizer response yields zero directives; make that visible
                # instead of a silent "no beat shortened".
                log.warning(
                    "reconcile_timeline relief: timing-director returned a non-dict output "
                    "(type=%s) — no directive proposed this pass.",
                    resp.output_type,
                )
                directives = []
            return await self._dispose_directives(draft, directives, allowed)

        result = await run_evaluator_optimizer(
            lines,
            critic_fn=critic_fn,
            optimizer_fn=optimizer_fn,
            threshold=cfg.relief_threshold,
            id_extractor=_spine_ids,
            max_passes=cfg.relief_iterations,
            stage_name=self.name,
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
        )

        # Re-solve the deterministic timeline against the (possibly shortened) spine so every
        # downstream number is honest for the copy narrate will actually speak.
        lines = result.draft
        demands = collect_demands(
            lines,
            clip_seconds=clip_seconds,
            wpm_band=cfg.wpm_band,
            wpm_natural=cfg.wpm_natural,
            target=cfg.target,
        )
        allotments = solve_timeline(
            demands,
            target=cfg.target,
            tolerance_pct=cfg.tolerance_pct,
            wpm_natural=cfg.wpm_natural,
            freeze_pad_ceiling_seconds=cfg.freeze_pad_ceiling_seconds,
        )
        # Honest convergence classification from the RE-SOLVED demands (spec 23 §7.7). The
        # loop-engine's ``result.converged`` reflects the relief SCORE threshold; feasibility is
        # a separate, deterministic fact recomputed here so the outcome is machine-assertable.
        overrun_seconds = _overrun_seconds(demands, cfg.target)
        convergence_outcome = _classify_convergence(lines, demands, cfg)
        # The un-cuttable floor on the spine's beat count: footage-pinned / claim-bearing /
        # author-locked / time_lock beats no in-place op may drop. Computed exactly as
        # _classify_convergence does (footage from the RE-SOLVED demands, protected via
        # _is_protected over the post-loop spine), and carried on the manifest so the PR-C
        # auto-converge hook clamps its target_beats to it — it can never ask a gen re-run to
        # emit fewer beats than the protected floor (which would re-render without shrinking).
        final_footage_ids = frozenset(d.id for d in demands if d.kind == "footage")
        protected_beat_count = sum(1 for line in lines if _is_protected(line, final_footage_ids))

        manifest = build_timeline_manifest(
            demands,
            allotments,
            target_seconds=cfg.target,
            card_seconds=cfg.card_seconds,
            loop_iterations=result.passes,
            converged=result.converged,
            pacing_score=result.score,
            convergence_outcome=convergence_outcome,
            structural_relief_requested=bool(result.rerun_stages),
            overrun_seconds=overrun_seconds,
            protected_beat_count=protected_beat_count,
            structural_rejected_ids=structural_rejected,
            structural_reallocated_ids=structural_reallocated_ids,
            structural_patches_applied=structural_patches_applied,
        )
        loop_notes = [
            f"reconcile_timeline: timeline_manifest={json.dumps(manifest, sort_keys=True)}"
        ]
        loop_degradations: list[Degradation] = []
        if structural_note is not None:
            loop_notes.append(f"reconcile_timeline: {structural_note}")
            # Every ``structural_note`` is a soft-degrade (the timing-director call failed
            # or answered off-contract) — see ``_apply_structural_prepass``.
            loop_degradations.append(
                degradation(TIMELINE_RELIEF_UNAVAILABLE, f"reconcile_timeline: {structural_note}")
            )
        # The user-facing degrade note is gated on the CONVERGENCE OUTCOME, not on
        # ``result.rerun_stages``. ``rerun_stages`` is only set when a *flagged* beat is
        # ineligible, and ``flagged`` excludes pinned beats — so a footage-dominated MIXED
        # spine (the exact case ``_classify_convergence`` was rewritten to catch) leaves it
        # FALSE and used to ship an over-target picture with the user told NOTHING. The spawn
        # policy already stopped trusting that flag (``structural_relief_policy`` §2); this
        # note now agrees with it.
        if convergence_outcome == "cant_converge":
            loop_notes.append(
                self._cant_converge_note(
                    ctx,
                    overrun_seconds=overrun_seconds,
                    rerun_stages=list(result.rerun_stages),
                    structural_relief_requested=bool(result.rerun_stages),
                    # The SAME manifest the spawner will re-read post-completion — so the
                    # note's forecast is computed from the spawner's own inputs.
                    manifest=manifest,
                )
            )
        # Typed twin of the same dict (identical keys) for the structured stage-output
        # field — the string note stays for humans/log greppers, this is what e2e asserts on.
        manifest_model = TimelineManifest.model_validate(manifest)
        return lines, demands, allotments, loop_notes, manifest_model, loop_degradations


def _expand_errored_note(errored: list[str]) -> str:
    """The note suffix naming the beats whose expander call FAILED (``""`` when none)."""
    if not errored:
        return ""
    return (
        f" {len(errored)} beat(s) ({', '.join(sorted(errored))}) were NOT re-asked: "
        f"the WM/LLM call failed, so they keep their original text."
    )


def _expand_nothing_usable_note(
    eligible: Mapping[str, object], errored: list[str], errored_note: str
) -> str:
    """The note when the underspoken re-ask applied NO patch.

    Every eligible beat errored ⇒ the re-ask was UNAVAILABLE, worded exactly as the
    wrapper's catch-all words a whole-loop outage. Otherwise some replies came back and
    none was usable, and any beats that errored are named after that.
    """
    if errored and len(errored) == len(eligible):
        return (
            "reconcile_timeline: UNDERSPOKEN RE-ASK UNAVAILABLE (WM/LLM call failed) — kept "
            "the deterministic allotments; the underspoken beats were not lengthened this run."
        )
    return (
        f"reconcile_timeline: UNDERSPOKEN RE-ASK produced nothing usable over "
        f"{', '.join(sorted(eligible))} — every rewrite came back empty, "
        f"not-longer, or past the overshoot ceiling. The beats keep their "
        f"original text and stay flagged; narration on them is still delivered "
        f"at the natural-rate floor.{errored_note}"
    )
