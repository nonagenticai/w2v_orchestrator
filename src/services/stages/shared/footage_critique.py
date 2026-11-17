"""Shared footage-critique evaluator-optimizer (L4) — score → nudge → re-render.

An OPT-IN, SOFT-gate stage that scores an already-compiled directed-footage video
shot-by-shot, nudges the direction plan for the shots that scored badly, and
re-renders — bounded to ≤2 passes by the shared ``loop_engine`` and ALWAYS
accepting the current best artifact. It never hard-blocks the deliverable: an
analyzer outage, a worker error, no flagged shots, or a failed re-render all
degrade to *accept the current artifact* and emit an empty/OK critique. That degrade
is HONEST, not silent: whenever the footage was not actually reviewed — a critic batch
that returned no verdict, OR an analyzer that never returned a reading — the retained
record carries ``critique_unavailable`` + ``unavailable_reason``, the stage emits a note
(⇒ a ``degraded`` run) AND a coded ``footage_critique_unavailable`` degradation, so an
empty ``flagged_shots`` from a critic that never ran is distinguishable from one that ran
and found nothing — both to a human and to a query. The code is not decoration: until
2026-09-15 this path raised prose only, the DAG executor stamped it
``unclassified_note`` with a NULL error, and 56 of 115 completed runs over 14 days
(48.7%, 55 of them scoring ZERO shots) shipped an unreviewed picture that nothing could
count.

Upstream failures are classified by the ONE policy in ``stages/upstream.py``
(``guarded``/``classify``), never by a local ``except httpx.HTTPStatusError``: a
transient upstream spends the configured ``llm_retries`` before the critique is called
unavailable, a permanent one fails fast with the real upstream sentence.

Structurally a critique twin of ``shared/footage_direction.py``: a subclass sets
``name`` + ``worker_name`` and supplies the data sources (the starting compiled
artifact, the L1 footage-direction plans, the per-shot timeline windows) plus the
re-render closure via the abstract hooks; the analyzer submit/poll, the whole-MP4
time→shot mapping, the WM evaluator call, plan validation and the bounded loop all
live here.

Authoritative-out invariant (mirrors footage_direction / sound_director): the
model is advisory. Its per-shot verdict is validated — unknown shot ids dropped,
scores clamped, strings bounded — and the plan nudges it drives are restricted to
INTENT-only fields (zoom / reframe / transition / callout). COORDINATES ARE NOT
THE MODEL'S TO SET: the compiler resolves every bbox from the captured focus
track, never from this critique.

Time→shot precision (read this before trusting a per-shot number): the analyzer
scores the WHOLE compiled MP4 on one time axis; there is no per-shot concept on
its side. We reconstruct each shot's ``[t0, t1)`` window from the ordered line
spine + real narration durations (cumulative running sum) and bucket the analyzer's
time-indexed samples into it. Spliced intro/outro title cards, a cold-open reframe
and disclaimer cards shift absolute time, so the mapping is approximate — good
enough for an advisory soft critic, never a hard gate.

LLM calls route through ``worker_management`` (the ``WMClient`` gateway) — never
inference-router / a provider directly; the model behind the worker is config.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import batched
from math import ceil
from typing import TYPE_CHECKING, ClassVar, TypedDict, cast

import httpx
from pydantic import Field
from w2v_common.stages import (
    StageRetriesExhaustedError,
    StageTimeoutError,
    stage_call,
)

from src.clients.schemas.video_analyzer import (
    VideoAnalyzerQualityMetrics,
)
from src.clients.video_analyzer_client import VideoAnalyzerClient
from src.clients.video_compile import VideoCompileClient
from src.services.llm import WMClient, WMResponse
from src.services.stages.base import (
    ArtifactUrl,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    StagePreconditionError,
    StageSkippedError,
)
from src.services.stages.llm_budget import (
    attempt_plan,
    downshift_reasoning,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
)
from src.services.stages.shared.cold_open import _motion_xy
from src.services.stages.shared.degradation import (
    FOOTAGE_BELOW_BAR,
    FOOTAGE_CRITIQUE_UNAVAILABLE,
    Degradation,
    degradation,
)
from src.services.stages.shared.loop_engine import Critique, run_evaluator_optimizer
from src.services.stages.upstream import (
    PermanentUpstreamError,
    TransientUpstreamError,
    guarded,
)
from src.utils import describe_exception

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

log = logging.getLogger(__name__)

# Per-shot scores live on a 1-5 scale, exactly like the retention critic. A shot
# scoring below ``_THRESHOLD`` is "bad" and gets nudged + re-rendered; when the
# worst shot clears the bar the loop converges and keeps the current artifact.
_SCORE_MIN = 1.0
_SCORE_MAX = 5.0
_THRESHOLD = 4.0
_EPSILON = 0.25
_MAX_PASSES = 2
# Same idiom as ``demo_video.final_critique._FAILED_EVAL_SCORE``: an analyzer that
# never returned a reading measured NOTHING, and must never read as a pass.
# ``score=_THRESHOLD`` satisfies ``score >= threshold`` in ``loop_engine``'s
# default gate — structurally identical to "the critic reviewed the footage and
# found nothing wrong" — so a 4xx from the analyzer's SSRF allow-list converged
# the loop as cleanly as a genuinely good shot. Below ``_SCORE_MIN`` so it can
# never be confused with a real (if poor) 1-5 verdict.
_FAILED_EVAL_SCORE = _SCORE_MIN - 1.0


# String hardening for the model's free-text verdict fields.
_MAX_ISSUE_CHARS = 300
_MAX_FIX_CHARS = 300
# Plan enum members the nudge may set (must match footage_direction's coercion).
_ZOOM_ON = "auto"
_REFRAME_ON = "follow-cursor"
_CUT = "cut"

# video_analyzer analysis config for a whole-video critique read: quality metrics
# + motion timeline + OCR + frame descriptions, no walkthrough/inventory. Bounds
# the frame sample so the vision pass stays affordable on an opt-in ≤2-pass loop.
_ANALYSIS_MAX_FRAMES = 48
# 2026-08-21: 180.0 -> 420.0. 2026-09-01: 420.0 -> 600.0.
#
# 180 was the OUTLIER for the same analyzer and the same job shape. Measured on run
# 01a021c6-01ba-70dc-a9cb-0aa194de7eb1 (2026-08-21): the poll timed out, the stage ran
# 3m09s, scored ZERO shots and still reported ``completed`` — a vacuous critic. 420
# fixed that case and still truncates a real tail: measured 2026-09-01 over 14 days,
# 536 analyzer jobs, p90 227s, max 1384s, and 27 of the 536 (5.0%) ran longer than
# 420s. 600 recovers the 420-600s band of that tail — the part this stage can afford.
#
# ⛔ 600 IS A CEILING, NOT A PREFERENCE, AND TWO INDEPENDENT BOUNDS SET IT:
#
#   1. TRANSPORT. ``BaseServiceClient`` clamps every poll deadline in
#      ``_clamp_timeout`` (w2v_common/clients/base.py) against ``_MAX_TIMEOUT`` =
#      600.0 and RAISES ``ValueError`` above it — it does not silently cap. So any
#      value over 600 does not merely fail to help, it makes the poll blow up before
#      the first GET. (Only ``CLIENT_MAX_TIMEOUT_S`` in the environment moves that
#      ceiling, and it is set nowhere in this repo or in gitops.)
#   2. THIS STAGE'S OWN ATTEMPT CLOCK, which is NOT the StageSpec 2500. The poll runs
#      inside ``critic_fn``, which ``run_evaluator_optimizer`` wraps in a
#      ``stage_call`` of ``llm_budget_seconds()`` = 950s PER ATTEMPT; the 2500 is the
#      ≤2-pass + re-render envelope around all of that. This value is the WORST CASE
#      the batch waves must survive: ``_batch_phase_seconds`` sizes the slice from the
#      analyzer's MEASURED elapsed, so the poll bound only shrinks the waves when the
#      analyzer actually spends it. Raising the bound therefore lowers the worst-case
#      slice, and past ~693s that worst case no longer funds one wave's full retry
#      ladder at ≥120s per attempt — the critic waves refuse to start
#      (``_CRITIC_FLOOR_SECONDS``) and the stage degrades to exactly the 0-shot
#      ``completed`` critique the 2026-08-21 fix existed to prevent.
#
# ⚠️ THIS STAGE AND ``demo_video/final_critique`` DO NOT DERIVE FROM EACH OTHER, even
# though they poll the SAME client for the SAME whole-video read: final_critique does
# its LLM work AFTER the poll and owns its whole StageSpec clock, whereas here the
# poll is one of two awaits inside a 950s LLM attempt. Their SIZING is independent;
# they simply both land on bound (1), the transport ceiling, which is the smaller
# bound for both. Treating one as a derivation of the other is the category error that
# produced an UNSATISFIABLE guard in ``test_footage_critique.py`` (it asserted
# ``polled >= 1100`` against a sibling that was itself above the clamp, AND
# ``polled <= 600``). The surviving sibling relation is a FLOOR only: this stage must
# not be handed LESS time than final_critique for the identical read.
#
# At 600 the by-construction identity
# ``analyzer_elapsed + batch_phase + _VALIDATION_HEADROOM_S == outer`` holds for every
# elapsed, and the WORST case (600 + 335 + 15 == 950) still yields 166s attempts from
# ``attempt_plan(335, 1)`` — above ``_CRITIC_FLOOR_SECONDS``.
# Asserted by ``test_critic_phase_still_funds_the_batch_waves`` and
# ``test_analyzer_poll_bound_is_not_below_the_sibling_stages``; the transport bound is
# asserted for EVERY stage, this one included, by
# ``tests/services/stages/test_poll_timeouts_within_client_ceiling.py``.
_ANALYSIS_POLL_TIMEOUT_S = 600.0

# Reserved out of the outer per-attempt budget for the RETURN PATH, so the batch
# phase cannot expand into the moment the pass needs to hand its verdicts back.
# What has to fit in it is pure CPU over at most ~25 records — ``_validate``, the
# spine sort, ``_republish``, building the ``Critique`` — plus the small skew
# between the outer ``stage_call`` starting its clock and ``critic_fn`` capturing
# ``entry``. Milliseconds of real work, rounded up hard: it is ~1.6% of a 950s
# attempt, so it cannot starve a wave, and being generous here is strictly cheaper
# than an outer cancel (which is what used to DISCARD the scored verdicts).
# NOT a tuned fraction and not derived from anything that drifts.
_VALIDATION_HEADROOM_S = 15.0

# The smallest per-attempt budget worth starting a critic call with. A critic call
# with reasoning OFF was MEASURED at 34-67s (``llm_budget.downshift_reasoning``,
# #873/#883); this is the TOP of that window, so below it an attempt is not
# reliably satisfiable and the whole ladder — including ``downshift_reasoning``'s
# reasoning-off retry, which is the fast one — is doomed before the first byte.
#
# Production ran the doomed case for 14 days: a 23.2s per-attempt budget, 2.9x under
# this floor, spent on two attempts that could not finish, after which the stage
# completed with ZERO shots scored and `critique_unavailable` reported as an
# unclassified prose note. Refusing to start and saying WHY is strictly better than
# burning the calls: same (absent) verdict, no WM spend, and a reason a reader can
# act on. Deliberately a REFUSAL and not a budget increase — the phase arithmetic
# above is what fixes the apportionment; this floor exists so that when the budget
# is genuinely too small (a real 600s analyzer read on a many-wave shot list) the
# stage says so instead of pretending to review the footage.
_CRITIC_FLOOR_SECONDS = 67.0

# Max shots scored per WM critic call. The single small vLLM times out when one
# call must emit an unbounded ``shots`` array (8 fields x N directed shots), which
# silently degrades the SOFT critique to an empty, no-flag verdict (no re-render
# pass fires). Splitting the shots into bounded batches keeps each call's output
# schema small so the worker returns real scores. Sized like the course_outline
# split — a handful of items per call, several small calls instead of one big one.
_CRITIC_BATCH_SIZE = 5

# How many critic batches may be IN FLIGHT at once. Same reasoning, same shape and
# same derivation as ``segment_select._SELECT_BATCH_CONCURRENCY`` — read the long
# comment there; only the arithmetic differs. The batches bound each call's
# prompt/answer; they were still awaited SERIALLY inside one evaluator-optimizer
# pass, so a demo with N directed shots cost ceil(N/5) sequential model calls.
# 5 in flight covers a 25-shot demo in one wave, which is the top of what
# ``footage_direction`` plans today.
#
# NO speedup figure is claimed: the per-request token rate at k=5 on the one small
# vLLM is unmeasured. What is claimed is bounded fan-out — never more than 5
# concurrent calls from this stage, whatever the shot count.
_CRITIC_BATCH_CONCURRENCY = 5

# Explicit per-call OUTPUT budget handed to WM. This is NOT an optimisation — it is
# the fix for a measured prod outage (2026-08-15) in which every critic call was
# rejected 422 ("Model token limit (6000) exceeded before any response was
# generated").
#
# ⚠️ THIS COMMENT USED TO EXPLAIN THAT WRONG, AND THE WRONG EXPLANATION SET THE
# NUMBER. It said WM "RESERVES 6000 output tokens against the small self-hosted
# model's context window ... the RESERVATION was the problem, not the payload", and
# concluded the budget should sit "far under WM's 6000 default" — which is how this
# constant ended up at 1800. There is no reservation: the model runs at
# ``--max-model-len=131072``. The 422 is raised post-generation by
# ``pydantic_ai/_agent_graph.py:1100-1113`` when ``finish_reason == 'length'`` and
# every returned part is a ``ThinkingPart`` — the reasoning model spent the whole
# budget THINKING and never emitted the JSON. Measured live 2026-08-19, one prompt,
# only max_tokens varying: 500 -> 422, 1760 -> 422, 6000 -> no 422.
#
# So the budget is a FLOOR, and a value below WM's own default is strictly worse
# than passing nothing. The per-shot arithmetic below is still the right way to size
# the ANSWER; ``output_token_budget`` adds the reasoning headroom on top.
# See ``stages/llm_budget.py``.
#
# Sizing — READ THIS BEFORE TUNING. ``_MAX_ISSUE_CHARS`` / ``_MAX_FIX_CHARS`` are applied
# in ``_validate`` AFTER the response is parsed, so on their own they constrain what we
# KEEP, not what the model may EMIT. The bound reaches the model through the two fields'
# ``description`` in ``_OUTPUT_SCHEMA`` — deliberately NOT a ``maxLength`` keyword, which
# vLLM guided decoding cannot enforce and which WM strips before building the grammar
# while still validating against the unstripped schema at ``retries=0`` (one over-long
# string would then permanently kill the stage; see tests/test_output_schema_bounds.py).
# So the arithmetic below is a budget for a WELL-BEHAVED reply plus headroom, and the
# honest-degrade markers are what catch a badly-behaved one: a truncated generation is
# not a smaller answer, it is invalid JSON, which used to surface as a clean
# "critic passed it" verdict.
#
# Worst-case output per shot verdict, counted against those bounds:
#   "issue"          300 chars, at a CONSERVATIVE 2 chars/token for JSON-escaped,
#                    punctuation-heavy text  → 150 tok, + ~4 tok for the key/quotes
#   "suggested_fix"  300 chars, same          → 150 tok, + ~4 tok
#   "shot_id"        key + a short id                    → ~8 tok
#   5 numbers        score/eye_directed/boring_cut/app_demoed/framing, ~6 tok each → 30
#   object braces, commas, whitespace                    → ~6 tok
#   ------------------------------------------------------------------------------
#   total                                                ≈ 352 tok/shot
# Rounded to 360/shot, which also amortises the ~10-token ``{"shots": [ … ]}`` envelope
# across the batch. DERIVED FROM _CRITIC_BATCH_SIZE on purpose so the two CANNOT drift:
# raising the batch size raises the budget with it.
_CRITIC_MAX_TOKENS_PER_SHOT = 360
_CRITIC_ANSWER_TOKENS = _CRITIC_BATCH_SIZE * _CRITIC_MAX_TOKENS_PER_SHOT  # 5 * 360 = 1800
_CRITIC_MAX_TOKENS = output_token_budget(answer_tokens=_CRITIC_ANSWER_TOKENS)

_ANALYSIS_CONFIG: dict[str, object] = {
    "generate_walkthrough": False,
    "generate_ui_inventory": False,
    "generate_frame_index": False,
    "compute_quality_metrics": True,
    "extract_ocr": True,
    "extract_frame_descriptions": True,
    "max_frames": _ANALYSIS_MAX_FRAMES,
}

_SYSTEM_PROMPT = (
    "You are a footage-quality critic for a short screen-recording demo video. You "
    "receive, per shot, MEASURED numbers taken off the ACTUAL compiled video — a "
    "motion score over the shot's time window, whole-video sharpness/brightness, "
    "any on-screen text (OCR), and the shot's current framing plan (zoom, reframe, "
    "callout). Judge EACH shot on four dimensions, scoring every one 1-5: "
    "eye_directed (is the viewer's eye guided to what matters), boring_cut (does the "
    "shot hold attention or drag), app_demoed (is the product actually shown doing "
    "something), framing (is the important region well framed). Reason ONLY from the "
    "measured numbers, not vibes; prefer restraint — most shots are fine, flag only "
    "genuinely weak ones. A shot MAY carry `idle_trim`: frozen stretches of its source "
    "recording (no pixel change) were ALREADY cut to a short hold before compile — "
    "`idle_fraction_after` is the stillness that remains. Do not mark boring_cut down "
    "for idle that was removed; judge what is left. "
    "Coordinates and geometry are NOT yours: NEVER emit bboxes, "
    "pixel positions, or crops — the compiler owns all geometry from the captured "
    "track. Return one entry per shot_id you were given."
)

# Object-rooted structured-output schema (WM convention): a required ``shots``
# array, each entry a per-shot verdict. WM returns ``response.output`` parsed to it.
#
# The two free-text fields state their length bound in ``description``, from the SAME
# constants ``_validate`` clamps to, so what the model is ASKED for and what we KEEP
# cannot drift. Keep the two in lockstep — an over-long emit against a fixed
# ``max_tokens`` truncates mid-JSON, and a truncated body is a decode error, not a
# shorter verdict.
_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["shots"],
    "properties": {
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["shot_id", "score"],
                "properties": {
                    "shot_id": {"type": "string"},
                    # The length bound lives in `description`, NOT in a `maxLength`
                    # keyword. vLLM guided decoding constrains grammar and type only, so
                    # `maxLength` cannot stop a long string being generated — and WM
                    # STRIPS the keyword before building the grammar while still
                    # validating the reply against the unstripped schema, with
                    # `retries=0`. A single over-long string would then permanently kill
                    # the stage. The model reads the description; `_validate` clamps on
                    # consumption. Both derive from the same constants, which is what
                    # makes `_CRITIC_MAX_TOKENS` a budget rather than a guess.
                    "issue": {
                        "type": "string",
                        "description": f"At most {_MAX_ISSUE_CHARS} characters.",
                    },
                    "suggested_fix": {
                        "type": "string",
                        "description": f"At most {_MAX_FIX_CHARS} characters.",
                    },
                    "score": {"type": "number"},
                    "eye_directed": {"type": "number"},
                    "boring_cut": {"type": "number"},
                    "app_demoed": {"type": "number"},
                    "framing": {"type": "number"},
                },
            },
        }
    },
}


class ShotWindow(TypedDict):
    """One shot's slice of the whole-video time axis: ``[t0, t1)`` in seconds."""

    shot_id: str
    t0: float
    t1: float


class _MeasuredShot(TypedDict, total=False):
    """The measured numbers handed to the critic for one shot (prompt row shape)."""

    shot_id: str
    t0: float
    t1: float
    motion_mean: float
    motion_samples: int
    ocr_text: str
    plan: Mapping[str, object]
    overall_sharpness: float
    overall_brightness: float
    overall_motion: float
    overall_score: float
    idle_trim: Mapping[str, object]


@dataclass(frozen=True)
class _FootageDraft:
    """One point in the loop: a compiled artifact + the plan that produced it.

    ``plans`` are the footage-direction plan rows (INTENT only) the compiler
    consumed for this artifact. The loop keeps the highest-scoring draft; carrying
    the plan on the draft is what lets the optimizer nudge from the retained best's
    plan rather than always from the L1 baseline.
    """

    artifact_url: str
    duration_seconds: float
    plans: tuple[Mapping[str, object], ...]


class ShotCritique(StageOutput):
    """One shot's validated per-dimension verdict (authoritative-out)."""

    shot_id: str
    score: float
    issue: str = ""
    suggested_fix: str = ""
    eye_directed: float | None = None
    boring_cut: float | None = None
    app_demoed: float | None = None
    framing: float | None = None


class TokenUsage(StageOutput):
    """Aggregate WM token spend across every critic call in the loop."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0


class FootageCritiqueRecord(StageOutput):
    """The structured outcome of the loop for the retained (best) artifact."""

    score: float
    passes: int
    converged: bool
    reason: str
    flagged_shots: list[str]
    shots: list[ShotCritique]
    # Honest-degrade markers. A critic batch that never returned a verdict used to be
    # indistinguishable from a critic that looked and found nothing wrong — the run
    # went green with `flagged_shots: []` either way (the "skip-and-continue ⇒ silent
    # GREEN" defect class). These two fields let a reader of ``stage_outputs`` tell
    # "the critic passed this footage" from "the critic never actually ran on it".
    # The stage still NEVER fails the run (orchestrator #812): the critic is advisory.
    critique_unavailable: bool = False
    unavailable_reason: str = ""
    # What the optimizer's re-render actually did, for the retained loop. ``""`` =
    # NOT RECORDED (the accept-current fallbacks, and every row persisted before this
    # field existed) — never read as "no re-render". Without it a failed re-render and
    # a re-render that did not help are indistinguishable: the optimizer hands back the
    # SAME draft on failure, so delta is 0 and the loop stops as ``plateau`` either way.
    rerender_outcome: str = ""


#: ``FootageCritiqueRecord.rerender_outcome`` values (see the field comment).
RERENDER_NOT_ATTEMPTED = "not_attempted"
RERENDER_FAILED = "failed"
RERENDER_RENDERED = "rendered"
#: Skipped on purpose: ``boring_cut`` was the ONLY dimension below the bar, and its one
#: lever (``transition_signal = cut``) cannot change what is on screen in the shots, so
#: a second paid render could not move a min-across-shots score. See ``_only_cut_is_low``.
RERENDER_SKIPPED_CUT_ONLY = "skipped_cut_only"


class FootageCritiqueInput(StageInput):
    """No typed inputs — everything is read from ``ctx`` (prior output + config)."""


class FootageCritiqueOutput(StageOutputWithNotes):
    """Best artifact seen + the structured critique + token spend.

    ``artifact_url`` is ALWAYS the best artifact the loop saw (the current compile
    artifact when nothing improved / the loop soft-degraded). ``render_manifest``
    depends on this stage and reads THIS ``artifact_url`` (falling back to
    ``demo_video_compile`` when the stage skipped-clean), so the accepted-best
    re-render reaches the deliverable/provenance ledger. The blueprint's
    ``terminal_artifact_stage`` stays ``demo_video_compile`` (the reorder cut +
    finalizer existence probe); ``notes`` (from ``StageOutputWithNotes``) surface a
    soft degradation as a ``degraded`` run without failing it.
    """

    # Declared output contract — enforced centrally in
    # `runs/dag_executor.py::_enforce_produces`. `produces_required` because this
    # stage is promo_v1's `terminal_artifact_stage`: `run_finalizer` probes THIS
    # `artifact_url` for the deliverable, and when it is absent the finalizer
    # silently disables the existence probe for the whole run. The invariant in the
    # docstring above holds on every path — the not-requested and no-directed-footage
    # branches both accept-current and return the UPSTREAM url — so empty means the
    # pass-through broke, never "nothing to critique".
    # `critique` / `token_usage` / `footage_plans` are all legitimately empty (a
    # skipped-clean or no-nudge loop), so only the artifact is declared.
    produces: ClassVar[tuple[str, ...]] = ("artifact_url",)
    produces_required: ClassVar[bool] = True

    artifact_url: ArtifactUrl
    duration_seconds: float
    critique: FootageCritiqueRecord
    token_usage: TokenUsage
    # The nudged best draft's footage-direction plan ROWS (INTENT only — SAME shape
    # as ``demo_video_footage_direction.plans``), persisted so a later re-compile
    # (encode_master) rebuilds the footage from the critique's improved plans instead
    # of the L1 baseline. EMPTY when the loop made no nudge (the pass-through /
    # accept-current path) — an empty list means "no persisted override", and the
    # compiler falls back to the base footage-direction plans (byte-identical).
    footage_plans: list[dict[str, object]] = Field(default_factory=list)
    # Per-shot idle measurement from the source recording (``idle_trim`` on the record
    # stage's clips: frozen seconds/fraction before vs after the pre-compile trim),
    # keyed by shot id — the same numbers the critic was shown. Empty when the
    # subclass supplies none or the record stage did not run the idle pass.
    idle_trim_by_shot: dict[str, dict[str, object]] = Field(default_factory=dict)


def _clamp_score(value: object) -> float:
    """Coerce a model score into the ``[1, 5]`` band; junk ⇒ the low end."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return _SCORE_MIN
    try:
        score = float(value)
    except TypeError, ValueError:
        return _SCORE_MIN
    if score != score:  # NaN
        return _SCORE_MIN
    return max(_SCORE_MIN, min(_SCORE_MAX, score))


def _opt_score(value: object) -> float | None:
    """Optional sub-dimension score: ``None`` when the model omitted it."""
    if value is None:
        return None
    return _clamp_score(value)


async def _score_critic_batch(
    *,
    wm: WMClient,
    worker_name: str,
    stage_name: str,
    message: str,
    batch_number: int,
    limit: asyncio.Semaphore,
    deadline: float,
    share_seconds: float,
    retries: int,
) -> tuple[list[object], str | None, Mapping[str, object] | None]:
    """ONE critic call, at module level so ``critic_fn`` stays under ruff's C901.

    A batch failure returns ``(no shots, reason, no usage)`` and NEVER raises —
    that is what the stage's SOFT contract rests on. Anything outside those
    branches (a bug) still propagates; the caller re-raises it in batch order
    after the gather.

    ⚠️ The call is routed through ``upstream.guarded`` + ``stage_call`` rather than
    catching ``httpx.HTTPStatusError`` here and deciding alone. That local decision
    was the defect: it spent ZERO of the configured ``llm_retries`` on a genuinely
    transient upstream — a 429/408/409, a 5xx blip, or the 422 "Exceeded maximum
    output retries" bad DRAW — and degraded the critique to "unavailable" on the
    first failure. ``guarded`` applies the ONE policy in ``stages/upstream.py``:

    * ``PermanentUpstreamError`` (402/422-contract/404/...) — no retry can help.
      Reported as unavailable IMMEDIATELY, logged at ERROR with the real sentence.
    * ``TransientUpstreamError`` / transport — retried by ``stage_call`` up to
      ``retries`` times before it is reported as unavailable.

    Either way the reason string is non-empty, so the critique is marked
    ``critique_unavailable`` and can never read as "the critic reviewed this and
    found no issues".

    Usage metrics are RETURNED rather than accumulated here, so the summation
    happens in batch order in the caller and cannot depend on which call landed
    first.

    ⚠️ DEADLINE — the reason this takes ``deadline``/``share_seconds`` instead of a
    ``timeout_seconds``. This call is nested INSIDE the evaluator-optimizer's own
    ``stage_call``; handing it the caller's full per-attempt budget with the caller's
    full ``retries`` authorised ``attempts x budget`` of inner ladder inside ONE outer
    slice of the same size. The outer clock therefore always fired first: the inner
    second attempt was dead code, and the cancellation discarded every batch already
    scored so the retried outer attempt re-paid for all of them. The slice is now
    sized by :func:`attempt_plan` to fit ``attempts x per_attempt + worst-case
    backoff <= budget``, with a hard ``asyncio.timeout(budget)`` around the whole
    ladder that holds however the ladder is sized.
    """
    # The batch is ADMITTED before its budget is sized: the size depends on what is
    # left of the shared deadline when this batch actually starts, and a second-wave
    # batch sized against the FIRST wave's remaining would overrun the phase.
    #
    # ⚠️ The semaphore now spans the whole ladder. It used to be acquired inside the
    # retried factory so a backoff sleep did not hold a slot hostage from a sibling —
    # that is deliberately traded away, because a slot released mid-ladder makes the
    # batch's wall clock unaccountable, and the per-wave share below is only true while
    # a slot is held for at most ``share_seconds`` (which the timeout guarantees).
    async with limit:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0.0:
            # STOP cleanly. Starting a batch that provably cannot finish only trades
            # the outer cancel for nothing — and a non-empty reason keeps the critique
            # marked unavailable rather than reading as "reviewed, found nothing".
            log.warning(
                "%s: critic deadline exhausted before batch %d, not starting it",
                stage_name,
                batch_number,
            )
            return [], f"batch {batch_number}: critic deadline exhausted before the call", None
        budget = min(share_seconds, remaining)
        # Retries are paid out of THIS batch's own slice, never out of a sibling's or
        # out of the outer loop's: ``attempt_plan`` shrinks the ladder (to a single
        # attempt if need be) so attempts + backoff fit inside ``budget``.
        per_attempt, batch_retries = attempt_plan(budget, retries)
        if per_attempt < _CRITIC_FLOOR_SECONDS:
            # REFUSE. A critic call with reasoning off was MEASURED at 34-67s, so an
            # attempt sized below that window cannot finish — and neither can the
            # retry, whatever ``downshift_reasoning`` does to it. Production spent 14
            # days proving it: 23.2s per attempt, two attempts, zero shots scored, and
            # the run reported ``completed``. Burning the calls buys nothing the
            # refusal does not, so say WHY instead: a non-empty reason keeps the
            # critique marked unavailable and names the arithmetic, which is what makes
            # an empty ``flagged_shots`` here readable as "not reviewed" rather than
            # "reviewed, found nothing".
            log.warning(
                "%s: critic batch %d budget %.1fs allows only a %.1fs attempt, below the "
                "measured %.0fs critic floor — refusing to start it",
                stage_name,
                batch_number,
                budget,
                per_attempt,
                _CRITIC_FLOOR_SECONDS,
            )
            return (
                [],
                (
                    f"batch {batch_number}: per-attempt budget {per_attempt:.1f}s is below "
                    f"the measured {_CRITIC_FLOOR_SECONDS:.0f}s critic floor "
                    f"(slice {budget:.1f}s) — not started"
                ),
                None,
            )

        async def _once(enable_reasoning: bool | None) -> WMResponse:
            return await guarded(
                lambda: wm.message(
                    enable_reasoning=enable_reasoning,
                    worker_name=worker_name,
                    message=message,
                    system_prompt=_SYSTEM_PROMPT,
                    output_schema=_OUTPUT_SCHEMA,
                    # See _CRITIC_MAX_TOKENS: omitting this let WM reserve its
                    # 6000-token default, which a reasoning model can burn entirely
                    # on thinking before emitting any JSON.
                    max_tokens=_CRITIC_MAX_TOKENS,
                )
            )

        try:
            # The hard bound: one slow batch must not eat the phase and take every
            # other batch's verdicts down with it.
            async with asyncio.timeout(budget):
                resp = await stage_call(
                    downshift_reasoning(_once),
                    timeout_seconds=per_attempt,
                    retries=batch_retries,
                    retry_on=(TransientUpstreamError, httpx.TransportError),
                    stage_name=f"{stage_name}:critic_batch_{batch_number}",
                )
        except PermanentUpstreamError as exc:
            # A PERMANENT upstream fault (bad worker name, a token budget the model
            # cannot honour, a rejected schema, no LLM provider configured) — NOT an
            # outage, and no retry can fix it. Laundering it through the outage branch
            # is how a permanently-broken critic reported "all good" on every run.
            # Logged at ERROR with the real upstream sentence; still NOT raised — the
            # critic is advisory and must never kill a finished run (orchestrator #812).
            detail = str(exc)[:300]
            log.error(
                "%s: critic worker PERMANENT upstream fault on batch %d — %s "
                "(critique will be reported as unavailable, run not failed)",
                stage_name,
                batch_number,
                detail,
            )
            return [], f"batch {batch_number}: permanent upstream fault — {detail}", None
        except StageRetriesExhaustedError as exc:
            # Transient upstream, retried to the end of the budget and still failing.
            detail = describe_exception(exc.last_error)[:300]
            log.warning(
                "%s: critic worker transient upstream failure on batch %d after %d attempt(s): %s",
                stage_name,
                batch_number,
                exc.attempts,
                detail,
            )
            return (
                [],
                f"batch {batch_number}: transient upstream failure after "
                f"{exc.attempts} attempt(s) — {detail}",
                None,
            )
        except StageTimeoutError as exc:
            log.warning(
                "%s: critic worker timed out on batch %d: %s", stage_name, batch_number, exc
            )
            return [], f"batch {batch_number}: StageTimeoutError — {str(exc)[:200]}", None
        except TimeoutError:
            # The per-batch CLAMP fired: the ladder was still running when this
            # batch's slice ran out. Distinct from ``StageTimeoutError`` (one
            # ATTEMPT's clock) on purpose — this one says the batch was cut off to
            # protect its siblings' share, and it is a non-empty reason, so the
            # critique is marked unavailable rather than reading as a clean pass.
            log.warning(
                "%s: critic batch %d exceeded its %.1fs share of the critic deadline",
                stage_name,
                batch_number,
                budget,
            )
            return [], f"batch {batch_number}: exceeded its {budget:.1f}s share", None
        except (httpx.HTTPError, RuntimeError) as exc:
            # SOFT: a genuine WM worker OUTAGE (transport error) must not fail the
            # run. A bug in how we build the message (ValueError) or parse the
            # response propagates instead. ``TimeoutError`` is NOT in this tuple any
            # more — the dedicated branch above owns it, because inside the clamp a
            # bare TimeoutError is the clamp firing far more often than it is a raw
            # socket timeout (httpx's own is a ``TransportError`` ⇒ ``HTTPError``).
            log.warning(
                "%s: critic worker error (%s) on batch %d: %s",
                stage_name,
                type(exc).__name__,
                batch_number,
                exc,
            )
            return [], f"batch {batch_number}: {type(exc).__name__}", None
        batch_raw = resp.output.get("shots") if isinstance(resp.output, Mapping) else None
        usage = resp.usage_metrics if isinstance(resp.usage_metrics, Mapping) else None
        return (list(batch_raw) if isinstance(batch_raw, list) else []), None, usage


def _usage_add(acc: dict[str, int], usage: Mapping[str, object] | None) -> None:
    """Best-effort accumulate WM ``usage_metrics`` into ``acc`` (never raises).

    WM's ``usage_metrics`` is an untyped JSON blob (dict | None); provider payloads
    vary, so we sum whichever of the common key spellings are present and count the
    call regardless, defaulting an unknown shape to a zero-token but counted call.
    """
    acc["calls"] = acc.get("calls", 0) + 1
    if not isinstance(usage, Mapping):
        return
    for field, keys in (
        ("prompt_tokens", ("prompt_tokens", "input_tokens")),
        ("completion_tokens", ("completion_tokens", "output_tokens")),
        ("total_tokens", ("total_tokens", "tokens")),
    ):
        for key in keys:
            raw = usage.get(key)
            if isinstance(raw, bool):  # bool is an int subclass — not a token count
                continue
            if isinstance(raw, (int, float)):
                acc[field] = acc.get(field, 0) + int(raw)
                break


def _bucket_motion(
    motion_timeline: Sequence[object], windows: Sequence[ShotWindow]
) -> dict[str, list[float]]:
    """Bucket motion samples into each shot's ``[t0, t1)`` window.

    A sample lands in the first window whose half-open interval contains its ``t``.
    Each sample is normalised through :func:`_motion_xy`, so BOTH the typed
    ``VideoAnalyzerMotionSample`` (the client-schema path) AND a raw ``Mapping`` row
    (an untyped analyzer JSON blob) are accepted; a malformed/non-finite entry is
    skipped rather than raising. See the module docstring on absolute-time imprecision.
    """
    buckets: dict[str, list[float]] = {w["shot_id"]: [] for w in windows}
    for sample in motion_timeline:
        xy = _motion_xy(sample)
        if xy is None:
            continue
        t, motion = xy
        for w in windows:
            if w["t0"] <= t < w["t1"]:
                buckets[w["shot_id"]].append(motion)
                break
    return buckets


def _bucket_ocr(frames: object, windows: Sequence[ShotWindow]) -> dict[str, list[str]]:
    """Bucket per-frame OCR text into each shot's window (best-effort, never raises).

    The analyzer's frame records are an untyped list on its side; a frame contributes
    its OCR/description text to the window its timestamp falls in. Missing timestamps
    or a non-list ⇒ empty buckets (the critic simply reasons without OCR for those).
    """
    buckets: dict[str, list[str]] = {w["shot_id"]: [] for w in windows}
    if not isinstance(frames, list):
        return buckets
    for raw in frames:
        if not isinstance(raw, Mapping):
            continue
        item = cast("Mapping[str, object]", raw)
        t = item.get("timestamp")
        if t is None:
            t = item.get("t")
        if isinstance(t, bool) or not isinstance(t, (int, float)):
            continue
        text = item.get("ocr_text") or item.get("text") or item.get("description")
        if not isinstance(text, str) or not text.strip():
            continue
        t_f = float(t)
        for w in windows:
            if w["t0"] <= t_f < w["t1"]:
                buckets[w["shot_id"]].append(text.strip())
                break
    return buckets


class FootageCritiqueStage(StageHandler[FootageCritiqueInput, FootageCritiqueOutput]):
    """Base footage-quality critique loop. Subclasses set ``name`` + ``worker_name``
    and implement the four data/render hooks. WM auto-creates ``worker_name`` on the
    first call (provider/model live on the WM worker config, not here)."""

    input_model = FootageCritiqueInput
    output_model = FootageCritiqueOutput
    worker_name: str = "footage-critic"
    # When True (demo/showcase), an absent `direction.footage.critique` skips the
    # stage clean (StageSkippedError). When False, the stage instead passes the
    # current terminal artifact through unchanged (used by blueprints that make
    # this stage the terminal_artifact_stage and have no render_manifest reader —
    # e.g. promo). Byte-identical delivered artifact in both cases.
    skip_when_unrequested: bool = True

    def __init__(
        self,
        client: WMClient | None = None,
        analyzer: VideoAnalyzerClient | None = None,
        compile_client: VideoCompileClient | None = None,
    ) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())
        self._analyzer = analyzer or VideoAnalyzerClient()
        self._compile_client = compile_client or VideoCompileClient()

    def _batch_phase_seconds(self, analyzer_elapsed: float) -> float:
        """The slice of ONE outer critic attempt the batch waves may spend, given how
        long the analyzer read ACTUALLY took.

        The evaluator-optimizer wraps ``critic_fn`` in a ``stage_call`` of
        ``self._llm_timeout`` per attempt, and that attempt has to cover TWO bounded
        awaits, not one: the analyzer read — hard-bounded by
        ``_ANALYSIS_POLL_TIMEOUT_S`` — and the critic batch waves. So the phase is
        what is LEFT of the attempt once the analyzer has been paid for, minus an
        explicit reserve for the return path::

            batch_phase = outer - analyzer_elapsed - _VALIDATION_HEADROOM_S

        which makes ``analyzer_elapsed + batch_phase + headroom == outer`` an identity
        for EVERY value of ``analyzer_elapsed``, not just the worst case. The waves can
        therefore never run into the outer deadline, and the headroom is what lets the
        pass RETURN instead of being cancelled on the way out — the cancel is what used
        to discard every verdict it had scored.

        ⚠️ ``analyzer_elapsed`` is MEASURED, and it is measured for a reason. This used
        to be a constant ``outer - _ANALYSIS_POLL_TIMEOUT_S`` (950 - 600 = 350) captured
        BEFORE the analyzer poll, and the docstring here claimed that "whatever the
        analyzer did NOT spend is left over as headroom". It claimed the OPPOSITE of what
        the code did: a deadline captured pre-poll means the analyzer's real elapsed time
        was deducted from the BATCH PHASE as well as from its own 600s reserve —
        double-charged. Production back-solves it exactly: a logged per-attempt budget of
        23.2s implies a wave share of 47.6s, i.e. the analyzer had eaten ~302 of the 350,
        and the stage's own elapsed was 349.79s — the whole phase. Reasoning-off critic
        calls measure 34-67s (``llm_budget.downshift_reasoning``), so 23.2s x 2 attempts
        was unsatisfiable BY CONSTRUCTION, and the stage completed having scored zero
        shots. That was 48.7% of completed runs over 14 days.

        Sizing from the real elapsed both removes the double charge and pays the common
        case what it is owed: an analyzer that answers in 30s now funds a ~905s batch
        phase instead of 320s. The worst case is unchanged in spirit — an analyzer that
        spends its entire 600s bound still leaves ``950 - 600 - headroom``.

        It is deliberately NOT a tuned fraction: the first two terms are the real outer
        bound and a real measurement, and the third is a fixed reserve for a pure-CPU
        return path (see :data:`_VALIDATION_HEADROOM_S`).
        """
        return max(0.0, self._llm_timeout - analyzer_elapsed - _VALIDATION_HEADROOM_S)

    # ── Subclass hooks ─────────────────────────────────────────────────────────
    def starting_artifact(self, ctx: RunContext) -> tuple[str, float]:
        """``(artifact_url, duration_seconds)`` of the already-compiled video to
        critique. Raise ``StagePreconditionError`` if the compile output is absent."""
        raise NotImplementedError

    def base_plans(self, ctx: RunContext) -> list[Mapping[str, object]]:
        """The L1 footage-direction plan rows (INTENT only), one per directed shot.
        Empty when directed footage produced no plan — the loop then accepts the
        current artifact unchanged."""
        raise NotImplementedError

    def shot_windows(self, ctx: RunContext) -> list[ShotWindow]:
        """Per-shot ``[t0, t1)`` windows over the whole-video time axis, in play
        order. Built from the ordered spine + real narration durations."""
        raise NotImplementedError

    def idle_trim_by_shot(self, ctx: RunContext) -> dict[str, dict[str, object]]:
        """Per-shot idle measurement of the shot's SOURCE recording, keyed by shot id.

        Optional hook: the default (no measurement) leaves every critic row
        byte-identical. Called once, guarded — an error here degrades to ``{}``.
        """
        return {}

    async def rerender(
        self, ctx: RunContext, plans: Sequence[Mapping[str, object]], pass_idx: int
    ) -> tuple[str, float]:
        """Re-render the video with ``plans`` applied to the directed footage,
        returning the new ``(artifact_url, duration_seconds)``. Reuses the canonical
        compile path — it must NOT duplicate compile logic.

        ``pass_idx`` is the 1-based critique re-render ordinal (the Nth actual
        re-render submit in this loop); the override builds a resume-stable compile
        discriminator (``critique:pass{pass_idx}``) from it, so a wholesale handler
        re-run after a resumable timeout reuses the SAME key for its Nth re-render."""
        raise NotImplementedError

    # ── Handler ────────────────────────────────────────────────────────────────
    async def handle(  # noqa: C901 — critic/optimizer closures capture the loop's shared soft-degrade state (usage, records_by_url, windows); splitting hurts clarity
        self, payload: FootageCritiqueInput, ctx: RunContext
    ) -> FootageCritiqueOutput:
        # Opt-in + skip-clean: the critique sub-flag lives UNDER the footage block
        # (`direction.footage.critique`). Absent/false ⇒ StageSkippedError ⇒ the
        # stage is `skipped` with empty output, so the terminal-artifact probe and
        # the render_manifest fallback both read compile's artifact unchanged
        # (byte-identical to a run that never had this stage).
        footage = (ctx.config_snapshot.get("direction") or {}).get("footage")
        critique_cfg = footage.get("critique") if isinstance(footage, Mapping) else None
        if not critique_cfg:
            if self.skip_when_unrequested:
                raise StageSkippedError(
                    f"{self.name}: direction.footage.critique not requested (opt-in stage)",
                    missing=["direction.footage.critique"],
                )
            # Pass-through: this blueprint makes the critique its terminal artifact
            # stage and has no render_manifest reader, so an absent critique block
            # must still deliver the current (promo_render) artifact unchanged.
            start_url, start_duration = self.starting_artifact(ctx)
            return self._accept_current(
                _FootageDraft(artifact_url=start_url, duration_seconds=start_duration, plans=()),
                {},
                reason="critique_not_requested",
                note="",
            )

        start_url, start_duration = self.starting_artifact(ctx)
        plans = self.base_plans(ctx)
        windows = self.shot_windows(ctx)
        try:
            idle_by_shot = self.idle_trim_by_shot(ctx)
        except Exception as exc:  # noqa: BLE001 — advisory context; never fail the critique
            log.warning("%s: idle_trim_by_shot failed: %s", self.name, exc)
            idle_by_shot = {}
        valid_ids = {str(p["line_id"]) for p in plans if p.get("line_id") is not None}

        usage: dict[str, int] = {}
        # Retained per-artifact verdicts — the loop's LoopResult exposes only the
        # retained draft + its rerun_stages, so we key the full per-shot records by
        # artifact_url here and look the best draft's records back up after the loop.
        records_by_url: dict[str, list[ShotCritique]] = {}
        # Per-artifact RAW verdict accumulator behind ``records_by_url``. Published
        # before the first await and extended as each batch lands (see the DURABILITY
        # comment in ``critic_fn``), so a retried critic attempt ACCUMULATES onto the
        # previous attempt's scored shots instead of discarding them and re-paying WM
        # for every batch it had already bought.
        raw_by_url: dict[str, list[object]] = {}
        # Per-artifact "the critic did not actually run (fully)" marker, same keying as
        # ``records_by_url``. Non-empty ⇒ at least one critic batch returned no verdict
        # for that draft, so its critique is INCOMPLETE and must not read as a clean pass.
        unavailable_by_url: dict[str, str] = {}
        # Artifacts for which NOTHING was ever scored — either the analyzer call
        # itself never returned a reading, or every WM critic batch failed with zero
        # survivors. See the short-circuit note in ``critic_fn``. Deliberately
        # narrower than ``unavailable_by_url``: a PARTIAL batch failure (some shots
        # scored, some not) is excluded, because a fresh pass-2 retry against the WM
        # worker is a real recovery path there, not a repeat of pass 1.
        critic_never_scored_urls: set[str] = set()
        # The verdict of a draft whose re-render the optimizer SKIPPED (cut-only, see
        # ``_only_cut_is_low``). The loop engine re-scores whatever the optimizer hands
        # back; a skipped re-render hands back the same artifact, so re-paying the
        # analyzer + every WM batch for it would buy the identical verdict.
        reuse_critique_by_url: dict[str, Critique] = {}

        initial = _FootageDraft(
            artifact_url=start_url,
            duration_seconds=start_duration,
            plans=tuple(plans),
        )

        # Nothing directed to critique (footage_direction produced no plan, or no
        # timeline windows resolved) ⇒ accept the current artifact, empty critique.
        if not valid_ids or not windows:
            return self._accept_current(
                initial,
                usage,
                reason="no_directed_footage",
                note="footage_critique: no directed-footage shots to critique",
            )

        async def _score_draft(draft: _FootageDraft) -> Critique:
            # DURABILITY (bug: every batch already scored was thrown away by an outer
            # cancel). Both per-draft containers are published into the handler's dicts
            # BEFORE the first await and then MUTATED IN PLACE, so the loop engine's
            # per-attempt ``stage_call`` timeout unwinding this coroutine still leaves
            # every verdict scored so far visible to ``handle``. ``setdefault`` (not
            # assignment) so a RETRIED attempt ACCUMULATES onto the previous attempt's
            # work — that retry used to re-pay WM for every batch it had already bought.
            records = records_by_url.setdefault(draft.artifact_url, [])
            raw_accum = raw_by_url.setdefault(draft.artifact_url, [])

            # SHORT-CIRCUIT a repeat probe of an artifact for which NOTHING was ever
            # scored (below). ``_FAILED_EVAL_SCORE`` no longer satisfies loop_engine's
            # ``score >= threshold`` gate, so this no longer converges on pass 1 — the
            # loop tries a pass 2. With nothing scored there is nothing to nudge, so
            # ``optimizer_fn`` returns the SAME draft rather than re-rendering, and
            # without this guard pass 2 would re-run the analyzer AND every WM critic
            # batch against an endpoint that, by definition, never answered at all —
            # doubling wall-clock cost for zero new information and blowing the outer
            # per-attempt budget (measured: 1.51s of a 1.00s slice).
            #
            # Scoped ONLY to ``critic_never_scored_urls``, never the broader
            # ``unavailable_by_url`` (which also covers a PARTIAL batch outage — some
            # shots scored, some not, analyzer working fine): a batch timeout there is
            # plausibly transient within its own retry ladder, and a genuinely fresh
            # pass-2 attempt against the WM worker (a fresh retry budget, not a repeat
            # of pass 1's) is how
            # ``test_successful_second_pass_clears_stale_unavailable_marker`` recovers
            # — that path must keep re-probing, not skip it.
            if draft.artifact_url in critic_never_scored_urls:
                return Critique(score=_FAILED_EVAL_SCORE, rerun_stages=[], directives=records)

            # DEADLINE (bug: the inner ladder could not fit inside the outer slice).
            # ``entry`` is the monotonic start of THIS outer attempt; every batch slice
            # below is clamped against what is actually left of it.
            #
            # ⚠️ Only the ENTRY TIME is captured here. The batch-phase deadline is
            # computed AFTER the analyzer poll below, from the analyzer's MEASURED
            # elapsed — capturing the deadline here instead deducted that elapsed from
            # the batch phase as well as from the analyzer's own reserve, which starved
            # the critic waves on 48.7% of completed runs. See ``_batch_phase_seconds``.
            clock = asyncio.get_running_loop()
            entry = clock.time()

            # SOFT: any analyzer/worker failure returns a passing, no-flag verdict so
            # the loop converges on the current artifact instead of raising.
            try:
                metrics, frames = await guarded(lambda: self._analyze(draft.artifact_url))
            except (
                PermanentUpstreamError,
                TransientUpstreamError,
                httpx.HTTPError,
                RuntimeError,
                TimeoutError,
                FileNotFoundError,
            ) as exc:
                # SOFT: a genuine analyzer OUTAGE (transport/HTTP/timeout/job-lost) must not
                # fail the run. A programming bug (KeyError/AttributeError/TypeError) is NOT
                # caught here — it propagates instead of being mislabelled "analyzer unavailable".
                #
                # ⚠️ It is recorded, not swallowed, and it must not score as a pass. This
                # path used to return ``score=_THRESHOLD`` — which satisfies loop_engine's
                # own ``score >= threshold`` gate, i.e. structurally IDENTICAL to "the
                # critic reviewed the footage and found nothing wrong" — so a 4xx from the
                # analyzer (its SSRF allow-list rejecting the artifact host is the live
                # shape) converged the loop as a clean pass. ``_FAILED_EVAL_SCORE`` sits
                # below the 1-5 clamp floor so "measured nothing" can never be confused
                # with a real verdict. The deliverable still ships the current artifact
                # unchanged (this stage never hard-blocks); only the honesty of the score
                # and the report changes.
                detail = describe_exception(exc)[:300]
                log.warning(
                    "%s: analyzer unavailable (%s), accepting current: %s",
                    self.name,
                    type(exc).__name__,
                    exc,
                )
                # ``records`` is already published (and may already carry a previous
                # attempt's verdicts) — nothing to seed here.
                unavailable_by_url[draft.artifact_url] = (
                    f"the video analyzer never returned a reading — {detail}; no shot was scored"
                )
                critic_never_scored_urls.add(draft.artifact_url)
                return Critique(score=_FAILED_EVAL_SCORE, rerun_stages=[], directives=[])
            # THE APPORTIONMENT. Sized here, after the poll, from what the analyzer
            # ACTUALLY spent — so the waves get every second the analyzer did not use,
            # and the analyzer's elapsed is charged exactly once.
            phase_deadline = clock.time() + self._batch_phase_seconds(clock.time() - entry)
            measured = self._measure(
                metrics, frames, windows, draft.plans, idle_by_shot=idle_by_shot
            )
            # Score the shots in bounded batches (≤ _CRITIC_BATCH_SIZE per WM call)
            # so no single call carries an unbounded ``shots`` output schema — the
            # small vLLM otherwise times out and the SOFT critique silently degrades
            # to an empty verdict. Each batch's ``shots`` list is accumulated and
            # validated once at the end (same merge + convergence semantics as the
            # former single call). SOFT contract preserved: a batch failure NEVER
            # raises. It no longer aborts the fan-out either — the batches run
            # CONCURRENTLY (``_CRITIC_BATCH_CONCURRENCY``) and the successful ones count; only when NOTHING came
            # back do we fall back to the passing, no-flag accept-current verdict.
            critic_batches = list(batched(measured, _CRITIC_BATCH_SIZE, strict=False))
            batch_count = len(critic_batches)
            limit = asyncio.Semaphore(_CRITIC_BATCH_CONCURRENCY)
            # Built HERE, outside the call helper: a ValueError from message-building
            # is a BUG and must propagate, exactly as it did when it sat inside the
            # serial loop's ``try``.
            messages = [self._build_message(batch) for batch in critic_batches]
            # Play order of the shots this pass measured — the spine ``_republish``
            # sorts the retained records by, so the list a reviewer reads is identical
            # whatever order the concurrent batches happened to land in.
            spine = {m["shot_id"]: i for i, m in enumerate(measured)}

            def _republish() -> None:
                """Re-derive the published records from everything accumulated so far.

                In place (``records[:]``), because ``records`` is the SAME list object
                already published in ``records_by_url`` — reassigning it would put the
                durability property back the way it was broken.

                ⚠️ REVERSED, and that is the whole dedup policy. ``_validate`` keeps the
                FIRST verdict per shot_id, so feeding it the accumulator backwards makes
                the NEWEST verdict win. Accumulating is what makes a retried attempt keep
                the batches it already paid for; last-write-wins is what stops that
                accumulation freezing a stale score when a genuine SECOND PASS re-scores
                the same url (the optimizer hands back the same draft when a re-render
                fails) — the pass-1 verdict must not outrank the pass-2 one.
                """
                validated = self._validate(list(reversed(raw_accum)), valid_ids)
                records[:] = sorted(validated, key=lambda r: spine.get(r.shot_id, len(spine)))

            # Per-WAVE share of what is LEFT of the critic phase. Batches inside a wave
            # run CONCURRENTLY (they share one wall clock); waves run one after another,
            # so the budget divides by the WAVE count, not the batch count:
            #
            #     waves = ceil(batch_count / _CRITIC_BATCH_CONCURRENCY)
            #     share = remaining / waves          ⇒  waves * share == remaining
            #
            # No floor and no fudge factor ON THE SHARE — raising it would break exactly
            # the equality this exists to buy (see ``audiobook/qa.py``'s
            # ``_MIN_JUDGE_BATCH_SECONDS`` comment for what a floor costs).
            # ``_CRITIC_FLOOR_SECONDS`` is NOT that: it never enlarges a share, it makes
            # a batch whose share is provably too small REFUSE instead of burning two
            # unsatisfiable attempts. The equality is untouched — a refused batch simply
            # does not spend its share. Each batch clamps ``share`` AGAIN against
            # the live remaining when it is admitted, so a wave that overruns steals from
            # itself, never from the next one.
            waves = max(1, ceil(batch_count / _CRITIC_BATCH_CONCURRENCY))
            share = max(0.0, phase_deadline - clock.time()) / waves

            failure_slots: list[str | None] = [None] * batch_count

            async def _run_batch(index: int) -> None:
                batch_shots, failure, batch_usage = await _score_critic_batch(
                    wm=self._wm,
                    worker_name=self.worker_name,
                    stage_name=self.name,
                    message=messages[index],
                    batch_number=index + 1,
                    limit=limit,
                    # The ONE upstream policy: guarded()/classify() decide
                    # permanent-vs-transient; the SLICE decides what may be spent.
                    deadline=phase_deadline,
                    share_seconds=share,
                    retries=self._llm_retries,
                )
                # DURABILITY: fold this batch in the MOMENT it lands, not after the
                # gather. An outer cancel between here and the gather's return used to
                # discard every batch already scored and then re-pay for all of them.
                failure_slots[index] = failure
                if failure is None:
                    # Only a call that SPOKE is counted (``_usage_add`` bumps ``calls``
                    # even on a None payload). Its sums are commutative, so folding in
                    # completion order is value-identical to folding in batch order.
                    _usage_add(usage, batch_usage)
                raw_accum.extend(batch_shots)
                _republish()

            # ``return_exceptions=True`` is REQUIRED here — the soft contract says one
            # batch's failure must not kill the pass, and it also keeps a sibling from
            # being left as an orphaned task. ``failure_slots`` is indexed by BATCH, so
            # the reason string a reviewer reads is in batch order whatever order the
            # calls completed in.
            settled = await asyncio.gather(
                *(_run_batch(index) for index in range(batch_count)),
                return_exceptions=True,
            )
            for result in settled:
                if isinstance(result, BaseException):
                    raise result
            failures = [reason for reason in failure_slots if reason is not None]
            if failures:
                # Record WHICH batches never spoke, keyed by the draft they were scoring,
                # so ``handle`` can surface it on the retained artifact's record.
                unavailable_by_url[draft.artifact_url] = (
                    f"{len(failures)} of {batch_count} critic batch(es) returned no verdict — "
                    + "; ".join(failures[:3])
                )
            else:
                # CLEAR a marker left by an EARLIER pass on this same artifact_url. The
                # optimizer returns the SAME draft when a re-render fails, so pass 2 can
                # re-critique the identical url; ``records_by_url`` is republished
                # per batch above, and a marker that is only ever SET would make a
                # fully successful second pass report ``critique_unavailable: True`` and
                # degrade the run on stale information. Set and cleared together, keyed the
                # same way, so the marker always describes the LATEST verdict for this url.
                unavailable_by_url.pop(draft.artifact_url, None)
            if failures and not records:
                # NOTHING came back at all ⇒ preserve the historical soft contract exactly:
                # a passing, no-flag verdict so the loop converges on the current artifact
                # (now carrying the unavailable marker so it is no longer a silent green).
                # Tested against ``records``, not this pass's raw shots, so a retried
                # attempt that adds nothing still reports the EARLIER attempt's verdicts
                # instead of falling back to the empty one.
                critic_never_scored_urls.add(draft.artifact_url)
                return Critique(score=_FAILED_EVAL_SCORE, rerun_stages=[], directives=[])
            # A failing batch NO LONGER discards the batches that already succeeded, and
            # neither does an outer cancel: ``records`` was republished as each batch
            # landed, so the verdict here is built from whatever came back.
            flagged = [r.shot_id for r in records if r.score < _THRESHOLD]
            # Convergence score = the WORST shot: any shot below the bar keeps the
            # loop iterating so a flagged shot actually gets a re-render pass.
            #
            # ⚠️ KNOWN LIMITATION (documented, deliberately NOT fixed here): on a PARTIAL
            # batch — some batches answered, some did not — ``records`` covers only the
            # shots that were actually scored. An unscored shot is therefore
            # indistinguishable from a passing one: it contributes no entry, so it cannot
            # be flagged, and this ``min()`` over the SURVIVORS can sit above _THRESHOLD
            # and converge the loop while the un-reviewed shots were never looked at.
            # Nothing here can score a shot the critic never answered on; the honest
            # mitigation is the ``critique_unavailable`` / ``unavailable_reason`` marker
            # set above, which makes the incompleteness visible in ``stage_outputs`` and
            # degrades the run — read those before treating a converged score or an empty
            # ``flagged_shots`` as evidence that the footage is good.
            score = min((r.score for r in records), default=_SCORE_MAX)
            # A COPY: ``records`` keeps being mutated by a later attempt on the same
            # url, and a Critique holding the live list would silently re-describe
            # itself.
            return Critique(score=score, rerun_stages=flagged, directives=list(records))

        async def critic_fn(draft: _FootageDraft) -> Critique:
            # A draft whose re-render was skipped (cut-only) keeps its pass-1 verdict.
            reused = reuse_critique_by_url.pop(draft.artifact_url, None)
            if reused is not None:
                return reused
            return await _score_draft(draft)

        # Loop-position render-pass ordinal: increments ONLY when an actual re-render
        # is submitted (the unflagged early-return does NOT bump it). The ordinal is
        # loop-position-derived, so on a wholesale handler re-run after a resumable
        # timeout the Nth re-render gets the SAME ``pass{n}`` key regardless of the
        # (non-deterministic) verdict content — the compiler then dedupes it instead
        # of paying for a second render.
        # NOTE: today ``run_evaluator_optimizer`` admits at most ONE optimize call
        # (``_MAX_PASSES=2`` → one revision), so this is always 1 in practice. The
        # counter is the correct GENERAL form: were the ceiling ever raised, a genuine
        # 2nd revision would get its own ``pass2`` key rather than being wrongly deduped
        # to the 1st (a hardcoded ``1`` would reintroduce that wrong-dedup).
        render_pass = 0
        # The re-render outcome of the LAST optimizer call. Starts NOT_ATTEMPTED: a loop
        # that converged on pass 1 (or flagged nothing) never submitted one.
        rerender_outcome = RERENDER_NOT_ATTEMPTED

        async def optimizer_fn(draft: _FootageDraft, critique: Critique) -> _FootageDraft:
            # SOFT: a re-render failure returns the SAME draft ⇒ the loop plateaus
            # and keeps the best artifact — never raises.
            records = critique.directives if isinstance(critique.directives, list) else []
            by_id = {r.shot_id: r for r in records if isinstance(r, ShotCritique)}
            flagged = {sid for sid, r in by_id.items() if r.score < _THRESHOLD}
            nonlocal render_pass, rerender_outcome
            if not flagged:
                rerender_outcome = RERENDER_NOT_ATTEMPTED
                return draft
            if _only_cut_is_low(by_id, flagged):
                # The only lever left is ``transition_signal = cut``: it changes how a
                # shot is ENTERED, not what is on screen, so the shot the critic scored
                # would be re-rendered unchanged. Skip the paid pass; the record and the
                # below-bar degradation say why.
                log.info(
                    "%s: boring_cut is the only dimension below the bar on %d flagged "
                    "shot(s) — skipping the re-render (it cannot change on-screen content)",
                    self.name,
                    len(flagged),
                )
                rerender_outcome = RERENDER_SKIPPED_CUT_ONLY
                reuse_critique_by_url[draft.artifact_url] = critique
                return draft
            render_pass += 1
            nudged = _nudge_plans(draft.plans, by_id, flagged)
            try:
                new_url, new_duration = await self.rerender(ctx, nudged, render_pass)
            except (httpx.HTTPError, RuntimeError, TimeoutError) as exc:
                # SOFT: a genuine re-render OUTAGE (incl. a lost compile job, RuntimeError)
                # ⇒ keep the best artifact. A bug in nudge/plan construction propagates.
                log.warning(
                    "%s: re-render failed (%s), keeping current: %s",
                    self.name,
                    type(exc).__name__,
                    exc,
                )
                rerender_outcome = RERENDER_FAILED
                return draft
            rerender_outcome = RERENDER_RENDERED
            return _FootageDraft(
                artifact_url=new_url, duration_seconds=new_duration, plans=tuple(nudged)
            )

        # Belt-and-suspenders: critic/optimizer already swallow their own failures,
        # but a defect in the loop itself must STILL accept the current artifact —
        # the soft gate never lets this stage fail the deliverable.
        try:
            result = await run_evaluator_optimizer(
                initial,
                critic_fn=critic_fn,
                optimizer_fn=optimizer_fn,
                threshold=_THRESHOLD,
                epsilon=_EPSILON,
                max_passes=_MAX_PASSES,
                stage_name=self.name,
                timeout_seconds=self._llm_timeout,
                retries=self._llm_retries,
            )
        except StageSkippedError, StagePreconditionError:
            # Control-flow signals are NOT failures — a skip/precondition must reach the
            # run engine, not be swallowed into a degraded "accept current" (matches the
            # sibling guard in demo_video/final_critique.py).
            raise
        except Exception as exc:
            # A loop error means the critic produced NO verdict for this artifact — most
            # commonly a truncated/invalid JSON body from the worker (a decode ValueError
            # escapes ``critic_fn``'s narrow catches by design). Historically this path
            # built a record with the unavailable markers left at their defaults, so a
            # generation that never parsed reported as ``critique_unavailable: False`` +
            # ``flagged_shots: []`` — i.e. a clean pass. It must be reported exactly like
            # a batch that returned nothing: honest, visible, and still NOT a run failure.
            detail = describe_exception(exc)[:300]
            log.error(
                "%s: critic loop failed, accepting current artifact and reporting the "
                "critique as UNAVAILABLE (run not failed): %s",
                self.name,
                detail,
                exc_info=True,
            )
            # DURABILITY, the visible half: ``critic_fn`` publishes each batch's
            # verdicts as they land, so whatever was scored before the loop died is
            # still here. Reporting it beats reporting nothing — and it is what stops
            # the next attempt (and the operator) treating the work as never done.
            partial = records_by_url.get(initial.artifact_url, [])
            return self._accept_current(
                initial,
                usage,
                reason="loop_error",
                note=(
                    f"footage_critique: critique INCOMPLETE — the critic loop failed "
                    f"({detail}). {len(partial)} shot(s) were scored before it failed; the "
                    "footage was NOT fully reviewed, so an empty flagged_shots list here is "
                    "not evidence of good footage."
                ),
                unavailable_reason=f"critic loop error — {detail}",
                shots=partial,
            )

        best = result.draft
        best_records = records_by_url.get(best.artifact_url, [])
        unavailable = unavailable_by_url.get(best.artifact_url, "")
        critique_record = FootageCritiqueRecord(
            score=result.score,
            passes=result.passes,
            converged=result.converged,
            reason=result.reason,
            flagged_shots=list(result.rerun_stages),
            shots=best_records,
            critique_unavailable=bool(unavailable),
            unavailable_reason=unavailable,
            rerender_outcome=rerender_outcome,
        )
        notes: list[str] = []
        degradations: list[Degradation] = []
        if unavailable:
            # BOTH channels. ``notes`` is prose for a human and is auto-translated into
            # a ``degraded`` run by runs/_shared.py; ``degradations`` is the queryable
            # half. Prose ALONE is what made this invisible: the DAG executor stamped
            # every such row ``unclassified_note`` with a NULL error, so 48.7% of
            # completed runs shipped an unreviewed picture that no query could find.
            # The deliverable still ships either way.
            degradations.append(degradation(FOOTAGE_CRITIQUE_UNAVAILABLE, unavailable))
            notes.append(
                f"footage_critique: critique INCOMPLETE — {unavailable}. "
                f"{len(best_records)} shot(s) scored; the footage was NOT fully reviewed, "
                "so an empty flagged_shots list here is not evidence of good footage."
            )
        if not result.converged and best_records:
            # BOTH channels again. The note alone is how a 2.0-against-4.0 critique
            # (run 01a0ab23) shipped with nothing any query, badge or alert could bind
            # to. The code carries the discriminator the bare ``reason`` cannot: a FAILED
            # re-render hands back the same draft, so it too stops as ``plateau``.
            # Never terminal (see FOOTAGE_BELOW_BAR) and never another paid pass.
            below = _below_bar_detail(result.score, result.passes, result.reason, rerender_outcome)
            degradations.append(degradation(FOOTAGE_BELOW_BAR, below))
            notes.append(f"footage_critique: accepted best artifact — {below}")
        return FootageCritiqueOutput(
            artifact_url=best.artifact_url,
            duration_seconds=best.duration_seconds,
            critique=critique_record,
            token_usage=_build_usage(usage),
            notes=notes,
            degradations=degradations,
            # Persist the retained best draft's plan rows so encode_master's re-compile
            # reproduces THIS artifact's footage (the initial draft carries the L1
            # baseline plans; a nudged draft carries the improved ones).
            footage_plans=[dict(p) for p in best.plans],
            idle_trim_by_shot={k: dict(v) for k, v in idle_by_shot.items()},
        )

    def _accept_current(
        self,
        draft: _FootageDraft,
        usage: dict[str, int],
        *,
        reason: str,
        note: str,
        unavailable_reason: str = "",
        shots: Sequence[ShotCritique] = (),
    ) -> FootageCritiqueOutput:
        """Emit the current artifact with an empty/OK critique (the soft fallback).

        ``unavailable_reason`` is the honest-degrade carrier: pass it on any fallback
        where the critic was ASKED and did not deliver a verdict (a loop error, a
        truncated/undecodable response), and the emitted record carries
        ``critique_unavailable=True`` + the reason instead of defaulting to a clean pass.
        Leave it empty for the fallbacks where there was nothing to critique in the first
        place (``critique_not_requested`` / ``no_directed_footage``) — those are honest
        no-ops, not degrades. This never raises either way: the critic is advisory.

        ``shots`` carries any verdicts that WERE scored before the fallback was taken
        (the loop-error path can now hold a partial set, because ``critic_fn`` publishes
        each batch as it lands). Empty by default, which is every other caller. Note the
        two are orthogonal and BOTH are reported: partial verdicts do not make the
        critique available, and ``unavailable_reason`` stays the thing that distinguishes
        "the critic did not finish" from "reviewed, found nothing" (#864).
        """
        scored = list(shots)
        return FootageCritiqueOutput(
            artifact_url=draft.artifact_url,
            duration_seconds=draft.duration_seconds,
            critique=FootageCritiqueRecord(
                score=min((r.score for r in scored), default=_SCORE_MAX),
                passes=0,
                converged=True,
                reason=reason,
                flagged_shots=[r.shot_id for r in scored if r.score < _THRESHOLD],
                shots=scored,
                critique_unavailable=bool(unavailable_reason),
                unavailable_reason=unavailable_reason,
            ),
            token_usage=_build_usage(usage),
            notes=[note] if note else [],
            # The queryable twin of the note, on exactly the fallbacks where the critic
            # WAS asked and did not deliver (``unavailable_reason`` is the discriminator
            # this docstring already defines). The honest no-ops — nothing was requested,
            # nothing was directed — carry neither channel and stay clean.
            degradations=(
                [degradation(FOOTAGE_CRITIQUE_UNAVAILABLE, unavailable_reason)]
                if unavailable_reason
                else []
            ),
            # No nudge happened on this path — carry the draft's own plans through
            # (the pass-through draft has none ⇒ ``[]`` ⇒ no persisted override, so a
            # re-compile falls back to the base footage-direction plans).
            footage_plans=[dict(p) for p in draft.plans],
        )

    async def _analyze(
        self, artifact_url: str
    ) -> tuple[VideoAnalyzerQualityMetrics | None, object]:
        """Submit the artifact to video_analyzer, poll, and return
        ``(quality_metrics, frame_analyses)`` from the full result body.

        ``quality_metrics`` is ``None`` when the analyzer did not run the quality
        stage (it always should here — ``_ANALYSIS_CONFIG`` requests it).
        ``frame_analyses`` is the analyzer's real per-frame field (each item
        carries ``timestamp`` + ``ocr_text``); it was previously read as ``frames``,
        a key the server never sends, so the per-shot OCR bucket was always empty.
        """
        job_id = await self._analyzer.start_analysis(
            video_url=artifact_url,
            config_overrides=_ANALYSIS_CONFIG,
        )
        await self._analyzer.poll_analysis(job_id, timeout=_ANALYSIS_POLL_TIMEOUT_S)
        result = await self._analyzer.get_result(job_id)
        return result.quality_metrics, result.frame_analyses

    def _measure(
        self,
        metrics: VideoAnalyzerQualityMetrics | None,
        frames: object,
        windows: Sequence[ShotWindow],
        plans: Sequence[Mapping[str, object]],
        *,
        idle_by_shot: Mapping[str, Mapping[str, object]] | None = None,
    ) -> list[_MeasuredShot]:
        """Map whole-MP4 time-indexed signals → per-shot measured numbers.

        ``idle_by_shot`` (optional) adds the shot's source-recording idle measurement as
        ``idle_trim``; a shot without one gets no key, so its row is unchanged.

        See the module docstring on absolute-time imprecision: title cards / cold
        open shift the axis, so a shot's motion bucket is approximate.
        """
        motion_by_shot = _bucket_motion(metrics.motion_timeline if metrics else [], windows)
        ocr_by_shot = _bucket_ocr(frames, windows)
        plan_by_id = {str(p["line_id"]): p for p in plans if p.get("line_id") is not None}
        overall_sharpness = metrics.mean_sharpness if metrics else 0.0
        overall_brightness = metrics.mean_brightness if metrics else 0.0
        overall_motion = metrics.motion_score if metrics else 0.0
        overall_score = metrics.overall_score if metrics else 0.0
        out: list[_MeasuredShot] = []
        for w in windows:
            shot_id = w["shot_id"]
            plan = plan_by_id.get(shot_id)
            # Only DIRECTED shots (those carrying a footage plan) are critiquable —
            # a shot with no plan has no INTENT field to nudge. Card/undirected lines
            # still occupied their window above so the time axis stays correct; we
            # just don't ask the critic to score them.
            if plan is None:
                continue
            samples = motion_by_shot.get(shot_id, [])
            row: _MeasuredShot = {
                "shot_id": shot_id,
                "t0": round(w["t0"], 2),
                "t1": round(w["t1"], 2),
                "motion_mean": round(sum(samples) / len(samples), 4) if samples else 0.0,
                "motion_samples": len(samples),
                "ocr_text": " ".join(ocr_by_shot.get(shot_id, []))[:_MAX_ISSUE_CHARS],
                "overall_sharpness": overall_sharpness,
                "overall_brightness": overall_brightness,
                "overall_motion": overall_motion,
                "overall_score": overall_score,
                "plan": {
                    "zoom": plan.get("zoom"),
                    "reframe": plan.get("reframe"),
                    "transition_signal": plan.get("transition_signal"),
                    "callout": plan.get("callout"),
                },
            }
            idle = (idle_by_shot or {}).get(shot_id)
            if idle:
                row["idle_trim"] = dict(idle)
            out.append(row)
        return out

    def _build_message(self, measured: Sequence[_MeasuredShot]) -> str:
        body = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in measured)
        return (
            "SHOTS (one JSON object per shot — measured numbers off the compiled "
            "video + its current framing plan):\n" + body
        )

    def _validate(self, raw: Sequence[object], valid_ids: set[str]) -> list[ShotCritique]:
        """Keep only verdicts on a KNOWN shot_id, clamp every score to ``[1, 5]``,
        and bound the free-text fields — the model is advisory, this stage is
        authoritative (mirrors footage_direction._validate)."""
        out: list[ShotCritique] = []
        seen: set[str] = set()
        for raw_item in raw:
            if not isinstance(raw_item, Mapping):
                continue
            item = cast("Mapping[str, object]", raw_item)
            shot_id = str(item.get("shot_id") or "")
            if shot_id not in valid_ids or shot_id in seen:
                continue
            seen.add(shot_id)
            out.append(
                ShotCritique(
                    shot_id=shot_id,
                    score=_clamp_score(item.get("score")),
                    issue=str(item.get("issue") or "")[:_MAX_ISSUE_CHARS].strip(),
                    suggested_fix=str(item.get("suggested_fix") or "")[:_MAX_FIX_CHARS].strip(),
                    eye_directed=_opt_score(item.get("eye_directed")),
                    boring_cut=_opt_score(item.get("boring_cut")),
                    app_demoed=_opt_score(item.get("app_demoed")),
                    framing=_opt_score(item.get("framing")),
                )
            )
        return out


def _below_bar_detail(score: float, passes: int, reason: str, rerender_outcome: str) -> str:
    """One sentence for the ``footage_below_bar`` degradation: score, bar, stop reason and
    what the re-render did — worded so "the re-render failed" never reads like "the
    re-render ran and did not help"."""
    if rerender_outcome == RERENDER_FAILED:
        cause = "the re-render FAILED, so no improved footage was ever scored"
    elif rerender_outcome == RERENDER_RENDERED:
        cause = "a re-render ran and did not lift the score over the bar"
    elif rerender_outcome == RERENDER_SKIPPED_CUT_ONLY:
        cause = (
            "the re-render was SKIPPED: boring_cut was the only dimension below the bar, "
            "and its one lever (a hard-cut transition) cannot change on-screen content"
        )
    else:
        cause = "no re-render was submitted"
    return (
        f"footage scored {score:.1f} against a bar of {_THRESHOLD:.1f} after {passes} "
        f"pass(es); stop={reason}; rerender={rerender_outcome} — {cause}"
    )


def _low(value: float | None) -> bool:
    """A sub-dimension reads as 'weak' when present and below the threshold."""
    return value is not None and value < _THRESHOLD


def _only_cut_is_low(by_id: Mapping[str, ShotCritique], flagged: set[str]) -> bool:
    """True when EVERY flagged shot's only weak dimension is ``boring_cut``.

    ``_nudge_plans`` answers ``boring_cut`` with ``transition_signal = cut`` alone, which
    changes the join, not the pixels the critic scored — so a re-render cannot lift a
    score that is the MIN across shots. A flagged shot with any other weak dimension,
    or with NO weak dimension (it gets the default zoom-on nudge), is actionable.
    """
    for shot_id in flagged:
        rec = by_id.get(shot_id)
        if rec is None or not _low(rec.boring_cut):
            return False
        if _low(rec.eye_directed) or _low(rec.framing) or _low(rec.app_demoed):
            return False
    return bool(flagged)


def _nudge_plans(
    plans: Sequence[Mapping[str, object]],
    by_id: Mapping[str, ShotCritique],
    flagged: set[str],
) -> list[dict[str, object]]:
    """Return a NEW plan list with INTENT-only nudges on the flagged shots only.

    Coordinates are never touched — only enum/intent fields the compiler already
    understands: a weak eye/framing forces zoom on (the compiler resolves the bbox
    from the captured track, or no-ops when there is none); a boring cut asks for a
    hard cut transition; a weak app-demo asks the reframe to follow the cursor. A
    shot flagged with no per-dimension signal gets the conservative default of
    zoom-on. Non-flagged plans pass through byte-identical.
    """
    out: list[dict[str, object]] = []
    for plan in plans:
        row = dict(plan)
        shot_id = str(plan.get("line_id") or "")
        if shot_id in flagged:
            rec = by_id.get(shot_id)
            touched = False
            if rec is not None:
                if _low(rec.eye_directed) or _low(rec.framing):
                    row["zoom"] = _ZOOM_ON
                    touched = True
                if _low(rec.boring_cut):
                    row["transition_signal"] = _CUT
                    touched = True
                if _low(rec.app_demoed):
                    row["reframe"] = _REFRAME_ON
                    touched = True
            if not touched:
                # Flagged but no actionable sub-dimension ⇒ direct the eye by default.
                row["zoom"] = _ZOOM_ON
        out.append(row)
    return out


def _build_usage(acc: Mapping[str, int]) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=acc.get("prompt_tokens", 0),
        completion_tokens=acc.get("completion_tokens", 0),
        total_tokens=acc.get("total_tokens", 0),
        calls=acc.get("calls", 0),
    )
