"""segment_select — the LLM picks which seconds of the footage are worth keeping.

The editorial decision of the product. Given each source's MEASURED profile from
``shot_analysis`` (duration, the cut points already in the footage, the motion
curve, the quality aggregate) the model returns an ordered list of candidate
segments: an in/out window, a 0..1 score, and a one-sentence rationale.

A source the analyzer FAILED on is still part of the corpus. It reaches the
model marked ``"unmeasured": true`` with its duration and nothing else, because
"never measured" and "measured, and dull" are different facts and only one of
them is a reason to leave the operator's footage out of their reel; the count
goes into the stage notes so the review gate can read a thin reel's cause. The
only source dropped is an unmeasured one with no known duration — no window can
be cut from a clip of unknown length — and that drop is counted too.

The model is ADVISORY; this stage is AUTHORITATIVE — the same posture as
``audiobook_cast`` and ``sound_director``. Every returned segment is validated
before it can reach the cut planner:

* an unknown ``source_id`` is dropped (the model may not invent footage);
* the window is clamped into ``[0, duration]`` of the source it names, and an
  inverted or zero-length window after clamping is dropped (a negative-length
  clip is a compositor 422, not an edit);
* a window shorter than ``_MIN_SEGMENT_SECONDS`` is dropped — a 0.3s flash is a
  glitch, never a shot — and one longer than ``_MAX_SEGMENT_SECONDS`` is
  truncated from its head, which is where the model's own rationale says the
  moment starts;
* the count is capped so a model that returns "the whole clip, sliced" cannot
  turn a 90-second reel into a 400-shot render bill;
* segments that would land in the reel as the SAME SHOT TWICE — same source,
  overlapping or near-adjacent windows — are dropped, because nothing downstream
  compares two windows to each other and a reel of one kiosk from one angle is
  the most visible way this product fails while every stage reports success
  (see ``_drop_near_duplicates``). That guard DROPS; it never re-orders — the
  order of the returned list is the model's edit, and ``cut_plan`` reads a
  source change between adjacent segments as a section change, so re-ordering
  here would silently move the beat placement of every reel.

A SHORT draw is a failure too, and it used to be an invisible one: the retry
only fired on an EMPTY draw, so a live 10-clip run that returned 18.8 s of
segments against an 80.0 s brief passed here without a word and surfaced one
stage later as ``cut_plan.shortfall_seconds``. This stage now measures its own
draw against WHAT THE REEL NEEDS: materially short with untouched footage left
buys ONE corrective re-ask that names the numbers, the larger of the two draws
is kept, and a result that still cannot fill the reel carries a note saying so —
see ``_should_reask``.

The yardstick is deliberately NOT the ~2x the brief ASKS for
(``_COVERAGE_FACTOR``). Judging the answer against the ask made the re-ask fire
on essentially every real run: 10 clips totalling 97.1 s against a 45 s reel puts
the ask at min(90.0, 97.1) = 90.0 s — 92.7% of ALL the footage — so a healthy
50 s editorial draw that fills the 45 s reel twice over read as a shortfall,
bought a second full pass (5 extra worker-management calls against a slow vLLM),
and told the reviewer "SHORT SELECTION: 50.0s ... against the 90.0s the brief
asked for". Asking for headroom is legitimate; measuring the answer against the
ask instead of against the need is the bug. See ``_REASK_COVERAGE_MULTIPLE``.

**This stage pauses**: it is the first output a human can read as an EDIT
("keep 4.2s-7.8s of clip 2 because that is the jump"). Every later stage is
mechanical by comparison, and a wrong selection reviewed here costs nothing
while the same selection reviewed after the compile costs a full render.

Rationale is not decoration. It is what makes the pause reviewable — a list of
timecodes with no reasons cannot be judged, only re-run — so a segment whose
rationale is empty still passes (we never drop footage over prose) but the
prompt asks for it explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import batched
from typing import TYPE_CHECKING, ClassVar, cast
from uuid import uuid7

from w2v_common.stages import stage_call

from src.services.llm import WMClient
from src.services.stages.action_reel.sampling import MOTION_SAMPLES_PER_SOURCE
from src.services.stages.action_reel.target_duration import target_seconds
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    StagePreconditionError,
)
from src.services.stages.batch_ledger import BatchLedger
from src.services.stages.llm_budget import (
    LLM_RETRY_ON,
    llm_budget_seconds,
    llm_client_timeout_seconds,
    llm_retries,
    output_token_budget,
)
from src.services.stages.primitives.script_stage import non_empty_array
from src.services.stages.upstream import guarded

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.services.runs.run_context import RunContext

# Editorial floors/ceilings. A reel shot is a beat, not a scene: under 0.4s reads
# as a glitch, over 12s stops being a cut and starts being a clip.
_MIN_SEGMENT_SECONDS = 0.4
_MAX_SEGMENT_SECONDS = 12.0
# Hard cap on candidates. The cut planner picks a subset anyway; this only stops
# a degenerate draw from turning into an unbounded render.
_MAX_SEGMENTS = 60
# How many motion samples to show per source — see ``action_reel/sampling.py``.
_MOTION_SAMPLES_PER_SOURCE = MOTION_SAMPLES_PER_SOURCE
# How many scene-cut times to show per source. The analyzer emits one entry per
# detected boundary and a long, busy clip produces hundreds of them — unbounded
# prompt input sitting right next to a curve we were already careful to bound.
# Thinned by even striding for the same reason as the motion curve: head
# truncation would hide every cut after the first seconds of the clip.
_SCENE_CUTS_PER_SOURCE = 24
# How many source clips go into ONE model call. The prompt is the whole corpus
# concatenated, so a 10-clip run used to hand a small self-hosted model every
# clip's duration + cut list + motion curve in a single message.
#
# Two separate failures live here and only one of them is fixed by a token
# budget. Sizing the OUTPUT budget (below) keeps the model from running out of
# tokens mid-reasoning; it does nothing about the INPUT side, where the prompt
# itself grows without bound as clips are added — and a longer prompt makes the
# model think LONGER, spending that budget faster. Batching is what bounds the
# prompt. Same shape as ``footage_critique._CRITIC_BATCH_SIZE``.
_SELECT_BATCH_SOURCES = 2
# How many of those batch calls may be IN FLIGHT at once.
#
# Batching bounds each call's PROMPT. It does nothing about the WALL CLOCK, and
# ``_draw`` used to await the batches one after another inside a SINGLE
# ``stage_call(timeout_seconds=llm_budget_seconds())``: the stage's worst case
# was ceil(sources/2) serial calls against one fixed budget, i.e. a tail that
# grows with the corpus while every individual call stays comfortably in budget.
# Nothing caps the corpus either — ``_MAX_SEGMENTS`` caps the merged draw, not
# the number of clips.
#
# 5 is 10 / ``_SELECT_BATCH_SOURCES``: one wave covers the 10-clip corpus this
# module's docstring is written about (the live run that returned 18.8 s), so the
# number of SEQUENTIAL waves stops tracking corpus size over the range this stage
# is actually run at. Past 10 clips the tail grows again, one wave per further
# 10 — concurrency bounds the arithmetic, it does not make it free.
#
# What is KNOWN about this value: the target is ONE small vLLM doing continuous
# batching, which decodes concurrent sequences in the same forward step, so k
# calls in flight do not cost k x one call; and 5 x ``_SELECT_MAX_TOKENS`` of KV
# cache is negligible against that server's ``--max-model-len=131072``.
# What is NOT known, and is deliberately NOT claimed anywhere here: the
# per-request token rate at k=5. The 11.5 tok/s in ``llm_budget.py`` was measured
# on SERIAL calls under fleet load; concurrency raises AGGREGATE throughput and
# lowers the per-request rate by an amount nobody has measured. No speedup figure
# is asserted — the test bounds a call by the DOWNSTREAM READ DEADLINE, which
# holds however slowly an individual call decodes.
_SELECT_BATCH_CONCURRENCY = 5
# Worst-case output tokens for ONE returned segment: source_id + two numbers +
# score + a one-sentence rationale, JSON-escaped, plus keys/braces/commas.
_MAX_TOKENS_PER_SEGMENT = 110
# How many segments one source can realistically justify. The system prompt tells
# the editor a 20-second clip yields three or four distinct moments; 8 is that
# with headroom, and it is what the budget is sized for rather than a limit the
# validator enforces (``_MAX_SEGMENTS`` is that limit, over the merged draw).
_SEGMENTS_PER_SOURCE = 8
# Worst-case size of the ANSWER for one batch. DERIVED FROM ``_SELECT_BATCH_SOURCES``
# on purpose so the two CANNOT drift: raising the batch size raises the estimate
# with it. 2 * 8 * 110 = 1760.
_SELECT_ANSWER_TOKENS = _SELECT_BATCH_SOURCES * _SEGMENTS_PER_SOURCE * _MAX_TOKENS_PER_SEGMENT
# The explicit per-call OUTPUT budget handed to WM.
#
# ⚠️ THIRD VALUE FOR THIS CONSTANT IN THREE DAYS. The mechanism is settled and
# #854 already states it correctly (thinking is charged against ``max_tokens``;
# the 422 is raised post-generation by ``pydantic_ai/_agent_graph.py:1100-1113``
# when every decoded part is a ``ThinkingPart``; the direction is UP). What kept
# moving was the NUMBER, because each version hardcoded its own ceiling from its
# own stale reading of the wall clock:
#
#   #851  1760  = the answer alone, zero headroom          -> 422 in prod
#   #854  7760  = answer + a 6000 literal named here       -> 37 tok/s was an
#                 idle-box figure for the RETIRED MoE model, and the 300 s
#                 router deadline it cites is overridden to 1200 in the cluster
#
# So this no longer carries a ceiling of its own. ``output_token_budget`` derives
# it from ``llm_budget_seconds()`` — the SAME per-call wall clock this stage
# hands to ``stage_call`` — times a measured token rate. Raising the clock in
# GitOps raises the token budget with it, and neither can drift from the other
# because there is only one number. Full derivation: ``stages/llm_budget.py``.
_SELECT_MAX_TOKENS = output_token_budget(answer_tokens=_SELECT_ANSWER_TOKENS)
# How much more than the target to ask for. The cut planner drops whatever does
# not fit the beat grid, so a selection sized exactly to the target can only come
# out short; it is capped at the footage that actually exists so a corpus smaller
# than the reel asks for "everything usable" rather than an impossible number.
_COVERAGE_FACTOR = 2.0
# How far apart two windows CUT FROM THE SAME CLIP have to be before they can be
# two different shots. Not a tolerance on the numbers — a statement about what an
# audience sees: an action-cam clip is one continuous camera move, so two windows
# taken from it less than this far apart show the same subject at nearly the same
# framing, and land in the reel as the same shot played twice.
#
# 1.5s is the shortest musical bar this product cuts to (four beats at 160 BPM,
# the top of the beat grid `cut_plan` solves against). Below one bar the footage
# in between cannot have carried the subject anywhere new, so no gap smaller than
# this can buy visual variety. Above it we stop guessing and let the model edit.
_NEAR_DUPLICATE_GAP_SECONDS = 1.5
# The floor the near-duplicate guard may not de-duplicate the selection below,
# as a multiple of the target reel length.
#
# It is deliberately NOT ``_COVERAGE_FACTOR``. Setting the floor to the same
# quantity the BRIEF asks the model for makes the guard INERT at its designed
# operating point: a model that obeys the brief returns ~2x the target, the
# unique windows on their own are then under 2x, and every duplicate is
# immediately re-admitted. The guard would only ever bite when the model
# OVER-delivered or when no length was requested — i.e. never, on the runs it
# was written for.
#
# 1.0x is what the cut planner actually needs, which is a different number from
# what the model is asked for. ``cut_plan`` walks a re-take FIFO (``_Take``), so
# shot COUNT is no longer bounded by segment count — the binding constraint is
# total approved FOOTAGE seconds. Its repair ladder trims an over-long segment
# to its beat window and plays it at 1.0x, so ~1 second of approved footage
# yields ~1 second of reel; ``target`` seconds of selection is therefore enough
# to fill a ``target`` second reel. The extra 1x asked for in the brief is slack
# for the PLANNER'S OWN CHOICE (skipping windows that fit the grid badly), not
# material the reel needs to exist — so it is exactly the margin dedupe is
# allowed to spend. Capped at the footage that exists, same as the brief.
_DEDUPE_COVERAGE_FLOOR_FACTOR = 1.0
# How much of what the REEL NEEDS a draw has to cover before the stage stops
# going back for more. NOTHING used to notice a short draw: ``_attempt``
# re-prompted only when the draw was EMPTY, so the live 10-clip run that returned
# 18.8s against an 80.0s brief passed here silently and surfaced one stage later
# as ``cut_plan.shortfall_seconds: 21.6`` — by which point the selection a
# reviewer would have re-run is buried under a plan built on top of it.
#
# ⚠️ This multiplies ``coverage_floor`` (``_DEDUPE_COVERAGE_FLOOR_FACTOR`` x
# target, capped at the footage that exists) — NOT ``requested_floor``, the ~2x
# the brief asks the model for. The previous version multiplied the ASK by 0.75,
# which on the real corpus this module is written about (10 clips, 97.1 s total,
# a 45 s reel) put the bar at 0.75 x min(90.0, 97.1) = 67.5 s: two thirds of ALL
# the footage in the run. A perfectly healthy 50 s draw — enough to fill the 45 s
# reel with slack to spare — fell under it, so the re-ask fired on essentially
# every real run and cost a whole extra pass, i.e. ``_SELECT_BATCH_CONCURRENCY``
# = 5 more worker-management calls against a slow, flaky vLLM, every time.
#
# ⚠️ CONFIRMED LIVE. A 10-clip / 45 s production run on 2026-08-20 (97.1 s of
# footage) came back with 45.4 s on the first draw and re-asked — correctly: the
# variety guard went on to drop 12 near-duplicates out of that draw, so 45.4 s of
# raw picks was NOT 45.4 s of usable reel, which is exactly the margin the 25%
# buys. The second draw returned 65.4 s and was kept. What the old yardstick then
# did was print, at a PAUSED review gate, "SHORT SELECTION: 65.4s of segments
# against the 90.0s the brief asked for ... cut_plan will report the remainder as
# shortfall" — and cut_plan went on to report `shortfall_seconds: 0.0` on a
# 47.08 s plan against a 45 s target. The note predicted a shortfall that did not
# happen. Under this constant the same run re-asks identically (45.4 < 56.25) and
# emits no note (65.4 >= 45.0), which is the truth about that run.
#
# 1.25 is the 1.0x the cut planner actually needs (see
# ``_DEDUPE_COVERAGE_FLOOR_FACTOR``: ~1 s of approved footage yields ~1 s of reel)
# plus 25% headroom for the material ``cut_plan`` will drop as near-duplicates
# before it ever reaches the grid. Under that the selection genuinely cannot
# cover the reel once dedupe has taken its cut, which is the failure worth a
# second paid call. Above it, the extra footage is planner SLACK — nice to have,
# never worth doubling the stage's cost to buy.
_REASK_COVERAGE_MULTIPLE = 1.25
# How much footage the model did NOT take has to be left before a re-ask can
# possibly help. The brief's floor is already capped at the footage that exists,
# so "is there more?" cannot be answered by comparing the floor to the available
# total — that comparison is true by construction. What a re-ask actually needs is
# UNTOUCHED material to build the missing seconds out of, and one full-length shot
# (``_MAX_SEGMENT_SECONDS``) is the smallest amount that could move the number at
# all. A three-clip corpus whose footage the model has already spent gets the note
# below and no second call: asking again for footage that does not exist is a paid
# request for the same answer.
_SHORTFALL_REASK_MIN_UNSELECTED_SECONDS = _MAX_SEGMENT_SECONDS

_SYSTEM_PROMPT = (
    "You are an action-sports video editor. You receive a set of raw source clips, "
    "each with its duration in seconds, the scene-cut times already present in the "
    "footage, and a motion curve (motion is 0..1 per timestamp; high motion is "
    "action, flat low motion is dead air). Pick the segments genuinely worth "
    "keeping: the jumps, carves, fly-throughs and reveals. For each, give the "
    "source_id it comes from, in_seconds and out_seconds (both inside that "
    "source's duration, out strictly greater than in), a score from 0 to 1 for how "
    "strong the moment is, and a one-sentence rationale a human reviewer can "
    # Spelled out rather than interpolated (an f-string here trips ruff's S608
    # on the word "select"), and pinned to the constants by a test: a prompt that
    # disagrees with `_validate` spends model effort on windows we then drop.
    "judge. Keep every segment at least 0.4 seconds and at most 12.0 seconds "
    "long. "
    "Draw from EVERY source clip you are given unless a clip is genuinely "
    "unusable — mining one clip while another goes untouched is a failed edit, "
    "not a strict one. When you are told a target reel length, select "
    "materially MORE total footage than that target — about twice it — because a "
    "later planner cuts the reel down to length from what you return and needs "
    "real choice; selecting only the target's worth guarantees a reel that runs "
    "short. This is still an edit, not a dump: keep the quality bar, rank the "
    "moments by score, and leave out dead air. "
    "VARIETY is part of the quality bar, not a nicety: consecutive picks should "
    "differ in subject, framing or motion character, and two segments that would "
    "read on screen as the SAME shot — the same subject held at the same framing, "
    "or two windows barely a second apart — are a defect; keep the stronger one. "
    "But variety is a rule about RESEMBLANCE, never a budget of one pick per "
    "clip. Aerial and action-camera footage is typically one continuous move for "
    "the whole clip, and its beginning, middle and end are still different shots: "
    "the subject, the framing and the speed have all changed by then. Take as "
    "many separated, non-overlapping windows from a single clip as that clip has "
    "distinct moments — a 20-second clip can easily yield three or four. If the "
    "variety rule and the coverage quota ever seem to conflict, THE QUOTA WINS: "
    "returning less footage than asked for guarantees a reel that runs short, "
    "which is a worse failure than two picks that rhyme. "
    "Never invent a source_id that was not given to you."
)

# Object-rooted structured-output schema (WM convention): WM returns
# ``response.output`` already parsed to this shape.
#
# ⚠️ ``rationale`` is in ``required`` on purpose, and it is the ONLY new constraint
# here — see the long note below for the one that was tried and reverted. Required
# PROPERTIES are expressible in a CFG, so guided decoding actually enforces them;
# an optional field is one the model can drop to save tokens, and this is the field
# the review gate shows the operator.
#
# ⚠️⚠️ DO NOT add ``minItems`` to ``segments``. It was added here on 2026-08-19 and
# reverted the same day, and the reasoning that produced it was seductive:
#
#   Measured against the REAL prompt and schema, thinking off, ``max_tokens=8000``:
#   ``{"segments": []}`` satisfied this schema in SIX completion tokens and the model
#   took that answer in 3 of 4 samples (2.7 s / 1.4 s / 12.5 s). The stage then raised
#   "no valid segments survived validation", the call was retried, and it surfaced as
#   ``Exceeded maximum output retries (3)``. Re-probed with ``minItems: 3`` the empty
#   draw vanished — 0 of 4 empty, 4 of 4 real draws.
#
# That probe went DIRECT to vLLM and is therefore the wrong path for this question.
# In production the call goes through worker-management, which STRIPS ``minItems``
# before building the grammar while still validating against the UNSTRIPPED schema —
# so the constraint is invisible to generation and mandatory at validation. That
# asymmetry is what killed every ``demo_video`` run at ``retention_plan:critic`` on
# 2026-07-20, and ``tests/test_output_schema_bounds.py`` bans the whole keyword family
# because of it. Read that test before touching this schema.
#
# The grammar floor is ``prefixItems``, which WM KEEPS: ``segments`` is a
# ``non_empty_array`` (``prefixItems: [ITEM]`` + ``items: ITEM``), so the CFG itself
# requires the first segment. Measured THROUGH worker-management on the script
# stages on 2026-10-06 (same WM + local vLLM/xgrammar path, ``disable_any_whitespace``):
# a plain ``{"type": "array", "items": ITEM}`` was steered from ``[`` straight to
# ``]`` — ``{"lines": []}`` on 6 of 6 calls — while the ``prefixItems`` shape
# returned full lists 2 of 2. Evidence: ``w2v/.proof-archive/2026-10-06/round4/
# r0-replay/``; see ``primitives.script_stage.non_empty_array``. (A 2026-08-20
# through-WM sample on THIS stage's batch prompt saw 0 empty draws in 12 — whether
# the grammar steers to ``]`` depends on the prompt, so the grammar must forbid it
# rather than rely on the prompt not triggering it.)
#
# The PARSER floor stays as the backstop for a provider without grammar
# enforcement: an all-empty draw raises the retryable "no valid segments survived
# validation" in ``_draw`` (ledger reset first, so ``stage_call`` re-prompts every
# batch) and the stage never ships an empty selection. Pinned through the real
# ``WMClient`` wire path in ``tests/.../test_segment_select_empty_draw_wire.py``.
_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "segments": non_empty_array(
            {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "in_seconds": {"type": "number"},
                    "out_seconds": {"type": "number"},
                    "score": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": ["source_id", "in_seconds", "out_seconds", "score", "rationale"],
            }
        )
    },
    "required": ["segments"],
}


class SegmentSelectInput(StageInput):
    """No typed inputs — the corpus is read from ``shot_analysis``."""


class SelectedSegment(StageOutput):
    """One candidate cut. ``id`` addresses it for annotations/regeneration; the
    cut planner keys its shots off it so a re-plan can be traced to a selection."""

    id: str
    source_id: str
    in_seconds: float
    out_seconds: float
    seconds: float
    score: float = 0.0
    rationale: str = ""


class SegmentSelectOutput(StageOutputWithNotes):
    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. Empty here means the stage
    # reported success having produced nothing, which degrades the run.
    # `cut_plan` builds the shot list out of `segments` — an empty selection is an
    # empty timeline.
    produces: ClassVar[tuple[str, ...]] = ("segments",)

    segments: list[SelectedSegment]
    #: How many segments the near-duplicate guard actually REMOVED. A structured
    #: field and not only a note, for the same reason ``cut_plan.shortfall_seconds``
    #: is one: a selection that got quietly smaller is the degradation a reviewer
    #: is most likely to skim past, because what is left still looks like a valid
    #: edit.
    near_duplicates_dropped: int = 0
    #: How many detected near-duplicates were KEPT ANYWAY because dropping them
    #: would have starved the cut planner (see ``_DEDUPE_COVERAGE_FLOOR_FACTOR``).
    #: The mirror of the field above, and it exists for the same reason
    #: ``cut_plan.overrun_seconds`` mirrors ``shortfall_seconds``: the drop count
    #: alone is one-sided. ``near_duplicates_dropped == 0`` used to be documented
    #: as "the model's own draw was already varied" while it was ALSO what a
    #: selection full of re-admitted duplicates reported — so a reviewer filtering
    #: on it got exactly the repeated-shot reel this guard exists to surface.
    #:
    #: Read them as a pair. ``(0, 0)`` is the only "no duplicates existed" reading;
    #: ``(0, n>0)`` means duplicates existed and every one of them is still in the
    #: reel. Their sum is how many the guard detected.
    near_duplicates_kept_for_coverage: int = 0


class SegmentSelectStage(StageHandler[SegmentSelectInput, SegmentSelectOutput]):
    name = "segment_select"
    input_model = SegmentSelectInput
    output_model = SegmentSelectOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"),)
    worker_name = "action-reel-segment-selector"

    def __init__(self, client: WMClient | None = None) -> None:
        self._llm_timeout = llm_budget_seconds()
        self._llm_retries = llm_retries()
        # NOT `self._llm_timeout`. That is the number this stage hands `stage_call` as its
        # PER-ATTEMPT budget below, and handing the same number to the HTTP client is the
        # measured defect (run 01a01d41-3147-70b4-afe9-9eb0beacb659: "all 2 attempts timed
        # out (per-attempt budget 950.0s)"). The stage clock starts first, so on a tie the
        # retryable httpx.ReadTimeout can never be raised. See llm_client_timeout_seconds.
        self._wm = client or WMClient(timeout=llm_client_timeout_seconds())

    async def handle(self, payload: SegmentSelectInput, ctx: RunContext) -> SegmentSelectOutput:
        sources, corpus_notes = _corpus_to_select_from(ctx.prior_output("shot_analysis") or {})
        durations = {str(s["id"]): float(s.get("duration_s") or 0.0) for s in sources}
        target = target_seconds(ctx)
        available = sum(durations.values())
        # What the cut planner needs to FILL the reel — NOT the ~2x the brief asks
        # the model for. See ``_DEDUPE_COVERAGE_FLOOR_FACTOR``: a floor set to the
        # brief's own number re-admits every duplicate the guard just found.
        coverage_floor = (
            0.0 if target is None else min(target * _DEDUPE_COVERAGE_FLOOR_FACTOR, available)
        )
        # What the BRIEF asks the model for (~2x the reel, capped at the footage
        # that exists) — the number a draw is measured against below. Deliberately
        # a different quantity from ``coverage_floor`` above, which is what the cut
        # planner needs to FILL the reel; see ``_DEDUPE_COVERAGE_FLOOR_FACTOR``.
        requested_floor = 0.0 if target is None else min(target * _COVERAGE_FACTOR, available)

        batches = [list(b) for b in batched(sources, _SELECT_BATCH_SOURCES, strict=False)]

        async def _draw(
            shortfall_hint: str | None = None,
            ledger: BatchLedger[list[dict], list[object]] | None = None,
        ) -> SegmentSelectOutput:
            """One draw = one pass over the corpus in bounded batches.

            The whole corpus used to go into ONE message. Both halves of that
            failed on the live 4-clip run: the model exhausted its output
            budget while reasoning and 422'd without emitting JSON (fixed by
            ``max_tokens=_SELECT_MAX_TOKENS``), and the prompt itself grows
            without bound as clips are added (fixed by batching). A token budget
            alone would still blow the INPUT side on a 10-clip corpus.

            The raw segments are accumulated across batches and handed to the
            EXISTING ``_validate`` / ``_drop_near_duplicates`` ladder ONCE, on the
            merged list — the guards are about the whole selection (the
            per-source duplicate check, the ``_MAX_SEGMENTS`` cap) and would be
            wrong applied per batch.
            """
            # A ledger is supplied only for the BASE draw — the one a retry
            # repeats verbatim. The re-ask draw builds a different prompt from
            # this attempt's own result, so resuming its batches across attempts
            # would pin answers to a hint that no longer applies.
            draw_ledger = ledger if ledger is not None else BatchLedger()

            async def _one_batch(index: int, batch: list[dict]) -> list[object]:
                """ONE model call. Raises exactly what the serial loop raised.

                Concurrency is bounded by the ledger's ``concurrency=`` argument
                now rather than by a semaphore held here — the same
                ``_SELECT_BATCH_CONCURRENCY`` ceiling, with one owner instead of
                two.
                """
                resp = await self._wm.message(
                    worker_name=self.worker_name,
                    message=self._build_message(
                        batch,
                        target,
                        shortfall_hint=shortfall_hint,
                        quota_seconds=_batch_quota(batch, requested_floor, available)
                        if len(batches) > 1
                        else None,
                        batch_label=(index, len(batches)) if len(batches) > 1 else None,
                    ),
                    system_prompt=_SYSTEM_PROMPT,
                    output_schema=_OUTPUT_SCHEMA,
                    # See _SELECT_MAX_TOKENS: omitting this leaves WM's 6000
                    # default, which a thinking model can burn before emitting
                    # any JSON. The budget must clear reasoning, not just the
                    # answer — passing too LITTLE is what kills the call.
                    max_tokens=_SELECT_MAX_TOKENS,
                    # Chain-of-thought OFF. Measured through worker-management on the
                    # production path (Job `wmprobe-segsel`, 2026-08-20), same prompt and
                    # schema, `max_tokens=8000`: with reasoning ON the call hit the stage's
                    # own 950s deadline and returned NOTHING -- twice, on both output
                    # schemas. With reasoning OFF the same call answered in 34-67s with
                    # real segments, every time. The model spends the whole budget thinking
                    # and never emits the answer, which surfaces one layer up as
                    # `Exceeded maximum output retries (3)`.
                    #
                    # This is only sendable because worker-management #188 added the field
                    # and `w2v_common` 0.61.0 can carry it -- `MessageRequest` is
                    # `extra="forbid"`, so sending it to an older WM would 422 the call.
                    # Omit it and w2v's calls take WM's in-memory auto-create branch
                    # (`model_provider="vllm"`), which goes DIRECT to vLLM and never
                    # touches inference-router -- the only component that would otherwise
                    # turn thinking off. See memory `w2v-wm-calls-bypass-inference-router`.
                    enable_reasoning=False,
                )
                data = resp.output
                if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
                    # Retryable through stage_call — re-prompt for a conforming object.
                    raise ValueError("segment_select: WM output missing segments")
                return list(data["segments"])

            # BOUNDED-PARALLEL, and reassembled in BATCH ORDER regardless of which
            # call finished first (see ``_SELECT_BATCH_CONCURRENCY``). Order is the
            # edit here: ``_merge_batch_draws``'s sort is stable, so completion
            # order would otherwise decide the reel's order within a tied score.
            #
            # ``return_exceptions=True`` so a failing batch cannot leave its
            # siblings as orphaned tasks whose exceptions surface later against a
            # dead run — the FIRST failure in batch order is then re-raised
            # unchanged, which is the same exception, from the same batch, that the
            # serial loop raised into ``stage_call``'s retry budget. The one honest
            # difference: the siblings were already in flight, so they are paid for
            # rather than skipped. With one wave covering the corpus that was true
            # of the serial loop's in-flight call too, only for fewer calls.
            # Through the LEDGER, so batches that already answered are not
            # re-issued by a retry. ``per_batch`` used to be local to this
            # function: attempt 1 could pay for k of N batches, time out, and
            # attempt 2 started again at batch 0 — re-paying k WM calls it
            # already had answers for and reaching the same place. A corpus
            # whose N batches did not fit inside ONE attempt could therefore
            # never complete, however many retries it was given. This stage is
            # the one that has been timing out in production.
            #
            # Batch boundaries are preserved, not flattened: a score is only
            # comparable to the other scores from the SAME model call, so which
            # batch an entry came from is load-bearing for
            # ``_merge_batch_draws``. Order is by batch index, not completion
            # order, for the same reason as before.
            parts = await draw_ledger.run_parallel(
                batches,
                # ``_one_batch`` labels prompts 1-based; the ledger indexes 0-based.
                lambda index, batch: _one_batch(index + 1, batch),
                concurrency=_SELECT_BATCH_CONCURRENCY,
            )
            per_batch: list[list[object]] = [list(part) for part in parts]
            segments, notes = _validate(_merge_batch_draws(per_batch), durations)
            if not segments:
                # An empty draw is retryable too: the model saw real footage and
                # returned nothing usable, which a re-prompt often fixes. Emitting
                # zero segments would instead hand cut_plan an empty timeline and
                # produce a "successful" run holding no video.
                #
                # ⚠️ CLEAR THE LEDGER FIRST, or the re-prompt above is a LIE. Every
                # batch here RETURNED — ``{"segments": []}`` is a successful call,
                # and the measured behaviour of the real model on the real prompt
                # (see the schema note near the top of this file) — so the ledger
                # banked all N of them. Without this, the retry's ``run_parallel``
                # finds nothing pending, issues ZERO WM calls, replays the same
                # empty answers into ``_validate``, and re-raises this identical
                # error until the budget runs out: a PAID stage failing
                # deterministically with no second look at the footage.
                #
                # Only THIS path resets. The ledger's purpose — resuming after a
                # partial or timed-out draw so answered batches are not re-bought —
                # is untouched on every other path, because this is the one case
                # where the whole draw produced nothing usable, which is precisely
                # the case the re-prompt exists for.
                draw_ledger.reset()
                raise ValueError("segment_select: no valid segments survived validation")
            varied = _drop_near_duplicates(segments, coverage_floor)
            return SegmentSelectOutput(
                segments=varied.segments,
                near_duplicates_dropped=varied.dropped,
                near_duplicates_kept_for_coverage=varied.kept_for_coverage,
                notes=notes + varied.notes,
            )

        # OUTSIDE the attempt factory, so a retry RESUMES the base draw instead
        # of re-paying it. Inside would be the bug; on ``self`` would be worse —
        # a second run of this stage would inherit the first run's answers.
        base_ledger: BatchLedger[list[dict], list[object]] = BatchLedger()

        async def _attempt() -> SegmentSelectOutput:
            best = await _draw(ledger=base_ledger)
            selected = _total_seconds(best.segments)
            # Judged against ``coverage_floor`` — what the reel NEEDS — and never
            # against ``requested_floor``, the 2x the brief asks for. The re-ask
            # itself still quotes the brief's number, because that is the
            # instruction the model was given; only the DECISION to spend a second
            # pass is made on the need. See ``_REASK_COVERAGE_MULTIPLE``.
            if _should_reask(selected, coverage_floor, available):
                # AT MOST ONE extra call per attempt. This is a PAID stage, and the
                # only thing that makes a second call defensible is that the first
                # one demonstrably under-filled a brief the corpus CAN fill.
                best, selected = await _reask_once(
                    _draw, best, selected, requested_floor, available
                )
            best.notes = [*best.notes, *_short_selection_note(selected, coverage_floor, available)]
            return best

        selection = await stage_call(
            lambda: guarded(_attempt),
            timeout_seconds=self._llm_timeout,
            retries=self._llm_retries,
            retry_on=LLM_RETRY_ON,
            stage_name=self.name,
        )
        # FIRST, and outside the retry: these describe the CORPUS this stage was
        # handed, not the draw it got back, so they must survive every attempt and
        # be the first thing the reviewer reads at the pause. A thin reel whose
        # cause is "most of the footage was never measured" is unreadable from the
        # segments alone.
        selection.notes = [*corpus_notes, *selection.notes]
        return selection

    def _build_message(
        self,
        sources: list[dict],
        target: float | None = None,
        *,
        shortfall_hint: str | None = None,
        quota_seconds: float | None = None,
        batch_label: tuple[int, int] | None = None,
    ) -> str:
        """One JSON object per source: identity + duration + the two curves.

        The motion curve is DOWNSAMPLED (see ``_MOTION_SAMPLES_PER_SOURCE``) —
        the analyzer emits one sample per analyzed frame, and handing a small
        vLLM thousands of floats is how the call times out and the stage silently
        degrades to no segments.

        The BRIEF line is what stops the "few strong segments" instinct from
        under-filling the reel: without a stated target the model has no way to
        know that 13.5 s of drone footage is being cut to 20 s, and 2.4 s of
        selection looks like good taste rather than the shortfall it is. The
        available total is stated alongside it so an honestly-short corpus reads
        as "take nearly everything" instead of an impossible quota.

        ``shortfall_hint`` is the ONE corrective re-ask (see
        ``_REASK_COVERAGE_MULTIPLE``): the same brief with the previous draw's
        own numbers appended, so the model is told what it actually returned
        rather than asked the identical question twice. ``None`` — every first
        call, and every call on a run with no requested length — must produce a
        byte-identical message to the one this stage has always sent; a test pins
        that, because the brief text and the constants are agreed by hand.

        ``quota_seconds`` / ``batch_label`` carry BATCHING (see
        ``_SELECT_BATCH_SOURCES``). ``sources`` is then one batch, not the corpus,
        so the footage total and clip count stated here are the batch's — the
        model can only draw from what it was shown, and stating the whole
        corpus's numbers against a two-clip prompt is the "brief is fiction"
        failure the cap on the ask exists to avoid. The reel length stays the REAL
        reel length and the batch line says which slice of the corpus this is;
        the quota is that batch's proportional share, so the batch quotas sum to
        the whole ask. Both default to ``None`` — a corpus that fits in ONE batch
        must still produce the byte-identical message above.
        """
        blocks = []
        for src in sources:
            block: dict[str, object] = {
                "source_id": str(src.get("id") or ""),
                "duration_s": round(float(src.get("duration_s") or 0.0), 2),
            }
            if src.get("error"):
                # ``"unmeasured": true`` and the measurement keys OMITTED — never
                # zeros. A quality of 0.0 beside an empty motion curve is what a
                # measured, genuinely dead clip looks like, so writing that for a
                # clip nobody measured tells the model the exact opposite of the
                # truth and guarantees it is passed over.
                block["unmeasured"] = True
            else:
                block["scene_cuts"] = _downsample_scene_cuts(src.get("scene_boundaries") or [])
                block["motion"] = _downsample_motion(src.get("motion_timeline") or [])
                block["quality"] = round(float(src.get("overall_score") or 0.0), 3)
            blocks.append(json.dumps(block, ensure_ascii=False))
        header = "SOURCE CLIPS (one JSON object per clip):\n" + "\n".join(blocks)
        if any(src.get("error") for src in sources):
            # Only when at least one is unmeasured: a fully measured corpus must
            # still send the byte-identical message it has always sent.
            header = (
                f'{header}\n\nSome clips above are marked "unmeasured": true. The '
                "analyzer could not measure those, so they carry no motion curve, no "
                "scene cuts and no quality score — that is MISSING INFORMATION, not a "
                "score of zero and not a verdict on the footage. Treat them as ordinary "
                "candidates judged on their duration, and take windows from them the "
                "same way you would from a measured clip; skipping them because they "
                "are unmeasured is how a ten-clip shoot becomes a two-clip reel."
            )
        if batch_label is not None:
            index, total = batch_label
            header = (
                f"{header}\n\nThis is batch {index} of {total} of this run's source clips. "
                # Deliberately NOT the word "select" inside an f-string — it trips
                # ruff's S608 (SQL-injection) heuristic; see _SYSTEM_PROMPT.
                "Draw ONLY from the clips above — the other batches are being edited "
                "separately and the picks are merged afterwards, so the length asked "
                "for below is this batch's share of the reel, not the whole reel."
            )
        if target is None:
            # No length was requested anywhere, so there is no quota to state and
            # nothing to over-select against — the brief line would be an invented
            # number, which is the one thing the resolver refuses to do.
            return header if shortfall_hint is None else f"{header}\n\n{shortfall_hint}"
        available = sum(float(src.get("duration_s") or 0.0) for src in sources)
        ask = min(target * _COVERAGE_FACTOR if quota_seconds is None else quota_seconds, available)
        brief = (
            f"{header}\n\nBRIEF: the finished reel must run {target:.1f} seconds. "
            f"There are {available:.1f} seconds of source footage across "
            f"{len(sources)} clip(s). Select at least "
            f"{ask:.1f} seconds of segments in "
            "total, spread across all of the clips, so the cut planner has room to "
            "choose."
        )
        return brief if shortfall_hint is None else f"{brief}\n\n{shortfall_hint}"


def _corpus_to_select_from(analysis: dict) -> tuple[list[dict], list[str]]:
    """The sources this stage may select from, plus what to tell the reviewer.

    An analyzer failure is MISSING MEASUREMENT, not a missing clip. Dropping
    errored sources here deleted the operator's own footage from the corpus: 10
    clips uploaded, 8 analyzer timeouts, a reel cut from 2 — five green gates,
    ``unfetchable_segments`` and ``unplaced_segments`` both 0, and nothing
    anywhere saying why the reel was thin. An unmeasured source still carries the
    one thing selection cannot do without, its DURATION (see
    ``shot_analysis.AnalyzedSource``, and that module's docstring: "that source
    simply carries no measurements — the selector can still use it on duration
    alone").

    The ONE source that genuinely cannot be offered is an unmeasured one with no
    known duration: every window named against it clamps to [0, 0] and dies in
    ``_validate`` anyway, so putting it in the prompt buys paid tokens for a
    candidate that cannot survive. Those are COUNTED, never silently forgotten.
    """
    corpus = [s for s in (analysis.get("sources") or []) if isinstance(s, dict)]
    if not corpus:
        # NOT a skip: selection is the product.
        raise StagePreconditionError(
            "segment_select: shot_analysis emitted no sources to select from",
            missing=["shot_analysis.sources"],
        )
    if all(s.get("error") for s in corpus):
        # The invariant ``shot_analysis`` already enforces at its own exit, held
        # again here because this stage can also run against a REPLAYED prior
        # output. A selection made with zero measured signal is a coin toss
        # wearing a rationale — duration-only is honest for SOME of the corpus and
        # meaningless for ALL of it.
        raise StagePreconditionError(
            "segment_select: every source is unmeasured; no measured signal to select from",
            missing=["shot_analysis.sources"],
        )
    offered = [s for s in corpus if not s.get("error") or float(s.get("duration_s") or 0.0) > 0.0]
    return offered, _unmeasured_corpus_notes(corpus, offered)


def _unmeasured_corpus_notes(corpus: list[dict], offered: list[dict]) -> list[str]:
    """Say how much of the operator's footage reached the editor UNMEASURED.

    The degradation this stage is most likely to hide. The segments that come
    back are individually valid whatever the corpus looked like, so a reel cut
    from two of ten clips reads at the review gate exactly like a reel cut from
    two clips — ``unfetchable_segments`` and ``unplaced_segments`` are both 0,
    every gate is green, and the thin reel has no stated cause. Same posture as
    ``_short_selection_note``: the numbers, and what a reviewer can do with them.
    """
    unmeasured = [s for s in corpus if s.get("error")]
    if not unmeasured:
        return []
    offered_ids = {str(s.get("id") or "") for s in offered}
    dropped = [s for s in unmeasured if str(s.get("id") or "") not in offered_ids]
    notes = [
        f"{len(unmeasured)} of {len(corpus)} sources could not be measured by the analyzer; "
        "they were offered to the editor on duration alone, with no motion curve, scene cuts "
        "or quality score — review whether the reel is thin because the footage is thin or "
        "because the analyzer was down"
    ]
    if dropped:
        notes.append(
            f"{len(dropped)} of those unmeasured sources also carry no known duration and could "
            "not be offered at all — nothing can be cut from a clip of unknown length; re-run "
            "shot_analysis or re-upload them"
        )
    return notes


def _batch_quota(
    batch: list[dict], requested_floor: float, available_seconds: float
) -> float | None:
    """This batch's proportional share of the whole corpus's ask.

    Scaling by FOOTAGE and not by clip count is what makes the per-batch quotas
    sum back to ``requested_floor``: a batch holding a 60 s clip and a 3 s one
    can honestly be asked for more than a batch of two 4 s clips, and asking each
    batch for the whole reel's worth would over-select by the batch count.
    """
    if requested_floor <= 0.0 or available_seconds <= 0.0:
        return None
    batch_available = sum(float(src.get("duration_s") or 0.0) for src in batch)
    return requested_floor * (batch_available / available_seconds)


def _merge_batch_draws(per_batch: list[list[object]]) -> list[object]:
    """Merge the per-batch draws into ONE ordered candidate list.

    Order is the EDIT here (see the module docstring): ``cut_plan`` reads a
    ``source_id`` change between adjacent segments as a section change, AND it
    walks this list in order and stops once the target is met — so anything left
    in the tail is never rendered. Plain concatenation would group every batch's
    segments together, clumping each clip's picks into one run of the reel.

    **Sorted by score DESC — over scores NORMALISED PER BATCH first.** The sort
    was always right; what was broken is that it sorted numbers that are not
    comparable. Each batch is an INDEPENDENT model call
    (``_SELECT_BATCH_SOURCES`` clips per call, up to
    ``_SELECT_BATCH_CONCURRENCY`` in flight) which scores the material it was
    shown on its own scale, with no knowledge of the other batches. Measured on
    the live 10-clip / 45 s run of 2026-08-20, the raw scores land in DISJOINT
    per-batch bands — 0.65/0.55, 0.55, 0.55/0.45, 0.50/0.45, 0.42/0.36/0.30,
    0.35/0.30/0.26, 0.35, 0.30, 0.30 — so a global raw sort hands the head of the
    list to whichever call happened to self-score generously and leaves the modest
    ones in a tail ``cut_plan`` never reaches. Whole UPLOADED CLIPS go missing from
    the reel on a scoring quirk rather than on an edit, with every gate green.

    Min-max per batch: ``(score - batch_min) / (batch_max - batch_min)``. Every
    batch's best becomes 1.0 and therefore ties at the top of the merged list, so
    every call is represented in the head whatever band it scored in — which is
    the whole defect. Min-max rather than RANK-within-batch on purpose: rank
    throws away the batch's own spacing, so a batch that said "one standout, then
    three also-rans" is flattened into the same evenly-spread ladder as a batch
    that liked everything equally, and with the batches all the same size rank
    degenerates into exactly the round-robin rejected below. Min-max keeps the
    shape the model expressed and only relocates the scale.

    A batch whose scores are ALL EQUAL (including a single-entry batch) has a zero
    span and is mapped entirely to 1.0 — never a division. That is the honest
    reading: such a call expressed no preference between its own picks, so every
    one of them is equally its best, and the tie is then broken by ``sorted``'s
    stability, i.e. by batch order and by the model's own order inside a batch.

    **Round-robin was considered and REJECTED.** It fixes the incomparability just
    as well, but it also changes the reel: ``cut_plan.py:1101`` charges
    ``if self.section_change and (self.index + span) not in self.downbeat_indices:
    value += 0.12``, and a mechanically interleaved list makes ``section_change``
    true on nearly every shot instead of occasionally, biasing span choice and
    beat phase on EVERY shot. The docstring this replaced was right about that
    mechanism and wrong about its magnitude: interleaving drops nothing and
    re-orders nothing downstream, so the harm is far smaller than "moves the
    downbeat preference of every reel" implied. It is still an editorial change,
    and an editorial change has no business riding along inside a correctness fix.
    Normalisation removes the incomparability and touches beat phase not at all.

    ``sorted`` is stable, so within one normalised score the accumulation order —
    batch order, and inside a batch the model's own order, the only order it
    actually expressed — survives untouched.

    A corpus that fitted in ONE batch is returned UNTOUCHED. There is no
    cross-batch scale to reconcile there, and the single model call DID express an
    edit over the whole corpus; re-scoring it would silently overrule the model on
    every run that never needed batching.
    """
    if len(per_batch) <= 1:
        return per_batch[0] if per_batch else []
    scaled: list[tuple[float, object]] = []
    for batch in per_batch:
        scores = [_entry_score(entry) for entry in batch]
        if not scores:
            continue
        bottom, top = min(scores), max(scores)
        span = top - bottom
        scaled.extend(
            (1.0 if span <= 0.0 else (score - bottom) / span, entry)
            for entry, score in zip(batch, scores, strict=True)
        )
    # `list.sort` is stable and the key reads ONLY the normalised score, so no two
    # opaque model entries are ever compared to each other.
    scaled.sort(key=lambda pair: -pair[0])
    return [entry for _, entry in scaled]


def _entry_score(entry: object) -> float:
    """The 0..1 score off one raw model entry, before ``_validate`` has run.

    A malformed entry (not an object) scores 0.0 rather than raising: it is
    ``_validate``'s job to drop it and to say so in the notes, and a merge that
    threw here would lose the whole draw over one bad element.
    """
    if not isinstance(entry, dict):
        return 0.0
    # Same cast as ``_validate``: isinstance-narrowing an ``object`` yields
    # ``dict[Unknown, Unknown]``, so the string-keyed read needs the opaque
    # LLM-segment shape spelled out to type-check.
    return _as_float(cast("dict[str, object]", entry).get("score"))


def _total_seconds(segments: list[SelectedSegment]) -> float:
    """How much footage a draw actually selected. The number the brief is about —
    segment COUNT says nothing, since ten 1-second picks and one 12-second pick
    are the same count of very different edits."""
    return sum(seg.seconds for seg in segments)


def _should_reask(selected_seconds: float, coverage_floor: float, available_seconds: float) -> bool:
    """Is one corrective re-ask worth a second PAID model call?

    Three conditions, all of them required (see ``_REASK_COVERAGE_MULTIPLE``):

    * a length was requested at all — with no target there is no floor and
      therefore no such thing as a short draw, only an edit;
    * the draw cannot cover THE REEL with dedupe headroom. ``coverage_floor`` is
      what the cut planner needs to fill the reel, capped at the footage that
      exists — emphatically not ``requested_floor``, the ~2x the brief asks for.
      Measuring the answer against the ask made this fire on a draw that filled
      the reel comfortably, on essentially every real run;
    * there is untouched footage left to build the missing seconds out of. The
      floor is already capped at the corpus, so "does more footage exist" cannot
      be asked by comparing the two — it is asked of what the model did NOT take.
      A corpus that genuinely cannot fill the reel gets the honest note and one
      call, never a loop.
    """
    if coverage_floor <= 0.0:
        return False
    if selected_seconds >= coverage_floor * _REASK_COVERAGE_MULTIPLE:
        return False
    return available_seconds - selected_seconds >= _SHORTFALL_REASK_MIN_UNSELECTED_SECONDS


async def _reask_once(
    draw: Callable[[str], Awaitable[SegmentSelectOutput]],
    first: SegmentSelectOutput,
    first_seconds: float,
    requested_floor: float,
    available_seconds: float,
) -> tuple[SegmentSelectOutput, float]:
    """Go back to the editor ONCE with the numbers, and keep the LARGER draw.

    "Larger" is the whole point. A model told to add material can lose some
    instead, and a guard that let the second answer overwrite the first would
    hand the run LESS footage than it had before the guard existed — a
    regression wearing a fix's clothes.

    A corrective call that itself fails is not a stage failure: we already hold a
    valid, merely short selection, so a transient WM failure degrades to it
    rather than discarding the paid first draw and re-running the whole attempt.
    ``PermanentUpstreamError`` is deliberately absent from ``LLM_RETRY_ON`` and
    still fails the stage loudly — a 402 is not a short draw.

    Either way the notes say what happened: which draw was kept is exactly the
    kind of decision that is invisible in the output it produces.
    """
    second: SegmentSelectOutput | None = None
    try:
        second = await draw(_shortfall_hint(first_seconds, requested_floor, available_seconds))
    except LLM_RETRY_ON:
        second = None
    kept, kept_seconds, took_second = first, first_seconds, False
    second_label = "nothing usable"
    if second is not None:
        second_seconds = _total_seconds(second.segments)
        second_label = f"{second_seconds:.1f}s"
        if second_seconds > first_seconds:
            kept, kept_seconds, took_second = second, second_seconds, True
    kept.notes = [
        *kept.notes,
        f"re-asked the editor once: the first draw returned {first_seconds:.1f}s against the "
        f"{requested_floor:.1f}s the brief asked for; the second returned {second_label} and "
        f"the {'second' if took_second else 'first'} draw was kept",
    ]
    return kept, kept_seconds


def _short_selection_note(
    selected_seconds: float, coverage_floor: float, available_seconds: float
) -> list[str]:
    """Say it out loud when the selection cannot fill THE REEL.

    THE mechanism the live defect was missing. 18.8 s against an 80 s reel
    reached ``cut_plan`` as a normal-looking edit and only became visible there,
    as ``shortfall_seconds``, on top of a plan already built from it. Same shape
    as the near-duplicate notes: the numbers, and what a reviewer can do about
    them.

    ⚠️ The number quoted is ``coverage_floor`` — the seconds the reel NEEDS,
    capped at the footage that exists — and never ``requested_floor``, the ~2x the
    brief asks the model for. This note is what a HUMAN READS AT THE PAUSE
    (``segment_select`` has ``default_pause_after=True``), and quoting the ask
    made it lie in the direction that wastes a reviewer's time: a 50.0 s draw that
    fills a 45 s reel with room to spare was reported as "SHORT SELECTION: 50.0s
    ... against the 90.0s the brief asked for". A note that cries wolf on a
    healthy edit is worse than no note, because the reviewer learns to skip it —
    and this is the one note that has to be believed. It now fires only when the
    reel genuinely cannot be filled, and says which way out exists.
    """
    if coverage_floor <= 0.0 or selected_seconds >= coverage_floor:
        return []
    # A corpus SMALLER than the reel is a different problem from a thin draw off a
    # big corpus, and the reviewer's next move differs: shoot/upload more footage
    # vs. re-run the selection. The floor is capped at ``available_seconds``, so
    # the two cases are distinguishable right here.
    need = (
        "the reel needs" if coverage_floor < available_seconds else "all the footage this run has"
    )
    return [
        f"SHORT SELECTION: {selected_seconds:.1f}s of segments against the "
        f"{coverage_floor:.1f}s {need} to fill it, out of {available_seconds:.1f}s of "
        "available footage — cut_plan will report the remainder as shortfall; review "
        "whether more of the footage is usable, re-run the selection, or shorten the reel"
    ]


def _shortfall_hint(
    selected_seconds: float, requested_floor: float, available_seconds: float
) -> str:
    """The corrective paragraph appended to the brief on the ONE re-ask.

    It names all three numbers on purpose. "Select more" is the instruction the
    first call already carried and did not land; what the model is missing is that
    its own answer was small against the ask, and that the footage to fix it is
    sitting in the clips it drew one window from.
    """
    return (
        f"CORRECTION — your previous answer was too short. You returned "
        f"{selected_seconds:.1f}s of segments in total; the brief needs at least "
        f"{requested_floor:.1f}s and there are {available_seconds:.1f}s of footage "
        "available. Go back to the LONGER clips and take more separated, "
        "non-overlapping windows from each of them — the beginning, middle and end of "
        "one continuous camera move are different shots. Return the COMPLETE "
        "selection again, not only the extra windows."
    )


def _even_indices(count: int, cap: int) -> list[int]:
    """``cap`` indices spanning ``range(count)`` end to end, first and LAST included.

    Index arithmetic rather than a stride, because a stride cannot do this. The
    stride these callers used was ``count // cap``, which is 1 for every ``count``
    in ``[cap + 1, 2 * cap - 1]`` — so ``points[::1][:cap]`` returned the FIRST
    ``cap`` items and dropped everything after them, the exact head-truncation
    both docstrings promised not to do. That band has never been reached in
    production (see ``_downsample_motion``); this is a latent contract bug, not a
    live one. ``round(i * (count - 1) / (cap - 1))``
    always lands on ``count - 1`` at ``i == cap - 1``, so the end of the clip —
    often a drone shot's best moment, and previously unreachable — is always shown.
    """
    if cap <= 0 or count <= 0:
        return []
    if count <= cap:
        return list(range(count))
    if cap == 1:
        return [count - 1]
    return [round(i * (count - 1) / (cap - 1)) for i in range(cap)]


def _downsample_motion(timeline: list) -> list[list[float]]:
    """Evenly thin the motion curve to at most ``_MOTION_SAMPLES_PER_SOURCE``
    ``[t, motion]`` pairs, RESAMPLED across the whole clip with the final sample
    always kept (see ``_even_indices``). Truncating would hide every moment after
    the first seconds, which on a drone clip is most of the footage.

    This used to stride by ``len // cap``, which for any length in
    ``[cap + 1, 2 * cap - 1]`` (25-47 here) gave a stride of 1 and therefore the
    first 24 points only — head truncation, in the band where the code
    contradicted this docstring. Worst case (len 47) the model saw 51% of the clip
    beside the full ``duration_s``.

    LATENT, never fired: measured over 23 ``shot_analysis`` rows / 147 real source
    clips, motion points per clip run min 4, median 4, max 5 — every one at or
    under the cap, none in the 25-47 band. Reaching it needs ~78-144 s source
    clips and the longest real clip is 18.02 s. Fixed because the code disagreed
    with its own contract and the fix is cheap, not because a user is affected.
    """
    points = [p for p in timeline if isinstance(p, dict)]
    if not points:
        return []
    return [
        [
            round(float(points[i].get("t") or 0.0), 2),
            round(float(points[i].get("motion") or 0.0), 3),
        ]
        for i in _even_indices(len(points), _MOTION_SAMPLES_PER_SOURCE)
    ]


def _downsample_scene_cuts(boundaries: list) -> list[float]:
    """Evenly thin the analyzer's cut list to at most ``_SCENE_CUTS_PER_SOURCE``
    times. Unbounded prompt input otherwise: a long, busy clip carries hundreds
    of boundaries straight into the message, beside a motion curve that was
    already bounded for exactly this reason. Resampled across the whole clip with
    the LAST cut always kept (see ``_even_indices``), so the times describe the
    whole clip rather than only its first seconds.

    The previous ``len // cap`` stride was 1 for any length in
    ``[cap + 1, 2 * cap - 1]`` (25-47 here) and so returned the first 24 cuts —
    head truncation, the thing this docstring already said it did not do. Latent
    like the motion curve above: production maxes out at 6 boundaries per clip,
    nowhere near the band."""
    times = [round(_as_float(t), 2) for t in boundaries]
    if not times:
        return []
    picked: list[float] = []
    seen: set[float] = set()
    for i in _even_indices(len(times), _SCENE_CUTS_PER_SOURCE):
        moment = times[i]
        # Two resampled indices can round to the same 2dp time on a dense cut
        # list. De-duplicated in WALK ORDER (not via `sorted(set(...))`, which
        # would re-sort a list whose order is the clip's chronology) and NOT
        # padded back up: a cut list is a set of moments, and inventing a
        # neighbouring index to reach exactly `cap` would report a boundary the
        # analyzer never found.
        if moment not in seen:
            seen.add(moment)
            picked.append(moment)
    return picked


def _validate(
    raw: list[object], durations: dict[str, float]
) -> tuple[list[SelectedSegment], list[str]]:
    """Keep only segments naming a real source with a usable, in-bounds window.

    Returns ``(segments, notes)`` — the notes record what was dropped and why, so
    the paused review shows the reviewer that the model over-reached rather than
    silently presenting a shorter list.
    """
    out: list[SelectedSegment] = []
    notes: list[str] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        # isinstance-narrowing an `object` element yields dict[Unknown, Unknown];
        # cast to the opaque LLM-segment shape so the string-keyed reads type-check.
        seg = cast("dict[str, object]", entry)
        source_id = str(seg.get("source_id") or "")
        duration = durations.get(source_id)
        if duration is None:
            notes.append(f"dropped segment for unknown source_id {source_id!r}")
            continue
        window = _clamp_window(seg, duration)
        if window is None:
            notes.append(f"dropped unusable window on source {source_id}")
            continue
        if len(out) >= _MAX_SEGMENTS:
            notes.append(f"capped at {_MAX_SEGMENTS} segments; later candidates dropped")
            break
        start, end = window
        out.append(
            SelectedSegment(
                id=str(uuid7()),
                source_id=source_id,
                in_seconds=round(start, 3),
                out_seconds=round(end, 3),
                seconds=round(end - start, 3),
                score=_clamp_unit(seg.get("score")),
                rationale=str(seg.get("rationale") or ""),
            )
        )
    return out, notes


@dataclass(frozen=True)
class _Deduped:
    """The variety guard's verdict. A record and not a tuple because the two
    counts are only honest read together (see ``SegmentSelectOutput``)."""

    segments: list[SelectedSegment]
    dropped: int
    kept_for_coverage: int
    notes: list[str]


def _admission_rank(indexed: tuple[int, SelectedSegment]) -> tuple[float, float, str, int]:
    """The total order the guard resolves collisions in. Explicit, and total.

    Order is the EDIT on a beat-cut product, so nothing here may depend on dict
    iteration or on the order the model happened to emit its draw in. The keys,
    in order: score DESC (the stronger window survives a collision), then
    ``in_seconds`` ASC, then ``source_id``, then the model's own index as the
    last resort. The first three already separate any two segments that differ
    as an EDIT; the index only breaks a tie between two segments naming the same
    window of the same source at the same score, where either choice is by
    definition the same footage.
    """
    index, seg = indexed
    return (-seg.score, seg.in_seconds, seg.source_id, index)


def _drop_near_duplicates(
    segments: list[SelectedSegment], coverage_floor_seconds: float
) -> _Deduped:
    """Remove segments that would land in the reel as the same shot twice.

    The model is advisory here as everywhere else in this stage: on the live
    10-clip run it returned several near-identical static windows of one kiosk,
    every one of them individually valid, and nothing downstream compares two
    windows to each other — ``cut_plan`` places what it is given.

    Two segments collide when they come from the SAME source and their windows
    overlap or sit within ``_NEAR_DUPLICATE_GAP_SECONDS`` of each other. Windows
    from DIFFERENT sources never collide however close their timestamps are:
    two clips shot at the same second of their own footage are two cameras, which
    is the variety we are trying to buy, not duplication.

    **This function decides what to DROP. It does not re-order the edit.** The
    survivors are emitted in the order the model returned them. An earlier
    version interleaved them round-robin across sources, which looked like free
    variety and was not: ``cut_plan`` derives ``section_change`` from
    "does the NEXT shot change source", and a perfectly interleaved list makes
    that true on nearly every shot instead of occasionally, moving the downbeat
    preference — i.e. the beat placement — of every action reel. Which segments
    reach the planner is this stage's decision; what order the reel plays them in
    is the model's edit and the reviewer's to change.

    Never trims silently: both counts are structured fields and each drop is
    named in the notes, mirroring ``cut_plan.shortfall_seconds`` /
    ``overrun_seconds`` — a selection that quietly got smaller is exactly the
    degradation a reviewer skims past, because what remains still looks like a
    valid edit.

    ``coverage_floor_seconds`` is what the cut planner needs to FILL the reel
    (``_DEDUPE_COVERAGE_FLOOR_FACTOR`` x target, capped at the footage that
    exists) — deliberately not the ~2x the BRIEF asks the model for, which would
    re-admit every duplicate the guard just found. Starving the planner trades
    one visible defect for another — a reel that runs short — so when
    de-duplicating would take the selection under that floor the higher-scoring
    duplicates are re-admitted, loudly, until it is met again.
    """
    ranked = sorted(enumerate(segments), key=_admission_rank)
    kept: list[SelectedSegment] = []
    collisions: list[tuple[SelectedSegment, SelectedSegment]] = []
    for _, seg in ranked:
        twin = next((k for k in kept if _reads_as_the_same_shot(k, seg)), None)
        if twin is None:
            kept.append(seg)
        else:
            collisions.append((seg, twin))
    if not collisions:
        return _Deduped(segments=list(segments), dropped=0, kept_for_coverage=0, notes=[])

    kept_seconds = sum(seg.seconds for seg in kept)
    unique_seconds = kept_seconds
    notes: list[str] = []
    dropped = 0
    readmitted = 0
    # `collisions` is already in `_admission_rank` order, so the duplicate a
    # starved floor gets back is the strongest one available — deterministically,
    # not "whichever the walk reached first".
    for duplicate, twin in collisions:
        if kept_seconds < coverage_floor_seconds:
            kept.append(duplicate)
            readmitted += 1
            kept_seconds += duplicate.seconds
            continue
        dropped += 1
        notes.append(
            f"dropped near-duplicate segment on source {duplicate.source_id} "
            f"({duplicate.in_seconds:.2f}s-{duplicate.out_seconds:.2f}s reads as the same "
            f"shot as {twin.in_seconds:.2f}s-{twin.out_seconds:.2f}s)"
        )
    if readmitted:
        notes.append(
            f"KEPT {readmitted} near-duplicate segment(s) despite the variety guard: "
            f"dropping them would leave {unique_seconds:.1f}s of selection against the "
            f"{coverage_floor_seconds:.1f}s the cut planner needs to fill the reel — "
            "the higher-scoring duplicate of each pair was kept; review the reel for "
            "repeated shots"
        )
    survivors = {seg.id for seg in kept}
    # The MODEL'S order, filtered. See the note about `section_change` above.
    return _Deduped(
        segments=[seg for seg in segments if seg.id in survivors],
        dropped=dropped,
        kept_for_coverage=readmitted,
        notes=notes,
    )


def _reads_as_the_same_shot(a: SelectedSegment, b: SelectedSegment) -> bool:
    """True when ``b`` would show the audience what ``a`` already showed them.

    Same source, and either overlapping windows (a negative gap) or a gap under
    ``_NEAR_DUPLICATE_GAP_SECONDS``. Different sources are never the same shot.
    """
    if a.source_id != b.source_id:
        return False
    gap = max(a.in_seconds, b.in_seconds) - min(a.out_seconds, b.out_seconds)
    return gap < _NEAR_DUPLICATE_GAP_SECONDS


def _clamp_window(seg: dict[str, object], duration: float) -> tuple[float, float] | None:
    """Clamp ``(in, out)`` into ``[0, duration]`` and enforce the length bounds.

    ``None`` means "drop this segment": the window is inverted, degenerate, or
    shorter than a shot after clamping. A too-LONG window is truncated rather
    than dropped — the model's rationale describes the moment at the head, so
    keeping the head is closer to its intent than discarding the edit entirely.
    """
    if duration <= 0:
        return None
    start = max(0.0, min(_as_float(seg.get("in_seconds")), duration))
    end = max(0.0, min(_as_float(seg.get("out_seconds")), duration))
    if end - start < _MIN_SEGMENT_SECONDS:
        return None
    return start, min(end, start + _MAX_SEGMENT_SECONDS)


def _clamp_unit(value: object) -> float:
    return max(0.0, min(1.0, _as_float(value)))


def _as_float(value: object) -> float:
    try:
        return float(value)  # ty: ignore[invalid-argument-type]
    except TypeError, ValueError:
        return 0.0
