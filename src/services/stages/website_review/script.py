"""website_review_script — narration script over the critique + redesign story.

Writes a timed voiceover that walks the viewer through the headline critique
findings and how the redesign addresses them. Each line may reference the
critique finding id it speaks to (``critique_item_id``) for downstream
annotation. Mirrors the app_review script stage.
"""

from __future__ import annotations

import contextlib
import logging
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final, cast

from w2v_common.stages import StageRetriesExhaustedError

from src.clients.design_client import DesignClient
from src.services.stages.base import (
    SlotUse,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    find_missing_verbatim,
    items_for_stage,
    verbatim_prompt_block,
)
from src.services.stages.generic.grounding import grounding_prompt_block
from src.services.stages.primitives.dialogue import (
    assign_dialogue_speakers,
    dialogue_cast,
    voice_for_speaker,
)
from src.services.stages.primitives.script_stage import (
    LLMScriptStageBase,
    ScriptDrawUnderFloorError,
)
from src.services.stages.shared.compute_plan import beat_word_budget
from src.services.stages.shared.degradation import (
    SCRIPT_ACT_IMBALANCE,
    SCRIPT_CLAIMS_UNGROUNDED,
    Degradation,
    advisory,
    degradation,
)
from src.services.stages.shared.empty_draw import empty_draw_failure, require_elements
from src.services.stages.shared.narration_budget import (
    DEFAULT_TOLERANCE,
    NarrationEstimate,
    TrimPlan,
    budget_words,
    build_narration_estimate,
    closer_to_order,
    estimate_seconds,
    narration_degradation,
    narration_estimate_advisory,
    narration_verdict,
    plan_trim,
    speech_rate_degradation,
    steer_verdict,
    trim_floor_degradation,
    word_target_sentence,
    wpm_is_usable,
)
from src.services.stages.shared.narration_budget import (
    count_words as count_narration_words,
)
from src.services.stages.shared.placeholder_text import text_or_empty
from src.services.stages.shared.shotlist import ShotSpec, normalize_shotlist
from src.services.stages.shared.speech_rate import (
    AUTHORED,
    CALIBRATION_PASSAGE,
    CALIBRATION_SENTENCE_TERMINATORS,
    CALIBRATION_TEXT,
    CALIBRATION_WORDS,
    RESOLUTION_ERROR,
    ScriptMeasurement,
    SpeechRate,
    measure_script_lines,
    resolve_effective_wpm,
    script_measurement_degradation,
)
from src.services.stages.shared.time_budget import DEFAULT_WPM_NATURAL
from src.services.stages.shared.video_gate import require_video
from src.services.stages.website_review.grounding import (
    MIN_AFTER_SHARE,
    RedesignInventory,
    ScriptLineView,
    ScriptViolations,
    build_inventory,
    find_script_violations,
    inventory_is_band_only,
)
from src.services.stages.website_review.order import (
    ReservedTitleCards,
    narration_budget_seconds,
    reserve_title_cards,
    website_review_ordered_seconds,
)
from src.services.stages.website_review.record import WebsiteReviewRecordStage

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.clients.tts_client import TTSClient
    from src.services.llm import WMClient
    from src.services.runs.run_context import RunContext

logger = logging.getLogger(__name__)

#: The stage whose ``synthesize_and_store`` call actually speaks these lines.
#: Read off the class rather than repeated as a literal so the two cannot drift: the
#: measurement below stores its clips under THIS name, and ``deterministic_clip_id``
#: then makes the measured clip and the narrated clip one object — see
#: :data:`NARRATE_SPEED` and :meth:`WebsiteReviewScriptStage._measure`.
NARRATE_STAGE_NAME: Final = WebsiteReviewRecordStage.name

#: The ``speed`` ``website_review_record`` requests: it inherits
#: ``NarrationStageBase.speed_for``, which returns ``None``, so the argument is
#: omitted entirely and ``deterministic_clip_id`` hashes ``None``. The measurement
#: MUST request the same thing — ``1.0`` is a different clip id AND a different
#: measurement (speed scales duration linearly backend-side), so measuring at 1.0
#: would both mis-measure the voice and be reused by nothing.
NARRATE_SPEED: Final[float | None] = None

#: How many CORRECTIVE attempts one draw may cost, and in which order they are
#: spent: attempt 1 is the LLM re-ask, attempt 2 the deterministic trim. So the
#: LLM budget for the whole stage is exactly TWO calls — the opening draw and one
#: re-ask — and the second attempt is free, because the trim runs on per-line
#: seconds this stage has already measured. THE SHORT SIDE is the documented
#: exception — see :data:`MAX_SHORT_REASK_ROUNDS`.
MAX_CORRECTIVE_ATTEMPTS: Final = 2

#: How many EXTRA short-side rounds :meth:`_short_reask_step` may spend, on top of
#: the one directional re-ask every draw gets. Each round is a real LLM ask for
#: more grounded material (never padding), guarded to fire only while the draw is
#: STILL short and no long candidate exists to trim instead — so a round that does
#: not help stops the loop on its own; this cap only bounds the case where every
#: round IS helping.
#:
#: Raised from an effective 1 to 2 after prod run 01a0b2f5-ed32-73f2-8a6e-2158a9ccf37a
#: (website_review, 120s order, example.com): first draft 58.5s (-51%), ONE short
#: reask recovered it to only 88.9s (-25.9%, still out of the +/-10% band), and the
#: single 1b attempt was the last one available — the draw shipped short with no
#: further lever. The single-round cap was measured and fixed only against 60s
#: orders (see :meth:`_short_reask_step`'s docstring); a bigger initial gap needs
#: more than one extra ask to close, and a second round is still an ASK, not a
#: mechanical lengthening.
MAX_SHORT_REASK_ROUNDS: Final = 2

#: A comfortable narration line, in words — ONE sentence of the shape delivered
#: narration is measured at. Not chosen here: it is the words-per-sentence of
#: ``speech_rate.CALIBRATION_TEXT``, the passage that module shaped to "the 12-14
#: band measured off delivered narration", so the two cannot drift. Used ONLY to
#: turn a word budget into a line count (:func:`after_line_allowance`); the prompt
#: still states no per-line word bound.
COMFORTABLE_LINE_WORDS: Final = CALIBRATION_WORDS // sum(
    CALIBRATION_TEXT.count(mark) for mark in CALIBRATION_SENTENCE_TERMINATORS
)

#: The language every narration line is written in. NOT a style choice — it is the
#: only language the live narrator can speak:
#:
#: * kokoro (``tts_backend`` on mini2) has NO Dutch/German/etc. voice at all, and
#: * the backend phonemizes EVERY request as American English (``KPipeline`` built
#:   once with ``lang_code="a"``; the onnx engine passes no ``lang``), so even its
#:   French/Spanish voices would mangle their own language.
#:
#: Prod run 01a09f82 (cliffordtraining.nl) let the LLM follow the Dutch findings and
#: shipped a Dutch script read by ``af_heart`` as English phonemes. Narrating a
#: non-English site in its own language needs a TTS provider/backend that speaks it —
#: a backend change, not something this prompt can fake.
NARRATION_LANGUAGE: Final = "English"


def language_instruction() -> str:
    """The prompt line pinning the script to what the narrator can actually say."""
    return (
        f"Language: write EVERY line in {NARRATION_LANGUAGE}, whatever language the site, "
        f"its findings or any grounding material are in — the narrator voice speaks only "
        f"{NARRATION_LANGUAGE}. Keep brand names, product names and short quoted UI labels "
        "verbatim, but never write a sentence in another language.\n"
    )


def _target_seconds(snapshot: dict) -> float:
    """The seconds this script may SPEAK — the order minus the compile's fixed cards.

    Resolved by ``website_review/order.py`` (the ONE resolver compile + the create
    door share): ``deliverable.target_duration_seconds`` → ``brief.duration_seconds``
    → 120, less the intro/outro title-card seconds the compile splices around the
    lines. Budgeting the whole order as narration is how a 10s order shipped 15.1s
    (prod run 01a09f82). Extracted so ``build_output`` and the reconciliation
    re-derive the SAME number the prompt was budgeted against.
    """
    return narration_budget_seconds(snapshot)


def word_budget(target_seconds: float, wpm: float) -> int:
    """Whole-script word budget = ``duration_s / 60 * wpm``, rounded.

    Thin wrapper over the EXISTING arithmetic in ``compute_plan.beat_word_budget``
    (per-beat only by naming; the formula is the one a whole-script budget needs).
    """
    return round(beat_word_budget(float(target_seconds), float(wpm)).value)


def count_words(texts: list[str]) -> int:
    """Actual spoken word count of the drawn script (whitespace-split).

    Delegates to ``narration_budget.count_words`` so the number recorded on the
    output and the number the reconciliation arithmetic works from are produced by
    ONE counting rule, not two that can drift apart.
    """
    return count_narration_words(texts)


def word_budget_sentence(target_seconds: float, wpm: float) -> str:
    """The countable half of "a N-second script".

    An LLM cannot honour a duration — it has no model of how long its own words
    take to say, which is why this stage's films overran their order badly. A word
    count it CAN honour. ``wpm`` is the MEASURED rate for this run's voice
    (``speech_rate.resolve_natural_wpm``), not the 150.0 constant: the live kokoro
    voice measures ~94, so a budget built on 150 is ~60% too generous.

    Stated as a TARGET band (:func:`narration_budget.word_target_sentence`), not a
    ceiling: "a HARD CEILING … write fewer if the story is done sooner" made the
    model under-write, and a film is exactly its narration.
    """
    return word_target_sentence(target_seconds, wpm, word_budget(target_seconds, wpm))


@dataclass(frozen=True, slots=True)
class _Draw:
    """One candidate script AND the per-line seconds it was judged at.

    The two travel together because the defect this stage is fixing is a draw
    judged at another draw's rate: the old loop estimated the FIRST draft from a
    cast rate and then graded the re-asked one — different words, and possibly a
    different speaker mix — at that same number.

    ``line_seconds`` is parallel to ``output.lines``, so a trim of the candidate
    is an index selection on both at once and needs no second measurement.
    """

    output: WebsiteReviewScriptOutput
    line_seconds: tuple[float, ...]
    #: Per line: is that seconds a FACT (a real synthesis, or an exactly-0.0s
    #: blank line) rather than words-over-a-rate?
    measured: tuple[bool, ...]
    #: ``sum(line_seconds)`` — the number every verdict here is taken against.
    estimate: float
    #: The rate THIS draw's seconds are expressed at, and where it came from.
    wpm: float
    basis: str
    #: What THIS draw's measurement was faulty about — the entries that belong on
    #: the degrading tier if and only if this draw is the one that SHIPS.
    #:
    #: They travel on the draw for exactly the reason the seconds do. The entries
    #: used to be accumulated across every draw the stage looked at, so a re-ask
    #: whose measurement failed and was then DISCARDED still flipped the run
    #: ``degraded`` — while the script that shipped had been measured end to end,
    #: and the text claimed every duration number "was budgeted at the ...
    #: calibration_passage rate", which was false of the draft that shipped.
    #: Advisories never leaked between candidates; degradations must not either.
    #: Pinned by
    #: ``test_a_discarded_re_asks_failed_measurement_never_degrades_the_shipped_draw``.
    rate_entries: tuple[Degradation, ...] = ()

    @property
    def measured_lines(self) -> int:
        return sum(1 for flag in self.measured if flag)

    def select(self, indices: Sequence[int]) -> _Draw:
        """This draw reduced to ``indices`` — lines, seconds and flags together.

        A trim needs no second measurement precisely because of this: the kept
        lines are the very lines that were synthesized, so their seconds are the
        same facts, selected.
        """
        seconds = tuple(self.line_seconds[i] for i in indices)
        return _Draw(
            output=_with_lines(self.output, [self.output.lines[i] for i in indices]),
            line_seconds=seconds,
            measured=tuple(self.measured[i] for i in indices),
            estimate=math.fsum(seconds),
            wpm=self.wpm,
            basis=self.basis,
            rate_entries=self.rate_entries,
        )


def _pool_label(draw: _Draw, pool: list[_Draw]) -> str:
    """Which draft ``draw`` is, for an action string a human reads later."""
    return "first draw" if pool and draw is pool[0] else "re-asked draw"


def _floor_note(plan: TrimPlan) -> str:
    """The action-string half of "the trim ran and could not reach the band"."""
    reason = (
        "next drop would land UNDER the band"
        if plan.declined_under_band
        else f"{plan.protected_count} protected line(s)"
    )
    return f"STILL LONG — trim floored by {reason}"


def _trimmable_seconds(draw: _Draw) -> list[float] | None:
    """``draw``'s per-line seconds for ``plan_trim`` — or ``None`` to use words.

    ``plan_trim`` RAISES on a seconds list that is not a measurement (non-finite,
    negative, mis-length), and rightly so: silently mis-aligning seconds with
    lines cuts somebody's script by the wrong weights. But this is a paid run's
    happy path, so the check happens HERE and a measurement that is not one
    simply falls back to the word proxy rather than aborting the reconciliation.
    """
    seconds = draw.line_seconds
    if len(seconds) != len(draw.output.lines):
        return None
    if not all(math.isfinite(value) and value >= 0.0 for value in seconds):
        return None
    return list(seconds)


def _finalize(
    output: WebsiteReviewScriptOutput,
    *,
    degradations: list[Degradation],
    advisories: list[Degradation] | None = None,
    estimate: NarrationEstimate | None = None,
) -> WebsiteReviewScriptOutput:
    """``output`` carrying what this run CONCLUDED — on the right three channels.

    ``degradations`` is evidence of a bad delivery and flips the run ``degraded``;
    ``advisories`` is the same entry shape on the tier that does not (a pre-audio
    estimate is not a verdict on audio nobody has heard yet); ``narration_estimate``
    is the numbers themselves, recorded on EVERY reconciled run so a clean one is
    distinguishable from one nobody measured.
    """
    return output.model_copy(
        update={
            "degradations": [*output.degradations, *degradations],
            "advisories": [*output.advisories, *(advisories or [])],
            "narration_estimate": estimate,
        }
    )


class WebsiteReviewScriptInput(StageInput):
    pass


class ReviewLine(StageOutput):
    id: str
    text: str
    seconds: float
    critique_item_id: str | None = None
    # Two-act narrative phase: "before" lines critique the CURRENT design (play
    # over the original screenshots); "after" lines present the redesign (play
    # over the after screenshots). Normalised into a clean before→after arc in
    # build_output — see _assign_phases.
    phase: str = "before"  # "before" | "after"
    # Two-hander speaker tag ("a" asks / "b" answers). The model's own tags are kept
    # when it tagged both roles, else derived from the line role in build_output
    # (``assign_dialogue_speakers``). Cast to distinct voices by
    # website_review_record only when ``dialogue_cast`` says the run IS a dialogue.
    speaker: str = "a"  # "a" | "b"
    # AFTER-act lines: the rendered redesign section (``redesign-section-<name>``
    # visual id) this line is about. Only ids of sections that were actually
    # rendered survive build_output; website_review_visual_director honours it.
    section_id: str | None = None


class WebsiteReviewScriptOutput(StageOutputWithNotes):
    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. Empty here means the stage
    # reported success having produced nothing, which degrades the run.
    produces: ClassVar[tuple[str, ...]] = ("lines",)

    lines: list[ReviewLine]
    total_seconds: float
    # Ordered-vs-delivered accounting. `total_seconds` is the model's OWN estimate
    # of its speaking time and has never been checked against anything; these two
    # are countable. RECORDING ONLY — nothing here raises or truncates.
    word_budget: int = 0
    actual_words: int = 0
    #: What the narration budget CONCLUDED, as numbers — the rate, where it came
    #: from, the first draw's estimate, and what each correction did. Present on
    #: EVERY reconciled run, clean ones included: "this run was measured and it
    #: landed" and "nobody ever measured this run" are otherwise the same absence.
    #: ``None`` only when reconciliation could not run at all.
    narration_estimate: NarrationEstimate | None = None
    #: The title-card seconds THIS script took out of its narration budget — what was
    #: reserved (after the 20% cap), not what the ``title_card`` block configures. The
    #: compile renders exactly this and never re-plans it, so a compile that runs
    #: under a different cap rule than the script (a resume or a regenerate across a
    #: deploy) still puts on screen the cards the narration was written around.
    #: ``None`` = a script produced before this record existed, which the compile
    #: reads as "budgeted for the cards as configured". A script that reserved NO
    #: cards records zeros, never ``None``.
    reserved_title_cards: ReservedTitleCards | None = None


class WebsiteReviewScriptStage(
    LLMScriptStageBase[WebsiteReviewScriptInput, WebsiteReviewScriptOutput]
):
    name = "website_review_script"
    input_model = WebsiteReviewScriptInput
    output_model = WebsiteReviewScriptOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (
        SlotUse("llm"),
        SlotUse("tts"),
    )
    element_key = "lines"
    element_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "seconds": {"type": "number"},
            "critique_item_id": {"type": ["string", "null"]},
            "phase": {"type": "string", "enum": ["before", "after"]},
            "speaker": {"type": "string", "enum": ["a", "b"]},
            "section_id": {"type": ["string", "null"]},
        },
        "required": ["text", "seconds", "phase"],
    }

    #: An EMPTY draw is invalid, not a result. Live 2026-10-06, run
    #: ``01a10e19-145f-7475-a34a-aa1fc8d3113f`` (50s order): the model answered
    #: HTTP 200 with ``{"lines": []}`` (7 completion tokens), the primitive's default
    #: floor of 0 accepted it, the stage persisted 0 lines as a success, four
    #: downstream stages no-op'd and the run died three stages later in compile with
    #: "no shots to compile". The floor is applied by the PARSER
    #: (``LLMScriptStageBase._attempt``), inside the retry budget — it cannot live in
    #: the output schema, because worker-management strips ``minItems``
    #: (``tests/test_output_schema_bounds.py``). Every attempt empty ⇒
    #: :class:`~src.services.stages.shared.empty_draw.EmptyScriptDrawError` out of
    #: ``_first_draw``.
    min_elements: ClassVar[int] = 1

    #: This run's narration rate — the number the ASKS are written from (the word
    #: budget in the prompt, the target in the re-ask). Overwritten per execution in
    #: ``handle`` with the PHASE-A prior (the calibration passage measured at the
    #: run's configured voice), and again in ``_reconcile`` with the rate pooled off
    #: the drafted lines themselves once they have been synthesized. It is NOT what a
    #: verdict is taken against any more — that is ``_Draw.estimate``, real measured
    #: seconds. The class default is the fallback so a direct ``build_prompt`` call
    #: (tests, tooling) still works.
    _wpm: float = DEFAULT_WPM_NATURAL

    #: The Phase-A prior, with its provenance. Kept beside ``_wpm`` because the number
    #: alone cannot say whether anybody measured it — and because the full-script
    #: measurement needs something to fall back TO when it cannot land.
    _rate: SpeechRate = SpeechRate(wpm=DEFAULT_WPM_NATURAL, measured=False, reason=RESOLUTION_ERROR)

    #: Where ``_wpm`` came from — moved in lockstep with it. A WORD FLOOR is stated at
    #: :func:`floor_wpm` of the pair, never at a calibration-passage rate directly.
    _wpm_basis: str = CALIBRATION_PASSAGE

    #: Appended to the prompt by ``build_prompt`` for the ONE bounded re-ask (see
    #: :data:`MAX_CORRECTIVE_ATTEMPTS` — attempt two is the trim, not a second
    #: re-ask, so this is set at most once per run). Empty for the first draw, and
    #: cleared again after the re-ask, so a prompt inspected outside reconciliation
    #: is unchanged.
    _reask_block: str = ""

    #: What the rendered redesign contains (sections + page features), read once per
    #: execution in ``handle``. ``None`` until then; ``_inventory_for`` then derives
    #: the section list from the redesign visuals alone (no HTML, features unknown),
    #: so a direct ``build_prompt`` call still grounds on the rendered sections.
    _inventory: RedesignInventory | None = None

    #: Appended for the ONE grounding re-ask: the draft's violations, verbatim. Kept
    #: set through the duration reconciliation so a length rewrite cannot quietly
    #: reintroduce the claims it was asked to remove.
    _grounding_block: str = ""

    #: The largest after-act line allowance any prompt of THIS execution has stated
    #: (:meth:`_after_allowance`). ``_wpm`` moves between the first draw and the
    #: re-asks, and the allowance is derived from it — so it only ever grows: a draw
    #: is never judged against a tighter rule than the prompt it answered.
    _allowance: int = 0

    def __init__(
        self,
        client: WMClient | None = None,
        tts: TTSClient | None = None,
        design: DesignClient | None = None,
    ) -> None:
        """``tts`` is the client the full-script measurement synthesizes through.

        ``None`` (production) lets ``measure_script_lines`` construct its own,
        exactly as the Phase-A probe already does. It exists so a test can measure a
        real duration off a fake backend instead of stubbing the measurement out — a
        stubbed measurement proves the wiring and nothing about the arithmetic that
        motivated it.
        """
        super().__init__(client)
        self._tts = tts
        self._design = design

    async def handle(
        self, payload: WebsiteReviewScriptInput, ctx: RunContext
    ) -> WebsiteReviewScriptOutput:
        # critique-only runs (generate_video=false) skip narration scripting; the
        # redesign it depends on was skipped too, so running on would fail loud.
        require_video(ctx)
        # `build_prompt` is SYNCHRONOUS on `LLMScriptStageBase` (eight stages
        # inherit it), so the wpm probe — the one await this stage needs — is taken
        # here and stashed. `dag_executor._run_stage` constructs a fresh handler per
        # stage execution, so instance state is safe (see `action_reel/compile.py`).
        self._rate = (await resolve_effective_wpm(ctx, texts=(), default=DEFAULT_WPM_NATURAL)).rate
        self._wpm = self._rate.wpm
        self._wpm_basis = self._rate.basis
        self._reask_block = ""
        self._grounding_block = ""
        self._inventory = await self._read_inventory(ctx)
        first = await self._grounded_draw(payload, ctx)
        # The parser floor (``min_elements``) already refuses an empty LLM draw;
        # this refuses a zero-line OUTPUT however it came about, before a second
        # is spent measuring it. Nothing below may turn a script into no script.
        require_elements(first.lines, stage=self.name, context=self._draw_facts(ctx))
        try:
            reconciled = await self._reconcile(first, payload, ctx)
        except Exception:  # a paid run must never fail on length arithmetic
            logger.warning(
                "website_review_script narration reconciliation failed — "
                "shipping the unreconciled script",
                exc_info=True,
            )
            reconciled = first
        require_elements(reconciled.lines, stage=self.name, context=self._draw_facts(ctx))
        return self._record_violations(reconciled, ctx)

    # ------------------------------------------------------------------ #
    # grounding
    # ------------------------------------------------------------------ #

    async def _read_inventory(self, ctx: RunContext) -> RedesignInventory:
        """The redesign's rendered sections, enriched from its HTML when readable.

        Soft by design: the HTML is fetched from html_render by job id, and any
        failure falls back to the sections the visuals name, with page features
        UNKNOWN — so a claim about page content is then not judged at all rather
        than judged against an inventory nobody took.
        """
        redesign = ctx.prior_output("website_redesign") or {}
        job_id = str(redesign.get("redesign_job_id") or "").strip()
        html: str | None = None
        if job_id and any(s for s in redesign.get("visuals") or []):
            # A client this stage constructs is closed here (same shape as
            # ``design_client.design_html_exists``); an injected one is the caller's.
            owned = self._design is None
            client = self._design or DesignClient()
            try:
                html = await client.get_design_html(job_id)
            except Exception:  # an unreadable inventory must never fail the script
                logger.warning(
                    "website_review_script: redesign HTML for job %s unreadable — "
                    "grounding on section names only",
                    job_id,
                    exc_info=True,
                )
            finally:
                if owned:
                    with contextlib.suppress(Exception):
                        await client.close()
        return build_inventory(redesign, html)

    def _inventory_for(self, ctx: RunContext) -> RedesignInventory:
        if self._inventory is None:
            self._inventory = build_inventory(ctx.prior_output("website_redesign") or {}, None)
        return self._inventory

    def _violations(self, output: WebsiteReviewScriptOutput, ctx: RunContext) -> ScriptViolations:
        brief = ctx.config_snapshot.get("brief") or {}
        return find_script_violations(
            [
                ScriptLineView(text=line.text, phase=line.phase, section_id=line.section_id)
                for line in output.lines
            ],
            self._inventory_for(ctx),
            authored_before=len(normalize_shotlist(brief)),
            after_allowance=self._after_allowance(ctx),
        )

    def _after_allowance(self, ctx: RunContext) -> int:
        """This run's after-act line allowance, from the numbers ``build_prompt`` states.

        :func:`after_line_allowance` at the prompt's own word ceiling, held at the
        largest value seen this execution (see :attr:`_allowance`). ONE resolver for
        the prompt's rule, the post-check that judges a draw against it and the
        empty-draw error that reports it.
        """
        shots = normalize_shotlist(ctx.config_snapshot.get("brief") or {})
        self._allowance = max(
            self._allowance,
            after_line_allowance(
                len(self._inventory_for(ctx).sections),
                word_budget(int(_target_seconds(ctx.config_snapshot)), self._wpm),
                authored_before=len(shots),
                authored_words=count_words([shot.narration for shot in shots]),
            ),
        )
        return self._allowance

    async def _grounded_draw(
        self, payload: WebsiteReviewScriptInput, ctx: RunContext
    ) -> WebsiteReviewScriptOutput:
        """The first draw — re-asked ONCE when it says what the redesign does not show.

        The re-ask carries the violations verbatim. The re-asked draft is kept only
        when it has FEWER violations; a failed re-ask keeps the first draw (a
        correction must never kill a paid run that already holds a script).
        """
        first = await self._first_draw(payload, ctx)
        found = self._violations(first, ctx)
        if not found:
            return first
        texts = [line.text for line in first.lines]
        self._grounding_block = _grounding_reask_block(
            found.describe(texts),
            shared_sections=self._after_allowance(ctx) > len(self._inventory_for(ctx).sections),
            band_only=inventory_is_band_only(self._inventory_for(ctx)),
        )
        try:
            # A correction of a draw already in hand: no floor, one call, and an
            # empty answer simply loses to ``first`` below.
            with self.corrective_draw():
                second = await super().handle(payload, ctx)
        except Exception:  # a failed correction degrades to the first draw
            logger.warning(
                "website_review_script grounding re-ask failed — keeping the first draw",
                exc_info=True,
            )
            return first
        if second.lines and self._violations(second, ctx).count() < found.count():
            return second
        return first

    async def _first_draw(
        self, payload: WebsiteReviewScriptInput, ctx: RunContext
    ) -> WebsiteReviewScriptOutput:
        """The draw every later step corrects — which therefore has to EXIST.

        ``min_elements = 1`` makes the primitive re-prompt an empty answer within
        the retry budget (``llm_retries() + 1`` calls). When every one of them came
        back empty the stage fails HERE, as the same ``StageRetriesExhaustedError``
        it always raised, with the cause renamed to what happened — an empty script
        draw — plus what the prompt actually handed the model (:meth:`_draw_facts`).

        Only the FIRST draw is floored. The corrective re-asks (grounding,
        duration) already hold a usable script, so they run inside
        ``corrective_draw()``: an empty answer there costs one call, is never
        re-prompted, and is recorded as a re-ask that was sent and discarded.

        An authored shotlist does NOT make an empty draw legitimate. The operator
        wrote the BEFORE act; the model is asked for the AFTER act and told its
        first line "must clearly PRESENT the redesign" — the reveal the film exists
        for, over a redesign this run already paid to render. "Nothing to add" is
        not an answer that prompt admits, so it is re-asked and then fails like any
        other empty draw rather than shipping a review that never shows the fix.
        """
        try:
            return await super().handle(payload, ctx)
        except StageRetriesExhaustedError as exc:
            failure = empty_draw_failure(exc, stage=self.name, context=self._draw_facts(ctx))
            if failure is None:
                raise
            raise failure from exc

    def _draw_facts(self, ctx: RunContext) -> str:
        """What the script prompt HAD, as a clause for an empty-draw error.

        Only numbers ``build_prompt`` itself states, read the same way it reads
        them: the rendered redesign sections, the narration seconds and word
        budget, and the line count its own rules cap the script at
        (:func:`max_lines_by_rules`). NOT the site capture: this prompt is built
        from the critique, the redesign, its inventory and ``source_ingest`` and
        never contains the scraped page text, so "the capture was thin" explains
        nothing here (that clause shipped in this error on 2026-10-06, #1068).
        """
        snapshot = ctx.config_snapshot
        sections = len(self._inventory_for(ctx).sections)
        authored = len(normalize_shotlist(snapshot.get("brief") or {}))
        narration = int(_target_seconds(snapshot))
        allowance = self._after_allowance(ctx)
        cap = max_lines_by_rules(sections, authored_before=authored, after_allowance=allowance)
        if cap is None:
            rules = "no rendered sections, so its rules cap no line count"
        elif allowance > sections:
            rules = (
                f"its rules allow at most {cap} line(s) (up to {allowance} after-act "
                f"line(s) over {sections} section(s)"
                + ("" if authored else ", at least half of all lines after-act")
                + ")"
            )
        elif authored:
            rules = f"its rules allow at most {cap} line(s) (one after-act line per section)"
        else:
            rules = (
                f"its rules allow at most {cap} line(s) (one after-act line per section, "
                "at least half of all lines after-act)"
            )
        return (
            f"; the script prompt had {sections} rendered redesign section(s) for a "
            f"{website_review_ordered_seconds(snapshot):g}s order — {narration}s of "
            f"narration, {word_budget(narration, self._wpm)} words at {self._wpm:.0f} wpm "
            f"— and {rules}"
        )

    def _record_violations(
        self, output: WebsiteReviewScriptOutput, ctx: RunContext
    ) -> WebsiteReviewScriptOutput:
        """What the SHIPPED script still gets wrong, on the right tier.

        A claim the picture does not show is a bad delivery → degradation. An after
        act too thin for the redesign it presents is truthful but weak → advisory.
        A line merely missing its ``section_id`` is not recorded — the visual
        director still maps it.
        """
        found = self._violations(output, ctx)
        texts = [line.text for line in output.lines]
        grounding = found.grounding_problems(texts)
        degradations = (
            [degradation(SCRIPT_CLAIMS_UNGROUNDED, f"{self.name}: " + "; ".join(grounding))]
            if grounding
            else []
        )
        advisories = (
            [advisory(SCRIPT_ACT_IMBALANCE, f"{self.name}: " + "; ".join(found.imbalance))]
            if found.imbalance
            else []
        )
        if not (degradations or advisories):
            return output
        return output.model_copy(
            update={
                "degradations": [*output.degradations, *degradations],
                "advisories": [*output.advisories, *advisories],
            }
        )

    # ------------------------------------------------------------------ #
    # narration reconciliation
    # ------------------------------------------------------------------ #

    def _cast_segments(
        self, output: WebsiteReviewScriptOutput, ctx: RunContext
    ) -> list[tuple[str, str | None]]:
        """``(text, voice_id)`` per line — the voice ``website_review_record`` WILL use.

        THE multi-voice difference from the single-narrator siblings, and the reason
        it is load-bearing twice over:

        * the measurement below synthesizes each line AT THIS VOICE, so a wrong voice
          measures audio nobody will hear;
        * ``deterministic_clip_id`` hashes ``(voice, speed, text)``, so a wrong voice
          also breaks the clip REUSE — ``website_review_record`` would find no
          existing clip and the run would pay for every line twice.

        Deliberately the SAME call ``website_review/record.py::voice_for`` makes, on
        the SAME shared ``DEFAULT_CAST`` (taken via ``voice_for_speaker``'s own
        default, not a copy of the mapping): a cast resolved by a parallel copy of
        the rule is a cast that goes wrong the first time somebody edits one of the
        two.

        The prompt, ``build_output``'s speaker derivation and this cast all read the
        ONE predicate ``dialogue_cast``: a run is a two-hander only when dialogue is
        explicitly enabled AND two distinct voices resolve. (They used to disagree —
        the prompt defaulted dialogue ON while casting required the explicit flag,
        so an unset flag shipped a two-speaker conversation read by ONE voice.)
        """
        voice = ctx.config_snapshot.get("voice") or {}
        brief = ctx.config_snapshot.get("brief") or {}
        return [
            (line.text, voice_for_speaker({"speaker": line.speaker}, voice, brief))
            for line in output.lines
        ]

    async def _measure(
        self, output: WebsiteReviewScriptOutput, ctx: RunContext
    ) -> ScriptMeasurement:
        """Synthesize EVERY line of this draft and return what it really runs to.

        This replaces the old estimate-at-a-blended-cast-rate, and the difference is
        not one of degree: a rate times a word count is a projection — several
        percent of error on exactly the axis this stage is graded on, and on a
        two-hander the blend is an average of voices that speak unequal shares.
        Measuring every line at its OWN cast voice yields the film's delivered
        length, because a website review IS the sum of its narration clips
        (``website_review/compile.py`` slaves the picture to the voice-over).

        It is very nearly free. ``speed`` and ``stage_name`` are the ones
        ``website_review_record`` will REQUEST (see :data:`NARRATE_STAGE_NAME` and
        :data:`NARRATE_SPEED`) and the voices are the ones it will CAST, so
        ``deterministic_clip_id`` makes every clip measured here the very clip
        record is about to ask for: each surviving line's synthesis is paid once,
        here, instead of once, later. Only the lines a re-ask replaces or a trim
        drops are genuinely spent.

        AN AUTHORED RATE IS NOT MEASURED AGAINST. When the operator wrote
        ``direction.timing.wpm.natural`` the ladder hands back an ``authored`` basis,
        and that rate "outranks every measurement" — so synthesizing the script here
        would buy a number this stage is contractually forbidden to use, at the cost
        of a full script of TTS. The estimate falls back to the prior-derived
        arithmetic, at the authored rate, and carries ``reason=None`` because nothing
        FAILED: the measurement was declined, not attempted and lost. (The clips are
        not lost either — ``website_review_record`` synthesizes them later, as it did
        before Phase C existed.)

        Never raises: ``measure_script_lines`` absorbs every failure into a
        prior-derived measurement carrying a ``reason``.
        """
        segments = self._cast_segments(output, ctx)
        if self._rate.basis == AUTHORED:
            return self._authored_measurement([text for text, _ in segments])
        return await measure_script_lines(
            ctx,
            texts=[text for text, _ in segments],
            voices=[voice for _, voice in segments],
            prior=self._rate,
            speed=NARRATE_SPEED,
            stage_name=NARRATE_STAGE_NAME,
            client=self._tts,
        )

    def _authored_measurement(self, texts: Sequence[str]) -> ScriptMeasurement:
        """The estimate at the AUTHORED rate, in ``ScriptMeasurement`` shape.

        The same SHAPE ``measure_script_lines`` returns when it declines — a per-line
        ``words / wpm`` estimate carrying the prior, ``measured_count=0``, no failed
        index — except for ``reason``, which is ``None`` here BECAUSE NOTHING FAILED.
        A reason would put a ``SPEECH_RATE_UNUSABLE`` degradation on a run whose rate
        the operator supplied, i.e. report the one case this stage handled exactly as
        instructed as the case it handled worst.

        NOT byte-for-byte, and the difference is the rate guard. ``_prior_measurement``
        divides by the prior's ``wpm`` directly; the seconds here go through
        ``estimate_seconds``, whose ``_safe_wpm`` SUBSTITUTES
        ``DEFAULT_WORDS_PER_MINUTE`` for any rate outside the usable band. So an
        operator who authors 400 gets a ``ScriptMeasurement`` whose ``wpm`` says 400
        and whose ``total_seconds`` was computed at the default — the two fields
        disagree. What this guarantees is only that the seconds are finite and
        positive for ANY authored number. The disagreement never reaches a run:
        ``_draw_of`` tests ``wpm_is_usable`` on exactly that ``wpm`` and falls to the
        word-count path, and the substitution is recorded as a degradation.

        ONE rate for a CAST script, deliberately. An authored rate is the operator's
        statement about the delivery, not about a voice, so it applies to every line
        regardless of who speaks it — the per-voice measurement that makes the cast
        blend honest is exactly what was declined.

        A blank line's ``0.0`` is exact, so it counts as measured — the same rule
        ``_prior_measurement`` applies, and what keeps ``measured_lines`` honest.
        """
        per_line = tuple(
            estimate_seconds([text], words_per_minute=self._rate.wpm) for text in texts
        )
        blank = tuple(not str(text).split() for text in texts)
        return ScriptMeasurement(
            per_line_seconds=per_line,
            measured=blank,
            failed_indices=(),
            extrapolated_indices=tuple(i for i, is_blank in enumerate(blank) if not is_blank),
            total_seconds=math.fsum(per_line),
            measured_count=0,
            wpm=self._rate.wpm,
            basis=self._rate.basis,
            prior=self._rate,
            reason=None,
        )

    def _draw_of(self, output: WebsiteReviewScriptOutput, measurement: ScriptMeasurement) -> _Draw:
        """A candidate paired with the seconds it will be JUDGED at.

        When the measured rate is unusable the measurement's own seconds are not
        either — a 0/NaN rate yields 0.0 per line, which would read as a 0.0s script
        and send a perfectly good draw off to be lengthened. That case falls back to
        word-count arithmetic, where ``estimate_seconds`` substitutes the documented
        default and ``speech_rate_degradation`` records that it did. Nothing here is
        silent.
        """
        texts = [line.text for line in output.lines]
        entries = tuple(self._rate_entries(measurement))
        if wpm_is_usable(measurement.wpm) and len(measurement.per_line_seconds) == len(texts):
            return _Draw(
                output=output,
                line_seconds=measurement.per_line_seconds,
                measured=measurement.measured,
                estimate=measurement.total_seconds,
                wpm=measurement.wpm,
                basis=measurement.basis,
                rate_entries=entries,
            )
        seconds = tuple(
            estimate_seconds([text], words_per_minute=measurement.wpm) for text in texts
        )
        return _Draw(
            output=output,
            line_seconds=seconds,
            measured=(False,) * len(texts),
            estimate=math.fsum(seconds),
            wpm=measurement.wpm,
            basis=measurement.basis,
            rate_entries=entries,
        )

    def _rate_entries(self, measurement: ScriptMeasurement) -> list[Degradation]:
        """The two REAL faults a measurement can carry, as ``degradations``.

        Either can hold without the other: the measurement did not land
        (``reason``), and the rate finally used is outside the budgetable band
        (``wpm_is_usable``). Both mean every duration number on this run describes
        audio nobody measured — a fault, not advice, which is why these stay on the
        degrading tier while the estimate MISS moves off it.
        """
        entries: list[Degradation] = []
        unmeasured = script_measurement_degradation(measurement, stage=self.name)
        if unmeasured is not None:
            entries.append(unmeasured)
        if not wpm_is_usable(measurement.wpm):
            entries.append(
                speech_rate_degradation(stage=self.name, words_per_minute=measurement.wpm)
            )
        return entries

    async def _reask(
        self,
        draw: _Draw,
        ordered: float,
        payload: WebsiteReviewScriptInput,
        ctx: RunContext,
    ) -> WebsiteReviewScriptOutput | None:
        """Ask the model again, through this stage's existing LLM call.

        Serves BOTH corrections — attempt one in either direction, and the short
        side's second attempt (1b). It is stateless between calls: ``_reask_block``
        is rebuilt on entry and cleared in ``finally``, and ``target_words`` is
        recomputed from ``self._wpm``, so a second call aims at the rate measured
        off the draw it is correcting rather than a stale one.

        ``None`` when the call failed — a re-ask is a correction, and a paid run
        that already holds a usable draw must not die because the correction did.

        A SHORT draw is re-asked FLOORED: the block states the minimum word count
        the band's lower edge needs (:func:`short_draw_floor_words`), not only an
        "approximately N" target. The standing prompt once called the budget a "HARD
        CEILING … write fewer if the story is done sooner", an instruction to
        under-write; an ask that named no minimum left that instruction standing
        and is how a 50s order shipped ~15s. The standing prompt now states a
        target band, and the floor stays as the re-ask's countable minimum.
        """
        short = steer_verdict(draw.estimate, ordered) == "short"
        self._reask_block = _reask_prompt_block(
            produced_words=count_words([line.text for line in draw.output.lines]),
            estimated_seconds=draw.estimate,
            ordered_seconds=ordered,
            # SHORT side: the corrected draw's OWN arithmetic (as app_review's
            # explicit-target re-ask does), because ``self._wpm`` is the rate
            # measured LAST — possibly off a draft that was discarded.
            target_words=budget_words(ordered, words_per_minute=draw.wpm if short else self._wpm),
            floor_words=(
                short_draw_floor_words(ordered, floor_wpm(draw.wpm, draw.basis)) if short else None
            ),
        )
        try:
            # A correction, not a draw: parsed with no floor, so an empty answer
            # costs ONE call and returns as zero elements for the caller to record
            # as a failed re-ask (``LLMScriptStageBase.corrective_draw``).
            with self.corrective_draw():
                return await super().handle(payload, ctx)
        except Exception:
            # Keeps the draw it holds; the CALLER names the failure in ``action``
            # and, if what ships is below the floor, degrades the run for it.
            logger.warning(
                "website_review_script duration re-ask failed — keeping the draw in hand",
                exc_info=True,
            )
            return None
        finally:
            self._reask_block = ""

    async def _short_reask_step(
        self,
        best: _Draw,
        pool: list[_Draw],
        ordered: float,
        payload: WebsiteReviewScriptInput,
        ctx: RunContext,
        action: str,
    ) -> tuple[_Draw, str, bool]:
        """ATTEMPT TWO+, SHORT SIDE — one more re-ask for a draw that is still short.

        Called in a loop by :meth:`_reconcile`, up to :data:`MAX_SHORT_REASK_ROUNDS`
        times; each call is one round and stops the loop itself (``kept=False``) the
        moment it is no longer short, a long candidate appears, or a round fails to
        land closer — so most runs still take exactly one extra round, as before.

        :func:`_trim_step` is LONG-only by construction (it filters the pool to
        candidates whose verdict is "long" and returns untouched when none
        qualify), so a draw that came in short and is STILL short after one re-ask
        had no second attempt at all: it shipped as drawn. That is the asymmetry,
        and it is what the live misses were made of. Measured on three 60s orders
        2026-09-15 — delivered 45.7s, 51.8s and 51.5s (-23.8%, -13.6%, -14.1%) —
        and ALL THREE had already re-asked and KEPT the re-ask (``reasked=true,
        reask_kept=true``) while still recording ``verdict="short"``. One
        correction was not enough, and there was no second one to take.

        Why the words go missing: the budget is sized at the calibration prior,
        which drew 135-149 words on the three misses against 174 on the runs that
        landed, while the model habitually writes FEWER words than budgeted (6 of
        7 runs). A high budget absorbs that under-write; a budget already equal to
        the words the rate needs has no slack, so the under-write lands straight
        out of band. The re-ask is the lever that recovers it, because
        :meth:`_reask` re-targets at ``self._wpm`` — the rate MEASURED off this
        run's own lines, which matched delivery to within 0.3% on all seven runs.

        This is an ASK, not padding. Nothing is repeated and nothing is stretched;
        the model is asked for more material, which is precisely what the "no
        mechanical lengthening" rule allows. It reuses this stage's OWN re-ask, so
        the grounding and two-act constraints that bind the first correction bind
        this one unchanged. ``closer_to_order`` keeps whichever draw is nearer the
        order, so a second ask that overshoots long is discarded, not shipped.

        Guarded on THE TRIM HAVING NOTHING TO DO — still short, and no LONG
        candidate anywhere in the pool. Whenever a long draw exists the trim can
        reach it deterministically and for free, and a paid LLM call must not
        pre-empt a free lever.

        ⚠️ EVERY OUTCOME IS NAMED IN ``action``, because the first cut of this
        method named only the winning one. Measured 2026-09-15 on the first wave
        after it shipped: seven 60s runs, three still short (-30.3%, -15.3%,
        -10.1%), and NOT ONE row carried the "re-asked twice" fingerprint. That is
        consistent with two completely different worlds — the guard never opened,
        or it opened every time and the second draw was never closer — and the
        persisted data could not tell them apart, because only the KEPT path wrote
        a string. A lever whose failure is silent cannot be evaluated, only
        believed. So a stand-down, a failed call and a rejected candidate now each
        say so, and the ``pool`` they were judged against is recorded alongside.
        """
        if steer_verdict(best.estimate, ordered) != "short":
            return best, action, False
        if any(steer_verdict(draw.estimate, ordered) == "long" for draw in pool):
            # Stood down ON PURPOSE so the free trim can take the long candidate.
            # Said out loud: a silent stand-down is indistinguishable from a step
            # that never existed, which is the whole reason this method is
            # instrumented (see the ⚠️ note above).
            longs = ", ".join(
                f"{draw.estimate:.1f}s"
                for draw in pool
                if steer_verdict(draw.estimate, ordered) == "long"
            )
            return best, f"{action}; second re-ask stood down for the trim ({longs})", False

        third = await self._reask(best, ordered, payload, ctx)
        if third is None:
            return best, f"{action}; second re-ask failed", False
        if not third.lines:
            return best, f"{action}; second re-ask came back empty", False

        measurement = await self._measure(third, ctx)
        self._wpm = measurement.wpm
        self._wpm_basis = measurement.basis
        candidate = self._draw_of(third, measurement)
        pool.append(candidate)
        if not closer_to_order(candidate.estimate, best.estimate, ordered):
            return (
                best,
                f"{action}; second re-ask not closer "
                f"({candidate.estimate:.1f}s vs {best.estimate:.1f}s, order {ordered:.1f}s)",
                False,
            )
        return candidate, "re-asked twice", True

    async def _reconcile(
        self,
        first: WebsiteReviewScriptOutput,
        payload: WebsiteReviewScriptInput,
        ctx: RunContext,
    ) -> WebsiteReviewScriptOutput:
        """Steer at the ORDER, correct until in band or out of attempts, never pad.

        Measured live 2026-09-12: FIVE consecutive 60s orders delivered +36.5%,
        +45.0%, +36.5%, +52.8% and +51.4% — the worst type on the board. The cause
        was not that the loop was absent but that every number in it was a
        PROJECTION: the word budget and the overrun check both ran at the
        calibration rate (171-175 wpm) while the cast delivers 120-134 wpm, so the
        budget was ~20s too generous on a 60s order and the check never fired.

        Three properties, each of which was a measured live miss:

        1. **It aims at the target, not at the band edge.** The decision to correct
           is :func:`steer_verdict` (long side +4%), not the +/-10% band that GRADES
           the master — a draw accepted at +9% has nothing left for the estimate's
           own error. The +/-10% band still decides what is REPORTED, so a
           conformant film is never flagged.
        2. **It iterates.** Up to :data:`MAX_CORRECTIVE_ATTEMPTS` corrections — the
           re-ask, then the trim — and a draw is shipped only when it is IN BAND or
           no attempt improved it. "Closer" alone was the bug: a re-ask that swings
           a +34% draft to -18.4% is closer and would end the loop, while the +34%
           draft it is holding is TRIMMABLE at no cost.
        3. **Every candidate is judged at ITS OWN measured seconds** — synthesized
           at the voices that will speak it.

        Still no padding. Nothing lengthens a short script MECHANICALLY — repeated
        or stretched audio is a worse defect than a short film. Asking the model for
        more material is not that, and a short draw gets several asks before it
        ships: see (1b). Each of them is FLOORED (an explicit minimum word count —
        see :meth:`_reask`), and a short draw that survives them all, or whose
        re-ask FAILED, still ships (never raised, never padded) but as a
        ``duration_out_of_tolerance`` DEGRADATION, not only an advisory.
        """
        ordered = _target_seconds(ctx.config_snapshot)
        if ordered <= 0.0:
            # No order means no band and no verdict; inventing one is the error
            # this refuses to make. This is the ONLY early exit: "no lines" used to
            # leave here too, which let a zero-line draw skip the short-draw floor
            # it is the limiting case of. ``handle`` now refuses that script before
            # this is called (``require_elements``).
            return first

        measurement = await self._measure(first, ctx)
        self._wpm = measurement.wpm
        self._wpm_basis = measurement.basis

        best = self._draw_of(first, measurement)
        pool = [best]
        first_estimate = best.estimate
        action = "kept first draw"
        reasked = False
        reask_kept: bool | None = None

        # (1) ATTEMPT ONE — the re-ask, in whichever direction the steer points.
        if steer_verdict(best.estimate, ordered) in ("long", "short"):
            best, action, reasked, reask_kept = await self._first_reask_step(
                best, pool, ordered, payload, ctx
            )

        # (1b) ATTEMPT TWO+, SHORT SIDE — repeat while a round keeps improving.
        # See _short_reask_step. Each round re-checks its own guard (still short,
        # no long candidate in the pool), so a round that no longer applies or did
        # not help returns ``kept=False`` and the loop stops itself; the range()
        # only bounds the case where rounds keep landing closer.
        for _ in range(MAX_SHORT_REASK_ROUNDS):
            best, action, second_kept = await self._short_reask_step(
                best, pool, ordered, payload, ctx, action
            )
            if second_kept:
                reask_kept = True
            else:
                break

        # (2) ATTEMPT TWO — the trim. Deterministic, free (it selects lines whose
        #     seconds are already measured), and the only lever that can rescue a
        #     draw the re-ask overshot: the LONG draft in the pool is trimmable
        #     even when the SHORT one is currently winning.
        floored: TrimPlan | None = None
        trimmed_lines = 0
        if steer_verdict(best.estimate, ordered) != "ok":
            best, action, floored, trimmed_lines = self._trim_step(best, pool, ordered, ctx, action)

        estimate = build_narration_estimate(
            estimate_seconds=best.estimate,
            ordered_seconds=ordered,
            wpm=best.wpm,
            wpm_basis=best.basis,
            first_draw_estimate_seconds=first_estimate,
            reasked=reasked,
            reask_kept=reask_kept,
            trimmed_lines=trimmed_lines,
            measured_lines=best.measured_lines,
        )

        # Reported against the OUTER band — the one that grades the delivered
        # master — so a film this stage steered to +2% raises nothing, and a film
        # the steer merely improved to +6% raises nothing either. It IS conformant;
        # a second, worse, pre-audio opinion saying otherwise is how a good
        # delivery gets read as broken.
        # The SHIPPED draw's measurement outcome, never the pool's. A candidate
        # that was discarded took its faults with it.
        rate_entries = list(best.rate_entries)
        if narration_verdict(best.estimate, ordered) == "ok":
            return _finalize(best.output, degradations=rate_entries, estimate=estimate)

        detail = f"first draw estimated {first_estimate:.1f}s; final {best.estimate:.1f}s."
        advisories = [
            narration_estimate_advisory(
                stage=self.name,
                ordered_seconds=ordered,
                estimated_seconds=best.estimate,
                action=action,
                detail=detail,
            )
        ]
        degradations = list(rate_entries)
        if floored is not None:
            degradations.append(trim_floor_degradation(stage=self.name, plan=floored))
        degradations.extend(self._short_floor_entries(best, ordered, action, detail))
        return _finalize(
            best.output, degradations=degradations, advisories=advisories, estimate=estimate
        )

    async def _first_reask_step(
        self,
        best: _Draw,
        pool: list[_Draw],
        ordered: float,
        payload: WebsiteReviewScriptInput,
        ctx: RunContext,
    ) -> tuple[_Draw, str, bool, bool | None]:
        """ATTEMPT ONE — the directional re-ask, with EVERY outcome named.

        Returns ``(best, action, reasked, reask_kept)``. A call that FAILED or came
        back empty used to leave ``action`` reading "kept first draw" — the same
        string a draw that never needed correcting carries — so a failed correction
        was invisible in the persisted record. It now says so.
        """
        second = await self._reask(best, ordered, payload, ctx)
        if second is None:
            return best, "re-ask failed; kept first draw", False, None
        if not second.lines:
            # SENT and DISCARDED: ``reask_kept=None`` documents "no re-ask
            # happened", which paired with ``reasked=True`` is a state the field's
            # own contract forbids (app_review and promo already record False).
            return best, "re-ask came back empty; kept first draw", True, False
        measurement = await self._measure(second, ctx)
        self._wpm = measurement.wpm
        self._wpm_basis = measurement.basis
        candidate = self._draw_of(second, measurement)
        pool.append(candidate)
        if closer_to_order(candidate.estimate, best.estimate, ordered):
            return candidate, "re-asked", True, True
        return best, "re-ask not closer; kept first draw", True, False

    def _short_floor_entries(
        self, best: _Draw, ordered: float, action: str, detail: str
    ) -> list[Degradation]:
        """THE SHORT-SIDE FLOOR: a draw shipping below the band is a DEGRADATION.

        Every ask is spent (or failed) and the draw is still below the band's lower
        edge. The film IS its narration, so this is the delivered length, not a
        steering note: it goes on the tier that flips the run ``degraded`` — the
        same outcome app_review records for a long draw no lever could move
        (``trim_floor_degradation``). NOT a raise: a paid run holding a usable
        script must not die because a correction did not land. The
        ``narration_estimate_off_target`` advisory is kept alongside, unchanged.
        """
        if narration_verdict(best.estimate, ordered) != "short":
            return []
        return [
            narration_degradation(
                stage=self.name,
                ordered_seconds=ordered,
                estimated_seconds=best.estimate,
                action=action,
                detail=(
                    f"{detail} STILL SHORT — re-asked for at least "
                    f"{short_draw_floor_words(ordered, floor_wpm(best.wpm, best.basis))} "
                    "words, shipped "
                    f"{count_words([line.text for line in best.output.lines])}; "
                    "nothing lengthens a short script mechanically."
                ),
            )
        ]

    def _trim_step(
        self,
        best: _Draw,
        pool: list[_Draw],
        ordered: float,
        ctx: RunContext,
        action: str,
    ) -> tuple[_Draw, str, TrimPlan | None, int]:
        """The trim, accepted only if it improves — and the floor if it did not land.

        **It trims any LONG candidate in the pool, not only the incumbent.** When a
        re-ask has swung a long draft to a shorter-but-still-wrong one, that shorter
        draw wins on "closer" and a trim of it is then impossible (nothing shortens
        a short script) — so the loop used to stop with both drafts out of band
        while one of them was one dropped line from conformant. Trimming a candidate
        costs no synthesis and no LLM call. The incumbent is tried FIRST, and ties
        keep it, so a single-candidate run is unchanged.

        ``floored`` is the plan when the band is STILL out of reach afterwards,
        which "``+ trimmed``" alone reads exactly like a landed trim — the live
        ``website_review_v1`` report that said "trimmed 2 line(s)" while delivering
        77.46s against a 60s order.

        **A plan that was NOT APPLIED is never reported as a floor.** A rejected
        trim leaves the script exactly as it was, so the plan's
        ``estimated_seconds_after`` describes a script that was never shipped. A
        plan that dropped NOTHING is not a rejection: there was nothing to apply,
        and the floor it names is the truth about this script.
        """
        incumbent = best
        sources = [
            draw
            for draw in [incumbent, *(d for d in pool if d is not incumbent)]
            if steer_verdict(draw.estimate, ordered) == "long"
        ]
        if not sources:
            return best, action, None, 0

        results = [(source, *self._apply_trim(source, ordered, ctx)) for source in sources]
        winner: tuple[_Draw, _Draw, TrimPlan] | None = None
        for source, trimmed, plan, _ in results:
            if trimmed is None or not closer_to_order(trimmed.estimate, best.estimate, ordered):
                continue
            if winner is None or closer_to_order(trimmed.estimate, winner[1].estimate, ordered):
                winner = (source, trimmed, plan)

        if winner is not None:
            source, trimmed, plan = winner
            dropped = len(source.output.lines) - len(trimmed.output.lines)
            scope = "" if source is incumbent else f" the {_pool_label(source, pool)}"
            action = f"{action} + trimmed{scope} {dropped} line(s)"
            if plan.exceeds_limit and narration_verdict(trimmed.estimate, ordered) == "long":
                return trimmed, f"{action}; {_floor_note(plan)}", plan, dropped
            return trimmed, action, None, dropped

        # Nothing was adopted. The report follows the INCUMBENT's own plan — the
        # first source, which is the incumbent whenever the incumbent is long —
        # because that is the script that actually ships.
        _, trimmed, plan, rejected = results[0]
        if trimmed is not None:
            rejected = "it landed no closer to the ordered duration"
        if rejected is not None:
            return best, f"{action}; trim REJECTED — {rejected}", None, 0
        if not (plan.exceeds_limit and narration_verdict(best.estimate, ordered) == "long"):
            return best, action, None, 0
        return best, f"{action}; {_floor_note(plan)}", plan, 0

    def _apply_trim(
        self, draw: _Draw, ordered: float, ctx: RunContext
    ) -> tuple[_Draw | None, TrimPlan, str | None]:
        """The trim plan applied, or ``(None, ...)`` when it must not be applied.

        The PLAN is always returned as the third element, applied or not: whether
        the band was reachable is a property of the plan, and a caller that only
        ever saw the applied output could not tell a floored trim from a landed one.
        The fourth element is the REJECTION, in the caller's words.

        **It plans on MEASURED seconds, not on a word proxy.** Two lines of equal
        word count do not take equal time — and on a two-hander they are not even
        spoken by the same voice, so a word-weighted plan can drop the wrong pair
        and land outside the band it was aiming at. ``line_seconds`` is handed to
        ``plan_trim`` whenever the measurement is usable; it falls back to words
        when it is not, which is the pre-measurement behaviour exactly.

        ``_protected_indices`` pins the authored before act, the reveal line and
        every line carrying a required verbatim phrase; the verbatim result is
        re-checked afterwards anyway, because a script that silently lost the
        user's exact words is worse than a long one.
        """
        plan = plan_trim(
            [line.text for line in draw.output.lines],
            ordered_seconds=ordered,
            # THIS draw's rate, never ``self._wpm``. ``self._wpm`` holds whatever
            # was measured LAST, which after a re-ask is the rate of a draft that
            # may have been discarded — and ``plan_trim`` falls back to the word
            # proxy the moment ``_trimmable_seconds`` refuses the measurement, at
            # which point the wrong rate silently decides which lines are cut.
            # Pinned by
            # ``test_the_trim_plans_at_the_shipped_draws_own_rate_not_the_last_measured_one``.
            words_per_minute=draw.wpm,
            # The OUTER +/-10% band, deliberately — NOT the 4% the steer aims at.
            # The steer decides WHETHER to correct, keeping headroom for the
            # estimate's own error; the trim decides what a cut is allowed to LAND
            # on, and it declines any drop falling below ``ordered x (1 -
            # tolerance)``. Narrowing this to the steer band makes it refuse cuts it
            # should take — the only available drop lands at, say, -8%, which is
            # conformant, and the script then ships its full overrun instead.
            #
            # The known cost, accepted: ``plan_trim`` stops at the FEWEST drops that
            # land inside this band rather than at the closest possible landing, so
            # a deeper cut nearer the order can be passed over. ``app_review`` made
            # the same trade deliberately. Pinned by
            # ``test_the_trim_aims_at_the_outer_band_not_at_the_steer_band``.
            tolerance=DEFAULT_TOLERANCE,
            protected_indices=_protected_indices(draw.output.lines, ctx),
            line_seconds=_trimmable_seconds(draw),
        )
        if not plan.dropped_indices:
            return None, plan, None
        kept = draw.select(plan.keep_indices)
        items = items_for_stage(ctx.config_snapshot.get("verbatim_items") or [], self.name)
        if find_missing_verbatim(" ".join(line.text for line in kept.output.lines), items):
            logger.warning("%s: trim would drop a required verbatim phrase — skipped", self.name)
            return None, plan, "it would have dropped a required verbatim phrase"
        return kept, plan, None

    def build_prompt(self, ctx: RunContext) -> str:
        # The redesign is the video's visual spine (the before→after reveal), so it
        # is required. The critique is OPTIONAL: a redesign-only-plus-video run
        # (generate_critique=false) has no findings, and the script simply walks the
        # redesign without finding-by-finding commentary.
        redesign = ctx.prior_output("website_redesign")
        if not redesign:
            raise RuntimeError("website_redesign output required for the review script")
        critique = ctx.prior_output("website_critique") or {}
        brief = ctx.config_snapshot.get("brief") or {}
        url = critique.get("url") or brief.get("target_url") or ""
        target_s = int(_target_seconds(ctx.config_snapshot))
        voice = ctx.config_snapshot.get("voice", {}) or {}
        items = items_for_stage(ctx.config_snapshot.get("verbatim_items") or [], self.name)
        notes = ctx.annotations_for("website_review_script")
        notes_block = (
            "\n\nApply reviewer notes:\n" + "\n".join(f"- {n['text']}" for n in notes)
            if notes
            else ""
        )
        findings = [
            {
                "id": f.get("id"),
                "category": f.get("category"),
                "severity": f.get("severity"),
                "title": f.get("title"),
                "recommendation": f.get("recommendation"),
            }
            for f in headline_findings(critique.get("findings") or [])
        ]
        dialogue_on = dialogue_cast(voice, brief) is not None
        personas = (brief.get("dialogue") or {}).get("personas") or {}
        persona_a = personas.get("a") or "a sharp reviewer who points out what's wrong"
        persona_b = personas.get("b") or "the designer who explains how the redesign fixes it"
        mode_block = (
            (
                "Write this as a natural TWO-PERSON conversation, not a monologue:\n"
                f'- speaker "a" = {persona_a} — asks pointed questions about each problem.\n'
                f'- speaker "b" = {persona_b} — answers, concrete and grounded.\n'
                'Tag every line with its `speaker` ("a" or "b") — the tag decides which '
                "voice reads it. Most ASK lines end with a question mark.\n"
            )
            if dialogue_on
            else 'This is a SINGLE narrator. Set "speaker" to "a" on every line.\n'
        )
        # When the operator authored a shotlist, the BEFORE act is fixed — the LLM
        # writes ONLY the AFTER (redesign) act; otherwise it writes both acts.
        shots = normalize_shotlist(brief)
        structure_block = _after_only_structure_block(shots) if shots else _TWO_ACT_STRUCTURE_BLOCK
        inventory = self._inventory_for(ctx)
        return (
            f"Write a {target_s}-second website review voiceover script for "
            f"{url}.\n"
            f"{word_budget_sentence(target_s, self._wpm)}"
            f"{self._minimum_sentence(ctx)}"
            f"{language_instruction()}"
            f"{_tone_line(voice)}"
            f"{_critique_header(critique)}"
            f"Key findings (id, category, severity, title, recommendation): {findings}\n"
            f"{_redesign_line(redesign)}"
            f"{inventory.prompt_block()}"
            f"{mode_block}"
            f"{structure_block}"
            f"{_section_rule(inventory, self._after_allowance(ctx), authored_before=len(shots))}"
            "For each line, include the critique finding id it references when relevant.\n"
            'Output JSON: {"lines":[{"text":"...","seconds":<num>,'
            '"critique_item_id":"..."|null,"phase":"before"|"after",'
            '"speaker":"a"|"b","section_id":"..."|null}, ...]}.'
            f"{notes_block}"
            f"{grounding_prompt_block(ctx)}"
            f"{verbatim_prompt_block(items)}"
            f"{self._grounding_block}"
            f"{self._reask_block}"
        )

    def _first_floor_words(self, ctx: RunContext) -> int:
        """This run's first-draw word minimum, at the CONTENT-basis rate (:func:`floor_wpm`)."""
        shots = normalize_shotlist(ctx.config_snapshot.get("brief") or {})
        return first_draw_floor_words(
            int(_target_seconds(ctx.config_snapshot)),
            floor_wpm(self._wpm, self._wpm_basis),
            authored_words=count_words([shot.narration for shot in shots]),
        )

    def _minimum_sentence(self, ctx: RunContext) -> str:
        """This run's first-draw minimum, from the numbers ``build_prompt`` budgets with."""
        return first_draw_minimum_sentence(self._first_floor_words(ctx))

    def under_floor_retry_block(self, error: ScriptDrawUnderFloorError, ctx: RunContext) -> str:
        """The primitive's corrective block, plus this stage's minimums as COUNTS.

        Live 2026-10-06 (runs ``01a11125`` and the r3b kill-capture run): attempt two
        was attempt one plus a paragraph restating "at least ONE line … at least N
        words" — and came back ``{"lines": []}`` again at temperature 0.1. The words
        were already there; what the retry never said is how many LINES that takes
        and how they split across the acts. :func:`minimum_line_plan` turns the word
        minimum into those counts, inside the prompt's own line cap, so the retry is a
        different, concrete ask rather than the same rule louder. (No sampling knob
        can change instead: ``WMClient.message`` carries no temperature and
        worker-management's ``MessageRequest`` accepts none per message.)
        """
        shots = normalize_shotlist(ctx.config_snapshot.get("brief") or {})
        floor = self._first_floor_words(ctx)
        cap = max_lines_by_rules(
            len(self._inventory_for(ctx).sections),
            authored_before=len(shots),
            after_allowance=self._after_allowance(ctx),
        )
        return (
            f"{super().under_floor_retry_block(error, ctx)} {self._minimum_sentence(ctx).rstrip()}"
            f"{minimum_line_plan(floor, cap=cap, authored_before=bool(shots))}"
        )

    def make_element(self, raw: dict, *, element_id: str) -> ReviewLine:
        return ReviewLine(
            id=element_id,
            text=str(raw["text"]),
            seconds=float(raw["seconds"]),
            critique_item_id=raw.get("critique_item_id"),
            phase=(str(raw.get("phase") or "before").strip().lower() or "before"),
            speaker=(str(raw.get("speaker") or "a").strip().lower() or "a"),
            section_id=(str(raw.get("section_id") or "").strip() or None),
        )

    def build_output(
        self, elements: list[StageOutput], ctx: RunContext
    ) -> WebsiteReviewScriptOutput:
        brief = ctx.config_snapshot.get("brief") or {}
        after = cast("list[ReviewLine]", elements)
        shots = normalize_shotlist(brief)
        if shots:
            # Authored BEFORE act (deterministic, verbatim, line_id = shot-NNN) +
            # AI-generated AFTER act. The boundary is KNOWN (the end of the authored
            # block), so skip the narration-pivot detection (_assign_phases) the
            # fully-AI path needs: prepend the before lines and pin every LLM line
            # to "after". The before line ids match site_capture's per-screen
            # capture_ids so the compile binds each before line to its own screen.
            before = [_shot_to_before_line(s) for s in shots]
            lines = [*before, *(line.model_copy(update={"phase": "after"}) for line in after)]
        else:
            # Phase first (the split point), THEN derive speakers within line order
            # so the pivot question lands at the before→after boundary.
            lines = _assign_phases(after)
        # ONE dialogue predicate for the prompt, the speakers and record's casting:
        # a two-hander only when explicitly enabled AND two distinct voices resolve.
        if dialogue_cast(ctx.config_snapshot.get("voice") or {}, brief) is not None:
            lines = assign_dialogue_speakers(lines)
        else:
            lines = [line.model_copy(update={"speaker": "a"}) for line in lines]
        lines = _keep_rendered_sections(lines, self._inventory_for(ctx))
        target_seconds = _target_seconds(ctx.config_snapshot)
        return WebsiteReviewScriptOutput(
            lines=lines,
            total_seconds=sum(line.seconds for line in lines),
            word_budget=word_budget(target_seconds, self._wpm),
            actual_words=count_words([line.text for line in lines]),
            # Same snapshot, same call, as `target_seconds` above (`order.py`:
            # budget = order - reserve_title_cards().total_seconds), so the record
            # IS the subtraction, not a description of it.
            reserved_title_cards=reserve_title_cards(ctx.config_snapshot),
        )


#: How many critique findings the prompt carries.
MAX_PROMPT_FINDINGS: Final = 12

_SEVERITY_RANK: Final = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def headline_findings(findings: Sequence[object]) -> list[dict[str, object]]:
    """The :data:`MAX_PROMPT_FINDINGS` most SEVERE findings, severity first.

    Sliced AFTER sorting: the critique returns findings grouped by category, not by
    severity, so a plain ``[:12]`` handed the prompt twelve seo/performance lows
    and dropped prod run 01a0a054's single ``critical`` ("No contact form on
    homepage", index 36). Stable within a severity; an unknown severity sorts last.
    """
    usable = [f for f in findings if isinstance(f, dict)]
    ranked = sorted(
        usable,
        key=lambda f: _SEVERITY_RANK.get(str(f.get("severity") or "").strip().lower(), 5),
    )
    return ranked[:MAX_PROMPT_FINDINGS]


def _tone_line(voice: Mapping[str, object]) -> str:
    """The voice/tone instruction: the descriptor AND ``voice.direction.tone``.

    The Studio writes the operator's tone to ``voice.direction.tone`` (prod run
    01a0a054: ``confident``); reading only ``descriptor`` sent every such run to
    the model as "neutral".
    """
    descriptor = str(voice.get("descriptor") or "").strip()
    direction = voice.get("direction")
    tone = str(direction.get("tone") or "").strip() if isinstance(direction, dict) else ""
    if descriptor and tone:
        return f"Voice/tone: {descriptor}; tone: {tone}.\n"
    return f"Voice/tone: {descriptor or tone or 'neutral'}.\n"


def after_line_allowance(
    sections: int, budget_words: int, *, authored_before: int = 0, authored_words: int = 0
) -> int:
    """How many after-act lines the prompt allows — ``0`` when no sections rendered.

    ``max(sections, ceil(after_words / COMFORTABLE_LINE_WORDS))``, where
    ``after_words`` is the share of the order's word budget the after act has to
    be ABLE to carry: :data:`MIN_AFTER_SHARE` of it on the fully-AI path (the same
    "at least HALF" the structure block states), all of what the authored before
    act left when the model writes the after act alone.

    One line per section is the rule whenever the page has enough sections for the
    order — the first term wins and nothing changes. It is the FEW-section page the
    second term exists for: a one-section page on a 50s order was capped at ONE
    after-act line, hence two lines in all, under a ~150-word minimum — two ~75-word
    "lines" about a section known by its slug. Formally satisfiable, and the model
    answered ``{"lines": []}`` on every attempt (run 01a11125, 2026-10-06); the same
    scenario had passed that morning only by BREAKING the rule (two after-act lines
    for one section). With this allowance ``max_lines_by_rules x
    COMFORTABLE_LINE_WORDS`` always covers the word budget, so the stated minimum is
    reachable at a line length narration is actually delivered at.
    """
    if sections <= 0:
        return 0
    share = 1.0 if authored_before else MIN_AFTER_SHARE
    after_words = max(0, budget_words - max(0, authored_words)) * share
    return max(sections, math.ceil(after_words / COMFORTABLE_LINE_WORDS))


def _section_rule(
    inventory: RedesignInventory, allowance: int | None = None, *, authored_before: int = 0
) -> str:
    """The after-act ↔ rendered-section contract, when sections were rendered.

    ``allowance`` above the section count (:func:`after_line_allowance`) is the
    few-section page: the rule then says plainly that several lines may cover the
    same section, and states the act split as COUNTS the model can check itself
    against. Otherwise — every page with enough sections for its order — the text is
    the one-line-per-section rule, byte for byte.
    """
    if not inventory.sections:
        return ""
    n = len(inventory.sections)
    if allowance is not None and allowance > n:
        total = max_lines_by_rules(n, authored_before=authored_before, after_allowance=allowance)
        split = (
            ""
            if authored_before
            else (
                f" Counted out: up to {allowance} after-act lines, so up to {allowance} "
                f"before-act lines and {total} lines in all — the after act still "
                "carries at least half of whatever number you write."
            )
        )
        page = "only ONE section" if n == 1 else f"only {n} sections"
        return (
            f"AFTER-ACT RULES: the after act walks the rendered sections above. This page "
            f"has {page}, and this order needs more narration than one line per section "
            f"can carry — so SEVERAL after-act lines may describe the SAME section, each "
            f"making a different point about it. Write at most {allowance} after-act "
            f'lines, in page order, and set each after-act line\'s "section_id" to the '
            "section_id it describes — only ids from the list; repeating an id across "
            f'lines is expected here. Before-act lines set "section_id" to null.{split}\n'
        )
    return (
        f"AFTER-ACT RULES: the after act walks the rendered sections above. Write at most "
        f"ONE after-act line per section (at most {n} after-act lines), in page order, "
        'and set each after-act line\'s "section_id" to the section_id it describes — '
        'only ids from the list. Before-act lines set "section_id" to null.\n'
    )


def max_lines_by_rules(
    sections: int, *, authored_before: int = 0, after_allowance: int | None = None
) -> int | None:
    """The most lines the prompt's OWN rules let the model write — ``None`` = uncapped.

    Two stated rules, and nothing else, bound the count: :func:`_section_rule`
    (at most ``after`` after-act lines — one per section, or the order-scaled
    ``after_allowance`` when that is larger, see :func:`after_line_allowance`) and,
    on the fully-AI path only, ``_TWO_ACT_STRUCTURE_BLOCK`` ("at least HALF of all
    lines are after-act lines", :data:`MIN_AFTER_SHARE`). So ``after`` after-act
    lines at most, and at most as many before-act lines again. With an authored
    shotlist the model writes the after act alone. No sections rendered ⇒ no
    ``AFTER-ACT RULES`` block ⇒ no cap.

    The prompt states NO per-line word bound, so no cap is ever formally
    unsatisfiable — but "formally" is how a one-section page got a two-line cap
    under a ~150-word minimum and an empty draw. Passed the allowance the prompt
    states, this cap times :data:`COMFORTABLE_LINE_WORDS` always covers the word
    budget (``tests/.../test_script_rules_satisfiable.py``).
    """
    if sections <= 0:
        return None
    after = max(sections, after_allowance or 0)
    if authored_before:
        return after
    return math.floor(after / MIN_AFTER_SHARE)


def _keep_rendered_sections(
    lines: list[ReviewLine], inventory: RedesignInventory
) -> list[ReviewLine]:
    """Clear any ``section_id`` that is not a rendered section on an after-act line."""
    valid = inventory.section_ids
    return [
        line
        if line.section_id is None or (line.phase == "after" and line.section_id in valid)
        else line.model_copy(update={"section_id": None})
        for line in lines
    ]


def _grounding_reask_block(
    problems: Sequence[str], *, shared_sections: bool = False, band_only: bool = False
) -> str:
    """The addendum for the ONE grounding re-ask: each violation, stated concretely.

    ``shared_sections`` is the few-section page (:func:`after_line_allowance` above
    the section count): the re-ask must not take back the "several lines may
    describe the same section" the prompt above it just granted.

    ``band_only`` (``grounding.inventory_is_band_only``): every section is a
    content-less ``band-NN`` viewport band, so the re-ask must not repeat "ONLY
    sections and elements listed" — nothing nameable is listed. It permits general visual terms screen by screen instead, and
    says an empty script is never the fix; the violations above still bind.
    """
    listed = "\n".join(f"- {p}" for p in problems)
    per_section = (
        "several lines may share a section, each with its section_id"
        if shared_sections
        else "one line per section, each with its section_id"
    )
    after_rule = (
        "After-act lines walk the redesigned page SCREEN BY SCREEN in the listed "
        "section order: the sections carry no named headings or elements, so describe "
        "each screen in general visual terms (layout, hierarchy, spacing, typography, "
        f"colour) without naming a feature the list does not show, {per_section}. An "
        "empty script is never a correct answer."
        if band_only
        else "After-act lines describe ONLY sections and elements listed under "
        f"REDESIGN CONTENT, {per_section}."
    )
    return (
        "\n\nREWRITE — your previous draft narrated things the video will NOT show:\n"
        f"{listed}\n"
        "Rewrite the complete script. Before-act lines state PROBLEMS with the current "
        f"page only. {after_rule} Return the full script again in the same JSON shape."
    )


def short_draw_floor_words(ordered_seconds: float, wpm: float) -> int:
    """The fewest words that reach the band's LOWER edge at ``wpm``.

    ``budget_words`` of ``ordered x (1 - tolerance)`` — the shared arithmetic, at
    the edge :func:`narration_verdict` grades "short" from — plus one word so its
    rounding can never state a floor that itself lands under the band. Capped at
    the full-order budget so the floor is never above the target it accompanies.
    """
    edge = budget_words(float(ordered_seconds) * (1.0 - DEFAULT_TOLERANCE), words_per_minute=wpm)
    return max(0, min(edge + 1, budget_words(ordered_seconds, words_per_minute=wpm)))


def floor_wpm(wpm: float, basis: str) -> float:
    """The rate a word FLOOR is stated at — never a calibration-passage rate as such.

    A Phase-A rate (:data:`~speech_rate.CALIBRATION_PASSAGE`) measures the fixed
    calibration passage, which the speech-rate module documents reads ~47% FASTER
    than real narration (175.2 vs 119.4 wpm on one voice). As a CEILING that errs
    safe — the reconcile measures the draft and trims a long one for free. As a
    FLOOR it over-asks: live 2026-10-06 the passage measured ~198 wpm, the 50s
    order's floor became 149 words, and the content then measured 138-140 wpm, so
    the minimum alone was a ~64s script — a floor that forbids an in-band answer.

    So a calibration-basis rate is held to at most :data:`DEFAULT_WPM_NATURAL`, the
    narration default the module already falls back to. Every other basis already IS
    content (``measured_all``/``measured_partial``/``script_sample``) or is the
    operator's word (``authored``) and passes through unchanged; so does a
    calibration rate slower than the default, since a lower floor never over-asks.
    """
    if basis == CALIBRATION_PASSAGE:
        return min(float(wpm), DEFAULT_WPM_NATURAL)
    return float(wpm)


def minimum_line_plan(floor_words: int, *, cap: int | None, authored_before: bool) -> str:
    """The word minimum as LINE COUNTS a retry can check itself against — ``""`` if none.

    ``ceil(floor_words / COMFORTABLE_LINE_WORDS)`` lines (one delivered-narration
    sentence each), never above the prompt's own rule cap (:func:`max_lines_by_rules`,
    which :func:`after_line_allowance` sizes to cover the budget) and never below
    one. On the fully-AI path the count is split so the after act carries at least
    :data:`MIN_AFTER_SHARE`; with an authored before act every line is after-act.
    """
    if floor_words < 1:
        return ""
    lines = max(1, math.ceil(floor_words / COMFORTABLE_LINE_WORDS))
    if cap is not None:
        lines = max(1, min(lines, cap))
    if authored_before:
        split = f"all {lines} of them after-act lines"
    else:
        after = math.ceil(lines * MIN_AFTER_SHARE)
        split = f"{lines - after} before-act line(s) then {after} after-act line(s)"
    return (
        f" Concretely: {floor_words} words takes at least {lines} lines — {split}. "
        f'Return a "lines" list with at least {lines} elements.'
    )


#: Why a stated minimum is binding (it once had to outrank a ceiling sentence's "write
#: fewer if the story is done sooner"; the budget is now a target band, but the
#: minimum still states the cost of missing it). ONE wording, shared by the first
#: draw's minimum (:func:`first_draw_minimum_sentence`) and the short-draw re-ask's
#: (:func:`_reask_prompt_block`), so the prompt never states the same rule two ways.
_HARD_MINIMUM_OVERRIDES: Final = (
    "This minimum is binding — a script under it is delivered as a film shorter than "
    "the customer ordered. Reach it with more grounded material about the findings "
    "and the redesign, never with filler or repetition."
)


def first_draw_floor_words(target_seconds: float, wpm: float, *, authored_words: int = 0) -> int:
    """The word minimum the FIRST draw states — ``0`` when none can honestly be stated.

    :func:`short_draw_floor_words` (the band's lower edge, the same number the
    short-draw re-ask floors with), held STRICTLY under the ceiling
    :func:`word_budget_sentence` states for the same ``target_seconds`` and ``wpm``
    — that helper only caps its floor AT the budget, and on a tiny order the two
    would otherwise be the same number, or the floor the larger. Net of
    ``authored_words``: with an operator shotlist the model writes the after act
    alone, and the authored before act already counts toward the order.
    """
    ceiling = word_budget(target_seconds, wpm)
    floor = min(short_draw_floor_words(target_seconds, wpm), ceiling - 1)
    return max(0, floor - max(0, authored_words))


def first_draw_minimum_sentence(floor_words: int) -> str:
    """The LOWER bound of the first draw — the only rule an empty answer breaks.

    The other length rules in the prompt were upper bounds (the word CEILING — now a
    target band — and "at most ONE after-act line per section"), so ``{"lines": []}`` violated none of
    them — and that is what the model answered on every attempt of the 2026-10-06
    website_review E2E. The output schema cannot carry the floor (worker-management
    strips ``minItems``), so it is stated here, in the re-ask's own vocabulary.
    """
    words = f" — and at least {floor_words} words in total" if floor_words >= 1 else ""
    return (
        'HARD MINIMUM: you must write at least ONE line — an empty "lines" list is '
        f"never a valid answer{words}. {_HARD_MINIMUM_OVERRIDES}\n"
    )


def _reask_prompt_block(
    *,
    produced_words: int,
    estimated_seconds: float,
    ordered_seconds: float,
    target_words: int,
    floor_words: int | None = None,
) -> str:
    """The addendum for the ONE bounded re-ask: the miss, stated as COUNTABLE facts.

    Everything here is a number the model can check itself against. The first draft
    already carried the word ceiling as prose and missed it; repeating the prose
    would re-run the same failure, so this states what was produced, what it costs
    in seconds, and what the order affords.
    """
    verb = "TOO LONG" if estimated_seconds > ordered_seconds else "TOO SHORT"
    floor = (
        ""
        if floor_words is None
        else (
            f" HARD MINIMUM: the rewritten script must contain at least {floor_words} "
            f"words in total. {_HARD_MINIMUM_OVERRIDES}"
        )
    )
    return (
        "\n\nREWRITE — your previous draft was "
        f"{verb}. It contained {produced_words} words, which at this narrator's "
        f"measured pace takes about {estimated_seconds:.0f} seconds to say. The "
        f"order is {ordered_seconds:.0f} seconds, i.e. {target_words} words for the "
        "WHOLE script (all lines combined). Rewrite the complete script to "
        f"approximately {target_words} words — keep the same structure and the same "
        "story, cut or expand the CONTENT of the lines rather than padding with "
        f"filler or repetition.{floor} Return the full script again in the same JSON shape."
    )


def _with_lines(
    output: WebsiteReviewScriptOutput, lines: list[ReviewLine]
) -> WebsiteReviewScriptOutput:
    """``output`` with ``lines`` replaced and every derived total re-derived.

    ``total_seconds`` and ``actual_words`` are sums OVER the lines; leaving them
    at their pre-trim values would make the stage output self-contradicting, and
    ``total_seconds`` is what downstream reporting reads as the script's length.
    """
    return output.model_copy(
        update={
            "lines": lines,
            "total_seconds": sum(line.seconds for line in lines),
            "actual_words": count_words([line.text for line in lines]),
        }
    )


def _protected_indices(lines: list[ReviewLine], ctx: RunContext) -> tuple[int, ...]:
    """Line indices the trim may NOT drop, because the picture is keyed off them.

    ``website_review/compile.py`` binds visuals to lines structurally, so a trim
    that removes the wrong line does not shorten the video — it breaks it:

    * **The authored BEFORE act** (``_shot_to_before_line``, ids ``shot-NNN``).
      These are the operator's own words, and site_capture captures one live
      screen per shot under the SAME id — dropping one orphans its screen and
      deletes text the operator wrote by hand. They are the first ``len(shots)``
      lines (``build_output`` prepends them).
    * **The reveal line** — the FIRST ``phase == "after"`` line.
      ``compile._key_lines`` puts the original page behind "before" lines and the
      redesign behind "after" lines, so if every "after" line is trimmed away the
      redesign — the entire point of the video — never appears on screen.
      ``plan_trim`` drops last-first, i.e. straight through the after act, so
      without this pin a hard overrun would deliver a critique with no redesign.
    * **Any line carrying a required verbatim phrase**, which
      ``assert_verbatim_present`` has already gated the draw on. Trimming it here
      would silently ship the video without a phrase the user marked required —
      the exact outcome that gate exists to prevent.

    NOT protected: the per-line question/answer alternation. ``_key_live_screens``
    advances the live screen on answer lines and holds through questions, but it
    resolves EVERY before line to some screen (modulo the pool), so a shorter
    before act shows fewer screens — which is what shortening means — rather than
    leaving a line without a visual.
    """
    protected: set[int] = set()
    brief = ctx.config_snapshot.get("brief") or {}
    authored = len(normalize_shotlist(brief))
    protected.update(range(authored))
    after = [i for i, line in enumerate(lines) if line.phase == "after"]
    if after:
        protected.add(after[0])
    if after and not authored and len(after) * 2 < len(lines):
        # Fully-AI path, and the redesign is ALREADY a minority of the script.
        # ``plan_trim`` prefers the LATEST lines, i.e. the after act, so without this
        # a trim widens the imbalance — prod run 01a0a054 shipped 5 before lines over
        # 1 after line. Trim the critique instead. (On the shotlist path the before
        # act is authored and pinned above, so pinning the after act too would leave
        # nothing to trim.)
        protected.update(after)
    items = items_for_stage(
        ctx.config_snapshot.get("verbatim_items") or [], WebsiteReviewScriptStage.name
    )
    if items:
        protected.update(
            i
            for i, line in enumerate(lines)
            if len(find_missing_verbatim(line.text, items)) < len(items)
        )
    return tuple(sorted(protected))


# The fully-AI path writes BOTH acts in one draft; the narration-content pivot
# (_assign_phases) then finds the before→after boundary.
_TWO_ACT_STRUCTURE_BLOCK = (
    "Structure the script as TWO acts and tag each line with its `phase`:\n"
    '- Phase "before": the CURRENT design\'s headline problems (the critique '
    "findings). These play OVER THE ORIGINAL PAGE, so state PROBLEMS ONLY — never "
    "describe a fix, a change or the redesign in this act (the old page is on "
    "screen). Keep it tight.\n"
    '- Phase "after": present the redesign. These play OVER THE REDESIGNED PAGE. '
    "The after act is the payoff: at least HALF of all lines are after-act lines.\n"
    'Do NOT interleave the phases — emit all "before" lines first, then the '
    'reveal, then all "after" lines. The REVEAL line must clearly PRESENT '
    'the redesign and contain the word "redesign" or "new design/layout" '
    '(e.g. "Here is the new redesign…", "Take a look at the redesign…"). That '
    'reveal line and everything after it is phase "after" — the redesigned '
    "page is on screen from that line onward, so do NOT keep narrating the "
    'redesign while still tagged "before". A line that only ASKS to see it '
    '("show me the redesign") stays "before" — the reveal is your NEXT line.\n'
)


def _after_only_structure_block(shots: list[ShotSpec]) -> str:
    """Prompt the LLM to write ONLY the AFTER act when the BEFORE act is authored.

    The operator's shotlist IS the before act; passing it as read-only context
    keeps the AI continuation coherent without letting it rewrite or repeat the
    authored lines. The first AI line must PRESENT the redesign — it is the moment
    the redesigned page appears on screen.
    """
    before = "\n".join(f'  - "{s.narration}"' for s in shots)
    return (
        "The BEFORE act (critiquing the CURRENT design over the original page) is "
        "ALREADY AUTHORED by the operator — do NOT rewrite, repeat or re-tag it. "
        "For continuity, these before-act lines play first, in order:\n"
        f"{before}\n"
        'Write ONLY the AFTER act and tag EVERY line `phase`:"after". Your FIRST '
        'line must clearly PRESENT the redesign (contain the word "redesign" or '
        '"new design/layout", e.g. "Here is the new redesign…") — it is the moment '
        "the redesigned page appears on screen. Then walk through how the redesign "
        "fixes the headline problems, picking up naturally where the authored "
        "before act leaves off.\n"
    )


def _shot_to_before_line(shot: ShotSpec) -> ReviewLine:
    """One authored shot → a verbatim BEFORE-act line keyed by its shot id.

    ``id == shot.line_id`` (shot-NNN) is the cross-stage join key: site_capture
    captures each URL shot under the same id, so website_review_compile binds this
    line to its own screen, and website_review_slides keys slide shots the same way.
    """
    return ReviewLine(
        id=shot.line_id,
        text=shot.narration,
        seconds=shot.seconds,
        critique_item_id=None,
        phase="before",
    )


# The narration ALWAYS reveals the redesign with a line that PRESENTS it
# ("here is the new redesign", "take a look at the redesign", "we've overhauled…").
# That line is the moment the redesign UI must appear on screen — so it is the
# authoritative before→after boundary, NOT the LLM's per-line `phase` enum (which
# tags the reveal line "before" and the 60/40 fallback, which lands ~2 lines late —
# the cause of the audio-talks-about-redesign-15s-before-it-appears bug).
_REVEAL_RE = re.compile(
    r"\b(here'?s|here is|this is|take a look|let me show|i'?ve|we'?ve|introducing|behold|"
    r"now,? (?:for|onto)|moving on to)\b",
    re.IGNORECASE,
)
_REDESIGN_RE = re.compile(
    r"\b(redesign|new (?:design|layout|version|look|page|site|homepage)|overhaul|rebuil|refreshed)\b",
    re.IGNORECASE,
)
# A line that merely ASKS to see the redesign ("show me the redesign") is still the
# OLD design on screen — the reveal is the NEXT line, so requests never pivot.
_REQUEST_RE = re.compile(
    r"\b(show me|let'?s see|can you show|let me see|what does|how (?:did|have|would) you|"
    r"walk me through|show us)\b",
    re.IGNORECASE,
)


def _find_redesign_pivot(lines: list[ReviewLine]) -> int | None:
    """Index of the line that PRESENTS the redesign (where the after act starts).

    A reveal = a presentation verb + a redesign noun, and NOT a request to see it.
    Returns None when no clear reveal line exists (the caller falls back to a split).
    """
    for i, line in enumerate(lines):
        text = line.text or ""
        if _REDESIGN_RE.search(text) and _REVEAL_RE.search(text) and not _REQUEST_RE.search(text):
            return i
    return None


def _fallback_split(lines: list[ReviewLine]) -> int:
    """Boundary index when there is no detectable reveal line: respect the LLM's
    tags only if it emitted a clean monotonic before→after, else a ~60/40 split."""
    n = len(lines)
    emitted = [line.phase for line in lines]
    if "before" in emitted and "after" in emitted:
        first_after = emitted.index("after")
        monotonic = all(p == "before" for p in emitted[:first_after]) and all(
            p == "after" for p in emitted[first_after:]
        )
        if monotonic and first_after >= 1:
            return first_after
    return max(1, min(n - 1, round(n * 0.6)))


def _assign_phases(lines: list[ReviewLine]) -> list[ReviewLine]:
    """Normalise the per-line ``phase`` into a clean two-act arc: a contiguous run
    of "before" (critiquing the current design) followed by "after" (presenting
    the redesign), with the boundary at the line that REVEALS the redesign.

    Deriving the boundary from the narration content — not the LLM's `phase` enum,
    which it tags unreliably (same lesson as ``_assign_dialogue_speakers``) — keeps
    the visual switch in lockstep with the audio: the redesign UI appears exactly
    when the narrator presents it, instead of ~15s later.
    """
    n = len(lines)
    if n <= 1:
        return lines  # nothing to reveal — a single act stays "before"
    pivot = _find_redesign_pivot(lines)
    if pivot is None or not (1 <= pivot < n):
        pivot = _fallback_split(lines)
    return [
        line.model_copy(update={"phase": "before" if i < pivot else "after"})
        for i, line in enumerate(lines)
    ]


def _critique_header(critique: dict) -> str:
    """``Overall score: … Summary: …`` — each half only when the critique carries it.

    An absent value interpolated straight into the prompt reads ``Summary: None``,
    which the model can (and in the promo sibling, did) narrate verbatim.
    """
    parts = []
    score = text_or_empty(critique.get("overall_score"))
    if score:
        parts.append(f"Overall score: {score}. ")
    summary = text_or_empty(critique.get("summary"))
    if summary:
        parts.append(f"Summary: {summary}")
    return ("".join(parts).rstrip() + "\n") if parts else ""


def _redesign_line(redesign: dict) -> str:
    """The redesign sentence, naming the style only when one was recorded."""
    style = text_or_empty(redesign.get("style"))
    styled = f" (style '{style}')" if style else ""
    return f"A redesigned version{styled} was generated in response to these findings.\n"
