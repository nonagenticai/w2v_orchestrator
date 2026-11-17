"""app_review_script — narrative script tying lines to fact_id / critique_item_id."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final, cast

from w2v_common.stages import StageRetriesExhaustedError

from src.services.stages.app_review.record import AppReviewRecordStage
from src.services.stages.base import (
    SlotUse,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    items_for_stage,
    verbatim_prompt_block,
)
from src.services.stages.primitives.script_stage import LLMScriptStageBase
from src.services.stages.shared.compute_plan import beat_word_budget
from src.services.stages.shared.degradation import Degradation
from src.services.stages.shared.duration_conformance import review_ordered_seconds
from src.services.stages.shared.empty_draw import empty_draw_failure, require_elements
from src.services.stages.shared.narration_budget import (
    DEFAULT_TOLERANCE,
    NarrationEstimate,
    TrimPlan,
    budget_words,
    build_narration_estimate,
    closer_to_order,
    estimate_seconds,
    narration_estimate_advisory,
    narration_verdict,
    plan_trim,
    speech_rate_degradation,
    steer_verdict,
    trim_floor_degradation,
    word_target_sentence,
    wpm_is_usable,
)
from src.services.stages.shared.speech_rate import (
    AUTHORED,
    RESOLUTION_ERROR,
    ScriptMeasurement,
    SpeechRate,
    measure_script_lines,
    resolve_effective_wpm,
    script_measurement_degradation,
)
from src.services.stages.shared.time_budget import DEFAULT_WPM_NATURAL
from src.services.stages.verbatim import find_missing_verbatim

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.clients.tts_client import TTSClient
    from src.services.llm import WMClient
    from src.services.runs.run_context import RunContext

logger = logging.getLogger(__name__)

#: The stage whose ``synthesize_and_store`` call actually speaks these lines.
#: Read off the class rather than repeated as a literal so the two cannot drift: the
#: Phase-B sample is stored under THIS name, and ``deterministic_clip_id`` then makes
#: the sampled clip and the narrated clip one object — see :data:`NARRATE_SPEED`.
NARRATE_STAGE_NAME: Final = AppReviewRecordStage.name

#: The ``speed`` ``app_review_record`` requests: it omits the argument entirely, so
#: the backend default holds and ``deterministic_clip_id`` hashes ``None``. The sample
#: MUST request the same thing — ``1.0`` is a different clip id AND a different
#: measurement (speed scales duration linearly backend-side), so a sample taken at 1.0
#: would both mis-measure the voice and be reused by nothing.
NARRATE_SPEED: Final[float | None] = None

#: How many CORRECTIVE attempts one draw may cost, and in which order they are
#: spent: attempt 1 is the LLM re-ask, attempt 2 the deterministic trim. So the
#: LLM budget for the whole stage is exactly TWO calls — the opening draw and one
#: re-ask — and the second attempt is free, because the trim runs on per-line
#: seconds this stage has already measured. A third draw is not a cheaper way to
#: land in band than trimming the two drafts it already paid for.
MAX_CORRECTIVE_ATTEMPTS: Final = 2


@dataclass(frozen=True, slots=True)
class _Draw:
    """One candidate script AND the per-line seconds it was judged at.

    The two travel together because the whole defect this stage is fixing is a
    draw judged at another draw's rate: the old loop measured a sample of the
    FIRST draft and then graded the re-asked one — different words entirely — at
    that number. Here a candidate cannot be compared without the seconds that
    were measured FOR IT.

    ``line_seconds`` is parallel to ``output.lines``, so a trim of the candidate
    is an index selection on both at once and needs no second measurement.
    """

    output: AppReviewScriptOutput
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
    #: and the text claimed its rate "could not be measured off this script's own
    #: lines", which was false of the shipped draft. Advisories never leaked
    #: between candidates; degradations must not either.
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
        lines = [self.output.lines[i] for i in indices]
        seconds = tuple(self.line_seconds[i] for i in indices)
        return _Draw(
            output=self.output.model_copy(
                update={
                    "lines": lines,
                    "total_seconds": sum(line.seconds for line in lines),
                    "actual_words": count_words([line.text for line in lines]),
                }
            ),
            line_seconds=seconds,
            measured=tuple(self.measured[i] for i in indices),
            estimate=math.fsum(seconds),
            wpm=self.wpm,
            basis=self.basis,
            rate_entries=self.rate_entries,
        )


def _target_seconds(brief: dict) -> float:
    """The ordered review length in seconds — precedence UNCHANGED.

    ``brief.duration_seconds`` (default 120) is the only key this stage has ever
    read; extracted so ``build_output`` re-derives the SAME number the prompt was
    budgeted against.
    """
    return float(int(brief.get("duration_seconds", 120)))


def word_budget(target_seconds: float, wpm: float) -> int:
    """Whole-script word budget = ``duration_s / 60 * wpm``, rounded.

    Thin wrapper over the EXISTING arithmetic in ``compute_plan.beat_word_budget``
    (per-beat only by naming; the formula is the one a whole-script budget needs).
    """
    return round(beat_word_budget(float(target_seconds), float(wpm)).value)


def count_words(texts: list[str]) -> int:
    """Actual spoken word count of the drawn script (whitespace-split)."""
    return sum(len(t.split()) for t in texts)


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


def _carries_verbatim(text: str, items: list[dict]) -> bool:
    """True when ``text`` alone satisfies at least one required verbatim phrase."""
    if not items:
        return False
    return len(find_missing_verbatim(text, items)) < len(items)


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
    output: AppReviewScriptOutput,
    *,
    degradations: list[Degradation],
    advisories: list[Degradation] | None = None,
    estimate: NarrationEstimate | None = None,
) -> AppReviewScriptOutput:
    """``output`` carrying what this run CONCLUDED — on the right three channels.

    ``degradations`` is evidence of a bad delivery and flips the run ``degraded``;
    ``advisories`` is the same entry shape on the tier that does not (a pre-audio
    estimate is not a verdict on audio nobody has heard yet); ``narration_estimate``
    is the numbers themselves, recorded on EVERY run so a clean one is
    distinguishable from one nobody measured.

    A single ``model_copy``, because the fields are one fact about one run.
    """
    return output.model_copy(
        update={
            "degradations": [*output.degradations, *degradations],
            "advisories": [*output.advisories, *(advisories or [])],
            "narration_estimate": estimate,
        }
    )


class AppReviewScriptInput(StageInput):
    pass


class ReviewLine(StageOutput):
    id: str
    text: str
    seconds: float
    fact_id: str | None = None
    critique_item_id: str | None = None


class AppReviewScriptOutput(StageOutputWithNotes):
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


class AppReviewScriptStage(LLMScriptStageBase[AppReviewScriptInput, AppReviewScriptOutput]):
    name = "app_review_script"
    input_model = AppReviewScriptInput
    output_model = AppReviewScriptOutput
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
            "fact_id": {"type": ["string", "null"]},
            "critique_item_id": {"type": ["string", "null"]},
        },
        "required": ["text", "seconds"],
    }

    #: This run's narration rate — the number the ASKS are written from (the word
    #: budget in the prompt, the target in the re-ask). Overwritten per execution in
    #: ``handle`` with the PHASE-A prior (the calibration passage measured at the
    #: run's voice), and again in ``_reconcile`` with the rate pooled off the drafted
    #: lines themselves once they have been synthesized. It is NOT what a verdict is
    #: taken against any more — that is ``_Draw.estimate``, real measured seconds.
    #: The class default is the fallback so a direct ``build_prompt`` call (tests,
    #: tooling) still works.
    _wpm: float = DEFAULT_WPM_NATURAL

    #: The Phase-A prior, with its provenance. Kept beside ``_wpm`` because the number
    #: alone cannot say whether anybody measured it — and because the full-script
    #: measurement needs something to fall back TO, and to report both rates when it
    #: cannot answer.
    _rate: SpeechRate = SpeechRate(wpm=DEFAULT_WPM_NATURAL, measured=False, reason=RESOLUTION_ERROR)

    def __init__(self, client: WMClient | None = None, tts: TTSClient | None = None) -> None:
        """``tts`` is the client the full-script measurement synthesizes through.

        ``None`` (production) lets ``measure_script_lines`` construct its own,
        exactly as the Phase-A probe already does. It exists so a test can measure a
        real duration off a fake backend instead of stubbing the measurement out — a
        stubbed measurement proves the wiring and nothing about the arithmetic that
        motivated it.
        """
        super().__init__(client)
        self._tts = tts

    #: Extra prompt paragraph for the ONE bounded re-ask (see
    #: :data:`MAX_CORRECTIVE_ATTEMPTS` — attempt two is the trim, not a second
    #: re-ask, so this is set at most once per run). Empty on the first
    #: draw; ``build_prompt`` appends it verbatim so the re-ask travels through
    #: the SAME ``LLMScriptStageBase`` call (same worker, schema, retry ladder) —
    #: this stage opens no second client.
    _reask_block: str = ""

    #: An EMPTY draw is invalid, not a result: the parser re-prompts it inside the
    #: retry budget and the stage fails when every attempt is empty. See
    #: ``shared/empty_draw.py`` for the live run this closes.
    #: This stage has no verbatim bypass, so every run reaches the parser.
    min_elements: ClassVar[int] = 1

    async def handle(self, payload: AppReviewScriptInput, ctx: RunContext) -> AppReviewScriptOutput:
        """Resolve the measured narration rate, run the LLM tail, then RECONCILE.

        ``build_prompt`` is SYNCHRONOUS on ``LLMScriptStageBase`` and is inherited
        by eight stages, so the one await this stage needs is taken here and the
        result stashed on the instance. ``dag_executor._run_stage`` constructs a
        fresh handler per stage execution, so instance state is safe (same
        precedent as ``action_reel/compile.py``).

        Reconciliation is wrapped whole: a paid run must never fail because the
        length arithmetic misbehaved, so any exception returns the draw the model
        actually produced.
        """
        self._rate = (await resolve_effective_wpm(ctx, texts=(), default=DEFAULT_WPM_NATURAL)).rate
        self._wpm = self._rate.wpm
        self._reask_block = ""
        try:
            first = await super().handle(payload, ctx)
        except StageRetriesExhaustedError as exc:
            # ``min_elements = 1``: every attempt drew empty. Named for what it is.
            failure = empty_draw_failure(exc, stage=self.name)
            if failure is None:
                raise
            raise failure from exc
        require_elements(first.lines, stage=self.name)
        try:
            reconciled = await self._reconcile(first, payload, ctx)
        except Exception:  # noqa: BLE001 — a paid run must never fail on length arithmetic
            logger.warning("%s: narration reconciliation failed — keeping draw", self.name)
            reconciled = first
        require_elements(reconciled.lines, stage=self.name)
        return reconciled

    # ------------------------------------------------------------------ #
    # narration reconciliation
    # ------------------------------------------------------------------ #

    def _reask_prompt_block(
        self, *, words: int, estimated: float, ordered: float, target_words: int | None = None
    ) -> str:
        target = (
            budget_words(ordered, words_per_minute=self._wpm)
            if target_words is None
            else target_words
        )
        direction = "TOO LONG" if estimated > ordered else "TOO SHORT"
        return (
            f"\n\nREVISION REQUIRED. Your previous draft was {direction}. It contained "
            f"{words} spoken words, which at this narrator's measured pace of "
            f"{self._wpm:.0f} words per minute takes about {estimated:.0f} seconds. The "
            f"order is {ordered:.0f} seconds, i.e. {target} words. Rewrite the WHOLE "
            f"script so the combined spoken text is as close to {target} words as you "
            "can make it — keep the strongest material and cut or expand the rest. Do "
            "not repeat a line to fill time."
        )

    async def _measure(self, texts: list[str], ctx: RunContext) -> ScriptMeasurement:
        """Synthesize EVERY line of this draft and return what it really runs to.

        This replaces the old four-line SAMPLE, and the difference is not one of
        degree: a sample yields a RATE, and a rate times a word count is still a
        projection — several percent of error, on exactly the axis this stage is
        being graded on. Measuring every line yields the film's delivered length,
        because an app_review film IS the sum of its narration clips.

        It is very nearly free. ``speed`` and ``stage_name`` are the ones
        ``app_review_record`` will REQUEST (see :data:`NARRATE_STAGE_NAME` and
        :data:`NARRATE_SPEED`), so ``deterministic_clip_id`` makes every clip
        measured here the very clip record is about to ask for: each surviving
        line's synthesis is paid once, here, instead of once, later. Only the
        lines a re-ask replaces or a trim drops are genuinely spent.

        app_review is SINGLE-VOICE — one narrator for the whole script — so every
        line carries the same ``voice_id`` ``app_review_record`` passes
        (``record.py``: the kokoro voice id, never the free-text descriptor).

        AN AUTHORED RATE IS NOT MEASURED AGAINST. When the operator wrote
        ``direction.timing.wpm.natural`` the ladder hands back an ``authored`` basis,
        and that rate "outranks every measurement" — so synthesizing the script here
        would buy a number this stage is contractually forbidden to use, at the cost
        of a full script of TTS. The estimate falls back to the prior-derived
        arithmetic, at the authored rate, and carries ``reason=None`` because nothing
        FAILED: the measurement was declined, not attempted and lost. (The clips are
        not lost either — ``app_review_record`` synthesizes them later, as it did
        before Phase C existed.)

        Never raises: ``measure_script_lines`` absorbs every failure into a
        prior-derived measurement carrying a ``reason``.
        """
        if self._rate.basis == AUTHORED:
            return self._authored_measurement(texts)
        voice_id = (ctx.config_snapshot.get("voice") or {}).get("voice_id")
        return await measure_script_lines(
            ctx,
            texts=texts,
            voices=[voice_id] * len(texts),
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
        word-count path, and ``_rate_entries`` records the substitution.

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

    def _draw_of(self, output: AppReviewScriptOutput, measurement: ScriptMeasurement) -> _Draw:
        """A candidate paired with the seconds it will be JUDGED at.

        When the measured rate is unusable the measurement's own seconds are not
        either — ``_seconds_at`` yields 0.0 per line for a 0/NaN rate, which would
        read as a 0.0s script and send a perfectly good draw off to be lengthened.
        That case falls back to word-count arithmetic, where ``estimate_seconds``
        substitutes the documented default and ``speech_rate_degradation`` records
        that it did. Nothing here is silent.
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
        audio nobody measured — a fault, not advice, which is why these stay on
        the degrading tier while the estimate MISS moves off it.
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
        payload: AppReviewScriptInput,
        ctx: RunContext,
        target_words: int | None = None,
    ) -> AppReviewScriptOutput | None:
        """ATTEMPT ONE: ask the model again, through this stage's existing LLM call.

        ``None`` when the call failed — a re-ask is a correction, and a paid run
        that already holds a usable draw must not die because the correction did.

        ``target_words`` overrides the budget the correction asks for; the
        trim-floored re-ask (step 2b) passes the SHIPPING draw's own arithmetic.
        """
        self._reask_block = self._reask_prompt_block(
            words=count_words([line.text for line in draw.output.lines]),
            estimated=draw.estimate,
            ordered=ordered,
            target_words=target_words,
        )
        try:
            # A correction, not a draw: parsed with no floor, so an empty answer
            # costs ONE call and returns as zero elements for the caller to record
            # as a failed re-ask (``LLMScriptStageBase.corrective_draw``).
            with self.corrective_draw():
                return await super().handle(payload, ctx)
        except Exception:  # noqa: BLE001 — a failed re-ask degrades to the first draw
            logger.warning("%s: narration re-ask failed — keeping first draw", self.name)
            return None
        finally:
            self._reask_block = ""

    async def _reconcile(
        self,
        first: AppReviewScriptOutput,
        payload: AppReviewScriptInput,
        ctx: RunContext,
    ) -> AppReviewScriptOutput:
        """Steer at the ORDER, correct until in band or out of attempts, never pad.

        Three properties, each of which was a measured live miss:

        1. **It aims at the target, not at the band edge.** The decision to
           correct is :func:`steer_verdict` (long side +4%), not the +/-10% band
           that GRADES the master — a draw accepted at +9% has nothing left for
           the estimate's own error. The +/-10% band still decides what is
           REPORTED, so a conformant film is never flagged.
        2. **It iterates.** Up to :data:`MAX_CORRECTIVE_ATTEMPTS` corrections —
           the re-ask, then the trim — and a draw is shipped only when it is IN
           BAND or no attempt improved it. "Closer" alone was the bug: run
           ``01a09751`` re-asked a +34% draft to -18.4%, accepted it as closer,
           and delivered -20.84%. Here that -18.4% draw does not end the loop; the
           +34% draft it is holding is TRIMMABLE, and the trim runs on seconds
           already measured, so it costs nothing to find that out.
        3. **Every candidate is judged at ITS OWN measured seconds.** A re-asked
           draft is new text; grading it at the rate sampled off the draft it
           replaced is how a "correction" lands 20% out.

        Still no padding. Nothing lengthens a short script mechanically — the
        re-ask may ask for more material, but repeated audio is a worse defect
        than a short film, so a short draw is ACCEPTED and recorded.
        """
        ordered = review_ordered_seconds(ctx.config_snapshot)
        if ordered is None or ordered <= 0.0:
            # No order means no band and no verdict; nothing was measured, so
            # there is no estimate to record either. Inventing one is the exact
            # error `duration_conformance` refuses by returning None.
            return first

        measurement = await self._measure([line.text for line in first.lines], ctx)
        self._wpm = measurement.wpm

        best = self._draw_of(first, measurement)
        pool = [best]
        first_estimate = best.estimate
        action = "kept first draw"
        reasked = False
        reask_kept: bool | None = None

        # (1) ATTEMPT ONE — the re-ask, in whichever direction the steer points.
        if steer_verdict(best.estimate, ordered) in ("long", "short"):
            second = await self._reask(best, ordered, payload, ctx)
            reasked = second is not None
            if second is not None and second.lines:
                second_measurement = await self._measure([line.text for line in second.lines], ctx)
                self._wpm = second_measurement.wpm
                candidate = self._draw_of(second, second_measurement)
                pool.append(candidate)
                if closer_to_order(candidate.estimate, best.estimate, ordered):
                    best = candidate
                    reask_kept = True
                    action = "re-asked"
                else:
                    reask_kept = False
                    action = "re-ask not closer; kept first draw"
            elif second is not None:
                # An EMPTY re-ask is a re-ask that was DISCARDED, and the record
                # has to say so: ``NarrationEstimate.reask_kept`` documents
                # ``None`` as "no re-ask happened", so leaving it None here made
                # ``reasked=True, reask_kept=None`` a state the field's own
                # contract calls impossible — unreadable after the fact, and the
                # one shape that cost an LLM call for nothing.
                reask_kept = False
                action = "re-ask returned no lines; kept first draw"

        # (2) ATTEMPT TWO — the trim. Deterministic, free (it selects lines whose
        #     seconds are already measured), and the only lever that can rescue a
        #     draw the re-ask overshot: the LONG draft in the pool is trimmable
        #     even when the SHORT one is currently winning.
        floored: TrimPlan | None = None
        trimmed_lines = 0
        # STOP EARLY. An incumbent the steer already calls ``ok`` is DONE, and the
        # guard is not equivalent to letting the trim run and lose: the trim scans
        # the whole pool, so a long sibling trimmed to a hair nearer the order
        # would displace a perfectly good in-band draw with one that had lines
        # amputated out of it. Pinned by
        # ``test_an_in_band_incumbent_is_not_displaced_by_a_trimmed_sibling``.
        if steer_verdict(best.estimate, ordered) != "ok":
            best, action, floored, trimmed_lines = self._trim_step(best, pool, ordered, ctx, action)

        # (2b) ATTEMPT THREE, LONG SIDE — see ``_floored_reask_step``. Fired ONLY
        #      when the trim could not act at all, so a trim that DID drop something
        #      is never pre-empted by a paid LLM call.
        if trimmed_lines == 0 and floored is not None:
            best, kept, action, floored = await self._floored_reask_step(
                best, pool, ordered, payload, ctx, action, floored
            )
            reasked = reasked or kept is not None
            reask_kept = True if kept else reask_kept

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
        # the steer merely improved to +6% raises nothing either. It IS
        # conformant; a second, worse, pre-audio opinion saying otherwise is how
        # a good delivery gets read as broken.
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
        return _finalize(
            best.output, degradations=degradations, advisories=advisories, estimate=estimate
        )

    async def _floored_reask_step(
        self,
        best: _Draw,
        pool: list[_Draw],
        ordered: float,
        payload: AppReviewScriptInput,
        ctx: RunContext,
        action: str,
        floored: TrimPlan,
    ) -> tuple[_Draw, bool | None, str, TrimPlan | None]:
        """ATTEMPT THREE, LONG SIDE: re-ask with an EXPLICIT WORD TARGET.

        Port of lesson's step (2b) (PR #1016). ``plan_trim`` can decline with
        nothing dropped — the duration quantum of one line is wider than the band,
        so every drop lands UNDER it (``declined_under_band``) — and before this the
        stage then shipped the long draw as-is: one corrective attempt on the long
        side, where the trim was supposed to be the second.

        **Only when the trim genuinely could not act** (the caller gates on zero
        lines dropped AND a floored plan; this gates on the draw still being LONG),
        so a paid call never pre-empts the free lever. The target is this draw's
        OWN arithmetic, ``ordered * draw.wpm / 60``, not ``self._wpm`` (the rate
        measured LAST, possibly a discarded draft). ``closer_to_order`` still
        decides, so a worse draw is discarded rather than shipped; when the new
        draw IS kept the floor is cleared — it describes a script no longer shipped.

        Returns ``(best, kept, action, floored)``; ``kept is None`` means no usable
        draw came back (call failed or empty), which is not a judged rejection.
        """
        if steer_verdict(best.estimate, ordered) != "long":
            return best, None, action, floored

        target_words = round(ordered * best.wpm / 60.0)
        logger.info(
            "%s: trim floored having dropped nothing — re-asking at an explicit "
            "%d-word target (estimate %.2fs vs ordered %.2fs)",
            self.name,
            target_words,
            best.estimate,
            ordered,
        )
        drawn = await self._reask(best, ordered, payload, ctx, target_words=target_words)
        if drawn is None:
            return best, None, action, floored
        if not drawn.lines:
            return best, False, action, floored
        measurement = await self._measure([line.text for line in drawn.lines], ctx)
        self._wpm = measurement.wpm
        candidate = self._draw_of(drawn, measurement)
        pool.append(candidate)
        if not closer_to_order(candidate.estimate, best.estimate, ordered):
            return best, False, action, floored
        return (
            candidate,
            True,
            f"{action}; re-asked at an explicit {target_words}-word target",
            None,
        )

    def _trim_step(
        self,
        best: _Draw,
        pool: list[_Draw],
        ordered: float,
        ctx: RunContext,
        action: str,
    ) -> tuple[_Draw, str, TrimPlan | None, int]:
        """The trim, accepted only if it improves — and the floor if it did not land.

        **It trims any LONG candidate in the pool, not only the incumbent.** That
        is the fix for the shape that shipped -20.84%: the re-ask had swung a +34%
        draft to -18.4%, which is closer and therefore wins, and a trim of the
        winner is then impossible (nothing shortens a short script) — so the loop
        used to stop with both drafts out of band while one of them was one
        dropped line from conformant. Trimming a candidate costs no synthesis and
        no LLM call, so there is no reason to look at only one. The incumbent is
        tried FIRST, and ties keep it, so a single-candidate run is unchanged.

        ``floored`` is the plan when the band is STILL out of reach afterwards,
        which "``+ trimmed``" alone reads exactly like a landed trim.

        **A plan that was NOT APPLIED is never reported as a floor.** A rejected
        trim (a verbatim phrase would have been lost, or the result was no closer
        to the order) leaves the script exactly as it was, so the plan's
        ``estimated_seconds_after`` describes a script that was never shipped.
        Reporting it put two contradicting entries on one run — ``narration
        estimated at 82.0s ... final 82.0s`` next to ``trim FLOORED at 80.0s — 2
        line(s) dropped`` when zero lines were dropped. A rejection is its own
        fact and is recorded as one, in the action string, with no floor entry.

        A plan that dropped NOTHING is not a rejection: there was nothing to
        apply, ``estimated_seconds_after`` equals the script that did ship, and
        the floor it names (every droppable line protected, or every remaining
        drop landing under the band) is the truth about this script.
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
            action = f"{action} + trimmed{scope}"
            if plan.exceeds_limit and narration_verdict(trimmed.estimate, ordered) == "long":
                return trimmed, f"{action}; {_floor_note(plan)}", plan, dropped
            return trimmed, action, None, dropped

        # Nothing was adopted. The report follows the first source's plan, which
        # IS the incumbent whenever the incumbent is long. When the incumbent is
        # SHORT it is not a source at all, so the first source is a different
        # draft — and an unqualified "trim REJECTED" then described a draw that
        # was never going to ship. Name the draw the note is about.
        source, trimmed, plan, rejected = results[0]
        scope = "" if source is incumbent else f" of the {_pool_label(source, pool)}"
        if trimmed is not None:
            rejected = "it landed no closer to the ordered duration"
        if rejected is not None:
            return best, f"{action}; trim{scope} REJECTED — {rejected}", None, 0
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

        The fourth element is the REJECTION, in the caller's words — ``None`` when
        the plan is safe to apply (including a plan that drops nothing, which is
        not a rejection). A ``None`` output with a rejection means the script was
        left untouched on purpose, and nothing about the plan may then be reported
        as if it had happened.

        A required verbatim phrase must survive: ``LLMScriptStageBase.handle``
        already asserted it on the FULL draw, so dropping lines here could ship a
        script that silently lost the user's exact words. Lines carrying a phrase
        are protected, and the result is re-checked before it is accepted.

        **It plans on MEASURED seconds, not on a word proxy.** Two lines of equal
        word count do not take equal time — a version string, an acronym, a clause
        the voice pauses after — so a word-weighted plan can drop the wrong pair
        and land outside the band it was aiming at. ``line_seconds`` is handed to
        ``plan_trim`` whenever the measurement is usable; it falls back to words
        when it is not, which is the pre-measurement behaviour exactly.
        """
        texts = [line.text for line in draw.output.lines]
        items = items_for_stage(ctx.config_snapshot.get("verbatim_items") or [], self.name)
        protected = tuple(
            index for index, text in enumerate(texts) if _carries_verbatim(text, items)
        )
        plan = plan_trim(
            texts,
            ordered_seconds=ordered,
            # THIS draw's rate, never ``self._wpm``. ``self._wpm`` holds whatever
            # was measured LAST, which after a re-ask is the rate of a draft that
            # may have been discarded — and ``plan_trim`` falls back to the word
            # proxy the moment ``_trimmable_seconds`` refuses the measurement, at
            # which point the wrong rate silently decides which lines are cut.
            words_per_minute=draw.wpm,
            # The OUTER +/-10% band, deliberately — NOT the 4% the steer aims at.
            # The steer decides WHETHER to correct; the trim decides what a cut is
            # allowed to land on, and it declines any drop falling below
            # ``ordered x (1 - tolerance)``. Narrowing this to the steer band makes
            # it refuse cuts it should take — the only available drop lands at,
            # say, -8%, which is conformant, and the script then ships its full
            # overrun instead. Pinned by
            # ``test_the_trim_aims_at_the_outer_band_not_at_the_steer_band``.
            tolerance=DEFAULT_TOLERANCE,
            protected_indices=protected,
            line_seconds=_trimmable_seconds(draw),
        )
        if not plan.dropped_indices:
            return None, plan, None
        kept = draw.select(plan.keep_indices)
        if find_missing_verbatim(" ".join(line.text for line in kept.output.lines), items):
            logger.warning("%s: trim would drop a required verbatim phrase — skipped", self.name)
            return None, plan, "it would have dropped a required verbatim phrase"
        return kept, plan, None

    def build_prompt(self, ctx: RunContext) -> str:
        inspect = ctx.prior_output("app_inspect")
        critique = ctx.prior_output("app_critique")
        if not (inspect and critique):
            raise RuntimeError("app_inspect + app_critique required")
        brief = ctx.config_snapshot.get("brief") or {}
        target_s = int(_target_seconds(brief))
        items = items_for_stage(ctx.config_snapshot.get("verbatim_items") or [], self.name)
        notes = ctx.annotations_for("app_review_script")
        notes_block = (
            "\n\nApply reviewer notes:\n" + "\n".join(f"- {n['text']}" for n in notes)
            if notes
            else ""
        )
        return (
            f"Write a {target_s}-second video review script.\n"
            f"{word_budget_sentence(target_s, self._wpm)}"
            f"Voice: "
            f"{ctx.config_snapshot.get('voice', {}).get('descriptor', 'neutral')}.\n"
            f"Facts: {inspect['facts']}\n"
            f"Pros: {critique['pros']}\n"
            f"Cons: {critique['cons']}\n"
            f"Recommendations: {critique['recommendations']}\n"
            "For each line, include the fact_id or critique_item_id it references.\n"
            'Output JSON: {"lines":[{"text":"...","seconds":<num>,'
            '"fact_id":"..."|null,"critique_item_id":"..."|null}, ...]}.'
            f"{notes_block}"
            f"{verbatim_prompt_block(items)}"
            f"{self._reask_block}"
        )

    def make_element(self, raw: dict, *, element_id: str) -> ReviewLine:
        return ReviewLine(
            id=element_id,
            text=str(raw["text"]),
            seconds=float(raw["seconds"]),
            fact_id=raw.get("fact_id"),
            critique_item_id=raw.get("critique_item_id"),
        )

    def build_output(self, elements: list[StageOutput], ctx: RunContext) -> AppReviewScriptOutput:
        lines = cast("list[ReviewLine]", elements)
        target_seconds = _target_seconds(ctx.config_snapshot.get("brief") or {})
        return AppReviewScriptOutput(
            lines=lines,
            total_seconds=sum(line.seconds for line in lines),
            word_budget=word_budget(target_seconds, self._wpm),
            actual_words=count_words([line.text for line in lines]),
        )
