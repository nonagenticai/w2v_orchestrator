"""lesson_script — long-form learning script (paragraphs, not single lines).

Output exposes ``paragraphs[*].id`` for annotation pinning and
``paragraphs[*].speaker`` ("a"/"b") for the two-voice Socratic mode (opt-in via
``brief.dialogue``).

When ``brief.dialogue.enabled`` is set the script is written as a Socratic
lesson — a curious STUDENT (speaker "a") asks short questions, an INSTRUCTOR
(speaker "b") teaches and answers. ``lesson_narrate`` then casts each speaker to
a distinct kokoro voice. When the flag is off the prompt asks for a single
narrator (the instructor) and the per-paragraph speaker is ignored downstream.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final, cast
from uuid import NAMESPACE_URL, uuid5, uuid7

from w2v_common.stages import StageRetriesExhaustedError

from src.services.stages.base import (
    SlotUse,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    assert_verbatim_present,
    find_missing_verbatim,
    items_for_stage,
    normalize_source_script,
    verbatim_prompt_block,
)
from src.services.stages.generic.grounding import grounding_prompt_block
from src.services.stages.lesson.narrate import _DEFAULT_CAST, LessonNarrateStage
from src.services.stages.primitives.dialogue import assign_dialogue_speakers, voice_for_speaker
from src.services.stages.primitives.script_stage import LLMScriptStageBase
from src.services.stages.shared.compute_plan import beat_word_budget
from src.services.stages.shared.degradation import Degradation
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
    word_band,
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

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.clients.tts_client import TTSClient
    from src.services.llm import WMClient
    from src.services.runs.run_context import RunContext

logger = logging.getLogger(__name__)

# Default Socratic personas (overridable via brief.dialogue.personas). Speaker
# "a" is the student who asks the questions; speaker "b" is the instructor who
# teaches and answers. The role split matches the question-mark heuristic in
# ``assign_dialogue_speakers`` (a question → "a").
_DEFAULT_PERSONA_A = (
    "Sam — a curious student meeting this material for the first time. Asks "
    "short, honest clarifying questions a real learner would ask."
)
_DEFAULT_PERSONA_B = (
    "Dr. Lee — a calm, authoritative instructor. Teaches and answers Sam's "
    "questions clearly and concretely, carrying the bulk of the explanation."
)

#: The stage whose ``synthesize_and_store`` call actually speaks these paragraphs.
#: It is ``lesson_narrate``, NOT ``lesson_record`` — ``lesson/compile.py`` reads its
#: audio out of ``ctx.prior_output("lesson_narrate")["clips"]`` and record only
#: renders slides. Read off the class rather than repeated as a literal so the two
#: cannot drift: the measurement below stores its clips under THIS name, and
#: ``deterministic_clip_id`` then makes the measured clip and the narrated clip one
#: object — see :data:`NARRATE_SPEED`.
NARRATE_STAGE_NAME: Final = LessonNarrateStage.name

#: The ``speed`` ``lesson_narrate`` requests: it does not override
#: ``NarrationStageBase.speed_for``, which returns ``None``, so the argument is
#: omitted entirely and ``deterministic_clip_id`` hashes ``None``. The measurement
#: MUST request the same thing — ``1.0`` is a different clip id AND a different
#: measurement (speed scales duration linearly backend-side), so measuring at 1.0
#: would both mis-measure the voice and be reused by nothing.
NARRATE_SPEED: Final[float | None] = None

#: How many CORRECTIVE attempts one draw may cost. Attempt 1 is the LLM re-ask in
#: whichever direction the steer points. Attempt 2 DEPENDS ON THE DIRECTION:
#:
#: * LONG  -> the deterministic trim. Free: it runs on per-paragraph seconds this
#:   stage has already measured, so the budget stays at TWO calls (draw+re-ask)
#:   whenever the trim ACTS. When it cannot act at all — every subset of the
#:   paragraphs lands under the band, so it drops nothing and floors — the long
#:   side falls back to a THIRD call too, a re-ask carrying an explicit word
#:   target. Step (2b); see the live run cited there.
#: * SHORT -> a second LLM re-ask, because the trim is long-only and would be a
#:   no-op. Stage LLM budget on this path is THREE calls (draw + two re-asks).
#:
#: That third call is a real cost, paid only by a draw that is still short after
#: the first re-ask. It buys the short side the same two attempts the long side
#: always had; before it, a short draw shipped whatever one re-ask reached.
#:
#: NOTE: this constant is documentation — the code branches on
#: :func:`steer_verdict`, it does not read this value as a counter.
MAX_CORRECTIVE_ATTEMPTS: Final = 2


#: Seconds of narration one paragraph should carry — one paragraph is one
#: narration clip and (on a short lesson) one slide, so this sets the slide pace.
SECONDS_PER_PARAGRAPH: Final = 20.0
#: The deterministic floor is looser than the ask: it only rescues a draw that
#: ignored the ask outright (run 01a156a8-978b: ONE 183-word paragraph for a 60s
#: lesson → one slide held 70s, and a trim with nothing to drop).
_FLOOR_SECONDS_PER_PARAGRAPH: Final = 30.0
_SENTENCE_SPLIT: Final = re.compile(r"(?<=[.!?])\s+")


def paragraph_count_floor(target_seconds: float) -> int:
    """The fewest paragraphs a lesson of ``target_seconds`` may ship with (>= 2)."""
    if not math.isfinite(target_seconds) or target_seconds <= 0.0:
        return 1
    return max(2, math.ceil(target_seconds / _FLOOR_SECONDS_PER_PARAGRAPH))


def _split_into(paragraph: LessonScriptParagraph, parts: int) -> list[LessonScriptParagraph]:
    """``paragraph`` split into up to ``parts`` pieces on sentence boundaries, by words.

    Each cut sits on the sentence boundary nearest an equal-words target, so the
    pieces are balanced. The first piece keeps the paragraph's id (annotations pinned
    to it stay pinned); the others get DETERMINISTIC ids (``uuid5`` of the parent id),
    so a re-entry replaying the same draw mints the same ids. ``seconds`` (the model's
    own estimate) is shared by word count. ``[paragraph]`` when it is one sentence.
    """
    sentences = [x for x in _SENTENCE_SPLIT.split(" ".join(paragraph.text.split())) if x]
    parts = min(parts, len(sentences))
    if parts < 2:
        return [paragraph]
    counts = [len(x.split()) for x in sentences]
    total = sum(counts) or 1
    boundaries = [sum(counts[:i]) for i in range(1, len(sentences))]  # words before cut i
    cuts: list[int] = []
    for j in range(1, parts):
        target = total * j / parts
        # After the previous cut, leaving one sentence for each cut still to place.
        lo = cuts[-1] + 1 if cuts else 1
        free = range(lo, len(sentences) - (parts - 1 - j))
        cuts.append(min(free, key=lambda i: abs(boundaries[i - 1] - target)))
    edges = [0, *cuts, len(sentences)]
    pieces: list[LessonScriptParagraph] = []
    for k in range(parts):
        chunk = sentences[edges[k] : edges[k + 1]]
        share = sum(counts[edges[k] : edges[k + 1]]) / total
        update: dict[str, object] = {"text": " ".join(chunk), "seconds": paragraph.seconds * share}
        if k:
            update["id"] = str(uuid5(NAMESPACE_URL, f"{paragraph.id}:split:{k}"))
        pieces.append(paragraph.model_copy(update=update))
    return pieces


def enforce_paragraph_floor(
    paragraphs: list[LessonScriptParagraph], target_seconds: float
) -> list[LessonScriptParagraph]:
    """Split the longest multi-sentence paragraphs until the floor is met.

    Never pads and never changes a word: only sentence boundaries move between
    paragraphs, so the narration is the same text and a lesson's slides and its
    trim both get the granularity the draw failed to provide. Stops early when no
    paragraph has two sentences left to separate.
    """
    floor = paragraph_count_floor(target_seconds)
    out = list(paragraphs)
    while len(out) < floor:
        ranked = sorted(range(len(out)), key=lambda i: len(out[i].text.split()), reverse=True)
        for index in ranked:
            pieces = _split_into(out[index], floor - len(out) + 1)
            if len(pieces) > 1:
                out[index : index + 1] = pieces
                break
        else:
            break
    return out


def _target_seconds(brief: dict) -> float:
    """The ordered lesson length in seconds — precedence UNCHANGED.

    ``brief.duration_minutes`` (default 6) is the only key this stage has ever
    read; extracted so ``build_output`` re-derives the SAME number the prompt was
    budgeted against.
    """
    return float(brief.get("duration_minutes", 6)) * 60.0


def word_budget(target_seconds: float, wpm: float) -> int:
    """Whole-script word budget = ``duration_s / 60 * wpm``, rounded.

    Thin wrapper over the EXISTING arithmetic in ``compute_plan.beat_word_budget``
    (that function is per-beat only by naming; the formula is the same one a
    whole-script budget needs).
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

    A TARGET with a stated floor, not a ceiling. The ceiling wording ("a HARD
    CEILING … write fewer if the story is done sooner") was read literally: run
    01a158a6-0f2c (2026-10-20, 60s ordered, ceiling 198 words) drew 66 words, and a
    live replay of that prompt drew 57/120/70/58 — a lesson is graded at +/-10% in
    BOTH directions, so an instruction that only forbids one of them buys the other.
    """
    words = word_budget(target_seconds, wpm)
    low, high = word_band(words)
    return (
        f"Total spoken length: at this narrator's measured pace of {wpm:.0f} words per "
        f"minute, {target_seconds:.0f} seconds is {words} words. Write {words} words in "
        f"total (all paragraphs combined) — no fewer than {low} and no more than {high}. "
        "A short script is as wrong as a long one: the customer ordered "
        f"{target_seconds:.0f} seconds. Reach the length with substance — concrete "
        "examples, a worked case, why it matters — never filler or repetition.\n"
    )


#: Spoken words per sentence the paragraph plan converts a word share with.
_WORDS_PER_SENTENCE: Final = 15.0


def paragraph_plan_sentence(target_seconds: float, wpm: float) -> str:
    """How many paragraphs to write (one per ~20s) and how many WORDS each carries.

    The per-paragraph share is the whole budget split evenly, stated as words and
    as the sentence count that holds them. "2-3 sentences each" (the previous
    wording) is ~40 words, a third short of the ~55 words 20s of this narrator
    takes — and the model obeyed it over the total.
    """
    count = max(
        paragraph_count_floor(target_seconds), round(target_seconds / SECONDS_PER_PARAGRAPH)
    )
    per = max(1, round(word_budget(target_seconds, wpm) / count))
    sentences = max(2, round(per / _WORDS_PER_SENTENCE))
    return (
        f"Write {count} paragraphs of about {per} words each ({sentences}-{sentences + 1} "
        "sentences). Each paragraph becomes its own narrated slide, so open each one "
        "with a short sentence that works as that slide's headline.\n"
    )


def corrective_block(
    verdict: str,
    *,
    produced_words: int,
    estimated_seconds: float,
    ordered_seconds: float,
    target_words: int,
    previous: Sequence[str] = (),
) -> str:
    """The ONE bounded re-ask: name the measurement, not the disappointment.

    The first draw already carried the word budget as prose and missed it. The
    only new information a second ask can carry is the ARITHMETIC of the miss —
    what was written, how long that takes to say, what was ordered, and the
    number to hit. "Try harder" is not a correction.

    SHORT with the ``previous`` draft: the ask is a DELTA over that draft — "it
    needs M more words, expand each paragraph by m" — with the draft quoted. A
    from-scratch "rewrite to about N words" re-drew short again live (run
    01a158a6-0f2c: asked 164 → 115, asked 151 → 127, shipped -23.7%); the same
    finding as the beat-expander's (#1125: absolute count grew 0/9, delta 9/9).
    """
    if verdict == "short" and previous:
        produced = max(produced_words, 0)
        missing = max(target_words - produced, 1)
        each = max(1, math.ceil(missing / len(previous)))
        draft = "\n".join(f"{i}. {text}" for i, text in enumerate(previous, start=1))
        return (
            f"\n\nCORRECTION — your previous draft (below) was TOO SHORT. It contained "
            f"{produced} words, which at this narrator's measured pace takes about "
            f"{estimated_seconds:.0f} seconds to speak. The customer ordered "
            f"{ordered_seconds:.0f} seconds, which is about {target_words} words, so the "
            f"script needs about {missing} MORE words. Keep its {len(previous)} "
            f"paragraphs in order and EXPAND each one by about {each} words — a concrete "
            "example, a worked number, or why it matters for the learning objective. "
            "Returning a paragraph unchanged or shorter is a failure. Do NOT pad with "
            "filler, repetition or restatement.\nPrevious draft:\n"
            f"{draft}"
        )
    direction = "TOO LONG" if verdict == "long" else "TOO SHORT"
    instruction = (
        "Cut content — remove or shorten whole paragraphs, keeping the opening "
        "and the learning objective intact."
        if verdict == "long"
        else "Add substance — teach the objective more fully, with concrete examples. "
        "Do NOT pad with filler, repetition or restatement."
    )
    return (
        f"\n\nCORRECTION — your previous draft was {direction}. It contained "
        f"{produced_words} words, which at this narrator's measured pace takes about "
        f"{estimated_seconds:.0f} seconds to speak. The customer ordered "
        f"{ordered_seconds:.0f} seconds. Rewrite the WHOLE script to total about "
        f"{target_words} words. {instruction}"
    )


@dataclass(frozen=True, slots=True)
class _Draw:
    """One candidate script AND the per-paragraph seconds it was judged at.

    The two travel together because the defect this stage is fixing is a draw
    judged at another draw's rate: the old loop estimated the FIRST draft from a
    blended cast rate and then graded the re-asked one — different words, and
    possibly a different speaker mix — at that same number.

    ``paragraph_seconds`` is parallel to ``output.paragraphs``, so a trim of the
    candidate is an index selection on both at once and needs no second
    measurement.
    """

    output: LessonScriptOutput
    paragraph_seconds: tuple[float, ...]
    #: Per paragraph: is that seconds a FACT (a real synthesis, or an exactly-0.0s
    #: blank paragraph) rather than words-over-a-rate?
    measured: tuple[bool, ...]
    #: ``sum(paragraph_seconds)`` — the number every verdict here is taken against.
    estimate: float
    #: The rate THIS draw's seconds are expressed at, and where it came from.
    wpm: float
    basis: str

    @property
    def measured_paragraphs(self) -> int:
        return sum(1 for flag in self.measured if flag)

    def select(self, indices: Sequence[int]) -> _Draw:
        """This draw reduced to ``indices`` — paragraphs, seconds and flags together.

        A trim needs no second measurement precisely because of this: the kept
        paragraphs are the very paragraphs that were synthesized, so their seconds
        are the same facts, selected.
        """
        seconds = tuple(self.paragraph_seconds[i] for i in indices)
        return _Draw(
            output=_with_paragraphs(self.output, [self.output.paragraphs[i] for i in indices]),
            paragraph_seconds=seconds,
            measured=tuple(self.measured[i] for i in indices),
            estimate=math.fsum(seconds),
            wpm=self.wpm,
            basis=self.basis,
        )


def _with_paragraphs(
    output: LessonScriptOutput, paragraphs: list[LessonScriptParagraph]
) -> LessonScriptOutput:
    """``output`` with ``paragraphs`` replaced and every derived total re-derived.

    ``total_seconds`` and ``actual_words`` are sums OVER the paragraphs; leaving
    them at their pre-trim values would make the stage output self-contradicting,
    and ``total_seconds`` is what downstream reporting reads as the script's length.
    """
    return output.model_copy(
        update={
            "paragraphs": paragraphs,
            "total_seconds": sum(p.seconds for p in paragraphs),
            "actual_words": count_words([p.text for p in paragraphs]),
        }
    )


def _protected_indices(
    paragraphs: list[LessonScriptParagraph], items: list[dict]
) -> tuple[int, ...]:
    """Paragraph indices the trim may NOT drop.

    A lesson pins NOTHING structurally, unlike its siblings: ``lesson/compile.py``
    rescales the slides to whatever audio came back, so there is no reveal line and
    no authored shotlist prefix whose removal would break the picture rather than
    shorten it. Passing no protection at all was therefore correct while the trim
    weighted paragraphs by WORDS, and it stays correct now that it weights them by
    MEASURED SECONDS — the unit the plan is expressed in has nothing to do with
    which paragraphs are structurally load-bearing.

    One thing IS load-bearing and was never protected: a required VERBATIM phrase.
    ``LLMScriptStageBase.handle`` gates the whole draw on it, and the steer's
    tighter long side makes this trim fire on drafts the old +/-10% gate let
    through, so dropping the paragraph that carries the customer's exact words is
    now reachable where it barely was before. app_review and website_review both
    pin it; lesson did not. This is that gap closed, and nothing else.
    """
    if not items:
        return ()
    return tuple(
        i
        for i, p in enumerate(paragraphs)
        if len(find_missing_verbatim(p.text, items)) < len(items)
    )


def _merge_entries(existing: list[Degradation], new: list[Degradation]) -> list[Degradation]:
    """``existing + new``, minus entries that say exactly the same thing.

    Two drafts measured through the same dead backend produce two byte-identical
    entries. One fact, one row: a reader counting rows must not read "the backend
    failed twice" as a worse outage than "the backend was down".
    """
    seen = {(entry["code"], entry["detail"]) for entry in existing}
    return [*existing, *(e for e in new if (e["code"], e["detail"]) not in seen)]


def _pool_label(draw: _Draw, pool: list[_Draw]) -> str:
    """Which draft ``draw`` is, for an action string a human reads later."""
    return "first draw" if pool and draw is pool[0] else "re-asked draw"


def _floor_note(plan: TrimPlan) -> str:
    """The action-string half of "the trim ran and could not reach the band"."""
    reason = (
        "next drop would land UNDER the band"
        if plan.declined_under_band
        else f"{plan.protected_count} protected paragraph(s)"
    )
    return f"STILL LONG — trim floored by {reason}"


def _trimmable_seconds(draw: _Draw) -> list[float] | None:
    """``draw``'s per-paragraph seconds for ``plan_trim`` — or ``None`` for words.

    ``plan_trim`` RAISES on a seconds list that is not a measurement (non-finite,
    negative, mis-length), and rightly so: silently mis-aligning seconds with
    paragraphs cuts somebody's lesson by the wrong weights. But this is a paid
    run's happy path, so the check happens HERE and a measurement that is not one
    simply falls back to the word proxy rather than aborting the reconciliation.
    """
    seconds = draw.paragraph_seconds
    if len(seconds) != len(draw.output.paragraphs):
        return None
    if not all(math.isfinite(value) and value >= 0.0 for value in seconds):
        return None
    return list(seconds)


def _finalize(
    output: LessonScriptOutput,
    *,
    degradations: list[Degradation],
    advisories: list[Degradation] | None = None,
    estimate: NarrationEstimate | None = None,
) -> LessonScriptOutput:
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


def _first_reask_record(
    drawn: LessonScriptOutput | None, kept: bool | None, action: str
) -> tuple[bool | None, str]:
    """``(reask_kept, action)`` for the first re-ask, with EVERY outcome named.

    ``kept`` is :meth:`LessonScriptStage._consider`'s verdict, ``None`` when there
    was no usable draw to judge. That covers two different facts: the call FAILED
    (``drawn is None`` — ``reasked`` is False and ``reask_kept`` stays ``None``),
    or it was SENT and came back EMPTY. The second is a re-ask that was discarded,
    and the record has to say so: ``reask_kept=None`` documents "no re-ask
    happened", so pairing it with ``reasked=True`` is a state the field's own
    contract forbids.
    """
    if kept is not None:
        return kept, "re-asked" if kept else "re-ask not closer; kept first draw"
    if drawn is not None:
        return False, "re-ask returned no paragraphs; kept first draw"
    return None, action


class LessonScriptInput(StageInput):
    pass


class LessonScriptParagraph(StageOutput):
    id: str
    text: str
    seconds: float
    # Two-voice speaker tag ("a"=student, "b"=instructor). Defaults to the
    # instructor (the single-narrator voice) and is forward-compat with any
    # persisted pre-dialogue script output; ignored by lesson_narrate unless
    # brief.dialogue.enabled.
    speaker: str = "b"


class LessonScriptOutput(StageOutputWithNotes):
    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. Empty here means the stage
    # reported success having produced nothing, which degrades the run.
    produces: ClassVar[tuple[str, ...]] = ("paragraphs",)

    paragraphs: list[LessonScriptParagraph]
    total_seconds: float
    learning_objective: str
    # Ordered-vs-delivered accounting. `total_seconds` is the model's OWN estimate
    # of its speaking time and has never been checked against anything; these two
    # are countable. RECORDING ONLY — nothing here raises or truncates.
    word_budget: int = 0
    actual_words: int = 0
    #: What the narration budget CONCLUDED, as numbers — the rate, where it came
    #: from, the first draw's estimate, and what each correction did. Present on
    #: EVERY reconciled run, clean ones included: "this run was measured and it
    #: landed" and "nobody ever measured this run" are otherwise the same absence.
    #: ``None`` only when reconciliation could not run at all (the verbatim
    #: short-circuit, or a defect inside the reconciler).
    narration_estimate: NarrationEstimate | None = None


class LessonScriptStage(LLMScriptStageBase[LessonScriptInput, LessonScriptOutput]):
    name = "lesson_script"
    input_model = LessonScriptInput
    output_model = LessonScriptOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (
        SlotUse("llm"),
        SlotUse("tts"),
    )
    element_key = "paragraphs"
    # Paragraphs, not one-line captions: a lesson element carries several
    # sentences, so its worst-case JSON is far larger than the default...
    tokens_per_element: ClassVar[int] = 220
    # ...and correspondingly there are FEWER of them. A lesson is a handful of
    # taught paragraphs, not 45 captions. Inheriting the base count gave a
    # worst-case answer of 220*45 = 9900 tokens — larger than the whole budget the
    # box can deliver, i.e. a request that could only ever truncate.
    # test_script_stage_token_budget caught that; this is the correction.
    max_expected_elements: ClassVar[int] = 20
    # Per-paragraph shape carries a speaker enum so the Socratic mode can
    # alternate "a" (student) / "b" (instructor). Single-narrator lessons still
    # emit it (WM only emits schema-named fields) but lesson_narrate ignores it
    # unless dialogue is enabled.
    element_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "seconds": {"type": "number"},
            "speaker": {"type": "string", "enum": ["a", "b"]},
        },
        "required": ["text", "seconds", "speaker"],
    }

    @property
    def worker_name(self) -> str:
        # NOT the base default "lesson-script": that worker was pinned to a
        # lines-shaped first call during 2026-06-02 diagnostics and now returns
        # {"lines": …} regardless of the per-call paragraphs schema (WM pins a
        # worker's output shape from its first message; the name-based reset
        # 404s). "-v2" is a fresh worker that pins correctly to `paragraphs`.
        # This is the ONLY script stage whose element_key is not "lines".
        return "lesson-script-v2"

    #: This run's narration rate — the number the ASKS are written from (the word
    #: budget in the prompt, the target in the corrective re-ask). Overwritten per
    #: execution in ``handle`` with the PHASE-A prior (the calibration passage
    #: measured at the run's configured voice), and again in ``_reconcile_narration``
    #: with the rate pooled off the drafted paragraphs themselves once they have
    #: been synthesized. It is NOT what a verdict is taken against any more — that is
    #: ``_Draw.estimate``, real measured seconds. The class default is the fallback
    #: so a direct ``build_prompt`` call (tests, tooling) still works.
    _wpm: float = DEFAULT_WPM_NATURAL

    #: The Phase-A prior, with its provenance. Kept beside ``_wpm`` because the
    #: number alone cannot say whether anybody measured it — and because the
    #: full-script measurement needs something to fall back TO when it cannot land.
    _rate: SpeechRate = SpeechRate(wpm=DEFAULT_WPM_NATURAL, measured=False, reason=RESOLUTION_ERROR)

    #: Appended to the prompt for the ONE corrective re-ask (see
    #: :data:`MAX_CORRECTIVE_ATTEMPTS` — attempt two is the trim, not a second
    #: re-ask, so this is set at most once per run). Empty on the first draw, and
    #: cleared again after the re-ask, so a prompt inspected outside reconciliation
    #: is unchanged.
    _correction: str = ""

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

    #: An EMPTY draw is invalid, not a result: the parser re-prompts it inside the
    #: retry budget and the stage fails when every attempt is empty. See
    #: ``shared/empty_draw.py`` for the live run this closes.
    #: The verbatim bypass (``brief.source_script``) returns from ``short_circuit``
    #: before the parser runs, so the floor never judges the customer's own text.
    min_elements: ClassVar[int] = 1

    async def handle(self, payload: LessonScriptInput, ctx: RunContext) -> LessonScriptOutput:
        """Resolve the narration-rate prior, run the LLM tail, then RECONCILE.

        ``build_prompt`` is SYNCHRONOUS on ``LLMScriptStageBase`` and is inherited
        by eight stages, so the one await this stage needs is taken here and the
        result stashed on the instance. ``dag_executor._run_stage`` constructs a
        fresh handler per stage execution, so instance state is safe (same
        precedent as ``action_reel/compile.py``). Skipped on the verbatim-bypass
        path, which builds no prompt and must not pay for a probe — and which is
        also NOT reconciled: that script is the customer's own words, and
        trimming it would delete text they supplied verbatim.

        Reconciliation is wrapped whole: a paid run must never fail because the
        length arithmetic misbehaved, so any exception returns the draw the model
        actually produced.
        """
        if self.short_circuit(ctx) is not None:
            return await super().handle(payload, ctx)
        self._rate = (await resolve_effective_wpm(ctx, texts=(), default=DEFAULT_WPM_NATURAL)).rate
        self._wpm = self._rate.wpm
        self._correction = ""
        try:
            first = await super().handle(payload, ctx)
        except StageRetriesExhaustedError as exc:
            # ``min_elements = 1``: every attempt drew empty. Named for what it is.
            failure = empty_draw_failure(exc, stage=self.name)
            if failure is None:
                raise
            raise failure from exc
        require_elements(first.paragraphs, stage=self.name)
        try:
            reconciled = await self._reconcile_narration(first, payload, ctx)
        except Exception:  # a paid run must never fail on length arithmetic
            logger.warning(
                "lesson_script narration reconciliation failed — shipping the unreconciled script",
                exc_info=True,
            )
            reconciled = first
        require_elements(reconciled.paragraphs, stage=self.name)
        return reconciled

    def _cast_segments(
        self, out: LessonScriptOutput, ctx: RunContext
    ) -> list[tuple[str, str | None]]:
        """``(text, voice_id)`` per paragraph — the voice ``lesson_narrate`` WILL use.

        Deliberately the same call ``lesson/narrate.py::voice_for`` makes, with the
        same ``_DEFAULT_CAST``, rather than a second implementation of the mapping:
        a budget resolved by a parallel copy of the cast rule is a budget that goes
        wrong the first time somebody edits one of the two. Dialogue-off runs take
        ``voice_for_speaker``'s own gate and come back as the single
        ``voice.voice_id`` for every paragraph.
        """
        voice = ctx.config_snapshot.get("voice") or {}
        brief = ctx.config_snapshot.get("brief") or {}
        return [
            (
                p.text,
                voice_for_speaker({"speaker": p.speaker}, voice, brief, default_cast=_DEFAULT_CAST),
            )
            for p in out.paragraphs
        ]

    async def _measure(self, out: LessonScriptOutput, ctx: RunContext) -> ScriptMeasurement:
        """Synthesize EVERY paragraph of this draft and return what it really runs to.

        This replaces the old estimate-at-a-blended-cast-rate, and the difference is
        not one of degree: a rate times a word count is a projection — several
        percent of error on exactly the axis this stage is graded on, and on a
        Socratic two-hander the blend is an average of voices that speak unequal
        shares. Measuring every paragraph at its OWN cast voice yields the lesson's
        delivered length, because a lesson IS the sum of its narration clips
        (``lesson/compile.py`` rescales the slides to whatever audio came back).

        It is very nearly free. ``speed`` and ``stage_name`` are the ones
        ``lesson_narrate`` will REQUEST (see :data:`NARRATE_STAGE_NAME` and
        :data:`NARRATE_SPEED`) and the voices are the ones it will CAST, so
        ``deterministic_clip_id`` makes every clip measured here the very clip
        narrate is about to ask for: each surviving paragraph's synthesis is paid
        once, here, instead of once, later. Only the paragraphs a re-ask replaces or
        a trim drops are genuinely spent.

        AN AUTHORED RATE IS NOT MEASURED AGAINST. When the operator wrote
        ``direction.timing.wpm.natural`` the ladder hands back an ``authored`` basis,
        and that rate "outranks every measurement" — so synthesizing the script here
        would buy a number this stage is contractually forbidden to use, at the cost
        of a whole lesson of TTS. The estimate falls back to the prior-derived
        arithmetic, at the authored rate, and carries ``reason=None`` because nothing
        FAILED: the measurement was declined, not attempted and lost. (The clips are
        not lost either — ``lesson_narrate`` synthesizes them later, as it did before
        Phase C existed.)

        Never raises: ``measure_script_lines`` absorbs every failure into a
        prior-derived measurement carrying a ``reason``.
        """
        segments = self._cast_segments(out, ctx)
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

        The same SHAPE ``measure_script_lines`` returns when it declines — a
        per-paragraph ``words / wpm`` estimate carrying the prior, ``measured_count=0``,
        no failed index — except for ``reason``, which is ``None`` here BECAUSE NOTHING
        FAILED. A reason would put a ``SPEECH_RATE_UNUSABLE`` degradation on a run whose
        rate the operator supplied, i.e. report the one case this stage handled exactly
        as instructed as the case it handled worst.

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
        statement about the delivery, not about a voice, so it applies to every
        paragraph regardless of who speaks it — the per-voice measurement that makes
        the cast blend honest is exactly what was declined.

        A blank paragraph's ``0.0`` is exact, so it counts as measured — the same
        rule ``_prior_measurement`` applies, and what keeps ``measured_lines`` honest.
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

    def _draw_of(self, out: LessonScriptOutput, measurement: ScriptMeasurement) -> _Draw:
        """A candidate paired with the seconds it will be JUDGED at.

        When the measured rate is unusable the measurement's own seconds are not
        either — a 0/NaN rate yields 0.0 per paragraph, which would read as a 0.0s
        script and send a perfectly good draw off to be lengthened. That case falls
        back to word-count arithmetic, where ``estimate_seconds`` substitutes the
        documented default and ``speech_rate_degradation`` records that it did.
        Nothing here is silent.
        """
        texts = [p.text for p in out.paragraphs]
        if wpm_is_usable(measurement.wpm) and len(measurement.per_line_seconds) == len(texts):
            return _Draw(
                output=out,
                paragraph_seconds=measurement.per_line_seconds,
                measured=measurement.measured,
                estimate=measurement.total_seconds,
                wpm=measurement.wpm,
                basis=measurement.basis,
            )
        seconds = tuple(
            estimate_seconds([text], words_per_minute=measurement.wpm) for text in texts
        )
        return _Draw(
            output=out,
            paragraph_seconds=seconds,
            measured=(False,) * len(texts),
            estimate=math.fsum(seconds),
            wpm=measurement.wpm,
            basis=measurement.basis,
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
        verdict: str,
        payload: LessonScriptInput,
        ctx: RunContext,
        target_words: int | None = None,
    ) -> LessonScriptOutput | None:
        """ATTEMPT ONE: ask the model again, through this stage's existing LLM call.

        ``None`` when the call failed — a re-ask is a correction, and a paid run
        that already holds a usable draw must not die because the correction did.

        ``target_words`` overrides the budget the correction asks for. The default
        — ``budget_words`` at ``self._wpm`` — is the rate LAST measured, which is
        the right number for the first correction. The trim-floored re-ask in step
        (2b) passes the target derived from the SHIPPING draw's own measured rate
        instead, because that is the draw whose word count is demonstrably wrong.
        """
        self._correction = corrective_block(
            verdict,
            produced_words=count_words([p.text for p in draw.output.paragraphs]),
            estimated_seconds=draw.estimate,
            ordered_seconds=ordered,
            target_words=(
                budget_words(ordered, words_per_minute=self._wpm)
                if target_words is None
                else target_words
            ),
            previous=[p.text for p in draw.output.paragraphs],
        )
        try:
            # A correction, not a draw: parsed with no floor, so an empty answer
            # costs ONE call and returns as zero elements for the caller to record
            # as a failed re-ask (``LLMScriptStageBase.corrective_draw``).
            with self.corrective_draw():
                return await super().handle(payload, ctx)
        except Exception:  # a failed re-ask degrades to the first draw
            logger.warning("lesson_script corrective re-ask failed", exc_info=True)
            return None
        finally:
            self._correction = ""

    async def _reconcile_narration(
        self,
        first: LessonScriptOutput,
        payload: LessonScriptInput,
        ctx: RunContext,
    ) -> LessonScriptOutput:
        """Steer at the ORDER, correct until in band or out of attempts, never pad.

        Measured live 2026-09-13: FIVE consecutive 120s orders delivered +4.21%,
        +4.48%, +11.03%, +24.03% and +26.35%, and two of those ALSO carried a false
        ``duration_out_of_tolerance`` raised from the pre-audio estimate while the
        delivered film was in band. Both halves of that come from the same cause:
        every number in the loop was a PROJECTION — ``words / wpm`` at a blended
        cast rate — so the check fired on films that were fine and missed films that
        were not.

        Three properties, each of which was a measured live miss:

        1. **It aims at the target, not at the band edge.** The decision to correct
           is :func:`steer_verdict` (long side +4%), not the +/-10% band that GRADES
           the master — a draw accepted at +9% has nothing left for the estimate's
           own error. The +/-10% band still decides what is REPORTED, so a
           conformant lesson is never flagged.
        2. **It iterates, and it spends the same number of attempts in both
           directions.** Up to :data:`MAX_CORRECTIVE_ATTEMPTS` corrections. A LONG
           draw gets the re-ask, then the trim. A SHORT draw gets the re-ask, then
           a SECOND re-ask — because the trim is long-only by construction, so
           without it a short draw had one attempt where a long draw had two, and
           shipped whatever the single re-ask reached. "Closer" alone was the bug:
           a re-ask that swings a +26% draft to -18% is closer and would end the
           loop, while the +26% draft it is holding is TRIMMABLE at no cost.
        3. **Every candidate is judged at ITS OWN measured seconds** — synthesized
           at the voices that will speak it.

        Still no padding. Nothing lengthens a short script MECHANICALLY: repeated
        audio is a worse defect than a short lesson. Asking the model for more
        material is not padding, and a short draw now gets that ask twice before it
        is ACCEPTED and recorded — which is the whole of the change. A draw that is
        still short after both asks ships exactly as it did before, with the same
        advisory.
        """
        ordered = _target_seconds(ctx.config_snapshot.get("brief") or {})
        if ordered <= 0.0:
            # No order means no band and no verdict; inventing one is the error
            # this refuses to make. The ONLY early exit: "no paragraphs" used to
            # leave here too, and ``handle`` now refuses that script before this
            # is called.
            return first

        measurement = await self._measure(first, ctx)
        self._wpm = measurement.wpm
        rate_entries = self._rate_entries(measurement)

        best = self._draw_of(first, measurement)
        pool = [best]
        first_estimate = best.estimate
        action = "kept first draw"
        reasked = False
        reask_kept: bool | None = None

        # (1) ATTEMPT ONE — the re-ask, in whichever direction the steer points.
        steer = steer_verdict(best.estimate, ordered)
        if steer in ("long", "short"):
            second = await self._reask(best, ordered, steer, payload, ctx)
            reasked = second is not None
            best, rate_entries, kept = await self._consider(
                second, best, pool, ordered, ctx, rate_entries
            )
            reask_kept, action = _first_reask_record(second, kept, action)

        # (1b) ATTEMPT TWO, SHORT SIDE — a second re-ask.
        #
        #      The trim below is LONG-only by construction (``_trim_step`` filters
        #      the pool to candidates whose verdict is "long"), so a draw that came
        #      in short and is STILL short after one re-ask has no second attempt at
        #      all: it ships as drawn. That is the asymmetry. Measured on the three
        #      live 120s lessons that missed, one re-ask moved 69.0s -> 101.3s,
        #      88.6s -> 105.7s and 78.6s -> 107.8s — real progress that still landed
        #      at -15.6%, -11.9% and -10.2%, all outside the +/-10% band that grades
        #      the master.
        #
        #      This is an ASK, not padding. Nothing is repeated and nothing is
        #      stretched; the model is asked for more material, which is precisely
        #      what the "no mechanical lengthening" rule above already allows.
        #      ``closer_to_order`` keeps whichever draw is nearer the order, so a
        #      second ask that overshoots long is discarded rather than shipped.
        #
        #      Guarded on THE TRIM HAVING NOTHING TO DO — still short, and no LONG
        #      candidate anywhere in the pool. Whenever a long draw exists (a short
        #      first draft re-asked into a long one, or the reverse), ``_trim_step``
        #      can reach it deterministically and for free, and a paid LLM call must
        #      not pre-empt a free lever. Keying this on the ORIGINAL steer instead
        #      was wrong for exactly that case, and the tie-keeps-the-first test is
        #      what says so.
        if steer_verdict(best.estimate, ordered) == "short" and not any(
            steer_verdict(draw.estimate, ordered) == "long" for draw in pool
        ):
            third = await self._reask(best, ordered, "short", payload, ctx)
            best, rate_entries, kept = await self._consider(
                third, best, pool, ordered, ctx, rate_entries
            )
            if kept:
                reask_kept = True
                action = "re-asked twice"
            elif third is not None and not third.paragraphs:
                action = f"{action}; second re-ask returned no paragraphs"

        # (2) ATTEMPT TWO, LONG SIDE — the trim. Deterministic, free (it selects
        #     paragraphs whose seconds are already measured), and the only lever
        #     that can rescue a draw the re-ask overshot: the LONG draft in the pool
        #     is trimmable even when the SHORT one is currently winning.
        floored: TrimPlan | None = None
        trimmed_paragraphs = 0
        if steer_verdict(best.estimate, ordered) != "ok":
            best, action, floored, trimmed_paragraphs = self._trim_step(
                best, pool, ordered, ctx, action
            )

        # (2b) ATTEMPT THREE, LONG SIDE — see ``_floored_reask_step``. Fired ONLY
        #      when the trim could not act at all, so a trim that DID drop something
        #      is never pre-empted by a paid LLM call.
        if trimmed_paragraphs == 0 and floored is not None:
            best, rate_entries, kept, action, floored = await self._floored_reask_step(
                best, pool, ordered, payload, ctx, rate_entries, action, floored
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
            trimmed_lines=trimmed_paragraphs,
            measured_lines=best.measured_paragraphs,
        )

        # Reported against the OUTER band — the one that grades the delivered
        # master — so a lesson this stage steered to +2% raises nothing, and a
        # lesson the steer merely improved to +6% raises nothing either. It IS
        # conformant; a second, worse, pre-audio opinion saying otherwise is how a
        # good delivery gets read as broken, and it is exactly what two of the five
        # live 120s runs did.
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

    async def _consider(
        self,
        drawn: LessonScriptOutput | None,
        best: _Draw,
        pool: list[_Draw],
        ordered: float,
        ctx: RunContext,
        rate_entries: list[Degradation],
    ) -> tuple[_Draw, list[Degradation], bool | None]:
        """Measure a re-asked draw, pool it, and keep it ONLY if it is closer.

        Shared by both asks so the two cannot drift apart: a draw is measured at
        the voices that will speak it, its rate degradations are merged, it joins
        the pool (the trim reaches pooled LONG draws even when a shorter one is
        winning), and :func:`closer_to_order` — strict ``<``, ties keep the
        incumbent — decides whether it displaces ``best``.

        Returns the possibly-updated ``best``, the merged entries, and whether the
        draw was kept: ``None`` means there was no usable draw to judge at all,
        which is NOT the same as a draw that was judged and rejected.
        """
        if drawn is None or not drawn.paragraphs:
            return best, rate_entries, None

        measurement = await self._measure(drawn, ctx)
        self._wpm = measurement.wpm
        merged = _merge_entries(rate_entries, self._rate_entries(measurement))
        candidate = self._draw_of(drawn, measurement)
        pool.append(candidate)
        if closer_to_order(candidate.estimate, best.estimate, ordered):
            return candidate, merged, True
        return best, merged, False

    async def _floored_reask_step(
        self,
        best: _Draw,
        pool: list[_Draw],
        ordered: float,
        payload: LessonScriptInput,
        ctx: RunContext,
        rate_entries: list[Degradation],
        action: str,
        floored: TrimPlan,
    ) -> tuple[_Draw, list[Degradation], bool | None, str, TrimPlan | None]:
        """ATTEMPT THREE, LONG SIDE: re-ask with an EXPLICIT WORD TARGET.

        The mirror of step (1b) on the long side. ``_trim_step`` is the long side's
        free second attempt — but it can come back having done NOTHING, and then
        the long side has had exactly one attempt where the short side has two, and
        the stage ships whatever the single re-ask reached.

        Live run ``01a0a34a-b050-753c-93a3-fefb5dc6fa99``, a 120s lesson: the first
        draw estimated 85.57s, the re-ask overcorrected to 137.795s (+14.85%), and
        ``plan_trim``'s exhaustive subset-sum over the four paragraphs (85/79/84/93
        words at 148.4814 wpm) found that NONE of the fifteen subsets lands inside
        [108.0, 132.0] — the best single drop reaches 105.87s, 2.13s UNDER the
        floor. So ``declined_under_band`` was True, ``dropped_indices`` was empty,
        ``trimmed_lines`` was 0, and the stage shipped +14.85%. That was the end of
        the long side. This is the attempt that was missing.

        **Only when the trim genuinely could not act.** The caller gates on zero
        paragraphs dropped AND a floored plan; this gates on the draw still being
        LONG. A trim that DID drop something has already spent the free lever, and
        a paid LLM call must never pre-empt a free one — precisely the property
        (1b)'s short-side guard protects, and
        ``test_a_trim_that_dropped_something_is_never_pre_empted_by_a_paid_re_ask``
        is what says so here.

        **The target is this draw's OWN arithmetic**, ``ordered * draw.wpm / 60``,
        not ``budget_words`` at ``self._wpm`` (whatever was measured LAST, possibly
        a discarded draft). For the live case: ``120.0 * 148.4814 / 60`` = 297
        words, against the 341 actually written. A word count is a thing the model
        can honour; "it is too long" demonstrably is not.

        ``_consider`` still decides with ``closer_to_order``, so a draw further
        from the order than the incumbent is DISCARDED rather than shipped — the
        invariant PR #995 added, and it holds here unchanged. When the new draw IS
        kept, the floor is cleared: the floored plan describes a script that no
        longer ships, and ``_trim_step`` already refuses to report a floor for a
        script it did not ship.
        """
        if steer_verdict(best.estimate, ordered) != "long":
            return best, rate_entries, None, action, floored

        target_words = round(ordered * best.wpm / 60.0)
        logger.info(
            "%s: trim floored having dropped nothing — re-asking at an explicit "
            "%d-word target (estimate %.2fs vs ordered %.2fs)",
            self.name,
            target_words,
            best.estimate,
            ordered,
        )
        drawn = await self._reask(best, ordered, "long", payload, ctx, target_words=target_words)
        best, rate_entries, kept = await self._consider(
            drawn, best, pool, ordered, ctx, rate_entries
        )
        if not kept:
            return best, rate_entries, kept, action, floored
        return (
            best,
            rate_entries,
            kept,
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

        **It trims any LONG candidate in the pool, not only the incumbent.** When a
        re-ask has swung a long draft to a shorter-but-still-wrong one, that shorter
        draw wins on "closer" and a trim of it is then impossible (nothing shortens
        a short script) — so the loop used to stop with both drafts out of band
        while one of them was one dropped paragraph from conformant. Trimming a
        candidate costs no synthesis and no LLM call. The incumbent is tried FIRST,
        and ties keep it, so a single-candidate run is unchanged.

        ``floored`` is the plan when the band is STILL out of reach afterwards,
        which "``+ trimmed``" alone reads exactly like a landed trim.

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
            dropped = len(source.output.paragraphs) - len(trimmed.output.paragraphs)
            scope = "" if source is incumbent else f" the {_pool_label(source, pool)}"
            action = f"{action} + trimmed{scope} {dropped} paragraph(s)"
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

        The PLAN is always returned as the second element, applied or not: whether
        the band was reachable is a property of the plan, and a caller that only
        ever saw the applied output could not tell a floored trim from a landed one.
        The third element is the REJECTION, in the caller's words.

        **It plans on MEASURED seconds, not on a word proxy.** Two paragraphs of
        equal word count do not take equal time — and in Socratic mode they are not
        even spoken by the same voice, so a word-weighted plan can drop the wrong
        pair and land outside the band it was aiming at. ``paragraph_seconds`` is
        handed to ``plan_trim`` whenever the measurement is usable; it falls back to
        words when it is not, which is the pre-measurement behaviour exactly.

        **The picture pins nothing in a lesson** — ``lesson/compile.py`` rescales
        the slides to whatever audio came back, so no paragraph is a structural
        anchor the way website_review's reveal line is, and the trim needs no
        phase/shotlist protection. A required VERBATIM phrase is different:
        ``LLMScriptStageBase.handle`` has already gated the draw on it, so dropping
        its paragraph here would silently ship a lesson without words the customer
        marked required. Those paragraphs are pinned, and the result is re-checked
        afterwards anyway.
        """
        items = items_for_stage(ctx.config_snapshot.get("verbatim_items") or [], self.name)
        plan = plan_trim(
            [p.text for p in draw.output.paragraphs],
            ordered_seconds=ordered,
            # THIS draw's rate, never ``self._wpm``. ``self._wpm`` holds whatever
            # was measured LAST, which after a re-ask is the rate of a draft that
            # may have been discarded — and ``plan_trim`` falls back to the word
            # proxy the moment ``_trimmable_seconds`` refuses the measurement, at
            # which point the wrong rate silently decides which paragraphs are cut.
            # Pinned by
            # ``test_the_trim_plans_at_the_shipped_draws_own_rate_not_the_last_measured_one``.
            words_per_minute=draw.wpm,
            tolerance=DEFAULT_TOLERANCE,
            protected_indices=_protected_indices(draw.output.paragraphs, items),
            line_seconds=_trimmable_seconds(draw),
        )
        if not plan.dropped_indices:
            return None, plan, None
        kept = draw.select(plan.keep_indices)
        if find_missing_verbatim(" ".join(p.text for p in kept.output.paragraphs), items):
            logger.warning("%s: trim would drop a required verbatim phrase — skipped", self.name)
            return None, plan, "it would have dropped a required verbatim phrase"
        return kept, plan, None

    def _objective(self, ctx: RunContext) -> str:
        brief = ctx.config_snapshot.get("brief") or {}
        return brief.get("learning_objective", "Learn the topic.")

    def short_circuit(self, ctx: RunContext) -> LessonScriptOutput | None:
        brief = ctx.config_snapshot.get("brief") or {}
        items = items_for_stage(ctx.config_snapshot.get("verbatim_items") or [], self.name)
        segs = normalize_source_script(brief.get("source_script") or "", split="paragraph")
        if not segs:
            return None
        paragraphs = [LessonScriptParagraph(id=str(uuid7()), text=t, seconds=s) for t, s in segs]
        if (brief.get("dialogue") or {}).get("enabled"):
            paragraphs = assign_dialogue_speakers(paragraphs)
        assert_verbatim_present(" ".join(p.text for p in paragraphs), items, stage=self.name)
        return LessonScriptOutput(
            paragraphs=paragraphs,
            total_seconds=sum(s for _, s in segs),
            learning_objective=self._objective(ctx),
        )

    @staticmethod
    def _dialogue_block(brief: dict) -> str:
        """Mode instructions: a Socratic two-voice lesson (opt-in) or a single narrator.

        Both branches still require a ``speaker`` per paragraph (the schema does);
        the single-narrator branch pins it to "b" (the instructor) so
        lesson_narrate — which only casts when dialogue is enabled — has a
        well-formed but inert value. The "student ends with ?, instructor does
        not" rule is load-bearing: ``assign_dialogue_speakers`` re-derives the
        speaker from exactly that question-mark heuristic, so the prompt and the
        deterministic re-derivation agree.
        """
        dialogue = brief.get("dialogue") or {}
        if not dialogue.get("enabled"):
            return '\n\nThis is a SINGLE narrator (the instructor). Set "speaker" to "b" on every paragraph.'
        personas = dialogue.get("personas") or {}
        persona_a = str(personas.get("a") or _DEFAULT_PERSONA_A)
        persona_b = str(personas.get("b") or _DEFAULT_PERSONA_B)
        return (
            "\n\nWrite this as a SOCRATIC two-voice lesson — a curious student and "
            "an instructor — NOT a single lecture. The two speakers:\n"
            f'- speaker "a" (student): {persona_a}\n'
            f'- speaker "b" (instructor): {persona_b}\n'
            "Rules:\n"
            '- The student "a" asks SHORT clarifying questions; every student turn '
            "ENDS with a question mark.\n"
            '- The instructor "b" carries most of the content: each instructor turn '
            "teaches/answers in 2-4 sentences and does NOT end with a question.\n"
            "- Alternate student question → instructor answer; the instructor may "
            "take several consecutive teaching turns between questions.\n"
            '- Start and end with the instructor "b" so the lesson opens and closes '
            "with teaching."
        )

    def build_prompt(self, ctx: RunContext) -> str:
        cfg = ctx.config_snapshot
        brief = cfg.get("brief") or {}
        voice = cfg.get("voice") or {}
        objective = self._objective(ctx)
        items = items_for_stage(cfg.get("verbatim_items") or [], self.name)
        duration_min = float(brief.get("duration_minutes", 6))
        target_seconds = _target_seconds(brief)
        notes = ctx.annotations_for("lesson_script")
        notes_block = ""
        if notes:
            joined = "\n".join(f"- {n['text']}" for n in notes)
            notes_block = "\n\nApply reviewer notes:\n" + joined
        return (
            f"Write a {duration_min}-minute teaching script.\n"
            f"{word_budget_sentence(target_seconds, self._wpm)}"
            f"{paragraph_plan_sentence(target_seconds, self._wpm)}"
            f"Voice: "
            f"{voice.get('descriptor', 'neutral')}. Learning objective: "
            f'"{objective}". Output JSON: '
            f'{{"paragraphs":[{{"text":"...","seconds":<num>,"speaker":"a"|"b"}}, ...]}}.'
            f"{self._dialogue_block(brief)}"
            f"{notes_block}"
            f"{verbatim_prompt_block(items)}"
            f"{grounding_prompt_block(ctx)}"
            # Empty on the first draw; carries the arithmetic of the miss on the
            # single corrective re-ask (``_reconcile_narration``). LAST so it is
            # the final instruction the model reads.
            f"{self._correction}"
        )

    def make_element(self, raw: dict, *, element_id: str) -> LessonScriptParagraph:
        return LessonScriptParagraph(
            id=element_id,
            text=str(raw["text"]),
            seconds=float(raw["seconds"]),
            speaker=(str(raw.get("speaker") or "b").strip().lower() or "b"),
        )

    def build_output(self, elements: list[StageOutput], ctx: RunContext) -> LessonScriptOutput:
        paragraphs = cast("list[LessonScriptParagraph]", elements)
        brief = ctx.config_snapshot.get("brief") or {}
        target_seconds = _target_seconds(brief)
        # BEFORE the speaker pass and before reconciliation measures anything: the
        # split halves are what get synthesized, sliced and (if long) trimmed.
        paragraphs = enforce_paragraph_floor(paragraphs, target_seconds)
        if (brief.get("dialogue") or {}).get("enabled"):
            paragraphs = assign_dialogue_speakers(paragraphs)
        return LessonScriptOutput(
            paragraphs=paragraphs,
            total_seconds=sum(p.seconds for p in paragraphs),
            learning_objective=self._objective(ctx),
            word_budget=word_budget(target_seconds, self._wpm),
            actual_words=count_words([p.text for p in paragraphs]),
        )
