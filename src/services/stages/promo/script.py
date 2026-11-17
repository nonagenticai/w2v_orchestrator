"""promo_script — conditioned on promo_analyze insights + CTA + reviewer notes.

Output exposes ``lines[*].id`` for annotation pinning and ``lines[*].speaker``
("a"/"b") for the two-hander dialogue mode (opt-in via ``brief.dialogue``).

When ``brief.dialogue.enabled`` is set the script is written as a natural
conversation between two people watching the product together — ``promo_narrate``
then casts each speaker to a distinct kokoro voice, so the promo reads as a real
dialogue rather than a single AI narrator. When the flag is off the prompt asks
for a single narrator and the per-line speaker is ignored downstream.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final, cast
from uuid import uuid7

from w2v_common.stages import StageRetriesExhaustedError

from src.config import get_settings
from src.services.stages.base import (
    SlotUse,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    assert_verbatim_present,
    items_for_stage,
    normalize_source_script,
    verbatim_prompt_block,
)
from src.services.stages.errors import StagePreconditionError
from src.services.stages.generic.grounding import grounding_prompt_block
from src.services.stages.primitives.dialogue import voice_for_speaker
from src.services.stages.primitives.script_stage import LLMScriptStageBase
from src.services.stages.promo.narrate import _DEFAULT_CAST, PromoNarrateStage
from src.services.stages.shared.compute_plan import beat_word_budget
from src.services.stages.shared.degradation import Degradation
from src.services.stages.shared.empty_draw import (
    empty_draw_failure,
    require_elements,
    thin_capture_clause,
)
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
from src.services.stages.shared.placeholder_text import text_or_empty
from src.services.stages.shared.shotlist import normalize_shotlist
from src.services.stages.shared.speech_rate import (
    RESOLUTION_ERROR,
    ScriptMeasurement,
    SpeechRate,
    measure_script_lines,
    resolve_cast_wpm,
    resolve_natural_wpm_detail,
    script_measurement_degradation,
    speech_rate_fallback_degradation,
)
from src.services.stages.shared.time_budget import DEFAULT_WPM_NATURAL
from src.services.stages.verbatim import find_missing_verbatim

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.clients.tts_client import TTSClient
    from src.services.llm import WMClient
    from src.services.runs.run_context import RunContext

logger = logging.getLogger(__name__)

#: The stage whose ``synthesize`` call actually speaks these lines. Read off the
#: class rather than repeated as a literal so the two cannot drift: the measurement
#: below stores its clips under THIS name, and ``deterministic_clip_id`` then makes
#: the measured clip and the narrated clip ONE object — see :data:`NARRATE_SPEED`.
NARRATE_STAGE_NAME: Final = PromoNarrateStage.name

#: The ``speed`` ``promo_narrate`` requests: it inherits ``NarrationStageBase.speed_for``,
#: which returns ``None``, so the argument is omitted entirely and
#: ``deterministic_clip_id`` hashes ``None``. The measurement MUST request the same
#: thing — ``1.0`` is a different clip id AND a different measurement (speed scales
#: duration linearly backend-side), so measuring at 1.0 would both mis-measure the
#: voice and be reused by nothing. Pinned by
#: ``test_the_measurement_requests_exactly_what_promo_narrate_will_request``.
NARRATE_SPEED: Final[float | None] = None

#: How many CORRECTIVE attempts one draw may cost, and in which order they are
#: spent: attempt 1 is the LLM re-ask, attempt 2 the deterministic trim. So the LLM
#: budget for the whole stage is exactly TWO calls — the opening draw and one re-ask
#: — and the second attempt is free, because the trim selects lines whose seconds
#: this stage has already measured.
MAX_CORRECTIVE_ATTEMPTS: Final = 2

# Default two-hander personas (overridable via brief.dialogue.personas). Speaker
# "a" asks the viewer's questions; speaker "b" guides and answers.
_DEFAULT_PERSONA_A = (
    "Maya — a sharp, slightly skeptical operator at a mid-size company. "
    "She asks the blunt questions a real viewer would."
)
_DEFAULT_PERSONA_B = (
    "Daniel — the guide who helped build this. Warm and concrete; he walks Maya "
    "through the live product on screen and answers her questions."
)


# site_capture statuses that mean the scrape did NOT succeed (as opposed to
# "succeeded and the page genuinely had no copy", which is "empty").
_GROUNDING_FAILURE_STATUSES = frozenset({"blocked", "fetch_failed"})

# Stated absence, not silence: when the site yielded no readable copy the prompt
# must SAY so, otherwise the LLM fills the gap with invented product facts.
_NO_SITE_COPY_BLOCK = (
    "\n\nNo copy could be read from the live product site. Do NOT invent product "
    "features, metrics, integrations, or customer names; stay strictly within the "
    "brief's stated value proposition."
)

# Surfaced on the stage output so an ungrounded-but-permitted run reads as degraded.
_NO_SITE_COPY_NOTE = (
    "site grounding unavailable: the live product site yielded no readable copy; "
    "the script was written from the brief alone"
)

# "thin" is the JS-only-SPA case: a real 200, but all we recovered is the <title>
# and the noscript line. Handing that over as "copy from the live product site"
# is the same falsehood as inventing — so we pass the fragment AND cap what may
# be built on it. Statuses that must degrade the run, mapped to their notes.
_THIN_SITE_COPY_CAUTION = (
    "\n\nThat is the ONLY copy that could be read from the live site — a fragment, "
    "not the page. Treat it as a hint about naming and tone only. Do NOT invent "
    "product features, metrics, integrations, or customer names beyond it; for "
    "everything else stay strictly within the brief's stated value proposition."
)
_THIN_SITE_COPY_NOTE = (
    "site grounding thin: only a fragment of the live product site was readable "
    "(likely a JavaScript-rendered page); the script leans on the brief"
)
_DEGRADING_STATUS_NOTES = {
    "empty": _NO_SITE_COPY_NOTE,
    "thin": _THIN_SITE_COPY_NOTE,
}


def _site_grounding_failure_reason(brief: dict, status: str) -> str:
    """Failure text naming the scrape target, when the brief carries one.

    ``site_capture`` grounds on ``brief.source_url`` and falls back to
    ``brief.target_url`` (the website-review shape), so read both rather than
    inventing a lookup; omit the URL entirely if neither is authored.
    """
    url = str(brief.get("source_url") or brief.get("target_url") or "").strip()
    target = f"the scrape of {url}" if url else "the site scrape"
    return (
        f"promo_script: site grounding required but {target} did not succeed ({status}); "
        "set PIPELINE_PROMO_REQUIRE_SITE_GROUNDING=false to generate an ungrounded "
        "script anyway"
    )


def _target_seconds(cfg: dict, brief: dict) -> float:
    """The ordered promo length in seconds — precedence UNCHANGED.

    Extracted verbatim from ``_build_prompt`` so ``build_output`` can re-derive
    the SAME number it budgeted the prompt against, without either copy drifting.
    """
    deliverable = cfg.get("deliverable") or {}
    return float(
        brief.get("target_duration_seconds") or deliverable.get("target_duration_seconds") or 30
    )


def word_budget_sentence(target_seconds: float, wpm: float) -> str:
    """The countable half of "approximately N seconds".

    An LLM cannot honour a duration — it has no model of how long its own words
    take to say, which is why promos ordered at 45s delivered 58.2s (129%). A word
    count it CAN honour. ``beat_word_budget`` is the existing arithmetic
    (``duration/60*wpm``); ``wpm`` is the MEASURED rate for this run's voice
    (``speech_rate.resolve_natural_wpm``), not the 150 constant, because the live
    kokoro voice measures ~94 and a budget built on 150 is ~60% too generous.

    Stated as a TARGET band (:func:`narration_budget.word_target_sentence`), not a
    ceiling: "a HARD CEILING … write fewer if the story is done sooner" made the
    model under-write, and a film is exactly its narration.
    """
    return word_target_sentence(target_seconds, wpm, word_budget(target_seconds, wpm))


def word_budget(target_seconds: float, wpm: float) -> int:
    """Whole-script word budget = ``duration_s / 60 * wpm``, rounded."""
    return round(beat_word_budget(float(target_seconds), float(wpm)).value)


def count_words(texts: list[str]) -> int:
    """Actual spoken word count of the drawn script (whitespace-split)."""
    return sum(len(t.split()) for t in texts)


def _reask_prompt_block(
    *,
    produced_words: int,
    estimated_seconds: float,
    ordered_seconds: float,
    target_words: int,
) -> str:
    """The addendum for the ONE bounded re-ask: the miss, as COUNTABLE facts.

    Everything here is a number the model can check itself against. The first draft
    already carried the word ceiling as prose and missed it — measured live over
    five 45s promos at +4.50%, +8.30%, +14.52%, +19.00% and +32.36% — so repeating
    the prose would re-run the same failure.

    ``estimated_seconds`` is a MEASUREMENT (every line of the previous draft
    synthesized at the voice that will speak it), not a projection, and
    ``target_words`` is taken at the rate that measurement pooled. That is what
    makes the re-ask safe here at all: promo's old word-only accounting could not
    tell a draw that was short in WORDS but long in SECONDS (a live run drew 12%
    under budget and delivered +13.87%), so a word-driven re-ask would have asked
    that run for MORE words and made the film longer still.
    """
    verb = "TOO LONG" if estimated_seconds > ordered_seconds else "TOO SHORT"
    return (
        "\n\nREWRITE — your previous draft was "
        f"{verb}. It contained {produced_words} words, which at this narrator's "
        f"measured pace takes about {estimated_seconds:.0f} seconds to say. The order "
        f"is {ordered_seconds:.0f} seconds, i.e. {target_words} words for the WHOLE "
        f"script (all lines combined). Rewrite the complete script to approximately "
        f"{target_words} words — keep the same structure and the same story, and cut "
        "or expand the CONTENT of the lines rather than padding with filler or "
        "repetition. Keep the closing call-to-action verbatim. Return the full script "
        "again in the same JSON shape."
    )


class PromoScriptInput(StageInput):
    pass


class PromoScriptLine(StageOutput):
    id: str
    text: str
    seconds: float
    # Two-hander speaker tag ("a"/"b"). Defaulted for single-narrator promos and
    # forward-compat with any persisted pre-dialogue script output; ignored by
    # promo_narrate unless brief.dialogue.enabled.
    speaker: str = "a"


class PromoScriptOutput(StageOutputWithNotes):
    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. Empty here means the stage
    # reported success having produced nothing, which degrades the run.
    produces: ClassVar[tuple[str, ...]] = ("lines",)

    lines: list[PromoScriptLine]
    #: The script's length in seconds. On a RECONCILED run this is the MEASURED
    #: length of the shipped draw (``_finalize`` stamps it, and it equals
    #: ``narration_estimate.estimate_seconds``). On the bypasses and on a run that
    #: could not be reconciled at all it falls back to the sum of ``lines[*].seconds``
    #: — the model's own guess, which `make_element` copies straight off the LLM and
    #: nothing validates. ``narration_estimate`` is how a reader tells the two apart.
    total_seconds: float
    # Ordered-vs-delivered accounting in WORDS; countable either way.
    word_budget: int = 0
    actual_words: int = 0
    #: What the narration budget CONCLUDED, as numbers — the rate, where it came
    #: from, the first draw's estimate, and what each correction did. Present on
    #: EVERY reconciled run, clean ones included: "this run was measured and it
    #: landed" and "nobody ever measured this run" are otherwise the same absence.
    #: ``None`` only when reconciliation could not run at all (the verbatim /
    #: shot-list bypass, or no order).
    narration_estimate: NarrationEstimate | None = None


@dataclass(frozen=True, slots=True)
class _Draw:
    """One candidate script AND the per-line seconds it was judged at.

    The two travel together because the whole defect this stage is fixing is a draw
    judged at another draw's number: promo used to compare a WORD count against a
    word budget built from a rate probed off a calibration passage, and grade a
    re-asked draft — different words, possibly a different speaker mix — at it.

    ``line_seconds`` is parallel to ``output.lines``, so a trim of the candidate is
    an index selection on both at once and needs no second measurement.
    """

    output: PromoScriptOutput
    line_seconds: tuple[float, ...]
    #: Per line: is that seconds a FACT (a real synthesis, or an exactly-0.0s blank
    #: line) rather than words-over-a-rate?
    measured: tuple[bool, ...]
    #: ``sum(line_seconds)`` — the number every verdict here is taken against.
    estimate: float
    #: The rate THIS draw's seconds are expressed at, and where it came from.
    wpm: float
    basis: str
    #: What THIS draw's measurement was faulty about — the entries that belong on the
    #: degrading tier if and only if this draw is the one that SHIPS. They travel on
    #: the draw for exactly the reason the seconds do: a re-ask whose measurement
    #: failed and was then DISCARDED must not flip the run ``degraded`` while the
    #: script that shipped was measured end to end.
    rate_entries: tuple[Degradation, ...] = ()

    @property
    def measured_lines(self) -> int:
        return sum(1 for flag in self.measured if flag)

    def select(self, indices: Sequence[int]) -> _Draw:
        """This draw reduced to ``indices`` — lines, seconds and flags together.

        A trim needs no second measurement precisely because of this: the kept lines
        are the very lines that were synthesized, so their seconds are the same
        facts, selected. Speakers are NOT re-derived: ``_assign_dialogue_speakers``
        already stamped them, the measurement was taken AT those voices, and
        re-deriving would silently re-cast a line whose audio is already measured
        and whose ``deterministic_clip_id`` promo_narrate is about to reuse.
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


def _with_lines(output: PromoScriptOutput, lines: list[PromoScriptLine]) -> PromoScriptOutput:
    """``output`` with ``lines`` replaced and every derived total re-derived.

    ``total_seconds`` and ``actual_words`` are sums OVER the lines; leaving them at
    their pre-trim values would make the stage output self-contradicting.

    The ``total_seconds`` written here is still the LLM's own arithmetic and is a
    placeholder: ``_finalize`` re-stamps it from the shipped draw's MEASURED
    seconds before the output leaves the stage. It is re-summed anyway so that an
    output built through this helper is internally consistent on its own.
    """
    return output.model_copy(
        update={
            "lines": lines,
            "total_seconds": sum(line.seconds for line in lines),
            "actual_words": count_words([line.text for line in lines]),
        }
    )


def _pool_label(draw: _Draw, pool: list[_Draw]) -> str:
    """Which draft ``draw`` is, for an action string a human reads later."""
    return "first draw" if pool and draw is pool[0] else "re-asked draw"


def _floor_note(plan: TrimPlan) -> str:
    """The action-string half of "the trim ran and could not reach the band"."""
    reason = (
        "next drop would land UNDER the band"
        if plan.declined_under_band
        else f"{plan.protected_count} protected turn(s)"
    )
    return f"STILL LONG — trim floored by {reason}"


def _trimmable_seconds(draw: _Draw) -> list[float] | None:
    """``draw``'s per-line seconds for ``plan_trim`` — or ``None`` to use words.

    ``plan_trim`` RAISES on a seconds list that is not a measurement (non-finite,
    negative, mis-length), and rightly so: silently mis-aligning seconds with lines
    cuts somebody's script by the wrong weights. But this is a paid run's happy
    path, so the check happens HERE and a measurement that is not one simply falls
    back to the word proxy rather than aborting the reconciliation.
    """
    seconds = draw.line_seconds
    if len(seconds) != len(draw.output.lines):
        return None
    if not all(math.isfinite(value) and value >= 0.0 for value in seconds):
        return None
    return list(seconds)


def protected_items(ctx: RunContext) -> list[dict]:
    """The phrases a promo trim may NOT drop — INCLUDING the call-to-action.

    THE promo-specific hazard. The CTA is requested in the prompt only
    (``End with this CTA verbatim: "..."``); it is not a ``verbatim_items`` entry,
    so ``assert_verbatim_present`` never gates it on the LLM path — it runs only on
    the two ``short_circuit`` bypasses. And ``plan_trim`` prefers dropping the
    LATEST lines, while ``_dialogue_block`` instructs speaker "b" to deliver "the
    closing call-to-action". A trim with no knowledge of the CTA therefore eats it
    FIRST, and ships a promo that advertises nothing — a worse defect than the
    overrun the trim was fixing.

    Modelled as a verbatim item rather than ONLY as "protect the last line": the CTA
    is a PHRASE, and where it lands is the model's choice. If the draft put it
    mid-script, protecting index ``-1`` alone would protect the wrong line. The two
    rules are a UNION, not an alternative — see :func:`_protected_blocks`.
    """
    items = list(items_for_stage(ctx.config_snapshot.get("verbatim_items") or [], "promo_script"))
    cta = _cta_of(ctx)
    if cta:
        items.append({"text": cta})
    return items


def _cta_of(ctx: RunContext) -> str:
    """The brief's call-to-action, or ``""`` when this promo did not order one."""
    return text_or_empty((ctx.config_snapshot.get("brief") or {}).get("call_to_action"))


def _carries_verbatim(text: str, items: list[dict]) -> bool:
    """True when ``text`` alone satisfies at least one required phrase."""
    if not items:
        return False
    return len(find_missing_verbatim(text, items)) < len(items)


def _protected_blocks(
    texts: list[str],
    blocks: tuple[tuple[int, ...], ...],
    items: list[dict],
    *,
    cta: str,
) -> tuple[int, ...]:
    """Which turn blocks the trim may NOT drop: verbatim carriers UNION the LAST one.

    **The verbatim half is not enough, and that is a REAL defect, not a theory.**
    :func:`_carries_verbatim` fires only when the model wrote the brief's CTA
    character-for-character. The prompt asks for it verbatim
    (``End with this CTA verbatim: "..."``) and nothing enforces that on the LLM
    path — ``assert_verbatim_present`` runs on the ``short_circuit`` bypasses only —
    so a paraphrase is a normal, unflagged draw. Brief CTA ``"Get the playbook."``,
    model writes ``"Grab the playbook today"``: no block is protected, ``plan_trim``
    prefers dropping the LATEST blocks, and a 3-line draft against a 20s order ships
    ONE line with no call to action at all. The post-trim re-check in
    :meth:`PromoScriptStage._apply_trim` cannot catch it either — it subtracts what
    was missing BEFORE the trim, and the paraphrased CTA was already missing, so the
    set difference is empty and the trim is approved.

    **So the LAST block is protected unconditionally whenever a CTA was ordered.**
    Chosen over the two alternatives on one property: it cannot be defeated by
    paraphrase.

    * A fuzzy or token-overlap match would need a threshold, and a threshold is a
      number somebody has to be right about — too loose and it pins an unrelated
      closing line, too tight and ``"Grab the playbook today"`` still slips. A full
      paraphrase sharing no tokens defeats it outright. A guard whose calibration
      decides whether a paid deliverable keeps its CTA is not a guard.
    * REFUSING any trim that would drop the final block (rather than protecting it)
      kills the WHOLE plan — the overrun then ships with every other droppable block
      still in it. Protection is strictly better: the trim keeps firing on the
      blocks it may legitimately take and simply routes around the closer.

    The structural claim it rests on: promo's closer IS the last block. The prompt
    says "End with this CTA", ``_dialogue_block`` assigns the "closing
    call-to-action" to speaker "b", and ``_turn_blocks`` groups ``b`` with the ``a``
    that precedes it — so on a two-hander the protected unit is the whole closing
    exchange. The verbatim half of the union is what still covers a CTA the model
    chose to put mid-script.

    Cost when the model DID write the CTA verbatim at the end: none — the two halves
    name the same block. Cost otherwise: one block of trim reach, which
    ``plan_trim`` absorbs by searching the remaining blocks, and reports honestly via
    ``protected_count`` when it cannot reach the band without it.

    ⚠️ KNOWN, NOT FIXED HERE — protection is decided per LINE, while the trim plans
    per BLOCK. A CTA SPLIT ACROSS A TURN PAIR (``a``: "Grab the playbook" / ``b``:
    "today, free.") satisfies neither line on its own, so ``_carries_verbatim`` says
    no for both. Inside the last block that no longer matters (the block is
    protected regardless); a split CTA the model placed MID-script is still
    unprotected, and the post-trim re-check then rejects the entire plan and ships
    the overrun. The fix would be to match against the JOINED block text — the same
    string ``plan_trim`` is given — but that widens what counts as a carrier for
    every verbatim item, not just the CTA, so it is deliberately left alone.
    """
    protected = {
        index
        for index, block in enumerate(blocks)
        if any(_carries_verbatim(texts[i], items) for i in block)
    }
    if cta and blocks:
        protected.add(len(blocks) - 1)
    return tuple(sorted(protected))


def _turn_blocks(lines: list[PromoScriptLine], *, dialogue: bool) -> tuple[tuple[int, ...], ...]:
    """The units the trim is allowed to drop: single lines, or whole TURN PAIRS.

    THE second promo-specific hazard. promo's lines are DIALOGUE TURNS, not
    narration beats: ``_assign_dialogue_speakers`` stamps a/b and ``promo_narrate``
    casts them to two distinct kokoro voices. Dropping ONE line out of ``a b a b``
    leaves ``a a b`` — the same voice speaking twice in a row, an answer with no
    question, a two-hander that reads broken. That is worse than an 8% overrun.

    So in dialogue mode the trim plans over ADJACENT PAIRS: the script is grouped
    ``(0,1) (2,3) …`` (a trailing odd line forms a block of its own) and a whole
    block is kept or dropped together.

    **Why that is sufficient, and why there is no second "did alternation survive?"
    check beside it.** Dropping a 2-line block removes three adjacencies —
    ``(end_{k-1}, start_k)``, ``(start_k, end_k)``, ``(end_k, start_{k+1})`` — and
    adds one, ``(end_{k-1}, start_{k+1})``. Suppose the added one is a same-speaker
    pair, i.e. ``end_{k-1} == start_{k+1} == v``, and that none of the three removed
    ones was: then ``start_k != v`` and ``end_k != v``, so over the two-symbol
    alphabet ``{a, b}`` both equal the other symbol — contradicting
    ``start_k != end_k``. So a whole-block drop can NEVER increase the number of
    same-speaker neighbours, whatever the draft looked like going in. A trailing
    1-line block sits at the end and removes an adjacency without adding one.

    A counting check would therefore have been arithmetically unfalsifiable — a
    guard that cannot fail is not a guard. The property is enforced by the BLOCKS,
    and that is what ``test_a_trim_in_dialogue_mode_keeps_the_a_b_alternation``
    pins: mutate this function to one line per block and the test goes red.

    Single-narrator promos (the default — ``brief.dialogue.enabled`` is opt-in) get
    one line per block, i.e. exactly the line-wise trim the siblings run. There is
    no two-hander to break: ``promo_narrate.voice_for`` returns ``voice.voice_id``
    for every line regardless of any ``speaker`` the model emitted.
    """
    if not dialogue:
        return tuple((i,) for i in range(len(lines)))
    return tuple(tuple(range(i, min(i + 2, len(lines)))) for i in range(0, len(lines), 2))


def _finalize(
    output: PromoScriptOutput,
    *,
    degradations: list[Degradation],
    advisories: list[Degradation] | None = None,
    estimate: NarrationEstimate | None = None,
) -> PromoScriptOutput:
    """``output`` carrying what this run CONCLUDED — on the right three channels.

    ``degradations`` is evidence of a bad delivery and flips the run ``degraded``;
    ``advisories`` is the same entry shape on the tier that does not (a pre-audio
    estimate is not a verdict on audio nobody has heard yet); ``narration_estimate``
    is the numbers themselves, recorded on EVERY reconciled run so a clean one is
    distinguishable from one nobody measured.

    **And ``total_seconds`` is re-stamped from the MEASUREMENT.** Everywhere else it
    is built it is a sum over ``lines[*].seconds``, which ``make_element`` copies
    straight off the LLM and nothing has ever checked — the model's guess at its own
    pace. A trim re-derives that sum (``_with_lines``) and so persists a smaller
    fiction rather than a fact: a live-shaped case has the shipped script recorded
    at 3.0s when its lines were measured at 30.0s, an order of magnitude out, with
    nothing in the output saying which number to believe. Once a draw has been
    synthesized at the voices that will speak it there IS a measured length, so that
    is what the field carries.

    Deliberately NOT propagated down onto ``lines[*].seconds``: those are read by
    promo_storyboard/render for per-beat timing, and re-basing them is a change to
    the delivered film, not to a report. So on a reconciled run ``total_seconds`` is
    the measured whole and need not equal the sum of the per-line guesses — it is
    the same number as ``narration_estimate.estimate_seconds``, which is where a
    reader can see the two agree.

    ``estimate is None`` is the un-reconciled path (no lines, no order, no LLM draw)
    where nothing was measured, and a non-finite measurement lands
    ``estimate_seconds=None``; both leave the sum alone rather than persisting a
    number nobody took.
    """
    carried = output.model_copy(
        update={
            "degradations": [*output.degradations, *degradations],
            "advisories": [*output.advisories, *(advisories or [])],
            "narration_estimate": estimate,
        }
    )
    if estimate is None or estimate.estimate_seconds is None:
        return carried
    return carried.model_copy(update={"total_seconds": float(estimate.estimate_seconds)})


class PromoScriptStage(LLMScriptStageBase[PromoScriptInput, PromoScriptOutput]):
    """promo_script — draw the narration, MEASURE it, then correct it to the order.

    **Why promo now has a corrective lever, when it deliberately did not.** The old
    accounting compared WORDS to a word budget and recorded the miss. It refused to
    act on that because the two halves disagreed on DIRECTION: the live run that
    drew 115 words against a 131-word budget was 12% SHORT in words and delivered
    51.240s against a 45.0s order — +13.87%, i.e. LONG in seconds. A word-driven
    re-ask would have asked that run for MORE words and made the film longer.

    Measuring SECONDS resolves that ambiguity, and that is the whole reason a lever
    is safe now. Every line of every draft is synthesized at the voice
    ``promo_narrate`` will cast it to, so the number a verdict is taken against is
    the film's delivered length rather than a projection of it — a promo IS its
    narration. Measured live over five 45s orders the old path delivered +4.50%,
    +8.30%, +14.52%, +19.00% and +32.36%, one of them drawing 152 words against a
    131-word budget.

    The loop is the siblings' (``app_review``, ``website_review``), with two
    promo-specific guards that are NOT optional:

    * :func:`protected_items` pins the CALL-TO-ACTION. It is requested in the prompt
      only and gated by nothing on the LLM path, and ``plan_trim`` drops the LATEST
      lines — where the CTA lives. A trim without this ships a promo that advertises
      nothing.
    * :func:`_turn_blocks` makes the trim drop whole TURN PAIRS in dialogue mode, so
      the a/b alternation survives, and :func:`_adjacent_repeats` refuses any trim
      that would leave more same-speaker neighbours than it found.

    The measurement is very nearly free: :data:`NARRATE_SPEED` and
    :data:`NARRATE_STAGE_NAME` are what ``promo_narrate`` will REQUEST, so
    ``deterministic_clip_id`` makes each clip measured here the very clip narrate is
    about to reuse. Only the lines a re-ask replaces or a trim drops are spent.

    Still no padding. Nothing lengthens a short script mechanically — the re-ask may
    ask for more material, but repeated audio is a worse defect than a short film,
    so a short draw is ACCEPTED and recorded.
    """

    name = "promo_script"
    input_model = PromoScriptInput
    output_model = PromoScriptOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (
        SlotUse("llm"),
        SlotUse("tts"),
    )
    element_key = "lines"
    # Per-line shape carries a speaker enum so dialogue mode can alternate "a"/"b".
    # Single-narrator promos still emit it (WM only emits schema-named fields) but
    # promo_narrate ignores it unless dialogue is enabled.
    element_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "seconds": {"type": "number"},
            "speaker": {"type": "string", "enum": ["a", "b"]},
        },
        "required": ["text", "seconds", "speaker"],
    }

    #: This run's narration rate — the number the ASKS are written from (the word
    #: budget in the prompt, the target in the re-ask). Overwritten per execution in
    #: ``handle`` with the PHASE-A cast prior, and again in ``_reconcile`` with the
    #: rate pooled off the drafted lines themselves once they have been synthesized.
    #: It is NOT what a verdict is taken against any more — that is ``_Draw.estimate``,
    #: real measured seconds. The class default is the fallback so a direct
    #: ``build_prompt``/``_build_prompt`` call (tests, tooling) still works.
    _wpm: float = DEFAULT_WPM_NATURAL

    #: The Phase-A prior, with its provenance. Kept beside ``_wpm`` because the number
    #: alone cannot say whether anybody measured it — and because the full-script
    #: measurement needs something to fall back TO when it cannot land.
    _rate: SpeechRate = SpeechRate(wpm=DEFAULT_WPM_NATURAL, measured=False, reason=RESOLUTION_ERROR)

    #: Appended to the prompt by ``build_prompt`` for the ONE bounded re-ask (see
    #: :data:`MAX_CORRECTIVE_ATTEMPTS` — attempt two is the trim, not a second
    #: re-ask). Empty for the first draw, and cleared again afterwards, so a prompt
    #: inspected outside reconciliation is unchanged.
    _reask_block: str = ""

    #: Per-voice calibration faults noticed while resolving the PROMPT budget. They
    #: are held on the instance rather than pushed onto the first draw's output
    #: because the draw that ships may be the RE-ASKED one, which never saw them —
    #: ``_finalize`` re-attaches them to whichever output actually ships.
    _cast_entries: tuple[Degradation, ...] = ()

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
    #: The verbatim bypass (authored shotlist / ``source_script``) returns from
    #: ``short_circuit`` before the parser runs, so the floor never judges it.
    min_elements: ClassVar[int] = 1

    async def handle(self, payload: PromoScriptInput, ctx: RunContext) -> PromoScriptOutput:
        """Resolve the cast rate, run the shared LLM tail, then RECONCILE.

        ``build_prompt`` is SYNCHRONOUS on ``LLMScriptStageBase`` and is inherited
        by eight stages, so the one await this stage needs is taken here and the
        result stashed on the instance. That is safe: ``dag_executor._run_stage``
        constructs a fresh handler per stage execution (same precedent as
        ``action_reel/compile.py``).

        The verbatim / shot-list bypass is untouched end to end: it builds no
        prompt, pays for no probe, measures nothing and is reconciled against
        nothing. An authored shotlist IS the picture length — correcting the
        operator's own narration toward a target they did not ask for would
        manufacture a defect out of their decision.

        Reconciliation is wrapped whole: a paid run must never fail because the
        length arithmetic misbehaved, so any exception returns the draw the model
        actually produced.
        """
        self._cast_entries = ()
        self._reask_block = ""
        bypassed = self.short_circuit(ctx) is not None
        if not bypassed:
            pending: list[Degradation] = []
            self._rate = await self._budget_rate(ctx, pending)
            self._wpm = self._rate.wpm
            self._cast_entries = tuple(pending)
        try:
            first = await super().handle(payload, ctx)
        except StageRetriesExhaustedError as exc:
            # ``min_elements = 1``: every attempt drew empty. Named for what it is.
            failure = empty_draw_failure(exc, stage=self.name, context=thin_capture_clause(ctx))
            if failure is None:
                raise
            raise failure from exc
        if bypassed:
            return first
        require_elements(first.lines, stage=self.name, context=thin_capture_clause(ctx))
        try:
            reconciled = await self._reconcile(first, payload, ctx)
        except Exception:
            logger.warning(
                "promo_script narration reconciliation failed — shipping the unreconciled script",
                exc_info=True,
            )
            reconciled = _finalize(first, degradations=list(self._cast_entries))
        require_elements(reconciled.lines, stage=self.name, context=thin_capture_clause(ctx))
        return reconciled

    async def _budget_rate(self, ctx: RunContext, pending: list[Degradation]) -> SpeechRate:
        """The rate to budget the prompt at: the CAST's blended rate, or the voice's.

        ``promo_narrate`` speaks a dialogue promo in TWO voices, and they do not talk
        at the same speed — the live pair measures ``af_heart`` at 175.24 wpm against
        ``am_michael`` at 156.60. A budget taken at
        ``config_snapshot["voice"]["voice_id"]`` alone therefore describes a voice
        that may never speak: on the sibling lesson stage that exact gap turned a
        +23.07% overrun into a +7.6% estimate, the verdict came back ``ok``, and the
        trim was never reached.

        **Single-narrator and uncast promos are untouched.** When the cast collapses
        to the configured voice — every ``brief.dialogue.enabled``-off promo, and any
        run whose cast ids are absent from ``VOICE_CATALOG`` — this returns exactly
        the single-voice rate the stage resolved before, without probing a second
        voice, so those runs keep the identical prompt.

        **Only the NARRATION BUDGET moves.** This changes the word ceiling stated in
        the prompt (and the ``word_budget`` recorded beside it); it does not touch
        ``lines[*].seconds``, the authored shotlist (``short_circuit`` returns before
        this is ever called), or anything the picture is cut against.

        **It is the PRIOR, not the verdict rate.** Under the #998 rule only the
        FIRST draft's budget may be written from a prior; the re-ask target, the
        verdict and the trim all run on the rate measured off the drafted lines
        (``_reconcile``). This number's job is to put the opening draw in the right
        order of magnitude, nothing more.
        """
        base = await resolve_natural_wpm_detail(ctx, default=DEFAULT_WPM_NATURAL)
        segments = _cast_segments(ctx)
        configured = (ctx.config_snapshot.get("voice") or {}).get("voice_id")
        distinct = {voice for _, voice in segments}
        if not distinct or distinct == {configured}:
            return base

        rate = await resolve_cast_wpm(ctx, segments=segments, default=base.wpm)
        pending.extend(
            speech_rate_fallback_degradation(stage=self.name, rate=per, voice=voice)
            for voice, per in rate.per_voice.items()
            if per.assumed
        )
        if not rate.weighted:
            return base
        logger.info(
            "promo_script cast rate %.1f wpm over %d voices (configured %r was %.1f wpm)",
            rate.wpm,
            len(rate.per_voice),
            configured,
            base.wpm,
        )
        # A blend is only a MEASUREMENT when every voice in it was measured; one
        # unprobeable voice contributed the caller's default, and `assumed` is what
        # `speech_rate_fallback_degradation` above has already reported.
        assumed = [per for per in rate.per_voice.values() if per.assumed]
        if assumed:
            return SpeechRate(wpm=rate.wpm, measured=False, reason=assumed[0].reason)
        return SpeechRate(wpm=rate.wpm, measured=True)

    # ------------------------------------------------------------------ #
    # narration reconciliation
    # ------------------------------------------------------------------ #

    def _line_voices(self, output: PromoScriptOutput, ctx: RunContext) -> list[str | None]:
        """The voice ``promo_narrate`` WILL speak each line in — one per line.

        Resolved through the SAME ``voice_for_speaker`` rule ``promo_narrate.voice_for``
        applies, on ``promo_narrate``'s OWN ``_DEFAULT_CAST`` (imported, never
        copied). It is load-bearing twice over:

        * the measurement synthesizes each line AT THIS VOICE, so a wrong voice
          measures audio nobody will hear;
        * ``deterministic_clip_id`` hashes ``(text, voice, speed)``, so a wrong voice
          also breaks the clip REUSE — ``promo_narrate`` would find no existing clip
          and the run would pay for every line TWICE.

        The dialogue gate comes along with the rule, so a single-narrator promo
        measures every line at ``voice.voice_id`` exactly as it will narrate.
        """
        voice = ctx.config_snapshot.get("voice") or {}
        brief = ctx.config_snapshot.get("brief") or {}
        return [
            voice_for_speaker({"speaker": line.speaker}, voice, brief, default_cast=_DEFAULT_CAST)
            for line in output.lines
        ]

    async def _measure(self, output: PromoScriptOutput, ctx: RunContext) -> ScriptMeasurement:
        """Synthesize EVERY line of this draft and return what it really runs to.

        Never raises: ``measure_script_lines`` absorbs every failure into a
        prior-derived measurement carrying a ``reason``.
        """
        return await measure_script_lines(
            ctx,
            texts=[line.text for line in output.lines],
            voices=self._line_voices(output, ctx),
            prior=self._rate,
            speed=NARRATE_SPEED,
            stage_name=NARRATE_STAGE_NAME,
            client=self._tts,
        )

    def _draw_of(self, output: PromoScriptOutput, measurement: ScriptMeasurement) -> _Draw:
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
        payload: PromoScriptInput,
        ctx: RunContext,
        target_words: int | None = None,
    ) -> PromoScriptOutput | None:
        """ATTEMPT ONE: ask the model again, through this stage's existing LLM call.

        ``None`` when the call failed — a re-ask is a correction, and a paid run
        that already holds a usable draw must not die because the correction did.

        ``target_words`` overrides the budget the correction asks for; the
        trim-floored re-ask (step 2b) passes the SHIPPING draw's own arithmetic.
        """
        self._reask_block = _reask_prompt_block(
            produced_words=count_words([line.text for line in draw.output.lines]),
            estimated_seconds=draw.estimate,
            ordered_seconds=ordered,
            target_words=(
                budget_words(ordered, words_per_minute=self._wpm)
                if target_words is None
                else target_words
            ),
        )
        try:
            # A correction, not a draw: parsed with no floor, so an empty answer
            # costs ONE call and returns as zero elements for the caller to record
            # as a failed re-ask (``LLMScriptStageBase.corrective_draw``).
            with self.corrective_draw():
                return await super().handle(payload, ctx)
        except Exception:
            logger.warning(
                "promo_script duration re-ask failed — keeping the first draw", exc_info=True
            )
            return None
        finally:
            self._reask_block = ""

    async def _reconcile(
        self,
        first: PromoScriptOutput,
        payload: PromoScriptInput,
        ctx: RunContext,
    ) -> PromoScriptOutput:
        """Steer at the ORDER, correct until in band or out of attempts, never pad.

        Three properties, each of which was a measured live miss:

        1. **It aims at the target, not at the band edge.** The decision to correct
           is :func:`steer_verdict` (long side +4%), not the +/-10% band that GRADES
           the master — a draw accepted at +9% has nothing left for the estimate's
           own error, and promo's live spread reached +32.36%. The +/-10% band still
           decides what is REPORTED, so a conformant film is never flagged.
        2. **It iterates.** Up to :data:`MAX_CORRECTIVE_ATTEMPTS` corrections — the
           re-ask, then the trim — and a draw ships only when it is IN BAND or no
           attempt improved it. "Closer" alone is the known bug: a re-ask that swings
           a +34% draft to -18.4% is closer and would end the loop, while the +34%
           draft it is holding is TRIMMABLE at no cost.
        3. **Every candidate is judged at ITS OWN measured seconds** — synthesized at
           the voices that will speak it.
        """
        ordered = _target_seconds(ctx.config_snapshot, ctx.config_snapshot.get("brief") or {})
        cast_entries = list(self._cast_entries)
        if ordered <= 0.0:
            # No order means no band and no verdict; inventing one is the error
            # this refuses to make, and with nothing measured there is no estimate
            # to record. The ONLY early exit: "no lines" used to leave here too,
            # and ``handle`` now refuses that script before this is called.
            return _finalize(first, degradations=cast_entries)

        measurement = await self._measure(first, ctx)
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
                second_measurement = await self._measure(second, ctx)
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
                # An EMPTY re-ask is a re-ask that was DISCARDED, and the record has
                # to say so: `reask_kept=None` documents "no re-ask happened", so
                # leaving it None here would assert a state the field's own contract
                # calls impossible.
                reask_kept = False
                action = "re-ask returned no lines; kept first draw"

        # (2) ATTEMPT TWO — the trim. Deterministic, free (it selects lines whose
        #     seconds are already measured), and the only lever that can rescue a
        #     draw the re-ask overshot.
        floored: TrimPlan | None = None
        trimmed_lines = 0
        # STOP EARLY. An incumbent the steer already calls `ok` is DONE, and the
        # guard is not equivalent to letting the trim run and lose: the trim scans
        # the whole pool, so a long sibling trimmed to a hair nearer the order would
        # displace a perfectly good in-band draw with one that had turns amputated.
        if steer_verdict(best.estimate, ordered) != "ok":
            best, action, floored, trimmed_lines = self._trim_step(best, pool, ordered, ctx, action)

        # (2b) ATTEMPT THREE, LONG SIDE — see ``_floored_reask_step``. Fired ONLY
        #     when the trim could not act at all, so a trim that DID drop something
        #     is never pre-empted by a paid LLM call.
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

        # Reported against the OUTER band — the one that grades the delivered master
        # — so a film this stage steered to +2% raises nothing, and a film the steer
        # merely improved to +6% raises nothing either. The SHIPPED draw's
        # measurement outcome, never the pool's: a discarded candidate took its
        # faults with it.
        degradations = [*cast_entries, *best.rate_entries]
        if narration_verdict(best.estimate, ordered) == "ok":
            return _finalize(best.output, degradations=degradations, estimate=estimate)

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
        payload: PromoScriptInput,
        ctx: RunContext,
        action: str,
        floored: TrimPlan,
    ) -> tuple[_Draw, bool | None, str, TrimPlan | None]:
        """ATTEMPT THREE, LONG SIDE: re-ask with an EXPLICIT WORD TARGET.

        Port of lesson's step (2b) (PR #1016). ``plan_trim`` can decline with
        nothing dropped — the duration quantum of one line is wider than the band,
        so every drop lands UNDER it (``declined_under_band``) — and before this the
        stage then shipped the long draw as-is.

        **Only when the trim genuinely could not act** (the caller gates on zero
        lines dropped AND a floored plan; this gates on the draw still being LONG),
        so a paid call never pre-empts the free lever. The target is this draw's
        OWN arithmetic, ``ordered * draw.wpm / 60``. ``closer_to_order`` still
        decides, so a worse draw is discarded rather than shipped; when the new
        draw IS kept the floor is cleared — it describes a script no longer shipped.

        Returns ``(best, kept, action, floored)``; ``kept is None`` means no usable
        draw came back (call failed), which is not a judged rejection.
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
        measurement = await self._measure(drawn, ctx)
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

        **It trims any LONG candidate in the pool, not only the incumbent.** When a
        re-ask has swung a long draft to a shorter-but-still-wrong one, that shorter
        draw wins on "closer" and a trim of it is then impossible (nothing shortens a
        short script) — so the loop would stop with both drafts out of band while one
        of them was one dropped turn from conformant. Trimming a candidate costs no
        synthesis and no LLM call. The incumbent is tried FIRST, and ties keep it, so
        a single-candidate run is unchanged.

        **A plan that was NOT APPLIED is never reported as a floor.** A rejection —
        the CTA would have gone, or the two-hander's alternation would have broken,
        or the result landed no closer — leaves the script exactly as it was, so the
        plan's ``estimated_seconds_after`` describes a script that was never shipped.
        A plan that dropped NOTHING is not a rejection: there was nothing to apply,
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

        # Nothing was adopted. The report follows the first source's plan, which IS
        # the incumbent whenever the incumbent is long. When the incumbent is SHORT
        # it is not a source at all, so name the draw the note is about.
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

        The PLAN is always returned as the third element, applied or not: whether the
        band was reachable is a property of the plan. The fourth element is the
        REJECTION, in the caller's words — ``None`` when the plan is safe to apply
        (including a plan that drops nothing, which is not a rejection).

        **It plans over TURN BLOCKS, on MEASURED seconds.** :func:`_turn_blocks` is
        what keeps a two-hander a two-hander (whole ``a``/``b`` pairs are dropped
        together); ``line_seconds`` is what keeps the arithmetic honest, because two
        lines of equal word count do not take equal time — and on a two-hander they
        are not even spoken by the same voice, so a word-weighted plan can drop the
        wrong pair and land outside the band it aimed at. The word proxy remains the
        fallback for a measurement that is not one.

        **Two rejections promo needs and the siblings do not:**

        * the CALL-TO-ACTION. :func:`_protected_blocks` pins it — the blocks carrying
          a required phrase VERBATIM, plus the closing block unconditionally,
          because a PARAPHRASED CTA carries no verbatim match and would otherwise be
          eaten first. The result is re-checked afterwards against what was missing
          BEFORE the trim — a straight ``find_missing_verbatim`` on the kept text
          would refuse every trim on a draft whose model never wrote the CTA
          verbatim in the first place, which is a guard that fires on the wrong
          fact; that same subtraction is exactly why the re-check alone cannot
          protect a paraphrase, and why the protection has to be structural.
        * the ALTERNATION, enforced structurally by :func:`_turn_blocks` rather than
          by a check — see its docstring for why a counting check here would have
          been unfalsifiable.
        """
        lines = draw.output.lines
        texts = [line.text for line in lines]
        items = protected_items(ctx)
        brief = ctx.config_snapshot.get("brief") or {}
        blocks = _turn_blocks(lines, dialogue=bool((brief.get("dialogue") or {}).get("enabled")))
        seconds = _trimmable_seconds(draw)
        plan = plan_trim(
            [" ".join(texts[i] for i in block) for block in blocks],
            ordered_seconds=ordered,
            # THIS draw's rate, never ``self._wpm``. ``self._wpm`` holds whatever was
            # measured LAST, which after a re-ask is the rate of a draft that may
            # have been discarded — and ``plan_trim`` falls back to the word proxy
            # the moment ``_trimmable_seconds`` refuses the measurement, at which
            # point the wrong rate silently decides which turns are cut.
            words_per_minute=draw.wpm,
            # The OUTER +/-10% band, deliberately — NOT the 4% the steer aims at. The
            # steer decides WHETHER to correct, keeping headroom for the estimate's
            # own error; the trim decides what a cut is allowed to LAND on, and it
            # declines any drop falling below ``ordered x (1 - tolerance)``.
            # Narrowing this to the steer band makes it refuse cuts it should take —
            # the only available drop lands at, say, -8%, which is conformant, and
            # the script then ships its full overrun instead.
            tolerance=DEFAULT_TOLERANCE,
            # Verbatim carriers UNION the closing block — a PARAPHRASED CTA carries
            # no verbatim match, and the post-trim re-check below cannot see it
            # either (it was already missing). See `_protected_blocks`.
            protected_indices=_protected_blocks(texts, blocks, items, cta=_cta_of(ctx)),
            line_seconds=(
                None
                if seconds is None
                else [math.fsum(seconds[i] for i in block) for block in blocks]
            ),
        )
        if not plan.dropped_indices:
            return None, plan, None
        kept = draw.select([i for index in plan.keep_indices for i in blocks[index]])
        kept_lines = kept.output.lines
        newly_missing = set(
            find_missing_verbatim(" ".join(line.text for line in kept_lines), items)
        ) - set(find_missing_verbatim(" ".join(texts), items))
        if newly_missing:
            logger.warning("%s: trim would drop required verbatim text — skipped", self.name)
            return None, plan, "it would have dropped the call-to-action or a required phrase"
        return kept, plan, None

    def short_circuit(self, ctx: RunContext) -> PromoScriptOutput | None:
        """Verbatim source-script bypass: emit the EXACT supplied lines (no LLM).

        Mirrors lesson's bypass (``lesson/script.py``). When the brief carries
        ``source_script`` (newline-separated), each non-blank line becomes one
        ``PromoScriptLine`` verbatim — used to reuse an already-approved
        narration (e.g. the two-voice Q&A the user wants kept) while a fresh run
        re-derives screens/visuals around it. Dialogue speakers are re-derived
        deterministically from the line role (the #178 lesson: never trust an
        LLM enum to alternate), so the two voices survive the bypass. Returns
        ``None`` (→ normal LLM generation) when no source script is supplied.
        """
        brief = ctx.config_snapshot.get("brief") or {}
        items = items_for_stage(ctx.config_snapshot.get("verbatim_items") or [], self.name)

        # Authored shotlist wins over a flat source_script: it is the more
        # specific per-shot contract. Emit one verbatim line per shot, in order,
        # with the deterministic shot line id (shot-NNN) every downstream stage
        # re-derives — so footage/slide bind to narration by construction.
        shots = normalize_shotlist(brief)
        if shots:
            lines = [
                PromoScriptLine(id=s.line_id, text=s.narration, seconds=s.seconds) for s in shots
            ]
            if (brief.get("dialogue") or {}).get("enabled"):
                lines = _assign_dialogue_speakers(lines)
            assert_verbatim_present(" ".join(ln.text for ln in lines), items, stage=self.name)
            return PromoScriptOutput(lines=lines, total_seconds=sum(ln.seconds for ln in lines))

        segs = normalize_source_script(brief.get("source_script") or "", split="line")
        if not segs:
            return None
        lines = [PromoScriptLine(id=str(uuid7()), text=t, seconds=s) for t, s in segs]
        if (brief.get("dialogue") or {}).get("enabled"):
            lines = _assign_dialogue_speakers(lines)
        assert_verbatim_present(" ".join(ln.text for ln in lines), items, stage=self.name)
        return PromoScriptOutput(lines=lines, total_seconds=sum(ln.seconds for ln in lines))

    def build_prompt(self, ctx: RunContext) -> str:
        analyze = ctx.prior_output("promo_analyze")
        if not analyze:
            raise RuntimeError("promo_analyze output missing")
        cfg = ctx.config_snapshot
        brief = cfg.get("brief") or {}
        voice = cfg.get("voice") or {}
        notes = ctx.annotations_for("promo_script")
        items = items_for_stage(cfg.get("verbatim_items") or [], self.name)
        # Ground the script in the real product: explicit user instructions and
        # (when site_capture ran) the scraped site copy. Both are optional.
        capture = ctx.prior_output("site_capture") or {}
        page_text = str(capture.get("page_text") or "")
        # WHY: an empty page_text alone cannot distinguish "the site has no copy"
        # from "the scrape was blocked/failed". Only the explicit status can, and
        # a hard failure must not be laundered into a confidently ungrounded script.
        status = str(capture.get("page_text_status") or "not_attempted")
        if (
            status in _GROUNDING_FAILURE_STATUSES
            and get_settings().pipeline.promo_require_site_grounding
        ):
            raise StagePreconditionError(_site_grounding_failure_reason(brief, status))
        return (
            self._build_prompt(
                cfg,
                brief,
                voice,
                analyze,
                notes,
                page_text=page_text,
                page_text_status=status,
                wpm=self._wpm,
            )
            + grounding_prompt_block(ctx)
            + verbatim_prompt_block(items)
            + self._reask_block
        )

    @staticmethod
    def _dialogue_block(brief: dict) -> str:
        """Mode instructions: a two-hander dialogue (opt-in) or a single narrator.

        Both branches still require a ``speaker`` per line (the schema does); the
        single-narrator branch just pins it to "a" so promo_narrate — which only
        casts when dialogue is enabled — has a well-formed but inert value.
        """
        dialogue = brief.get("dialogue") or {}
        if not dialogue.get("enabled"):
            return '\n\nThis is a SINGLE narrator. Set "speaker" to "a" on every line.'
        personas = dialogue.get("personas") or {}
        persona_a = str(personas.get("a") or _DEFAULT_PERSONA_A)
        persona_b = str(personas.get("b") or _DEFAULT_PERSONA_B)
        return (
            "\n\nWrite this as a natural, fast-moving DIALOGUE between two people "
            "watching the product on screen together — NOT ad copy, NOT a single "
            "narrator. It should sound like a real, slightly unscripted conversation "
            "between two colleagues. The two speakers:\n"
            f'- speaker "a": {persona_a}\n'
            f'- speaker "b": {persona_b}\n'
            "Rules:\n"
            '- Alternate turns (a, b, a, b, …), starting with "a".\n'
            "- Keep each turn short and spoken — 1-2 sentences, like real talk.\n"
            "- Use contractions and casual connective talk (you'd, it's, that's, "
            'so, okay, right). Let "a" interrupt and react — "wait —", "huh", '
            '"okay so…", "hold on", "no way" — instead of asking tidy full '
            "questions every time.\n"
            "- They are narrating over a live screen-recording walking through the "
            "product, so each of b's turns moves to the NEXT screen in order and "
            "reacts to what just happened there. KEEP the screens in tour order so "
            "the visuals line up.\n"
            "- VARY how each of b's turns opens. Do NOT start turns with the same "
            'phrase. Specifically BAN the repetitive "Look at this / See this / '
            'Watch this / Check out this <thing>" opener. Lead with the reaction or '
            'the point first, then mention the screen naturally ("and that just '
            'shipped — straight from the dashboard", "yeah, and the leaderboard '
            'updates live"), not "Look at this dashboard."\n'
            '- Speaker "b" delivers the closing call-to-action.'
        )

    @classmethod
    def _build_prompt(
        cls,
        cfg: dict,
        brief: dict,
        voice: dict,
        analyze: dict,
        notes: list[dict],
        *,
        page_text: str = "",
        page_text_status: str = "not_attempted",
        wpm: float = DEFAULT_WPM_NATURAL,
    ) -> str:
        # Target length lives on the deliverable (where the FE sets it); accept a
        # brief-level override too. Reading only `brief` defaulted every promo to
        # 30s regardless of the requested duration — the deliverable value was
        # silently ignored.
        target_seconds = _target_seconds(cfg, brief)
        notes_block = ""
        if notes:
            joined = "\n".join(f"- {n['text']}" for n in notes)
            notes_block = "\n\nApply reviewer notes:\n" + joined
        instructions = str(brief.get("instructions") or "").strip()
        instructions_block = (
            f"\n\nFollow these specific instructions for the script:\n{instructions}"
            if instructions
            else ""
        )
        # The status only has to discriminate the EMPTY case: "we read nothing" and
        # "we never looked" are different facts, and only one of them is safe to leave
        # unsaid. Copy that IS present is always groundable, so this branch must stay
        # keyed on page_text alone — every site_capture row persisted before
        # `page_text_status` existed carries text with the "not_attempted" default, and
        # requiring status == "ok" here silently dropped their grounding (caught by
        # test_script_grounds_on_instructions_and_capture_page_text). A gated-off
        # blocked/fetch_failed carries no text and lands in the final branch: today's
        # silent behaviour, but now only as an explicit operator opt-in.
        if page_text:
            site_block = (
                "\n\nGround the script in this copy from the live product site "
                f"(use it for accurate features/value/problems, do not quote verbatim):\n{page_text}"
            )
            # "thin" keeps the grounded block (the fragment is real and worth having)
            # but appends a ceiling on it. Without this, 12 characters of <title> read
            # to the model as the site's product copy and it confabulates the rest —
            # the exact failure the empty-case fix was meant to close, except the
            # empty case almost never fires on a real SPA.
            if page_text_status == "thin":
                site_block += _THIN_SITE_COPY_CAUTION
        elif page_text_status == "empty":
            site_block = _NO_SITE_COPY_BLOCK
        else:
            site_block = ""
        hooks = "\n".join(f"- {h['text']}" for h in analyze["hook_candidates"])
        pains = "\n".join(f"- {p['text']}" for p in analyze["pain_points"])
        # Every optional field is emitted ONLY when present. Interpolating an absent
        # one rendered the literal "None" into the prompt, and for the CTA the model
        # obeyed: `End with this CTA verbatim: "None"` closed a customer's promo on a
        # spoken "None". A missing CTA is a legitimate brief (see `_cta_of`), so the
        # close becomes a natural takeaway rather than a placeholder.
        title = text_or_empty(cfg.get("title"))
        descriptor = text_or_empty(voice.get("descriptor"))
        value_prop = text_or_empty(brief.get("value_proposition"))
        cta = text_or_empty(brief.get("call_to_action"))
        close_line = (
            f'End with this CTA verbatim: "{cta}"\n'
            if cta
            else (
                "End with one natural closing line: a concrete takeaway about the product. "
                "There is no call-to-action to quote — do not invent a URL, a sign-up "
                "instruction, or any placeholder text.\n"
            )
        )
        return (
            f"Write a promo script of approximately {target_seconds:.0f} seconds.\n"
            f"{word_budget_sentence(target_seconds, wpm)}"
            + (f"Title: {title}\n" if title else "")
            + (f"Voice: {descriptor}\n" if descriptor else "")
            + (
                "Convey this value (weave it into the lines, name the offering types where "
                f"they fit; do not quote verbatim): {value_prop}\n"
                if value_prop
                else ""
            )
            + f"Open with one of these hooks (use verbatim where possible):\n{hooks}\n"
            f"Touch on these audience pain points:\n{pains}\n"
            f"{close_line}"
            'Output JSON: {"lines":[{"text":"...","seconds":<num>,"speaker":"a"|"b"}, ...]}.'
            f"{cls._dialogue_block(brief)}"
            f"{instructions_block}"
            f"{site_block}"
            f"{notes_block}"
        )

    def make_element(self, raw: dict, *, element_id: str) -> PromoScriptLine:
        return PromoScriptLine(
            id=element_id,
            text=str(raw["text"]),
            seconds=float(raw["seconds"]),
            speaker=(str(raw.get("speaker") or "a").strip().lower() or "a"),
        )

    def build_output(self, elements: list[StageOutput], ctx: RunContext) -> PromoScriptOutput:
        lines = cast("list[PromoScriptLine]", elements)
        brief = ctx.config_snapshot.get("brief") or {}
        if (brief.get("dialogue") or {}).get("enabled"):
            lines = _assign_dialogue_speakers(lines)
        # `_build_prompt` is a classmethod with no handle on the output, so the
        # degradation note is re-derived here from the same site_capture status it
        # read. A run that produced a script the site could not ground must surface
        # as `degraded` (non-empty notes → degraded, see StageOutputWithNotes).
        capture = ctx.prior_output("site_capture") or {}
        notes: list[str] = []
        degraded_note = _DEGRADING_STATUS_NOTES.get(str(capture.get("page_text_status") or ""))
        if degraded_note is not None:
            notes.append(degraded_note)
        target_seconds = _target_seconds(ctx.config_snapshot, brief)
        # The word budget is RECORDING, not a verdict: it is the ordered duration
        # expressed in words at whatever rate this draw was ASKED at — the cast
        # prior on the first draw, the measured rate on a re-ask (``_reconcile``
        # overwrites ``_wpm`` before ``_reask``, per the #998 rule). What the draw
        # is GRADED against is `_Draw.estimate`, real measured seconds, in
        # ``_reconcile``. The old word-vs-budget degradation raised here is gone:
        # it fired on a proxy that could not tell a draw short in WORDS from one
        # long in SECONDS, and reported a pre-audio miss on the degrading tier.
        return PromoScriptOutput(
            lines=lines,
            total_seconds=sum(line.seconds for line in lines),
            notes=notes,
            word_budget=word_budget(target_seconds, self._wpm),
            actual_words=count_words([line.text for line in lines]),
        )


def _cast_segments(ctx: RunContext) -> list[tuple[str, str | None]]:
    """``(text, voice_id)`` per speaker the promo WILL be narrated in.

    Resolved through the SAME ``voice_for_speaker`` rule ``promo_narrate.voice_for``
    applies, on ``promo_narrate``'s OWN ``_DEFAULT_CAST`` (imported, never copied):
    a budget resolved by a parallel copy of the cast rule is a budget that goes
    wrong the first time somebody edits one of the two. The dialogue gate, the
    ``config.voice.cast`` override and the ``VOICE_CATALOG`` validation all come
    along with it, so a typo'd cast id budgets at ``voice_id`` exactly as it will
    narrate at ``voice_id``.

    **The weighting is one word per speaker — i.e. EQUAL, and that is an assumption,
    stated here rather than hidden.** Unlike the post-draw stages (lesson,
    website_review) this budget is spent BEFORE a line exists, so the real per-voice
    word split is unknowable here. Equal is not a guess pulled from the air: the
    prompt this number feeds mandates it — ``_dialogue_block`` instructs "Alternate
    turns (a, b, a, b, …)" with "each turn short — 1-2 sentences" for BOTH speakers.
    The error is bounded by the spread between the two voices (~12% on the live
    default pair) and lands BETWEEN them; budgeting at ``voice_id`` alone can be
    wrong by that whole spread and is wrong in a single, systematically optimistic
    direction. ``_assign_dialogue_speakers`` guarantees both roles actually appear
    (it falls back to strict a/b parity), so neither speaker is hypothetical.
    """
    brief = ctx.config_snapshot.get("brief") or {}
    voice = ctx.config_snapshot.get("voice") or {}
    return [
        ("word", voice_for_speaker({"speaker": speaker}, voice, brief, default_cast=_DEFAULT_CAST))
        for speaker in ("a", "b")
    ]


def _assign_dialogue_speakers(lines: list[PromoScriptLine]) -> list[PromoScriptLine]:
    """Derive the speaker tag from each line's role: a question is the asker
    ("a"), anything else is the answerer ("b").

    The LLM reliably writes the dialogue CONTENT as Q&A but does NOT reliably
    alternate the per-line ``speaker`` field — in practice it tends to emit every
    line as "a", which would collapse the two voices back into a single narrator
    (the exact "obviously AI-generated" feel we're escaping). Deriving the speaker
    from "does this line ask a question?" guarantees two distinct voices that
    match the brief's "one asks, the other answers". Falls back to strict a/b
    parity if the heuristic can't find both roles (e.g. a script with no question
    marks), so dialogue mode always yields two voices.
    """
    roles = ["a" if line.text.rstrip().endswith("?") else "b" for line in lines]
    if "a" not in roles or "b" not in roles:
        roles = ["a" if i % 2 == 0 else "b" for i in range(len(lines))]
    return [
        line.model_copy(update={"speaker": role}) for line, role in zip(lines, roles, strict=True)
    ]
