"""Script draft — typed LLM call via the shared LLMScriptStageBase.

Output exposes ``lines[*].id`` so annotations can pin to individual lines.
The only thing this stage adds over its siblings is ``min_elements = 1``: an
empty ``lines`` list otherwise flows into storyboard → zero shots →
video-compile 422 (bug-bash B1+B2, 2026-05-09).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, ClassVar, Final, cast
from uuid import uuid7

from src.services.stages.base import (
    SlotUse,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
    assert_verbatim_present,
    find_missing_verbatim,
    items_for_stage,
    normalize_source_script,
    partition_for_stage,
    verbatim_prompt_block,
)
from src.services.stages.generic.grounding import grounding_prompt_block
from src.services.stages.generic.line_fit import (
    LineFit,
    LineTake,
    fit_script,
    verbatim_rewritable,
)
from src.services.stages.primitives.narration import synthesize_and_store_clip
from src.services.stages.primitives.script_stage import LLMScriptStageBase
from src.services.stages.shared.authored_storyboard import storyboard_panels
from src.services.stages.shared.degradation import DURATION_OUT_OF_TOLERANCE
from src.services.stages.shared.narration_budget import (
    DEFAULT_STEER,
    DEFAULT_TOLERANCE,
    TrimPlan,
    budget_words,
    count_words,
    estimate_seconds,
    narration_degradation,
    narration_verdict,
    plan_trim,
    word_band,
)
from src.services.stages.shared.speech_rate import (
    AUTHORED,
    ScriptMeasurement,
    SpeechRate,
    measure_script_lines,
    resolve_effective_wpm,
)
from src.services.stages.shared.time_budget import DEFAULT_WPM_NATURAL

if TYPE_CHECKING:
    from src.clients.tts_client import TTSClient
    from src.services.llm import WMClient
    from src.services.runs.run_context import RunContext

logger = logging.getLogger(__name__)

#: The generic narrate stage. Phase-B sample clips are stored under it so the clips
#: measured here are the clips ``narrate`` will request (deterministic clip ids).
NARRATE_STAGE_NAME = "narrate"

#: Length re-asks one execution may spend after the opening draw. Each is judged at
#: its own measured seconds and kept only when strictly closer to the order, so a
#: round that does not help ends the loop; this bounds only rounds that keep
#: helping. 2, the same short-side allowance website_review needed (run
#: 01a0b2f5-ed32: one re-ask recovered -51% only to -25.9%).
MAX_LENGTH_REASKS = 2

#: Owner decision 2026-10-20: ASK LONG, THEN TRIM. Every ask (the opening draw and
#: each re-ask) is sized at the order x this factor, and an over-long draft is cut
#: back deterministically by whole sentences (:func:`plan_sentence_trim`). Live run
#: ``01a159b1-785e`` (30s order) drew 18.47s, 39.84s and 21.34s and shipped -28.7%:
#: the model cannot hit a word count and re-asks oscillate around it, but a LONG
#: draft can always be made to fit, and a short one never can — nothing downstream
#: pads narration.
ASK_LONG_FACTOR: Final = 1.15

#: Candidate sentence boundary: terminal punctuation (optionally closed by a quote
#: or bracket) followed by whitespace. :func:`split_sentences` then requires the
#: next word to start with an uppercase letter or digit and the word ending in the
#: period not to be an abbreviation or an initial.
_BOUNDARY: Final = re.compile(r"[.!?][\"'\u201d\u2019)\]]?\s+")
_OPENERS: Final = "\"'\u201c\u2018(["
_CLOSERS: Final = "\"'\u201d\u2019)]"

#: Lower-cased tokens (period included) that end in a period without ending a
#: sentence — English and Dutch. A trim that split "Ask Dr. Smith" would cut
#: half a sentence, which is exactly what the sentence-level trim promises not to.
ABBREVIATIONS: Final = frozenset(
    {
        "dr.", "mr.", "mrs.", "ms.", "st.", "prof.", "jr.", "sr.", "mt.", "no.",
        "e.g.", "i.e.", "etc.", "vs.", "approx.", "inc.", "ltd.", "co.",
        "bijv.", "o.a.", "d.w.z.", "m.b.t.", "dhr.", "mevr.", "nr.", "ca.", "t.a.v.",
        "i.p.v.", "m.a.w.", "e.d.", "evt.", "incl.", "excl.",
    }
)  # fmt: skip


def _ends_with_abbreviation(head: str) -> bool:
    """True when the word ending ``head`` (with its period) is not a sentence end."""
    words = head.rstrip(_CLOSERS).split()
    word = words[-1].lstrip(_OPENERS).lower() if words else ""
    if not word.endswith("."):
        return False
    if word in ABBREVIATIONS:
        return True
    # An initial ("J. Smith").
    return len(word) == 2 and word[0].isalpha()


def split_sentences(text: str) -> list[str]:
    """Whole sentences of ``text``, whitespace-normalised, never an empty string.

    A boundary is terminal punctuation + whitespace + a word starting with an
    uppercase letter or digit (after any opening quote/bracket), and never after
    a known abbreviation or an initial. "Ask Dr. Smith, e.g. about 3.5 percent.
    Bel bijv. de heer Jansen." is TWO sentences.
    """
    flat = " ".join(str(text).split())
    out: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(flat):
        rest = flat[match.end() :].lstrip(_OPENERS)
        if not rest or not (rest[0].isupper() or rest[0].isdigit()):
            continue
        head = flat[start : match.end()].rstrip()
        if match.group().startswith(".") and _ends_with_abbreviation(head):
            continue
        out.append(head)
        start = match.end()
    out.append(flat[start:])
    return [part for part in out if part]


def verbatim_protected_units(
    units: list[tuple[int, int, str]], texts: list[str], items: list[dict]
) -> set[tuple[int, int]]:
    """``(line, sentence)`` keys a trim must keep so every verbatim item survives.

    Same matcher as the gate (:func:`find_missing_verbatim`). An item inside one
    sentence pins that sentence; one spanning sentences of a line pins the whole
    line; one spanning lines (or absent) pins everything — nothing is trimmed.
    """
    keep: set[tuple[int, int]] = set()
    for item in items:
        hit = {(li, si) for li, si, t in units if not find_missing_verbatim(t, [item])}
        if not hit:
            lines = {li for li, t in enumerate(texts) if not find_missing_verbatim(t, [item])}
            hit = {(li, si) for li, si, _t in units if li in lines or not lines}
        keep |= hit
    return keep


def plan_sentence_trim(
    texts: list[str],
    *,
    ordered_seconds: float,
    words_per_minute: float,
    verbatim_items: list[dict] | None = None,
) -> tuple[list[str], TrimPlan]:
    """Cut whole sentences from an over-long script until it lands in the band.

    Reuses :func:`narration_budget.plan_trim` (the exhaustive, overcorrection-safe
    drop-set search) with SENTENCES as its units instead of lines:

    * never mid-sentence — a unit is a whole sentence;
    * never empties a line — the FIRST sentence of every line is protected, which
      also keeps the opening line's hook;
    * keeps every sentence carrying a required verbatim phrase
      (:func:`verbatim_protected_units`);
    * keeps the closing call-to-action — the LAST sentence of the last line is
      protected;
    * prefers cutting from the END of the LONGEST lines — units are handed to the
      search ordered so those sit at the highest indices, which is what its
      tie-break (fewest cuts, then the latest indices) drops first;
    * aims at the order — the search runs first against the tighter
      :data:`DEFAULT_STEER` band and falls back to the +/-:data:`DEFAULT_TOLERANCE`
      grading band only when nothing lands in the tight one.

    Returns the rebuilt line texts (same count, same order) and the plan that was
    applied. A script already inside the band, or short, comes back unchanged.
    """
    units: list[tuple[int, int, str]] = []
    for li, text in enumerate(texts):
        units.extend((li, si, sentence) for si, sentence in enumerate(split_sentences(text)))
    line_words = [len(str(t).split()) for t in texts]
    last_line = max((li for li, _si, _t in units), default=-1)
    last_sentence = max((si for li, si, _t in units if li == last_line), default=-1)
    # Least-preferred-to-cut first: short lines before long ones, a line's early
    # sentences before its later ones.
    order = sorted(range(len(units)), key=lambda u: (line_words[units[u][0]], units[u][1]))
    ordered_units = [units[u] for u in order]
    pinned = verbatim_protected_units(units, list(texts), list(verbatim_items or []))
    protected = [
        i
        for i, (li, si, _t) in enumerate(ordered_units)
        if si == 0 or (li == last_line and si == last_sentence) or (li, si) in pinned
    ]
    unit_texts = [t for _li, _si, t in ordered_units]
    plan = plan_trim(
        unit_texts,
        ordered_seconds=ordered_seconds,
        words_per_minute=words_per_minute,
        tolerance=DEFAULT_STEER,
        protected_indices=protected,
    )
    if plan.exceeds_limit or not plan.dropped_indices:
        plan = plan_trim(
            unit_texts,
            ordered_seconds=ordered_seconds,
            words_per_minute=words_per_minute,
            tolerance=DEFAULT_TOLERANCE,
            protected_indices=protected,
        )
    if not plan.dropped_indices:
        return list(texts), plan
    kept = sorted(ordered_units[i][:2] for i in plan.keep_indices)
    by_line: dict[int, list[str]] = {}
    for li, si in kept:
        by_line.setdefault(li, []).append(next(t for a, b, t in units if a == li and b == si))
    return [" ".join(by_line[li]) if li in by_line else texts[li] for li in range(len(texts))], plan


#: The fewest lines a whole-line trim may leave: the hook and the closer.
MIN_LINES_AFTER_TRIM: Final = 2


def plan_line_trim(
    texts: list[str],
    *,
    ordered_seconds: float,
    words_per_minute: float,
    verbatim_items: list[dict] | None = None,
) -> tuple[list[int], TrimPlan]:
    """Tier two of the ask-long trim: drop WHOLE lines once sentences are exhausted.

    Live run ``01a15a56-0306`` (30s order) drew 9 lines of ONE sentence each,
    58.3s. :func:`plan_sentence_trim` protects the first sentence of every line,
    so it had nothing to cut and the film shipped +94.8%. This tier drops whole
    middle lines through the same :func:`narration_budget.plan_trim` search:

    * never the FIRST line (the hook) nor the LAST (the call-to-action);
    * never a line holding a required verbatim phrase — and if a phrase is held
      by no single line (it spans lines, or is absent), nothing is dropped;
    * never below :data:`MIN_LINES_AFTER_TRIM` lines;
    * prefers the LONGEST middle lines — units are ordered so they sit at the
      highest indices, which the search's tie-break drops first;
    * tight :data:`DEFAULT_STEER` band first, then the +/-:data:`DEFAULT_TOLERANCE`
      grading band.

    Returns the ORIGINAL indices of the surviving lines, ascending (so callers
    keep line order and line ids), and the plan applied.
    """
    n = len(texts)
    items = list(verbatim_items or [])
    pinned: set[int] = {0, n - 1} if n else set()
    for item in items:
        holders = {li for li, t in enumerate(texts) if not find_missing_verbatim(t, [item])}
        pinned |= holders or set(range(n))
    line_words = [len(str(t).split()) for t in texts]
    # Least-preferred-to-drop first: pinned lines, then short before long.
    order = sorted(range(n), key=lambda li: (li not in pinned, line_words[li], li))
    unit_texts = [texts[li] for li in order]
    protected = [i for i, li in enumerate(order) if li in pinned]
    if n <= MIN_LINES_AFTER_TRIM:
        protected = list(range(n))
    plan = plan_trim(
        unit_texts,
        ordered_seconds=ordered_seconds,
        words_per_minute=words_per_minute,
        tolerance=DEFAULT_STEER,
        protected_indices=protected,
    )
    if plan.exceeds_limit or not plan.dropped_indices:
        plan = plan_trim(
            unit_texts,
            ordered_seconds=ordered_seconds,
            words_per_minute=words_per_minute,
            tolerance=DEFAULT_TOLERANCE,
            protected_indices=protected,
        )
    kept = sorted(order[i] for i in plan.keep_indices)
    if len(kept) < MIN_LINES_AFTER_TRIM:  # defensive: the pins already guarantee it
        return list(range(n)), plan
    return kept, plan


#: The no-source fact rule. Without ``source_ingest`` content the model has only the
#: brief, and the length rule's old "concrete detail, a worked example" drew invented
#: specifics (a made-up "London 14-day deposit rule"). Nothing true can be added
#: from nowhere, so the rule forbids the falsifiable kinds of claim outright.
NO_SOURCE_FACT_RULE: Final = (
    "\n\nFACTS: no source material was provided, so the brief above is your only "
    "source of facts. Do NOT state numbers, statistics, percentages, prices, laws, "
    "regulations, rules, deadlines, dates, or names of places, organisations, products "
    "or people unless they appear in the brief. Prefer general, non-falsifiable "
    'phrasing ("check the rules where you live", not an invented rule or figure).'
)

#: The with-source fact rule: the sources widen what may be claimed, nothing else does.
SOURCE_FACT_RULE: Final = (
    "\n\nFACTS: only state facts — numbers, rules, deadlines, prices, dates, names — "
    "that are present in the SOURCE MATERIAL or the brief. Never add a specific the "
    "sources do not contain; where they are silent, stay general."
)


def ask_long_sentence(
    target_seconds: float, wpm: float, words: int, *, grounded: bool = False
) -> str:
    """The prompt's length rule: a LONG target with a floor, and why long is safe.

    ``grounded`` (source material attached) keeps the "concrete detail, worked
    example" filler-avoidance advice, scoped to the sources. Without sources that
    advice is an instruction to invent specifics, so it asks for explanation instead.
    """
    substance = (
        "concrete detail and worked examples taken only from the source material, why it matters"
        if grounded
        else "explanation, context, why it matters to this audience"
    )
    low, high = word_band(words)
    return (
        f"Total spoken length: at this narrator's measured pace of {wpm:.0f} words per "
        f"minute, write {words} words in total (all lines combined) — no fewer than "
        f"{low} and no more than {high}. That is deliberately a little longer than "
        f"{target_seconds:.0f} seconds: the script is cut to the exact length "
        "afterwards by removing whole sentences, so write complete sentences that "
        "each stand on their own, open with the hook, and end on the closing line. "
        f"A short script cannot be fixed. Reach the length with substance — {substance} "
        "— never filler or repetition."
    )


class ScriptDraftInput(StageInput):
    pass


class ScriptLine(StageOutput):
    id: str
    text: str
    seconds: float
    #: The TTS speed this line's clip was MEASURED at, set by the per-line fit
    #: (:mod:`line_fit`) when it re-synthesized the line inside the speed clamp.
    #: ``narrate`` requests exactly this speed, so ``deterministic_clip_id`` (voice +
    #: speed + text) hands it the measured clip — measured == shipped. ``None`` =
    #: the backend default, which is what narrate sends when the field is absent.
    speed: float | None = None


class ScriptDraftOutput(StageOutputWithNotes):
    # Declared output contract — enforced once, centrally, in
    # `runs/dag_executor.py::_enforce_produces`. Empty here means the stage
    # reported success having produced nothing, which degrades the run.
    produces: ClassVar[tuple[str, ...]] = ("lines",)

    lines: list[ScriptLine]
    total_seconds: float


class ScriptDraftStage(LLMScriptStageBase[ScriptDraftInput, ScriptDraftOutput]):
    name = "script_draft"
    input_model = ScriptDraftInput
    output_model = ScriptDraftOutput
    # ``tts``: the word budget is written at the run voice's MEASURED rate
    # (``resolve_effective_wpm`` synthesizes a calibration passage, cached per
    # voice) — the same source and the same declaration as lesson/app_review/promo
    # script and demo_video gen. 150 wpm unmeasured is ~60% generous on kokoro.
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("llm"), SlotUse("tts"))
    element_key = "lines"
    min_elements = 1

    #: This run's narration rate, the number the word budget is written from.
    #: Overwritten per execution in ``handle`` with the measured Phase-A prior; the
    #: class default keeps a direct ``build_prompt`` call (tests, tooling) working.
    _wpm: float = DEFAULT_WPM_NATURAL
    #: Extra paragraph for the ONE bounded length re-ask; empty on the first draw.
    _reask_block: str = ""
    #: The Phase-A prior with its provenance — what the full-script measurement
    #: falls back to, and whether the rate was AUTHORED (then nothing is measured).
    _rate: SpeechRate = SpeechRate(wpm=DEFAULT_WPM_NATURAL, measured=False)
    #: The last full-script measurement, keyed by the exact texts it measured, so the
    #: per-line fit reuses it instead of asking again.
    _measured: tuple[tuple[str, ...], ScriptMeasurement] | None = None

    def __init__(self, client: WMClient | None = None, tts: TTSClient | None = None) -> None:
        """``tts`` is the client the per-line fit synthesizes through; ``None``
        (production) builds the default one on first use, as narrate does."""
        super().__init__(client)
        self._tts = tts

    async def handle(self, payload: ScriptDraftInput, ctx: RunContext) -> ScriptDraftOutput:
        """Budget the draw to the ORDER in words, then re-ask while it misses.

        "Write a 30-second script" is a length an LLM cannot honour — it has no
        model of how long its words take to say. Live run ``01a156a4-e906``
        (2026-10-20) drew 12.25s of narration for a 30s order (-59%) and nothing
        noticed. A word count it CAN honour, at the run voice's MEASURED rate.
        The authored / source_script bypasses are the author's words and are
        never re-asked. A failed re-ask keeps the best draw in hand: a paid run
        must not die on its length correction.

        ⛔ EVERY draft is judged at the rate measured on ITS OWN text. Live run
        ``01a15968-48fd`` (2026-10-20, 30s order): the opening ask was sized at the
        158.1 wpm Phase-A prior (79 words); the draft came back at 48 words, which
        Phase B measured at 199.9 wpm (14.4s) — short, re-asked for 100 words. The
        re-ask drew 65 words, and the verdict reused draft ONE's 199.9 wpm (19.5s,
        "closer" — shipped) while narrate spoke those 65 words at 166.8 wpm:
        **23.47s delivered, -21.8%**. Per-content rate varies by ±20% on the same
        voice (short sentences pause more), so a rate measured on different words is
        not a measurement of these. And one re-ask was the only lever, so a re-ask
        that helped but did not REACH the band shipped as if it had.

        ⭐ ASK LONG, THEN TRIM (owner decision, live run ``01a159b1-785e``: draws of
        18.47s / 39.84s / 21.34s for a 30s order, shipped -28.7%). Every ask is
        sized at ``order x ASK_LONG_FACTOR``; re-asks fire only while the LONGEST
        draft is under the band floor; the longest draft is then cut back by whole
        sentences (:meth:`_fit_to_order`). A long draft is always fixable, a short
        one never is.
        """
        if self.short_circuit(ctx) is not None:
            return await super().handle(payload, ctx)
        self._rate = (await resolve_effective_wpm(ctx, texts=(), default=DEFAULT_WPM_NATURAL)).rate
        self._wpm = self._rate.wpm
        self._measured = None
        self._reask_block = ""
        ordered = float(
            (ctx.prior_output("brief_intake") or {}).get("target_duration_seconds") or 0
        )
        best = await super().handle(payload, ctx)
        best_wpm, best_estimate = await self._measure(best, ctx)
        # Re-ask ONLY while the longest draft is under the band floor. Once any
        # draft reaches the floor, asking again is the oscillation that shipped
        # 01a159b1 at -28.7%; an over-long draft is trimmed below instead.
        for _ in range(MAX_LENGTH_REASKS):
            if narration_verdict(best_estimate, ordered) != "short":
                break
            # The re-ask is sized at the rate measured on the draft it corrects —
            # the best evidence of how THIS model's prose for THIS brief is spoken.
            self._wpm = best_wpm
            texts = [line.text for line in best.lines]
            self._reask_block = self._reask_prompt_block(
                words=count_words(texts),
                lines=len([t for t in texts if t.strip()]),
                estimated=best_estimate,
                ordered=ordered,
            )
            try:
                with self.corrective_draw():
                    candidate = await super().handle(payload, ctx)
            except Exception:  # noqa: BLE001 — a failed re-ask degrades to the best draw
                logger.warning("%s: length re-ask failed — keeping the best draw", self.name)
                break
            finally:
                self._reask_block = ""
            if not candidate.lines:
                break
            candidate_wpm, candidate_estimate = await self._measure(candidate, ctx)
            # Keep the LONGEST draft: too long is fixable by the trim, too short is
            # not. A re-ask that did not come back longer ends the loop — another
            # ask from the same draft is the same ask.
            if not candidate_estimate > best_estimate:
                break
            best, best_wpm, best_estimate = candidate, candidate_wpm, candidate_estimate
        self._wpm = best_wpm
        fitted = await self._fit_to_order(
            best, ctx, ordered=ordered, wpm=best_wpm, estimate=best_estimate
        )
        return await self._fit_lines(fitted, payload, ctx, ordered=ordered)

    async def _fit_to_order(
        self,
        draft: ScriptDraftOutput,
        ctx: RunContext,
        *,
        ordered: float,
        wpm: float,
        estimate: float,
    ) -> ScriptDraftOutput:
        """Trim an over-long draft by whole sentences, then whole lines; record a miss.

        In band: untouched. Short after every re-ask: shipped as the longest draft
        with a ``duration_out_of_tolerance`` degradation (nothing pads narration).
        Long: :func:`plan_sentence_trim` at the draft's MEASURED rate and, if that
        still reads long, :func:`plan_line_trim` (whole middle lines); then the
        trimmed lines are re-measured — the measurement synthesizes them under the
        narrate stage's clip ids, so it is the audio narrate will play, paid once —
        and re-planned once at that rate if they still read long.
        """
        verdict = narration_verdict(estimate, ordered)
        if verdict == "short":
            return draft.model_copy(
                update={
                    "degradations": [
                        *draft.degradations,
                        narration_degradation(
                            stage=self.name,
                            ordered_seconds=ordered,
                            estimated_seconds=estimate,
                            action="shipped_longest_draft",
                            detail=f"{MAX_LENGTH_REASKS} length re-ask(s) did not reach the floor.",
                        ),
                    ]
                }
            )
        if verdict != "long":
            return draft
        items, _dropped = partition_for_stage(
            ctx.config_snapshot.get("verbatim_items") or [], self.name
        )
        before = estimate
        for _ in range(2):
            texts = [line.text for line in draft.lines]
            # Tier one: non-first sentences.
            trimmed, plan = plan_sentence_trim(
                texts, ordered_seconds=ordered, words_per_minute=wpm, verbatim_items=items
            )
            # Tier two: still over the band (one-sentence lines leave tier one
            # nothing to cut — live 01a15a56-0306) -> drop whole middle lines.
            kept = list(range(len(trimmed)))
            if (
                narration_verdict(estimate_seconds(trimmed, words_per_minute=wpm), ordered)
                == "long"
            ):
                kept, _line_plan = plan_line_trim(
                    trimmed, ordered_seconds=ordered, words_per_minute=wpm, verbatim_items=items
                )
            if not plan.dropped_indices and len(kept) == len(trimmed):
                break
            survivors = [trimmed[i] for i in kept]
            # The base gate ran on the UNTRIMMED draw, so the trim is re-gated with
            # the same check and fails the same way.
            assert_verbatim_present(" ".join(survivors), items, stage=self.name)
            draft = self._with_texts(draft, trimmed, wpm=wpm)
            draft = self._keep_lines(draft, kept)
            wpm, estimate = await self._measure(draft, ctx)
            self._wpm = wpm
            if narration_verdict(estimate, ordered) != "long":
                break
        logger.info(
            "%s: ask-long trim %.1fs -> %.1fs against an ordered %.1fs",
            self.name,
            before,
            estimate,
            ordered,
        )
        if narration_verdict(estimate, ordered) == "ok":
            return draft
        return draft.model_copy(
            update={
                "degradations": [
                    *draft.degradations,
                    narration_degradation(
                        stage=self.name,
                        ordered_seconds=ordered,
                        estimated_seconds=estimate,
                        action="sentence_trim_floored",
                        detail=(
                            f"Trimmed from {before:.1f}s by whole sentences, then whole "
                            "middle lines; the hook line, the closing line and every "
                            "verbatim line remain."
                        ),
                    ),
                ]
            }
        )

    @staticmethod
    def _with_texts(draft: ScriptDraftOutput, texts: list[str], *, wpm: float) -> ScriptDraftOutput:
        """The draft with new line texts; ids kept, a changed line's seconds re-estimated."""
        lines = [
            line
            if line.text == text
            else line.model_copy(
                update={"text": text, "seconds": estimate_seconds([text], words_per_minute=wpm)}
            )
            for line, text in zip(draft.lines, texts, strict=True)
        ]
        return draft.model_copy(
            update={"lines": lines, "total_seconds": sum(line.seconds for line in lines)}
        )

    @staticmethod
    def _keep_lines(draft: ScriptDraftOutput, kept: list[int]) -> ScriptDraftOutput:
        """The draft with only the ``kept`` lines — order and line ids unchanged.

        narrate / captions / video_render key on line id and run after this stage,
        so survivors keep their ids; a dropped line is simply absent.
        """
        if len(kept) == len(draft.lines):
            return draft
        lines = [draft.lines[i] for i in kept]
        return draft.model_copy(
            update={"lines": lines, "total_seconds": sum(line.seconds for line in lines)}
        )

    async def _measure(self, draft: ScriptDraftOutput, ctx: RunContext) -> tuple[float, float]:
        """``(wpm, seconds)`` for THIS draft: the SUM of every line's real synthesis.

        ⛔ Not a pooled rate projected onto word counts. That path (Phase B,
        ``resolve_effective_wpm(texts=...)``) refuses any draft averaging under
        ``MIN_SAMPLE_AVG_WORDS_PER_LINE`` (10) words a line — most generic drafts —
        and substitutes the calibration prior: run 61ae budgeted at 158 wpm a script
        spoken at 138, estimated 31.1s, delivered 35.63s (+19%). Phase C
        (:func:`measure_script_lines`) keeps each line's real duration, so the
        seconds ARE what narrate will play. Each clip is stored under the narrate
        stage at the speed narrate requests (the line's own ``speed``), so
        ``deterministic_clip_id`` makes it the clip narrate reuses: paid once.

        An AUTHORED rate is never measured against (the operator's rate outranks
        every measurement) — the estimate is words at that rate. Never raises.
        """
        texts = [line.text for line in draft.lines]
        if self._rate.basis == AUTHORED:
            return self._rate.wpm, estimate_seconds(texts, words_per_minute=self._rate.wpm)
        voice = (ctx.config_snapshot.get("voice") or {}).get("voice_id")
        measurement = await measure_script_lines(
            ctx,
            texts=texts,
            voices=[voice] * len(texts),
            prior=self._rate,
            speed=None,
            stage_name=NARRATE_STAGE_NAME,
            client=self._tts,
        )
        self._measured = (tuple(texts), measurement)
        return measurement.wpm, measurement.total_seconds

    def _tts_client(self) -> TTSClient:
        if self._tts is None:
            from src.clients.tts_client import TTSClient

            self._tts = TTSClient()
        return self._tts

    async def _fit_lines(
        self,
        draft: ScriptDraftOutput,
        payload: ScriptDraftInput,
        ctx: RunContext,
        *,
        ordered: float,
    ) -> ScriptDraftOutput:
        """The per-line closed loop (:func:`line_fit.fit_script`) over the MEASURED lines.

        Each line is budgeted its word share of the order and judged at its real
        synthesized seconds: kept (+/-5%), re-synthesized once inside the [0.9, 1.1]
        speed clamp (+/-12%), or rewritten alone by the LLM (<= 2 rewrites, each ONE
        model call — :meth:`redraw_elements` does not retry) and then clamped.
        The last two lines absorb the residual. Every take is synthesized under the
        narrate stage at the speed persisted on the line, so narrate replays the
        measured clip. Lines carrying a required verbatim phrase are never
        rewritten. Skipped without an order, on an authored rate, or when the
        measurement did not land on every line (an extrapolated second is not one a
        line can be fitted to). Wrapped whole: a paid run never fails on this.
        """
        if ordered <= 0.0 or self._rate.basis == AUTHORED or not draft.lines:
            return draft
        try:
            return await self._fit_lines_inner(draft, ctx, ordered=ordered)
        except Exception:
            logger.warning(
                "%s: per-line fit failed — shipping the trimmed draft", self.name, exc_info=True
            )
            return draft

    async def _fit_lines_inner(
        self, draft: ScriptDraftOutput, ctx: RunContext, *, ordered: float
    ) -> ScriptDraftOutput:
        texts = [line.text for line in draft.lines]
        if self._measured is not None and self._measured[0] == tuple(texts):
            measurement = self._measured[1]
        else:
            await self._measure(draft, ctx)
            measurement = self._measured[1] if self._measured is not None else None
        if measurement is None or not measurement.complete:
            logger.info(
                "%s: per-line fit skipped — the measurement did not land on every line",
                self.name,
            )
            return draft
        items, _dropped = partition_for_stage(
            ctx.config_snapshot.get("verbatim_items") or [], self.name
        )
        rewritable = verbatim_rewritable(texts, items)
        voice = (ctx.config_snapshot.get("voice") or {}).get("voice_id")
        run_id = str(ctx.run_id)

        async def _synth(text: str, speed: float | None) -> LineTake:
            clip = await synthesize_and_store_clip(
                self._tts_client(),
                text=text,
                voice=voice,
                run_id=run_id,
                stage_name=NARRATE_STAGE_NAME,
                speed=speed,
            )
            return LineTake(
                text=text,
                seconds=float(clip.seconds),
                speed=speed,
                pause_seconds=clip.service_pause_seconds,
            )

        async def _rewrite(
            index: int, text: str, measured: float, budget: float, want: int
        ) -> str | None:
            message = self._line_rewrite_prompt(
                ctx, texts, index=index, text=text, measured=measured, budget=budget, want=want
            )
            with self.corrective_draw():
                elements = await self.redraw_elements(ctx, message)
            new = " ".join(str(getattr(e, "text", "")).strip() for e in elements).strip()
            return new or None

        takes = [
            LineTake(text=t, seconds=float(s))
            for t, s in zip(texts, measurement.per_line_seconds, strict=True)
        ]
        fits = await fit_script(
            takes, ordered, synth=_synth, rewrite=_rewrite, rewritable=rewritable
        )
        return self._with_fits(
            draft, fits, ordered=ordered, items=items, measured_seconds=[t.seconds for t in takes]
        )

    def _with_fits(
        self,
        draft: ScriptDraftOutput,
        fits: list[LineFit],
        *,
        ordered: float,
        items: list[dict],
        measured_seconds: list[float],
    ) -> ScriptDraftOutput:
        """The draft carrying each fit's text, MEASURED seconds and speed; ids kept."""
        lines = [
            line.model_copy(
                update={"text": fit.take.text, "seconds": fit.take.seconds, "speed": fit.take.speed}
            )
            for line, fit in zip(draft.lines, fits, strict=True)
        ]
        assert_verbatim_present(" ".join(line.text for line in lines), items, stage=self.name)
        total = sum(line.seconds for line in lines)
        before = sum(measured_seconds)
        changed = [f for f in fits if f.rewrites or f.resynthesized]
        logger.info(
            "%s: per-line fit %d line(s) changed (%d rewrite(s), %d speed re-synth) — "
            "%.2fs -> %.2fs measured against an ordered %.1fs",
            self.name,
            len(changed),
            sum(f.rewrites for f in fits),
            sum(1 for f in fits if f.resynthesized),
            before,
            total,
            ordered,
        )
        degradations = list(draft.degradations)
        if narration_verdict(total, ordered) == "ok":
            # Measured in band now: an earlier ESTIMATE's out-of-band verdict is stale.
            degradations = [d for d in degradations if d.get("code") != DURATION_OUT_OF_TOLERANCE]
        elif not any(d.get("code") == DURATION_OUT_OF_TOLERANCE for d in degradations):
            degradations.append(
                narration_degradation(
                    stage=self.name,
                    ordered_seconds=ordered,
                    estimated_seconds=total,
                    action="line_fit_floored",
                    detail="Every line was fitted (speed clamp 0.9-1.1, <=2 rewrites a line).",
                )
            )
        return draft.model_copy(
            update={"lines": lines, "total_seconds": total, "degradations": degradations}
        )

    def _line_rewrite_prompt(
        self,
        ctx: RunContext,
        texts: list[str],
        *,
        index: int,
        text: str,
        measured: float,
        budget: float,
        want: int,
    ) -> str:
        brief = ctx.prior_output("brief_intake") or {}
        words = len(text.split())
        delta = want - words
        verb = f"ADD {delta}" if delta > 0 else f"CUT {-delta}"
        # An "ADD N words" ask with no fact rule draws invented specifics (the round-9
        # "London 14-day deposit rule"), so additions are explanation, never new facts,
        # and the same fact rule as the main prompt applies — with the source block
        # when there is one, since the sources are the only licence to be specific.
        how = (
            " Add only explanation or context for what the line already says — no new "
            "facts, figures, rules or names."
            if delta > 0
            else ""
        )
        grounding = grounding_prompt_block(ctx)
        fact_rule = grounding + (SOURCE_FACT_RULE if grounding else NO_SOURCE_FACT_RULE)
        direction = "TOO LONG" if measured > budget else "TOO SHORT"
        role = (
            " It is the OPENING line: keep it a hook."
            if index == 0
            else " It is the CLOSING line: keep it the close."
            if index == len(texts) - 1
            else ""
        )
        script = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        notes = ctx.annotations_for("script_draft")
        notes_block = (
            "\n\nReviewer notes that still apply:\n" + "\n".join(f"- {n['text']}" for n in notes)
            if notes
            else ""
        )
        return (
            f"Rewrite ONE line of the narration script for the video "
            f'"{brief.get("title", "")}" (angle: {brief.get("angle", "")}; audience: '
            f"{brief.get('audience') or 'general'}; voice: {brief.get('voice_descriptor', '')}).\n\n"
            f"The whole script, for context only — do NOT rewrite the other lines:\n{script}\n\n"
            f'Line {index + 1} reads: "{text}"\n'
            f"Spoken by this narrator it MEASURED {measured:.1f} seconds; its slot is "
            f"{budget:.1f} seconds, so it is {direction} by {abs(measured - budget):.1f}s. "
            f"It has {words} words: rewrite it to exactly {want} words ({verb} words).{how}{role} "
            "Keep its meaning, its facts and its place in the flow; complete sentences; "
            "no filler, no repetition, no new claims. Output JSON with exactly ONE line: "
            '{"lines":[{"text":"...","seconds":<num>}]}.' + fact_rule + notes_block
        )

    def _reask_prompt_block(
        self, *, words: int, lines: int, estimated: float, ordered: float
    ) -> str:
        target = budget_words(ordered * ASK_LONG_FACTOR, words_per_minute=self._wpm)
        direction = "TOO LONG" if estimated > ordered else "TOO SHORT"
        # Live, the model drew 48 and then 65 words when asked for 79 and 100 (~62%).
        # A line count at its own line length is a second, coarser handle on the same
        # target. UNPROVEN live that it lands closer — it only restates the target.
        per_line = words / lines if lines > 0 and words > 0 else 0.0
        line_hint = (
            f" Your {lines} line(s) averaged {per_line:.0f} words; at that length the "
            f"order needs about {max(1, round(target / per_line))} line(s)."
            if per_line > 0
            else ""
        )
        return (
            f"\n\nREVISION REQUIRED. Your previous draft was {direction}. It contained "
            f"{words} spoken words, which at this narrator's measured pace of "
            f"{self._wpm:.0f} words per minute takes about {estimated:.0f} seconds. The "
            f"order is {ordered:.0f} seconds; write {target} words.{line_hint} Rewrite "
            f"the WHOLE script so the combined spoken text is at least {target} words — "
            "slightly long is fine (it is trimmed by whole sentences afterwards), short "
            "is not. Expand the CONTENT of the lines; do not repeat a line to fill time."
        )

    def short_circuit(self, ctx: RunContext) -> ScriptDraftOutput | None:
        brief = ctx.prior_output("brief_intake") or {}
        if not brief:
            raise RuntimeError("brief_intake output missing — DAG ordering broken")
        items = items_for_stage(ctx.config_snapshot.get("verbatim_items") or [], self.name)

        # WS-C: an authored/approved storyboard drives the spine directly (no LLM).
        # The storyboard lives on config_snapshot["brief"] (folded in by run_setup) —
        # NOT on the brief_intake stage output above, which does not carry it. Panels
        # carry the clip-<n> id + narration + duration; project onto ScriptLine so
        # storyboard_draft/narrate join on the same ids as the LLM/source_script path.
        panels = storyboard_panels(ctx.config_snapshot.get("brief") or {})
        if panels is not None:
            sb_lines = [
                ScriptLine(id=p.id or str(uuid7()), text=p.text, seconds=p.seconds) for p in panels
            ]
            assert_verbatim_present(
                " ".join(line.text for line in sb_lines), items, stage=self.name
            )
            return ScriptDraftOutput(
                lines=sb_lines, total_seconds=sum(line.seconds for line in sb_lines)
            )

        # Legacy: free-text source_script → line-split (unchanged).
        segs = normalize_source_script(brief.get("source_script") or "", split="line")
        if not segs:
            return None
        lines = [ScriptLine(id=str(uuid7()), text=t, seconds=s) for t, s in segs]
        assert_verbatim_present(" ".join(line.text for line in lines), items, stage=self.name)
        return ScriptDraftOutput(lines=lines, total_seconds=sum(s for _, s in segs))

    def build_prompt(self, ctx: RunContext) -> str:
        brief = ctx.prior_output("brief_intake") or {}
        notes = ctx.annotations_for("script_draft")
        items = items_for_stage(ctx.config_snapshot.get("verbatim_items") or [], self.name)
        grounding = grounding_prompt_block(ctx)
        return (
            self._build_prompt(brief, notes, wpm=self._wpm, grounded=bool(grounding))
            + grounding
            + (SOURCE_FACT_RULE if grounding else NO_SOURCE_FACT_RULE)
            + verbatim_prompt_block(items)
            + self._reask_block
        )

    @staticmethod
    def _build_prompt(
        brief: dict,
        notes: list[dict],
        *,
        wpm: float = DEFAULT_WPM_NATURAL,
        grounded: bool = False,
    ) -> str:
        notes_block = ""
        if notes:
            joined = "\n".join(f"- {n['text']}" for n in notes)
            notes_block = "\n\nApply these reviewer notes from the prior run:\n" + joined
        target = float(brief.get("target_duration_seconds", 60) or 0)
        # The countable half of "an N-second script": a word count the model can
        # actually hit, at this narrator's measured pace (not the 150 constant) —
        # sized LONG (ASK_LONG_FACTOR), because the draft is trimmed afterwards.
        words = budget_words(target * ASK_LONG_FACTOR, words_per_minute=wpm)
        budget = (
            (" " + ask_long_sentence(target, wpm, words, grounded=grounded)) if words > 0 else ""
        )
        return (
            f"Write a script for a {brief.get('target_duration_seconds', 60)}-second "
            f"video. Title: {brief['title']}. Angle: {brief['angle']}. "
            f"Audience: {brief.get('audience') or 'general'} — write for them: their "
            "vocabulary, what they already know, and what they care about. "
            f"Voice: {brief['voice_descriptor']}.{budget} Output JSON: "
            f'{{"lines":[{{"text":"...","seconds":<num>}}, ...]}}.{notes_block}'
        )

    def make_element(self, raw: dict, *, element_id: str) -> ScriptLine:
        return ScriptLine(id=element_id, text=str(raw["text"]), seconds=float(raw["seconds"]))

    def build_output(self, elements: list[StageOutput], ctx: RunContext) -> ScriptDraftOutput:
        lines = cast("list[ScriptLine]", elements)
        return ScriptDraftOutput(lines=lines, total_seconds=sum(line.seconds for line in lines))
