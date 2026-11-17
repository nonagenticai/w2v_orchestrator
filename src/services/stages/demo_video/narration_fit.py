"""Fit each demo_video / showcase narration line to the footage it backs — MEASURED.

THE DEFECT. ``reconcile_timeline`` judged every beat's narration length as
``word_count / wpm * 60`` at ONE pooled rate, and its only lengthening lever was a
word-count re-ask sized at that rate. Real per-line rates spread far wider than the pool:
showcase run ``01a15b57-9403`` (2026-10-20, 30s order, three ~10s clips) measured
165.6 wpm on clip-1 and 181.4 on clip-4 against a 164.2 pooled budget; run
``01a15bda-2d21`` read clip-4 at 197.9. So a re-ask that "hit" its word count still
landed seconds short, nothing ever synthesized the text before narrate, and narrate can
only move SPEED — clamped at the 0.97 drag floor — so the beats shipped
``narration_rate_floor_clamped`` + ``timeline_underspoken`` with the picture holding the
rest.

THE FIX is the generic per-line loop (:mod:`src.services.stages.generic.line_fit`,
#1134), reused rather than forked: each spoken line's budget is the seconds its beat's
picture plays (the solved allotment — the footage window, after any authored lead /
out-point and any footage tail trim), its real clip is synthesized and MEASURED, and it
is kept (+/-5%), re-synthesized inside the [0.9, 1.1] speed clamp under kokoro's
CALIBRATED speed response (plus one corrective re-synthesis if that take misses), or
rewritten alone (<= 2 rewrites, told its measured length, budget and the exact
words to add or cut; a reply off its word count by more than :data:`FIT_WORD_TOLERANCE`
is re-asked once with its real count). Every take is stored under the narrate stage
at the speed persisted on the line as ``vo_rate``, so narrate's
``deterministic_clip_id(text, voice, speed)`` resolves to the measured clip: measured ==
shipped, paid once.

Pure helpers only; the stage owns the TTS client and the WM call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from src.services.stages.generic.line_fit import ACCEPT_BAND, LineFit, verbatim_rewritable
from src.services.stages.primitives.dialogue import voice_for_speaker
from src.services.stages.shared.beat_grounding import beat_source

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.services.stages.shared.time_budget import Allotment, TimeDemand

#: The speed every line's FIRST take is synthesized at: natural pace, stated explicitly
#: (never ``None``) because it is persisted as the line's ``vo_rate`` and narrate hashes
#: the requested speed into the clip id — ``None`` and ``1.0`` name different clips.
FIRST_TAKE_SPEED = 1.0


@dataclass(frozen=True, slots=True)
class FitTarget:
    """One spoken line the measured fit will synthesize, and what it is fitted to."""

    index: int
    line_id: str
    text: str
    #: The seconds this line's picture plays — the solved allotment.
    budget: float
    #: The solver-granted freeze-pad past the clip (footage only, else 0.0).
    pad: float
    voice: str | None
    rewritable: bool
    #: What the beat's recording performs (scenario + gherkin) — the only material a
    #: rewrite may add; ``None`` for a beat with no recorded script.
    source: str | None


def _finite_positive(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    out = float(value)
    return out if math.isfinite(out) and out > 0.0 else None


def fit_targets(
    lines: Sequence[Mapping[str, object]],
    allotments: Sequence[Allotment],
    *,
    voice: Mapping[str, object],
    brief: Mapping[str, object],
    verbatim_items: Sequence[dict],
) -> list[FitTarget]:
    """The lines the measured fit covers, in spine order.

    A line is fitted when it has words and a positive allotment, and carries no
    author ``rate`` (narrate plays an authored rate verbatim, so a fitted speed would
    never be requested — the measured clip could not be the shipped one). It is
    REWRITABLE unless it is author-``locked``, carries a grounded claim
    (``claim_status`` — the same exclusions as the underspoken re-ask), or holds a
    required verbatim phrase (:func:`verbatim_rewritable`).
    """
    by_alloc = {str(a.id): a for a in allotments}
    texts = [str(line.get("text") or "") for line in lines]
    verbatim_ok = verbatim_rewritable(texts, list(verbatim_items))
    out: list[FitTarget] = []
    for i, line in enumerate(lines):
        text = texts[i].strip()
        allotment = by_alloc.get(str(line.get("id")))
        if not text or allotment is None or line.get("rate") is not None:
            continue
        budget = _finite_positive(allotment.seconds)
        if budget is None:
            continue
        out.append(
            FitTarget(
                index=i,
                line_id=str(line["id"]),
                text=text,
                budget=budget,
                pad=max(float(allotment.freeze_pad_seconds or 0.0), 0.0),
                voice=voice_for_speaker(dict(line), dict(voice), dict(brief)),
                rewritable=(
                    verbatim_ok[i] and not line.get("locked") and not line.get("claim_status")
                ),
                source=beat_source(line),
            )
        )
    return out


def measured_demands(
    demands: Sequence[TimeDemand],
    measured: Mapping[str, float],
    wpm_natural: float,
) -> list[TimeDemand]:
    """``demands`` with each fitted beat's word count replaced by its MEASURED length.

    Every estimate in the solver (``narration_seconds``, ``vo_rate_for``'s
    underspoken / unsayable flags, the projection the structural-relief spawn reads)
    is ``word_count / wpm_natural * 60``. Re-expressing each fitted line's measured
    seconds as the word count that estimate would need makes all of it read the
    measurement instead of the pool — without a second code path. ``min/ideal/max``
    are untouched: a footage beat stays pinned at its clip, and an elastic beat keeps
    the bounds its allotment (= the fit's budget) was solved inside.
    """
    if wpm_natural <= 0.0:
        return list(demands)
    return [
        replace(d, word_count=max(1, round(measured[d.id] * wpm_natural / 60.0)))
        if d.id in measured and d.word_count > 0
        else d
        for d in demands
    ]


def in_band(fit: LineFit) -> bool:
    return abs(fit.miss) <= ACCEPT_BAND


def hold_for(fit: LineFit, pad: float) -> float:
    """The ``hold_seconds`` that makes narrate KEEP the measured clip.

    narrate fits to its measurement clamped into ``[window - hold, window]``
    (``window = seconds + freeze_pad``) and keeps anything within 5% of that target.
    In band with no pad, ``hold = 0`` already keeps it (|miss| <= 5% of ``seconds``).
    Otherwise the hold is the window the measured clip leaves unspoken, so the clamp's
    lower edge IS the measurement: no re-synthesis, no drag to a floor. A footage beat's
    picture is the clip whatever the hold says (``shot_assembly`` caps the held end at
    the allotment), so this never lengthens footage; on a non-footage beat the hold is
    the residual the picture holds, exactly as before.
    """
    if in_band(fit) and pad <= 0.0:
        return 0.0
    return max(0.0, fit.budget + pad - fit.take.seconds)


def fitted_line(
    line: Mapping[str, object], fit: LineFit, *, pad: float, footage: bool
) -> dict[str, object]:
    """``line`` carrying the fit's text, its MEASURED speed and hold, and the marker
    (``fitted_seconds``) + leading-pause slack narrate uses to replay that exact clip."""
    speed = fit.take.speed if fit.take.speed is not None else FIRST_TAKE_SPEED
    hold = hold_for(fit, pad)
    slack = max(0.0, fit.budget - fit.take.seconds - hold) if footage else 0.0
    return {
        **line,
        "text": fit.take.text,
        "vo_rate": float(speed),
        "hold_seconds": hold,
        "fitted_seconds": fit.take.seconds,
        "fitted_slack_seconds": slack,
    }


def measured_underspoken_note(fits: Sequence[tuple[str, LineFit]]) -> str | None:
    """The measured twin of the estimate-based UNDERSPOKEN note — or ``None``.

    Names every line whose FITTED clip still speaks short of its window by more than
    :data:`ACCEPT_BAND` — what the re-ask's word arithmetic used to only guess at.
    """
    short = [(lid, f) for lid, f in fits if f.miss < -ACCEPT_BAND]
    if not short:
        return None
    named = ", ".join(
        f"{lid} speaks {f.take.seconds:.2f}s of its {f.budget:.2f}s window ({f.miss:+.1%})"
        for lid, f in short
    )
    gap = sum(f.budget - f.take.seconds for _, f in short)
    return (
        f"reconcile_timeline: UNDERSPOKEN narration (MEASURED) — {named}, after the "
        f"per-line fit (speed clamp 0.9-1.1, <=2 rewrites a line). The picture holds the "
        f"remaining {gap:.2f}s; the voice is not dragged to fill it."
    )


#: A fit rewrite whose reply is more than this many words off its ask is re-asked ONCE.
#:
#: The rewrite loop's word target is right; the model's word COUNT is not. Recorded
#: showcase runs 01a15d7c-ca2d / 01a15e2b-2a88 / 01a15e77-e3a4 (2026-10-21), 12
#: rewrites: the model returned the asked count +6.7 words on average (+22.7% of the
#: ask; asked 24 -> got 31 twice, 34 -> 46/43, 26 -> 36/34, 33 -> 44) and the takes
#: missed their windows by +22% to +35%. Had each reply carried exactly the asked
#: count, at that reply's own measured pace, every one of the 12 lands within -7.1% to
#: +3.9% of its window (mean |miss| 3.8%) — inside the speed clamp's reach. The
#: pace the target is computed from was off by only -3.0% on average (max 7.9%); no
#: fixed lead/tail silence (<= 0.06s in every sidecar) skews it. So the miss is the
#: count, and the count is what is checked.
FIT_WORD_TOLERANCE = 1

#: System prompt for the measured fit's rewrites, BOTH directions. The fit used the
#: underspoken re-ask's prompts, which ask for a MINIMUM ("at or just above") or a
#: MAXIMUM — not the exact count :func:`fit_message` states. All 12 recorded fit
#: rewrites were lengthens under the MINIMUM prompt; 11 came back over their ask.
FIT_SYSTEM = (
    "You rewrite ONE narration beat to an EXACT number of words, because it was "
    "measured too long or too short for the seconds of screen recording it is spoken "
    "over. The word count you are given is the measurement, not a suggestion: a reply "
    "with more words runs over the picture and one with fewer leaves it silent, so "
    "count your words before you answer. Return ONLY the rewritten narration — no id, "
    "no quotes, no preamble, no explanation. Keep the original's meaning and its "
    "subject. ⛔ NEVER introduce a claim, statistic, number, feature, or "
    "superlative that neither the original nor the recorded steps you are given "
    "contain. Do NOT pad with filler or repetition. ⛔ NEVER write a URL, web address, "
    "domain or file path — refer to a site or page by its name instead."
)


def _grounding(source: str | None) -> str:
    return (
        f"\n\nWhat the screen recording under this beat actually shows (its scripted "
        f"steps):\n{source}\n\nUse ONLY what the beat already says and what these steps "
        f"show. Never mention a feature, action, item, outcome or claim that is not in them."
        if source
        else ""
    )


def _todo(delta: int, uncovered: Sequence[str]) -> str:
    """The not-yet-narrated steps, offered as MATERIAL for ``delta`` words — not a list
    to narrate in full. "Take any added words from them" read as "narrate them all":
    on 01a15e77-e3a4 clip-4 an ask of +5 words came back +16, every listed step said."""
    if delta <= 0 or not uncovered:
        return ""
    return (
        "\n\nThe recording ALSO performs these steps, which the beat does not narrate "
        f"yet. Take the {delta} added words from them, in order, in plain spoken language "
        f"— narrate only as many of them as {delta} words allow and leave the rest out; "
        "leaving steps out is correct, going over the word count is not:\n"
        + "\n".join(f"- {step}" for step in uncovered)
    )


def _verb(delta: int) -> str:
    return f"ADD {delta}" if delta > 0 else f"CUT {-delta}"


def fit_message(
    text: str,
    *,
    measured: float,
    budget: float,
    want: int,
    source: str | None,
    uncovered: Sequence[str] = (),
) -> str:
    """The one-line rewrite ask: measured length, the window, and the EXACT word delta.

    A word target alone is what the old re-ask asked; what was missing is the
    MEASUREMENT — the model is told how long its words actually took this narrator, so
    "add 4 words" is a correction of a known miss rather than a guess at a pooled rate.
    Additions come only from the recorded steps (``source``/``uncovered``), the same
    grounding the underspoken re-ask enforces, and are refused downstream otherwise.
    """
    words = len(text.split())
    delta = want - words
    direction = "TOO LONG" if measured > budget else "TOO SHORT"
    how = (
        " Add only explanation of what the viewer sees happen — no new facts."
        if delta > 0
        else " Say the same thing in fewer words."
    )
    return (
        f"This narration beat is spoken over a screen recording that plays for "
        f"{budget:.1f} seconds. Spoken by this narrator it MEASURED {measured:.1f} seconds, "
        f"so it is {direction} by {abs(measured - budget):.1f}s. It has {words} words: "
        f"rewrite it to exactly {want} words ({_verb(delta)} words) — not more than "
        f"{want + FIT_WORD_TOLERANCE}, not fewer than {max(1, want - FIT_WORD_TOLERANCE)}."
        f"{how} Keep its meaning and its place in the flow; complete sentences; no filler, "
        f"no repetition. Do not write any URL, web address or file path. Return only the "
        f"full rewritten text.{_grounding(source)}{_todo(delta, uncovered)}"
        f"\n\nBeat ({words} words): {text}"
    )


def word_miss(text: str, want: int) -> int:
    """Signed words a reply carries beyond its ask, counted as :func:`spoken_words`
    counts them (whitespace tokens — the unit tts_backend's ``word_count`` reports)."""
    return len(text.split()) - want


def needs_recount(text: str, want: int) -> bool:
    return abs(word_miss(text, want)) > FIT_WORD_TOLERANCE


def recount_message(
    reply: str,
    *,
    want: int,
    budget: float,
    source: str | None,
    uncovered: Sequence[str] = (),
) -> str:
    """The ONE re-ask for a reply that missed its word count by more than
    :data:`FIT_WORD_TOLERANCE`: the model's own reply, its real count, and the exact
    words to cut or add from IT (not from the original)."""
    has = len(reply.split())
    delta = want - has
    how = (
        " Remove words or clauses; keep the steps in the order they happen."
        if delta < 0
        else " Add only explanation of what the viewer sees happen — no new facts."
    )
    return (
        f"Your rewrite of this narration beat has {has} words, but the screen recording "
        f"it is spoken over plays for {budget:.1f} seconds, which is exactly {want} words "
        f"for this narrator, so rewrite it to exactly {want} words ({_verb(delta)} words) — "
        f"not more than {want + FIT_WORD_TOLERANCE}, not fewer than "
        f"{max(1, want - FIT_WORD_TOLERANCE)}.{how} Complete sentences; no filler. Do not "
        f"write any URL, web address or file path. Return only the full rewritten text."
        f"{_grounding(source)}{_todo(delta, uncovered)}\n\nBeat ({has} words): {reply}"
    )


def fit_summary(fits: Sequence[tuple[str, LineFit]]) -> str:
    """One log line: per line, window, shipped take and how — and, for a line that was
    worked, every measured candidate (words, seconds, predicted speed and miss), so a
    measurement in the log can be traced to the clip it belongs to."""
    parts = []
    for lid, f in fits:
        part = (
            f"{lid} {f.budget:.2f}s: {f.take.seconds:.2f}s ({f.miss:+.1%}, "
            f"{f.initial_band}, {f.rewrites} rewrite(s){', re-synth' if f.resynthesized else ''}"
            f"{', corrected' if f.corrected else ''}"
            f", speed {f.take.speed}, {len(f.take.text.split())}w)"
        )
        if f.rewrites:
            part += " [" + "; ".join(a.describe() for a in f.attempts) + "]"
        parts.append(part)
    return "; ".join(parts)
