"""Deterministic craft rubric for the ORCH-1.2 ``script_doctor`` loop.

Pure, I/O-free story-craft scoring over a video script's beats. Each metric is
a small free function returning a :class:`MetricResult` (higher score == better,
0-100), so the WM-LLM doctor loop can *gate* on measured targets instead of a
vibe score. Reuses ``_text.py``'s deterministic helpers directly (per
``02-narrative-script.md`` §T2 — the legacy ``evaluate()`` protocol has since
been retired entirely) so readability / TTR / sentence-variance / preamble
become gates rather than passive insights.

The single public entry point is :func:`score_script`, which returns a
``ScriptReport`` dict. Everything here is deterministic given its inputs — no
randomness, no LLM, no DB, no network.

Convention note (composite inversion): ``abstraction_ratio`` and
``cross_beat_redundancy`` are "lower raw value is better". Their raw ratio is
inverted into a 0-100 *goodness* at metric-construction time (so ``.score``
follows the system-wide "100 == good" convention and the flag rule ``score < 70``
works uniformly). :func:`composite` therefore takes a plain weighted mean of the
already-goodness scores — it must NOT re-invert.
"""

from __future__ import annotations

import math
import re
import string
from typing import Any

from src.services.evaluation.models import MetricResult
from src.services.evaluation.rubrics._kernel import metric
from src.services.evaluation.rubrics._text import (
    _lexical_diversity,
    _preamble_contamination,
    _readability_score,
    _sentence_length_variance,
    _split_sentences,
)

# --------------------------------------------------------------------------- #
# Lexicons + constants (all deterministic, curated)
# --------------------------------------------------------------------------- #

CLICHE_LEXICON = frozenset(
    {
        "transformation",
        "synergy",
        "empower",
        "leverage",
        "seamless",
        "robust",
        "innovative",
        "unlock",
        "elevate",
        "holistic",
        "cutting-edge",
        "revolutionary",
        "game-changing",
        "best-in-class",
        "next-generation",
        "world-class",
        "state-of-the-art",
    }
)

# Strong pivot signals — the "turn" and hook-contrast tests.
_TURN_MARKERS = frozenset(
    {"but", "until", "then", "instead", "however", "yet", "suddenly", "although"}
)
_HOOK_CONTRAST = frozenset(
    {"but", "yet", "however", "until", "instead", "unlike", "no", "not", "never", "without"}
)
_QUESTION_WORDS = frozenset(
    {"what", "why", "how", "who", "when", "where", "imagine", "did", "can", "will"}
)

# Imperative openers for the CTA "single imperative" gate.
_IMPERATIVE_VERBS = frozenset(
    {
        "start",
        "get",
        "try",
        "join",
        "see",
        "discover",
        "learn",
        "build",
        "create",
        "download",
        "sign",
        "book",
        "visit",
        "explore",
        "make",
        "become",
        "claim",
        "grab",
        "unlock",
        "watch",
        "read",
        "call",
        "buy",
        "subscribe",
        "register",
    }
)

# Small concrete-noun lexicon — the deterministic stand-in for POS tagging in
# specificity density (digits + proper nouns + concrete nouns).
_CONCRETE_NOUNS = frozenset(
    {
        "dashboard",
        "button",
        "screen",
        "chart",
        "graph",
        "table",
        "form",
        "page",
        "click",
        "cursor",
        "video",
        "image",
        "photo",
        "file",
        "report",
        "invoice",
        "email",
        "map",
        "code",
        "server",
        "database",
        "task",
        "ticket",
        "order",
        "product",
        "camera",
        "phone",
        "laptop",
        "keyboard",
        "widget",
        "row",
        "column",
        "pixel",
        "frame",
        "clip",
        "slider",
        "menu",
        "icon",
        "badge",
        "card",
    }
)

# Content-word stopwords for the abstraction-ratio denominator.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "as",
        "at",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "you",
        "your",
        "we",
        "our",
        "they",
        "their",
        "i",
        "he",
        "she",
        "his",
        "her",
        "not",
        "no",
        "so",
        "up",
        "out",
        "into",
        "from",
        "will",
        "can",
        "do",
        "does",
    }
)

# ⚠️ A FALLBACK, NOT THE RATE. `word_budget` and `_duration_budget` make a
# PHYSICAL prediction — "can this voice say these words inside these seconds?" —
# not an aesthetic judgement, so the reference has to be the rate the voice
# actually delivers. `score_script(..., wpm=...)` supplies the measured rate
# (`shared/speech_rate.resolve_natural_wpm`, resolved once per doctoring run at
# `stages/shared/script_doctor.py::_doctor`); this constant is what remains when
# no caller states one, and it is deliberately the same 150.0 as
# `time_budget.DEFAULT_WPM_NATURAL` so an unstated rate behaves exactly as this
# rubric did before the seam existed.
#
# WHY A FIXED YARDSTICK IS WRONG *HERE*, when elsewhere it would be right: a
# scoring constant that moves with the thing it measures is usually worthless,
# and that argument does not apply to this metric. The thing being scored is the
# WORD COUNT; the yardstick is the voice's SPEED, which is an independent,
# measurable property of the TTS backend. Holding it at 150 while the live voice
# delivers ~120 wpm does not make the score stricter — it inverts it. A beat that
# will overrun its clip scores a clean pass, and `script_doctor`'s rewrite loop,
# which is driven by exactly these flags, is then told the beat is UNDER budget
# and asked to ADD words. That is the overrun class this rubric should be
# catching, amplified by the gate meant to catch it.
_WPM = 150.0
_FLAG_THRESHOLD = 70.0  # metric.threshold != "pass" (score < 70) == flagged
_ABSTRACTION_CAP = 0.15  # cliches / content words that maps to score 0
_SPECIFICITY_TARGET = 10.0  # specific tokens per 100 words for full score
_HOOK_HEAD_WORDS = 8  # first ~3s at 150 wpm

# Composite metric weights (sum == 1.0). abstraction/redundancy scores are
# already goodness (see module docstring) → plain weighted mean here.
_COMPOSITE_WEIGHTS: dict[str, float] = {
    "hook_strength": 0.18,
    "turn_presence": 0.12,
    "specificity_density": 0.12,
    "abstraction_ratio": 0.10,
    "one_idea_per_beat": 0.10,
    "word_budget": 0.10,
    "cross_beat_redundancy": 0.08,
    "opener_variety": 0.06,
    "readability": 0.06,
    "cta_craft": 0.04,
    "sentence_cv": 0.02,
    "ttr": 0.02,
}


# --------------------------------------------------------------------------- #
# Tokenisation helpers
# --------------------------------------------------------------------------- #


def _words(text: str) -> list[str]:
    """Lowercased word tokens, keeping intra-word hyphens (``cutting-edge``)."""
    return re.findall(r"[a-z][a-z'-]*", text.lower())


def _opener_key(text: str) -> str:
    """The first-3-words signature of a beat, for opener-variety collisions."""
    return " ".join(re.findall(r"[a-z0-9']+", text.lower())[:3])


def _trigrams(text: str) -> set[tuple[str, ...]]:
    """Word trigram set of a beat, for cross-beat Jaccard overlap."""
    toks = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(toks[i : i + 3]) for i in range(len(toks) - 2)}


def _jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    """Jaccard similarity of two trigram sets (0.0 when either is empty)."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _verbatim_texts(verbatim_items: list[Any]) -> set[str]:
    """Extract the pinned phrase strings from ``verbatim_items`` defensively."""
    out: set[str] = set()
    for item in verbatim_items or []:
        if isinstance(item, str):
            out.add(item)
        elif isinstance(item, dict):
            for key in ("text", "phrase", "value", "content"):
                value = item.get(key)
                if isinstance(value, str):
                    out.add(value)
                    break
    return out


# --------------------------------------------------------------------------- #
# Per-line metrics
# --------------------------------------------------------------------------- #


def word_budget(text: str, clip_seconds: float | None, wpm: float = _WPM) -> MetricResult:
    """Pace a beat against its clip: ``target = clip_seconds * wpm/60`` words.

    ``wpm`` is THIS RUN'S measured speaking rate when the caller has one (see
    :data:`_WPM` for why a fixed reference is the wrong instrument for this
    particular metric), and :data:`_WPM` otherwise.

    Flags (score < 70) when the word count deviates > 20% from the budget.
    Neutral 75 when the beat has no bound clip seconds — ``threshold`` is
    ``"insufficient_evidence"``, NOT ``"pass"``: nothing was actually measured
    (there is no clip length to pace against), and labelling that a pass reads
    identically to "paced correctly" to anything consulting the threshold
    string directly rather than the raw score. ``75.0`` sits above
    ``_FLAG_THRESHOLD`` so it still does not FLAG the line either way — this is
    a labelling fix, not a gating one.
    """
    words = len(text.split())
    if not clip_seconds or clip_seconds <= 0:
        return metric(
            "word_budget",
            75.0,
            0.10,
            raw_value={"words": words, "target": 0},
            detail=f"{words} words; no clip seconds for budget",
            threshold="insufficient_evidence",
        )
    target = clip_seconds * wpm / 60.0
    deviation = abs(words - target) / target * 100 if target else 0.0
    score = 100.0 - deviation / 20 * 15 if deviation <= 20 else max(20.0, 70.0 - (deviation - 20))
    score = round(score, 1)
    return metric(
        "word_budget",
        score,
        0.10,
        raw_value={
            "words": words,
            "target": round(target, 1),
            "deviation_pct": round(deviation, 1),
        },
        detail=f"{words} words vs budget {target:.0f} ({deviation:.0f}% dev)",
    )


def _specificity_counts(text: str) -> tuple[int, int, int, int]:
    """Return ``(digit_tokens, proper_nouns, concrete_nouns, total_words)``.

    Proper nouns are capitalised tokens that are *not* the first token of their
    sentence (sentence-initial capitals are skipped).
    """
    proper = 0
    for sentence in _split_sentences(text):
        tokens = sentence.split()
        for pos, token in enumerate(tokens):
            clean = token.strip(string.punctuation)
            if pos > 0 and clean[:1].isupper() and clean[1:].islower() and len(clean) > 1:
                proper += 1
    all_words = text.split()
    digits = sum(1 for token in all_words if any(char.isdigit() for char in token))
    concrete = sum(1 for word in _words(text) if word in _CONCRETE_NOUNS)
    return digits, proper, concrete, len(all_words)


def specificity_density(text: str) -> MetricResult:
    """Specific tokens (digits + proper nouns + concrete nouns) per 100 words.

    Low density → "too generic" flag. Full score at >= 10 specific/100 words.
    """
    digits, proper, concrete, total = _specificity_counts(text)
    if total == 0:
        return metric(
            "specificity_density",
            50.0,
            0.12,
            raw_value={"density": 0},
            detail="empty beat",
            threshold="warn",
        )
    density = (digits + proper + concrete) / total * 100
    score = round(min(100.0, density / _SPECIFICITY_TARGET * 100), 1)
    return metric(
        "specificity_density",
        score,
        0.12,
        raw_value={
            "density": round(density, 1),
            "digits": digits,
            "proper": proper,
            "concrete": concrete,
        },
        detail=f"{density:.1f} specific tokens/100 words (target >= {_SPECIFICITY_TARGET:.0f})",
    )


def abstraction_ratio(text: str) -> MetricResult:
    """Cliche/abstract hits per content word (show-don't-tell gate).

    Lower raw ratio is better; the score is the inverted goodness so high
    abstraction → low score → flag.
    """
    tokens = _words(text)
    content = [word for word in tokens if word not in _STOPWORDS]
    hits = sum(1 for word in tokens if word in CLICHE_LEXICON)
    if not content:
        return metric(
            "abstraction_ratio",
            100.0,
            0.10,
            raw_value={"ratio": 0, "hits": hits},
            detail="no content words",
            threshold="pass",
        )
    ratio = hits / len(content)
    score = round(max(0.0, 100.0 * (1.0 - min(1.0, ratio / _ABSTRACTION_CAP))), 1)
    return metric(
        "abstraction_ratio",
        score,
        0.10,
        raw_value={"ratio": round(ratio, 3), "hits": hits, "content_words": len(content)},
        detail=f"{hits} cliche(s) / {len(content)} content words (ratio {ratio:.3f})",
    )


def one_idea_per_beat(text: str) -> MetricResult:
    """Count independent clauses; > 1 idea → split/trim flag.

    Ideas = sentence count + semicolons + coordinating conjunctions after a comma.
    """
    sentences = _split_sentences(text)
    semicolons = text.count(";")
    coord = len(re.findall(r",\s+(?:and|but|or|nor|for|so|yet)\s", text.lower()))
    ideas = len(sentences) + semicolons + coord
    score = 100.0 if ideas <= 1 else round(max(20.0, 100.0 - (ideas - 1) * 45.0), 1)
    return metric(
        "one_idea_per_beat",
        score,
        0.10,
        raw_value={
            "ideas": ideas,
            "sentences": len(sentences),
            "semicolons": semicolons,
            "coord": coord,
        },
        detail=f"{ideas} independent idea(s) (target 1)",
    )


def _opener_metric(text: str, seen_openers: set[str]) -> MetricResult:
    """Opener-variety for one beat vs the openers seen in earlier beats."""
    key = _opener_key(text)
    collision = bool(key) and key in seen_openers
    score = 30.0 if collision else 100.0
    detail = f"opener '{key}' repeats an earlier beat" if collision else f"opener '{key}' is unique"
    return metric(
        "opener_variety",
        score,
        0.06,
        raw_value={"opener": key, "collision": collision},
        detail=detail,
    )


def _redundancy_metric(text: str, prior_trigrams: list[set[tuple[str, ...]]]) -> MetricResult:
    """Cross-beat redundancy: max trigram Jaccard vs earlier beats (inverted)."""
    tris = _trigrams(text)
    max_jac = max((_jaccard(tris, prior) for prior in prior_trigrams), default=0.0)
    score = round(max(0.0, 100.0 * (1.0 - max_jac)), 1)
    return metric(
        "cross_beat_redundancy",
        score,
        0.08,
        raw_value={"max_jaccard": round(max_jac, 3)},
        detail=f"max trigram overlap {max_jac:.2f} vs earlier beats",
    )


def hook_strength(first_line: str) -> MetricResult:
    """Beat-1 cold-open: number / contrast / question in the first ~3s of words."""
    head = " ".join(first_line.split()[:_HOOK_HEAD_WORDS])
    head_words = set(re.findall(r"[a-z']+", head.lower()))
    signals = 0
    signals += 1 if any(char.isdigit() for char in head) else 0
    signals += 1 if "?" in head else 0
    signals += 1 if head_words & _HOOK_CONTRAST else 0
    signals += 1 if head_words & _QUESTION_WORDS else 0
    score = min(100.0, 40.0 + signals * 35.0)
    return metric(
        "hook_strength",
        score,
        0.18,
        raw_value={"signals": signals, "head": head},
        detail=f"{signals} hook signal(s) in first {_HOOK_HEAD_WORDS} words",
    )


def cta_craft(last_line: str, verbatim_items: list[Any]) -> MetricResult:
    """Last beat: a single verbatim-pinned imperative with no competing asks."""
    text = str(last_line)
    n_sentences = len(_split_sentences(text))
    tokens = re.findall(r"[a-z']+", text.lower())
    imperatives = sum(1 for word in tokens if word in _IMPERATIVE_VERBS)
    verbatims = _verbatim_texts(verbatim_items)
    pinned = any(phrase.lower() in text.lower() for phrase in verbatims) if verbatims else None

    score, reasons = _cta_penalties(n_sentences, imperatives, pinned)
    detail = "clean imperative CTA" if not reasons else "; ".join(reasons)
    return metric(
        "cta_craft",
        round(score, 1),
        0.04,
        raw_value={
            "sentences": n_sentences,
            "imperatives": imperatives,
            "verbatim_pinned": pinned,
        },
        detail=detail,
    )


def _cta_penalties(
    n_sentences: int, imperatives: int, pinned: bool | None
) -> tuple[float, list[str]]:
    """Apply the CTA deductions; returns ``(score, reasons)``."""
    score = 100.0
    reasons: list[str] = []
    if n_sentences > 1:
        score -= 40.0
        reasons.append("multiple sentences (competing asks)")
    if imperatives == 0:
        score -= 25.0
        reasons.append("no imperative verb")
    elif imperatives > 1:
        score -= 20.0
        reasons.append("multiple imperatives")
    if pinned is False:
        score -= 30.0
        reasons.append("not verbatim-pinned")
    return max(0.0, min(100.0, score)), reasons


# --------------------------------------------------------------------------- #
# Per-story metrics
# --------------------------------------------------------------------------- #


def turn_presence(lines: list[dict[str, Any]]) -> MetricResult:
    """Require exactly one strong pivot beat across the arc.

    Zero turns → flat (flag); more than one → diffuse (partial credit).
    """
    turn_ids: list[str] = []
    for idx, line in enumerate(lines):
        words = set(re.findall(r"[a-z']+", str(line.get("text", "")).lower()))
        if words & _TURN_MARKERS:
            turn_ids.append(str(line.get("id", idx)))
    count = len(turn_ids)
    if count == 1:
        score = 100.0
    elif count == 0:
        score = 30.0
    else:
        score = max(40.0, 100.0 - (count - 1) * 20.0)
    return metric(
        "turn_presence",
        round(score, 1),
        0.12,
        raw_value={"turn_count": count, "turn_line_ids": turn_ids},
        detail=f"{count} strong turn(s) detected (target 1)",
    )


def _duration_budget(
    lines: list[dict[str, Any]], target_seconds: float, wpm: float = _WPM
) -> MetricResult:
    """Global pacing: total word count vs the deliverable target duration.

    Same rate seam, and for the same reason, as :func:`word_budget`: this is the
    whole-script version of the identical physical question.
    """
    total_words = sum(len(str(line.get("text", "")).split()) for line in lines)
    target_words = target_seconds * wpm / 60.0 if target_seconds else 0.0
    if target_words <= 0:
        return metric(
            "duration_budget",
            75.0,
            0.0,
            raw_value={"words": total_words, "target": 0},
            detail=f"{total_words} words; no target duration",
            threshold="pass",
        )
    deviation = abs(total_words - target_words) / target_words * 100
    if deviation <= 10:
        score = 100.0
    elif deviation <= 20:
        score = 100.0 - (deviation - 10) / 10 * 30
    else:
        score = max(20.0, 70.0 - (deviation - 20))
    return metric(
        "duration_budget",
        round(score, 1),
        0.0,
        raw_value={
            "words": total_words,
            "target": round(target_words),
            "deviation_pct": round(deviation, 1),
        },
        detail=f"{total_words} words vs {target_words:.0f} budget ({deviation:.0f}% dev)",
    )


def _rewrap(source: MetricResult, name: str, weight: float) -> MetricResult:
    """Re-key a reused ``_text.py`` metric under this rubric's metric name."""
    return metric(
        name,
        source.score,
        weight,
        raw_value=source.raw_value,
        detail=source.detail,
        threshold=source.threshold,
    )


def _story_metrics(
    lines: list[dict[str, Any]], target_duration_seconds: float, wpm: float = _WPM
) -> dict[str, MetricResult]:
    """The per-story gate bundle over the whole concatenated script."""
    full = " ".join(str(line.get("text", "")) for line in lines)
    return {
        "readability": _rewrap(_readability_score(full), "readability", 0.06),
        "sentence_cv": _rewrap(_sentence_length_variance(full), "sentence_cv", 0.02),
        "ttr": _rewrap(_lexical_diversity(full), "ttr", 0.02),
        "preamble": _rewrap(_preamble_contamination(full), "preamble", 0.0),
        "turn_presence": turn_presence(lines),
        "duration_budget": _duration_budget(lines, target_duration_seconds, wpm),
    }


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def _line_metrics(
    text: str,
    clip_seconds: float | None,
    seen_openers: set[str],
    prior_trigrams: list[set[tuple[str, ...]]],
    *,
    is_first: bool,
    is_last: bool,
    verbatim_items: list[Any],
    wpm: float = _WPM,
) -> dict[str, MetricResult]:
    """All per-line metrics for one beat (hook only on beat-1, CTA only last)."""
    metrics: dict[str, MetricResult] = {
        "word_budget": word_budget(text, clip_seconds, wpm),
        "specificity_density": specificity_density(text),
        "abstraction_ratio": abstraction_ratio(text),
        "one_idea_per_beat": one_idea_per_beat(text),
        "opener_variety": _opener_metric(text, seen_openers),
        "cross_beat_redundancy": _redundancy_metric(text, prior_trigrams),
    }
    if is_first:
        metrics["hook_strength"] = hook_strength(text)
    if is_last:
        metrics["cta_craft"] = cta_craft(text, verbatim_items)
    return metrics


def composite(report: dict[str, Any]) -> float:
    """Weighted mean of the normalized (0-100) metric scores.

    Per-line metrics are averaged across beats; per-story metrics contribute
    their single score. abstraction/redundancy are already goodness — no
    re-inversion here.
    """
    by_name: dict[str, list[float]] = {}
    for line_metrics in report["per_line"].values():
        for name, result in line_metrics.items():
            by_name.setdefault(name, []).append(result.score)
    for name, result in report["per_story"].items():
        by_name.setdefault(name, []).append(result.score)

    weighted_sum = 0.0
    total_weight = 0.0
    for name, weight in _COMPOSITE_WEIGHTS.items():
        scores = by_name.get(name)
        if not scores:
            continue
        weighted_sum += sum(scores) / len(scores) * weight
        total_weight += weight
    return round(weighted_sum / total_weight, 2) if total_weight else 0.0


def score_script(
    lines: list[dict],
    *,
    verbatim_items: list[Any],
    target_duration_seconds: float,
    director: dict[str, Any] | None = None,
    wpm: float | None = None,
) -> dict[str, Any]:
    """Score a video script's beats against the deterministic craft rubric.

    Returns a ``ScriptReport``::

        {"composite": float,                     # 0-100
         "per_line": {line_id: {metric: MetricResult}},
         "per_story": {metric: MetricResult},
         "flags": {line_id: [metric_name, ...]}}  # locked lines never appear;
                                                    # a failing "preamble"
                                                    # (per_story) is attributed
                                                    # to the FIRST line's id

    ``director`` is accepted for contract stability (it tunes the LLM loop's
    aggressiveness upstream) but does not influence any deterministic score.

    ``wpm`` is the run's MEASURED speaking rate, and it does influence two scores
    (``word_budget`` per line, ``duration_budget`` per story) because both of them
    ask a physical question the voice's real speed is the only correct answer to —
    see :data:`_WPM`. ``None`` (or a non-positive/non-finite value, which is what a
    failed probe upstream would degrade to) falls back to :data:`_WPM`, so an
    unstated rate reproduces this rubric's pre-seam behaviour exactly rather than
    scoring every beat against a nonsense target.
    """
    rate = wpm if wpm is not None and math.isfinite(wpm) and wpm > 0.0 else _WPM
    per_line: dict[str, dict[str, MetricResult]] = {}
    flags: dict[str, list[str]] = {}
    seen_openers: set[str] = set()
    prior_trigrams: list[set[tuple[str, ...]]] = []
    count = len(lines)

    for idx, line in enumerate(lines):
        line_id = str(line.get("id", idx))
        text = str(line.get("text", ""))
        clip_seconds = line.get("seconds")
        metrics = _line_metrics(
            text,
            clip_seconds,
            seen_openers,
            prior_trigrams,
            is_first=idx == 0,
            is_last=idx == count - 1,
            verbatim_items=verbatim_items,
            wpm=rate,
        )
        per_line[line_id] = metrics
        seen_openers.add(_opener_key(text))
        prior_trigrams.append(_trigrams(text))

        if not line.get("locked", False):
            flagged = [name for name, result in metrics.items() if result.score < _FLAG_THRESHOLD]
            if flagged:
                flags[line_id] = flagged

    per_story = _story_metrics(lines, target_duration_seconds, rate)

    # ``preamble`` is a per_story metric (scored over the whole concatenated
    # script), but ``flags`` is keyed by line_id — the shape script_doctor's
    # ``_flagged_lines``/``_critique_msg`` actually consume
    # (`src/services/stages/shared/script_doctor.py`). Deliberately NOT added
    # to ``_COMPOSITE_WEIGHTS`` (see module docstring's "convention note" +
    # the comment there — that would force rebalancing all 12 weights and
    # shift every composite score fleet-wide). Instead, attribute a failing
    # preamble score to the FIRST line: ``_preamble_contamination`` regex-
    # matches against the *start* of the space-joined ``full`` text
    # (`_story_metrics`'s ``" ".join(...)`` + `^`-anchored patterns in
    # ``_text.py::_preamble_contamination``), which is structurally beat 0's
    # opening words — so the first line is the correct (and only sound)
    # attribution target, not an arbitrary choice. A locked first line is
    # left unflagged, matching the per-line loop's own lock rule above (a
    # locked line must never appear in ``flags``, and would be filtered out
    # of the revise set regardless).
    if lines and per_story["preamble"].score < _FLAG_THRESHOLD:
        first_line = lines[0]
        if not first_line.get("locked", False):
            first_id = str(first_line.get("id", 0))
            flags.setdefault(first_id, [])
            if "preamble" not in flags[first_id]:
                flags[first_id].append("preamble")

    # NOTE: ``duration_budget`` (per_story, weight 0.0, absent from
    # ``_COMPOSITE_WEIGHTS``) has the IDENTICAL inert shape as preamble did.
    # Deliberately left inert here per explicit scope instruction — flagging
    # it would make script_doctor loop on scripts it currently accepts
    # whenever total word count drifts from the deliverable target, a new
    # behaviour change riding along with this bug fix. Report separately.

    report: dict[str, Any] = {
        "composite": 0.0,
        "per_line": per_line,
        "per_story": per_story,
        "flags": flags,
    }
    report["composite"] = composite(report)
    return report
