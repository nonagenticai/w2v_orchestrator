"""Deterministic claim-grounding lint (advisory only) — ORCH-0.7.

Pure helpers, no DB, no client, no ``ctx``. Mirrors the frozen-dataclass +
free-function style of ``retention.py``: every function is a total function of
its arguments, so the same line/sentence lists always yield byte-identical
output and the module can never issue a render or a network call.

The lint scans authored narration *lines* for marketing superlatives ("the
best", "#1", "world-class") and for quantitative assertions the superlative
lexicon cannot see ("40% faster", "3x cheaper", "in 30 seconds"), then checks
whether the operator's ingested source material actually *supports* each claim.
A claim with no supporting source sentence is flagged ``unsourced``; one echoed
by a source sentence is ``grounded``; a line with no claim is ``none`` (never
flagged). Purely advisory — nothing here gates a run; the caller attaches the
insight strings to a stage output (never ``notes``, which would auto-degrade the
run) or, like ``claim_finish``, feeds ``flagged`` to a WM critic as bounded
evidence.

Inputs are duck-typed: a *line* may be a ``dict`` (the JSONB shape
``ctx.prior_output`` returns) or any object exposing ``id`` / ``text`` — the
``AuthoredLine`` field set — without importing that model (kept decoupled, like
``retention.py``). *Sentences* are plain strings (see
``source_ingest.grounding_sentences``).

Inertness contract: empty ``lines`` OR empty ``sentences`` ⇒ ``insights == []``,
so a run with no grounding material (or no lines) is byte-identical to before —
the lint only speaks when there is source material to check a claim against.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import cast

__all__ = [
    "ClaimAdvisory",
    "ClaimGroundingReport",
    "DisclaimerReq",
    "analyze_claims",
    "disclaimer_requirements",
    "find_quantitative",
    "find_superlatives",
    "is_grounded",
    "support_score",
]

# Marketing superlatives that assert a factual, checkable claim. Lower-cased;
# matched case-insensitively. Alpha-only terms match on word boundaries (so
# "only" never trips inside "commonly"); terms carrying spaces/symbols match as a
# substring. Author-overridable via the ``terms`` argument.
_SUPERLATIVES: frozenset[str] = frozenset(
    {
        "best",
        "fastest",
        "cheapest",
        "smartest",
        "safest",
        "easiest",
        "only",
        "#1",
        "number one",
        "world-class",
        "world class",
        "leading",
        "market-leading",
        "unmatched",
        "unbeatable",
        "unrivaled",
        "guaranteed",
        "revolutionary",
        "top-rated",
        "the most",
    }
)

# Quantitative / comparative assertions the superlative lexicon cannot see: a
# percentage ("40%"), a multiple ("3x"), or a duration ("in 30 seconds"). These
# are the *checkable numbers* a marketing script invents most readily, so they
# are flagged on exactly the same footing as a superlative.
_QUANTITATIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\d+(?:\.\d+)?%"),
    re.compile(r"\b\d+(?:\.\d+)?x\b", re.IGNORECASE),
    re.compile(
        r"\b\d+(?:\.\d+)?\s+(?:seconds?|minutes?|hours?|days?)\b",
        re.IGNORECASE,
    ),
)

# Function words carry no evidential weight, so they are dropped before the
# token-overlap score — otherwise every sentence "supports" every claim.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "to",
        "we",
        "with",
    }
)

# Tokeniser for the overlap score: a word/numeral, keeping an internal dot or
# hyphen ("3.5x", "world-class") but never a trailing sentence period.
_WORD_RE = re.compile(r"[a-z0-9%#]+(?:[.-][a-z0-9%#]+)*")


@dataclass(frozen=True)
class ClaimAdvisory:
    """One checkable claim found in a line, with its grounding verdict.

    ``claim_status`` ∈ {"grounded", "unsourced", "none"} (contract C6 verbatim):
    ``grounded`` — a source sentence supports it (``claim_source`` carries that
    sentence); ``unsourced`` — no source sentence supports it; ``none`` — the
    line carried no claim (never emitted into ``flagged``). ``matched_term`` is
    the superlative term or the matched numeral.
    """

    line_id: str
    text: str
    claim_status: str
    claim_source: str = ""
    matched_term: str = ""


@dataclass(frozen=True)
class ClaimGroundingReport:
    """Deterministic claim-grounding read-out for one line list.

    ``insights`` is the flat, human-readable, JSON-serializable list the caller
    surfaces (advisory). ``flagged`` carries one ``ClaimAdvisory``-shaped dict per
    claim found — superlative or quantitative, grounded + unsourced alike — in
    spine order, superlatives before numerals within a line. ``lines_checked`` is
    the count of lines carrying at least one claim.
    """

    lines_checked: int
    flagged: list[dict[str, object]]
    insights: list[str]


# ------------------------------------------------------------------ accessors --
def _field(line: object, name: str) -> object:
    """Read ``name`` off a line that may be a ``dict`` or an attribute-bag."""
    if isinstance(line, dict):
        return cast("dict[str, object]", line).get(name)
    return getattr(line, name, None)


def _text(line: object) -> str:
    return str(_field(line, "text") or "")


def _line_id(line: object) -> str:
    return str(_field(line, "id") or "")


# ------------------------------------------------------------------ functions --
def find_superlatives(text: str, *, terms: frozenset[str] | None = None) -> list[str]:
    """Return the superlative terms present in ``text``, in deterministic order.

    Empty/blank text → ``[]``. Alpha-only terms match on word boundaries; terms
    with spaces/symbols match as a substring. The result is sorted so the same
    text always yields the same order.
    """
    hay = str(text or "").lower()
    if not hay.strip():
        return []
    pool = terms if terms is not None else _SUPERLATIVES
    return [term for term in sorted(pool) if _term_present(term.lower(), hay)]


def _term_present(term: str, hay: str) -> bool:
    if term.isalpha():
        return re.search(rf"\b{re.escape(term)}\b", hay) is not None
    return term in hay


def find_quantitative(text: str) -> list[str]:
    """Return the quantitative claims present in ``text``, in deterministic order.

    Catches the checkable numbers the superlative lexicon misses: percentages
    (``40%``), multiples (``3x``) and durations (``30 seconds``). Empty/blank text
    → ``[]``. Matches are de-duplicated and sorted, so the same text always yields
    the same order.
    """
    hay = str(text or "")
    if not hay.strip():
        return []
    found: set[str] = set()
    for pattern in _QUANTITATIVE_PATTERNS:
        found.update(match.group(0).strip() for match in pattern.finditer(hay))
    return sorted(found)


def _content_words(text: str) -> frozenset[str]:
    """The evidential tokens of ``text``: lower-cased words minus function words."""
    return frozenset(
        word for word in _WORD_RE.findall(str(text or "").lower()) if word not in _STOPWORDS
    )


def support_score(term: str, sentences: Sequence[str]) -> tuple[float, str]:
    """How strongly does the corpus support ``term``? → (score, closest sentence).

    A graded companion to the binary :func:`is_grounded`: the Jaccard overlap of
    content words between ``term`` and each source sentence, maximised across the
    corpus, returned with the best-matching sentence. ``0.0`` means the corpus says
    nothing on the subject; a substring hit (what ``is_grounded`` tests) always
    scores ``> 0``. Empty ``term`` or empty ``sentences`` → ``(0.0, "")``, so a run
    with no ingested corpus scores every claim at zero rather than guessing.
    """
    needle = _content_words(term)
    if not needle:
        return (0.0, "")
    best_score = 0.0
    best_sentence = ""
    for sentence in sentences or []:
        haystack = _content_words(str(sentence))
        if not haystack:
            continue
        score = len(needle & haystack) / len(needle | haystack)
        if score > best_score:
            best_score, best_sentence = score, str(sentence).strip()
    return (best_score, best_sentence)


def is_grounded(term: str, sentences: Sequence[str]) -> tuple[bool, str]:
    """Is ``term`` supported by any source sentence? → (grounded, matching snippet).

    A claim is grounded when its term appears (case-insensitive substring) in a
    source sentence; the first such sentence is returned as the source snippet.
    No support (or empty ``term``/``sentences``) → ``(False, "")``.
    """
    needle = str(term or "").lower().strip()
    if not needle:
        return (False, "")
    for sentence in sentences or []:
        if needle in str(sentence).lower():
            return (True, str(sentence).strip())
    return (False, "")


def analyze_claims(
    lines: Sequence[object],
    sentences: Sequence[str],
    *,
    terms: frozenset[str] | None = None,
) -> ClaimGroundingReport:
    """Compose the claim-grounding read-out for a line list against source sentences.

    Each claim found in a line — a superlative (:func:`find_superlatives`) or a
    quantitative assertion (:func:`find_quantitative`) — becomes a
    ``ClaimAdvisory`` (grounded when a source sentence supports it, else
    unsourced) appended to ``flagged``. Empty ``lines`` OR empty ``sentences`` ⇒
    ``insights == []`` (the lint only speaks when there is source material to check
    against), so an opted-out / grounding-less run stays byte-identical to before.
    """
    line_list = list(lines or [])
    sentence_list = list(sentences or [])

    flagged: list[dict[str, object]] = []
    lines_checked = 0
    for line in line_list:
        line_text = _text(line)
        found = [*find_superlatives(line_text, terms=terms), *find_quantitative(line_text)]
        if not found:
            continue
        lines_checked += 1
        line_id = _line_id(line)
        for term in found:
            grounded, source = is_grounded(term, sentence_list)
            advisory = ClaimAdvisory(
                line_id=line_id,
                text=line_text,
                claim_status="grounded" if grounded else "unsourced",
                claim_source=source,
                matched_term=term,
            )
            flagged.append(asdict(advisory))

    insights: list[str] = []
    if line_list and sentence_list:
        for row in flagged:
            if row["claim_status"] == "unsourced":
                insights.append(
                    f'unsourced claim "{row["matched_term"]}" in {row["line_id"] or "a line"}: '
                    "no ingested source supports it — soften the wording or cite a source"
                )
    return ClaimGroundingReport(lines_checked=lines_checked, flagged=flagged, insights=insights)


# ------------------------------------------------------ disclaimer rule engine --
@dataclass(frozen=True)
class DisclaimerReq:
    """One disclaimer a claim class mandates, and the lines that triggered it.

    ``kind`` ∈ :data:`DISCLAIMER_KINDS`. ``text`` is a FIXED template — never LLM
    prose. ``line_ids`` is the sorted set of spine lines whose claims triggered it.
    """

    kind: str
    text: str
    line_ids: tuple[str, ...]


#: Deterministic rules: claim class → the disclaimer it mandates. No LLM decides
#: WHETHER a disclaimer is needed, and none writes WHAT it says either: the spec's
#: `disclaimer-writer` worker is deliberately not used. Free-form legal prose from a
#: language model is a liability, not a feature — a fixed, reviewed template is safer,
#: reproducible (the manifest hash depends on it) and free. Order here is the emitted
#: order and the card's reading order.
_DISCLAIMER_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "comparative",
        re.compile(
            r"\b(?:than|vs\.?|versus|compared to|better|worse|more|less|"
            r"best|worst|fastest|cheapest|leading|#1|number one)\b",
            re.IGNORECASE,
        ),
        "Comparison based on publicly available information.",
    ),
    (
        "pricing",
        re.compile(
            r"[$€£]\d|\b(?:price|pricing|priced|free|per (?:month|seat|user)|/mo)\b",
            re.IGNORECASE,
        ),
        "Pricing subject to change.",
    ),
    (
        "performance",
        re.compile(
            r"\d+\s?%|\b(?:faster|save|saves|saved|increase|boost|roi|uptime)\b",
            re.IGNORECASE,
        ),
        "Results may vary.",
    ),
    (
        "testimonial",
        re.compile(
            r"\b(?:customers say|users report|loved|testimonial)\b|[\"“][^\"”]{8,}[\"”]",
            re.IGNORECASE,
        ),
        "Individual results may vary.",
    ),
)

#: The fixed emission order (also the tie-break for a stable manifest hash).
DISCLAIMER_KINDS: tuple[str, ...] = tuple(kind for kind, _, _ in _DISCLAIMER_RULES)


def disclaimer_requirements(flagged: Sequence[dict[str, object]]) -> list[DisclaimerReq]:
    """Which disclaimers do the flagged claims mandate? (A18 §4, AC-8.)

    A deterministic rule engine over the rows :func:`analyze_claims` already
    produced — a comparative claim mandates a comparison disclaimer, a price claim a
    pricing one, a performance number a results one, a quoted/testimonial line an
    individual-results one. One requirement per matched class (deduped, line ids
    merged), in :data:`DISCLAIMER_KINDS` order.

    Empty ``flagged`` ⇒ ``[]`` — no claims, no card, byte-identical shot list.
    """
    by_kind: dict[str, set[str]] = {}
    for row in flagged or []:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "")
        if not text.strip():
            continue
        line_id = str(row.get("line_id") or "")
        for kind, pattern, _template in _DISCLAIMER_RULES:
            if pattern.search(text):
                by_kind.setdefault(kind, set()).add(line_id)

    return [
        DisclaimerReq(kind=kind, text=template, line_ids=tuple(sorted(by_kind[kind])))
        for kind, _pattern, template in _DISCLAIMER_RULES
        if kind in by_kind
    ]
