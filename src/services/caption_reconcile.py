"""Word-timed caption reconcile core: align an authored script to ASR word timings.

The soft/burned caption tracks in :mod:`src.services.captions` are *line*-timed:
each spoken line gets the whole duration of its narration clip. That is accurate
at the line boundary but flat inside the line — every word of a 12-word sentence
shares one ``[start, end)`` span, so a per-word "karaoke" caption can't be minted.

This module lifts the timing to the *word* level. Given the authored script words
of a clip (the surface text we must display, punctuation and all) and the Whisper
ASR words for that clip's synthesised audio (accurate per-word ``start``/``end``),
it produces one :class:`TimedToken` per *script* word carrying a real timestamp:

* words the ASR recognised are timed directly from the ASR span (``source="asr"``);
* words the ASR dropped/substituted are timed by *interpolation* across the gap
  between their timed neighbours (``source="interp"``);
* if the alignment is too weak to trust at all, every script word is spread evenly
  across the clip (``source="even"``) — a graceful degrade, never a crash.

INVARIANT: :attr:`TimedToken.text` is always the **script** surface word (with its
own punctuation) — never the raw ASR transcription. ASR supplies *timing only*; the
authored script is authoritative for *what the caption says*.

Pure functions, no I/O. :func:`build_asr_words` adapts Whisper's per-word dicts;
:func:`align_script_to_words` runs the alignment. Downstream,
:func:`src.services.caption_cues.group_tokens_into_cues` packs the tokens into
wrapped on-screen cues.
"""

from __future__ import annotations

import string
import unicodedata
from dataclasses import dataclass

# Needleman-Wunsch scoring: reward exact/variant matches, penalise gaps + subs.
_MATCH = 2
_MISMATCH = -1
_GAP = -1

_PUNCT = string.punctuation

_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


@dataclass(frozen=True)
class AsrWord:
    """One ASR-recognised word with its audio-relative timing + confidence."""

    start: float
    end: float
    word: str
    probability: float = 1.0


@dataclass(frozen=True)
class TimedToken:
    """One *script* word with a resolved ``[start, end)`` timestamp.

    ``text`` is the authored script surface word (its own punctuation included) —
    INVARIANT: never the raw ASR surface. ``source`` records how the timing was
    resolved: ``"asr"`` (matched an ASR word directly), ``"interp"`` (interpolated
    across a gap of dropped/substituted words), or ``"even"`` (whole alignment
    discarded; word spread evenly across the clip).
    """

    text: str
    start: float
    end: float
    source: str


def _norm(s: str) -> str:
    """Fold ``s`` to a comparison key: NFKD, strip combining marks, lowercase, trim punctuation.

    A pure-punctuation (or empty) token normalises to ``""``.
    """
    decomposed = unicodedata.normalize("NFKD", s)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.lower().strip(_PUNCT).strip()


def _int_to_words(n: int) -> str:
    """Spell a non-negative integer 0..9999 in English words (``2024`` -> ``two thousand twenty four``)."""
    if n < 0 or n > 9999:
        return ""
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens = _TENS[n // 10]
        return tens if n % 10 == 0 else f"{tens} {_ONES[n % 10]}"
    if n < 1000:
        head = f"{_ONES[n // 100]} hundred"
        rem = n % 100
        return head if rem == 0 else f"{head} {_int_to_words(rem)}"
    head = f"{_ONES[n // 1000]} thousand"
    rem = n % 1000
    return head if rem == 0 else f"{head} {_int_to_words(rem)}"


def _variants(token: str) -> set[str]:
    """Normalised surface forms of ``token`` that an ASR word might legitimately match.

    Beyond the token's own :func:`_norm` key, a pure-integer script token also
    yields its spelled-out English forms so a digit like ``"5"`` matches the ASR's
    ``"five"``, and a 4-digit year like ``"2024"`` also emits the two-group reading
    ``"twenty twenty four"`` (and the ``"oh"``-inflected ``"twenty oh five"`` form
    for years whose last pair is < 10).

    LIMITATION (documented, intentional): multi-*word* spellings are included as
    whole-string variants but only match when the ASR happens to emit the number as
    a single token. True absorption of several consecutive ASR words into one script
    number token is NOT implemented — such a token falls through to a substitution
    and receives interpolated timing, which is visually fine for captions. Single-
    word number matches (0-19, the tens, and any number whose reading is one word)
    work exactly.
    """
    base = _norm(token)
    out: set[str] = set()
    if base:
        out.add(base)
    if base.isdigit():
        n = int(base)
        if 0 <= n <= 9999:
            spelled = _norm(_int_to_words(n))
            if spelled:
                out.add(spelled)
        if 1000 <= n <= 9999:
            hi, lo = divmod(n, 100)
            if lo != 0:
                out.add(_norm(f"{_int_to_words(hi)} {_int_to_words(lo)}"))
                if lo < 10:
                    out.add(_norm(f"{_int_to_words(hi)} oh {_int_to_words(lo)}"))
    return out


def build_asr_words(segment_words: list[dict]) -> list[AsrWord]:
    """Adapt Whisper per-word dicts (``{start, end, word, probability}``) into typed rows.

    Rows without a numeric ``start``/``end`` are skipped (they carry no timing);
    a missing/invalid ``probability`` defaults to ``1.0``. The raw ``word`` surface
    (often carrying Whisper's leading space) is preserved verbatim — normalisation
    happens at match time, not here.
    """
    rows: list[AsrWord] = []
    for entry in segment_words:
        try:
            start = float(entry["start"])
            end = float(entry["end"])
        except KeyError, TypeError, ValueError:
            continue
        word = str(entry.get("word") or "")
        raw_prob = entry.get("probability")
        try:
            probability = float(raw_prob) if raw_prob is not None else 1.0
        except TypeError, ValueError:
            probability = 1.0
        rows.append(AsrWord(start=start, end=end, word=word, probability=probability))
    return rows


def _clamp_monotonic(tokens: list[TimedToken], *, floor: float) -> list[TimedToken]:
    """Force a non-decreasing, non-overlapping timeline: each token starts >= the prior end."""
    out: list[TimedToken] = []
    cursor = floor
    for tok in tokens:
        start = max(tok.start, cursor)
        end = max(tok.end, start)
        out.append(TimedToken(text=tok.text, start=start, end=end, source=tok.source))
        cursor = end
    return out


def _nw_timeline(  # noqa: C901 — Needleman-Wunsch fill + traceback; a single DP over a shared score matrix, splitting further hurts clarity
    norm_script: list[str],
    variants_script: list[set[str]],
    asr: list[AsrWord],
    norm_asr: list[str],
) -> tuple[list[float], list[float], list[str]]:
    """Needleman-Wunsch align script→ASR; return per-script ``(starts, ends, sources)``.

    ``sources[k]`` is ``"asr"`` for a direct match (timed from the ASR word) or
    ``"pending"`` for a substitution / ASR-drop (left for the caller to interpolate).
    Extra ASR words (no script counterpart) are discarded — the script is authoritative.
    """
    m = len(norm_script)
    n = len(asr)

    def _matched(i: int, j: int) -> bool:
        ns = norm_script[i]
        na = norm_asr[j]
        if ns and ns == na:
            return True
        return bool(na) and na in variants_script[i]

    score = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        score[i][0] = i * _GAP
    for j in range(1, n + 1):
        score[0][j] = j * _GAP
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag = score[i - 1][j - 1] + (_MATCH if _matched(i - 1, j - 1) else _MISMATCH)
            up = score[i - 1][j] + _GAP  # consume script word, ASR gap -> ASR-drop
            left = score[i][j - 1] + _GAP  # consume ASR word, script gap -> ASR-extra
            score[i][j] = max(diag, up, left)

    # traceback (diagonal preferred on ties); "pending" == not yet timed.
    starts: list[float] = [0.0] * m
    ends: list[float] = [0.0] * m
    sources: list[str] = ["pending"] * m
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            step = _MATCH if _matched(i - 1, j - 1) else _MISMATCH
            if score[i][j] == score[i - 1][j - 1] + step:
                if step == _MATCH:
                    starts[i - 1] = asr[j - 1].start
                    ends[i - 1] = asr[j - 1].end
                    sources[i - 1] = "asr"
                # else: substitution -> leave pending for interpolation
                i -= 1
                j -= 1
                continue
        if i > 0 and score[i][j] == score[i - 1][j] + _GAP:
            i -= 1  # ASR-drop: script word with no ASR counterpart -> pending
            continue
        j -= 1  # ASR-extra: discard the ASR word
    return starts, ends, sources


def _interpolate_pending(
    starts: list[float],
    ends: list[float],
    sources: list[str],
    *,
    clip_start: float,
    clip_end: float,
) -> None:
    """Time each un-timed ("pending") run by spreading it across its timed neighbours."""
    m = len(sources)
    k = 0
    while k < m:
        if sources[k] != "pending":
            k += 1
            continue
        run_end = k
        while run_end < m and sources[run_end] == "pending":
            run_end += 1
        t0 = ends[k - 1] if k > 0 else clip_start  # end of prior resolved word
        t1 = starts[run_end] if run_end < m else clip_end  # start of next timed word
        if t1 < t0:
            t1 = t0
        count = run_end - k
        width = (t1 - t0) / count
        for offset in range(count):
            starts[k + offset] = t0 + offset * width
            ends[k + offset] = t0 + (offset + 1) * width
            sources[k + offset] = "interp"
        k = run_end


def align_script_to_words(
    script_tokens: list[str],
    asr_words: list[AsrWord],
    *,
    clip_start: float,
    clip_end: float,
    min_match: float = 0.6,
) -> list[TimedToken]:
    """Align authored ``script_tokens`` to ``asr_words``, timing each script word.

    Pure-punctuation / empty script tokens are dropped (they never become standalone
    captioned words). The remaining script words are aligned to the ASR words with
    Needleman-Wunsch over normalised forms (match ``+2``, mismatch/gap ``-1``,
    traceback prefers the diagonal on ties). Diagonal matches take the ASR word's
    span (``source="asr"``); substitutions and ASR-drops are left un-timed and then
    interpolated across the gap between their timed neighbours (``source="interp"``);
    extra ASR words (no script counterpart) are discarded — the script is authoritative.

    If the fraction of directly-matched script words falls below ``min_match``, the
    whole alignment is thrown away and every script word is distributed evenly across
    ``[clip_start, clip_end)`` (``source="even"``). The returned timeline is always
    monotonic and non-overlapping.
    """
    script = [tok for tok in script_tokens if _norm(tok)]
    m = len(script)
    if m == 0:
        return []

    starts, ends, sources = _nw_timeline(
        [_norm(tok) for tok in script],
        [_variants(tok) for tok in script],
        asr_words,
        [_norm(a.word) for a in asr_words],
    )

    matched_count = sum(1 for source in sources if source == "asr")
    match_ratio = matched_count / max(1, m)

    # --- degrade: too weak to trust -> even distribution -----------------------
    if match_ratio < min_match:
        width = (clip_end - clip_start) / m
        even = [
            TimedToken(
                text=script[k],
                start=clip_start + k * width,
                end=clip_start + (k + 1) * width,
                source="even",
            )
            for k in range(m)
        ]
        return _clamp_monotonic(even, floor=clip_start)

    # --- interpolate pending runs across their timed neighbours ----------------
    _interpolate_pending(starts, ends, sources, clip_start=clip_start, clip_end=clip_end)

    tokens = [
        TimedToken(text=script[k], start=starts[k], end=ends[k], source=sources[k])
        for k in range(m)
    ]
    return _clamp_monotonic(tokens, floor=clip_start)
