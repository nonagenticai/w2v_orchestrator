"""Pack word-timed :class:`~src.services.caption_reconcile.TimedToken`s into on-screen cues.

:func:`src.services.caption_reconcile.align_script_to_words` produces one timed
token per script word. A caption *cue* is a short, wrapped block shown on screen for
a span of time — at most :data:`max_lines` lines, each at most :data:`max_chars_per_line`
characters. This module packs the tokens into such cues.

Packing rule (run 01a11295, 2026-10-06): a cue never spans a sentence end
(``.``, ``?``, ``!``); within a sentence the breaks are chosen together — a
clause boundary (``, ; : —``) first, then a break right before a clause opener
(where/which/that/and…), never a function word at a cue's end, never a stranded
1-2 word tail, and (run 01a11693) never a break a few words after a clause
opener, between that clause's subject and its verb ("…where targets" /
"already pay attention."). Lines inside a cue are wrapped greedily (video_compiler
re-wraps the cue text with its own greedy wrap anyway).

Token timestamps are consumed as-is: they are already in whatever timebase the align
step received via ``clip_start`` (clip-relative in the caller's usage). A cue's
``start`` is its first token's ``start``; its ``end`` is its last token's ``end``,
plus at most :data:`CUE_TAIL_SECONDS` when another cue of the clip follows (never
past that cue's start). The emitted cues are guaranteed monotonic and non-overlapping.

Pure functions, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.services.caption_reconcile import TimedToken
from src.services.stages.promo.headline import DETERMINERS, PREPOSITIONS, SUBJECT_PRONOUNS, is_verb

_PREPOSITIONS = PREPOSITIONS

_SENTENCE_FINAL = (".", "?", "!")
#: A cue may end right after one of these: the clause/phrase boundary it marks.
_CLAUSE_FINAL = (",", ";", ":", "\u2014", "\u2013", "-")
#: Words a cue must never END on — they bind to what follows ("…code, not" /
#: "language models."; "…scraped or" / "bought lists.", run 01a11295).
_FUNCTION_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "if",
        "in", "into", "is", "its", "nor", "not", "of", "on", "onto", "or", "our",
        "so", "than", "that", "the", "their", "these", "this", "those", "to", "was",
        "were", "which", "who", "with", "your",
    }
)  # fmt: skip
#: A cue that is followed by another cue of the same clip holds on for at most
#: this long after its last word (never past the next cue's start), so the box
#: does not blink off between two cues of one sentence. The LAST cue of a clip
#: gets none: it ends on its last word, never in the silence or the hold after.
CUE_TAIL_SECONDS = 0.2

# Break costs (lower is better). A clause boundary is free; any other word is a
# plain break; ending on a function word or stranding a 1-2 word tail is a
# defect the packer only accepts when nothing else fits.
_COST_CUE = 5.0
_COST_PLAIN_BREAK = 30.0
#: A break right BEFORE a word that opens a clause ("…assets | and places…",
#: "…them | where targets…") is nearly as good as a comma.
_COST_BEFORE_OPENER = 10.0
#: A break right before a preposition ("…decision back | to the exact rule…")
#: keeps the phrase it opens whole; a plain break may cut a noun phrase in two
#: ("…back to the exact" / "rule that allowed it.", render 01a1179d).
_COST_BEFORE_PREPOSITION = 20.0
#: A break just AFTER a clause opener, before its verb ("…where targets" /
#: "already pay attention.", run 01a11693): the clause is cut between its
#: subject and its verb. Only accepted when nothing else fits.
_COST_INSIDE_OPENED_CLAUSE = 200.0
#: How many words after an opener a break still counts as inside its subject.
_OPENED_CLAUSE_WORDS = 3
#: Coordinators: they open a clause only before a new subject or a verb.
_COORDINATORS = frozenset({"and", "but", "or"})
#: Words that open a clause (relative, subordinate, coordinating).
_CLAUSE_OPENERS = frozenset(
    {
        "where", "which", "who", "whom", "whose", "that", "when", "while", "whereas",
        "because", "before", "after", "until", "unless", "if", "and", "but", "or",
    }
)  # fmt: skip
_COST_FUNCTION_WORD_END = 1000.0
_COST_STRANDED_TAIL = 500.0
_STRANDED_TAIL_WORDS = 2


@dataclass(frozen=True)
class ShotCue:
    """One on-screen caption cue: ``[start, end)`` seconds of up-to-``max_lines`` wrapped ``lines``.

    Times are relative to the clip start (whatever timebase the align step received),
    matching the per-clip caption overlay the compiler burns in.

    ``words`` is the flat, in-order list of ``(text, start, end)`` for every token
    packed into this cue — the per-word timings the compiler needs for karaoke
    ``{\\k}`` highlight. It carries the SAME clip-relative timebase as the tokens;
    ``lines``/``start``/``end`` are unchanged by its presence.
    """

    start: float
    end: float
    lines: list[str]
    words: list[tuple[str, float, float]] = field(default_factory=list)


def _is_sentence_final(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in _SENTENCE_FINAL


def _is_clause_final(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and (stripped[-1] in _SENTENCE_FINAL or stripped.endswith(_CLAUSE_FINAL))


def _is_function_word(text: str) -> bool:
    return text.strip("\"'()[]").lower() in _FUNCTION_WORDS


def _wrap(words: list[str], max_chars: int) -> list[str]:
    """Greedy wrap of ``words`` into lines of at most ``max_chars`` (a longer word
    gets a line of its own)."""
    lines: list[str] = []
    cur = ""
    for word in words:
        candidate = word if not cur else f"{cur} {word}"
        if cur and len(candidate) > max_chars:
            lines.append(cur)
            cur = word
        else:
            cur = candidate
    if cur:
        lines.append(cur)
    return lines


def _break_cost(words: list[str], end: int) -> float:
    """Cost of a cue that ends after ``words[end - 1]``."""
    cost = _COST_CUE
    if end == len(words):
        return cost  # the sentence's own end
    last = words[end - 1]
    if _is_clause_final(last):
        return cost
    if _is_function_word(last):
        return cost + _COST_FUNCTION_WORD_END
    start = max(0, end - _OPENED_CLAUSE_WORDS)
    if any(_opens_clause(words, i) for i in range(start, end)):
        return cost + _COST_INSIDE_OPENED_CLAUSE
    if _opens_clause(words, end):
        return cost + _COST_BEFORE_OPENER
    if _bare_word(words[end]) in _PREPOSITIONS:
        return cost + _COST_BEFORE_PREPOSITION
    return cost + _COST_PLAIN_BREAK


def _opens_clause(words: list[str], index: int) -> bool:
    """``words[index]`` opens a clause. A coordinator only does when what follows
    is a new subject or a verb ("…assets | and places them…"); "and" inside one
    noun phrase ("financial and consent decisions") opens nothing."""
    word = _bare_word(words[index])
    if word not in _CLAUSE_OPENERS:
        return False
    if word not in _COORDINATORS:
        return True
    if index + 1 >= len(words):
        return False
    following = words[index + 1]
    return _bare_word(following) in DETERMINERS | SUBJECT_PRONOUNS or is_verb(following)


def _bare_word(text: str) -> str:
    return text.strip("\"'()[],.;:!?").lower()


def _split_sentence(words: list[str], *, max_chars_per_line: int, max_lines: int) -> list[int]:
    """Cue sizes for one sentence: clause boundaries first, never a function word
    at a cue's end, never a stranded 1-2 word tail — each cue fitting
    ``max_lines`` lines of ``max_chars_per_line`` (fewest, cheapest breaks win)."""
    n = len(words)
    best: list[float] = [0.0] + [float("inf")] * n
    back: list[int] = [0] * (n + 1)
    for end in range(1, n + 1):
        for begin in range(end - 1, -1, -1):
            if end - begin > 1 and len(_wrap(words[begin:end], max_chars_per_line)) > max_lines:
                break  # a longer cue only wraps to more lines
            cost = best[begin] + _break_cost(words, end)
            if end == n and begin > 0 and end - begin <= _STRANDED_TAIL_WORDS:
                cost += _COST_STRANDED_TAIL
            if cost < best[end]:
                best[end] = cost
                back[end] = begin
    sizes: list[int] = []
    end = n
    while end > 0:
        sizes.append(end - back[end])
        end = back[end]
    return sizes[::-1]


def _sentences(tokens: list[TimedToken]) -> list[list[TimedToken]]:
    """``tokens`` (blank ones dropped) cut after every sentence-final token."""
    out: list[list[TimedToken]] = []
    cur: list[TimedToken] = []
    for tok in tokens:
        if not tok.text:
            continue
        cur.append(tok)
        if _is_sentence_final(tok.text):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def group_tokens_into_cues(
    tokens: list[TimedToken],
    *,
    max_chars_per_line: int = 42,
    max_lines: int = 2,
) -> list[ShotCue]:
    """Pack ``tokens`` into wrapped, monotonic, non-overlapping cues.

    A cue never spans a sentence end. Inside a sentence the cue breaks are chosen
    together (:func:`_split_sentence`): a clause boundary (``, ; : —``) first, a
    function word never at a cue's end, a 1-2 word tail never left alone, each
    cue at most ``max_lines`` lines of ``max_chars_per_line``. Greedy filling
    broke run 01a11295's captions mid-clause ("…deterministic code, not" /
    "language models.").

    Timing: a cue starts on its first word and ends on its last word's end — plus
    :data:`CUE_TAIL_SECONDS`, capped at the next cue's start, when another cue of
    the same clip follows. The last cue ends on its last word: never in the
    silence or the hold after the narration. Empty-text tokens are skipped;
    ``[]`` for empty input.
    """
    cues: list[ShotCue] = []
    for sentence in _sentences(tokens):
        words = [tok.text for tok in sentence]
        at = 0
        for size in _split_sentence(
            words, max_chars_per_line=max_chars_per_line, max_lines=max_lines
        ):
            part = sentence[at : at + size]
            at += size
            cues.append(
                ShotCue(
                    start=part[0].start,
                    end=part[-1].end,
                    lines=_wrap([tok.text for tok in part], max_chars_per_line),
                    words=[(tok.text, tok.start, tok.end) for tok in part],
                )
            )
    return _clamp_cues_monotonic(_with_tails(cues))


def _with_tails(cues: list[ShotCue]) -> list[ShotCue]:
    """Each cue but the last held :data:`CUE_TAIL_SECONDS` longer, up to the next start."""
    out: list[ShotCue] = []
    for index, cue in enumerate(cues):
        end = cue.end
        if index + 1 < len(cues):
            end = max(end, min(end + CUE_TAIL_SECONDS, cues[index + 1].start))
        out.append(ShotCue(start=cue.start, end=end, lines=cue.lines, words=cue.words))
    return out


def _clamp_cues_monotonic(cues: list[ShotCue]) -> list[ShotCue]:
    """Force a monotonic, non-overlapping cue timeline (each cue starts ≥ prior end)."""
    result: list[ShotCue] = []
    cursor = float("-inf")
    for cue in cues:
        start = max(cue.start, cursor)
        end = max(cue.end, start)
        result.append(ShotCue(start=start, end=end, lines=cue.lines, words=cue.words))
        cursor = end
    return result
