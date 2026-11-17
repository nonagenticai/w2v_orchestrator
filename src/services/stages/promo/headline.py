"""On-screen phrase grammar for the promo motion brief — every phrase reads as one.

Why (frame QA of run 01a11511, 2026-10-07): on a platform-written script of
plain sentences the brief put fragments on screen that read as broken copy —
"Merqi runs motion / you keep final say" (words dropped from INSIDE a clause),
"Ever requesting contact" (a stranded adverb opening the phrase, cut from
"before ever requesting contact"), "Builds multimedia assets tailored" (ending on
a dangling participle). Every one of them passed the "words from the line"
rule, because that rule is a bag of words.

:func:`phrase_problem` is the deterministic gate both the rule author and the
model's answers go through. A phrase on a card must be:

1. **a span of the line** — its words appear in the line consecutively, as
   written (never words dropped from inside a clause). Two derived forms count
   as the line: a shared-head coordination distributes ("critical financial and
   consent decisions" also says "financial decisions" and "consent decisions"),
   and a spelled-out number may be shown as digits ("over thirty" → "30", "30+");
2. **cut at a boundary** — it does not stop inside a noun phrase (the line's
   next word is not a content word that continues it, and not "of"), and does
   not cut a word in half;
3. **not opened by a stranded word** — never a conjunction / subordinator
   ("and", "while", "before", "that") or a stranded adverb ("ever", "also");
4. **not ended by a dangling word** — never a determiner, preposition,
   conjunction, auxiliary, subject pronoun, a participle ("tailored"), or an
   adjective ("financial", "highest").

The ``label`` slot — a handoff's route ("Route automatically", "Tailored") and
a ladder's end rungs ("Prohibited", "Automatic") — names an action or a level,
which is a verb group or a lone modifier by nature, so rule 4's
participle/adjective clause does not apply to it.

A small closed lexicon, no NLP dependency: function words are closed classes;
participles and adjectives are recognised by suffix with a list of the nouns
that share the suffix; a short list of verbs (with their inflections) marks
where a subject ends.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

_TOKEN = re.compile(r"[a-z0-9]+")

#: Spelled-out numbers a card may show as digits ("thirty" → "30").
_NUMBER_WORDS: dict[str, str] = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100", "thousand": "1000",
}  # fmt: skip
#: "over thirty" / "more than 30" — the line's own way of saying "30+".
_AT_LEAST = re.compile(
    r"\b(?:over|more than|at least|upwards of)\s+(\d[\d,]*)(?![\d+%])", re.IGNORECASE
)

DETERMINERS = frozenset(
    {
        "the", "a", "an", "each", "every", "this", "that", "these", "those", "some", "any",
        "all", "our", "your", "their", "its", "my", "his", "her",
    }
)  # fmt: skip
PREPOSITIONS = frozenset(
    {
        "to", "of", "for", "at", "in", "on", "by", "with", "from", "into", "onto", "via",
        "through", "over", "under", "about", "after", "before", "between", "without",
        "within", "across", "per", "than", "as", "toward", "towards", "upon", "against",
        "around", "along", "behind", "beyond", "during", "among", "inside", "outside",
    }
)  # fmt: skip
CONJUNCTIONS = frozenset(
    {
        "and", "or", "but", "nor", "so", "yet", "while", "whereas", "because", "although",
        "though", "before", "after", "until", "unless", "if", "when", "whenever", "once",
        "since", "than", "that", "which", "who", "whom", "whose", "where", "then",
    }
)  # fmt: skip
AUXILIARIES = frozenset(
    {
        "is", "are", "was", "were", "be", "been", "being", "am", "can", "could", "will",
        "would", "shall", "should", "may", "might", "must", "do", "does", "did", "has",
        "have", "had",
    }
)  # fmt: skip
SUBJECT_PRONOUNS = frozenset({"i", "we", "they", "he", "she", "it", "you"})
#: Adverbs that only mean something attached to the words around them.
STRANDED_ADVERBS = frozenset(
    {"ever", "also", "even", "just", "still", "already", "too", "very", "only", "more",
     "most", "less", "least", "next", "now", "again", "then", "first"}
)  # fmt: skip
#: Words that turn what follows them in their clause into its opposite (or bound
#: it). "no" is one of them, not a determiner: "No campaign goes live…" →
#: "Campaign goes live…" says the opposite. ``n't`` is matched by suffix.
NEGATIONS = frozenset(
    {
        "no", "not", "never", "nothing", "none", "nobody", "nowhere", "neither", "nor",
        "without", "cannot", "hardly", "barely", "seldom", "rarely", "fewer", "less",
    }
)  # fmt: skip
#: Limiters: they bound only the words right after them ("only public sources").
LIMITERS = frozenset({"only", "just", "solely", "merely", "except"})
#: Words that make a phrase a REFUSED one (a ✗ row, a rejected panel may carry it).
NEGATING_CONTEXT = NEGATIONS | frozenset(
    {"avoiding", "avoid", "avoids", "excluding", "rejecting", "refusing", "instead", "rather"}
)
#: Verb particles a sentence may end on ("slips through", "signs off").
_PARTICLES = frozenset({"through", "off", "out", "up", "down", "over", "away", "back", "in", "on"})
#: Where a negation's scope ends: punctuation (on the word before) or one of these.
_SCOPE_BREAKS = frozenset({"while", "whereas", "but", "because", "although", "though", "yet"})
FUNCTION_WORDS = (
    DETERMINERS | PREPOSITIONS | CONJUNCTIONS | AUXILIARIES | SUBJECT_PRONOUNS | NEGATIONS
)

_START_FORBIDDEN = CONJUNCTIONS | (STRANDED_ADVERBS - {"only", "next", "first"})
_END_FORBIDDEN = (
    DETERMINERS
    | PREPOSITIONS
    | CONJUNCTIONS
    | AUXILIARIES
    | (SUBJECT_PRONOUNS - {"you", "it"})
    | (STRANDED_ADVERBS - {"next", "now", "again", "first"})
    | NEGATIONS
)

#: Nouns that end like a participle or an adjective.
_ED_NOUNS = frozenset({"need", "seed", "speed", "feed", "bed", "red", "shed", "greed", "embed"})
_ADJ_SUFFIXES = ("al", "ic", "ive", "ous", "ful", "able", "ible", "est")
_SUFFIX_NOUNS = frozenset(
    {
        "proposal", "approval", "signal", "portal", "capital", "journal", "deal", "goal",
        "total", "rival", "arrival", "renewal", "referral", "removal", "trial", "manual",
        "terminal", "material", "potential", "tutorial", "editorial", "interval", "festival",
        "topic", "logic", "music", "traffic", "metric", "clinic", "public", "graphic",
        "graphics", "republic", "mechanic", "relative", "objective", "initiative",
        "executive", "representative", "alternative", "incentive", "directive", "motive",
        "archive", "detective", "narrative", "native", "creative", "table", "cable", "label",
        "variable", "test", "interest", "request", "contest", "forest", "guest", "chest",
        "quest", "harvest", "manifest", "protest", "nest", "rest", "west", "crest", "digest",
        "vest", "best", "latest", "pest",
    }
)  # fmt: skip

#: Verbs (base forms) — inflected below. Only used to find where a subject ends
#: and that a phrase may stop before a verb; a noun that is also a verb here
#: ("route", "request") is harmless in both uses.
_VERB_BASES = (
    "run", "build", "create", "make", "earn", "request", "score", "rank", "find", "route",
    "handle", "use", "respect", "set", "keep", "send", "hand", "escalate", "pass", "go",
    "validate", "merge", "review", "approve", "publish", "write", "generate", "produce",
    "design", "decide", "own", "manage", "track", "measure", "test", "ship", "start", "get",
    "give", "take", "show", "help", "let", "deliver", "read", "learn", "pick", "choose",
    "surface", "spot", "identify", "prioritize", "prioritise", "automate", "connect", "sync",
    "scale", "grow", "sell", "buy", "book", "call", "reply", "answer", "invite", "close",
    "open", "launch", "plan", "schedule", "check", "flag", "block", "refuse", "reject",
    "accept", "protect", "store", "scrape", "avoid", "follow", "lead", "qualify", "nurture",
    "capture", "place", "convert", "need", "want", "work", "reach", "tailor", "rank",
    "say", "turn", "drive", "power", "focus", "fix", "map", "draft", "draw", "match", "move",
    "collect", "gather", "analyze", "analyse", "audit", "monitor", "watch", "notify",
    "alert", "forward", "transfer", "assign", "operate", "control", "limit", "enforce",
    "verify", "prove", "replace", "cut", "save", "earn", "win", "lose", "pay", "charge",
)  # fmt: skip
_IRREGULAR = frozenset(
    {
        "ran", "built", "made", "found", "kept", "sent", "went", "goes", "done", "got",
        "gave", "given", "took", "taken", "wrote", "written", "chose", "chosen", "sold",
        "bought", "led", "held", "shown", "drew", "drawn", "won", "lost", "paid", "said",
        "does", "says",
    }
)  # fmt: skip


def _inflections(base: str) -> set[str]:
    stem = base[:-1] if base.endswith("e") else base
    third = base + ("es" if base.endswith(("s", "sh", "ch", "x", "z", "o")) else "s")
    if base.endswith("y") and base[-2:-1] not in "aeiou":
        third = base[:-1] + "ies"
    past = base + "d" if base.endswith("e") else base + "ed"
    return {base, third, past, stem + "ing"}


VERBS = frozenset(word for base in _VERB_BASES for word in _inflections(base)) | _IRREGULAR


def bare(word: str) -> str:
    """``word`` lower-cased without punctuation ("Target's" → "targets", "files." → "files")."""
    return re.sub(r"[^\w-]", "", word.lower())


def tokens(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens; a spelled-out number is its digits.

    ``30+`` → ``30``, ``merqi.nl`` → two tokens, ``thirty`` → ``30`` — so a card
    that shows "30" for a line that says "thirty" shows the line's own number.
    """
    return [_NUMBER_WORDS.get(token, token) for token in _TOKEN.findall(text.lower())]


def with_figures(text: str) -> str:
    """``text`` with its numbers in digits and "over N" also written "N+".

    The form a card's figure is checked against: "for over thirty capabilities"
    → "for over 30+ capabilities" — the line's own claim, in a card's notation.
    """
    words = re.sub(
        r"[A-Za-z]+",
        lambda match: _NUMBER_WORDS.get(match.group(0).lower(), match.group(0)),
        text,
    )
    return _AT_LEAST.sub(lambda match: f"{match.group(0)}+", words)


def is_participle(word: str) -> bool:
    """A past participle by suffix ("tailored", "routed"), not a noun like "speed"."""
    word = bare(word)
    return len(word) > 4 and word.endswith("ed") and word not in _ED_NOUNS


def is_adjective(word: str) -> bool:
    """An adjective by suffix ("financial", "critical", "highest"), not a noun like "proposal"."""
    word = bare(word).split("-")[-1]
    return (
        len(word) > 4
        and word.endswith(_ADJ_SUFFIXES)
        and word not in _SUFFIX_NOUNS
        and word not in VERBS
    )


def is_verb(word: str) -> bool:
    return bare(word) in VERBS


def is_content(word: str) -> bool:
    """A word that can continue a noun phrase (not a function word, verb, -ly adverb)."""
    word_bare = bare(word)
    return bool(word_bare) and not (
        word_bare in FUNCTION_WORDS
        or word_bare in STRANDED_ADVERBS
        or word_bare in VERBS
        or word_bare.endswith("ly")
        or is_participle(word_bare)
    )


# ── what the line says ──────────────────────────────────────────────────────


_COORDINATION = re.compile(r"\b([\w-]+)\s+(and|or)\s+([\w'-]+)\s+([\w'-]+)", re.IGNORECASE)


def coordination_forms(text: str) -> list[str]:
    """The line with a shared-head coordination distributed.

    "critical financial and consent decisions" also says "critical financial
    decisions" and "critical consent decisions". Only modifier coordinations: the
    first conjunct is not a plural noun and the second not a verb (so "public
    sources and respects robots.txt" is left alone).
    """
    forms: list[str] = []
    for match in _COORDINATION.finditer(text):
        first, second, head = match.group(1), match.group(3), match.group(4)
        if any(is_negator(word) for word in text[: match.start()].split()):
            continue  # a stand-alone form would shed the negation
        if bare(first).endswith("s") and not bare(first).endswith("ss"):
            continue
        if (is_verb(second) and bare(second).endswith("s")) or bare(head) in FUNCTION_WORDS:
            continue
        if bare(second) in FUNCTION_WORDS or bare(first) in FUNCTION_WORDS:
            continue
        start, end = match.span()
        forms.append(f"{text[:start]}{first} {head}{text[end:]}")
        forms.append(f"{text[: match.start(1)]}{second} {head}{text[end:]}")
        # each conjunct with the head, as a phrase of its own ("financial decisions")
        forms.extend((f"{first} {head}", f"{second} {head}"))
    return forms


@dataclass(frozen=True)
class Source:
    """A text the card may quote from: its words, and each word's tokens."""

    words: tuple[str, ...]

    @classmethod
    def of(cls, text: str) -> Source:
        return cls(tuple(with_figures(text).split()))

    def stream(self) -> list[tuple[int, int, str]]:
        """``(word index, token index within the word, token)`` in reading order."""
        return [
            (index, position, token)
            for index, word in enumerate(self.words)
            for position, token in enumerate(tokens(word))
        ]


def sources_for(line_text: str, *extra: str) -> list[Source]:
    """The line, its distributed coordinations and any extra text (the brand name)."""
    derived = coordination_forms(line_text)
    return [Source.of(text) for text in (line_text, *derived, *extra)]


#: ``negated``: a phrase the card itself marks as refused (a checklist row ✗, a
#: rejected panel) — it may be the words a negation governs.
Slot = Literal["headline", "label", "negated"]


def is_negator(word: str) -> bool:
    raw = word.lower().replace("\u2019", "'")
    return bare(word) in NEGATIONS or raw.rstrip(",;:.!?").endswith("n't")


def governing_negation(words: Sequence[str], first_word: int) -> str | None:
    """The negator or limiter that governs a span starting at ``first_word``.

    Frame audit of PR #1080: a span cut just after a negator says the opposite
    of the line — "No campaign goes live without your approval" → "Campaign goes
    live without your approval". Any negator earlier in the same clause (back to
    punctuation or a "while"/"but"…) governs the span; a limiter ("only") governs
    the words right after it. ``None`` when nothing does.
    """
    for index in range(first_word - 1, -1, -1):
        word = words[index]
        if re.search(r"[,;:.!?]$", word) or bare(word) in _SCOPE_BREAKS:
            return None
        if is_negator(word):
            return bare(word) or word
        if bare(word) in LIMITERS and index == first_word - 1:
            return bare(word)
    return None


#: Words that open a subordinate condition clause. A headline never cuts inside
#: one (audit of b86a1b23 — no verb detection: the clause runs to a boundary).
CONDITION_WORDS = frozenset(
    {"until", "unless", "without", "if", "when", "whenever", "before", "after", "except"}
)


def _clause_boundary_after(words: Sequence[str], index: int) -> bool:
    """A clause ends after ``words[index]``: punctuation, a dash, or a coordinating
    and/but/or that opens a new clause (followed by a subject pronoun). Used for
    a host clause and a LEADING condition; a trailing condition ends only at a
    hard boundary (:func:`_hard_boundary_after`)."""
    if index + 1 >= len(words) or re.search(r"[,;:.!?]$", words[index]):
        return True
    following = words[index + 1]
    if following in {"-", "\u2013", "\u2014"}:
        return True
    return (
        bare(following) in {"and", "but", "or"}
        and index + 2 < len(words)
        and bare(words[index + 2]) in SUBJECT_PRONOUNS
    )


def _non_restrictive(words: Sequence[str], index: int) -> bool:
    """``words[index]`` opens a non-restrictive relative clause (", which …")."""
    return (
        index > 0
        and bare(words[index]) in {"which", "who", "whom", "whose"}
        and bool(re.search(r",$", words[index - 1]))
    )


def condition_clauses(words: Sequence[str]) -> list[tuple[int, int]]:
    """``(first, last)`` word index of every condition clause: from its condition
    word to the next clause boundary."""
    out: list[tuple[int, int]] = []
    for start, word in enumerate(words):
        if _non_restrictive(words, start):
            # ", which a human approves before anything ships": the safeguard the
            # line rests on — kept whole, like a condition (audit of d6f223d9)
            end = next(i for i in range(start, len(words)) if _hard_boundary_after(words, i))
            out.append((start, end))
            continue
        if bare(word) not in CONDITION_WORDS:
            continue
        leading = (
            start == 0
            or bool(re.search(r"[;:.!?]$", words[start - 1]))
            or (words[start - 1] in _DASHES)
        )
        if leading:
            # "Unless you approve it, | nothing is sent": a leading condition ends
            # at its comma — the main clause follows it.
            end = next(i for i in range(start, len(words)) if _clause_boundary_after(words, i))
        else:
            # A trailing condition runs to the sentence end, ; : or a dash — never
            # to a comma: "until the reviewer who owns the account, and the legal
            # team, have both signed off" is ONE condition (audit of 603186d3).
            end = next(i for i in range(start, len(words)) if _hard_boundary_after(words, i))
        out.append((start, end))
    return out


_DASHES = frozenset({"-", "\u2013", "\u2014"})


def _hard_boundary_after(words: Sequence[str], index: int) -> bool:
    """A sentence end, ; : or a dash after ``words[index]`` (never a comma)."""
    if index + 1 >= len(words) or re.search(r"[;:.!?]$", words[index]):
        return True
    return words[index + 1] in _DASHES


def _host_range(words: Sequence[str], clause: tuple[int, int]) -> tuple[int, int]:
    """The clause a condition clause modifies: the one after it when it leads
    ("Unless you approve it, | nothing is sent"), else the one before it."""
    start, end = clause
    if re.search(r"[,;:]$", words[end]) and end + 1 < len(words):
        stop = next(i for i in range(end + 1, len(words)) if _clause_boundary_after(words, i))
        return end + 1, stop
    # The clause before it — a comma that sets the condition off is not a
    # boundary of its host ("…respects rate limits, | without stored cookies…").
    begin = max((i + 1 for i in range(start - 1) if _clause_boundary_after(words, i)), default=0)
    return begin, start - 1


def claim_guarded(words: Sequence[str]) -> bool:
    """A line whose claim a cut condition would invert: it negates or limits
    ("only when"). Nothing may be cut INSIDE its condition (phrase level)."""
    return any(is_negator(word) or bare(word) == "only" for word in words)


def condition_guarded(words: Sequence[str]) -> bool:
    """A line whose card must carry its condition with its main clause: a guarded
    claim, or any condition / non-restrictive relative clause ("Merqi books a
    meeting WHEN the prospect has asked for one" — the condition is the consent;
    "…, which a human approves…" — the safeguard; audit of d6f223d9)."""
    return claim_guarded(words) or bool(condition_clauses(words))


def condition_card_problem(line_text: str, screen: Sequence[str]) -> str | None:
    """A card that shows a condition without its main clause, or (on a line whose
    claim hangs on its condition) the main clause without the condition.

    "Merqi books a meeting only when the prospect has asked for one" → "The
    prospect has asked for one" has lost what the line is about; "No campaign
    goes live" has dropped the exception the claim rests on.
    """
    words = line_text.strip().split()
    shown = {token for text in screen for token in tokens(text)}

    def content(lo: int, hi: int) -> set[str]:
        """Tokens of the clause's words that say something (verbs included)."""
        return {
            token
            for i in range(lo, hi + 1)
            if bare(words[i]) not in FUNCTION_WORDS | STRANDED_ADVERBS
            and bare(words[i]) not in CONDITION_WORDS
            for token in tokens(words[i])
        }

    for clause in condition_clauses(words):
        host = _host_range(words, clause)
        cond, main = shown & content(*clause), shown & content(*host)
        if cond and not main:
            return f"shows the condition {bare(words[clause[0]])!r}… without its main clause"
        if main and not cond and condition_guarded(words):
            return f"drops the condition {bare(words[clause[0]])!r}… the line puts on it"
    return None


def condition_problem(words: Sequence[str], first: int, last: int) -> str | None:
    """A span that cuts inside a condition clause (on a line whose claim hangs on it).

    "Nothing ships until the customer | is ready" says more than the line. That
    the CARD also carries the whole condition when it shows its main clause is
    checked over the whole card (:func:`condition_card_problem`): a checklist
    may carry it on other rows.
    """
    if not condition_guarded(words):
        return None
    for start, end in condition_clauses(words):
        overlaps = first <= end and last >= start
        whole = first <= start and last >= end
        # the clause without only its opener ("before ever | requesting contact")
        body = (
            first > start
            and last >= end
            and not any(is_content(words[i]) for i in range(start, first))
        )
        if overlaps and not whole and not body:
            return f"cuts inside the condition clause {bare(words[start])!r}…"
    return None


def _cuts_negation_scope(words: Sequence[str], first: int, last: int) -> str | None:
    """A span that carries a negator must run to the end of its clause.

    "No campaign goes live" is not "No campaign goes live without your
    approval", and "Nothing ships" is not "Nothing ships until a human signs
    off": the condition is part of what is negated.
    """
    negators = [i for i in range(first, last + 1) if is_negator(words[i])]
    if not negators or re.search(r"[,;:.!?]$", words[last]) or last + 1 >= len(words):
        return None
    following = bare(words[last + 1])
    if following in _SCOPE_BREAKS:
        return None
    # "never A or B" says "never A": cutting a disjunct keeps the claim true.
    if following in {"or", "nor"}:
        return None
    negators = [words[i] for i in negators]
    return (
        f"cuts the scope of {bare(negators[0])!r} short (the line goes on: "
        f"{bare(words[last + 1])!r}) — the card would claim more than the line"
    )


def _boundary_problem(words: Sequence[str], first: int, last: int, size: int) -> str | None:
    """Why the span ``words[first..last]`` is cut inside a phrase, or ``None``."""
    word = words[last]
    if last + 1 < len(words) and not re.search(r"[,;:.!?)]$", word):
        following = words[last + 1]
        if bare(following) == "of" or (is_content(following) and bare(word) not in FUNCTION_WORDS):
            return f"stops inside a phrase (the line goes on: {bare(following)!r})"
    if first > 0 and size > 1:
        before = words[first - 1]
        if (
            not re.search(r"[,;:.!?(]$|'s$|s'$", before)
            and is_content(before)
            and not re.search(r"\d", before)  # "30+ | capabilities": a figure, then its noun
            and is_content(words[first])
        ):
            return f"starts inside a phrase (the line says {bare(before)!r} before it)"
    if bare(words[first]) == "not" and first > 0 and not re.search(r"[,;:.!?]$", words[first - 1]):
        # "Not send email…" cut from "does not send email…" ("X, not Y" is fine)
        return "opens inside a verb group ('not' without its verb)"
    return None


def negating_context(words: Sequence[str], first_word: int) -> str | None:
    """The word that makes the span starting at ``first_word`` a refused one, in its clause."""
    for index in range(first_word - 1, -1, -1):
        word = words[index]
        if bare(word) in NEGATING_CONTEXT or is_negator(word):
            return bare(word)
        if re.search(r"[,;:.!?]$", word) or bare(word) in _SCOPE_BREAKS:
            return None
    return None


def _span_problem(phrase_tokens: list[str], source: Source, *, negated: bool = False) -> str | None:
    """``None`` when the phrase is a clean span of ``source``, else why not.

    ``negated``: the card marks the phrase as refused, so it may sit under a negation."""
    stream = source.stream()
    size = len(phrase_tokens)
    reason = "drops words from inside the line (it is not a run of the line's words)"
    for start in range(len(stream) - size + 1):
        if [token for _w, _p, token in stream[start : start + size]] != phrase_tokens:
            continue
        first_word, first_pos, _ = stream[start]
        last_word, last_pos, _ = stream[start + size - 1]
        if first_pos != 0 or last_pos != len(tokens(source.words[last_word])) - 1:
            reason = "cuts a word in half"
            continue
        problem = _boundary_problem(source.words, first_word, last_word, size)
        if problem is None and negated and negating_context(source.words, first_word) is None:
            # a ✗ row / rejected panel may only carry what the line itself refuses
            problem = "is marked refused but the line does not negate it"
        negator = None if negated else governing_negation(source.words, first_word)
        if problem is None and negator is not None:
            problem = f"drops the {negator!r} that governs it — the card would say the opposite"
        if problem is None and not negated:  # a refused row names what the line refuses
            problem = condition_problem(source.words, first_word, last_word)
        if problem is None:
            problem = _cuts_negation_scope(source.words, first_word, last_word)
        if problem is not None:
            reason = problem
            continue
        return None
    return reason


#: Quantifier determiners: a legitimate subject opener before a noun ("Most
#: leads go cold within a week", "Few teams have time to…", "No campaign…").
QUANTIFIERS = frozenset(
    {"most", "many", "some", "all", "few", "every", "each", "no", "none", "more"}
)


def _quantified_subject(words: Sequence[str]) -> bool:
    """``words`` opens on a quantifier followed by its noun (or "of")."""
    if len(words) < 2 or bare(words[0]) not in QUANTIFIERS:
        return False
    following = bare(words[1])
    return following == "of" or following not in FUNCTION_WORDS | STRANDED_ADVERBS


def phrase_problem(
    phrase: str, sources: Sequence[Source], *, slot: Slot = "headline"
) -> str | None:
    """Why ``phrase`` would read as broken copy on a card; ``None`` when it reads as a phrase.

    See the module docstring for the four rules. An empty phrase is not shown,
    so it has no problem.
    """
    words = [word for word in phrase.split() if bare(word)]
    phrase_tokens = tokens(phrase)
    if not words or not phrase_tokens:
        return None
    negated = slot == "negated"
    reasons = [_span_problem(phrase_tokens, source, negated=negated) for source in sources]
    if all(reasons):
        return f"{phrase!r} {reasons[0]}"
    first, last = bare(words[0]), bare(words[-1])
    if all(bare(word) in FUNCTION_WORDS for word in words):
        return f"{phrase!r} has no content word"
    if first in _START_FORBIDDEN and not _quantified_subject(words):
        return f"{phrase!r} opens on a stranded {first!r}"
    particle = last in _PARTICLES and len(words) > 1 and bare(words[-2]) not in FUNCTION_WORDS
    if last in _END_FORBIDDEN and not particle:
        return f"{phrase!r} ends on a dangling {last!r}"
    if slot == "headline" and len(words) > 1 and (is_participle(last) or is_adjective(last)):
        return f"{phrase!r} ends on a dangling {last!r}"
    if slot == "headline" and len(words) == 1 and (is_adjective(last) or is_participle(last)):
        return f"{phrase!r} is a lone modifier"
    before_last = words[-2] if len(words) > 1 else ""
    if (
        last.endswith("ing")
        and before_last
        and not is_content(before_last)
        and not is_verb(before_last)
        and bare(before_last) not in DETERMINERS
    ):
        # "…automatically routing", "…to merging": a participle left hanging
        return f"{phrase!r} ends on a dangling {last!r}"
    return None


def problems(phrases: Iterable[tuple[str, Slot]], sources: Sequence[Source]) -> list[str]:
    """Every problem of ``phrases`` (``(text, slot)``), in order."""
    return [
        problem
        for text, slot in phrases
        if (problem := phrase_problem(text, sources, slot=slot)) is not None
    ]


# ── building phrases that pass ──────────────────────────────────────────────


def predicate(words: Sequence[str]) -> list[str]:
    """``words`` without their subject: from the first verb on ("The system earns
    attention" → "earns attention", "It uses only…" → "uses only…").

    Unchanged when no verb is found after a subject.
    """
    if words and bare(words[0]) in SUBJECT_PRONOUNS and len(words) > 1:
        return list(words[1:])
    for index, word in enumerate(words):
        if index and is_verb(word) and not is_participle(word):
            return list(words[index:])
    return list(words)


def subject(words: Sequence[str]) -> list[str]:
    """The words before the first verb ("Deterministic code handles…" → "Deterministic code")."""
    for index, word in enumerate(words):
        if index and is_verb(word) and not is_participle(word):
            return list(words[:index])
    return []


def strip_stranded(words: Sequence[str]) -> list[str]:
    """``words`` without leading conjunctions / stranded adverbs ("ever requesting" → "requesting")."""
    out = list(words)
    while out and bare(out[0]) in _START_FORBIDDEN:
        out = out[1:]
    return out


def strip_determiner(words: Sequence[str]) -> list[str]:
    """``words`` without a leading determiner ("your marketing surfaces" → "marketing surfaces")."""
    return list(words[1:]) if len(words) > 1 and bare(words[0]) in DETERMINERS else list(words)


def trim_trailing(words: Sequence[str]) -> list[str]:
    """``words`` without a trailing dangling word (a stranded adverb, preposition…)."""
    out = list(words)
    while out and (bare(out[-1]) in _END_FORBIDDEN or bare(out[-1]) in STRANDED_ADVERBS):
        out = out[:-1]
    return out


def clean_spans(text: str, *, max_words: int) -> list[list[str]]:
    """Every span of ``text`` of at most ``max_words`` words that reads as a phrase.

    Best first: a span that ends at a clause boundary, then the one with the
    most content words, then the earliest.
    """
    sources = sources_for(text)
    found: list[tuple[tuple[int, int, int], list[str]]] = []
    words = text.strip().split()
    for start in range(len(words)):
        for end in range(start + 1, min(len(words), start + max_words) + 1):
            span = words[start:end]
            phrase = " ".join(span).strip(" ,;:.!?")
            if phrase_problem(phrase, sources) is not None:
                continue
            boundary = end == len(words) or bool(re.search(r"[,;:.!?]$", words[end - 1]))
            content = sum(1 for word in span if is_content(word))
            found.append(((0 if boundary else 1, -content, start), span))
    return [span for _key, span in sorted(found, key=lambda pair: pair[0])]


# ── multi-element cards: every element at a legal start, in the line's order ─

#: Subordinators that open a clause an element may start at — the element must
#: then run to that clause's end ("…motion | while you keep the final say").
SUBORDINATORS = frozenset({"while", "whereas", "because", "although", "though", "since"})
#: Words that open a relative clause: never a step of its own.
RELATIVES = frozenset({"which", "who", "whom", "whose", "that"})


def legal_starts(words: Sequence[str]) -> set[int]:
    """Word indexes an element of a multi-element card may start at.

    The start of the line; right after a clause boundary (, ; : or a dash);
    right after a coordinating and/but/or; a condition word, or right after
    one (the element must then run to the condition's end — see
    :func:`element_order_problem`). Never at a relative pronoun (audit of
    d6f223d9: "…the copy, which a human approves…" → "Approves").
    """
    starts = {0}
    for i in range(1, len(words)):
        before = words[i - 1]
        if (
            re.search(r"[,;:]$", before)
            or before in _DASHES
            or bare(before) in {"and", "but", "or"}
        ):
            starts.add(i)
        if bare(words[i]) in CONDITION_WORDS | SUBORDINATORS:
            starts.update({i, i + 1})
    return {i for i in starts if i < len(words) and bare(words[i]) not in RELATIVES}


def locate(words: Sequence[str], element: str) -> list[tuple[int, int]]:
    """Every ``(first, last)`` word span of ``words`` whose tokens are ``element``'s."""
    return _locate(words, element, 0)


def _locate(words: Sequence[str], element: str, after: int) -> list[tuple[int, int]]:
    """``(first, last)`` word spans of ``words`` (from ``after`` on) whose tokens are ``element``'s."""
    wanted = tokens(element)
    out: list[tuple[int, int]] = []
    for first in range(after, len(words)):
        got: list[str] = []
        for last in range(first, len(words)):
            got += tokens(words[last])
            if got == wanted:
                out.append((first, last))
            if len(got) >= len(wanted):
                break
    return out


Role = Literal["node", "connector"]


def starts_with_verb(text: str) -> bool:
    """A node may not open on a verb or an auxiliary ("Approves", "Has asked…").
    A gerund ("Requesting contact") is a noun phrase and may."""
    first = bare(text.split()[0]) if text.split() else ""
    if first in AUXILIARIES:
        return True
    return first in VERBS and not first.endswith("ing")


def _opens_on_verb(words: Sequence[str], first: int) -> bool:
    """The span at ``first`` opens on a verb or auxiliary — unless the line puts a
    determiner before it, which makes it a noun ("every | draft")."""
    if first > 0 and bare(words[first - 1]) in DETERMINERS:
        return False
    return starts_with_verb(words[first])


def element_order_problem(line_text: str, elements: Sequence[tuple[str, Role]]) -> str | None:
    """Why the elements of a multi-element card misread the line, or ``None``.

    Every element is a run of the line, and the elements follow the line's
    order (element i+1 starts after element i ends) — a reversed order reverses
    the claim. Roles (audit of d6f223d9, option 2):

    * a NODE (a circle, a panel, a step, a headline) is a noun phrase of the
      line — never opening on a verb or auxiliary ("Approves", "Has asked for
      one" lose their subject) — OR a whole clause/series item starting at a
      legal start (:func:`legal_starts`: "Merqi drafts, | reviews, | and
      schedules posts"); one that starts at or right after a condition word
      runs to the condition's end;
    * a CONNECTOR (the label on an arrow) may be the line's verb phrase
      ("Tailored", "Route automatically").

    No element starts at a relative pronoun.
    """
    words = line_text.strip().split()
    starts = legal_starts(words)
    clauses = {start: end for start, end in condition_clauses(words)}
    for start, word in enumerate(words):
        if bare(word) in SUBORDINATORS:
            clauses[start] = next(
                i for i in range(start, len(words)) if _hard_boundary_after(words, i)
            )
    cursor = 0
    for element, role in elements:
        if not tokens(element):
            continue
        spans = _locate(words, element, cursor)
        if not spans:
            return f"{element!r} is not in the line's order — the card would reorder the claim"
        legal = [
            (first, last)
            for first, last in spans
            # a relative clause is never a step of its own — except a whole
            # non-restrictive one as the sub-line it was split into
            if (bare(words[first]) not in RELATIVES or _non_restrictive(words, first))
            and _condition_whole(words, clauses, first, last)
            and (role == "connector" or first in starts or not _opens_on_verb(words, first))
        ]
        if not legal:
            return (
                f"{element!r} opens on a verb or inside a clause — a {role} must be a noun "
                "phrase or a whole clause of the line"
            )
        cursor = legal[0][1] + 1
    return None


def _condition_whole(words: Sequence[str], clauses: dict[int, int], first: int, last: int) -> bool:
    for start in (first, first - 1):
        opener = bare(words[start]) in CONDITION_WORDS | SUBORDINATORS or (
            _non_restrictive(words, start)
        )
        if start in clauses and opener:
            return last >= clauses[start]
    return True


# ── a single headline starts at a subject (audit of the first live render) ──

#: Pronouns that are an object, never the start of a headline ("Them where…").
OBJECT_PRONOUNS = frozenset({"them", "it", "us", "him", "her", "me", "you", "itself", "themselves"})


def clause_starts(words: Sequence[str]) -> set[int]:
    """Where a clause starts: the line's start, or right after , ; : or a dash."""
    starts = {0}
    for i in range(1, len(words)):
        before = words[i - 1]
        comma = before.endswith(",") and not modifier_list(before, words[i])
        if comma or re.search(r"[;:]$", before) or before in _DASHES:
            starts.add(i)
    return {i for i in starts if i < len(words)}


def modifier_list(before: str, word: str) -> bool:
    """ "graduated, governed autonomy": a comma between two modifiers of one noun
    phrase does not start a clause (render 01a1179d)."""
    return (is_adjective(before) or is_participle(before)) and (
        is_adjective(word) or is_participle(word) or is_content(word)
    )


def _has_verb(words: Sequence[str]) -> bool:
    return any(
        bare(word) in AUXILIARIES or (is_verb(word) and not is_participle(word)) for word in words
    )


def clause_end(words: Sequence[str], first: int) -> int:
    """The last word of the clause starting at ``first``: up to the sentence end,
    ; : or a dash, or the word before an and/but/or that opens a clause with its
    own subject. A comma does not end it, nor does a coordinated predicate ("is
    single use AND expires in minutes" is one clause)."""
    for i in range(first, len(words)):
        if re.search(r"[;:.!?]$", words[i]) or i + 1 >= len(words):
            return i
        following = words[i + 1]
        if following in _DASHES:
            return i
        if (
            bare(following) in {"and", "but", "or", "nor"}
            and i + 2 < len(words)
            and bare(words[i + 2]) in DETERMINERS | SUBJECT_PRONOUNS
        ):
            return i
    return len(words) - 1


def _stops_mid_clause(words: Sequence[str], first: int, last: int) -> str | None:
    """A clause headline that stops before its clause ends (audit of #1084: "Every
    approval is single" out of "…is single use and expires in minutes"). A
    headline without a verb is the clause's subject noun phrase and may stop
    where the noun phrase does."""
    if not _has_verb(words[first : last + 1]):
        return None
    end = clause_end(words, first)
    if last >= end:
        return None
    return f"stops mid-clause (the clause runs on to {bare(words[end])!r})"


_ARTICLES_DROPPABLE = frozenset({"the", "a", "an"})


def _dropped_modifier(words: Sequence[str], first: int) -> str | None:
    """A noun-phrase headline keeps its modifiers: the word before it may be an
    article, never a possessive / quantifier ("your | whole revenue motion") nor
    a modifier set off by a comma ("graduated, | governed autonomy")."""
    if first == 0:
        return None
    before = words[first - 1]
    word = bare(before)
    if word in DETERMINERS and word not in _ARTICLES_DROPPABLE | {"only"}:
        return word
    if before.endswith(",") and modifier_list(before, words[first]):
        return word
    return None


def _noun_phrase_problem(words: Sequence[str], first: int, starts: set[int]) -> str | None:
    """A mid-clause headline as the line's subject/object noun phrase: whole (its
    own modifiers kept) and following the line's verb — or why not."""
    dropped = _dropped_modifier(words, first)
    if dropped:
        return f"drops its own modifier {dropped!r} (render 01a1179d)"
    before = first - 1
    while before >= 0 and bare(words[before]) in DETERMINERS | {"only"}:
        before -= 1  # the noun phrase's own determiners
    if before < 0 or before + 1 in starts or is_verb(words[before]):
        return None
    return f"is not the line's subject or object (it follows {bare(words[before])!r})"


def headline_start_problem(line_text: str, headline: str) -> str | None:
    """Why ``headline`` does not start where a headline may, or ``None``.

    Run 01a11693 (first live per-line render) showed "Them where targets already
    pay attention" and "Under a single governed engine" as headlines: a span
    that starts with an object pronoun or a preposition, mid-clause. A headline
    is either a WHOLE clause from its start (the line's start, or after , ; :
    —) to its end (:func:`clause_end`), or the subject/object noun phrase of the
    line: it opens on neither a pronoun, a preposition, a conjunction, an adverb
    nor a verb, and (past its own determiners) follows the line's verb.
    """
    words = line_text.strip().split()
    spans = _locate(words, headline, 0)
    if not spans:
        return None  # not a run of the line: the phrase gate reports it
    starts = clause_starts(words)
    reasons: list[str] = []
    for first, last in spans:
        if first in starts:
            cut = _stops_mid_clause(words, first, last)
            if cut is None:
                return None
            reasons.append(cut)
            continue
        opener = bare(words[first])
        if opener in OBJECT_PRONOUNS | SUBJECT_PRONOUNS:
            reasons.append(f"opens on the pronoun {opener!r}")
            continue
        # ("only" may lead its noun phrase: the limiter stays on the card)
        if (
            opener in PREPOSITIONS
            or opener in CONJUNCTIONS
            or opener in STRANDED_ADVERBS - {"only"}
        ):
            reasons.append(f"opens on {opener!r} mid-clause")
            continue
        # (a participle may lead a noun phrase: "tailored multimedia assets")
        verb_led = _opens_on_verb(words, first) and not is_participle(words[first])
        if (opener.endswith("ly") and opener != "only") or verb_led:
            reasons.append(f"opens on {opener!r} mid-clause")
            continue
        np_problem = _noun_phrase_problem(words, first, starts)
        if np_problem is None:
            # past its article it may still be a clause ("Platform scores every
            # landing page" out of "The platform scores…"): then it runs to its end
            cut = _stops_mid_clause(words, first, last)
            if cut is None:
                return None
            np_problem = cut
        reasons.append(np_problem)
    return f"headline {headline!r} " + (reasons[0] if reasons else "starts mid-clause")
