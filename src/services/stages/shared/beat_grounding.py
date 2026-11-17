"""Grounding a rewritten narration beat in what its recording actually shows.

Shared by ``demo_video_reconcile_timeline``'s beat-expander and ``script_doctor``'s
revise step: both ask a model to REWRITE a beat, and on showcase run 01a156a6
(2026-10-20) both rewrites invented content the clip never shows ("refactor the entire
component tree", "break the demo in three ways") over a TodoMVC recording that adds
todos. The beat's scenario name + gherkin steps are what the recording performs; a
rewrite that talks about things neither the beat nor those steps mention is refused.
"""

from __future__ import annotations

import re
from collections.abc import Mapping


def beat_source(line: Mapping[str, object]) -> str | None:
    """What a beat's recording actually PERFORMS: its scenario name + gherkin steps.

    The bdd-generated spine carries both on every line (``scenario_name`` /
    ``gherkin``, see ``demo_video_gen``). ``None`` when neither is present — an
    authored beat with no scenario has no recorded script to ground against, and a
    fabricated one would be worse than none.
    """
    pieces = [str(line.get(key) or "").strip() for key in ("scenario_name", "gherkin")]
    joined = "\n".join(p for p in pieces if p)
    return joined or None


#: Above this many content words absent from BOTH the beat's text and its
#: :func:`beat_source`, a lengthened beat is refused as UNGROUNDED. Calibrated on the
#: live model (2026-10-20 replay, run 01a156a6): grounded replies introduce 0-3 such
#: words ("entry", "list", "confirm"); the invented beats this run shipped introduce
#: 11-12 ("refactor", "component", "re-renders", ...) and the ungrounded-prompt replies
#: 6 ("demonstrate", "handles", "accordingly", ...).
MAX_UNGROUNDED_WORDS = 4

#: Function / narration-scaffold words (4+ letters; shorter never count) that carry no
#: claim — never counted as novel.
_GROUNDING_STOPWORDS = frozenset(
    {
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "there",
        "here",
        "your",
        "with",
        "from",
        "into",
        "onto",
        "were",
        "been",
        "being",
        "will",
        "would",
        "could",
        "should",
        "might",
        "does",
        "done",
        "have",
        "next",
        "first",
        "finally",
        "also",
        "just",
        "very",
        "more",
        "most",
        "each",
        "every",
        "some",
        "three",
        "which",
        "what",
        "when",
        "where",
        "lets",
        "start",
        "begin",
        "show",
        "watch",
        "look",
        "take",
        "make",
        "again",
        "once",
        "while",
        "after",
        "before",
        "about",
        "over",
        "under",
        "down",
        "back",
        "time",
        "here's",
        "let's",
    }
)

_WORD_RE = re.compile(r"[a-z][a-z'-]+")


def _grounding_stem(word: str) -> str:
    """A crude stem so "type"/"typing", "open"/"opening", "click"/"clicked" match."""
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if word.endswith(suffix) and not word.endswith("ss") and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    return word.rstrip("e")[:5]


def ungrounded_words(new_text: str, original: str, source: str) -> list[str]:
    """Content words the re-ask INTRODUCED that neither the beat nor its recording has.

    A deliberately simple lexical check — it cannot prove a reply is true, only that it
    talks about things the beat and its scripted steps never mention, which is the
    exact failure seen live ("isolate the state management layer" over a clip that adds
    a todo). Words of fewer than 4 letters and :data:`_GROUNDING_STOPWORDS` never count.
    """
    known = {_grounding_stem(w) for w in _WORD_RE.findall(f"{original} {source}".lower())}
    return sorted(
        {
            w
            for w in _WORD_RE.findall(new_text.lower())
            if len(w) >= 4 and w not in _GROUNDING_STOPWORDS and _grounding_stem(w) not in known
        }
    )


_STEP_KEYWORD_RE = re.compile(r"^\s*(given|when|then|and|but)\s+", re.IGNORECASE)
_STEP_URL_RE = re.compile(r"\s*(?:at\s+)?\"?(?:https?://|www\.)[^\s\"]+\"?", re.IGNORECASE)
_QUOTED_RE = re.compile(r"\"[^\"]*\"")
_STEP_FIRST_PERSON_RE = re.compile(r"^i\s+(?:am\s+)?", re.IGNORECASE)

#: A recorded step counts as NARRATED once at least this share of its content words
#: appear (by :func:`_grounding_stem`) in the beat text.
_STEP_COVERED_SHARE = 0.5


def _content_stems(text: str) -> set[str]:
    return {
        _grounding_stem(w)
        for w in _WORD_RE.findall(text.lower())
        if len(w) >= 4 and w not in _GROUNDING_STOPWORDS
    }


def uncovered_steps(text: str, gherkin: str | None) -> list[str]:
    """The recorded ACTION steps the beat's narration does not mention yet, in order.

    Showcase run 01a158a5-e826 (2026-10-20): beat clip-4 recorded "Add a todo and mark
    it completed" but its narration only said "add a second task". The beat-expander
    was told to ELABORATE on what the beat already says, so it padded ("the list of
    items currently displayed") and the grounding check refused 5 of 9 replies; the
    completion step was never narrated. These steps are the grounded material a
    too-short beat should be lengthened WITH.

    Only the steps from the scenario's first When/Then on count: the leading Given/And
    block is the setup every bdd scenario shares ("open the demo", "see the heading"),
    and narrating it on every beat is the "Let's start by navigating to the todos
    page" repetition. Each step loses its keyword, any URL (the voice would spell it)
    and a leading first-person "I"/"I am". A step is covered when at least half its
    content words outside quotes already appear in ``text`` (quoted item names like
    "Send email" recur across steps, so they cannot prove the ACTION was narrated) AND,
    when it quotes something, a word of the quote appears too (so "see the link
    'Remo H. Jansen'" is not covered by a beat that names a different link); a repeat of an earlier step is skipped. ``[]`` when ``gherkin`` is empty.
    """
    if not gherkin:
        return []
    steps = [raw.strip() for raw in gherkin.splitlines() if _STEP_KEYWORD_RE.match(raw)]
    first_action = next(
        (i for i, step in enumerate(steps) if re.match(r"(when|then)\b", step, re.IGNORECASE)),
        0,
    )
    known = _content_stems(text)
    out: list[str] = []
    seen: set[str] = set()
    for raw in steps[first_action:]:
        step = _STEP_KEYWORD_RE.sub("", raw).strip()
        step = _STEP_URL_RE.sub("", step).strip()
        step = _STEP_FIRST_PERSON_RE.sub("", step).strip()
        stems = _content_stems(_QUOTED_RE.sub(" ", step))
        quoted = _content_stems(" ".join(_QUOTED_RE.findall(step)))
        key = step.lower()
        if not (stems or quoted) or key in seen:
            continue
        seen.add(key)
        action_covered = not stems or len(stems & known) / len(stems) >= _STEP_COVERED_SHARE
        object_covered = not quoted or bool(quoted & known)
        if not (action_covered and object_covered):
            out.append(step)
    return out
