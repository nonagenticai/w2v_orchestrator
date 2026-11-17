"""Burned-in caption cues of promo render 01a1179d (2026-10-07, REJECTED at frame QA).

"…back to the exact" / "rule that allowed it.": a plain break cut a noun phrase
in two. A break right before a preposition ("…decision back | to the exact rule
that allowed it.") keeps the phrase it opens whole and now beats a plain break.
The line is the run's ``promo_script`` line verbatim; the break choice depends
only on the words and the 2 x 42-character budget, so the tokens carry even
0.3 s timings.
"""

from __future__ import annotations

from src.services.caption_cues import group_tokens_into_cues
from src.services.caption_reconcile import TimedToken

_TRACE = (
    "You can freeze anything in flight and trace every decision back to the exact rule that "
    "allowed it."
)


def _cues(text: str) -> list[str]:
    tokens = [
        TimedToken(text=word, start=i * 0.3, end=i * 0.3 + 0.25, source="word")
        for i, word in enumerate(text.split())
    ]
    return [" ".join(cue.lines) for cue in group_tokens_into_cues(tokens)]


def test_a_noun_phrase_is_never_cut_by_a_cue_break() -> None:
    assert _cues(_TRACE) == [
        "You can freeze anything in flight and trace every decision back",
        "to the exact rule that allowed it.",
    ]
