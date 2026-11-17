"""Burned-in caption cues of promo run 01a11693 (2026-10-07, first live per-line render).

Frame QA: "…and places them where targets" / "already pay attention." — the cue
packer broke the where-clause between its subject ("targets") and its verb
("already pay"). Both breaks it weighed ("…assets | and places…" and "…where
targets | already…") were plain breaks of equal cost. A break right before a
clause opener now beats a plain break, and a break a few words after an opener
(inside that clause's subject) is a defect accepted only when nothing else fits.

The script lines are the run's ``promo_script`` lines verbatim. The break choice
depends only on the words and the 2 x 42-character budget, not on the timings,
so the tokens carry even 0.3 s timings.
"""

from __future__ import annotations

import pytest

from src.services.caption_cues import group_tokens_into_cues
from src.services.caption_reconcile import TimedToken

_WHERE = (
    "It generates tailored multimedia assets and places them where targets already pay attention."
)


def _cues(text: str) -> list[str]:
    tokens = [
        TimedToken(text=word, start=i * 0.3, end=i * 0.3 + 0.25, source="word")
        for i, word in enumerate(text.split())
    ]
    return [" ".join(cue.lines) for cue in group_tokens_into_cues(tokens)]


def test_a_where_clause_is_never_cut_between_its_subject_and_its_verb() -> None:
    cues = _cues(_WHERE)

    assert cues == [
        "It generates tailored multimedia assets",
        "and places them where targets already pay attention.",
    ]
    assert not any(cue.endswith("where targets") for cue in cues)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Merqi executes your entire revenue motion under a single governed engine.",
         ["Merqi executes your entire revenue motion under a single governed engine."])],
)  # fmt: skip
def test_the_run_s_other_lines(text: str, expected: list[str]) -> None:
    assert _cues(text) == expected
