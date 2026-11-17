"""Placeholder text must never reach a customer's video.

A missing optional field interpolated with ``f"{d.get('x')}"`` renders the
string ``"None"``. In a prompt the LLM obeys it: a promo whose caller sent no
``call_to_action`` was told ``End with this CTA verbatim: "None"`` and closed
the video on a spoken "None". The same family leaked a hardcoded house brand
("Common Practice") as the slide eyebrow when the brand had no name.

This module is the one vocabulary for "this text is a placeholder, not
content", used at three layers:

- prompt building (:func:`text_or_empty`) — never interpolate an absent value;
- script parsing (:func:`is_placeholder_text`) — drop a line the model emitted
  anyway;
- review (:func:`is_placeholder_text`) — a shipped narration line that is a
  placeholder fails the run instead of scoring it 2/5.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "PLACEHOLDER_TOKENS",
    "is_placeholder_text",
    "slide_anchor",
    "text_or_empty",
]

#: Lower-cased, stripped values that mean "nothing was here". Trailing
#: punctuation is stripped before the lookup, so ``"None."`` matches too.
#: ``untitled`` is the draft-project default title (``api/schemas/projects.py``).
PLACEHOLDER_TOKENS: frozenset[str] = frozenset(
    {"", "none", "null", "nil", "undefined", "n/a", "na", "nan", "tbd", "todo", "untitled"}
)

# Curly quotes as escapes (ruff RUF001): left/right double, left/right single.
_CURLY = "\u201c\u201d\u2018\u2019"
_TRAILING_PUNCT = " \t\r\n.,;:!?\"'`()[]{}" + _CURLY


def is_placeholder_text(value: object) -> bool:
    """True when ``value`` carries no content: None, blank, or a null-ish token."""
    if value is None:
        return True
    return str(value).strip(_TRAILING_PUNCT).lower() in PLACEHOLDER_TOKENS


def text_or_empty(value: object) -> str:
    """``str(value).strip()``, or ``""`` when the value is a placeholder."""
    return "" if is_placeholder_text(value) else str(value).strip()


def slide_anchor(config_snapshot: Mapping[str, Any]) -> str:
    """The slide eyebrow / deck title for a run — or ``""`` for no eyebrow.

    Order: an explicit ``deliverable.title`` → ``brand.name`` → the project
    ``title``. Deliberately NOT ``deliverable.name``: that is an internal label
    the caller picks for the row (merqi sends ``"marketing-video"``), not copy
    meant for the screen. And never a hardcoded house brand — an unbranded run
    renders no eyebrow rather than someone else's name.
    """
    deliverable = config_snapshot.get("deliverable") or {}
    brand = config_snapshot.get("brand") or {}
    for value in (deliverable.get("title"), brand.get("name"), config_snapshot.get("title")):
        text = text_or_empty(value)
        if text:
            return text
    return ""
