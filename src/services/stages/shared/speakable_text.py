"""Model-generated narration must be SPEAKABLE: no URL or raw path reaches the voice.

Run ``01a0b007``: bdd's draft and script_doctor's rewrite both carried
``…application at "https://demo.playwright.dev/todomvc".`` on clip-5. ``split()`` counts
that URL as ONE word; kokoro spells it out character by character. The word budget
under-priced the line several-fold (delivered at 82.3 wpm against a solved 136.06) and
narrate had to speed it up 1.22-1.47x.

One rule, two uses:

* :func:`contains_url` — the DETECTOR (reconcile_timeline's re-ask refuses a reply that
  trips it).
* :func:`speakable_text` — the deterministic REWRITE applied to MODEL-GENERATED line
  text only: ``demo_video_gen``'s bdd line build (before gen samples its rate) and
  ``reconcile_timeline``'s read of the final plan lines (before it measures/budgets), on a
  run whose spine is not owner-authored (:func:`~.authored_storyboard.authored_lines`).

What counts as a URL — ONLY what is evidenced and unambiguous: a whitespace token
(surrounding quotes/brackets and trailing punctuation stripped) that starts with
``http://`` / ``https://`` or ``www.``. The live DB held 103 full ``http(s)://`` URLs
across 284 spine lines and ZERO bare paths. Bare ``/paths`` and ``host/path`` tokens are
deliberately NOT matched: ``redirected to /checkout/success``, ``Node.js/Express``,
``ASP.NET/C#``, ``README.md/`` are prose a voice reads fine, and rewriting them garbled
the sentence ("I am on the the site page").
"""

from __future__ import annotations

import re
from typing import Final

_URL_TOKEN_RE: Final = re.compile(r"^(?:https?://\S+|www\.\S+)$", re.IGNORECASE)
_QUOTES: Final = "\"'`\u201c\u201d\u2018\u2019"
_OPENERS: Final = "([{<"
_EDGE_PUNCTUATION: Final = _QUOTES + _OPENERS + ")]}>.,;:!?"

#: What a URL that is not a removable "at <url>" clause is spoken as.
URL_SPOKEN_AS: Final = "the site"

_SENTINEL: Final = "\x00URL\x00"
# ", which is located at the URL <url>" / " at <url>" / " available at <url>" — a trailing
# location clause that adds nothing a listener can use. Only "at"/"via": "on <url>",
# "to <url>" and "from <url>" carry the sentence's verb and get the phrase instead.
_CLAUSE_RE: Final = re.compile(
    r"(?:,\s*)?(?:\bwhich\s+is\s+)?(?:\b(?:located|found|available|hosted|live)\s+)?"
    r"\b(?P<prep>at|via)\s+(?:the\s+(?:url|address|link|web\s+address|page)\s+)?"
    + re.escape(_SENTINEL)
    # …and only when the clause ENDS the sentence/phrase: "Sign up at <url> today" keeps
    # its preposition as "Sign up at the site today".
    + r"(?=[.,;:!?)\]}>]|$)",
    re.IGNORECASE,
)
# A clause is only removed when it does not carry the sentence: "I am at <url>" keeps
# its meaning only as "I am at the site".
_VERB_BEFORE_CLAUSE_RE: Final = re.compile(
    r"\b(?:am|is|are|was|were|be|been|go|goes|went|navigate|navigates|arrive|arrives|"
    r"look|looks|start|starts|begin|begins)\s*$",
    re.IGNORECASE,
)


def _is_url_token(token: str) -> bool:
    return bool(_URL_TOKEN_RE.match(token.strip(_EDGE_PUNCTUATION)))


def contains_url(text: str) -> bool:
    """True when ``text`` carries a URL or raw path by this module's rule."""
    return any(_is_url_token(token) for token in text.split())


def _mark(token: str) -> str:
    """Replace a URL token by the sentinel, keeping surrounding brackets and punctuation."""
    leading = ""
    core = token
    while core and core[0] in _QUOTES + _OPENERS:
        if core[0] in _OPENERS:
            leading += core[0]
        core = core[1:]
    trailing = ""
    while core and core[-1] in _QUOTES + ".,;:!?)]}>":
        if core[-1] not in _QUOTES:
            trailing = core[-1] + trailing
        core = core[:-1]
    return leading + _SENTINEL + trailing


def speakable_text(text: str) -> str:
    """``text`` with every URL / raw path rewritten deterministically. Idempotent.

    * a trailing ``at <url>`` / ``which is located at the URL <url>`` clause is REMOVED
      (``…application at "https://…".`` → ``…application.``), unless the word before it
      is the sentence's verb (``I am at <url>``);
    * any other URL becomes :data:`URL_SPOKEN_AS` (``Go to <url>`` → ``Go to the site``).

    Never empties a line: a line that was only a URL becomes the phrase. Text with no
    URL is returned unchanged (same object semantics: equal string).
    """
    if not contains_url(text):
        return text
    marked = " ".join(_mark(t) if _is_url_token(t) else t for t in text.split())

    def _clause(match: re.Match[str]) -> str:
        if _VERB_BEFORE_CLAUSE_RE.search(marked[: match.start()]):
            return f" {match.group('prep')} {URL_SPOKEN_AS}"
        return ""

    out = _CLAUSE_RE.sub(_clause, marked)
    # "on the <url> page" must not become "on the the site page".
    out = re.sub(r"\b(the)\s+" + re.escape(_SENTINEL), r"\1 site", out, flags=re.IGNORECASE)
    out = out.replace(_SENTINEL, URL_SPOKEN_AS)
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out or URL_SPOKEN_AS
