"""Per-book pronunciation lexicon for audiobook narration.

The TTS backend (`tts_backend`, kokoro) already normalizes numbers, times,
decimals, symbols and initialisms on every synth. What it *cannot* know is how
a given book wants its proper nouns and coined words voiced — "Koster",
"Doorn", "Merqi", "café". This module applies a per-book ``{written: spoken}``
lexicon once, at manuscript-parse time, rewriting those tokens to a spoken
respelling the phonemizer reads correctly.

Because the rewritten text flows unchanged into the content-addressed clip id
(`tts_storage.deterministic_clip_id`) and the gateway idempotency key, a lexicon
change automatically yields new clip ids → clean re-synth. No PREPROC_VERSION
bump is needed for a lexicon change (that constant is only for *backend-side*
preprocessing changes that don't alter the request text).

Matching is whole-word (so ``Bos`` never matches inside ``Boston``),
case-insensitive, longest-key-first (so a multi-word key like ``de Jong`` wins
over its fragments), and preserves quote characters so the downstream
voice-acting span splitter (`parsing.split_spans`, quote-keyed) is undisturbed.
"""

from __future__ import annotations

import re

# Global respellings that apply to every book. Kept intentionally minimal —
# book-specific names belong in the project brief (``brief.pronunciations``),
# not here. Extend only with genuinely universal respellings.
DEFAULT_LEXICON: dict[str, str] = {}


def build_lexicon(brief_pronunciations: dict[str, str] | None) -> dict[str, str]:
    """Merge the global default lexicon with a book's brief-supplied map.

    The brief wins on conflicts, so a book can override or extend the defaults.
    Non-string / empty keys and values are dropped defensively (the brief is a
    free-form dict authored outside this codebase).
    """
    merged = dict(DEFAULT_LEXICON)
    for written, spoken in (brief_pronunciations or {}).items():
        if isinstance(written, str) and isinstance(spoken, str) and written.strip():
            merged[written] = spoken
    return merged


def _compile(lexicon: dict[str, str]) -> tuple[re.Pattern[str], dict[str, str]] | None:
    """Compile a single whole-word alternation over the lexicon keys.

    Keys are escaped and sorted longest-first so a longer key (``de Jong``)
    is preferred over a shorter one it contains (``Jong``). Returns the pattern
    plus a lowercased-key → value lookup, or ``None`` for an empty lexicon.
    """
    keys = [k for k in lexicon if k.strip()]
    if not keys:
        return None
    keys.sort(key=len, reverse=True)
    # ``(?<!\w)…(?!\w)`` rather than ``\b…\b``: a plain ``\b`` fails on keys that
    # begin or end with a non-word char (e.g. a dotted ``C.A.B.``), because the
    # boundary would fall between two non-word chars. The negative-lookaround form
    # means "not flanked by word chars", which still blocks partial matches
    # (``Bos`` inside ``Boston``) but works for keys carrying punctuation.
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)",
        re.IGNORECASE,
    )
    lookup = {k.lower(): v for k, v in lexicon.items()}
    return pattern, lookup


def apply_lexicon(text: str, lexicon: dict[str, str]) -> str:
    """Rewrite every whole-word lexicon key in ``text`` to its spoken form.

    Case-insensitive; the canonical spoken value is emitted regardless of the
    matched casing. A no-op when the lexicon is empty.
    """
    compiled = _compile(lexicon)
    if compiled is None:
        return text
    pattern, lookup = compiled
    return pattern.sub(lambda m: lookup.get(m.group(0).lower(), m.group(0)), text)
