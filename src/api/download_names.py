"""Person-facing download file names and their ``Content-Disposition`` header.

A download named ``captions.srt`` for every run is indistinguishable from the
next one in a Downloads folder, and a server ``Content-Disposition`` filename
OVERRIDES the browser's ``<a download="…">`` hint — so the name has to be right
here, not only in the frontend. The ASCII stem matches the frontend's
``downloadBaseName`` (``src/lib/utils/output-labels.ts``) so both agree.

The header carries two names (RFC 6266 §4.3): an ASCII ``filename=`` every
client understands, and an RFC 5987 ``filename*=UTF-8''…`` that keeps a
non-ASCII project title intact for the clients that read it.
"""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import quote

_MAX_STEM = 60
_FALLBACK_STEM = "video"


def _cap(stem: str) -> str:
    return stem[:_MAX_STEM].rstrip("-")


def ascii_stem(title: str | None) -> str:
    """Lowercase ASCII letters/digits joined by single dashes; ``video`` if empty.

    The SAME algorithm, step for step, as the frontend's ``downloadBaseName``
    (``w2v_frontend/src/lib/utils/output-labels.ts``): NFKD, drop the combining
    marks U+0300 to U+036F, lowercase, collapse every run outside ``[a-z0-9]`` to
    one dash, trim dashes, cap at 60. Both repos test the identical vector file
    (``download-name-vectors.json``), so "Straße" or "a日b" cannot name the
    caption file differently from the video beside it.
    """
    folded = unicodedata.normalize("NFKD", title or "")
    folded = "".join(ch for ch in folded if not "\u0300" <= ch <= "\u036f")
    stem = _cap(re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-"))
    return stem or _FALLBACK_STEM


def unicode_stem(title: str | None) -> str:
    """Like :func:`ascii_stem` but keeps non-ASCII letters (``café-launch``).

    NFC first: a decomposed ``e`` + U+0301 would otherwise lose its accent, since a
    lone combining mark is not a word character and becomes a dash."""
    text = unicodedata.normalize("NFC", title or "").lower()
    stem = _cap(re.sub(r"[\W_]+", "-", text, flags=re.UNICODE).strip("-"))
    return stem or _FALLBACK_STEM


def attachment_disposition(title: str | None, suffix: str) -> str:
    """``attachment`` disposition for ``<title-stem><suffix>``, e.g. ``-captions.srt``.

    ``suffix`` is code-controlled ASCII; the title is untrusted, so it only ever
    reaches the header through the two stem functions above (no quotes, no
    backslashes, no control characters survive them) and percent-encoding.
    """
    ascii_name = f"{ascii_stem(title)}{suffix}"
    utf8_name = f"{unicode_stem(title)}{suffix}"
    header = f'attachment; filename="{ascii_name}"'
    if utf8_name != ascii_name:
        header += f"; filename*=UTF-8''{quote(utf8_name, safe='')}"
    return header
