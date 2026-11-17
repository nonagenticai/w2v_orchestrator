"""One-line, printable text from a stranger + the signup locale.

Requester-typed ``name`` / ``company`` / ``message`` end up in a Telegram message
George acts on. A newline, a bidi override or a zero-width character there would
let a requester forge lines of that message ("Signup id: …") or visually reorder
it. So the fields are normalised at INPUT VALIDATION and again at render
(``telegram.py``). Port of fr ``services/signup/untrusted-text.ts``.
"""

from __future__ import annotations

import re
import unicodedata

from src.config import SIGNUP_LOCALES

_SPACES = re.compile(r"\s+")


def _category(ch: str) -> str:
    return unicodedata.category(ch)


def single_line(value: str) -> str:
    """Controls / line + paragraph separators → space, format characters (bidi,
    zero-width, BOM) removed, whitespace collapsed, trimmed. Never a line break."""
    out: list[str] = []
    for ch in value:
        cat = _category(ch)
        if cat in ("Cc", "Zl", "Zp"):
            out.append(" ")
        elif cat == "Cf":
            continue
        elif cat == "Zs":
            out.append(" ")
        else:
            out.append(ch)
    return _SPACES.sub(" ", "".join(out)).strip()


def multi_line(value: str) -> str:
    """Keep ``\\n`` (normalised from CR/CRLF/U+2028/U+2029), tabs → space, drop every
    other control and format character, trim."""
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    for ch in value:
        cat = _category(ch)
        if ch == "\n" or cat in ("Zl", "Zp"):
            out.append("\n")
        elif ch == "\t":
            out.append(" ")
        elif cat in ("Cc", "Cf"):
            continue
        else:
            out.append(ch)
    return "".join(out).strip()


def excerpt(value: str | None, limit: int) -> str:
    """The first ``limit`` characters of ``value`` on one line, ``…`` when cut."""
    flat = single_line(value or "")
    return f"{flat[:limit]}…" if len(flat) > limit else flat


def normalise_locale(value: object) -> str | None:
    """A caller-supplied language tag → one of :data:`SIGNUP_LOCALES`, or None.
    Lenient on purpose: ``de-DE``, ``DE`` and ``de_AT`` mean German, an unknown
    language is "not told" (→ English), never a refused signup."""
    if not isinstance(value, str):
        return None
    primary = re.split(r"[-_]", value.strip().lower(), maxsplit=1)[0]
    return primary if primary in SIGNUP_LOCALES else None
