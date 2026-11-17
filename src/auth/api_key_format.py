"""Tenant API key format: ``w2v_sk_<prefix>_<secret>`` (P2 tenancy, migration 0074).

* ``prefix`` — 8 lowercase RFC 4648 base32 characters (40 random bits). Public: it
  is stored in clear, UNIQUE, and is the lookup key (``tenant_api_keys.prefix``).
* ``secret`` — ``secrets.token_urlsafe(32)``: 32 random bytes, 43 urlsafe-base64
  characters (``A-Z a-z 0-9 - _``). Shown to the caller ONCE, never stored.
* stored hash — SHA-256 of the FULL key string (32 bytes,
  ``tenant_api_keys.key_hash``). Compared with :func:`hmac.compare_digest`.

The prefix has a FIXED length, so parsing never has to guess where it ends even
though the secret alphabet contains ``_``. Pure functions only — no I/O.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass

KEY_SCHEME = "w2v_sk_"
PREFIX_LENGTH = 8
SECRET_BYTES = 32
#: ``token_urlsafe(32)`` = ceil(32 * 4 / 3) = 43 chars, unpadded.
SECRET_LENGTH = 43

_PREFIX_RE = re.compile(r"^[a-z2-7]{8}$")
_SECRET_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")


@dataclass(frozen=True, slots=True)
class ParsedApiKey:
    prefix: str
    secret: str


@dataclass(frozen=True, slots=True)
class GeneratedApiKey:
    """A freshly minted key. ``full_key`` is returned to the caller once; only
    ``prefix`` and ``key_hash`` are persisted."""

    full_key: str
    prefix: str
    key_hash: bytes

    def __repr__(self) -> str:  # never leak the secret into logs / tracebacks
        return f"GeneratedApiKey(prefix={self.prefix!r}, full_key='***')"


def looks_like_api_key(token: str) -> bool:
    """Cheap scheme check — lets an auth lane route ``w2v_sk_…`` bearers to the
    key verifier instead of the JWT decoder. Does NOT validate the key."""
    return token.startswith(KEY_SCHEME)


def new_prefix() -> str:
    # 5 random bytes = 40 bits = exactly 8 base32 characters, no padding.
    return base64.b32encode(secrets.token_bytes(5)).decode("ascii").lower()


def hash_api_key(full_key: str) -> bytes:
    return hashlib.sha256(full_key.encode("utf-8")).digest()


def format_api_key(prefix: str, secret: str) -> str:
    return f"{KEY_SCHEME}{prefix}_{secret}"


def generate_api_key() -> GeneratedApiKey:
    prefix = new_prefix()
    full_key = format_api_key(prefix, secrets.token_urlsafe(SECRET_BYTES))
    return GeneratedApiKey(full_key=full_key, prefix=prefix, key_hash=hash_api_key(full_key))


def parse_api_key(token: str) -> ParsedApiKey | None:
    """Split a presented key into ``(prefix, secret)``; ``None`` if malformed.

    Strict: exact scheme, an 8-char base32 prefix, one ``_`` separator and a
    43-char urlsafe secret. Anything else (wrong lengths, whitespace, uppercase
    prefix) is rejected before a DB lookup is spent on it.
    """
    if not token.startswith(KEY_SCHEME):
        return None
    rest = token[len(KEY_SCHEME) :]
    if len(rest) != PREFIX_LENGTH + 1 + SECRET_LENGTH or rest[PREFIX_LENGTH] != "_":
        return None
    prefix, secret = rest[:PREFIX_LENGTH], rest[PREFIX_LENGTH + 1 :]
    if not _PREFIX_RE.match(prefix) or not _SECRET_RE.match(secret):
        return None
    return ParsedApiKey(prefix=prefix, secret=secret)


def verify_api_key(presented: str, stored_hash: bytes) -> bool:
    """Constant-time compare of ``sha256(presented)`` against the stored hash."""
    return hmac.compare_digest(hash_api_key(presented), stored_hash)
