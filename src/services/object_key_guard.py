"""Object-key confinement — the one place a caller-supplied storage key is
re-derived against the prefix its owner is allowed to name.

``w2v_common.storage.StorageService`` has no tenant concept: ``get_bytes(key)`` /
``retrieve(ref)`` / ``get_object_stream(bucket, key)`` take a raw key and read it.
Confinement is therefore the *caller's* job, and "the caller proved it owns
something" is only half of it — proving ownership of a run/tenant while still
passing an attacker-supplied key straight through reads someone else's object
with our credentials.

The rule this module implements (the ``frontend_test_runner_service`` pattern):
after ownership is proven, the untrusted string may only ever name a **suffix
under the owned prefix**. The comparison is on path *segments*, never on raw
string prefixes — ``documents/<tenant>evil/x`` must not pass a
``documents/<tenant>`` test — and the input is decoded + normalised first so
``..`` traversal, absolute paths, percent- and double-percent-encoded separators
(``%2e%2e``, ``%252f``) and ``s3://`` / ``minio://`` / ``file://`` schemes cannot
smuggle an escape past the check.

**Decode to decide, never to rewrite.** Percent-decoding is a *safety probe*, not
a transform on the identifier: an object legitimately keyed
``discount%2Ddeal.pdf`` (a literal ``%`` in the key) presigns to the URL path
``discount%252Ddeal.pdf``, and a decode-to-fixpoint would collapse both to
``discount-deal.pdf`` — a key that matches the allowlist and then 404s at MinIO
because it does not exist. So :func:`normalize_key` returns the caller's own
spelling (path-normalised only) and runs every refusal test against the decoded
probe. Callers holding a *URL path* (which carries exactly one encoding layer)
strip that one layer with :func:`decode_ref_path` first, so both sides of a
comparison speak the same true object key.
"""

from __future__ import annotations

import posixpath
from urllib.parse import unquote

# Percent-decoding is applied repeatedly so a double-encoded separator
# (``%252f`` → ``%2f`` → ``/``) is seen as the separator it becomes. A key that
# is still changing after this many rounds is not a key anyone legitimately
# minted — we reject rather than keep peeling.
_MAX_DECODE_ROUNDS = 3

# Characters that never appear in a key we wrote and that each defeat the
# segment comparison in their own way: a URL scheme would redirect the read, a
# backslash is a separator on the way through some SDKs, a NUL truncates.
#
# ``\`` is deliberately a blanket refusal even though S3/MinIO accept it as an
# ordinary key byte. DECISION: backslash-bearing keys are an UNSUPPORTED key
# shape here. Justification: nothing in this service can mint one — every key is
# an f-string of a uuid, a stage name and a sanitised filename
# (``artifacts/{job}/{stage}/{file}``) — so refusing it costs zero reachable
# objects, while accepting it costs a separator that ``posixpath`` does not treat
# as one but Windows-flavoured path handling downstream does, i.e. a live
# smuggling surface for ``..\``-shaped escapes past the segment comparison. If a
# writer ever legitimately emits ``a\b.png``, fold ``\`` only where it could
# smuggle a separator (``..\`` / leading ``\``) rather than lifting this outright.
_FORBIDDEN_SUBSTRINGS = ("://", "\\", "\x00")


class KeyConfinementError(ValueError):
    """A caller-supplied key escaped (or could not be tied to) its owner's prefix."""


def _decode(value: str) -> str:
    """Percent-decode until stable; raise if it never stabilises."""
    seen = value
    for _ in range(_MAX_DECODE_ROUNDS):
        decoded = unquote(seen)
        if decoded == seen:
            return seen
        seen = decoded
    raise KeyConfinementError("key is encoded beyond any legitimate depth")


def normalize_prefix(prefix: str) -> str:
    """Normalise an *owned* prefix (trusted input) to its segment form.

    Trailing/leading slashes are dropped so ``confine_key`` can do a single
    ``prefix + "/"`` segment comparison. Raises when the prefix would degenerate
    to "everything" — an empty or ``.``/``/`` prefix would confine nothing, and
    silently allowing that is how a confinement check becomes decorative.
    """
    candidate = posixpath.normpath((prefix or "").strip().strip("/"))
    if not candidate or candidate in (".", "/") or candidate.startswith(".."):
        raise KeyConfinementError("refusing to confine against an empty prefix")
    return candidate


def decode_ref_path(path: str) -> str:
    """Strip the ONE percent-encoding layer a URL path carries, and no more.

    A stored artifact ref is the path component of a presigned URL, so the true
    object key is exactly one ``unquote`` away — a key containing a literal ``%``
    arrives as ``%25``. Decoding further would not recover a key, it would invent
    a different one (see the module docstring). Use this on anything that came out
    of a URL *before* handing it to :func:`normalize_key`; a key that arrived as a
    path parameter has already been decoded once by Starlette and must NOT be run
    through this a second time.
    """
    return unquote(path)


def _assert_plain_relative_key(candidate: str, original: str) -> str:
    """Run every refusal test on ``candidate``; return its path-normalised form."""
    if not candidate:
        raise KeyConfinementError("empty object key")
    if any(bad in candidate for bad in _FORBIDDEN_SUBSTRINGS):
        raise KeyConfinementError(f"refusing non-key reference: {original!r}")
    if candidate.startswith("/"):
        raise KeyConfinementError(f"refusing absolute object key: {original!r}")
    # normpath collapses ``.``/``..`` *before* any comparison, so an escaping key
    # ends up outside its prefix as a plain path and is caught by the segment
    # test rather than sneaking through as a literal string prefix.
    normalized = posixpath.normpath(candidate)
    if normalized in (".", "..") or normalized.startswith("../"):
        raise KeyConfinementError(f"refusing traversing object key: {original!r}")
    return normalized


def normalize_key(key: str) -> str:
    """Normalise an untrusted object key into its comparison form.

    Every confinement decision — prefix containment *and* set membership —
    compares this form, so a key that reaches storage is always the same key that
    was checked. Raises :class:`KeyConfinementError` for anything that is not a
    plain relative object key.

    The returned value is the caller's OWN spelling with ``.``/``..`` segments
    collapsed — never the percent-decoded one. Decoding runs to a fixpoint only
    to *judge* the key (``%252f`` must be seen as the separator it becomes), and
    both the caller's form and that decoded probe must pass every refusal test;
    the probe is then thrown away so a legitimate ``discount%2Ddeal.pdf`` is
    matched and fetched under the name it actually has. ``normpath`` is
    identifier-preserving for real keys (no writer emits a ``./`` or ``../``
    segment), so the identifier that reaches storage is the one that was checked.
    """
    raw = (key or "").strip()
    normalized = _assert_plain_relative_key(raw, key)
    # Safety probe only: never returned, never fetched.
    probe = _decode(raw)
    _assert_plain_relative_key(probe, key)
    _assert_decoding_does_not_resegment(raw, probe, key)
    return normalized


def _assert_decoding_does_not_resegment(raw: str, probe: str, original: str) -> None:
    """Refuse a key whose percent-decoding would change its *path structure*.

    This is what keeps "return the caller's spelling" from becoming a hole. A key
    may absolutely contain an escaped byte — ``discount%2Ddeal.pdf`` decodes to a
    different *name* but the same one-segment path, and is served under the name
    it really has. What it may not do is decode into a different *shape*:
    ``a%2Fb`` gains a segment, ``%2e%2e/x`` turns a segment into ``..``. Those are
    separator smuggling, not naming — a downstream layer that decodes once more
    than we did would then walk somewhere the segment comparison never approved.
    Rejected outright rather than sanitised, so there is nothing left to disagree
    about.
    """
    if probe == raw:
        return
    raw_segments = raw.split("/")
    probe_segments = probe.split("/")
    if len(raw_segments) != len(probe_segments) or any(
        decoded in (".", "..") and decoded != literal
        for literal, decoded in zip(raw_segments, probe_segments, strict=True)
    ):
        raise KeyConfinementError(f"refusing encoded path separator in object key: {original!r}")


def key_segments(key: str) -> list[str]:
    """The normalised key's path segments — for rules that match a fixed slot."""
    return normalize_key(key).split("/")


def is_within(prefix: str, normalized_key: str) -> bool:
    """Segment-wise containment of an already-normalised key under ``prefix``.

    Segment-wise is the whole point: ``documents/<tenant>evil/x`` shares a string
    prefix with ``documents/<tenant>`` and must still be refused.
    """
    base = normalize_prefix(prefix)
    return normalized_key == base or normalized_key.startswith(base + "/")


def confine_key(prefix: str, key: str) -> str:
    """Return the normalised ``key``, proven to live at or under ``prefix``.

    ``prefix`` is trusted (derived from the run / tenant we already authorised);
    ``key`` is not. Raises :class:`KeyConfinementError` on anything that is not a
    plain relative key inside the prefix.
    """
    normalized = normalize_key(key)
    if not is_within(prefix, normalized):
        raise KeyConfinementError(f"refusing to access outside {prefix}: {key!r}")
    return normalized
