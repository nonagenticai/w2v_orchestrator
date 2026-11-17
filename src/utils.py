"""Shared utility functions for w2v_orchestrator."""

from __future__ import annotations

import math
import re
from typing import SupportsFloat


def format_exception_message(exc: BaseException) -> str:
    """Format an exception into a non-empty human-readable string.

    Some exception subclasses (e.g. bare ``OSError``, ``RuntimeError()``)
    produce empty strings when cast with ``str()``.  This helper always
    returns something useful by falling back to ``repr()``.
    """
    msg = str(exc)
    if msg:
        return msg
    return f"{type(exc).__name__}: {exc!r}"


# ── "Type: detail" for run/stage error messages — NEVER a trailing empty colon ─
# ``f"{type(exc).__name__}: {exc}"`` is how a failure becomes ``runs.error`` and the
# stage row's error, i.e. the only text a person reading a failed run gets. Several
# exceptions stringify to "" — httpx's ``ReadTimeout`` most often (prod 2026-10-18:
# "...retries exhausted after 2 attempts: ReadTimeout:" and nothing else), bare
# ``TimeoutError()``, ``CancelledError()`` — and ``w2v_common``'s
# ``StageRetriesExhaustedError`` bakes its ``last_error`` into its message with that
# same f-string, so the empty detail is frozen in before we ever see it. These
# helpers rebuild the message from the exception's own attributes instead.

#: httpx timeout class -> the ``request.extensions["timeout"]`` key it ran out of.
#: Matched by NAME so this module keeps importing only the standard library.
_HTTPX_TIMEOUT_PHASE = {
    "ReadTimeout": "read",
    "WriteTimeout": "write",
    "ConnectTimeout": "connect",
    "PoolTimeout": "pool",
}
_TIMEOUT_PHRASE = {
    "read": "no response within {t}",
    "write": "request body not sent within {t}",
    "connect": "could not connect within {t}",
    "pool": "no free connection within {t}",
}


def _request_target_and_timeout(exc: BaseException, phase: str | None) -> tuple[str, str]:
    """``(target, timeout)`` from an httpx exception's ``.request``; ``""`` when unknown.

    httpx's ``.request`` is a property that RAISES when no request was attached, so
    it is read defensively. The target is host + path only — never the query
    string, which can carry presigned credentials.
    """
    try:
        request = getattr(exc, "request", None)
    except RuntimeError:
        return "", ""
    if request is None:
        return "", ""
    target = ""
    url = getattr(request, "url", None)
    host = getattr(url, "host", "") if url is not None else ""
    if host:
        port = getattr(url, "port", None)
        target = f"{host}:{port}" if port else str(host)
        target += str(getattr(url, "path", "") or "")
    timeout = ""
    extensions = getattr(request, "extensions", None)
    if phase and isinstance(extensions, dict):
        budget = extensions.get("timeout")
        seconds = budget.get(phase) if isinstance(budget, dict) else None
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
            timeout = f"{seconds:g}s"
    return target, timeout


def exception_detail(exc: BaseException) -> str:
    """The part after ``"Type: "`` — guaranteed non-empty.

    ``str(exc)`` when it has text; otherwise a description built from what the
    exception carries (timeout phase + budget + target for httpx, the cause chain
    otherwise). A ``StageRetriesExhaustedError`` is always rebuilt so its embedded
    ``last_error`` gets the same treatment.
    """
    # Local import: w2v_common.stages.errors is dependency-free, but keeping it out
    # of module import time preserves this module's stdlib-only import surface.
    from w2v_common.stages.errors import StageRetriesExhaustedError

    if isinstance(exc, StageRetriesExhaustedError):
        return (
            f"stage {exc.stage_name!r}: retries exhausted after {exc.attempts} attempts: "
            f"{describe_exception(exc.last_error)}"
        )
    message = str(exc)
    if message.strip():
        return message
    phase = _HTTPX_TIMEOUT_PHASE.get(type(exc).__name__)
    target, timeout = _request_target_and_timeout(exc, phase)
    if phase is not None:
        detail = _TIMEOUT_PHRASE[phase].format(t=timeout or "the client timeout")
    elif isinstance(exc, TimeoutError):
        detail = "timed out (no further detail)"
    else:
        detail = "raised with no error message"
    if target:
        detail += f" from {target}" if phase in (None, "read") else f" to {target}"
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc and str(cause).strip():
        detail += f" (caused by {type(cause).__name__}: {cause})"
    return detail


def describe_exception(exc: BaseException) -> str:
    """``"Type: detail"`` for a run/stage error message, never ``"Type: "``.

    Drop-in for ``f"{type(exc).__name__}: {exc}"`` wherever the text is persisted
    or shown to a person.
    """
    return f"{type(exc).__name__}: {exception_detail(exc)}"


# ── Exception text that is safe to STORE or SHOW ─────────────────────────────
# Lives here, not in ``src/middleware/audit.py`` where it was born. It has two
# consumers now — the audit middleware and the preview-proxy sweeper — and the
# sweeper is a background loop that must NOT import the HTTP middleware stack
# (``audit`` pulls in starlette and ``src.api.routes._common``). A private symbol
# reached across that layer boundary by a function-local import worked, but a
# rename would only have failed at exception time, in production. This module
# imports nothing but the standard library, so both layers can depend on it.

#: An exception message is unvetted text; keep the stored projection bounded.
MAX_EXC_MESSAGE_CHARS = 500

#: Credential shapes that turn up inside exception strings: a DSN's userinfo, a
#: bearer token echoed by an HTTP client, a `password=`/`token=`/`api_key=` pair.
#:
#: ⚠️ This does NOT cover SigV4 URL QUERY credentials (``X-Amz-Credential`` /
#: ``X-Amz-Signature``). ``src/lifecycle/preview_proxy.py`` layers its own
#: query-parameter scrub ON TOP for that, because its exceptions routinely carry a
#: presigned MinIO URL. Do not "simplify" that layering away.
_CREDENTIAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)://[^/\s:@]+:[^/\s@]+@"), "://[redacted]@"),
    (re.compile(r"(?i)\bbearer\s+[\w\-._~+/]+=*"), "Bearer [redacted]"),
    (
        re.compile(r"(?i)\b(password|passwd|pwd|token|api[_-]?key|secret)\s*[=:]\s*\S+"),
        r"\1=[redacted]",
    ),
)


def safe_exc_message(exc: Exception) -> str:
    """A bounded, credential-scrubbed rendering of an exception's message.

    `str(exc)` on a bare `TimeoutError()` is empty, so fall back to the repr the
    same way the persist-failure log line does.
    """
    message = str(exc) or repr(exc)
    for pattern, replacement in _CREDENTIAL_PATTERNS:
        message = pattern.sub(replacement, message)
    return message[:MAX_EXC_MESSAGE_CHARS]


# ── Float coercion: TWO deliberately separate families ───────────────────────
# STRICT (``as_float`` / ``opt_float``) guards with isinstance and is for values
# that are already supposed to BE numbers (analyzer scalars, measured loudness,
# configured ceilings) — a string or a bool arriving there is a wire-shape bug, not
# a value to rescue, and a NaN would poison downstream arithmetic.
# PERMISSIVE (``coerce_float``) parses whatever an LLM/JSON boundary hands back.
# They disagree on five live inputs — ``"1.5"``, ``True``, ``float("nan")``,
# ``float("inf")`` and ``Decimal("2")`` — so merging them silently changes behaviour at
# every call site. ``tests/test_utils.py`` pins that divergence. Do NOT collapse these.
#
# NON-FINITE REJECTION (both strict helpers): ``inf`` / ``-inf`` are rejected exactly
# like NaN. Every strict call site is a MEASUREMENT (analyzer scalars, compiler
# loudness, configured ceilings) where an infinity is a wire-shape bug, and letting one
# through is not inert: it propagates into ``statistics.pstdev`` (which raises
# ``ValueError`` on an infinite variance — one bad frame then discards a whole gestalt),
# into Prometheus histogram ``_sum``, and into JSON serialisation (``Infinity`` is not
# valid JSON). ``default`` / ``None`` is always the safer read.


def as_float(value: object, default: float = 0.0) -> float:
    """STRICT coercion to float; junk / ``None`` / ``bool`` / non-finite ⇒ ``default``.

    Never raises. Numeric strings are REJECTED (``"1.5"`` ⇒ ``default``) and so are the
    non-finite floats ``nan`` / ``inf`` / ``-inf`` — see the family note above.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    f = float(value)
    return f if math.isfinite(f) else default  # NaN / ±inf ⇒ default


def opt_float(value: object) -> float | None:
    """STRICT optional scalar: ``None`` passes through; junk / ``bool`` / non-finite ⇒ ``None``.

    The optional twin of :func:`as_float`, kept SEPARATE on purpose: folding it in via
    a ``None`` default would widen ``as_float``'s return type to ``float | None`` and
    poison every caller that legitimately relies on getting a real float back.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None  # NaN / ±inf ⇒ None


def coerce_float(value: object) -> float | None:
    """PERMISSIVE best-effort float coercion of an untrusted JSON/LLM/ffmpeg value.

    Parses numeric strings (``"1.5"`` ⇒ ``1.5``), accepts ``bool`` (``True`` ⇒ ``1.0``)
    and any ``SupportsFloat`` (e.g. ``Decimal``), and passes NaN through unchanged.
    ``None`` or an unparseable value ⇒ ``None``; never raises. Use this only at wire
    boundaries where a provider may legitimately hand back a stringified number.
    """
    if value is None:
        return None
    if not isinstance(value, (SupportsFloat, str)):
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


# ── Rubric-score clamping: the bound the SCHEMA cannot carry ─────────────────
# vLLM guided decoding constrains an LLM's output GRAMMAR and TYPE, not its numeric
# range — a JSON-Schema ``minimum``/``maximum`` on an integer property is therefore
# unenforced at decode time and then FATAL at post-hoc validation. Worker-management
# sets ``retries=0`` for the ``inference_router`` provider deliberately, so the FIRST
# out-of-range integer kills the stage permanently (prod ``demo_video`` runs failed
# 100% at ``retention_plan:critic`` on ``overall_score: 12``, contaminated by the hard
# numbers in the critic's own evidence block). The band therefore lives in the schema
# ``description`` + the system prompt, where it STEERS the model, and is ENFORCED here,
# at the consumer. ``tests/test_output_schema_bounds.py`` stops the schema form coming
# back.


def clamp_score(
    value: object, *, low: float = 1.0, high: float = 5.0, default: float = 0.0
) -> float:
    """Read an untrusted LLM rubric score; anything not inside ``[low, high]`` ⇒ ``default``.

    PERMISSIVE at the parse step (built on :func:`coerce_float`, so a stringified
    ``"4"`` from a provider still counts) but ``bool`` is rejected outright — JSON
    ``true`` is not a measurement, exactly as in the strict family above.

    An out-of-band number is treated as NO ANSWER, not as the nearest bound. This
    is the whole point: the observed failure was a critic emitting ``12`` for a 1-5
    score, having leaked a magnitude out of the evidence block in its prompt. Such
    a value is contaminated, not merely large — and saturating it to ``high`` would
    be the worst available reading, since a top score clears every convergence gate
    and so silently certifies the draft as perfect on the first pass, skipping the
    optimizer. Folding it to ``default`` instead makes a garbage answer look like
    the absent answer it really is, and the improvement pass still runs.

    ``default`` sits OUTSIDE the band on purpose. Every call site already read an
    absent-or-junk score as ``0.0``, and that zero is load-bearing: it must stay
    distinguishable from a genuine floor score of ``low`` so a critic that answered
    nothing never outranks one that scored the draft badly but honestly. Pass an
    explicit ``default`` only when a call site wants the missing case inside the band.
    """
    if isinstance(value, bool):
        return default
    score = coerce_float(value)
    if score is None or score != score:  # unparseable, or NaN (coerce_float passes NaN)
        return default
    if not (low <= score <= high):
        return default
    return score
