"""Classify upstream (worker-management / LLM) HTTP failures as permanent or transient.

WHY this module exists — three failures observed live, all treated identically today:

1. WM **402** ``byo_llm_not_configured`` — "Configure your LLM provider in Settings
   before running agents." The tenant has no LLM provider. Retrying cannot help; the
   orchestrator retried twice and then buried the sentence under a generic
   ``StageRetriesExhaustedError`` ("retries exhausted"), so the user never saw the one
   line that would have told them what to do.
2. WM **500** "Model token limit (4000) exceeded before any response was generated."
   Deterministic: the same prompt against the same worker config will exceed the same
   limit on every attempt. Retrying burns wall-clock and tokens for a guaranteed loss.
3. WM **500** "Connection error." — genuinely transient (pgx/inference-router blip).
   This one MUST keep retrying; it is the reason the retry budget exists at all.
4. WM **422** "Exceeded maximum output retries (3)" — a 4xx that is NOT our bug: the
   model failed structured-output validation three times in a row. A different draw of
   the same request succeeds. See ``TRANSIENT_4XX_MARKERS`` below.

So: a stage needs to distinguish "this will never work" (fail fast, surface the real
sentence) from "try again" (spend the budget). ``classify`` does that; ``guarded``
wraps a call factory so the distinction happens before ``stage_call``'s ``retry_on``
ever sees an ``httpx.HTTPStatusError``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from src.models.failure_class import (
    EXTERNAL,
    TRANSIENT_STATUSES,
    UNKNOWN,
    FailureClass,
    fault_for_status,
)
from src.utils import describe_exception

# Re-exported for import-path stability: this module has always been where
# callers and tests reach for the fault-axis status map. The definitions moved
# to ``models.failure_class`` so that an unstamped ``httpx.HTTPStatusError``
# reaching ``attribute_exception`` can use the SAME map without importing a
# service module from a leaf model.
__all__ = [
    "PERMANENT_5XX_MARKERS",
    "PERMANENT_STATUSES",
    "TRANSIENT_4XX_MARKERS",
    "TRANSIENT_STATUSES",
    "UPSTREAM_AUTH",
    "UPSTREAM_AUTH_5XX_MARKERS",
    "PermanentUpstreamError",
    "TransientUpstreamError",
    "UpstreamAuthError",
    "classify",
    "classify_transport",
    "fault_for_status",
    "guarded",
]

log = logging.getLogger(__name__)


class _AttributedUpstreamError(Exception):
    """Mixin for the FAULT attribution carried alongside the retryability verdict.

    ⚠️ Two ORTHOGONAL axes travel on one exception, and conflating them is the
    mistake this attribute exists to prevent. The CLASS says whether a retry can
    help (permanent vs transient); ``failure_class`` says whose fault it was.
    Our own 500 and a customer's 400 are both ``PermanentUpstreamError`` and have
    completely different faults; a customer's unconfigured LLM provider is
    permanent while an inference-router blip is transient, and neither pair
    shares an attribution. Nothing here may ever derive one axis from the other.

    It is stamped HERE, in ``classify``/``classify_transport``, because this is
    where the HTTP status actually is — by the time the error reaches
    ``run_dispatcher``'s failure site the response is long gone. The default is
    ``UNKNOWN`` so an instance raised anywhere else in the codebase (there are
    such call sites) asserts nothing rather than inheriting an attribution.
    """

    failure_class: FailureClass = UNKNOWN


class TransientUpstreamError(_AttributedUpstreamError):
    """Upstream failed in a way that a retry can plausibly fix. Retry me."""


class PermanentUpstreamError(_AttributedUpstreamError):
    """Upstream failed in a way no retry can fix. ``str(self)`` IS the user-facing message.

    LOAD-BEARING: this deliberately sits under ``Exception`` and **not**
    ``RuntimeError`` (and therefore not ``w2v_common``'s ``StageError``, which extends
    ``RuntimeError``). Retry wrappers in this repo list broad retryable types — any of
    which would silently swallow a permanent failure back into the retry budget if this
    class sat under ``RuntimeError``. Keep it off that branch of the hierarchy. Do not
    "tidy" this into a StageError subclass. ``_AttributedUpstreamError`` (its only
    base) extends ``Exception`` directly for exactly this reason — do not re-base it.
    """


#: ``failure_detail.code`` of an upstream AUTH/LICENCE refusal (see ``UpstreamAuthError``).
UPSTREAM_AUTH = "upstream_auth"


class UpstreamAuthError(PermanentUpstreamError):
    """An upstream refused the call's CREDENTIALS or LICENCE — no retry can fix it.

    Raised for a direct 401/403 and for a 5xx whose body is a wrapped 401/403 from
    the hop behind it (UI review 2026-10-17 §0: worker_management answered **500**
    wrapping inference-router ``401 Caller licence could not be introspected`` after
    licences answered 409 ``ambiguous_bundle``; ``classify`` called that transient and
    every LLM stage burned its whole retry budget on a failure that needed a config
    change).

    Carries a structured ``failure_detail`` (``runs.failure_detail``, read by
    ``failure_class.failure_detail_for``) so the frontend can branch on
    ``code == "upstream_auth"`` + ``scope`` instead of parsing prose:

    * ``scope="platform"`` — OUR credential/licence (WM/inference-router S2S licence,
      the static vLLM token). The tenant cannot fix it; support must.
    * ``scope="tenant"`` — the tenant's own BYO provider key was rejected; the tenant
      fixes it on the Providers page.

    ``str(self)`` stays the diagnostic sentence (status + url + upstream detail) for
    ``runs.error``; the user-facing line is ``failure_detail["message"]``.
    """

    def __init__(self, message: str, *, scope: str = "platform") -> None:
        super().__init__(message)
        self.scope = scope

    @property
    def failure_detail(self) -> dict[str, str]:
        if self.scope == "tenant":
            text = (
                "Your AI provider rejected its API key. "
                "Update the key on the Providers page, then restart the run."
            )
        else:
            text = (
                "The AI service refused this request's licence or credentials. "
                "Retrying will not help — this needs a platform configuration fix; "
                "contact support."
            )
        return {"code": UPSTREAM_AUTH, "scope": self.scope, "message": text}


# 4xx that can never be fixed by repeating the identical request.
PERMANENT_STATUSES = frozenset({400, 401, 402, 403, 404, 405, 413, 415, 422, 501})

# Substrings that mark a 5xx body as a DETERMINISTIC failure (prompt/config too big for
# the worker's token budget) rather than an infrastructure blip. WM should really return
# 422 for these — a follow-up will fix WM's status code, after which this list becomes
# belt-and-braces rather than load-bearing. Until then it is the only signal we have.
PERMANENT_5XX_MARKERS = ("token limit", "exceeded before any response", "context length")

#: Statuses that mean "your credentials / licence were refused" on THIS hop.
AUTH_STATUSES = frozenset({401, 403})

# Substrings (lowercase) that mark a **5xx** body as a WRAPPED auth/licence refusal
# from the hop behind the one we called — WM turns an inference-router 401 into its
# own 500. Matched against the body WM *asserted* (``_envelope_detail``) and the raw
# body, same convention as ``PERMANENT_5XX_MARKERS``.
#
# * ``could not be introspected`` / ``caller is not authorised`` — inference-router's
#   own 401 sentence (``inference-router/src/security.py``: "Caller licence could not
#   be introspected; caller is not authorised"). Observed live 2026-10-17.
# * ``ambiguous_bundle`` — licences' 409 code when the tenant holds two bundles that
#   both reach inference-router; WM drops the token and the call 401s downstream.
# * ``error code: 401`` / ``error code: 403`` — the openai SDK's ``APIStatusError``
#   rendering; ``status_code: 401`` / ``status_code: 403`` — pydantic-ai's
#   ``ModelHTTPError`` rendering. These are the two shapes WM's agent path can wrap.
#
# Deliberately NOT a bare "401"/"unauthorized": a 5xx whose prose merely MENTIONS a
# number is not evidence, and a false-permanent kills a paid run (bias to transient).
UPSTREAM_AUTH_5XX_MARKERS = (
    "could not be introspected",
    "caller is not authorised",
    "ambiguous_bundle",
    "error code: 401",
    "error code: 403",
    "status_code: 401",
    "status_code: 403",
)

# Substrings that mark a **422** body as model SAMPLING non-determinism rather than a bad
# request. WM classifies pydantic-ai's "Exceeded maximum output retries (3)" as a
# deterministic failure and answers 422, but it is a bad DRAW, not a bad REQUEST: the model
# lost structured-output validation three times at temperature 0.1. Prod refutes
# "deterministic" — ``demo_video_footage_direction`` is 103 completed vs 3 failed, including
# a success on an identical two-clip input shape hours before run
# 01a00a4e-22ad-74dc-9c53-3e39fd0a6094 died on it in EIGHT SECONDS having spent zero of its
# configured ``llm_retries``. Same request, different draw — exactly what a retry is for,
# and what ``classify``'s "bias to TRANSIENT when unsure" is for.
#
# ⚠️ Keep this list minimal and keep the escape scoped to 422. A 4xx is normally OUR bug;
# retrying a real contract fault burns money and hides the defect. Do NOT "correct" this
# back into PERMANENT_STATUSES, and do NOT widen it to 4xx in general.
#
# ⚠️ KNOWN OVER-REACH, accepted deliberately: this marker is a SUPERSET of the sampling
# case. pydantic-ai does not inspect ``finish_reason`` before re-asking, so a reply that
# was TRUNCATED at ``max_tokens`` mid-JSON is indistinguishable from a malformed one and
# surfaces as this same sentence. That half IS deterministic, and we now pay one extra
# attempt on a guaranteed loss. Accepted because the alternative -- keeping it permanent
# -- discards a whole PAID run on a recoverable draw, which is the strictly worse trade at
# a measured 103:1 success ratio. The real cure is an explicit ``max_tokens`` for
# ``footage-director`` (it currently inherits DEFAULT_SELF_HOSTED_MAX_TOKENS, itself raised
# 4000->6000 for this very worker); do that and this over-reach shrinks to nothing.
TRANSIENT_4XX_MARKERS = ("maximum output retries",)

# ⚠️ NO STRUCTURAL FIELD EXISTS UPSTREAM — verified 2026-08-19 against
# ``nonagenticai/worker_management`` @ 2e3e131. This is why the marker above is prose.
#
# WM's agent branch raises a bare ``HTTPException(status_code=422, detail=str(e))``
# (``src/api/workers/message_endpoints.py:826-827``) and its generic handler stamps
# ``error.code`` purely from a status→name map (``src/api/error_handlers.py:198-214``:
# ``422: "VALIDATION_ERROR"``). So a model that lost the structured-output draw and a
# genuinely malformed request arrive with the SAME ``error.code``. There is no ``type``,
# ``kind`` or domain ``code`` anywhere in the envelope to key on. Do not invent one here;
# the real cure is a WM-side domain code on that raise.
#
# What DOES exist is a checkable relationship. WM answers 422 from that branch **only**
# when its own ``_is_deterministic_failure`` matched, and it matches by substring against
# the very ``str(e)`` it then puts in ``detail`` — so any 422 WM produces this way is
# guaranteed to contain one of the strings below. Vendored verbatim from
# ``worker_management/src/api/workers/_helpers.py::_DETERMINISTIC_FAILURE_MARKERS``:
WM_DETERMINISTIC_FAILURE_MARKERS = (
    # n=3136. The prompt leaves no room for output under this model's max_tokens.
    # GENUINELY deterministic — deliberately NOT rescued by TRANSIENT_4XX_MARKERS.
    "exceeded before any response was generated",
    # n=211. The agent exhausted its own structured-output retry budget. A bad DRAW.
    # This is the one TRANSIENT_4XX_MARKERS must keep covering.
    "exceeded maximum output retries",
)

# The exact sentences WM has been observed to emit for the rescued class. Pinned so a
# reword upstream breaks CI HERE, loudly, instead of silently killing a paid run in prod.
# If one of these fails: WM's marker tuple changed → re-vendor it and re-derive
# TRANSIENT_4XX_MARKERS. Do NOT delete the failing case.
WM_OBSERVED_TRANSIENT_422_MESSAGES = (
    "Exceeded maximum output retries (3)",
    "Exceeded maximum output retries (0)",
)


def _attributed[E: _AttributedUpstreamError](error: E, fault: FailureClass) -> E:
    """Stamp ``fault`` on ``error`` and return it, so callers stay one-liners."""
    error.failure_class = fault
    return error


def _human(exc: httpx.HTTPStatusError) -> str:
    """Extract the innermost human sentence from WM's error envelope.

    WM's envelope is DOUBLE-NESTED for structured errors::

        {"error": {"code": "ERROR",
                   "message": {"code": "byo_llm_not_configured",
                               "message": "Configure your LLM provider in Settings ..."},
                   "status": 402}}

    ...but flat for internal ones::

        {"error": {"code": "INTERNAL_ERROR", "message": "Connection error.", "status": 500}}

    The body may also not be JSON at all (proxy/ingress HTML). Parse defensively and fall
    back to raw text. The status code + request URL stay in the returned string so the
    run's ``error`` field remains diagnosable.
    """
    status = exc.response.status_code
    try:
        payload = exc.response.json()
    except ValueError, UnicodeDecodeError:
        payload = None

    detail = _detail_from_payload(payload) if isinstance(payload, dict) else ""
    if not detail:
        detail = (exc.response.text or "").strip()[:500] or exc.response.reason_phrase

    return f"upstream {status} from {exc.request.url}: {detail}"


def _detail_from_payload(payload: dict[str, Any]) -> str:
    """Dig the innermost human sentence out of a parsed WM error body."""
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, dict):
            inner = message.get("message")
            return str(inner) if inner else str(message.get("code") or "")
        if message is not None:
            return str(message)
        if error.get("code"):
            return str(error["code"])
    elif isinstance(error, str):
        return error

    for key in ("detail", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _body_text(exc: httpx.HTTPStatusError) -> str:
    try:
        return exc.response.text or ""
    except ValueError, UnicodeDecodeError:  # pragma: no cover - defensive
        return ""


def _envelope_detail(exc: httpx.HTTPStatusError) -> str:
    """The innermost sentence WM *asserted*, never text it merely echoed back.

    Match markers against this, NOT the raw body. A genuine FastAPI 422 embeds the
    offending value under ``detail[].input``, so raw-body matching lets a run whose own
    script says "maximum output retries" flip its own validation error to transient --
    reproduced in review. ``_detail_from_payload`` reads WM's ``error.message`` envelope
    and returns "" when ``detail`` is a list, which is exactly the FastAPI shape, so the
    echoed value can never reach a marker.
    """
    try:
        payload = exc.response.json()
    except ValueError, UnicodeDecodeError:  # pragma: no cover - defensive
        return ""
    return _detail_from_payload(payload) if isinstance(payload, dict) else ""


def classify(exc: httpx.HTTPStatusError) -> Exception:
    """Return (do NOT raise) a Permanent or Transient error for an upstream HTTP status.

    Bias to TRANSIENT when unsure: a false-permanent kills a 20-minute run outright,
    while a false-transient costs at most a couple of extra upstream calls.
    """
    status = exc.response.status_code
    fault = fault_for_status(status)

    if status in TRANSIENT_STATUSES:
        return _attributed(TransientUpstreamError(_human(exc)), fault)

    # Checked BEFORE PERMANENT_STATUSES: 422 is in that set, and only a marker-matched
    # body escapes it. Same matching convention as PERMANENT_5XX_MARKERS — lowercase
    # markers, plain substring, against the lowercased raw body.
    if status == 422:
        detail = _envelope_detail(exc).lower()
        matched = next((m for m in TRANSIENT_4XX_MARKERS if m in detail), None)
        if matched is not None:
            # LOUD ON PURPOSE. This prose match is the only thing standing between a
            # recoverable bad model draw and a dead paid run, and it is coupled to a
            # sentence w2v does not own. Log it at WARNING so we can SEE the fallback
            # working (and see it stop working) in prod, per-run, without a deploy.
            log.warning(
                "upstream.422_rescued_by_prose_marker marker=%r detail=%r url=%s",
                matched,
                detail[:200],
                exc.request.url,
            )
            # FAULT: ``external``, NOT the ``fault_for_status(422)`` = ``unknown``
            # this branch used to carry. 422 is unattributable *from the status
            # alone* — it means both "our request was malformed" and "the model
            # lost its draw" — which is exactly why it is absent from
            # ``_STATUS_FAULT``. But reaching HERE is a strictly stronger signal
            # than the status: WM has ASSERTED, in its own error envelope, that
            # this was a sampling non-determinism (``TRANSIENT_4XX_MARKERS``), so
            # the "could be our malformed request" reading is already excluded.
            # A dependency that could not produce a valid answer to a request it
            # understood is ``external`` by ``failure_class``'s own definition —
            # the identical verdict, for the identical reason, that
            # ``classify_transport`` stamps a request the server never saw.
            #
            # ⚠️ SCOPED TO THIS BRANCH ON PURPOSE. The two axes are still not
            # collapsed: this is not "transient ⇒ external". The permanent 422
            # below stays ``unknown``, and so does the bias-to-transient
            # fallthrough at the end of this function — those are transient/
            # permanent verdicts made when UNSURE, and a wrong attribution is
            # worse than an absent one. Only the marker-matched branch, where the
            # ambiguity has been positively resolved by the upstream itself,
            # earns a fault.
            return _attributed(TransientUpstreamError(_human(exc)), EXTERNAL)

    if status in AUTH_STATUSES:
        return _attributed(UpstreamAuthError(_human(exc)), fault)

    if status in PERMANENT_STATUSES:
        if status == 422:
            # The other side of the same coin: an unrescued 422 ends the run with ZERO
            # retries. If WM rewords the sampling-failure sentence this is where the
            # money burns, so make it visible rather than indistinguishable from a real
            # contract fault.
            log.warning(
                "upstream.422_classified_permanent detail=%r url=%s",
                _envelope_detail(exc)[:200],
                exc.request.url,
            )
        return _attributed(PermanentUpstreamError(_human(exc)), fault)

    if status >= 500:
        body = _body_text(exc).lower()
        if any(marker in body for marker in PERMANENT_5XX_MARKERS):
            return _attributed(PermanentUpstreamError(_human(exc)), fault)
        if any(marker in body for marker in UPSTREAM_AUTH_5XX_MARKERS):
            # LOUD: a 5xx promoted to permanent on prose. If the upstream rewords its
            # sentence this branch silently stops firing (and we are back to burning
            # the retry budget) — the WARNING makes both states visible per run.
            log.warning(
                "upstream.5xx_wrapped_auth_classified_permanent status=%s url=%s",
                status,
                exc.request.url,
            )
            # FAULT unchanged (``fault_for_status`` → ``external`` for a 5xx): a
            # licence/credential refusal between two platform services is neither
            # this run's code path nor the customer's input. The retry verdict is
            # what changes; the two axes stay independent.
            return _attributed(UpstreamAuthError(_human(exc)), fault)

    return _attributed(TransientUpstreamError(_human(exc)), fault)


def classify_transport(exc: httpx.RequestError) -> Exception:
    """Name a transport failure from the SAME taxonomy as ``classify``.

    A connect/read/write/pool error never reached the application, so it says
    nothing about whether the request would succeed on a retry — it is transient
    by construction. There is no permanent branch here on purpose: inventing one
    would be guessing about a request the server never saw.

    This exists because the four names in the failure record
    (``HTTPStatusError``, ``TransientUpstreamError``, ``PermanentUpstreamError``,
    ``ReadError``) were a function of WHICH CALL SITE the failure came through,
    not of what went wrong. ``httpx.ReadError`` in particular was classified
    nowhere in this package, so identical upstream trouble was unaggregatable.

    Retry behaviour is deliberately unchanged: ``httpx.TransportError`` is
    already in ``llm_budget.LLM_RETRY_ON`` and ``TransientUpstreamError`` is too,
    so an LLM stage retries exactly as many times as before. Only the name moves.
    """
    # A request the server never saw is EXTERNAL by construction: it says nothing
    # about our code path and nothing about the customer's input, only that a
    # dependency was unreachable.
    return _attributed(TransientUpstreamError(describe_exception(exc)), EXTERNAL)


async def guarded[R](factory: Callable[[], Awaitable[R]]) -> R:
    """Await ``factory()``, translating upstream failures into Permanent/Transient.

    Catches ``HTTPStatusError`` (the server answered with a status) and
    ``TransportError`` (it never answered). Everything else propagates untouched
    — ``ValueError`` re-prompts, ``RuntimeError`` is poll_job's terminal-job
    signal, cancellation keeps its meaning.

    ``RequestError`` is caught SECOND and separately rather than folded into one
    ``except`` tuple: the two carry different information and must not be given a
    single message shape. Note ``httpx.HTTPStatusError`` is NOT a subclass of
    ``RequestError`` (both derive from ``HTTPError``), so the order of these two
    handlers is not load-bearing — but keeping them distinct is.

    ``httpx.RequestError``, NOT ``httpx.TransportError``: ``TransportError`` is only
    a MIDDLE layer of httpx's hierarchy (``RequestError -> TransportError ->
    NetworkError -> ReadError``). Two of ``RequestError``'s children sit OUTSIDE it
    — ``DecodingError`` (a corrupt / truncated ``Content-Encoding`` body, which
    istio + cloudflared in front of every ``.test`` hop can produce) and
    ``TooManyRedirects``. Both escaped this guard entirely: they reached the run
    failure classifier unstamped, so ``attribute_exception`` returned ``UNKNOWN``
    (exactly the unaggregatability the ``classify_transport`` docstring above
    claims to have fixed for ``ReadError``), and neither is in
    ``llm_budget.LLM_RETRY_ON`` nor a ``ValueError``, so a stage got ZERO retries on
    a failure that is transient by construction — the application never delivered a
    body. ``RequestError`` is the boundary that docstring's own reasoning ("it never
    answered") actually describes.
    """
    try:
        return await factory()
    except httpx.HTTPStatusError as exc:
        raise classify(exc) from exc
    except httpx.RequestError as exc:
        raise classify_transport(exc) from exc
