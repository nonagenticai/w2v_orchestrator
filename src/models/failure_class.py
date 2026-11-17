"""Fault attribution for a failed run: whose problem is it?

``runs`` carries ``status`` plus a free-text ``error`` and nothing else, so every
question of the form "did WE break this run, or did the customer?" has to be
re-derived by hand from prose. Measured 2026-09-01: bucketing the 72 failed runs
inside retention needed an ad-hoc regex over ``runs.error`` and still left 36 of
the 72 unclassifiable. This module is the taxonomy that makes that a column read.

⚠️ This is a DIFFERENT AXIS from ``src/services/stages/upstream.py::classify``.
That one splits PERMANENT vs TRANSIENT — *can a retry fix it?* — and it is the
right axis for the retry budget. Fault attribution is orthogonal: our own 500 and
a customer's 400 are both PERMANENT, and a customer's misconfigured LLM provider
is permanent while an inference-router blip is transient, yet neither of those
pairs shares a fault. The two must never be collapsed into one enum. What they DO
share is a call site: ``classify`` is where the HTTP status is in hand, so it
stamps ``failure_class`` on the error it returns (see ``upstream.fault_for_status``)
and this module reads it back off the exception at the persistence site.

⛔ NOT A BILLING FEATURE. There is deliberately no ``billable`` flag, no refund
path and no money in this module. It exists so a human can answer "ours or
theirs?" in one query instead of an afternoon.

The cardinal rule: **a wrong attribution is worse than an absent one.** A bucket
nobody trusts is a bucket nobody uses, and a run mis-filed as ``our_defect``
sends an engineer hunting a bug that does not exist. So every mapping here is an
ALLOWLIST keyed on a structural signal (an exception type, an HTTP status), never
on prose, and every unmatched failure returns ``unknown``.

This module is a LEAF, and lives in ``src/models/`` for the same reason
``showcase_line`` does: the API schemas, the database repository and the
dispatcher all depend on this vocabulary, and none of them may depend on each
other. Define the Literal here; ``api.schemas._shared`` re-exports it under the
name API readers expect, so the two can never drift.
"""

from __future__ import annotations

from typing import Final, Literal

import httpx

# ⚠️ Must stay in EXACT parity with the DB's ``ck_runs_failure_class`` CHECK
# (``Run.__table_args__`` + migration 0066). A DB-only value 500s every read that
# validates it (``RunDetail.model_validate``); a schema-only value is a dead
# branch the DB rejects on write. Guarded by
# ``tests/contracts/test_failure_class_check_constraint_parity.py`` — widen the
# CHECK, this Literal and the model predicate in ONE PR, or that guard fails.
#
# This estate has already shipped that exact bug once on the other axis:
# migration 0059 widened ``ck_stage_outputs_status`` to admit ``'cancelled'``,
# ``StageStatus`` was never widened, and every read of an affected run 500'd.
FailureClass = Literal["our_defect", "customer_input", "external", "unknown"]

#: The fault is in code or configuration WE own and can fix by shipping a change.
#: Reached only from signals that cannot mean anything else — a Python
#: programming error raised inside our own process, or a run that reported every
#: stage terminal-good and then had no deliverable in storage.
OUR_DEFECT: Final[FailureClass] = "our_defect"

#: The fault is in tenant-supplied input, configuration or account state. No code
#: change we could ship would make this run succeed; the customer has to act.
CUSTOMER_INPUT: Final[FailureClass] = "customer_input"

#: A dependency outside this process failed at runtime — it 5xx'd, rate-limited
#: us, or never answered at all. It earns its own member rather than folding into
#: ``our_defect`` because infrastructure flake and a shipped bug demand completely
#: different responses, and merging them would inflate the defect count with noise
#: until nobody read it. (Some of those dependencies are ours to *operate*; none
#: of them are this run's code path, which is what is being attributed.)
EXTERNAL: Final[FailureClass] = "external"

#: Not attributable from the evidence in hand. The DEFAULT everywhere, and a
#: first-class answer rather than a failure of the taxonomy: it is the honest
#: label for the 36-of-72 that no regex could place, and it is what keeps the
#: other three members trustworthy.
UNKNOWN: Final[FailureClass] = "unknown"

#: The whole vocabulary as a runtime tuple. Spelled out rather than derived with
#: ``get_args`` so ``ty`` keeps each element at its literal type (``get_args``
#: erases to ``tuple[Any, ...]``, which would silently un-type every membership
#: narrowing below). The parity guard asserts this tuple equals
#: ``get_args(FailureClass)`` AND equals the DB CHECK, so a member added to one
#: and not the others fails there.
FAILURE_CLASSES: Final[tuple[FailureClass, ...]] = (
    OUR_DEFECT,
    CUSTOMER_INPUT,
    EXTERNAL,
    UNKNOWN,
)

# Exception types that can only be reached by a defect in code we wrote. A
# ``TypeError``/``AttributeError``/``NameError`` in our own process is never
# something a customer can type into a brief and never something an upstream
# service can hand us — it is a line of Python that is wrong. ``KeyError`` and
# ``IndexError`` are included on the same reasoning even when the missing key
# came from tenant data: failing to validate or default tenant input before
# indexing it IS the defect.
#
# ⚠️ Deliberately NOT here: ``ValueError`` (the re-prompt signal on the stage
# paths, and pydantic's validation error base — routinely raised BY tenant data),
# ``RuntimeError`` (``StageError``/``StageRetriesExhaustedError`` extend it, and
# those wrap a cause whose own attribution is what matters), and every kind of
# ``TimeoutError`` (a budget we set too small and a dependency that hung are
# indistinguishable from the type alone — see ``_EXTERNAL_TYPES`` below). Each of
# those would be a guess, so each one lands in ``unknown``.
_OUR_DEFECT_TYPES: Final[tuple[type[BaseException], ...]] = (
    TypeError,
    AttributeError,
    NameError,
    UnboundLocalError,
    IndexError,
    KeyError,
    ZeroDivisionError,
    AssertionError,
    NotImplementedError,
)

# Exception types that mean "a dependency outside this process did not deliver a
# response", independent of WHERE they were raised.
#
# WHY this exists, measured: ``upstream.guarded`` already converts these into a
# stamped ``TransientUpstreamError`` — but only on the calls that go THROUGH it.
# Every httpx call that does not (a ``w2v_common`` client raising its own
# transport error, a stage calling httpx directly) delivered the exception to the
# persistence site unstamped, where an ``httpx.ReadError`` matched nothing and
# came back ``unknown``. That is 11 of the 47 armed-era failures — the largest
# family in the taxonomy that is NOT genuinely ambiguous.
#
# This is the SAME boundary and the SAME verdict as
# ``upstream.classify_transport``: ``httpx.RequestError`` is the class of "the
# application never delivered a body", which says nothing about our code path and
# nothing about the customer's input, only that a dependency was unreachable.
# Deliberately keeping the two identical means a call that missed ``guarded``
# attributes exactly as it would have inside it — the attribution stops being a
# function of which call site the failure came through, which was the whole point
# of ``classify_transport``.
#
# The builtin ``ConnectionError`` (``ConnectionReset``/``Refused``/``BrokenPipe``)
# is included on the same reasoning for the non-httpx clients (redis, psycopg).
#
# ⚠️ Deliberately NOT here: ``OSError`` at large (a missing local file is ours),
# ``socket.gaierror`` (a name that does not resolve is usually OUR config), and
# every ``TimeoutError`` — including ``httpx.TimeoutException``, which is a
# ``TransportError`` but is excluded explicitly below. A timeout cannot be
# attributed from its type: our per-attempt budget being too small and a
# dependency hanging produce the identical object. Those stay ``unknown``.
_EXTERNAL_TYPES: Final[tuple[type[BaseException], ...]] = (
    httpx.RequestError,
    ConnectionError,
)

#: Subtracted from ``_EXTERNAL_TYPES``. Checked FIRST, so a subclass listed here
#: falls through to ``unknown`` rather than inheriting ``external`` from its base.
_NOT_EXTERNAL_TYPES: Final[tuple[type[BaseException], ...]] = (
    httpx.TimeoutException,
    # A URL this process constructed with a scheme httpx cannot speak is our
    # configuration, not a dependency failing. It is a ``TransportError``, so it
    # would otherwise be swept up as ``external``.
    httpx.UnsupportedProtocol,
)

# ── Fault attribution for an HTTP status ────────────────────────────────────
#
# Lives HERE rather than in ``services.stages.upstream`` (which re-exports it for
# import-path stability) so that a raw ``httpx.HTTPStatusError`` reaching the
# persistence site — one that never passed through ``guarded``, and therefore
# carries no stamp — is attributed from the status it is still carrying, by the
# same map that would have stamped it. ``models`` is a leaf and ``upstream``
# already imports from it, so the dependency can only point this way.
#
# Statuses whose fault is UNAMBIGUOUS. Everything absent from this map, and every
# 4xx not covered below, is ``unknown`` ON PURPOSE — a wrong attribution is worse
# than an absent one, so this is an allowlist of the cases that can only mean one
# thing, never a best-guess partition of the status space.
#
# ⚠️ Resist widening this by intuition. Each entry below is a case that CANNOT be
# read the other way; the ones deliberately left out are left out because they
# genuinely can:
#   * 400 — could equally be OUR malformed request or THEIR malformed brief.
#   * 401/403 — S2S auth is our config, but a licence/quota refusal on the same
#     status is the tenant's account state. Two faults, one code.
#   * 404 — our wrong URL, or their missing resource.
#   * 422 — WM answers 422 both for a real contract fault (ours) and for a lost
#     structured-output draw (neither). See ``upstream.TRANSIENT_4XX_MARKERS``.
# All four land in ``unknown``, which is the honest answer.
_STATUS_FAULT: Final[dict[int, FailureClass]] = {
    # WM ``byo_llm_not_configured``: "Configure your LLM provider in Settings
    # before running agents." The tenant has no provider. No change we can ship
    # makes this run succeed — only the customer can act. The single documented
    # unambiguous customer case on this path.
    402: CUSTOMER_INPUT,
}

#: 4xx that are retryable (timeout, contention, too-early, rate limit). Mirrored
#: from ``upstream.TRANSIENT_STATUSES``, which imports this name rather than
#: defining its own — a dependency that is busy, contended or rate-limiting us is
#: neither our code nor their input.
TRANSIENT_STATUSES: Final[frozenset[int]] = frozenset({408, 409, 425, 429})


def fault_for_status(status: int) -> FailureClass:
    """Attribute an upstream HTTP status on the FAULT axis.

    Not the retryability axis — see ``upstream.classify``. The overlap is
    coincidental and partial: ``TRANSIENT_STATUSES`` all happen to be
    ``external``, but 402 is permanent AND ``customer_input`` while a permanent
    500 is also ``external``, so neither axis can be computed from the other.

    5xx is ``external`` rather than ``our_defect`` even for services this org
    operates: what is being attributed is THIS run's code path, and "a service we
    called returned 500" is a materially different finding from "the orchestrator
    has a bug". Folding infrastructure flake into ``our_defect`` would inflate the
    defect count with noise until the bucket stopped being read.
    """
    mapped = _STATUS_FAULT.get(status)
    if mapped is not None:
        return mapped
    if status in TRANSIENT_STATUSES or status >= 500:
        return EXTERNAL
    return UNKNOWN


# A cause chain longer than this is pathological; the bound also makes a
# self-referential ``__cause__`` impossible to loop on (belt to the seen-set's
# braces).
_MAX_CHAIN = 16


def _chain(exc: BaseException) -> list[BaseException]:
    """``exc`` followed by its ``__cause__`` links, deduped and bounded.

    Only ``__cause__`` (explicit ``raise ... from ...``) is walked, never
    ``__context__``. ``__context__`` is set implicitly by ANY raise inside an
    ``except`` block, so it routinely links a failure to a completely unrelated
    exception that merely happened to be in flight — attributing off that would
    manufacture exactly the wrong answers this module exists to avoid.
    ``upstream.guarded`` raises its classified errors with an explicit ``from``,
    which is what makes the cause chain load-bearing here.
    """
    out: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and len(out) < _MAX_CHAIN and id(current) not in seen:
        seen.add(id(current))
        out.append(current)
        current = current.__cause__
    return out


def attribute_exception(exc: BaseException) -> FailureClass:
    """Attribute ``exc`` on the fault axis, or ``unknown`` if it cannot be.

    Four passes over the cause chain, in this order and for this reason:

    1. **An explicit ``failure_class`` attribute wins**, anywhere in the chain.
       That attribute is stamped where the evidence actually was — ``upstream``
       has the HTTP status, we do not — so it must outrank any inference made
       here from the exception's type. ``unknown`` attributes are skipped rather
       than accepted, so a wrapper that could not attribute itself does not veto
       a definite answer from its cause.
    2. **A programming-error type means ``our_defect``**, from ``_OUR_DEFECT_TYPES``
       only. Ranked above the two evidence passes below because a ``KeyError``
       raised while handling a dependency's answer is still OUR bug: the
       dependency answered, we mishandled it.
    3. **An unstamped ``httpx.HTTPStatusError`` is attributed from the status it
       is still carrying**, by the same ``fault_for_status`` map ``guarded``
       would have used. Reached only when the call did NOT go through
       ``upstream.guarded`` — a stamped error is already settled by pass 1.
    4. **A transport-layer type means ``external``**, from ``_EXTERNAL_TYPES``
       minus ``_NOT_EXTERNAL_TYPES``.

    Anything else returns ``unknown``. There is no fifth pass and no prose
    matching: the function is total, and every path that is not an explicit
    allowlist hit falls through to the ``return UNKNOWN`` at the bottom.
    """
    chain = _chain(exc)
    for pass_ in (_by_stamp, _by_our_defect_type, _by_http_status, _by_external_type):
        found = pass_(chain)
        if found is not None:
            return found
    return UNKNOWN


def _by_stamp(chain: list[BaseException]) -> FailureClass | None:
    """Pass 1 — an explicit ``failure_class`` attribute anywhere in the chain."""
    for link in chain:
        stamped = getattr(link, "failure_class", None)
        # Compared against the members rather than tested with ``in``: an
        # arbitrary exception's attribute is untyped, and matching it to a known
        # member is what narrows it back to ``FailureClass`` without an ignore.
        for member in FAILURE_CLASSES:
            if stamped == member and member != UNKNOWN:
                return member
    return None


def _by_our_defect_type(chain: list[BaseException]) -> FailureClass | None:
    """Pass 2 — a Python programming error in our own process."""
    for link in chain:
        if isinstance(link, _OUR_DEFECT_TYPES):
            return OUR_DEFECT
    return None


def _by_http_status(chain: list[BaseException]) -> FailureClass | None:
    """Pass 3 — an UNSTAMPED ``HTTPStatusError``, read by the status it carries."""
    for link in chain:
        if isinstance(link, httpx.HTTPStatusError):
            fault = fault_for_status(link.response.status_code)
            # ``unknown`` does NOT stop the walk: an ambiguous 4xx must not veto
            # a transport failure deeper in the same chain.
            if fault != UNKNOWN:
                return fault
    return None


def _by_external_type(chain: list[BaseException]) -> FailureClass | None:
    """Pass 4 — a transport-layer type, minus the explicitly subtracted ones."""
    for link in chain:
        if isinstance(link, _NOT_EXTERNAL_TYPES):
            continue
        if isinstance(link, _EXTERNAL_TYPES):
            return EXTERNAL
    return None


def failure_detail_for(exc: BaseException) -> dict[str, str] | None:
    """The structured what-to-do a failure carries, or ``None``.

    Read off the FIRST link of the ``__cause__`` chain (same walk as
    :func:`attribute_exception`) that exposes a ``failure_detail`` mapping of
    string keys/values with a non-empty ``code`` — e.g.
    ``SlotNotConfiguredError``'s ``{"code": "byo_provider_required",
    "modality": "tts", "message": "Configure your own TTS provider to continue"}``.
    Structural, never prose: an exception without the attribute yields ``None``.
    """
    for link in _chain(exc):
        detail = getattr(link, "failure_detail", None)
        if not isinstance(detail, dict):
            continue
        clean = {k: v for k, v in detail.items() if isinstance(k, str) and isinstance(v, str)}
        if clean.get("code"):
            return clean
    return None
