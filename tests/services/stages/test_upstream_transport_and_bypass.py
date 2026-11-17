"""Transport failures get a name, and the design stages stop bypassing the taxonomy.

The bug these lock down
-----------------------
Production failure records carried FOUR different names for the same upstream
trouble — ``HTTPStatusError``, ``TransientUpstreamError``, ``PermanentUpstreamError``
and ``ReadError`` — decided purely by WHICH CALL SITE the failure came through:

* ``guarded()`` caught only ``httpx.HTTPStatusError``, so ``httpx.ReadError`` was
  classified nowhere in the package (a grep for ``ReadError|ConnectError`` across
  ``src/services/stages/`` returned zero hits).
* the ``web_design`` / ``web_design_v2`` / ``course.stitch`` stages never called
  ``guarded()`` at all, so raw httpx exceptions reached ``stage_call``.
* ``design.result`` used the default ``retry_on=(Exception,)``, which retried an
  html_render **401** — an expired service licence, permanent by definition — and
  then buried it under "retries exhausted".

Fakes only, no network.
"""

from __future__ import annotations

import httpx
import pytest
from w2v_common.stages import stage_call
from w2v_common.stages.errors import StageRetriesExhaustedError

from src.services.stages.llm_budget import LLM_RETRY_ON
from src.services.stages.upstream import (
    PermanentUpstreamError,
    TransientUpstreamError,
    classify_transport,
    guarded,
)

_URL = "http://html-render.w2v:8080/api/designs"


def _http_error(status: int, body: str = "") -> httpx.HTTPStatusError:
    request = httpx.Request("GET", _URL)
    response = httpx.Response(status, text=body, request=request)
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


# ---------------------------------------------------------------------------
# Transport errors now come from the same taxonomy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadError("connection reset"),
        httpx.ConnectError("no route to host"),
        httpx.WriteError("broken pipe"),
        httpx.PoolTimeout("pool exhausted"),
    ],
)
async def test_transport_errors_are_named_transient(exc: httpx.TransportError) -> None:
    """MUTATION: delete the ``except httpx.TransportError`` arm of ``guarded``.

    Every one of these is a subclass of ``httpx.TransportError``, so a single arm
    covers the family — but only if it is actually there. Before this change the
    raw exception escaped and the failure record said "ReadError".
    """

    async def _boom() -> None:
        raise exc

    with pytest.raises(TransientUpstreamError) as caught:
        await guarded(_boom)

    # The original type survives in the message: the point is a COMMON class with
    # the specific cause still legible, not a class that erases what happened.
    assert type(exc).__name__ in str(caught.value)


async def test_transport_classification_has_no_permanent_branch() -> None:
    """A request the server never saw says nothing about a retry succeeding.

    MUTATION: add a permanent branch to ``classify_transport`` (e.g. for
    ConnectError) and this fails. Guessing "permanent" here would hard-fail a
    20-minute run over a transient network blip.
    """
    assert isinstance(classify_transport(httpx.ConnectError("x")), TransientUpstreamError)
    assert not isinstance(classify_transport(httpx.ConnectError("x")), PermanentUpstreamError)


async def test_http_status_errors_still_classify_as_before() -> None:
    """The new arm must not shadow the existing one.

    ``HTTPStatusError`` is NOT a subclass of ``TransportError`` (both derive from
    ``HTTPError``), so adding the transport arm cannot steal status failures — this
    pins that, since getting it wrong would silently make every 402/401 transient.
    """

    async def _boom() -> None:
        raise _http_error(402, "byo_llm_not_configured")

    with pytest.raises(PermanentUpstreamError):
        await guarded(_boom)


# ---------------------------------------------------------------------------
# Retry behaviour is deliberately UNCHANGED for transport failures
# ---------------------------------------------------------------------------


async def test_a_named_transport_error_still_retries_exactly_as_before() -> None:
    """``httpx.TransportError`` and ``TransientUpstreamError`` are BOTH in LLM_RETRY_ON.

    So renaming the exception must not change how many upstream calls are made.
    MUTATION: drop ``TransientUpstreamError`` from ``LLM_RETRY_ON`` and this stops
    at 1 attempt — a silent loss of resilience hiding inside a "naming" change.
    """
    attempts = 0

    async def _flaky() -> str:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadError("connection reset")

    with pytest.raises(StageRetriesExhaustedError):
        await stage_call(
            lambda: guarded(_flaky),
            timeout_seconds=5.0,
            retries=2,
            retry_on=LLM_RETRY_ON,
            backoff_initial=0.0,
            stage_name="test.transport",
        )

    assert attempts == 3  # initial + 2 retries


# ---------------------------------------------------------------------------
# The 401 that was being retried
# ---------------------------------------------------------------------------


async def test_a_401_fails_fast_and_carries_the_upstream_message() -> None:
    """THE reported defect: html_render 401 (expired licence) retried as a blip.

    MUTATION: restore ``retry_on=(Exception,)`` at the ``design.result`` call site,
    or map 401 to transient in ``PERMANENT_STATUSES`` — either way the attempt
    count goes above 1 and this fails.
    """
    attempts = 0

    async def _expired() -> str:
        nonlocal attempts
        attempts += 1
        raise _http_error(401, "license token expired")

    with pytest.raises(PermanentUpstreamError) as caught:
        await stage_call(
            lambda: guarded(_expired),
            timeout_seconds=5.0,
            retries=3,
            retry_on=LLM_RETRY_ON,
            backoff_initial=0.0,
            stage_name="design.result",
        )

    assert attempts == 1, "a permanent auth failure must not spend the retry budget"
    assert "401" in str(caught.value)


async def test_a_5xx_from_the_same_site_still_retries() -> None:
    """The mirror image: tightening 401 must not make genuine blips fail fast.

    MUTATION: classify all 5xx as permanent and this drops to 1 attempt.
    """
    attempts = 0

    async def _blip() -> str:
        nonlocal attempts
        attempts += 1
        raise _http_error(503, "upstream busy")

    with pytest.raises(StageRetriesExhaustedError):
        await stage_call(
            lambda: guarded(_blip),
            timeout_seconds=5.0,
            retries=2,
            retry_on=LLM_RETRY_ON,
            backoff_initial=0.0,
            stage_name="design.result",
        )

    assert attempts == 3


# ---------------------------------------------------------------------------
# What guarded() must NOT touch
# ---------------------------------------------------------------------------


async def test_runtime_error_passes_through_untouched() -> None:
    """``poll_job`` raises RuntimeError for a server-marked-failed job.

    ``design.wait`` keeps ``retry_on=()`` so that signal propagates on attempt 1.
    MUTATION: widen ``guarded`` to catch ``Exception`` and this fails — a dead job
    would be re-polled until the budget is gone, the exact bug retry-truth fixed.
    """

    async def _terminal() -> None:
        raise RuntimeError("job failed: verification rejected the design")

    with pytest.raises(RuntimeError) as caught:
        await guarded(_terminal)

    assert not isinstance(caught.value, (TransientUpstreamError, PermanentUpstreamError))


async def test_value_error_passes_through_untouched() -> None:
    """``primitives/script_stage.py`` raises ValueError INSIDE the budget to re-prompt."""

    async def _short_draw() -> None:
        raise ValueError("draft too short")

    with pytest.raises(ValueError):
        await guarded(_short_draw)


async def test_cancellation_is_never_reclassified() -> None:
    """A cancelled stage must stay cancelled, not become a retryable transient."""

    async def _cancelled() -> None:
        raise TimeoutError("cancelled")

    with pytest.raises(TimeoutError):
        await guarded(_cancelled)


@pytest.mark.parametrize(
    "exc",
    [
        httpx.DecodingError("gzip stream truncated"),
        httpx.TooManyRedirects("exceeded max redirects"),
    ],
)
async def test_non_transport_request_errors_are_also_named_transient(exc: httpx.RequestError):
    """``TransportError`` is only a MIDDLE layer of httpx's hierarchy.

    ``RequestError -> TransportError -> NetworkError -> ReadError``: the arm above
    catches the ``TransportError`` subtree, but two of ``RequestError``'s children
    sit OUTSIDE it. ``DecodingError`` is the reachable one — a truncated or corrupt
    ``Content-Encoding`` body, which istio + cloudflared in front of every
    ``.test`` hop can produce.

    Both escaped ``guarded`` entirely: unstamped, so ``attribute_exception``
    returned ``UNKNOWN`` (the exact unaggregatability this module claims to have
    fixed for ``ReadError``), and absent from ``llm_budget.LLM_RETRY_ON`` and not a
    ``ValueError``, so a paid stage got ZERO retries on a failure that is transient
    by construction — the application never delivered a body.

    MUTATION: narrow ``guarded``'s second arm back to ``httpx.TransportError`` and
    both parametrisations raise the raw httpx error instead.
    """
    assert not isinstance(exc, httpx.TransportError), (
        "this test is vacuous unless the exception is OUTSIDE the TransportError subtree"
    )

    async def _boom() -> None:
        raise exc

    with pytest.raises(TransientUpstreamError) as caught:
        await guarded(_boom)

    assert type(exc).__name__ in str(caught.value)
