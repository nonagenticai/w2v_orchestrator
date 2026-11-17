"""The design stages must actually ROUTE through the taxonomy — not merely have one.

Why this file exists separately from ``test_upstream_transport_and_bypass.py``
-----------------------------------------------------------------------------
That file proves ``guarded()``/``classify_transport()`` behave correctly. It does
NOT prove any stage calls them. Verified by mutation: reverting
``web_design_v2/design.py`` to its unguarded form with the default
``retry_on=(Exception,)`` left **1945 tests passing**. The whole fix could have
shipped inert at exactly the line it was written for.

So these tests drive the real stage handlers through a faked client and assert the
two things the bug was actually about:
  * the exception a caller SEES is classified, not a raw ``httpx`` type, and
  * a permanent failure costs exactly ONE upstream call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from w2v_common.stages.errors import StageRetriesExhaustedError

from src.services.runs.run_context import RunContext
from src.services.stages.upstream import PermanentUpstreamError
from src.services.stages.web_design_v2.design import WebDesignV2Stage

_URL = "http://html-render.w2v:8080/api/designs"


def _http_error(status: int, body: str = "") -> httpx.HTTPStatusError:
    request = httpx.Request("GET", _URL)
    response = httpx.Response(status, text=body, request=request)
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


def _ctx() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": {"description": "a landing page for Acme"}, "title": "Acme"},
        prior_run_outputs={},
    )


async def test_design_start_401_is_recorded_under_a_classified_name() -> None:
    """MUTATION: remove ``guarded`` from the ``design.start`` call site.

    ``design.start`` runs with ``retries=0``, so ``stage_call`` always wraps the
    cause in ``StageRetriesExhaustedError`` — the classification therefore shows up
    in the recorded MESSAGE rather than the raised type. That message is exactly
    what the failure records aggregate on, so it is the thing worth pinning:
    unguarded it reads ``HTTPStatusError``, guarded it reads ``UpstreamAuthError``
    (the ``PermanentUpstreamError`` subclass a 401/403 classifies to since the
    UI review 2026-10-17).
    """
    client = AsyncMock()
    client.start_design.side_effect = _http_error(401, "license token expired")

    with pytest.raises(StageRetriesExhaustedError) as caught:
        await WebDesignV2Stage(client=client).run({}, ctx=_ctx())

    assert "UpstreamAuthError" in str(caught.value)
    assert "HTTPStatusError" not in str(caught.value)


async def test_design_start_readerror_is_recorded_under_a_classified_name() -> None:
    """MUTATION: remove the ``TransportError`` arm, or drop ``guarded`` here.

    ``ReadError`` was the name that appeared in production with no classifier
    anywhere in the package; it must now arrive as a ``TransientUpstreamError``
    with the original cause still legible.
    """
    client = AsyncMock()
    client.start_design.side_effect = httpx.ReadError("connection reset")

    with pytest.raises(StageRetriesExhaustedError) as caught:
        await WebDesignV2Stage(client=client).run({}, ctx=_ctx())

    assert "TransientUpstreamError" in str(caught.value)
    assert "ReadError" in str(caught.value)  # cause preserved, not erased


async def test_design_result_401_costs_exactly_one_upstream_call() -> None:
    """THE call-site regression: ``design.result`` used ``retry_on=(Exception,)``.

    MUTATION: restore that default (or drop ``guarded`` here) and the call count
    goes to 2 — the expired-licence 401 retried as though it were a blip, then
    reported as "retries exhausted" with the real cause buried.
    """
    client = AsyncMock()
    client.start_design.return_value = "job-1"
    client.wait_for_design.return_value = type("S", (), {"accessibility_score": None})()
    client.get_design_result.side_effect = _http_error(401, "license token expired")

    with pytest.raises(PermanentUpstreamError):
        await WebDesignV2Stage(client=client).run({}, ctx=_ctx())

    assert client.get_design_result.await_count == 1


async def test_design_result_503_still_retries() -> None:
    """The mirror: narrowing the 401 must not stop genuine blips retrying.

    MUTATION: put ``PermanentUpstreamError`` into ``LLM_RETRY_ON``, or classify
    5xx as permanent, and this drops to a single call.
    """
    client = AsyncMock()
    client.start_design.return_value = "job-1"
    client.wait_for_design.return_value = type("S", (), {"accessibility_score": None})()
    client.get_design_result.side_effect = _http_error(503, "upstream busy")

    with pytest.raises(Exception):  # noqa: B017 - StageRetriesExhaustedError or the classified type
        await WebDesignV2Stage(client=client).run({}, ctx=_ctx())

    assert client.get_design_result.await_count > 1


async def test_design_wait_terminal_failure_still_propagates_on_first_attempt() -> None:
    """``design.wait`` keeps ``retry_on=()``; ``guarded`` must not swallow RuntimeError.

    MUTATION: widen ``guarded`` to catch ``Exception`` and a server-marked-failed
    job gets re-polled until the budget is gone — the corpse-polling bug.
    """
    client = AsyncMock()
    client.start_design.return_value = "job-1"
    client.wait_for_design.side_effect = RuntimeError("job failed: verification rejected")

    with pytest.raises(RuntimeError):
        await WebDesignV2Stage(client=client).run({}, ctx=_ctx())

    assert client.wait_for_design.await_count == 1
