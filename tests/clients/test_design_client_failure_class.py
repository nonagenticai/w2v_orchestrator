"""``DesignClient.wait_for_design`` must not leak an unattributed ``RuntimeError``.

Sibling of ``tests/models/test_failure_class_real_shapes.py`` shape 2
(``VideoCompileClient`` / ``VideoCompileJobFailedError``), for the fourth live
shape that file's own docstring says it does NOT cover: "StageRetriesExhaustedError:
stage 'web_design': ... RuntimeError: design job ... failed: generate_full_page:
LLM reply is not a storable page (does not end on a closed tag (truncated))".

``BaseServiceClient.poll_job`` (the shared w2v_common helper) raises a bare,
unstamped ``RuntimeError`` on a terminal ``failed``/``error`` status —
``upstream.guarded`` documents that it lets this type through untouched — so it
reached ``models.failure_class.attribute_exception`` unstamped and came back
``unknown``. These tests drive the REAL poll loop (``httpx.MockTransport``),
not a hand-built wrapper, because the defect was that the object prod actually
raises carried no attribution — a test that constructs
``DesignJobFailedError`` directly would only prove ``attribute_exception`` can
read an attribute, never in doubt.

⚠️ NOT marked ``integration``: this repo's pytest ``addopts`` deselects that
marker, so an ``integration`` test here would never execute in CI. Everything
below is in-process and needs no service.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from w2v_common.clients.base import JobCancelledError
from w2v_common.stages import StageRetriesExhaustedError

from src.clients.design_client import DesignClient, DesignJobFailedError
from src.models.failure_class import EXTERNAL, attribute_exception

pytestmark = pytest.mark.asyncio

_IDEM_KEY = "33333333-3333-7333-8333-333333333333:web_design"


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse poll_job's retry/backoff sleeps so the poll loop runs instantly."""

    async def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _noop)


def _client_scripted(poll_steps: list[httpx.Response]) -> DesignClient:
    """GET /api/designs/jobs/{id} -> pop the next scripted poll response."""
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        step = poll_steps[min(state["i"], len(poll_steps) - 1)]
        state["i"] += 1
        return step

    client = DesignClient(base_url="http://design.test", license_token="t")
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        timeout=client.timeout,
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )
    return client


_DETAIL = "generate_full_page: LLM reply is not a storable page (does not end on a closed tag (truncated))"


async def test_terminal_failed_status_raises_an_attributable_type() -> None:
    """Drive the REAL poll loop: a terminal 'failed' job must not raise a bare RuntimeError."""
    client = _client_scripted([httpx.Response(200, json={"status": "failed", "error": _DETAIL})])
    try:
        with pytest.raises(DesignJobFailedError) as caught:
            await client.wait_for_design("job-1", timeout=30.0)
    finally:
        await client.close()

    # Still a RuntimeError, so every existing `except RuntimeError` catcher in the
    # stage layer is unaffected: this change moves attribution only, never
    # control flow.
    assert isinstance(caught.value, RuntimeError)
    # The message reaches an operator verbatim via runs.error — must not drift.
    assert str(caught.value) == f"design job job-1 failed: {_DETAIL}"


async def test_web_design_retries_exhausted_on_a_failed_job_is_external() -> None:
    """The full live chain: stage_call wrapper -> design job failure -> 'external'.

    ``external``, not ``our_defect`` or ``unknown``: html_render ran the job and
    told us it failed — the same finding, and the same verdict, as a video-compile
    job reporting terminal 'failed' (``VideoCompileJobFailedError``).
    """
    client = _client_scripted([httpx.Response(200, json={"status": "failed", "error": _DETAIL})])
    try:
        with pytest.raises(DesignJobFailedError) as caught:
            await client.wait_for_design("job-1", timeout=30.0)
    finally:
        await client.close()

    try:
        raise StageRetriesExhaustedError(
            stage_name="web_design", attempts=1, last_error=caught.value
        ) from caught.value
    except StageRetriesExhaustedError as exc:
        assert attribute_exception(exc) == EXTERNAL
        assert "design job job-1 failed" in str(exc)


async def test_cancelled_status_is_not_reattributed_as_a_failure() -> None:
    """A user-cancelled job must keep raising JobCancelledError untouched — a
    cancel is not an unattributed failure, and re-typing it here would make an
    intentional stop look like an outage."""
    client = _client_scripted([httpx.Response(200, json={"status": "cancelled"})])
    try:
        with pytest.raises(JobCancelledError):
            await client.wait_for_design("job-1", timeout=30.0)
    finally:
        await client.close()
