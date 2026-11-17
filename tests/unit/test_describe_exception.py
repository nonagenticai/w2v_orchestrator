"""Regression: a run/stage error message never ends in an empty ``"Type: "``.

Live 2026-10-18, Marketing scorecard run 01a14efb-3f75…: the stage error read
``StageRetriesExhaustedError: stage 'marketing_scorecard_scan.score': retries
exhausted after 2 attempts: ReadTimeout:`` — nothing after the colon, because
``str(httpx.ReadTimeout(""))`` is empty and both ``w2v_common``'s
``StageRetriesExhaustedError`` and our own ``f"{type(exc).__name__}: {exc}"``
interpolate it verbatim.
"""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
from w2v_common.stages import StageRetriesExhaustedError

from src.services.blueprints.base import DAGBlueprint, StageSpec
from src.services.runs.dag_executor import DAGExecutor
from src.services.runs.run_context import RunContext
from src.services.stages.base import StageHandler, StageInput, StageOutput
from src.utils import describe_exception, exception_detail


def _read_timeout(read_seconds: float = 30.0) -> httpx.ReadTimeout:
    """An empty-message ReadTimeout carrying the request httpx really attaches."""
    client = httpx.Client(timeout=httpx.Timeout(10.0, read=read_seconds))
    request = client.build_request(
        "POST", "http://analysis-service.w2v:6303/api/scorecard?token=s3cr3t"
    )
    client.close()
    return httpx.ReadTimeout("", request=request)


def test_empty_read_timeout_names_the_budget_and_target() -> None:
    assert str(_read_timeout()) == ""  # the premise: httpx gives us nothing
    assert describe_exception(_read_timeout()) == (
        "ReadTimeout: no response within 30s from analysis-service.w2v:6303/api/scorecard"
    )


def test_target_never_includes_the_query_string() -> None:
    assert "s3cr3t" not in describe_exception(_read_timeout())


def test_retries_exhausted_with_empty_last_error_is_rebuilt() -> None:
    exc = StageRetriesExhaustedError("marketing_scorecard_scan.score", 2, _read_timeout(45))
    assert describe_exception(exc) == (
        "StageRetriesExhaustedError: stage 'marketing_scorecard_scan.score': "
        "retries exhausted after 2 attempts: ReadTimeout: no response within 45s from "
        "analysis-service.w2v:6303/api/scorecard"
    )


def test_connect_timeout_says_connect() -> None:
    request = httpx.Request(
        "GET", "http://kag.kag.svc:8900/x", extensions={"timeout": {"connect": 5}}
    )
    assert describe_exception(httpx.ConnectTimeout("", request=request)) == (
        "ConnectTimeout: could not connect within 5s to kag.kag.svc:8900/x"
    )


def test_timeout_without_request_still_has_text() -> None:
    assert describe_exception(httpx.ReadTimeout("")) == (
        "ReadTimeout: no response within the client timeout"
    )


def test_bare_timeout_error_and_blank_runtime_error() -> None:
    assert describe_exception(TimeoutError()) == "TimeoutError: timed out (no further detail)"
    assert describe_exception(RuntimeError("  ")) == "RuntimeError: raised with no error message"


def test_empty_message_surfaces_a_texted_cause() -> None:
    try:
        try:
            raise ValueError("bad payload shape")
        except ValueError as inner:
            raise RuntimeError() from inner
    except RuntimeError as exc:
        assert exception_detail(exc) == (
            "raised with no error message (caused by ValueError: bad payload shape)"
        )


def test_a_message_with_text_is_unchanged() -> None:
    exc = ValueError("boom")
    assert describe_exception(exc) == f"{type(exc).__name__}: {exc}"
    wrapped = StageRetriesExhaustedError("s", 3, exc)
    assert describe_exception(wrapped) == f"{type(wrapped).__name__}: {wrapped}"


class _NoopIn(StageInput):
    pass


class _NoopOut(StageOutput):
    value: int = 0


class _ScoreTimesOut(StageHandler):
    name = "marketing_scorecard_scan"
    input_model = _NoopIn
    output_model = _NoopOut

    async def handle(self, payload, ctx):
        raise StageRetriesExhaustedError("marketing_scorecard_scan.score", 2, _read_timeout())


@pytest.mark.asyncio
async def test_executor_persists_and_emits_a_non_empty_stage_error() -> None:
    """The stage row's error and the ``stage_failed`` event — what the run page shows."""
    persisted: list[str | None] = []
    emitted: list[str] = []

    async def emit(event, **kw):
        if event == "stage_failed":
            emitted.append(kw["error"])

    async def persist(stage_name, status, output=None, error=None):
        if status == "failed":
            persisted.append(error)

    bp = DAGBlueprint(
        id="t",
        project_type="generic",
        stages=[
            StageSpec(name="marketing_scorecard_scan", handler_factory=lambda d: _ScoreTimesOut())
        ],
    )
    executor = DAGExecutor(bp, emit=emit, persist=persist)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})
    with pytest.raises(StageRetriesExhaustedError):
        await executor.run(ctx, pause_after_stages=set())

    expected = (
        "StageRetriesExhaustedError: stage 'marketing_scorecard_scan.score': retries "
        "exhausted after 2 attempts: ReadTimeout: no response within 30s from "
        "analysis-service.w2v:6303/api/scorecard"
    )
    assert persisted == [expected]
    assert emitted == [expected]
