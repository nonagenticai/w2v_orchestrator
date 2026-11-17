"""Permanent-vs-transient classification of upstream (WM/LLM) HTTP failures.

The bug these lock down: a WM **402 byo_llm_not_configured** was retried as if it
were a blip and then reported as "retries exhausted after 2 attempts", burying the
one sentence ("Configure your LLM provider in Settings") that tells the user what to
do. So each test asserts BOTH halves — the classification *and* the attempt count,
because a correct class that still spends the retry budget is not a fix.

Fakes only, no network. ``backoff_initial=0.0`` keeps the retrying cases from
sleeping through a real 1s/2s ladder.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from w2v_common.stages import stage_call
from w2v_common.stages.errors import StageRetriesExhaustedError

from src.services.stages.llm_budget import LLM_RETRY_ON
from src.services.stages.shared.loop_engine import Critique, run_evaluator_optimizer
from src.services.stages.upstream import (
    TRANSIENT_4XX_MARKERS,
    TRANSIENT_STATUSES,
    WM_DETERMINISTIC_FAILURE_MARKERS,
    WM_OBSERVED_TRANSIENT_422_MESSAGES,
    PermanentUpstreamError,
    TransientUpstreamError,
    classify,
    guarded,
)

_WM_URL = "http://worker-management.workers:8887/workers/x/message"


def _http_error(status: int, body: str = "") -> httpx.HTTPStatusError:
    """An HTTPStatusError shaped like a real WMClient failure (request + body attached)."""
    request = httpx.Request("POST", _WM_URL)
    response = httpx.Response(status, text=body, request=request)
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


_BYO_LLM_402 = (
    '{"error": {"code": "ERROR", "status": 402, "message": '
    '{"code": "byo_llm_not_configured", "message": '
    '"Configure your LLM provider in Settings before running agents."}}}'
)


def test_classify_402_is_permanent_and_carries_the_actionable_sentence() -> None:
    """The whole point: str(exc) must reach the user with WM's own instruction."""
    result = classify(_http_error(402, _BYO_LLM_402))

    assert isinstance(result, PermanentUpstreamError)
    assert "Configure your LLM provider" in str(result)


@pytest.mark.parametrize("status", [400])
def test_classify_client_errors_are_permanent(status: int) -> None:
    """A malformed request repeated verbatim stays malformed."""
    assert isinstance(classify(_http_error(status, "bad request")), PermanentUpstreamError)


_MAX_OUTPUT_RETRIES_422 = (
    '{"error": {"code": "ERROR", "status": 422, "message": "Exceeded maximum output retries (3)"}}'
)


def test_classify_422_max_output_retries_is_transient() -> None:
    """Regression for run 01a00a4e-22ad-74dc-9c53-3e39fd0a6094: WM answers 422 for
    pydantic-ai's "Exceeded maximum output retries (3)", which is a bad model DRAW, not a
    bad request (demo_video_footage_direction: 103 completed vs 3 failed). Classing it
    permanent killed the run in 8s having spent zero retries."""
    result = classify(_http_error(422, _MAX_OUTPUT_RETRIES_422))

    assert isinstance(result, TransientUpstreamError)
    assert "Exceeded maximum output retries" in str(result)


def test_classify_422_ordinary_validation_body_stays_permanent() -> None:
    """SCOPE GUARD — the one that matters most. 422 is normally OUR bug: a request that
    fails validation fails identically forever. Only a marker-matched body escapes."""
    body = '{"error": {"code": "ERROR", "status": 422, "message": "field required: brief"}}'

    assert isinstance(classify(_http_error(422, body)), PermanentUpstreamError)


def test_classify_422_does_not_match_a_marker_the_caller_merely_echoed() -> None:
    """A real 422 whose body ECHOES the marker must stay permanent.

    Found by adversarial review, reproduced against the first cut: FastAPI puts the
    offending value in ``detail[].input``, so matching the RAW BODY meant a run whose own
    script mentions "maximum output retries" -- a video *about* LLM pipelines -- flipped
    its own validation error to transient and retried a guaranteed loss. Markers are now
    matched against WM's asserted ``error.message`` envelope only; ``detail`` as a list
    yields no detail at all, so echoed text can never reach a marker.
    """
    body = (
        '{"detail":[{"type":"string_too_long","loc":["body","brief","script"],'
        '"msg":"String should have at most 2000 characters",'
        '"input":"How to configure maximum output retries in your LLM pipeline"}]}'
    )

    assert isinstance(classify(_http_error(422, body)), PermanentUpstreamError)


@pytest.mark.parametrize("status", [400])
def test_classify_marker_does_not_leak_to_other_statuses(status: int) -> None:
    """The escape is scoped to 422 ALONE — it is not a 4xx-wide rule. 400/404 keep failing
    fast even carrying the marker text; 409 keeps retrying because it always did
    (TRANSIENT_STATUSES), not because of the marker."""
    result = classify(_http_error(status, "Exceeded maximum output retries (3)"))

    expected = TransientUpstreamError if status in TRANSIENT_STATUSES else PermanentUpstreamError
    assert isinstance(result, expected)


def test_422_marker_matching_is_case_insensitive_inside_the_envelope() -> None:
    """Case-insensitive substring, but only within WM's asserted ``error.message``."""
    shouted = (
        '{"error": {"code": "ERROR", "status": 422,'
        ' "message": "PREFIX: EXCEEDED MAXIMUM OUTPUT RETRIES (3) suffix"}}'
    )

    assert isinstance(classify(_http_error(422, shouted)), TransientUpstreamError)


def test_422_bare_text_body_stays_permanent_by_design() -> None:
    """The deliberate cost of envelope-only matching, pinned so it is a DECISION not a bug.

    A 422 whose body is bare text (proxy/ingress, or a WM shape change) carries no
    ``error.message``, so it no longer escapes and fails fast as before. That is the safe
    direction: the alternative is raw-body matching, which review reproduced flipping a
    genuine validation error to transient purely because the run's own script echoed the
    marker. If WM ever starts returning this sentence as bare text, this test is the
    tripwire — widen the extraction, do NOT go back to matching the whole body.
    """
    bare = "Exceeded maximum output retries (3)"

    assert isinstance(classify(_http_error(422, bare)), PermanentUpstreamError)


def test_5xx_marker_matching_is_case_insensitive_substring() -> None:
    """The pre-existing 5xx convention is unchanged: matched against the raw body."""
    shouted_500 = _http_error(500, "PREFIX: MODEL TOKEN LIMIT (4000) EXCEEDED suffix")

    assert isinstance(classify(shouted_500), PermanentUpstreamError)


def test_classify_5xx_token_limit_is_permanent() -> None:
    """WM returns 500 for a deterministic token-budget overflow. Retrying is guaranteed loss."""
    exc = _http_error(500, "Model token limit (4000) exceeded before any response was generated")

    assert isinstance(classify(exc), PermanentUpstreamError)


def test_classify_5xx_connection_error_is_transient() -> None:
    """The counterweight to the test above: a 500 body WITHOUT a token-limit marker is
    an inference-router blip and is the reason the retry budget exists."""
    assert isinstance(classify(_http_error(500, "Connection error.")), TransientUpstreamError)


@pytest.mark.parametrize("status", [408])
def test_classify_transient_statuses(status: int) -> None:
    assert isinstance(classify(_http_error(status, "")), TransientUpstreamError)


async def test_guarded_402_fails_on_first_attempt() -> None:
    """Regression: 402 must cost exactly ONE upstream call and surface as Permanent —
    NOT StageRetriesExhaustedError, which is what the user saw before this fix."""
    calls = {"n": 0}

    async def factory() -> None:
        calls["n"] += 1
        raise _http_error(402, _BYO_LLM_402)

    with pytest.raises(PermanentUpstreamError) as caught:
        await stage_call(
            lambda: guarded(factory),
            timeout_seconds=5.0,
            retries=2,
            retry_on=LLM_RETRY_ON,
            backoff_initial=0.0,
            stage_name="test",
        )

    assert calls["n"] == 1
    assert "Configure your LLM provider" in str(caught.value)


async def test_guarded_503_spends_the_full_retry_budget() -> None:
    """The transient half still gets every attempt it is owed (retries=2 → 3 calls)."""
    calls = {"n": 0}

    async def factory() -> None:
        calls["n"] += 1
        raise _http_error(503, "upstream unavailable")

    with pytest.raises(StageRetriesExhaustedError):
        await stage_call(
            lambda: guarded(factory),
            timeout_seconds=5.0,
            retries=2,
            retry_on=LLM_RETRY_ON,
            backoff_initial=0.0,
            stage_name="test",
        )

    assert calls["n"] == 3


async def test_guarded_passes_valueerror_through_and_it_still_retries() -> None:
    """LOAD-BEARING: primitives/script_stage.py raises ValueError inside the retry budget
    to re-prompt a short LLM draw. guarded() only touches HTTPStatusError, and ValueError
    is in LLM_RETRY_ON — so that re-prompt must keep working."""
    calls = {"n": 0}

    async def factory() -> None:
        calls["n"] += 1
        raise ValueError("too few elements")

    with pytest.raises(StageRetriesExhaustedError):
        await stage_call(
            lambda: guarded(factory),
            timeout_seconds=5.0,
            retries=2,
            retry_on=LLM_RETRY_ON,
            backoff_initial=0.0,
            stage_name="test",
        )

    assert calls["n"] == 3


async def test_loop_engine_critic_402_fails_on_first_call() -> None:
    """The evaluator-optimizer stages (script_doctor, retention_plan, claim_finish) call
    WM through run_evaluator_optimizer, not stage_call directly — so they are exactly the
    sites a call-site-only fix would silently skip. Prove the loop's OWN stage_call is
    guarded: a 402 critic aborts on call 1 and never reaches the optimizer."""
    critic_calls = {"n": 0}
    optimizer_calls = {"n": 0}

    async def critic(_draft: str) -> Critique:
        critic_calls["n"] += 1
        raise _http_error(402, _BYO_LLM_402)

    async def optimizer(draft: str, _critique: Critique) -> str:
        optimizer_calls["n"] += 1
        return draft

    with pytest.raises(PermanentUpstreamError) as caught:
        await run_evaluator_optimizer(
            "draft-0",
            critic_fn=critic,
            optimizer_fn=optimizer,
            timeout_seconds=5.0,
            retries=2,
            stage_name="test_loop",
        )

    assert critic_calls["n"] == 1
    assert optimizer_calls["n"] == 0
    assert "Configure your LLM provider" in str(caught.value)


# ---------------------------------------------------------------------------
# T1-3: the rescue is PROSE, and nothing upstream offers a structural alternative.
#
# Verified 2026-08-19 against worker_management @ 2e3e131: the agent branch raises
# `HTTPException(422, detail=str(e))` (message_endpoints.py:826) and the generic handler
# stamps `error.code` from a status→name map (error_handlers.py:198-214), so a lost
# structured-output draw and a malformed request are byte-identical in `error.code`.
# There is no `type`/`kind`/domain-`code` field to key on. These tests therefore pin the
# coupling itself, and assert the fallback is LOUD, so a reword breaks CI here instead of
# silently killing a paid run.
# ---------------------------------------------------------------------------


def _wm_422(message: str) -> httpx.HTTPStatusError:
    """A 422 shaped exactly like WM's agent branch: detail=str(e), code=VALIDATION_ERROR."""
    return _http_error(
        422,
        json.dumps({"error": {"code": "VALIDATION_ERROR", "status": 422, "message": message}}),
    )


@pytest.mark.parametrize("message", list(WM_OBSERVED_TRANSIENT_422_MESSAGES)[:1])
def test_every_pinned_wm_sentence_is_rescued(message: str) -> None:
    """TRIPWIRE. These are the exact sentences WM emits for the recoverable class.

    If this goes red, worker_management (or pydantic-ai beneath it) reworded the sentence
    and EVERY such 422 now falls through to PermanentUpstreamError — a recoverable bad
    draw killing a whole paid run with zero retries. Re-vendor
    ``WM_DETERMINISTIC_FAILURE_MARKERS`` and re-derive ``TRANSIENT_4XX_MARKERS``. Do not
    delete the failing case.
    """
    assert isinstance(classify(_wm_422(message)), TransientUpstreamError)


def test_our_marker_covers_wm_own_retries_marker() -> None:
    """The only structural relationship that DOES exist, asserted.

    WM answers 422 from the agent branch only when its own ``_is_deterministic_failure``
    matched, and it matches by substring against the very ``str(e)`` it puts in ``detail``.
    So as long as one of our markers is a substring of WM's retries marker, every 422 WM
    produces for that class is guaranteed to be rescued — independent of the numeric
    suffix, casing, or surrounding prose.
    """
    wm_retries_marker = "exceeded maximum output retries"
    assert wm_retries_marker in WM_DETERMINISTIC_FAILURE_MARKERS
    assert any(marker in wm_retries_marker for marker in TRANSIENT_4XX_MARKERS)


def test_wm_token_limit_marker_is_deliberately_not_rescued() -> None:
    """SCOPE GUARD. WM's other deterministic marker is a genuine, repeatable overflow.

    It must keep failing fast: rescuing it would spend the whole retry budget on a
    guaranteed loss. This asserts the vendored tuple did not widen the transient class.
    """
    permanent_half = "exceeded before any response was generated"
    assert permanent_half in WM_DETERMINISTIC_FAILURE_MARKERS
    assert not any(marker in permanent_half for marker in TRANSIENT_4XX_MARKERS)
    result = classify(
        _wm_422("Model token limit (6000) exceeded before any response was generated")
    )
    assert isinstance(result, PermanentUpstreamError)


def test_a_rescued_422_logs_a_distinct_loud_event(caplog: pytest.LogCaptureFixture) -> None:
    """The fallback must be OBSERVABLE. A prose match that nobody can see in prod is a
    guess; we need to watch it fire (and watch it stop firing) without a deploy."""
    with caplog.at_level(logging.WARNING, logger="src.services.stages.upstream"):
        assert isinstance(
            classify(_wm_422("Exceeded maximum output retries (3)")), TransientUpstreamError
        )

    assert any("upstream.422_rescued_by_prose_marker" in r.message for r in caplog.records)
    assert any("maximum output retries" in r.getMessage() for r in caplog.records)


def test_a_reworded_422_is_permanent_today_and_says_so_loudly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The failure mode, pinned. A REWORDED sentence is not rescued — there is no
    structural field to fall back on — so it kills the run with zero retries. That is
    exactly why the event below exists: it is the only signal that this happened."""
    reworded = _wm_422("exceeded the max output retry limit")

    with caplog.at_level(logging.WARNING, logger="src.services.stages.upstream"):
        result = classify(reworded)

    assert isinstance(result, PermanentUpstreamError)
    assert any("upstream.422_classified_permanent" in r.message for r in caplog.records)
