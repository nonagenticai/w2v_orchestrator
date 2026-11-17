"""``failure_class`` must be non-``unknown`` for the shapes prod ACTUALLY produces.

Measured on the live DB 2026-09-10, last 7 days: ``runs.failure_class`` was
**NULL 334 / 'unknown' 42 / 'external' 2**. Not one failure carried an
actionable class, so nothing was alertable and no trend was measurable — the
column was arithmetically present and informationally empty.

These tests pin the three failure shapes observed live in the preceding 48h to
the class each must land in, driving the REAL raise sites rather than
constructing the terminal exception by hand — a test that builds the wrapper
itself proves only that ``attribute_exception`` can read an attribute, which was
never in doubt. What was in doubt is whether the objects prod actually raises
carry one.

⚠️ NOT marked ``integration``: this repo's pytest ``addopts`` deselects that
marker, so an ``integration`` test here would never execute in CI. Everything
below is in-process (``httpx.MockTransport``) and needs no service.
"""

from __future__ import annotations

import httpx
import pytest
from w2v_common.stages import StageRetriesExhaustedError

from src.clients.video_compile import VideoCompileClient, VideoCompileJobFailedError
from src.models.failure_class import EXTERNAL, OUR_DEFECT, UNKNOWN, attribute_exception
from src.services.stages.base import StageProducedNothingError
from src.services.stages.upstream import (
    PermanentUpstreamError,
    TransientUpstreamError,
    classify,
)

_WM_URL = "http://worker-management.w2v:9101/workers/vd/message"


def _wm_422(payload: dict[str, object]) -> httpx.HTTPStatusError:
    """A real WM 422, in the envelope shape ``_envelope_detail`` reads."""
    request = httpx.Request("POST", _WM_URL)
    response = httpx.Response(422, request=request, json=payload)
    return httpx.HTTPStatusError("422", request=request, response=response)


def _exhausted(stage_name: str, attempts: int, cause: BaseException) -> StageRetriesExhaustedError:
    """The wrapper ``w2v_common.stage_call`` raises, chained exactly as it chains.

    ``stage_call`` does ``raise StageRetriesExhaustedError(...) from exc``, so
    the cause chain ``attribute_exception`` walks is load-bearing here.
    """
    try:
        raise StageRetriesExhaustedError(
            stage_name=stage_name, attempts=attempts, last_error=cause
        ) from cause
    except StageRetriesExhaustedError as exc:
        return exc


# ── Shape 1: a stage reported success and produced nothing ──────────────────
# Live: "StageProducedNothingError: stage 'promo_slides' reported success but
# produced nothing for its required output field(s): slides" (x4).


def test_stage_produced_nothing_is_our_defect() -> None:
    """A genuine upstream contract breach is OURS, and stays ours.

    This shape was ALREADY attributed correctly before this change; the test is
    a regression guard, because the two edits alongside it both add ``external``
    stamps and ``external`` is precisely the wrong bucket for this one. A
    ``promo_slides`` run that swallowed its own error and returned a
    success-shaped empty output is a defect we ship a fix for, not infra flake.
    """
    exc = StageProducedNothingError("promo_slides", ["slides"])
    assert attribute_exception(exc) == OUR_DEFECT


def test_stage_produced_nothing_outranks_a_transport_cause() -> None:
    """Even chained under a transport error, the stamp wins over pass 4.

    Guards the ordering ``attribute_exception`` documents: the explicit stamp
    (pass 1) must beat the ``httpx.RequestError`` -> ``external`` inference
    (pass 4). Without that ordering the swallowed-error case — the one this
    error exists to catch — would file itself as somebody else's fault.
    """
    try:
        raise StageProducedNothingError("promo_slides", ["slides"]) from httpx.ReadError("reset")
    except StageProducedNothingError as exc:
        assert attribute_exception(exc) == OUR_DEFECT


# ── Shape 2: a downstream service hard-failed the job ───────────────────────
# Live: "StageRetriesExhaustedError: stage 'lesson_compile': retries exhausted
# after 1 attempts: RuntimeError: video_compile job ... failed: ...".


@pytest.mark.asyncio
async def test_compiler_terminal_failure_raises_an_attributable_type() -> None:
    """Drive the REAL poll loop: a terminal 'failed' job must not raise a bare RuntimeError.

    This is the mutation-sensitive half of shape 2. ``RuntimeError`` is
    deliberately absent from ``_OUR_DEFECT_TYPES`` (``StageError`` extends it),
    so a bare one carries no structural signal and walks out of
    ``attribute_exception`` as ``unknown`` — which is why this shape was
    unattributable in prod.
    """
    client = _client_scripted(
        [httpx.Response(200, json={"status": "failed", "error": "mastering gate rejected"})]
    )
    try:
        with pytest.raises(VideoCompileJobFailedError) as caught:
            await client.compile(shots=[], timeout=30.0, idempotency_key=_IDEM_KEY)
    finally:
        await client.close()

    # Still a RuntimeError, so every existing `except RuntimeError` /
    # `retry_on=(..., RuntimeError)` catcher in the stage layer is unaffected:
    # this change moves attribution only, never control flow.
    assert isinstance(caught.value, RuntimeError)
    # The message reaches an operator verbatim via runs.error — it must not drift.
    assert str(caught.value) == "video_compile job job-1 failed: mastering gate rejected"


def test_lesson_compile_retries_exhausted_on_a_failed_job_is_external() -> None:
    """The full live chain: stage_call wrapper -> compiler job failure -> 'external'.

    ``external`` and not ``our_defect``: what is attributed is THIS run's code
    path. "The compiler ran the job and reported it failed" is a materially
    different finding from "the orchestrator has a bug", and the compiler being
    a service this org operates does not make it this run's code path — the same
    reasoning ``fault_for_status`` already applies to a 5xx.
    """
    cause = VideoCompileJobFailedError("job-1", "mastering gate rejected")
    exc = _exhausted("lesson_compile", 1, cause)

    assert attribute_exception(exc) == EXTERNAL
    # The prose an operator reads still names the real cause, unchanged in shape.
    assert "video_compile job job-1 failed" in str(exc)


# ── Shape 3: a transient upstream 4xx that exhausted our retries ────────────
# Live: "StageRetriesExhaustedError: stage 'voice_director': retries exhausted
# after 2 attempts: TransientUpstreamError: upstream 422 from
# http://worker-management...".


def test_marker_rescued_422_is_stamped_external_by_classify() -> None:
    """A 422 WM itself calls a lost sampling draw is a dependency failing, not us.

    ``fault_for_status(422)`` is ``unknown`` ON PURPOSE — from the status alone a
    422 means both "our request was malformed" and "the model lost its draw".
    Reaching the marker branch resolves that ambiguity with the upstream's OWN
    assertion, so the fault is knowable here and nowhere later: by the time the
    error reaches ``run_dispatcher``'s persistence site the response is gone.
    """
    classified = classify(_wm_422({"error": {"message": "Exceeded maximum output retries (3)"}}))

    assert isinstance(classified, TransientUpstreamError)
    assert getattr(classified, "failure_class", None) == EXTERNAL


def test_voice_director_retries_exhausted_on_a_lost_draw_is_external() -> None:
    """The full live chain: stage_call wrapper -> classified 422 -> 'external'."""
    hse = _wm_422({"error": {"message": "Exceeded maximum output retries (3)"}})
    classified = classify(hse)
    try:
        raise classified from hse
    except Exception as inner:  # noqa: BLE001 -- reproducing the real raise/catch shape
        exc = _exhausted("voice_director", 2, inner)

    assert attribute_exception(exc) == EXTERNAL


# ── Non-regression: the ambiguity that must STAY ambiguous ──────────────────
#
# The cardinal rule of this taxonomy is that a wrong attribution is worse than
# an absent one. These pin the boundary of the change above: only the branch
# where the upstream positively resolved the ambiguity earns a fault. Without
# them, "transient => external" could be widened by a later reader into exactly
# the axis-collapse ``models/failure_class.py`` forbids.


def test_a_real_422_contract_fault_stays_unattributed() -> None:
    """An UNRESCUED 422 (a FastAPI validation body) must remain 'unknown'.

    This is the case that could equally be our malformed request or their
    malformed brief. Filing it as ``external`` would send an engineer looking
    for an outage that never happened.
    """
    classified = classify(_wm_422({"detail": [{"loc": ["body", "x"], "msg": "field required"}]}))

    assert isinstance(classified, PermanentUpstreamError)
    assert getattr(classified, "failure_class", None) == UNKNOWN
    assert attribute_exception(_exhausted("voice_director", 2, classified)) == UNKNOWN


def test_402_still_attributes_to_the_customer() -> None:
    """The one documented unambiguous customer case must survive the change.

    402 is ``byo_llm_not_configured`` — the tenant has no LLM provider, and no
    code we ship makes the run succeed. It is a PERMANENT status, so it must not
    be swept into the transient branch this change touched.
    """
    request = httpx.Request("POST", _WM_URL)
    response = httpx.Response(402, request=request, json={"error": {"message": "byo_llm"}})
    classified = classify(httpx.HTTPStatusError("402", request=request, response=response))

    assert getattr(classified, "failure_class", None) == "customer_input"


# ── Local harness (mirrors tests/unit/test_video_compile_poll.py) ───────────

_IDEM_KEY = "22222222-2222-7222-8222-222222222222:lesson_compile"


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse retry/backoff sleeps so the poll loop runs instantly."""
    import asyncio

    async def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _noop)


def _client_scripted(poll_steps: list[httpx.Response]) -> VideoCompileClient:
    """POST -> submit (job_id); each GET pops the next scripted poll response."""
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"job_id": "job-1"})
        step = poll_steps[min(state["i"], len(poll_steps) - 1)]
        state["i"] += 1
        return step

    client = VideoCompileClient(base_url="http://compile.test", license_token="t")
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        timeout=client.timeout,
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )
    return client
