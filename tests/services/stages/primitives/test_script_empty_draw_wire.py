"""app_review / promo / lesson script — the EMPTY draw, through the REAL ``WMClient`` wire.

The website_review incident (``website_review/test_script_empty_draw_wire.py``:
HTTP 200, ``{"lines": []}``, persisted as a zero-line SUCCESS, run dead three
stages later in compile) was not website_review's alone. These three stages sit on
the same primitive with the same default floor of 0, and promo and lesson carried
the same "no lines → return early" door in their reconciliation.

One case per stage, same shape: the model answers empty on every attempt ⇒ the
SCRIPT stage fails after exactly ``llm_retries() + 1`` calls; empty first, valid
second ⇒ it recovers at the cost of one call. The floor lives in the PARSER — it
cannot live in the grammar, the xgrammar strip drops ``minItems``
(``tests/test_output_schema_bounds.py``).

What the floor must NOT touch is pinned here too: the verbatim bypasses
(``source_script``) never draw from the model at all, so an always-empty model
cannot fail them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid7

import pytest
from w2v_common.stages import StageRetriesExhaustedError

import src.services.stages.app_review.script as app_review_script
import src.services.stages.lesson.script as lesson_script
import src.services.stages.promo.script as promo_script
from src.services.llm import WMClient, vllm_direct
from src.services.runs.run_context import RunContext
from src.services.stages.llm_budget import llm_retries
from src.services.stages.primitives.script_stage import (
    LLMScriptStageBase,
)
from tests.services.stages.app_review.rate_stubs import stub_speech_rate as stub_app_review_rate
from tests.services.stages.lesson.rate_stubs import stub_speech_rate as stub_lesson_rate
from tests.services.stages.promo.rate_stubs import stub_speech_rate as stub_promo_rate
from tests.support.platform_llm import (
    PlatformChat,
    system_text,
    user_text,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

#: 1 word = 1 second, so a draw's length is its word count.
_WPM = 60.0


def _words(count: int) -> str:
    return " ".join(["word"] * count)


@dataclass(frozen=True)
class _Case:
    module: ModuleType
    stage_cls: type[LLMScriptStageBase[Any, Any]]
    stub_rate: Callable[..., Any]
    element_key: str
    worker_name: str
    snapshot: dict[str, Any]
    prior: dict[str, Any]
    valid: list[dict[str, Any]]


_CASES = {
    "app_review": _Case(
        module=app_review_script,
        stage_cls=app_review_script.AppReviewScriptStage,
        stub_rate=stub_app_review_rate,
        element_key="lines",
        worker_name="app-review-script",
        snapshot={"brief": {"duration_seconds": 60}, "voice": {"descriptor": "neutral"}},
        prior={
            "app_inspect": {"facts": [{"id": "f1", "kind": "title", "text": "T"}]},
            "app_critique": {
                "pros": [{"id": "p1", "text": "fast"}],
                "cons": [{"id": "c1", "text": "sparse"}],
                "recommendations": [{"id": "r1", "text": "ship"}],
                "overall_take": "ok",
            },
        },
        valid=[{"text": _words(30), "seconds": 30.0}, {"text": _words(30), "seconds": 30.0}],
    ),
    "promo": _Case(
        module=promo_script,
        stage_cls=promo_script.PromoScriptStage,
        stub_rate=stub_promo_rate,
        element_key="lines",
        worker_name="promo-script",
        snapshot={
            "title": "Anti-churn",
            "brief": {"target_duration_seconds": 60},
            "voice": {"descriptor": "crisp"},
        },
        prior={
            "promo_analyze": {
                "pain_points": [{"id": "p1", "text": "high churn"}],
                "hook_candidates": [{"id": "h1", "text": "Stop the bleeding."}],
                "tone_hints": ["urgent"],
            },
        },
        # ``speaker`` is schema-required: the vLLM path validates the reply against
        # the grammar (WM never did), so the valid script must carry it.
        valid=[
            {"text": _words(30), "seconds": 30.0, "speaker": "a"},
            {"text": _words(30), "seconds": 30.0, "speaker": "a"},
        ],
    ),
    "lesson": _Case(
        module=lesson_script,
        stage_cls=lesson_script.LessonScriptStage,
        stub_rate=stub_lesson_rate,
        element_key="paragraphs",
        worker_name="lesson-script-v2",
        snapshot={
            "brief": {"duration_minutes": 1, "learning_objective": "Diagnose churn drivers."},
            "voice": {"descriptor": "calm"},
        },
        prior={},
        valid=[
            {"text": _words(30), "seconds": 30.0, "speaker": "b"},
            {"text": _words(30), "seconds": 30.0, "speaker": "b"},
        ],
    ),
}

pytestmark = pytest.mark.usefixtures("platform_llm_socket")

_each_stage = pytest.mark.parametrize("case", list(_CASES.values()), ids=list(_CASES))


def _ctx(case: _Case, *, brief: dict[str, Any] | None = None, **prior: Any) -> RunContext:
    snapshot = {**case.snapshot, "brief": {**case.snapshot["brief"], **(brief or {})}}
    return RunContext(
        run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot=snapshot,
        prior_run_outputs={**case.prior, **prior},
    )


class _WireWM:
    """The platform LLM (Chat Completions) behind an ``httpx2.MockTransport``.

    Answers the first ``empty_calls`` calls with an empty element list — the
    schema-valid, HTTP-200 answer the live run got — and every later one with the
    case's valid script. Records every request body and the worker each call named.
    """

    def __init__(
        self, case: _Case, *, empty_calls: int, first: list[dict[str, Any]] | None = None
    ) -> None:
        self.case = case
        self.empty_calls = empty_calls
        #: When set: call 1 answers THIS, and every later call answers empty.
        self.first = first
        self.chat = PlatformChat(self.answer)
        #: ``worker_name`` of every ``WMClient.message`` call (not on the vLLM wire).
        self.workers: list[str] = []

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return self.chat.bodies

    @property
    def paths(self) -> list[str]:
        return self.chat.paths

    def elements(self, n: int) -> list[dict[str, Any]]:
        if self.first is not None:
            return self.first if n == 0 else []
        return [] if n < self.empty_calls else self.case.valid

    def answer(self, _body: dict[str, Any], n: int) -> dict[str, Any]:
        return {self.case.element_key: self.elements(n)}

    def client(self) -> WMClient:
        wire = self

        class _Recording(WMClient):
            async def message(self, worker_name: str, *args: Any, **kwargs: Any) -> Any:
                wire.workers.append(worker_name)
                return await super().message(worker_name, *args, **kwargs)

        # Route the platform socket here; everything above the transport (body
        # building, schema strip, response parsing) is real.
        self.chat.activate()
        return _Recording(timeout=30.0)


async def _run(
    monkeypatch: pytest.MonkeyPatch, case: _Case, wire: _WireWM, ctx: RunContext | None = None
) -> Any:
    case.stub_rate(monkeypatch, case.module, _WPM)
    return await case.stage_cls(client=wire.client()).run({}, ctx=ctx or _ctx(case))


@_each_stage
async def test_an_empty_first_attempt_recovers_on_the_re_prompt(
    monkeypatch: pytest.MonkeyPatch, case: _Case
) -> None:
    wire = _WireWM(case, empty_calls=1)
    out = await _run(monkeypatch, case, wire)

    elements = getattr(out, case.element_key)
    assert [element.text for element in elements] == [raw["text"] for raw in case.valid]
    # Exactly ONE extra call. Compared against the same model answering validly
    # from the start, so anything else the stage does cancels out.
    baseline = _WireWM(case, empty_calls=0)
    await _run(monkeypatch, case, baseline)
    assert len(wire.bodies) == len(baseline.bodies) + 1


async def test_promo_names_a_thin_site_capture_in_the_empty_draw_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """promo_script is grounded on ``site_capture`` — a thin read is the likely cause."""
    case = _CASES["promo"]
    capture = {"clips": [], "page_text": "x" * 171, "page_text_status": "thin", "notes": []}
    wire = _WireWM(case, empty_calls=10**6)

    with pytest.raises(StageRetriesExhaustedError) as caught:
        await _run(monkeypatch, case, wire, _ctx(case, site_capture=capture))

    assert "empty script draw" in str(caught.value)
    assert "site capture read 171 chars" in str(caught.value)


@pytest.mark.parametrize("name", ["promo"])
async def test_a_verbatim_source_script_never_draws_so_the_floor_never_judges_it(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """The legitimate "the model wrote nothing" case: it was never asked.

    ``short_circuit`` returns the customer's own text before the parser runs, so a
    model that would answer empty forever cannot fail an authored script.
    """
    case = _CASES[name]
    wire = _WireWM(case, empty_calls=10**6)
    ctx = _ctx(case, brief={"source_script": "First authored line.\n\nSecond authored line."})

    out = await _run(monkeypatch, case, wire, ctx)

    assert [element.text for element in getattr(out, case.element_key)] == [
        "First authored line.",
        "Second authored line.",
    ]
    assert wire.bodies == []


@pytest.mark.parametrize("name", ["promo"])
async def test_no_order_is_the_only_unreconciled_exit(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """The one early return left in each reconciliation: no order ⇒ no band ⇒ no verdict.

    "No elements" used to leave by the same door. A script WITH elements and no
    order still ships unreconciled (``narration_estimate is None``, one call); a
    script with none never reaches reconciliation (tests above).
    """
    case = _CASES[name]
    monkeypatch.setattr(case.module, "_target_seconds", lambda *_args: 0.0)
    wire = _WireWM(case, empty_calls=0)

    out = await _run(monkeypatch, case, wire)

    assert len(getattr(out, case.element_key)) == len(case.valid)
    assert out.narration_estimate is None
    assert len(wire.bodies) == 1


#: Calls made by ``origin/main`` on the adversarial sequence below — a valid but
#: far-too-short first draw, then an empty answer to EVERY corrective re-ask. One
#: call for the draw plus one per re-ask the stage's own ladder allows. The floor
#: must not add to these: a correction that draws empty is a failed correction,
#: not a new draw to re-prompt (it briefly cost ``llm_retries() + 1`` each).
_MAIN_WORST_CASE_CALLS = {"app_review": 2, "promo": 2, "lesson": 3}


@_each_stage
async def test_an_empty_corrective_re_ask_costs_one_call_and_is_recorded_as_discarded(
    monkeypatch: pytest.MonkeyPatch, case: _Case
) -> None:
    short = [{**raw, "text": _words(5), "seconds": 5.0} for raw in case.valid]
    wire = _WireWM(case, empty_calls=0, first=short)

    out = await _run(monkeypatch, case, wire)

    # Not a failure: the stage holds a usable script and ships it.
    assert [element.text for element in getattr(out, case.element_key)] == [_words(5)] * 2
    assert len(wire.bodies) == _MAIN_WORST_CASE_CALLS[case.stage_cls.name.removesuffix("_script")]
    # The re-ask WAS sent and WAS discarded — never "no re-ask happened".
    estimate = out.narration_estimate
    assert estimate is not None
    assert (estimate.reasked, estimate.reask_kept) == (True, False)
    assert estimate.verdict == "short"
    assert any("no " in entry["detail"] for entry in out.advisories), out.advisories


# --- a retry after an under-floor draw is a DIFFERENT ask -------------------------

_RETRY_MARKER = "RETRY — your previous answer"


@_each_stage
async def test_the_retry_after_an_empty_draw_appends_a_corrective_block(
    monkeypatch: pytest.MonkeyPatch, case: _Case
) -> None:
    """origin/main re-POSTed the byte-identical prompt on every attempt."""
    wire = _WireWM(case, empty_calls=10**6)

    with pytest.raises(StageRetriesExhaustedError, match="empty script draw"):
        await _run(monkeypatch, case, wire)

    first, *retries = [user_text(body) for body in wire.bodies]
    assert retries
    # The FIRST draw of these stages is untouched: no corrective text, and no
    # claim that an earlier answer existed.
    assert _RETRY_MARKER not in first
    for retry in retries:
        assert retry != first
        assert retry.startswith(first)
        block = retry[len(first) :]
        assert _RETRY_MARKER in block
        assert f"0 `{case.element_key}`" in block
        assert "HARD MINIMUM" in block
    # One model call per attempt, as before.
    assert len(wire.bodies) == llm_retries() + 1


@_each_stage
async def test_a_retry_for_any_other_reason_re_sends_the_same_prompt(
    monkeypatch: pytest.MonkeyPatch, case: _Case
) -> None:
    """Only an under-floor draw earns the block — a shape error says nothing about length."""

    # The vLLM path re-asks a grammar violation itself (``OUTPUT_RETRIES``, the
    # violations appended to the SAME call); only when those are spent does the
    # STAGE retry. So the first draw stays malformed through all of them.
    transport_calls = vllm_direct.OUTPUT_RETRIES + 1

    class _MalformedThenValid(_WireWM):
        def elements(self, n: int) -> list[dict[str, Any]]:
            return [{"seconds": 3.0}] if n < transport_calls else self.case.valid

    wire = _MalformedThenValid(case, empty_calls=0)
    await _run(monkeypatch, case, wire)

    assert len(wire.bodies) == transport_calls + 1
    first, retry = wire.bodies[0], wire.bodies[transport_calls]
    # The stage's retry is a fresh call with the SAME prompt — nothing appended.
    assert retry["messages"] == first["messages"]
    assert all(_RETRY_MARKER not in user_text(body) for body in wire.bodies)


@_each_stage
async def test_an_under_floor_draw_is_logged_with_its_prompt_and_completion(
    monkeypatch: pytest.MonkeyPatch, case: _Case, caplog: pytest.LogCaptureFixture
) -> None:
    import hashlib

    wire = _WireWM(case, empty_calls=1)
    with caplog.at_level("WARNING", logger="src.services.stages.primitives.script_stage"):
        await _run(monkeypatch, case, wire)

    messages = [
        r.getMessage()
        for r in caplog.records
        if r.levelname == "WARNING" and "under-floor script draw" in r.getMessage()
    ]
    assert len(messages) == 1
    payload = json.loads(messages[0].split("under-floor script draw ", 1)[1])
    # What the model received, rebuilt from the log line alone: the prompt is the
    # user turn and the system prompt closes the system turn (after the worker seed).
    assert user_text(wire.bodies[0]) == payload["prompt"]
    assert system_text(wire.bodies[0]).endswith(payload["system_prompt"])
    prompt = payload["prompt"]
    assert prompt and "Total spoken length" in prompt
    assert payload["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert payload["prompt_chars"] == len(prompt)
    assert json.loads(payload["completion"]) == {case.element_key: []}
    assert "test-vllm-licence-token" not in caplog.text
