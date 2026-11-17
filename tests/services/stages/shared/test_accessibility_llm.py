"""Unit tests for the A19 accessibility Tier-2 LLM layer.

Cover the pure word-budget helpers, the two WM-calling functions (with a fake
``WMClient``), and the growth-bundle ``_attach_a11y_llm`` opt-in gate + never-gates
degrade path.
"""

from collections.abc import Iterator
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.shared import speech_rate
from src.services.stages.shared.accessibility import A11yPolicy, a11y_report
from src.services.stages.shared.accessibility_llm import (
    clamp_description_words,
    describe_gaps,
    gap_word_budget,
    rank_findings,
)


class _NoNetworkTTSClient:
    """Stands in for ``TTSClient`` in this unit test: constructing one is an error.

    ``describe_gaps`` now threads ``ctx`` through to a Phase-A wpm resolution
    (``_phase_a_wpm`` in ``accessibility_llm.py``), which — like every other
    speech-rate call site — builds a real ``TTSClient`` when no client is injected.
    Mirrors ``tests/services/stages/demo_video/conftest.py``'s guard so this test
    stays a unit test rather than real network I/O with a multi-second timeout.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "speech_rate tried to build a real TTSClient in a unit test — "
            "test_accessibility_llm.py has no fake client to inject here."
        )


@pytest.fixture(autouse=True)
def _no_speech_rate_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(speech_rate, "TTSClient", _NoNetworkTTSClient)
    speech_rate.clear_measured_wpm_cache()
    yield
    speech_rate.clear_measured_wpm_cache()


def _wm_response(output: object) -> WMResponse:
    return WMResponse(
        output=output,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


# ------------------------------------------------------------- pure helpers ---
def test_gap_word_budget_150_wpm() -> None:
    assert gap_word_budget(60_000.0) == 150  # a full minute → 150 words
    assert gap_word_budget(2_000.0) == 5  # 2s → 5 words
    assert gap_word_budget(100.0) == 1  # tiny gap floors to 1, never 0


def test_gap_word_budget_rounds_rather_than_truncates() -> None:
    """1040ms @ 150wpm = 2.6 words. round() -> 3; a truncating int() would give 2
    and under-fill the gap, leaving residual dead-air (the bug this guards)."""
    assert gap_word_budget(1_040.0) == 3


def test_clamp_description_words_truncates_over_budget() -> None:
    assert clamp_description_words("one two three four", 2) == "one two"


def test_clamp_description_words_leaves_within_budget() -> None:
    assert clamp_description_words("one two", 5) == "one two"


# ------------------------------------------------------ WM-calling functions ---
@pytest.mark.asyncio
async def test_rank_findings_returns_worker_json() -> None:
    wm = AsyncMock()
    wm.message.return_value = _wm_response(
        {"summary": "1 issue", "priority_findings": [{"line_id": "L1", "severity": "high"}]}
    )
    policy = A11yPolicy()
    report = a11y_report([], [], policy)  # empty is fine — the message is what matters
    out = await rank_findings(wm, report, policy)
    assert out == {
        "summary": "1 issue",
        "priority_findings": [{"line_id": "L1", "severity": "high"}],
    }


@pytest.mark.asyncio
async def test_rank_findings_non_object_degrades_to_empty() -> None:
    wm = AsyncMock()
    wm.message.return_value = _wm_response("not an object")
    out = await rank_findings(wm, a11y_report([], [], A11yPolicy()), A11yPolicy())
    assert out == {}


@pytest.mark.asyncio
async def test_describe_gaps_clamps_and_drops_unknown_ids() -> None:
    wm = AsyncMock()
    wm.message.return_value = _wm_response(
        {
            "lines": [
                # 2000ms gap → 5-word budget; this 7-word line must be clamped.
                {"after_line_id": "L1", "text": "a b c d e f g"},
                # id not in the gap set → dropped (never invents a beat).
                {"after_line_id": "GHOST", "text": "invented"},
            ]
        }
    )
    gaps = [{"after_line_id": "L1", "gap_ms": 2000.0}]
    out = await describe_gaps(wm, gaps, {"L1": "prior narration"})
    assert len(out) == 1
    assert out[0]["after_line_id"] == "L1"
    assert out[0]["text"] == "a b c d e"  # clamped to 5 words


@pytest.mark.asyncio
async def test_describe_gaps_no_gaps_makes_no_call() -> None:
    wm = AsyncMock()
    out = await describe_gaps(wm, [], {})
    assert out == []
    wm.message.assert_not_called()


# ------------------------------------------------- _attach_a11y_llm gating ---
def _ctx(clips: list[dict] | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={"demo_video_narrate": {"clips": clips or []}},
    )


# A shot whose text/bg contrast fails AA, so a11y_report yields an insight the
# critic has something to rank.
_FAILING_GEN = {
    "lines": [
        {"id": "L1", "line_id": "L1", "text": "hi", "text_color": "#808080", "bg_color": "#888888"}
    ]
}
