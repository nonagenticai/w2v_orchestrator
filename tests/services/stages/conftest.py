"""Shared fixtures for stage-handler tests.

Since 2026-06-01 the LLM stages shape-validate *inside* the ``stage_call``
retry budget (so a structurally-valid-but-missing-a-field WM reply re-prompts
instead of failing the run — the stage raises a plain ``ValueError``, which
``stage_call``'s default ``retry_on=(Exception,)`` re-tries). That means any
test feeding a malformed/missing-key output now exhausts the retry budget,
and the real exponential backoff (1s, 2s, …) would add multiple seconds of
wall-clock per such test. Zero the backoff delay in the stage-test tree: it
changes only timing, never retry behaviour.
"""

from __future__ import annotations

from typing import Any

import pytest
from w2v_common.stages import execution


@pytest.fixture(autouse=True)
def _no_stage_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(execution, "_backoff_delay", lambda *a, **k: 0.0)


@pytest.fixture(autouse=True)
def _stub_brand_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default the best-effort brand extraction to a no-op so stage tests stay
    hermetic (no network to analysis_service). Tests that exercise the wiring
    re-patch these names with a populated return.
    """

    async def _none(*_args: Any, **_kwargs: Any) -> tuple[None, None]:
        return None, None

    for module in (
        "src.services.stages.website_review.redesign",
        "src.services.stages.promo.html_segments",
    ):
        monkeypatch.setattr(f"{module}.resolve_brand_context", _none)


@pytest.fixture(autouse=True)
def _seal_speech_rate_tts(monkeypatch: pytest.MonkeyPatch):
    """Seal the speech-rate measurement's TTS client for EVERY stage test.

    The script stages measure narration by really synthesizing it
    (``speech_rate.measure_script_lines`` and the wpm probes) through a ``TTSClient``
    built at ``speech_rate``'s OWN import site. A stage test that does not stub it
    (e.g. primitives/test_script_stage_step_ledger.py,
    demo_video/test_record_failed_scenario_tail.py) reached the live ``tts.test``
    unit on mini2 from a dev box and hung until pytest-timeout — the same escape
    tests/integration/conftest.py ``_stub_speech_rate_tts`` seals for the DAG tests,
    with the same fake (a duration derived from each line's word count, so the
    measurement arithmetic still runs; only the network hop is replaced).

    Autouse so a stage newly reaching for the measurement is covered the day it is
    written. A test that must exercise the client patches over it locally (later
    ``monkeypatch.setattr`` wins). The process-wide measured-wpm cache is cleared on
    both sides so one test's stubbed rate never serves the next.
    """
    from src.services.stages.shared import speech_rate
    from tests.integration._dispatch_harness import _speech_rate_tts_mock

    speech_rate.clear_measured_wpm_cache()
    monkeypatch.setattr(speech_rate, "TTSClient", lambda *a, **k: _speech_rate_tts_mock())
    yield
    speech_rate.clear_measured_wpm_cache()
