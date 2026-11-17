"""The Phase-A prior is a NARRATION rate, not the calibration passage's raw rate.

Live 2026-10-20: the passage measured 197.6 wpm while the same voice delivered the
drafted narration at 151-166 wpm, so every script prompt asked for ~20-30% more
words than the order affords. The single source (``resolve_natural_wpm_detail``)
now converts the measured passage rate with ``PASSAGE_TO_NARRATION_RATIO``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

import src.services.stages.shared.speech_rate as speech_rate
from src.services.runs.run_context import RunContext
from src.services.stages.shared.time_budget import DEFAULT_WPM_NATURAL

#: The live 2026-10-20 measurements this fix is calibrated against.
LIVE_PASSAGE_WPM = 197.6
LIVE_NARRATION_WPM = (151.0, 166.0)


def _ctx() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"voice": {"voice_id": "af_heart"}},
        prior_run_outputs={},
    )


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    speech_rate.clear_measured_wpm_cache()
    monkeypatch.setattr(speech_rate, "TTSClient", MagicMock())
    fake = AsyncMock(return_value=LIVE_PASSAGE_WPM)
    monkeypatch.setattr(speech_rate, "measure_natural_wpm", fake)
    return fake


def test_the_live_passage_rate_maps_into_the_measured_narration_range() -> None:
    low, high = LIVE_NARRATION_WPM
    assert low <= speech_rate.narration_prior_wpm(LIVE_PASSAGE_WPM) <= high


@pytest.mark.asyncio
async def test_the_resolved_prior_is_the_narration_rate_not_the_passage_rate(
    probe: Any,
) -> None:
    """FAILS BEFORE: the resolver returned 197.6 — the passage rate as the prior."""
    rate = await speech_rate.resolve_natural_wpm_detail(_ctx(), default=DEFAULT_WPM_NATURAL)

    low, high = LIVE_NARRATION_WPM
    assert rate.measured is True
    assert rate.basis == speech_rate.CALIBRATION_PASSAGE
    assert low <= rate.wpm <= high
    # Every Phase-A entry point reads the same single source.
    assert await speech_rate.resolve_natural_wpm(_ctx(), default=DEFAULT_WPM_NATURAL) == rate.wpm
    effective = await speech_rate.resolve_effective_wpm(_ctx(), texts=(), default=150.0)
    assert effective.rate.wpm == rate.wpm


@pytest.mark.asyncio
async def test_the_prior_scales_with_the_voice_never_a_hardcoded_rate(probe: Any) -> None:
    """A slower voice's prior moves with its own measurement (a ratio, not a constant)."""
    probe.return_value = 120.0
    rate = await speech_rate.resolve_natural_wpm_detail(_ctx(), default=DEFAULT_WPM_NATURAL)
    assert rate.wpm == pytest.approx(120.0 * speech_rate.PASSAGE_TO_NARRATION_RATIO)
    assert rate.wpm < DEFAULT_WPM_NATURAL


@pytest.mark.asyncio
async def test_a_fallback_default_is_not_rescaled(probe: Any) -> None:
    """The ratio converts a PASSAGE measurement only; an assumed default is untouched."""
    probe.return_value = None
    rate = await speech_rate.resolve_natural_wpm_detail(_ctx(), default=DEFAULT_WPM_NATURAL)
    assert rate.assumed is True
    assert rate.wpm == DEFAULT_WPM_NATURAL


@pytest.mark.asyncio
async def test_an_authored_rate_is_not_rescaled() -> None:
    ctx = _ctx()
    ctx.config_snapshot["direction"] = {"timing": {"wpm": {"natural": 140.0}}}
    effective = await speech_rate.resolve_effective_wpm(ctx, texts=(), default=150.0)
    assert effective.rate.basis == speech_rate.AUTHORED
    assert effective.rate.wpm == 140.0
