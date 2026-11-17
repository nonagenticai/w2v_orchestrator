"""Pin the generic ``script_draft`` speech-rate probe for every test in this dir.

``script_draft`` resolves the run voice's measured words-per-minute before it
budgets the draw (a real TTS synthesis in production). No test here is about
that probe, so it is stubbed at the stage's own import site to a fixed rate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from unittest.mock import AsyncMock

import pytest

from src.services.stages.generic.line_fit import SPEED_EXPONENT
from src.services.stages.primitives.narration import SynthesizedClip
from src.services.stages.shared.speech_rate import (
    BASIS_MEASURED_ALL,
    ScriptMeasurement,
    ScriptSpeechRate,
    SpeechRate,
)
from src.services.tts_storage import deterministic_clip_id

#: The rate every generic test "speaks" at — 150 wpm == 2.5 words/second.
STUB_WPM = 150.0


@pytest.fixture(autouse=True)
def _stub_generic_speech_rate(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    rate = SpeechRate(wpm=STUB_WPM, measured=True)
    stub = AsyncMock(return_value=ScriptSpeechRate(rate=rate, prior=rate))
    monkeypatch.setattr("src.services.stages.generic.script_draft.resolve_effective_wpm", stub)
    return stub


def measurement_at(texts: Sequence[str], wpm: float) -> ScriptMeasurement:
    """What ``measure_script_lines`` returns for ``texts`` spoken at exactly ``wpm``."""
    per_line = tuple(len(str(t).split()) * 60.0 / wpm for t in texts)
    return ScriptMeasurement(
        per_line_seconds=per_line,
        measured=(True,) * len(per_line),
        failed_indices=(),
        extrapolated_indices=(),
        total_seconds=math.fsum(per_line),
        measured_count=sum(1 for t in texts if str(t).split()),
        wpm=wpm,
        basis=BASIS_MEASURED_ALL,
        prior=SpeechRate(wpm=wpm, measured=True),
    )


@pytest.fixture(autouse=True)
def _stub_generic_measurement(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Phase C (every line synthesized) at the stub rate — no real TTS.

    A test that needs a content-dependent rate sets ``side_effect`` to its own
    ``async (ctx, *, texts, **kw) -> measurement_at(texts, wpm)``.
    """

    async def _measure(_ctx: object, *, texts: Sequence[str], **_: object) -> ScriptMeasurement:
        return measurement_at(texts, STUB_WPM)

    stub = AsyncMock(side_effect=_measure)
    monkeypatch.setattr("src.services.stages.generic.script_draft.measure_script_lines", stub)
    return stub


@pytest.fixture(autouse=True)
def _stub_generic_line_synth(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """The per-line fit's TTS: words at the stub rate, scaled by kokoro's MEASURED
    response to the speed asked (``speed ** -SPEED_EXPONENT``, not ``1/speed``)."""

    async def _synth(
        _client: object,
        *,
        text: str,
        voice: str | None,
        run_id: str,
        stage_name: str,
        speed: float | None = None,
        **_: object,
    ) -> SynthesizedClip:
        words = len(text.split())
        seconds = words * 60.0 / STUB_WPM * (speed or 1.0) ** -SPEED_EXPONENT
        clip_id = deterministic_clip_id(text, voice, speed)
        return SynthesizedClip(clip_id, f"/clips/{stage_name}/{clip_id}", seconds, speed, words)

    stub = AsyncMock(side_effect=_synth)
    monkeypatch.setattr("src.services.stages.generic.script_draft.synthesize_and_store_clip", stub)
    return stub
