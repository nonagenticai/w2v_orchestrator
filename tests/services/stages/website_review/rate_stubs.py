"""Pin BOTH speech-rate phases for a website_review test that is not about rates.

``website_review_script`` resolves its narration rate twice: a Phase-A calibration
prior before the draw, and then a full-script MEASUREMENT of every drafted line
before the verdict. A test about the steer/re-ask/trim loop, or about the
word-budget sentence, wants known rates and no TTS at all — so this stubs both.

**The measurement stub is PER VOICE**, and that is the whole difference from the
single-narrator siblings: website_review casts each line to its speaker's voice
(``primitives/dialogue.voice_for_speaker``), so a stub returning one rate for the
whole script would make the cast invisible and the clip-reuse contract
untestable. Each line's seconds is its own word count at ITS OWN voice's rate, so
a test's arithmetic is the arithmetic the stage does and a trim over measured
seconds is exercised rather than bypassed.

Deliberately NOT a fixture: the tests here stub inside their own ``_run`` helpers,
and a module that only defines helpers (no ``test_`` prefix) is not collected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

from src.services.stages.shared import speech_rate as _speech_rate
from src.services.stages.shared.speech_rate import (
    BASIS_MEASURED_ALL,
    ScriptMeasurement,
    SpeechRate,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

    import pytest

    from src.services.stages.shared.speech_rate import FallbackReason


def _rate_for(voice: str | None, rates: dict[str | None, float], default: float) -> float:
    return rates.get(voice, default)


def measurement_at(
    texts: Sequence[str],
    voices: Sequence[str | None],
    rates: dict[str | None, float],
    default: float,
    prior: SpeechRate,
) -> ScriptMeasurement:
    """A LANDED full-script measurement: every line at ITS VOICE's rate.

    ``ScriptMeasurement.wpm`` is the POOLED rate — total words over total measured
    seconds — exactly as the real ``_measure_script_lines`` computes it, because
    that is the number the stage then writes the re-ask target and the word-proxy
    trim fallback from. A stub returning one voice's rate there would hide a
    mis-blended pool.

    A rate ``<= 0`` is passed through unchanged rather than substituted — the
    stage's own ``wpm_is_usable`` guard is exactly what such a test exercises, and
    a stub that quietly fixed the rate would make that guard untestable.
    """
    per_line: list[float] = []
    for text, voice in zip(texts, voices, strict=True):
        rate = _rate_for(voice, rates, default)
        words = len(str(text).split())
        per_line.append(words * 60.0 / rate if rate > 0.0 else 0.0)
    total = math.fsum(per_line)
    words_total = sum(len(str(t).split()) for t in texts)
    pooled = (words_total / total * 60.0) if total > 0.0 else prior.wpm
    return ScriptMeasurement(
        per_line_seconds=tuple(per_line),
        measured=(True,) * len(per_line),
        failed_indices=(),
        extrapolated_indices=(),
        total_seconds=total,
        measured_count=len(per_line),
        wpm=pooled,
        basis=BASIS_MEASURED_ALL,
        prior=prior,
    )


def refused_measurement(
    texts: Sequence[str], prior: SpeechRate, reason: FallbackReason
) -> ScriptMeasurement:
    """What ``measure_script_lines`` returns when it could NOT measure.

    Shaped exactly like its ``_prior_measurement``: every spoken line extrapolated
    from ``prior``, ``measured_count=0``, and a ``reason`` — which is the whole
    point, because that reason is what ``script_measurement_degradation`` turns
    into a row on the degrading tier.
    """
    per_line = tuple(
        (len(str(text).split()) * 60.0 / prior.wpm) if prior.wpm > 0.0 else 0.0 for text in texts
    )
    return ScriptMeasurement(
        per_line_seconds=per_line,
        measured=(False,) * len(per_line),
        failed_indices=tuple(range(len(per_line))),
        extrapolated_indices=tuple(range(len(per_line))),
        total_seconds=math.fsum(per_line),
        measured_count=0,
        wpm=prior.wpm,
        basis=prior.basis,
        prior=prior,
        reason=reason,
    )


@dataclass
class MeasurementRecorder:
    """Every ``measure_script_lines`` call the stage made, as ``(texts, voices)``.

    The CAST is the assertion this exists for: ``website_review_record`` casts each
    line to its speaker's voice, and ``deterministic_clip_id`` hashes that voice —
    so a measurement taken at any other voice measures audio nobody will hear AND
    stores clips under ids record will not look up, making the run pay for every
    line twice. A test can only see that by reading what was actually passed.
    """

    calls: list[tuple[tuple[str, ...], tuple[str | None, ...]]] = field(default_factory=list)
    #: ``speed`` and ``stage_name`` of every call — the other two thirds of the
    #: clip id.
    speeds: list[float | None] = field(default_factory=list)
    stage_names: list[str] = field(default_factory=list)

    @property
    def voices(self) -> tuple[str | None, ...]:
        """The per-line voices of the LAST measurement."""
        return self.calls[-1][1] if self.calls else ()

    @property
    def texts(self) -> tuple[str, ...]:
        return self.calls[-1][0] if self.calls else ()


def stub_speech_rate(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    default_wpm: float,
    rates: dict[str | None, float] | None = None,
) -> MeasurementRecorder:
    """Make the prior AND the full-script measurement answer, with no backend.

    ``rates`` maps a kokoro voice id to its rate; every voice absent from it
    (including ``None``, the backend default) delivers ``default_wpm``. The
    measurement stub reports ``basis=measured_all`` and ``reason=None`` — a LANDED
    measurement — because that is the production happy path, and a stub claiming
    ``calibration_passage`` with no reason would assert a state the real code never
    produces.
    """
    prior = SpeechRate(wpm=default_wpm, measured=True)
    # Patched on the SPEECH_RATE module, not on the stage: the stage now takes its
    # prior from `resolve_effective_wpm`, whose FIRST rung is the operator's authored
    # rate. Stubbing the stage's own name would replace that ladder with this file's
    # idea of it — and the authored rung is precisely what these stages are being
    # tested for. Patching the Phase-A probe underneath it removes the TTS call and
    # leaves the real precedence running.
    monkeypatch.setattr(_speech_rate, "resolve_natural_wpm_detail", AsyncMock(return_value=prior))
    recorder = MeasurementRecorder()

    async def _measure(
        _ctx: Any,
        *,
        texts: Sequence[str],
        voices: Sequence[str | None],
        speed: float | None = None,
        stage_name: str = "",
        **_kwargs: Any,
    ) -> ScriptMeasurement:
        recorder.calls.append((tuple(texts), tuple(voices)))
        recorder.speeds.append(speed)
        recorder.stage_names.append(stage_name)
        return measurement_at(texts, voices, rates or {}, default_wpm, prior)

    monkeypatch.setattr(module, "measure_script_lines", _measure)
    return recorder


def stub_failed_measurement(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, wpm: float, reason: FallbackReason
) -> None:
    """Phase A answers ``wpm``; the full-script measurement REFUSES with ``reason``."""
    prior = SpeechRate(wpm=wpm, measured=True)
    # Patched on the SPEECH_RATE module, not on the stage: the stage now takes its
    # prior from `resolve_effective_wpm`, whose FIRST rung is the operator's authored
    # rate. Stubbing the stage's own name would replace that ladder with this file's
    # idea of it — and the authored rung is precisely what these stages are being
    # tested for. Patching the Phase-A probe underneath it removes the TTS call and
    # leaves the real precedence running.
    monkeypatch.setattr(_speech_rate, "resolve_natural_wpm_detail", AsyncMock(return_value=prior))

    async def _measure(_ctx: Any, *, texts: Sequence[str], **_kwargs: Any) -> ScriptMeasurement:
        return refused_measurement(texts, prior, reason)

    monkeypatch.setattr(module, "measure_script_lines", _measure)
