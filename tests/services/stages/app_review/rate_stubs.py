"""Pin BOTH speech-rate phases for an app_review test that is not about rates.

``app_review_script`` resolves its narration rate twice: a Phase-A calibration
prior before the draw, and then a full-script MEASUREMENT of every drafted line
before the verdict. A test about the steer/re-ask/trim loop, or about the
word-budget sentence, wants one known rate and no TTS at all — so this stubs both
to the same number.

The measurement stub is deliberately not a constant: it returns the per-line
seconds ``wpm`` implies for each line's own word count, so a test's arithmetic is
the same arithmetic the stage does and a trim over measured seconds is exercised
rather than bypassed. A stub returning one flat total would make every per-line
decision (which line to drop) untestable.

Deliberately NOT a fixture: the tests here stub inside their own ``_run`` helpers,
and a module that only defines helpers (no ``test_`` prefix) is not collected.
"""

from __future__ import annotations

import math
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


def measurement_at(texts: Sequence[str], wpm: float, prior: SpeechRate) -> ScriptMeasurement:
    """A LANDED full-script measurement: every line at ``wpm``, nothing failed.

    ``wpm <= 0`` is passed through unchanged rather than substituted — the stage's
    own ``wpm_is_usable`` guard is exactly what such a test exercises, and a stub
    that quietly fixed the rate would make that guard untestable.
    """
    per_line = tuple((len(str(text).split()) * 60.0 / wpm) if wpm > 0.0 else 0.0 for text in texts)
    return ScriptMeasurement(
        per_line_seconds=per_line,
        measured=(True,) * len(per_line),
        failed_indices=(),
        extrapolated_indices=(),
        total_seconds=math.fsum(per_line),
        measured_count=len(per_line),
        wpm=wpm,
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
    per_line = tuple(len(str(text).split()) * 60.0 / prior.wpm for text in texts)
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


def stub_speech_rate(monkeypatch: pytest.MonkeyPatch, module: ModuleType, wpm: float) -> None:
    """Make the prior AND the full-script measurement answer ``wpm``, with no backend.

    The measurement stub reports ``basis=measured_all`` and ``reason=None`` — a
    LANDED measurement — because that is the production happy path, and because a
    stub claiming ``calibration_passage`` with no reason would assert a state the
    real code never produces.
    """
    prior = SpeechRate(wpm=wpm, measured=True)
    # Patched on the SPEECH_RATE module, not on the stage: the stage now takes its
    # prior from `resolve_effective_wpm`, whose FIRST rung is the operator's authored
    # rate. Stubbing the stage's own name would replace that ladder with this file's
    # idea of it — and the authored rung is precisely what these stages are being
    # tested for. Patching the Phase-A probe underneath it removes the TTS call and
    # leaves the real precedence running.
    monkeypatch.setattr(_speech_rate, "resolve_natural_wpm_detail", AsyncMock(return_value=prior))

    async def _measure(_ctx: Any, *, texts: Sequence[str], **_kwargs: Any) -> ScriptMeasurement:
        return measurement_at(texts, wpm, prior)

    monkeypatch.setattr(module, "measure_script_lines", _measure)


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
