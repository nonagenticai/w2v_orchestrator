"""Pin BOTH speech-rate phases for a promo test that is not about rates.

``promo_script`` resolves its narration rate twice: a Phase-A calibration prior
(blended across the CAST when the promo is a two-hander) before the draw, and then
a full-script MEASUREMENT of every drafted line before the verdict. A test about
the steer/re-ask/trim loop, or about the word-budget sentence, wants known rates
and no TTS at all — so this stubs both.

**The measurement stub is PER VOICE.** ``promo_narrate`` casts each line to its
speaker's voice (``primitives/dialogue.voice_for_speaker`` on promo's own
``_DEFAULT_CAST``), so a stub returning one rate for the whole script would make
the cast invisible and the clip-reuse contract untestable. Each line's seconds is
its own word count at ITS OWN voice's rate, so a test's arithmetic is the
arithmetic the stage does and a trim over measured seconds is exercised rather
than bypassed.

Deliberately NOT a fixture (beyond the autouse hermetic default in ``conftest.py``):
tests that are ABOUT the rate stub inside their own ``_run`` helpers, and a module
that only defines helpers (no ``test_`` prefix) is not collected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

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
    seconds — exactly as the real ``measure_script_lines`` computes it, because that
    is the number the stage then writes the re-ask target and the word-proxy trim
    fallback from. A stub returning one voice's rate there would hide a mis-blended
    pool.

    A rate ``<= 0`` is passed through unchanged rather than substituted — the
    stage's own ``wpm_is_usable`` guard is exactly what such a test exercises.
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
    point, because that reason is what ``script_measurement_degradation`` turns into
    a row on the degrading tier.
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

    The CAST is the assertion this exists for: ``promo_narrate`` casts each line to
    its speaker's voice, and ``deterministic_clip_id`` hashes that voice — so a
    measurement taken at any other voice measures audio nobody will hear AND stores
    clips under ids narrate will not look up, making the run pay for every line
    twice. A test can only see that by reading what was actually passed.
    """

    calls: list[tuple[tuple[str, ...], tuple[str | None, ...]]] = field(default_factory=list)
    #: ``speed`` and ``stage_name`` of every call — the other two thirds of the clip id.
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

    ``resolve_cast_wpm`` is stubbed too: promo blends its PROMPT budget across the
    cast, and leaving that one live would send a two-hander test at a real TTS
    backend through the very path the prior stub was meant to close.
    """
    prior = SpeechRate(wpm=default_wpm, measured=True)
    monkeypatch.setattr(module, "resolve_natural_wpm_detail", AsyncMock(return_value=prior))
    recorder = MeasurementRecorder()

    async def _cast(
        _ctx: Any, *, segments: Sequence[tuple[str, str | None]], default: float, **_kw: Any
    ) -> Any:
        from src.services.stages.shared.speech_rate import CastSpeechRate

        per_voice = {
            voice: SpeechRate(wpm=_rate_for(voice, rates or {}, default_wpm), measured=True)
            for _, voice in segments
        }
        words = [len(str(text).split()) for text, _ in segments]
        seconds = math.fsum(
            count / per_voice[voice].wpm * 60.0
            for count, (_, voice) in zip(words, segments, strict=True)
            if count > 0
        )
        if seconds <= 0.0:
            return CastSpeechRate(wpm=default, seconds=0.0, per_voice=per_voice, weighted=False)
        return CastSpeechRate(
            wpm=sum(words) / (seconds / 60.0),
            seconds=seconds,
            per_voice=per_voice,
            weighted=True,
        )

    monkeypatch.setattr(module, "resolve_cast_wpm", _cast)

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


def stub_measurement_at_order(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, ordered: float, default_wpm: float
) -> None:
    """Every draft measures EXACTLY ``ordered`` seconds — in band, so nothing steers.

    For tests that are about something else entirely (the element-shape retry
    ladder, the bypass-vs-LLM branch) and whose stub draws are a handful of words
    against a 30s order, i.e. wildly "short" and therefore re-asked. Those tests
    assert exact ``fake.message`` call counts, and a correction they are not testing
    would silently redefine what they pin. Pinning the estimate ON the order is the
    honest way to hold the duration loop still: it exercises the real
    ``steer_verdict`` and gets ``"ok"`` from it.
    """
    prior = SpeechRate(wpm=default_wpm, measured=True)
    monkeypatch.setattr(module, "resolve_natural_wpm_detail", AsyncMock(return_value=prior))

    async def _measure(_ctx: Any, *, texts: Sequence[str], **_kwargs: Any) -> ScriptMeasurement:
        count = max(1, len(tuple(texts)))
        per_line = tuple(float(ordered) / count for _ in range(count))
        return ScriptMeasurement(
            per_line_seconds=per_line,
            measured=(True,) * count,
            failed_indices=(),
            extrapolated_indices=(),
            total_seconds=math.fsum(per_line),
            measured_count=count,
            wpm=default_wpm,
            basis=BASIS_MEASURED_ALL,
            prior=prior,
        )

    monkeypatch.setattr(module, "measure_script_lines", _measure)


def stub_failed_measurement(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, wpm: float, reason: FallbackReason
) -> None:
    """Phase A answers ``wpm``; the full-script measurement REFUSES with ``reason``."""
    prior = SpeechRate(wpm=wpm, measured=True)
    monkeypatch.setattr(module, "resolve_natural_wpm_detail", AsyncMock(return_value=prior))

    async def _measure(_ctx: Any, *, texts: Sequence[str], **_kwargs: Any) -> ScriptMeasurement:
        return refused_measurement(texts, prior, reason)

    monkeypatch.setattr(module, "measure_script_lines", _measure)
