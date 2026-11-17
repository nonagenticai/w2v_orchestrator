"""A measurement that EQUALS the default is still a measurement.

``ReconcileTimelineStage.handle`` used to decide whether the speech-rate probe had
answered by comparing its return value to ``DEFAULT_WPM_NATURAL``::

    measured = await resolve_natural_wpm(...)
    if measured != DEFAULT_WPM_NATURAL:   # ← 150.0 means "the probe failed"

The fallback constant and a genuine measurement are the same Python float. The rate
measured off the live voice today is ~151.6 wpm against a 150.0 default, so a voice
that really does speak at exactly 150.0 is one plausible draw away — and that draw was
filed as a probe failure and thrown away.

The stage now asks :func:`resolve_natural_wpm_detail`, which reports measured-ness
explicitly, and branches on ``rate.measured``. The two contracts pinned here:

* a probe that measures EXACTLY 150.0 re-resolves the timing config against the
  measurement (it is a value, not a sentinel);
* a probe that cannot answer re-resolves NOTHING — ``cfg`` keeps the value the
  pre-measurement stage produced, byte-identical, which is the deliberate
  never-fail-a-run guarantee of ``shared/speech_rate.py``.

Both are asserted on the CALLS to ``_resolve_timing_config`` rather than on the
resolved numbers, because at ``natural == 150.0`` the two branches produce the same
config — which is exactly why the defect survived: it is invisible in the output and
lethal to anything that later hangs off "did we measure?".
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pytest

import src.services.stages.demo_video.reconcile_timeline as reconcile_timeline
from src.clients.tts_client import TTSClient
from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.reconcile_timeline import ReconcileTimelineStage
from src.services.stages.shared.speech_rate import CALIBRATION_WORDS, clear_measured_wpm_cache
from src.services.stages.shared.time_budget import DEFAULT_WPM_NATURAL

#: A synthesis duration that makes the probe measure the default rate to the bit.
_DURATION_AT_EXACTLY_THE_DEFAULT = CALIBRATION_WORDS * 60.0 / DEFAULT_WPM_NATURAL

_LINE = "This beat introduces the product in one short line."


class _FakeTTS:
    """``duration`` is the seconds this backend reports for the ``CALIBRATION_WORDS``
    passage; every other call (Phase B samples a different-length script line) scales
    proportionally off the SAME implied wpm, so a voice this fixture models answers
    any text at one consistent rate rather than a duration pinned to one text's word
    count. ``duration=0.0`` (unusable) still yields 0.0 for every call, unchanged."""

    def __init__(self, *, duration: float, error: Exception | None = None) -> None:
        self.duration = duration
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def synthesize(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.duration <= 0.0:
            return type("_Synth", (), {"duration_seconds": self.duration})()
        wpm = CALIBRATION_WORDS * 60.0 / self.duration
        words = len(str(kwargs.get("text", "")).split())
        return type("_Synth", (), {"duration_seconds": words * 60.0 / wpm})()


@pytest.fixture(autouse=True)
def _passage_rate_is_the_prior(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the passage->narration ratio at 1.0: these tests prove the measured-rate
    PLUMBING with arithmetic written at the raw passage rate. The ratio itself is
    pinned by ``tests/services/stages/shared/test_speech_rate_narration_prior.py``."""
    from src.services.stages.shared import speech_rate

    monkeypatch.setattr(speech_rate, "PASSAGE_TO_NARRATION_RATIO", 1.0)


@pytest.fixture(autouse=True)
def _clean_cache() -> Any:
    clear_measured_wpm_cache()
    yield
    clear_measured_wpm_cache()


@pytest.fixture
def resolutions(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every ``_resolve_timing_config`` call without changing its behaviour."""
    calls: list[dict[str, Any]] = []
    real = reconcile_timeline._resolve_timing_config

    def _spy(ctx: RunContext, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return real(ctx, **kwargs)

    monkeypatch.setattr(reconcile_timeline, "_resolve_timing_config", _spy)
    return calls


def _ctx() -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "direction": {"timing": {"mode": "auto"}},
            "deliverable": {"target_duration_seconds": 8.0},
            "voice": {"voice_id": "af_heart"},
        },
        prior_run_outputs={
            "demo_video_gen": {
                "lines": [
                    {"id": "clip-1", "text": _LINE, "seconds": 3.0, "clip": None},
                    {"id": "clip-2", "text": _LINE, "seconds": 3.0, "clip": None},
                ],
                "bdd_job_id": "",
                "features_generated": 0,
                "package_url": "",
            },
            "demo_video_record": {"clips": []},
        },
    )


@pytest.mark.asyncio
async def test_a_probe_measuring_exactly_the_default_counts_as_measured(
    resolutions: list[dict[str, Any]],
) -> None:
    """150.0 off a real synthesis is a VALUE, not the no-measurement sentinel."""
    fake = _FakeTTS(duration=_DURATION_AT_EXACTLY_THE_DEFAULT)
    stage = ReconcileTimelineStage(tts_client=cast("TTSClient", fake))

    await stage.handle(stage.input_model(), _ctx())

    # The probe really did run and really did measure the default rate.
    assert len(fake.calls) == 1
    # First resolution is the opt-in check (no measurement yet); the SECOND only
    # happens when the stage accepted the measurement.
    assert resolutions == [{}, {"default_natural": DEFAULT_WPM_NATURAL}]


@pytest.mark.asyncio
async def test_a_probe_that_cannot_answer_re_resolves_nothing(
    resolutions: list[dict[str, Any]],
) -> None:
    """The never-fail guarantee: no measurement ⇒ ``cfg`` is left exactly as resolved."""
    fake = _FakeTTS(duration=0.0, error=RuntimeError("tts down"))
    stage = ReconcileTimelineStage(tts_client=cast("TTSClient", fake))

    await stage.handle(stage.input_model(), _ctx())

    assert len(fake.calls) == 1
    assert resolutions == [{}]


@pytest.mark.asyncio
async def test_an_unusable_duration_is_not_mistaken_for_a_measurement(
    resolutions: list[dict[str, Any]],
) -> None:
    """A backend that answers with a zero duration measured nothing either."""
    fake = _FakeTTS(duration=0.0)
    stage = ReconcileTimelineStage(tts_client=cast("TTSClient", fake))

    await stage.handle(stage.input_model(), _ctx())

    assert resolutions == [{}]
