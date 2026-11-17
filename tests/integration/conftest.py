"""Fixtures shared by the in-process full-DAG dispatch tests.

Holds exactly one thing: the autouse seal over the speech-rate measurement's TTS
client. See `_stub_speech_rate_tts` for why it lives here and is autouse rather
than being another `patch(...)` line in each archetype test's mock stack.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.services.stages.shared import speech_rate
from tests.integration._dispatch_harness import _speech_rate_tts_mock


@pytest.fixture(autouse=True)
def _stub_speech_rate_tts():
    """Seal the speech-rate measurement seam for EVERY test in this package.

    The script stages no longer guess how long a narration line takes — they
    synthesize it and measure it (`speech_rate.measure_script_lines`, plus the
    `resolve_natural_wpm` / `resolve_script_sample_wpm` probes). That client is
    built at `speech_rate`'s own import site, which is NOT the
    `primitives.narration.TTSClient` (nor `app_review.record.TTSClient`) that the
    archetype tests patch — so without this fixture the measurement escapes to the
    live `tts.test` unit on mini2 and the DAG test times out with its run
    still `running` (observed: `app_review` + `website_review`, ~37s to a poll
    budget exhaustion, unbounded when the backend is slow rather than absent).

    Autouse and package-wide ON PURPOSE. Per-test patches are what produced the
    bug: `app_review`'s measurement shipped to main and no archetype test's mock
    stack grew a line for it, because nothing forces one. A stage newly reaching
    for the measurement is covered here the day it is written; a test that must
    exercise the real client can still opt out by patching over it locally.

    Not a no-op: `_speech_rate_tts_mock` returns real per-line durations derived
    from each line's word count, so the measurement/extrapolation code path runs
    for real and only the network hop is replaced.

    The memoised rate cache is cleared on both sides — it is process-wide and
    keyed by (provider, voice), so one test's stubbed rate would otherwise be
    served to the next test (and to any unit test sharing the process) without a
    single synthesis being attempted.
    """
    speech_rate.clear_measured_wpm_cache()
    with patch.object(speech_rate, "TTSClient", return_value=_speech_rate_tts_mock()):
        yield
    speech_rate.clear_measured_wpm_cache()
