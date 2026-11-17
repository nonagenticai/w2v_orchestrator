"""Keep the lesson stage tests OFF the real TTS backend.

``lesson_script`` now measures every drafted paragraph through TTS on top of the
Phase-A calibration probe it already took. Both build a real ``TTSClient``
against ``tts.test`` when nothing stubs them, and both absorb the failure —
so an unstubbed test does not go red, it goes SLOW and then silently budgets at
the 150.0 default.

Measured on this directory: NINE tests in ``test_script.py`` and
``test_script_word_budget.py`` called ``stage.run()`` with nothing stubbed in the
speech-rate layer and each burned the probe's own 30.00s timeout — the whole
lesson suite took 204.17s before this fixture existed, and would have taken far
longer once the stage synthesizes EVERY line rather than one calibration
passage.

So the default here is a LANDED measurement at the documented default rate, which
is the number those runs were already getting after the timeout: same outputs, no
socket, no 30s. A test that is ABOUT the rate re-patches these names in its own
body (monkeypatch applies in order, so the test's patch wins), and a test that
wants the REAL resolution path restores it explicitly — see
``test_script_word_budget.py::test_unmeasurable_voice_falls_back_to_exactly_the_default_wpm``.
"""

from __future__ import annotations

import pytest

import src.services.stages.lesson.script as lesson_script
from src.services.stages.shared.time_budget import DEFAULT_WPM_NATURAL
from tests.services.stages.lesson.rate_stubs import stub_speech_rate


@pytest.fixture(autouse=True)
def _hermetic_speech_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_speech_rate(monkeypatch, lesson_script, DEFAULT_WPM_NATURAL)
