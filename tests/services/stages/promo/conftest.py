"""Keep the promo stage tests OFF the real TTS backend.

``promo_script`` now measures every drafted line through TTS (Phase C) on top of
the Phase-A calibration probe it already took. Both build a real ``TTSClient``
against ``tts.test`` when nothing stubs them, and both absorb the failure — so
an unstubbed test does not go red, it goes SLOW and then silently budgets at the
150.0 default. This directory measured 95.53s for 166 tests before this fixture
existed, essentially all of it probe timeouts inside tests that are not about
rates at all.

So the default here is a LANDED measurement at the documented default rate, which
is the number those runs were already getting after the timeout: same outputs, no
socket, no wait. A test that is ABOUT the rate re-patches these names in its own
body (monkeypatch applies in order, so the test's patch wins), and a test that
wants the REAL resolution path restores it explicitly — see
``test_script_cast_word_budget.py::_arm_cast``.
"""

from __future__ import annotations

import pytest

import src.services.stages.promo.script as promo_script
from src.services.stages.shared.time_budget import DEFAULT_WPM_NATURAL
from tests.services.stages.promo.rate_stubs import stub_speech_rate


@pytest.fixture(autouse=True)
def _hermetic_speech_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_speech_rate(monkeypatch, promo_script, DEFAULT_WPM_NATURAL)
