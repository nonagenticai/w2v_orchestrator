"""Keep the website_review stage tests OFF the real TTS backend.

``website_review_script`` now measures every drafted line through TTS (Phase C)
on top of the Phase-A calibration probe it already took. Both build a real
``TTSClient`` against ``tts.test`` when nothing stubs them, and both absorb
the failure — so an unstubbed test does not go red, it goes SLOW and then
silently budgets at the 150.0 default. Measured on this directory before this
fixture existed: ``test_script.py::test_output_schema_names_phase_and_critique_item_id``
and ``test_script_shotlist.py::test_before_lines_authored_verbatim_with_shot_ids``
each burned exactly 30.00s — the probe's own timeout — and passed.

So the default here is a LANDED measurement at the documented default rate, which
is the number those runs were already getting after the timeout: same outputs, no
socket, no 30s. A test that is ABOUT the rate re-patches these names in its own
body (monkeypatch applies in order, so the test's patch wins), and a test that
wants the REAL resolution path restores it explicitly — see
``test_script_word_budget.py::test_unmeasurable_voice_falls_back_to_exactly_the_default_wpm``.
"""

from __future__ import annotations

import pytest

import src.services.stages.website_review.script as website_review_script
from src.services.stages.shared.time_budget import DEFAULT_WPM_NATURAL
from tests.services.stages.website_review.rate_stubs import stub_speech_rate


@pytest.fixture(autouse=True)
def _hermetic_speech_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_speech_rate(monkeypatch, website_review_script, DEFAULT_WPM_NATURAL)
