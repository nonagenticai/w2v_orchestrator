"""Unit-test guards for the demo_video stages.

``ReconcileTimelineStage`` calibrates the voice's real speaking rate by synthesizing
one short sentence (``shared/speech_rate.py``). A stage constructed WITHOUT an explicit
``tts_client`` builds a real :class:`~src.clients.tts_client.TTSClient` on demand — so
in a unit test that lazy construction reaches for tts_backend over the network, hangs
until the probe's own deadline, and adds that wall-clock to every suite that touches the
stage. It is also, plainly, network I/O in a unit test.

The fixture below makes that construction IMPOSSIBLE rather than merely fast: the
default-client factory raises, ``speech_rate`` catches it exactly as it catches any
other probe failure, and the stage falls back to ``DEFAULT_WPM_NATURAL`` — the same
answer the hanging network attempt eventually produced, minus the hang.

A test that wants to exercise the MEASURED path injects its own fake client
(``ReconcileTimelineStage(tts_client=...)``), which never touches this factory.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, NoReturn

import pytest

from src.services.stages.shared import speech_rate


class _NoNetworkTTSClient:
    """Stands in for ``TTSClient`` in unit tests: constructing one is an error."""

    def __init__(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise RuntimeError(
            "speech_rate tried to build a real TTSClient in a unit test. Inject a fake "
            "via ReconcileTimelineStage(tts_client=...) to exercise the measured path."
        )


@pytest.fixture(autouse=True)
def _no_speech_rate_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Block the calibration probe's default client, and never memoise across tests."""
    monkeypatch.setattr(speech_rate, "TTSClient", _NoNetworkTTSClient)
    speech_rate.clear_measured_wpm_cache()
    yield
    speech_rate.clear_measured_wpm_cache()
