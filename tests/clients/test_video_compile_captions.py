"""Unit tests for VideoCompileClient.compile captions_render threading (no DB).

``compile`` folds the optional ``captions_render`` mode (burned_in|overlay)
alongside the ``captions`` cues, telling the compiler whether to BURN the cues
into the picture or MUX them as a selectable soft mov_text track. These pin the
byte-identity contract: ``captions_render`` only ever lands in the payload WITH
captions; a compile without captions never carries it (so the request is
byte-identical to before this wiring). The submit→poll wrapper is stubbed so no
socket opens — the assertion is purely on the payload handed to it.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.clients.video_compile import VideoCompileClient


def _client_capturing(monkeypatch: pytest.MonkeyPatch) -> tuple[VideoCompileClient, dict[str, Any]]:
    # base_url + license_token bypass settings/discovery; no real socket opens.
    client = VideoCompileClient(base_url="http://compile.test", license_token="t")
    captured: dict[str, Any] = {}

    async def _fake_submit_and_await(
        endpoint: str,
        payload: dict[str, Any],
        *,
        timeout: float,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return {"status": "completed", "artifact_url": "s3://out.mp4", "duration_seconds": 1.0}

    monkeypatch.setattr(client, "_submit_and_await", _fake_submit_and_await)
    return client, captured


_IDEM_KEY = "11111111-1111-7111-8111-111111111111:demo_video_compile"

_CAPS = [{"shot_index": 0, "cues": [{"text": "hi", "start": 0.0, "end": 1.0}]}]


async def test_captions_render_rides_with_cues(monkeypatch: pytest.MonkeyPatch) -> None:
    """overlay mode + cues → both land; the compiler muxes a soft track."""
    client, captured = _client_capturing(monkeypatch)

    await client.compile(
        shots=[{"clip": 1}], captions=_CAPS, captions_render="overlay", idempotency_key=_IDEM_KEY
    )

    assert captured["payload"]["captions"] == _CAPS
    assert captured["payload"]["captions_render"] == "overlay"


async def test_captions_render_burned_in_rides_with_cues(monkeypatch: pytest.MonkeyPatch) -> None:
    client, captured = _client_capturing(monkeypatch)

    await client.compile(
        shots=[{"clip": 1}],
        captions=_CAPS,
        captions_render="burned_in",
        idempotency_key=_IDEM_KEY,
    )

    assert captured["payload"]["captions_render"] == "burned_in"


async def test_captions_render_omitted_without_cues(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mode with NO cues (the common none-mode compile) must never send
    captions_render — the request stays byte-identical to before this wiring."""
    client, captured = _client_capturing(monkeypatch)

    await client.compile(
        shots=[{"clip": 1}], captions=None, captions_render="none", idempotency_key=_IDEM_KEY
    )

    assert "captions" not in captured["payload"]
    assert "captions_render" not in captured["payload"]


async def test_captions_render_default_omits_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default call (no captions_render kwarg) with cues never sends the key ⇒
    the compiler applies its own default (``overlay`` since compiler #247;
    subtitles are never burned in unless asked)."""
    client, captured = _client_capturing(monkeypatch)

    await client.compile(shots=[{"clip": 1}], captions=_CAPS, idempotency_key=_IDEM_KEY)

    assert captured["payload"]["captions"] == _CAPS
    assert "captions_render" not in captured["payload"]
