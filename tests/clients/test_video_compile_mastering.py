"""Unit tests for VideoCompileClient.compile mastering-block threading (no DB).

``compile`` folds an optional ``mastering`` dict into the submit payload,
mirroring the existing ``music`` / ``ambient`` request-level beds. These pin
that: a supplied block lands in the request payload; ``None`` omits the key so
the request is byte-identical to today. The submit→poll wrapper is stubbed so no
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


async def test_mastering_block_lands_in_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    client, captured = _client_capturing(monkeypatch)
    mastering = {"loudness_target": -12.5, "crf": 18, "colorspace": "bt709"}

    await client.compile(shots=[{"clip": 1}], mastering=mastering, idempotency_key=_IDEM_KEY)

    assert captured["payload"]["mastering"] == mastering
    assert captured["payload"]["shots"] == [{"clip": 1}]


async def test_mastering_none_omits_key(monkeypatch: pytest.MonkeyPatch) -> None:
    client, captured = _client_capturing(monkeypatch)

    await client.compile(shots=[{"clip": 1}], mastering=None, idempotency_key=_IDEM_KEY)

    assert "mastering" not in captured["payload"]


async def test_mastering_default_omits_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default call (no mastering kwarg) never sends a `mastering` key."""
    client, captured = _client_capturing(monkeypatch)

    await client.compile(shots=[{"clip": 1}], idempotency_key=_IDEM_KEY)

    assert "mastering" not in captured["payload"]
