"""Unit tests for VideoCompileClient.compile finish-block threading (no DB).

``compile`` folds an optional ``finish`` dict into the submit payload, mirroring
the ``mastering`` block exactly. These pin that: a supplied block lands in the
request payload; ``None`` omits the key so the request is byte-identical to
before the color-finish wiring. The submit→poll wrapper is stubbed so no socket
opens — the assertion is purely on the payload handed to it.
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
        captured["idempotency_key"] = idempotency_key
        return {"status": "completed", "artifact_url": "s3://out.mp4", "duration_seconds": 1.0}

    monkeypatch.setattr(client, "_submit_and_await", _fake_submit_and_await)
    return client, captured


_IDEM_KEY = "11111111-1111-7111-8111-111111111111:demo_video_compile"


async def test_finish_block_lands_in_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    client, captured = _client_capturing(monkeypatch)
    finish = {"letterbox_color": "#280983"}

    await client.compile(shots=[{"clip": 1}], finish=finish, idempotency_key=_IDEM_KEY)

    assert captured["payload"]["finish"] == finish
    assert captured["payload"]["shots"] == [{"clip": 1}]


async def test_finish_none_omits_key(monkeypatch: pytest.MonkeyPatch) -> None:
    client, captured = _client_capturing(monkeypatch)

    await client.compile(shots=[{"clip": 1}], finish=None, idempotency_key=_IDEM_KEY)

    assert "finish" not in captured["payload"]


async def test_finish_default_omits_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default call (no finish kwarg) never sends a `finish` key."""
    client, captured = _client_capturing(monkeypatch)

    await client.compile(shots=[{"clip": 1}], idempotency_key=_IDEM_KEY)

    assert "finish" not in captured["payload"]


async def test_finish_does_not_disturb_mastering(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two blocks are independent — threading finish must not drop mastering."""
    client, captured = _client_capturing(monkeypatch)

    await client.compile(
        shots=[{"clip": 1}],
        mastering={"loudness_target": -12.5},
        finish={"letterbox_color": "#000000"},
        idempotency_key=_IDEM_KEY,
    )

    assert captured["payload"]["mastering"] == {"loudness_target": -12.5}
    assert captured["payload"]["finish"] == {"letterbox_color": "#000000"}
