"""Unit tests for VideoCompileClient.compile emit_shot_timeline threading (no DB).

``compile`` folds an opt-in ``emit_shot_timeline`` flag into the submit payload
(A11 momentum consume). These pin that: passing ``True`` lands the key so the
compiler runs its authoritative per-beat timeline pass; the default omits the
key so a non-gated compile sends a byte-identical payload. The submit→poll
wrapper is stubbed so no socket opens — the assertion is purely on the payload.
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


async def test_emit_shot_timeline_true_lands_in_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    client, captured = _client_capturing(monkeypatch)

    await client.compile(shots=[{"clip": 1}], emit_shot_timeline=True, idempotency_key=_IDEM_KEY)

    assert captured["payload"]["emit_shot_timeline"] is True
    assert captured["payload"]["shots"] == [{"clip": 1}]


async def test_emit_shot_timeline_default_omits_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default call (no emit_shot_timeline kwarg) never sends the key (byte-identical)."""
    client, captured = _client_capturing(monkeypatch)

    await client.compile(shots=[{"clip": 1}], idempotency_key=_IDEM_KEY)

    assert "emit_shot_timeline" not in captured["payload"]
