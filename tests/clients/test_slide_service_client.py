"""Unit tests for SlideServiceClient.start_deck design-token threading (no DB).

``start_deck`` folds the run's resolved C23 design-token bundle into the deck
submit payload, mirroring the ``mastering`` / ``music`` optional blocks on
``VideoCompileClient.compile``. These pin that: a supplied bundle lands in the
request payload as ``tokens``; a falsy bundle omits the key so the request is
byte-identical to today. The submit wrapper is stubbed so no socket opens — the
assertion is purely on the payload handed to it.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.clients.slide_service_client import SlideServiceClient
from src.services.stages.shared.design_system import DEFAULT_TOKENS, as_params


def _client_capturing(monkeypatch: pytest.MonkeyPatch) -> tuple[SlideServiceClient, dict[str, Any]]:
    # base_url + license_token bypass settings/discovery; no real socket opens.
    client = SlideServiceClient(base_url="http://slides.test", license_token="t")
    captured: dict[str, Any] = {}

    async def _fake_submit_job(endpoint: str, payload: dict[str, Any]) -> str:
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return "deck-1"

    monkeypatch.setattr(client, "submit_job", _fake_submit_job)
    return client, captured


async def test_tokens_land_in_the_deck_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    client, captured = _client_capturing(monkeypatch)
    tokens = as_params(DEFAULT_TOKENS)

    job_id = await client.start_deck({"title": "T", "slides": []}, tokens=tokens)

    assert job_id == "deck-1"
    assert captured["endpoint"] == "/api/decks"
    assert captured["payload"]["tokens"] == tokens
    # Pin the key SET, not a count — a magic number silently drifts when the
    # design system grows a token, and says nothing about which one went missing.
    assert set(captured["payload"]["tokens"]) == set(as_params(DEFAULT_TOKENS))
    assert captured["payload"]["title"] == "T"  # the rest of the payload is untouched


async def test_no_tokens_omits_the_key_and_forwards_the_payload_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, captured = _client_capturing(monkeypatch)
    payload = {"title": "T", "slides": []}

    await client.start_deck(payload)

    assert "tokens" not in captured["payload"]
    # Strict identity: the very same object is submitted, zero keys added.
    assert captured["payload"] is payload


async def test_empty_tokens_is_treated_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    client, captured = _client_capturing(monkeypatch)
    payload = {"title": "T", "slides": []}

    await client.start_deck(payload, tokens={})

    assert "tokens" not in captured["payload"]
    assert captured["payload"] is payload
