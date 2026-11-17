"""MediaDiscoveryClient — direct media-discovery ``POST /api/search`` (KAG removal)."""

from __future__ import annotations

from typing import Any

import pytest

from src.clients.base import LicensedServiceClient
from src.clients.media_discovery_client import MediaDiscoveryClient
from src.clients.schemas.media_discovery import MediaSearchResponse

pytestmark = pytest.mark.asyncio

_WIRE = {
    "results": [
        {
            "id": "m1",
            "title": "Skyline",
            "url": "https://upload.wikimedia.org/x.jpg",
            "media_type": "image",
            "source": "wikimedia",
            "thumbnail_url": "https://upload.wikimedia.org/x-t.jpg",
            "quality_score": None,
            "tenant_id": "dropped",
        }
    ],
    "total": 1,
    "sources_queried": ["wikimedia"],
    "sources_failed": [],
    "search_time_ms": 4.5,
}


async def test_resolves_the_media_discovery_peer_by_default() -> None:
    client = MediaDiscoveryClient(license_token="t")
    assert isinstance(client, LicensedServiceClient)
    assert client.discovery_name == "media_discovery"
    assert client.base_url.startswith("http://media-discovery.media-discovery:8809")


async def test_search_posts_api_search_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((method, path, kwargs))
        return _WIRE

    client = MediaDiscoveryClient(license_token="t")
    monkeypatch.setattr(client, "_request", fake_request)
    resp = await client.search(
        "city skyline", sources=["wikimedia"], media_types=["image"], limit=5
    )
    assert calls == [
        (
            "POST",
            "/api/search",
            {
                "json": {
                    "query": "city skyline",
                    "limit": 5,
                    "sources": ["wikimedia"],
                    "media_types": ["image"],
                }
            },
        )
    ]
    assert isinstance(resp, MediaSearchResponse)
    assert resp.results[0].quality_score is None
    assert not hasattr(resp.results[0], "tenant_id")


async def test_search_omits_unset_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs["json"])
        return _WIRE

    client = MediaDiscoveryClient(license_token="t")
    monkeypatch.setattr(client, "_request", fake_request)
    await client.search("dogs")
    assert seen == {"query": "dogs", "limit": 20}


@pytest.mark.parametrize(("asked", "sent"), [(500, 100)])
async def test_search_clamps_limit_to_the_server_bound(
    monkeypatch: pytest.MonkeyPatch, asked: int, sent: int
) -> None:
    seen: dict[str, Any] = {}

    async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs["json"])
        return _WIRE

    client = MediaDiscoveryClient(license_token="t")
    monkeypatch.setattr(client, "_request", fake_request)
    await client.search("dogs", limit=asked)
    assert seen["limit"] == sent
