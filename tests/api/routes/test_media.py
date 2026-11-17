"""/api/media/search route — trimmed media-discovery search for the wizard panel."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.clients.schemas.media_discovery import MediaSearchResponse

pytestmark = pytest.mark.asyncio


async def test_search_returns_trimmed_results(client) -> None:
    upstream = MediaSearchResponse.model_validate(
        {
            "results": [
                {
                    "id": "m1",
                    "title": "A dog",
                    "url": "https://img/1.jpg",
                    "thumbnail_url": "https://img/1-t.jpg",
                    "media_type": "image",
                    "source": "unsplash",
                    "author_name": "Jane",
                    "quality_score": 0.8,
                    "extra_field_ignored": "x",
                },
                # Empty id → dropped by the route (the server contract makes a
                # MISSING id impossible — response_model requires it).
                {"id": "", "title": "no id — dropped", "url": "u", "media_type": "", "source": ""},
            ],
            "total": 1,
            "sources_queried": ["unsplash"],
            "sources_failed": [],
            "search_time_ms": 3.2,
        }
    )
    with patch(
        "src.api.routes.media.MediaDiscoveryClient",
        return_value=AsyncMock(search=AsyncMock(return_value=upstream)),
    ):
        r = await client.post("/api/media/search", json={"query": "dog", "limit": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["results"]) == 1
    item = body["results"][0]
    assert item["id"] == "m1"
    assert item["thumbnail_url"] == "https://img/1-t.jpg"
    assert "extra_field_ignored" not in item
    assert body["sources_queried"] == ["unsplash"]


async def test_search_rejects_empty_query(client) -> None:
    r = await client.post("/api/media/search", json={"query": ""})
    assert r.status_code == 422


async def test_search_upstream_failure_is_502(client) -> None:
    with patch(
        "src.api.routes.media.MediaDiscoveryClient",
        return_value=AsyncMock(search=AsyncMock(side_effect=RuntimeError("media-discovery down"))),
    ):
        r = await client.post("/api/media/search", json={"query": "dog"})
    assert r.status_code == 502


# ─── GET /api/media/thumb — same-origin thumbnail proxy ─────────────────────

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


async def test_thumb_proxies_an_allowed_cdn_image(client) -> None:
    fetch = AsyncMock(return_value=(_PNG, "png"))
    with patch("src.api.routes.media.fetch_image_bytes", fetch):
        r = await client.get(
            "/api/media/thumb", params={"url": "https://i.ytimg.com/vi/abc/hqdefault.jpg"}
        )
    assert r.status_code == 200, r.text
    assert r.content == _PNG
    assert r.headers["content-type"] == "image/png"
    assert r.headers["cache-control"] == "private, max-age=86400"
    args, kwargs = fetch.call_args
    assert args[0] == "https://i.ytimg.com/vi/abc/hqdefault.jpg"
    assert kwargs["max_bytes"] == 5 * 1024 * 1024


async def test_thumb_rejects_a_host_off_the_allow_list(client) -> None:
    fetch = AsyncMock(return_value=(_PNG, "png"))
    with patch("src.api.routes.media.fetch_image_bytes", fetch):
        r = await client.get("/api/media/thumb", params={"url": "https://evil.example/x.png"})
    assert r.status_code == 400
    fetch.assert_not_called()


async def test_thumb_refusal_is_a_typed_body_the_fe_can_render(client) -> None:
    """The refusal used to be the bare string "thumbnail host is not allowed", which
    reached the operator as a broken <img> with nothing to say. The FE switches on
    ``reason`` and renders ``message``; ``host`` names the CDN to consider admitting."""
    with patch("src.api.routes.media.fetch_image_bytes", AsyncMock()):
        r = await client.get(
            "/api/media/thumb", params={"url": "https://artwork.somepodcast.example/a.jpg"}
        )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["reason"] == "host_not_allowed"
    assert detail["host"] == "artwork.somepodcast.example"
    assert "not supported" in detail["message"]

    with patch("src.api.routes.media.fetch_image_bytes", AsyncMock()):
        bad = await client.get("/api/media/thumb", params={"url": "file:///etc/passwd"})
    assert bad.status_code == 400
    assert bad.json()["detail"]["reason"] == "invalid_url"


async def test_thumb_allow_list_comes_from_config_not_a_module_literal(client, monkeypatch) -> None:
    """A newly federated adapter's CDN is admitted from the gitops ConfigMap, without a
    code deploy — and the added entry is still matched on a dot boundary."""
    from src.config import clear_settings_cache

    monkeypatch.setenv(
        "API_THUMB_ALLOWED_HOST_SUFFIXES", '["ytimg.com", "cdn.newprovider.example"]'
    )
    clear_settings_cache()
    try:
        fetch = AsyncMock(return_value=(_PNG, "png"))
        with patch("src.api.routes.media.fetch_image_bytes", fetch):
            ok = await client.get(
                "/api/media/thumb", params={"url": "https://img.cdn.newprovider.example/1.jpg"}
            )
            spoof = await client.get(
                "/api/media/thumb", params={"url": "https://evil-cdn.newprovider.example/1.jpg"}
            )
            dropped = await client.get(
                "/api/media/thumb", params={"url": "https://i.vimeocdn.com/video/1.jpg"}
            )
        assert ok.status_code == 200, ok.text
        assert spoof.status_code == 400
        # vimeocdn is a DEFAULT entry, not a configured one — config REPLACES the list.
        assert dropped.status_code == 400
    finally:
        monkeypatch.delenv("API_THUMB_ALLOWED_HOST_SUFFIXES", raising=False)
        clear_settings_cache()


def test_a_single_label_suffix_is_refused_at_config_load(monkeypatch) -> None:
    """``com`` would allow-list an entire TLD — i.e. turn the proxy open. It must fail
    at boot, not widen the allow-list at runtime."""
    import pytest as _pytest

    from src.config import APISettings

    monkeypatch.setenv("API_THUMB_ALLOWED_HOST_SUFFIXES", '["com"]')
    with _pytest.raises(ValueError, match="no dot"):
        APISettings()
    monkeypatch.setenv("API_THUMB_ALLOWED_HOST_SUFFIXES", '["*.ytimg.com", ".vimeocdn.com"]')
    assert APISettings().thumb_allowed_host_suffixes == ["ytimg.com", "vimeocdn.com"]


@pytest.mark.parametrize(
    "entry",
    ["ytimg.com:8080"],
)
def test_an_entry_that_is_not_a_bare_hostname_is_refused_at_config_load(
    monkeypatch, entry: str
) -> None:
    """Each of these passes the dot check and then matches NOTHING — the operator reads
    an admitted CDN, the proxy has a dead entry. That is the same "hole that looks
    closed" class the normaliser exists to prevent, so it must fail at boot too."""
    from src.config import APISettings

    monkeypatch.setenv("API_THUMB_ALLOWED_HOST_SUFFIXES", json.dumps([entry]))
    with pytest.raises(ValueError, match="not a bare hostname"):
        APISettings()


async def test_thumb_upstream_failure_is_502(client) -> None:
    with patch(
        "src.api.routes.media.fetch_image_bytes", AsyncMock(side_effect=ValueError("not an image"))
    ):
        r = await client.get(
            "/api/media/thumb", params={"url": "https://i.vimeocdn.com/video/1.jpg"}
        )
    assert r.status_code == 502


async def test_thumb_refuses_svg_as_a_typed_400_not_a_server_error(client) -> None:
    """A live "Find media" result is ``…Microsoft_Office_Teams_(2025-present).svg``. The
    asset is healthy; we decline to proxy script-bearing markup as an image. That is OUR
    refusal, so it gets the same typed shape as ``host_not_allowed`` — the FE can say
    "can't preview a vector image" instead of showing a server error.

    The bytes are still never served: assert the response body is not the SVG."""
    svg = b"<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>"
    with patch("src.api.routes.media.fetch_image_bytes", AsyncMock(return_value=(svg, "jpg"))):
        r = await client.get(
            "/api/media/thumb",
            params={"url": "https://upload.wikimedia.org/a/Microsoft_Office_Teams.svg"},
        )
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert detail["reason"] == "vector_image_unsupported"
    assert detail["host"] == "upload.wikimedia.org"
    assert "vector image" in detail["message"]
    assert svg not in r.content


_SVG_DOC = "<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>"


@pytest.mark.parametrize(
    ("label", "body"),
    [("utf-16le + BOM", b"\xff\xfe" + _SVG_DOC.encode("utf-16-le"))],
)
async def test_thumb_refuses_utf16_svg_as_the_typed_400_not_a_502(
    client, label: str, body: bytes
) -> None:
    """A UTF-16 encoded SVG is a HEALTHY asset we simply decline to preview — the exact
    class of confusing server-error this PR set out to remove. The sniff matched bytes
    literally, so ``<\x00s\x00v\x00g\x00`` never matched ``b"<svg"`` and every one of
    these fell through to the 502. (``str.encode("utf-16-le"/"-be")`` emits no BOM, so
    the BOM cases prepend it explicitly.)

    The bytes are still NEVER served — asserted on every case. Recognising more shapes
    can only refuse more, never proxy: ``looks_like_svg`` is read inside the branch where
    the bytes already failed ``sniff_image_format``, and both arms of it are refusals."""
    with patch("src.api.routes.media.fetch_image_bytes", AsyncMock(return_value=(body, "jpg"))):
        r = await client.get(
            "/api/media/thumb",
            params={"url": "https://upload.wikimedia.org/a/Teams.svg"},
        )
    assert r.status_code == 400, f"{label}: {r.text}"
    detail = r.json()["detail"]
    assert detail["reason"] == "vector_image_unsupported"
    assert detail["host"] == "upload.wikimedia.org"
    assert body not in r.content


async def test_thumb_non_raster_that_is_not_svg_is_still_502(client) -> None:
    """The typed SVG refusal must not swallow a genuinely broken payload — HTML from a
    hotlink-blocking CDN is a FAILURE, not a deliberate refusal, and stays a 502."""
    with patch(
        "src.api.routes.media.fetch_image_bytes",
        AsyncMock(return_value=(b"<html><body>403 Forbidden</body></html>", "jpg")),
    ):
        r = await client.get(
            "/api/media/thumb", params={"url": "https://upload.wikimedia.org/a/b.jpg"}
        )
    assert r.status_code == 502


@pytest.mark.parametrize(
    ("url", "allowed"),
    [("https://i.ytimg.com/vi/x/0.jpg", True)],
)
def test_thumb_host_allow_list(url: str, allowed: bool) -> None:
    from src.api.routes.media import thumb_host_allowed

    assert thumb_host_allowed(url) is allowed
