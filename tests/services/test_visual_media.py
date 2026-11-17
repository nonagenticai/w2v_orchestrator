"""visual_media — real-media fetch/re-publish + collection helpers."""

from __future__ import annotations

import ipaddress
import socket
from unittest.mock import AsyncMock

import httpx
import pytest
from w2v_common.net import safe_fetch

from src.services import visual_media
from src.services.visual_media import (
    MEDIA_FETCH_USER_AGENT,
    assign_real_media_by_line,
    assign_real_media_videos_by_line,
    collect_ai_video_urls,
    collect_motion_clips_by_line,
    collect_motion_slides_by_line,
    collect_real_media_urls,
    collect_real_media_videos,
    fetch_image_bytes,
    fetch_video_bytes,
    is_fetchable_url,
    looks_like_svg,
    republish_image,
    republish_video,
    sniff_image_format,
)


def _shots(*line_ids: str) -> list[dict[str, str]]:
    return [{"id": f"s{i}", "line_id": lid} for i, lid in enumerate(line_ids)]


def test_collect_motion_clips_by_line_splits_keyed_and_pool() -> None:
    out = {
        "clips": [
            {"line_id": "shot-001", "video_url": "https://m/a.mp4"},
            {"line_id": None, "video_url": "https://m/b.mp4"},
            {"video_url": "https://m/c.mp4"},  # no line_id key → pool
            {"line_id": "shot-002", "video_url": "/api/runs/r/artifact/d.mp4"},  # proxy ok
            {"line_id": "shot-003", "video_url": "file:///nope.mp4"},  # unfetchable → dropped
            {"line_id": "shot-001", "video_url": "https://m/a.mp4"},  # dup → dropped
        ]
    }
    keyed, pool = collect_motion_clips_by_line(out)
    assert keyed == {"shot-001": "https://m/a.mp4", "shot-002": "/api/runs/r/artifact/d.mp4"}
    assert pool == ["https://m/b.mp4", "https://m/c.mp4"]


def test_collect_motion_clips_by_line_empty_when_skipped() -> None:
    assert collect_motion_clips_by_line(None) == ({}, [])
    assert collect_motion_clips_by_line({}) == ({}, [])


def test_collect_motion_slides_by_line_keys_fetchable_clips() -> None:
    slides = {
        "slides": [
            {
                "line_id": "l0",
                "image_url": "https://s/0.png",
                "motion_video_url": "https://m/0.mp4",
            },
            {"line_id": "l1", "image_url": "https://s/1.png"},  # no motion → not keyed
            {"line_id": "l2", "image_url": "https://s/2.png", "motion_video_url": "file:///x.mp4"},
            {"line_id": "l3", "image_url": "https://s/3.png", "motion_video_url": None},
        ]
    }
    assert collect_motion_slides_by_line(slides) == {"l0": "https://m/0.mp4"}


def test_collect_motion_slides_by_line_empty_when_absent() -> None:
    assert collect_motion_slides_by_line(None) == {}
    assert collect_motion_slides_by_line({}) == {}
    assert collect_motion_slides_by_line({"slides": []}) == {}


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# The fetch guard is now w2v_common.net.safe_get, which RESOLVES every host (the
# old local guard never did — that was the bypass). These tests must therefore
# resolve offline and deterministically.
_STUB_DNS = {
    "cdn": "93.184.216.34",
    "upload.wikimedia.org": "185.15.59.240",
    "redirected.cdn": "93.184.216.35",
    "localhost": "127.0.0.1",
    "kag.internal": "192.0.2.11",  # Tailnet CGNAT — is_private is False for this
    "minio": "10.43.0.5",
}


@pytest.fixture(autouse=True)
def _stub_dns(monkeypatch) -> None:
    def fake_getaddrinfo(host, port, *args, **kwargs):
        addr = _STUB_DNS.get(host)
        if addr is None:
            try:
                ipaddress.ip_address(host)  # a literal IP resolves to itself
            except ValueError:
                raise socket.gaierror(f"stubbed DNS: unknown host {host!r}") from None
            addr = host
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, port or 0))]

    monkeypatch.setattr(safe_fetch.socket, "getaddrinfo", fake_getaddrinfo)


def test_is_fetchable_url() -> None:
    assert is_fetchable_url("https://minio.internal/x.png") is True
    assert is_fetchable_url("http://cdn/x.jpg") is True
    assert is_fetchable_url("/api/runs/abc/artifact/x.png") is True
    assert is_fetchable_url("/api/concept-images/a/download") is False
    assert is_fetchable_url("file:///tmp/x.png") is False
    assert is_fetchable_url(None) is False


def test_collect_real_media_urls_priority_dedup_and_filter() -> None:
    urls = collect_real_media_urls(
        source_ingest={
            "pinned_media": [
                {"ref": "https://x/pin1", "composite_url": "https://minio/p1.png"},
                {"ref": "https://x/pin2", "composite_url": ""},  # not an image → skipped
                {"ref": "https://x/pin3"},  # no composite_url → skipped
                # a re-published VIDEO pin → belongs to videos, NOT images
                {"ref": "https://x/v", "composite_url": "https://minio/p.mp4", "kind": "video"},
            ],
            "images": [
                {"url": "https://minio/extracted1.png"},
                {"url": "/api/concept-images/bad"},  # non-fetchable → skipped
                {"url": "https://minio/p1.png"},  # dup of pinned → deduped
            ],
        },
        media_discovery={
            "items": [
                {"composite_url": "https://minio/disc1.png"},
                {"composite_url": ""},  # failed → skipped
                # a discovered VIDEO → belongs to videos, NOT images
                {"composite_url": "https://minio/disc.mp4", "media_type": "video"},
            ],
        },
    )
    # Priority order: pinned → extracted → discovered, deduped; videos excluded.
    assert urls == [
        "https://minio/p1.png",
        "https://minio/extracted1.png",
        "https://minio/disc1.png",
    ]


def test_collect_real_media_urls_empty_when_stages_skipped() -> None:
    assert collect_real_media_urls(source_ingest=None, media_discovery=None) == []
    assert collect_real_media_urls(source_ingest={}, media_discovery={}) == []


def test_assign_real_media_by_line_once_each_in_order() -> None:
    image_by_line, fit_by_line = assign_real_media_by_line(
        _shots("l1", "l2", "l3", "l4", "l5"),
        ["https://m/a.png", "https://m/b.png", "https://m/c.png"],
    )
    # Filled the first three lines, once each, in priority order; rest untouched.
    assert image_by_line == {
        "l1": "https://m/a.png",
        "l2": "https://m/b.png",
        "l3": "https://m/c.png",
    }
    assert fit_by_line == {"l1": "cover", "l2": "cover", "l3": "cover"}


def test_assign_real_media_by_line_more_urls_than_shots_drops_extras() -> None:
    image_by_line, fit_by_line = assign_real_media_by_line(
        _shots("l1", "l2"),
        ["https://m/a.png", "https://m/b.png", "https://m/c.png"],
    )
    assert image_by_line == {"l1": "https://m/a.png", "l2": "https://m/b.png"}
    assert set(fit_by_line) == {"l1", "l2"}


def test_assign_real_media_by_line_honours_skip_line_ids() -> None:
    image_by_line, _ = assign_real_media_by_line(
        _shots("l1", "l2", "l3"),
        ["https://m/a.png", "https://m/b.png"],
        skip_line_ids={"l1"},
    )
    # l1 is protected; media flows to the next eligible lines instead.
    assert image_by_line == {"l2": "https://m/a.png", "l3": "https://m/b.png"}


def test_assign_real_media_by_line_empty_is_noop() -> None:
    assert assign_real_media_by_line(_shots("l1", "l2"), []) == ({}, {})


def test_assign_real_media_by_line_distribute_spreads_evenly() -> None:
    image_by_line, fit_by_line = assign_real_media_by_line(
        _shots("l0", "l1", "l2", "l3", "l4", "l5"),
        ["https://m/a.png", "https://m/b.png"],
        distribute=True,
    )
    # Spread across the timeline (bucket centres l2, l4), priority order preserved
    # left-to-right; the other four lines untouched.
    assert image_by_line == {"l2": "https://m/a.png", "l4": "https://m/b.png"}
    assert fit_by_line == {"l2": "cover", "l4": "cover"}


def test_assign_real_media_by_line_distribute_excludes_skip_then_spreads() -> None:
    # Eligible set excludes the skipped line BEFORE striding, so the spread is
    # computed over the remaining lines only.
    image_by_line, _ = assign_real_media_by_line(
        _shots("l0", "l1", "l2", "l3", "l4", "l5"),
        ["https://m/a.png"],
        skip_line_ids={"l0"},
        distribute=True,
    )
    # 5 eligible lines [l1..l5], one url → bucket centre index 2 → l3.
    assert image_by_line == {"l3": "https://m/a.png"}


def test_assign_real_media_by_line_distribute_dense_when_more_urls_than_lines() -> None:
    # More media than lines → fills every line in priority order (== front-load).
    image_by_line, _ = assign_real_media_by_line(
        _shots("l0", "l1"),
        ["https://m/a.png", "https://m/b.png", "https://m/c.png"],
        distribute=True,
    )
    assert image_by_line == {"l0": "https://m/a.png", "l1": "https://m/b.png"}


def test_assign_real_media_by_line_distribute_false_is_front_loaded() -> None:
    # Default (distribute=False) is unchanged: first K lines, in order.
    image_by_line, _ = assign_real_media_by_line(
        _shots("l0", "l1", "l2", "l3", "l4", "l5"),
        ["https://m/a.png", "https://m/b.png"],
    )
    assert image_by_line == {"l0": "https://m/a.png", "l1": "https://m/b.png"}


@pytest.mark.asyncio
async def test_fetch_image_bytes_happy_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "image/png"}, content=b"PNGDATA")

    data, fmt = await fetch_image_bytes("https://cdn/x", client=_client(handler))
    assert data == b"PNGDATA"
    assert fmt == "png"


_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


@pytest.mark.asyncio
async def test_wikimedia_403s_the_default_ua_and_200s_ours() -> None:
    """The live defect, reproduced: 11 of 22 "Find media" thumbnails were broken and ALL
    11 were ``upload.wikimedia.org`` — an always-allow-listed host. Wikimedia's UA policy
    403s httpx's default UA (proven in-pod: 403 text/plain 126 bytes vs 200 image/png
    33486 bytes for an identifying UA), and this module sent no ``User-Agent`` at all.

    The handler enforces the SAME rule the origin does, so the assertion can only pass
    if the header actually reaches the wire.
    """
    seen: list[str] = []

    def wikimedia(request: httpx.Request) -> httpx.Response:
        ua = request.headers.get("user-agent", "")
        seen.append(ua)
        if ua.startswith("python-httpx") or not ua:
            return httpx.Response(403, headers={"content-type": "text/plain"}, content=b"no")
        return httpx.Response(200, headers={"content-type": "image/png"}, content=_PNG_BYTES)

    data, fmt = await fetch_image_bytes(
        "https://upload.wikimedia.org/wikipedia/commons/5/50/Microsoft_Teams.png",
        client=_client(wikimedia),
    )
    assert data == _PNG_BYTES
    assert fmt == "png"
    # The header is SENT, not merely defined as a constant somewhere.
    assert seen == [MEDIA_FETCH_USER_AGENT]


@pytest.mark.asyncio
async def test_the_video_fetch_sends_the_same_user_agent() -> None:
    """The identical defect, one function away: ``fetch_video_bytes`` sent no UA at all,
    so a pinned Wikimedia ``.webm`` 403s in ``republish_video`` and the shot soft-fails —
    while its thumbnail, fixed by this same PR, renders fine. A half-fixed pair reads as
    a video-specific fault when it is one UA policy.

    The handler enforces the SAME rule the origin does (403 for httpx's default UA), so
    this can only pass if the header reaches the WIRE — not if a constant is referenced.
    """
    seen: list[str] = []

    def wikimedia(request: httpx.Request) -> httpx.Response:
        ua = request.headers.get("user-agent", "")
        seen.append(ua)
        if ua.startswith("python-httpx") or not ua:
            return httpx.Response(403, headers={"content-type": "text/plain"}, content=b"no")
        return httpx.Response(200, headers={"content-type": "video/webm"}, content=b"WEBMDATA")

    data, fmt = await fetch_video_bytes(
        "https://upload.wikimedia.org/wikipedia/commons/a/ab/Clip.webm",
        client=_client(wikimedia),
    )
    assert data == b"WEBMDATA"
    assert fmt == "webm"
    assert seen == [MEDIA_FETCH_USER_AGENT]


@pytest.mark.asyncio
async def test_the_video_user_agent_survives_a_cross_origin_redirect() -> None:
    """Wikimedia redirects media paths across origins and ``safe_get`` rebuilds the
    request on each hop. The video path must keep the UA across that hop too."""
    seen: list[tuple[str, str]] = []

    def hopping(request: httpx.Request) -> httpx.Response:
        seen.append((request.headers.get("host", ""), request.headers.get("user-agent", "")))
        if len(seen) == 1:
            return httpx.Response(302, headers={"location": "https://redirected.cdn/x.webm"})
        return httpx.Response(200, headers={"content-type": "video/webm"}, content=b"WEBMDATA")

    data, _fmt = await fetch_video_bytes("https://cdn/x.webm", client=_client(hopping))
    assert data == b"WEBMDATA"
    assert [host for host, _ in seen] == ["cdn", "redirected.cdn"]
    assert [ua for _, ua in seen] == [MEDIA_FETCH_USER_AGENT, MEDIA_FETCH_USER_AGENT]


def test_the_user_agent_identifies_us_and_does_not_impersonate_a_browser() -> None:
    """Wikimedia's policy asks for an identifying UA with a contact. A spoofed browser
    string would also get a 200 — and is dishonest and fragile, so it must not be what
    we send."""
    assert "w2v-orchestrator/" in MEDIA_FETCH_USER_AGENT
    assert "https://studio.nonagentic.ai" in MEDIA_FETCH_USER_AGENT
    lowered = MEDIA_FETCH_USER_AGENT.lower()
    assert not any(
        token in lowered for token in ("mozilla", "chrome", "safari", "applewebkit", "gecko")
    )


@pytest.mark.asyncio
async def test_the_user_agent_survives_a_cross_origin_redirect() -> None:
    """Wikimedia redirects thumbnail paths across origins, and ``safe_get`` re-builds the
    request on every hop, reducing headers to its cross-origin-safe set. ``user-agent`` is
    in that set (and absent from ``_CALLER_FORBIDDEN_HEADERS``), so NO ``w2v_common``
    change is needed — this test is the proof, and would go red if that ever changed."""
    seen: list[tuple[str, str]] = []

    def hopping(request: httpx.Request) -> httpx.Response:
        seen.append((request.headers.get("host", ""), request.headers.get("user-agent", "")))
        if len(seen) == 1:
            return httpx.Response(302, headers={"location": "https://redirected.cdn/x.png"})
        return httpx.Response(200, headers={"content-type": "image/png"}, content=_PNG_BYTES)

    data, _fmt = await fetch_image_bytes("https://cdn/x.png", client=_client(hopping))
    assert data == _PNG_BYTES
    assert [host for host, _ in seen] == ["cdn", "redirected.cdn"]
    assert [ua for _, ua in seen] == [MEDIA_FETCH_USER_AGENT, MEDIA_FETCH_USER_AGENT]


@pytest.mark.parametrize(
    ("data", "expected"),
    [(b"<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>", True)],
)
def test_looks_like_svg(data: bytes, expected: bool) -> None:
    assert looks_like_svg(data) is expected


@pytest.mark.parametrize(
    "data",
    [b"\x89PNG\r\n\x1a\n" + b"\x00" * 24],
)
def test_the_utf16_sniff_cannot_shadow_a_real_raster(data: bytes) -> None:
    """The UTF-16 de-interleave looks for a ``\\x00`` in one half of a pair — and raster
    magic is full of ``\\x00``. A raster must still sniff as itself and must NOT be
    mistaken for a vector, or the PR would start refusing images it used to proxy."""
    assert sniff_image_format(data) is not None
    assert looks_like_svg(data) is False


@pytest.mark.asyncio
async def test_fetch_image_bytes_rejects_non_image() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html>")

    with pytest.raises(ValueError, match="not an image"):
        await fetch_image_bytes("https://cdn/x", client=_client(handler))


@pytest.mark.asyncio
async def test_non_media_answers_raise_the_classification_error() -> None:
    """``text/html`` is a CLASSIFICATION ("this is a page"), distinguishable from a
    fetch failure — and still a ValueError for every existing soft-fail caller."""
    from src.services.visual_media import NotMediaContentError, fetch_video_bytes

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html>")

    with pytest.raises(NotMediaContentError) as img:
        await fetch_image_bytes("https://cdn/x", client=_client(handler))
    assert img.value.content_type == "text/html" and img.value.expected == "image"
    with pytest.raises(NotMediaContentError, match="not a video") as vid:
        await fetch_video_bytes("https://cdn/x", client=_client(handler))
    assert vid.value.expected == "video"
    assert isinstance(vid.value, ValueError)


@pytest.mark.parametrize(
    ("url", "host"),
    [("https://www.youtube.com/watch?v=abc", "youtube.com")],
)
def test_video_page_host(url: object, host: str | None) -> None:
    from src.services.visual_media import video_page_host

    assert video_page_host(url) == host


@pytest.mark.asyncio
async def test_fetch_image_bytes_rejects_oversize() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "image/jpeg"}, content=b"x" * 100)

    with pytest.raises(ValueError, match="exceeds"):
        await fetch_image_bytes("https://cdn/x", max_bytes=10, client=_client(handler))


@pytest.mark.asyncio
async def test_fetch_image_bytes_rejects_non_http_scheme() -> None:
    with pytest.raises(ValueError, match="non-http"):
        await fetch_image_bytes("ftp://cdn/x")


@pytest.mark.asyncio
async def test_fetch_image_bytes_rejects_private_hosts() -> None:
    for url in (
        "http://localhost/x.png",
        "http://127.0.0.1/x.png",
        "http://10.0.0.5/x.png",
        "http://192.168.1.10/x.png",
        "http://kag.internal/x.png",
        "http://minio/x.png",
    ):
        # Estate names (.internal/.svc/localhost) are refused pre-resolution by
        # NAME; literal internal IPs are refused by resolved ADDRESS. Both are
        # rejections — accept either wording.
        with pytest.raises(ValueError, match=r"Blocked hostname|blocked address"):
            await fetch_image_bytes(url)


@pytest.mark.asyncio
async def test_fetch_image_bytes_allow_hosts_opens_exactly_one_estate_host() -> None:
    """Our OWN artifact store is an estate-internal name the guard blocks by design.

    That is why the storyboard PDF export embedded a placeholder for every frame
    it ever had: ``minio.internal`` raised ``SSRFError`` on every fetch and the
    export still 200'd. ``allow_hosts`` is the opt-in that lets a caller which has
    already proven the URL is ours fetch it — and it opens ONLY the named host,
    so the same call still refuses every other internal name.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "image/png"}, content=b"PNGDATA")

    # Default (no allowlist) — unchanged, still blocked.
    with pytest.raises(ValueError, match="Blocked hostname"):
        await fetch_image_bytes("http://minio.internal/w2v/x.png", client=_client(handler))

    data, fmt = await fetch_image_bytes(
        "http://minio.internal/w2v/x.png",
        client=_client(handler),
        allow_hosts=("minio.internal", "minio.internal:8389"),
    )
    assert (data, fmt) == (b"PNGDATA", "png")

    # The hole is exactly one host wide: a different estate name is still refused
    # by the very same call shape.
    with pytest.raises(ValueError, match="Blocked hostname"):
        await fetch_image_bytes(
            "http://kag.internal/x.png",
            client=_client(handler),
            allow_hosts=("minio.internal", "minio.internal:8389"),
        )


_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _mislabelled(body: bytes = _PNG):
    """Our artifact store, serving real image bytes as ``application/json``."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, content=body)

    return handler


@pytest.mark.asyncio
async def test_confined_artifact_is_read_by_magic_bytes_not_declared_type() -> None:
    """A real PNG our store mislabels as ``application/json`` must still embed.

    ``store_stage_artifact`` declared ``application/json`` for EVERY object it
    ever wrote, and MinIO persists the declared type — fixing the write side does
    not rewrite the objects already there. So the storyboard PDF, which fetches
    the object directly rather than through the panel proxy, kept rejecting real
    frames as "not an image" even after the SSRF block was lifted. Confirmed live:
    the export stayed at its placeholder size with
    ``ValueError: not an image (application/json)`` in the pod log.
    """
    data, fmt = await fetch_image_bytes(
        "http://minio.internal/w2v/x.png",
        client=_client(_mislabelled()),
        allow_hosts=("minio.internal",),
    )
    assert fmt == "png"
    assert data.startswith(b"\x89PNG")


@pytest.mark.asyncio
async def test_mislabelled_bytes_are_trusted_only_for_a_confined_caller() -> None:
    """Sniffing is not a general relaxation of the content-type guard.

    Without ``allow_hosts`` the caller has proven nothing about the URL, so a
    mislabelled body is still refused — otherwise any operator-supplied URL could
    smuggle bytes past the check by lying about its type.
    """
    # `cdn` is the fixture's public host — this must fail on the TYPE, not on DNS.
    with pytest.raises(ValueError, match="not an image"):
        await fetch_image_bytes("http://cdn/x.png", client=_client(_mislabelled()))


@pytest.mark.asyncio
async def test_confined_caller_still_refuses_bytes_that_are_not_an_image() -> None:
    """The allowlist buys a magic-byte check, not a free pass."""
    with pytest.raises(ValueError, match="not an image"):
        await fetch_image_bytes(
            "http://minio.internal/w2v/x.png",
            client=_client(_mislabelled(b'{"not": "an image"}')),
            allow_hosts=("minio.internal",),
        )


@pytest.mark.asyncio
async def test_republish_image_stores_and_returns_url(monkeypatch) -> None:
    store = AsyncMock(return_value="https://minio/run/img.png")
    monkeypatch.setattr(visual_media, "store_capture_image", store)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "image/webp"}, content=b"WEBP")

    url = await republish_image(
        "run-1", "media_discovery", "https://cdn/x", client=_client(handler)
    )
    assert url == "https://minio/run/img.png"
    _, kwargs = store.call_args
    assert kwargs["image_bytes"] == b"WEBP"
    assert kwargs["image_format"] == "webp"
    assert kwargs["run_id"] == "run-1"


@pytest.mark.asyncio
async def test_republish_image_raises_on_non_image(monkeypatch) -> None:
    monkeypatch.setattr(visual_media, "store_capture_image", AsyncMock())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "video/mp4"}, content=b"MP4")

    with pytest.raises(ValueError, match="not an image"):
        await republish_image("run-1", "src", "https://cdn/clip", client=_client(handler))


# --- real-media VIDEO clips -------------------------------------------------


def test_collect_real_media_videos_priority_dedup_and_filter() -> None:
    videos = collect_real_media_videos(
        source_ingest={
            "pinned_media": [
                {"composite_url": "https://minio/v1.mp4", "kind": "video"},
                {"composite_url": "https://minio/p1.png"},  # image pin → not a video
                {"composite_url": "", "kind": "video"},  # failed → skipped
            ],
        },
        media_discovery={
            "items": [
                {"composite_url": "https://minio/v2.webm", "media_type": "video"},
                {"composite_url": "https://minio/disc1.png", "media_type": "image"},  # not a video
                {"composite_url": "https://minio/v1.mp4", "media_type": "video"},  # dup → deduped
            ],
        },
    )
    # Priority: pinned videos → discovered videos, deduped; images excluded.
    assert videos == ["https://minio/v1.mp4", "https://minio/v2.webm"]


def test_collect_real_media_videos_empty_when_stages_skipped() -> None:
    assert collect_real_media_videos(source_ingest=None, media_discovery=None) == []
    assert collect_real_media_videos(source_ingest={}, media_discovery={}) == []


def test_collect_ai_video_urls_keeps_fetchable_dedups_and_drops_relative() -> None:
    urls = collect_ai_video_urls(
        {
            "videos": [
                {"video_url": "https://minio/ai1.mp4"},
                {"video_url": "/api/content/videos/jobs/j/download"},  # republish failed → dropped
                {"video_url": "https://minio/ai1.mp4"},  # dup → deduped
                {"video_url": None},  # nothing → skipped
                {"video_url": "/api/runs/r/artifact/ai2.mp4"},  # artifact-proxy path → kept
                "not-a-dict",  # defensive
            ]
        }
    )
    assert urls == ["https://minio/ai1.mp4", "/api/runs/r/artifact/ai2.mp4"]


def test_collect_ai_video_urls_empty_when_skipped() -> None:
    assert collect_ai_video_urls(None) == []
    assert collect_ai_video_urls({}) == []
    assert collect_ai_video_urls({"videos": []}) == []


def test_assign_real_media_videos_by_line_once_each_in_order() -> None:
    video_by_line, fit_by_line = assign_real_media_videos_by_line(
        _shots("l1", "l2", "l3"),
        ["https://m/a.mp4", "https://m/b.mp4"],
    )
    assert video_by_line == {"l1": "https://m/a.mp4", "l2": "https://m/b.mp4"}
    assert fit_by_line == {"l1": "cover", "l2": "cover"}


def test_assign_real_media_videos_by_line_honours_skip_line_ids() -> None:
    # Lines an image already claimed are skipped so a clip never collides with it.
    video_by_line, _ = assign_real_media_videos_by_line(
        _shots("l1", "l2", "l3"),
        ["https://m/a.mp4", "https://m/b.mp4"],
        skip_line_ids={"l1"},
    )
    assert video_by_line == {"l2": "https://m/a.mp4", "l3": "https://m/b.mp4"}


def test_assign_real_media_videos_by_line_empty_is_noop() -> None:
    assert assign_real_media_videos_by_line(_shots("l1", "l2"), []) == ({}, {})


def test_assign_real_media_videos_by_line_distribute_spreads_evenly() -> None:
    video_by_line, fit_by_line = assign_real_media_videos_by_line(
        _shots("l0", "l1", "l2", "l3", "l4", "l5"),
        ["https://m/a.mp4", "https://m/b.mp4"],
        distribute=True,
    )
    # Clips spread across the timeline (bucket centres l2, l4), not front-loaded.
    assert video_by_line == {"l2": "https://m/a.mp4", "l4": "https://m/b.mp4"}
    assert fit_by_line == {"l2": "cover", "l4": "cover"}


def test_assign_real_media_videos_by_line_distribute_false_is_front_loaded() -> None:
    video_by_line, _ = assign_real_media_videos_by_line(
        _shots("l0", "l1", "l2", "l3", "l4", "l5"),
        ["https://m/a.mp4", "https://m/b.mp4"],
    )
    assert video_by_line == {"l0": "https://m/a.mp4", "l1": "https://m/b.mp4"}


@pytest.mark.asyncio
async def test_fetch_video_bytes_happy_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "video/mp4"}, content=b"MP4DATA")

    data, fmt = await fetch_video_bytes("https://cdn/clip", client=_client(handler))
    assert data == b"MP4DATA"
    assert fmt == "mp4"


@pytest.mark.asyncio
async def test_fetch_video_bytes_maps_webm_and_mov() -> None:
    def webm(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "video/webm"}, content=b"W")

    def mov(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "video/quicktime"}, content=b"M")

    assert (await fetch_video_bytes("https://cdn/a", client=_client(webm)))[1] == "webm"
    assert (await fetch_video_bytes("https://cdn/b", client=_client(mov)))[1] == "mov"


@pytest.mark.asyncio
async def test_fetch_video_bytes_rejects_non_video() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "image/png"}, content=b"PNG")

    with pytest.raises(ValueError, match="not a video"):
        await fetch_video_bytes("https://cdn/x", client=_client(handler))


@pytest.mark.asyncio
async def test_fetch_video_bytes_aborts_oversize() -> None:
    # The streamed download must abort the instant it crosses the cap (OOM guard).
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "video/mp4"}, content=b"x" * 100)

    with pytest.raises(ValueError, match="exceeds"):
        await fetch_video_bytes("https://cdn/x", max_bytes=10, client=_client(handler))


@pytest.mark.asyncio
async def test_fetch_video_bytes_rejects_private_hosts() -> None:
    # `.internal` is refused by NAME, before any DNS lookup (w2v_common >=0.52.0):
    # split-horizon DNS means the same name can resolve publicly from outside and
    # to a tailnet address from inside a pod, so the name itself is the thing to
    # refuse. A resolved-address rejection would also be acceptable here.
    with pytest.raises(ValueError, match=r"Blocked hostname|blocked address"):
        await fetch_video_bytes("http://kag.internal/clip.mp4")


@pytest.mark.asyncio
async def test_republish_video_stores_and_returns_url(monkeypatch) -> None:
    store = AsyncMock(return_value="https://minio/run/clip.mp4")
    monkeypatch.setattr(visual_media, "store_capture_clip", store)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "video/mp4"}, content=b"MP4")

    url = await republish_video(
        "run-1", "media_discovery", "https://cdn/c", client=_client(handler)
    )
    assert url == "https://minio/run/clip.mp4"
    _, kwargs = store.call_args
    assert kwargs["video_bytes"] == b"MP4"
    assert kwargs["video_format"] == "mp4"
    assert kwargs["run_id"] == "run-1"
