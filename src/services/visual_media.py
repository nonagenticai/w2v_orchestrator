"""visual_media — turn recorded *real-media* candidates into compiler-safe shots.

The promo pipeline records three sources of real (non-AI-generated) imagery that,
until now, never reached the rendered video:

* operator-**pinned** media (``source_ingest.pinned_media`` — items the user picked
  in the wizard's "Find media" panel),
* document-/page-**extracted** images (``source_ingest.images``), and
* auto-**discovered** media (``media_discovery.items`` — media-discovery search results).

The compiler is anonymous and downloads each shot's URL itself, so an *external*
URL it cannot fetch (hotlink-blocked, egress-blocked, expired) would crash the
whole compile — exactly the failure ``render._fetchable`` warns about. The fix is
the same re-publish pattern the rest of the pipeline uses: fetch the external
image once, store it in this run's MinIO bucket, and hand the compiler the
in-cluster URL instead. ``extracted`` images are already re-published by
``source_ingest``; ``pinned`` + ``discovered`` are re-published here (in their
producing stages) via :func:`republish_image`, which records the result on a
``composite_url`` field.

:func:`collect_real_media_urls` then gathers the compiler-safe image URLs in
priority order (pinned → extracted → discovered) for a render to lay over the
image-suitable lines. Pinned/discovered **video** clips are re-published the same
way via :func:`republish_video` (a streamed, size-capped download so the pod
never buffers an oversized file) and gathered by
:func:`collect_real_media_videos`; :func:`assign_real_media_videos_by_line` keys
them per line as ``video_url`` (which the compositor honours over ``image_url``).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse
from uuid import uuid7

import httpx
from w2v_common.net import safe_get

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

from src.services.still_fit import fit_ingested_still
from src.services.tts_storage import store_capture_clip, store_capture_image

log = logging.getLogger(__name__)

# A single external image: cap the download and the wait so one slow/huge asset
# can't stall the producing stage (which has its own DAG timeout on top).
_MAX_IMAGE_BYTES = 12 * 1024 * 1024  # 12 MiB
#: Raw-body bound for an image fetched by the INGEST path (``republish_image``
#: with ``fit_to``) — and only there. A 48 MP phone original is 15-30 MB, so the
#: general 12 MiB cap dropped exactly the photos users upload. The body is still
#: streamed and size-capped, and it is fitted to the film frame
#: (``still_fit.fit_ingested_still``) BEFORE it is stored, so what reaches run
#: storage — and every downstream re-fetch — stays under ``_MAX_IMAGE_BYTES``.
_MAX_INGEST_IMAGE_BYTES = 40 * 1024 * 1024
_FETCH_TIMEOUT_S = 20.0

# Real-media video clips (pinned / discovered). The download is STREAMED and
# aborted the instant it exceeds the cap, so the orchestrator never buffers an
# oversized clip — critical given the pod's tight memory budget (the OOM history
# in html_render etc.). 64 MiB comfortably covers stock clips (pexels/unsplash
# are typically a few MB) while staying far below an OOM. Re-publishes run
# sequentially in their producing stages, so peak buffering is ~one clip.
_MAX_VIDEO_BYTES = 64 * 1024 * 1024  # 64 MiB
_VIDEO_FETCH_TIMEOUT_S = 40.0

#: The ``User-Agent`` every outbound media fetch identifies itself with.
#:
#: Not cosmetic — it is the difference between a thumbnail and a 502. Wikimedia's
#: User-Agent policy **403s httpx's default UA outright**, and 11 of the 22 results
#: in one live "Find media" panel were ``upload.wikimedia.org`` — an
#: ALWAYS-allow-listed host — so every one of them came back as a broken image.
#: Proven from inside the live orchestrator pod against the same URL and egress:
#: httpx default → ``403 Forbidden`` (text/plain, 126 bytes); this UA → ``200``
#: (image/png, 33486 bytes).
#:
#: Deliberately NOT a spoofed browser UA. Wikimedia asks for an identifying string
#: with a way to reach the operator; impersonating Chrome is both dishonest and
#: fragile (it breaks the moment the policy starts matching on it). The contact is
#: the orchestrator's own public surface, which already appears throughout this
#: repo — nothing invented.
#:
#: ``w2v_common.net.safe_get`` carries this through: ``user-agent`` is in its
#: ``_CROSS_ORIGIN_SAFE_HEADERS`` (so it survives a cross-origin redirect hop) and
#: is absent from ``_CALLER_FORBIDDEN_HEADERS`` (so the caller may set it). No
#: change to ``w2v_common`` is needed.
MEDIA_FETCH_USER_AGENT = "w2v-orchestrator/1.0 (+https://studio.nonagentic.ai)"

# SSRF: media URLs legitimately originate from media-discovery (public CDNs:
# unsplash/pexels/...) or an operator's "Find media" pin — never from inside the
# cluster. The guard is ``w2v_common.net.safe_get``: it resolves the host once,
# rejects private/loopback/link-local/**CGNAT** (this estate's
# Tailnet: postgres/registry/k3s all live there — ``ip.is_private`` is False for
# those), pins the connection to the validated IP (no DNS-rebinding TOCTOU), and
# follows redirects MANUALLY so every hop is re-validated. The previous local
# guard did neither: it never resolved DNS (so any hostname passed) and the
# client ran ``follow_redirects=True`` (so a 302 escaped it entirely).


class NotMediaContentError(ValueError):
    """The URL answered, but with something that is not the expected media type.

    A CLASSIFICATION, not a fetch failure: a YouTube watch URL answers
    ``text/html`` because it is a web PAGE that embeds a player, not a clip anyone
    can download. Callers that soft-fail on every ``ValueError`` keep working (it
    subclasses it); callers that care can tell "this is not media" apart from
    "this media could not be fetched" and record the first without a
    failure-looking degradation (prod run 01a09f82's ``media_discovery``).
    """

    def __init__(self, url: str, content_type: str, *, expected: str) -> None:
        self.url = url
        self.content_type = content_type or "unknown"
        self.expected = expected
        super().__init__(
            f"not a{'n' if expected == 'image' else ''} {expected} ({self.content_type}): {url!r}"
        )


#: Hosts whose URLs are video WATCH PAGES (HTML with an embedded player), never a
#: downloadable media file. Classified up front so the pipeline neither downloads
#: an HTML body into the video size cap nor reports the page as a failed clip.
_VIDEO_PAGE_HOSTS: tuple[str, ...] = (
    "youtube.com",
    "youtu.be",
    "youtube-nocookie.com",
    "vimeo.com",
    "dailymotion.com",
    "dai.ly",
    "tiktok.com",
    "twitch.tv",
)


def video_page_host(url: object) -> str | None:
    """The watch-page host ``url`` belongs to, or ``None`` when it is not one.

    ``player.vimeo.com`` / ``www.youtube.com`` / ``m.youtube.com`` all match their
    registrable domain. CDN hosts that serve real files (``i.ytimg.com``,
    ``vimeocdn.com``) are different domains and do not match.
    """
    if not isinstance(url, str):
        return None
    host = (urlparse(url).hostname or "").lower()
    for page_host in _VIDEO_PAGE_HOSTS:
        if host == page_host or host.endswith("." + page_host):
            return page_host
    return None


def is_fetchable_url(url: object) -> bool:
    """True for URLs the video-compile service can actually fetch.

    Two valid shapes, because the orchestrator artifact store yields whichever
    the backend is configured for: absolute ``http(s)://`` (e.g. a MinIO
    presigned URL) and the orchestrator artifact-proxy path
    ``/api/runs/<id>/artifact/...``. Anything else (e.g. an ACG-relative
    ``/api/concept-images/...`` URL that failed to re-publish, or a ``file://``
    local-backend fallback) is rejected so the caller degrades to a slide
    fallback rather than 500-ing the whole compile.
    """
    return isinstance(url, str) and (
        url.startswith(("http://", "https://")) or url.startswith("/api/runs/")
    )


async def fetch_image_bytes(
    url: str,
    *,
    timeout: float = _FETCH_TIMEOUT_S,
    max_bytes: int = _MAX_IMAGE_BYTES,
    client: httpx.AsyncClient | None = None,
    allow_hosts: Iterable[str] | None = None,
) -> tuple[bytes, str]:
    """GET an external image, enforcing scheme/host/content-type/size guards.

    Returns ``(image_bytes, image_format)`` where ``image_format`` is a short
    extension (``png``/``jpg``/``webp``/``gif``). Raises ``ValueError`` for a
    rejected URL (``SSRFError`` subclasses it), a non-image content-type, or an
    over-size body (``ResponseTooLargeError``, also a ``ValueError``) — callers
    soft-fail on any exception.

    ``allow_hosts`` (bare hostname or ``host:port``, forwarded verbatim to
    ``safe_get``) names hosts whose estate-internal address is fetched anyway. It
    is **opt-in per call site and defaults to none** on purpose: the callers that
    fetch an operator- or tenant-supplied URL (real-media pins, media discoveries,
    reference images) must keep the full guard, and a module-level allowlist would
    hand every one of them a hole. Only a caller that has independently proven the
    URL addresses OUR OWN artifact store — e.g. the storyboard PDF export, which
    runs each frame through ``api.panel_image_url.panel_object_ref`` first — may
    pass one.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"refusing non-http(s) media URL: {url!r}")

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=timeout)
    try:
        resp = await safe_get(
            client,
            url,
            max_fetch_bytes=max_bytes,
            allow_hosts=allow_hosts,
            headers={"User-Agent": MEDIA_FETCH_USER_AGENT},
        )
        resp.raise_for_status()
        content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        data = resp.content
        sniffed = _ext_for_magic(data)
        if not content_type.startswith("image/"):
            # A declared type we do not trust is only acceptable when the caller
            # has proven the object is OURS and the bytes are self-evidently an
            # image. Our own artifact store served every object it ever wrote as
            # `application/json` until the write side was fixed, and those stored
            # types are never rewritten — so a real PNG the storyboard PDF must
            # embed still arrives mislabelled. Sniffing the magic bytes is what
            # makes those historical frames usable; an operator-supplied URL has
            # no `allow_hosts` and so still fails here on the declared type.
            if allow_hosts is None or sniffed is None:
                raise NotMediaContentError(url, content_type, expected="image")
            log.info(
                "trusting magic bytes over declared %r for confined artifact: %s",
                content_type or "unknown",
                url.split("?")[0],
            )
            return data, sniffed
    finally:
        if owns_client:
            await client.aclose()
    return data, _ext_for_content_type(content_type)


#: Magic-byte prefixes for the formats ``fetch_image_bytes`` returns. JPEG is
#: matched on SOI alone (``FF D8 FF``) because the fourth byte varies by marker.
_MAGIC_PREFIXES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"\xff\xd8\xff", "jpg"),
)


def _ext_for_magic(data: bytes) -> str | None:
    """The image format the BYTES actually are, or ``None`` if not an image.

    Deliberately independent of any declared content type — this exists precisely
    for objects whose declared type is known to be wrong.
    """
    for prefix, ext in _MAGIC_PREFIXES:
        if data.startswith(prefix):
            return ext
    # WEBP is RIFF-framed: "RIFF" <4-byte size> "WEBP".
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    return None


def sniff_image_format(data: bytes) -> str | None:
    """Public view of :func:`_ext_for_magic`: ``png``/``jpg``/``webp``/``gif`` or ``None``."""
    return _ext_for_magic(data)


def _single_byte_head(data: bytes) -> bytes:
    """*data*'s leading bytes as single-byte text, de-interleaving UTF-16.

    A UTF-16 document stores each ASCII character as a two-byte unit whose other
    half is ``\\x00``, so a byte-wise ``b"<svg" in head`` can never match one —
    the bytes read ``<\\x00s\\x00v\\x00g\\x00``. Both a BOM and the bare
    null-interleaving (a BOM is not required) are recognised, and anything else is
    returned unchanged so the UTF-8 path is untouched.

    Detection is deliberately narrow: it needs a ``\\x00`` in the position the
    encoding puts one AND a non-``\\x00`` in the other, which no PNG/JPEG/GIF/WEBP
    magic satisfies. It could not shadow a raster in any case — every caller
    consults :func:`looks_like_svg` only AFTER :func:`sniff_image_format` has
    already returned ``None``.
    """
    head = data[:4096]
    if head[:2] == b"\xff\xfe":  # UTF-16LE BOM: ASCII is the low byte of each unit
        return head[2::2]
    if head[:2] == b"\xfe\xff":  # UTF-16BE BOM: ASCII is the high byte of each unit
        return head[3::2]
    if len(head) >= 4:
        if head[0] and not head[1] and not head[3]:  # BOM-less UTF-16LE
            return head[0::2]
        if not head[0] and head[1] and not head[2]:  # BOM-less UTF-16BE
            return head[1::2]
    return head


def looks_like_svg(data: bytes) -> bool:
    """True when ``data`` is an SVG document.

    SVG is markup that carries ``<script>`` and remote references, so it is
    deliberately NOT in :data:`_MAGIC_PREFIXES` and must never be proxied as an
    image. This exists purely to tell that DELIBERATE REFUSAL apart from "the
    upstream fetch broke" — a real Wikimedia result
    (``Microsoft_Office_Teams_(2025-present).svg``) is a perfectly healthy asset we
    simply cannot preview, and answering 502 told the operator the server was
    broken. It does not, and must not, make SVG fetchable.

    Matched structurally, not by content-type or file extension (both are
    attacker/CDN-controlled): normalise a UTF-16 body to single-byte text
    (:func:`_single_byte_head`), skip a UTF-8 BOM and leading whitespace, require
    the body to OPEN with ``<``, then look for an ``<svg`` tag within the XML prolog
    window. A PNG/JPEG/GIF/WEBP never starts with ``<``, so this cannot shadow
    :func:`sniff_image_format`. Gzip-compressed ``.svgz`` is not matched (it is
    binary) and falls through to the generic non-raster refusal, which is correct —
    we still do not serve it.

    **Widening this can only ever refuse MORE, never proxy.** A ``True`` here is
    read at exactly one place — ``routes/media.thumb``, inside the branch where
    ``sniff_image_format(data) is None`` — and it selects a typed ``400``
    ``vector_image_unsupported`` instead of a bare ``502``. Both are refusals; the
    bytes are returned to the caller down neither. Recognising UTF-16 SVG (a real
    encoding for a healthy asset that previously fell through to the 502) therefore
    moves a source from "server error" to "we can't preview a vector image" and
    does not make SVG fetchable.
    """
    head = _single_byte_head(data).lstrip(b"\xef\xbb\xbf").lstrip()[:2048]
    if not head.startswith(b"<"):
        return False
    return b"<svg" in head.lower()


def _ext_for_content_type(content_type: str) -> str:
    if "png" in content_type:
        return "png"
    if "webp" in content_type:
        return "webp"
    if "gif" in content_type:
        return "gif"
    return "jpg"


async def republish_image(
    run_id: str,
    stage_name: str,
    url: str,
    *,
    image_id: str | None = None,
    client: httpx.AsyncClient | None = None,
    fit_to: tuple[int, int] | None = None,
) -> str:
    """Fetch an external image and re-publish it to run storage.

    Returns the compiler-fetchable MinIO URL. Raises on any fetch/store failure
    so the caller can soft-fail (record nothing / a note) without compositing an
    asset the compiler can't reach.

    ``fit_to`` (the film's output ``(width, height)``) normalises the still at
    ingest — EXIF rotation baked in, cover-fitted to ``still_fit.STILL_HEADROOM``
    x the frame, photo re-encoded as JPEG — so no stage downstream ever handles a
    full-size camera original (see ``src/services/still_fit.py``). With it the raw
    body may be up to ``_MAX_INGEST_IMAGE_BYTES`` (streamed, capped), because it
    is fitted under ``_MAX_IMAGE_BYTES`` before it is stored; a bomb, an
    undecodable oversize body or an unfittable still raises
    ``StillRejectedError``. Omitted, the general cap applies and the bytes are
    stored exactly as fetched.
    """
    if fit_to is None:
        data, fmt = await fetch_image_bytes(url, client=client)
    else:
        data, fmt = await fetch_image_bytes(url, client=client, max_bytes=_MAX_INGEST_IMAGE_BYTES)
        data, fmt = await asyncio.to_thread(
            fit_ingested_still, data, fmt, *fit_to, max_out_bytes=_MAX_IMAGE_BYTES
        )
    return await store_capture_image(
        run_id=run_id,
        stage_name=stage_name,
        image_id=image_id or str(uuid7()),
        image_bytes=data,
        image_format=fmt,
    )


def _ext_for_video_content_type(content_type: str) -> str:
    if "webm" in content_type:
        return "webm"
    if "quicktime" in content_type or "mov" in content_type:
        return "mov"
    if "ogg" in content_type:
        return "ogv"
    return "mp4"


async def fetch_video_bytes(
    url: str,
    *,
    timeout: float = _VIDEO_FETCH_TIMEOUT_S,
    max_bytes: int = _MAX_VIDEO_BYTES,
    client: httpx.AsyncClient | None = None,
) -> tuple[bytes, str]:
    """GET an external video, enforcing scheme/host/content-type/size guards.

    Returns ``(video_bytes, video_format)`` (``mp4``/``webm``/``mov``/``ogv``).
    The body is STREAMED by ``safe_get`` and the download aborts the moment it
    crosses ``max_bytes`` — the orchestrator never holds an oversized clip in
    memory. Raises ``ValueError`` for a rejected URL (``SSRFError``), a non-video
    content-type, or an over-size body (``ResponseTooLargeError``) — callers
    soft-fail on any exception.

    Sends :data:`MEDIA_FETCH_USER_AGENT` for the same reason
    :func:`fetch_image_bytes` does, and it is the SAME defect: the video path
    fetches the SAME third-party hosts from the SAME pinned "Find media" results.
    A pinned Wikimedia ``.webm`` 403s httpx's default UA exactly as a ``.png``
    does, so ``republish_video`` soft-fails the shot while its thumbnail — fixed
    one function away — renders fine. Half-fixed is worse than broken: it reads
    like a video-specific fault when it is the same UA policy.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"refusing non-http(s) media URL: {url!r}")

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=timeout)
    try:
        resp = await safe_get(
            client,
            url,
            max_fetch_bytes=max_bytes,
            headers={"User-Agent": MEDIA_FETCH_USER_AGENT},
        )
        resp.raise_for_status()
        content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        if not content_type.startswith("video/"):
            raise NotMediaContentError(url, content_type, expected="video")
        data = resp.content
    finally:
        if owns_client:
            await client.aclose()
    return data, _ext_for_video_content_type(content_type)


async def republish_video(
    run_id: str,
    stage_name: str,
    url: str,
    *,
    clip_id: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Fetch an external video and re-publish it to run storage.

    Returns the compiler-fetchable MinIO URL. Raises on any fetch/store failure
    so the caller can soft-fail (record nothing / a note) without compositing a
    clip the anonymous compiler can't reach.
    """
    data, fmt = await fetch_video_bytes(url, client=client)
    return await store_capture_clip(
        run_id=run_id,
        stage_name=stage_name,
        clip_id=clip_id or str(uuid7()),
        video_bytes=data,
        video_format=fmt,
    )


def collect_real_media_urls(
    *,
    source_ingest: dict[str, Any] | None,
    media_discovery: dict[str, Any] | None,
) -> list[str]:
    """Gather compiler-fetchable real-media URLs in priority order.

    Priority: operator-**pinned** (explicit curation) → document-**extracted** →
    auto-**discovered** (already quality-ranked). Deduped by URL; only URLs
    ``is_fetchable_url`` accepts are kept (pinned/discovered carry their
    re-published ``composite_url``; extracted carry their already-republished
    ``url``). Returns ``[]`` when both stages skipped/produced nothing, so the
    caller's behaviour is unchanged for runs without sources.
    """
    si = source_ingest or {}
    md = media_discovery or {}
    out: list[str] = []
    seen: set[str] = set()

    def _add(url: object) -> None:
        if isinstance(url, str) and is_fetchable_url(url) and url not in seen:
            seen.add(url)
            out.append(url)

    # Video pins/discoveries also carry a composite_url now (re-published as a
    # clip); they belong to collect_real_media_videos, not here — skip them so an
    # mp4 never lands on an image line.
    for pin in si.get("pinned_media") or []:
        if isinstance(pin, dict) and pin.get("kind") != "video":
            _add(pin.get("composite_url"))
    for img in si.get("images") or []:
        if isinstance(img, dict):
            _add(img.get("url"))
    for item in md.get("items") or []:
        if isinstance(item, dict) and item.get("media_type") != "video":
            _add(item.get("composite_url"))
    return out


def collect_real_media_videos(
    *,
    source_ingest: dict[str, Any] | None,
    media_discovery: dict[str, Any] | None,
) -> list[str]:
    """Gather compiler-fetchable real-media VIDEO URLs in priority order.

    Priority: operator-**pinned** video clips → auto-**discovered** clips. Both
    carry their re-published, in-cluster ``composite_url`` (set by
    ``source_ingest`` / ``media_discovery`` via :func:`republish_video`).
    Documents never yield video. Deduped by URL; only ``is_fetchable_url`` URLs
    are kept. Returns ``[]`` when no sources produced a clip, so the caller's
    behaviour is unchanged for image-only / no-source runs.
    """
    si = source_ingest or {}
    md = media_discovery or {}
    out: list[str] = []
    seen: set[str] = set()

    def _add(url: object) -> None:
        if isinstance(url, str) and is_fetchable_url(url) and url not in seen:
            seen.add(url)
            out.append(url)

    for pin in si.get("pinned_media") or []:
        if isinstance(pin, dict) and pin.get("kind") == "video":
            _add(pin.get("composite_url"))
    for item in md.get("items") or []:
        if isinstance(item, dict) and item.get("media_type") == "video":
            _add(item.get("composite_url"))
    return out


def collect_ai_video_urls(ai_video_gen: dict[str, Any] | None) -> list[str]:
    """Gather compiler-fetchable AI-generated VIDEO clip URLs, deduped.

    ``ai_video_gen`` re-publishes each clip's ACG-relative URL to run storage
    (see ``ai_video_gen._republish_relative``); this collects those
    ``video_url`` values, keeping only the ones ``is_fetchable_url`` accepts —
    a republish that failed (leaving an unfetchable ACG-relative URL) is
    dropped so the render degrades to its fallback rather than 500-ing the
    compile. Returns ``[]`` when the stage skipped / produced no usable clip,
    so the caller's behaviour is unchanged for runs without generative video.
    """
    out: list[str] = []
    seen: set[str] = set()
    for vid in (ai_video_gen or {}).get("videos") or []:
        if not isinstance(vid, dict):
            continue
        url = vid.get("video_url")
        if isinstance(url, str) and is_fetchable_url(url) and url not in seen:
            seen.add(url)
            out.append(url)
    return out


def collect_motion_clips_by_line(
    motion_graphics_gen: dict[str, Any] | None,
) -> tuple[dict[str, str], list[str]]:
    """Split motion-graphics clips into a per-line keyed map and an unkeyed pool.

    ``motion_graphics_gen`` re-publishes each clip to run storage (see
    ``store_generated_video``); this collects the ``video_url`` values that
    ``is_fetchable_url`` accepts. A clip carrying a ``line_id`` joins 1:1 (keyed
    map); a clip without one (a single base render, or extra clips) goes into the
    round-robin pool. Returns ``({}, [])`` when the stage skipped, so callers are
    unchanged for runs without motion graphics.
    """
    keyed: dict[str, str] = {}
    pool: list[str] = []
    seen: set[str] = set()
    for clip in (motion_graphics_gen or {}).get("clips") or []:
        if not isinstance(clip, dict):
            continue
        url = clip.get("video_url")
        if not (isinstance(url, str) and is_fetchable_url(url)) or url in seen:
            continue
        seen.add(url)
        line_id = clip.get("line_id")
        if line_id:
            keyed[str(line_id)] = url
        else:
            pool.append(url)
    return keyed, pool


def collect_motion_slides_by_line(slides: dict[str, Any] | None) -> dict[str, str]:
    """Per-line animated-slide clip URLs from ``promo_slides`` output.

    When motion graphics are enabled, ``promo_slides`` renders each line's slide
    as a branded motion clip alongside the still PNG (``motion_video_url``). This
    collects the fetchable ones keyed by ``line_id`` so the render can supersede a
    line's still slide with its animated version. The still PNG (``image_url``)
    stays the degrade path — an animated slide only ever replaces its OWN still.
    Returns ``{}`` when no slide carries a motion clip (motion disabled / failed).
    """
    keyed: dict[str, str] = {}
    for slide in (slides or {}).get("slides") or []:
        if not isinstance(slide, dict):
            continue
        url = slide.get("motion_video_url")
        line_id = slide.get("line_id")
        if line_id and isinstance(url, str) and is_fetchable_url(url):
            keyed[str(line_id)] = url
    return keyed


def overlay_motion_slides(
    image_by_line: dict[str, str],
    video_by_line: dict[str, str],
    fit_by_line: dict[str, str],
    *,
    slide_by_line: dict[str, str],
    motion_slide_by_line: dict[str, str],
) -> None:
    """Replace a line's still slide with its animated version, in place.

    An animated slide supersedes ONLY its own still: a line is upgraded when it
    is still showing its fallback slide (``image_by_line[line] == slide_by_line[line]``)
    and no higher-precedence video already claimed it. The still PNG stays the
    degrade path, and a line that took a dedicated image (html / ai-image /
    curated) or any clip (footage / curated / motion-graphics / AI) is untouched.
    ``"contain"`` keeps the slide text uncropped. Empty input → no change.

    Shared by promo_render and website_review_compile.
    """
    for line_id, motion_url in motion_slide_by_line.items():
        if line_id in video_by_line:
            continue
        if image_by_line.get(line_id) == slide_by_line.get(line_id):
            video_by_line[line_id] = motion_url
            fit_by_line[line_id] = "contain"


def _distribute_indices(n: int, m: int) -> list[int]:
    """Evenly-spaced slot indices for ``m`` media items across ``n`` lines.

    Places each item at the centre of its equal ``n/m`` bucket —
    ``round((k+0.5)*n/m)`` — so the ``m`` items land spread through the timeline
    rather than on the leading ``m``. (Bucket-centre spacing beats the
    ``(k+1)*n/(m+1)`` interior stride used by :func:`render._assign_footage`,
    which collapses to adjacent slots when ``m`` approaches ``n/2`` — e.g. m=3
    over n=6 → ``[1, 3, 5]`` here vs ``[2, 3, 4]`` there.) Collision-advance
    (wraps modulo ``n``) guards the rare rounding tie; ``m<n`` guarantees a free
    slot. For ``m>=n`` every slot fills in priority order (dense ``range(n)``), so
    striding degrades gracefully to the front-load result when there is more media
    than lines. Returned indices follow priority order (item ``k`` → ``out[k]``),
    so the caller assigns the highest-priority URL to the earliest strided slot.
    """
    if n <= 0 or m <= 0:
        return []
    if m >= n:
        return list(range(n))  # dense: first n media over all n lines, in order
    used: set[int] = set()
    out: list[int] = []
    for k in range(m):
        idx = min(n - 1, max(0, round((k + 0.5) * n / m)))
        while idx in used:
            idx = (idx + 1) % n
        used.add(idx)
        out.append(idx)
    return out


def assign_real_media_by_line(
    shots: list[dict[str, Any]],
    real_media_urls: list[str],
    *,
    skip_line_ids: Iterable[str] = (),
    distribute: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    """Lay real-media URLs over shot lines, once each, in priority order.

    Returns ``(image_by_line, fit_by_line)`` — a keyed image URL per ``line_id``
    (which beats the round-robin AI pool in :func:`assemble_shots`) and a matching
    ``fit="cover"`` so the real photo fills the frame. URLs are consumed in order
    (``real_media_urls`` is already priority-ranked by
    :func:`collect_real_media_urls`), one per non-skipped shot line, until either
    the URLs or the shots run out — extras are dropped.

    By default media front-loads onto the first non-skipped lines (real media that
    *leads*). With ``distribute=True`` the same URLs are instead **spread evenly
    across** the non-skipped lines (priority order preserved left-to-right) — used
    for full-frame cutaways that should recur through the timeline rather than
    cluster up front (see :func:`_distribute_indices`).

    ``skip_line_ids`` names lines that already carry a dedicated visual and must
    not be overridden (the generic caller passes none — every generic line only
    holds a round-robin AI image, which real media is meant to displace). Empty
    ``real_media_urls`` → ``({}, {})`` so the caller is byte-for-byte unchanged.
    """
    skip = set(skip_line_ids)
    image_by_line: dict[str, str] = {}
    fit_by_line: dict[str, str] = {}
    if distribute:
        eligible = [str(s["line_id"]) for s in shots if str(s["line_id"]) not in skip]
        for k, idx in enumerate(_distribute_indices(len(eligible), len(real_media_urls))):
            image_by_line[eligible[idx]] = real_media_urls[k]
            fit_by_line[eligible[idx]] = "cover"
        return image_by_line, fit_by_line
    media = iter(real_media_urls)
    for shot in shots:
        line_id = str(shot["line_id"])
        if line_id in skip:
            continue
        url = next(media, None)
        if url is None:
            break
        image_by_line[line_id] = url
        fit_by_line[line_id] = "cover"
    return image_by_line, fit_by_line


def assign_real_media_videos_by_line(
    shots: list[dict[str, Any]],
    real_media_video_urls: list[str],
    *,
    skip_line_ids: Iterable[str] = (),
    distribute: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    """Lay real-media VIDEO URLs over shot lines, once each, in priority order.

    Returns ``(video_by_line, fit_by_line)`` — a keyed video URL per ``line_id``
    (which wins as the shot's ``video_url`` in :func:`assemble_shots`, and the
    compositor honours ``video_url`` over any ``image_url``) and a matching
    ``fit="cover"`` so the clip fills the frame. URLs are consumed in order, one
    per non-skipped shot line, until either runs out.

    By default clips front-load onto the first non-skipped lines. With
    ``distribute=True`` they are **spread evenly across** the non-skipped lines
    (priority order preserved left-to-right) — used for AI-video / lesson cutaways
    that should recur through the timeline rather than cluster up front (see
    :func:`_distribute_indices`).

    ``skip_line_ids`` names lines already claimed by another visual (the generic
    caller passes the lines real-media *images* took, so an image and a clip
    never fight for the same line — videos take their own lines). Empty
    ``real_media_video_urls`` → ``({}, {})`` so the caller is byte-for-byte
    unchanged.
    """
    skip = set(skip_line_ids)
    video_by_line: dict[str, str] = {}
    fit_by_line: dict[str, str] = {}
    if distribute:
        eligible = [str(s["line_id"]) for s in shots if str(s["line_id"]) not in skip]
        for k, idx in enumerate(_distribute_indices(len(eligible), len(real_media_video_urls))):
            video_by_line[eligible[idx]] = real_media_video_urls[k]
            fit_by_line[eligible[idx]] = "cover"
        return video_by_line, fit_by_line
    media = iter(real_media_video_urls)
    for shot in shots:
        line_id = str(shot["line_id"])
        if line_id in skip:
            continue
        url = next(media, None)
        if url is None:
            break
        video_by_line[line_id] = url
        fit_by_line[line_id] = "cover"
    return video_by_line, fit_by_line


@dataclass(frozen=True)
class GenericLineMedia:
    """Which generic-film lines carry a real/motion/AI-video visual, and what.

    ``image_by_line`` / ``video_by_line`` are the keyed visuals ``video_render``
    hands to :func:`assemble_shots`; ``fit_by_line`` their matching fits. Every
    line in :attr:`covered_line_ids` shows that visual INSTEAD of its per-line
    AI still, so ``ai_image_gen`` must not pay to render one for it.
    """

    image_by_line: dict[str, str]
    video_by_line: dict[str, str]
    fit_by_line: dict[str, str]

    @property
    def covered_line_ids(self) -> frozenset[str]:
        return frozenset(self.image_by_line) | frozenset(self.video_by_line)


def assign_generic_line_media(ctx: RunContext) -> GenericLineMedia:
    """The ONE generic coverage rule, shared by ``video_render`` and ``ai_image_gen``.

    Precedence, highest first: real-media images (pinned → extracted →
    discovered) front-load the leading lines; real-media clips take their own
    lines; keyed motion-graphics clips join 1:1 on any line still free, then the
    unkeyed motion pool and finally ``ai_video_gen`` clips spread evenly over
    the remaining lines. Pure function of ``storyboard_draft`` + the four opt-in
    media stages' outputs, so a resumed or regenerated run recomputes the
    identical assignment. Every opt-in stage skipped (or no storyboard) → all
    maps empty → nothing covered.
    """
    story = ctx.prior_output("storyboard_draft") or {}
    shots = story.get("shots") or []
    source_ingest = ctx.prior_output("source_ingest")
    media_discovery = ctx.prior_output("media_discovery")
    real_media_urls = collect_real_media_urls(
        source_ingest=source_ingest,
        media_discovery=media_discovery,
    )
    real_media_videos = collect_real_media_videos(
        source_ingest=source_ingest,
        media_discovery=media_discovery,
    )
    image_by_line, image_fit = assign_real_media_by_line(shots, real_media_urls)
    # Videos take their own lines (skip those an image already claimed) so an
    # image and a clip never fight for one line; video_url beats image_url.
    video_by_line, video_fit = assign_real_media_videos_by_line(
        shots, real_media_videos, skip_line_ids=image_by_line.keys()
    )
    # Branded motion-graphics clips (motion_graphics_gen, opt-in): above AI
    # video, below footage/curated media. Keyed clips join 1:1; unkeyed clips
    # spread round-robin.
    motion_keyed, motion_pool = collect_motion_clips_by_line(
        ctx.prior_output("motion_graphics_gen")
    )
    motion_fit: dict[str, str] = {}
    for line_id, url in motion_keyed.items():
        if line_id not in image_by_line and line_id not in video_by_line:
            video_by_line[line_id] = url
            motion_fit[line_id] = "cover"
    motion_pool_by_line, motion_pool_fit = assign_real_media_videos_by_line(
        shots,
        motion_pool,
        skip_line_ids=set(image_by_line) | set(video_by_line),
        distribute=True,
    )
    video_by_line.update(motion_pool_by_line)
    motion_fit.update(motion_pool_fit)
    # AI-generated clips (ai_video_gen, opt-in) are the lowest-priority motion:
    # they fill only the lines no curated image or clip claimed, SPREAD EVENLY
    # (``distribute=True``) so generative motion recurs through the timeline.
    ai_vid_by_line, ai_vid_fit = assign_real_media_videos_by_line(
        shots,
        collect_ai_video_urls(ctx.prior_output("ai_video_gen")),
        skip_line_ids=set(image_by_line) | set(video_by_line),
        distribute=True,
    )
    video_by_line.update(ai_vid_by_line)
    return GenericLineMedia(
        image_by_line=image_by_line,
        video_by_line=video_by_line,
        fit_by_line={**image_fit, **video_fit, **motion_fit, **ai_vid_fit},
    )
