"""Media routes — media-discovery search for the creation wizard's "Find media" panel.

The wizard searches media-discovery's federated media sources live so the
operator can pin real images/clips onto a project's ``brief.sources``.
media-discovery stays internal on the ``media_discovery`` discovery peer (called
directly; KAG, which used to proxy it, is retired); this re-exposes a trimmed
search on the orchestrator's public ``w2v.nonagentic.ai`` surface.

The orchestrator's own service license token authorizes the S2S call (media
search hits external/public media adapters, not tenant-private data), so the
route only requires an authenticated caller — ``MediaDiscoveryClient`` carries
the token. This is the operator-curated surface; ``media_discovery`` is the
pipeline-auto one.
"""

from __future__ import annotations

from typing import Annotated, Final, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator

from src.api.routes._common import TenantIdDep
from src.clients.media_discovery_client import MediaDiscoveryClient
from src.clients.schemas.media_discovery import MediaItem
from src.config import get_settings
from src.services.visual_media import fetch_image_bytes, looks_like_svg, sniff_image_format

router = APIRouter(prefix="/api/media", tags=["media"])

_SEARCH_TIMEOUT_S = 60.0

THUMB_MAX_BYTES: Final = 5 * 1024 * 1024
_THUMB_TIMEOUT_S: Final = 15.0
_THUMB_MEDIA_TYPES: Final = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}


class ThumbRefusal(BaseModel):
    """The 400 body when ``GET /api/media/thumb`` will not proxy a URL.

    A MACHINE-READABLE ``reason`` plus a human sentence, because the bare
    ``"thumbnail host is not allowed"`` string this replaces reached the operator as a
    broken <img> with no explanation. ``host`` is the PARSED host (empty when the URL
    did not parse), so support can see which CDN to consider admitting.

    The contract this body OFFERS a client: branch on ``reason`` (a closed Literal,
    safe to switch on), and render ``message`` verbatim rather than composing prose
    from ``reason``. No consumer does so yet — w2v_frontend PR #443 is the first and
    is still open, so nothing on its ``main`` reads either field today.
    """

    reason: Literal["host_not_allowed", "invalid_url", "vector_image_unsupported"]
    host: str = ""
    message: str = ""

    @model_validator(mode="after")
    def _fill_message(self) -> ThumbRefusal:
        if self.message:
            return self
        if self.reason == "invalid_url":
            self.message = "Thumbnail unavailable — that is not a usable image address."
        elif self.reason == "vector_image_unsupported":
            # NOT a server error: the asset is fine, we decline to proxy script-bearing
            # markup as an image. The operator should read "can't preview", not "broken".
            self.message = "No preview — this source is a vector image (SVG) we can't display here."
        else:
            named = f" ({self.host})" if self.host else ""
            self.message = (
                f"Thumbnail unavailable — this source{named} is not supported for preview."
            )
        return self


def thumb_allowed_host_suffixes() -> tuple[str, ...]:
    """The configured allow-list (``API_THUMB_ALLOWED_HOST_SUFFIXES``).

    Read per call rather than bound at import so a ConfigMap change reaches the route
    on the next pod, and so a test can exercise a different list without reloading the
    module. ``get_settings`` is ``lru_cache``d, so this is a dict lookup.
    """
    return tuple(get_settings().api.thumb_allowed_host_suffixes)


def thumb_refusal(url: str) -> ThumbRefusal | None:
    """``None`` when ``url`` may be proxied, else the typed reason it may not.

    The host is taken from :func:`urlsplit`'s PARSED ``hostname`` — never from the raw
    string — and matched as ``host == suffix`` or ``host.endswith("." + suffix)``. That
    dot-boundary is what makes the allow-list unspoofable by suffix: ``evil-ytimg.com``
    and ``ytimg.com.evil.example`` both fail BOTH arms, and a userinfo trick
    (``https://i.ytimg.com@evil.example/x.jpg``) parses to ``hostname='evil.example'``
    and is additionally refused outright for carrying credentials. A substring or
    ``re.search`` test would admit every one of those, which is why this is neither.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return ThumbRefusal(reason="invalid_url", host="")
    host = (parts.hostname or "").lower().rstrip(".")
    if parts.scheme not in ("http", "https") or not host or parts.username or parts.password:
        return ThumbRefusal(reason="invalid_url", host=host)
    if any(host == s or host.endswith("." + s) for s in thumb_allowed_host_suffixes()):
        return None
    return ThumbRefusal(reason="host_not_allowed", host=host)


def thumb_host_allowed(url: str) -> bool:
    """True for an http(s) URL whose host is (a subdomain of) an allowed CDN."""
    return thumb_refusal(url) is None


class MediaSearchBody(BaseModel):
    query: str = Field(min_length=1, max_length=240)
    sources: list[str] | None = None
    media_types: list[str] | None = None
    limit: int = Field(default=12, ge=1, le=50)


class MediaResult(BaseModel):
    id: str
    title: str = ""
    url: str = ""
    thumbnail_url: str = ""
    media_type: str = ""
    source: str = ""
    author_name: str = ""
    quality_score: float = 0.0


class MediaSearchResult(BaseModel):
    results: list[MediaResult]
    total: int = 0
    sources_queried: list[str] = []
    sources_failed: list[str] = []


def _to_result(item: MediaItem) -> MediaResult | None:
    if not item.id:
        return None
    return MediaResult(
        id=item.id,
        title=item.title,
        url=item.url,
        thumbnail_url=item.thumbnail_url or "",
        media_type=item.media_type,
        source=item.source,
        author_name=item.author_name or "",
        # Nullable server-side (unscored items) — fold to 0.0 for the FE.
        quality_score=item.quality_score or 0.0,
    )


@router.post("/search", operation_id="media.search")
async def search_media(body: MediaSearchBody, tenant_id: TenantIdDep) -> MediaSearchResult:
    """Search media-discovery and return a trimmed, FE-friendly result set."""
    client = MediaDiscoveryClient(timeout=_SEARCH_TIMEOUT_S)
    try:
        resp = await client.search(
            body.query,
            sources=body.sources,
            media_types=body.media_types,
            limit=body.limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"media search failed: {exc}") from exc
    finally:
        await client.close()
    parsed = (_to_result(item) for item in resp.results)
    results = [m for m in parsed if m is not None]
    return MediaSearchResult(
        results=results,
        total=resp.total or len(results),
        sources_queried=resp.sources_queried,
        sources_failed=resp.sources_failed,
    )


@router.get("/thumb", operation_id="media.thumb")
async def media_thumbnail(
    url: Annotated[str, Query(min_length=1, max_length=2048)],
    tenant_id: TenantIdDep,
) -> Response:
    """Same-origin proxy for a "Find media" result's external ``thumbnail_url``.

    The SPA's CSP is ``img-src 'self' data: blob:``, so an ``<img>`` pointing at
    ``i.ytimg.com`` never loads. The FE calls
    ``GET /api/media/thumb?url=<encodeURIComponent(thumbnail_url)>`` instead.

    400 — with a :class:`ThumbRefusal` body, not a bare string — when the host is not
    on the configured allow-list (``API_THUMB_ALLOWED_HOST_SUFFIXES``, defaulting to
    :data:`~src.config.DEFAULT_THUMB_ALLOWED_HOST_SUFFIXES`), and again — reason
    ``vector_image_unsupported`` — when the bytes are an SVG. SVG is still never
    proxied (script-bearing markup); it simply stops being reported as a server fault,
    because the asset is healthy and the refusal is ours.

    502 only for a genuine failure: the upstream fetch broke, the bytes are neither a
    raster image (png/jpeg/webp/gif, judged by the BYTES) nor an SVG, or the body
    exceeds :data:`THUMB_MAX_BYTES`. The fetch goes through the SSRF-guarded
    ``fetch_image_bytes`` (``w2v_common.net.safe_get``), which identifies itself with
    :data:`~src.services.visual_media.MEDIA_FETCH_USER_AGENT` — without it Wikimedia
    403s the default httpx UA and every ``upload.wikimedia.org`` thumbnail 502s.
    """
    refusal = thumb_refusal(url)
    if refusal is not None:
        raise HTTPException(status_code=400, detail=refusal.model_dump())
    try:
        data, _declared = await fetch_image_bytes(
            url, timeout=_THUMB_TIMEOUT_S, max_bytes=THUMB_MAX_BYTES
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail="thumbnail fetch failed") from exc
    sniffed = sniff_image_format(data)
    if sniffed is None:
        if looks_like_svg(data):
            # A DELIBERATE refusal, so it gets the typed 400 shape, not a 5xx. SVG stays
            # unproxied (script-bearing markup); the FE can now say so instead of
            # rendering a server error over a perfectly healthy Wikimedia asset.
            raise HTTPException(
                status_code=400,
                detail=ThumbRefusal(
                    reason="vector_image_unsupported",
                    host=(urlsplit(url).hostname or "").lower(),
                ).model_dump(),
            )
        raise HTTPException(status_code=502, detail="thumbnail is not a raster image")
    return Response(
        content=data,
        media_type=_THUMB_MEDIA_TYPES[sniffed],
        headers={"Cache-Control": "private, max-age=86400"},
    )
