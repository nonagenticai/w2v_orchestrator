"""Open one MinIO object and stream it to the browser.

The HTTP-layer plumbing shared by the orchestrator's object proxies: open a
range-aware read with the orchestrator's OWN MinIO credentials, whitelist the
headers that cross in each direction, and stream the body in bounded chunks,
releasing the upstream connection when the client disconnects.

Deliberately knows nothing about *authorization*. Deciding which bucket + key a
given caller may address is the calling route's whole job — this module is
reached only after that decision, and it will fetch whatever key it is handed.

The inbound query string is ignored on purpose: a stored presigned URL carries
the issuing service's signature with a 1-hour TTL, so forwarding it makes an
object unreachable an hour after it was written. Reading with our own
credentials has no TTL.

``Range`` / ``If-Range`` / ``If-Modified-Since`` / ``If-None-Match`` round-trip
so ``<video>`` seeking and conditional fetches work without a server-side cache.

The object's stored ``Content-Type`` is forwarded as-is by default, which is only
as trustworthy as whatever wrote the object. Historical stage artifacts were all
written as ``application/json`` regardless of extension (see
``storage_service.guess_artifact_content_type``), and re-writing them is not on the
table — so a caller that knows what the object is may pass ``media_type`` to
:func:`stream_object` and have that served instead.

Failure mapping (matching ``routes/artifact_proxy.py``, whose ``_open_minio_object``
this is modelled on): an uninitialised/streaming-incapable storage backend is a
**503**, MinIO's 403 and 404 both collapse to a **404** so a probe cannot tell
"absent" from "not permitted", and any other S3 or transport error is a **502**.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator, Mapping
from typing import Protocol

from fastapi import HTTPException, Request
from starlette.responses import Response, StreamingResponse

from src.services.tts_storage import (
    _get_storage,  # why: reuse the process-wide lazy MinIO singleton already used by TTS/slide storage
)

log = logging.getLogger(__name__)

# Headers forwarded from the inbound request to MinIO. Whitelist only —
# anything else (cookie, authorization) would be a leak.
FORWARDED_REQUEST_HEADERS = frozenset({"range", "if-range", "if-modified-since", "if-none-match"})

# Headers forwarded from MinIO back to the browser. Whitelist again — MinIO
# stamps various ``X-Amz-*`` headers that mean nothing to the SPA.
FORWARDED_RESPONSE_HEADERS = frozenset(
    {
        "content-type",
        "content-length",
        "content-range",
        "accept-ranges",
        "etag",
        "last-modified",
        "cache-control",
    }
)

# 64 KiB keeps memory bounded on a long object without paying syscall overhead
# per packet.
STREAM_CHUNK_BYTES = 64 * 1024


class StreamedLike(Protocol):
    """The streaming read handle ``StorageService.get_object_stream`` returns.

    Spelled structurally rather than importing ``w2v_common.storage.StreamedObject``
    so a test fake satisfies it without subclassing, and so this module does not
    grow a hard dependency on a facade class it only reads three members of.
    """

    status: int
    headers: Mapping[str, str]

    def iter_chunks(self, chunk_size: int) -> AsyncIterator[bytes]: ...

    async def aclose(self) -> None: ...


def filter_headers(items: Mapping[str, str], allowed: frozenset[str]) -> dict[str, str]:
    """Whitelist-filter request or response headers (case-insensitive).

    Accepts both ``starlette.datastructures.Headers`` and ``multidict.CIMultiDict``
    (aiohttp), which both satisfy ``Mapping[str, str]``.
    """
    return {name: value for name, value in items.items() if name.lower() in allowed}


async def open_object(
    bucket: str, object_key: str, request_headers: dict[str, str]
) -> StreamedLike:
    """Open a range-aware streaming read against MinIO; raise ``HTTPException`` on failure.

    The caller MUST release the handle via ``aclose()`` — :func:`stream_object`
    does that in the body generator's ``finally``.
    """
    try:
        storage = await _get_storage()
    except Exception as exc:
        # No backend at all (misconfigured / not yet initialised). 503 rather
        # than an empty 200, which would render as a silently broken image.
        log.warning("object stream: storage singleton unavailable: %s", exc)
        raise HTTPException(503, "artifact storage not initialised") from exc

    # ``get_object_stream`` is beyond the minimal ``_StorageLike`` protocol, so
    # reach it via getattr (same house pattern as the other non-protocol storage
    # methods) — a storage without it is treated as uninitialised.
    open_stream = getattr(storage, "get_object_stream", None)
    if open_stream is None:
        raise HTTPException(503, "artifact storage not initialised")

    # miniopy_async raises ``S3Error`` for any HTTP-level error; the class
    # carries ``code`` ("NoSuchKey", "AccessDenied", …) and ``status_code``.
    try:
        from miniopy_async.error import S3Error
    except ImportError as exc:  # pragma: no cover -- prod always has it
        raise HTTPException(500, "minio sdk missing") from exc

    try:
        stream = await open_stream(bucket, object_key, request_headers=request_headers or None)
    except RuntimeError as exc:
        # Backend not initialised (or not MinIO) — the facade raises RuntimeError.
        raise HTTPException(503, "artifact storage not initialised") from exc
    except S3Error as exc:
        status = getattr(exc, "status_code", 500) or 500
        log.info(
            "object stream: MinIO S3Error for %s/%s: code=%s status=%s",
            bucket,
            object_key,
            getattr(exc, "code", "?"),
            status,
        )
        if status in (403, 404):
            raise HTTPException(404, "artifact not found") from exc
        raise HTTPException(502, "artifact unreachable") from exc
    except Exception as exc:
        # Transport-level / connection / DNS / TLS — never let an opaque 500
        # surface; the browser would just retry, hammering MinIO.
        log.warning("object stream: MinIO transport error for %s/%s: %s", bucket, object_key, exc)
        raise HTTPException(502, "artifact unreachable") from exc

    return stream


async def stream_object(
    bucket: str,
    object_key: str,
    request: Request,
    *,
    media_type: str | None = None,
) -> Response:
    """Stream ``{bucket}/{object_key}`` to the caller, forwarding range headers.

    Authorization is the caller's responsibility (see the module docstring).

    ``media_type`` overrides the ``Content-Type`` MinIO reports for the object,
    for a caller that knows what the object is and does not trust the stored
    metadata. Keyword-only, and unset by default: forwarding upstream's type stays
    the behaviour for every generic proxy. See the module docstring for why a
    caller would ever need it.
    """
    upstream = await open_object(
        bucket, object_key, filter_headers(request.headers, FORWARDED_REQUEST_HEADERS)
    )
    response_headers = filter_headers(upstream.headers, FORWARDED_RESPONSE_HEADERS)
    if media_type is not None:
        # Drop the forwarded type case-insensitively before re-adding ours: aiohttp
        # hands back ``istr`` keys whose casing is the server's ("Content-Type"), and
        # Starlette lowercases raw header names AFTER building them — so leaving the
        # forwarded entry in place would emit two Content-Type headers, with MinIO's
        # stored (wrong) one first.
        for name in [n for n in response_headers if n.lower() == "content-type"]:
            del response_headers[name]
        response_headers["content-type"] = media_type

    async def _body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.iter_chunks(STREAM_CHUNK_BYTES):
                yield chunk
        finally:
            # Best-effort connection release — the body iter is already
            # exhausted, so any error here is the aiohttp internal pool tearing
            # down a half-spent connection. Nothing actionable.
            with contextlib.suppress(Exception):
                await upstream.aclose()

    return StreamingResponse(
        _body(),
        status_code=upstream.status,
        media_type=response_headers.get("content-type", "application/octet-stream"),
        headers=response_headers,
    )
