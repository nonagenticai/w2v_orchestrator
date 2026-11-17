"""Refresh shot asset URLs to a fresh presigned TTL just before compile.

Every video-compile stage builds shots whose ``image_url`` / ``video_url`` /
``audio_url`` — plus the nested ``overlay_clips[].url`` and
``split.second_source_url`` — are presigned MinIO URLs read from prior stage
outputs. Those URLs were signed at generation time with a 1h TTL
(``presign_expiry_seconds``, the GLOBAL default — see ``config.py``). A run
paused or iterated past that hour hands video_compiler a dead signature → its
``renderer._fetch`` 403s → the whole compile 500s. A fresh autopilot run (~25min)
never hits it.

This helper re-mints each owned-MinIO asset URL with the orchestrator's own
credentials at the moment shots are handed to video_compiler — seconds before
the fetch — using ``compile_budget.PRESIGN_TTL_SECONDS`` (NOT the global
``presign_expiry_seconds``), because a compile the orchestrator itself now
waits up to ``compile_budget.COMPILE_WAIT_CEILING_SECONDS`` for can legitimately
still be running when the global 1h TTL would have expired (2026-09-17
incident, ``shot_asset_unfetchable`` — a master shipped missing narration).
Deliberately scoped to THIS helper rather than raising the global default:
every call site below feeds a compile/render/concat/composite request
(video_compiler, audiobook concat, slide_service) — none is browser-facing —
so this is the one and only presign path that should ever hand out the long
TTL. Browser-facing presigns (``uploads_video`` ``original_url``, storyboard
panel thumbnails, the final-artifact rewrite path) go through
``StorageService.get_presigned_url``/``resign_presigned_url`` directly and stay
on the global 3600s default. It is the consumption-side twin of the browser
``artifact_proxy`` and the YouTube ``_presigned_video_url`` re-sign paths,
which likewise stay on the global default.

Best-effort by design: a non-owned / unparseable URL, a storage backend that
can't re-sign (the local/test fakes), or any storage error keeps the original
value. The helper can only refresh a URL or leave it untouched — it must never
fail a compile that would otherwise have proceeded.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterator, MutableMapping
from typing import Any
from urllib.parse import unquote, urlparse

from src.services.compile_budget import PRESIGN_TTL_SECONDS
from src.services.tts_storage import _get_storage

logger = logging.getLogger(__name__)

#: Top-level shot keys that carry a fetchable asset URL the compositor downloads.
_ASSET_URL_KEYS = ("image_url", "video_url", "audio_url")

#: ``(container_key, url_key)`` pairs where the URL lives one level down, inside a
#: LIST of dicts on the shot. ``overlay_clips`` is the animated-graphic track: the
#: renderer fetches each clip's ``url`` and, when the fetch fails, logs
#: ``overlay_clip.dropped`` and RENDERS ON — so an expired signature here ships a
#: reel with no graphics, no error, and a green run. That silence is why the
#: nested walk exists at all: the three top-level keys fail loudly (a dead
#: ``video_url`` 500s the compile), this one does not.
_NESTED_LIST_URL_KEYS: tuple[tuple[str, str], ...] = (("overlay_clips", "url"),)

#: Same idea for a single nested DICT. ``split.second_source_url`` is the B-side
#: of a split-screen shot, stamped by ``shared/footage_plans`` from the same
#: presigned MinIO pool as ``video_url`` — and it is emitted TODAY (promo_render,
#: demo_video footage direction), so unlike ``overlay_clips`` this one is a live
#: expiry hole, not a pre-emptive one.
_NESTED_DICT_URL_KEYS: tuple[tuple[str, str], ...] = (("split", "second_source_url"),)


def _url_slots(shot: MutableMapping[str, Any]) -> Iterator[tuple[MutableMapping[str, Any], str]]:
    """Every ``(container, key)`` in *shot* that may hold a re-signable URL.

    Yields the OWNING dict rather than the value, so the caller writes the fresh
    URL back where it found it without knowing how deep it was. Separated from the
    re-sign loop because "where can a URL hide in a shot" is a payload-shape
    question that grows with the compiler's Shot model, while the loop below is
    fixed policy (cache, soft-fail, keep-the-original).

    A malformed container — ``overlay_clips`` that is not a list of dicts, a
    ``split`` that is not a dict — yields nothing rather than raising: this
    helper's contract is that it can only refresh a URL or leave the payload
    alone, never reshape it or fail a compile that would have proceeded.
    """
    for key in _ASSET_URL_KEYS:
        yield shot, key
    for container_key, url_key in _NESTED_DICT_URL_KEYS:
        nested = shot.get(container_key)
        if isinstance(nested, dict):
            yield nested, url_key
    for container_key, url_key in _NESTED_LIST_URL_KEYS:
        entries = shot.get(container_key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                yield entry, url_key


async def _resign_one(
    url: str,
    *,
    resign: Callable[[str], Awaitable[str | None]],
    get_presigned: Callable[..., Awaitable[str | None]] | None,
    ttl_seconds: int,
) -> str:
    """Re-sign a single URL, upgrading to *ttl_seconds* when possible.

    Split out of :func:`refresh_shot_asset_urls` purely to keep that function's
    branch count under the repo's complexity ceiling — no behaviour lives here
    that didn't live inline before.
    """
    signed = await resign(url)
    if signed is None:
        # Declined (non-owned host/bucket or unparseable URL): keep the
        # original. Logged so a later downstream 403 isn't a silent mystery —
        # the re-sign was attempted and deliberately skipped.
        logger.debug("refresh_shot_asset_urls: re-sign declined for %s (not an owned URL?)", url)
        return url
    if get_presigned is None:
        # No per-call-TTL capable method on this storage (a fake without it):
        # keep the default-TTL signature rather than failing the compile over
        # a TTL we can't upgrade.
        return signed
    # `resign` above already proved `url` is ours (bucket we own, host we
    # presign for) — reuse that ownership check rather than re-deriving it,
    # and just re-derive the {bucket}/{key} split (the same split
    # `resign_presigned_url` does internally) to mint a SECOND signature at
    # the compile TTL. A cheap local SigV4 sign (no network I/O) — the wasted
    # first signature is the price of not duplicating the ownership guard
    # here.
    bucket, _, obj_key = urlparse(url).path.lstrip("/").partition("/")
    bucket, obj_key = unquote(bucket), unquote(obj_key)
    if not obj_key:
        return signed
    long_signed = await get_presigned(obj_key, expires=ttl_seconds, bucket=bucket)
    return long_signed or signed


async def refresh_shot_asset_urls(
    shots: list[dict[str, Any]],
    *,
    storage: object = None,
    ttl_seconds: int = PRESIGN_TTL_SECONDS,
) -> list[dict[str, Any]]:
    """Re-sign each owned-MinIO asset URL in *shots* in place; return *shots*.

    ``ttl_seconds`` defaults to ``compile_budget.PRESIGN_TTL_SECONDS`` — every
    caller of this helper is a compile/render/concat/composite input path (see
    the module docstring), so that is the correct default for all of them.
    Pass an explicit shorter value only for a caller that is NOT one of those
    (none exist today).

    Reuses the process-wide storage singleton
    (``StorageService.resign_presigned_url`` for OWNERSHIP validation,
    ``StorageService.get_presigned_url`` for the actual TTL-bearing signature —
    ``resign_presigned_url`` itself has no per-call TTL override, it always
    signs at the service's configured ``presign_expiry_seconds``). Falls back
    to a no-op when storage is unavailable or can't re-sign (local backend / a
    test fake without the method). Repeated URLs — e.g. one redesign section
    reused across several after-lines — are re-signed once via the cache.

    Walks the three top-level asset keys AND the nested ones an asset can hide in
    (``overlay_clips[].url``, ``split.second_source_url``); see the key tables
    above for why the nested ones matter more than their obscurity suggests.
    """
    if storage is None:
        try:
            storage = await _get_storage()
        except Exception as exc:  # noqa: BLE001 -- never fail compile on a storage blip
            logger.warning(
                "refresh_shot_asset_urls: storage unavailable, skipping re-sign: %s", exc
            )
            return shots
    resign = getattr(storage, "resign_presigned_url", None)
    if resign is None:
        return shots
    get_presigned = getattr(storage, "get_presigned_url", None)
    # Keyed by ORIGINAL url, and deliberately shared across every shot and every
    # nesting level: the compiler dedupes its per-job fetches by URL, so two
    # references to one source MUST come back as the same new URL. Minting a fresh
    # signature per call site would silently double the downloads of a reused clip.
    cache: dict[str, str] = {}

    for shot in shots:
        for container, key in _url_slots(shot):
            url = container.get(key)
            if not isinstance(url, str) or not url:
                continue
            if url not in cache:
                try:
                    cache[url] = await _resign_one(
                        url, resign=resign, get_presigned=get_presigned, ttl_seconds=ttl_seconds
                    )
                except Exception as exc:  # noqa: BLE001 -- a per-URL sign failure keeps the original
                    logger.warning("refresh_shot_asset_urls: re-sign failed for %s: %s", url, exc)
                    cache[url] = url
            container[key] = cache[url]
    return shots
