"""Re-sign a panel's stored media URL at READ time.

``media_ref.url`` is a presigned MinIO URL minted when the image was generated,
with the storage service's ``presign_expiry_seconds`` TTL (**1 hour**). The
storyboard is not a run: a board is opened, edited and re-opened for days, so by
design almost every read of a generated panel happens AFTER that signature has
expired. Serving the stored string then yields ``Request has expired`` on every
board thumbnail — the image is fine, the pointer is dead.

So the stored value is treated as a durable *object identity*, and the fetchable
URL is minted per response. This is the read-side twin of
``stages/shot_presign.refresh_shot_asset_urls`` (which re-signs shot URLs just
before compile) and of the browser ``artifact_proxy`` — same helper
(``StorageService.resign_presigned_url``), same ownership guard, same
best-effort posture.

Best-effort by design: a non-owned/unparseable URL, a backend that cannot
re-sign (local/test fakes), or any storage error keeps the ORIGINAL value. This
helper can only refresh a URL or leave it untouched — a storage blip must never
turn a board read into a 500.
"""

from __future__ import annotations

import logging
from typing import TypedDict

from src.services.tts_storage import _get_storage

log = logging.getLogger(__name__)


class MediaRefJson(TypedDict, total=False):
    """The JSONB shape of ``StoryboardPanel.media_ref`` — the dict twin of
    :class:`src.models.showcase_line.MediaRef`.

    Spelled as a ``TypedDict`` rather than ``dict[str, Any]`` because the repo
    ratchets the number of ``dict[str, Any]`` annotations (``tests/test_any_ratchet.py``)
    — and because the three keys here are genuinely known. ``total=False``: a
    text-only Phase-0 panel persists an empty ``{}``.
    """

    type: str | None
    asset_id: str | None
    url: str | None


async def refresh_panel_media_url(
    media_ref: MediaRefJson | None, *, storage: object = None
) -> MediaRefJson | None:
    """Return *media_ref* with a freshly presigned ``url``, or unchanged.

    Never mutates the argument: the caller's dict is often the SQLAlchemy row's
    JSONB value, and rewriting it in place would mark the ORM object dirty and
    flush a pointless UPDATE (writing a URL that is stale again in an hour).
    """
    if not media_ref:
        return media_ref
    url = media_ref.get("url")
    if not isinstance(url, str) or not url:
        return media_ref
    if storage is None:
        try:
            storage = await _get_storage()
        except Exception as exc:  # noqa: BLE001 -- never fail a board read on a storage blip
            log.warning("panel media re-sign skipped, storage unavailable: %s", exc)
            return media_ref
    resign = getattr(storage, "resign_presigned_url", None)
    if resign is None:
        return media_ref
    try:
        fresh = await resign(url)
    except Exception as exc:  # noqa: BLE001 -- a sign failure keeps the (possibly stale) original
        log.warning("panel media re-sign failed for %s: %s", url, exc)
        return media_ref
    if not fresh or fresh == url:
        # ``None`` = declined (not an owned URL) — e.g. an operator-pasted
        # external image, which needs no signature and must not be rewritten.
        return media_ref
    return {**media_ref, "url": fresh}
