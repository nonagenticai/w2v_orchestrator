"""Bound a real-world still to the film it will appear in — resolution AND encoding.

A pinned or discovered photo arrives at whatever size its camera produced: a
phone's 4032x3024 JPEG (12 MP, ~3.7 MB) is the COMMON case, not the edge one.
Nothing downstream needs those pixels — the local Ken Burns renderer draws at
most a 1920px long edge (``ken_burns_local.LOCAL_MAX_LONG_EDGE``) and the film is
at most 4K — but every stage that re-encodes the still pays for them. Live run
01a1639e (2026-10-22): the cover-crop pass re-encoded that 3.7 MB JPEG as a
**lossless PNG** at native 4032x2268, the PNG weighed more than the 12 MiB fetch
cap of ``ken_burns._fetch_own_asset``, the local render could not read it back,
and the shot shipped as a frozen still (``ken_burns_local_fallback_failed``).

The fix is to never hand a downstream stage more still than the frame can use:
cover-fit to :data:`STILL_HEADROOM` x the output frame (room for the Ken Burns
push-in and a pan's travel) and encode a photo as JPEG. Alpha keeps PNG — a JPEG
would flatten it — but is bounded the same way.
"""

from __future__ import annotations

import io
import logging

from PIL import Image, ImageOps

log = logging.getLogger(__name__)

#: How many output frames' worth of pixels a still may keep, per axis. The Ken
#: Burns move zooms ~12-14% and pans within that; 2x the frame is ample and still
#: sharp at the 4K tier.
STILL_HEADROOM = 2.0
#: JPEG quality of a re-encoded photo still. A cover-fitted 3840x2160 photo lands
#: at ~2-5 MB — a quarter of the fetch cap — and is visually lossless at the
#: <=1920px the local renderer draws.
STILL_JPEG_QUALITY = 90
_EXIF_ORIENTATION = 0x0112


def bound_to_frame(
    image: Image.Image, width: int, height: int, *, headroom: float = STILL_HEADROOM
) -> Image.Image:
    """``image`` Lanczos-downscaled to just cover ``headroom`` x ``width`` x ``height``.

    The COVER factor (``max``), so the result still fills the bounded frame on
    both axes under any fit; never upscales. Returns ``image`` itself when it is
    already within the bound.
    """
    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0 or width <= 0 or height <= 0:
        return image
    factor = max(width * headroom / src_w, height * headroom / src_h)
    if factor >= 1.0:
        return image
    size = (max(1, round(src_w * factor)), max(1, round(src_h * factor)))
    return image.resize(size, Image.Resampling.LANCZOS)


def encode_still(image: Image.Image) -> tuple[bytes, str]:
    """``(bytes, ext)`` — JPEG for an opaque still, PNG only when it has alpha."""
    out = io.BytesIO()
    if image.mode in {"RGBA", "LA"}:
        image.save(out, format="PNG", optimize=True)
        return out.getvalue(), "png"
    image.convert("RGB").save(out, format="JPEG", quality=STILL_JPEG_QUALITY, optimize=True)
    return out.getvalue(), "jpg"


#: Step-down ladder when an encoded still is over its byte cap: JPEG quality
#: first (invisible at the <=1920px the local renderer draws), then the pixel
#: headroom. At the 4K tier 2x the frame is 7680x4320 (33 MP), where q90 of a
#: grainy photo or any alpha PNG can exceed 12 MiB.
_QUALITY_STEPS = (STILL_JPEG_QUALITY, 80, 70)
_SCALE_STEPS = (1.0, 0.75, 0.5)


def encode_still_under(image: Image.Image, max_bytes: int) -> tuple[bytes, str] | None:
    """:func:`encode_still`, stepped down until it fits ``max_bytes``; ``None`` if never.

    Alpha stays PNG (only the scale steps apply); an opaque still walks the JPEG
    quality steps at each scale.
    """
    alpha = image.mode in {"RGBA", "LA"}
    src_w, src_h = image.size
    for scale in _SCALE_STEPS:
        img = image
        if scale < 1.0:
            size = (max(1, round(src_w * scale)), max(1, round(src_h * scale)))
            img = image.resize(size, Image.Resampling.LANCZOS)
        if alpha:
            data, ext = encode_still(img)
            if len(data) <= max_bytes:
                return data, ext
            continue
        rgb = img if img.mode == "RGB" else img.convert("RGB")
        for quality in _QUALITY_STEPS:
            out = io.BytesIO()
            rgb.save(out, format="JPEG", quality=quality, optimize=True)
            if out.tell() <= max_bytes:
                return out.getvalue(), "jpg"
    return None


class StillRejectedError(ValueError):
    """An ingested image that cannot be made safe to store (bomb, undecodable, too big)."""


#: Decode ceiling for an ingested still, checked from the header BEFORE any pixel
#: is decoded. A 48 MP phone original is 48M px; 100M px leaves room for a
#: panorama while refusing a decompression bomb (a tiny file declaring a
#: gigapixel canvas) before it can allocate. Pillow's own guard only raises at
#: 2 x ``Image.MAX_IMAGE_PIXELS`` (~179M px).
MAX_STILL_PIXELS = 100_000_000


def _fit(
    data: bytes, width: int, height: int, *, force: bool, max_bytes: int | None = None
) -> tuple[bytes, str] | None:
    """Shared body of :func:`normalise_still` / :func:`fit_ingested_still`; raises.

    ``None`` when there is nothing to do (within the bound, no rotation) and not
    ``force``, or the image is animated and not ``force``. With ``max_bytes`` the
    encode steps down (:func:`encode_still_under`) and raises if it never fits.
    """
    with Image.open(io.BytesIO(data)) as src:
        src_w, src_h = src.size
        if src_w * src_h > MAX_STILL_PIXELS:
            raise StillRejectedError(
                f"{src_w}x{src_h} is {src_w * src_h:,} px, over the {MAX_STILL_PIXELS:,} px "
                "decode limit"
            )
        # MPO (iPhone/Samsung portrait, depth and gain-map JPEGs) reports several
        # frames but is a still: frame 0 is the photo.
        if getattr(src, "n_frames", 1) > 1 and src.format != "MPO":
            if force:
                raise StillRejectedError("animated image too large to store as-is")
            return None
        orientation = src.getexif().get(_EXIF_ORIENTATION, 1)
        rotated = orientation not in (0, 1)
        has_alpha = src.mode in {"RGBA", "LA", "PA"} or (
            src.mode == "P" and "transparency" in src.info
        )
        if src.format in {"JPEG", "MPO"}:
            # Decode at the smallest DCT scale that still covers the bound (in
            # STORED orientation): a 48 MP original decodes at 1/2 or 1/4 size
            # instead of ~150 MB of RGB per full-size copy.
            want_w, want_h = round(width * STILL_HEADROOM), round(height * STILL_HEADROOM)
            if orientation in (5, 6, 7, 8):
                want_w, want_h = want_h, want_w
            src.draft(None, (want_w, want_h))
        image = ImageOps.exif_transpose(src) or src
        bounded = bound_to_frame(image, width, height)
        if bounded is image and not rotated and not force:
            return None
        mode = "RGBA" if has_alpha else "RGB"
        final = bounded if bounded.mode == mode else bounded.convert(mode)
        if max_bytes is None:
            return encode_still(final)
        encoded = encode_still_under(final, max_bytes)
        if encoded is None:
            raise StillRejectedError(
                f"fitted still never fit the {max_bytes:,} byte cap, even stepped down"
            )
        return encoded


def normalise_still(data: bytes, width: int, height: int) -> tuple[bytes, str] | None:
    """``data`` bounded to the frame (see :func:`bound_to_frame`) and re-encoded.

    ``None`` means "keep the original bytes": the still is already within the
    bound and needs no rotation, it is animated (re-encoding would keep frame 0
    only), or it cannot be decoded (the caller's own guards decide what an
    undecodable body means). EXIF orientation is baked into the pixels first — a
    later Pillow crop ignores the tag, so a portrait-tagged phone photo would
    otherwise be cropped on the wrong axis.
    """
    try:
        return _fit(data, width, height, force=False)
    except Exception as exc:  # noqa: BLE001 -- best effort: the original still stands
        log.info("still normalise skipped: %s: %s", type(exc).__name__, exc)
        return None


def fit_ingested_still(
    data: bytes, fmt: str, width: int, height: int, *, max_out_bytes: int
) -> tuple[bytes, str]:
    """An ingested image fitted to the film frame, guaranteed ``<= max_out_bytes``.

    The ingest path fetches images up to a LARGER raw bound than every other
    fetch (48 MP phone originals are 15-30 MB), so it must never store what it
    fetched before it is fitted: whatever this returns is what downstream stages
    re-fetch under the general cap. Raises :class:`StillRejectedError` (a
    ``ValueError``, so callers' soft-fail paths record it) for a decompression
    bomb (ours at :data:`MAX_STILL_PIXELS` or Pillow's), an undecodable body over
    ``max_out_bytes``, or a still that stays over it even stepped down. A small undecodable body is returned unchanged — the
    pre-existing behaviour for an ``image/*`` Pillow cannot read (e.g. SVG).
    """
    oversized = len(data) > max_out_bytes
    try:
        out = _fit(data, width, height, force=oversized, max_bytes=max_out_bytes)
    except StillRejectedError:
        raise
    except Image.DecompressionBombError as exc:
        # Pillow's own guard (~179M px) fires inside ``Image.open``, BEFORE
        # ``_fit``'s pixel check — a tiny bomb file must still be refused.
        raise StillRejectedError(f"decompression bomb refused (decode limit): {exc}") from exc
    except Exception as exc:
        if not oversized:
            return data, fmt
        raise StillRejectedError(
            f"undecodable image of {len(data):,} bytes: {type(exc).__name__}: {exc}"
        ) from exc
    return (data, fmt) if out is None else out
