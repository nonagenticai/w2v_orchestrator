"""A full-size pinned phone photo must animate, not ship as a frozen still.

Live run 01a1639e-c6dd-7191-b7ee-7c169493404a (generic, 2026-10-22) pinned one
4032x3024 Wikimedia JPEG (3.7 MB). ``crop_cover_stills_to_frame`` re-encoded it
as a LOSSLESS PNG at native 4032x2268, that PNG was over the 12 MiB cap of
``ken_burns._fetch_own_asset``, the local render could not re-fetch it, and the
shot shipped frozen with ``ken_burns_local_fallback_failed``.

The synthetic photo below has camera-like grain, so its sizes behave like a
real photo's: ~5 MB as JPEG, ~18 MB as a 16:9 PNG crop.
"""

from __future__ import annotations

import io
import random
import socket
import struct
import zlib
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from PIL import Image
from w2v_common.net import ResponseTooLargeError

from src.services import visual_media
from src.services.runs.run_context import RunContext
from src.services.stages.shared import ken_burns_local
from src.services.stages.shared.degradation import KEN_BURNS_LOCAL_FALLBACK_FAILED, Degradation
from src.services.still_fit import (
    STILL_HEADROOM,
    StillRejectedError,
    encode_still_under,
    fit_ingested_still,
    normalise_still,
)

_PHONE_W, _PHONE_H = 4032, 3024


def _phone_photo(width: int = _PHONE_W, height: int = _PHONE_H) -> Image.Image:
    rnd = random.Random(7)  # noqa: S311 -- deterministic test pixels, not crypto
    coarse = (width // 8, height // 8)
    base = Image.frombytes("RGB", coarse, rnd.randbytes(coarse[0] * coarse[1] * 3))
    base = base.resize((width, height), Image.Resampling.BICUBIC)
    grain = Image.effect_noise((width, height), 24).convert("RGB")
    return Image.blend(base, grain, 0.25)


def _jpeg(image: Image.Image, quality: int = 90, **kw: Any) -> bytes:
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=quality, **kw)
    return out.getvalue()


def _ctx(resolution: str) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"deliverable": {"format": "16x9"}, "resolution": resolution},
        prior_run_outputs={},
    )


@pytest.fixture
def _object_store(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """An in-memory MinIO whose read-back enforces the REAL ken_burns fetch cap."""
    objects: dict[str, bytes] = {}

    async def _fetch(url: str, _client: object) -> tuple[bytes, str]:
        data = objects[url]
        if len(data) > ken_burns_local.kb._MAX_IMAGE_BYTES:
            raise ResponseTooLargeError(
                f"{len(data)} bytes > {ken_burns_local.kb._MAX_IMAGE_BYTES}"
            )
        return data, "image/jpeg"

    async def _store_image(**kw: Any) -> str:
        url = f"https://minio.test/i/{kw['image_id']}.{kw['image_format']}"
        objects[url] = kw["image_bytes"]
        return url

    async def _store_video(**kw: Any) -> str:
        return f"https://minio.test/v/{kw['video_id']}.mp4"

    monkeypatch.setattr(ken_burns_local.kb, "_fetch_own_asset", _fetch)
    monkeypatch.setattr(ken_burns_local, "store_generated_image", _store_image)
    monkeypatch.setattr(ken_burns_local, "store_generated_video", _store_video)
    return objects


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resolution", "crop_size"),
    [("1080p", (3840, 2160))],
)
async def test_a_full_size_pinned_photo_gets_a_ken_burns_clip_not_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
    _object_store: dict[str, bytes],
    resolution: str,
    crop_size: tuple[int, int],
) -> None:
    """The live 01a1639e chain: cover-crop, then the local render's re-fetch."""
    pin_url = "https://minio.test/capture/pin.jpg"
    _object_store[pin_url] = _jpeg(_phone_photo())
    render = AsyncMock(return_value=b"\x00mp4")
    monkeypatch.setattr(ken_burns_local, "render_local_ken_burns", render)
    ctx = _ctx(resolution)
    shots: list[dict[str, Any]] = [
        {"id": "pin", "image_url": pin_url, "fit": "cover", "seconds": 4.0}
    ]
    notes: list[str] = []
    degradations: list[Degradation] = []

    cropped = await ken_burns_local.crop_cover_stills_to_frame(
        ctx, cast(Any, shots), notes=notes, degradations=degradations
    )
    out = await ken_burns_local.rescue_frozen_stills(
        ctx,
        cropped,
        notes=notes,
        advisories=[],
        budget_seconds=600.0,
        degradations=degradations,
        local_first=True,
    )

    assert out[0]["video_url"].startswith("https://minio.test/v/")
    assert [d["code"] for d in degradations] == []
    assert not any(KEN_BURNS_LOCAL_FALLBACK_FAILED in n or "failed" in n for n in notes)
    crop_bytes = _object_store[cropped[0]["image_url"]]
    with Image.open(io.BytesIO(crop_bytes)) as crop:
        assert crop.format == "JPEG"
        assert crop.size == crop_size
    assert len(crop_bytes) < ken_burns_local.kb._MAX_IMAGE_BYTES // 2
    (call,) = render.await_args_list
    assert call.args[0] == crop_bytes


def test_normalise_still_cover_fits_a_phone_photo_to_twice_the_frame() -> None:
    out = normalise_still(_jpeg(_phone_photo()), 1920, 1080)
    assert out is not None
    data, ext = out
    assert ext == "jpg"
    with Image.open(io.BytesIO(data)) as img:
        # Cover-fit: the long axis hits 2x frame width, the short one stays >= 2x height.
        assert img.size == (round(1920 * STILL_HEADROOM), 2880)


def test_normalise_still_leaves_a_frame_sized_still_alone() -> None:
    assert normalise_still(_jpeg(_phone_photo(1920, 1080)), 1920, 1080) is None
    assert normalise_still(b"not an image", 1920, 1080) is None


def test_normalise_still_bakes_in_exif_rotation() -> None:
    """A portrait-tagged phone photo is stored rotated, so a later crop cuts the right axis."""
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    data = _jpeg(_phone_photo(800, 600), exif=exif.tobytes())
    out = normalise_still(data, 1920, 1080)
    assert out is not None
    with Image.open(io.BytesIO(out[0])) as img:
        assert img.size == (600, 800)
        assert img.getexif().get(0x0112, 1) == 1


# ---------------------------------------------------------------------------
# Photos OVER the general 12 MiB cap (48 MP phone originals are 15-30 MB)
# ---------------------------------------------------------------------------
_STUB_HOSTS = {"cdn": "93.184.216.34"}


@pytest.fixture
def _stub_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    from w2v_common.net import safe_fetch

    def fake_getaddrinfo(host: str, port: int | None, *args: Any, **kwargs: Any) -> list[Any]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (_STUB_HOSTS[host], port or 0))]

    monkeypatch.setattr(safe_fetch.socket, "getaddrinfo", fake_getaddrinfo)


def _serving(body: bytes) -> httpx.AsyncClient:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "image/jpeg"}, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_a_pin_over_the_general_cap_is_fitted_stored_and_animated(
    monkeypatch: pytest.MonkeyPatch, _stub_dns: None, _object_store: dict[str, bytes]
) -> None:
    """A >12 MiB phone original is fetched under the INGEST bound, fitted before it
    is stored, and the stored still then animates under the general cap."""
    big = _jpeg(_phone_photo(6000, 4500), quality=97)
    assert len(big) > visual_media._MAX_IMAGE_BYTES
    stored: dict[str, Any] = {}

    async def _store(**kw: Any) -> str:
        stored.update(kw)
        url = f"https://minio.test/capture/{kw['image_id']}.{kw['image_format']}"
        _object_store[url] = kw["image_bytes"]
        return url

    monkeypatch.setattr(visual_media, "store_capture_image", _store)
    pin_url = await visual_media.republish_image(
        "run-1", "source_ingest", "https://cdn/phone.jpg", client=_serving(big), fit_to=(1920, 1080)
    )
    assert stored["image_format"] == "jpg"
    assert len(stored["image_bytes"]) <= visual_media._MAX_IMAGE_BYTES
    with Image.open(io.BytesIO(stored["image_bytes"])) as img:
        assert img.size == (3840, 2880)

    render = AsyncMock(return_value=b"\x00mp4")
    monkeypatch.setattr(ken_burns_local, "render_local_ken_burns", render)
    ctx = _ctx("1080p")
    shots = cast(Any, [{"id": "pin", "image_url": pin_url, "fit": "cover", "seconds": 4.0}])
    degradations: list[Degradation] = []
    cropped = await ken_burns_local.crop_cover_stills_to_frame(
        ctx, shots, degradations=degradations
    )
    out = await ken_burns_local.rescue_frozen_stills(
        ctx, cropped, budget_seconds=600.0, degradations=degradations, local_first=True
    )
    assert out[0]["video_url"].startswith("https://minio.test/v/")
    assert degradations == []


@pytest.mark.asyncio
async def test_the_general_cap_still_applies_without_fit_to(_stub_dns: None) -> None:
    big = b"\xff\xd8\xff" + b"\x00" * (visual_media._MAX_IMAGE_BYTES + 1)
    with pytest.raises(ResponseTooLargeError):
        await visual_media.republish_image("run-1", "x", "https://cdn/p.jpg", client=_serving(big))


@pytest.mark.asyncio
async def test_a_pin_over_the_ingest_bound_is_refused_with_a_note_naming_it(
    monkeypatch: pytest.MonkeyPatch, _stub_dns: None
) -> None:
    """Over the ingest bound: never silently dropped — the run records WHY, by ref."""
    from src.services.stages.generic.source_ingest import SourceIngestStage

    big = b"\xff\xd8\xff" + b"\x00" * (visual_media._MAX_INGEST_IMAGE_BYTES + 1)
    real = visual_media.republish_image

    async def _republish(*args: Any, **kw: Any) -> str:
        return await real(*args, client=_serving(big), **kw)

    monkeypatch.setattr("src.services.stages.generic.source_ingest.republish_image", _republish)
    monkeypatch.setattr(
        "src.services.stages.generic.source_ingest.republish_video",
        AsyncMock(
            side_effect=visual_media.NotMediaContentError("u", "image/jpeg", expected="video")
        ),
    )
    ref = "https://cdn/huge.jpg"
    out = await SourceIngestStage(client=AsyncMock()).run(
        {},
        ctx=RunContext(
            run_id=uuid4(),
            deliverable_id=uuid4(),
            config_snapshot={"brief": {"sources": [{"kind": "media", "ref": ref}]}},
            prior_run_outputs={},
        ),
    )
    (pin,) = out.pinned_media
    assert pin["composite_url"] == "" and pin["composite_status"] == "failed"
    (note,) = [n for n in out.notes if ref in n]
    assert "ResponseTooLargeError" in note


def _png_header_only(width: int, height: int) -> bytes:
    """A PNG that DECLARES ``width x height`` — the bomb shape: tiny file, huge canvas."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


# Pillow emits DecompressionBombWarning (a RuntimeWarning) from Image.open for
# anything over MAX_IMAGE_PIXELS (~89M px), BEFORE our own 100M px check runs.
# It is only a warning in production, but the pre-push hook runs pytest with
# `-W error::RuntimeWarning`, which turned dims0 into an exception and failed
# the hook on macOS while CI (no -W error) stayed green. Ignore that one
# warning here: what this test pins is OUR refusal, not Pillow's warning.
@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
@pytest.mark.parametrize(
    "dims",
    [(12_000, 9_000)],
)
def test_a_decompression_bomb_is_refused_before_decode(dims: tuple[int, int]) -> None:
    bomb = _png_header_only(*dims)
    assert len(bomb) < 100
    with pytest.raises(StillRejectedError, match="decode limit"):
        fit_ingested_still(bomb, "png", 1920, 1080, max_out_bytes=12 * 1024 * 1024)


def test_an_mpo_phone_photo_is_fitted_and_rotated_not_treated_as_animated() -> None:
    """iPhone/Samsung portrait + gain-map JPEGs are MPO: several frames, one photo."""
    exif = Image.Exif()
    exif[0x0112] = 6
    out = io.BytesIO()
    _phone_photo().save(
        out, format="MPO", save_all=True, append_images=[_phone_photo(400, 300)], exif=exif
    )
    with Image.open(io.BytesIO(out.getvalue())) as probe:
        assert probe.format == "MPO" and getattr(probe, "n_frames", 1) == 2
    fitted = normalise_still(out.getvalue(), 1920, 1080)
    assert fitted is not None
    with Image.open(io.BytesIO(fitted[0])) as img:
        assert img.size[1] > img.size[0]  # portrait after the EXIF rotation
        assert img.size == (3024, 4032)  # rotated; already within the cover bound


def test_an_over_cap_encode_steps_down_instead_of_failing() -> None:
    image = _phone_photo()
    first = _jpeg(image)
    cap = len(first) // 3
    encoded = encode_still_under(image, cap)
    assert encoded is not None
    assert encoded[1] == "jpg" and len(encoded[0]) <= cap


def test_an_undecodable_body_over_the_cap_is_refused() -> None:
    with pytest.raises(StillRejectedError, match="undecodable"):
        fit_ingested_still(b"\x00" * 200, "jpg", 1920, 1080, max_out_bytes=100)
    # Under the cap it is stored as fetched, exactly as before (e.g. an SVG).
    assert fit_ingested_still(b"<svg/>", "svg", 1920, 1080, max_out_bytes=100) == (b"<svg/>", "svg")


def test_every_cap_downstream_of_ingest_is_the_same_cap() -> None:
    assert visual_media._MAX_IMAGE_BYTES == ken_burns_local.kb._MAX_IMAGE_BYTES
    assert visual_media._MAX_INGEST_IMAGE_BYTES > visual_media._MAX_IMAGE_BYTES
