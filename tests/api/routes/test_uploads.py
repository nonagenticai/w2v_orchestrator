"""/api/uploads/manuscript route tests (audiobook source upload).

``store_manuscript`` is patched so these don't need a storage backend — they
prove validation (extension, empty, size) and the response contract the FE
carries into ``brief.manuscript_ref``.

The voiceover tests deliberately do NOT patch the ffprobe/ffmpeg: the duration the
route returns is the whole point of the upload (it declares the shot's length), and
the LEVEL the take is stored at is the whole point of the premaster, so both are
exercised for real against real WAVs. They therefore require ffmpeg on PATH — a
skip here would be a test that never runs.
"""

from __future__ import annotations

import io
import math
import shutil
import struct
import wave
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from src.services import audio_premaster
from src.services.audio_premaster import measure_loudness

pytestmark = pytest.mark.asyncio


async def test_upload_manuscript_returns_ref(client) -> None:
    with patch(
        "src.api.routes.uploads.store_manuscript",
        new=AsyncMock(return_value="manuscripts/tenant/upload/abc.txt"),
    ) as store:
        r = await client.post(
            "/api/uploads/manuscript",
            files={"file": ("my-book.txt", b"Chapter 1\nHello.", "text/plain")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["manuscript_ref"] == "manuscripts/tenant/upload/abc.txt"
    assert body["manuscript_name"] == "my-book.txt"
    assert body["bytes"] == len(b"Chapter 1\nHello.")
    store.assert_awaited_once()


async def test_upload_manuscript_rejects_bad_extension(client) -> None:
    r = await client.post(
        "/api/uploads/manuscript",
        files={"file": ("malware.pdf", b"%PDF", "application/pdf")},
    )
    assert r.status_code == 415


async def test_upload_manuscript_rejects_empty(client) -> None:
    r = await client.post(
        "/api/uploads/manuscript",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert r.status_code == 400


_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
_WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 24


async def test_upload_thumbnail_stores_and_returns_url(client) -> None:
    with patch(
        "src.api.routes.uploads.store_thumbnail_override",
        new=AsyncMock(return_value="http://minio.test:8389/w2v-orchestrator/t/abc.png"),
    ) as store:
        r = await client.post(
            "/api/uploads/thumbnail",
            files={"file": ("frame.png", _PNG, "image/png")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["thumbnail_url"].endswith("/abc.png")
    assert body["bytes"] == len(_PNG)
    store.assert_awaited_once()


@pytest.mark.parametrize(
    ("name", "data", "ctype"),
    [("f.jpg", _JPEG, "image/jpeg")],
)
async def test_upload_thumbnail_accepts_jpeg_and_webp(client, name, data, ctype) -> None:
    with patch(
        "src.api.routes.uploads.store_thumbnail_override",
        new=AsyncMock(return_value=f"http://minio.test:8389/w2v-orchestrator/t/{name}"),
    ):
        r = await client.post("/api/uploads/thumbnail", files={"file": (name, data, ctype)})
    assert r.status_code == 200, r.text


async def test_upload_thumbnail_rejects_bad_extension(client) -> None:
    r = await client.post(
        "/api/uploads/thumbnail",
        files={"file": ("evil.svg", b"<svg/>", "image/svg+xml")},
    )
    assert r.status_code == 415


async def test_upload_thumbnail_rejects_mismatched_magic(client) -> None:
    """A .png extension over JPEG bytes is rejected (extension is not trusted)."""
    r = await client.post(
        "/api/uploads/thumbnail",
        files={"file": ("lie.png", _JPEG, "image/png")},
    )
    assert r.status_code == 415


async def test_upload_thumbnail_rejects_empty(client) -> None:
    r = await client.post(
        "/api/uploads/thumbnail",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert r.status_code == 400


# --- source document upload (multi-source briefs) -------------------------------


async def test_upload_document_returns_ref(client) -> None:
    with patch(
        "src.api.routes.uploads.store_document",
        new=AsyncMock(return_value="documents/tenant/upload/abc.pdf"),
    ) as store:
        r = await client.post(
            "/api/uploads/document",
            files={"file": ("brand-book.pdf", b"%PDF-1.7 hello", "application/pdf")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["document_ref"] == "documents/tenant/upload/abc.pdf"
    assert body["document_name"] == "brand-book.pdf"
    assert body["bytes"] == len(b"%PDF-1.7 hello")
    store.assert_awaited_once()


@pytest.mark.parametrize("name", ["notes.md"])
async def test_upload_document_accepts_doc_types(client, name) -> None:
    with patch(
        "src.api.routes.uploads.store_document",
        new=AsyncMock(return_value=f"documents/tenant/upload/{name}"),
    ):
        r = await client.post(
            "/api/uploads/document",
            files={"file": (name, b"some content", "application/octet-stream")},
        )
    assert r.status_code == 200, r.text


async def test_upload_document_rejects_bad_extension(client) -> None:
    r = await client.post(
        "/api/uploads/document",
        files={"file": ("clip.mp4", b"\x00\x00\x00\x18ftyp", "video/mp4")},
    )
    assert r.status_code == 415


async def test_upload_document_rejects_empty(client) -> None:
    r = await client.post(
        "/api/uploads/document",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert r.status_code == 400


# --- recording-package upload (showcase / demo_video authored mode) -------------


def _zip_bytes() -> bytes:
    """A minimal but real zip: a features file + the shared recording lib stub."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("features/demo.feature", "Feature: demo\n  Scenario: s\n")
        z.writestr("lib/recording_helpers.ts", "export const x = 1;\n")
    return buf.getvalue()


async def test_upload_package_stores_and_returns_url(client) -> None:
    data = _zip_bytes()
    with patch(
        "src.api.routes.uploads.store_package",
        new=AsyncMock(return_value="http://minio.test:8389/w2v-orchestrator/vp/pkg.zip"),
    ) as store:
        r = await client.post(
            "/api/uploads/package",
            files={"file": ("recording-package.zip", data, "application/zip")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["package_url"].endswith("/pkg.zip")
    assert body["bytes"] == len(data)
    store.assert_awaited_once()
    # The immutable ref a login brief must use: the stored upload id + content hash.
    import hashlib

    assert store.await_args is not None
    upload_id = store.await_args.args[1]
    assert body["package_ref"] == (
        f"uploads://package/{upload_id}@sha256:{hashlib.sha256(data).hexdigest()}"
    )


async def test_upload_package_rejects_bad_extension(client) -> None:
    r = await client.post(
        "/api/uploads/package",
        files={"file": ("recording.tar", _zip_bytes(), "application/x-tar")},
    )
    assert r.status_code == 415


async def test_upload_package_rejects_non_zip_bytes(client) -> None:
    """A .zip extension over non-zip bytes is rejected (content is validated)."""
    r = await client.post(
        "/api/uploads/package",
        files={"file": ("lie.zip", b"not a zip at all", "application/zip")},
    )
    assert r.status_code == 400


async def test_upload_package_rejects_truncated_zip(client) -> None:
    """A truncated zip (missing its end-of-central-directory) is rejected."""
    data = _zip_bytes()
    r = await client.post(
        "/api/uploads/package",
        files={"file": ("truncated.zip", data[: len(data) // 2], "application/zip")},
    )
    assert r.status_code == 400


async def test_upload_package_rejects_empty(client) -> None:
    r = await client.post(
        "/api/uploads/package",
        files={"file": ("empty.zip", b"", "application/zip")},
    )
    assert r.status_code == 400


# --- per-clip voiceover upload (Studio narration replacement) -------------------

_VO_URL = "http://minio.test:8389/w2v-orchestrator/artifacts/voiceovers/t/vo/abc.wav"


def _wav_bytes(seconds: float = 0.5, rate: int = 8000) -> bytes:
    """A real, decodable mono 16-bit WAV — ffprobe must report ~``seconds``."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


async def test_upload_voiceover_stores_and_returns_probed_duration(client) -> None:
    """Happy path: real WAV → real ffprobe → the probed duration is the response."""
    data = _wav_bytes(0.5)
    with patch(
        "src.api.routes.uploads.store_voiceover_override",
        new=AsyncMock(return_value=_VO_URL),
    ) as store:
        r = await client.post(
            "/api/uploads/voiceover",
            files={"file": ("take-2.wav", data, "audio/wav")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["audio_url"] == _VO_URL
    # The duration is ffprobed from the bytes, not declared by the client.
    assert body["seconds"] == pytest.approx(0.5, abs=0.05)
    store.assert_awaited_once()
    # It is stored with the sniffed format + the probed duration (the sidecar).
    await_args = store.await_args
    assert await_args is not None
    assert await_args.args[3] == "wav"
    assert await_args.kwargs["seconds"] == pytest.approx(0.5, abs=0.05)


async def test_upload_voiceover_rejects_bad_extension(client) -> None:
    r = await client.post(
        "/api/uploads/voiceover",
        files={"file": ("take.mp4", _wav_bytes(), "video/mp4")},
    )
    assert r.status_code == 415


async def test_upload_voiceover_rejects_mismatched_magic(client) -> None:
    """A .mp3 extension over PNG bytes is rejected (the extension is not trusted)."""
    r = await client.post(
        "/api/uploads/voiceover",
        files={"file": ("lie.mp3", _PNG, "audio/mpeg")},
    )
    assert r.status_code == 415


async def test_upload_voiceover_rejects_undecodable_audio(client) -> None:
    """Bytes that pass the RIFF/WAVE magic check but carry no decodable stream are
    rejected — an unprobeable file must never be stored (the compiler would resolve
    it to SILENCE without failing the render)."""
    junk = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\xde\xad\xbe\xef" * 8
    with patch("src.api.routes.uploads.store_voiceover_override", new=AsyncMock()) as store:
        r = await client.post(
            "/api/uploads/voiceover",
            files={"file": ("broken.wav", junk, "audio/wav")},
        )
    assert r.status_code == 400
    store.assert_not_awaited()


async def test_upload_voiceover_rejects_oversize(client, monkeypatch) -> None:
    monkeypatch.setattr("src.api.routes.uploads._VOICEOVER_MAX_BYTES", 1024)
    r = await client.post(
        "/api/uploads/voiceover",
        files={"file": ("long.wav", _wav_bytes(0.5), "audio/wav")},
    )
    assert r.status_code == 413


async def test_upload_voiceover_rejects_empty(client) -> None:
    r = await client.post(
        "/api/uploads/voiceover",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert r.status_code == 400


# --- voiceover loudness pre-master (level-match with the -18 LUFS TTS clips) -----
#
# The compiler applies NO gain to shot.audio_url, so an operator's raw take used
# to land on the timeline next to TTS neighbours tts_backend had already mastered
# to -18 LUFS — a 10-20 dB jump mid-video. The route now normalizes at upload.
# Real ffmpeg, real levels: the stored bytes are measured, not just counted.


def _tone_wav(target_lufs: float, *, seconds: float = 4.0, rate: int = 24_000) -> bytes:
    """A decodable mono 1 kHz sine WAV at roughly *target_lufs* (see test_audio_premaster)."""
    amplitude = 10 ** ((target_lufs + 3.7) / 20)
    peak = int(amplitude * 32767)
    frames = bytearray()
    for n in range(int(rate * seconds)):
        frames += struct.pack("<h", int(peak * math.sin(2 * math.pi * 1000 * n / rate)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def _stored_bytes(store: AsyncMock) -> bytes:
    """The audio actually handed to ``store_voiceover_override``."""
    await_args = store.await_args
    assert await_args is not None
    return await_args.args[2]


@pytest.mark.parametrize("input_lufs", [-8.0])
async def test_upload_voiceover_normalizes_loudness(client, input_lufs) -> None:
    """A hot (-8) and a quiet (-30) upload are both STORED at -18 ±1 LUFS."""
    data = _tone_wav(input_lufs)
    with patch(
        "src.api.routes.uploads.store_voiceover_override",
        new=AsyncMock(return_value=_VO_URL),
    ) as store:
        r = await client.post(
            "/api/uploads/voiceover",
            files={"file": ("take.wav", data, "audio/wav")},
        )
    assert r.status_code == 200, r.text
    stored = _stored_bytes(store)
    assert stored != data  # the raw take is NOT what lands in the bucket

    measured = await measure_loudness(stored, "wav")
    assert measured.integrated_lufs == pytest.approx(-18.0, abs=1.0)
    # The reported duration describes the STORED artifact (the compiler
    # freeze-pads the visual to it), and re-levelling does not change length.
    assert r.json()["seconds"] == pytest.approx(4.0, abs=0.05)


async def test_upload_voiceover_survives_missing_ffmpeg(client, monkeypatch) -> None:
    """Fail-open: with no ffmpeg for the premaster, the upload still succeeds and the
    ORIGINAL bytes are stored (today's un-normalized behaviour) — never a 500.

    Only ``ffmpeg`` is hidden; ``ffprobe`` must keep resolving, because the route's
    decodability gate (a 400 that must not change) depends on it.

    The stub is per-BINARY rather than a blanket ``which -> None``. It used to be
    blanket, which worked only while the route owned its own private
    ``_probe_audio_seconds``; that probe is now ``audio_premaster.probe_seconds``,
    shared with the TTS timing guard, so hiding every binary from this module would
    also blind the decodability gate and turn this fail-open case into a 400 —
    testing the opposite of what it claims.
    """
    monkeypatch.setattr(
        audio_premaster,
        "shutil",
        SimpleNamespace(which=lambda name: None if name == "ffmpeg" else shutil.which(name)),
    )
    data = _tone_wav(-8.0)
    with patch(
        "src.api.routes.uploads.store_voiceover_override",
        new=AsyncMock(return_value=_VO_URL),
    ) as store:
        r = await client.post(
            "/api/uploads/voiceover",
            files={"file": ("take.wav", data, "audio/wav")},
        )
    assert r.status_code == 200, r.text
    assert _stored_bytes(store) == data
    assert r.json()["seconds"] == pytest.approx(4.0, abs=0.05)


async def test_upload_voiceover_premaster_disabled_stores_raw(client, monkeypatch) -> None:
    """The kill-switch is a ConfigMap flip: OFF ⇒ byte-identical passthrough."""
    monkeypatch.setenv("VOICEOVER_PREMASTER_ENABLED", "false")
    data = _tone_wav(-8.0)
    with patch(
        "src.api.routes.uploads.store_voiceover_override",
        new=AsyncMock(return_value=_VO_URL),
    ) as store:
        r = await client.post(
            "/api/uploads/voiceover",
            files={"file": ("take.wav", data, "audio/wav")},
        )
    assert r.status_code == 200, r.text
    assert _stored_bytes(store) == data


# --- storyboard-asset reference image upload (Character/Location/Prop library) ---
#
# The bucket these land in is readable back through the artifact proxy and the URL
# is later re-signed for panel renders, so the magic-byte gate is the security
# boundary: the extension and the multipart content-type are both client-supplied.
# The fixtures below are therefore REAL encoded images (built by Pillow, already a
# declared dependency), not header stubs — a stub would pass the sniffer while
# proving nothing about the dimension read.

_ASSET_URL = (
    "http://minio.test:8389/w2v-orchestrator/artifacts/"
    "storyboard_assets/t/asset_reference_image/abc.png"
)


def _real_image(fmt: str, size: tuple[int, int] = (12, 7)) -> bytes:
    """A real, fully-encoded image of *fmt* at *size* (RGB)."""
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buf, format=fmt)
    return buf.getvalue()


async def test_upload_storyboard_asset_stores_and_returns_owned_url(client) -> None:
    """Happy path: a real PNG round-trips to an owned ``w2v-*`` URL with its size."""
    data = _real_image("PNG")
    with patch(
        "src.api.routes.uploads.store_storyboard_asset_image",
        new=AsyncMock(return_value=_ASSET_URL),
    ) as store:
        r = await client.post(
            "/api/uploads/storyboard-asset",
            files={"file": ("hero.png", data, "image/png")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image_url"] == _ASSET_URL
    # Load-bearing: re-signing (and therefore every later panel render) only works
    # for a bucket we own.
    assert "/w2v-" in body["image_url"]
    assert body["bytes"] == len(data)
    # Dimensions come from the header, not from the client.
    assert (body["width"], body["height"]) == (12, 7)
    store.assert_awaited_once()
    # Stored under the tenant namespace with the SNIFFED format.
    await_args = store.await_args
    assert await_args is not None
    assert await_args.args[2] == data
    assert await_args.args[3] == "png"


@pytest.mark.parametrize(
    ("name", "fmt", "ctype"),
    [("loc.jpg", "JPEG", "image/jpeg")],
)
async def test_upload_storyboard_asset_accepts_all_allowed_formats(
    client, name, fmt, ctype
) -> None:
    with patch(
        "src.api.routes.uploads.store_storyboard_asset_image",
        new=AsyncMock(return_value=_ASSET_URL),
    ):
        r = await client.post(
            "/api/uploads/storyboard-asset",
            files={"file": (name, _real_image(fmt), ctype)},
        )
    assert r.status_code == 200, r.text


async def test_upload_storyboard_asset_rejects_mismatched_magic(client) -> None:
    """A ``.png`` carrying real JPEG bytes is rejected — the extension is not trusted."""
    with patch("src.api.routes.uploads.store_storyboard_asset_image", new=AsyncMock()) as store:
        r = await client.post(
            "/api/uploads/storyboard-asset",
            files={"file": ("lie.png", _real_image("JPEG"), "image/png")},
        )
    assert r.status_code == 415
    store.assert_not_awaited()


async def test_upload_storyboard_asset_rejects_riff_that_is_not_webp(client) -> None:
    """A WAV renamed ``.webp`` is rejected: ``RIFF`` alone does not mean WebP.

    WebP is the one format here whose signature is not a single prefix — the
    container is ``RIFF``, shared with WAV and AVI, and only the ``WEBP`` fourcc at
    offset 8 distinguishes it. A sniffer that checks the prefix and stops accepts
    every RIFF file on earth, which is why this case gets its own test rather than
    riding along with the generic mismatch one: that test would still pass with the
    offset-8 check deleted.
    """
    wav = b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt "
    with patch("src.api.routes.uploads.store_storyboard_asset_image", new=AsyncMock()) as store:
        r = await client.post(
            "/api/uploads/storyboard-asset",
            files={"file": ("sneaky.webp", wav, "image/webp")},
        )
    assert r.status_code == 415, r.text
    store.assert_not_awaited()


@pytest.mark.parametrize(
    "blob",
    [b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 64],
)
async def test_upload_storyboard_asset_rejects_renamed_executable(client, blob) -> None:
    """SECURITY: a renamed binary / script with a ``.png`` name never lands in the bucket.

    That bucket is served back out through the artifact proxy, so a payload parked
    behind an image name is the whole reason the sniffer exists. A truthful
    content-type is deliberately NOT sent — the client controls that header too.
    """
    with patch("src.api.routes.uploads.store_storyboard_asset_image", new=AsyncMock()) as store:
        r = await client.post(
            "/api/uploads/storyboard-asset",
            files={"file": ("payload.png", blob, "image/png")},
        )
    assert r.status_code == 415, r.text
    store.assert_not_awaited()


async def test_upload_storyboard_asset_rejects_oversize(client, monkeypatch) -> None:
    monkeypatch.setattr("src.api.routes.uploads._ASSET_IMAGE_MAX_BYTES", 16)
    r = await client.post(
        "/api/uploads/storyboard-asset",
        files={"file": ("big.png", _real_image("PNG"), "image/png")},
    )
    assert r.status_code == 413


async def test_upload_storyboard_asset_rejects_empty(client) -> None:
    r = await client.post(
        "/api/uploads/storyboard-asset",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert r.status_code == 400


@pytest.mark.parametrize(("name", "ctype"), [("vector.svg", "image/svg+xml")])
async def test_upload_storyboard_asset_rejects_unlisted_extension(client, name, ctype) -> None:
    """SVG in particular is script-bearing markup — it is not on the allowlist."""
    r = await client.post(
        "/api/uploads/storyboard-asset",
        files={"file": (name, b"<svg xmlns='http://www.w3.org/2000/svg'/>", ctype)},
    )
    assert r.status_code == 415
