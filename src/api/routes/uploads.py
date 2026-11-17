"""Uploads — POST /api/uploads/{manuscript,thumbnail,document,package,voiceover,storyboard-asset}.

The audiobook wizard uploads the source manuscript *before* the project is
created, so this is tenant-scoped (not project-scoped). The file is staged via
``manuscript_storage.store_manuscript`` and the opaque storage ``ref`` is
returned for the FE to carry into ``brief.manuscript_ref``; ``audiobook_ingest``
reads it back by that ref (by key — no SSRF surface).

The thumbnail upload stages an operator-chosen course cover (a captured video
frame or an uploaded image) and returns a fetchable artifact URL the FE then
PATCHes onto ``course_stitch`` via ``runs.stages.thumbnail.patch``.

The voiceover upload stages a per-clip narration replacement (Studio direct
editor): the audio is magic-byte validated, ffprobed for its true duration,
loudness-normalized to the same -18 LUFS the TTS path masters to, and stored in
an owned bucket; the FE carries the returned URL + duration onto a ``ClipEdit``
(``audio_url`` / ``audio_seconds``), which replaces that shot's TTS narration and
declares that shot's length.

The storyboard-asset upload stages a reference image for a project's Asset
library (Character / Location / Prop): the image is magic-byte validated and
stored in an owned bucket, and the returned URL is what the caller appends to
that asset's ``reference_images``. Without it there is no way for a client to
produce a value for that field.
"""

from __future__ import annotations

import io
import logging
import zipfile
import zlib
from collections.abc import Callable
from uuid import uuid7

from fastapi import APIRouter, HTTPException, UploadFile, status
from PIL import Image
from pydantic import BaseModel

from src.api.routes._common import TenantIdDep
from src.services.audio_premaster import premaster, probe_seconds
from src.services.document_storage import store_document
from src.services.manuscript_storage import store_manuscript
from src.services.package_refs import make_package_ref
from src.services.tts_storage import (
    store_package,
    store_storyboard_asset_image,
    store_thumbnail_override,
    store_voiceover_override,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/uploads", tags=["uploads"])

_ALLOWED_EXTENSIONS = {"txt", "docx", "epub"}
_MAX_BYTES = 25 * 1024 * 1024  # 25 MiB — generous for a book-length manuscript

# Source documents (multi-source briefs): a superset of the manuscript types
# plus PDF/markdown — anything source_ingest's in-process parser handles.
_DOCUMENT_EXTENSIONS = {"pdf", "txt", "docx", "epub", "md", "markdown"}
_DOCUMENT_MAX_BYTES = 25 * 1024 * 1024  # 25 MiB

# Recording packages: a zipped Playwright-BDD recording bundle (features + steps +
# assets + the shared recording lib). Larger than a manuscript — it can carry image
# assets and fixtures — so a more generous cap.
_PACKAGE_MAX_BYTES = 100 * 1024 * 1024  # 100 MiB

# Image thumbnails: small (a single frame / cover), one of three web formats,
# validated by magic bytes (the extension alone is not trusted).
_THUMBNAIL_MAX_BYTES = 8 * 1024 * 1024
_THUMBNAIL_MAGIC: dict[str, tuple[bytes, ...]] = {
    "png": (b"\x89PNG\r\n\x1a\n",),
    "jpg": (b"\xff\xd8\xff",),
    "jpeg": (b"\xff\xd8\xff",),
    "webp": (b"RIFF",),  # + "WEBP" at offset 8, checked below
}


# Per-clip voiceover replacements: a single narration take for ONE shot. Same
# 25 MiB cap as a manuscript (a 25 MiB wav is ~2.5h of speech — far beyond any
# single shot), validated by magic bytes (the extension alone is not trusted) and
# then ffprobed for its true duration.
_VOICEOVER_MAX_BYTES = 25 * 1024 * 1024
# The duration probe itself (`audio_premaster.probe_seconds`, shared with the TTS
# timing guard) owns its own timeout; it used to live here as
# `_probe_audio_seconds` + `_VOICEOVER_PROBE_TIMEOUT_S`.


def _is_mpeg_audio(data: bytes) -> bool:
    """MP3: an ID3v2 tag, or a bare MPEG frame sync (11 set bits)."""
    if data.startswith(b"ID3"):
        return True
    return len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0


def _is_aac(data: bytes) -> bool:
    """AAC: an ADTS frame sync (0xFFFx), an ADIF header, or an MP4 ``ftyp`` box
    (an ``.aac`` that is really an MP4-wrapped stream)."""
    if data.startswith(b"ADIF") or data[4:8] == b"ftyp":
        return True
    return len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xF0) == 0xF0


# Storyboard-asset reference images (Character / Location / Prop libraries): a
# handful of stills per asset that later panel renders condition on. A reference
# image is a photo/render, not a print master — 10 MiB comfortably holds a 4K PNG
# while keeping a mistaken "upload the source PSD" out of the bucket.
_ASSET_IMAGE_MAX_BYTES = 10 * 1024 * 1024


def _is_webp(data: bytes) -> bool:
    """WebP: a RIFF container whose fourcc at offset 8 is ``WEBP``.

    Both halves are required — ``RIFF`` alone is also WAV and AVI.
    """
    return data.startswith(b"RIFF") and data[8:12] == b"WEBP"


_IMAGE_MAGIC: dict[str, Callable[[bytes], bool]] = {
    "png": lambda data: data.startswith(b"\x89PNG\r\n\x1a\n"),
    "jpg": lambda data: data.startswith(b"\xff\xd8\xff"),
    "jpeg": lambda data: data.startswith(b"\xff\xd8\xff"),
    "webp": _is_webp,
    "gif": lambda data: data.startswith((b"GIF87a", b"GIF89a")),
}


_VOICEOVER_MAGIC: dict[str, Callable[[bytes], bool]] = {
    "mp3": _is_mpeg_audio,
    # RIFF container + the WAVE fourcc at offset 8 (RIFF alone is also WEBP/AVI).
    "wav": lambda data: data.startswith(b"RIFF") and data[8:12] == b"WAVE",
    # ISO-BMFF: the ``ftyp`` box type sits at offset 4, after its size.
    "m4a": lambda data: data[4:8] == b"ftyp",
    "aac": _is_aac,
    "ogg": lambda data: data.startswith(b"OggS"),
    "flac": lambda data: data.startswith(b"fLaC"),
}


class ManuscriptUploadResponse(BaseModel):
    manuscript_ref: str
    manuscript_name: str
    bytes: int


class ThumbnailUploadResponse(BaseModel):
    thumbnail_url: str
    bytes: int


class DocumentUploadResponse(BaseModel):
    document_ref: str
    document_name: str
    bytes: int


class PackageUploadResponse(BaseModel):
    package_url: str
    # Immutable, tenant-bound reference (uploads://package/<id>@sha256:<hash>). The
    # ONLY package form a brief with a login may use — see services/package_refs.
    package_ref: str
    bytes: int


class VoiceoverUploadResponse(BaseModel):
    audio_url: str
    seconds: float


class StoryboardAssetImageUploadResponse(BaseModel):
    """The URL an operator drops onto ``StoryboardAsset.reference_images``.

    ``width``/``height`` are best-effort pixel dimensions read from the image
    HEADER (no full decode). They are ``None`` when the header is unreadable —
    the upload is never failed on a dimension read, because the bytes have
    already passed the magic-byte gate and the field is advisory (the FE uses it
    to lay the thumbnail out).
    """

    image_url: str
    bytes: int
    width: int | None = None
    height: int | None = None


def _ext(filename: str) -> str:
    _, _, ext = (filename or "").lower().rpartition(".")
    return ext


def _thumbnail_format(data: bytes, ext: str) -> str | None:
    """Return the normalized image format iff *data*'s magic bytes match *ext*.

    Guards against an extension that lies about the bytes. WEBP needs the
    ``WEBP`` fourcc at offset 8 in addition to the ``RIFF`` header."""
    magics = _THUMBNAIL_MAGIC.get(ext)
    if not magics or not any(data.startswith(m) for m in magics):
        return None
    if ext == "webp" and data[8:12] != b"WEBP":
        return None
    return "jpg" if ext == "jpeg" else ext


def _audio_format(data: bytes, ext: str) -> str | None:
    """Return the normalized audio format iff *data*'s magic bytes match *ext*.

    Same posture as ``_thumbnail_format``: the extension (and the multipart
    content-type header, which is client-controlled) is not trusted — the bytes
    must actually look like the claimed container. A mismatch is rejected before
    anything is written or probed."""
    check = _VOICEOVER_MAGIC.get(ext)
    if check is None or not check(data):
        return None
    return ext


def _image_format(data: bytes, ext: str) -> str | None:
    """Return the normalized image format iff *data*'s magic bytes match *ext*.

    Same posture as ``_audio_format``: neither the extension nor the multipart
    content-type header (both client-controlled) is trusted — the bytes must
    actually look like the claimed image. This is what stops a renamed script or
    ELF binary from being parked in the bucket behind a ``.png`` name and served
    back out through the artifact proxy.
    """
    check = _IMAGE_MAGIC.get(ext)
    if check is None or not check(data):
        return None
    return "jpg" if ext == "jpeg" else ext


def _image_dimensions(data: bytes) -> tuple[int | None, int | None]:
    """Best-effort ``(width, height)`` from the image header, ``(None, None)`` on failure.

    ``Image.open`` parses only the header — it does not decode the pixel data — so
    this is cheap and does not hand a hostile payload to a full decoder. Pillow is
    already a declared dependency of this service; no new one is introduced.

    Advisory only: any unreadable/hostile header degrades to ``None`` rather than
    failing an upload whose bytes already passed the magic-byte gate.
    ``DecompressionBombError`` is caught explicitly because Pillow raises it from
    the header parse (on an absurd declared size) and it does not derive from
    ``OSError``.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
    except OSError, ValueError, Image.DecompressionBombError:
        return None, None
    return width, height


@router.post(
    "/manuscript",
    response_model=ManuscriptUploadResponse,
    operation_id="uploads.manuscript",
    summary="Stage an uploaded audiobook manuscript (.txt/.docx/.epub).",
)
async def upload_manuscript(file: UploadFile, tenant_id: TenantIdDep) -> ManuscriptUploadResponse:
    filename = file.filename or "manuscript"
    ext = _ext(filename)
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported manuscript type {ext!r}; allowed: {sorted(_ALLOWED_EXTENSIONS)}",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty manuscript upload")
    if len(data) > _MAX_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"manuscript exceeds {_MAX_BYTES // (1024 * 1024)} MiB limit",
        )
    ref = await store_manuscript(str(tenant_id), filename, data)
    return ManuscriptUploadResponse(manuscript_ref=ref, manuscript_name=filename, bytes=len(data))


@router.post(
    "/thumbnail",
    response_model=ThumbnailUploadResponse,
    operation_id="uploads.thumbnail",
    summary="Stage an operator-chosen course thumbnail (.png/.jpg/.webp).",
)
async def upload_thumbnail(file: UploadFile, tenant_id: TenantIdDep) -> ThumbnailUploadResponse:
    ext = _ext(file.filename or "thumbnail.png")
    if ext not in _THUMBNAIL_MAGIC:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported thumbnail type {ext!r}; allowed: png, jpg, jpeg, webp",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty thumbnail upload")
    if len(data) > _THUMBNAIL_MAX_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"thumbnail exceeds {_THUMBNAIL_MAX_BYTES // (1024 * 1024)} MiB limit",
        )
    image_format = _thumbnail_format(data, ext)
    if image_format is None:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"thumbnail bytes do not match a {ext!r} image",
        )
    url = await store_thumbnail_override(str(tenant_id), str(uuid7()), data, image_format)
    return ThumbnailUploadResponse(thumbnail_url=url, bytes=len(data))


@router.post(
    "/document",
    response_model=DocumentUploadResponse,
    operation_id="uploads.document",
    summary="Stage a source document (.pdf/.txt/.docx/.epub/.md) for multi-source ingest.",
)
async def upload_document(file: UploadFile, tenant_id: TenantIdDep) -> DocumentUploadResponse:
    filename = file.filename or "document"
    ext = _ext(filename)
    if ext not in _DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported document type {ext!r}; allowed: {sorted(_DOCUMENT_EXTENSIONS)}",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty document upload")
    if len(data) > _DOCUMENT_MAX_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"document exceeds {_DOCUMENT_MAX_BYTES // (1024 * 1024)} MiB limit",
        )
    ref = await store_document(str(tenant_id), filename, data)
    return DocumentUploadResponse(document_ref=ref, document_name=filename, bytes=len(data))


@router.post(
    "/package",
    response_model=PackageUploadResponse,
    operation_id="uploads.package",
    summary="Stage a Playwright recording-package (.zip) for a showcase/demo_video brief.",
)
async def upload_package(file: UploadFile, tenant_id: TenantIdDep) -> PackageUploadResponse:
    """Stage a recording-package zip and return a presigned ``package_url``.

    The HTTP replacement for the in-pod ``host_package.py`` script: an authored
    recording bundle becomes a ``brief.package_url`` over plain multipart, so
    neither an agent nor the frontend needs pod access to drive a showcase run.
    """
    if _ext(file.filename or "package.zip") != "zip":
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "unsupported package type; expected a .zip recording package",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty package upload")
    if len(data) > _PACKAGE_MAX_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"package exceeds {_PACKAGE_MAX_BYTES // (1024 * 1024)} MiB limit",
        )
    # Validate it is a real, uncorrupted zip before it reaches the record stage
    # (which downloads + unpacks it live). Opening validates the container;
    # testzip() reads every member — returning the first bad member's name, or
    # raising on a broken deflate stream. Both mean "don't hand this to record".
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        bad_member = archive.testzip()
    except zipfile.BadZipFile, zlib.error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "upload is not a valid zip") from None
    if bad_member is not None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "corrupt zip: failed CRC check")
    upload_id = str(uuid7())
    url = await store_package(str(tenant_id), upload_id, data)
    return PackageUploadResponse(
        package_url=url, package_ref=make_package_ref(upload_id, data), bytes=len(data)
    )


@router.post(
    "/voiceover",
    response_model=VoiceoverUploadResponse,
    operation_id="uploads.voiceover",
    summary="Stage a per-clip voiceover replacing one shot's TTS narration.",
)
async def upload_voiceover(file: UploadFile, tenant_id: TenantIdDep) -> VoiceoverUploadResponse:
    """Stage an uploaded narration take and return its URL + probed duration.

    Tenant-scoped like the thumbnail upload (the upload precedes the clip-edit
    PATCH that references it). The returned ``audio_url`` goes onto a ``ClipEdit``
    as the shot's ``audio_url`` — it REPLACES that shot's TTS narration — and
    ``seconds`` becomes its ``audio_seconds``, which declares the shot's length
    (the compiler ffprobes the audio and freeze-pads the visual to it, so the FE
    must show the caller how far the clip will stretch).

    The duration is probed HERE, server-side: an unprobeable file is a 400 and is
    never stored, because the compiler resolves an unfetchable/undecodable
    ``audio_url`` to SILENCE without failing the render.

    The take is then LOUDNESS-NORMALIZED to -18 LUFS / -1.5 dBTP — the exact level
    ``tts_backend`` masters every synthesized clip to — before it is stored. The
    compiler applies no gain to ``shot.audio_url``, so storing an operator's raw
    take (a phone recording at -30 LUFS, a hot bounce at -8) next to pre-mastered
    TTS neighbours is a 10-20 dB level jump mid-video. See
    ``src.services.audio_premaster``; it fails open, so a premaster that can't run
    degrades to today's un-normalized behaviour and never fails the upload.

    **Probe ordering:** the ORIGINAL bytes are probed first (that probe is the
    decodability gate whose 400 must not change), premastering runs second, and
    the NORMALIZED bytes are re-probed third — because a re-encode can shift the
    duration by a frame (encoder priming/padding on mp3/aac) and ``seconds`` must
    describe the artifact that is actually stored, which is what the compiler
    freeze-pads the visual to. If the re-probe can't read the normalized output,
    that is treated as one more fail-open path: the ORIGINAL bytes and their
    probed duration are stored.
    """
    ext = _ext(file.filename or "voiceover.mp3")
    if ext not in _VOICEOVER_MAGIC:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported voiceover type {ext!r}; allowed: {sorted(_VOICEOVER_MAGIC)}",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty voiceover upload")
    if len(data) > _VOICEOVER_MAX_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"voiceover exceeds {_VOICEOVER_MAX_BYTES // (1024 * 1024)} MiB limit",
        )
    audio_format = _audio_format(data, ext)
    if audio_format is None:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"voiceover bytes do not match a {ext!r} audio file",
        )
    seconds = await probe_seconds(data, audio_format)
    if seconds is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "voiceover is not decodable audio")

    stored, _metrics = await premaster(data, audio_format)
    if stored is not data:
        normalized_seconds = await probe_seconds(stored, audio_format)
        if normalized_seconds is None:
            # The normalized bytes don't probe — ffmpeg produced something we
            # can't vouch for. Fall back to the original take rather than store an
            # artifact the compiler might resolve to SILENCE.
            logger.warning("premastered voiceover failed re-probe; storing the original take")
            stored = data
        else:
            seconds = normalized_seconds

    url = await store_voiceover_override(
        str(tenant_id), str(uuid7()), stored, audio_format, seconds=seconds
    )
    return VoiceoverUploadResponse(audio_url=url, seconds=seconds)


@router.post(
    "/storyboard-asset",
    response_model=StoryboardAssetImageUploadResponse,
    operation_id="uploads.storyboard_asset",
    summary="Stage a storyboard-asset reference image (.png/.jpg/.webp/.gif).",
)
async def upload_storyboard_asset(
    file: UploadFile, tenant_id: TenantIdDep
) -> StoryboardAssetImageUploadResponse:
    """Stage a reference image for a project's Asset library and return its URL.

    Tenant-scoped like the thumbnail and voiceover uploads: the upload precedes
    the ``POST /api/projects/{id}/storyboard/assets`` (or the PATCH) that
    references it, so it cannot be project-scoped. The returned ``image_url``
    is what the caller appends to the asset's ``reference_images`` — that list is
    a ``list[str]`` of fetchable URLs, and until this route existed there was no
    way for a client to produce one.

    The bytes are magic-byte validated (``_image_format``) before anything is
    written: the extension and the multipart content-type are both
    attacker-controlled, and this bucket is readable back through the artifact
    proxy, so a renamed script must never land in it.
    """
    ext = _ext(file.filename or "reference.png")
    if ext not in _IMAGE_MAGIC:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported reference image type {ext!r}; allowed: {sorted(_IMAGE_MAGIC)}",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty reference image upload")
    if len(data) > _ASSET_IMAGE_MAX_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"reference image exceeds {_ASSET_IMAGE_MAX_BYTES // (1024 * 1024)} MiB limit",
        )
    image_format = _image_format(data, ext)
    if image_format is None:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"reference image bytes do not match a {ext!r} image",
        )
    width, height = _image_dimensions(data)
    url = await store_storyboard_asset_image(str(tenant_id), str(uuid7()), data, image_format)
    return StoryboardAssetImageUploadResponse(
        image_url=url, bytes=len(data), width=width, height=height
    )
