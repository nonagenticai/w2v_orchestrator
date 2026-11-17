"""Video-source uploads — project-scoped ingest of raw camera/drone footage.

The ``action_reel`` product starts from footage the operator shot: a handful of
4K clips, 60-150 MiB each, that an LLM later cuts to music. Each clip is
registered as a ``video_sources`` row and its bytes are parked in the own bucket
under a tenant+project confined key; a later stage fills in the preview /poster/
sprite derivatives (the row is born ``preview_status = 'pending'``).

The ``probe`` blob, by contrast, is filled HERE and nowhere else. It is the only
point in the fleet that still holds the clip as a local file, so ffprobing costs
one subprocess against a path we already have — whereas any later reader would
have to pull a 134 MiB original back out of MinIO into a pod sized for JSON,
which is exactly what this module exists to avoid. ``footage_probe`` (the stage
that flattens these rows into the DAG) is written against that contract and
re-probes nothing: an empty blob there means zeroed duration/fps and a
"no probed duration" note on every clip, so leaving it unwritten is not a
deferral, it is a permanent loss of the measurement.

**WHY THIS IS NOT IN ``uploads.py``.** Every handler in that module does
``data = await file.read()`` — the whole upload into a single ``bytes`` — and
then length-checks it. That pattern is correct for a 25 MiB manuscript and fatal
here: a 134 MiB HEVC clip would be materialised in the worker's heap in full,
and ten concurrent uploads (the normal shape of "drag the shoot into the
browser") would be over a gigabyte of resident memory in a pod sized for JSON.
Worse, the cap is only observable AFTER the bytes are already in RAM, so the
413 does not actually protect anything. This module therefore streams: the body
is read in fixed 1 MiB chunks straight to a temp file, the running total is
checked per chunk so an over-cap upload is abandoned mid-flight, and the temp
file is handed to ``StorageService.put_file`` which streams it into MinIO via
``fput_object`` (chunked multipart). Peak memory is one chunk, whatever the
clip's size. Nothing in here may grow an unbounded ``read()``.

**WHY THERE IS A SECOND, CHUNKED SURFACE.** Streaming fixes this pod's memory;
it does not fix the EDGE. Every request reaches the app through Cloudflare, and
Cloudflare refuses a body over 100 MB with its own 413 — the request never
arrives, so no amount of care in this module can see it. That was measured
against the real public host rather than inferred from the docs: PUT bodies of
32, 48, 64 and 96 MiB all reached the app (they got the app's own 401), and
100 MB+ got a 413 stamped by the edge. A 134 MiB 4K drone clip therefore cannot
be uploaded in ONE request at all, and 4 of the 10 clips in the demo corpus are
un-uploadable through the product.

So the same bytes also have a three-call path — create an upload, PUT N parts,
complete — where the server composes the parts server-side into the exact same
final object under the exact same key, and answers ``complete`` with the exact
same ``VideoSourceResponse`` body the single-shot route returns. The single-shot
route is deliberately left ALONE: it is the right call for a 20 MiB clip (one
request, one round trip) and every one of its behaviours is pinned by tests. The
chunked routes reuse its helpers — ``_require_project`` (tenant isolation),
``_source_prefix`` (the confined key tree), ``_is_isobmff_video`` (the magic-byte
gate), ``_probe_video`` (the measurement) — rather than re-deriving any of them,
because a second implementation of a security or contract rule is a second thing
to get wrong.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid7

from fastapi import APIRouter, HTTPException, Request, Response, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from src.api.routes._common import SessionDep, TenantIdDep
from src.database.models import VideoSource, VideoSourceUpload
from src.database.repositories.project_repository import ProjectRepository
from src.database.session import bind_session_tenant
from src.services.storage_service import ComposeConstraintError
from src.services.tts_storage import _get_storage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(tags=["video-sources"])

# 4 GiB — a minute of 4K HEVC off a drone is ~130 MiB, and operators routinely
# hand us a single 1.9 GB clip straight off the card, which the previous 512 MiB
# cap refused outright. 4 GiB holds a long uninterrupted take with headroom while
# still refusing an obvious "uploaded the whole card" mistake. Enforced as a
# RUNNING total (see the stream loop), not after the fact, which is the entire
# point of this module. Nothing downstream had to change to carry it: the bytes
# never accumulate in the heap (one `_CHUNK_BYTES` chunk at a time) and the
# chunked surface slices it into `_UPLOAD_PART_BYTES` parts, which at this cap is
# 128 parts — two orders of magnitude under the compose ceiling
# (`storage_service.MAX_COMPOSE_PARTS` = 10_000).
_VIDEO_MAX_BYTES = 4 * 1024 * 1024 * 1024

# Read granularity. Big enough that a 134 MiB clip is ~134 awaits rather than
# tens of thousands, small enough that peak RSS per in-flight upload is trivial.
_CHUNK_BYTES = 1024 * 1024

_VIDEO_EXTENSIONS = {"mp4", "mov", "m4v"}

# 32 MiB per part. Squeezed between two hard numbers, neither of them ours:
#
#  * CEILING — Cloudflare 413s a body over 100 MB before the app is reached. The
#    MEASURED boundary on the real public host is that 32/48/64/96 MiB PUT bodies
#    all arrive (they get the app's 401) and 100 MB+ does not. 32 MiB sits a
#    factor of three under the smallest measured failure, which is the margin a
#    plan-change, a proxy in front of the tunnel, or a base64-ish transfer
#    encoding can eat without silently breaking every upload in the product.
#  * FLOOR — server-side compose is S3/MinIO multipart underneath, and every part
#    but the LAST must be at least 5 MiB or the compose is refused. 32 MiB clears
#    that by 6x, so the floor can never be the thing that fails a compose.
#
# It also keeps the part COUNT sane: at the 4 GiB cap a whole upload is 128
# parts, so ``complete`` composes a list that is still far under the S3/MinIO
# 10,000-source compose ceiling and a retry costs one part, not the file. Raising this trades edge headroom for round trips; lowering it trades
# round trips for a longer compose list. Do not lower it under 5 MiB.
_UPLOAD_PART_BYTES = 32 * 1024 * 1024

# How long the client should assume an open upload stays useful.
#
# ⚠ ADVISORY, and deliberately labelled as such: this is a hint returned to the
# client so a browser knows when to stop resuming and start over. NOTHING on the
# server enforces it today — there is no reaper for ``video_source_uploads``, and
# an abandoned upload's cost is the staging objects under its own prefix, which
# bucket lifecycle reclaims. If an enforced expiry is ever wanted, it belongs in
# the same place as the sweep (a background pass that flips ``open`` → ``aborted``
# and deletes the prefix), not as a check bolted onto these handlers, which would
# only reject the honest client that came back late.
_UPLOAD_TTL = timedelta(hours=24)

# ffprobe reads container headers, not the whole file, so even a 4 GiB clip is
# milliseconds — a probe that runs past this is a pathological/adversarial input
# and must not pin a worker slot. Same posture (and same order of magnitude) as
# ``uploads._VOICEOVER_PROBE_TIMEOUT_S``; the difference is what a timeout MEANS:
# there a probe miss rejects the upload, here it only costs the measurement.
_VIDEO_PROBE_TIMEOUT_S = 20.0

# ISO-BMFF (MP4/MOV/M4V) brands we accept. ``qt  `` is QuickTime and really does
# carry two trailing SPACES — a stripped comparison would silently widen the
# allowlist. ``iso4``/``iso5``/``iso6`` are the later base-media revisions real
# cameras stamp.
#
# ⚠ These are matched against the major brand AND the compatible-brands list, not
# the major brand alone — see ``_is_isobmff_video``. Checking only the major brand
# rejects real camera footage: every clip from the Sony drone this product was
# designed around announces major brand ``XAVC`` (a vendor brand that is not, and
# should not be, on this list) while declaring ``mp42``/``iso6`` as compatible.
# A vendor-brand allowlist is unmaintainable; "declares a base brand we support"
# is the check the spec actually intends.
_ISOBMFF_BRANDS = frozenset(
    {
        b"isom",
        b"iso2",
        b"iso4",
        b"iso5",
        b"iso6",
        b"mp42",
        b"mp41",
        b"qt  ",
        b"hev1",
        b"hvc1",
        b"avc1",
    }
)


class VideoSourceResponse(BaseModel):
    """One registered source clip.

    ``original_url`` is a presigned GET for the stored original (``None`` on a
    backend that cannot presign). ``preview_status`` is always ``pending`` on
    create — the derivative pass flips it to ``ready``/``failed`` later, and the
    FE renders a placeholder until then.
    """

    source_id: UUID
    filename: str
    bytes: int
    content_type: str | None = None
    preview_status: str
    original_url: str | None = None


class VideoSourceListResponse(BaseModel):
    items: list[VideoSourceResponse]
    total: int


def _ext(filename: str) -> str:
    _, _, ext = (filename or "").lower().rpartition(".")
    return ext


def _is_isobmff_video(head: bytes) -> bool:
    """True iff *head* (the FIRST bytes of the upload) looks like MP4/MOV/M4V.

    Same posture as ``uploads._image_format``: neither the extension nor the
    multipart content-type is trusted — both are attacker-controlled, and this
    bucket is readable back out through the artifact proxy, so a renamed script
    must never land in it. Checked on the first chunk only, BEFORE the rest of
    the body is drained, so a hostile upload is rejected without ever being
    stored.

    Layout of the ``ftyp`` box we parse::

        0..4    box size (big-endian uint32, includes these 4 bytes)
        4..8    b"ftyp"
        8..12   major brand
        12..16  minor version (ignored — a version number, not a brand)
        16..    compatible brands, 4 bytes each, until the box ends

    We accept when the major brand OR **any** compatible brand is one we support.
    Matching the major brand alone looks stricter but is simply wrong: cameras
    stamp a VENDOR major brand (Sony writes ``XAVC``) and declare the real base
    brands as compatible — which is exactly what the compatible-brands list is
    specified for. The size field is honoured rather than assuming 32 bytes,
    since the list is variable-length, and is clamped to what we actually have so
    a truncated or hostile header can never over-read.
    """
    if head[4:8] != b"ftyp":
        return False
    if head[8:12] in _ISOBMFF_BRANDS:
        return True
    box_size = int.from_bytes(head[0:4], "big") if len(head) >= 4 else 0
    # Clamp: never trust a declared size past the bytes in hand.
    end = min(box_size, len(head))
    return any(head[off : off + 4] in _ISOBMFF_BRANDS for off in range(16, max(end - 3, 16), 4))


def _source_prefix(tenant_id: UUID, project_id: UUID, source_id: UUID) -> str:
    """Object-key tree for one source clip — tenant AND project confined.

    Both segments are load-bearing: the key is the only thing that keeps one
    tenant's footage out of another's listing, and ``StorageService`` has no
    tenant concept of its own to fall back on.
    """
    return f"tenants/{tenant_id}/projects/{project_id}/video-sources/{source_id}/"


async def _require_project(session: AsyncSession, project_id: UUID, tenant_id: UUID) -> None:
    """404 unless ``project_id`` belongs to the caller's EFFECTIVE tenant.

    Same helper (and same repository chokepoint) as ``storyboard._require_project``.
    The router is additionally mounted under ``asgi._tenant_gate``, which binds
    the RLS GUC from the very same ``get_tenant_id`` dep this handler reads — a
    handler-only check would be inert against a gate resolving a different
    tenant, so the two must stay on one resolution path.
    """
    if await ProjectRepository(session).get(project_id, tenant_id=tenant_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "project not found")


async def _presign(key: str) -> str | None:
    """Best-effort presigned GET for ``key``; ``None`` when unavailable."""
    storage = await _get_storage()
    return await storage.get_presigned_url(key)


def _row_resp(row: VideoSource, original_url: str | None = None) -> VideoSourceResponse:
    return VideoSourceResponse(
        source_id=row.id,
        filename=row.filename,
        bytes=row.bytes,
        content_type=row.content_type,
        preview_status=row.preview_status,
        original_url=original_url,
    )


async def _drain_to_path(
    read: Callable[[int], Awaitable[bytes]],
    dest: Path,
    *,
    ext: str,
    check_magic: bool,
    max_bytes: int,
    what: str = "video",
    empty_detail: str = "empty video upload",
) -> int:
    """Drain ``read`` onto ``dest`` in fixed-size chunks; return the byte count.

    THE streaming loop, and the reason this module exists. Never call ``read()``
    with no argument (or ``request.body()``) anywhere near it — that is the
    buffered ``uploads.py`` pattern, which materialises the whole upload as one
    ``bytes`` and cannot carry a 134 MiB 4K clip. Peak memory here is one chunk
    no matter how large the body is, and that property is the whole contract.

    Written against a bare ``read(size)`` callable rather than an ``UploadFile``
    so the SAME loop serves both ingest surfaces: the single-shot multipart route
    (``UploadFile.read``) and the chunked part PUT, whose body is a raw
    ``application/octet-stream`` stream with no ``UploadFile`` anywhere in sight.
    Duplicating the loop for the second surface would have meant two places where
    an unbounded read could creep back in, and two places to keep the mid-stream
    rejects correctly ordered.

    Both rejects that must happen MID-STREAM live here, in the order that makes
    them meaningful:

    * ``check_magic`` — the container sniff on the FIRST chunk, so a hostile body
      is refused after 1 MiB rather than after 500. Off for parts 2..N, whose
      first bytes are the middle of an mdat and are not a container head; only
      part 1 carries the ``ftyp`` box, so only part 1 can be judged by it.
    * ``max_bytes`` — the running total, so a file we have already refused stops
      being written to disk instead of being measured after the fact.
    """
    total = 0
    first = True
    with dest.open("wb") as out:
        while chunk := await read(_CHUNK_BYTES):
            if first:
                first = False
                if check_magic and not _is_isobmff_video(chunk):
                    raise HTTPException(
                        status.HTTP_400_BAD_REQUEST,
                        f"upload bytes do not match a {ext!r} video container",
                    )
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    status.HTTP_413_CONTENT_TOO_LARGE,
                    f"{what} exceeds {max_bytes // (1024 * 1024)} MiB limit",
                )
            # Disk IO off the event loop: at the 4 GiB cap this is ~4096 writes
            # and a blocking write on a slow volume would stall every request.
            await asyncio.to_thread(out.write, chunk)
    if total == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, empty_detail)
    return total


async def _stream_to_path(file: UploadFile, dest: Path, ext: str) -> int:
    """Drain a multipart ``UploadFile`` onto ``dest``; return the byte count.

    The single-shot route's binding of ``_drain_to_path``: the whole clip arrives
    in one body, so the magic check applies and the running cap is the whole-video
    cap. Behaviour (including every message and status code) is unchanged from
    when the loop was inlined here — the extraction exists to serve the chunked
    part PUT, not to alter this path.

    ``_VIDEO_MAX_BYTES`` is read HERE, at call time, rather than bound as a
    default argument: tests monkeypatch the module constant to drive the running
    cap without generating half a gigabyte, and a default evaluated at def time
    would silently ignore them.
    """
    return await _drain_to_path(
        file.read,
        dest,
        ext=ext,
        check_magic=True,
        max_bytes=_VIDEO_MAX_BYTES,
    )


async def _probe_video(path: str) -> dict[str, object]:
    """ffprobe the staged clip; ``{}`` when it cannot be measured.

    **Never raises, and never fails the upload.** By the time this runs the bytes
    are already durably in MinIO and the operator's footage is safe — a missing
    measurement is a degraded row (``footage_probe`` reports the clip with a
    zeroed duration and a note), while a raised exception would 500 a request
    whose only remaining work is an INSERT. So every failure mode collapses to
    the same empty blob: no ffprobe on PATH, a non-zero exit (a container whose
    ``ftyp`` passed our magic check but that no decoder accepts), a hang past
    ``_VIDEO_PROBE_TIMEOUT_S``, or output that is not a JSON object.

    ``-show_format -show_streams -print_format json`` is deliberately the WHOLE
    blob rather than a couple of ``-show_entries`` fields: the column is
    free-form JSONB precisely because ffprobe's shape varies per container, and
    ``footage_probe._read_probe`` reads ``format.duration``, the first
    ``codec_type == "video"`` stream's ``codec_name``/``width``/``height``, and
    its RATIONAL ``avg_frame_rate``/``r_frame_rate``. Storing ffprobe's own
    output verbatim is what keeps this producer and that consumer from drifting —
    a hand-rolled shape would have to be re-agreed on both sides.

    Reads from the staging PATH, which is why the call site sits inside the
    ``try`` that the ``finally`` unlinks: seekable input is required for the
    MP4/MOV family (the ``moov`` atom may sit at the end of the file), and the
    original in MinIO would have to be dragged back through this pod's heap.
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        logger.info("no ffprobe on PATH; storing video source without a probe blob")
        return {}
    proc = await asyncio.create_subprocess_exec(
        ffprobe,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-print_format",
        "json",
        path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with asyncio.timeout(_VIDEO_PROBE_TIMEOUT_S):
            stdout, _ = await proc.communicate()
    except TimeoutError:
        # Reap it: an abandoned ffprobe on a pathological file would keep reading
        # for the pod's lifetime.
        proc.kill()
        await proc.wait()
        logger.warning("ffprobe timed out after %ss on %s", _VIDEO_PROBE_TIMEOUT_S, path)
        return {}
    if proc.returncode != 0:
        logger.warning("ffprobe exited %s on %s; storing an empty probe", proc.returncode, path)
        return {}
    try:
        parsed = json.loads(stdout.decode("utf-8", "replace"))
    except ValueError as exc:
        logger.warning("ffprobe produced unparseable JSON for %s: %s", path, exc)
        return {}
    if not isinstance(parsed, dict):
        # The column is NOT NULL with a ``'{}'`` default and the consumer indexes
        # it by key, so a bare list/scalar must be normalised here rather than
        # stored and re-narrowed at every read.
        logger.warning("ffprobe JSON for %s is not an object; storing an empty probe", path)
        return {}
    return parsed


@router.post(
    "/api/projects/{project_id}/video-sources",
    response_model=VideoSourceResponse,
    status_code=status.HTTP_201_CREATED,
    operation_id="uploads.video_source",
    summary="Upload one raw source clip (.mp4/.mov/.m4v) for an action_reel project.",
)
async def upload_video_source(
    project_id: UUID,
    file: UploadFile,
    request: Request,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> VideoSourceResponse:
    """Stream ONE source clip to storage and register it as a ``video_sources`` row.

    One file per request by design: a multi-file part would either serialise the
    uploads behind one HTTP request (no browser progress per clip, one failure
    kills the batch) or force the whole set through this worker at once. Ten
    parallel single-clip POSTs is what the FE actually wants.

    Validation runs cheapest-and-most-decisive first, so a doomed upload dies
    before its body is drained:

    1. extension + ``video/*`` content-type → 415 (zero bytes read);
    2. a declared ``Content-Length`` already over the cap → 413 (zero bytes read
       — this is what stops a 4 GB body from being streamed to disk only to be
       rejected at the end);
    3. magic bytes on the FIRST chunk → 400 (nothing stored);
    4. the running byte total → 413, aborting mid-stream;
    5. an empty body → 400.
    """
    await _require_project(session, project_id, tenant_id)
    # Release the request's DB connection before the body stream: a clip upload
    # takes minutes, and holding the ownership-check transaction open for all of
    # it pins one pgbouncer/pool connection (idle-in-transaction) per in-flight
    # upload. The row write below starts a fresh transaction.
    await session.commit()

    filename = file.filename or "clip.mp4"
    ext = _ext(filename)
    if ext not in _VIDEO_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported video type {ext!r}; allowed: {sorted(_VIDEO_EXTENSIONS)}",
        )
    content_type = file.content_type or ""
    if not content_type.startswith("video/"):
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported content type {content_type!r}; expected video/*",
        )

    # Declared size gate. The header describes the whole multipart body (a little
    # larger than the file), so it can only ever over-estimate — meaning a
    # Content-Length at or under the cap proves nothing and the running check
    # below is still the real enforcement, but one already OVER the cap is a
    # certain reject and costs us nothing to honour here.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > _VIDEO_MAX_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"video exceeds {_VIDEO_MAX_BYTES // (1024 * 1024)} MiB limit",
        )

    source_id = uuid7()
    object_key = f"{_source_prefix(tenant_id, project_id, source_id)}original.{ext}"

    # mkstemp (not NamedTemporaryFile): it hands back a raw fd + path rather than
    # a live handle, so the staging file can be written under a `with` and then
    # REOPENED BY PATH by put_file — fput_object takes a path, not a handle, and a
    # context-managed NamedTemporaryFile would already have destroyed it. Closing
    # the fd immediately keeps _stream_to_path the only writer.
    fd, tmp_path = tempfile.mkstemp(prefix="w2v-vsrc-", suffix=f".{ext}")
    os.close(fd)
    try:
        total = await _stream_to_path(file, Path(tmp_path), ext)

        storage = await _get_storage()
        # put_file is beyond the minimal ``_StorageLike`` protocol (which
        # describes the TTS/slide clip surface the test fakes implement), so it
        # is reached via getattr — the same house pattern as youtube.py's
        # resign_presigned_url and artifact_proxy's get_object_stream.
        put_file = getattr(storage, "put_file", None)
        if put_file is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "artifact storage not initialised"
            )
        await put_file(object_key, tmp_path, content_type)
        # Measure LAST, and inside this ``try``: the staging file is the only
        # local copy and the ``finally`` below is about to reclaim it, so a probe
        # placed after the block would have nothing to read. Deliberately after
        # put_file rather than before — the durable store is the operator's
        # promise, the measurement is a best-effort bonus on top of it.
        probe = await _probe_video(tmp_path)
    finally:
        # Always reclaim the staging file: a leaked 134 MiB temp per upload fills
        # the pod's ephemeral disk within a shoot.
        try:
            await asyncio.to_thread(os.unlink, tmp_path)
        except OSError:
            logger.warning("could not remove staging file %s", tmp_path)

    # New transaction after the long stream: re-bind the effective tenant so the
    # INSERT passes RLS under enforce (the after_begin listener also re-binds from
    # ``session.info``; this keeps the write correct without relying on it alone).
    await bind_session_tenant(session, tenant_id)
    row = VideoSource(
        id=source_id,
        project_id=project_id,
        filename=filename,
        object_key=object_key,
        content_type=content_type,
        bytes=total,
        preview_status="pending",
        # Raw ffprobe output, stored verbatim (``{}`` when unmeasurable). This is
        # the ONLY writer of the column in the fleet — see the module docstring.
        probe=probe,
    )
    session.add(row)
    await session.flush()
    # Commit before responding — `get_db_session` commits in its ``yield`` teardown,
    # which FastAPI runs AFTER the response is sent (CLAUDE.md § "A write handler
    # MUST commit before it responds"). Here it is not only the caller's next
    # request: the out-of-process preview/derivative worker picks this row up from
    # ANOTHER connection, so a merely-flushed row is invisible to it.
    await session.commit()
    logger.info("video source stored: project=%s source=%s bytes=%d", project_id, source_id, total)
    return _row_resp(row, await _presign(object_key))


@router.get(
    "/api/projects/{project_id}/video-sources",
    response_model=VideoSourceListResponse,
    operation_id="uploads.video_sources.list",
    summary="List a project's uploaded source clips.",
)
async def list_video_sources(
    project_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> VideoSourceListResponse:
    """List the project's source clips, oldest first (upload order = shoot order).

    Each row carries a freshly presigned ``original_url``: the presign TTL is an
    hour, so URLs must be minted per read rather than stored on the row.
    """
    await _require_project(session, project_id, tenant_id)
    rows = (
        (
            await session.execute(
                select(VideoSource)
                .where(VideoSource.project_id == project_id)
                .order_by(VideoSource.created_at)
            )
        )
        .scalars()
        .all()
    )
    items = [_row_resp(row, await _presign(row.object_key)) for row in rows]
    return VideoSourceListResponse(items=items, total=len(items))


@router.delete(
    "/api/projects/{project_id}/video-sources/{source_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="uploads.video_sources.delete",
    summary="Remove an uploaded source clip from a project.",
)
async def delete_video_source(
    project_id: UUID,
    source_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> Response:
    """Delete the row; sweep the objects best-effort.

    The ROW is the record of truth the FE and the cut stage read, so its removal
    is what the caller is promised. The bucket sweep (original + every derived
    preview/poster/sprite under the same key tree) is deliberately fail-open:
    MinIO being unreachable must not leave the operator unable to remove a clip
    they uploaded by mistake, and a stranded object is bucket-lifecycle garbage
    that no one can address anyway once the row is gone.
    """
    await _require_project(session, project_id, tenant_id)
    row = (
        await session.execute(
            select(VideoSource).where(
                VideoSource.id == source_id, VideoSource.project_id == project_id
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "video source not found")

    try:
        storage = await _get_storage()
        # Off-protocol like put_file above — resolved via getattr so a partial
        # test fake is simply treated as "nothing to sweep".
        delete_prefix = getattr(storage, "delete_object_prefix", None)
        if delete_prefix is not None:
            await delete_prefix(_source_prefix(tenant_id, project_id, source_id))
    except Exception as exc:  # noqa: BLE001 -- object sweep is best-effort, never fatal
        logger.warning("video source %s: object sweep failed (%s)", source_id, exc)

    await session.delete(row)
    await session.commit()  # durable before the 204 — see `upload_video_source`
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ==============================================================================
# Chunked upload — create / PUT part / complete / abort
#
# The three-call path that exists because Cloudflare will not carry a 134 MiB
# body (see the module docstring for the MEASURED edge behaviour). The client
# slices the file into ``part_size`` chunks and PUTs them; the server composes
# them into one object server-side. Nothing above this line changes.
# ==============================================================================


class VideoUploadCreateRequest(BaseModel):
    """What the client knows before it sends a single byte.

    ``bytes`` is the file's REAL size as the browser reports it, and it is used
    for three things: refusing an upload that could never fit under the cap
    (before anything is transferred), telling the client how many parts to slice,
    and — at ``complete`` — proving that what arrived is the whole file.

    ⚠ That third use makes this field LOAD-BEARING, and it did not used to be.
    This docstring previously said "a declaration, not a promise — ``complete``
    composes what actually arrived rather than what was announced here", and that
    was precisely the hole: a client that sent 4 of 16 parts and called
    ``complete`` got a 128 MiB fragment registered as a perfectly successful
    ``VideoSource``. ffprobe on a truncated MP4 returns ``{}`` (the ``moov`` atom
    is at the end), ``footage_probe`` re-probes nothing, and the clip carries a
    zeroed duration forever with nothing anywhere flagging it. ``complete`` now
    requires parts ``1..part_count`` AND ``sum(parts) == bytes`` — see its
    docstring for why the sum check is worth the coupling.
    """

    filename: str = Field(min_length=1, max_length=512)
    content_type: str = Field(min_length=1, max_length=255)
    bytes: int = Field(gt=0)


class VideoUploadCreateResponse(BaseModel):
    """The plan the client executes.

    ``source_id`` is handed out HERE, before any byte moves, so the final object
    key is fixed at create time. That is what makes the part keys and the final
    key derivable by both sides without a second round trip, and what makes
    ``complete`` idempotent: the winner's row already has the id the loser would
    have had to invent.

    ``expires_at`` is advisory — see ``_UPLOAD_TTL``.
    """

    upload_id: UUID
    source_id: UUID
    part_size: int
    part_count: int
    expires_at: datetime


class VideoUploadPartResponse(BaseModel):
    """Acknowledgement of one stored part, plus the running state of the whole.

    The running totals are returned on every part so a client that lost track
    (tab reload, retried part, parallel PUTs finishing out of order) can resume
    without a separate status call.
    """

    part_number: int
    bytes: int
    received_parts: int
    received_bytes: int


def _upload_parts_prefix(
    tenant_id: UUID, project_id: UUID, source_id: UUID, upload_id: UUID
) -> str:
    """Staging key tree for one upload's parts.

    Deliberately NESTED INSIDE ``_source_prefix``: the parts are the same tenant's
    and the same project's bytes as the final object, so they must live under the
    same confined tree and inherit the same isolation. Keeping staging in a
    sibling "tmp/" bucket would have created a second key space with its own
    (missable) tenant rules — the exact shape of every cross-tenant leak.
    """
    return f"{_source_prefix(tenant_id, project_id, source_id)}uploads/{upload_id}/parts/"


def _part_key(parts_prefix: str, part_number: int) -> str:
    """Zero-padded, DETERMINISTIC key for one part.

    Deterministic is the whole idempotency story: a part that is retried (flaky
    wifi, a browser that gave up on a slow PUT and re-sent it) overwrites its own
    object rather than adding a second one, so the composed result cannot contain
    a duplicated segment. Zero-padded to 5 so a lexical listing of the prefix is
    also the compose order — the cap allows 16 parts today, and padding means a
    future cap raise cannot make ``10`` sort before ``9``.
    """
    return f"{parts_prefix}{part_number:05d}"


def _parts_of(upload: VideoSourceUpload) -> dict[int, int]:
    """The ``parts`` JSONB as ``{part_number: bytes}``.

    JSON object keys are strings, so the column stores ``{"1": 33554432}``; every
    reader here wants integers to sort and to sum. Narrowed in ONE place rather
    than at each use, and defensively (a non-numeric key or value is dropped
    rather than raised on) because this blob is the only state a resumed upload
    has: a single malformed entry must degrade to "that part is missing, send it
    again", never to a 500 that strands the operator's footage forever.
    """
    out: dict[int, int] = {}
    for key, value in (upload.parts or {}).items():
        try:
            out[int(key)] = int(value)
        except TypeError, ValueError:
            logger.warning("upload %s: dropping malformed part entry %r", upload.id, key)
    return out


def _assert_whole_file_arrived(upload: VideoSourceUpload) -> int:
    """400 unless the ledger holds the WHOLE file; return the byte total.

    Two checks, and neither is optional — a compose that runs on anything less
    registers a truncated clip as a complete one, which is the single worst
    outcome this module has (a valid file of a plausible size that ``footage_probe``
    reports with a zeroed duration forever, and that nothing anywhere flags):

    * every part ``1..part_count`` is present. Checking instead that the parts in
      hand are contiguous from 1 — which is what this used to do — is not a
      completeness check at all: 4 of 16 parts pass it, and compose a 128 MiB
      fragment. The PUT route already bounds each part number to
      ``1..part_count``, so this is the other half of that same fence, not a new
      constraint on the client.
    * the parts SUM to ``bytes_declared``. The count check cannot see a truncated
      final part (16 of 16 arrived, the last one died after 5 bytes) and every
      per-part rule is satisfied in that case. The size declared at create is the
      only "this is the whole file" statement anyone ever makes, so it is enforced
      — the price being that a client which misreports its own file's size cannot
      complete, which is the right trade against registering a partial clip.

    400 for both: the fix is entirely on the client (re-send the named parts) and
    the upload deliberately stays ``open`` so it can.
    """
    stored = _parts_of(upload)
    if not stored:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "no parts uploaded")

    missing = [num for num in range(1, upload.part_count + 1) if num not in stored]
    if missing:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"upload is missing part(s) {missing}; "
            f"parts 1..{upload.part_count} must all be present",
        )

    total = sum(stored.values())
    if total != upload.bytes_declared:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"upload declared {upload.bytes_declared} bytes but {total} arrived across "
            f"{len(stored)} part(s); re-send the short or missing part(s)",
        )
    return total


async def _storage_op(name: str) -> Callable[..., Awaitable[object]]:
    """Resolve an off-protocol ``StorageService`` method or 503.

    ``compose_object``/``get_file``/``object_size``/``put_file``/
    ``delete_object_prefix`` are all
    beyond the minimal ``_StorageLike`` protocol the test fakes implement, so they
    are reached via ``getattr`` — the same house pattern as ``youtube.py``'s
    ``resign_presigned_url``. A backend that cannot compose is a 503 (the caller
    may retry against a healthy replica), never a 500 and never a silent
    single-part fallback that would truncate the clip to its first 32 MiB.
    """
    storage = await _get_storage()
    op = getattr(storage, name, None)
    if op is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"artifact storage does not support {name}",
        )
    return op


async def _sweep_parts(parts_prefix: str) -> None:
    """Best-effort delete of one upload's staging tree.

    Fail-open for the same reason ``delete_video_source``'s sweep is: the DB row
    is the record of truth, and MinIO being unreachable must not turn a completed
    (or abandoned) upload into an error the operator cannot get past. A stranded
    part is bucket-lifecycle garbage under a prefix nobody reads.
    """
    try:
        storage = await _get_storage()
        delete_prefix = getattr(storage, "delete_object_prefix", None)
        if delete_prefix is not None:
            await delete_prefix(parts_prefix)
    except Exception as exc:  # noqa: BLE001 -- staging sweep is best-effort, never fatal
        logger.warning("upload staging sweep failed for %s (%s)", parts_prefix, exc)


class _RequestBodyReader:
    """Presents a raw ASGI request body as the ``read(size)`` ``_drain_to_path`` wants.

    ``request.stream()`` yields whatever the server hands up — h11/uvicorn emit
    network-sized frames (tens of KiB), not our 1 MiB chunks — so the frames are
    accumulated into a small buffer and sliced to the requested size. The buffer
    never exceeds ``size`` plus one network frame, so peak memory on the part path
    stays at ~one chunk exactly like the multipart path.

    This class is the ONLY reason the part route can refuse ``await
    request.body()``. That call would materialise a whole 32 MiB part per
    in-flight request, and the point of the chunked surface is that ten parallel
    uploads still cost this pod almost nothing.
    """

    def __init__(self, request: Request) -> None:
        self._chunks = request.stream().__aiter__()
        self._buf = bytearray()
        self._exhausted = False

    async def read(self, size: int = -1) -> bytes:
        while not self._exhausted and len(self._buf) < size:
            try:
                self._buf += await self._chunks.__anext__()
            except StopAsyncIteration:
                self._exhausted = True
        taken = bytes(self._buf[:size])
        del self._buf[:size]
        return taken


async def _load_upload(
    session: AsyncSession,
    project_id: UUID,
    upload_id: UUID,
    *,
    lock: bool = False,
) -> VideoSourceUpload:
    """Fetch one upload, 404 unless it belongs to ``project_id``.

    The project has ALREADY been checked against the caller's tenant by
    ``_require_project``, and this filters the upload to that project — so the
    two together are the tenant fence. Addressing another tenant's upload_id is
    indistinguishable from a typo: 404, never 403, never a partial answer that
    confirms the id exists.

    ``lock=True`` takes ``FOR UPDATE`` and forces ``populate_existing``. The
    second half is not optional: without it SQLAlchemy's identity map returns the
    instance already loaded in this session with its PRE-LOCK attribute values, so
    a read-modify-write of ``parts`` under the lock would merrily write back a
    snapshot taken before the concurrent sibling committed — the lost update the
    lock was taken to prevent.
    """
    stmt = select(VideoSourceUpload).where(
        VideoSourceUpload.id == upload_id,
        VideoSourceUpload.project_id == project_id,
    )
    if lock:
        stmt = stmt.with_for_update().execution_options(populate_existing=True)
    upload = (await session.execute(stmt)).scalar_one_or_none()
    if upload is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "upload not found")
    return upload


@router.post(
    "/api/projects/{project_id}/video-sources/uploads",
    response_model=VideoUploadCreateResponse,
    status_code=status.HTTP_201_CREATED,
    operation_id="uploads.video_source.create_chunked",
    summary="Begin a chunked upload of one source clip (for files over the edge body limit).",
)
async def create_video_upload(
    project_id: UUID,
    body: VideoUploadCreateRequest,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> VideoUploadCreateResponse:
    """Reserve an upload and hand back the slicing plan. ZERO bytes are read.

    Every reject a doomed upload can earn is earned HERE, on a JSON body of a few
    hundred bytes, before the client spends ten minutes pushing 500 MiB up a
    domestic uplink:

    1. extension not in the allowlist → 415;
    2. content type not ``video/*`` → 415;
    3. declared size over ``_VIDEO_MAX_BYTES`` → 413.

    The magic-byte gate cannot run here — no bytes exist yet — so it runs on the
    first chunk of part 1, which is the earliest moment a container head can be
    judged. That is the same guarantee the single-shot route gives, just deferred
    by one request.

    ``source_id`` is pre-allocated so the FINAL key is fixed now. The alternative
    (mint the id at ``complete``) would make the part keys unrelated to the final
    object, which in turn makes an interrupted ``complete`` unrecoverable: nothing
    would tie the orphaned staging tree to the row that should have owned it.
    """
    await _require_project(session, project_id, tenant_id)

    ext = _ext(body.filename)
    if ext not in _VIDEO_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported video type {ext!r}; allowed: {sorted(_VIDEO_EXTENSIONS)}",
        )
    if not body.content_type.startswith("video/"):
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported content type {body.content_type!r}; expected video/*",
        )
    if body.bytes > _VIDEO_MAX_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"video exceeds {_VIDEO_MAX_BYTES // (1024 * 1024)} MiB limit",
        )

    part_count = math.ceil(body.bytes / _UPLOAD_PART_BYTES)
    upload = VideoSourceUpload(
        id=uuid7(),
        project_id=project_id,
        source_id=uuid7(),
        filename=body.filename,
        content_type=body.content_type,
        ext=ext,
        bytes_declared=body.bytes,
        part_size=_UPLOAD_PART_BYTES,
        part_count=part_count,
        status="open",
        parts={},
    )
    session.add(upload)
    await session.flush()
    # Committed before responding: the client's very next request is a part PUT
    # that lands on ANOTHER replica (2+ pods behind one Service), which reads this
    # row on its own connection. A merely-flushed row 404s that PUT — this is not
    # the microsecond race the single-shot route guards against, it is a certainty.
    await session.commit()
    logger.info(
        "chunked upload opened: project=%s upload=%s source=%s parts=%d declared=%d",
        project_id,
        upload.id,
        upload.source_id,
        part_count,
        body.bytes,
    )
    return VideoUploadCreateResponse(
        upload_id=upload.id,
        source_id=upload.source_id,
        part_size=_UPLOAD_PART_BYTES,
        part_count=part_count,
        expires_at=datetime.now(UTC) + _UPLOAD_TTL,
    )


@router.put(
    "/api/projects/{project_id}/video-sources/uploads/{upload_id}/parts/{part_number}",
    response_model=VideoUploadPartResponse,
    operation_id="uploads.video_source.put_part",
    summary="Upload one part of a chunked video upload (raw application/octet-stream body).",
)
async def put_video_upload_part(
    project_id: UUID,
    upload_id: UUID,
    part_number: int,
    request: Request,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> VideoUploadPartResponse:
    """Stream ONE raw part to its staging key and record its size.

    The body is ``application/octet-stream``, NOT multipart. That is deliberate:
    multipart would add a base64-ish envelope and a boundary scan to every part
    for no benefit (there is exactly one field), and the envelope inflates the
    body — pushing a 32 MiB part closer to the edge limit this whole surface
    exists to stay under. It is drained through ``_drain_to_path`` in 1 MiB
    chunks like everything else here; ``await request.body()`` is as forbidden on
    this path as ``await file.read()`` is on the other.

    **Idempotent.** The key is derived from ``part_number`` alone, so a retried
    part overwrites its own object and re-records the same size. There is no
    "part already uploaded" error to get stuck behind, which matters because the
    common failure this surface exists to survive is a part dying mid-flight.

    Three caps, in the order they can fire:

    * a declared ``Content-Length`` already over the per-part allowance → 413
      with zero bytes read;
    * the running per-part total → 413 mid-stream (so a lying Content-Length
      cannot write more than one part's worth to the pod's disk);
    * the running WHOLE-UPLOAD total across every stored part → 413, and because
      that upload can now never succeed, it is ABORTED and its staging tree swept
      in the same breath. The distinction is deliberate: an oversized single part
      is a client bug the client can fix by re-slicing, while an oversized total
      is a file that was never going to fit and whose remaining parts must not be
      allowed to keep filling the bucket.
    """
    await _require_project(session, project_id, tenant_id)
    upload = await _load_upload(session, project_id, upload_id)
    if upload.status != "open":
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"upload is {upload.status}, not open for parts"
        )
    if not 1 <= part_number <= upload.part_count:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"part_number must be within 1..{upload.part_count}",
        )

    # One chunk of slack over the negotiated part size. Not generosity: a client
    # is allowed to be a little off (a final part, a re-slice after a resume) and
    # the number that actually protects the pod is the WHOLE-upload cap below.
    part_cap = upload.part_size + _CHUNK_BYTES
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > part_cap:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"upload part exceeds {part_cap // (1024 * 1024)} MiB limit",
        )

    parts_prefix = _upload_parts_prefix(tenant_id, project_id, upload.source_id, upload_id)
    fd, tmp_path = tempfile.mkstemp(prefix="w2v-vpart-", suffix=f".{upload.ext}")
    os.close(fd)
    try:
        part_bytes = await _drain_to_path(
            _RequestBodyReader(request).read,
            Path(tmp_path),
            ext=upload.ext,
            # ONLY part 1 carries the ``ftyp`` box. Sniffing part 2 would reject
            # every legitimate upload; not sniffing part 1 would let a renamed
            # payload into a bucket that is readable back through the artifact
            # proxy. Same gate as the single-shot route, on the only chunk that
            # can carry it — and it runs before a single byte is stored.
            check_magic=part_number == 1,
            max_bytes=part_cap,
            what="upload part",
            empty_detail="empty upload part",
        )

        stored = _parts_of(upload)
        # This part REPLACES any earlier attempt at the same number, so its old
        # size must leave the total — otherwise a client that retried a part three
        # times would be charged for it three times and 413 on a legal file.
        other_bytes = sum(size for num, size in stored.items() if num != part_number)
        if other_bytes + part_bytes > _VIDEO_MAX_BYTES:
            await _sweep_parts(parts_prefix)
            upload.status = "aborted"
            await session.commit()
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"video exceeds {_VIDEO_MAX_BYTES // (1024 * 1024)} MiB limit",
            )

        put_file = await _storage_op("put_file")
        await put_file(_part_key(parts_prefix, part_number), tmp_path, "application/octet-stream")
    finally:
        try:
            await asyncio.to_thread(os.unlink, tmp_path)
        except OSError:
            logger.warning("could not remove staging part file %s", tmp_path)

    # Re-read under a row lock before merging. ``parts`` is a JSONB blob and this
    # is a read-modify-write: parallel part PUTs are the DESIGNED client shape, so
    # two handlers merging the dict they each read at entry would silently drop one
    # part and the upload would fail at ``complete`` with no trace of why.
    upload = await _load_upload(session, project_id, upload_id, lock=True)
    merged = _parts_of(upload)
    merged[part_number] = part_bytes
    upload.parts = {str(num): size for num, size in sorted(merged.items())}
    # Durable before the response, and before the NEXT part lands on another
    # replica: an uncommitted part is a part ``complete`` will call missing.
    await session.commit()
    return VideoUploadPartResponse(
        part_number=part_number,
        bytes=part_bytes,
        received_parts=len(merged),
        received_bytes=sum(merged.values()),
    )


@router.post(
    "/api/projects/{project_id}/video-sources/uploads/{upload_id}/complete",
    response_model=VideoSourceResponse,
    status_code=status.HTTP_201_CREATED,
    operation_id="uploads.video_source.complete",
    summary="Compose the uploaded parts into one source clip and register it.",
)
async def complete_video_upload(
    project_id: UUID,
    upload_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> VideoSourceResponse:
    """Compose the parts, measure the result, and register the ``video_sources`` row.

    Answers with the BYTE-IDENTICAL ``VideoSourceResponse`` (and 201) that the
    single-shot route returns. That is a contract, not a nicety: the frontend has
    one upload-completed handler and one generated type, and a chunked upload that
    answered with its own shape would fork both — the whole point is that the two
    surfaces differ in transport and in nothing else.

    Order of operations, each step justified by what it prevents:

    1. ``SELECT … FOR UPDATE`` on the upload row — the single-winner fence, see
       below;
    2. COMPLETENESS check over parts ``1..part_count`` → 400, and ``sum(parts)``
       against ``bytes_declared`` → 400. A gap means the composed object would be a
       clip with a hole punched in it that still plays for its first few seconds;
       a short tail means the same thing with the hole at the end. Both are the
       worst possible outcome, because nothing downstream would flag either — see
       "why the declared size is enforced" below;
    3. if the final object ALREADY EXISTS, skip the compose. A previous attempt
       composed it and then died before its row was durable (see "crash-safety");
    4. otherwise compose in ASCENDING part order into the final key. Ordering is
       the entire correctness of the composition: parts are opaque byte ranges,
       and any other order produces a file whose ``moov`` no longer describes its
       ``mdat``;
    5. pull the composed object back to a LOCAL file and ``_probe_video`` it;
    6. INSERT the row + flip the status, ``commit`` — and only THEN sweep the
       staging tree.

    **Crash-safety: the parts are swept AFTER the commit, never before.** They
    used to be swept first, which opened a window — a SIGTERM (routine: the
    deployment rolls with ``maxSurge: 1``) or any commit failure between the sweep
    and the commit rolled the row back with the parts already deleted. The upload
    row was still ``open``, so the client retried, the completeness check still
    passed against the ledger, and the compose then failed forever because its
    sources no longer existed: a permanent 500 on footage that was in fact safely
    composed, plus an orphaned object under ``object_key`` with no row pointing at
    it. The asymmetry is the whole argument: a leftover part set is recoverable
    garbage under a prefix nobody reads, while a missing one is unrecoverable. So
    the sweep is the LAST thing this handler does, and step 3 makes the retry that
    reaches it converge rather than recompose.

    **Why the declared size is enforced.** ``part_count`` and ``bytes_declared``
    both come from the browser, and it is tempting to treat them as untrusted
    hints (this handler used to check only that the parts it happened to have were
    contiguous from 1, never how many there should have been). That is what let 4
    of 16 parts compose into a 128 MiB fragment and register as a success. The
    count check alone still leaves the tail: 16 of 16 parts where the last one was
    truncated mid-flight is the same silent, permanently-zeroed clip. The sum is
    the only statement of "this is the whole file" anyone ever makes, so it is
    enforced, and the cost is understood: a client that lies about its own file's
    size cannot complete. That is the correct outcome — the alternative is
    registering a truncated clip as whole.

    **Exactly one winner across replicas.** Two ``complete`` calls racing (a
    double-click, a retry after a slow first attempt, two browser tabs) hit the
    same row: the first takes the lock and holds it to COMMIT, the second blocks,
    and — because the lock release re-reads the row under READ COMMITTED — the
    second then observes ``status='completed'`` and returns the winner's row
    instead of composing again. Without the lock both would INSERT a
    ``video_sources`` row for the same pre-allocated ``source_id`` and the loser
    would surface a duplicate-key 500 on a clip that uploaded perfectly.

    **Why the probe pulls the WHOLE object back and not a byte range.** An MP4's
    ``moov`` atom may sit at the END of the file (anything not written with
    ``+faststart``, which includes most camera output), so ffprobe needs a
    genuinely seekable local copy. A header-only range fetch does not fail loudly:
    ffprobe returns nothing usable, ``_probe_video`` degrades to ``{}`` as it is
    designed to, and ``footage_probe`` — which never re-probes — permanently
    reports zero duration and zero fps for that clip. A silent, permanent loss of
    the measurement is far more expensive than one streamed download.
    """
    await _require_project(session, project_id, tenant_id)
    upload = await _load_upload(session, project_id, upload_id, lock=True)

    if upload.status == "aborted":
        raise HTTPException(status.HTTP_409_CONFLICT, "upload was aborted")
    if upload.status == "completed":
        # The loser of the race. The winner's row IS the answer — returning it is
        # what makes ``complete`` safely retryable from a client that never saw
        # the first response.
        row = (
            await session.execute(select(VideoSource).where(VideoSource.id == upload.source_id))
        ).scalar_one_or_none()
        if row is None:
            # Completed, but the source has since been deleted. Not an error to
            # paper over by composing again: the parts are swept and the operator
            # deliberately removed the clip.
            raise HTTPException(status.HTTP_409_CONFLICT, "upload already completed")
        return _row_resp(row, await _presign(row.object_key))

    total = _assert_whole_file_arrived(upload)

    parts_prefix = _upload_parts_prefix(tenant_id, project_id, upload.source_id, upload_id)
    object_key = f"{_source_prefix(tenant_id, project_id, upload.source_id)}original.{upload.ext}"
    ordered_keys = [_part_key(parts_prefix, num) for num in range(1, upload.part_count + 1)]

    compose_object = await _storage_op("compose_object")
    get_file = await _storage_op("get_file")
    object_size = await _storage_op("object_size")

    # Idempotent against an already-composed object. A previous attempt that
    # composed and then died before committing its row leaves exactly this state,
    # and it must converge: the parts it staged may already be gone (they are swept
    # after the commit now, but a pod that died mid-sweep, or a bucket lifecycle
    # rule, can still have taken them), and recomposing from absent sources is a
    # permanent 500 on footage that is in fact safely stored.
    if await object_size(object_key) is None:
        try:
            await compose_object(object_key, ordered_keys, upload.content_type)
        except ComposeConstraintError as exc:
            # 409, not 400 and not 500. Not 500: the storage layer raises this
            # BEFORE issuing any compose, precisely so the caller can be told which
            # key is wrong — its own docstring promises "a precise, human-readable
            # 4xx naming the offending key", and letting it escape made that a lie
            # (a client PUTting parts under the 5 MiB S3 floor got an opaque 500).
            # Not 400: this request has NO body, so nothing in the request is
            # malformed — what conflicts is the STAGED STATE (a part below the
            # floor, or a source that has vanished). 409 also carries the right
            # recovery: the upload stays ``open``, so re-PUTting the offending part
            # and calling ``complete`` again is the fix.
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    else:
        logger.info(
            "chunked upload %s: %s already composed; skipping compose and re-registering",
            upload_id,
            object_key,
        )

    # mkstemp for the same reason the single-shot route uses it: ``get_file`` takes
    # a PATH, so the local copy must outlive any handle we opened it with.
    fd, tmp_path = tempfile.mkstemp(prefix="w2v-vsrc-", suffix=f".{upload.ext}")
    os.close(fd)
    try:
        probe: dict[str, object] = {}
        if await get_file(object_key, tmp_path):
            probe = await _probe_video(tmp_path)
        else:
            # Same posture as every other probe failure: the composed original is
            # already durable, so a failed read-back costs the measurement and
            # nothing else. It must never fail an upload whose bytes are safe.
            logger.warning("could not read back %s for probing; storing an empty probe", object_key)
    finally:
        try:
            await asyncio.to_thread(os.unlink, tmp_path)
        except OSError:
            logger.warning("could not remove staging file %s", tmp_path)

    row = VideoSource(
        id=upload.source_id,
        project_id=project_id,
        filename=upload.filename,
        object_key=object_key,
        content_type=upload.content_type,
        bytes=total,
        preview_status="pending",
        probe=probe,
    )
    session.add(row)
    upload.status = "completed"
    await session.flush()
    # One commit for both the row and the status flip — they are the same fact.
    # It also releases the FOR UPDATE lock, which is the moment the racing sibling
    # wakes up and sees ``completed``. Durable before responding, exactly as in
    # ``upload_video_source``: the out-of-process preview worker reads this row on
    # another connection.
    await session.commit()
    logger.info(
        "chunked upload completed: project=%s upload=%s source=%s parts=%d bytes=%d",
        project_id,
        upload_id,
        upload.source_id,
        upload.part_count,
        total,
    )

    # Swept LAST, AFTER the commit — never before it. The parts are the only thing
    # a retry can recompose from, and until this transaction is durable the retry
    # is still a possibility: a SIGTERM (routine on a ``maxSurge: 1`` rolling
    # deploy) or a commit failure between a pre-commit sweep and here rolled the
    # row back with the sources already deleted, leaving an ``open`` upload that
    # can never complete and an orphaned composed object nothing points at. Leaving
    # parts behind, by contrast, costs bucket-lifecycle garbage under a prefix
    # nobody reads. Best-effort by design (``_sweep_parts`` never raises), so a
    # MinIO blip here cannot turn a completed upload into an error either.
    await _sweep_parts(parts_prefix)
    return _row_resp(row, await _presign(object_key))


@router.delete(
    "/api/projects/{project_id}/video-sources/uploads/{upload_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="uploads.video_source.abort",
    summary="Abandon a chunked upload and discard its staged parts.",
)
async def abort_video_upload(
    project_id: UUID,
    upload_id: UUID,
    session: SessionDep,
    tenant_id: TenantIdDep,
) -> Response:
    """Discard an in-flight upload: sweep the staging tree, flip to ``aborted``.

    The browser calls this when the user cancels, and it is the only thing that
    reclaims the staged parts promptly (nothing else reaps them — see
    ``_UPLOAD_TTL``). Idempotent by design: aborting an already-aborted upload is
    a 204, because a cancel that can fail is a cancel the UI has to explain.

    An already-COMPLETED upload keeps its status. Flipping it would contradict the
    ``video_sources`` row that exists and make ``complete`` recomposable — the way
    to remove a finished clip is ``DELETE /video-sources/{source_id}``, which
    sweeps the real object tree. The staging sweep still runs (it is a no-op on a
    completed upload, and free insurance if the completing pod died before its
    own sweep).
    """
    await _require_project(session, project_id, tenant_id)
    upload = await _load_upload(session, project_id, upload_id, lock=True)

    await _sweep_parts(_upload_parts_prefix(tenant_id, project_id, upload.source_id, upload_id))
    if upload.status == "open":
        upload.status = "aborted"
    await session.commit()  # durable before the 204 — see `upload_video_source`
    return Response(status_code=status.HTTP_204_NO_CONTENT)
