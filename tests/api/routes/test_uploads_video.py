"""DB-backed tests for the streaming video-source routes (``uploads_video.py``).

Covers the project-scoped surface the ``action_reel`` product uploads into:

* ``POST   /api/projects/{project_id}/video-sources`` (stream one clip)
* ``GET    /api/projects/{project_id}/video-sources``
* ``DELETE /api/projects/{project_id}/video-sources/{source_id}``

Storage is faked (``_get_storage`` is monkeypatched on the route module, the same
pattern ``test_artifact_proxy.py`` uses) so nothing here needs MinIO. The fake's
``put_file`` reads back the staging file it is handed, which is what makes the
happy-path assertions statements about the STREAM and not merely about the
response body: the bytes asserted are the bytes that reached storage.

Two behaviours cannot be reached through an HTTP request at all and are driven
straight at ``_stream_to_path``: the RUNNING byte cap (a multipart body is always
larger than the file it carries, so a request whose declared length is under the
cap can never carry a file over it — the declared-length gate would fire first),
and the "first chunk only" property of the magic-byte check.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID, uuid4, uuid7

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from src.api.routes import uploads_video
from src.database.models import Project, VideoSource
from src.services.storage_service import MAX_COMPOSE_PARTS

if TYPE_CHECKING:
    from fastapi import UploadFile

pytestmark = pytest.mark.asyncio


# --- fixtures / helpers ---------------------------------------------------------


def _mp4(payload: bytes = b"", *, brand: bytes = b"isom") -> bytes:
    """A minimal ISO-BMFF head (size, ``ftyp``, brand) plus *payload*.

    Only the first 12 bytes are load-bearing for ``_is_isobmff_video``; the rest
    stands in for the clip's real mdat.
    """
    return (32).to_bytes(4, "big") + b"ftyp" + brand + b"\x00" * 4 + payload


class _FakeStorage:
    """Records every storage call the route makes; presigns deterministically."""

    def __init__(self) -> None:
        self.puts: list[tuple[str, str, str]] = []
        self.staged: list[bytes] = []
        self.prefixes: list[str] = []
        self.sweep_error: Exception | None = None
        self.presign_calls: list[tuple[str, int | None]] = []

    async def put_file(self, key: str, path: str, content_type: str) -> None:
        self.puts.append((key, path, content_type))
        # Read the staging file back HERE — after the handler returns it is
        # unlinked, so this is the only place the streamed bytes are observable.
        self.staged.append(Path(path).read_bytes())

    async def get_presigned_url(self, key: str, expires: int | None = None) -> str:
        self.presign_calls.append((key, expires))
        return f"https://minio.test/{key}?sig=abc"

    async def delete_object_prefix(self, prefix: str) -> None:
        self.prefixes.append(prefix)
        if self.sweep_error is not None:
            raise self.sweep_error


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch) -> _FakeStorage:
    """Patch the route module's ``_get_storage`` with a recording fake."""
    fake = _FakeStorage()

    async def _get_storage() -> _FakeStorage:
        return fake

    monkeypatch.setattr(uploads_video, "_get_storage", _get_storage)
    return fake


# A real, decodable clip is required to exercise the ffprobe path: ``_mp4()`` is a
# container HEAD only, which passes the route's magic-byte gate (as it should —
# that gate reads 12 bytes) but which no decoder accepts. Generated rather than
# committed so the repo does not carry a binary whose provenance and licence
# nobody can state, and kept tiny (a 1s 320x240 testsrc is ~10 KiB).
_CLIP_SECONDS = 1.0
_CLIP_FPS = 25.0
_CLIP_WIDTH = 320
_CLIP_HEIGHT = 240


@pytest.fixture(scope="session")
def real_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Path to a generated, genuinely decodable MP4.

    Session-scoped: the encode is the same every time and costs an ffmpeg
    process, so it is paid once for the whole run.

    Skips (never fails) when ffmpeg/ffprobe are absent: an image without ffmpeg
    cannot express the difference between "the probe path is broken" and "there
    is nothing to probe with", and the route's own contract in that situation is
    the degraded one, which the sibling test covers WITHOUT needing the tools.
    """
    # Resolved to an ABSOLUTE path and passed as argv[0]: a bare "ffmpeg" would be
    # re-resolved against whatever PATH exec sees, which is not necessarily the one
    # this check consulted (S607).
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not on PATH; cannot generate or verify a real clip")
    path = tmp_path_factory.mktemp("real-clip") / "clip.mp4"
    subprocess.run(  # noqa: S603 -- fixed argv, no shell, path is pytest's own tmp dir
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={_CLIP_WIDTH}x{_CLIP_HEIGHT}:rate={_CLIP_FPS:.0f}"
            f":duration={_CLIP_SECONDS}",
            # yuv420p + faststart: the widely-decodable pixel format, and the moov
            # atom moved to the front — the same shape a real camera file has.
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(path),
        ],
        capture_output=True,
        check=True,
        timeout=60,
    )
    return path


@pytest.fixture
def staging_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point ``tempfile.mkstemp`` at an empty private directory.

    Lets a test assert the staging file was reclaimed by listing the directory —
    a scan of the real system temp dir would race every other process on the box.
    """
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(staging))
    return staging


async def _seed_project(db_session, *, tenant_id: UUID, subject_id: UUID) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        # NOT ``action_reel``: that product type does not exist yet (it arrives with
        # its blueprint + the ``ck_projects_type`` CHECK migration), and seeding it
        # here would fail the constraint. Deliberate — the route is project-scoped,
        # never type-scoped, so ANY existing type exercises it identically.
        type="generic",
        title="video-source-upload",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    db_session.add(project)
    await db_session.commit()
    return project


async def _own_project(client, db_session) -> Project:
    return await _seed_project(
        db_session,
        tenant_id=UUID(client.tenant_id),
        subject_id=UUID(client.subject_id),
    )


async def _post(client, project_id, name="take-1.mp4", data=None, ctype="video/mp4"):
    return await client.post(
        f"/api/projects/{project_id}/video-sources",
        files={"file": (name, _mp4() if data is None else data, ctype)},
    )


class _ChunkedUpload:
    """Minimal ``UploadFile`` stand-in that hands back queued chunks in order.

    ``reads`` counts the ``read`` calls so a test can prove a reject happened
    MID-STREAM (the remaining chunks were never drained).
    """

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)
        self.reads = 0

    async def read(self, _size: int = -1) -> bytes:
        self.reads += 1
        return self._chunks.pop(0) if self._chunks else b""


# --- POST: happy path -----------------------------------------------------------


async def test_upload_video_source_creates_row_and_streams_to_storage(
    client, db_session, storage, staging_dir
) -> None:
    """201 + a pending row + the bytes under a tenant/project-confined key."""
    project = await _own_project(client, db_session)
    data = _mp4(b"mdat-payload" * 4)

    r = await client.post(
        f"/api/projects/{project.id}/video-sources",
        files={"file": ("take-1.mp4", data, "video/mp4")},
    )

    assert r.status_code == 201, r.text
    body = r.json()
    source_id = body["source_id"]
    expected_key = (
        f"tenants/{client.tenant_id}/projects/{project.id}/video-sources/{source_id}/original.mp4"
    )
    assert body["filename"] == "take-1.mp4"
    assert body["bytes"] == len(data)
    assert body["preview_status"] == "pending"
    assert body["content_type"] == "video/mp4"
    assert body["original_url"] == f"https://minio.test/{expected_key}?sig=abc"

    # Storage saw the confined key, the declared content type, and the real bytes.
    assert len(storage.puts) == 1
    assert storage.puts[0][0] == expected_key
    assert storage.puts[0][2] == "video/mp4"
    assert storage.staged == [data]

    row = (
        await db_session.execute(select(VideoSource).where(VideoSource.id == UUID(source_id)))
    ).scalar_one()
    assert row.project_id == project.id
    assert row.object_key == expected_key
    assert row.bytes == len(data)
    assert row.preview_status == "pending"
    assert row.preview_key is None and row.poster_key is None and row.sprite_key is None
    # The derivatives are filled by a later pass, so they are NULL on create.
    # ``probe`` is empty for a different reason: it IS written on create (see the
    # ffprobe tests below), but ``_mp4()`` is a bare container head with no
    # decodable payload, so ffprobe exits non-zero and the blob degrades to {}.
    assert row.probe == {}

    # The staging file is reclaimed on the success path too.
    assert list(staging_dir.iterdir()) == []


async def test_upload_releases_db_transaction_while_streaming(
    client, db_session, storage, monkeypatch
) -> None:
    """The ownership-check transaction is committed BEFORE the (multi-minute)
    body stream, so an in-flight upload does not pin a DB connection; the row
    is still written afterwards in a fresh transaction."""
    project = await _own_project(client, db_session)
    seen: dict[str, Any] = {}

    real_require = uploads_video._require_project
    real_stream = uploads_video._stream_to_path

    async def _capturing_require(session, project_id, tenant_id):
        seen["session"] = session
        await real_require(session, project_id, tenant_id)

    async def _observing_stream(file, dest, ext):
        seen["in_txn_during_stream"] = seen["session"].in_transaction()
        return await real_stream(file, dest, ext)

    monkeypatch.setattr(uploads_video, "_require_project", _capturing_require)
    monkeypatch.setattr(uploads_video, "_stream_to_path", _observing_stream)

    r = await _post(client, project.id)

    assert r.status_code == 201, r.text
    assert seen["in_txn_during_stream"] is False
    row = (
        await db_session.execute(
            select(VideoSource).where(VideoSource.id == UUID(r.json()["source_id"]))
        )
    ).scalar_one()
    assert row.project_id == project.id


@pytest.mark.parametrize(
    ("name", "ctype", "brand"),
    [("shot.mov", "video/quicktime", b"qt  ")],
)
async def test_upload_accepts_every_allowed_extension_and_brand(
    client, db_session, storage, name, ctype, brand
) -> None:
    """The allowlisted containers/brands land, keyed by their OWN extension."""
    project = await _own_project(client, db_session)

    r = await _post(client, project.id, name=name, data=_mp4(brand=brand), ctype=ctype)

    assert r.status_code == 201, r.text
    ext = name.rsplit(".", 1)[1]
    assert storage.puts[0][0].endswith(f"/original.{ext}")


# --- POST: the ffprobe blob -----------------------------------------------------
# The upload is the ONLY writer of ``video_sources.probe`` in the fleet, and
# ``footage_probe`` is written on the premise that the blob is authoritative and
# never re-probes. These two tests hold both halves of that contract: that a real
# clip is measured, in the shape the consumer parses, and that a clip we cannot
# measure still lands.


async def test_upload_succeeds_with_an_empty_probe_when_ffprobe_is_unavailable(
    client, db_session, storage, staging_dir, monkeypatch, real_clip
) -> None:
    """A probe failure degrades to ``{}``; it never fails the upload.

    ``shutil.which`` is stubbed to ``None`` — the "no ffprobe on this image" case,
    which is also the cheapest stand-in for the other three failure modes (non-zero
    exit, timeout, unparseable output) since all four collapse to the same blob.
    The clip is a REAL, probeable one, so an empty blob here can only come from
    the degradation path and not from an undecodable payload.

    The point is the 201: by the time the probe runs the bytes are already in
    MinIO, so refusing the request would throw away footage that is safely stored
    over a measurement nobody asked for.
    """
    monkeypatch.setattr(uploads_video.shutil, "which", lambda _name: None)
    project = await _own_project(client, db_session)

    r = await client.post(
        f"/api/projects/{project.id}/video-sources",
        files={"file": ("real.mp4", real_clip.read_bytes(), "video/mp4")},
    )

    assert r.status_code == 201, r.text
    row = (
        await db_session.execute(
            select(VideoSource).where(VideoSource.id == UUID(r.json()["source_id"]))
        )
    ).scalar_one()
    assert row.probe == {}
    # The bytes still reached storage, and the staging file was still reclaimed —
    # the degraded path costs the measurement and nothing else.
    assert len(storage.puts) == 1
    assert list(staging_dir.iterdir()) == []


# --- POST: rejects --------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "ctype"),
    [("clip.avi", "video/x-msvideo")],
)
async def test_upload_rejects_unlisted_extension_415(
    client, db_session, storage, name, ctype
) -> None:
    project = await _own_project(client, db_session)
    r = await _post(client, project.id, name=name, ctype=ctype)
    assert r.status_code == 415, r.text
    assert storage.puts == []


async def test_upload_rejects_non_video_content_type_415(client, db_session, storage) -> None:
    """A ``.mp4`` name over a non-``video/*`` part is refused before any read."""
    project = await _own_project(client, db_session)
    r = await _post(client, project.id, ctype="application/octet-stream")
    assert r.status_code == 415, r.text
    assert storage.puts == []


async def test_upload_rejects_bytes_that_are_not_an_isobmff_container_400(
    client, db_session, storage, staging_dir
) -> None:
    """The security gate: a ``.mp4`` carrying arbitrary bytes never reaches storage.

    Both the extension and the multipart content-type are caller-controlled, and
    this bucket is readable back through the artifact proxy — so the magic-byte
    check on the first chunk is what stands between a renamed payload and the
    bucket. The empty staging dir proves the partial write was reclaimed.
    """
    project = await _own_project(client, db_session)
    r = await _post(client, project.id, data=b"not a video")
    assert r.status_code == 400, r.text
    assert "video container" in r.json()["detail"]
    assert storage.puts == []
    assert list(staging_dir.iterdir()) == []


async def test_upload_rejects_isobmff_with_an_unlisted_brand_400(
    client, db_session, storage
) -> None:
    """``ftyp`` alone is not enough — a brand must be on the allowlist."""
    project = await _own_project(client, db_session)
    r = await _post(client, project.id, data=_mp4(brand=b"evil"))
    assert r.status_code == 400, r.text
    assert storage.puts == []


async def test_upload_accepts_a_vendor_major_brand_via_compatible_brands(
    client, db_session, storage
) -> None:
    """A VENDOR major brand is accepted when a compatible brand is supported.

    Regression test for a real rejection, not a hypothetical: every clip from the
    Sony drone this product was built around stamps major brand ``XAVC`` and lists
    ``XAVC mp42 nras iso6`` as compatible. Matching only the major brand 400s the
    entire demo corpus. Byte-for-byte the real 32-byte ``ftyp`` box those files
    carry.
    """
    header = (
        (32).to_bytes(4, "big")
        + b"ftyp"
        + b"XAVC"  # major brand — deliberately NOT on the allowlist
        + b"\x01\x0b\x1f\xff"  # minor version
        + b"XAVC"
        + b"mp42"  # ← the one that must carry the decision
        + b"nras"
        + b"iso6"
    )
    project = await _own_project(client, db_session)
    r = await _post(client, project.id, data=header + b"\x00" * 64)
    assert r.status_code == 201, r.text
    assert len(storage.puts) == 1


async def test_upload_rejects_empty_body_400(client, db_session, storage) -> None:
    project = await _own_project(client, db_session)
    r = await _post(client, project.id, data=b"")
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == "empty video upload"
    assert storage.puts == []


async def test_the_whole_video_cap_is_4_gib_and_slices_under_the_compose_ceiling() -> None:
    """Pin the SHIPPED cap, which no other test in this file can see.

    Every 413 test here monkeypatches ``_VIDEO_MAX_BYTES`` down to a handful of
    bytes so the running check can be driven without generating gigabytes. That
    makes them statements about the enforcement PATH and deliberately blind to
    the constant's real value — so a revert of the cap itself would leave all of
    them green. This is the only assertion that would go red.

    4 GiB is the number the product needs: operators hand us a single ~1.9 GB
    clip straight off the drone's card, which the previous 512 MiB cap refused
    outright.

    The second half is the reason the cap could be raised without touching
    anything else. ``_UPLOAD_PART_BYTES`` stays at 32 MiB (it is pinned by
    Cloudflare's unraisable ~100 MB edge body limit, NOT by our cap), so raising
    the cap raises the PART COUNT instead — and a part count over
    ``MAX_COMPOSE_PARTS`` would fail at ``complete``, after the client has
    already paid for the whole upload.
    """
    assert uploads_video._VIDEO_MAX_BYTES == 4 * 1024 * 1024 * 1024
    # Unchanged, and not derived from the cap: the edge limit sets it.
    assert uploads_video._UPLOAD_PART_BYTES == 32 * 1024 * 1024

    worst_case_parts = math.ceil(uploads_video._VIDEO_MAX_BYTES / uploads_video._UPLOAD_PART_BYTES)
    assert worst_case_parts == 128
    assert worst_case_parts <= MAX_COMPOSE_PARTS


async def test_declared_content_length_over_cap_413_before_any_byte_is_read(
    client, db_session, storage, staging_dir, monkeypatch
) -> None:
    """A body whose declared length already busts the cap is refused unread.

    The payload is deliberately NOT a video: if the route had streamed it, the
    magic-byte check would have produced a 400. A 413 can therefore only come
    from the ``Content-Length`` gate, and the empty staging dir proves no temp
    file was even created.
    """
    monkeypatch.setattr(uploads_video, "_VIDEO_MAX_BYTES", 8)
    project = await _own_project(client, db_session)

    r = await _post(client, project.id, data=b"not a video, and far over the cap")

    assert r.status_code == 413, r.text
    assert "MiB limit" in r.json()["detail"]
    assert storage.puts == []
    assert list(staging_dir.iterdir()) == []


async def test_upload_on_storage_without_put_file_is_503_and_still_cleans_staging(
    client, db_session, monkeypatch, staging_dir
) -> None:
    """``put_file`` is off-protocol (reached via ``getattr``) — its absence is a
    503, and the ``finally`` still reclaims the staged clip."""

    class _NoPutFile:
        async def get_presigned_url(self, key: str) -> str:
            return f"https://minio.test/{key}"

    async def _get_storage() -> _NoPutFile:
        return _NoPutFile()

    monkeypatch.setattr(uploads_video, "_get_storage", _get_storage)
    project = await _own_project(client, db_session)

    r = await _post(client, project.id)

    assert r.status_code == 503, r.text
    assert list(staging_dir.iterdir()) == []


# --- POST: project ownership ----------------------------------------------------


async def test_upload_to_unknown_project_404(client, storage) -> None:
    r = await _post(client, uuid7())
    assert r.status_code == 404, r.text
    assert r.json()["detail"] == "project not found"
    assert storage.puts == []


async def test_upload_to_another_tenants_project_404(client, db_session, storage) -> None:
    """Cross-tenant is indistinguishable from missing — never a 403, never a 201."""
    foreign = await _seed_project(db_session, tenant_id=uuid4(), subject_id=uuid4())
    r = await _post(client, foreign.id)
    assert r.status_code == 404, r.text
    assert storage.puts == []


# --- the streaming loop: reachable only below the HTTP layer --------------------
# A multipart body is always LARGER than the file it carries, so a request that
# clears the declared-length gate can never carry a file over the cap. The
# running check is therefore driven directly.


async def test_stream_to_path_aborts_mid_stream_when_the_running_total_busts_the_cap(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(uploads_video, "_VIDEO_MAX_BYTES", 40)
    dest = tmp_path / "clip.mp4"
    upload = _ChunkedUpload([_mp4(b"a" * 20), b"b" * 32, b"c" * 32])

    with pytest.raises(HTTPException) as exc:
        await uploads_video._stream_to_path(cast("UploadFile", upload), dest, "mp4")

    assert exc.value.status_code == 413
    # Aborted on the SECOND chunk: the third was never read, and the over-cap
    # chunk was never written.
    assert upload.reads == 2
    assert dest.read_bytes() == _mp4(b"a" * 20)


async def test_stream_to_path_rejects_bad_magic_before_draining_the_body(tmp_path) -> None:
    """The magic check runs on the first chunk only, so a hostile 500 MiB upload
    is refused after one chunk rather than after the whole body."""
    dest = tmp_path / "clip.mp4"
    upload = _ChunkedUpload([b"not a video", b"more", b"and more"])

    with pytest.raises(HTTPException) as exc:
        await uploads_video._stream_to_path(cast("UploadFile", upload), dest, "mp4")

    assert exc.value.status_code == 400
    assert upload.reads == 1
    assert dest.read_bytes() == b""


async def test_stream_to_path_accepts_a_multi_chunk_body_and_checks_magic_once(
    tmp_path,
) -> None:
    """A body spanning several chunks round-trips whole; only chunk 1 is sniffed
    (chunk 2 is not a container head and must not be judged as one)."""
    dest = tmp_path / "clip.mp4"
    chunks = [_mp4(b"x" * 8), b"not a container head", b"tail"]
    upload = _ChunkedUpload(chunks)

    total = await uploads_video._stream_to_path(cast("UploadFile", upload), dest, "mp4")

    assert total == sum(len(c) for c in chunks)
    assert dest.read_bytes() == b"".join(chunks)


async def test_stream_to_path_rejects_an_empty_body(tmp_path) -> None:
    dest = tmp_path / "clip.mp4"
    with pytest.raises(HTTPException) as exc:
        await uploads_video._stream_to_path(cast("UploadFile", _ChunkedUpload([])), dest, "mp4")
    assert exc.value.status_code == 400
    assert exc.value.detail == "empty video upload"


# --- GET: list ------------------------------------------------------------------


async def test_list_video_sources_returns_every_clip_with_a_fresh_presign(
    client, db_session, storage
) -> None:
    project = await _own_project(client, db_session)
    first = (await _post(client, project.id, name="a.mp4")).json()
    second = (await _post(client, project.id, name="b.mov", ctype="video/quicktime")).json()

    r = await client.get(f"/api/projects/{project.id}/video-sources")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 2
    assert {i["source_id"] for i in body["items"]} == {first["source_id"], second["source_id"]}
    assert {i["filename"] for i in body["items"]} == {"a.mp4", "b.mov"}
    # Minted per read (TTL is an hour), so every row carries a URL.
    assert all(i["original_url"].startswith("https://minio.test/") for i in body["items"])
    assert all(i["preview_status"] == "pending" for i in body["items"])


async def test_list_video_sources_is_empty_for_a_project_with_no_uploads(
    client, db_session, storage
) -> None:
    project = await _own_project(client, db_session)
    r = await client.get(f"/api/projects/{project.id}/video-sources")
    assert r.status_code == 200, r.text
    assert r.json() == {"items": [], "total": 0}


async def test_list_video_sources_of_another_tenants_project_404(
    client, db_session, storage
) -> None:
    foreign = await _seed_project(db_session, tenant_id=uuid4(), subject_id=uuid4())
    r = await client.get(f"/api/projects/{foreign.id}/video-sources")
    assert r.status_code == 404, r.text


# --- DELETE ---------------------------------------------------------------------


async def test_delete_video_source_removes_the_row_and_sweeps_the_key_tree(
    client, db_session, storage
) -> None:
    """The row is the promise; the object sweep covers the whole source prefix
    (original + every derived preview/poster/sprite)."""
    project = await _own_project(client, db_session)
    source_id = (await _post(client, project.id)).json()["source_id"]

    r = await client.delete(f"/api/projects/{project.id}/video-sources/{source_id}")

    assert r.status_code == 204, r.text
    assert r.content == b""
    assert storage.prefixes == [
        f"tenants/{client.tenant_id}/projects/{project.id}/video-sources/{source_id}/"
    ]
    remaining = (
        await db_session.execute(select(VideoSource).where(VideoSource.id == UUID(source_id)))
    ).scalar_one_or_none()
    assert remaining is None


async def test_delete_video_source_survives_a_failing_object_sweep(
    client, db_session, storage
) -> None:
    """Fail-open by design: MinIO being unreachable must not strand the row, or
    the operator can never remove a clip they uploaded by mistake."""
    project = await _own_project(client, db_session)
    source_id = (await _post(client, project.id)).json()["source_id"]
    storage.sweep_error = RuntimeError("minio unreachable")

    r = await client.delete(f"/api/projects/{project.id}/video-sources/{source_id}")

    assert r.status_code == 204, r.text
    remaining = (
        await db_session.execute(select(VideoSource).where(VideoSource.id == UUID(source_id)))
    ).scalar_one_or_none()
    assert remaining is None


async def test_delete_unknown_video_source_404(client, db_session, storage) -> None:
    project = await _own_project(client, db_session)
    r = await client.delete(f"/api/projects/{project.id}/video-sources/{uuid7()}")
    assert r.status_code == 404, r.text
    assert r.json()["detail"] == "video source not found"
    assert storage.prefixes == []


async def test_delete_video_source_from_a_different_project_404(
    client, db_session, storage
) -> None:
    """The source must belong to the project in the path — owning both is not
    enough to address one project's clip through the other's URL."""
    project = await _own_project(client, db_session)
    other = await _own_project(client, db_session)
    source_id = (await _post(client, project.id)).json()["source_id"]

    r = await client.delete(f"/api/projects/{other.id}/video-sources/{source_id}")

    assert r.status_code == 404, r.text
    assert storage.prefixes == []
    remaining = (
        await db_session.execute(select(VideoSource).where(VideoSource.id == UUID(source_id)))
    ).scalar_one_or_none()
    assert remaining is not None


async def test_delete_video_source_in_another_tenants_project_404(
    client, db_session, storage
) -> None:
    foreign = await _seed_project(db_session, tenant_id=uuid4(), subject_id=uuid4())
    r = await client.delete(f"/api/projects/{foreign.id}/video-sources/{uuid7()}")
    assert r.status_code == 404, r.text
    assert r.json()["detail"] == "project not found"
    assert storage.prefixes == []


async def test_original_url_presign_uses_the_global_default_ttl_not_compile(storage) -> None:
    """(c) Representative browser-facing presign: ``original_url`` is handed
    straight to the SPA, so it must NOT get the long compile-input TTL —
    ``_presign`` must call ``get_presigned_url`` with no explicit ``expires``
    override, which is what lets it fall through to the (now-reverted) global
    ``StorageSettings.presign_expiry_seconds`` default (3600s).

    MUTATION PROOF: passing ``expires=compile_budget.PRESIGN_TTL_SECONDS`` (or
    any explicit override) at this call site turns this red.
    """
    url = await uploads_video._presign("tenants/t/projects/p/video-sources/s/original.mp4")

    assert url == "https://minio.test/tenants/t/projects/p/video-sources/s/original.mp4?sig=abc"
    assert storage.presign_calls == [("tenants/t/projects/p/video-sources/s/original.mp4", None)]
