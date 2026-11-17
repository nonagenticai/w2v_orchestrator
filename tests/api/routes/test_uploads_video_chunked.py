"""DB-backed tests for the CHUNKED video-source upload surface (``uploads_video.py``).

The three-call path that exists because Cloudflare 413s any body over 100 MB, so a
134 MiB 4K drone clip — 4 of the 10 clips in the demo corpus — cannot be uploaded
in one request at all:

* ``POST   /api/projects/{pid}/video-sources/uploads``                       (create)
* ``PUT    /api/projects/{pid}/video-sources/uploads/{uid}/parts/{n}``       (part)
* ``POST   /api/projects/{pid}/video-sources/uploads/{uid}/complete``        (compose)
* ``DELETE /api/projects/{pid}/video-sources/uploads/{uid}``                 (abort)

Same harness as ``test_uploads_video.py``: ``_get_storage`` is monkeypatched on the
route module with a partial fake (``compose_object`` / ``get_file`` / ``put_file`` /
``delete_object_prefix`` are all off-protocol and reached through ``getattr``, so a
partial fake is the house pattern, not a shortcut). The fake keeps a real
key → bytes map so a compose is observable as CONCATENATION and not merely as a
call, which is what lets the ordering test assert on the composed video rather
than on the arguments alone.

Every test's docstring states what breaks if the behaviour it pins is reverted.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path
from uuid import UUID, uuid4, uuid7

import pytest
from sqlalchemy import select

from src.api.routes import uploads_video
from src.database.models import Project, VideoSource, VideoSourceUpload
from src.services.storage_service import ComposeConstraintError

pytestmark = pytest.mark.asyncio


# --- fixtures / helpers ---------------------------------------------------------


def _mp4(payload: bytes = b"", *, brand: bytes = b"isom") -> bytes:
    """A minimal ISO-BMFF head (size, ``ftyp``, brand) plus *payload*.

    Same helper as the single-shot suite: only the first 12 bytes are load-bearing
    for ``_is_isobmff_video``; the rest stands in for the clip's mdat.
    """
    return (32).to_bytes(4, "big") + b"ftyp" + brand + b"\x00" * 4 + payload


class _FakeChunkStorage:
    """Records every storage call AND keeps a real key → bytes map.

    The map is what makes ``compose_object`` observable as an actual
    concatenation: a test can read the composed object back and prove the parts
    were stitched in the order the route asked for, rather than only proving that
    a call happened.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.puts: list[tuple[str, str, str]] = []
        self.composes: list[tuple[str, list[str], str]] = []
        self.prefixes: list[str] = []
        self.gets: list[str] = []

    async def put_file(self, key: str, path: str, content_type: str) -> str:
        self.puts.append((key, path, content_type))
        # Read the staging file back HERE — the handler unlinks it in its
        # ``finally``, so this is the only place the streamed bytes are visible.
        self.objects[key] = Path(path).read_bytes()
        return key

    async def get_file(self, key: str, local_path: str) -> bool:
        self.gets.append(key)
        data = self.objects.get(key)
        if data is None:
            # Absence is a RETURN VALUE on the real service, not an exception —
            # the route branches on it to degrade the probe.
            return False
        Path(local_path).write_bytes(data)
        return True

    async def compose_object(
        self, key: str, source_keys: list[str], content_type: str = "application/octet-stream"
    ) -> str:
        self.composes.append((key, list(source_keys), content_type))
        self.objects[key] = b"".join(self.objects[src] for src in source_keys)
        return key

    async def object_size(self, key: str) -> int | None:
        """A metadata HEAD: the size, or ``None`` when the object is absent.

        The route probes the FINAL key with this before composing, so a retry after
        a crash between compose and commit converges instead of re-stitching parts
        that may already be gone.
        """
        data = self.objects.get(key)
        return None if data is None else len(data)

    async def get_presigned_url(self, key: str) -> str:
        return f"https://minio.test/{key}?sig=abc"

    async def delete_object_prefix(self, prefix: str) -> None:
        self.prefixes.append(prefix)
        for key in [k for k in self.objects if k.startswith(prefix)]:
            del self.objects[key]


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch) -> _FakeChunkStorage:
    """Patch the route module's ``_get_storage`` with a recording fake."""
    fake = _FakeChunkStorage()

    async def _get_storage() -> _FakeChunkStorage:
        return fake

    monkeypatch.setattr(uploads_video, "_get_storage", _get_storage)
    return fake


@pytest.fixture
def staging_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point ``tempfile.mkstemp`` at an empty private directory.

    Lets a test assert the staging file was reclaimed by listing the directory —
    scanning the real system temp dir would race every other process on the box.
    """
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(staging))
    return staging


@pytest.fixture(scope="session")
def real_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Path to a generated, genuinely decodable MP4.

    Re-declared here rather than imported from ``test_uploads_video.py``: a
    fixture defined in a sibling test MODULE is not visible to this one, and
    hoisting it into ``tests/conftest.py`` would make every unrelated test session
    pay for (and depend on) an ffmpeg encode.

    Skips — never fails — when ffmpeg/ffprobe are absent: an image without them
    cannot tell "the probe path is broken" from "there is nothing to probe with",
    and the degraded contract is covered by a sibling test that needs neither.
    """
    # Absolute path as argv[0]: a bare "ffmpeg" would be re-resolved against
    # whatever PATH exec sees, not the one this check consulted (S607).
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not on PATH; cannot generate or verify a real clip")
    path = tmp_path_factory.mktemp("chunked-real-clip") / "clip.mp4"
    subprocess.run(  # noqa: S603 -- fixed argv, no shell, path is pytest's own tmp dir
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=25:duration=1.0",
            # yuv420p + faststart: the widely-decodable pixel format, and the moov
            # atom at the FRONT — which also means part 1 of the split below really
            # does carry the container head, exactly like a real camera file.
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


async def _seed_project(db_session, *, tenant_id: UUID, subject_id: UUID) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        # NOT ``action_reel`` — that product type has no ``ck_projects_type`` entry
        # yet. The route is project-scoped, never type-scoped, so any existing type
        # exercises it identically. Same choice as the single-shot suite.
        type="generic",
        title="chunked-video-upload",
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


async def _create(client, project_id, *, name="take-1.mp4", ctype="video/mp4", size=64):
    return await client.post(
        f"/api/projects/{project_id}/video-sources/uploads",
        json={"filename": name, "content_type": ctype, "bytes": size},
    )


async def _put_part(client, project_id, upload_id, part_number: int, data: bytes):
    """PUT one raw part. ``content=`` (not ``files=``) — the body is
    ``application/octet-stream``; a multipart envelope would inflate every part
    toward the very edge limit this surface exists to stay under."""
    return await client.put(
        f"/api/projects/{project_id}/video-sources/uploads/{upload_id}/parts/{part_number}",
        content=data,
    )


async def _complete(client, project_id, upload_id):
    return await client.post(
        f"/api/projects/{project_id}/video-sources/uploads/{upload_id}/complete"
    )


def _parts_prefix(client, project_id, source_id, upload_id) -> str:
    return (
        f"tenants/{client.tenant_id}/projects/{project_id}"
        f"/video-sources/{source_id}/uploads/{upload_id}/parts/"
    )


async def _reload_upload(db_session, upload_id) -> VideoSourceUpload:
    """Read the upload row through the TEST's own session.

    Deliberately an independent session from the one each request used: the whole
    reason ``video_source_uploads`` is a table and not a dict in the pod is that
    part *n* and part *n+1* land on different replicas. Asserting through this
    session is asserting what a DIFFERENT pod would see.
    """
    # ``populate_existing`` for the same reason the route's ``_load_upload`` sets it:
    # without it the identity map hands back the instance loaded EARLIER in this
    # test, with its pre-request attribute values, and the assertion would be about
    # a stale snapshot rather than about what is committed in Postgres.
    return (
        await db_session.execute(
            select(VideoSourceUpload)
            .where(VideoSourceUpload.id == UUID(str(upload_id)))
            .execution_options(populate_existing=True)
        )
    ).scalar_one()


# --- create ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "ctype"),
    [("clip.avi", "video/x-msvideo")],
)
async def test_create_rejects_unsupported_type_415_with_zero_bytes_read(
    client, db_session, storage, name, ctype
) -> None:
    """A doomed upload is refused on a few hundred bytes of JSON, before transfer.

    REVERTED ⇒ the reject moves to the first part PUT (or to ``complete``), so the
    operator pushes 500 MiB up a domestic uplink for ten minutes and is told
    afterwards that the container was never acceptable. Nothing is stored and no
    session row is opened, so there is nothing to sweep either.
    """
    project = await _own_project(client, db_session)

    r = await _create(client, project.id, name=name, ctype=ctype)

    assert r.status_code == 415, r.text
    assert storage.puts == []
    # No session row: a rejected create must not leave an ``open`` upload behind
    # for the 24 h sweeper to reap.
    assert (await db_session.execute(select(VideoSourceUpload))).scalars().all() == []


async def test_create_rejects_declared_bytes_over_the_cap_413_with_zero_bytes_read(
    client, db_session, storage, monkeypatch
) -> None:
    """The declared size busts the whole-video cap ⇒ 413 before a byte moves.

    REVERTED ⇒ the only remaining enforcement is the running per-part total, which
    fires part by part: a client declaring ten times the cap would push a full
    32 MiB part into the bucket before anything stopped it, and the staging tree
    would have to be swept after the fact rather than never written.
    """
    monkeypatch.setattr(uploads_video, "_VIDEO_MAX_BYTES", 64)
    project = await _own_project(client, db_session)

    r = await _create(client, project.id, size=65)

    assert r.status_code == 413, r.text
    assert "MiB limit" in r.json()["detail"]
    assert storage.puts == []
    assert (await db_session.execute(select(VideoSourceUpload))).scalars().all() == []


async def test_create_returns_the_slicing_plan_and_opens_one_session(
    client, db_session, storage
) -> None:
    """201 + the plan, and ``source_id`` is allocated BEFORE any byte moves.

    REVERTED (id minted at ``complete`` instead) ⇒ the parts are written under a
    key unrelated to the final object, so an interrupted ``complete`` leaves an
    orphaned staging tree nothing can tie back to the row that should have owned
    it.
    """
    project = await _own_project(client, db_session)

    r = await _create(client, project.id, size=100 * 1024 * 1024)

    assert r.status_code == 201, r.text
    body = r.json()
    assert body["part_size"] == uploads_video._UPLOAD_PART_BYTES
    # 100 MiB at 32 MiB/part is 4 parts (ceil, not floor — a floor would silently
    # drop the tail of every file whose size is not a multiple of the part size).
    assert body["part_count"] == 4
    assert body["expires_at"]

    row = await _reload_upload(db_session, body["upload_id"])
    assert row.project_id == project.id
    assert str(row.source_id) == body["source_id"]
    assert row.status == "open"
    assert row.parts == {}
    assert row.bytes_declared == 100 * 1024 * 1024


# --- part PUT -------------------------------------------------------------------


async def test_part_one_that_is_not_an_isobmff_container_is_400_and_stores_nothing(
    client, db_session, storage, staging_dir
) -> None:
    """The magic-byte gate on part 1 — the security half of this surface.

    Both the filename and the declared content type are attacker-controlled, and
    this bucket is readable back out through the artifact proxy.

    REVERTED ⇒ a renamed script lands in a bucket that is readable back out
    through the artifact proxy. The empty staging dir proves the partial write was
    reclaimed, and the empty ``puts`` proves the reject happened before storage
    was touched at all.
    """
    project = await _own_project(client, db_session)
    created = (await _create(client, project.id)).json()

    r = await _put_part(client, project.id, created["upload_id"], 1, b"not a video at all")

    assert r.status_code == 400, r.text
    assert "video container" in r.json()["detail"]
    assert storage.puts == []
    assert storage.objects == {}
    assert list(staging_dir.iterdir()) == []
    # And the ledger stays empty — a rejected part must not be counted as received.
    assert (await _reload_upload(db_session, created["upload_id"])).parts == {}


async def test_part_two_is_not_magic_checked_because_only_part_one_carries_ftyp(
    client, db_session, storage, monkeypatch
) -> None:
    """Parts 2..N are the middle of an mdat and must NOT be sniffed.

    REVERTED (sniff every part) ⇒ every legitimate multi-part upload 400s on part
    2, i.e. the entire feature is dead for exactly the files it was built for.
    """
    monkeypatch.setattr(uploads_video, "_UPLOAD_PART_BYTES", 32)
    project = await _own_project(client, db_session)
    created = (await _create(client, project.id, size=64)).json()

    first = await _put_part(client, project.id, created["upload_id"], 1, _mp4(b"a" * 8))
    assert first.status_code == 200, first.text
    r = await _put_part(client, project.id, created["upload_id"], 2, b"raw mdat continuation")

    assert r.status_code == 200, r.text
    assert r.json()["received_parts"] == 2


async def test_running_total_across_parts_is_enforced_from_the_persisted_ledger(
    client, db_session, storage, monkeypatch
) -> None:
    """The cumulative cap survives a pod hop — it is summed from ``parts`` in Postgres.

    The orchestrator runs replicas:2 behind an HPA, so part *n* and part *n+1*
    land on different pods essentially every time. Each request here already opens
    its OWN session, and the assertion below reads the ledger through a THIRD,
    independent session — so this pins the durable ledger, not an in-process
    counter.

    REVERTED (in-memory/per-pod counter) ⇒ the running total resets to zero on
    every pod hop and a client can stream unbounded bytes into the bucket, one
    part at a time, forever.
    """
    # 16-byte parts under a 40-byte whole-video cap: three legal part slots, but
    # the second part's bytes already bust the total.
    monkeypatch.setattr(uploads_video, "_UPLOAD_PART_BYTES", 16)
    monkeypatch.setattr(uploads_video, "_VIDEO_MAX_BYTES", 40)
    project = await _own_project(client, db_session)
    created = (await _create(client, project.id, size=40)).json()
    upload_id, source_id = created["upload_id"], created["source_id"]
    assert created["part_count"] == 3

    first = _mp4(b"a" * 8)  # 24 bytes
    assert (await _put_part(client, project.id, upload_id, 1, first)).status_code == 200
    # The ledger is DURABLE after part 1 — this is the state the next pod reads.
    assert (await _reload_upload(db_session, upload_id)).parts == {"1": len(first)}

    r = await _put_part(client, project.id, upload_id, 2, b"b" * 24)

    assert r.status_code == 413, r.text
    assert "MiB limit" in r.json()["detail"]
    row = await _reload_upload(db_session, upload_id)
    # The upload can never succeed now, so it is terminal and its staging tree is
    # swept in the same breath — its remaining parts must not keep filling the
    # bucket.
    assert row.status == "aborted"
    assert row.parts == {"1": len(first)}, "the rejected part must not enter the ledger"
    assert _parts_prefix(client, project.id, source_id, upload_id) in storage.prefixes
    # Part 2's bytes never reached storage: the cap check precedes the put.
    assert [key for key, _, _ in storage.puts] == [
        f"{_parts_prefix(client, project.id, source_id, upload_id)}00001"
    ]


async def test_retried_part_replaces_its_own_bytes_in_the_running_total(
    client, db_session, storage, monkeypatch
) -> None:
    """A re-PUT of part *n* overwrites it; the total is not charged twice.

    REVERTED (accumulate instead of replace) ⇒ a client on flaky wifi that retried
    one part three times 413s on a file that fits comfortably — i.e. the worse the
    connection, the more certainly the upload is refused, which is exactly
    backwards for the surface that exists to survive bad connections.
    """
    monkeypatch.setattr(uploads_video, "_UPLOAD_PART_BYTES", 16)
    monkeypatch.setattr(uploads_video, "_VIDEO_MAX_BYTES", 40)
    project = await _own_project(client, db_session)
    created = (await _create(client, project.id, size=40)).json()
    upload_id = created["upload_id"]

    payload = _mp4(b"a" * 8)  # 24 bytes; 3 x 24 would be 72 > 40
    for _ in range(3):
        r = await _put_part(client, project.id, upload_id, 1, payload)
        assert r.status_code == 200, r.text
        assert r.json()["received_parts"] == 1
        assert r.json()["received_bytes"] == len(payload)

    assert (await _reload_upload(db_session, upload_id)).parts == {"1": len(payload)}


async def test_part_number_outside_the_negotiated_range_is_400(client, db_session, storage) -> None:
    """Part numbers are bounded by the plan the server handed out.

    REVERTED ⇒ an arbitrary part index writes an object under a key ``complete``
    will never compose — unbounded staging garbage nobody reads or reaps.
    """
    project = await _own_project(client, db_session)
    created = (await _create(client, project.id, size=64)).json()

    assert (await _put_part(client, project.id, created["upload_id"], 0, _mp4())).status_code == 400
    assert (await _put_part(client, project.id, created["upload_id"], 9, _mp4())).status_code == 400
    assert storage.puts == []


# --- complete -------------------------------------------------------------------


async def test_complete_composes_parts_in_ascending_numeric_order(
    client, db_session, storage, monkeypatch
) -> None:
    """A 12-part upload composes 1,2,…,12 — never ``"10"`` before ``"2"``.

    ``compose_object``'s source order IS the byte order and it does NOT sort;
    ordering is the entire correctness of the composition, because the parts are
    opaque byte ranges.

    REVERTED (lexical sort anywhere up the chain) ⇒ silently corrupt video: the
    object is the right SIZE and uploads fine, ffprobe returns ``{}``, the
    duration lands as 0, and the DAG carries a permanently zeroed clip length
    downstream. Nothing flags it.

    Parts are PUT out of order here on purpose — arrival order is the client's
    business (parallel PUTs are the designed shape) and must not leak into the
    compose plan.
    """
    # A 16-byte part size so the plan really does allow a 2-digit part number
    # without moving 12 x 32 MiB. The declared size is the EXACT sum of the parts
    # below — ``complete`` enforces that, so a plan and a payload that disagree
    # would be rejected before the compose this test is about.
    monkeypatch.setattr(uploads_video, "_UPLOAD_PART_BYTES", 16)
    project = await _own_project(client, db_session)
    payloads = {n: (_mp4() if n == 1 else bytes([n]) * 16) for n in range(1, 12)}
    payloads[12] = bytes([12]) * 5  # a short FINAL part, as every real upload has
    created = (await _create(client, project.id, size=sum(map(len, payloads.values())))).json()
    upload_id, source_id = created["upload_id"], created["source_id"]
    assert created["part_count"] == 12
    for num in [1, 10, 2, 11, 3, 12, 4, 9, 5, 8, 6, 7]:
        assert (
            await _put_part(client, project.id, upload_id, num, payloads[num])
        ).status_code == 200, num

    r = await _complete(client, project.id, upload_id)

    assert r.status_code == 201, r.text
    prefix = _parts_prefix(client, project.id, source_id, upload_id)
    key, ordered, ctype = storage.composes[0]
    assert ordered == [f"{prefix}{n:05d}" for n in range(1, 13)]
    assert key == (
        f"tenants/{client.tenant_id}/projects/{project.id}/video-sources/{source_id}/original.mp4"
    )
    assert ctype == "video/mp4"
    # And the composed OBJECT — not just the argument list — is the parts in
    # numeric order. This is the assertion a lexical sort actually fails.
    assert storage.objects[key] == b"".join(payloads[n] for n in range(1, 13))


async def test_complete_writes_a_non_empty_probe_and_the_single_shot_response_shape(
    client, db_session, storage, real_clip, monkeypatch
) -> None:
    """The composed clip is measured, and the body matches the single-shot route.

    Two contracts in one test because they are the same promise: a chunked upload
    must be indistinguishable from a single-shot one once it lands.

    REVERTED (no read-back probe, or a header-only range fetch) ⇒ ``footage_probe``
    — the only consumer, which RE-PROBES NOTHING — reports zeroed duration and fps
    for that clip forever. It is a permanent loss of the measurement, not a
    deferral, and it is silent.

    REVERTED (its own response shape) ⇒ the frontend's single upload-completed
    handler and its one generated type fork in two.
    """
    clip = real_clip.read_bytes()
    # The generated fixture clip is ~10 KiB, so at the production 32 MiB part size
    # it would plan exactly ONE part and never exercise a compose at all. Shrink
    # the part size to half the clip so the plan is genuinely multi-part. Rounded
    # UP, not down: the route plans ``ceil(bytes / part_size)`` parts and requires
    # every one of them, so an odd-length clip split at ``len // 2`` would plan
    # THREE parts and this test would (correctly) be refused as incomplete.
    half = len(clip) - len(clip) // 2
    monkeypatch.setattr(uploads_video, "_UPLOAD_PART_BYTES", half)
    project = await _own_project(client, db_session)
    created = (await _create(client, project.id, name="real.mp4", size=len(clip))).json()
    upload_id, source_id = created["upload_id"], created["source_id"]
    assert created["part_count"] == 2

    # Split the REAL clip so the composed object is a genuinely decodable file:
    # a probe over a concatenation of junk would pass the "non-empty" assertion
    # for the wrong reason.
    assert (await _put_part(client, project.id, upload_id, 1, clip[:half])).status_code == 200
    assert (await _put_part(client, project.id, upload_id, 2, clip[half:])).status_code == 200

    r = await _complete(client, project.id, upload_id)

    assert r.status_code == 201, r.text
    body = r.json()
    expected_key = (
        f"tenants/{client.tenant_id}/projects/{project.id}/video-sources/{source_id}/original.mp4"
    )
    assert body["source_id"] == source_id
    assert body["filename"] == "real.mp4"
    # The authoritative size is the SUM OF PARTS, never the client's declaration.
    assert body["bytes"] == len(clip)
    assert body["preview_status"] == "pending"
    assert body["original_url"] == f"https://minio.test/{expected_key}?sig=abc"

    row = (
        await db_session.execute(select(VideoSource).where(VideoSource.id == UUID(source_id)))
    ).scalar_one()
    assert row.object_key == expected_key
    assert row.probe, "a decodable composed clip must not be stored with an empty probe blob"
    assert "format" in row.probe, "the blob must be ffprobe's OWN output, not a hand-rolled shape"

    # Byte-identical response SHAPE to the single-shot route — asserted against a
    # live single-shot upload rather than a hand-copied key list, so the two
    # cannot drift while both tests stay green.
    single = await client.post(
        f"/api/projects/{project.id}/video-sources",
        files={"file": ("single.mp4", clip, "video/mp4")},
    )
    assert single.status_code == 201, single.text
    assert set(body) == set(single.json())


async def test_complete_sweeps_the_staging_parts_after_a_successful_compose(
    client, db_session, storage
) -> None:
    """The staged parts are reclaimed once the final object exists.

    REVERTED ⇒ every completed upload leaves a full second copy of the clip in the
    bucket under a prefix nothing reads — the footage bucket doubles in size per
    shoot, and only bucket lifecycle (if configured) ever notices.
    """
    project = await _own_project(client, db_session)
    # The declared size is the exact length of the single part below: ``complete``
    # requires ``sum(parts) == bytes`` (a short tail is a silently truncated clip).
    created = (await _create(client, project.id, size=len(_mp4(b"z" * 8)))).json()
    upload_id, source_id = created["upload_id"], created["source_id"]
    assert (await _put_part(client, project.id, upload_id, 1, _mp4(b"z" * 8))).status_code == 200

    assert (await _complete(client, project.id, upload_id)).status_code == 201

    prefix = _parts_prefix(client, project.id, source_id, upload_id)
    assert prefix in storage.prefixes
    # Swept AFTER the compose returned (the final object exists, so the parts are
    # genuinely garbage) and the final object itself is untouched by the sweep.
    assert not [key for key in storage.objects if key.startswith(prefix)]
    assert any(key.endswith("/original.mp4") for key in storage.objects)


async def test_complete_skips_the_compose_when_the_final_object_already_exists(
    client, db_session, storage
) -> None:
    """``complete`` is IDEMPOTENT against an already-composed object.

    The state set up here is exactly what a SIGTERM between the compose and the
    commit leaves behind — routine, because the deployment rolls with
    ``maxSurge: 1``: the final object exists, the parts are gone, and no
    ``video_sources`` row was ever committed, so the upload is still ``open`` and
    the client retries.

    REVERTED ⇒ the retry recomposes from sources that no longer exist, which fails
    for good: a PERMANENT 500 on footage that is in fact safely stored in MinIO,
    plus an orphaned composed object under ``object_key`` with no row and no
    lifecycle rule to reclaim it. The parts are deleted below precisely so a
    reverted guard cannot pass by quietly re-stitching them.
    """
    project = await _own_project(client, db_session)
    payload = _mp4(b"k" * 8)
    created = (await _create(client, project.id, size=len(payload))).json()
    upload_id, source_id = created["upload_id"], created["source_id"]
    assert (await _put_part(client, project.id, upload_id, 1, payload)).status_code == 200

    object_key = (
        f"tenants/{client.tenant_id}/projects/{project.id}/video-sources/{source_id}/original.mp4"
    )
    # The dead attempt's leftovers: composed object present, staging tree swept.
    storage.objects[object_key] = payload
    staged = _parts_prefix(client, project.id, source_id, upload_id)
    for key in [k for k in storage.objects if k.startswith(staged)]:
        del storage.objects[key]

    r = await _complete(client, project.id, upload_id)

    assert r.status_code == 201, r.text
    assert storage.composes == [], "the object was already there; composing again is the bug"
    rows = (
        (await db_session.execute(select(VideoSource).where(VideoSource.project_id == project.id)))
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert rows[0].object_key == object_key
    assert (await _reload_upload(db_session, upload_id)).status == "completed"


async def test_the_staging_parts_are_swept_only_after_the_row_is_committed(
    client, db_session, storage, monkeypatch
) -> None:
    """The sweep runs AFTER the ``video_sources`` row is durable, never before.

    Asserted by reading the row from an INDEPENDENT session at the moment the
    sweep is invoked — under READ COMMITTED that query sees the row only if the
    handler's transaction has already committed, so this pins the real ordering
    rather than the order two mocks happen to be called in.

    REVERTED (sweep first, as it was) ⇒ a SIGTERM in that window — routine on a
    ``maxSurge: 1`` rolling deploy — or any commit failure rolls the row back with
    the parts ALREADY DELETED. The upload row is still ``open``, so the client
    retries, the completeness check still passes against the ledger, and the
    compose then fails forever on sources that no longer exist: an unrecoverable
    500 plus an orphaned composed object. A leftover part set is recoverable
    garbage; a missing one is not.
    """
    project = await _own_project(client, db_session)
    payload = _mp4(b"c" * 8)
    created = (await _create(client, project.id, size=len(payload))).json()
    upload_id, source_id = created["upload_id"], created["source_id"]
    assert (await _put_part(client, project.id, upload_id, 1, payload)).status_code == 200

    original_sweep = uploads_video._sweep_parts
    row_was_committed: list[bool] = []

    async def _observing_sweep(prefix: str) -> None:
        found = (
            await db_session.execute(select(VideoSource).where(VideoSource.id == UUID(source_id)))
        ).scalar_one_or_none()
        row_was_committed.append(found is not None)
        await original_sweep(prefix)

    monkeypatch.setattr(uploads_video, "_sweep_parts", _observing_sweep)

    assert (await _complete(client, project.id, upload_id)).status_code == 201

    assert row_was_committed == [True], "the parts were swept before the row was durable"
    assert _parts_prefix(client, project.id, source_id, upload_id) in storage.prefixes


async def test_a_compose_constraint_violation_is_a_409_naming_the_key_not_a_500(
    client, db_session, storage, monkeypatch
) -> None:
    """``ComposeConstraintError`` is translated, not leaked.

    It is raised by ``storage_service._assert_compose_part_sizes`` for a plan that
    S3/MinIO cannot legally assemble — most reachably, a client that ignores the
    negotiated ``part_size`` and PUTs non-final parts under the 5 MiB multipart
    floor. Every one of those parts clears the route's own caps, so nothing rejects
    them earlier.

    REVERTED (uncaught) ⇒ that client gets an opaque 500 and no way to learn which
    part was wrong, while ``storage_service``'s own docstring promises "a precise,
    human-readable 4xx naming the offending key". 409 rather than 400 because this
    request has no body to be malformed — what conflicts is the staged state — and
    the upload must stay ``open`` and its parts unswept so re-PUTting the offending
    part and calling ``complete`` again is the fix.
    """
    project = await _own_project(client, db_session)
    payload = _mp4(b"s" * 8)
    created = (await _create(client, project.id, size=len(payload))).json()
    upload_id, source_id = created["upload_id"], created["source_id"]
    assert (await _put_part(client, project.id, upload_id, 1, payload)).status_code == 200
    offending = f"{_parts_prefix(client, project.id, source_id, upload_id)}00001"

    async def _refuse(key: str, source_keys: list[str], content_type: str) -> str:
        raise ComposeConstraintError(
            f"compose source part 0 ({offending!r}) is 1048576 bytes; every part except "
            f"the last must be >= 5242880 bytes (5 MiB) or MinIO rejects the compose "
            f"with EntityTooSmall after the whole upload has been paid for"
        )

    monkeypatch.setattr(storage, "compose_object", _refuse)

    r = await _complete(client, project.id, upload_id)

    assert r.status_code == 409, r.text
    assert offending in r.json()["detail"], "the 4xx must name the part the client has to re-send"
    assert (
        await db_session.execute(select(VideoSource).where(VideoSource.id == UUID(source_id)))
    ).scalar_one_or_none() is None
    # Resumable: the ledger and the staged parts both survive a rejected compose.
    assert (await _reload_upload(db_session, upload_id)).status == "open"
    assert storage.prefixes == []


async def test_complete_with_four_of_sixteen_parts_is_400_and_composes_nothing(
    client, db_session, storage, monkeypatch
) -> None:
    """Completeness is measured against ``part_count``, not against ``max(part)``.

    REVERTED (contiguous-from-1 only, as it was) ⇒ 4 of 16 parts compose into a
    128 MiB FRAGMENT that is registered as a perfectly successful ``VideoSource``.
    ffprobe on a truncated MP4 returns ``{}`` (the ``moov`` atom sits at the end),
    ``_probe_video`` degrades to an empty blob exactly as designed, and
    ``footage_probe`` — which re-probes nothing — carries a zeroed duration for
    that clip forever. Nothing anywhere flags it: it is a valid file of a plausible
    size, which the module docstring calls the worst possible outcome.
    """
    monkeypatch.setattr(uploads_video, "_UPLOAD_PART_BYTES", 16)
    project = await _own_project(client, db_session)
    created = (await _create(client, project.id, size=16 * 16)).json()
    upload_id, source_id = created["upload_id"], created["source_id"]
    assert created["part_count"] == 16

    for num in range(1, 5):
        body = _mp4() if num == 1 else bytes([num]) * 16
        assert (await _put_part(client, project.id, upload_id, num, body)).status_code == 200, num

    r = await _complete(client, project.id, upload_id)

    assert r.status_code == 400, r.text
    assert "missing part(s) [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]" in r.json()["detail"]
    assert storage.composes == []
    assert (
        await db_session.execute(select(VideoSource).where(VideoSource.id == UUID(source_id)))
    ).scalar_one_or_none() is None
    # Resumable, and nothing swept — the operator already paid to send parts 1..4.
    assert (await _reload_upload(db_session, upload_id)).status == "open"
    assert storage.prefixes == []


async def test_complete_with_a_truncated_final_part_is_400_on_the_declared_size(
    client, db_session, storage, monkeypatch
) -> None:
    """All ``part_count`` parts present, but the bytes do not add up ⇒ 400.

    The count check cannot see a short TAIL: 16 of 16 parts arrived, the last one
    died mid-flight after 5 bytes, and every per-part rule is satisfied.

    REVERTED ⇒ the same permanently zeroed-duration clip as a missing part, only
    harder to spot, because the ledger looks complete. ``bytes`` declared at create
    is the only statement of "this is the whole file" that is ever made.
    """
    monkeypatch.setattr(uploads_video, "_UPLOAD_PART_BYTES", 16)
    project = await _own_project(client, db_session)
    created = (await _create(client, project.id, size=16 * 16)).json()
    upload_id, source_id = created["upload_id"], created["source_id"]

    for num in range(1, 17):
        body = _mp4() if num == 1 else bytes([num % 251]) * (5 if num == 16 else 16)
        assert (await _put_part(client, project.id, upload_id, num, body)).status_code == 200, num

    r = await _complete(client, project.id, upload_id)

    assert r.status_code == 400, r.text
    assert "declared 256 bytes but 245 arrived" in r.json()["detail"]
    assert storage.composes == []
    assert (
        await db_session.execute(select(VideoSource).where(VideoSource.id == UUID(source_id)))
    ).scalar_one_or_none() is None
    assert (await _reload_upload(db_session, upload_id)).status == "open"


async def test_complete_with_a_missing_part_is_400_and_creates_no_video_source(
    client, db_session, storage, monkeypatch
) -> None:
    """A gap in parts 1..N refuses the compose outright.

    REVERTED ⇒ the composed object is a clip with a hole punched in it that still
    plays for its first few seconds. That is the worst available outcome, because
    nothing downstream flags it — it is a valid file of a plausible size.
    """
    monkeypatch.setattr(uploads_video, "_UPLOAD_PART_BYTES", 16)
    project = await _own_project(client, db_session)
    created = (await _create(client, project.id, size=48)).json()
    upload_id, source_id = created["upload_id"], created["source_id"]
    assert created["part_count"] == 3

    assert (await _put_part(client, project.id, upload_id, 1, _mp4(b"a"))).status_code == 200
    assert (await _put_part(client, project.id, upload_id, 3, b"cccc")).status_code == 200

    r = await _complete(client, project.id, upload_id)

    assert r.status_code == 400, r.text
    assert "missing part(s) [2]" in r.json()["detail"]
    assert storage.composes == []
    assert (
        await db_session.execute(select(VideoSource).where(VideoSource.id == UUID(source_id)))
    ).scalar_one_or_none() is None
    # Still ``open``: a fixable client error must leave the upload resumable —
    # flipping it terminal would strand parts the operator already paid to send.
    assert (await _reload_upload(db_session, upload_id)).status == "open"


async def test_complete_with_no_parts_at_all_is_400(client, db_session, storage) -> None:
    """An upload nobody sent bytes for cannot be composed into an empty object."""
    project = await _own_project(client, db_session)
    created = (await _create(client, project.id, size=64)).json()

    r = await _complete(client, project.id, created["upload_id"])

    assert r.status_code == 400, r.text
    assert r.json()["detail"] == "no parts uploaded"
    assert storage.composes == []


async def test_two_concurrent_completes_yield_exactly_one_video_source_row(
    client, db_session, storage
) -> None:
    """The single-winner fence: ``SELECT … FOR UPDATE`` + a status transition.

    Both calls address the same pre-allocated ``source_id``, so without the lock
    both INSERT that primary key and the loser surfaces a duplicate-key 500 on a
    clip that uploaded perfectly (a double-click, a retry after a slow first
    attempt, or two browser tabs is all it takes).

    REVERTED ⇒ 500 on the loser, one composed object, and an upload row whose
    status depends on which transaction lost — plus a second compose against
    parts the winner has already swept.
    """
    project = await _own_project(client, db_session)
    created = (await _create(client, project.id, size=len(_mp4(b"y" * 8)))).json()
    upload_id, source_id = created["upload_id"], created["source_id"]
    assert (await _put_part(client, project.id, upload_id, 1, _mp4(b"y" * 8))).status_code == 200

    first, second = await asyncio.gather(
        _complete(client, project.id, upload_id),
        _complete(client, project.id, upload_id),
    )

    assert {first.status_code, second.status_code} == {201}, (first.text, second.text)
    # The loser returns the WINNER's row — that is what makes ``complete`` safely
    # retryable from a client that never saw the first response.
    assert first.json()["source_id"] == second.json()["source_id"] == source_id
    assert first.json() == second.json()
    rows = (
        (await db_session.execute(select(VideoSource).where(VideoSource.project_id == project.id)))
        .scalars()
        .all()
    )
    assert len(rows) == 1
    # Composed once, not twice: the loser never re-stitched already-swept parts.
    assert len(storage.composes) == 1
    assert (await _reload_upload(db_session, upload_id)).status == "completed"


async def test_parts_and_complete_are_refused_once_the_upload_is_terminal(
    client, db_session, storage
) -> None:
    """An aborted upload accepts no further parts and cannot be completed.

    REVERTED ⇒ a cancelled upload keeps accepting bytes into a staging tree the
    abort already swept, and can still materialise a ``VideoSource`` the operator
    explicitly cancelled.
    """
    project = await _own_project(client, db_session)
    created = (await _create(client, project.id, size=64)).json()
    upload_id = created["upload_id"]
    assert (
        await client.delete(f"/api/projects/{project.id}/video-sources/uploads/{upload_id}")
    ).status_code == 204

    assert (await _put_part(client, project.id, upload_id, 1, _mp4())).status_code == 409
    assert (await _complete(client, project.id, upload_id)).status_code == 409
    assert storage.puts == []
    assert storage.composes == []


# --- abort ----------------------------------------------------------------------


async def test_abort_open_upload_returns_204_and_flips_status_to_aborted(
    client, db_session, storage
) -> None:
    """Cancel sweeps the staged parts and marks the session terminal.

    Nothing else reaps staging promptly (the TTL is advisory and there is no
    reaper — see ``_UPLOAD_TTL``), so this is the only prompt reclamation path.

    REVERTED ⇒ every cancelled upload's parts sit in the bucket until (if ever)
    bucket lifecycle collects them, and the row stays ``open`` so the surface
    still accepts parts for an upload the user cancelled.
    """
    project = await _own_project(client, db_session)
    created = (await _create(client, project.id, size=64)).json()
    upload_id, source_id = created["upload_id"], created["source_id"]
    assert (await _put_part(client, project.id, upload_id, 1, _mp4(b"q" * 8))).status_code == 200

    r = await client.delete(f"/api/projects/{project.id}/video-sources/uploads/{upload_id}")

    assert r.status_code == 204, r.text
    assert r.content == b""
    prefix = _parts_prefix(client, project.id, source_id, upload_id)
    assert prefix in storage.prefixes
    assert not [key for key in storage.objects if key.startswith(prefix)]
    assert (await _reload_upload(db_session, upload_id)).status == "aborted"

    # Idempotent: a cancel that can fail is a cancel the UI has to explain.
    again = await client.delete(f"/api/projects/{project.id}/video-sources/uploads/{upload_id}")
    assert again.status_code == 204, again.text
    assert (await _reload_upload(db_session, upload_id)).status == "aborted"


# --- tenant isolation -----------------------------------------------------------


async def test_chunked_routes_on_another_tenants_project_are_404(
    client, db_session, storage
) -> None:
    """Part PUT / complete / abort against a foreign project are all 404.

    REVERTED ⇒ a cross-tenant WRITE into someone else's key tree: the part key is
    built from the path's ``project_id`` and the CALLER's tenant, so a handler
    that skipped ``_require_project`` would happily stage bytes under, and compose
    an object into, another tenant's project.

    404 and never 403: an id belonging to another tenant must be indistinguishable
    from a typo.
    """
    foreign = await _seed_project(db_session, tenant_id=uuid4(), subject_id=uuid4())
    upload_id = uuid7()

    assert (await _create(client, foreign.id)).status_code == 404
    assert (await _put_part(client, foreign.id, upload_id, 1, _mp4())).status_code == 404
    assert (await _complete(client, foreign.id, upload_id)).status_code == 404
    assert (
        await client.delete(f"/api/projects/{foreign.id}/video-sources/uploads/{upload_id}")
    ).status_code == 404
    assert storage.puts == []
    assert storage.composes == []
    assert storage.prefixes == []


async def test_an_upload_cannot_be_addressed_through_a_different_project(
    client, db_session, storage
) -> None:
    """Owning both projects is not enough to reach one's upload via the other's URL.

    REVERTED ⇒ the project segment of the key tree stops being load-bearing and an
    upload's parts can be composed into a project that never opened it.
    """
    project = await _own_project(client, db_session)
    other = await _own_project(client, db_session)
    created = (await _create(client, project.id, size=64)).json()
    upload_id = created["upload_id"]

    assert (await _put_part(client, other.id, upload_id, 1, _mp4())).status_code == 404
    assert (await _complete(client, other.id, upload_id)).status_code == 404
    assert (
        await client.delete(f"/api/projects/{other.id}/video-sources/uploads/{upload_id}")
    ).status_code == 404
    assert (await _reload_upload(db_session, upload_id)).status == "open"


# --- storage degradation --------------------------------------------------------


async def test_complete_on_a_backend_without_compose_is_503_and_never_truncates(
    client, db_session, monkeypatch
) -> None:
    """A backend that cannot compose is a 503 — never a single-part fallback.

    REVERTED (silent fallback to "just use part 1") ⇒ every multi-part clip is
    truncated to its first 32 MiB and registered as if it were whole.
    """

    class _NoCompose:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}

        async def put_file(self, key: str, path: str, content_type: str) -> str:
            self.objects[key] = Path(path).read_bytes()
            return key

        async def get_presigned_url(self, key: str) -> str:
            return f"https://minio.test/{key}"

    fake = _NoCompose()

    async def _get_storage() -> _NoCompose:
        return fake

    monkeypatch.setattr(uploads_video, "_get_storage", _get_storage)
    project = await _own_project(client, db_session)
    created = (await _create(client, project.id, size=len(_mp4(b"w" * 8)))).json()
    upload_id, source_id = created["upload_id"], created["source_id"]
    assert (await _put_part(client, project.id, upload_id, 1, _mp4(b"w" * 8))).status_code == 200

    r = await _complete(client, project.id, upload_id)

    assert r.status_code == 503, r.text
    assert (
        await db_session.execute(select(VideoSource).where(VideoSource.id == UUID(source_id)))
    ).scalar_one_or_none() is None
    # Still resumable — a transient backend fault must not burn the upload.
    assert (await _reload_upload(db_session, upload_id)).status == "open"


async def test_complete_stores_an_empty_probe_when_the_read_back_fails(
    client, db_session, storage, staging_dir, monkeypatch
) -> None:
    """A failed read-back costs the measurement and nothing else.

    REVERTED (raise instead of degrade) ⇒ a 5xx on an upload whose bytes are
    already durably composed in MinIO, i.e. the operator is told their footage
    failed when it did not, and retrying re-composes parts that were swept.
    """

    async def _no_read_back(key: str, local_path: str) -> bool:
        return False

    monkeypatch.setattr(storage, "get_file", _no_read_back)
    project = await _own_project(client, db_session)
    created = (await _create(client, project.id, size=len(_mp4(b"p" * 8)))).json()
    upload_id, source_id = created["upload_id"], created["source_id"]
    assert (await _put_part(client, project.id, upload_id, 1, _mp4(b"p" * 8))).status_code == 200

    r = await _complete(client, project.id, upload_id)

    assert r.status_code == 201, r.text
    row = (
        await db_session.execute(select(VideoSource).where(VideoSource.id == UUID(source_id)))
    ).scalar_one()
    assert row.probe == {}
    assert list(staging_dir.iterdir()) == [], "the local probe copy is reclaimed on the fail path"


# --- the single-shot route is untouched -----------------------------------------


async def test_single_shot_route_still_streams_and_registers_unchanged(
    client, db_session, storage, staging_dir
) -> None:
    """The pre-existing one-request upload behaves exactly as before.

    The chunked surface was added ALONGSIDE this route, reusing its helpers
    (``_require_project``, ``_source_prefix``, ``_is_isobmff_video``,
    ``_probe_video``) rather than re-deriving them.

    REVERTED (the chunked work having disturbed the shared helpers) ⇒ the 20 MiB
    single-request path — the right call for a small clip, and the one every
    existing client uses — regresses in the same commit that fixes large files.
    """
    project = await _own_project(client, db_session)
    data = _mp4(b"mdat-payload" * 4)

    r = await client.post(
        f"/api/projects/{project.id}/video-sources",
        files={"file": ("take-1.mp4", data, "video/mp4")},
    )

    assert r.status_code == 201, r.text
    body = r.json()
    expected_key = (
        f"tenants/{client.tenant_id}/projects/{project.id}"
        f"/video-sources/{body['source_id']}/original.mp4"
    )
    assert body["filename"] == "take-1.mp4"
    assert body["bytes"] == len(data)
    assert body["preview_status"] == "pending"
    assert body["original_url"] == f"https://minio.test/{expected_key}?sig=abc"
    # It streams to the SAME confined key tree, with no compose in sight.
    assert [key for key, _, _ in storage.puts] == [expected_key]
    assert storage.objects[expected_key] == data
    assert storage.composes == []
    assert list(staging_dir.iterdir()) == []

    row = (
        await db_session.execute(
            select(VideoSource).where(VideoSource.id == UUID(body["source_id"]))
        )
    ).scalar_one()
    assert row.object_key == expected_key
    assert row.bytes == len(data)
    assert row.preview_status == "pending"
