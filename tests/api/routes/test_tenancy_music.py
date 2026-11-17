"""P6 ORCH-4 — tenant music library on the REAL request path.

Real RS256 JWT → ``AuthMiddleware`` → ``TenancyMiddleware`` (membership) → the REAL
``/api/tenant/music`` router → real ``tenant_sound_assets`` / ``admin_audit_log`` rows
in Postgres → the manifest written to an in-memory storage fake (the repo's
``tts_storage.set_storage_for_tests`` seam) → read back by the REAL
``sound_catalog.load_sound_library``. ffprobe is real (CI installs ffmpeg).

Non-vacuity: every refusal asserts nothing was stored AND is paired with a control
through the same call shape that does store.
"""

from __future__ import annotations

import asyncio
import io
import json
import uuid
import wave
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx
import pytest
from sqlalchemy import select

from src.api.routes.music import list_music_tracks
from src.config import clear_settings_cache, get_settings
from src.database.models import (
    AdminAuditLog,
    Deliverable,
    Project,
    Run,
    TenantSoundAsset,
)
from src.services import tts_storage
from src.services.stages import sound_catalog
from src.services.stages.sound_catalog import load_sound_library, tenant_manifest_key
from tests.api.routes._tenancy_harness import (  # noqa: F401 — fixtures
    _PUBLIC_PEER,
    Harness,
    harness_fixture,
    signing_key_fixture,
)

pytestmark = pytest.mark.asyncio

PATH = "/api/tenant/music"


class _FakeStream:
    """The members ``minio_stream.StreamedLike`` reads. MinIO's stored type is
    deliberately generic so the route's ``media_type`` override is observable."""

    def __init__(self, payload: bytes) -> None:
        self.status = 200
        self.headers = {
            "content-type": "application/octet-stream",
            "content-length": str(len(payload)),
        }
        self._payload = payload

    async def iter_chunks(self, chunk_size: int) -> AsyncIterator[bytes]:
        for start in range(0, len(self._payload), chunk_size):
            yield self._payload[start : start + chunk_size]

    async def aclose(self) -> None:
        return None


class _MemStorage:
    """In-memory own-bucket fake. ``put_object`` yields to the loop so concurrent
    writers genuinely interleave; presigns carry a counter so a re-sign is visible."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.opened: list[tuple[str, str]] = []
        self.signed = 0
        self.fail_put_suffix: str | None = None

    async def initialize(self) -> None:  # pragma: no cover - protocol filler
        return None

    async def store_stage_artifact(  # pragma: no cover - protocol filler
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        raise AssertionError("not used")

    async def put_object(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        await asyncio.sleep(0.02)
        if self.fail_put_suffix and key.endswith(self.fail_put_suffix):
            raise OSError("minio down")
        self.objects[key] = (data, content_type)
        return key

    async def get_object_bytes(self, key: str) -> bytes | None:
        hit = self.objects.get(key)
        return hit[0] if hit else None

    async def delete_object(self, key: str) -> bool:
        self.objects.pop(key, None)
        return True

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        self.signed += 1
        return f"https://minio.test/{path}?sig={self.signed}"

    async def get_object_stream(
        self, bucket: str, object_key: str, request_headers: dict[str, str] | None = None
    ) -> _FakeStream:
        self.opened.append((bucket, object_key))
        return _FakeStream(self.objects[object_key][0])

    def manifest(self, tenant: uuid.UUID) -> dict[str, Any]:
        raw = self.objects.get(tenant_manifest_key(str(tenant)))
        return json.loads(raw[0]) if raw else {}

    def audio_keys(self, tenant: uuid.UUID) -> set[str]:
        prefix = f"sound_library/tenants/{tenant}/"
        return {k for k in self.objects if k.startswith(prefix) and not k.endswith(".json")}


@pytest.fixture
def storage() -> Any:
    mem = _MemStorage()
    sound_catalog.clear_sound_library_cache()
    tts_storage.set_storage_for_tests(cast(Any, mem))
    yield mem
    tts_storage.set_storage_for_tests(None)
    sound_catalog.clear_sound_library_cache()


def _wav(seconds: float = 0.5) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * int(8000 * seconds))
    return buf.getvalue()


async def _user(h: Harness, tenant: uuid.UUID, role: str = "tenant_admin") -> tuple[str, uuid.UUID]:
    sub = uuid.uuid7()
    await h.member(sub, tenant, role, email=f"{role}-{sub}@acme.test")
    return h.token(sub), sub


async def _upload(
    h: Harness,
    bearer: str,
    data: bytes,
    *,
    filename: str = "Our Anthem.wav",
    content_type: str = "audio/wav",
    **form: str,
) -> httpx.Response:
    fields = {"rights_attested": "true", **form}
    transport = httpx.ASGITransport(app=h.app, client=_PUBLIC_PEER)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        return await c.post(
            PATH,
            files={"file": (filename, data, content_type)},
            data=fields,
            headers={"Authorization": f"Bearer {bearer}"},
        )


async def _rows(h: Harness, tenant: uuid.UUID) -> list[TenantSoundAsset]:
    async with h.factory() as s:
        return list(
            (
                await s.execute(
                    select(TenantSoundAsset)
                    .where(TenantSoundAsset.tenant_id == tenant)
                    .order_by(TenantSoundAsset.created_at)
                )
            )
            .scalars()
            .all()
        )


async def _audits(h: Harness, tenant: uuid.UUID, action: str) -> list[AdminAuditLog]:
    async with h.factory() as s:
        return list(
            (
                await s.execute(
                    select(AdminAuditLog).where(
                        AdminAuditLog.target_tenant_id == tenant, AdminAuditLog.action == action
                    )
                )
            )
            .scalars()
            .all()
        )


async def _run(h: Harness, tenant: uuid.UUID, status: str) -> uuid.UUID:
    async with h.factory() as s:
        project = Project(
            id=uuid.uuid7(),
            tenant_id=tenant,
            owner_subject_id=uuid.uuid7(),
            visibility="private",
            type="promo",
            title="p",
        )
        deliverable = Deliverable(id=uuid.uuid7(), project_id=project.id, name="d", format="16x9")
        run = Run(
            id=uuid.uuid7(), deliverable_id=deliverable.id, blueprint_id="promo_v1", status=status
        )
        s.add_all([project, deliverable, run])
        await s.commit()
        return run.id


async def _set_run_status(h: Harness, run_id: uuid.UUID, status: str) -> None:
    async with h.factory() as s:
        run = await s.get(Run, run_id)
        assert run is not None
        run.status = status
        await s.commit()


# ─── upload → library → pipeline listing ─────────────────────────────────


async def test_upload_lands_in_the_tenant_library_only(harness: Harness, storage) -> None:
    a, b = await harness.tenant(), await harness.tenant()
    admin, admin_sub = await _user(harness, a)
    member, _ = await _user(harness, a, "member")

    r = await _upload(harness, admin, _wav(0.5), description="our anthem")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["sound_id"] == "tenant_our_anthem"
    assert body["category"] == "music"
    assert body["rights"] == "tenant-declared"
    assert body["rights_attested_by"] == str(admin_sub)
    assert body["content_type"] == "audio/wav"
    assert abs(body["duration_s"] - 0.5) < 0.05, "real ffprobe duration"

    [row] = await _rows(harness, a)
    assert row.object_key == f"sound_library/tenants/{a}/music/{row.id}.wav"
    assert storage.audio_keys(a) == {row.object_key}
    manifest = storage.manifest(a)
    assert manifest == {
        "tenant_our_anthem": {
            "key": row.object_key,
            "description": "our anthem",
            "category": "music",
            "duration_s": row.duration_s,
            "retired": False,
        }
    }
    assert len(await _audits(harness, a, "music.upload")) == 1

    lib_a = await load_sound_library(tenant_id=str(a), use_cache=False)
    lib_b = await load_sound_library(tenant_id=str(b), use_cache=False)
    assert "tenant_our_anthem" in lib_a.ids_in("music")
    assert lib_a.ids_in("music") > lib_b.ids_in("music")  # bundled beds still there
    assert "corporate_uplift" in lib_a.ids_in("music")
    assert lib_b.get("tenant_our_anthem") is None

    # The studio picker endpoint lists it for A, not for B.
    tracks_a = {t.id for t in (await list_music_tracks(tenant_id=a)).tracks}
    tracks_b = {t.id for t in (await list_music_tracks(tenant_id=b)).tracks}
    assert "tenant_our_anthem" in tracks_a
    assert "tenant_our_anthem" not in tracks_b

    # A member lists it. The url is the same-origin proxy path, NOT a presigned
    # MinIO URL (whose minio.test host a browser cannot resolve).
    r1 = await harness.call("GET", PATH, member)
    assert r1.status_code == 200, r1.text
    [t1] = r1.json()["tracks"]
    assert t1["url"] == f"{PATH}/{row.id}/content"
    assert storage.signed == 0, "the list never mints a presigned MinIO URL"
    # ...and that path really serves the track's bytes to the member, from the
    # own bucket, with the row's (sniffed) type rather than MinIO's stored one.
    played = await harness.call("GET", t1["url"], member)
    assert played.status_code == 200, played.text
    assert played.content == storage.objects[row.object_key][0]
    assert played.headers["content-type"] == "audio/wav"
    assert storage.opened[-1] == (get_settings().storage.minio_bucket, row.object_key)
    # Tenant B's member sees nothing of A's — neither in the list nor via the proxy.
    b_member, _ = await _user(harness, b, "member")
    rb = await harness.call("GET", PATH, b_member)
    assert rb.status_code == 200 and rb.json()["tracks"] == []
    opened = len(storage.opened)
    rb = await harness.call("GET", t1["url"], b_member)
    assert rb.status_code == 404, rb.text
    assert len(storage.opened) == opened, "a foreign track never reaches MinIO"


async def test_list_exposes_the_enforced_upload_limits(
    harness: Harness, storage, monkeypatch
) -> None:
    tenant = await harness.tenant()
    member, _ = await _user(harness, tenant, "member")
    monkeypatch.setenv("STORAGE_TENANT_MUSIC_MAX_BYTES", "12345")
    clear_settings_cache()
    r = await harness.call("GET", PATH, member)
    assert r.status_code == 200, r.text
    limits = r.json()["limits"]
    assert limits["max_bytes"] == 12345, "read from settings, not hardcoded"
    assert limits["max_duration_s"] is None
    assert limits["max_description_length"] == 500
    assert limits["categories"] == ["accent", "ambient", "music", "transition"]
    assert {f["format"]: f["content_type"] for f in limits["formats"]} == {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
    }
    # The advertised cap is the enforced one: one byte over it is refused.
    admin, _ = await _user(harness, tenant)
    monkeypatch.setenv("STORAGE_TENANT_MUSIC_MAX_BYTES", str(len(_wav()) - 1))
    clear_settings_cache()
    advertised = (await harness.call("GET", PATH, member)).json()["limits"]["max_bytes"]
    r = await _upload(harness, admin, _wav())
    assert r.status_code == 422 and r.json()["error"]["details"]["max_bytes"] == advertised, r.text


async def test_content_of_a_deleted_track_is_404(harness: Harness, storage) -> None:
    tenant = await harness.tenant()
    admin, _ = await _user(harness, tenant)
    r = await _upload(harness, admin, _wav())
    assert r.status_code == 201, r.text
    url = r.json()["url"]
    assert (await harness.call("GET", url, admin)).status_code == 200  # control
    assert (await harness.call("DELETE", f"{PATH}/{r.json()['id']}", admin)).status_code == 204
    gone = await harness.call("GET", url, admin)
    assert gone.status_code == 404, gone.text


async def test_type_is_sniffed_not_trusted(harness: Harness, storage) -> None:
    tenant = await harness.tenant()
    admin, _ = await _user(harness, tenant)

    # Non-audio bytes with an audio name + header: refused, nothing stored.
    r = await _upload(
        harness, admin, b"<html>not audio</html>" * 10, filename="x.mp3", content_type="audio/mpeg"
    )
    assert r.status_code == 422, r.text
    assert "UNSUPPORTED_AUDIO" in r.text
    # Magic bytes right but not decodable: refused by the ffprobe gate.
    r = await _upload(harness, admin, b"RIFF\x00\x00\x00\x00WAVEjunkjunk", filename="y.wav")
    assert r.status_code == 422, r.text
    assert "UNDECODABLE_AUDIO" in r.text
    assert await _rows(harness, tenant) == []
    assert storage.audio_keys(tenant) == set()

    # Control: real WAV bytes under a misleading name AND header are accepted and
    # stored as what they really are.
    r = await _upload(
        harness, admin, _wav(), filename="bed.mp3", content_type="image/png", slug="bed"
    )
    assert r.status_code == 201, r.text
    assert r.json()["content_type"] == "audio/wav"
    [row] = await _rows(harness, tenant)
    assert row.object_key.endswith(".wav")


async def test_size_cap_and_attestation(harness: Harness, storage, monkeypatch) -> None:
    tenant = await harness.tenant()
    admin, _ = await _user(harness, tenant)
    data = _wav(0.5)  # ~8 KB
    monkeypatch.setenv("STORAGE_TENANT_MUSIC_MAX_BYTES", str(len(data) - 1))
    clear_settings_cache()
    r = await _upload(harness, admin, data, slug="big")
    assert r.status_code == 422, r.text
    assert "limit" in r.text

    monkeypatch.setenv("STORAGE_TENANT_MUSIC_MAX_BYTES", str(len(data)))
    clear_settings_cache()
    r = await _upload(harness, admin, data, slug="big", rights_attested="false")
    assert r.status_code == 422, r.text
    assert "rights" in r.text
    assert await _rows(harness, tenant) == []
    assert storage.audio_keys(tenant) == set()

    r = await _upload(harness, admin, data, slug="big")  # control: exactly at the cap
    assert r.status_code == 201, r.text


async def test_member_cannot_write(harness: Harness, storage) -> None:
    tenant = await harness.tenant()
    admin, _ = await _user(harness, tenant)
    member, _ = await _user(harness, tenant, "member")

    r = await _upload(harness, member, _wav())
    assert r.status_code == 403, r.text
    assert await _rows(harness, tenant) == [] and storage.objects == {}

    r = await _upload(harness, admin, _wav())  # control
    assert r.status_code == 201, r.text
    track_id = r.json()["id"]
    r = await harness.call("PATCH", f"{PATH}/{track_id}", member, json_body={"description": "x"})
    assert r.status_code == 403, r.text
    r = await harness.call("DELETE", f"{PATH}/{track_id}", member)
    assert r.status_code == 403, r.text
    [row] = await _rows(harness, tenant)
    assert row.deleted_at is None and row.description == ""


async def test_other_tenant_cannot_touch_the_track(harness: Harness, storage) -> None:
    a, b = await harness.tenant(), await harness.tenant()
    admin_a, _ = await _user(harness, a)
    admin_b, _ = await _user(harness, b)
    r = await _upload(harness, admin_a, _wav())
    track_id = r.json()["id"]

    r = await harness.call("DELETE", f"{PATH}/{track_id}", admin_b)
    assert r.status_code == 404, r.text
    r = await harness.call("PATCH", f"{PATH}/{track_id}", admin_b, json_body={"description": "x"})
    assert r.status_code == 404, r.text
    [row] = await _rows(harness, a)
    assert row.deleted_at is None and row.object_key in storage.objects

    r = await harness.call("PATCH", f"{PATH}/{track_id}", admin_a, json_body={"description": "y"})
    assert r.status_code == 200, r.text  # control
    assert storage.manifest(a)["tenant_our_anthem"]["description"] == "y"


async def test_duplicate_slug_is_409(harness: Harness, storage) -> None:
    tenant = await harness.tenant()
    admin, _ = await _user(harness, tenant)
    assert (await _upload(harness, admin, _wav(), slug="bed")).status_code == 201
    r = await _upload(harness, admin, _wav(), slug="bed")
    assert r.status_code == 409, r.text
    assert len(await _rows(harness, tenant)) == 1
    assert len(storage.audio_keys(tenant)) == 1, "the refused upload's object is not stranded"


# ─── delete + the pinning rule ───────────────────────────────────────────


async def test_delete_without_active_runs_removes_everything(harness: Harness, storage) -> None:
    tenant = await harness.tenant()
    admin, _ = await _user(harness, tenant)
    await _run(harness, tenant, "completed")  # terminal runs do not pin
    track_id = (await _upload(harness, admin, _wav())).json()["id"]

    r = await harness.call("DELETE", f"{PATH}/{track_id}", admin)
    assert r.status_code == 204, r.text
    assert await _rows(harness, tenant) == []
    assert storage.audio_keys(tenant) == set()
    assert storage.manifest(tenant) == {}
    [audit] = await _audits(harness, tenant, "music.delete")
    assert audit.detail is not None and audit.detail["object_deleted"] is True
    lib = await load_sound_library(tenant_id=str(tenant), use_cache=False)
    assert lib.get("tenant_our_anthem") is None


async def test_delete_under_an_active_run_retires_until_the_run_ends(
    harness: Harness, storage
) -> None:
    tenant = await harness.tenant()
    admin, _ = await _user(harness, tenant)
    track_id = (await _upload(harness, admin, _wav())).json()["id"]
    run_id = await _run(harness, tenant, "running")

    r = await harness.call("DELETE", f"{PATH}/{track_id}", admin)
    assert r.status_code == 204, r.text
    [row] = await _rows(harness, tenant)
    assert row.deleted_at is not None
    assert row.object_key in storage.objects, "the running run can still fetch it"
    assert storage.manifest(tenant)["tenant_our_anthem"]["retired"] is True
    lib = await load_sound_library(tenant_id=str(tenant), use_cache=False)
    assert "tenant_our_anthem" not in lib.ids_in("music"), "never offered for a new pick"
    assert lib.resolves("tenant_our_anthem", category="music"), "a pinned run still resolves it"
    assert lib.key_for("tenant_our_anthem") == row.object_key
    # Gone from the tenant's list; the id cannot be reused while it is retiring.
    assert (await harness.call("GET", PATH, admin)).json()["tracks"] == []
    assert (await _upload(harness, admin, _wav())).status_code == 409

    # The run ends; the next library change purges the retired track.
    await _set_run_status(harness, run_id, "completed")
    assert (await _upload(harness, admin, _wav(), slug="next")).status_code == 201
    rows = await _rows(harness, tenant)
    assert [r.sound_id for r in rows] == ["tenant_next"]
    assert row.object_key not in storage.objects
    assert set(storage.manifest(tenant)) == {"tenant_next"}


async def test_concurrent_uploads_give_one_consistent_manifest(harness: Harness, storage) -> None:
    tenant = await harness.tenant()
    admin, _ = await _user(harness, tenant)
    results = await asyncio.gather(
        *(_upload(harness, admin, _wav(), slug=f"bed_{i}") for i in range(4))
    )
    assert [r.status_code for r in results] == [201] * 4, [r.text for r in results]
    rows = await _rows(harness, tenant)
    assert {r.sound_id for r in rows} == {f"tenant_bed_{i}" for i in range(4)}
    manifest = storage.manifest(tenant)
    assert set(manifest) == {r.sound_id for r in rows}
    assert {v["key"] for v in manifest.values()} == {r.object_key for r in rows}


async def test_manifest_write_failure_rolls_back_and_cleans_up(harness: Harness, storage) -> None:
    tenant = await harness.tenant()
    admin, _ = await _user(harness, tenant)
    storage.fail_put_suffix = "manifest.json"
    r = await _upload(harness, admin, _wav())
    assert r.status_code == 503, r.text
    assert await _rows(harness, tenant) == []
    assert storage.audio_keys(tenant) == set()
    storage.fail_put_suffix = None
    assert (await _upload(harness, admin, _wav())).status_code == 201  # control


# ─── review fixes ────────────────────────────────────────────────────────


async def test_an_id3_wrapped_playlist_is_not_accepted_as_audio(
    harness: Harness, storage, tmp_path
) -> None:
    """ffprobe auto-detects the demuxer AFTER skipping an ID3v2 tag, so ``ID3`` +
    an HLS playlist sniffs as ``mp3`` and (on ffmpeg builds without the m3u8
    extension check) is probed as HLS — following a ``file://`` reference on the
    pod and reporting ITS duration. Stored, it is a playlist the compiler would
    open over http. The probe must decode the bytes as the sniffed container."""
    tenant = await harness.tenant()
    admin, _ = await _user(harness, tenant)
    target = tmp_path / "target.wav"
    target.write_bytes(_wav(2.0))
    playlist = (
        b"#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:2,\n"
        + f"file://{target}\n".encode()
        + b"#EXT-X-ENDLIST\n"
        + b"\n" * 64
    )
    polyglot = b"ID3\x03\x00\x00\x00\x00\x00\x00" + playlist
    r = await _upload(harness, admin, polyglot, filename="x.mp3", slug="poly")
    assert r.status_code == 422, r.text
    assert "UNDECODABLE_AUDIO" in r.text
    assert await _rows(harness, tenant) == []
    assert storage.audio_keys(tenant) == set()


async def test_patch_is_audited(harness: Harness, storage) -> None:
    tenant = await harness.tenant()
    admin, sub = await _user(harness, tenant)
    r = await _upload(harness, admin, _wav(), slug="bed")
    track_id = r.json()["id"]
    r = await harness.call(
        "PATCH",
        f"{PATH}/{track_id}",
        admin,
        json_body={"description": "new", "category": "ambient"},
    )
    assert r.status_code == 200, r.text
    [audit] = await _audits(harness, tenant, "music.update")
    assert audit.actor_subject_id == sub
    assert audit.detail["id"] == track_id
    assert audit.detail["changed"] == ["category", "description"]


async def _bed_for(library: Any, monkeypatch: pytest.MonkeyPatch, music_id: str) -> tuple:
    """Drive the REAL ``resolve_audio_bed`` with a run that pinned ``music_id``."""
    from src.services.runs.run_context import RunContext
    from src.services.stages.shared import sound_design

    async def _load(*_a: object, **_k: object) -> Any:
        return library

    async def _presign(sfx_id: str) -> str:
        return f"https://minio.test/{library.key_for(sfx_id)}?sig=x"

    monkeypatch.setattr(sound_design, "load_sound_library", _load)
    monkeypatch.setattr(sound_design, "presign_sound_asset", _presign)
    cfg = {"enabled": True, "music": music_id, "ambient": False}
    ctx = RunContext(
        run_id=uuid.uuid7(),
        deliverable_id=uuid.uuid7(),
        config_snapshot={"sound_effects": cfg},
        prior_run_outputs={},
    )
    degradations: list[Any] = []
    bed = await sound_design.resolve_audio_bed(
        ctx, [{"seconds": 5.0}], cfg, degradations=degradations
    )
    return bed, degradations


async def test_patch_category_under_an_active_run_keeps_the_pinned_pick(
    harness: Harness, storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run pins its config for its whole lifetime: a running run that picked the
    track as music keeps its bed after an admin re-files it as ambient. New picks
    see only the live category; the pin is cleared once no run is active."""
    tenant = await harness.tenant()
    admin, _ = await _user(harness, tenant)
    track_id = (await _upload(harness, admin, _wav(), slug="bed")).json()["id"]
    run_id = await _run(harness, tenant, "running")

    r = await harness.call("PATCH", f"{PATH}/{track_id}", admin, json_body={"category": "ambient"})
    assert r.status_code == 200, r.text
    assert r.json()["category"] == "ambient"
    [row] = await _rows(harness, tenant)
    assert row.pinned_categories == ["music"]
    assert storage.manifest(tenant)["tenant_bed"]["pinned_categories"] == ["music"]

    lib = await load_sound_library(tenant_id=str(tenant), use_cache=False)
    assert lib.resolves("tenant_bed", category="music"), "the running run's pick resolves"
    assert not lib.is_valid("tenant_bed", category="music"), "never a NEW music pick"
    assert "tenant_bed" not in lib.ids_in("music")
    assert "tenant_bed" in lib.ids_in("ambient")
    bed, degradations = await _bed_for(lib, monkeypatch, "tenant_bed")
    assert bed is not None and bed["music"]["url"].startswith(
        f"https://minio.test/{row.object_key}"
    ), "the run keeps its music bed"
    assert degradations == []

    # The run ends; the next library change clears the pin.
    await _set_run_status(harness, run_id, "completed")
    assert (await _upload(harness, admin, _wav(), slug="next")).status_code == 201
    row = next(r for r in await _rows(harness, tenant) if r.sound_id == "tenant_bed")
    assert row.pinned_categories == []
    assert "pinned_categories" not in storage.manifest(tenant)["tenant_bed"]
    lib = await load_sound_library(tenant_id=str(tenant), use_cache=False)
    assert not lib.resolves("tenant_bed", category="music")


async def test_patch_category_without_an_active_run_pins_nothing(
    harness: Harness, storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control for the test above: same PATCH, no active run — the old category is
    gone at once and a run pinning it drops the bed (recorded as a degradation)."""
    tenant = await harness.tenant()
    admin, _ = await _user(harness, tenant)
    await _run(harness, tenant, "completed")
    track_id = (await _upload(harness, admin, _wav(), slug="bed")).json()["id"]
    r = await harness.call("PATCH", f"{PATH}/{track_id}", admin, json_body={"category": "ambient"})
    assert r.status_code == 200, r.text
    [row] = await _rows(harness, tenant)
    assert row.pinned_categories == []
    assert "pinned_categories" not in storage.manifest(tenant)["tenant_bed"]
    lib = await load_sound_library(tenant_id=str(tenant), use_cache=False)
    assert not lib.resolves("tenant_bed", category="music")
    bed, degradations = await _bed_for(lib, monkeypatch, "tenant_bed")
    assert not (bed or {}).get("music")
    assert len(degradations) == 1


@pytest.mark.parametrize("run_status", ["completed"])
async def test_concurrent_deletes_of_one_track_are_204_then_404(
    harness: Harness, storage, run_status: str
) -> None:
    """Two DELETEs of the same track race: one wins (204), the other must see the
    winner's commit and answer 404 — not 500 from a stale UPDATE (purge path) and
    not a second 204 + duplicate audit (retire path)."""
    tenant = await harness.tenant()
    admin, _ = await _user(harness, tenant)
    await _run(harness, tenant, run_status)
    track_id = (await _upload(harness, admin, _wav(), slug="bed")).json()["id"]
    results = await asyncio.gather(
        harness.call("DELETE", f"{PATH}/{track_id}", admin),
        harness.call("DELETE", f"{PATH}/{track_id}", admin),
    )
    assert sorted(r.status_code for r in results) == [204, 404], [r.text for r in results]
    assert len(await _audits(harness, tenant, "music.delete")) == 1
    rows = await _rows(harness, tenant)
    if run_status == "running":
        assert len(rows) == 1 and rows[0].deleted_at is not None
    else:
        assert rows == [] and storage.audio_keys(tenant) == set()
