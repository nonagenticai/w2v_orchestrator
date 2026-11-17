"""P6 ORCH-6 — tenant offboarding (export + delete) on the REAL request path.

Real RS256 JWT → ``AuthMiddleware`` → ``TenancyMiddleware`` (membership) → the REAL
``/api/tenant/offboarding`` + ``/api/admin/...`` routers → real rows in Postgres in
EVERY orchestrator table → an in-memory own-bucket storage fake (the repo's
``tts_storage.set_storage_for_tests`` seam). The export background task really runs
(Starlette runs it before the ASGI call returns); its archive is opened and read back.

Non-vacuity: the seed puts at least one row of BOTH tenants in every table the plan
covers (asserted), and every refusal is paired with the call that succeeds.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import io
import json
import uuid
import zipfile
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from sqlalchemy import select, text

from src.api.routes import _common
from src.clients.schemas.licenses import ActiveLicenseView, ActivePlanRef
from src.config import clear_settings_cache, get_settings
from src.database.models import AdminAuditLog, Tenant, TenantOffboarding
from src.database.repositories.admin_repository import AdminRepository
from src.services import tenant_offboarding as svc
from src.services import tts_storage
from src.services.tenant_provisioning import TenantConflictError, create_tenant
from tests.api.routes._tenancy_harness import (  # noqa: F401 — fixtures
    Harness,
    harness_fixture,
    signing_key_fixture,
)

pytestmark = pytest.mark.asyncio


# ─── fakes ───────────────────────────────────────────────────────────────────


class _FakeStream:
    """The members ``minio_stream.StreamedLike`` reads."""

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
    """In-memory own bucket. ``fail_prefix_once`` makes ONE delete of that prefix
    fail (a storage outage mid-purge); ``signed`` counts presigns."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.opened: list[tuple[str, str]] = []
        self.signed = 0
        self.fail_prefix_once: str | None = None

    async def initialize(self) -> None:  # pragma: no cover - protocol filler
        return None

    async def store_stage_artifact(  # pragma: no cover - protocol filler
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        raise AssertionError("not used")

    async def put_object(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        self.objects[key] = data
        return key

    async def get_object_bytes(self, key: str) -> bytes | None:
        return self.objects.get(key)

    async def list_object_keys(self, prefix: str) -> list[str]:
        return [k for k in self.objects if k.startswith(prefix)]

    async def delete_object_prefix(self, prefix: str) -> bool:
        if self.fail_prefix_once == prefix:
            self.fail_prefix_once = None
            return False
        for k in [k for k in self.objects if k.startswith(prefix)]:
            del self.objects[k]
        return True

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        self.signed += 1
        return f"https://minio.test/{path}?X-Amz-Signature=s{self.signed}&exp={expires}"

    async def get_object_stream(
        self, bucket: str, object_key: str, request_headers: dict[str, str] | None = None
    ) -> _FakeStream:
        self.opened.append((bucket, object_key))
        return _FakeStream(self.objects[object_key])


class _Licences:
    """Scripted licences service: ``state`` is none | active | down."""

    def __init__(self) -> None:
        self.state = "none"

    async def get_active_license(self, *, tenant_id: str, app_key: str) -> ActiveLicenseView | None:
        if self.state == "down":
            raise RuntimeError("licences unavailable")
        if self.state == "active":
            return ActiveLicenseView(active=True, expires_at=None, plan=ActivePlanRef(code="std"))
        return None

    async def list_user_licenses(self, subject_id: str | None, *, tenant_id: str) -> list[Any]:
        return []


@pytest.fixture
def storage() -> Any:
    mem = _MemStorage()
    tts_storage.set_storage_for_tests(cast(Any, mem))
    yield mem
    tts_storage.set_storage_for_tests(None)


@pytest.fixture
def licences(harness: Harness) -> _Licences:
    stub = _Licences()
    harness.app.dependency_overrides[_common.get_licenses_client] = lambda: stub
    return stub


# ─── seed ────────────────────────────────────────────────────────────────────

_SEED = [
    "INSERT INTO projects (id, tenant_id, owner_subject_id, visibility, type, title, brief, "
    "brand, voice, theme) VALUES (:p, :t, :sub, 'private', 'generic', 'proj', "
    '\'{"topic": "x", "auth_config": {"password": "hunter2"}}\', \'{}\', \'{}\', \'{}\')',  # pragma: allowlist secret
    "INSERT INTO projects (id, tenant_id, owner_subject_id, visibility, type, title, brief, "
    "brand, voice, theme) VALUES (:p2, :t, :sub, 'private', 'generic', 'child', '{}', '{}', "
    "'{}', '{}')",
    "INSERT INTO deliverables (id, project_id, name, format) VALUES (:d, :p, 'main', '16x9')",
    "INSERT INTO runs (id, deliverable_id, blueprint_id, status, config_snapshot) VALUES "
    '(:r, :d, \'generic\', \'completed\', \'{"keep": 1, "provider_slots": {"llm": "v1"}, '
    '"brief": {"auth_info": {"password": "hunter2"}}}\')',  # pragma: allowlist secret
    "INSERT INTO stage_outputs (id, run_id, stage_name, status) VALUES (:so, :r, 's', 'completed')",
    "INSERT INTO stage_step_results (id, run_id, stage_name, step_key, input_hash, result) "
    "VALUES (:ssr, :r, 's', 'k', 'h', '{}')",
    "INSERT INTO annotations (id, stage_output_id, text) VALUES (:an, :so, 'note')",
    "INSERT INTO run_usage (id, run_id, stage_name) VALUES (:ru, :r, 's')",
    "INSERT INTO verbatim_items (id, project_id, text, position) VALUES (:vi, :p, 'v', 0)",
    "INSERT INTO source_ingest_cache (fingerprint, tenant_id, kind, ref) VALUES (:fp, :t, 'k', 'r')",
    "INSERT INTO storyboards (id, project_id) VALUES (:sb, :p)",
    "INSERT INTO storyboard_scenes (id, project_id, storyboard_id, position) VALUES (:sc, :p, :sb, 0)",
    "INSERT INTO storyboard_panels (id, project_id, storyboard_id, shot_key, position) "
    "VALUES (:pn, :p, :sb, 'a', 0)",
    "INSERT INTO storyboard_panel_events (id, project_id, panel_id, event_type, revision) "
    "VALUES (:pe, :p, :pn, 'approved', 1)",
    "INSERT INTO storyboard_assets (id, project_id, kind, name) VALUES (:sa, :p, 'character', 'c')",
    "INSERT INTO storyboard_panel_assets (id, panel_id, asset_id, role) "
    "VALUES (:spa, :pn, :sa, 'character')",
    "INSERT INTO storyboard_snapshots (id, project_id, storyboard_id, kind) "
    "VALUES (:ss, :p, :sb, 'manual')",
    "INSERT INTO storyboard_branches (id, root_project_id, child_project_id, name, base_snapshot_id) "
    "VALUES (:br, :p, :p2, 'b', :ss)",
    "INSERT INTO video_sources (id, project_id, filename, object_key, content_type, bytes) "
    "VALUES (:vs, :p, 'f.mp4', :vskey, 'video/mp4', 1)",
    "INSERT INTO video_source_uploads (id, project_id, source_id, filename, content_type, ext, "
    "bytes_declared, part_size, part_count) VALUES (:vu, :p, :vs, 'f.mp4', 'video/mp4', 'mp4', 1, 1, 1)",
    "INSERT INTO tenant_invites (id, tenant_id, email, token_hash, expires_at) "
    "VALUES (:ti, :t, :email, :hash32, now() + interval '1 day')",
    "INSERT INTO tenant_api_keys (id, tenant_id, name, prefix, key_hash) "
    "VALUES (:ak, :t, 'k', :keyprefix, :hash32)",
    "INSERT INTO tenant_connections (id, tenant_id, slot, current_version) VALUES (:tc, :t, 'llm', 1)",
    "INSERT INTO tenant_connection_versions (connection_id, version, kind, secret_ciphertext, "
    "key_id, secret_fingerprint) VALUES (:tc, 1, 'openai', :hash32, 'key-1', 'abcdef012345')",
    "INSERT INTO tenant_defaults (tenant_id) VALUES (:t)",
    "INSERT INTO tenant_sound_assets (id, tenant_id, sound_id, category, object_key, duration_s, "
    "bytes, content_type, rights_attested_by) VALUES (:snd, :t, 'tenant_bed', 'music', "
    ":sndkey, 1.0, 1, 'audio/wav', :sub)",
    "INSERT INTO service_runs (id, tenant_id, subject_id, slot, status, request) "
    "VALUES (:sr, :t, :sub, 'image', 'succeeded', '{\"prompt\": \"a fox\"}')",
    "INSERT INTO provisioning_audit (id, tenant_id, target_email, payload) "
    "VALUES (:pa, :t, :email, '{}')",
    "INSERT INTO signup_approvals (email, name, source, status, decided_at, tenant_slug, "
    "tenant_id) VALUES (:email, 'N', 'web', 'approved', now(), "
    "'sa-' || substr(md5(CAST(:t AS text)), 1, 12), :t)",
    "INSERT INTO request_audit (id, claim_tenant_id, effective_tenant_id, event, method, path, "
    "status_code, credential_fingerprint, detail) "
    "VALUES (:ra, :t, :t, 'mutation', 'POST', '/x', 200, 'fp', '{}')",
    "INSERT INTO admin_audit_log (id, action, target_tenant_id, actor_subject_id, detail) "
    "VALUES (:al, 'defaults.update', :t, :sub, '{}')",
]


async def _seed(
    h: Harness, tenant: uuid.UUID, admin: uuid.UUID, storage: _MemStorage
) -> dict[str, Any]:
    keys = [
        "p",
        "p2",
        "d",
        "r",
        "so",
        "ssr",
        "an",
        "ru",
        "vi",
        "sb",
        "sc",
        "pn",
        "pe",
        "sa",
        "spa",
        "ss",
        "br",
        "vs",
        "vu",
        "ti",
        "ak",
        "tc",
        "snd",
        "sr",
        "pa",
        "ra",
        "al",
    ]
    ids: dict[str, Any] = {k: uuid.uuid7() for k in keys}
    ids.update(
        t=tenant,
        sub=admin,
        fp=f"fp-{tenant}",
        email=f"x-{tenant}@acme.test",
        hash32=tenant.bytes * 2,
        keyprefix=base64.b32encode(tenant.bytes[-5:]).decode().lower(),
        sndkey=f"sound_library/tenants/{tenant}/music/{ids['snd']}.wav",
        vskey=f"tenants/{tenant}/projects/{ids['p']}/video-sources/{ids['vs']}/src.mp4",
    )
    async with h.factory() as s:
        for sql in _SEED:
            await s.execute(text(sql), ids)
        await s.commit()
    prefixes = [
        *svc.tenant_object_prefixes(tenant),
        *svc.run_object_prefixes(ids["r"]),
        *svc.project_object_prefixes(ids["p"]),
    ]
    for prefix in prefixes:
        storage.objects[f"{prefix}obj-{tenant}.bin"] = f"bytes of {tenant}".encode()
    return ids


async def _admin(h: Harness, tenant: uuid.UUID) -> uuid.UUID:
    sub = uuid.uuid7()
    await h.member(sub, tenant, "tenant_admin", email=f"admin-{sub}@acme.test")
    return sub


async def _super(h: Harness) -> str:
    home = await h.tenant("platform-home")
    sub = uuid.uuid7()
    await h.member(sub, home, "tenant_admin", email="root@platform.test")
    return h.token(sub, roles=["super_admin"], email="root@platform.test")


async def _counts(h: Harness, tenant: uuid.UUID) -> dict[str, int]:
    async with h.factory() as s:
        return await svc.row_counts(s, tenant)


async def _set_status(h: Harness, tenant: uuid.UUID, value: str) -> None:
    async with h.factory() as s:
        await s.execute(
            text("UPDATE tenants SET status = :v WHERE id = :t"), {"v": value, "t": tenant}
        )
        await s.commit()


async def _jobs(h: Harness, tenant: uuid.UUID) -> list[TenantOffboarding]:
    async with h.factory() as s:
        return list(
            (
                await s.execute(
                    select(TenantOffboarding)
                    .where(TenantOffboarding.tenant_id == tenant)
                    .order_by(TenantOffboarding.created_at)
                )
            )
            .scalars()
            .all()
        )


async def _export(h: Harness, token: str, tenant: uuid.UUID) -> dict[str, Any]:
    r = await h.call("POST", f"/api/admin/tenants/{tenant}/export", token)
    assert r.status_code == 202, r.text
    await svc.drain()
    got = await h.call("GET", f"/api/admin/offboarding/{r.json()['id']}", token)
    assert got.status_code == 200, got.text
    return got.json()


def _archive(storage: _MemStorage, job: dict[str, Any]) -> zipfile.ZipFile:
    [key] = [k for k in storage.objects if k.endswith(f"{job['id']}.zip")]
    return zipfile.ZipFile(io.BytesIO(storage.objects[key]))


def _rows(zf: zipfile.ZipFile, table: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in zf.read(f"data/{table}.jsonl").splitlines() if line]


class _Finished:
    """A 202-accepted delete, observed after its detached job finished: the original
    status code with the job as ``GET /api/admin/offboarding/{id}`` reports it."""

    def __init__(self, status_code: int, body: dict[str, Any], text: str) -> None:
        self.status_code, self._body, self.text = status_code, body, text

    def json(self) -> dict[str, Any]:
        return self._body


async def _delete(h: Harness, token: str, tenant: uuid.UUID, confirm: str, force: bool = False):
    q = f"?confirm={confirm}" + ("&force=true" if force else "")
    r = await h.call("DELETE", f"/api/admin/tenants/{tenant}{q}", token)
    if r.status_code != 202:
        return r
    assert r.json()["status"] == "running", r.text  # answered before the job ran
    await svc.drain()
    got = await h.call("GET", f"/api/admin/offboarding/{r.json()['id']}", token)
    assert got.status_code == 200, got.text
    return _Finished(202, got.json(), got.text)


async def _two_tenants(h: Harness, storage: _MemStorage) -> tuple[uuid.UUID, uuid.UUID, dict, dict]:
    a, b = await h.tenant("acme"), await h.tenant("beta")
    ids_a = await _seed(h, a, await _admin(h, a), storage)
    ids_b = await _seed(h, b, await _admin(h, b), storage)
    return a, b, ids_a, ids_b


# ─── tests ───────────────────────────────────────────────────────────────────


async def test_seed_covers_every_planned_table(harness: Harness, storage: _MemStorage) -> None:
    a, _b, _ia, _ib = await _two_tenants(harness, storage)
    counts = await _counts(harness, a)
    empty = sorted(t for t, n in counts.items() if n == 0 and t != "tenant_offboarding")
    assert not empty, (
        f"seed leaves planned tables empty — isolation checks would be vacuous: {empty}"
    )


async def test_tenant_admin_can_only_request(
    harness: Harness, storage: _MemStorage, licences
) -> None:
    a, b = await harness.tenant("acme"), await harness.tenant("beta")
    admin_a, admin_b = await _admin(harness, a), await _admin(harness, b)
    member = uuid.uuid7()
    await harness.member(member, a, "member")
    tok_a = harness.token(admin_a)

    r = await harness.call(
        "POST", "/api/tenant/offboarding", harness.token(member), json_body={"kind": "export"}
    )
    assert r.status_code == 403, r.text
    r = await harness.call(
        "POST", "/api/tenant/offboarding", tok_a, json_body={"kind": "delete", "reason": "leaving"}
    )
    assert r.status_code == 201, r.text
    assert r.json()["status"] == "requested"
    dup = await harness.call("POST", "/api/tenant/offboarding", tok_a, json_body={"kind": "delete"})
    assert dup.status_code == 409 and "OFFBOARDING_ALREADY_REQUESTED" in dup.text
    assert await harness.audit_actions(a) == ["offboarding.request"]

    # Every execution route is super_admin only (control: the request above worked).
    for method, path in (
        ("POST", f"/api/admin/tenants/{a}/export"),
        ("GET", f"/api/admin/tenants/{a}/offboarding-plan"),
        ("DELETE", f"/api/admin/tenants/{a}?confirm=acme&force=true"),
        ("POST", f"/api/admin/offboarding/{r.json()['id']}/approve"),
        ("POST", f"/api/admin/offboarding/{r.json()['id']}/reject"),
        ("GET", "/api/admin/offboarding"),
    ):
        denied = await harness.call(method, path, tok_a, json_body={} if method == "POST" else None)
        assert denied.status_code == 403, (method, path, denied.text)
    assert [j.status for j in await _jobs(harness, a)] == ["requested"]
    # B's admin does not see A's request.
    listed = await harness.call("GET", "/api/tenant/offboarding", harness.token(admin_b))
    assert listed.status_code == 200 and listed.json()["items"] == []
    own = await harness.call("GET", "/api/tenant/offboarding", tok_a)
    assert [i["kind"] for i in own.json()["items"]] == ["delete"]


async def test_export_round_trip(harness: Harness, storage: _MemStorage) -> None:
    a, b, _ia, ids_b = await _two_tenants(harness, storage)
    sup = await _super(harness)
    job = await _export(harness, sup, a)
    assert job["status"] == "done", job
    # A same-origin proxy path, never a presigned minio.test URL.
    assert job["download_url"] == f"/api/admin/offboarding/{job['id']}/download"
    assert storage.signed == 0
    got = await harness.call("GET", job["download_url"], sup)
    assert got.status_code == 200, got.text
    assert got.headers["content-type"] == "application/zip"
    assert "attachment" in got.headers["content-disposition"]
    assert storage.opened == [
        (get_settings().storage.minio_bucket, f"exports/tenants/{a}/{job['id']}.zip")
    ]
    assert zipfile.ZipFile(io.BytesIO(got.content)).namelist(), "the real archive bytes"
    [row] = [j for j in await _jobs(harness, a) if j.kind == "export"]
    assert row.artifact_key == f"exports/tenants/{a}/{job['id']}.zip", "only the KEY is stored"

    zf = _archive(storage, job)
    names = set(zf.namelist())
    seeded = await _counts(harness, a)
    for scope in svc.TENANT_TABLES:
        assert f"data/{scope.table}.jsonl" in names, scope.table
        if scope.table not in ("tenant_offboarding", "admin_audit_log"):
            assert len(_rows(zf, scope.table)) == seeded[scope.table], scope.table
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["tables"]["projects"] == 2
    blob = b"".join(zf.read(n) for n in names)
    assert str(b).encode() not in blob, "tenant B leaked into A's export"
    assert str(ids_b["p"]).encode() not in blob
    # Secrets are not exported (the manifest only NAMES the redacted keys).
    data = b"".join(zf.read(n) for n in names if n.startswith("data/"))
    assert b"hunter2" not in data and b"auth_config" not in data and b"auth_info" not in data
    assert b"provider_slots" not in data
    assert all("key_hash" not in r for r in _rows(zf, "tenant_api_keys"))
    assert all("token_hash" not in r for r in _rows(zf, "tenant_invites"))
    assert all("secret_ciphertext" not in r for r in _rows(zf, "tenant_connection_versions"))
    assert _rows(zf, "runs")[0]["config_snapshot"]["keep"] == 1  # control: the rest is there
    # Every object of A, none of B.
    objects = {n.removeprefix("objects/") for n in names if n.startswith("objects/")}
    expected = {k for k in storage.objects if k.endswith(f"obj-{a}.bin")}
    assert expected and objects == expected
    assert zf.read(f"objects/{sorted(expected)[0]}") == f"bytes of {a}".encode()
    assert "offboarding.export" in await harness.audit_actions(a)


async def test_delete_refusals_and_force(harness: Harness, storage: _MemStorage, licences) -> None:
    a, _b, ids_a, _ib = await _two_tenants(harness, storage)
    sup = await _super(harness)

    plan = (await harness.call("GET", f"/api/admin/tenants/{a}/offboarding-plan", sup)).json()
    assert plan["confirm_token"] == "acme" and plan["deletable"] is False
    assert {b["code"] for b in plan["blockers"]} == {"TENANT_NOT_SUSPENDED", "EXPORT_MISSING"}
    assert plan["rows"]["projects"] == 2 and any(
        p.startswith("tenants/") for p in plan["object_prefixes"]
    )

    r = await _delete(harness, sup, a, "wrong", force=True)
    assert r.status_code == 422 and "CONFIRMATION_MISMATCH" in r.text
    r = await _delete(harness, sup, a, "acme", force=True)
    assert r.status_code == 409 and "TENANT_NOT_SUSPENDED" in r.text, "force never skips suspension"

    await _set_status(harness, a, "suspended")
    r = await _delete(harness, sup, a, "acme")
    assert r.status_code == 409 and "EXPORT_MISSING" in r.text
    # An active run blocks even with force.
    async with harness.factory() as s:
        await s.execute(text("UPDATE runs SET status = 'running' WHERE id = :r"), {"r": ids_a["r"]})
        await s.commit()
    r = await _delete(harness, sup, a, "acme", force=True)
    assert r.status_code == 409 and "ACTIVE_RUNS" in r.text
    async with harness.factory() as s:
        await s.execute(
            text("UPDATE runs SET status = 'completed' WHERE id = :r"), {"r": ids_a["r"]}
        )
        await s.commit()
    assert (await _export(harness, sup, a))["status"] == "done"
    # A live licence (and an unverifiable one) blocks unless forced.
    licences.state = "active"
    r = await _delete(harness, sup, a, "acme")
    assert r.status_code == 409 and "LICENCE_ACTIVE" in r.text
    licences.state = "down"
    r = await _delete(harness, sup, a, "acme")
    assert r.status_code == 409 and "LICENCE_UNVERIFIED" in r.text
    assert await _counts(harness, a) != {}  # nothing was touched by any refusal
    assert (await _counts(harness, a))["projects"] == 2

    r = await _delete(harness, sup, a, "acme", force=True)
    assert r.status_code == 202, r.text
    assert r.json()["status"] == "done" and r.json()["force"] is True
    async with harness.factory() as s:
        details = (
            (
                await s.execute(
                    select(AdminAuditLog.detail).where(
                        AdminAuditLog.target_tenant_id == a,
                        AdminAuditLog.action == "offboarding.delete",
                    )
                )
            )
            .scalars()
            .all()
        )
    [rows_deleted] = [d for d in details if d.get("phase") == "rows_deleted"]
    assert rows_deleted["forced_over"] == ["LICENCE_UNVERIFIED"] and rows_deleted["force"] is True


async def test_partner_tenant_needs_force(
    harness: Harness, storage: _MemStorage, licences, monkeypatch
) -> None:
    partner_licence = uuid.uuid7()
    monkeypatch.setenv("W2V_PARTNER_LICENSE_IDS", str(partner_licence))
    clear_settings_cache()
    a = await harness.tenant("partnerco")
    async with harness.factory() as s:
        await s.execute(
            text("UPDATE tenants SET status = 'suspended', licence_ref = :l WHERE id = :t"),
            {"l": partner_licence, "t": a},
        )
        await s.commit()
    sup = await _super(harness)
    assert (await _export(harness, sup, a))["status"] == "done"
    r = await _delete(harness, sup, a, "partnerco")
    assert r.status_code == 409 and "PARTNER_TENANT" in r.text
    r = await _delete(harness, sup, a, "partnerco", force=True)
    assert r.status_code == 202, r.text


async def test_delete_isolation_completeness_and_idempotence(
    harness: Harness, storage: _MemStorage, licences
) -> None:
    a, b, ids_a, _ib = await _two_tenants(harness, storage)
    sup = await _super(harness)
    # B has an open request of its own — a job row that must survive A's delete.
    tok_b = harness.token(await _admin(harness, b))
    assert (
        await harness.call("POST", "/api/tenant/offboarding", tok_b, json_body={"kind": "export"})
    ).status_code == 201
    assert (await _export(harness, sup, a))["status"] == "done"
    await _set_status(harness, a, "suspended")
    before_b = await _counts(harness, b)
    objects_b = {k: v for k, v in storage.objects.items() if str(b) in k}
    assert objects_b

    r = await _delete(harness, sup, a, "acme")
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["status"] == "done"
    result = job["result"]
    assert result["complete"] is False
    assert {d["service"] for d in result["not_deleted"]} >= {
        "youtube_upload_service",
        "video_production_service",
        "tts_gateway",
        "video_compiler_service",
    }
    assert result["deleted_rows"]["projects"] == 2

    # A: no rows left except the deliberate retentions.
    after_a = await _counts(harness, a)
    assert after_a.pop("tenants") == 1
    assert after_a.pop("tenant_offboarding") == 2  # the export + the delete job
    audit_left = after_a.pop("admin_audit_log")
    assert all(n == 0 for n in after_a.values()), after_a
    actions = await harness.audit_actions(a)
    assert len(actions) == audit_left and all(x.startswith("offboarding.") for x in actions)
    assert "defaults.update" not in actions
    async with harness.factory() as s:
        tomb = await s.get(Tenant, a)
        assert tomb is not None and tomb.status == "deleted" and tomb.name is None
    # A: no objects left under any prefix, except the retained export archive.
    leftover = sorted(k for k in storage.objects if str(a) in k)
    assert leftover == [
        f"exports/tenants/{a}/{j.id}.zip" for j in await _jobs(harness, a) if j.kind == "export"
    ]
    for run_prefix in svc.run_object_prefixes(ids_a["r"]):
        assert not await storage.list_object_keys(run_prefix)
    # B: every table and every object untouched.
    assert await _counts(harness, b) == before_b
    assert {k: v for k, v in storage.objects.items() if str(b) in k} == objects_b

    # Idempotent: same job back, nothing new written. The name is gone, so the
    # confirm token is now the id (the plan says so).
    plan = (await harness.call("GET", f"/api/admin/tenants/{a}/offboarding-plan", sup)).json()
    assert plan["confirm_token"] == str(a) and plan["blockers"][0]["code"] == "ALREADY_DELETED"
    again = await _delete(harness, sup, a, str(a))
    assert again.status_code == 200 and again.json()["id"] == job["id"]
    assert await harness.audit_actions(a) == actions
    # The tombstone is final: no re-provisioning, no scope-switch, no export.
    async with harness.factory() as s:
        assert not await AdminRepository(s).tenant_switchable(a)
        with pytest.raises(TenantConflictError):
            await create_tenant(
                s, tenant_id=a, name="again", created_via="super_admin", actor_subject_id=None
            )
    r = await harness.call("POST", f"/api/admin/tenants/{a}/export", sup)
    assert r.status_code == 409 and "TENANT_DELETED" in r.text


async def test_delete_resumes_after_storage_failure_and_crash(
    harness: Harness, storage: _MemStorage, licences
) -> None:
    a, b, ids_a, _ib = await _two_tenants(harness, storage)
    sup = await _super(harness)
    assert (await _export(harness, sup, a))["status"] == "done"
    await _set_status(harness, a, "suspended")
    failing = svc.run_object_prefixes(ids_a["r"])[0]
    storage.fail_prefix_once = failing

    r = await _delete(harness, sup, a, "acme")
    assert r.status_code == 202 and r.json()["status"] == "failed", r.text
    [job] = [j for j in await _jobs(harness, a) if j.kind == "delete"]
    assert job.status == "failed" and failing in (job.error or "")
    assert job.progress["rows_deleted"] is True
    assert cast(list[str], job.progress["prefixes_pending"])[0] == failing
    assert job.progress["prefixes_done"], "prefixes before the failure were recorded as done"
    assert (await _counts(harness, a))["projects"] == 0, "phase 1 committed atomically"
    assert await storage.list_object_keys(failing), "the failed prefix still holds its object"

    # Simulate a crash mid-phase-2: a RUNNING job with a fresh heartbeat is refused …
    async with harness.factory() as s:
        await s.execute(
            text(
                "UPDATE tenant_offboarding SET status = 'running', updated_at = now() WHERE id = :j"
            ),
            {"j": job.id},
        )
        await s.commit()
    r = await _delete(harness, sup, a, str(a))
    assert r.status_code == 409 and "DELETE_IN_PROGRESS" in r.text
    # … and once its lease is stale it is re-claimed and resumed where it stopped.
    async with harness.factory() as s:
        await s.execute(
            text("UPDATE tenant_offboarding SET updated_at = :old WHERE id = :j"),
            {"j": job.id, "old": datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1)},
        )
        await s.commit()
    r = await _delete(harness, sup, a, str(a))
    assert r.status_code == 202, r.text
    assert r.json()["id"] == str(job.id) and r.json()["status"] == "done"
    assert not [k for k in storage.objects if str(a) in k and not k.startswith("exports/")]
    assert [k for k in storage.objects if str(b) in k], "control: B's objects remain"


async def test_approve_reject_flow(harness: Harness, storage: _MemStorage, licences) -> None:
    a, _b, _ia, _ib = await _two_tenants(harness, storage)
    sup = await _super(harness)
    tok = harness.token(await _admin(harness, a))
    exp = await harness.call("POST", "/api/tenant/offboarding", tok, json_body={"kind": "export"})
    dele = await harness.call("POST", "/api/tenant/offboarding", tok, json_body={"kind": "delete"})
    assert exp.status_code == dele.status_code == 201

    queue = (await harness.call("GET", "/api/admin/offboarding?status=requested", sup)).json()
    assert {i["id"] for i in queue["items"]} >= {exp.json()["id"], dele.json()["id"]}

    r = await harness.call(
        "POST", f"/api/admin/offboarding/{exp.json()['id']}/approve", sup, json_body={}
    )
    assert r.status_code == 202, r.text
    await svc.drain()
    # The tenant admin downloads its export; the deciding operator is not named.
    own = (await harness.call("GET", "/api/tenant/offboarding", tok)).json()["items"]
    [done] = [i for i in own if i["kind"] == "export"]
    assert done["status"] == "done" and done["decided_by"] is None
    assert done["download_url"] == f"/api/tenant/offboarding/{done['id']}/download"
    got = await harness.call("GET", done["download_url"], tok)
    assert got.status_code == 200, got.text
    assert got.content == storage.objects[f"exports/tenants/{a}/{done['id']}.zip"]
    # Another tenant's admin cannot download it, and nothing is read from MinIO.
    opened = len(storage.opened)
    other = await harness.call(
        "GET", done["download_url"], harness.token(await _admin(harness, _b))
    )
    assert other.status_code == 404, other.text
    assert len(storage.opened) == opened
    # An unfinished (requested) job has no download.
    pending = await harness.call(
        "GET", f"/api/tenant/offboarding/{dele.json()['id']}/download", tok
    )
    assert pending.status_code == 404, pending.text

    r = await harness.call(
        "POST",
        f"/api/admin/offboarding/{dele.json()['id']}/reject",
        sup,
        json_body={"reason": "no"},
    )
    assert r.status_code == 200 and r.json()["status"] == "rejected"
    r = await harness.call(
        "POST", f"/api/admin/offboarding/{dele.json()['id']}/approve", sup, json_body={}
    )
    assert r.status_code == 409 and "OFFBOARDING_NOT_PENDING" in r.text
    actions = await harness.audit_actions(a)
    assert actions.count("offboarding.request") == 2 and "offboarding.reject" in actions
    assert (await _counts(harness, a))["projects"] == 2, "a rejected delete deleted nothing"


# ─── adversarial review (2026-10-09) ─────────────────────────────────────────


async def test_cross_tenant_request_audit_belongs_to_the_accessed_tenant(
    harness: Harness, storage: _MemStorage, licences
) -> None:
    """A principal homed in A acting INTO B (``claim=A, effective=B``) is B's security
    record of that access. A's offboarding must neither hand it to A (it names B and
    B's paths) nor erase it from B's audit trail."""
    a, b, _ia, _ib = await _two_tenants(harness, storage)
    into_b = uuid.uuid7()
    async with harness.factory() as s:
        await s.execute(
            text(
                "INSERT INTO request_audit (id, claim_tenant_id, effective_tenant_id, event, "
                "method, path, status_code, cross_tenant, detail) VALUES (:id, :a, :b, "
                "'mutation', 'POST', :path, 200, true, '{}')"
            ),
            {"id": into_b, "a": a, "b": b, "path": f"/api/projects/secret-of-{b}"},
        )
        await s.commit()
    sup = await _super(harness)
    job = await _export(harness, sup, a)
    assert job["status"] == "done", job
    exported = {r["id"] for r in _rows(_archive(storage, job), "request_audit")}
    assert exported, "control: A's own request_audit rows are exported"
    assert str(into_b) not in exported, "B's access record leaked into A's export"

    await _set_status(harness, a, "suspended")
    r = await _delete(harness, sup, a, "acme")
    assert r.status_code == 202, r.text
    async with harness.factory() as s:
        kept = await s.execute(text("SELECT 1 FROM request_audit WHERE id = :id"), {"id": into_b})
        assert kept.first() is not None, "A's delete erased B's audit of a cross-tenant access"
        own = await s.execute(
            text("SELECT count(*) FROM request_audit WHERE effective_tenant_id = :a"), {"a": a}
        )
        assert own.scalar_one() == 0, "control: A's own request_audit rows are deleted"


class _SessionWatchingStorage(_MemStorage):
    """Records whether the export's DB session still holds a transaction while each
    object is downloaded (prod: ``idle_in_transaction_session_timeout = 5min``)."""

    def __init__(self) -> None:
        super().__init__()
        self.session: Any = None
        self.in_txn_during_fetch: list[bool] = []

    async def get_object_bytes(self, key: str) -> bytes | None:
        if not key.startswith("exports/"):
            self.in_txn_during_fetch.append(bool(self.session.in_transaction()))
        return await super().get_object_bytes(key)


async def test_export_does_not_hold_a_db_transaction_across_object_downloads(
    harness: Harness,
) -> None:
    watch = _SessionWatchingStorage()
    tts_storage.set_storage_for_tests(cast(Any, watch))
    try:
        a, _b, _ia, _ib = await _two_tenants(harness, watch)

        async def _beat(_progress: dict[str, object]) -> None:
            return None

        async with harness.factory() as s:
            watch.session = s
            summary = await svc.build_export_archive(s, watch, a, uuid.uuid7(), _beat)
        assert summary["object_count"] == len(watch.in_txn_during_fetch) > 0
        assert not any(watch.in_txn_during_fetch), (
            "the REPEATABLE READ snapshot is held open while MinIO objects stream — "
            "a large tenant trips idle_in_transaction_session_timeout and the export dies"
        )
    finally:
        tts_storage.set_storage_for_tests(None)


class _LocalBackendStorage(_MemStorage):
    """``w2v_common`` local backend semantics: ``delete_prefix`` of a prefix that holds
    nothing returns ``False`` (``dir_path.exists()`` is false)."""

    async def delete_object_prefix(self, prefix: str) -> bool:
        if not [k for k in self.objects if k.startswith(prefix)]:
            return False
        return await super().delete_object_prefix(prefix)


async def test_delete_succeeds_when_a_prefix_was_already_empty(harness: Harness, licences) -> None:
    local = _LocalBackendStorage()
    tts_storage.set_storage_for_tests(cast(Any, local))
    try:
        a, b, _ia, _ib = await _two_tenants(harness, local)
        empty = svc.tenant_object_prefixes(a)[-1]
        for k in [k for k in local.objects if k.startswith(empty)]:
            del local.objects[k]
        sup = await _super(harness)
        assert (await _export(harness, sup, a))["status"] == "done"
        await _set_status(harness, a, "suspended")
        r = await _delete(harness, sup, a, "acme")
        assert r.status_code == 202, r.text
        assert r.json()["status"] == "done"
        assert not [k for k in local.objects if str(a) in k and not k.startswith("exports/")]
        assert [k for k in local.objects if str(b) in k], "control: B's objects remain"
    finally:
        tts_storage.set_storage_for_tests(None)


class _GatedStorage(_MemStorage):
    """Blocks every prefix delete until ``release`` is set; ``entered`` fires when the
    job reaches its first object delete (phase 1 committed)."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def delete_object_prefix(self, prefix: str) -> bool:
        self.entered.set()
        await self.release.wait()
        return await super().delete_object_prefix(prefix)


async def _job_row(h: Harness, job_id: str) -> TenantOffboarding:
    async with h.factory() as s:
        job = await s.get(TenantOffboarding, uuid.UUID(job_id))
        assert job is not None
        return job


async def test_delete_answers_202_at_once_and_reports_progress(
    harness: Harness, licences, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delete can take minutes; behind Cloudflare (100 s → 524) the HTTP call must
    not wait for it. The job runs detached, heartbeats, and GET reports it."""
    gated = _GatedStorage()
    tts_storage.set_storage_for_tests(cast(Any, gated))
    monkeypatch.setattr(svc, "HEARTBEAT_SECONDS", 0.05)
    try:
        a, b, _ia, _ib = await _two_tenants(harness, gated)
        sup = await _super(harness)
        gated.release.set()  # the export also uses the storage
        assert (await _export(harness, sup, a))["status"] == "done"
        gated.release.clear()
        await _set_status(harness, a, "suspended")

        r = await harness.call("DELETE", f"/api/admin/tenants/{a}?confirm=acme", sup)
        assert r.status_code == 202, r.text
        assert r.json()["status"] == "running"
        job_id = r.json()["id"]
        await asyncio.wait_for(gated.entered.wait(), 10)
        first = await _job_row(harness, job_id)
        await asyncio.sleep(0.3)
        mid = (await harness.call("GET", f"/api/admin/offboarding/{job_id}", sup)).json()
        assert mid["status"] == "running" and mid["progress"]["rows_deleted"] is True
        assert mid["progress"]["prefixes_pending"], "objects still pending while blocked"
        assert (await _job_row(harness, job_id)).updated_at > first.updated_at, "heartbeat"

        gated.release.set()
        await svc.drain()
        done = (await harness.call("GET", f"/api/admin/offboarding/{job_id}", sup)).json()
        assert done["status"] == "done", done
        assert done["result"]["complete"] is False and done["result"]["not_deleted"]
        assert not [k for k in gated.objects if str(a) in k and not k.startswith("exports/")]
        assert [k for k in gated.objects if str(b) in k], "control: B's objects remain"
    finally:
        gated.release.set()
        await svc.drain()
        tts_storage.set_storage_for_tests(None)


async def test_delete_killed_mid_job_resumes_on_restart(harness: Harness, licences) -> None:
    from src.lifecycle.offboarding_resume import resume_offboarding_once

    gated = _GatedStorage()
    tts_storage.set_storage_for_tests(cast(Any, gated))
    try:
        a, b, _ia, _ib = await _two_tenants(harness, gated)
        sup = await _super(harness)
        gated.release.set()
        assert (await _export(harness, sup, a))["status"] == "done"
        gated.release.clear()
        await _set_status(harness, a, "suspended")
        r = await harness.call("DELETE", f"/api/admin/tenants/{a}?confirm=acme", sup)
        assert r.status_code == 202, r.text
        job_id = r.json()["id"]
        await asyncio.wait_for(gated.entered.wait(), 10)

        # The pod dies mid-purge: every in-process task is gone, the row says running.
        for task in list(svc._TASKS):
            task.cancel()
        await svc.drain()
        died = await _job_row(harness, job_id)
        assert died.status == "running" and died.progress["rows_deleted"] is True
        assert [k for k in gated.objects if str(a) in k and not k.startswith("exports/")]

        # A fresh lease is NOT stolen (another pod may still be working it) …
        gated.release.set()
        assert await resume_offboarding_once(harness.factory) == []
        # … an expired one is re-claimed at startup and finished from its progress.
        async with harness.factory() as s:
            await s.execute(
                text("UPDATE tenant_offboarding SET updated_at = :old WHERE id = :j"),
                {
                    "j": died.id,
                    "old": datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1),
                },
            )
            await s.commit()
        assert await resume_offboarding_once(harness.factory) == [died.id]
        await svc.drain()
        done = await _job_row(harness, job_id)
        assert done.status == "done", (done.status, done.error)
        assert done.result is not None and done.result["complete"] is False
        assert not [k for k in gated.objects if str(a) in k and not k.startswith("exports/")]
        assert [k for k in gated.objects if str(b) in k], "control: B's objects remain"
        assert await resume_offboarding_once(harness.factory) == [], "claimed once"
    finally:
        gated.release.set()
        await svc.drain()
        tts_storage.set_storage_for_tests(None)
