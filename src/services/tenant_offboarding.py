"""Tenant offboarding (P6 ORCH-6): export a tenant's data, then delete it.

The single source of truth for WHAT a tenant owns in the orchestrator:

* :data:`TENANT_TABLES` — every orchestrator table, each with the predicate that
  selects tenant ``:t``'s rows and what deletion does to them (``delete``,
  ``delete_keep_offboarding``, ``tombstone``, ``retain``). Ordered leaf-first, so a
  child's selector still finds its parents when it is deleted. A table missing here
  fails ``tests/services/test_tenant_offboarding_coverage.py`` — a new table cannot be
  added without deciding how a tenant's rows in it are exported and deleted.
* :func:`tenant_object_prefixes` / :func:`run_object_prefixes` /
  :func:`project_object_prefixes` — every object-key tree in the orchestrator's bucket
  that holds tenant data (enumerated from the writers; see each helper).

Export (:func:`run_export`) writes ONE consistent snapshot (a REPEATABLE READ
transaction) to ``exports/tenants/{t}/{job}.zip``: ``data/<table>.jsonl`` per table
(secrets redacted — see :data:`_EXPORT_DROP_COLUMNS` / :func:`_redact`), every object
under the tenant's prefixes at ``objects/<key>``, and ``manifest.json``. Only the key is
stored; a download URL is presigned per read.

Delete (:func:`run_delete`) is resumable. Phase 1 (one transaction): re-check the
tenant has no active run, record the object prefixes (run/project ids are gone
afterwards), delete every tenant row, tombstone the ``tenants`` row and audit — all or
nothing. Phase 2: purge each prefix, committing progress after each one, and verify the
prefix is empty. A crash or a storage error leaves ``progress`` saying exactly what is
left; re-running the delete continues from there.

What this does NOT delete is reported, never hidden: :data:`DOWNSTREAM_NOT_DELETED`
(other w2v services keep their own tenant rows — blueprint limitation) and
:data:`RETAINED` (the tombstone, the offboarding trail, export archives).
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import logging
import shutil
import tempfile
import zipfile
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from uuid import UUID, uuid7

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import get_settings
from src.database.models import AdminAuditLog, Tenant, TenantOffboarding
from src.database.repositories.admin_audit_repository import AuditRepository, member_actor_clause
from src.database.session import bind_session_tenant
from src.services.auth_secrets import AUTH_BLOCK_KEYS
from src.services.document_storage import document_prefix
from src.services.license_provisioning import W2V_LICENCE_APP_KEY
from src.services.license_status import get_license_status
from src.services.stages.sound_catalog import tenant_prefix as sound_library_prefix
from src.services.storage_service import MEDIA_CACHE_ROOT
from src.services.tenant_sound_library import tenant_has_active_runs
from src.services.tts_storage import package_namespace

logger = logging.getLogger(__name__)

#: A ``running`` job whose ``updated_at`` is older than this is presumed crashed and
#: may be re-claimed (by a re-POST/re-DELETE, or by :func:`resume_stale_jobs`). A live
#: job's :func:`_heartbeating` wrapper bumps ``updated_at`` every
#: :data:`HEARTBEAT_SECONDS`, independent of how long one storage call takes.
RUNNING_LEASE_SECONDS = 180
HEARTBEAT_SECONDS = 30

Mode = Literal["delete", "delete_keep_offboarding", "tombstone", "retain"]


@dataclass(frozen=True)
class TableScope:
    table: str
    #: SQL predicate selecting tenant ``:t``'s rows (unqualified columns of ``table``).
    where: str
    mode: Mode
    why: str = ""


_P = "SELECT id FROM projects WHERE tenant_id = :t"
# Constant SQL composed from constants only; ``:t`` is the sole bound value.
_D = f"SELECT id FROM deliverables WHERE project_id IN ({_P})"  # noqa: S608
_R = f"SELECT id FROM runs WHERE deliverable_id IN ({_D})"  # noqa: S608
_SO = f"SELECT id FROM stage_outputs WHERE run_id IN ({_R})"  # noqa: S608
_PANELS = f"SELECT id FROM storyboard_panels WHERE project_id IN ({_P})"  # noqa: S608
_CONNS = "SELECT id FROM tenant_connections WHERE tenant_id = :t"

#: Every orchestrator table, leaf-first (the deletion order).
TENANT_TABLES: tuple[TableScope, ...] = (
    TableScope(
        "annotations", f"stage_output_id IN ({_SO}) OR applied_in_run_id IN ({_R})", "delete"
    ),
    TableScope("run_usage", f"run_id IN ({_R})", "delete"),
    TableScope("stage_step_results", f"run_id IN ({_R})", "delete"),
    TableScope("stage_outputs", f"run_id IN ({_R})", "delete"),
    TableScope("storyboard_panel_assets", f"panel_id IN ({_PANELS})", "delete"),
    TableScope("storyboard_panel_events", f"project_id IN ({_P})", "delete"),
    TableScope(
        "storyboard_branches",
        f"root_project_id IN ({_P}) OR child_project_id IN ({_P})",
        "delete",
    ),
    TableScope("storyboard_snapshots", f"project_id IN ({_P})", "delete"),
    TableScope("storyboard_panels", f"project_id IN ({_P})", "delete"),
    TableScope("storyboard_scenes", f"project_id IN ({_P})", "delete"),
    TableScope("storyboard_assets", f"project_id IN ({_P})", "delete"),
    TableScope("storyboards", f"project_id IN ({_P})", "delete"),
    TableScope("video_source_uploads", f"project_id IN ({_P})", "delete"),
    TableScope("video_sources", f"project_id IN ({_P})", "delete"),
    TableScope("verbatim_items", f"project_id IN ({_P})", "delete"),
    TableScope("runs", f"deliverable_id IN ({_D})", "delete"),
    TableScope("deliverables", f"project_id IN ({_P})", "delete"),
    TableScope("projects", "tenant_id = :t", "delete"),
    TableScope("source_ingest_cache", "tenant_id = :t", "delete"),
    TableScope(
        "service_runs",
        "tenant_id = :t",
        "delete",
        why="service-lab bench runs (a prompt, its output text, a media key): the tenant "
        "can regenerate any of it, and the media objects live under "
        "media-cache/tenants/<t>/ which tenant_object_prefixes already erases",
    ),
    TableScope("tenant_connection_versions", f"connection_id IN ({_CONNS})", "delete"),
    TableScope("tenant_connections", "tenant_id = :t", "delete"),
    TableScope("tenant_sound_assets", "tenant_id = :t", "delete"),
    TableScope("tenant_defaults", "tenant_id = :t", "delete"),
    TableScope("tenant_api_keys", "tenant_id = :t", "delete"),
    TableScope("tenant_invites", "tenant_id = :t", "delete"),
    TableScope("tenant_members", "tenant_id = :t", "delete"),
    TableScope("provisioning_audit", "tenant_id = :t", "delete"),
    TableScope(
        "signup_approvals",
        "tenant_id = :t AND NOT (status = 'removed' "
        "AND coalesce(removal->>'state', '') NOT IN ('done','resolved_manually'))",
        "delete",
        why="the early-access request that founded :t (the requester's name, email, "
        "message): exported and erased with the tenant it founded — EXCEPT a row whose "
        "REMOVAL is still running (services/signup/removal.py): it holds that cleanup's "
        "own state and erases the personal data itself when it finishes. A finished "
        "(done / resolved_manually) removal row goes with the tenant like any other",
    ),
    TableScope(
        "request_audit",
        "effective_tenant_id = :t",
        "delete",
        why="a row with claim_tenant_id = :t but another effective tenant is a principal "
        "of :t acting INTO that tenant — the accessed tenant's security record, never "
        "exported to nor erased with :t",
    ),
    TableScope(
        "admin_audit_log",
        "target_tenant_id = :t",
        "delete_keep_offboarding",
        why="offboarding.* rows are kept: they are the record that the deletion happened",
    ),
    TableScope(
        "tenant_offboarding",
        "tenant_id = :t",
        "retain",
        why="the job rows are the proof of deletion and hold no tenant content",
    ),
    TableScope(
        "tenants",
        "id = :t",
        "tombstone",
        why="status='deleted', name cleared: the id can never be re-provisioned",
    ),
)

#: Columns never exported (secret material or credential digests).
_EXPORT_DROP_COLUMNS: dict[str, tuple[str, ...]] = {
    "tenant_api_keys": ("key_hash",),
    "tenant_invites": ("token_hash",),
    "tenant_connection_versions": ("secret_ciphertext", "key_id", "secret_fingerprint"),
    "request_audit": ("credential_fingerprint",),
}

#: Downstream services that keep their OWN rows for a tenant (blueprint limitation:
#: each needs an SA-gated ``DELETE /api/internal/tenants/{t}`` follow-up). Enumerated
#: 2026-10-09 from each service's models (tables carrying ``tenant_id``).
DOWNSTREAM_NOT_DELETED: tuple[dict[str, object], ...] = (
    {"service": "youtube_upload_service", "data": ["oauth_tokens", "upload_jobs"]},
    {"service": "video_compiler_service", "data": ["compile_jobs"]},
    {
        "service": "video_production_service",
        "data": [
            "recording_jobs",
            "recording_usage",
            "compilation_jobs",
            "compilation_batches",
            "compilation_usage",
            "custom_presets",
            "compilation_templates",
        ],
    },
    {
        "service": "ai_content_generation_service",
        "data": [
            "video_generation_jobs",
            "video_generation_usage",
            "imagegen_jobs",
            "imagegen_batches",
        ],
    },
    {
        "service": "analysis_service",
        "data": [
            "analysis_jobs",
            "analysis_usage",
            "user_context_profiles",
            "brand_extraction_logs",
            "brands",
            "brand_versions",
            "brand_overrides",
            "logo_assets",
            "critique_jobs",
            "critique_usage",
            "critique_batches",
            "critique_findings",
            "finding_suppressions",
            "critique_schedules",
            "audit_logs",
            "icp_extractions",
            "review_jobs",
            "review_usage",
            "review_batches",
            "analyze_review_jobs",
        ],
    },
    {"service": "app_inspector_service", "data": ["inspection_jobs", "inspection_usage"]},
    {"service": "bdd_playwright_service", "data": ["bdd_jobs", "bdd_usage", "narration_templates"]},
    {
        "service": "content_service",
        "data": [
            "storyline_jobs",
            "narration_jobs",
            "script_jobs",
            "script_batches",
            "script_usage",
        ],
    },
    {
        "service": "frontend_test_runner_service",
        "data": ["test_execution_jobs", "test_runner_usage"],
    },
    {
        "service": "html_render_service",
        "data": [
            "redesign_jobs",
            "redesign_usage",
            "custom_styles",
            "batch_jobs",
            "webhook_delivery_log",
        ],
    },
    {"service": "slide_service", "data": ["slide_templates", "deck_jobs"]},
    {"service": "tts_gateway", "data": ["tts_jobs"]},
    {"service": "video_analyzer_service", "data": ["analysis_jobs"]},
    {"service": "licenses", "data": ["the tenant's licences"]},
    {"service": "keycloak", "data": ["member user accounts (identities are not tenant-owned)"]},
)


def retained(tenant_id: UUID) -> list[dict[str, str]]:
    """What deletion deliberately keeps, and why."""
    return [
        {"what": "tenants row (tombstone, status=deleted, name cleared)", "why": _why("tenants")},
        {"what": "tenant_offboarding job rows", "why": _why("tenant_offboarding")},
        {"what": "admin_audit_log rows with action offboarding.*", "why": _why("admin_audit_log")},
        {
            "what": f"export archives under {export_prefix(tenant_id)}",
            "why": "a handed-over download link must keep working; no lifecycle rule "
            "expires exports/ yet — delete them by hand once delivered",
        },
    ]


def _why(table: str) -> str:
    return next(s.why for s in TENANT_TABLES if s.table == table)


# ─── object prefixes ─────────────────────────────────────────────────────────


def tenant_object_prefixes(tenant_id: UUID) -> list[str]:
    """Every tenant-keyed object tree in the own bucket (each writer cited)."""
    t = str(tenant_id)
    return [
        f"tenants/{t}/",  # uploads_video: video sources + chunked uploads
        f"{sound_library_prefix(t)}/",  # tenant music library (ORCH-4)
        f"{MEDIA_CACHE_ROOT}/tenants/{t}/",  # cross-run media + ingest cache
        f"{document_prefix(t)}/",  # document uploads
        f"artifacts/manuscripts/{t}/",  # manuscript_storage
        f"artifacts/thumbnails/{t}/",  # tts_storage.store_thumbnail_override
        f"artifacts/storyboard_assets/{t}/",  # tts_storage.store_storyboard_asset_image
        f"artifacts/voiceovers/{t}/",  # tts_storage.store_voiceover_override
        f"artifacts/{package_namespace(t)}/",  # video package uploads
    ]


def run_object_prefixes(run_id: UUID) -> list[str]:
    """Run-keyed trees: stage artifacts (``store_stage_artifact``) + final videos."""
    return [f"artifacts/{run_id}/", f"videos/{run_id}/"]


def project_object_prefixes(project_id: UUID) -> list[str]:
    """Project-keyed trees: storyboard panel images (``render._namespace``)."""
    return [f"artifacts/storyboard/{project_id}/"]


def export_prefix(tenant_id: UUID) -> str:
    return f"exports/tenants/{tenant_id}/"


def export_key(tenant_id: UUID, job_id: UUID) -> str:
    return f"{export_prefix(tenant_id)}{job_id}.zip"


async def all_object_prefixes(session: AsyncSession, tenant_id: UUID) -> list[str]:
    """Tenant + run + project prefixes, from the rows as they are NOW."""
    runs = (await session.execute(text(_R), {"t": tenant_id})).scalars().all()
    projects = (await session.execute(text(_P), {"t": tenant_id})).scalars().all()
    prefixes = tenant_object_prefixes(tenant_id)
    for run_id in sorted(runs, key=str):
        prefixes.extend(run_object_prefixes(run_id))
    for project_id in sorted(projects, key=str):
        prefixes.extend(project_object_prefixes(project_id))
    return prefixes


# ─── storage ─────────────────────────────────────────────────────────────────


class OffboardingStorage(Protocol):
    async def put_object(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str: ...

    async def get_object_bytes(self, key: str) -> bytes | None: ...

    async def list_object_keys(self, prefix: str) -> list[str]: ...

    async def delete_object_prefix(self, prefix: str) -> bool: ...


_STORAGE_METHODS = (
    "put_object",
    "get_object_bytes",
    "list_object_keys",
    "delete_object_prefix",
)


def as_offboarding_storage(storage: object) -> OffboardingStorage | None:
    """``storage`` when it has every method offboarding needs, else ``None``."""
    if all(callable(getattr(storage, m, None)) for m in _STORAGE_METHODS):
        return cast(OffboardingStorage, storage)
    return None


async def _fetch_to_file(storage: OffboardingStorage, key: str, path: str) -> bool:
    """Stream ``key`` to ``path`` (``get_file`` when the backend has it — a large
    video never lands in RAM), else via bytes. False when the object is gone."""
    get_file = getattr(storage, "get_file", None)
    if callable(get_file):
        return bool(await get_file(key, path))
    data = await storage.get_object_bytes(key)
    if data is None:
        return False
    await asyncio.to_thread(_write_bytes, path, data)
    return True


def _write_bytes(path: str, data: bytes) -> None:
    Path(path).write_bytes(data)


def _read_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


async def _upload_file(storage: OffboardingStorage, key: str, path: str) -> None:
    put_file = getattr(storage, "put_file", None)
    if callable(put_file):
        await put_file(key, path, content_type="application/zip")
        return
    await storage.put_object(key, await asyncio.to_thread(_read_bytes, path), "application/zip")


# ─── jobs ────────────────────────────────────────────────────────────────────


def now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def is_stale(job: TenantOffboarding) -> bool:
    """A ``running`` job nobody has heartbeat recently (its worker crashed)."""
    age = (now_utc() - job.updated_at).total_seconds()
    return job.status == "running" and age > RUNNING_LEASE_SECONDS


async def latest_job(
    session: AsyncSession, tenant_id: UUID, kind: str, statuses: tuple[str, ...]
) -> TenantOffboarding | None:
    return (
        await session.execute(
            select(TenantOffboarding)
            .where(
                TenantOffboarding.tenant_id == tenant_id,
                TenantOffboarding.kind == kind,
                TenantOffboarding.status.in_(statuses),
            )
            .order_by(TenantOffboarding.created_at.desc(), TenantOffboarding.id.desc())
            .limit(1)
            .with_for_update()
        )
    ).scalar_one_or_none()


async def has_completed_export(session: AsyncSession, tenant_id: UUID) -> bool:
    return (
        await session.execute(
            select(TenantOffboarding.id)
            .where(
                TenantOffboarding.tenant_id == tenant_id,
                TenantOffboarding.kind == "export",
                TenantOffboarding.status == "done",
            )
            .limit(1)
        )
    ).first() is not None


async def _set_job(
    factory: async_sessionmaker[AsyncSession], job_id: UUID, **values: object
) -> None:
    """Update the job row in its OWN short transaction (heartbeat / progress)."""
    async with factory() as s:
        job = await s.get(TenantOffboarding, job_id, with_for_update=True)
        if job is None:  # pragma: no cover - job rows are never deleted
            return
        for name, value in values.items():
            setattr(job, name, value)
        job.updated_at = now_utc()
        await s.commit()


# ─── delete blockers ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Blocker:
    code: str
    message: str
    #: ``force=true`` (audited) may override it; a non-overridable one never yields.
    overridable: bool


async def delete_blockers(
    session: AsyncSession,
    tenant: Tenant,
    *,
    licence_state: str,
    partner_licence_ids: frozenset[UUID],
) -> list[Blocker]:
    """Why ``tenant`` may not be deleted right now (empty → it may)."""
    blockers: list[Blocker] = []
    if tenant.status != "suspended":
        blockers.append(
            Blocker(
                "TENANT_NOT_SUSPENDED",
                f"suspend the tenant first (status is {tenant.status!r})",
                overridable=False,
            )
        )
    await bind_session_tenant(session, tenant.id)
    if await tenant_has_active_runs(session, tenant.id):
        blockers.append(
            Blocker(
                "ACTIVE_RUNS",
                "the tenant has pending/running/paused runs; cancel them first",
                overridable=False,
            )
        )
    if not await has_completed_export(session, tenant.id):
        blockers.append(
            Blocker("EXPORT_MISSING", "no completed export exists for the tenant", True)
        )
    if tenant.licence_ref is not None and tenant.licence_ref in partner_licence_ids:
        blockers.append(
            Blocker("PARTNER_TENANT", "the tenant holds an allow-listed partner licence", True)
        )
    if licence_state in ("active", "expiring_soon"):
        blockers.append(
            Blocker("LICENCE_ACTIVE", f"the tenant's Studio licence is {licence_state}", True)
        )
    elif licence_state == "unknown":
        blockers.append(
            Blocker(
                "LICENCE_UNVERIFIED", "the licences service could not confirm no live licence", True
            )
        )
    return blockers


async def row_counts(session: AsyncSession, tenant_id: UUID) -> dict[str, int]:
    await bind_session_tenant(session, tenant_id)
    counts: dict[str, int] = {}
    for scope in TENANT_TABLES:
        n = await session.execute(
            text(f"SELECT count(*) FROM {scope.table} WHERE {scope.where}"),  # noqa: S608 - constant SQL
            {"t": tenant_id},
        )
        counts[scope.table] = int(n.scalar_one())
    return counts


# ─── export ──────────────────────────────────────────────────────────────────


def _strip_secrets(value: object) -> object:
    if isinstance(value, dict):
        return {k: _strip_secrets(v) for k, v in value.items() if k not in AUTH_BLOCK_KEYS}
    if isinstance(value, list):
        return [_strip_secrets(v) for v in value]
    return value


def _redact(table: str, row: dict[str, object], tenant_id: UUID) -> dict[str, object]:
    for column in _EXPORT_DROP_COLUMNS.get(table, ()):
        row.pop(column, None)
    if table == "runs":
        snapshot = row.get("config_snapshot")
        if isinstance(snapshot, dict):
            # Provider-slot pins reference the tenant's secret versions.
            snapshot.pop("provider_slots", None)
    if table == "request_audit" and row.get("claim_tenant_id") != str(tenant_id):
        row["actor_subject_id"] = None  # an operator acting INTO the tenant is not named
    redacted = _strip_secrets(row)
    return redacted if isinstance(redacted, dict) else {}


async def _table_rows(
    session: AsyncSession, scope: TableScope, tenant_id: UUID
) -> AsyncIterator[dict[str, object]]:
    if scope.table == "admin_audit_log":
        # The tenant-facing audit redaction (ORCH-5): an operator is not named.
        from src.api.routes.tenancy.audit import scrub_detail

        stmt = (
            select(
                AdminAuditLog.id,
                AdminAuditLog.created_at,
                AdminAuditLog.actor_subject_id,
                AdminAuditLog.action,
                AdminAuditLog.target_tenant_id,
                AdminAuditLog.detail,
                member_actor_clause(tenant_id).label("is_member"),
            )
            .where(AdminAuditLog.target_tenant_id == tenant_id)
            .order_by(AdminAuditLog.created_at, AdminAuditLog.id)
        )
        for row in (await session.execute(stmt)).all():
            member = bool(row.is_member)
            yield {
                "id": str(row.id),
                "created_at": row.created_at.isoformat(),
                "actor_subject_id": str(row.actor_subject_id) if member else None,
                "actor": "member" if member else "platform operator",
                "action": row.action,
                "target_tenant_id": str(row.target_tenant_id),
                "detail": scrub_detail(row.detail or {}, operator=not member),
            }
        return
    result = await session.stream(
        text(f"SELECT to_jsonb(x) AS r FROM {scope.table} x WHERE {scope.where}"),  # noqa: S608
        {"t": tenant_id},
    )
    async for (raw,) in result:
        yield _redact(scope.table, dict(raw), tenant_id)


Heartbeat = Callable[[dict[str, object]], Awaitable[None]]


async def build_export_archive(
    session: AsyncSession,
    storage: OffboardingStorage,
    tenant_id: UUID,
    job_id: UUID,
    heartbeat: Heartbeat,
) -> dict[str, object]:
    """Write the archive to ``export_key`` and return its summary.

    ``session`` must be fresh: it is switched to REPEATABLE READ so every table and
    the object prefix list come from ONE snapshot; that snapshot is rolled back (never
    committed) before any object is downloaded."""
    await session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
    await bind_session_tenant(session, tenant_id)
    workdir = tempfile.mkdtemp(prefix="w2v-offboarding-")
    archive = str(Path(workdir) / "export.zip")
    counts: dict[str, int] = {}
    objects: list[dict[str, object]] = []
    missing: list[str] = []
    try:
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for scope in TENANT_TABLES:
                n = 0
                with zf.open(f"data/{scope.table}.jsonl", "w") as fh:
                    async for row in _table_rows(session, scope, tenant_id):
                        fh.write(json.dumps(row, sort_keys=True, default=str).encode() + b"\n")
                        n += 1
                counts[scope.table] = n
                await heartbeat({"phase": "tables", "tables": dict(counts)})
            prefixes = await all_object_prefixes(session, tenant_id)
            # Every row is read: end the snapshot BEFORE streaming objects. Holding it
            # across a large tenant's downloads trips the primary's
            # ``idle_in_transaction_session_timeout`` (5 min) and pins a pgbouncer
            # server connection for the whole export.
            await session.rollback()
            for prefix in prefixes:
                for key in sorted(await storage.list_object_keys(prefix)):
                    tmp = str(Path(workdir) / "object.bin")
                    if not await _fetch_to_file(storage, key, tmp):
                        missing.append(key)
                        continue
                    size = Path(tmp).stat().st_size
                    await asyncio.to_thread(zf.write, tmp, f"objects/{key}")
                    await asyncio.to_thread(Path(tmp).unlink)
                    objects.append({"key": key, "bytes": size})
                await heartbeat(
                    {"phase": "objects", "tables": dict(counts), "objects": len(objects)}
                )
            manifest = {
                "tenant_id": str(tenant_id),
                "job_id": str(job_id),
                "generated_at": now_utc().isoformat(),
                "tables": counts,
                "objects": objects,
                "object_prefixes": prefixes,
                "objects_listed_but_missing": missing,
                "redacted": {
                    "columns": {k: list(v) for k, v in _EXPORT_DROP_COLUMNS.items()},
                    "json_keys": [*sorted(AUTH_BLOCK_KEYS), "runs.config_snapshot.provider_slots"],
                    "operators": "actors of platform-operator actions are not named",
                },
                "not_included": list(DOWNSTREAM_NOT_DELETED),
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        key = export_key(tenant_id, job_id)
        await _upload_file(storage, key, archive)
        return {
            "artifact_key": key,
            "tables": counts,
            "object_count": len(objects),
            "object_bytes": sum(cast(int, o["bytes"]) for o in objects),
            "archive_bytes": Path(archive).stat().st_size,
            "objects_listed_but_missing": missing,
        }
    finally:
        await asyncio.to_thread(shutil.rmtree, workdir, True)


async def run_export(
    factory: async_sessionmaker[AsyncSession],
    storage: OffboardingStorage,
    job_id: UUID,
) -> None:
    """The background export task. Never raises: the outcome is the job row."""
    async with factory() as s:
        job = await s.get(TenantOffboarding, job_id)
        if job is None or job.status != "running":
            return
        tenant_id, actor = job.tenant_id, job.decided_by

    async def heartbeat(progress: dict[str, object]) -> None:
        await _set_job(factory, job_id, progress=progress)

    try:
        async with factory() as read:
            summary = await build_export_archive(read, storage, tenant_id, job_id, heartbeat)
            await read.rollback()
    except Exception as exc:
        logger.exception("offboarding export %s for tenant %s failed", job_id, tenant_id)
        async with factory() as s:
            job = await s.get(TenantOffboarding, job_id, with_for_update=True)
            if job is not None:
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"[:2000]
                job.finished_at = job.updated_at = now_utc()
                await AuditRepository(s).record(
                    actor_subject_id=actor,
                    action="offboarding.export",
                    target_tenant_id=tenant_id,
                    detail={"job_id": str(job_id), "phase": "failed", "error": job.error},
                )
                await s.commit()
        return
    async with factory() as s:
        job = await s.get(TenantOffboarding, job_id, with_for_update=True)
        if job is None:  # pragma: no cover
            return
        job.status, job.error = "done", None
        job.artifact_key = cast(str, summary["artifact_key"])
        job.result = summary
        job.finished_at = job.updated_at = now_utc()
        await AuditRepository(s).record(
            actor_subject_id=actor,
            action="offboarding.export",
            target_tenant_id=tenant_id,
            detail={
                "job_id": str(job_id),
                "phase": "completed",
                "tables": summary["tables"],
                "object_count": summary["object_count"],
                "archive_bytes": summary["archive_bytes"],
            },
        )
        await s.commit()


# ─── delete ──────────────────────────────────────────────────────────────────


class OffboardingStepError(RuntimeError):
    """A delete phase failed; the job row is marked failed with resumable progress."""


async def purge_tenant_rows(session: AsyncSession, tenant_id: UUID) -> dict[str, int]:
    """Delete / tombstone every tenant row per :data:`TENANT_TABLES` (no commit)."""
    await bind_session_tenant(session, tenant_id)
    counts: dict[str, int] = {}
    for scope in TENANT_TABLES:
        params = {"t": tenant_id}
        if scope.mode == "retain":
            continue
        if scope.mode == "tombstone":
            res = await session.execute(
                text(
                    "UPDATE tenants SET status = 'deleted', name = NULL, updated_at = now() "
                    "WHERE id = :t"
                ),
                params,
            )
        elif scope.mode == "delete_keep_offboarding":
            res = await session.execute(
                text(
                    f"DELETE FROM {scope.table} WHERE ({scope.where}) "  # noqa: S608
                    "AND NOT starts_with(action, 'offboarding.')"
                ),
                params,
            )
        else:
            res = await session.execute(
                text(f"DELETE FROM {scope.table} WHERE {scope.where}"),  # noqa: S608
                params,
            )
        counts[scope.table] = int(getattr(res, "rowcount", 0) or 0)
    return counts


async def delete_rows_phase(
    session: AsyncSession, job: TenantOffboarding, *, forced_over: list[str]
) -> None:
    """Phase 1, in the CALLER's transaction (commit is the caller's): lock the
    tenant, refuse on an active run, record the prefixes, delete, tombstone, audit."""
    tenant_id = job.tenant_id
    await bind_session_tenant(session, tenant_id)
    tenant = (
        await session.execute(select(Tenant).where(Tenant.id == tenant_id).with_for_update())
    ).scalar_one()
    # The job runs in the background: re-check the hard blockers under the lock.
    if tenant.status != "suspended":
        raise OffboardingStepError(f"TENANT_NOT_SUSPENDED: status is {tenant.status!r}")
    if await tenant_has_active_runs(session, tenant_id):
        raise OffboardingStepError("ACTIVE_RUNS: a run of the tenant became active")
    prefixes = await all_object_prefixes(session, tenant_id)
    name_before, status_before = tenant.name, tenant.status
    counts = await purge_tenant_rows(session, tenant_id)
    progress: dict[str, object] = {
        "forced_over": forced_over,
        "rows_deleted": True,
        "deleted_rows": counts,
        "prefixes_pending": prefixes,
        "prefixes_done": [],
    }
    job.progress = progress
    job.updated_at = now_utc()
    await AuditRepository(session).record(
        actor_subject_id=job.decided_by,
        action="offboarding.delete",
        target_tenant_id=tenant_id,
        detail={
            "job_id": str(job.id),
            "phase": "rows_deleted",
            "tenant_name": name_before,
            "status_before": status_before,
            "deleted_rows": counts,
            "force": job.force,
            "forced_over": forced_over,
        },
    )


async def delete_objects_phase(
    session: AsyncSession, storage: OffboardingStorage, job: TenantOffboarding
) -> None:
    """Phase 2: purge each pending prefix, committing after each so a crash resumes
    at the next one. A prefix that cannot be emptied fails the job (resumable)."""
    while True:
        progress = dict(job.progress or {})
        pending = list(cast(list[str], progress.get("prefixes_pending") or []))
        if not pending:
            return
        prefix = pending[0]
        ok = await storage.delete_object_prefix(prefix)
        # The listing is the verdict, not ``ok``: the local backend reports False for a
        # prefix that never held anything, and a minio failure can leave nothing behind.
        left = await storage.list_object_keys(prefix)
        if left:
            raise OffboardingStepError(
                f"STORAGE: could not purge {prefix!r} ({len(left)} object(s) left"
                f"{'' if ok else ', delete reported failure'})"
            )
        progress["prefixes_pending"] = pending[1:]
        progress["prefixes_done"] = [*cast(list[str], progress.get("prefixes_done") or []), prefix]
        job.progress = progress
        job.updated_at = now_utc()
        await session.commit()


async def finish_delete(session: AsyncSession, job: TenantOffboarding) -> None:
    """Mark done, write the honest report and the completion audit (no commit)."""
    progress = job.progress or {}
    job.status, job.error = "done", None
    job.finished_at = job.updated_at = now_utc()
    job.result = {
        "deleted_rows": progress.get("deleted_rows", {}),
        "deleted_prefixes": progress.get("prefixes_done", []),
        "retained": retained(job.tenant_id),
        "not_deleted": list(DOWNSTREAM_NOT_DELETED),
        "complete": False,
        "note": "orchestrator data deleted; downstream services keep their own tenant "
        "rows (not_deleted) until each gains an internal tenant-delete endpoint",
    }
    await AuditRepository(session).record(
        actor_subject_id=job.decided_by,
        action="offboarding.delete",
        target_tenant_id=job.tenant_id,
        detail={
            "job_id": str(job.id),
            "phase": "completed",
            "deleted_prefixes": len(cast(list[str], progress.get("prefixes_done") or [])),
            "not_deleted": [d["service"] for d in DOWNSTREAM_NOT_DELETED],
        },
    )


async def mark_failed(factory: async_sessionmaker[AsyncSession], job_id: UUID, error: str) -> None:
    async with factory() as s:
        job = await s.get(TenantOffboarding, job_id, with_for_update=True)
        if job is None:  # pragma: no cover
            return
        job.status, job.error = "failed", error[:2000]
        job.updated_at = now_utc()
        await AuditRepository(s).record(
            actor_subject_id=job.decided_by,
            action="offboarding.delete" if job.kind == "delete" else "offboarding.export",
            target_tenant_id=job.tenant_id,
            detail={"job_id": str(job_id), "phase": "failed", "error": job.error},
        )
        await s.commit()


# ─── background execution ────────────────────────────────────────────────────

#: Strong refs to in-flight jobs (an un-referenced asyncio task can be GC'd mid-run).
_TASKS: set[asyncio.Task[None]] = set()


def spawn(coro: Awaitable[None], *, name: str) -> asyncio.Task[None]:
    """Run ``coro`` as a pod-lifetime task, detached from the HTTP request: the
    caller answers 202 at once (a delete can outlive Cloudflare's 100 s origin
    timeout). A pod that dies mid-job leaves it ``running``; the stale lease makes
    :func:`resume_stale_jobs` pick it up from its recorded progress."""

    async def _run() -> None:
        await coro

    task = asyncio.create_task(_run(), name=name)
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return task


async def drain() -> None:
    """Wait for every spawned job (tests, graceful shutdown)."""
    while _TASKS:
        await asyncio.gather(*list(_TASKS), return_exceptions=True)


async def _heartbeating(
    factory: async_sessionmaker[AsyncSession], job_id: UUID, work: Awaitable[None]
) -> None:
    """Await ``work`` while bumping the job's ``updated_at`` every
    :data:`HEARTBEAT_SECONDS`, so one long storage call never looks like a crash."""

    async def beat() -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            try:
                async with factory() as s:
                    await s.execute(
                        text(
                            "UPDATE tenant_offboarding SET updated_at = now() "
                            "WHERE id = :j AND status = 'running'"
                        ),
                        {"j": job_id},
                    )
                    await s.commit()
            except Exception:
                logger.warning("offboarding %s: heartbeat failed", job_id, exc_info=True)

    beater = asyncio.create_task(beat(), name=f"offboarding-heartbeat-{job_id}")
    try:
        await work
    finally:
        beater.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await beater


async def run_delete(
    factory: async_sessionmaker[AsyncSession], storage: OffboardingStorage, job_id: UUID
) -> None:
    """The background delete of a CLAIMED (``running``) job. Never raises: the
    outcome is the job row (``done`` with the report, or ``failed`` with resumable
    progress — re-DELETE to resume)."""
    tenant_id: UUID | None = None
    async with factory() as s:
        job = await s.get(TenantOffboarding, job_id)
        if job is None or job.kind != "delete" or job.status != "running":
            return
        tenant_id = job.tenant_id
        try:
            progress = dict(job.progress or {})
            if not progress.get("rows_deleted"):
                forced = cast(list[str], progress.get("forced_over") or [])
                await delete_rows_phase(s, job, forced_over=list(forced))
                await s.commit()
            await delete_objects_phase(s, storage, job)
            await finish_delete(s, job)
            await s.commit()
        except Exception as exc:
            logger.exception("offboarding delete %s for tenant %s failed", job_id, tenant_id)
            await s.rollback()
            message = str(exc) if isinstance(exc, OffboardingStepError) else None
            await mark_failed(factory, job_id, message or f"{type(exc).__name__}: {exc}")
            return
    # In-process positive cache of "known" tenants: drop it so this pod stops
    # admitting the tenant at once (other pods expire it within its 60 s TTL).
    from src.api.routes import _common

    _common._known_tenant_cache.pop(tenant_id, None)


async def execute_job(
    factory: async_sessionmaker[AsyncSession],
    storage: OffboardingStorage,
    job_id: UUID,
    kind: str,
) -> None:
    """Run one claimed job under a heartbeat."""
    work = run_delete if kind == "delete" else run_export
    await _heartbeating(factory, job_id, work(factory, storage, job_id))


# ─── starting a job (shared by the HTTP routes and the signup removal) ────────


class StartRefusedError(RuntimeError):
    """A delete / export could not be started: a stable ``code`` + the blockers."""

    def __init__(
        self, http: int, code: str, message: str, details: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.http = http
        self.code = code
        self.message = message
        self.details = details or {}


async def fresh_licence_state(licenses: object, tenant_id: UUID) -> str:
    """The tenant's ``w2v`` licence state from a FRESH licences answer (the status
    cache is bypassed): an offboarding verdict, never the gate-off ``active``."""
    result = await get_license_status(
        client=cast(Any, licenses),
        app_key=W2V_LICENCE_APP_KEY,
        tenant_id=str(tenant_id),
        subject_id=None,
        gate_enabled=True,
        now=now_utc(),
    )
    return result.state


async def delete_preflight(
    session: AsyncSession, licenses: object, tenant: Tenant, *, force: bool
) -> list[str]:
    """Refuse on any hard blocker, or on a soft one without ``force``; return the
    codes ``force`` overrode (audited with the deletion)."""
    blockers = await delete_blockers(
        session,
        tenant,
        licence_state=await fresh_licence_state(licenses, tenant.id),
        partner_licence_ids=get_settings().w2v_partner_license_ids,
    )
    hard = [b for b in blockers if not b.overridable]
    soft = [b for b in blockers if b.overridable]
    if hard or (soft and not force):
        raise StartRefusedError(
            409,
            "OFFBOARDING_BLOCKED",
            "; ".join(b.message for b in (hard or soft)),
            {"blockers": [b.__dict__ for b in blockers]},
        )
    return [b.code for b in soft]


@dataclass(frozen=True)
class DeleteStart:
    """``job`` to run (``started``), or the tenant's finished delete job when it
    was already deleted (``started=False``: nothing to run, idempotent re-run)."""

    job: TenantOffboarding
    started: bool


async def start_tenant_delete(
    session: AsyncSession,
    licenses: object,
    tenant: Tenant,
    actor: UUID | None,
    *,
    force: bool = False,
    job_hint: UUID | None = None,
    refuse_forced: bool = False,
) -> DeleteStart:
    """The ``DELETE /api/admin/tenants/{t}`` core, minus the confirmation (the
    caller checks it): find / claim the tenant's delete job, preflight it (a job
    whose rows are already gone resumes without one), mark it ``running`` and
    COMMIT. The caller then runs :func:`execute_job` (detached, or awaited).
    ``tenant`` must be locked in ``session`` (``SELECT … FOR UPDATE``).

    ``refuse_forced`` (the signup removal, never forced): an open job an admin
    earlier FORCED is not resumed — resuming it would keep ``force=true`` and so
    make this caller's delete forced (``FORCED_JOB_EXISTS``).

    Raises :class:`StartRefusedError` (``OFFBOARDING_NOT_PENDING``,
    ``DELETE_IN_PROGRESS``, ``ALREADY_DELETED``, ``OFFBOARDING_BLOCKED``,
    ``FORCED_JOB_EXISTS``)."""
    tenant_id = tenant.id
    job = await latest_job(session, tenant_id, "delete", ("requested", "running", "failed"))
    if job_hint is not None and (job is None or job.id != job_hint):
        raise StartRefusedError(409, "OFFBOARDING_NOT_PENDING", "the request is not open")
    if job is not None and job.status == "running" and not is_stale(job):
        raise StartRefusedError(
            409,
            "DELETE_IN_PROGRESS",
            "a delete of this tenant is already running",
            {"job_id": str(job.id)},
        )
    if refuse_forced and job is not None and job.force:
        raise StartRefusedError(
            409,
            "FORCED_JOB_EXISTS",
            "an open delete job of this tenant was forced by an admin",
            {"job_id": str(job.id), "job_status": job.status},
        )
    rows_done = bool(job is not None and (job.progress or {}).get("rows_deleted"))
    if tenant.status == "deleted" and not rows_done:
        done = await latest_job(session, tenant_id, "delete", ("done",))
        if done is None:
            raise StartRefusedError(409, "ALREADY_DELETED", "the tenant was deleted")
        return DeleteStart(job=done, started=False)
    # A resumed job already passed (and recorded) its preflight before its rows went.
    forced_over = (
        [] if rows_done else await delete_preflight(session, licenses, tenant, force=force)
    )
    now = now_utc()
    if job is None:
        job = TenantOffboarding(
            id=uuid7(),
            tenant_id=tenant_id,
            kind="delete",
            requested_by=actor,
            force=False,
            progress={},
            created_at=now,
        )
        session.add(job)
    job.status, job.decided_by, job.error = "running", actor, None
    job.force = bool(job.force) or force
    job.started_at = job.started_at or now
    job.updated_at = now
    if not rows_done:  # the background rows phase audits what force overrode
        job.progress = {**(job.progress or {}), "forced_over": forced_over}
    await session.commit()
    return DeleteStart(job=job, started=True)


async def start_tenant_export(
    session: AsyncSession, tenant: Tenant, actor: UUID | None, *, job_hint: UUID | None = None
) -> TenantOffboarding:
    """The ``POST /api/admin/tenants/{t}/export`` core: claim (or re-claim a
    stale) export job, mark it ``running``, audit and COMMIT. The caller runs
    :func:`execute_job`. ``tenant`` must be locked in ``session``.

    Raises :class:`StartRefusedError` (``TENANT_DELETED``, ``EXPORT_IN_PROGRESS``,
    ``OFFBOARDING_NOT_PENDING``)."""
    tenant_id = tenant.id
    if tenant.status == "deleted":
        raise StartRefusedError(409, "TENANT_DELETED", "the tenant was deleted")
    job = await latest_job(session, tenant_id, "export", ("requested", "running"))
    if job is not None and job.status == "running" and not is_stale(job):
        raise StartRefusedError(
            409,
            "EXPORT_IN_PROGRESS",
            "an export of this tenant is already running",
            {"job_id": str(job.id)},
        )
    if job_hint is not None and (job is None or job.id != job_hint):
        raise StartRefusedError(409, "OFFBOARDING_NOT_PENDING", "the request is not open")
    now = now_utc()
    if job is None:
        job = TenantOffboarding(
            id=uuid7(),
            tenant_id=tenant_id,
            kind="export",
            requested_by=actor,
            force=False,
            created_at=now,
        )
        session.add(job)
    resumed = job.status == "running"
    job.status, job.decided_by, job.error = "running", actor, None
    job.artifact_key, job.result, job.progress = None, None, {}
    job.started_at = job.updated_at = now
    await AuditRepository(session).record(
        actor_subject_id=actor,
        action="offboarding.export",
        target_tenant_id=tenant_id,
        detail={"job_id": str(job.id), "phase": "started", "reclaimed_stale": resumed},
    )
    await session.commit()
    return job


async def resume_stale_jobs(
    factory: async_sessionmaker[AsyncSession], storage: OffboardingStorage
) -> list[UUID]:
    """Atomically re-claim every ``running`` job whose lease expired (its pod died)
    and resume it in the background: a delete from its recorded progress, an export
    from scratch. One claimer wins per job (the conditional UPDATE)."""
    async with factory() as s:
        rows = (
            await s.execute(
                text(
                    "UPDATE tenant_offboarding SET updated_at = now() "
                    "WHERE status = 'running' "
                    "AND updated_at < now() - make_interval(secs => :lease) "
                    "RETURNING id, kind"
                ),
                {"lease": RUNNING_LEASE_SECONDS},
            )
        ).all()
        await s.commit()
    for job_id, kind in rows:
        logger.warning("offboarding %s job %s: lease expired, resuming", kind, job_id)
        spawn(execute_job(factory, storage, job_id, kind), name=f"offboarding-{kind}-{job_id}")
    return [r[0] for r in rows]
