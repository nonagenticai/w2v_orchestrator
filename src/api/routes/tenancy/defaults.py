"""Tenant production defaults (P6 ORCH-1).

- GET /api/tenant/defaults → the tenant's defaults + what the deployment uses where a
  field is ``null``. Any MEMBER may read it (the create-project wizard prefills from
  it), so it carries the full tenant data gate (``TENANT_GATE``: known tenant, RLS
  bound to the effective tenant, licence).
- PUT /api/tenant/defaults → replace the document. ``TenantAdminDep`` +
  ``LICENSE_GATE_ONLY`` + the RLS bind to the AUTHORIZED effective tenant (the
  tenant-admin tier posture, same as ``/api/tenant/connections``). ``expected_version``
  must equal the stored version (``0`` = no row yet), else 409 ``VERSION_CONFLICT``;
  the compare-and-set happens in the write statement itself, so a concurrent writer
  also gets the 409. Every successful write audits ``defaults.update`` with the
  changed fields (defaults carry no secrets). Values the pipeline would refuse are a
  422, not a stored default that fails every run: ``voice.voice_id`` must be a voice
  the tenant's narration path accepts (``_check_voice``), and
  ``model_tier.timing_director`` one of the router tiers (``TimingDirectorTier``).

A run pins the defaults at creation (``build_config_snapshot``); changing them never
affects a run that already exists.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, status
from fastapi import params as fastapi_params
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes._common import (
    LICENSE_GATE_ONLY,
    TENANT_GATE,
    AuditRepoDep,
    SessionDep,
    SubjectIdDep,
    TenantAdminDep,
    TenantIdDep,
)
from src.api.routes.tenancy.connections import bind_admin_tenant_rls
from src.api.routes.tenancy.errors import TenancyError
from src.api.schemas.tenant_defaults import (
    DeploymentDefaults,
    TenantDefaultsDocument,
    TenantDefaultsPut,
    TenantDefaultsView,
)
from src.config import get_settings
from src.database.models import TenantDefaults
from src.database.repositories.connections_repository import ConnectionsRepository
from src.database.repositories.tenant_defaults_repository import (
    SETTING_COLUMNS,
    TenantDefaultsRepository,
)
from src.services.stages.audiobook.voice_catalog import VOICE_CATALOG
from src.services.tenant_defaults import (
    DEPLOYMENT_CAPTIONS_MODE,
    DEPLOYMENT_RETENTION_MODE,
    DEPLOYMENT_VOICE_DIRECTION_TONE,
)

router = APIRouter(prefix="/api/tenant/defaults", tags=["tenancy"])

#: SQLSTATE ``foreign_key_violation`` — the tenant has no ``tenants`` row.
_FK_VIOLATION = "23503"

#: Entitlement + the RLS bind to the authorized effective tenant (see module doc).
_ADMIN_GATE: list[fastapi_params.Depends] = [
    *LICENSE_GATE_ONLY,
    Depends(bind_admin_tenant_rls),
]


def _err(http: int, code: str, message: str, **details: object) -> TenancyError:
    return TenancyError(http, code, message, details or None)


async def _check_voice(session: AsyncSession, tenant_id: UUID, voice_id: str | None) -> None:
    """Refuse a default ``voice_id`` the tenant's narration path would refuse.

    With no tenant ``tts`` connection, narration goes to the deployment kokoro backend
    (``tts.test``), which 500s on an id outside ``VOICE_CATALOG`` — the same check
    the project door applies (``schemas.projects.Voice``). A stored bad default would
    fail every run three stages deep, so it is a 422 here.

    With an ACTIVE tenant ``tts`` connection (openai / elevenlabs / openai-compatible),
    the voice namespace is the tenant provider's and cannot be enumerated (an
    openai-compatible server may be anything), so any id is accepted. Note that every
    tenant TTS kind requires its own ``voice`` option, which wins over this default in
    ``tts_inprocess``. Known gap: removing the ``tts`` connection later does not
    re-validate the stored default, so a non-catalog id saved under a tenant provider
    would then reach kokoro and fail narration.
    """
    if voice_id is None or voice_id in VOICE_CATALOG:
        return
    if await ConnectionsRepository(session).get_current(tenant_id, "tts") is not None:
        return
    raise _err(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "UNKNOWN_VOICE",
        f"voice_id {voice_id!r} is not a voice the narration backend accepts; "
        f"choose one of the {len(VOICE_CATALOG)} catalog voices, or configure a tenant "
        "TTS provider first",
        voice_id=voice_id,
    )


def _deployment_defaults() -> DeploymentDefaults:
    pipeline = get_settings().pipeline
    return DeploymentDefaults(
        voice_id=None,
        captions_mode=DEPLOYMENT_CAPTIONS_MODE,
        captions_karaoke=pipeline.captions_karaoke,
        voice_direction_tone=(
            DEPLOYMENT_VOICE_DIRECTION_TONE if pipeline.voice_direction_default_enabled else None
        ),
        retention_mode=(DEPLOYMENT_RETENTION_MODE if pipeline.retention_default_enabled else None),
        timing_director_model_tier=pipeline.timing_director_model_tier,
    )


def _view(row: TenantDefaults | None) -> TenantDefaultsView:
    deployment = _deployment_defaults()
    if row is None:
        return TenantDefaultsView(version=0, deployment=deployment)
    return TenantDefaultsView.model_validate(
        {
            **{col: getattr(row, col) for col in SETTING_COLUMNS},
            "version": row.version,
            "updated_at": row.updated_at,
            "updated_by": row.updated_by,
            "deployment": deployment,
        }
    )


def _columns(body: TenantDefaultsDocument) -> dict[str, object]:
    """The body as column values: each block without its ``null`` fields, and a block
    with nothing left stored as SQL NULL (so "all null" == "no default")."""
    out: dict[str, object] = {}
    for col in SETTING_COLUMNS:
        value = getattr(body, col)
        if value is None or isinstance(value, str):
            out[col] = value
            continue
        block = value.model_dump(mode="json", exclude_none=True)
        out[col] = block or None
    return out


def _changes(before: dict[str, object], after: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        col: {"from": before.get(col), "to": after.get(col)}
        for col in SETTING_COLUMNS
        if before.get(col) != after.get(col)
    }


@router.get(
    "",
    response_model=TenantDefaultsView,
    operation_id="tenant.defaults.get",
    dependencies=TENANT_GATE,
)
async def get_tenant_defaults(tenant_id: TenantIdDep, session: SessionDep) -> TenantDefaultsView:
    return _view(await TenantDefaultsRepository(session).get(tenant_id))


@router.put(
    "",
    response_model=TenantDefaultsView,
    operation_id="tenant.defaults.put",
    dependencies=_ADMIN_GATE,
)
async def put_tenant_defaults(
    body: TenantDefaultsPut,
    tenant_id: TenantAdminDep,
    actor: SubjectIdDep,
    audit: AuditRepoDep,
    session: SessionDep,
) -> TenantDefaultsView:
    await _check_voice(session, tenant_id, body.voice.voice_id if body.voice else None)
    values = _columns(body)
    repo = TenantDefaultsRepository(session)
    current = await repo.get(tenant_id)
    live_version = current.version if current is not None else 0
    if body.expected_version != live_version:
        raise _err(
            status.HTTP_409_CONFLICT,
            "VERSION_CONFLICT",
            f"expected version {body.expected_version}, current is {live_version}",
            expected=body.expected_version,
            current=live_version,
        )
    before = {col: getattr(current, col) for col in SETTING_COLUMNS} if current else {}
    try:
        if current is None:
            row = await repo.create(tenant_id, values, actor=actor)
        else:
            row = await repo.update_if_version(tenant_id, live_version, values, actor=actor)
    except IntegrityError as exc:
        await session.rollback()
        if getattr(exc.orig, "sqlstate", None) == _FK_VIOLATION:
            raise _err(
                status.HTTP_404_NOT_FOUND, "TENANT_NOT_FOUND", "tenant is not provisioned"
            ) from None
        raise
    if row is None:  # a concurrent writer won the compare-and-set
        await session.rollback()
        raise _err(
            status.HTTP_409_CONFLICT,
            "VERSION_CONFLICT",
            "the defaults changed concurrently; reload and retry",
            expected=body.expected_version,
        )
    await audit.record(
        actor_subject_id=actor,
        action="defaults.update",
        target_tenant_id=tenant_id,
        detail={"version": row.version, "changes": _changes(before, values)},
    )
    view = _view(row)
    await session.commit()
    return view
