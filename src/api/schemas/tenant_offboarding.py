"""P6 ORCH-6 schemas: tenant offboarding (export + delete).

``download_url`` is the same-origin download path of a finished export (tenant view:
``/api/tenant/offboarding/{id}/download``; operator view:
``/api/admin/offboarding/{id}/download``). It is never a presigned MinIO URL — that
host is ``minio.test``, which a browser cannot resolve. The job row keeps only
``artifact_key``.
"""

from __future__ import annotations

import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

_cfg = ConfigDict(populate_by_name=True, str_strip_whitespace=True, extra="forbid")

OffboardingKind = Literal["export", "delete"]
OffboardingStatus = Literal["requested", "running", "done", "failed", "rejected"]


class OffboardingRequestCreate(BaseModel):
    """A tenant_admin's request; a super_admin decides and executes it."""

    model_config = _cfg
    kind: OffboardingKind
    reason: str | None = Field(default=None, max_length=1000)


class OffboardingApprove(BaseModel):
    """Approve = execute. A delete needs the plan's ``confirm_token``; ``force``
    overrides the overridable blockers (missing export, live/unverified licence,
    partner tenant) and is audited."""

    model_config = _cfg
    confirm: str | None = None
    force: bool = False


class OffboardingReject(BaseModel):
    model_config = _cfg
    reason: str | None = Field(default=None, max_length=1000)


class OffboardingJob(BaseModel):
    id: UUID
    tenant_id: UUID
    kind: OffboardingKind
    status: OffboardingStatus
    requested_by: UUID | None = None
    #: The deciding super_admin; withheld (``None``) on the tenant_admin view.
    decided_by: UUID | None = None
    reason: str | None = None
    force: bool = False
    error: str | None = None
    progress: dict[str, object] = Field(default_factory=dict)
    result: dict[str, object] | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    started_at: datetime.datetime | None = None
    finished_at: datetime.datetime | None = None
    download_url: str | None = None


class OffboardingJobList(BaseModel):
    items: list[OffboardingJob]


class OffboardingBlocker(BaseModel):
    code: str
    message: str
    overridable: bool


class OffboardingPlan(BaseModel):
    """Step 1 of the two-step delete: what a delete would do, and the token step 2
    (``DELETE /api/admin/tenants/{id}?confirm=<token>``) must echo."""

    tenant_id: UUID
    tenant_name: str | None = None
    tenant_status: str
    confirm_token: str
    deletable: bool
    blockers: list[OffboardingBlocker]
    rows: dict[str, int]
    object_prefixes: list[str]
    retained: list[dict[str, str]]
    not_deleted: list[dict[str, object]]
