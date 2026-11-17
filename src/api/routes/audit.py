"""``POST /api/audit`` — one-command ``url`` → audit (A0 sugar route).

Creates a project + a scan-only audit deliverable + a started run in a SINGLE call
and returns the run id. The run's ``<type>_scan`` output IS the deliverable, and
``GET /api/runs/{run_id}/report/pdf`` renders the designed, enriched report. This is
the audit analogue of ``projects.py::create_project_from_intent`` (the one-command
video front door): the caller supplies a URL (and optionally an ``intent``) instead
of hand-assembling a project + deliverable + brief + run.

``include_llm`` defaults **on** here — the whole point of the one-command audit is the
GREAT enriched report (priority/prose/grounding/exec-summary). Set it false for the
deterministic titles-only scorecard.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

from src.api.routes._common import ProjectRepoDep, SessionDep, SubjectIdDep, TenantIdDep

router = APIRouter(prefix="/api/audit", tags=["audit"])

_camel = ConfigDict(populate_by_name=True, alias_generator=to_camel)

AuditType = Literal["seo_audit", "cro_audit", "marketing_scorecard"]


class AuditRequest(BaseModel):
    """One-command audit input — a URL, optionally steered by an ``intent``."""

    model_config = ConfigDict(
        populate_by_name=True, str_strip_whitespace=True, alias_generator=to_camel
    )

    # min_length=1 (after strip) so an empty/whitespace URL 422s at the door rather
    # than surviving to a StagePrecondition failure one stage deep — same contract as
    # AuditBrief.target_url.
    url: str = Field(min_length=1, description="Public HTTP(S) URL to audit")
    intent: str | None = Field(
        default=None,
        description="Optional audience / goal to steer the enriched report prose",
    )
    audit_type: AuditType = Field(
        default="marketing_scorecard",
        description="Which scorecard family to run (marketing = the composite go-to-market scorecard)",
    )
    include_llm: bool = Field(
        default=True,
        description="Produce the enriched (grounded, prose) report; false = deterministic titles only",
    )

    @field_validator("url")
    @classmethod
    def _require_http_scheme(cls, v: str) -> str:
        # Reject non-http(s) schemes (javascript:, ftp:, data:, a bare host) at the
        # door → 422, instead of letting them survive to a deep scan failure or the
        # analysis fetch (the docstring promises an http(s) URL; enforce it here).
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("url must be an http(s) URL")
        return v


class AuditResponse(BaseModel):
    """The created project + deliverable + started run, with URLs to fetch the report
    and stream live progress."""

    model_config = _camel

    project_id: UUID
    deliverable_id: UUID
    run_id: UUID
    report_pdf_url: str
    events_url: str


@router.post(
    "",
    response_model=AuditResponse,
    status_code=status.HTTP_201_CREATED,
    operation_id="audit.create",
)
async def create_audit(
    body: AuditRequest,
    request: Request,
    tenant_id: TenantIdDep,
    subject_id: SubjectIdDep,
    project_repo: ProjectRepoDep,
    session: SessionDep,
) -> AuditResponse:
    """Create + start a scan-only audit from a URL in one call (mirrors
    ``create_project_from_intent``)."""
    brief: dict[str, object] = {"target_url": body.url, "include_llm": body.include_llm}
    if body.intent:
        brief["intent"] = body.intent
    try:
        proj = await project_repo.create_with_deliverables(
            tenant_id=tenant_id,
            owner_subject_id=subject_id,
            type_=body.audit_type,
            title=f"Audit: {body.url}",
            brief=brief,
            brand={},
            voice={},
            theme={},
            visibility="tenant",
            parent_project_id=None,
            # Scan-only: no video renders, so `format` is inert metadata — use the
            # landscape default (the only ck_deliverables_format-valid values are
            # 16x9/9x16/1x1/4x5/audio; there is no report format).
            deliverables=[{"name": "Audit report", "format": "16x9"}],
        )
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    # Commit BEFORE start_run — same reason as `create_project_from_intent`: the
    # engine opens its OWN session and cannot see a merely-flushed deliverable.
    await session.commit()
    deliverable = proj.deliverables[0]
    engine = request.app.state.run_engine
    try:
        run = await engine.start_run(
            deliverable_id=deliverable.id,
            tenant_id=tenant_id,
            subject_id=subject_id,
            parent_run_id=None,
            pause_after_stages=None,
        )
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    return AuditResponse(
        project_id=proj.id,
        deliverable_id=deliverable.id,
        run_id=run.id,
        report_pdf_url=f"/api/runs/{run.id}/report/pdf",
        events_url=f"/api/runs/{run.id}/events",
    )
