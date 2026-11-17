"""Critique routes — proxy to analysis_service.

The orchestrator owns the public ``w2v.nonagentic.ai`` API surface; the
Istio VirtualService routes only to ``w2v-frontend → w2v-orchestrator``.
analysis_service stays internal on port 6303. This router re-exposes the
critique endpoints the frontend calls. Header forwarding + the thin proxy
helpers live in ``_proxy`` (shared with the redesign router).
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.api.routes._common import TenantIdDep
from src.api.routes._proxy import proxy_get, proxy_post, proxy_stream

router = APIRouter(prefix="/api/critique", tags=["critique"])


class CritiqueAuthInfo(BaseModel):
    """Mirror of analysis_service ``AuthInfo``: credentials analysis applies when
    fetching the target URL (basic / bearer / cookie replay). Same field rules."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["none", "basic", "bearer", "cookies"] = "none"
    username: str | None = None
    password: str | None = None
    token: str | None = None
    cookies: list[dict[str, str]] | None = None
    cookie_header: str | None = None
    login_url: str | None = None

    @model_validator(mode="after")
    def _required_fields_present(self) -> CritiqueAuthInfo:
        if self.type == "basic":
            if not self.username or self.password is None:
                raise ValueError("auth_info type=basic requires username and password")
        elif self.type == "bearer":
            if not self.token:
                raise ValueError("auth_info type=bearer requires token")
        elif self.type == "cookies" and not self.cookies and not self.cookie_header:
            raise ValueError("auth_info type=cookies requires cookies or cookie_header")
        return self


class CritiqueWebhookConfig(BaseModel):
    """Mirror of analysis_service ``WebhookConfig``."""

    model_config = ConfigDict(extra="forbid")

    events: list[str] | None = None
    severity_threshold: str | None = None


class CritiqueCreate(BaseModel):
    """Inbound body for starting a critique — mirrors analysis_service's
    ``CritiqueRequest`` (``src/domains/critique/schemas.py``) field-for-field,
    including its "exactly one content source" rule, so a body analysis would
    accept is accepted here and one it would reject is a 422 here (not a
    proxied 422 after a hop).

    Content source — exactly one of:

    * ``url``          crawl a live page (the w2v frontend's path);
    * ``html``         score pre-rendered HTML in place (merqi marketing's
                       branding-render critique + copy diagnostics, P9-7);
    * ``artifact_ref`` an object key in analysis's own bucket, tenant-confined
                       server-side by analysis.

    Size: ``ProxyBodyLimitMiddleware`` caps this exact route at analysis's own
    ``MAX_HTML_SIZE`` (10 MiB) + a 64 KiB JSON envelope — a body analysis would
    score is never refused here, a larger one is a 413 (declared or streamed),
    and the cap is a streaming count, so nothing is buffered before Auth.

    Deliberately NOT exposed (analysis accepts them; a public API-key caller
    must not): ``llm_config`` — its ``base_url`` + ``api_key_env`` would let a
    caller point analysis's LLM call at their own host carrying any env-named
    secret of analysis; ``site_analysis`` — orchestrator-computed context that
    steers which security checks run. ``extra="forbid"`` makes sending either
    (or any unknown field) a loud 422 instead of a silent drop.
    """

    model_config = ConfigDict(extra="forbid")

    url: str | None = None
    html: str | None = None
    artifact_ref: str | None = None
    categories: list[str] | None = None
    max_pages: int | None = Field(default=None, ge=1, le=50)
    instructions: str | None = None
    github_repo: str | None = None
    force_rescan: bool | None = None
    callback_url: str | None = None
    tags: list[str] | None = None
    webhook_config: CritiqueWebhookConfig | None = None
    auth_info: CritiqueAuthInfo | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> CritiqueCreate:
        # Same predicate as analysis's CritiqueRequest._exactly_one_source.
        sources = [
            bool(self.url and self.url.strip()),
            bool(self.html and self.html.strip()),
            bool(self.artifact_ref and self.artifact_ref.strip()),
        ]
        if sum(sources) != 1:
            raise ValueError("exactly one of 'url', 'html', or 'artifact_ref' is required")
        return self


@router.post("", operation_id="critique.create")
async def create_critique(
    body: CritiqueCreate,
    request: Request,
    tenant_id: TenantIdDep,
) -> dict:
    """Start a critique job. Forwards to analysis_service ``POST /api/critique``."""
    payload = body.model_dump(exclude_none=True)
    return await proxy_post("/api/critique", payload, effective_tenant=tenant_id)


@router.get("", operation_id="critique.list")
async def list_critiques(
    request: Request,
    tenant_id: TenantIdDep,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    status: Annotated[str | None, Query()] = None,
) -> dict:
    """List critique jobs for the calling tenant.

    Frontend uses ``page_size``; analysis_service uses ``per_page``. Translated here.
    """
    params: dict[str, str | int] = {"page": page, "per_page": page_size}
    if status:
        params["status"] = status
    return await proxy_get("/api/critique", params=params, effective_tenant=tenant_id)


@router.get("/compare", operation_id="critique.compare")
async def compare_critiques(
    request: Request,
    tenant_id: TenantIdDep,
    job_id_1: Annotated[str, Query(min_length=1, max_length=200)],
    job_id_2: Annotated[str, Query(min_length=1, max_length=200)],
) -> dict:
    """Compare two critique jobs (score + findings deltas).

    Declared BEFORE ``/{job_id}`` on purpose: otherwise ``GET /compare`` matches
    ``/{job_id}`` with ``job_id="compare"`` and is proxied WITHOUT its query
    string — analysis answers 422 (P9-7). Both ids are forwarded as query params;
    analysis scopes BOTH jobs to the delegated tenant."""
    return await proxy_get(
        "/api/critique/compare",
        params={"job_id_1": job_id_1, "job_id_2": job_id_2},
        effective_tenant=tenant_id,
    )


@router.get("/{job_id}", operation_id="critique.get")
async def get_critique(
    job_id: str,
    request: Request,
    tenant_id: TenantIdDep,
) -> dict:
    """Get a single critique job by id."""
    return await proxy_get(f"/api/critique/{job_id}", effective_tenant=tenant_id)


@router.get("/{job_id}/report/pdf", operation_id="critique.report.pdf")
async def get_critique_report_pdf(
    job_id: str,
    request: Request,
    tenant_id: TenantIdDep,
) -> StreamingResponse:
    """Stream the critique-report PDF from analysis_service."""
    return await proxy_stream(
        f"/api/critique/{job_id}/report/pdf",
        "application/pdf",
        effective_tenant=tenant_id,
    )
