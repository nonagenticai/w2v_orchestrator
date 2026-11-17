"""Redesign routes — proxy to html_render_service.

Companion to the critique router: re-exposes the web-design capability on the
public orchestrator surface. The inbound public path stays ``/api/redesign/*``
(unchanged for callers), but it now forwards to ``html_render_service``'s
canonical ``/api/designs/*`` surface (discovery name ``html_render``) — the
capability was lifted out of analysis_service into its own service. Redesign is
critique-first — the standalone tool feeds a completed critique's findings into
``critique_findings`` so the variants address what the critique flagged. Header
forwarding + thin proxy helpers are shared via ``_proxy`` (critique keeps the
default ``analysis`` peer).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.routes._common import TenantIdDep
from src.api.routes._proxy import proxy_get, proxy_post, proxy_stream

router = APIRouter(prefix="/api/redesign", tags=["redesign"])


class RedesignCreate(BaseModel):
    """Inbound body for starting a redesign. Mirrors the fields of
    analysis_service's ``RedesignRequest`` that the standalone tool exposes."""

    url: str
    scope: str = "full_page"
    style: str = "modern"
    styles: list[str] | None = Field(default=None, max_length=4)
    critique_findings: list[dict] | None = None
    html_content: str | None = None


@router.post("", operation_id="redesign.create")
async def create_redesign(
    body: RedesignCreate,
    request: Request,
    tenant_id: TenantIdDep,
) -> dict:
    """Start a redesign job. Forwards to html_render_service ``POST /api/designs``."""
    payload = body.model_dump(exclude_none=True)
    return await proxy_post(
        "/api/designs", payload, discovery_name="html_render", effective_tenant=tenant_id
    )


@router.get("/jobs/{job_id}", operation_id="redesign.get")
async def get_redesign(
    job_id: str,
    request: Request,
    tenant_id: TenantIdDep,
) -> dict:
    """Get redesign job status."""
    return await proxy_get(
        f"/api/designs/jobs/{job_id}", discovery_name="html_render", effective_tenant=tenant_id
    )


@router.get("/jobs/{job_id}/result", operation_id="redesign.result")
async def get_redesign_result(
    job_id: str,
    request: Request,
    tenant_id: TenantIdDep,
) -> dict:
    """Get the full redesign result (variants + artifacts) for a completed job."""
    return await proxy_get(
        f"/api/designs/jobs/{job_id}/result",
        discovery_name="html_render",
        effective_tenant=tenant_id,
    )


@router.get("/jobs/{job_id}/html", operation_id="redesign.html")
async def download_redesign_html(
    job_id: str,
    request: Request,
    # why: rendered in an <iframe src> (and opened via target=_blank), which
    # cannot set the X-Active-Tenant-Id header — scope-switch rides ?active_tenant=.
    # get_tenant_id reads both channels, so the router-level gates see it too.
    tenant_id: TenantIdDep,
) -> StreamingResponse:
    """Download the primary redesigned HTML for a completed job."""
    return await proxy_stream(
        f"/api/designs/jobs/{job_id}/html",
        "text/html",
        discovery_name="html_render",
        effective_tenant=tenant_id,
    )


@router.get("/jobs/{job_id}/pdf", operation_id="redesign.pdf")
async def download_redesign_pdf(
    job_id: str,
    request: Request,
    # why: reached as a plain <a href>, which cannot set the X-Active-Tenant-Id
    # header — the scope-switch rides ?active_tenant= instead. get_tenant_id
    # reads both channels, so the router-level gates see it too.
    tenant_id: TenantIdDep,
) -> StreamingResponse:
    """Download the generated page as PDF (html_render mints it on demand)."""
    return await proxy_stream(
        f"/api/designs/jobs/{job_id}/pdf",
        "application/pdf",
        discovery_name="html_render",
        effective_tenant=tenant_id,
    )


@router.get("/jobs/{job_id}/variants/{style}/html", operation_id="redesign.variant.html")
async def download_redesign_variant_html(
    job_id: str,
    style: str,
    request: Request,
    tenant_id: TenantIdDep,
) -> StreamingResponse:
    """Download a specific redesign variant's HTML for a completed job."""
    return await proxy_stream(
        f"/api/designs/jobs/{job_id}/variants/{style}/html",
        "text/html",
        discovery_name="html_render",
        effective_tenant=tenant_id,
    )
