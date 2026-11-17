"""Brand routes — proxy to analysis_service (P9-7).

merqi's marketing service called analysis (:6303) DIRECTLY for brand extraction
and the ICP; a tenant API key (``w2v_sk_``) cannot authenticate there (only the
orchestrator verifies keys), and decision 18 makes the orchestrator the cell's
only ingress. These routes re-expose exactly what marketing uses:

* ``POST /api/brand/from-url``     → analysis ``POST /api/brand/from-url``
* ``POST /api/brand/extract``      → analysis ``POST /api/brand/extract``
* ``POST /api/brand/icp-from-url`` → analysis ``POST /api/brand/icp-from-url``

Bodies are forwarded VERBATIM (analysis owns and validates the request schema —
a second copy here would drift). The caller is never forwarded as a credential:
``_proxy.forward_headers`` sends the orchestrator's service-account licence
(or a licence-lane caller's own licence token — never a ``w2v_sk_`` key) plus
``X-Effective-Tenant-Id`` = the resolved tenant and a fresh cell token.
ICP and extract scrape + call an LLM synchronously → 120 s upstream timeout.
Mounted with the full ``TENANT_GATE`` in ``src/asgi.py``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body
from pydantic import JsonValue

from src.api.routes._common import TenantIdDep
from src.api.routes._proxy import ANALYSIS_SLOW_TIMEOUT, proxy_post

router = APIRouter(prefix="/api/brand", tags=["brand"])

#: Any JSON object — analysis owns (and validates) the request schema.
JsonBody = Annotated[dict[str, JsonValue], Body()]


@router.post("/from-url", operation_id="brand.from_url")
async def brand_from_url(tenant_id: TenantIdDep, body: JsonBody) -> dict:
    """Extract a brand profile from a URL (analysis ``POST /api/brand/from-url``)."""
    return await proxy_post("/api/brand/from-url", body, effective_tenant=tenant_id)


@router.post("/extract", operation_id="brand.extract")
async def brand_extract(tenant_id: TenantIdDep, body: JsonBody) -> dict:
    """Extract a brand from analysis data (analysis ``POST /api/brand/extract``)."""
    return await proxy_post(
        "/api/brand/extract", body, effective_tenant=tenant_id, timeout=ANALYSIS_SLOW_TIMEOUT
    )


@router.post("/icp-from-url", operation_id="brand.icp_from_url")
async def brand_icp_from_url(tenant_id: TenantIdDep, body: JsonBody) -> dict:
    """Derive an ideal-customer profile from a URL (analysis ``POST /api/brand/icp-from-url``)."""
    return await proxy_post(
        "/api/brand/icp-from-url", body, effective_tenant=tenant_id, timeout=ANALYSIS_SLOW_TIMEOUT
    )
