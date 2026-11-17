"""HTTP client for Brand Extractor Service (port 6303 — served by Analysis Service)."""

from __future__ import annotations

from typing import Any

from src.clients.base import LicensedServiceClient
from src.clients.schemas.brand_extractor import BrandExtraction


class BrandExtractorClient(LicensedServiceClient):
    """Client for the Brand Extractor microservice.

    analysis_service is behind license-authz (only ``/health*`` is exempt), so
    the orchestrator's service-account licence token must be baked into
    ``X-Tenant-Id: Bearer <token>`` — anonymous calls get 401 ("Missing
    authentication token"), which made ``resolve_brand_context`` silently fall
    back to the generic blue palette. Same pattern as ``CritiqueClient`` /
    ``VideoCompileClient``; resolves to the ``analysis`` service via discovery.
    """

    def __init__(
        self,
        base_url: str | None = None,
        license_token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(
            service_name="brand_extractor",
            discovery_name="analysis",
            base_url=base_url,
            license_token=license_token,
            timeout=timeout,
        )

    async def extract_brand(
        self,
        url: str,
        *,
        download_logo: bool = False,
        llm_config: dict[str, object] | None = None,
    ) -> BrandExtraction:
        """Extract brand information from a URL (synchronous endpoint)."""
        payload: dict[str, Any] = {"url": url, "download_logo": download_logo}
        if llm_config:
            payload["llm_config"] = llm_config
        return BrandExtraction.model_validate(
            await self._request("POST", "/api/brand/from-url", json=payload)
        )
