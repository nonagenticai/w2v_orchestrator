"""HTTP client for BDD Playwright Service (port 6313)."""

from __future__ import annotations

from typing import Any

from src.clients.base import LicensedServiceClient
from src.clients.schemas.bdd_playwright import BddJobResult, BddJobStatus


class BDDPlaywrightClient(LicensedServiceClient):
    """Client for the BDD Playwright microservice.

    bdd_playwright_service is behind license-authz (only ``/health*`` is exempt), so
    the orchestrator's service-account token must be presented as ``X-Tenant-Id:
    Bearer <token>`` — an anonymous call gets 401. Same pattern as ``BrandExtractorClient``.
    """

    def __init__(
        self,
        base_url: str | None = None,
        license_token: str | None = None,
        **kwargs: Any,  # noqa: ANN401 — why: opaque pass-through to LicensedServiceClient (timeouts, retries); narrowing here would duplicate the base signature
    ) -> None:
        super().__init__(
            service_name="bdd_playwright",
            base_url=base_url,
            license_token=license_token,
            **kwargs,
        )

    async def start_qa_suite(self, payload: dict) -> str:
        """Start a QA test suite generation job, returns job_id."""
        return await self.submit_job("/api/bdd/qa-suite", payload)

    async def start_accessibility(self, payload: dict) -> str:
        """Start an accessibility audit generation job, returns job_id."""
        return await self.submit_job("/api/bdd/accessibility", payload)

    async def start_documentation(self, payload: dict) -> str:
        """Start an interactive documentation generation job, returns job_id."""
        return await self.submit_job("/api/bdd/documentation", payload)

    async def start_demo_video(self, payload: dict) -> str:
        """Start a demo/tutorial video generation job, returns job_id."""
        return await self.submit_job("/api/bdd/demo-video", payload)

    async def get_job(self, job_id: str) -> BddJobStatus:
        """Get BDD job status."""
        data = await self._request("GET", f"/api/bdd/{job_id}")
        return BddJobStatus.model_validate(data)

    async def get_job_result(self, job_id: str) -> BddJobResult:
        """Get full BDD job result."""
        data = await self._request("GET", f"/api/bdd/{job_id}/result")
        return BddJobResult.model_validate(data)

    async def wait_for_job(self, job_id: str, timeout: float = 600.0) -> BddJobStatus:
        """Poll until BDD job completes.

        ``poll_job`` polls ``GET /api/bdd/{job_id}``, so the settled value is the
        status envelope. It raises on downstream failure/timeout rather than
        returning a failed job.
        """
        data = await self.poll_job("/api/bdd", job_id, timeout=timeout)
        return BddJobStatus.model_validate(data)
