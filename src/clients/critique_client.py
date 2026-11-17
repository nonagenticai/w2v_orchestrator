"""HTTP client for Critique Service (port 6303 — served by Analysis Service)."""

from __future__ import annotations

from src.clients.base import LicensedServiceClient
from src.clients.schemas.critique import CritiqueComparison, CritiqueResult


class CritiqueClient(LicensedServiceClient):
    """Client for the Critique microservice.

    analysis_service is behind license-authz (only ``/health*`` is exempt), so
    the orchestrator's service-account licence token is baked into
    ``X-Tenant-Id: Bearer <token>`` by ``BaseServiceClient`` — anonymous calls
    get 401. Same pattern as ``BrandExtractorClient`` / ``SlideServiceClient``.
    """

    def __init__(
        self,
        base_url: str | None = None,
        license_token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(
            service_name="critique",
            discovery_name="analysis",
            base_url=base_url,
            license_token=license_token,
            timeout=timeout,
        )

    async def start_critique(
        self,
        url: str,
        categories: list[str] | None = None,
        llm_config: dict[str, object] | None = None,
        auth_info: dict[str, object] | None = None,
    ) -> str:
        """Start critique job (POST /api/critique), returns the job_id.

        ``auth_info`` is an opaque dict validated server-side against
        ``CritiqueRequest.auth_info`` (analysis_service ``schemas.py:190``);
        ``llm_config`` maps to ``CritiqueRequest.llm_config`` (:189).
        """
        payload: dict[str, object] = {"url": url}
        if categories:
            payload["categories"] = categories
        if llm_config:
            payload["llm_config"] = llm_config
        if auth_info:
            payload["auth_info"] = auth_info
        return await self.submit_job("/api/critique", payload)

    async def get_critique(self, job_id: str) -> CritiqueResult:
        """Get critique result (GET /api/critique/{job_id})."""
        data = await self._request("GET", f"/api/critique/{job_id}")
        return CritiqueResult.model_validate(data)

    async def compare_critiques(
        self, baseline_job_id: str, candidate_job_id: str
    ) -> CritiqueComparison:
        """Compare two critique jobs and return score/finding deltas."""
        data = await self._request(
            "GET",
            "/api/critique/compare",
            params={"job_id_1": baseline_job_id, "job_id_2": candidate_job_id},
        )
        return CritiqueComparison.model_validate(data)

    async def wait_for_critique(self, job_id: str, timeout: float = 300.0) -> CritiqueResult:
        """Poll until critique completes.

        Validates to the typed mirror ``CritiqueResult``
        (``src.clients.schemas.critique``; server file:line cited there).
        """
        data = await self.poll_job("/api/critique", job_id, timeout=timeout)
        return CritiqueResult.model_validate(data)
