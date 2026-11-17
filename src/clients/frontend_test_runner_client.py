"""HTTP client for Frontend Test Runner Service (port 6316)."""

from __future__ import annotations

from w2v_common.clients.base import raise_for_status_with_body

from src.clients.base import LicensedServiceClient


class FrontendTestRunnerClient(LicensedServiceClient):
    """Client for the frontend test runner microservice.

    frontend_test_runner is behind license-authz (only ``/health*`` is exempt), so
    the orchestrator's service-account token must be presented as ``X-Tenant-Id:
    Bearer <token>`` — an anonymous call gets 401. Same pattern as ``BrandExtractorClient``.
    """

    def __init__(
        self,
        base_url: str | None = None,
        license_token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(
            service_name="frontend_test_runner",
            base_url=base_url,
            license_token=license_token,
            timeout=timeout,
        )

    async def start_execution(
        self, payload: dict[str, object], *, idempotency_key: str | None = None
    ) -> str:
        """Start a test execution job, returns job_id.

        ``idempotency_key`` (sent as the Idempotency-Key header) makes a re-submit
        of the SAME execution — e.g. demo_video_record retrying after a resumable
        timeout — return the ORIGINAL job instead of spawning a second paid
        bddgen+Playwright recording run (FTR dedupes on it server-side).
        """
        return await self.submit_job(
            "/api/test-runner/execute", payload, idempotency_key=idempotency_key
        )

    async def get_job(self, job_id: str) -> dict[str, object]:
        """Get test execution job status."""
        return await self._request("GET", f"/api/test-runner/{job_id}")

    async def get_job_result(self, job_id: str) -> dict[str, object]:
        """Get full test execution result."""
        return await self._request("GET", f"/api/test-runner/{job_id}/result")

    async def wait_for_execution(self, job_id: str, timeout: float = 600.0) -> dict[str, object]:
        """Poll until test execution job completes."""
        return await self.poll_job("/api/test-runner", job_id, timeout=timeout)

    async def get_clip_video(self, job_id: str, clip_key: str) -> bytes:
        """Fetch a recorded per-scenario clip's raw bytes via the stream-through route.

        The orchestrator re-publishes these into its own storage (so the
        video-compile service can fetch them anonymously) — the same download +
        ``store_capture_clip`` pattern ``site_capture`` uses with VPS recordings.
        """
        client = await self._get_client()
        response = await client.get(f"/api/test-runner/{job_id}/clips/{clip_key}/video")
        raise_for_status_with_body(response, "frontend_test_runner")
        return response.content
