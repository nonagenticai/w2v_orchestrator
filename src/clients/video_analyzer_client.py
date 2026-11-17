"""HTTP client for video_analyzer_service."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

from src.clients.base import LicensedServiceClient
from src.clients.schemas.video_analyzer import VideoAnalyzerAudioGrid, VideoAnalyzerJobResult

logger = logging.getLogger(__name__)


class VideoAnalyzerClient(LicensedServiceClient):
    """Async client for the video_analyzer_service /api/analyze endpoints.

    video_analyzer_service is behind license-authz, so this extends
    ``LicensedServiceClient`` to bake in the orchestrator's service-account
    token (``X-Tenant-Id: Bearer <token>``) and resolve its base URL via
    discovery (``video_analyzer`` → in-cluster :6317) when none is passed.
    """

    def __init__(
        self,
        base_url: str | None = None,
        license_token: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(
            service_name="video_analyzer",
            base_url=base_url,
            license_token=license_token,
            timeout=timeout,
        )

    async def start_analysis(
        self,
        *,
        video_path: str | None = None,
        video_url: str | None = None,
        verification: dict[str, Any] | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> str:
        """POST /api/analyze and return the assigned ``job_id``."""
        payload: dict[str, Any] = {}
        if video_path is not None:
            payload["video_path"] = video_path
        if video_url is not None:
            payload["video_url"] = video_url
        if verification is not None:
            payload["verification"] = verification
        if config_overrides:
            payload.update(config_overrides)

        result = await self._request("POST", "/api/analyze", json=payload)
        job_id: str = result.get("job_id", "")
        if not job_id:
            raise RuntimeError("video_analyzer did not return a job_id")
        return job_id

    async def start_analysis_upload(
        self,
        *,
        file_path: Path,
        config: dict[str, Any] | None = None,
    ) -> str:
        """POST /api/analyze/upload (multipart) and return the assigned ``job_id``."""
        if not file_path.exists():
            raise FileNotFoundError(f"upload not found: {file_path}")
        client = await self._get_client()
        with file_path.open("rb") as fh:
            files = {"file": (file_path.name, fh, "video/mp4")}
            data: dict[str, str] = {}
            if config is not None:
                import json as _json

                data["config"] = _json.dumps(config)
            try:
                resp = await client.post("/api/analyze/upload", files=files, data=data)
            except httpx.HTTPError as exc:
                raise RuntimeError(f"video_analyzer upload failed: {exc}") from exc
        if resp.status_code >= 400:
            raise RuntimeError(
                f"video_analyzer upload returned {resp.status_code}: {resp.text[:500]}"
            )
        job_id: str = resp.json().get("job_id", "")
        if not job_id:
            raise RuntimeError("video_analyzer did not return a job_id")
        return job_id

    async def poll_analysis(
        self,
        job_id: str,
        *,
        timeout: float = 600.0,
        interval: float = 2.0,
    ) -> dict[str, Any]:
        """Poll the job until it completes or fails."""
        return await self.poll_job(
            "/api/analyze",
            job_id,
            interval=interval,
            timeout=timeout,
        )

    async def get_result(self, job_id: str) -> VideoAnalyzerJobResult:
        """Fetch the full job result, validated to the server's wire shape.

        The real field is ``frame_analyses`` (NOT ``frames``): the route declares
        ``response_model=JobResultResponse`` so FastAPI strips any other key.
        """
        data = await self._request("GET", f"/api/analyze/{job_id}/result")
        return VideoAnalyzerJobResult.model_validate(data)

    async def analyze_audio(self, *, audio_url: str) -> VideoAnalyzerAudioGrid:
        """POST /api/analyze/audio — the beat grid of a music bed, SYNCHRONOUSLY.

        Deliberately NOT submit→poll like ``start_analysis``: beat tracking a
        two-minute bed is a few seconds of DSP (no vision model, no frame
        extraction), so the analyzer answers on the request and a job envelope
        would only add two round-trips and a lost-job failure mode.

        ``audio_url`` is a presigned GET the analyzer fetches itself — the bed
        lives in the ORCHESTRATOR's own bucket (``sound_library/…``), which the
        analyzer has no credentials for, so handing it a raw object key would
        404 server-side. Mint the URL right before the call: presigns expire in
        an hour and a paused run would otherwise send a dead signature.
        """
        result = await self._request("POST", "/api/analyze/audio", json={"audio_url": audio_url})
        return VideoAnalyzerAudioGrid.model_validate(result)

    async def get_verification_result(self, job_id: str) -> dict[str, Any]:
        """Fetch the verification report (404 if no criteria were submitted)."""
        return await self._request("GET", f"/api/analyze/{job_id}/verification")

    async def download_artifact(self, job_id: str, name: str) -> bytes:
        """Stream a single artifact and return its raw bytes."""
        resp = await self._request_raw("GET", f"/api/analyze/{job_id}/artifact/{name}")
        return resp.content

    async def delete_job(self, job_id: str) -> None:
        await self._request("DELETE", f"/api/analyze/{job_id}")
