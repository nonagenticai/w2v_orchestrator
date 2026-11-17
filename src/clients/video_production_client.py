"""HTTP client for Video Production Service (port 6308).

Recording endpoints (``/api/record/*``) of the merged video_production_service.
The compilation-half methods (``/api/compile/*``) were dead code — every
compile path in the orchestrator goes through ``VideoCompileClient`` — and
were deleted rather than typed.
"""

from __future__ import annotations

from w2v_common.clients.base import raise_for_status_with_body

from src.clients.base import LicensedServiceClient


class VideoProductionClient(LicensedServiceClient):
    """Client for the merged Video Production microservice.

    Exposes the recording endpoints (``/api/record/*``).

    Like every other downstream client, the orchestrator's service-account
    license token is baked into ``X-Tenant-Id: Bearer <token>`` (and per-request
    identity ContextVars X-License-Id / X-Subject-Id are forwarded via the shared
    ``identity_event_hook``). Without this, anonymous calls hit license-authz and
    get back 401 — which is exactly what site_capture's ``/api/record/url`` call
    did until this token was wired in.
    """

    def __init__(
        self,
        base_url: str | None = None,
        license_token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(
            service_name="video_production",
            base_url=base_url,
            license_token=license_token,
            timeout=timeout,
        )

    # ── Recording endpoints ───────────────────────────────────────────

    async def record_url(
        self,
        url: str,
        part_id: str = "site",
        auth_info: dict[str, object] | None = None,
        recording_config: dict[str, object] | None = None,
        scroll_config: dict[str, object] | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        """Record a live URL (e.g. an authenticated SPA) as a video, returns job_id.

        ``idempotency_key`` (sent as the Idempotency-Key header) makes a re-submit
        of the SAME body return the ORIGINAL job instead of a second paid
        recording — see :meth:`record_animation`.
        """
        payload: dict[str, object] = {"url": url, "part_id": part_id}
        if auth_info:
            payload["auth_info"] = auth_info
        if recording_config:
            payload["recording_config"] = recording_config
        if scroll_config:
            payload["scroll_config"] = scroll_config
        result = await self._request(
            "POST", "/api/record/url", json=payload, idempotency_key=idempotency_key
        )
        return result.get("job_id", "")

    async def record_animation(
        self,
        *,
        template_id: str | None = None,
        html: str | None = None,
        params: dict[str, object] | None = None,
        part_id: str = "motion",
        fps: int = 30,
        width: int | None = None,
        height: int | None = None,
        duration_ms: int | None = None,
        output_format: str = "mp4",
        transparent: bool = False,
        idempotency_key: str | None = None,
        quality_preset: str | None = None,
    ) -> str:
        """Render a branded motion-graphics template to a clip, returns job_id.

        Supply either ``template_id`` (library template) or raw ``html``;
        ``params`` becomes ``window.PARAMS`` so one template serves every brand.
        Capture is deterministic (frame-stepped). Poll via
        :meth:`wait_for_content_recording` and fetch with
        :meth:`get_recording_video` (shared /jobs endpoints).

        ``idempotency_key`` (sent as the Idempotency-Key header) makes a
        re-submit of the SAME clip — e.g. ``motion_graphics_gen`` retrying its
        per-item callable after a transient wait failure — return the ORIGINAL
        job instead of spawning a SECOND paid recording (the service dedupes on
        it server-side). Without it, the re-submit double-bills.

        ``transparent`` renders the clip with a REAL ALPHA CHANNEL (VP9 WebM) so
        it can be composited OVER footage as a ``Shot.overlay_clips`` entry
        instead of cutting full-frame. Two service-side rules the caller has to
        respect: the service forces/requires a WebM container (an explicit
        ``output_format="mp4"``/``"gif"`` is REJECTED rather than silently
        flattened to an opaque clip — h264 has no alpha), and only
        overlay-designed templates accept it (``lower_third`` is the only one
        today; every other library template paints its own opaque stage and the
        route 400s). Inline ``html`` owns its own backdrop and is always allowed.

        The flag is sent ONLY when True, so every existing caller's payload is
        byte-identical to one issued before this parameter existed.

        ``quality_preset`` names the service's encode preset (e.g. the
        ``high_quality`` / ``2k`` / ``4k`` tiers site_capture records at). Without
        it the service encodes at ``standard`` (CRF 23) and the compiler
        re-encodes that — baked-in generational loss. Sent ONLY when not None, so
        existing callers stay byte-identical and an older service never sees the
        key; an unknown name is a 422 from the service.
        """
        payload: dict[str, object] = {
            "part_id": part_id,
            "fps": fps,
            "output_format": output_format,
        }
        if transparent:
            payload["transparent"] = True
        if template_id:
            payload["template_id"] = template_id
        if html:
            payload["html"] = html
        if params is not None:
            payload["params"] = params
        if width:
            payload["width"] = width
        if height:
            payload["height"] = height
        if duration_ms:
            payload["duration_ms"] = duration_ms
        if quality_preset is not None:
            payload["quality_preset"] = quality_preset
        result = await self._request(
            "POST", "/api/record/animation", json=payload, idempotency_key=idempotency_key
        )
        return result.get("job_id", "")

    async def capture_screens(
        self,
        url: str,
        steps: list[dict[str, object]],
        part_id: str = "screens",
        auth_info: dict[str, object] | None = None,
        recording_config: dict[str, object] | None = None,
        record_clips: bool = False,
        idempotency_key: str | None = None,
    ) -> str:
        """Capture per-screen assets for a demo in a single pass, returns job_id.

        Default mode produces a single ``.zip`` of ``<capture_id>.png`` stills.
        With ``record_clips=True`` the recorder also records a short interaction
        *clip* per screen (the click landing on it), so the zip holds BOTH
        ``<capture_id>.mp4`` and ``<capture_id>.png``. Either way the zip is
        downloaded via :meth:`get_recording_video` (shared /jobs + /video
        endpoints), so frame-synced promo render keys each screen to its
        narration line.

        ``idempotency_key``: as :meth:`record_url`.
        """
        payload: dict[str, object] = {"url": url, "part_id": part_id, "steps": steps}
        if auth_info:
            payload["auth_info"] = auth_info
        if recording_config:
            payload["recording_config"] = recording_config
        if record_clips:
            payload["record_clips"] = True
        result = await self._request(
            "POST", "/api/record/capture/screens", json=payload, idempotency_key=idempotency_key
        )
        return result.get("job_id", "")

    async def wait_for_content_recording(
        self, job_id: str, timeout: float = 300.0
    ) -> dict[str, object]:
        """Poll until a content / URL recording completes (shared /jobs endpoint)."""
        return await self.poll_job("/api/record/jobs", job_id, timeout=timeout)

    async def get_recording_video(self, job_id: str) -> tuple[bytes, str, str | None]:
        """Fetch recorded video bytes + response metadata."""
        client = await self._get_client()
        response = await client.get(f"/api/record/jobs/{job_id}/video")
        raise_for_status_with_body(response, "video_production")
        content_type = response.headers.get("content-type", "video/mp4")
        disposition = response.headers.get("content-disposition")
        return response.content, content_type, disposition
