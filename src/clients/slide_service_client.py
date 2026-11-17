"""HTTP client for Slide Service (port 6312).

Extends ``BaseServiceClient`` so the orchestrator's service-account license
token is baked into ``X-Tenant-Id: Bearer <token>``. Without it, anonymous
calls hit slide-service license-authz and get back 401 — surfaced by the
course/lesson E2E where ``lesson_record`` POSTs ``/api/decks`` and the call
was rejected before this client forwarded the token.
"""

from __future__ import annotations

from typing import Any

from w2v_common.clients.base import raise_for_status_with_body

from src.clients.base import LicensedServiceClient
from src.clients.schemas.slide_service import SlideDeckStatus


class SlideServiceClient(LicensedServiceClient):
    """Client for the slide deck generation microservice."""

    def __init__(
        self,
        base_url: str | None = None,
        license_token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(
            service_name="slide_service",
            base_url=base_url,
            license_token=license_token,
            timeout=timeout,
        )

    async def resolve_template_id(self, name: str) -> str:
        """Return the id of a seeded template by its machine ``name``.

        slide-service seeds system templates at startup (``title``, ``agenda``,
        ``feature_spotlight``, ``tutorial_step``, ...). Stages resolve by stable
        name rather than hardcoding a per-environment UUID.
        """
        data = await self._request("GET", "/api/templates")
        templates = data.get("templates", []) if isinstance(data, dict) else []
        for tpl in templates:
            if tpl.get("name") == name:
                return str(tpl["id"])
        available = ", ".join(sorted(str(t.get("name")) for t in templates))
        raise RuntimeError(f"slide template {name!r} not found. Available: {available}")

    async def start_deck(
        self, payload: dict[str, Any], *, tokens: dict[str, str] | None = None
    ) -> str:
        """Submit deck generation job (POST /api/decks), returns job_id.

        ``tokens`` is the run's resolved C23 design-token bundle (palette + type
        scale + grid + font/icon set; see ``design_system.as_params``). slide_service
        consumes it as CSS custom properties with today's values as fallbacks, so it
        is presence-is-opt-in: a falsy ``tokens`` submits the SAME ``payload`` object
        with zero keys added, keeping the request byte-identical to today.
        """
        body = {**payload, "tokens": tokens} if tokens else payload
        return await self.submit_job("/api/decks", body)

    async def wait_for_deck(self, job_id: str, timeout: float = 300.0) -> SlideDeckStatus:
        """Poll until deck generation completes.

        Validates to the typed mirror in ``src.clients.schemas.slide_service``
        (``DeckStatusResponse``; server file:line cited there).
        """
        data = await self.poll_job("/api/decks", job_id, timeout=timeout)
        return SlideDeckStatus.model_validate(data)

    async def get_slide_png(self, job_id: str, slide_index: int) -> bytes:
        """Download a single rendered slide preview as PNG bytes."""
        client = await self._get_client()
        response = await client.get(f"/api/decks/{job_id}/slides/{slide_index}.png")
        raise_for_status_with_body(response, "slide_service")
        return response.content
