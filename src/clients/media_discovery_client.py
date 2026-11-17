"""HTTP client for media-discovery's federated media search.

Two orchestrator consumers: ``routes/media.py`` (the wizard's "Find media"
panel) and ``stages/generic/media_discovery.py`` (the pipeline-auto surface).

media-discovery is called directly; KAG, which used to proxy it under
``/media/*``, is retired. media-discovery is behind licence auth (only
``/health*`` is exempt) and accepts ``X-Tenant-Id: Bearer <licence>``; the
licence comes from the ``media_discovery`` hop lane (``MEDIA_DISCOVERY_AUTH_MODE``,
app key ``w2v`` — w2v_common ``HOP_APP_KEYS``). It emits no cost events.
"""

from __future__ import annotations

from typing import Final

from src.clients.base import LicensedServiceClient
from src.clients.schemas.media_discovery import MediaSearchResponse

#: media-discovery's own bound: ``SearchRequest.limit = Field(default=20, ge=1, le=100)``
#: (media-discovery ``src/schemas.py``). Outside it the search is a 422.
MEDIA_DISCOVERY_MAX_LIMIT: Final = 100


def clamp_search_limit(limit: int) -> int:
    """``limit`` clamped into media-discovery's accepted ``[1, 100]``."""
    return max(1, min(limit, MEDIA_DISCOVERY_MAX_LIMIT))


class MediaDiscoveryClient(LicensedServiceClient):
    """Client for media-discovery ``POST /api/search``."""

    def __init__(
        self,
        base_url: str | None = None,
        license_token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(
            service_name="media_discovery",
            base_url=base_url,
            license_token=license_token,
            timeout=timeout,
            licence_hop="media_discovery",  # MEDIA_DISCOVERY_AUTH_MODE
        )

    async def search(
        self,
        query: str,
        *,
        sources: list[str] | None = None,
        media_types: list[str] | None = None,
        limit: int = 20,
    ) -> MediaSearchResponse:
        """Federated search across media-discovery's adapters (``POST /api/search``).

        ``sources`` names the adapters to query (e.g. youtube/wikimedia);
        ``media_types`` filters by kind (image/video/...). Both omitted ⇒ search
        all configured sources.
        """
        payload: dict[str, object] = {"query": query, "limit": clamp_search_limit(limit)}
        if sources:
            payload["sources"] = sources
        if media_types:
            payload["media_types"] = media_types
        data = await self._request("POST", "/api/search", json=payload)
        return MediaSearchResponse.model_validate(data)
