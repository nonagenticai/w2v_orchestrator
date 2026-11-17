"""Typed responses for media-discovery's federated search (``POST /api/search``).

Source of truth (read, not guessed): media-discovery ``src/schemas.py`` —
``SearchResponse`` (:86-91), items are ``MediaItemResponse`` (:43-66). The route
declares ``response_model=SearchResponse`` (media-discovery
``src/routers/search.py``, mounted at ``/api/search`` in ``src/app.py``), so every
declared field is guaranteed present on the wire and any undeclared key is
stripped.

The orchestrator calls media-discovery directly (KAG, which used to proxy this
JSON verbatim under ``/media/api/search``, is retired).
"""

from __future__ import annotations

from pydantic import BaseModel


class MediaItem(BaseModel):
    """One federated media search hit — mirrors ``MediaItemResponse``.

    Only the fields the orchestrator's consumers read (``routes/media.py`` +
    ``stages/generic/media_discovery.py``) are declared; pydantic's default
    ``extra="ignore"`` drops the rest (``tenant_id``, ``external_ids``,
    timestamps, ...) without coupling us to them. ``quality_score`` is nullable
    server-side (unscored items) — consumers fold it with ``or 0.0``.
    """

    id: str
    title: str
    url: str
    media_type: str
    source: str
    thumbnail_url: str | None = None
    author_name: str | None = None
    quality_score: float | None = None


class MediaSearchResponse(BaseModel):
    """``POST /api/search`` — mirrors ``SearchResponse``.

    All five fields are required server-side and serialized through
    ``response_model=SearchResponse``, so they are always on the wire.
    """

    results: list[MediaItem]
    total: int
    sources_queried: list[str]
    sources_failed: list[str]
    search_time_ms: float
