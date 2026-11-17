"""Typed responses for slide_service's deck API (port 6312).

Source of truth (read, not guessed) — slide_service ``src/api/schemas.py``:
- ``GET /api/decks/{job_id}`` -> ``DeckStatusResponse`` (:224-231); the route
  ``get_deck_status`` declares that return type (slide_service
  ``src/api/routes.py:657``), so FastAPI serializes through it and every field
  below is exactly what the server model declares — nothing here invents one.

``status`` is deliberately NOT ``Literal``: a new service-side status string
must not be able to hard-fail a running pipeline. The orchestrator's lone
consumer discards the body (it polls only to await terminal state), but
``status`` / ``slide_count`` are the fields the audit/lesson/promo mock harnesses
read, and they mirror the server model 1:1.
"""

from __future__ import annotations

from pydantic import BaseModel


class SlideDeckStatus(BaseModel):
    """``GET /api/decks/{job_id}`` — mirrors ``DeckStatusResponse``."""

    job_id: str
    status: str
    slide_count: int | None = None
    created_at: str | None = None
    error: str | None = None
