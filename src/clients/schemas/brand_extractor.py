"""Typed response for analysis_service's brand-extraction API (port 6303).

Source of truth (read, not guessed) — analysis_service
``src/domains/brand/schemas.py:160-195`` defines ``BrandResponse``, the
``response_model`` of ``POST /api/brand/from-url``.

This mirror is deliberately FOCUSED: only the 7 scalar fields
``resolve_brand_context`` actually reads are declared, each with the server's
own default value. The full ``BrandResponse`` carries nested sub-models and
``dict[str, Any] | None`` blobs — mirroring those would re-introduce ``Any``.
``extra="ignore"`` (pydantic's default) drops every field not declared here.
"""

from __future__ import annotations

from pydantic import BaseModel


class BrandExtraction(BaseModel):
    """Focused mirror of ``BrandResponse`` — the 7 scalar fields the orchestrator reads.

    Field defaults match the server (``schemas.py:160-195``): a field absent
    from the JSON body resolves to the same value the server model would have
    produced.
    """

    name: str = ""  # server :163
    logo_url: str = ""  # server :165
    primary_color: str = "#2563EB"  # server :166
    secondary_color: str = "#1E40AF"  # server :167
    font_family_primary: str = ""  # server :192
    font_family_secondary: str = ""  # server :193
    brand_voice: str = ""  # server :180
