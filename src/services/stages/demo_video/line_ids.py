"""The authored-line join id, shared by ``demo_video_gen`` and ``showcase_slides``.

Its own module so ``showcase_slides`` can use it without importing ``gen`` — whose
LLM/TTS clients would otherwise count against the slides stage's
``provider_slots`` (``tests/unit/test_stage_provider_slots.py`` scans imports).
"""

from __future__ import annotations

from typing import Any

from src.models import normalize_clip_key


def authored_line_id(raw: dict[str, Any], idx: int) -> str:
    """The join id for an authored brief line — shared by gen and showcase_slides
    so a card line's narration, slide image, and shot all key on the same id.

    Normalized to the FTR ``clip-<n>`` form (a leading ``@`` stripped) so an id given
    as the ``@clip-1`` tag still joins to its recorded footage — and so both callers
    derive the SAME id from the raw brief.
    """
    base = str(raw.get("id") or raw.get("line_id") or "").strip() or f"clip-{idx + 1}"
    return normalize_clip_key(base)
