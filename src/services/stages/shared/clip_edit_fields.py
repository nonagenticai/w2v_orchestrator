"""The per-clip fields a Studio edit may set on a clip-spine element.

A dependency-free leaf on purpose: ``runs/regenerate.py`` patches these onto the
spine (apply-edits) and every compile stage that rebuilds its shot dicts from the
spine must carry exactly these. Importing them from ``runs/elements`` put a compile
stage behind ``services.blueprints`` — whose package import registers the blueprints,
which import the compile stages — a circular import.

A compile stage that rebuilds the spine dict field-by-field and does NOT carry them
turns every edit into a green no-op re-stitch.
"""

from __future__ import annotations

from typing import Final

EDITABLE_CLIP_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "clip_lead_seconds",
        "clip_out_seconds",
        "clip_anchor",
        "caption",
        "fit",
        "speed",
        "transition",
        "adjust",
        "overlays",
        "audio",
        "audio_url",
        "audio_seconds",
    }
)
