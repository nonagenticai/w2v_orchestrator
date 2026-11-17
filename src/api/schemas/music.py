"""Music-track catalogue schemas — the music beds exposed to the studio UI.

Mirrors the ``category == "music"`` slice of the sound catalog
(``services/stages/sound_catalog``) so the frontend can render its music-bed
picker from the API instead of a hardcoded list.
"""

from __future__ import annotations

from pydantic import BaseModel

from ._shared import _camel


class MusicTrack(BaseModel):
    """One selectable music bed (a ``music``-category asset in the catalog).

    ``id`` is the catalog id a run's config block references; ``label`` is the
    human/UI-readable description; ``category`` is always ``"music"`` here.
    """

    model_config = _camel
    id: str
    label: str
    category: str


class MusicTracksResponse(BaseModel):
    """The music beds available to the caller's tenant (built-in + manifests)."""

    model_config = _camel
    tracks: list[MusicTrack]
