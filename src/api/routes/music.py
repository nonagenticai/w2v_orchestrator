"""Music-track catalogue — the music beds the studio can score a video with.

Read-only, tenant-aware surface over the manifest-driven sound catalog
(``services/stages/sound_catalog``). The frontend renders its music-bed picker
from this endpoint instead of a hardcoded id list; the catalog is the single
source of truth (built-in baseline layered with the shared and per-tenant
manifests).
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.routes._common import TenantIdDep
from src.api.schemas import MusicTrack, MusicTracksResponse
from src.services.stages.sound_catalog import load_sound_library

router = APIRouter(prefix="/api/music", tags=["music"])


@router.get("/tracks", response_model=MusicTracksResponse, operation_id="music.tracks.list")
async def list_music_tracks(tenant_id: TenantIdDep) -> MusicTracksResponse:
    """List the music beds available to the caller's tenant.

    Reuses ``load_sound_library`` — the tenant argument layers the caller's
    per-tenant manifest over the shared library — and returns only the
    ``music``-category assets (sorted by id), which are the beds a run's config
    block may score with.
    """
    library = await load_sound_library(tenant_id=str(tenant_id))
    tracks: list[MusicTrack] = []
    for track_id in library.ids_in("music"):
        asset = library.get(track_id)
        if asset is None:  # unreachable — id came from the library itself
            continue
        tracks.append(
            MusicTrack(id=track_id, label=asset.description or track_id, category=asset.category)
        )
    return MusicTracksResponse(tracks=tracks)
