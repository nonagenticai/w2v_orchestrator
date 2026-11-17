"""P6 ORCH-4 schemas: the tenant music library (``/api/tenant/music``).

A tenant track's rights are TENANT-DECLARED (``rights`` is always
``"tenant-declared"``): a tenant_admin attested them on upload. They are never
presented as verified royalty-free — that label belongs to the bundled library only.
``url`` is the same-origin ``/api/tenant/music/{id}/content`` proxy path (the browser
cannot resolve the ``minio.test`` presign host). ``limits`` carries the upload
constraints the server enforces, so the UI never hardcodes them.
"""

from __future__ import annotations

import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

_cfg = ConfigDict(populate_by_name=True, str_strip_whitespace=True, extra="forbid")

TenantSoundCategory = Literal["music", "ambient", "accent", "transition"]

#: Longest track description (upload form field and PATCH body).
DESCRIPTION_MAX_LENGTH = 500


class TenantMusicTrack(BaseModel):
    id: UUID
    #: The catalog id a run references (``tenant_<slug>``); also listed by
    #: ``GET /api/music/tracks`` for music-category tracks.
    sound_id: str
    category: TenantSoundCategory
    description: str
    duration_s: float
    bytes: int
    content_type: str
    rights: Literal["tenant-declared"] = "tenant-declared"
    rights_attested_by: UUID
    created_at: datetime.datetime
    url: str | None = None


class TenantMusicFormat(BaseModel):
    #: The sniffed container (``mp3`` / ``wav`` / ``ogg`` / ``flac``).
    format: str
    extension: str
    content_type: str


class TenantMusicLimits(BaseModel):
    #: ``STORAGE_TENANT_MUSIC_MAX_BYTES`` — a larger upload is a 422 ``FILE_TOO_LARGE``.
    max_bytes: int
    #: ``None``: no duration cap (any decodable, non-zero duration is accepted).
    max_duration_s: float | None = None
    max_description_length: int
    categories: list[TenantSoundCategory]
    #: Accepted containers. The format is SNIFFED from the bytes; the filename and
    #: client content-type are ignored (use these for the picker's ``accept``).
    formats: list[TenantMusicFormat]


class TenantMusicList(BaseModel):
    tracks: list[TenantMusicTrack]
    limits: TenantMusicLimits


class TenantMusicPatch(BaseModel):
    model_config = _cfg
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX_LENGTH)
    category: TenantSoundCategory | None = None
