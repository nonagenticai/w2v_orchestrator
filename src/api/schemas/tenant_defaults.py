"""P6 ORCH-1 schemas: tenant production defaults (``/api/tenant/defaults``).

Every field is optional and ``null`` means "use the deployment default". ``PUT``
replaces the whole document (an omitted block is cleared), guarded by
``expected_version``. Unknown keys are refused (``extra="forbid"``) so a typo cannot be
silently stored as a default nothing reads.

There is no logo field: the only brand-logo reader in the pipeline is the advisory
asset-rights ledger (``brand.logo_url``), which nothing renders, so a stored logo
would be a dead default. It comes back with the first stage that draws one.
"""

from __future__ import annotations

import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ._shared import CaptionsMode, CaptionsModeInput

_cfg = ConfigDict(populate_by_name=True, str_strip_whitespace=True, extra="forbid")

RetentionMode = Literal["auto", "critique_only", "off"]
HexColor = Annotated[str, Field(pattern=r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")]
#: The inference-router tier aliases the timing director can be routed to — the form
#: ``PIPELINE_TIMING_DIRECTOR_MODEL_TIER`` takes. The value travels as WM
#: ``model_override``; WM refuses a name the router's ``/model/info`` does not list, and
#: the router only lists the ``model_aliases`` of ENABLED providers. Source of truth:
#: inference-router ``providers.yaml`` (deepseek: ``tier-medium``/``tier-flash``,
#: xai: ``tier-high``). A tier added there must be added here. Any other value would
#: be stored and then fail every reconcile_timeline relief call of every run.
TimingDirectorTier = Literal["tier-flash", "tier-medium", "tier-high"]


class VoiceDefaults(BaseModel):
    model_config = _cfg
    #: Checked against the voices the tenant's narration path accepts by the PUT route
    #: (it needs the tenant's ``tts`` slot, so it cannot be a schema rule).
    voice_id: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    speed: Annotated[float, Field(ge=0.5, le=2.0)] | None = None


class BrandDefaults(BaseModel):
    model_config = _cfg
    primary_color: HexColor | None = None
    secondary_color: HexColor | None = None
    font_family: Annotated[str, Field(min_length=1, max_length=128)] | None = None


class CaptionsDefaults(BaseModel):
    model_config = _cfg
    mode: CaptionsModeInput | None = None
    karaoke: bool | None = None


class VoiceDirectionDefaults(BaseModel):
    model_config = _cfg
    tone: Annotated[str, Field(min_length=1, max_length=64)] | None = None


class ModelTierDefaults(BaseModel):
    model_config = _cfg
    timing_director: TimingDirectorTier | None = None


class TenantDefaultsDocument(BaseModel):
    """The settings document (shared by the PUT body and the GET view)."""

    model_config = _cfg
    voice: VoiceDefaults | None = None
    brand: BrandDefaults | None = None
    captions: CaptionsDefaults | None = None
    voice_direction: VoiceDirectionDefaults | None = None
    retention_mode: RetentionMode | None = None
    model_tier: ModelTierDefaults | None = None


class TenantDefaultsPut(TenantDefaultsDocument):
    #: The version the editor loaded (``0`` = the tenant has no defaults yet).
    expected_version: Annotated[int, Field(ge=0)]


class DeploymentDefaults(BaseModel):
    """What applies where the tenant leaves a field ``null`` (read-only)."""

    model_config = _cfg
    voice_id: str | None = None
    captions_mode: CaptionsMode
    captions_karaoke: bool
    voice_direction_tone: str | None
    retention_mode: RetentionMode | None
    timing_director_model_tier: str | None


class TenantDefaultsView(TenantDefaultsDocument):
    #: ``0`` when the tenant has never saved defaults.
    version: int
    updated_at: datetime.datetime | None = None
    updated_by: UUID | None = None
    deployment: DeploymentDefaults
