"""Declarative authoring front-door schemas (great-video A0).

A single, ergonomic request contract an author (or an upstream tool) submits to
describe the video they want — the ``VideoIntent`` — plus the ``PlanDocument`` the
pipeline emits back as the resolved, knob-by-knob plan. Both sit at the API door
only; the persisted shape stays the free-form ``brief: dict`` /
``config_snapshot`` (same contract as ``ShowcaseBrief`` in ``projects.py``).
``extra='allow'`` on the author-facing models keeps every not-yet-typed key
round-tripping untouched — only the typed fields are validated here.

``PlanDocument.to_brief_patch`` is the one behavioural bridge: a presence-opt-in
merge into ``config_snapshot['brief']`` that can only ADD authored values, never
blank an existing one (an empty plan is a no-op).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from src.models import AuthoredLine

from ._shared import ProjectType


class AspectStyle(StrEnum):
    """Per-aspect render style. PERMISSIVE at the door — accepted as
    ``AspectStyle | str`` (like every ``showcase_line`` panel enum) so an
    out-of-vocabulary style still round-trips instead of 422-ing at the model."""

    PRODUCT_DEMO_FORWARD = "product_demo_forward"
    SCREENCAST = "screencast"
    KINETIC_TYPOGRAPHY = "kinetic_typography"
    TALKING_HEAD = "talking_head"
    FOUNDER_STORY = "founder_story"


class PackagingIntent(BaseModel):
    """Authoring hints for the deliverable's packaging (title / description /
    thumbnail). ``extra='allow'`` — forward-compat for packaging knobs not yet
    typed here."""

    model_config = ConfigDict(extra="allow")
    title: str | None = None
    description: str | None = None
    thumbnail_text: str | None = None


class VideoIntent(BaseModel):
    """The declarative authoring front door: one ergonomic contract describing the
    video an author wants. Standalone — deliberately NOT registered in any
    per-type create-door model map (``projects._BRIEF_DOOR_MODELS``); it is its
    own module and does not touch the ``ProjectCreate`` path.

    ``populate_by_name=True`` lets ``target_url`` be supplied as either ``url`` (the
    author-friendly alias) or ``target_url``. ``extra='allow'`` round-trips any
    forward key untouched.
    """

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True, extra="allow")
    type: ProjectType
    title: str = Field(min_length=1, max_length=200)
    # Accept the friendly ``url`` OR the canonical ``target_url`` (populate_by_name).
    target_url: str = Field(min_length=1, alias="url")
    preset: str = "default"
    goal: str | None = None
    mood: list[str] = []
    # THE FOURTH ORDER DOOR. `projects.py` resolves the order as
    # `body.target_duration_seconds or intent.duration_seconds`, so this field sets the
    # order whenever the outer key is omitted — and the floor is computed from it.
    # Under `extra="allow"` a camelCase `durationSeconds` was not even dropped: it rode
    # into `brief["intent"]` as an extra key while the order fell back to the 120s
    # default, so a 9s order returned 201 and STARTED A RUN. Same alias treatment as
    # `projects._TARGET_DURATION_ALIASES`; `populate_by_name` already applies.
    duration_seconds: int | None = Field(
        default=None,
        ge=1,
        le=3600,
        validation_alias=AliasChoices("duration_seconds", "durationSeconds"),
    )
    must_say: list[str] = []
    must_show: list[str] = []
    # 'auto' → the pipeline casts; a dict pins speaker-role → voice/character.
    cast: Literal["auto"] | dict[str, str] = "auto"
    # 'from_url' brand derivation is DEFERRED this slice — only an explicit dict of
    # brand overrides is honored; the literal keeps the contract forward-compatible.
    brand: Literal["from_url"] | dict[str, object] = {}
    sources: list[str] = []
    # uploads://<ref> (from POST /api/uploads/package) OR an http(s) url.
    package: str | None = None
    # 'auto' is INERT this slice (no recording is triggered); 'off' is the default.
    record: Literal["auto", "off"] | str = "off"
    packaging: PackagingIntent | None = None
    aspect_styles: dict[str, AspectStyle | str] = {}
    overrides: dict[str, object] = {}


class KnobProvenance(BaseModel):
    """Serialized form of the pipeline's internal ``provenance.ProvenanceValue`` — how
    one resolved knob got its value, for the audit trail carried on
    ``PlanDocument.provenance``.

    ``source`` is one of ``provenance.VALID_SOURCES`` (e.g. authored / default /
    computed / clamped); ``computed_by`` names the stage/solver that derived it;
    ``clamp_of`` records the pre-clamp value when a bound was applied.
    """

    value: object
    source: str
    computed_by: str | None = None
    clamp_of: object = None


class PlanDocument(BaseModel):
    """The resolved plan the pipeline emits back for a ``VideoIntent`` — every knob
    decided, with provenance. Serializes straight into ``config_snapshot`` (audit-
    only fields) plus a presence-opt-in patch onto ``config_snapshot['brief']`` (see
    ``to_brief_patch``). ``extra='allow'`` round-trips forward keys untouched.
    """

    model_config = ConfigDict(extra="allow")
    lines: list[AuthoredLine] = []
    package_url: str | None = None
    dialogue: dict[str, object] = {}
    voice: dict[str, object] = {}
    sound_effects: dict[str, object] = {}
    title_card: dict[str, object] = {}
    ken_burns: dict[str, object] = {}
    theme: dict[str, object] = {}
    resolution: str | None = None
    preset: str = "default"
    # Audit-only: per-knob provenance + a hash of the resolving input. Both ride in
    # config_snapshot['plan_document'], NOT in the brief patch (see to_brief_patch).
    provenance: dict[str, KnobProvenance] = {}
    input_hash: str | None = None

    def to_brief_patch(self) -> dict[str, object]:
        """Presence-opt-in dict merged into ``config_snapshot['brief']``.

        Emits ONLY the keys with a non-empty / non-None value — an empty list, empty
        dict, ``None`` or empty string is skipped — so a minimal plan is a no-op
        merge and never overwrites an already-authored brief field. EXCLUDES
        ``provenance`` + ``input_hash`` (audit-only; they ride in
        ``config_snapshot['plan_document']``) and ``preset`` (not a brief key).
        """
        candidates: dict[str, object] = {
            "lines": [line.model_dump() for line in self.lines],
            "package_url": self.package_url,
            "dialogue": self.dialogue,
            "voice": self.voice,
            "sound_effects": self.sound_effects,
            "title_card": self.title_card,
            "ken_burns": self.ken_burns,
            "theme": self.theme,
            "resolution": self.resolution,
        }
        return {key: value for key, value in candidates.items() if value}
