"""P8 schemas: the static provider requirements manifest (``GET /api/provider-requirements``).

What a tenant (or a partner embedding w2v, e.g. merqi) must configure for w2v to run:
one entry per configurable provider slot with the kinds it accepts, each kind's form
fields, and which blueprint stages call it. No tenant data — the same for every
caller on a deployment.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ._shared import _camel


class KindField(BaseModel):
    """One form input. ``key``: ``base_url`` | ``model`` | ``options.<name>``.
    ``required`` is the server's rule (a PUT without it is 422), not a hint."""

    model_config = _camel
    key: str
    label: str
    required: bool
    help: str = ""
    format: Literal["text", "url"] = "text"


class KindRequirement(BaseModel):
    model_config = _camel
    label: str
    help: str = ""
    base_url: Literal["required", "optional", "forbidden"]
    default_base_url: str | None = None
    model_required: bool
    secret_required: bool
    secret_label: str
    required_options: list[str]
    fields: list[KindField]


class SlotUsage(BaseModel):
    """A blueprint stage that may call the slot. ``when``: ``always`` (whenever the
    stage runs) or the brief opt-in without which the stage skips."""

    model_config = _camel
    blueprint: str
    project_type: str
    stage: str
    when: str


class SlotRequirement(BaseModel):
    """``type``: ``connection`` = configured via ``PUT /api/tenant/connections``;
    ``grant`` = connected through its own flow (``publish.youtube`` → YouTube OAuth),
    never part of the bulk PUT."""

    model_config = _camel
    slot: str
    label: str
    type: Literal["connection", "grant"]
    default_kind: str
    default_kind_label: str
    allowed_kinds: list[str]
    byo_required: bool
    kinds: dict[str, KindRequirement] = Field(default_factory=dict)
    used_by: list[SlotUsage] = Field(default_factory=list)


class PlatformDependency(BaseModel):
    """A service w2v depends on that a tenant can NOT configure (owner 2026-10-09:
    platform services stay fixed deployment dependencies;
    KAG retired 2026-10, replaced by webscraper + media-discovery)."""

    model_config = _camel
    name: str
    label: str
    type: Literal["platform"] = "platform"
    configurable: bool = False
    description: str = ""


class ProviderRequirements(BaseModel):
    model_config = _camel
    app: str = "w2v"
    manifest_version: int = 1
    slots: list[SlotRequirement]
    dependencies: list[PlatformDependency]
