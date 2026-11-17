"""Verbatim items + growth bundle schemas."""

from __future__ import annotations

import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from ._shared import VerbatimKind, VerbatimScope, VerbatimStatus, _camel


class VerbatimItemCreate(BaseModel):
    model_config = _camel
    kind: VerbatimKind = "tts"
    text: Annotated[str, Field(min_length=1, max_length=2000)]
    hint: Annotated[str | None, Field(max_length=2000)] = None
    scope: VerbatimScope = "pipeline"
    scope_target: Annotated[str | None, Field(max_length=64)] = None
    position: int | None = None


class VerbatimItemUpdate(BaseModel):
    model_config = _camel
    kind: VerbatimKind | None = None
    text: Annotated[str | None, Field(min_length=1, max_length=2000)] = None
    hint: Annotated[str | None, Field(max_length=2000)] = None
    scope: VerbatimScope | None = None
    scope_target: Annotated[str | None, Field(max_length=64)] = None
    position: int | None = None


class VerbatimItemResponse(BaseModel):
    """SPA-facing verbatim row.

    None of these lifecycle/identity fields is stored on ``verbatim_items``
    (0009 created them, the v3 reset 0015 dropped the table CASCADE, 0017
    re-added only kind/hint/scope/scope_target). They resolve as follows:

    * ``tenant_id`` / ``subject_id`` / ``visibility`` — INHERITED from the
      parent Project by the route's ``_resp``.
    * ``status`` / ``placed_in_stage`` / ``placed_in_element_id`` — DERIVED at
      read time by ``src.services.verbatim_placement.derive_placements``, which
      re-matches the phrase against a run's currently-persisted carrier stage
      outputs. Derivation only happens on ``verbatim.list`` when the caller
      passes ``?run_id=`` (placement is a property of a RUN, not of the row);
      every other response — and a list with no ``run_id`` — correctly keeps the
      ``pending`` default, because without a run there is nothing to match
      against. All three ``VerbatimStatus`` values are reachable: ``placed``,
      ``pending`` ("an eligible stage ran and the phrase is not in it, or nothing
      has run yet") and ``rejected`` ("every stage that ran was forbidden from
      carrying this item's ``kind``" — today that is every ``slide`` item, since
      every stage in ``STAGE_VERBATIM_KIND`` consumes ``tts``).
    * ``retry_count`` — genuinely unimplemented; always 0.

    v3 dropped per-job verbatim, so job_id is always null. See migration 0017
    for the live column shape.
    """

    model_config = _camel
    id: UUID
    project_id: UUID | None
    job_id: UUID | None = None
    tenant_id: UUID | None = None
    subject_id: UUID | None = None
    visibility: str = "tenant"
    kind: VerbatimKind
    text: str
    hint: str | None = None
    scope: VerbatimScope
    scope_target: str | None = None
    status: VerbatimStatus = "pending"
    placed_in_stage: str | None = None
    placed_in_element_id: str | None = None
    retry_count: int = 0
    position: int
    created_at: datetime.datetime
    updated_at: datetime.datetime


class VerbatimItemListResponse(BaseModel):
    """List envelope the SPA's verbatim store expects (``{items, total}``)."""

    model_config = _camel
    items: list[VerbatimItemResponse]
    total: int


class GrowthBundleSnippet(BaseModel):
    model_config = _camel
    id: str
    channel: Literal["linkedin", "twitter", "email", "landing_page"]
    text: str
    cta_url: str | None = None


class GrowthBundleOutput(BaseModel):
    model_config = _camel
    snippets: list[GrowthBundleSnippet]
