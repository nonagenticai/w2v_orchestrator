"""Annotation schemas."""

from __future__ import annotations

import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field

from ._shared import AnnotationStatus, _camel


class AnnotationCreate(BaseModel):
    model_config = _camel
    text: Annotated[str, Field(min_length=1, max_length=2000)]
    element_id: str | None = None


class AnnotationUpdate(BaseModel):
    model_config = _camel
    text: Annotated[str | None, Field(min_length=1, max_length=2000)] = None
    status: AnnotationStatus | None = None


class AnnotationResponse(BaseModel):
    model_config = _camel
    id: UUID
    stage_output_id: UUID
    element_id: str | None
    text: str
    status: AnnotationStatus
    applied_in_run_id: UUID | None
    created_at: datetime.datetime
    updated_at: datetime.datetime


class StageOutputAnnotations(BaseModel):
    model_config = _camel
    stage_output_id: UUID
    items: list[AnnotationResponse]
