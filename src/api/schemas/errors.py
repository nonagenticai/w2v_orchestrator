"""Error response schema."""

from __future__ import annotations

from pydantic import BaseModel

from ._shared import _camel


class ErrorResponse(BaseModel):
    model_config = _camel
    error: str
    detail: str | None = None
