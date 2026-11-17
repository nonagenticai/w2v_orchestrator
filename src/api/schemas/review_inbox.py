"""Schemas for the run owner's in-app "needs your review" inbox."""

from __future__ import annotations

import datetime
from uuid import UUID

from pydantic import BaseModel

from ._shared import _camel


class ReviewInboxItem(BaseModel):
    """One run the caller owns that is paused at a review checkpoint."""

    model_config = _camel
    run_id: UUID
    project_id: UUID
    project_title: str
    deliverable_name: str | None = None
    #: The checkpoint to review — the frontend links to
    #: ``/projects/{project_id}/runs/{run_id}/stages/{stage_name}``.
    stage_name: str | None = None
    paused_at: datetime.datetime | None = None
    #: The run has waited 24h or more (or its wait start is unknown): the reminder (shown highlighted, and a
    #: dismissed notice comes back).
    reminder: bool = False


class ReviewInbox(BaseModel):
    """The caller's review notices, oldest wait first (the count is ``len(items)``)."""

    model_config = _camel
    items: list[ReviewInboxItem]
