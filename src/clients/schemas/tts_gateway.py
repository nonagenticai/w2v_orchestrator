"""Typed responses for the pooling TTS gateway claim-queue.

Mirrors the gateway's submit + poll wire shapes. ``extra="ignore"`` (pydantic's
default) drops any field the orchestrator's ``audiobook_narrate`` stage does not
read, so a gateway that ADDS a field can never break the orchestrator.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GatewayJobRef(BaseModel):
    """One job reference in a submit response (``ref`` links back to the chunk)."""

    ref: str | None = None
    job_id: str | None = None
    idempotency_key: str
    status: str


class SubmitBatchResult(BaseModel):
    """Response of ``POST /api/tts/batches`` — the enqueued batch + its job refs."""

    batch_id: str
    jobs: list[GatewayJobRef] = Field(default_factory=list)


class GatewayJobStatus(BaseModel):
    """One job's status in a batch poll (URLs presigned at poll time)."""

    job_id: str
    idempotency_key: str
    status: str
    result_url: str | None = None
    #: Measured audio length, or None when the gateway could not measure it.
    #: NEVER collapse this to ``0.0``: tts_gateway#38 stopped fabricating a zero
    #: precisely so "unknown length" and "zero seconds of audio" stay distinct.
    duration_seconds: float | None = None
    #: True when the gateway persisted SQL NULL for ``duration_seconds`` because
    #: the render could not be probed (tts_gateway#38). The audio itself is fine;
    #: only its LENGTH is unknown. Declared here — and not left to ``extra="ignore"``
    #: — because a dropped field is dropped SILENTLY: no error, no log, and the
    #: producer-side fix becomes inert one layer up.
    duration_unmeasured: bool = False
    error: str | None = None


class BatchStatusResult(BaseModel):
    """Response of ``GET /api/tts/batches/{batch_id}`` — per-chunk job statuses."""

    batch_id: str
    jobs: list[GatewayJobStatus] = Field(default_factory=list)
