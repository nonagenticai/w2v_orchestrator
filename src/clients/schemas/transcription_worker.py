"""Typed response for transcription_worker_service ``POST /api/transcribe``.

Mirrors the server ``TranscriptionResponse`` (Whisper word-timestamps).
``extra="ignore"`` (pydantic's default) drops any field the orchestrator's
word-timed-captions path does not read.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptWord(BaseModel):
    """One word with its timing + confidence.

    ``probability`` is server-nullable: Whisper legitimately emits ``null`` for
    some words. Declaring it non-nullable made a null word raise a
    ``ValidationError`` at the client boundary, which the caption path swallowed
    and silently downgraded a whole clip from word-timed to phrase captions.
    Downstream (``caption_reconcile``) folds a missing probability to ``1.0``.
    """

    start: float
    end: float
    word: str
    probability: float | None = None


class TranscriptSegment(BaseModel):
    """One transcription segment with its per-word timings."""

    id: int
    start: float
    end: float
    text: str
    words: list[TranscriptWord] = Field(default_factory=list)
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


class TranscriptionResult(BaseModel):
    """Full result of ``POST /api/transcribe`` — text + timed segments (+ optional SRT)."""

    backend: str
    model: str
    language: str | None = None
    duration: float
    text: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    srt: str | None = None
