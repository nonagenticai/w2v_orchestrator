"""Request / response models for the service lab (``/api/services/*``).

The service lab is a thin HTTP edge over ``src/services/service_lab/invoker.py``:
one endpoint per provider slot, plus a history read and a media stream for what
was generated. The models here are the ONLY wire contract — the invoker's
``InvokeRequest`` / ``InvokeResult`` never cross it (they carry raw bytes and a
provider dataclass), so a change there is not automatically a wire change.

Bounds are the abuse limits, not the providers' own: a 2000-character image
prompt and a 2048 px (64-multiple) canvas are what the always-on entry points
accept. ``seed`` is deliberately unbounded — providers truncate it themselves.

``width``/``height`` are validated to a multiple of 64 because every image
backend on the estate quantises to a 64-px latent grid; letting a 300 px request
through would silently produce something that is not the size the caller asked
for, and the caller would have no way to tell.

``ServiceRunView.request`` is the SANITISED echo persisted at run time (no
uploaded bytes, no provider secret) — see ``routes/service_lab.py``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from ._shared import _camel

#: The slots ``GET /api/services/{slot}/history`` accepts. Mirrors the invoker's
#: ``SLOTS``; declared here as a Literal so the path parameter is documented in
#: OpenAPI and FastAPI answers 422 for anything else.
ServiceLabSlot = Literal["image", "tts", "asr", "llm", "vlm", "ocr", "video"]

#: Prompt-length limits per generation endpoint (characters).
MAX_IMAGE_PROMPT: Final = 2000
MAX_TEXT_PROMPT: Final = 8000
MAX_TTS_TEXT: Final = 5000

#: Image canvas bounds — every provider on the estate works on a 64-px grid.
MIN_IMAGE_EDGE: Final = 256
MAX_IMAGE_EDGE: Final = 2048
IMAGE_EDGE_MULTIPLE: Final = 64


class ImageGenerateRequest(BaseModel):
    """``POST /api/services/image``."""

    model_config = _camel

    prompt: str = Field(min_length=1, max_length=MAX_IMAGE_PROMPT)
    width: int | None = Field(default=None, ge=MIN_IMAGE_EDGE, le=MAX_IMAGE_EDGE)
    height: int | None = Field(default=None, ge=MIN_IMAGE_EDGE, le=MAX_IMAGE_EDGE)
    seed: int | None = None

    @field_validator("width", "height")
    @classmethod
    def _multiple_of_64(cls, value: int | None) -> int | None:
        if value is not None and value % IMAGE_EDGE_MULTIPLE:
            raise ValueError(f"must be a multiple of {IMAGE_EDGE_MULTIPLE}")
        return value


class TtsGenerateRequest(BaseModel):
    """``POST /api/services/tts``."""

    model_config = _camel

    text: str = Field(min_length=1, max_length=MAX_TTS_TEXT)
    voice: str | None = None
    speed: float | None = Field(default=None, ge=0.25, le=4.0)


class LlmGenerateRequest(BaseModel):
    """``POST /api/services/llm``."""

    model_config = _camel

    prompt: str = Field(min_length=1, max_length=MAX_TEXT_PROMPT)
    system: str | None = None


class VideoGenerateRequest(BaseModel):
    """``POST /api/services/video``."""

    model_config = _camel

    prompt: str = Field(min_length=1, max_length=MAX_IMAGE_PROMPT)
    seconds: int | None = Field(default=None, ge=1, le=30)


class EnhancePromptRequest(BaseModel):
    """``POST /api/services/enhance``."""

    model_config = _camel

    prompt: str = Field(min_length=1, max_length=MAX_TEXT_PROMPT)
    target: Literal["image", "video", "llm", "vlm"]


class EnhancePromptResponse(BaseModel):
    """The rewritten prompt. Never the original — the caller already has it."""

    model_config = _camel

    prompt: str


class ServiceProviderView(BaseModel):
    """Where a run's output came from: ``deployment`` or ``tenant`` provider."""

    model_config = _camel

    source: str | None = None
    kind: str | None = None
    model: str | None = None


class ServiceErrorView(BaseModel):
    """The refusal a failed run recorded, in the same code the HTTP edge returns."""

    model_config = _camel

    code: str
    message: str


class ServiceRunView(BaseModel):
    """One persisted service-lab run.

    ``media_url`` is a path on THIS service (``/api/services/{slot}/history/{id}/media``)
    rather than a MinIO URL: the object is in the orchestrator's own bucket and is
    only reachable with its credentials, so the browser gets a proxied path it can
    put straight into ``<img>`` / ``<audio>`` / ``<video>``.
    """

    model_config = _camel

    id: UUID
    slot: str
    status: str
    created_at: datetime
    duration_ms: int | None = None
    provider: ServiceProviderView
    request: dict[str, object]
    output_kind: str | None = None
    text: str | None = None
    media_url: str | None = None
    media_content_type: str | None = None
    error: ServiceErrorView | None = None


class ServiceRunHistory(BaseModel):
    """``GET /api/services/{slot}/history`` — newest first."""

    model_config = _camel

    items: list[ServiceRunView]
