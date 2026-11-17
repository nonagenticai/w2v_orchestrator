"""P3 WP4 schemas: tenant provider connections (``/api/tenant/connections``).

Secrets are WRITE-ONLY: a request may carry ``secret``; no response model has a
field that could hold it (or its ciphertext) — only ``has_secret`` and the 12-hex
``key_fingerprint``.
"""

from __future__ import annotations

import datetime
import json
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, JsonValue, field_validator

from ._shared import _camel

#: Serialized ceiling for ``options`` — provider knobs, not a document store.
MAX_OPTIONS_BYTES = 4096


class ConnectionView(BaseModel):
    """The tenant's current connection on a slot. Never the secret."""

    model_config = _camel
    id: UUID
    kind: str
    base_url: str | None = None
    model: str | None = None
    options: dict[str, JsonValue] = Field(default_factory=dict)
    version: int
    has_secret: bool
    key_fingerprint: str | None = None
    last_test_at: datetime.datetime | None = None
    last_test_ok: bool | None = None
    last_test_error: str | None = None


class KindRequirements(BaseModel):
    """Fields a connection of one tenant kind must carry (drives the Providers form)."""

    model_config = _camel
    base_url: Literal["required", "optional", "forbidden"]
    model_required: bool
    secret_required: bool
    required_options: list[str]


class LlmAccessView(BaseModel):
    """The tenant's LLM access posture (licences ``byo_mode`` for app ``w2v``)."""

    model_config = _camel
    mode: Literal["local_allowed", "platform_only", "byo_required"]
    label: str


class SlotView(BaseModel):
    """One registry slot as the Providers page shows it.

    ``type``: ``connection`` = stored here (``writable`` says whether a tenant may);
    ``grant`` = connected through its own flow (``publish.youtube`` → the YouTube
    OAuth grant), never written through this API and never pinned per run."""

    model_config = _camel
    slot: str
    type: Literal["connection", "grant"] = "connection"
    writable: bool
    allowed_kinds: list[str]
    source: Literal["tenant", "deployment"]
    default_kind: str
    byo_required: bool
    #: Per allowed tenant kind; empty for non-writable slots.
    kind_requirements: dict[str, KindRequirements] = Field(default_factory=dict)
    connection: ConnectionView | None = None


class SlotList(BaseModel):
    model_config = _camel
    items: list[SlotView]
    llm_access: LlmAccessView | None = None


class ModelsRequest(BaseModel):
    """``POST /{slot}/models`` body. ``secret`` absent/``null`` = use the stored
    secret (only while kind and base URL are unchanged)."""

    model_config = _camel
    kind: str = Field(min_length=1, max_length=64)
    base_url: str | None = Field(default=None, max_length=2048)
    secret: str | None = Field(default=None, max_length=8192)

    @field_validator("base_url", mode="before")
    @classmethod
    def _blank_is_none(cls, value: str | None) -> str | None:
        return None if isinstance(value, str) and not value.strip() else value


class ModelsResult(BaseModel):
    model_config = _camel
    models: list[str]


class ConnectionDraft(BaseModel):
    """A connection configuration to test or store.

    ``secret``: absent (or ``null``) = keep the stored secret, ``""`` = clear it,
    any other value = set it. Keeping is refused when ``kind`` or the endpoint
    changes (the stored key was entrusted to one endpoint)."""

    model_config = _camel
    kind: str = Field(min_length=1, max_length=64)
    base_url: str | None = Field(default=None, max_length=2048)
    model: str | None = Field(default=None, max_length=256)
    options: dict[str, JsonValue] | None = None
    secret: str | None = Field(default=None, max_length=8192)

    @field_validator("base_url", "model")
    @classmethod
    def _blank_is_none(cls, value: str | None) -> str | None:
        return value or None

    @field_validator("options")
    @classmethod
    def _bounded_options(cls, value: dict[str, JsonValue] | None) -> dict[str, JsonValue] | None:
        if value is not None and len(json.dumps(value, default=str)) > MAX_OPTIONS_BYTES:
            raise ValueError(f"options must serialize to at most {MAX_OPTIONS_BYTES} bytes")
        return value


class ConnectionPut(ConnectionDraft):
    """``PUT`` body: a draft plus the optimistic-concurrency guard.

    ``expected_version`` is the version the client last saw (``0`` = no tenant
    connection yet); a mismatch is 409."""

    expected_version: int = Field(ge=0)


class ConnectionTestResult(BaseModel):
    """``ok`` is ``None`` when the kind has no auth-only endpoint (``probed=False``):
    nothing was sent, so neither a pass nor a failure is claimed."""

    model_config = _camel
    ok: bool | None
    probed: bool = True
    message: str
    upstream_status: int | None = None
    #: The connection version tested; ``None`` for a draft.
    version: int | None = None


# ─── P8 bulk subtree PUT (PUT /api/tenant/connections) ───────────────────


class ConnectionsBulkPut(BaseModel):
    """The whole tenant subtree in one write.

    ``connections``: slot → ``ConnectionPut`` (set) or ``null`` (delete → back to the
    deployment default). Slots not named are left untouched. ``dependencies``: w2v has
    no configurable dependency (webscraper / media-discovery are fixed), so anything non-empty is 422."""

    model_config = _camel
    connections: dict[str, ConnectionPut | None] = Field(default_factory=dict, max_length=32)
    dependencies: dict[str, JsonValue] = Field(default_factory=dict)


class BulkSlotResult(SlotView):
    """A slot of the bulk write after it committed. ``change``: what happened;
    ``test``: the live test (``None`` when nothing was tested — unchanged/deleted)."""

    change: Literal["created", "updated", "deleted", "unchanged"]
    test: ConnectionTestResult | None = None


class ConnectionsBulkResult(BaseModel):
    model_config = _camel
    items: list[BulkSlotResult]
