"""Provider slot registry — the closed vocabulary of slots and the kinds each accepts.

A *slot* is one external capability a run consumes (``llm``, ``tts``, …). Each slot
has exactly one *deployment* kind — the platform-provided implementation a tenant
gets when it has configured nothing — plus zero or more *tenant* kinds a tenant
may bring itself. Each tenant kind carries a :class:`KindSpec`: which fields a
connection of that kind needs (base URL, model, secret, named options).

Writable slots (P3 + P5): ``llm``, ``vlm``, ``tts``, ``asr``, ``image``,
``video_gen``, ``ocr``, ``device_farm``. ``publish.youtube`` is a **grant** slot: the tenant's YouTube OAuth
grant IS the provider, it lives in youtube_upload (not in the connection store),
it cannot be written through ``/api/tenant/connections`` and it is NOT pinned per
run (publishing happens after the run). Every other slot is listed so the
Providers page shows the full surface, but it is "platform provided": no tenant
kinds, so a tenant connection on it is invalid by construction.

**Model-provider slots** (``SlotSpec.byo_governed``, owner decision 2026-10-22):
every GPU/model service — ``llm``, ``vlm``, ``tts``, ``asr``, ``image``,
``video_gen``, ``ocr`` — is a plain web provider (URI + token + API shape), and
whether a tenant may use the PLATFORM's is its licences ``byo_mode``
(``providers/access.py``). That table is derived from this registry, never
from a per-slot ``if``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Literal

#: Deployment kind of the ``llm`` slot: today's worker-management path, unchanged.
KIND_NONAGENTIC_WM: Final = "nonagentic-wm"
#: Deployment kind of ``tts`` / ``image`` / ``video_gen``: today's platform call
#: (modality-gateway / ACG's configured providers), byte-for-byte.
KIND_NONAGENTIC: Final = "nonagentic"
#: Deployment kind of ``device_farm``: app_inspector's local automation, no farm.
KIND_LOCAL: Final = "local"
#: Deployment kind of every slot that is not tenant-writable.
KIND_PLATFORM: Final = "platform"

KIND_OPENAI_COMPATIBLE: Final = "openai-compatible"
KIND_ANTHROPIC: Final = "anthropic"
KIND_OPENAI: Final = "openai"
KIND_ELEVENLABS: Final = "elevenlabs"
KIND_RUNWAY: Final = "runway"
KIND_KLING: Final = "kling"
KIND_VEO: Final = "veo"
KIND_SORA: Final = "sora"
KIND_SEEDANCE: Final = "seedance"
KIND_BROWSERSTACK: Final = "browserstack"
#: A GPU service that speaks the PLATFORM protocol (modality-gateway's routes:
#: ``/api/tts/synthesize``, ``/api/asr/transcribe``, ``/api/image/generate``) at the
#: tenant's own ``base_url`` — e.g. ``https://api.nonagentic.ai/tts`` or a self-hosted
#: copy of the platform stack. The secret, when set, is sent as ``Bearer <secret>``.
KIND_NONAGENTIC_ENDPOINT: Final = "nonagentic_endpoint"

SlotType = Literal["connection", "grant"]
#: ``required``: no default endpoint exists. ``optional``: the provider's public API
#: is the default. ``forbidden``: the endpoint is fixed (the tenant's credential is
#: only ever sent to the provider's own host).
BaseUrlRule = Literal["required", "optional", "forbidden"]


#: Where a form field writes: ``base_url`` / ``model`` are top-level draft fields,
#: ``options.<k>`` is ``options[k]``.
FieldFormat = Literal["text", "url"]


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One input of a kind's connection form (P8: moved server-side from the FE's
    ``slot-fields.ts``). ``required`` is NOT stored here — it is derived from the
    :class:`KindSpec` rules the server enforces (:meth:`KindSpec.field_required`),
    so the form can never ask for less (or more) than a PUT accepts."""

    key: str
    label: str
    help: str = ""
    format: FieldFormat = "text"


@dataclass(frozen=True, slots=True)
class KindSpec:
    """What a tenant connection of one kind must carry, plus its form metadata."""

    base_url: BaseUrlRule = "optional"
    model_required: bool = False
    secret_required: bool = True
    #: ``options`` keys that must be present as non-empty strings.
    required_options: tuple[str, ...] = ()
    #: Display name of the kind (Providers page / requirements manifest).
    label: str = ""
    #: One-line explanation shown under the kind picker.
    help: str = ""
    #: The endpoint used when ``base_url`` is left empty (``None``: no default —
    #: either the URL is required, or the provider adapter downstream owns it).
    default_base_url: str | None = None
    #: Label of the write-only secret field ("API key", "Access key").
    secret_label: str = "API key"  # noqa: S105  # pragma: allowlist secret — a form label
    #: The form's inputs, in display order.
    fields: tuple[FieldSpec, ...] = ()

    def field_required(self, key: str) -> bool:
        """Whether a PUT is refused without ``key`` — the server's rule, not a hint."""
        if key == "base_url":
            return self.base_url == "required"
        if key == "model":
            return self.model_required
        if key.startswith("options."):
            return key.removeprefix("options.") in self.required_options
        return False


@dataclass(frozen=True, slots=True)
class SlotSpec:
    """One slot: its deployment kinds and the kinds a tenant may configure on it."""

    name: str
    deployment_kinds: frozenset[str]
    tenant_kind_specs: Mapping[str, KindSpec] = field(default_factory=dict)
    slot_type: SlotType = "connection"
    #: Display name of the slot (Providers page / requirements manifest).
    label: str = ""
    #: A model-provider slot governed by the tenant's licences ``byo_mode``
    #: (``providers/access.py``): ``byo_required`` ⇒ the tenant must bring it,
    #: ``platform_only`` ⇒ the tenant's own connection is ignored and unwritable.
    byo_governed: bool = False
    #: Short name used in user-facing sentences ("Configure your own TTS provider").
    short_label: str = ""

    @property
    def tenant_kinds(self) -> frozenset[str]:
        return frozenset(self.tenant_kind_specs)

    @property
    def writable(self) -> bool:
        """True when a tenant may store a connection on this slot."""
        return self.slot_type == "connection" and bool(self.tenant_kind_specs)

    @property
    def pinned(self) -> bool:
        """True when a run pins this slot at start (grant slots are not pinned)."""
        return self.slot_type == "connection"

    @property
    def kinds(self) -> frozenset[str]:
        return self.deployment_kinds | self.tenant_kinds

    @property
    def base_url_required_kinds(self) -> frozenset[str]:
        """Tenant kinds that cannot work without a ``base_url`` (nothing to default to)."""
        return frozenset(k for k, s in self.tenant_kind_specs.items() if s.base_url == "required")

    def kind_spec(self, kind: str) -> KindSpec | None:
        return self.tenant_kind_specs.get(kind)


def _platform(name: str, label: str) -> SlotSpec:
    return SlotSpec(name=name, deployment_kinds=frozenset({KIND_PLATFORM}), label=label)


_VOICE: Final = ("voice",)

#: Public default endpoints (the tester / in-process adapters use the same values;
#: ``tests/unit/test_provider_requirements.py`` pins the equality).
OPENAI_DEFAULT_BASE_URL: Final = "https://api.openai.com/v1"
ANTHROPIC_DEFAULT_BASE_URL: Final = "https://api.anthropic.com"
ELEVENLABS_DEFAULT_BASE_URL: Final = "https://api.elevenlabs.io"
VEO_DEFAULT_BASE_URL: Final = "https://generativelanguage.googleapis.com/v1beta"
#: Model an ``asr`` ``openai-compatible`` connection uses when none is set.
ASR_DEFAULT_MODEL: Final = "whisper-1"

#: Display names of every kind, deployment kinds included (the manifest labels
#: ``default_kind`` with these). Copied from the FE's ``slot-fields.ts`` KIND_LABELS.
KIND_LABELS: Final[dict[str, str]] = {
    KIND_OPENAI_COMPATIBLE: "OpenAI-compatible",
    KIND_ANTHROPIC: "Anthropic",
    KIND_NONAGENTIC_WM: "Platform (Nonagentic)",
    KIND_NONAGENTIC: "Platform (Nonagentic)",
    KIND_LOCAL: "Platform (local automation)",
    KIND_PLATFORM: "Platform (Nonagentic)",
    KIND_OPENAI: "OpenAI",
    KIND_ELEVENLABS: "ElevenLabs",
    KIND_RUNWAY: "Runway",
    KIND_KLING: "Kling",
    KIND_VEO: "Google Veo",
    KIND_SORA: "OpenAI Sora",
    KIND_SEEDANCE: "Seedance",
    KIND_BROWSERSTACK: "BrowserStack",
    KIND_NONAGENTIC_ENDPOINT: "Nonagentic-protocol GPU endpoint",
}


def kind_label(kind: str) -> str:
    return KIND_LABELS.get(kind, kind)


def _model(help_text: str) -> FieldSpec:
    return FieldSpec(key="model", label="Model", help=help_text)


def _base_url(help_text: str) -> FieldSpec:
    return FieldSpec(key="base_url", label="Base URL", help=help_text, format="url")


def _nonagentic_endpoint(example_url: str, route: str) -> KindSpec:
    """A platform-protocol GPU service at the tenant's own URL (``route`` under it)."""
    return KindSpec(
        base_url="required",
        secret_required=False,
        label=KIND_LABELS[KIND_NONAGENTIC_ENDPOINT],
        help=(
            f"A GPU service speaking the Nonagentic platform protocol ({route} under the "
            "base URL). The token, if set, is sent as 'Authorization: Bearer <token>'."
        ),
        secret_label="Bearer token",  # noqa: S106  # pragma: allowlist secret — a form label
        fields=(_base_url(example_url),),
    )


#: The provider_contracts OCR route (Mistral OCR shape; mlx_api serves it).
OCR_ROUTE: Final = "/v1/ocr"


def _video_kind(kind: str, default_base_url: str | None = None) -> KindSpec:
    return KindSpec(
        label=KIND_LABELS[kind],
        default_base_url=default_base_url,
        fields=(_model("provider default"), _base_url("the provider's public API")),
    )


SLOTS: Final[dict[str, SlotSpec]] = {
    spec.name: spec
    for spec in (
        SlotSpec(
            name="llm",
            byo_governed=True,
            short_label="LLM",
            label="Language model (LLM)",
            deployment_kinds=frozenset({KIND_NONAGENTIC_WM}),
            tenant_kind_specs={
                KIND_OPENAI_COMPATIBLE: KindSpec(
                    base_url="required",
                    secret_required=False,
                    label=KIND_LABELS[KIND_OPENAI_COMPATIBLE],
                    fields=(_model("e.g. gpt-4.1-mini"), _base_url("https://api.example.com/v1")),
                ),
                # P3 semantics kept: the live test reports a missing model / key.
                KIND_ANTHROPIC: KindSpec(
                    secret_required=False,
                    label=KIND_LABELS[KIND_ANTHROPIC],
                    default_base_url=ANTHROPIC_DEFAULT_BASE_URL,
                    fields=(
                        _model(""),
                        _base_url(ANTHROPIC_DEFAULT_BASE_URL),
                    ),
                ),
            },
        ),
        SlotSpec(
            name="vlm",
            byo_governed=True,
            short_label="VLM",
            label="Vision model (VLM)",
            deployment_kinds=frozenset({KIND_PLATFORM}),
            tenant_kind_specs={
                # Chat Completions with ``image_url`` content parts (vLLM / OpenAI).
                KIND_OPENAI_COMPATIBLE: KindSpec(
                    base_url="required",
                    model_required=True,
                    secret_required=False,
                    label=KIND_LABELS[KIND_OPENAI_COMPATIBLE],
                    help="OpenAI Chat Completions with image_url content parts.",
                    fields=(
                        _base_url("https://api.example.com/v1"),
                        _model("a vision-capable served model"),
                    ),
                ),
            },
        ),
        SlotSpec(
            name="tts",
            byo_governed=True,
            short_label="TTS",
            label="Text to speech",
            deployment_kinds=frozenset({KIND_NONAGENTIC}),
            tenant_kind_specs={
                KIND_OPENAI: KindSpec(
                    required_options=_VOICE,
                    label=KIND_LABELS[KIND_OPENAI],
                    default_base_url=OPENAI_DEFAULT_BASE_URL,
                    fields=(
                        _model("e.g. gpt-4o-mini-tts"),
                        FieldSpec(key="options.voice", label="Voice", help="e.g. alloy"),
                        _base_url(OPENAI_DEFAULT_BASE_URL),
                    ),
                ),
                KIND_ELEVENLABS: KindSpec(
                    required_options=_VOICE,
                    label=KIND_LABELS[KIND_ELEVENLABS],
                    default_base_url=ELEVENLABS_DEFAULT_BASE_URL,
                    fields=(
                        FieldSpec(
                            key="options.voice", label="Voice ID", help="the ElevenLabs voice_id"
                        ),
                        _model("e.g. eleven_multilingual_v2"),
                        _base_url(ELEVENLABS_DEFAULT_BASE_URL),
                    ),
                ),
                # Self-hosted (kokoro / openedai-speech …): endpoint + model are the
                # tenant's; a local server may run without a key.
                KIND_OPENAI_COMPATIBLE: KindSpec(
                    base_url="required",
                    model_required=True,
                    secret_required=False,
                    required_options=_VOICE,
                    label=KIND_LABELS[KIND_OPENAI_COMPATIBLE],
                    fields=(
                        _base_url("https://tts.example.com/v1"),
                        _model("the served model name"),
                        FieldSpec(key="options.voice", label="Voice"),
                    ),
                ),
                KIND_NONAGENTIC_ENDPOINT: _nonagentic_endpoint(
                    "https://api.nonagentic.ai/tts", "POST /api/tts/synthesize"
                ),
            },
        ),
        SlotSpec(
            name="asr",
            byo_governed=True,
            short_label="ASR",
            label="Speech to text",
            deployment_kinds=frozenset({KIND_PLATFORM}),
            tenant_kind_specs={
                # OpenAI ``POST /audio/transcriptions`` (Whisper API, faster-whisper-server …).
                KIND_OPENAI_COMPATIBLE: KindSpec(
                    base_url="required",
                    secret_required=False,
                    label=KIND_LABELS[KIND_OPENAI_COMPATIBLE],
                    help="OpenAI /audio/transcriptions (verbose_json with word timestamps).",
                    fields=(
                        _base_url("https://asr.example.com/v1"),
                        _model(f"defaults to {ASR_DEFAULT_MODEL}"),
                    ),
                ),
                KIND_NONAGENTIC_ENDPOINT: _nonagentic_endpoint(
                    "https://api.nonagentic.ai/asr", "POST /api/asr/transcribe"
                ),
            },
        ),
        SlotSpec(
            name="image",
            byo_governed=True,
            short_label="image",
            label="Image generation",
            deployment_kinds=frozenset({KIND_NONAGENTIC}),
            tenant_kind_specs={
                KIND_OPENAI: KindSpec(
                    label=KIND_LABELS[KIND_OPENAI],
                    default_base_url=OPENAI_DEFAULT_BASE_URL,
                    fields=(
                        _model("defaults to gpt-image-1"),
                        _base_url(OPENAI_DEFAULT_BASE_URL),
                    ),
                ),
                # Self-hosted OpenAI Images API (``POST {base_url}/images/generations``).
                # ai_content_generation names this kind ``openai_compatible`` on the
                # wire — see ``slot_payload._WIRE_KINDS``.
                KIND_OPENAI_COMPATIBLE: KindSpec(
                    base_url="required",
                    secret_required=False,
                    label=KIND_LABELS[KIND_OPENAI_COMPATIBLE],
                    help="OpenAI Images API (POST /images/generations) at your own base URL.",
                    fields=(
                        _base_url("https://image.example.com/v1"),
                        _model("the served model name (optional)"),
                    ),
                ),
                KIND_NONAGENTIC_ENDPOINT: _nonagentic_endpoint(
                    "https://api.nonagentic.ai/image", "POST /api/image/generate"
                ),
            },
        ),
        SlotSpec(
            name="video_gen",
            byo_governed=True,
            short_label="video generation",
            label="Video generation",
            deployment_kinds=frozenset({KIND_NONAGENTIC}),
            tenant_kind_specs={
                KIND_RUNWAY: _video_kind(KIND_RUNWAY),
                KIND_KLING: _video_kind(KIND_KLING),
                KIND_VEO: _video_kind(KIND_VEO, VEO_DEFAULT_BASE_URL),
                KIND_SORA: _video_kind(KIND_SORA),
                KIND_SEEDANCE: _video_kind(KIND_SEEDANCE),
            },
        ),
        SlotSpec(
            name="ocr",
            label="Document OCR",
            byo_governed=True,
            short_label="OCR",
            # The platform kind is the operator's configured OCR provider
            # (``W2V_PROVIDER_OCR_URL``, mlx_api PaddleOCR-VL today).
            deployment_kinds=frozenset({KIND_PLATFORM}),
            tenant_kind_specs={
                # A copy of the platform OCR service (mlx_api) at the tenant's URL.
                KIND_NONAGENTIC_ENDPOINT: KindSpec(
                    base_url="required",
                    secret_required=False,
                    label=KIND_LABELS[KIND_NONAGENTIC_ENDPOINT],
                    help=(
                        f"An OCR service speaking the Nonagentic platform protocol (POST "
                        f"{OCR_ROUTE} under the base URL, Mistral OCR request/response shape: "
                        "document_url data: URL in, pages[].markdown out). The token, if set, "
                        "is sent as 'Authorization: Bearer <token>'."
                    ),
                    secret_label="Bearer token",  # noqa: S106  # pragma: allowlist secret — a form label
                    fields=(
                        _base_url("https://ocr.example.com"),
                        _model("the served OCR model (optional)"),
                    ),
                ),
                # A Mistral-OCR-shaped API whose base URL already ends in /v1
                # (``POST {base_url}/ocr`` — e.g. https://api.mistral.ai/v1).
                KIND_OPENAI_COMPATIBLE: KindSpec(
                    base_url="required",
                    model_required=True,
                    secret_required=False,
                    label=KIND_LABELS[KIND_OPENAI_COMPATIBLE],
                    help=(
                        "Mistral OCR API shape at your own base URL: POST {base_url}/ocr "
                        "with a document_url, returning pages[].markdown "
                        "(e.g. https://api.mistral.ai/v1, model mistral-ocr-latest)."
                    ),
                    fields=(
                        _base_url("https://api.example.com/v1"),
                        _model("e.g. mistral-ocr-latest"),
                    ),
                ),
            },
        ),
        SlotSpec(
            name="publish.youtube",
            label="YouTube publishing",
            deployment_kinds=frozenset({KIND_PLATFORM}),
            slot_type="grant",
        ),
        SlotSpec(
            name="device_farm",
            label="Device farm",
            deployment_kinds=frozenset({KIND_LOCAL}),
            tenant_kind_specs={
                KIND_BROWSERSTACK: KindSpec(
                    base_url="forbidden",
                    required_options=("username",),
                    label=KIND_LABELS[KIND_BROWSERSTACK],
                    secret_label="Access key",  # noqa: S106  # pragma: allowlist secret — a form label
                    fields=(
                        FieldSpec(
                            key="options.username",
                            label="Username",
                            help="your BrowserStack username",
                        ),
                    ),
                )
            },
        ),
        _platform("scrape", "Web scraping"),
        _platform("media_search", "Media search"),
        _platform("doc_parse", "Document parsing"),
    )
}

#: Slot names in registry order (stable: the Providers page iterates it).
SLOT_NAMES: Final[tuple[str, ...]] = tuple(SLOTS)
#: Slots a run pins at start — every slot except grant slots (``publish.youtube``).
PINNED_SLOT_NAMES: Final[tuple[str, ...]] = tuple(n for n, s in SLOTS.items() if s.pinned)


def get_slot(slot: str) -> SlotSpec:
    """Return the spec for ``slot``; an unknown name is a programming error (``KeyError``)."""
    try:
        return SLOTS[slot]
    except KeyError:
        raise KeyError(f"unknown provider slot {slot!r}; known: {', '.join(SLOT_NAMES)}") from None


def writable_slots() -> tuple[str, ...]:
    """Slots a tenant may configure through the connection store."""
    return tuple(name for name, spec in SLOTS.items() if spec.writable)


#: Model-provider slots governed by the tenant's licences ``byo_mode`` (registry order).
BYO_GOVERNED_SLOTS: Final[tuple[str, ...]] = tuple(n for n, s in SLOTS.items() if s.byo_governed)


def slot_short_label(slot: str) -> str:
    """The slot's name inside a user-facing sentence ("TTS", "image", …)."""
    spec = SLOTS.get(slot)
    return (spec.short_label or spec.name) if spec is not None else slot
