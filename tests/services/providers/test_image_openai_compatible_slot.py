"""Image slot kind ``openai-compatible`` + its ACG wire name ``openai_compatible``.

ai_content_generation (#141) validates image ``provider_slot.kind`` against
``nonagentic_endpoint`` / ``openai_compatible`` (UNDERSCORE). The orchestrator's
registry spells the shared kind ``openai-compatible`` (hyphen), so the payload
builder maps it for the image slot — and ONLY there.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import SecretStr

from src.services.providers.registry import KIND_OPENAI_COMPATIBLE, get_slot
from src.services.providers.resolver import ResolvedSlot, SlotRef
from src.services.providers.slot_payload import slot_payload, wire_kind

CONN = UUID("01990000-0000-7000-8000-000000000004")


def _resolved(slot: str, kind: str, *, secret: str | None) -> ResolvedSlot:
    return ResolvedSlot(
        ref=SlotRef(slot=slot, source="tenant", kind=kind, connection_id=CONN, version=1),
        base_url="https://image.example.com/v1",
        model=None,
        options={},
        secret=SecretStr(secret) if secret else None,
        pinned_ip=None,
    )


def test_image_slot_accepts_openai_compatible_with_base_url_required_and_key_optional() -> None:
    spec = get_slot("image").kind_spec(KIND_OPENAI_COMPATIBLE)
    assert spec is not None, "image must offer the openai-compatible tenant kind"
    assert spec.base_url == "required"
    assert spec.secret_required is False
    assert spec.model_required is False
    assert KIND_OPENAI_COMPATIBLE in get_slot("image").base_url_required_kinds


def test_image_payload_sends_the_underscore_wire_kind_acg_expects() -> None:
    got = slot_payload(_resolved("image", KIND_OPENAI_COMPATIBLE, secret=None))
    assert got["kind"] == "openai_compatible"
    assert got["base_url"] == "https://image.example.com/v1"
    assert got["model"] is None
    assert got["secret"] is None


def test_the_wire_mapping_is_image_only() -> None:
    assert wire_kind("image", "openai-compatible") == "openai_compatible"
    assert wire_kind("image", "nonagentic_endpoint") == "nonagentic_endpoint"
    assert wire_kind("image", "openai") == "openai"
    # Other slots keep the registry spelling verbatim.
    assert wire_kind("vlm", "openai-compatible") == "openai-compatible"
    assert wire_kind("tts", "openai-compatible") == "openai-compatible"
