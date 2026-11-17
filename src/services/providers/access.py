"""ONE provider-access policy for every model slot, keyed by licences ``byo_mode``.

Owner decision 2026-10-22: every GPU/model service (``llm``, ``vlm``, ``tts``, ``asr``,
``image``, ``video_gen``, ``ocr``) is a plain web provider (URI + token + API shape).
Whether a tenant may use the PLATFORM's is decided at provisioning, CommonPractice-
style, via its licences ``byo_mode`` (w2v calls it ``llm_access``). The slots it
governs are the registry's ``byo_governed`` slots — this module never names a slot.

=============  =====================================  =========================
mode           which provider a governed slot uses    tenant writes on the slot
=============  =====================================  =========================
local_allowed  tenant connection if active, else       allowed
               the platform default
byo_required   tenant connection ONLY — no platform    allowed
               fall-through (preflight refuses the
               run up front; the resolver refuses
               again, defence in depth)
platform_only  the platform default — a saved tenant   refused (409)
               connection is IGNORED at pin/resolve
=============  =====================================  =========================

Non-governed slots (``device_farm``, ``publish.youtube``, the platform-provided
``scrape`` / ``media_search`` / ``doc_parse``) are unaffected by the mode.

Callers: ``resolver.pin_slots`` / ``resolver.resolve_slot`` / ``preflight.preflight_byo``
and the connection write routes (``api/routes/tenancy/connections.py``).
"""

from __future__ import annotations

from src.services.providers.byo_posture import ByoMode
from src.services.providers.registry import BYO_GOVERNED_SLOTS, get_slot

__all__ = [
    "governed",
    "platform_allowed",
    "required_slots",
    "tenant_connection_used",
    "tenant_writes_refused",
]


def governed(slot: str) -> bool:
    """True when ``slot`` is a model-provider slot subject to ``byo_mode``."""
    return get_slot(slot).byo_governed


def tenant_connection_used(mode: ByoMode, slot: str) -> bool:
    """Whether an active tenant connection on ``slot`` is used (False ⇒ ignored)."""
    return not (mode == "platform_only" and governed(slot))


def platform_allowed(mode: ByoMode, slot: str) -> bool:
    """Whether the platform default may serve ``slot`` for a tenant in ``mode``."""
    return not (mode == "byo_required" and governed(slot))


def tenant_writes_refused(mode: ByoMode, slot: str) -> bool:
    """Whether a write that SETS a tenant connection on ``slot`` is refused."""
    return mode == "platform_only" and governed(slot)


def required_slots(mode: ByoMode) -> frozenset[str]:
    """The slots a tenant in ``mode`` must bring itself (the Providers page badge)."""
    return frozenset(BYO_GOVERNED_SLOTS) if mode == "byo_required" else frozenset()
