"""Tenant provider slots: registry, policy, resolver and secret GC (P3)."""

from src.services.providers.errors import (
    ProviderKeyringUnavailableError,
    SlotEgressRefusedError,
    SlotNotConfiguredError,
    SlotUnavailableError,
)
from src.services.providers.registry import (
    PINNED_SLOT_NAMES,
    SLOT_NAMES,
    SLOTS,
    SlotSpec,
    get_slot,
    writable_slots,
)
from src.services.providers.resolver import (
    PROVIDER_SLOTS_KEY,
    ConnectionRepository,
    ConnectionVersionRecord,
    ResolvedSlot,
    SecretDecryptor,
    SlotRef,
    pin_slots,
    pins_to_snapshot,
    resolve_slot,
    secret_aad,
)
from src.services.providers.settings import ProviderPolicySettings, get_provider_policy

__all__ = [
    "PINNED_SLOT_NAMES",
    "PROVIDER_SLOTS_KEY",
    "SLOTS",
    "SLOT_NAMES",
    "ConnectionRepository",
    "ConnectionVersionRecord",
    "ProviderKeyringUnavailableError",
    "ProviderPolicySettings",
    "ResolvedSlot",
    "SecretDecryptor",
    "SlotEgressRefusedError",
    "SlotNotConfiguredError",
    "SlotRef",
    "SlotSpec",
    "SlotUnavailableError",
    "get_provider_policy",
    "get_slot",
    "pin_slots",
    "pins_to_snapshot",
    "resolve_slot",
    "secret_aad",
    "writable_slots",
]
