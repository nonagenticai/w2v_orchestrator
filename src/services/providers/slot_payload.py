"""Build the cross-repo ``provider_slot`` body field from the run-pinned slot (P5-2).

Downstream services (ai_content_generation, app_inspector) never store tenant
slots: the orchestrator resolves the slot the RUN pinned and passes it per
request. Contract (fixed, ``w2v-p5-slots-blueprint.md`` § "Cross-repo contract")::

    {"slot": "image"|"video_gen"|"device_farm", "kind": str, "base_url": str|null,
     "model": str|null, "options": {..}, "secret": str|null,
     "ref": {"connection_id": uuid, "version": int}}

Sent ONLY when the bound run pinned a ``source=tenant`` connection for the slot.
Unbound context (API request, reconciler), legacy run, or deployment pin →
``None`` and the caller sends today's body unchanged.

The returned dict carries the DECRYPTED secret: it exists only to be serialised
into the downstream request body. Never log it, never put it in an exception,
never persist it. :func:`redact_provider_slot` gives a log-safe copy.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Final

from w2v_common.auth.identity_context import get_tenant_id_from_context

from src.auth.run_providers import current_run_slot
from src.services.providers.errors import SlotUnavailableError
from src.services.providers.resolver import PROVIDER_SLOTS_KEY, ResolvedSlot, SlotRef, resolve_slot

__all__ = [
    "PROVIDER_SLOT_FIELD",
    "build_provider_slot",
    "redact_provider_slot",
    "resolve_pinned_slot",
    "slot_payload",
]

#: Request-body field name on the downstream side.
PROVIDER_SLOT_FIELD: Final = "provider_slot"
_REDACTED: Final = "[REDACTED]"

#: (slot, orchestrator kind) -> the kind name the DOWNSTREAM service expects.
#: The orchestrator's registry spells the self-hosted OpenAI kind
#: ``openai-compatible`` (hyphen, shared with llm/vlm/tts/asr); ai_content_generation
#: (#141) validates image kinds against ``nonagentic_endpoint`` / ``openai_compatible``
#: (underscore). Every other (slot, kind) is sent verbatim.
_WIRE_KINDS: Final[dict[tuple[str, str], str]] = {
    ("image", "openai-compatible"): "openai_compatible",
}


def wire_kind(slot: str, kind: str) -> str:
    """The ``kind`` value sent downstream for ``slot`` (see ``_WIRE_KINDS``)."""
    return _WIRE_KINDS.get((slot, kind), kind)


async def resolve_pinned_slot(tenant_id: uuid.UUID, slot: str, ref: SlotRef) -> ResolvedSlot:
    """Load + validate exactly the pinned version of ``slot`` for ``tenant_id``.

    Same path as the P3 ``llm`` dispatcher: tenant-bound session (RLS), the
    resolver re-checks tenant/slot/purge/policy/egress and decrypts the secret."""
    from src.config import get_settings
    from src.database.session import bind_session_tenant, get_db_session
    from src.services.providers.adapters import build_resolver_deps

    async with get_db_session() as session:
        await bind_session_tenant(session, tenant_id)
        deps = build_resolver_deps(session, get_settings())
        return await resolve_slot(
            deps.repo,
            deps.decryptor,
            tenant_id,
            slot,
            run_snapshot={PROVIDER_SLOTS_KEY: {slot: ref.to_dict()}},
            policy=deps.policy,
            posture=deps.posture,
        )


def _bound_tenant(slot: str) -> uuid.UUID:
    raw = get_tenant_id_from_context()
    try:
        return uuid.UUID(str(raw))
    except ValueError, TypeError:
        raise SlotUnavailableError(
            slot, "no tenant is bound to this call, so its provider cannot be used"
        ) from None


def slot_payload(resolved: ResolvedSlot) -> dict[str, object]:
    """The contract JSON for a resolved TENANT slot (secret in clear — see module doc)."""
    ref = resolved.ref
    if ref.source != "tenant" or ref.connection_id is None or ref.version is None:
        raise ValueError("provider_slot is only built for a tenant connection")
    return {
        "slot": ref.slot,
        "kind": wire_kind(ref.slot, ref.kind),
        "base_url": resolved.base_url,
        "model": resolved.model,
        "options": dict(resolved.options),
        "secret": resolved.secret.get_secret_value() if resolved.secret is not None else None,
        "ref": {"connection_id": str(ref.connection_id), "version": ref.version},
    }


async def build_provider_slot(slot_name: str) -> dict[str, object] | None:
    """``provider_slot`` for the bound run's pin of ``slot_name``, or ``None``.

    ``None`` unless the run pinned a tenant connection. A pinned tenant connection
    that cannot be used (purged, policy, egress, undecryptable) raises
    :class:`SlotUnavailableError` — never a silent fall back to the platform."""
    ref = current_run_slot(slot_name)
    if ref is None or ref.source != "tenant":
        return None
    resolved = await resolve_pinned_slot(_bound_tenant(slot_name), slot_name, ref)
    return slot_payload(resolved)


def redact_provider_slot(payload: Mapping[str, object] | None) -> dict[str, object] | None:
    """Log-safe copy: ``secret`` replaced when present."""
    if payload is None:
        return None
    out = dict(payload)
    if out.get("secret") is not None:
        out["secret"] = _REDACTED
    return out
