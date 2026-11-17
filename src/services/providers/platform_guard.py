"""The ONE choke point before any in-process client takes the PLATFORM path for a slot.

Run-start preflight (``providers/preflight.py``) only covers the slots a blueprint
DECLARES as required, at run START. It cannot cover:

* request-context calls with no run pin (storyboard assist, critique, previews …);
* optional slot uses the preflight deliberately skips (``when="optional"``);
* a run pinned to the platform BEFORE its tenant was flipped to ``byo_required``
  (a paused run resumed later).

So every in-process model client calls :func:`refuse_platform_for_bound_tenant`
right before its platform/deployment branch (LLM ``WMClient``, ``TTSClient``,
``TTSGatewayClient``, ``TranscriptionWorkerClient``, ACG image/video_gen). OCR
resolves through ``resolve_slot`` itself, which applies the same rule.

* A tenant-pinned slot is not the platform path → no-op (the caller uses the tenant).
* NO tenant bound (a cron / background system call) → no-op: CommonPractice's rule,
  no tenant ⇒ platform.
* Otherwise the tenant's licences ``byo_mode`` decides (``providers/access.py``):
  ``byo_required`` raises :class:`~src.services.providers.errors.SlotNotConfiguredError`
  naming the slot; an unreadable posture with no last-known value raises
  :class:`~src.services.providers.errors.ByoPostureUnavailableError`.
"""

from __future__ import annotations

import uuid

from w2v_common.auth.identity_context import get_tenant_id_from_context

__all__ = ["bound_tenant_or_none", "refuse_platform_for_bound_tenant"]


def bound_tenant_or_none() -> uuid.UUID | None:
    """The tenant bound to this context, or ``None`` (a system / no-tenant call)."""
    raw = get_tenant_id_from_context()
    if not raw:
        return None
    try:
        return uuid.UUID(str(raw))
    except ValueError, TypeError:
        return None


async def refuse_platform_for_bound_tenant(slot: str) -> None:
    """Raise when the bound tenant may not use the PLATFORM for ``slot`` (see module doc)."""
    from src.auth.run_providers import current_run_slot
    from src.services.providers import adapters
    from src.services.providers.resolver import refuse_platform_if_byo_required

    ref = current_run_slot(slot)
    if ref is not None and ref.source == "tenant":
        return
    tenant_id = bound_tenant_or_none()
    if tenant_id is None:
        return
    await refuse_platform_if_byo_required(adapters.get_byo_posture(), tenant_id, slot)
