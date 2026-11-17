"""Run-start BYO preflight: refuse a byo_required run BEFORE any stage spends money.

A run whose blueprint will call a model-provider slot (every ``byo_governed`` registry
slot: ``llm``, ``vlm``, ``tts``, ``asr``, ``image``, ``video_gen``, ``ocr``) for a
tenant that is licences ``byo_required`` and has no tenant connection of its own for
that slot could only fail later — after upstream paid stages already ran. So
``RunEngine.start_run`` asks here first, after the run's slots are pinned:

* the slots the run NEEDS come from the stage classes' ``provider_slots``
  declarations (the same source as the P8 requirements manifest), for every stage
  that will actually execute (preserved regenerate stages are excluded) and whose
  ``when`` opt-in is present in the snapshot;
* a needed slot already pinned to a ``tenant`` connection is satisfied;
* otherwise the tenant's posture is read ONCE: when ``platform_allowed`` is false
  for it (``providers/access.py``) the run is refused with ONE
  :class:`SlotNotConfiguredError` naming EVERY missing slot (``byo_provider_required``,
  ``modality`` = the first, ``slots`` = all, comma-separated); a posture read failure
  raises :class:`ByoPostureUnavailableError` (transient) — never a platform fallback.

``local_allowed`` / ``platform_only`` pass (the platform default serves the slot).
The resolver re-checks the same rule at resolve time, and modality_gateway's 402 stays
the backstop for every platform-GPU call.
"""

from __future__ import annotations

from collections.abc import Container, Mapping
from typing import TYPE_CHECKING
from uuid import UUID

from src.services.providers.access import platform_allowed
from src.services.providers.errors import SlotNotConfiguredError
from src.services.providers.registry import SLOTS
from src.services.providers.resolver import PROVIDER_SLOTS_KEY

if TYPE_CHECKING:
    from src.services.blueprints.base import DAGBlueprint
    from src.services.providers.byo_posture import ByoPostureReader

__all__ = ["preflight_byo", "required_byo_modalities"]


def _opted_in(when: str, config_snapshot: Mapping[str, object]) -> bool:
    """``always``, or the ``brief.<key>`` opt-in is present (snapshot or brief).

    ``optional`` (a best-effort call the stage degrades without) never requires a
    provider up front."""
    if when == "always":
        return True
    if when == "optional":
        return False
    key = when.removeprefix("brief.")
    brief = config_snapshot.get("brief")
    return bool(config_snapshot.get(key)) or (isinstance(brief, Mapping) and bool(brief.get(key)))


def required_byo_modalities(
    blueprint: DAGBlueprint,
    config_snapshot: Mapping[str, object],
    *,
    skip_stages: Container[str] = (),
) -> list[str]:
    """``byo_governed`` slots this run will call, in first-use (topological) order."""
    from src.services.blueprints.base import stage_handler_class

    by_name = {spec.name: spec for spec in blueprint.stages}
    needed: list[str] = []
    for name in blueprint.topological_order():
        if name in skip_stages:
            continue
        cls = stage_handler_class(by_name[name])
        if cls is None:
            continue
        for use in cls.provider_slots:
            spec = SLOTS.get(use.slot)
            if (
                spec is not None
                and spec.byo_governed
                and use.slot not in needed
                and _opted_in(use.when, config_snapshot)
            ):
                needed.append(use.slot)
    return needed


def _tenant_pinned(config_snapshot: Mapping[str, object], modality: str) -> bool:
    pins = config_snapshot.get(PROVIDER_SLOTS_KEY)
    entry = pins.get(modality) if isinstance(pins, Mapping) else None
    return isinstance(entry, Mapping) and entry.get("source") == "tenant"


async def preflight_byo(
    blueprint: DAGBlueprint,
    config_snapshot: Mapping[str, object],
    tenant_id: UUID,
    *,
    posture: ByoPostureReader,
    skip_stages: Container[str] = (),
) -> None:
    """Raise unless the tenant may run every model-provider slot this run needs."""
    missing = [
        m
        for m in required_byo_modalities(blueprint, config_snapshot, skip_stages=skip_stages)
        if not _tenant_pinned(config_snapshot, m)
    ]
    if not missing:
        return
    mode = await posture.byo_mode(tenant_id, slot=missing[0])
    refused = [m for m in missing if not platform_allowed(mode, m)]
    if refused:
        raise SlotNotConfiguredError(refused)
