"""Carry a run's PINNED provider slots + model tiers into its background task.

``build_config_snapshot`` pins every provider slot (``provider_slots``) and the
per-stage model tiers (``model_tier``) at run start. The dispatcher binds them
here — next to ``bind_run_attribution`` — so a call site deep inside a stage can
ask "which provider does THIS run use for ``llm``?" without threading the
snapshot through ~77 call sites. ``asyncio`` tasks copy the context, so stage
tasks spawned after the bind see it.

A legacy run (snapshot without the keys) binds "nothing pinned": callers then
fall back to live resolution / the env tier, exactly as before P3.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Final

from w2v_common.auth.identity_context import (
    get_correlation_id_from_context,
    get_stage_from_context,
    get_w2v_project_id_from_context,
    set_correlation_context,
    set_w2v_project_context,
)

from src.observability.logging import bound_correlation_id
from src.services.providers.resolver import PROVIDER_SLOTS_KEY, SlotRef

__all__ = [
    "MODEL_TIER_KEY",
    "bind_run_provider_slots",
    "bound_existing_run_scope",
    "clear_run_provider_slots",
    "current_run_model_tier",
    "current_run_slot",
]

#: ``config_snapshot`` key for the per-stage model tiers pinned at run start.
MODEL_TIER_KEY: Final = "model_tier"


@dataclass(frozen=True, slots=True)
class _Binding:
    slots: Mapping[str, object] | None = None
    model_tier: Mapping[str, object] | None = field(default=None)


_RUN_PROVIDERS: ContextVar[_Binding | None] = ContextVar("w2v_run_providers", default=None)


def _binding_from(snapshot: Mapping[str, object]) -> _Binding:
    slots = snapshot.get(PROVIDER_SLOTS_KEY)
    tiers = snapshot.get(MODEL_TIER_KEY)
    return _Binding(
        slots=dict(slots) if isinstance(slots, Mapping) else None,
        model_tier=dict(tiers) if isinstance(tiers, Mapping) else None,
    )


def bind_run_provider_slots(snapshot: Mapping[str, object]) -> None:
    """Bind ``snapshot``'s pins for the current context (always overwrites)."""
    _RUN_PROVIDERS.set(_binding_from(snapshot))


@contextmanager
def bound_existing_run_scope(
    snapshot: Mapping[str, object], *, run_id: str, stage: str | None
) -> Iterator[None]:
    """Act AS an existing run for one block in REQUEST context, then restore.

    A request-context path that calls a provider on behalf of an EXISTING run (the
    single-line re-narrate) has no dispatcher bind: without this the call sees no
    pin and silently takes the platform provider (a tenant that pinned its own TTS
    gets one platform clip spliced in), and its usage carries the HTTP request's
    correlation id instead of the run's. Binds the run's pinned slots + model
    tiers, the cost correlation (run id + stage) and the run's W2V project; every
    value is restored on exit so the rest of the request is untouched. The
    caller's actor / fr project stay the request's own (it IS the caller).
    """
    slots_token = _RUN_PROVIDERS.set(_binding_from(snapshot))
    prior_correlation = get_correlation_id_from_context()
    prior_stage = get_stage_from_context()
    prior_project = get_w2v_project_id_from_context()
    try:
        set_correlation_context(run_id, stage)
        project = snapshot.get("project_id")
        if isinstance(project, str) and project:
            set_w2v_project_context(project)
        with bound_correlation_id(run_id):
            yield
    finally:
        set_correlation_context(prior_correlation or None, prior_stage or None)
        set_w2v_project_context(prior_project or None)
        _RUN_PROVIDERS.reset(slots_token)


def clear_run_provider_slots() -> None:
    """Unbind (long-lived reconciler tasks, between runs)."""
    _RUN_PROVIDERS.set(None)


def current_run_slot(slot: str) -> SlotRef | None:
    """The bound run's pin for ``slot``; ``None`` when unbound / legacy / not pinned.

    A malformed pin raises ``SlotUnavailableError`` (fail the stage, never guess)."""
    binding = _RUN_PROVIDERS.get()
    if binding is None or binding.slots is None:
        return None
    entry = binding.slots.get(slot)
    if entry is None:
        return None
    if isinstance(entry, SlotRef):
        return entry
    if not isinstance(entry, Mapping):
        from src.services.providers.errors import SlotUnavailableError

        raise SlotUnavailableError(slot, "the run's pinned provider reference is malformed")
    return SlotRef.from_dict(slot, entry)


def current_run_model_tier(stage_key: str, default: str | None) -> str | None:
    """The tier the bound run pinned for ``stage_key``; ``default`` when it pinned none.

    "Pinned none" means unbound, a legacy snapshot without ``model_tier``, or the key
    absent. A PRESENT ``None`` is a real pin (the run started on the default tier)
    and is returned as ``None`` even if the env has changed since."""
    binding = _RUN_PROVIDERS.get()
    if binding is None or binding.model_tier is None or stage_key not in binding.model_tier:
        return default
    value = binding.model_tier[stage_key]
    return value if isinstance(value, str) else None
