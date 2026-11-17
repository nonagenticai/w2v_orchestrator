"""The reel's requested length, resolved once for every stage that needs it.

Two stages need this number and they must agree on it: ``segment_select`` tells
the editor how much material to find, and ``cut_plan`` fits that material to the
music and reports the ``shortfall_seconds`` when the selection could not fill the
request. Resolving it twice is how the selector and the planner end up aiming at
different reels.

It lives in its own leaf module rather than in ``cut_plan`` because the import
would otherwise run UPSTREAM — ``segment_select`` reaching into the stage that
consumes it — and any future edge back the other way would be a cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext


def target_seconds(ctx: RunContext) -> float | None:
    """The reel's target length, or ``None`` for "use every selected segment".

    Same three-source resolution the fleet already uses for this key
    (``shared/script_doctor._resolve_target``): the run config, then the
    deliverable descriptor, then the brief. No default is invented — an author
    who did not ask for a length gets the whole approved selection, which is the
    edit a human already signed off on.
    """
    snapshot: Mapping[str, object] = ctx.config_snapshot or {}
    for source in (
        snapshot,
        _mapping(snapshot.get("deliverable")),
        _mapping(snapshot.get("brief")),
    ):
        value = source.get("target_duration_seconds")
        if value is not None:
            seconds = _as_float(value)
            return seconds if seconds > 0 else None
    return None


def _mapping(value: object) -> Mapping[str, object]:
    """A nested descriptor, or an empty one. ``isinstance``-narrowing an
    ``object`` yields ``Mapping[Unknown, Unknown]``, so the cast names the shape
    a config snapshot's sub-objects actually have rather than widening it."""
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value)
    return {}


def _as_float(value: object) -> float:
    try:
        return float(value)  # ty: ignore[invalid-argument-type]
    except TypeError, ValueError:
        return 0.0
