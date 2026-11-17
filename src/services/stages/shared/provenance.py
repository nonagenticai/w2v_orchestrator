"""Provenance-tracked value resolution + deterministic seeding.

Pure helpers, no DB. Two concerns live here:

1. **Provenance resolution** — a stage often has three candidate values for a
   knob: an explicit per-run ``override``, a ``smart-auto`` value the pipeline
   computed, and a ``default``. ``resolve_with_provenance`` picks the winner via
   a fixed ladder (override → smart_auto → default) and records *where the value
   came from* in a :class:`ProvenanceValue`, so the UI/audit can explain every
   knob. ``clamp`` records when a value was pulled back inside a bound.

2. **Deterministic seeding** — ``master_seed`` derives one 31-bit seed from a
   stable run field; ``subseed`` fans it out into per-spine 32-bit seeds keyed by
   spine id (never positional), so a re-run of the same config reproduces the same
   stochastic choices and re-keying a spine cannot shuffle unrelated clips.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

# Private sentinel: distinguishes "argument not supplied" from a supplied
# ``None`` / ``0`` / ``False`` (all of which are valid resolved values).
_UNSET: Final = object()

# The closed set of provenance sources. Kept for callers/tests to assert against.
VALID_SOURCES: Final = frozenset({"default", "smart-auto", "override", "llm-clamped"})


@dataclass(frozen=True)
class ProvenanceValue:
    """A resolved value plus where it came from.

    ``source`` is one of :data:`VALID_SOURCES`. ``computed_by`` names the
    heuristic/model that produced a smart-auto or clamped value. ``clamp_of``
    carries the pre-clamp value when ``source == "llm-clamped"``. ``input_hash``
    optionally records a stable hash of the inputs the value was derived from.
    """

    value: Any
    source: str
    computed_by: str | None = None
    clamp_of: Any = None
    input_hash: str | None = None


def resolve_with_provenance(
    key: str,
    *,
    override: Any = _UNSET,
    smart_auto: Any = _UNSET,
    default: Any,
    computed_by: str | None = None,
    clamp_of: Any = None,
) -> ProvenanceValue:
    """Resolve ``key`` down the ladder override → smart_auto → default.

    ``override`` and ``smart_auto`` default to the private ``_UNSET`` sentinel,
    so an explicitly-supplied ``None`` / ``0`` / ``False`` is a real value and
    still wins its rung. ``key`` is accepted for symmetry with call sites that
    resolve many keys in a loop (and for future audit logging); it does not
    affect the resolved value.
    """
    if override is not _UNSET:
        return ProvenanceValue(
            value=override, source="override", computed_by=computed_by, clamp_of=clamp_of
        )
    if smart_auto is not _UNSET:
        return ProvenanceValue(
            value=smart_auto, source="smart-auto", computed_by=computed_by, clamp_of=clamp_of
        )
    return ProvenanceValue(
        value=default, source="default", computed_by=computed_by, clamp_of=clamp_of
    )


def clamp(value: float, lo: float, hi: float, *, computed_by: str | None) -> ProvenanceValue:
    """Clamp ``value`` into ``[lo, hi]`` and record whether it actually moved.

    Returns source ``"llm-clamped"`` (with ``clamp_of`` = the original value)
    ONLY when clamping changed the value; otherwise the value was already in
    range and the source is ``"smart-auto"``.
    """
    clamped = min(max(value, lo), hi)
    if clamped != value:
        return ProvenanceValue(
            value=clamped, source="llm-clamped", computed_by=computed_by, clamp_of=value
        )
    return ProvenanceValue(value=value, source="smart-auto", computed_by=computed_by)


def input_hash(*parts: object) -> str:
    """Stable sha256 over the canonical JSON of ``parts``.

    Canonicalization uses ``sort_keys=True`` so dict key ordering never changes
    the hash (order-insensitive per the contract). Positional ``parts`` order is
    significant. Non-JSON-native values fall back to ``str`` so callers can pass
    UUIDs etc. without pre-stringifying.
    """
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def master_seed(config_snapshot: dict[str, Any]) -> int:
    """Derive a deterministic 31-bit master seed from a stable run field.

    Prefers ``project_id`` (stable across re-runs of the same project), falling
    back to the brief id or the title so a snapshot lacking a project id still
    seeds deterministically. The result is in ``[0, 2**31)``.
    """
    brief = config_snapshot.get("brief") or {}
    seed_field = (
        config_snapshot.get("project_id")
        or (brief.get("id") if isinstance(brief, dict) else None)
        or config_snapshot.get("title")
        or ""
    )
    digest = hashlib.sha256(str(seed_field).encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**31)


def subseed(master: int, namespace: str, spine_id: str) -> int:
    """Derive a deterministic 32-bit subseed from (master, namespace, spine_id).

    Keyed BY the spine id (never a positional index) so re-keying or reordering
    the spine cannot shuffle the stochastic choices of unrelated clips. Result is
    in ``[0, 2**32)``.
    """
    digest = hashlib.sha256(f"{master}:{namespace}:{spine_id}".encode()).hexdigest()
    return int(digest, 16) % (2**32)
