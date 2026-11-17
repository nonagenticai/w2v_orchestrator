"""Deterministic (no-LLM) authoring front door — the A0 planning core.

Turns a high-level ``VideoIntent`` into a computed ``PlanDocument`` by walking
each knob down the provenance ladder (override > smart-auto > default), reusing
the pure planning substrate in :mod:`src.services.stages.shared.compute_plan`
and :mod:`src.services.stages.shared.provenance`. No DB, no HTTP, no
``RunContext`` lives here.
"""

from __future__ import annotations

# NOTE: `resolve_plan` is deliberately NOT re-exported here. It imports
# ``src.api.schemas.intent`` (the parallel A0 schema PR); importing it at package
# init would couple `presets` / `compute_levers` — and their tests — to that
# module landing first. Import it directly: ``from
# src.services.planning.resolve_plan import resolve_plan``.
from src.services.planning.presets import Preset, load_preset

__all__ = ["Preset", "load_preset"]
