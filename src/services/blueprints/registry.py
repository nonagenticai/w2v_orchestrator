"""Singleton-ish blueprint registry. Modules register at import time."""

from __future__ import annotations

from src.services.blueprints.base import DAGBlueprint


class BlueprintRegistry:
    def __init__(self) -> None:
        self._by_type: dict[str, DAGBlueprint] = {}
        self._by_id: dict[str, DAGBlueprint] = {}

    def register(self, bp: DAGBlueprint) -> None:
        if bp.project_type in self._by_type:
            raise ValueError(f"duplicate blueprint for project_type '{bp.project_type}'")
        # Wiring time. A blueprint whose terminal stage cannot supply the
        # deliverable handle the finalizer will look for never reaches the
        # registry — the import of its module raises, so CI and the service both
        # fail immediately instead of shipping a permanently-inert existence gate.
        bp.validate_deliverable_wiring()
        self._by_type[bp.project_type] = bp
        self._by_id[bp.id] = bp

    def resolve(self, project_type: str) -> DAGBlueprint:
        return self._by_type[project_type]

    def resolve_by_id(self, blueprint_id: str) -> DAGBlueprint:
        return self._by_id[blueprint_id]

    def all(self) -> list[DAGBlueprint]:
        return list(self._by_type.values())


_registry = BlueprintRegistry()


def get_registry() -> BlueprintRegistry:
    return _registry
