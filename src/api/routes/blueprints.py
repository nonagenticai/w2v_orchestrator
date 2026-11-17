"""Blueprint metadata — list all registered blueprints + fetch one by id."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

import src.services.blueprints  # noqa: F401  triggers blueprint registration
from src.api.schemas import BlueprintStage, BlueprintSummary
from src.config import get_settings
from src.services.blueprints.base import DAGBlueprint
from src.services.blueprints.registry import get_registry

router = APIRouter(prefix="/api/blueprints", tags=["blueprints"])


def _summarize(bp: DAGBlueprint) -> BlueprintSummary:
    """Map a blueprint's DAG (StageSpec → BlueprintStage) into its API summary."""
    # None on a spec means "inherit the runtime LLM base ceiling"; surface the
    # resolved effective value so the API never reports a null timeout.
    base_ceiling = get_settings().pipeline.llm_stage_base_ceiling_s
    stages = [
        BlueprintStage(
            name=s.name,
            depends_on=list(s.depends_on),
            parallel_group=s.parallel_group,
            default_timeout_seconds=(
                s.default_timeout_seconds if s.default_timeout_seconds is not None else base_ceiling
            ),
            default_pause_after=s.default_pause_after,
        )
        for s in bp.stages
    ]
    return BlueprintSummary(id=bp.id, project_type=bp.project_type, stages=stages)  # ty: ignore[invalid-argument-type]  # why: blueprint.project_type is a runtime str; pydantic Literal narrows at construction


@router.get("", response_model=list[BlueprintSummary], operation_id="blueprints.list")
async def list_blueprints() -> list[BlueprintSummary]:
    return [_summarize(bp) for bp in get_registry().all()]


@router.get("/{blueprint_id}", response_model=BlueprintSummary, operation_id="blueprints.get")
async def get_blueprint(blueprint_id: str) -> BlueprintSummary:
    """Fetch one blueprint's DAG shape by its **id** (e.g. ``seo_audit_v1``)."""
    try:
        bp = get_registry().resolve_by_id(blueprint_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"blueprint {blueprint_id!r} not found"
        ) from exc
    return _summarize(bp)
