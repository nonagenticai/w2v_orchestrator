"""Typed responses for html_render_service's design API (port 6319).

Source of truth (read, not guessed) — html_render_service ``src/api/schemas.py``:
- ``GET /api/designs/jobs/{job_id}``        -> ``JobStatusResponse``     (:574-600)
- ``GET /api/designs/jobs/{job_id}/result`` -> ``RedesignResultResponse`` (:523-561)

Nested models mirrored 1:1 from the same file:
``AccessibilityFindingResponse`` (:444), ``AccessibilityCategoryResponse`` (:455),
``AccessibilityScoreResponse`` (:463), ``VariantResult`` (:474),
``VisualAssetResponse`` (:485), ``ConceptImageAssetResponse`` (:498).

``status`` fields are deliberately NOT ``Literal``: a new service-side status
string must not be able to hard-fail a running pipeline. Blobs the orchestrator
never reads (``token_usage``, ``metadata``, ``html_validation``, ...) are
``dict[str, object]`` — never ``Any``. FastAPI serializes through
``response_model``, so a field absent from the server model never arrives;
nothing here invents one.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DesignAccessibilityFinding(BaseModel):
    """One accessibility finding — mirrors ``AccessibilityFindingResponse``."""

    category: str
    severity: str
    description: str
    element: str | None = None
    recommendation: str | None = None
    wcag_criterion: str | None = None


class DesignAccessibilityCategory(BaseModel):
    """Per-category accessibility score — mirrors ``AccessibilityCategoryResponse``."""

    category: str
    # None = this category was never scored (the LLM reply omitted it). The
    # server emits NULL here deliberately; a required float rejected the whole
    # result payload after the render was already paid for.
    score: float | None = None
    findings: list[DesignAccessibilityFinding] = Field(default_factory=list)


class DesignAccessibilityScore(BaseModel):
    """Accessibility assessment — mirrors ``AccessibilityScoreResponse``."""

    # None = UNSCORED (the assessment did not happen). Never substitute a
    # number — server ``accessibility_scorer.py`` emits NULL + ``unscored_reason``
    # on an unparseable LLM reply.
    overall_score: float | None = None
    unscored_reason: str | None = None
    categories: list[DesignAccessibilityCategory] = Field(default_factory=list)
    findings: list[DesignAccessibilityFinding] = Field(default_factory=list)
    html_was_fixed: bool = False
    auto_theme: dict[str, object] | None = None
    critique_adapter_validation: dict[str, object] | None = None


class DesignVariantResult(BaseModel):
    """One A/B style variant's outcome — mirrors ``VariantResult``."""

    variant_id: str
    style: str
    status: str
    redesigned_html_path: str | None = None
    screenshot_paths: dict[str, str] | None = None
    error: str | None = None


class DesignVisualAsset(BaseModel):
    """Structured visual asset for downstream consumers — mirrors ``VisualAssetResponse``."""

    asset_id: str
    title: str
    kind: str
    role: str
    url: str
    source: str
    page_url: str | None = None
    metadata: dict[str, object] | None = None


class DesignConceptImageAsset(BaseModel):
    """Resolved AI concept-image asset — mirrors ``ConceptImageAssetResponse``."""

    job_id: str | None = None
    asset_id: str
    provider: str | None = None
    status: str = "completed"
    prompt: str | None = None
    url: str
    download_url: str | None = None
    presigned_url: str | None = None
    image_url: str | None = None
    target_usage: str | None = None
    usage_context: dict[str, object] | None = None
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    file_size_bytes: int | None = None
    cost: float | None = None
    error: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    metadata: dict[str, object] | None = None


class DesignJobStatus(BaseModel):
    """``GET /api/designs/jobs/{job_id}`` — mirrors ``JobStatusResponse``.

    ``accessibility_score`` lives HERE (the status body), not on the result —
    consumers read it off the ``wait_for_design`` return, matching the server's
    placement.
    """

    job_id: str
    status: str
    url: str | None = None
    scope: str | None = None
    style: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    result_url: str | None = None
    iteration_number: int | None = None
    parent_job_id: str | None = None
    root_job_id: str | None = None
    refine_url: str | None = None
    # True + reason when concept images were requested and generation FAILED —
    # so an empty concept_image_assets is not read as "none requested".
    concept_image_degraded: bool = False
    concept_image_error: str | None = None
    accessibility_score: float | None = None
    batch_id: str | None = None
    token_usage: dict[str, object] | None = None


class DesignJobResult(BaseModel):
    """``GET /api/designs/jobs/{job_id}/result`` — mirrors ``RedesignResultResponse``."""

    job_id: str
    url: str
    scope: str
    style: str
    redesigned_html_path: str | None = None
    section_html_paths: list[str] | None = None
    screenshots: dict[str, str] | None = None
    # Discrete per-section redesign stills, section_name → fetchable URL (also
    # surfaced individually in visual_assets as redesign_section:<name>).
    section_screenshots: dict[str, str] | None = None
    concept_image_assets: list[DesignConceptImageAsset] | None = None
    # True + reason when concept images were requested and generation FAILED —
    # so an empty concept_image_assets is not read as "none requested".
    concept_image_degraded: bool = False
    concept_image_error: str | None = None
    visual_assets: list[DesignVisualAsset] | None = None
    variants: list[DesignVariantResult] | None = None
    component_html_path: str | None = None
    accessibility: DesignAccessibilityScore | None = None
    original_screenshots: dict[str, str] | None = None
    comparisons: dict[str, str] | None = None
    diffs: dict[str, str] | None = None
    html_validation: dict[str, object] | None = None
    verification_status: str | None = None
    verification_report: dict[str, object] | None = None
    token_usage: dict[str, object] | None = None
