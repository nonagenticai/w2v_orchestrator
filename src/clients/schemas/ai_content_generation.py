"""Typed responses for ai_content_generation_service (port 6310).

Source of truth (read, not guessed) — all models are plain ``BaseModel`` with
snake_case fields, no aliases, no ``by_alias``; wire keys == field names:
- VIDEO family, ``ai_content_generation_service/src/api/schemas.py``:
  - ``GET /api/content/videos/jobs/{id}``        -> ``JobStatusResponse``  (THIN)
  - ``GET /api/content/videos/jobs/{id}/result`` -> ``GenerationResultResponse`` (FAT)
- CONCEPT-IMAGE family, ``.../src/api/content_schemas.py`` (built explicitly in
  ``content_routes.py``): job / asset / batch-submission / batch models.

⚠️ The VIDEO status envelope is THIN on purpose. Unlike the concept-image status
(which carries the asset URL inline), ``JobStatusResponse`` has **no**
``video_url`` / ``asset_id`` / duration — only a ``result_url`` pointer to the
``/result`` endpoint, where ``GenerationResultResponse`` carries ``video_url``
and ``actual_duration_s``. A consumer that reads the asset off the *status* poll
gets ``None`` for every asset field. Encoding the two shapes as distinct models
makes that mistake a type error instead of a silent prod failure (it was one:
``ai_video_gen`` used to read ``video_url`` off the status poll and always
tripped its own "no usable video_url" gate).

Concept-image note: the ``duration``/``asset_id`` split does NOT apply there —
``ConceptImageJobResponse`` (the *status*) already carries ``asset_id`` / ``url``
/ ``image_url`` / ``download_url``, which is why the image path never needed a
second result fetch.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------


class AcgVideoJobStatus(BaseModel):
    """``GET /api/content/videos/jobs/{id}`` — the THIN poll/status envelope.

    Mirrors ``JobStatusResponse``. Carries no asset URL/duration: ``result_url``
    points at the ``/result`` endpoint that does. ``status`` is deliberately not
    a ``Literal`` — a new downstream status string must not hard-fail a run.
    """

    job_id: str
    status: str
    provider: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    result_url: str | None = None


class AcgVideoProviderInfo(BaseModel):
    """One entry of ``GET /api/content/videos/providers``.

    Mirrors ``ProviderInfoResponse``. ``available`` is NOT a static capability
    table: ACG's ``_init_video_provider_registry`` only ``register()``s a
    provider whose ``health_check()`` passed at start-up (a missing/invalid API
    key or an absent local GPU means it never enters the registry at all), and
    ``list_providers`` flips ``available`` to ``False`` if introspecting a
    registered provider then raises. So an EMPTY ``providers`` list is the
    truthful "this deployment cannot generate video" signal.
    """

    name: str
    available: bool
    models: list[str] | None = None
    styles: list[str] | None = None
    max_duration_s: float | None = None
    supports_image_to_video: bool = False


class AcgVideoProviderList(BaseModel):
    """``GET /api/content/videos/providers`` — mirrors ``ProvidersListResponse``.

    ``default_provider`` is echoed straight from ACG's CONFIG
    (``get_settings().generation.default_provider``, see the service's
    ``routes/providers.py``) — it is NOT reconciled against the registry, so it can
    name a provider that never registered. Only ``providers`` carries capability;
    never treat a non-null ``default_provider`` as evidence that anything is usable.
    """

    providers: list[AcgVideoProviderInfo] = Field(default_factory=list)
    default_provider: str | None = None

    @property
    def available_names(self) -> set[str]:
        """Names of providers that are registered AND introspectable."""
        return {p.name for p in self.providers if p.available}


class AcgVideoGenerationResult(BaseModel):
    """``GET /api/content/videos/jobs/{id}/result`` — the FAT result envelope.

    Mirrors ``GenerationResultResponse``. Only returned once the job is fully
    ``completed`` (the route 409s otherwise). ``video_url`` may be an
    ACG-relative download path (``/api/content/videos/jobs/{id}/download``) that
    the caller re-publishes; the duration field is ``actual_duration_s`` (NOT
    ``duration_s`` — a name the old consumer read and never found).
    """

    job_id: str
    status: str
    provider: str | None = None
    prompt: str | None = None
    video_path: str | None = None
    video_url: str | None = None
    file_size_bytes: int | None = None
    actual_duration_s: float | None = None
    generation_time_seconds: float | None = None
    cost: float | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Concept images
# ---------------------------------------------------------------------------


class AcgConceptImageJob(BaseModel):
    """``GET /api/concept-images/jobs/{id}`` — the FAT status envelope.

    Mirrors ``ConceptImageJobResponse``. Unlike the video status, this carries
    the asset inline (``asset_id`` / ``url`` / ``image_url`` / ``download_url``),
    so ``ai_image_gen`` / ``course_thumbnail`` read it directly off the poll.
    ``presigned_url`` is always ``None`` in the current serializer.
    """

    job_id: str
    asset_id: str | None = None
    status: str
    provider: str | None = None
    url: str | None = None
    image_url: str | None = None
    download_url: str | None = None
    presigned_url: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


class AcgConceptImageAsset(BaseModel):
    """``GET /api/concept-images/jobs/{id}/result`` and ``/assets/{id}``.

    Mirrors ``ConceptImageAssetResponse``. ``asset_id`` is required here (unlike
    on the status envelope). ``usage_context`` / ``metadata`` are opaque JSON
    blobs the orchestrator never reads — ``dict[str, object]``, never
    ``dict[str, Any]`` (per the schemas-package convention).
    """

    job_id: str
    asset_id: str
    status: str
    provider: str | None = None
    prompt: str | None = None
    url: str | None = None
    image_path: str | None = None
    image_url: str | None = None
    download_url: str | None = None
    presigned_url: str | None = None
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    file_size_bytes: int | None = None
    cost: float | None = None
    usage_context: dict[str, object] | None = None
    target_usage: str | None = None
    error: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    metadata: dict[str, object] | None = None


class AcgConceptImageBatchSubmission(BaseModel):
    """``POST /api/concept-images/batches`` — the batch-accepted envelope.

    Mirrors ``ConceptImageBatchSubmissionResponse``.
    """

    batch_id: str
    job_ids: list[str] = Field(default_factory=list)
    status: str = "pending"
    status_url: str | None = None
    result_url: str | None = None


class AcgConceptImageBatch(BaseModel):
    """``GET /api/concept-images/batches/{id}`` and ``/batches/{id}/result``.

    Mirrors ``ConceptImageBatchResponse``. ``results`` is populated only with
    completed/failed/cancelled children.
    """

    batch_id: str
    status: str
    job_ids: list[str] = Field(default_factory=list)
    job_count: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    pending_jobs: int = 0
    budget_limit: float | None = None
    total_cost: float = 0.0
    created_at: str | None = None
    completed_at: str | None = None
    results: list[AcgConceptImageAsset] = Field(default_factory=list)
