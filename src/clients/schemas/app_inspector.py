"""Typed responses for app_inspector_service (port 6311).

Source of truth (read, not guessed) — ``app_inspector_service/src/api/schemas.py``:
  - ``GET /api/inspect/{job_id}``          -> ``JobStatusResponse``       (THIN)
  - ``GET /api/inspect/{job_id}/result``   -> ``InspectionResultResponse`` (FAT)

⚠️ The status poll is THIN on purpose. ``JobStatusResponse`` carries only
``job_id / status / screens_discovered / created_at / completed_at / error /
result_url`` — a pointer, NOT the inspection. The FAT
``InspectionResultResponse`` at ``/result`` carries the screens, accessibility
score, and the inferred ``estimated_purpose``. A consumer that reads inspection
data off the *status* poll gets nothing — which is exactly the dead-signal bug
this module fixes (``app_review`` used to poll the status endpoint and read
``title/tagline/pricing`` fields that live on NEITHER model, so every fact was
silently empty). Same failure mode as
``ai_content_generation`` VIDEO's thin-status trap.

These are the CONSUMED SUBSET of the server model, not a full mirror:
``extra="ignore"`` means a server-side schema addition never 500s this client.
The drift guard is ``tests/contracts/test_app_inspector_contract.py``, which
walks this model's schema against a vendored copy of the server's — so a field
we read that the server renames or drops fails CI instead of going dead on the
wire.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ScreenSummary(BaseModel):
    """Consumed subset of the server's ``ScreenInfo`` (a discovered app screen)."""

    model_config = ConfigDict(extra="ignore")

    screen_id: str = ""
    screen_type: str = ""
    activity_name: str | None = None
    text_content: str | None = None
    headings: list[str] = Field(default_factory=list)
    # None means NOT MEASURED. It was `int = 100`, which fabricated a perfect
    # score for a screen nothing had scored — the same defect the server just
    # fixed by making these nullable (app_inspector PR #82). A non-nullable int
    # here would make pydantic REJECT the server's null and break the consumer,
    # so this must land before/with that change.
    accessibility_score: int | None = None
    # A pod-local FILE PATH under STORAGE_BACKEND=local, a presigned URL under
    # =minio. The extractor only surfaces it as a visual when it is URL-shaped.
    screenshot_path: str | None = None
    # Explicit capture outcome (app_inspector_service PR #92). The producer bug
    # it fixes: a FAILED capture still returned a non-empty ``screenshot_path``
    # pointing at a file that was never written, so ``_url_shaped`` said yes and
    # `app_review_record` shipped an all-text-card video on a clean green run —
    # its "no screenshot_url" note cannot fire when the url is a well-formed lie.
    # Default ``"unknown"`` deliberately mirrors the server's: absence is
    # UNMEASURED, not success, so a pre-#92 server (which omits the field) keeps
    # the old behaviour instead of having every screenshot suppressed.
    screenshot_status: str = "unknown"
    has_forms: bool = False
    has_media: bool = False


class NavigationEdge(BaseModel):
    """Consumed subset of the server's ``NavigationEdge`` (a screen transition)."""

    model_config = ConfigDict(extra="ignore")

    from_screen: str = ""
    to_screen: str = ""
    action: str = "tap"


class InspectionResult(BaseModel):
    """Consumed subset of the server's ``InspectionResultResponse`` (``/result``)."""

    model_config = ConfigDict(extra="ignore")

    job_id: str = ""
    status: str = ""
    platform: str | None = None
    app_path: str | None = None
    screens: list[ScreenSummary] = Field(default_factory=list)
    navigation_graph: list[NavigationEdge] = Field(default_factory=list)
    screens_discovered: int = 0
    # None means NOT MEASURED — see accessibility_score above. A zero-screen
    # inspection used to arrive here as a flat 100.
    overall_accessibility_score: int | None = None
    # Inferred, heuristic app purpose. Server COMPUTES + persists it; it reaches
    # /result only after app_inspector_service PR #60 (before that it is always
    # None on the wire). Optional here so this client never hard-blocks on the
    # server deploy order.
    estimated_purpose: str | None = None
    # Both DERIVED server-side from ``screens`` (app_inspector_service PR #92) —
    # a producer cannot overstate them, which is what makes them worth gating on.
    # ``artifacts_degraded`` is true when a screen reports a failed capture, or
    # screens were discovered and NONE carries a screenshot: the difference
    # between "complete in shape" and "actually carries artifacts", which a
    # well-formed but empty structure otherwise hides.
    screenshots_captured: int = 0
    artifacts_degraded: bool = False
    error: str | None = None
