"""Typed pydantic response models for the downstream-service client seam.

Every ``src/clients/*_client.py`` method that used to return ``dict[str, Any]``
should return a model defined here instead. The models mirror the *wire format*
of the downstream service's response — not its internal Python attribute names.

Why mirrors and not a shared package: the services are independently deployed
git repos with no shared response-schema package. Drift is guarded by
``contract_scanner_service`` rather than by import coupling.

Convention:
- Fields the orchestrator consumes get real types.
- Opaque JSON blobs the orchestrator never reads are ``dict[str, object]``
  (validates any JSON object, but forces a consumer to narrow) — never
  ``dict[str, Any]``, which the ANN401/count ratchets exist to drive to zero.
- Models keep pydantic's default ``extra="ignore"`` so a downstream service
  ADDING a field can never break the orchestrator.
"""

from __future__ import annotations

from src.clients.schemas.ai_content_generation import (
    AcgConceptImageAsset,
    AcgConceptImageBatch,
    AcgConceptImageBatchSubmission,
    AcgConceptImageJob,
    AcgVideoGenerationResult,
    AcgVideoJobStatus,
)
from src.clients.schemas.analysis_scorecard import (
    CroFindingRef,
    CroLlmJudgment,
    CroScoreResponse,
    CroSubsectionSummary,
    MarketingDimensionSummary,
    MarketingFindingRef,
    MarketingLlmAssessment,
    MarketingScorecardResponse,
    SeoAuditFindingRef,
    SeoAuditGroupSummary,
    SeoAuditResponse,
)
from src.clients.schemas.bdd_playwright import (
    BddJobResult,
    BddJobState,
    BddJobStatus,
    BddNarrationPart,
    BddNarrationScript,
)
from src.clients.schemas.brand_extractor import (
    BrandExtraction,
)
from src.clients.schemas.critique import (
    CritiqueComparison,
    CritiqueFinding,
    CritiqueResult,
)
from src.clients.schemas.design import (
    DesignAccessibilityCategory,
    DesignAccessibilityFinding,
    DesignAccessibilityScore,
    DesignConceptImageAsset,
    DesignJobResult,
    DesignJobStatus,
    DesignVariantResult,
    DesignVisualAsset,
)
from src.clients.schemas.keycloak import (
    KeycloakUser,
)
from src.clients.schemas.licenses import (
    ActiveLicenseView,
    AppView,
    LicenseView,
    PlanView,
)
from src.clients.schemas.media_discovery import (
    MediaItem,
    MediaSearchResponse,
)
from src.clients.schemas.slide_service import (
    SlideDeckStatus,
)
from src.clients.schemas.transcription_worker import (
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
)
from src.clients.schemas.tts import (
    SynthesizeWire,
    TTSSynthesis,
)
from src.clients.schemas.tts_gateway import (
    BatchStatusResult,
    GatewayJobRef,
    GatewayJobStatus,
    SubmitBatchResult,
)
from src.clients.schemas.video_analyzer import (
    VideoAnalyzerJobResult,
    VideoAnalyzerJobStatus,
    VideoAnalyzerMotionSample,
    VideoAnalyzerQualityMetrics,
    VideoAnalyzerScreenshotMatch,
    VideoAnalyzerVerification,
    VideoAnalyzerVerificationFailure,
)

__all__ = [
    "AcgConceptImageAsset",
    "AcgConceptImageBatch",
    "AcgConceptImageBatchSubmission",
    "AcgConceptImageJob",
    "AcgVideoGenerationResult",
    "AcgVideoJobStatus",
    "ActiveLicenseView",
    "AppView",
    "BatchStatusResult",
    "BddJobResult",
    "BddJobState",
    "BddJobStatus",
    "BddNarrationPart",
    "BddNarrationScript",
    "BrandExtraction",
    "CritiqueComparison",
    "CritiqueFinding",
    "CritiqueResult",
    "CroFindingRef",
    "CroLlmJudgment",
    "CroScoreResponse",
    "CroSubsectionSummary",
    "DesignAccessibilityCategory",
    "DesignAccessibilityFinding",
    "DesignAccessibilityScore",
    "DesignConceptImageAsset",
    "DesignJobResult",
    "DesignJobStatus",
    "DesignVariantResult",
    "DesignVisualAsset",
    "GatewayJobRef",
    "GatewayJobStatus",
    "KeycloakUser",
    "LicenseView",
    "MarketingDimensionSummary",
    "MarketingFindingRef",
    "MarketingLlmAssessment",
    "MarketingScorecardResponse",
    "MediaItem",
    "MediaSearchResponse",
    "PlanView",
    "SeoAuditFindingRef",
    "SeoAuditGroupSummary",
    "SeoAuditResponse",
    "SlideDeckStatus",
    "SubmitBatchResult",
    "SynthesizeWire",
    "TTSSynthesis",
    "TranscriptSegment",
    "TranscriptWord",
    "TranscriptionResult",
    "VideoAnalyzerJobResult",
    "VideoAnalyzerJobStatus",
    "VideoAnalyzerMotionSample",
    "VideoAnalyzerQualityMetrics",
    "VideoAnalyzerScreenshotMatch",
    "VideoAnalyzerVerification",
    "VideoAnalyzerVerificationFailure",
]
