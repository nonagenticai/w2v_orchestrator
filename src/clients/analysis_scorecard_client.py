"""One HTTP client for every analysis_service scorecard surface (port 6303).

``POST /api/cro/score``, ``/api/seo/audit`` and ``/api/marketing/scorecard`` are
all **synchronous** deterministic scorers (no job to poll) that take
``{url, include_llm, …}`` and return a score payload directly. They differ only
in the discovery ``service_name``, the endpoint path, and the response shape, so
one client serves all three — collapsing the former ``CroAuditClient`` /
``SeoAuditClient`` / ``MarketingScorecardClient`` clones — via one typed method
per surface. The SEO surface additionally accepts the opt-in ``competitor_urls`` /
``cross_page_linking`` / ``max_pages`` inputs; those keys are added to the payload
only when set, so the default request to every surface stays byte-identical.

The opt-in "great report" enrichment (analysis_service PR1-3 — priority/rank,
score projection, evidence, recommendation prose, claim-grounding, exec-summary,
report-critic) is forwarded via :class:`ScorecardEnrichment`, whose ``as_payload``
emits a key only when its flag is set. Scan stages build it with
:meth:`ScorecardEnrichment.from_brief`, which always sets the deterministic flags
(priority/projection/evidence/unbucketed) and adds the WM-backed ones only with
``include_llm`` (design 2026-10-17 §2.0).

Responses validate to the typed mirrors in ``src.clients.schemas.analysis_scorecard``
(server file:line citations live there).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.clients.base import LicensedServiceClient
from src.clients.schemas.analysis_scorecard import (
    CroScoreResponse,
    MarketingScorecardResponse,
    SeoAuditResponse,
)

#: httpx timeout for every scorecard POST. It must not undercut the longest
#: per-attempt ``stage_call`` budget the scan stages give it (150s, the AI /
#: advanced path) — with the 30s ``BaseServiceClient`` default an AI-analysis
#: scorecard (~60s live) hit ReadTimeout before the stage budget ever applied.
SCORECARD_HTTP_TIMEOUT_SECONDS = 150.0


@dataclass(frozen=True, slots=True)
class ScorecardEnrichment:
    """Opt-in "great report" flags forwarded to analysis_service.

    All flags default off → :meth:`as_payload` returns ``{}``. :meth:`from_brief`
    always sets the deterministic flags (priority/projection/evidence/unbucketed),
    which populate their response blocks unconditionally; the WM-backed flags (recommendations/grounding/exec_summary/critic)
    fail-soft to titles-only when their workers are unregistered or unavailable.
    """

    include_priority: bool = False
    include_projection: bool = False
    include_evidence: bool = False
    include_unbucketed: bool = False
    include_recommendations: bool = False
    include_grounding: bool = False
    include_exec_summary: bool = False
    include_critic: bool = False
    intent: str | None = None

    @classmethod
    def from_brief(cls, *, include_llm: bool, intent: str | None = None) -> ScorecardEnrichment:
        """The deterministic enrichment is ALWAYS on; ``include_llm`` adds only AI layers.

        Design 2026-10-17 §2.0 / F13: priority/rank, projection, evidence and
        ``include_unbucketed`` are deterministic, so they no longer ride
        ``include_llm`` — the AI toggle then changes only AI content (recommendation
        prose, grounding, exec summary, critic), which makes on/off honest.
        """
        deterministic = cls(
            include_priority=True,
            include_projection=True,
            include_evidence=True,
            include_unbucketed=True,
        )
        if not include_llm:
            return deterministic
        return cls(
            include_priority=True,
            include_projection=True,
            include_evidence=True,
            include_unbucketed=True,
            include_recommendations=True,
            include_grounding=True,
            include_exec_summary=True,
            include_critic=True,
            intent=intent,
        )

    def as_payload(self) -> dict[str, object]:
        """Return only the set keys (mirrors the SEO "add-only-when-set" pattern)."""
        payload: dict[str, object] = {}
        if self.include_priority:
            payload["include_priority"] = True
        if self.include_projection:
            payload["include_projection"] = True
        if self.include_evidence:
            payload["include_evidence"] = True
        if self.include_unbucketed:
            payload["include_unbucketed"] = True
        if self.include_recommendations:
            payload["include_recommendations"] = True
        if self.include_grounding:
            payload["include_grounding"] = True
        if self.include_exec_summary:
            payload["include_exec_summary"] = True
        if self.include_critic:
            payload["include_critic"] = True
        if self.intent:
            payload["intent"] = self.intent
        return payload


class AnalysisScorecardClient(LicensedServiceClient):
    """Client for the synchronous analysis_service scorecard endpoints.

    analysis_service is behind license-authz (only ``/health*`` is exempt), so the
    orchestrator's service-account licence token rides ``X-Tenant-Id: Bearer
    <token>`` (baked by ``BaseServiceClient``). ``service_name`` is the discovery
    key (``cro`` / ``seo`` / ``marketing``, all resolving via ``analysis``); the
    concrete POST path lives on each per-surface method.
    """

    def __init__(
        self,
        *,
        service_name: str,
        base_url: str | None = None,
        license_token: str | None = None,
    ) -> None:
        super().__init__(
            service_name=service_name,
            discovery_name="analysis",
            base_url=base_url,
            license_token=license_token,
            timeout=SCORECARD_HTTP_TIMEOUT_SECONDS,
        )

    async def score_cro(
        self,
        url: str,
        include_llm: bool = False,
        *,
        enrichment: ScorecardEnrichment | None = None,
        sources: list[str] | None = None,
    ) -> CroScoreResponse:
        """Score ``url`` for conversion-rate optimization (``POST /api/cro/score``)."""
        payload: dict[str, object] = {"url": url, "include_llm": include_llm}
        if enrichment:
            payload.update(enrichment.as_payload())
        if sources:
            payload["sources"] = list(sources)
        return CroScoreResponse.model_validate(
            await self._request("POST", "/api/cro/score", json=payload)
        )

    async def score_seo(
        self,
        url: str,
        include_llm: bool = False,
        *,
        competitor_urls: list[str] | None = None,
        cross_page_linking: bool = False,
        max_pages: int | None = None,
        enrichment: ScorecardEnrichment | None = None,
        sources: list[str] | None = None,
    ) -> SeoAuditResponse:
        """Audit ``url`` for search discoverability (``POST /api/seo/audit``).

        The three opt-in inputs drive analysis_service's SEO advanced analyzers;
        each is added to the payload only when set so the default-path request
        stays byte-identical to the cro/marketing surfaces.
        """
        payload: dict[str, object] = {"url": url, "include_llm": include_llm}
        if competitor_urls:
            payload["competitor_urls"] = list(competitor_urls)
        if cross_page_linking:
            payload["cross_page_linking"] = True
        if max_pages is not None:
            payload["max_pages"] = int(max_pages)
        if enrichment:
            payload.update(enrichment.as_payload())
        if sources:
            payload["sources"] = list(sources)
        return SeoAuditResponse.model_validate(
            await self._request("POST", "/api/seo/audit", json=payload)
        )

    async def score_marketing(
        self,
        url: str,
        include_llm: bool = False,
        *,
        enrichment: ScorecardEnrichment | None = None,
        sources: list[str] | None = None,
    ) -> MarketingScorecardResponse:
        """Score ``url`` as a go-to-market asset (``POST /api/marketing/scorecard``)."""
        payload: dict[str, object] = {"url": url, "include_llm": include_llm}
        if enrichment:
            payload.update(enrichment.as_payload())
        if sources:
            payload["sources"] = list(sources)
        return MarketingScorecardResponse.model_validate(
            await self._request("POST", "/api/marketing/scorecard", json=payload)
        )
