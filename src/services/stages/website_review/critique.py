"""website_critique — run analysis_service critique on the target URL.

Submits a critique job to analysis_service (port 6303) for ``brief.target_url``
and polls to completion, emitting the structured findings + scores. The
findings feed the redesign stage and the review script; ``critique_job_id`` is
surfaced so the frontend can link the downloadable critique report (PDF/JSON)
served by the existing W1 proxy routes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid7

from w2v_common.stages import stage_call

from src.clients.critique_client import CritiqueClient
from src.services.auth_secrets import auth_for_origin, unseal_auth
from src.services.stages.base import StageHandler, StageInput, StageOutput
from src.services.stages.shared.video_gate import require_critique

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

_START_TIMEOUT_SECONDS = 60.0
# poll_job hard-rejects a deadline > 600s (BaseServiceClient._clamp_timeout), so
# one poll waits up to ~590s; stage_call retries it once for a ~1180s ceiling.
_POLL_TIMEOUT_SECONDS = 590.0
_MAX_FINDINGS = 40


class WebsiteCritiqueInput(StageInput):
    pass


class WebsiteFinding(StageOutput):
    id: str
    category: str
    severity: str
    title: str
    description: str
    recommendation: str = ""


class WebsiteCritiqueOutput(StageOutput):
    critique_job_id: str
    url: str
    overall_score: float | None = None
    summary: str = ""
    category_scores: dict[str, float] = {}  # noqa: RUF012 — pydantic copies field defaults per-instance
    findings: list[WebsiteFinding] = []  # noqa: RUF012 — pydantic copies field defaults per-instance


class WebsiteCritiqueStage(StageHandler[WebsiteCritiqueInput, WebsiteCritiqueOutput]):
    name = "website_critique"
    input_model = WebsiteCritiqueInput
    output_model = WebsiteCritiqueOutput

    def __init__(self, client: CritiqueClient | None = None) -> None:
        self._client = client or CritiqueClient()

    async def handle(
        self,
        payload: WebsiteCritiqueInput,
        ctx: RunContext,
    ) -> WebsiteCritiqueOutput:
        require_critique(ctx)  # skip on a redesign-only / generate_critique=false run
        brief = ctx.config_snapshot.get("brief") or {}
        target = brief.get("target_url")
        if not target:
            raise RuntimeError("brief.target_url required for website_review")
        categories = brief.get("critique_categories") or None
        # auth_info travels as an opaque dict; analysis_service validates the
        # shape via its AuthInfo pydantic model and returns 422 on mismatch.
        # See analysis_service/src/domains/critique/schemas.py::AuthInfo.
        # Sealed at rest (auth_secrets); unsealed in memory for this one call.
        auth_info = auth_for_origin(unseal_auth(brief.get("auth_info")), str(target), brief)

        # Two explicit branches (rather than a dict-spread) so the named
        # parameters on start_critique stay narrowly typed for `ty` — and so
        # absent auth_info translates to literally no kwarg, preserving the
        # prior call shape for downstream mocks that assert on it.
        url_str = str(target)
        if auth_info:
            job_id = await stage_call(
                lambda: self._client.start_critique(
                    url=url_str, categories=categories, auth_info=auth_info
                ),
                timeout_seconds=_START_TIMEOUT_SECONDS,
                retries=0,  # non-idempotent submit: a retry would start a duplicate critique job
                stage_name="website_critique.start",
            )
        else:
            job_id = await stage_call(
                lambda: self._client.start_critique(url=url_str, categories=categories),
                timeout_seconds=_START_TIMEOUT_SECONDS,
                retries=0,  # non-idempotent submit: a retry would start a duplicate critique job
                stage_name="website_critique.start",
            )
        # retry_on=() keeps the single retry a poll-*timeout* retry only (stage_call's
        # TimeoutError branch ignores retry_on): a still-running critique keeps
        # waiting, a server-marked-failed job propagates the service's error text
        # immediately rather than re-polling the terminal job a second time.
        result = await stage_call(
            lambda: self._client.wait_for_critique(job_id, timeout=_POLL_TIMEOUT_SECONDS),
            timeout_seconds=_POLL_TIMEOUT_SECONDS + 10.0,
            retries=1,
            retry_on=(),
            stage_name="website_critique.wait",
        )

        findings = [
            WebsiteFinding(
                id=str(f.finding_id or uuid7()),
                category=str(f.category or "general"),
                severity=str(f.severity or "info"),
                title=str(f.title or ""),
                description=str(f.description or ""),
                recommendation=str(f.recommendation or ""),
            )
            for f in (result.findings or [])[:_MAX_FINDINGS]
        ]
        return WebsiteCritiqueOutput(
            critique_job_id=str(job_id),
            url=str(result.url or target),
            overall_score=result.overall_score,
            summary=str(result.summary or ""),
            category_scores=dict(result.category_scores or {}),
            findings=findings,
        )
