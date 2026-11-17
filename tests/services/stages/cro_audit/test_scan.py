from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.clients.schemas.analysis_scorecard import CroScoreResponse
from src.services.runs.run_context import RunContext
from src.services.stages.cro_audit.scan import CroAuditScanStage

_RESULT = {
    "overall_score": 82.0,
    "grade": "A",
    "llm_enriched": False,
    "subsection_scores": {"hero": 90.0, "form": 70.0},
    "subsections": [
        {"section": "hero", "score": 90.0, "weight": 0.18, "count": 1, "top_severity": "low"},
        {"section": "form", "score": 70.0, "weight": 0.20, "count": 2, "top_severity": "medium"},
    ],
    "quick_wins": [
        {
            "check_id": "cro-hero-no-trust-signal",
            "section": "hero",
            "title": "No trust signal",
            "severity": "medium",
        }
    ],
    "strategic": [],
    "llm_judgments": [],
    "source": "https://acme.test",
    "scored_at": "2026-07-05T00:00:00Z",
}


def _ctx(target: str | None = "https://acme.test", include_llm: bool = False) -> RunContext:
    brief: dict = {}
    if target is not None:
        brief["target_url"] = target
    if include_llm:
        brief["include_llm"] = True
    return RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={"brief": brief})


@pytest.mark.asyncio
async def test_scan_emits_score_and_subsections() -> None:
    fake = AsyncMock()
    fake.score_cro.return_value = CroScoreResponse.model_validate(_RESULT)
    stage = CroAuditScanStage(client=fake)

    out = await stage.run({}, ctx=_ctx())

    assert out.overall_score == 82.0
    assert out.grade == "A"
    assert out.url == "https://acme.test"
    assert out.subsection_scores["form"] == 70.0
    assert len(out.subsections) == 2
    assert out.subsections[0].section == "hero"
    assert out.quick_wins[0].check_id == "cro-hero-no-trust-signal"
    fake.score_cro.assert_awaited_once()
    assert fake.score_cro.await_args.kwargs["include_llm"] is False


@pytest.mark.asyncio
async def test_scan_forwards_include_llm_and_judgments() -> None:
    fake = AsyncMock()
    fake.score_cro.return_value = CroScoreResponse.model_validate(
        {
            **_RESULT,
            "llm_enriched": True,
            "llm_judgments": [
                {"element": "headline", "score": 80.0, "assessment": "ok", "suggestion": "tighten"}
            ],
        }
    )
    stage = CroAuditScanStage(client=fake)

    out = await stage.run({}, ctx=_ctx(include_llm=True))

    assert out.llm_enriched is True
    assert len(out.llm_judgments) == 1
    assert out.llm_judgments[0].element == "headline"
    assert fake.score_cro.await_args.kwargs["include_llm"] is True
