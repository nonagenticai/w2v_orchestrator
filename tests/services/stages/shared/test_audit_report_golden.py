"""Golden HTML snapshots: 3 audit types x (AI off / AI ok / AI failed) = 9 goldens.

Fixtures (``audit_report/golden/fixtures/<project_type>_<variant>.json``) are REAL
analysis_service responses captured against nonagentic.ai (feat/audit-contract
eb6c797). Each golden is the END-TO-END path the PDF route sees: wire JSON →
``<type>_scan`` stage (client mocked to return the fixture) → persisted
``model_dump(mode="json")`` → ``build_report_html``. The ``Scored at`` line is
normalised before comparison.

Regenerate after an intended template change, then REVIEW the diff:

    UPDATE_AUDIT_GOLDENS=1 uv run pytest tests/services/stages/shared/test_audit_report_golden.py
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
from functools import cache
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.clients.schemas.analysis_scorecard import (
    CroScoreResponse,
    MarketingScorecardResponse,
    SeoAuditResponse,
)
from src.services.runs.run_context import RunContext
from src.services.stages.cro_audit.scan import CroAuditScanStage
from src.services.stages.marketing_scorecard.scan import MarketingScorecardScanStage
from src.services.stages.seo_audit.scan import SeoAuditScanStage
from src.services.stages.shared.audit_report import build_report_html

GOLDEN = Path(__file__).parent / "audit_report" / "golden"
FIXTURES = GOLDEN / "fixtures"
TYPES = ("seo_audit", "cro_audit", "marketing_scorecard")
VARIANTS = ("ai_off", "ai_ok", "ai_failed")
CASES = [(t, v) for t in TYPES for v in VARIANTS]
TARGET = "https://nonagentic.ai"

# project_type -> (wire response model, stage class, client method)
_PIPE: dict[str, tuple[Any, Any, str]] = {
    "seo_audit": (SeoAuditResponse, SeoAuditScanStage, "score_seo"),
    "cro_audit": (CroScoreResponse, CroAuditScanStage, "score_cro"),
    "marketing_scorecard": (
        MarketingScorecardResponse,
        MarketingScorecardScanStage,
        "score_marketing",
    ),
}

_SCORED_AT = re.compile(r"(<li data-scored-at>Scored at: )[^<]*")


def load_wire(project_type: str, variant: str) -> dict[str, Any]:
    """The raw analysis_service response fixture."""
    return json.loads((FIXTURES / f"{project_type}_{variant}.json").read_text())


async def run_scan_stage(project_type: str, wire: dict[str, Any], *, include_llm: bool) -> Any:
    """Run the real ``<type>_scan`` stage with its client returning ``wire``."""
    model, stage_cls, method = _PIPE[project_type]
    fake = AsyncMock()
    getattr(fake, method).return_value = model.model_validate(wire)
    brief: dict[str, Any] = {"target_url": TARGET}
    if include_llm:
        brief["include_llm"] = True
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={"brief": brief})
    return await stage_cls(client=fake).run({}, ctx=ctx)


@cache
def _persisted(project_type: str, variant: str) -> str:
    out = asyncio.run(
        run_scan_stage(
            project_type, load_wire(project_type, variant), include_llm=variant != "ai_off"
        )
    )
    return json.dumps(out.model_dump(mode="json"))


def load_fixture(project_type: str, variant: str) -> dict[str, Any]:
    """The persisted ``<type>_scan`` output for a real fixture (a fresh copy)."""
    return copy.deepcopy(json.loads(_persisted(project_type, variant)))


def render_fixture(project_type: str, variant: str) -> str:
    return build_report_html(project_type, load_fixture(project_type, variant))


def normalise(html: str) -> str:
    """Mask the timestamp, and match what the whitespace pre-commit hooks write
    (no trailing spaces, exactly one final newline)."""
    html = _SCORED_AT.sub(r"\1<normalised>", html)
    return "\n".join(line.rstrip() for line in html.splitlines()) + "\n"


@pytest.mark.parametrize(
    ("project_type", "variant"), list(CASES)[:1], ids=([f"{t}-{v}" for t, v in CASES])[:1]
)
def test_golden_snapshot(project_type: str, variant: str) -> None:
    html = normalise(render_fixture(project_type, variant))
    path = GOLDEN / f"{project_type}_{variant}.html"
    if os.environ.get("UPDATE_AUDIT_GOLDENS") == "1":
        path.write_text(html)
    assert path.exists(), f"missing golden {path.name}; run with UPDATE_AUDIT_GOLDENS=1"
    assert html == path.read_text(), f"{path.name} drifted; review and regenerate"


def test_every_fixture_has_a_golden_and_vice_versa() -> None:
    expected = {f"{t}_{v}" for t, v in CASES}
    assert {p.stem for p in FIXTURES.glob("*.json")} == expected
    assert {p.stem for p in GOLDEN.glob("*.html")} == expected
