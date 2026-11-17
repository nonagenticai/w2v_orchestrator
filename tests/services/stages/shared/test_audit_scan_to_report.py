"""Real analysis_service fixture → ``<type>_scan`` stage → persisted output → report.

Proves lane E end to end on real wire data (design 2026-10-17 §3.3, §5.3): the
scan persists project_type / facts / checks / findings / ai / coverage, and an
AI-requested-but-unavailable response marks the stage degraded with
``audit_ai_analysis_unavailable`` instead of completing silently green.
"""

from __future__ import annotations

import pytest

from src.services.stages.shared.audit_report import build_report_html
from src.services.stages.shared.degradation import AUDIT_AI_ANALYSIS_UNAVAILABLE, is_degraded
from tests.services.stages.shared.test_audit_report_golden import (
    _PIPE,
    CASES,
    TYPES,
    load_wire,
    run_scan_stage,
)


@pytest.mark.parametrize(
    ("project_type", "variant"), list(CASES)[:1], ids=([f"{t}-{v}" for t, v in CASES])[:1]
)
async def test_real_fixture_through_scan_stage_then_renderer(
    project_type: str, variant: str
) -> None:
    wire = load_wire(project_type, variant)
    out = await run_scan_stage(project_type, wire, include_llm=variant != "ai_off")
    persisted = out.model_dump(mode="json")

    # The scan persists the audit-report contract 1:1.
    assert persisted["project_type"] == project_type
    assert len(persisted["findings"]) == len(wire["findings"]) > 0
    assert len(persisted["checks"]) == len(wire["checks"]) > 0
    assert persisted["facts"] == wire["facts"]
    assert persisted["ai"]["status"] == wire["ai"]["status"]
    assert persisted["scored_at"] == wire["scored_at"]
    if "coverage" in wire:
        assert persisted["coverage"] == wire["coverage"]

    # AI requested and unavailable → degraded with the coded entry; otherwise clean.
    codes = [d["code"] for d in persisted["degradations"]]
    if variant == "ai_failed":
        assert codes == [AUDIT_AI_ANALYSIS_UNAVAILABLE]
        assert "WM 401 Unauthorized" in persisted["degradations"][0]["detail"]
        assert is_degraded(persisted)
    else:
        assert codes == []
        assert not is_degraded(persisted)

    # ... and the persisted output renders through the type's own template.
    html = build_report_html(project_type, persisted)
    expected = {
        "ai_off": "AI analysis: not requested",
        "ai_ok": "AI analysis: complete",
        "ai_failed": "AI analysis unavailable:",
    }[variant]
    assert expected in html


@pytest.mark.parametrize("project_type", list(TYPES)[:1])
async def test_include_llm_with_no_ai_block_is_degraded(project_type: str) -> None:
    # An older analysis_service pod sends no ``ai`` block: never read as complete.
    wire = load_wire(project_type, "ai_ok")
    wire.pop("ai")
    out = await run_scan_stage(project_type, wire, include_llm=True)
    persisted = out.model_dump(mode="json")
    assert [d["code"] for d in persisted["degradations"]] == [AUDIT_AI_ANALYSIS_UNAVAILABLE]
    assert "did not report AI status" in persisted["degradations"][0]["detail"]
    assert "AI analysis: not reported" in build_report_html(project_type, persisted)


@pytest.mark.parametrize("project_type", list(TYPES)[:1])
async def test_partial_ai_is_degraded(project_type: str) -> None:
    wire = load_wire(project_type, "ai_ok")
    wire["ai"]["status"] = "partial"
    wire["ai"]["layers"][-1] = {**wire["ai"]["layers"][-1], "status": "failed", "reason": "WM 500"}
    out = await run_scan_stage(project_type, wire, include_llm=True)
    persisted = out.model_dump(mode="json")
    detail = persisted["degradations"][0]["detail"]
    assert persisted["degradations"][0]["code"] == AUDIT_AI_ANALYSIS_UNAVAILABLE
    assert detail.startswith("AI analysis partial — ") and "WM 500" in detail


# --- null scores: the parse failure origin/main hits on every new response ---


@pytest.mark.parametrize(
    ("project_type", "variant"), list(CASES)[:1], ids=([f"{t}-{v}" for t, v in CASES])[:1]
)
async def test_every_real_fixture_validates_through_mirror_and_stage_output(
    project_type: str, variant: str
) -> None:
    wire = load_wire(project_type, variant)
    model, stage_cls, _ = _PIPE[project_type]
    parsed = model.model_validate(wire)  # client mirror
    out = await run_scan_stage(project_type, wire, include_llm=variant != "ai_off")
    # The persisted JSON round-trips through the stage output model.
    reparsed = stage_cls.output_model.model_validate(out.model_dump(mode="json"))
    assert reparsed.overall_score == parsed.overall_score
    if project_type == "seo_audit":
        perf_wire = next(g for g in parsed.groups if g.group == "performance")
        perf_out = next(g for g in reparsed.groups if g.group == "performance")
        assert perf_wire.score is None and perf_wire.measured is False
        assert perf_out.score is None and perf_out.measured is False and perf_out.note
        assert reparsed.group_scores.get("performance", None) is None


async def test_cro_null_section_scores_validate_through_mirror_and_stage() -> None:
    # A page with no form / no footer: analysis_service sends null section scores.
    wire = load_wire("cro_audit", "ai_off")
    for sub in wire["subsections"]:
        if sub["section"] in {"form", "footer"}:
            sub.update(score=None, measured=False, note=f"no {sub['section']} on this page")
            wire["subsection_scores"][sub["section"]] = None
    out = await run_scan_stage("cro_audit", wire, include_llm=False)
    persisted = out.model_dump(mode="json")
    nulls = {s["section"] for s in persisted["subsections"] if s["score"] is None}
    assert nulls == {"form", "footer"}
    assert persisted["subsection_scores"]["form"] is None
    html = build_report_html("cro_audit", persisted)
    assert "Not measured — no form on this page" in html
