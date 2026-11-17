"""Per-type audit report templates — distinctness, dispatch, missing data, AI status,
and the design §6 acceptance checks (audit-templates-design.md).

Rendered from REAL analysis_service fixtures (nonagentic.ai) pushed through the
scan stage, exactly as the PDF route sees them (see test_audit_report_golden.py).
"""

from __future__ import annotations

import copy
import re
from typing import Any

import pytest

from src.services.blueprints.audit_family import AUDIT_PROJECT_TYPES
from src.services.stages.shared.audit_report import (
    RENDERERS,
    UnknownAuditTypeError,
    build_report_html,
)
from src.services.stages.shared.audit_report.primitives import TYPE_ACCENTS, host_of
from src.services.stages.shared.design_system import DEFAULT_TOKENS, as_params
from tests.services.stages.shared.test_audit_report_golden import TYPES, load_fixture

_H2 = re.compile(r"<h2>(.*?)</h2>")
_SEO, _CRO, _MKT = TYPES


def _h2s(html: str) -> set[str]:
    return set(_H2.findall(html))


def _render(project_type: str, variant: str) -> str:
    return build_report_html(project_type, load_fixture(project_type, variant))


def _section(html: str, key: str) -> str:
    return html.split(f'data-section="{key}"')[1].split("</section>")[0]


# --- dispatch ---------------------------------------------------------------


def test_every_audit_project_type_has_a_renderer() -> None:
    assert set(RENDERERS) == set(AUDIT_PROJECT_TYPES) == set(TYPES)


def test_unknown_type_raises_no_default() -> None:
    with pytest.raises(UnknownAuditTypeError):
        build_report_html("unknown", {"overall_score": 50})
    with pytest.raises(UnknownAuditTypeError):
        build_report_html("", {})


def test_dispatch_ignores_payload_shape() -> None:
    # A CRO payload rendered as seo_audit is an SEO report: the type comes from
    # project_type, never from which breakdown key is present.
    html = build_report_html(_SEO, load_fixture(_CRO, "ai_off"))
    assert "Search &amp; AI discoverability audit" in html
    assert "Conversion audit" not in html


# --- distinctness (§5.3) ----------------------------------------------------


@pytest.mark.parametrize("variant", ["ai_off"])
def test_renders_share_no_h2_except_methodology(variant: str) -> None:
    seo, cro, mkt = (_h2s(_render(t, variant)) for t in TYPES)
    assert seo & cro == {"Methodology"}
    assert seo & mkt == {"Methodology"}
    assert cro & mkt == {"Methodology"}


@pytest.mark.parametrize("variant", ["ai_off"])
def test_signature_elements(variant: str) -> None:
    seo, cro, mkt = (_render(t, variant) for t in TYPES)
    assert 'data-section="serp-preview"' in seo
    assert '<svg data-chart="funnel"' in cro
    assert '<svg data-chart="radar"' in mkt
    assert 'data-section="serp-preview"' not in cro + mkt
    assert 'data-chart="funnel"' not in seo + mkt
    assert 'data-chart="radar"' not in seo + cro


# --- §6 acceptance: page-1 identity -----------------------------------------


def test_titles_and_accents_differ_per_type() -> None:
    seo, cro, mkt = (_render(t, "ai_off") for t in TYPES)
    assert "<h1>Search &amp; AI discoverability audit</h1>" in seo
    assert "<h1>Conversion audit</h1>" in cro
    assert "<h1>Go-to-market scorecard</h1>" in mkt
    assert len(set(TYPE_ACCENTS.values())) == 3
    for html, t in zip((seo, cro, mkt), TYPES, strict=True):
        assert f"--type-accent:{TYPE_ACCENTS[t]};" in html


def test_seo_page1_indexable_badge_serp_preview_and_status_strip() -> None:
    html = _render(_SEO, "ai_off")
    assert ">INDEXABLE<" in html
    assert "checked: meta_robots, canonical, robots_txt" in html  # facts.indexable.checked
    preview = _section(html, "serp-preview")
    assert "Search result preview" in preview and "nonagentic.ai" in preview
    strip = _section(html, "indexability")
    for chip in ("✓ robots.txt", "✓ sitemap", "✓ canonical", "✓ HTTPS", "✓ viewport"):
        assert chip in strip


def test_cro_biggest_leak_journey_and_inventories() -> None:
    html = _render(_CRO, "ai_off")
    # facts.journey.weakest_stage == "trust", weakest section social 76.4
    assert "Biggest leak: Trust — social 76/100" in html
    assert "not measurable from a page fetch (no layout)" in html  # has_cta_above_fold=null
    cta = _section(html, "cta-inventory")
    assert "15 CTAs · 0 generic (0%)" in cta and "Browse the projects" in cta
    assert "Your message" in _section(html, "form-friction")
    assert "Customer logos" in _section(html, "trust")
    exps = _section(html, "experiments")
    assert "Fewer competing navigation links concentrate attention on the conversion goal." in exps


def test_marketing_weights_table_and_reference_block() -> None:
    html = _render(_MKT, "ai_off")
    one_pager = _section(html, "one-pager")
    assert 'class="weights-onepager"' in one_pager
    assert "22.2" in one_pager  # facts.dimension_sources[content].weighted_points 22.25
    assert "CRO scanner" in one_pager and "Marketing scanner" in one_pager
    ref = _section(html, "reference")
    assert "Run a CRO audit for the breakdown" in ref
    assert "Run an SEO audit for the breakdown" in ref
    # NotMeasuredFact reasons come from analysis_service, not static copy.
    assert "Not measured — No competitor pages analysed — add competitor URLs" in html


@pytest.mark.parametrize("variant", ["ai_off"])
def test_findings_are_scoped_to_their_own_engine(variant: str) -> None:
    seo, cro, mkt = (_render(t, variant) for t in TYPES)
    # The marketing buckets carry borrowed cro-/seo- refs; the scorecard lists none.
    assert "22 competing nav links" not in mkt
    assert "Primary keyword missing from the title tag" not in mkt
    assert "No comparison or alternatives content" in mkt
    assert "22 competing nav links" in cro
    assert "Primary keyword missing from the title tag" in seo


# --- §6 acceptance: AI on vs off --------------------------------------------


@pytest.mark.parametrize("project_type", list(TYPES)[:1])
def test_ai_status_line_per_variant(project_type: str) -> None:
    assert "AI analysis: not requested" in _render(project_type, "ai_off")
    assert "AI analysis: complete" in _render(project_type, "ai_ok")
    failed = _render(project_type, "ai_failed")
    line = failed.split('class="ai-status')[1].split("</p>")[0]
    assert "AI analysis unavailable:" in line and "WM 401 Unauthorized" in line


@pytest.mark.parametrize("project_type", list(TYPES)[:1])
@pytest.mark.parametrize("variant", ["ai_off"])
def test_ai_status_line_is_before_first_section(project_type: str, variant: str) -> None:
    html = _render(project_type, variant)
    assert html.index('class="ai-status') < html.index("<section")


@pytest.mark.parametrize("project_type", list(TYPES)[:1])
def test_ai_off_has_no_ai_content_but_keeps_deterministic_rank(project_type: str) -> None:
    html = _render(project_type, "ai_off")
    assert 'data-section="ai-analysis"' not in html
    assert 'class="rec"' not in html
    assert "grounded" not in html
    assert '<span class="rank">#1</span>' in html  # deterministic rank survives AI off


@pytest.mark.parametrize("project_type", [_SEO])
def test_ai_off_keeps_deterministic_projection(project_type: str) -> None:
    assert 'data-section="projection"' in _render(project_type, "ai_off")


def test_ai_ok_has_type_specific_ai_content() -> None:
    seo = _render(_SEO, "ai_ok")
    assert 'data-card="search_intent"' in seo and "commercial" in seo
    assert 'data-card="eeat_depth"' in seo and "names no author or credentials" in seo
    gap = seo.split('data-card="content_gap"')[1].split("data-card=")[0]
    assert "Not run for this scan — add competitor URLs" in gap
    cro = _render(_CRO, "ai_ok")
    assert 'data-card="judge-headline"' in cro and "Lead with the outcome the visitor gets." in cro
    mkt = _render(_MKT, "ai_ok")
    assert 'data-card="assess-headline"' in mkt
    assert "The page scores below target" in _section(mkt, "one-pager")  # narrative filled
    for html in (seo, cro, mkt):
        assert 'data-card="report-quality"' in html
    for html in (seo, cro):  # the marketing fixture has no prose on its own mkt-* rows
        assert 'class="rec"' in html
        assert '<span class="ground ok">grounded</span>' in html


@pytest.mark.parametrize("project_type", list(TYPES)[:1])
def test_ai_failed_names_reason_in_ai_section(project_type: str) -> None:
    html = _render(project_type, "ai_failed")
    assert "AI analysis unavailable: WM 401 Unauthorized" in _section(html, "ai-analysis")
    assert 'class="rec"' not in html


def test_partial_status_counts_unavailable_layers() -> None:
    scan = load_fixture(_CRO, "ai_ok")
    layers = scan["ai"]["layers"]
    critic = next(i for i, layer in enumerate(layers) if layer["layer"] == "critic")
    layers[critic] = {
        **layers[critic],
        "status": "failed",
        "reason": "WM 500 Internal Server Error",
    }
    scan["ai"]["status"] = "partial"
    scan.pop("critique")
    html = build_report_html(_CRO, scan)
    assert (
        f"AI analysis: partial — 1 of {len(layers)} layers unavailable "
        "(WM 500 Internal Server Error (critic))"
    ) in html
    assert "AI analysis unavailable: WM 500 Internal Server Error" in html  # report-quality card


def test_missing_ai_block_is_never_reported_complete() -> None:
    scan = load_fixture(_SEO, "ai_ok")
    scan.pop("ai")
    html = build_report_html(_SEO, scan)
    assert "AI analysis: not reported" in html
    assert "AI analysis: complete" not in html
    assert "AI analysis: not requested" not in html


# --- missing-data states (§5.3) ---------------------------------------------


def test_seo_unmeasured_group_renders_note_and_no_bar() -> None:
    scan = load_fixture(_SEO, "ai_off")
    for g in scan["groups"]:
        if g["group"] == "internal_linking":
            g.update(score=None, measured=False, note="Cross-page crawl failed: timeout after 20s")
    html = build_report_html(_SEO, scan)
    verdict = _section(html, "verdict")
    assert "Not measured — Cross-page crawl failed: timeout after 20s" in verdict
    assert not re.search(r'internal_linking</span><span class="track">', verdict)
    assert "Not measured — Cross-page crawl failed: timeout after 20s" in _section(html, "links")


def test_seo_performance_and_crawl_not_measured_with_server_reason() -> None:
    html = _render(_SEO, "ai_off")
    perf = _section(html, "performance")
    assert "Not measured — performance: not measured — no Lighthouse/Core Web Vitals" in perf
    assert "Not measured — site-wide linking not measured (cross-page crawl off)" in _section(
        html, "links"
    )


def test_seo_robots_not_fetched_renders_not_measured() -> None:
    scan = load_fixture(_SEO, "ai_off")
    scan["facts"]["robots"] = {"measured": False, "reason": "robots.txt unreachable"}
    html = build_report_html(_SEO, scan)
    strip = _section(html, "indexability")
    assert "— robots.txt" in strip and "Not measured — robots.txt unreachable" in strip


def test_cro_no_form_renders_not_measured() -> None:
    scan = load_fixture(_CRO, "ai_off")
    scan["facts"]["form"] = {"present": False}
    html = build_report_html(_CRO, scan)
    assert "Not measured — No conversion form on this page" in html


def test_cro_unmeasured_section_is_hatched_not_scored() -> None:
    scan = load_fixture(_CRO, "ai_off")
    for st in scan["facts"]["journey"]["stages"]:
        if st["stage"] == "commitment":
            st.update(score=None, measured=False, section_scores={"form": None})
    html = build_report_html(_CRO, scan)
    assert "Commitment · not measured" in _section(html, "journey")


def test_marketing_sibling_pointer_static_and_linked() -> None:
    scan = load_fixture(_MKT, "ai_off")
    scan["related_reports"] = {"cro_audit": {"completed_at": "2026-10-01"}}
    html = build_report_html(_MKT, scan)
    assert "Details: CRO audit of 2026-10-01" in html


@pytest.mark.parametrize("project_type", list(list(AUDIT_PROJECT_TYPES))[:1])
def test_empty_scan_renders_every_section_not_measured(project_type: str) -> None:
    html = build_report_html(project_type, {})
    assert html.startswith("<!doctype html>")
    assert "AI analysis: not reported" in html
    for body in html.split("<section")[1:]:
        assert "Not measured — " in body or "AI analysis" in body, body[:200]


# The pre-redesign persisted scan shapes (no project_type / facts / checks /
# findings / ai) — what every run stored before this branch.
_LEGACY: dict[str, dict[str, Any]] = {
    "seo_audit": {
        "url": "https://acme.test",
        "overall_score": 93.4,
        "grade": "A",
        "groups": [{"group": "on_page", "score": 90.0, "weight": 0.2, "count": 3}],
        "quick_wins": [
            {
                "check_id": "seo-og",
                "group": "on_page",
                "title": "Add OpenGraph tags",
                "severity": "medium",
            }
        ],
    },
    "cro_audit": {
        "url": "https://acme.test",
        "overall_score": 62.0,
        "grade": "C",
        "subsections": [{"section": "hero", "score": 80.0, "weight": 0.2, "count": 1}],
        "quick_wins": [
            {
                "check_id": "cro-x",
                "section": "hero",
                "title": "Move CTA above the fold",
                "severity": "medium",
            }
        ],
        "exec_summary": {"summary": "Legacy summary.", "roadmap": ["cro-x"]},
    },
    "marketing_scorecard": {
        "url": "https://acme.test",
        "overall_score": 74.0,
        "grade": "B",
        "dimensions": [{"dimension": "growth", "score": 68.0, "weight": 0.2, "count": 2}],
        "quick_wins": [
            {
                "check_id": "mkt-growth-no-analytics",
                "dimension": "growth",
                "title": "Add analytics tracking",
                "severity": "high",
            }
        ],
    },
}


@pytest.mark.parametrize("project_type", list(list(_LEGACY))[:1])
def test_legacy_scan_output_still_renders(project_type: str) -> None:
    html = build_report_html(project_type, copy.deepcopy(_LEGACY[project_type]))
    assert _LEGACY[project_type]["quick_wins"][0]["title"] in html
    assert "Not measured — passing checks — the check inventory is not in this scan" in html
    assert "AI analysis: not reported" in html


def test_legacy_cro_with_exec_summary_shows_it() -> None:
    html = build_report_html(_CRO, copy.deepcopy(_LEGACY["cro_audit"]))
    assert "Legacy summary." in html
    assert "<li>Move CTA above the fold</li>" in html  # id-locked roadmap resolved


@pytest.mark.parametrize("project_type", ["seo_audit"])
def test_finding_without_bucket_is_rendered_not_dropped(project_type: str) -> None:
    scan = {
        "overall_score": 50,
        "quick_wins": [{"check_id": "x-1", "title": "Orphan finding", "severity": "low"}],
    }
    html = build_report_html(project_type, scan)
    assert "<h3>Other findings</h3>" in html
    assert "Orphan finding" in html


# --- rendering hygiene ------------------------------------------------------


def test_html_escapes_untrusted_fields() -> None:
    html = build_report_html(
        _SEO,
        {
            "url": "https://acme.test/?a=<script>",
            "overall_score": 50,
            "grade": "C",
            "groups": [{"group": "<b>x</b>", "score": 50}],
            "quick_wins": [{"title": "<img src=x>", "severity": "low"}],
        },
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<img src=x>" not in html


def test_roadmap_drops_unknown_ids_and_dedupes() -> None:
    scan = load_fixture(_CRO, "ai_ok")
    scan["exec_summary"] = {
        "summary": "S.",
        "roadmap": ["cro-hero-nav-distraction", "cro-hero-nav-distraction", "alien.id"],
    }
    html = build_report_html(_CRO, scan)
    assert html.count("<li>22 competing nav links</li>") == 1
    assert "alien.id" not in html


def test_unverified_badge_shows_even_without_prose() -> None:
    scan = load_fixture(_SEO, "ai_ok")
    scan["grounding"] = [{"check_id": "seo-eeat-no-author", "supported": False, "evidence_ref": ""}]
    html = build_report_html(_SEO, scan)
    assert '<span class="ground warn">unverified</span>' in html


def test_bar_tone_keys_off_rounded_score() -> None:
    html = build_report_html(
        _SEO,
        {"overall_score": 80, "groups": [{"group": "technical", "score": 79.6, "weight": 0.18}]},
    )
    assert "width:80%;background:#16a34a" in html


def test_pagination_rules_present() -> None:
    html = _render(_SEO, "ai_off")
    assert "@page { size: A4; margin: 16mm; }" in html
    assert "section.major { break-before: page; }" in html


def test_explicit_tokens_override_default_brand() -> None:
    override = {**as_params(DEFAULT_TOKENS), "surface": "#123456"}
    html = build_report_html(_CRO, load_fixture(_CRO, "ai_off"), tokens=override)
    assert "--surface:#123456;" in html


@pytest.mark.parametrize(
    ("url", "host"),
    [("https://nonagentic.ai", "nonagentic.ai")],
)
def test_host_of_is_filename_safe(url: str, host: str) -> None:
    assert host_of(url) == host
