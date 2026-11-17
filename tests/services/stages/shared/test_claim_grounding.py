"""Tests for the deterministic claim-grounding lint (ORCH-0.7).

Pure — no DB, no client, no ctx. Mirrors ``test_retention.py``: everything runs
against fixed in-memory line/sentence lists, and identical inputs must yield
byte-identical outputs.
"""

from __future__ import annotations

from src.services.stages.shared.claim_grounding import (
    ClaimGroundingReport,
    analyze_claims,
    find_quantitative,
    find_superlatives,
    is_grounded,
    support_score,
)


def test_unsourced_superlative_is_flagged() -> None:
    # A superlative with NO supporting source sentence is flagged `unsourced`.
    lines = [{"id": "clip-1", "text": "We are the best CRM for teams.", "clip": None}]
    report = analyze_claims(lines, [])
    assert isinstance(report, ClaimGroundingReport)
    assert report.lines_checked == 1
    assert report.flagged, "the superlative should have produced a flagged row"
    row = report.flagged[0]
    assert row["claim_status"] == "unsourced"
    assert row["matched_term"] == "best"
    assert row["line_id"] == "clip-1"
    assert row["claim_source"] == ""
    # Deterministic.
    assert analyze_claims(lines, []) == report


def test_sourced_claim_is_grounded() -> None:
    # The same term echoed by a source sentence is `grounded`, with the snippet.
    lines = [{"id": "clip-1", "text": "It is the fastest onboarding around.", "clip": None}]
    sentences = ["Independent benchmarks rank it the fastest onboarding on the market."]
    report = analyze_claims(lines, sentences)
    row = report.flagged[0]
    assert row["claim_status"] == "grounded"
    assert row["matched_term"] == "fastest"
    assert row["claim_source"] == sentences[0]
    # A grounded claim is not an unsourced insight.
    assert report.insights == []


def test_grounded_and_unsourced_helpers() -> None:
    assert find_superlatives("the best and #1 tool") == ["#1", "best"]
    assert find_superlatives("a plain descriptive sentence") == []
    # Word-boundary: "only" must not trip inside "commonly".
    assert find_superlatives("commonly used") == []

    grounded, snippet = is_grounded("fastest", ["Rated the fastest in class."])
    assert grounded is True
    assert snippet == "Rated the fastest in class."
    assert is_grounded("fastest", ["a slow tool"]) == (False, "")
    assert is_grounded("", ["anything"]) == (False, "")


def test_empty_lines_or_sentences_yield_no_insights() -> None:
    # Empty lines → nothing checked, no insights.
    empty = analyze_claims([], [])
    assert empty.insights == []
    assert empty.flagged == []
    assert empty.lines_checked == 0

    # Superlative present but NO source material → flagged unsourced, but the lint
    # stays silent (no grounding to check against ⇒ byte-identical opt-out).
    lines = [{"id": "clip-1", "text": "the best", "clip": None}]
    no_sources = analyze_claims(lines, [])
    assert no_sources.insights == []
    assert no_sources.flagged[0]["claim_status"] == "unsourced"

    # Sentences present but no lines → nothing to check.
    assert analyze_claims([], ["the best product ever"]).insights == []


def test_unsourced_insight_surfaces_only_with_source_material() -> None:
    lines = [{"id": "clip-1", "text": "We are the cheapest option.", "clip": None}]
    sentences = ["Our pricing is competitive with the market."]
    report = analyze_claims(lines, sentences)
    # "cheapest" is not echoed by the source sentence → unsourced + surfaced.
    assert report.flagged[0]["claim_status"] == "unsourced"
    assert len(report.insights) == 1
    assert "cheapest" in report.insights[0]
    assert "clip-1" in report.insights[0]


def test_quantitative_claims_flagged() -> None:
    # The numbers a marketing script invents most readily — a percentage, a multiple,
    # a duration — are checkable claims the superlative lexicon cannot see.
    # De-duplicated and sorted, so the same text always yields the same order.
    assert find_quantitative("Ship 40% faster, 3x cheaper, in 30 seconds.") == [
        "30 seconds",
        "3x",
        "40%",
    ]
    assert find_quantitative("no numbers here") == []
    assert find_quantitative("") == []

    # They ride the same ClaimAdvisory rail as a superlative: unsourced when the
    # corpus does not echo the numeral, grounded when it does.
    lines = [{"id": "clip-1", "text": "Deploys land 40% faster.", "clip": None}]
    unsourced = analyze_claims(lines, ["Deploys land noticeably faster."])
    assert unsourced.lines_checked == 1
    assert [row["matched_term"] for row in unsourced.flagged] == ["40%"]
    assert unsourced.flagged[0]["claim_status"] == "unsourced"
    assert "40%" in unsourced.insights[0]

    sentences = ["Our benchmark shows deploys land 40% faster than the baseline."]
    grounded = analyze_claims(lines, sentences)
    assert grounded.flagged[0]["claim_status"] == "grounded"
    assert grounded.flagged[0]["claim_source"] == sentences[0]
    assert grounded.insights == []


def test_support_score_zero_without_sentences() -> None:
    # No corpus (or no term) ⇒ the corpus says nothing on the subject — never a guess.
    assert support_score("fastest", []) == (0.0, "")
    assert support_score("", ["Rated the fastest in class."]) == (0.0, "")

    # A substring hit always scores above zero and returns the closest sentence.
    score, closest = support_score("fastest onboarding", ["Rated the fastest onboarding here."])
    assert score > 0.0
    assert closest == "Rated the fastest onboarding here."

    # A corpus that is silent on the term scores it at zero.
    assert support_score("cheapest", ["Deploys land quickly."]) == (0.0, "")
