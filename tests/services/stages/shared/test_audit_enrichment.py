from __future__ import annotations

from src.services.stages.shared.audit_enrichment import (
    map_grounding,
    map_recommendation,
    resolve_intent,
)


def test_resolve_intent_reads_brief_then_direction() -> None:
    assert resolve_intent({}, {"intent": "  B2B devs  "}) == "B2B devs"
    assert resolve_intent({"direction": {"intent": "buyers"}}, {}) == "buyers"
    # brief wins over direction
    assert resolve_intent({"direction": {"intent": "buyers"}}, {"intent": "devs"}) == "devs"


def test_resolve_intent_ignores_non_string_values() -> None:
    # A non-string intent must be ignored, not coerced into a garbage repr string.
    assert resolve_intent({}, {"intent": ["a", "b"]}) is None
    assert resolve_intent({}, {"intent": 5}) is None
    assert resolve_intent({"direction": {"intent": 9}}, {}) is None
    assert resolve_intent({"direction": "not-a-dict"}, {}) is None


def test_resolve_intent_empty_is_none() -> None:
    assert resolve_intent({}, {}) is None
    assert resolve_intent({}, {"intent": "   "}) is None


def test_map_helpers_pass_none_through() -> None:
    # The mappers are the byte-identity seam: None in → None out.
    assert map_recommendation(None) is None
    assert map_grounding(None) is None
