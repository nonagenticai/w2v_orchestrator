"""A18 phase 2 — the deterministic disclaimer rule engine + its compile-tail card.

Pure: the rule engine is a total function of the flagged claim rows, and the splice
is a total function of ``(shots, card)``. A thin ``SimpleNamespace`` ctx carries the
``config_snapshot`` + ``prior_output`` the card builder reads — no DB, no I/O.

Pins AC-8: a card IS present when the copy mandates a disclaimer; NO card when it
mandates none. And the inertness half of AC-1: ``splice_disclaimer_card(shots, None)``
returns the shot list unchanged.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

from src.services.stages.shared.claim_grounding import (
    DISCLAIMER_KINDS,
    analyze_claims,
    disclaimer_requirements,
)
from src.services.stages.shared.disclaimer_card import (
    DISCLAIMER_LINE_ID,
    DISCLAIMER_MAX_CHARS,
    build_disclaimer_shot,
    splice_disclaimer_card,
)

_SHOTS: list[dict[str, Any]] = [
    {"id": "clip-1", "line_id": "clip-1", "seconds": 4.0, "description": "open"},
    {"id": "clip-2", "line_id": "clip-2", "seconds": 3.0, "description": "close"},
]


def _ctx(snapshot: dict[str, Any], prior: dict[str, Any]) -> Any:
    return SimpleNamespace(
        config_snapshot=snapshot,
        prior_output=lambda stage: prior.get(stage),
    )


# ----------------------------------------------------- the rule engine (§4) --
def _flagged(text: str) -> list[dict[str, Any]]:
    """The real ``analyze_claims`` rows for one line (the engine's actual input)."""
    return analyze_claims([{"id": "clip-1", "text": text}], ["Some source sentence."]).flagged


def test_comparative_claim_mandates_a_comparison_disclaimer() -> None:
    reqs = disclaimer_requirements(_flagged("It is the fastest tracker, 3x faster than Jira."))
    kinds = {req.kind for req in reqs}
    assert "comparative" in kinds
    assert any(req.text == "Comparison based on publicly available information." for req in reqs)


def test_requirements_are_deduped_and_kind_ordered() -> None:
    flagged = [
        *_flagged("The fastest tool, 40% faster than the rest."),
        *_flagged("Only $9 per month — the best price."),
    ]
    reqs = disclaimer_requirements(flagged)
    kinds = [req.kind for req in reqs]
    assert kinds == sorted(kinds, key=DISCLAIMER_KINDS.index)
    assert len(kinds) == len(set(kinds)), "one requirement per class, deduped"


def test_no_flagged_claims_mandates_nothing() -> None:
    assert disclaimer_requirements([]) == []
    assert disclaimer_requirements(_flagged("Let's dive in.")) == []


# --------------------------------------------------- AC-8: the card is spliced --
_DISCLAIMERS = [
    {
        "kind": "comparative",
        "text": "Comparison based on publicly available information.",
        "line_ids": ["clip-1"],
    },
    {"kind": "performance", "text": "Results may vary.", "line_ids": ["clip-1"]},
]


def test_ac8_card_present_when_requirements_exist() -> None:
    ctx = _ctx({"legal": {"mode": "auto"}}, {"claim_finish": {"disclaimers": _DISCLAIMERS}})
    card = build_disclaimer_shot(ctx)
    assert card is not None
    assert card["line_id"] == DISCLAIMER_LINE_ID
    assert card["seconds"] == 4.0
    assert "Comparison based on" in card["description"]
    assert "Results may vary." in card["description"]
    assert len(card["description"]) <= DISCLAIMER_MAX_CHARS

    spliced = splice_disclaimer_card(copy.deepcopy(_SHOTS), card)
    # Tail-spliced: the card is LAST, and every prior shot is byte-unchanged.
    assert spliced[:-1] == _SHOTS
    assert spliced[-1] == card


def test_ac8_no_card_when_requirements_are_empty() -> None:
    ctx = _ctx({"legal": {"mode": "auto"}}, {"claim_finish": {"disclaimers": []}})
    assert build_disclaimer_shot(ctx) is None


def test_ac8_no_card_when_claim_finish_skipped() -> None:
    # A skipped stage stores `{}` — the case for every run that never opted in.
    ctx = _ctx({"legal": {"trademarks": ["Acme®"]}}, {"claim_finish": {}})
    assert build_disclaimer_shot(ctx) is None
    # …and when the stage isn't in the DAG at all.
    assert build_disclaimer_shot(_ctx({}, {})) is None


def test_disclaimers_mode_off_declines_the_card() -> None:
    ctx = _ctx(
        {"legal": {"mode": "auto", "disclaimers": {"mode": "off"}}},
        {"claim_finish": {"disclaimers": _DISCLAIMERS}},
    )
    assert build_disclaimer_shot(ctx) is None


def test_card_body_is_clamped() -> None:
    long = [{"kind": "comparative", "text": "x" * 400, "line_ids": []}]
    ctx = _ctx({"legal": {"mode": "auto"}}, {"claim_finish": {"disclaimers": long}})
    card = build_disclaimer_shot(ctx)
    assert card is not None
    assert len(card["description"]) == DISCLAIMER_MAX_CHARS


# ------------------------------------------- AC-1: a None card changes nothing --
def test_ac1_splice_with_none_card_is_byte_identical() -> None:
    shots = copy.deepcopy(_SHOTS)
    assert splice_disclaimer_card(shots, None) == _SHOTS
    assert splice_disclaimer_card(shots, None) is shots
