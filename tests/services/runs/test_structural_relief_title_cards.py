"""Structural relief compares a carded film against the ORDER, not against the body.

reconcile_timeline reserves title cards INSIDE the order, so the manifest's
``run.target_seconds`` is the BODY (order minus ``run.card_seconds``) while the delivered
film carries the cards. Compared raw, a carded run delivering exactly its order read as
over by the card seconds — and with relief armed that is a PAID child render.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.services.runs.structural_relief_policy import decide_relief_shape
from src.services.stages.shared.time_budget import (
    Allotment,
    TimeDemand,
    TimelineManifest,
    build_timeline_manifest,
)

_ORDER = 50.0
_CARDS = 6.0
_BODY = _ORDER - _CARDS  # 44.0
_PER_BEAT = 10.632  # 5 pinned footage beats = 53.16s of body footage, +9.16s over
_BEATS = 5


def _run(card_seconds: float | None) -> dict[str, Any]:
    run: dict[str, Any] = {
        "target_seconds": _BODY,
        "overrun_seconds": _BEATS * _PER_BEAT - _BODY,
    }
    if card_seconds is not None:
        run["card_seconds"] = card_seconds
    return run


def _beats() -> list[dict[str, Any]]:
    return [
        {
            "id": f"clip-{i}",
            "seconds": _PER_BEAT,
            "min_seconds": _PER_BEAT,
            "max_seconds": _PER_BEAT,
        }
        for i in range(1, _BEATS + 1)
    ]


def _shape(delivered: float | None, card_seconds: float | None = _CARDS):
    return decide_relief_shape(
        run=_run(card_seconds),
        beats=_beats(),
        floor_ratio=0.5,
        delivered_seconds=delivered,
    )


def test_precondition_this_spine_spawns_when_delivery_is_unmeasured() -> None:
    assert _shape(None).verdict == "SPAWN"


def test_precondition_without_card_seconds_an_on_length_carded_film_spawns() -> None:
    """The defect: the body target vs the whole film reads +13.7% over and buys a child."""
    assert _shape(_ORDER + 0.016, card_seconds=None).verdict == "SPAWN"


@pytest.mark.parametrize(
    "delivered",
    [_ORDER + 0.016],
)
def test_a_carded_run_that_is_on_length_does_not_spawn_relief(delivered: float) -> None:
    shape = _shape(delivered)
    assert shape.verdict == "STOP_ALREADY_CONFORMING"
    assert shape.status_quo_seconds == pytest.approx(delivered)
    assert f"{_ORDER:.2f}s order" in (shape.stop_detail or "")


def test_card_seconds_ride_the_real_manifest_reconcile_builds() -> None:
    """The field the policy reads is the one ``build_timeline_manifest`` writes."""
    demand = TimeDemand(
        id="clip-1", kind="footage", min=10.0, ideal=10.0, max=10.0, elasticity=0.0, pinned=True
    )
    allot = Allotment(id="clip-1", seconds=10.0, source="default", vo_rate=None, infeasible=False)
    raw = build_timeline_manifest(
        [demand],
        [allot],
        target_seconds=_BODY,
        loop_iterations=0,
        converged=True,
        pacing_score=None,
        card_seconds=_CARDS,
    )
    manifest = TimelineManifest.model_validate(raw)
    assert manifest.run.card_seconds == pytest.approx(_CARDS)
    assert manifest.model_dump(mode="json")["run"]["card_seconds"] == pytest.approx(_CARDS)
