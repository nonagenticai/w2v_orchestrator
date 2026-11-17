"""An infeasibility detail may not attribute the dwell to a suite this run never had.

``_apply_feasibility_floor``'s INFEASIBLE-AT-ONE detail ended "...rather than trimming
the single scenario, whose recording dwell is owned by bdd". That clause is only true on
the bdd rung (``_BDD_GUARANTEED_YIELD``), where bdd exported
``narration_script.parts[].expected_recording_seconds`` for the suite it just generated.

``handle`` calls this function for BOTH spines. On the AUTHORED path there is no bdd
suite at all: the per-scenario figure is either the PARENT run's measurement carried on
``config_snapshot["structural_relief"]`` or nothing recorded. The detail nonetheless
named bdd as the owner — an attribution the run cannot support, in the channel that
records what was silently changed about a paid deliverable.

The sibling log line one block above was ALREADY conditional on the same rung
("bdd" vs "whoever sets the pacing"), so the two halves of one message disagreed.
"""

from __future__ import annotations

import re

import pytest

from src.services.stages.demo_video.gen import (
    _BDD_GUARANTEED_YIELD,
    _PARENT_MEASURED_YIELD,
    DemoVideoLine,
    _apply_feasibility_floor,
)

# 30.00s ordered against a 44.00s per-scenario figure ⇒ one scenario alone overruns.
_ORDERED = 30.0
_PER_SCENARIO = 44.0

_CREDITS_BDD = re.compile(r"\bowned by bdd\b", re.IGNORECASE)


def _lines(count: int) -> list[DemoVideoLine]:
    return [
        DemoVideoLine.model_validate({"id": f"clip-{i}", "text": f"Beat {i}."})
        for i in range(1, count + 1)
    ]


def _detail(yield_source: str | None) -> str:
    kept, degradations = _apply_feasibility_floor(
        _lines(2), {"target_duration_seconds": _ORDERED}, _PER_SCENARIO, 0.5, yield_source
    )
    assert len(kept) == 1, "the refit itself is unchanged by this wording fix"
    assert len(degradations) == 1
    return str(degradations[0]["detail"])


@pytest.mark.parametrize("yield_source", [_PARENT_MEASURED_YIELD])
def test_a_non_bdd_rung_never_credits_bdd_with_the_dwell(yield_source: str | None) -> None:
    """THE CLAIM. No bdd suite on this run ⇒ the detail may not name bdd as the owner."""
    detail = _detail(yield_source)
    assert not _CREDITS_BDD.search(detail), (
        f"detail credits bdd on the {yield_source!r} rung: {detail!r}"
    )
    # The honest half is still there: the figure, its provenance, the order, the refit.
    assert f"{_PER_SCENARIO:.2f}s" in detail
    assert f"ordered {_ORDERED:.2f}s" in detail
    assert "refitted to 1 scenario" in detail


def test_the_bdd_rung_keeps_the_attribution_it_can_support() -> None:
    """The other direction: on the rung where bdd DID guarantee the dwell, say so."""
    detail = _detail(_BDD_GUARANTEED_YIELD)
    assert _CREDITS_BDD.search(detail), f"the bdd rung must keep its attribution: {detail!r}"
