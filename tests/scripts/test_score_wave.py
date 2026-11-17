"""score_wave.py — lineage-role split + unreadable-decision-payload paths.

Two properties this tool must never get wrong again (see the module docstring
for the prior published-wrong-verdict incidents):

  1. LINEAGE SPLIT. A structural-relief PARENT is out of band BY CONSTRUCTION.
     `build_rows` must classify orig/parent/child correctly and
     `print_role_summary` must never pool them into one median.
  2. UNREADABLE DECISION PAYLOAD -> loud failure. A decision row whose output
     does not match either the nested or the top-level shape must make
     `print_relief_decisions` return exit code 3, not silently read as "no
     relief happened".
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.score_wave import ScoredRun, build_rows, print_relief_decisions, print_role_summary


def _write_stage_rows(snapdir: Path, run_id: str, stage_rows: list[dict]) -> None:
    stages = snapdir / "stages"
    stages.mkdir(parents=True, exist_ok=True)
    (stages / f"{run_id}.json").write_text(json.dumps(stage_rows))


def _conformance_stage(deviation_pct: float) -> dict:
    return {
        "stage_name": "demo_video_compile",
        "output": {
            "duration_conformance": {
                "deviation_pct": deviation_pct,
                "delivered_basis": "playable",
                "delivered_seconds": 44.0,
            }
        },
    }


def test_build_rows_splits_orig_parent_child_by_lineage(tmp_path: Path) -> None:
    runs = [
        {
            "id": "orig-1",
            "blueprint_id": "demo_video",
            "status": "completed",
            "ordered_seconds": 44,
        },
        {
            "id": "parent-1",
            "blueprint_id": "demo_video",
            "status": "completed",
            "ordered_seconds": 44,
        },
        {
            "id": "child-1",
            "blueprint_id": "demo_video",
            "status": "completed",
            "ordered_seconds": 44,
            "parent_run_id": "parent-1",
        },
    ]
    _write_stage_rows(tmp_path, "orig-1", [_conformance_stage(3.0)])
    # Parent is out of band by construction — a huge deviation is EXPECTED, not a defect.
    _write_stage_rows(tmp_path, "parent-1", [_conformance_stage(56.8)])
    _write_stage_rows(tmp_path, "child-1", [_conformance_stage(-2.5)])

    rows = build_rows(runs, tmp_path / "stages")
    by_id = {r.id: r for r in rows}

    assert by_id["orig-1"].role == "orig"
    assert by_id["parent-1"].role == "parent"
    assert by_id["child-1"].role == "child"
    # The precondition that makes it a parent must never leak into being an
    # `orig` too (i.e. no row is double-counted across roles).
    assert {r.role for r in rows} == {"orig", "parent", "child"}


def test_role_summary_never_pools_parent_into_orig_or_child(tmp_path: Path, capsys) -> None:
    """A parent's out-of-band deviation must not drag the orig/child medians."""
    rows = [
        ScoredRun(
            id="orig-1",
            blueprint="demo_video",
            status="completed",
            role="orig",
            srp=None,
            ordered=44,
            conf={"deviation_pct": 3.0, "delivered_basis": "playable", "delivered_seconds": 45.3},
            stage="demo_video_compile",
            decisions=[],
            captured=True,
        ),
        ScoredRun(
            id="parent-1",
            blueprint="demo_video",
            status="completed",
            role="parent",
            srp=True,
            ordered=44,
            conf={"deviation_pct": 56.8, "delivered_basis": "playable", "delivered_seconds": 69.0},
            stage="demo_video_compile",
            decisions=[],
            captured=True,
        ),
        ScoredRun(
            id="child-1",
            blueprint="demo_video",
            status="completed",
            role="child",
            srp=True,
            ordered=44,
            conf={"deviation_pct": -2.5, "delivered_basis": "playable", "delivered_seconds": 42.9},
            stage="demo_video_compile",
            decisions=[],
            captured=True,
        ),
    ]

    print_role_summary(rows)
    out = capsys.readouterr().out

    # Each role gets its OWN n=1 line — a merged/pooled summary would print n=3
    # for whichever role absorbed the others, or a median dragged by 56.8%.
    assert "orig    n=1" in out
    assert "parent  n=1" in out
    assert "child   n=1" in out
    assert "median= +56.80%" in out  # the parent's own line, out of band by construction
    assert "median=  +3.00%" in out  # the orig line — NOT averaged with the parent's 56.8%


def test_unreadable_decision_payload_returns_exit_code_3(capsys) -> None:
    row = ScoredRun(
        id="run-unreadable",
        blueprint="demo_video",
        status="completed",
        role="orig",
        srp=None,
        ordered=44,
        conf=None,
        stage=None,
        decisions=[
            {
                "stage_name": "structural_relief_decision",
                # Neither nested under "structural_relief_decision" nor carrying
                # any of the recognised top-level markers -> unreadable.
                "output": {"unexpected_shape": True},
            }
        ],
        captured=True,
    )

    rc = print_relief_decisions([row])

    assert rc == 3
    err = capsys.readouterr().err
    assert "DECISION ROWS COULD NOT BE READ" in err
    assert "run-unreadable" in err


def test_readable_decision_payload_returns_exit_code_0(capsys) -> None:
    row = ScoredRun(
        id="run-readable",
        blueprint="demo_video",
        status="completed",
        role="orig",
        srp=None,
        ordered=44,
        conf=None,
        stage=None,
        decisions=[
            {
                "stage_name": "structural_relief_decision",
                "output": {
                    "structural_relief_decision": {
                        "schema_version": 2,
                        "outcome": "spawned",
                        "refusal": None,
                        "inputs": {"shape": {"target_beats": 6, "measured_seconds_per_beat": 4.0}},
                        "spawned_child_run_id": "child-99",
                        "spawned_pass": 1,
                    }
                },
            }
        ],
        captured=True,
    )

    rc = print_relief_decisions([row])

    assert rc == 0
    out = capsys.readouterr().out
    assert "predicted=24.0" in out
