"""The final_critique -> video_analyzer poll budget, and what it costs the matrix.

`_ANALYSIS_POLL_TIMEOUT_S` has now been raised twice for the SAME bug: the client
stopped polling while the analyzer was still working, so the stage completed with
`gestalt_summary == {}` and the calibration corpus silently gained nothing. Both
previous values (180.0, then 420.0) were derived from a percentile of observed
analyzer runtimes, and both were obsolete within weeks — the analyzer's runtime
distribution moves with cluster contention and with pgx model swaps, so a constant
sized from a sample schedules its own next failure.

A test asserting `_ANALYSIS_POLL_TIMEOUT_S == 1100.0` would restate the line and
catch nothing. These assert the RELATIONS that actually have to hold:

1. the poll must claim the room the BINDING constraint gives it, and no more —
   that is min(stage spare budget, the client clamp in `BaseServiceClient`), and the
   clamp is what binds today. Deriving the floor from the stage budget alone is what
   shipped `1100.0`, a value the clamp RAISES on rather than caps;
2. the run wall-clock that buys must fit inside the E2E deadlines that pay for it;
3. the shortfall against the analyzer's own deadline must stay documented, so the
   next reader knows the bug is bounded rather than closed.

Together they mean a future lowering of the poll budget, or a raise of it without
the matching stage and matrix headroom, fails here in seconds rather than in a
75-minute nightly job.

The stage-budget relation itself is asserted against the live `StageSpec` in
`test_final_critique.py::
test_analyzer_poll_timeout_is_sized_from_measurement_and_fits_the_stage`.
"""

from __future__ import annotations

import json
from pathlib import Path

from w2v_common.clients.base import _MAX_TIMEOUT, _clamp_timeout

from src.services.stages.demo_video.final_critique import (
    _ANALYSIS_POLL_TIMEOUT_S,
    _ANALYZER_JOB_DEADLINE_S,
    _LLM_LADDER_WORST_CASE_S,
    _STAGE_TIMEOUT_S,
)

#: Matrix labels whose scenario runs the demo_video blueprint all the way through
#: `final_critique`, i.e. the entries that actually pay for the poll budget.
#: `demo-video-gen` is deliberately absent: its scenario sets
#: `pause_after_stages: ["demo_video_gen"]`, so it never reaches this stage and its
#: 600s deadline is unaffected.
_DEMO_VIDEO_FULL_CHAIN_LABELS = frozenset(
    {
        "demo-video-full-chain-todomvc",
        "demo-video-relief",
        "demo-video-footage-relief",
        "demo-video-claim-finish",
    }
)

#: Measured 2026-09-01 on the live orchestrator DB: over the 24 completed
#: `demo_video_v1` runs since the 2026-08-19 pgx model swap, the longest run
#: duration EXCLUDING its own final_critique stage was 2588s. This is the part of a
#: demo_video run the poll budget does NOT control, so it is the floor an E2E
#: deadline has to clear before final_critique spends a single second.
_MAX_DEMO_VIDEO_RUN_EXCLUDING_FINAL_CRITIQUE_S = 2588.0


def _matrix_deadlines() -> dict[str, int]:
    """`label -> deadline` for every matrix entry, narrowed to real types.

    `matrix.json` also carries `_doc` / `_doc2` prose keys; entries without a real
    `label` + integer `deadline` are skipped rather than trusted.
    """
    raw: object = json.loads(Path("scripts/e2e_scenarios/matrix.json").read_text())
    assert isinstance(raw, list)
    deadlines: dict[str, int] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label")
        deadline = entry.get("deadline")
        if isinstance(label, str) and isinstance(deadline, int):
            deadlines[label] = deadline
    return deadlines


def test_poll_budget_claims_the_room_the_stage_budget_allows() -> None:
    """The poll must spend the stage's spare budget — all of it, and only it.

    Upper bound: a poll that outlives the stage wrapper is worse than dead config.
    dag_executor raises StageResumableTimeoutError and re-dispatches the stage from
    scratch (a second analyzer job, possibly a second full compile); past
    `max_stage_attempts` the run persists `failed`. That converts a gate documented
    as never-hard-blocking into a hard block.

    Lower bound: every second of stage budget the poll does NOT claim is a gestalt
    sample thrown away while video_analyzer is still working — the exact defect
    180.0 and 420.0 each shipped.

    ⚠️ THE STAGE BUDGET IS THE UPPER BOUND ONLY. This test used to derive the LOWER
    bound from it too (`spare - 100 <= poll`), which forced the poll UP to 1100.0 — a
    value `BaseServiceClient._clamp_timeout` REJECTS with `ValueError` rather than
    capping, so it raised on every analyzer poll in production while this file stayed
    green. The floor is now the smaller of the stage's spare room and the CLIENT
    CLAMP, i.e. the constraint that actually binds.

    Mutation-checked: restoring `_ANALYSIS_POLL_TIMEOUT_S = 420.0` fails this, and so
    does raising it to 1100.0 or to the analyzer's 1800s deadline.
    """
    spare = _STAGE_TIMEOUT_S - _LLM_LADDER_WORST_CASE_S
    binding = min(spare, _MAX_TIMEOUT)

    assert spare > _ANALYSIS_POLL_TIMEOUT_S, (
        f"the poll budget ({_ANALYSIS_POLL_TIMEOUT_S}s) plus the LLM ladder "
        f"({_LLM_LADDER_WORST_CASE_S}s) must fit inside the stage budget "
        f"({_STAGE_TIMEOUT_S}s), or the stage wrapper fires first and the SOFT gate "
        "becomes a hard, re-dispatching failure"
    )
    _clamp_timeout(_ANALYSIS_POLL_TIMEOUT_S)  # raises above the transport ceiling
    assert binding - 100 <= _ANALYSIS_POLL_TIMEOUT_S, (
        f"the poll budget ({_ANALYSIS_POLL_TIMEOUT_S}s) leaves more than 100s of the "
        f"{binding}s the BINDING constraint allows unused (stage spare {spare}s, "
        f"client clamp {_MAX_TIMEOUT}s — the smaller one binds). Do not re-derive it "
        "from a percentile of analyzer runtimes — that is what 180.0 and 420.0 did and "
        "both went stale within weeks. Claim the room the binding contract gives you, "
        "and no more: above the clamp the poll raises rather than waits."
    )


def test_the_shortfall_against_the_analyzer_deadline_stays_documented() -> None:
    """This fix is bounded, not complete, and the next reader has to know that.

    The IDEAL key is the analyzer's own `job_deadline_seconds` (1800): every job
    provably resolves — completed or failed — by then, so a client that waits that
    long can never abandon work that was still coming. The stage budget does not
    permit it today, so jobs between the poll budget and 1800s are still truncated.

    This asserts the gap is real and acknowledged rather than silently assumed away.
    If someone raises `_STAGE_TIMEOUT_S` enough to close it, this test fails and they
    are made to delete it deliberately — at which point the poll should simply be
    keyed to `_ANALYZER_JOB_DEADLINE_S` and the whole class of bug ends.
    """
    assert _ANALYSIS_POLL_TIMEOUT_S < _ANALYZER_JOB_DEADLINE_S, (
        "the poll budget now reaches video_analyzer's own job deadline. Good — that "
        "closes the truncation class for real. Key _ANALYSIS_POLL_TIMEOUT_S directly "
        "to _ANALYZER_JOB_DEADLINE_S (plus a small margin) and delete this test."
    )
    assert _STAGE_TIMEOUT_S < _ANALYZER_JOB_DEADLINE_S + _LLM_LADDER_WORST_CASE_S, (
        "the stage budget is now large enough to poll for the analyzer's full "
        "deadline; raise _ANALYSIS_POLL_TIMEOUT_S to claim it"
    )


def test_demo_video_matrix_deadlines_cover_the_poll_budget() -> None:
    """The poll budget is spent out of the demo_video E2E deadline. Pin the coupling.

    Raising the poll from 420s to 1100s is a real, bounded increase in demo_video run
    wall-clock, and it lands where there is LEAST headroom: measured over the 24
    completed demo_video_v1 runs since the 2026-08-19 model swap, the six runs whose
    gestalt timed out at 420s were the six slowest runs in the set, because a slow
    analyzer and a slow run share one cause — contention.

    So a demo_video full-chain deadline must clear the part of the run the poll does
    not control PLUS a full poll budget. Left at the old 3600, the 2898s run would
    have landed at 3578s — 22s of nominal margin, which is not margin.

    Mutation-checked twice: raising `_ANALYSIS_POLL_TIMEOUT_S` without touching
    matrix.json fails this, and lowering any of these deadlines back to 3600 fails it
    too.
    """
    required = _MAX_DEMO_VIDEO_RUN_EXCLUDING_FINAL_CRITIQUE_S + _ANALYSIS_POLL_TIMEOUT_S

    deadlines = _matrix_deadlines()
    matched = _DEMO_VIDEO_FULL_CHAIN_LABELS & deadlines.keys()
    assert matched == _DEMO_VIDEO_FULL_CHAIN_LABELS, (
        "the demo_video full-chain matrix entries this guard protects have been "
        f"renamed or removed. Guard list: {sorted(_DEMO_VIDEO_FULL_CHAIN_LABELS)}. "
        f"Present in matrix.json: {sorted(matched)}. Revise the guard list "
        "deliberately — do not let it silently protect nothing."
    )

    offenders = [
        (label, deadlines[label])
        for label in sorted(_DEMO_VIDEO_FULL_CHAIN_LABELS)
        if deadlines[label] < required
    ]

    assert offenders == [], (
        f"these demo_video scenarios budget less than {required:.0f}s "
        f"({_MAX_DEMO_VIDEO_RUN_EXCLUDING_FINAL_CRITIQUE_S:.0f}s measured max run "
        f"excluding final_critique + {_ANALYSIS_POLL_TIMEOUT_S:.0f}s of analyzer "
        f"polling), so a slow-analyzer run can outlive its deadline and red the "
        f"matrix: {offenders}. Either raise these deadlines with the poll budget, or "
        "do not raise the poll budget."
    )
