"""Unit tests for the run-completion gate's terminal-stage accounting.

The dispatcher decides a run is 'completed' when every blueprint stage reached a
terminal state. 'skipped' (an opt-in stage whose precondition was absent) MUST
count as terminal alongside 'completed' — otherwise a run that legitimately
skips an optional stage hangs in 'running' forever. This pins that invariant on
the pure helper, with no database.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.services.runs.run_finalizer import (
    FAILED_STAGE_STATUSES,
    TERMINAL_STAGE_STATUSES,
    RunDecision,
    decide_run_terminal_state,
    pause_trigger_stage,
    terminal_stage_names,
)


@dataclass
class _Row:
    stage_name: str
    status: str


@dataclass
class _Blueprint:
    order: list[str]

    def topological_order(self) -> list[str]:
        return self.order


def test_completed_and_skipped_are_terminal() -> None:
    assert "completed" in TERMINAL_STAGE_STATUSES
    assert "skipped" in TERMINAL_STAGE_STATUSES
    assert len(TERMINAL_STAGE_STATUSES) == 2


def test_terminal_names_includes_skipped_excludes_inflight() -> None:
    rows = [
        _Row("a", "completed"),
        _Row("b", "skipped"),
        _Row("c", "failed"),
        _Row("d", "running"),
        _Row("e", "pending"),
    ]
    assert terminal_stage_names(rows) == {"a", "b"}


def test_all_skipped_is_terminal() -> None:
    # A run whose only optional stage skipped must still be considered done.
    rows = [_Row("only", "skipped")]
    assert terminal_stage_names(rows) == {"only"}


def test_empty_rows_yields_empty_set() -> None:
    assert terminal_stage_names([]) == set()


# The dispatch-entry pause guard keys off decide_run_terminal_state == PAUSED.
# These pin the semantic that makes the guard safe: an un-approved consumed
# checkpoint returns PAUSED (re-pause the raced orphan), but once resume strips
# the checkpoint from the pause set the SAME stage states return CONTINUE — so
# an approved-then-drained run resumes instead of wrongly re-pausing.
def test_consumed_checkpoint_still_in_pause_set_is_paused() -> None:
    bp = _Blueprint(order=["gate", "after"])
    rows = [_Row("gate", "completed"), _Row("after", "pending")]
    assert decide_run_terminal_state(bp, rows, {"gate"}) is RunDecision.PAUSED


def test_consumed_checkpoint_cleared_from_pause_set_continues() -> None:
    bp = _Blueprint(order=["gate", "after"])
    rows = [_Row("gate", "completed"), _Row("after", "pending")]
    # Resume cleared "gate" from pause_after_stages → no un-approved gate remains.
    assert decide_run_terminal_state(bp, rows, set()) is RunDecision.CONTINUE


# --- FAILED direction (the completion gate's missing "second direction") ---
# A stage row in a terminal-BAD state (failed/cancelled) can never advance to
# completed/skipped, so the run can never reach DAG_COMPLETE. Before this branch
# such a run fell through to CONTINUE and the reconciler re-dispatched it forever
# until the ~2h max-age backstop. It happens for real when a resumable-timeout
# drain lets an independent sibling persist 'failed' on a transient error and
# that sibling then succeeds on resume (its success write is a guarded no-op, so
# the row stays 'failed'). These pin FAILED, and that it never regresses the
# legitimate DAG_COMPLETE / PAUSED / CONTINUE verdicts.
def test_failed_status_constant() -> None:
    assert frozenset({"failed", "cancelled"}) == FAILED_STAGE_STATUSES


def test_failed_sibling_yields_failed_not_continue() -> None:
    # The regression: without the FAIL direction this is CONTINUE (wedge).
    bp = _Blueprint(order=["a", "b"])
    rows = [_Row("a", "completed"), _Row("b", "failed")]
    assert decide_run_terminal_state(bp, rows, set()) is RunDecision.FAILED


def test_cancelled_stage_row_yields_failed() -> None:
    # 'cancelled' (written only by the failure sweep) is also terminal-bad.
    bp = _Blueprint(order=["a", "b"])
    rows = [_Row("a", "completed"), _Row("b", "cancelled")]
    assert decide_run_terminal_state(bp, rows, set()) is RunDecision.FAILED


def test_all_terminal_good_still_dag_complete_even_with_failed_elsewhere() -> None:
    # DAG_COMPLETE is checked FIRST: a fully-good DAG never trips FAILED. (A stray
    # 'failed' row for a stage NOT in the blueprint's topo order is ignored.)
    bp = _Blueprint(order=["a", "b"])
    rows = [_Row("a", "completed"), _Row("b", "skipped"), _Row("ghost", "failed")]
    assert decide_run_terminal_state(bp, rows, set()) is RunDecision.DAG_COMPLETE


def test_failed_row_outside_topo_is_ignored() -> None:
    # A failed row whose stage is not part of this blueprint must not fail the run.
    bp = _Blueprint(order=["a", "b"])
    rows = [_Row("a", "completed"), _Row("b", "pending"), _Row("ghost", "failed")]
    assert decide_run_terminal_state(bp, rows, set()) is RunDecision.CONTINUE


def test_failed_wins_over_an_unapproved_pause_gate() -> None:
    # A terminal-bad row must fail the run even if a consumed checkpoint is still
    # in the pause set — a failed run can't sit paused-for-review.
    bp = _Blueprint(order=["gate", "after"])
    rows = [_Row("gate", "completed"), _Row("after", "failed")]
    assert decide_run_terminal_state(bp, rows, {"gate"}) is RunDecision.FAILED


def test_pending_without_failed_still_continues() -> None:
    # No terminal-bad row → the normal in-flight verdict is unchanged.
    bp = _Blueprint(order=["a", "b"])
    rows = [_Row("a", "completed"), _Row("b", "running")]
    assert decide_run_terminal_state(bp, rows, set()) is RunDecision.CONTINUE


# --- Which stage the pause names (pause_trigger_stage) ---
# `runs.current_stage` is otherwise only written at stage START, and a blueprint
# wave runs its independent stages CONCURRENTLY — so the last writer can be a
# no-dep leaf that then SKIPPED. Leaving that value in place on pause pointed the
# review banner + the stage-editor link at a stage that never ran (measured live:
# a demo_video run paused after `demo_video_gen` reported `source_ingest`, which
# was SKIPPED). These pin that the pause names the checkpoint instead.
def test_pause_names_the_completed_checkpoint_not_the_skipped_first_stage() -> None:
    bp = _Blueprint(order=["source_ingest", "demo_video_gen", "script_doctor"])
    rows = [
        _Row("source_ingest", "skipped"),
        _Row("demo_video_gen", "completed"),
        _Row("script_doctor", "pending"),
    ]
    pause_set = {"demo_video_gen"}
    assert decide_run_terminal_state(bp, rows, pause_set) is RunDecision.PAUSED
    assert pause_trigger_stage(bp, rows, pause_set) == "demo_video_gen"


def test_pause_trigger_never_returns_a_skipped_stage() -> None:
    # A *skipped* pause-stage produces nothing to review and never gates
    # (dag_executor), so it must never be offered as the review target.
    bp = _Blueprint(order=["a", "gate", "b"])
    rows = [_Row("a", "completed"), _Row("gate", "skipped"), _Row("b", "pending")]
    assert pause_trigger_stage(bp, rows, {"gate"}) is None


def test_pause_trigger_is_none_when_no_checkpoint_is_consumed() -> None:
    bp = _Blueprint(order=["a", "gate"])
    rows = [_Row("a", "completed"), _Row("gate", "pending")]
    assert pause_trigger_stage(bp, rows, {"gate"}) is None
    assert pause_trigger_stage(bp, rows, set()) is None


def test_pause_trigger_picks_the_furthest_consumed_checkpoint() -> None:
    # Two consumed gates (a re-paused raced re-dispatch): the run stopped at the
    # LAST one in topological order, so that is the stage to review.
    bp = _Blueprint(order=["gate1", "middle", "gate2", "after"])
    rows = [
        _Row("gate1", "completed"),
        _Row("middle", "completed"),
        _Row("gate2", "completed"),
        _Row("after", "pending"),
    ]
    assert pause_trigger_stage(bp, rows, {"gate1", "gate2"}) == "gate2"


def test_pause_trigger_tolerates_a_checkpoint_outside_the_blueprint() -> None:
    # An off-blueprint checkpoint cannot be topologically ordered; still name it
    # rather than falling back to whatever current_stage happens to hold.
    bp = _Blueprint(order=["a"])
    rows = [_Row("a", "pending"), _Row("ghost", "completed")]
    assert pause_trigger_stage(bp, rows, {"ghost"}) == "ghost"
