import logging
from uuid import uuid4

import pytest
from sqlalchemy import text

from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository
from src.database.repositories.stage_output_repository import StageOutputRepository

pytestmark = pytest.mark.asyncio


async def _setup_run(session):
    tenant = uuid4()
    repo = ProjectRepository(session)
    proj = await repo.create_with_deliverables(
        tenant_id=tenant,
        owner_subject_id=uuid4(),
        type_="generic",
        title="x",
        brief={},
        brand={},
        voice={},
        theme={},
        visibility="tenant",
        deliverables=[
            {
                "name": "d",
                "format": "16x9",
                "captions_mode": "none",
                "target_duration_seconds": 60,
            }
        ],
    )
    await session.commit()
    did = proj.deliverables[0].id
    run_repo = RunRepository(session)
    run = await run_repo.create(
        deliverable_id=did,
        blueprint_id="generic_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    await session.commit()
    return run


async def test_upsert_pending_idempotent(db_session) -> None:
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    so1 = await repo.upsert_pending(run.id, "script_draft")
    await db_session.commit()
    so2 = await repo.upsert_pending(run.id, "script_draft")
    await db_session.commit()

    # Same row returned both times
    assert so1.id == so2.id

    # Only one row in DB
    rows = await repo.list_for_run(run.id)
    assert len(rows) == 1
    assert rows[0].stage_name == "script_draft"
    assert rows[0].status == "pending"


async def test_mark_running_then_completed(db_session) -> None:
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    await repo.upsert_pending(run.id, "analyze")
    await db_session.commit()

    await repo.mark_running(run.id, "analyze")
    await db_session.commit()
    so = await repo.get(run.id, "analyze")
    assert so is not None
    assert so.status == "running"
    assert so.started_at is not None

    output_data = {"summary": "test output", "score": 42}
    await repo.mark_completed(run.id, "analyze", output_data)
    await db_session.commit()
    so = await repo.get(run.id, "analyze")
    assert so is not None
    assert so.status == "completed"
    assert so.output == output_data
    assert so.completed_at is not None
    assert so.started_at is not None


async def test_mark_pending_resets_running_but_not_completed(db_session) -> None:
    """mark_pending resets a 'running' stage (cancel-stranded) back to pending
    and clears its outcome fields, but must NEVER clobber a 'completed' stage —
    a late cancellation could otherwise wipe a stage that already finished.

    ``started_at`` is NOT cleared: see
    ``test_resumed_stage_keeps_its_first_attempt_start_time``."""
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    # running → pending (the cancel-stranded case)
    await repo.upsert_pending(run.id, "analyze")
    await repo.mark_running(run.id, "analyze")
    await db_session.commit()
    await repo.mark_pending(run.id, "analyze")
    await db_session.commit()
    so = await repo.get(run.id, "analyze")
    assert so is not None
    assert so.status == "pending"
    assert so.started_at is not None
    assert so.completed_at is None
    assert so.error is None

    # completed must be untouched by a stray mark_pending
    await repo.upsert_pending(run.id, "script_draft")
    await repo.mark_running(run.id, "script_draft")
    await repo.mark_completed(run.id, "script_draft", {"k": "v"})
    await db_session.commit()
    await repo.mark_pending(run.id, "script_draft")
    await db_session.commit()
    done = await repo.get(run.id, "script_draft")
    assert done is not None
    assert done.status == "completed"
    assert done.output == {"k": "v"}


async def test_increment_attempts_bumps_and_survives_mark_pending(db_session) -> None:
    """Durability P1: increment_attempts returns the new value and the counter is
    NOT reset by mark_running/mark_pending, so it survives a resume."""
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    await repo.upsert_pending(run.id, "growth_bundle")
    await repo.mark_running(run.id, "growth_bundle")
    await db_session.commit()

    assert await repo.increment_attempts(run.id, "growth_bundle") == 1
    await db_session.commit()
    assert await repo.increment_attempts(run.id, "growth_bundle") == 2
    await db_session.commit()

    # A resume resets the stage to pending — the counter must persist.
    await repo.mark_pending(run.id, "growth_bundle")
    await db_session.commit()
    so = await repo.get(run.id, "growth_bundle")
    assert so is not None
    assert so.status == "pending"
    assert so.attempts == 2

    # mark_running (next resume) also must not reset it.
    await repo.mark_running(run.id, "growth_bundle")
    await db_session.commit()
    so = await repo.get(run.id, "growth_bundle")
    assert so is not None
    assert so.attempts == 2


async def test_increment_attempts_missing_row_returns_zero(db_session) -> None:
    """Defensive: incrementing a non-existent stage row returns 0 (no crash)."""
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)
    assert await repo.increment_attempts(run.id, "never_seeded") == 0


async def test_increment_attempts_missing_row_warns(db_session, caplog) -> None:
    """The defensive 0 is an invariant breach (reconcile_blueprint_stage_rows
    guarantees the row) and must be SIGNALLED, not silent — 0 <= cap grants an
    unbounded resume. Behaviour is unchanged: still 0, still no raise."""
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    with caplog.at_level(
        logging.WARNING, logger="src.database.repositories.stage_output_repository"
    ):
        assert await repo.increment_attempts(run.id, "never_seeded") == 0

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "a missing stage_outputs row must emit a warning"
    msg = warnings[-1].getMessage()
    assert str(run.id) in msg
    assert "never_seeded" in msg


async def test_mark_completed_reports_whether_a_row_was_written(db_session) -> None:
    """mark_completed returns rowcount > 0 so a missing row is self-reporting
    instead of a silent no-op that drops the stage's computed output."""
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    await repo.upsert_pending(run.id, "analyze")
    await db_session.commit()

    assert await repo.mark_completed(run.id, "analyze", {"k": "v"}) is True
    await db_session.commit()

    # No row for this stage → the UPDATE touches nothing.
    assert await repo.mark_completed(run.id, "never_seeded", {"k": "v"}) is False
    await db_session.commit()
    assert await repo.get(run.id, "never_seeded") is None


async def test_mark_skipped_reports_whether_a_row_was_written(db_session) -> None:
    """mark_skipped returns rowcount > 0. A False means the terminal 'skipped'
    never landed, so the completion gate can never count the stage."""
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    await repo.upsert_pending(run.id, "ai_image_gen")
    await db_session.commit()

    assert await repo.mark_skipped(run.id, "ai_image_gen", "no config block") is True
    await db_session.commit()
    so = await repo.get(run.id, "ai_image_gen")
    assert so is not None
    assert so.status == "skipped"
    assert so.error == "no config block"

    assert await repo.mark_skipped(run.id, "never_seeded", "reason") is False
    await db_session.commit()
    assert await repo.get(run.id, "never_seeded") is None


# --- write-once contract: a terminal stage row is never clobbered ------------
#
# run_dispatcher documents mark_completed/mark_failed as "write-once-guarded",
# but only mark_pending actually carried a status guard. Every terminal writer
# now guards on ``status IN ('pending','running')`` (only a NON-terminal row may
# be mutated), so a racing/stale write across the 2 replicas — or a resume
# re-issuing a mark on an already-terminal row — is a benign no-op instead of a
# silent clobber. terminal = {completed, failed, skipped}.


async def test_mark_completed_never_clobbers_terminal(db_session) -> None:
    """A completed write must not overwrite an already-'failed' stage, and it
    self-reports the no-op as False."""
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    await repo.upsert_pending(run.id, "analyze")
    await repo.mark_running(run.id, "analyze")
    await repo.mark_failed(run.id, "analyze", "boom")
    await db_session.commit()

    wrote = await repo.mark_completed(run.id, "analyze", {"real": "result"})
    await db_session.commit()

    assert wrote is False
    so = await repo.get(run.id, "analyze")
    assert so is not None
    assert so.status == "failed"
    assert so.error == "boom"
    assert so.output is None


async def test_mark_failed_never_clobbers_terminal_and_reports_rowcount(db_session) -> None:
    """A failed write must not overwrite an already-'completed' stage, and
    mark_failed now returns a bool (rowcount > 0) like its siblings."""
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    await repo.upsert_pending(run.id, "analyze")
    await repo.mark_running(run.id, "analyze")
    await repo.mark_completed(run.id, "analyze", {"k": "v"})
    await db_session.commit()

    wrote = await repo.mark_failed(run.id, "analyze", "boom")
    await db_session.commit()

    assert wrote is False
    so = await repo.get(run.id, "analyze")
    assert so is not None
    assert so.status == "completed"
    assert so.error is None
    assert so.output == {"k": "v"}

    # And an absent row is self-reporting, not a silent None.
    assert await repo.mark_failed(run.id, "never_seeded", "x") is False


async def test_mark_running_never_reopens_terminal(db_session) -> None:
    """mark_running must not reopen a completed stage back to 'running'."""
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    await repo.upsert_pending(run.id, "analyze")
    await repo.mark_running(run.id, "analyze")
    await repo.mark_completed(run.id, "analyze", {"k": "v"})
    await db_session.commit()
    before = await repo.get(run.id, "analyze")
    assert before is not None
    started = before.started_at

    await repo.mark_running(run.id, "analyze")
    await db_session.commit()

    so = await repo.get(run.id, "analyze")
    assert so is not None
    assert so.status == "completed"
    assert so.started_at == started


async def test_record_progress_never_overwrites_completed(db_session) -> None:
    """Sub-DAG bookkeeping must not clobber a completed course row's real output
    on a resume that re-attaches to existing children."""
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    await repo.upsert_pending(run.id, "course_compile_lessons")
    await repo.mark_running(run.id, "course_compile_lessons")
    await repo.mark_completed(run.id, "course_compile_lessons", {"real": "result"})
    await db_session.commit()

    await repo.record_progress(run.id, "course_compile_lessons", {"children": ["a", "b"]})
    await db_session.commit()

    so = await repo.get(run.id, "course_compile_lessons")
    assert so is not None
    assert so.status == "completed"
    assert so.output == {"real": "result"}


# --- regression sentinels: GREEN before AND after the guard -------------------


async def test_record_progress_writes_while_running(db_session) -> None:
    """The guard still allows the intended write while the course stage runs."""
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    await repo.upsert_pending(run.id, "course_compile_lessons")
    await repo.mark_running(run.id, "course_compile_lessons")
    await db_session.commit()

    await repo.record_progress(run.id, "course_compile_lessons", {"children": ["a", "b"]})
    await db_session.commit()

    so = await repo.get(run.id, "course_compile_lessons")
    assert so is not None
    assert so.status == "running"
    assert so.output == {"children": ["a", "b"]}


async def test_skipped_stage_is_terminal_success_not_rerun_on_resume(db_session) -> None:
    """A 'skipped' stage is terminal-SUCCESS — the dispatcher seeds it into the
    resume completed-set (with an empty {} output, since a skip persists no output)
    so it is NOT re-dispatched. This mirrors run_finalizer.TERMINAL_STAGE_STATUSES =
    {'completed','skipped'}. Pins that a skip persists cleanly with NO output, so the
    dispatcher's `status == "skipped"` seed produces `{}` and never re-runs the stage.
    """
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    await repo.upsert_pending(run.id, "ai_image_gen")
    await repo.mark_running(run.id, "ai_image_gen")
    await repo.mark_skipped(run.id, "ai_image_gen", "no config block")
    await db_session.commit()

    so = await repo.get(run.id, "ai_image_gen")
    assert so is not None
    # Terminal-success skip: recorded skipped, carries the reason, holds NO output.
    assert so.status == "skipped"
    assert so.error == "no config block"
    assert not so.output  # None/{} — the dispatcher seeds this stage as {} on resume

    # Reproduce the dispatcher's resume seed (run_dispatcher.py prior_run_outputs):
    # a 'skipped' row IS included (with {}), so on resume it is already-completed
    # and never re-dispatched — unlike the old behaviour that dropped it and re-ran.
    stage_rows = await repo.list_for_run(run.id)
    prior_run_outputs = {
        r.stage_name: (dict(r.output) if (r.status == "completed" and r.output) else {})
        for r in stage_rows
        if (r.status == "completed" and r.output) or r.status == "skipped"
    }
    assert "ai_image_gen" in prior_run_outputs  # seeded → resume will NOT re-run it
    assert prior_run_outputs["ai_image_gen"] == {}


# --- bulk_cancel_non_terminal: sweep stranded stage rows on run hard-failure ---
#
# When a stage hard-fails, the DAG executor cancels only its current gather batch
# (resetting those siblings to 'pending'); any independent sibling branch or
# never-scheduled downstream stage is left 'pending'/'running' under a now-terminal
# 'failed' run that is never resumed → the rows strand forever and pollute the
# stage_outputs group-by. The run-failure site sweeps them to terminal 'cancelled'.


async def test_bulk_cancel_non_terminal_sweeps_pending_and_running_only(db_session) -> None:
    """Only 'pending'/'running' rows are swept to 'cancelled' (with completed_at);
    already-terminal 'completed'/'failed'/'skipped' rows are never clobbered."""
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    # A completed stage (terminal) — output must survive.
    await repo.upsert_pending(run.id, "analyze")
    await repo.mark_running(run.id, "analyze")
    await repo.mark_completed(run.id, "analyze", {"k": "v"})
    # A failed stage (terminal) — the stage that hard-failed the run.
    await repo.upsert_pending(run.id, "record")
    await repo.mark_running(run.id, "record")
    await repo.mark_failed(run.id, "record", "boom")
    # A skipped stage (terminal) — a legitimately opted-out stage.
    await repo.upsert_pending(run.id, "ai_image_gen")
    await repo.mark_skipped(run.id, "ai_image_gen", "no config block")
    # An independent sibling still 'pending' (never scheduled before the abort).
    await repo.upsert_pending(run.id, "claim_finish")
    # An in-flight sibling still 'running' when the run aborted.
    await repo.upsert_pending(run.id, "growth_bundle")
    await repo.mark_running(run.id, "growth_bundle")
    await db_session.commit()

    swept = await repo.bulk_cancel_non_terminal(run.id)
    await db_session.commit()

    # Exactly the two non-terminal rows were swept.
    assert swept == 2

    by_name = {r.stage_name: r for r in await repo.list_for_run(run.id)}
    # Swept → cancelled, completed_at stamped.
    assert by_name["claim_finish"].status == "cancelled"
    assert by_name["claim_finish"].completed_at is not None
    assert by_name["growth_bundle"].status == "cancelled"
    assert by_name["growth_bundle"].completed_at is not None
    # Terminal rows untouched — the write-once contract holds.
    assert by_name["analyze"].status == "completed"
    assert by_name["analyze"].output == {"k": "v"}
    assert by_name["record"].status == "failed"
    assert by_name["record"].error == "boom"
    assert by_name["ai_image_gen"].status == "skipped"


async def test_bulk_cancel_non_terminal_noop_when_all_terminal(db_session) -> None:
    """A run whose stages are all terminal sweeps nothing (returns 0) and mutates
    no rows — so calling it on an already-clean run is a safe no-op."""
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    await repo.upsert_pending(run.id, "analyze")
    await repo.mark_running(run.id, "analyze")
    await repo.mark_completed(run.id, "analyze", {"k": "v"})
    await db_session.commit()

    swept = await repo.bulk_cancel_non_terminal(run.id)
    await db_session.commit()

    assert swept == 0
    so = await repo.get(run.id, "analyze")
    assert so is not None
    assert so.status == "completed"


async def test_increment_attempts_never_bumps_a_swept_cancelled_row(db_session) -> None:
    """A row swept to terminal 'cancelled' by bulk_cancel_non_terminal must stay
    unwritable — increment_attempts carried NO status guard while every sibling
    writer did, so it could still bump the resume counter on a run that is
    already terminal-failed. Latent hardening: the sole caller
    (run_dispatcher.bump_attempts) runs before the sweep, so no reachable path
    hits this today. The guard makes it structurally impossible either way, and
    the no-op reports the usual defensive 0."""
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    await repo.upsert_pending(run.id, "growth_bundle")
    await repo.mark_running(run.id, "growth_bundle")
    await db_session.commit()
    assert await repo.increment_attempts(run.id, "growth_bundle") == 1
    await db_session.commit()

    # The run hard-fails; the leftover row is swept to terminal 'cancelled'.
    assert await repo.bulk_cancel_non_terminal(run.id) == 1
    await db_session.commit()

    # A stray bump must not mutate the swept row.
    assert await repo.increment_attempts(run.id, "growth_bundle") == 0
    await db_session.commit()

    so = await repo.get(run.id, "growth_bundle")
    assert so is not None
    assert so.status == "cancelled"
    assert so.attempts == 1  # unchanged — the guard held


async def test_increment_attempts_still_bumps_non_terminal_rows(db_session) -> None:
    """Regression fence for the guard above: the intended bump still works on
    both non-terminal states ('pending' before first dispatch, 'running' in
    flight). A guard that over-fired here would break timeout resume entirely."""
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    # 'pending' — seeded but not yet dispatched.
    await repo.upsert_pending(run.id, "claim_finish")
    await db_session.commit()
    assert await repo.increment_attempts(run.id, "claim_finish") == 1
    await db_session.commit()

    # 'running' — the live timeout-resume case.
    await repo.upsert_pending(run.id, "growth_bundle")
    await repo.mark_running(run.id, "growth_bundle")
    await db_session.commit()
    assert await repo.increment_attempts(run.id, "growth_bundle") == 1
    assert await repo.increment_attempts(run.id, "growth_bundle") == 2
    await db_session.commit()

    by_name = {r.stage_name: r for r in await repo.list_for_run(run.id)}
    assert by_name["claim_finish"].attempts == 1
    assert by_name["growth_bundle"].attempts == 2


async def test_mark_completed_on_cancelled_row_returns_false_and_warns(db_session, caplog) -> None:
    """A guarded no-op is invisible unless it SPEAKS: the bool return is the only
    signal, and some call sites discard it, so a stage result silently dropped on
    an already-terminal row leaves no trace. The repository now warns on every
    0-row terminal write, naming the run and stage."""
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    await repo.upsert_pending(run.id, "analyze")
    await repo.mark_running(run.id, "analyze")
    await db_session.commit()
    assert await repo.bulk_cancel_non_terminal(run.id) == 1
    await db_session.commit()

    with caplog.at_level(
        logging.WARNING, logger="src.database.repositories.stage_output_repository"
    ):
        wrote = await repo.mark_completed(run.id, "analyze", {"real": "result"})
    await db_session.commit()

    assert wrote is False
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "a 0-row terminal write must emit a warning"
    msg = warnings[-1].getMessage()
    assert "mark_completed" in msg
    assert str(run.id) in msg
    assert "analyze" in msg

    # And the swept row is untouched — 'cancelled' stays terminal.
    so = await repo.get(run.id, "analyze")
    assert so is not None
    assert so.status == "cancelled"
    assert so.output is None


async def test_bulk_cancel_non_terminal_scoped_to_run(db_session) -> None:
    """The sweep is scoped to its run_id — a concurrently-live OTHER run's
    pending/running rows are never touched."""
    run_a = await _setup_run(db_session)
    run_b = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    await repo.upsert_pending(run_a.id, "claim_finish")  # will be swept
    await repo.upsert_pending(run_b.id, "claim_finish")  # must survive
    await repo.mark_running(run_b.id, "claim_finish")
    await db_session.commit()

    swept = await repo.bulk_cancel_non_terminal(run_a.id)
    await db_session.commit()

    assert swept == 1
    a = await repo.get(run_a.id, "claim_finish")
    b = await repo.get(run_b.id, "claim_finish")
    assert a is not None and a.status == "cancelled"
    assert b is not None and b.status == "running"  # other run untouched


async def test_resumed_stage_keeps_its_first_attempt_start_time(db_session) -> None:
    """A resumable-timeout resume must NOT erase the stage's FIRST start.

    Nightly action_reel run 01a01d41 read as: shot_analysis completed 03:40:57,
    segment_select — its only runnable successor — started 04:13:12, i.e. 1935s
    with nothing running, then timed out against the harness deadline. There was
    no dispatch gap. Those 1935s WERE segment_select's first attempt: it hit its
    LLM budget (950s x 2 attempts), ``DAGExecutor._run_one`` reset it to
    'pending' and raised StageResumableTimeoutError, and the reconciler
    re-dispatched it. ``mark_pending`` nulled ``started_at`` and ``mark_running``
    re-stamped ``now()``, so the only surviving evidence of attempt #1 was
    destroyed and the stage table rendered a retry as a phantom 32-minute stall
    — the run's whole remaining budget attributed to the wrong mechanism.

    ``RunRepository.mark_started`` already COALESCEs for exactly this reason at
    the RUN level; the stage-level writer was the unguarded twin.
    """
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    await repo.upsert_pending(run.id, "segment_select")
    await repo.mark_running(run.id, "segment_select")
    await db_session.commit()
    first = await repo.get(run.id, "segment_select")
    assert first is not None
    first_start = first.started_at
    assert first_start is not None

    # The resumable-timeout path: bump the durable counter, reset to pending.
    assert await repo.increment_attempts(run.id, "segment_select") == 1
    await repo.mark_pending(run.id, "segment_select")
    await db_session.commit()
    reset = await repo.get(run.id, "segment_select")
    assert reset is not None
    assert reset.status == "pending"
    # The stage is awaiting resume, not un-started: when it began stays on record.
    assert reset.started_at == first_start
    assert reset.completed_at is None

    # The reconciler re-dispatches; attempt #2 must not rewrite history.
    await repo.mark_running(run.id, "segment_select")
    await db_session.commit()
    resumed = await repo.get(run.id, "segment_select")
    assert resumed is not None
    assert resumed.status == "running"
    assert resumed.started_at == first_start, (
        "a resumed stage re-stamped started_at: elapsed now reports only the last "
        "attempt, and the burnt attempt reads as a gap before the stage started"
    )
    assert resumed.attempts == 1


async def test_drained_resume_reports_per_execution_runtime_not_wall_clock(db_session) -> None:
    """A drained-and-resumed stage's per-execution runtime must be bounded by its
    ceiling, NOT by the wall clock since first dispatch.

    This is the whole point of ``last_execution_started_at``. The stage below is
    first dispatched, drained, sits idle for an hour (a pod roll), resumes and
    finishes in seconds. ``completed_at - started_at`` reports the hour — first
    dispatch → final completion, which is the deliberate and correct meaning of
    ``started_at`` — and is therefore useless for asking "did this handler exceed
    its budget". Two audits in one day read it as runtime and concluded a budget
    was incoherent. ``completed_at - last_execution_started_at`` answers that
    question honestly.
    """
    ceiling_seconds = 600.0
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    await repo.upsert_pending(run.id, "demo_video_record")
    await repo.mark_running(run.id, "demo_video_record")
    await db_session.commit()

    # Backdate BOTH stamps by an hour: the executor's own clock is the DB's, so
    # this is the only way to make a real elapsed span inside a fast test. It
    # stands in for "first dispatched an hour ago", not for a slow handler.
    await db_session.execute(
        text(
            "UPDATE stage_outputs SET started_at = started_at - interval '1 hour', "
            "last_execution_started_at = last_execution_started_at - interval '1 hour' "
            "WHERE run_id = :rid AND stage_name = 'demo_video_record'"
        ),
        {"rid": run.id},
    )
    await db_session.commit()
    # The session is expire_on_commit=False, so ``repo.get`` would otherwise hand
    # back the cached pre-UPDATE object and the backdate would be invisible. Expunge
    # (not expire) because an expired attribute would lazy-load outside greenlet.
    db_session.expunge_all()

    # Drained by a pod roll, then resumed by the reconciler.
    await repo.mark_pending(run.id, "demo_video_record")
    await db_session.commit()
    await repo.mark_running(run.id, "demo_video_record")
    await db_session.commit()
    assert await repo.mark_completed(run.id, "demo_video_record", {"ok": True})
    await db_session.commit()

    db_session.expunge_all()
    so = await repo.get(run.id, "demo_video_record")
    assert so is not None

    elapsed = so.elapsed_seconds
    assert elapsed is not None
    assert elapsed > 3000, (
        "the first-dispatch span should still cover the drained hour — "
        f"got {elapsed}s; started_at's semantics have been broken"
    )

    per_execution = so.last_execution_seconds
    assert per_execution is not None, (
        "no per-execution runtime is recoverable: last_execution_started_at was "
        "never stamped, so the only available number is the inflated span"
    )
    assert per_execution < ceiling_seconds, (
        "the resumed execution reports the WALL-CLOCK span "
        f"({per_execution}s) rather than its own runtime, so it appears to have "
        f"blown a {ceiling_seconds}s ceiling it never approached"
    )
    # And the two must not be the same number, or nothing has been gained.
    assert elapsed - per_execution > 3000


async def test_drained_resume_is_distinguishable_from_a_timeout_retry(db_session) -> None:
    """The two reset paths must leave DIFFERENT rows behind.

    Both a resumable timeout and a drain/cancel call ``mark_pending``, and before
    ``interruptions`` existed only the timeout path bumped a counter. So a drained
    stage resumed carrying ``attempts == 0`` — byte-identical to a stage that had
    never been interrupted at all. Since orchestrator pods roll constantly that is
    the COMMON case, which is why ``attempts == 0`` is a false filter for "this
    row's duration is trustworthy".
    """
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    # Stage A: drained by a pod roll.
    await repo.upsert_pending(run.id, "drained")
    await repo.mark_running(run.id, "drained")
    assert await repo.increment_interruptions(run.id, "drained") == 1
    await repo.mark_pending(run.id, "drained")
    await db_session.commit()

    # Stage B: a genuine resumable timeout.
    await repo.upsert_pending(run.id, "timed_out")
    await repo.mark_running(run.id, "timed_out")
    assert await repo.increment_attempts(run.id, "timed_out") == 1
    await repo.mark_pending(run.id, "timed_out")
    await db_session.commit()

    db_session.expunge_all()
    drained = await repo.get(run.id, "drained")
    timed_out = await repo.get(run.id, "timed_out")
    assert drained is not None and timed_out is not None

    assert (drained.interruptions, drained.attempts) == (1, 0), (
        "a drained stage is indistinguishable from a clean one: "
        f"interruptions={drained.interruptions} attempts={drained.attempts}"
    )
    assert (timed_out.interruptions, timed_out.attempts) == (0, 1), (
        "a timeout retry was recorded as a drain: "
        f"interruptions={timed_out.interruptions} attempts={timed_out.attempts}"
    )
    assert (drained.interruptions, drained.attempts) != (
        timed_out.interruptions,
        timed_out.attempts,
    )
    # Both rows are ambiguous to duration consumers, for two different reasons.
    assert drained.duration_is_ambiguous
    assert timed_out.duration_is_ambiguous


async def test_a_pre_0067_row_stays_ambiguous_and_is_never_backfilled(db_session) -> None:
    """NULL means "not recorded" and must never be read as "never drained".

    The column is nullable with NO server default precisely so the migration
    cannot assert, about every historical row, that it ran uninterrupted — that
    would be the ``attempts == 0`` mistake rebuilt in a new column. The INSERT
    writers seed 0 instead, so the two populations stay separable: a row created
    after 0067 is RECORDED as never-drained, a row that predates it is UNKNOWN.
    """
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)
    await repo.upsert_pending(run.id, "fresh_row")
    await repo.upsert_pending(run.id, "legacy_row")
    await db_session.commit()

    # A row as migration 0067 leaves it: the column exists, nothing filled it in.
    await db_session.execute(
        text(
            "UPDATE stage_outputs SET interruptions = NULL "
            "WHERE run_id = :rid AND stage_name = 'legacy_row'"
        ),
        {"rid": run.id},
    )
    await db_session.commit()
    db_session.expunge_all()

    fresh = await repo.get(run.id, "fresh_row")
    legacy = await repo.get(run.id, "legacy_row")
    assert fresh is not None and legacy is not None

    assert fresh.interruptions == 0, (
        "a row created after 0067 did not record its interruption count, so it is "
        "indistinguishable from a historical row whose history is genuinely unknown"
    )
    assert not fresh.duration_is_ambiguous, (
        "a recorded, never-drained, never-timed-out row was reported as ambiguous"
    )

    assert legacy.interruptions is None, (
        "the migration backfilled the column, fabricating the claim that every "
        "historical stage ran without a drain"
    )
    assert legacy.duration_is_ambiguous, (
        "an UNRECORDED row was treated as trustworthy — not-recorded is not the "
        "same as did-not-happen"
    )


async def test_resume_advances_last_execution_start_but_never_started_at(db_session) -> None:
    """The two stamps must move INDEPENDENTLY: one per execution, one per stage.

    Guards both halves at once. If ``started_at`` were re-stamped, the burnt
    execution would vanish and a retry would render as a phantom dispatch stall
    (the 1935s ``segment_select`` misread). If ``last_execution_started_at`` were
    COALESCEd like ``started_at``, it would be a duplicate of it and no
    per-execution runtime would exist at all.
    """
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    await repo.upsert_pending(run.id, "segment_select")
    await repo.mark_running(run.id, "segment_select")
    await db_session.commit()
    first = await repo.get(run.id, "segment_select")
    assert first is not None
    first_start = first.started_at
    first_exec_start = first.last_execution_started_at
    assert first_start is not None
    assert first_exec_start is not None
    assert first_exec_start == first_start, "on the FIRST execution the two coincide"

    await repo.mark_pending(run.id, "segment_select")
    await db_session.commit()
    db_session.expunge_all()
    reset = await repo.get(run.id, "segment_select")
    assert reset is not None
    assert reset.last_execution_started_at == first_exec_start, (
        "mark_pending erased the abandoned execution's start; the only record "
        "that it happened is gone"
    )

    # Make the two clocks separable — same reason as the backdate above.
    await db_session.execute(
        text(
            "UPDATE stage_outputs SET started_at = started_at - interval '1 hour', "
            "last_execution_started_at = last_execution_started_at - interval '1 hour' "
            "WHERE run_id = :rid AND stage_name = 'segment_select'"
        ),
        {"rid": run.id},
    )
    await db_session.commit()
    db_session.expunge_all()  # see the note on the sibling test above
    backdated = await repo.get(run.id, "segment_select")
    assert backdated is not None
    backdated_start = backdated.started_at
    backdated_exec_start = backdated.last_execution_started_at
    assert backdated_start is not None
    assert backdated_exec_start is not None

    await repo.mark_running(run.id, "segment_select")
    await db_session.commit()
    db_session.expunge_all()
    resumed = await repo.get(run.id, "segment_select")
    assert resumed is not None
    assert resumed.started_at == backdated_start, (
        "a resumed stage re-stamped started_at: first-dispatch semantics broken"
    )
    assert resumed.last_execution_started_at is not None
    assert resumed.last_execution_started_at > backdated_exec_start, (
        "last_execution_started_at did NOT advance on resume, so it is just a "
        "second copy of started_at and no per-execution runtime is recoverable"
    )


# --- list_for_run: DETERMINISTIC order ---------------------------------------
#
# Callers treat this list as ordered — the run-detail route serialises it into
# RunDetail.stage_outputs and Studio used to pick "the LAST stage carrying a clip
# array" out of it. Without an ORDER BY the SELECT returns physical heap order,
# so that pick flipped between requests (live run 01a08f79: clip reorder 422'd
# intermittently). created_at ALONE does not disambiguate: the rows are seeded in
# ONE transaction and share an identical timestamp, so stage_name is the key that
# actually makes the order stable.

_HEAP_ORDER = ["video_render", "storyboard_draft", "script_draft", "narrate", "brief_intake"]


async def test_list_for_run_is_sorted_by_stage_name_when_created_at_ties(db_session) -> None:
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    # Seed in ONE transaction, in REVERSE-alphabetical order. Both halves matter:
    # the single transaction makes every created_at identical (so created_at alone
    # cannot order them), and the reverse insertion order makes physical heap
    # order the exact opposite of the contracted order.
    for stage in _HEAP_ORDER:
        await repo.upsert_pending(run.id, stage)
    await db_session.commit()

    rows = await repo.list_for_run(run.id)
    assert len(rows) == len(_HEAP_ORDER)

    stamps = {r.created_at for r in rows}
    assert len(stamps) == 1, (
        f"fixture no longer ties created_at ({stamps}); this test would then pass "
        "on created_at alone and stop covering the stage_name tiebreak"
    )
    assert [r.stage_name for r in rows] == sorted(_HEAP_ORDER)
    assert [r.stage_name for r in rows] != _HEAP_ORDER  # i.e. NOT insertion/heap order


async def test_list_for_run_returns_the_same_order_on_consecutive_calls(db_session) -> None:
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)
    for stage in _HEAP_ORDER:
        await repo.upsert_pending(run.id, stage)
    await db_session.commit()

    first = [r.stage_name for r in await repo.list_for_run(run.id)]
    second = [r.stage_name for r in await repo.list_for_run(run.id)]
    assert first == second
    assert first == sorted(_HEAP_ORDER)


async def test_list_for_run_orders_by_created_at_before_stage_name(db_session) -> None:
    """created_at is the PRIMARY key of the order; stage_name only breaks ties.
    A row written later sorts last even when its name sorts first."""
    run = await _setup_run(db_session)
    repo = StageOutputRepository(db_session)

    for stage in ("script_draft", "video_render"):
        await repo.upsert_pending(run.id, stage)
    await db_session.commit()
    # Separate transaction → a strictly later now().
    await repo.upsert_pending(run.id, "aaa_written_last")
    await db_session.commit()

    rows = await repo.list_for_run(run.id)
    assert [r.stage_name for r in rows] == ["script_draft", "video_render", "aaa_written_last"]
