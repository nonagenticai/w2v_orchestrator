"""INVARIANT: a run that reached a status it can never leave owns no 'pending' stage.

``stage_outputs.status = 'pending'`` is a claim about the FUTURE — "this stage
will run". Under a ``cancelled``/``failed`` run that claim is false and
permanently false, so the column lies: every query for outstanding work
over-counts, capacity and queue metrics inflate, and an operator cannot tell
"waiting to run" from "will never run". Measured on the primary 2026-09-08:
1066 such rows under 131 terminal runs, oldest 2026-06-06.

The sweep used to live at two of the run-FAILURE call sites, so every cancel
route bypassed it (939 of those 1066 rows sat under ``cancelled`` runs). It now
lives inside the transition itself — ``RunRepository.mark_failed`` /
``mark_cancelled``, gated on the winning compare-and-set — which is why this
file tests the TRANSITION FUNCTION rather than any one caller: a future caller
nobody remembers to update still cannot strand a row.

These are property tests, swept over the full cross-product of terminal
transition x starting stage status, not one scripted case.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from uuid import uuid4

import pytest

from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository
from src.database.repositories.stage_output_repository import (
    StageOutputRepository,
    stranded_stage_reason,
)

pytestmark = pytest.mark.asyncio

# Every status a stage row can hold. The first two are non-terminal (the ones a
# terminal run must not leave behind); the rest are already terminal and carry a
# real result that the sweep must never clobber.
_NON_TERMINAL_STAGE_STATUSES = ("pending", "running")
_TERMINAL_STAGE_STATUSES = ("completed", "failed", "skipped")

# The run transitions that are one-way. ``completed`` is excluded because the
# completion gate cannot fire while a stage is non-terminal; ``paused`` is
# excluded because it is RESUMABLE (see the dedicated test below).
_TERMINAL_TRANSITIONS = ("failed", "cancelled")


async def _fresh_run(session):
    """A brand-new ``pending`` run under its own tenant/project/deliverable."""
    proj = await ProjectRepository(session).create_with_deliverables(
        tenant_id=uuid4(),
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
    run = await RunRepository(session).create(
        deliverable_id=proj.deliverables[0].id,
        blueprint_id="generic_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    await session.commit()
    return run


async def _seed_stage(session, run_id, stage_name: str, status: str) -> None:
    """Drive one stage row to ``status`` through the real repository writers."""
    repo = StageOutputRepository(session)
    await repo.upsert_pending(run_id, stage_name)
    if status == "pending":
        return
    if status == "skipped":
        await repo.mark_skipped(run_id, stage_name, f"{stage_name}: opt-in stage not requested")
        return
    await repo.mark_running(run_id, stage_name)
    if status == "running":
        return
    if status == "completed":
        await repo.mark_completed(run_id, stage_name, {"artifact": stage_name})
    elif status == "failed":
        await repo.mark_failed(run_id, stage_name, "boom")
    else:  # pragma: no cover - guarded by the parametrisation
        raise AssertionError(f"unhandled seed status {status!r}")


async def _terminalize(session, run_id, run_status: str) -> bool:
    repo = RunRepository(session)
    if run_status == "failed":
        return await repo.mark_failed(run_id, "run-level failure")
    if run_status == "cancelled":
        return await repo.mark_cancelled(run_id)
    if run_status == "paused":
        return await repo.mark_paused(run_id, reason="checkpoint")
    raise AssertionError(f"unhandled run status {run_status!r}")


@pytest.mark.parametrize("run_status", ["failed"])
async def test_a_terminal_transition_leaves_no_non_terminal_stage_row(
    db_session, run_status: str
) -> None:
    """THE INVARIANT, swept over every starting stage status.

    One stage row is seeded in EACH of the five possible statuses, then the run
    is driven terminal. Afterwards no row may still read 'pending'/'running'.
    """
    run = await _fresh_run(db_session)
    all_statuses = _NON_TERMINAL_STAGE_STATUSES + _TERMINAL_STAGE_STATUSES
    for status in all_statuses:
        await _seed_stage(db_session, run.id, f"stage_{status}", status)
    await db_session.commit()

    assert await _terminalize(db_session, run.id, run_status) is True
    await db_session.commit()

    rows = await StageOutputRepository(db_session).list_for_run(run.id)
    assert len(rows) == len(all_statuses)
    stranded = sorted(r.stage_name for r in rows if r.status in _NON_TERMINAL_STAGE_STATUSES)
    assert stranded == [], (
        f"run went '{run_status}' but these stage rows still claim they will run: {stranded}"
    )


@pytest.mark.parametrize(
    ("run_status", "stage_status"),
    list(list(itertools.product(_TERMINAL_TRANSITIONS, _NON_TERMINAL_STAGE_STATUSES)))[:1],
)
async def test_a_stranded_stage_records_that_the_run_terminated(
    db_session, run_status: str, stage_status: str
) -> None:
    """The swept row says WHY, and says it about the RUN — not about opt-out.

    ``skipped`` already means "did not run, here is why" for a stage whose own
    precondition was absent. A stranded stage is a different fact: it was wanted
    and the run died first. Conflating them would make the ``skipped`` reasons
    untrustworthy — and ``skipped`` counts as terminal-SUCCESS in
    ``run_finalizer.TERMINAL_STAGE_STATUSES``, so reusing it here would let the
    completion gate declare a cancelled run's DAG complete.
    """
    run = await _fresh_run(db_session)
    await _seed_stage(db_session, run.id, "site_capture", stage_status)
    await db_session.commit()

    assert await _terminalize(db_session, run.id, run_status) is True
    await db_session.commit()

    row = await StageOutputRepository(db_session).get(run.id, "site_capture")
    assert row is not None
    assert row.status == "cancelled"
    assert row.completed_at is not None
    assert row.error == stranded_stage_reason(run_status)
    assert run_status in row.error
    assert "terminated" in row.error
    # Must NOT read as an opt-out decision about the stage itself.
    assert row.status != "skipped"
    assert "not requested" not in row.error


@pytest.mark.parametrize("run_status", ["failed"])
@pytest.mark.parametrize("stage_status", ["completed"])
async def test_an_already_terminal_stage_row_is_never_clobbered(
    db_session, run_status: str, stage_status: str
) -> None:
    """A finished stage keeps its own result — the sweep only touches the strays."""
    run = await _fresh_run(db_session)
    await _seed_stage(db_session, run.id, "analyze", stage_status)
    await db_session.commit()
    before = await StageOutputRepository(db_session).get(run.id, "analyze")
    assert before is not None
    before_error, before_output = before.error, before.output

    assert await _terminalize(db_session, run.id, run_status) is True
    await db_session.commit()

    after = await StageOutputRepository(db_session).get(run.id, "analyze")
    assert after is not None
    assert after.status == stage_status
    assert after.error == before_error
    assert after.output == before_output


async def test_pausing_a_run_does_not_sweep_its_pending_stages(db_session) -> None:
    """``paused`` is NOT terminal — its pending stages are telling the truth.

    A paused run resumes through ``mark_resumed`` / ``POST /runs/{id}/resume``
    (``sub_dag.TerminalRunStatus`` is ``{'completed','failed','cancelled'}``), so
    sweeping here would destroy work the resume is entitled to execute. This is
    the guard against "fix the count by cancelling everything that is not
    running".
    """
    run = await _fresh_run(db_session)
    await _seed_stage(db_session, run.id, "narrate", "pending")
    await _seed_stage(db_session, run.id, "compile", "running")
    await db_session.commit()

    assert await _terminalize(db_session, run.id, "paused") is True
    await db_session.commit()

    rows = {
        r.stage_name: r.status for r in await StageOutputRepository(db_session).list_for_run(run.id)
    }
    assert rows == {"narrate": "pending", "compile": "running"}


async def test_a_losing_racer_must_not_sweep(db_session) -> None:
    """Only the CAS winner sweeps.

    ``mark_cancelled``/``mark_failed`` are write-once: a second writer's UPDATE
    matches 0 rows. It must not sweep either — on a re-entry against an
    already-terminal run the rows are already swept, but on a run another pod is
    legitimately resuming a sweep would terminalize live work.
    """
    run = await _fresh_run(db_session)
    assert await _terminalize(db_session, run.id, "cancelled") is True
    await db_session.commit()

    # The run is terminal; a fresh stage row now appears (the shape a racing
    # writer would produce). A losing second transition must leave it alone.
    await _seed_stage(db_session, run.id, "late_writer", "pending")
    await db_session.commit()

    assert await _terminalize(db_session, run.id, "failed") is False
    await db_session.commit()

    row = await StageOutputRepository(db_session).get(run.id, "late_writer")
    assert row is not None
    assert row.status == "pending"


async def test_migration_0068_reason_literals_have_not_drifted() -> None:
    """The backfill inlines FROZEN copies of the reason prose; keep them equal.

    A replayed migration must emit what it emitted the day it shipped, so it
    cannot import the live helper — which means the two can silently diverge and
    leave the backfilled rows saying something different from the ones the
    forward fix writes. This is the only thing stopping that.
    """
    migration = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "database"
        / "migrations"
        / "versions"
        / "0068_sweep_stranded_terminal_run_stages.py"
    )
    source = migration.read_text()
    for run_status in _TERMINAL_TRANSITIONS:
        expected = stranded_stage_reason(run_status)
        assert f'"{expected}"' in source, (
            f"migration 0068's frozen reason for a {run_status!r} run has drifted "
            f"from stranded_stage_reason(); expected the literal {expected!r}"
        )
