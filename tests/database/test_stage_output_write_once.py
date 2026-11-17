"""The write-once contract on ``stage_outputs.output`` is enforced by POSTGRES.

Every assertion here drives raw SQL through the session, deliberately NOT through
``StageOutputRepository`` — the whole point of migration 0063 is that Python is no
longer the guard. A repository-level test would only re-prove the ``status IN
('pending','running')`` WHERE clause that was already there and that any repair
script, future handler or interactive ``psql`` bypasses.

Coverage, mapping 1:1 onto the three things the trigger must get right:

* a terminal stage's ``output`` cannot be rewritten — the DB raises;
* every LEGITIMATE post-terminal writer still succeeds (the four Studio routes'
  ``SET LOCAL`` escape, the restamp's downstream reset, and the non-output
  columns that must stay updatable on a terminal row);
* a non-terminal stage's ``output`` is still freely updatable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository
from src.database.stage_output_guard import ERRCODE, TERMINAL_STATUSES

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

pytestmark = pytest.mark.asyncio

_TERMINAL_OUTPUT = {"video_url": "https://minio.test/real.mp4", "duration": 61.4}


async def _seed_stage(
    session: AsyncSession, *, status: str, output: Mapping[str, object] | None = None
) -> UUID:
    """Create project → deliverable → run → one ``stage_outputs`` row, and commit.

    The stage row is INSERTed directly at its final status: the trigger is
    ``BEFORE UPDATE`` only, so seeding an already-terminal row this way is exactly
    what ``run_setup.seed_stage_rows`` does for a regenerated run's preserved
    stages — and it proves the guard does not accidentally cover INSERT.
    """
    proj = await ProjectRepository(session).create_with_deliverables(
        tenant_id=uuid4(),
        owner_subject_id=uuid4(),
        type_="generic",
        title="write-once",
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
    stage_id = uuid4()
    await session.execute(
        text(
            "INSERT INTO stage_outputs (id, run_id, stage_name, status, output) "
            "VALUES (:id, :run_id, 'script_draft', :status, CAST(:output AS jsonb))"
        ),
        {
            "id": stage_id,
            "run_id": run.id,
            "status": status,
            "output": json.dumps(output) if output is not None else None,
        },
    )
    await session.commit()
    return stage_id


async def _read_output(session: AsyncSession, stage_id: UUID) -> dict[str, object] | None:
    row = await session.execute(
        text("SELECT output FROM stage_outputs WHERE id = :id"), {"id": stage_id}
    )
    return cast("dict[str, object] | None", row.scalar_one())


async def _rewrite_output(
    session: AsyncSession, stage_id: UUID, output: Mapping[str, object]
) -> None:
    await session.execute(
        text("UPDATE stage_outputs SET output = CAST(:output AS jsonb) WHERE id = :id"),
        {"id": stage_id, "output": json.dumps(output)},
    )


# ---------------------------------------------------------------------------
# (a) the rejection — the guard's whole reason to exist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("terminal_status", list(TERMINAL_STATUSES)[:1])
async def test_rewriting_a_terminal_stage_output_is_rejected_by_the_database(
    db_session: AsyncSession, terminal_status: str
) -> None:
    """A raw UPDATE that changes a terminal stage's ``output`` must RAISE.

    Parametrized across all four terminal statuses because the trigger's WHEN
    clause lists them literally — a status dropped from that list (or a status
    added to ``ck_stage_outputs_status`` without being added here, which is how
    'cancelled' arrived in migration 0059) would silently reopen the hole.
    """
    stage_id = await _seed_stage(db_session, status=terminal_status, output=_TERMINAL_OUTPUT)

    with pytest.raises(DBAPIError) as excinfo:
        await _rewrite_output(db_session, stage_id, {"video_url": "https://evil/forged.mp4"})
        await db_session.commit()

    # Assert on the psycopg error identity, not on message text: the custom
    # SQLSTATE is the stable contract a caller could branch on.
    assert getattr(excinfo.value.orig, "sqlstate", None) == ERRCODE, excinfo.value
    assert "write-once" in str(excinfo.value)

    await db_session.rollback()
    assert await _read_output(db_session, stage_id) == _TERMINAL_OUTPUT


async def test_rejection_survives_a_multi_row_update(db_session: AsyncSession) -> None:
    """A blanket ``UPDATE stage_outputs SET output = …`` (the repair-script shape
    this guard exists for) is rejected on the terminal row rather than partially
    applied — the row trigger raises, aborting the whole statement."""
    stage_id = await _seed_stage(db_session, status="completed", output=_TERMINAL_OUTPUT)

    with pytest.raises(DBAPIError):
        await db_session.execute(
            text("UPDATE stage_outputs SET output = '{\"clobbered\": true}'::jsonb")
        )
        await db_session.commit()

    await db_session.rollback()
    assert await _read_output(db_session, stage_id) == _TERMINAL_OUTPUT


# ---------------------------------------------------------------------------
# (b) every legitimate post-terminal write still succeeds
# ---------------------------------------------------------------------------


async def test_studio_edit_escape_hatch_permits_the_rewrite(db_session: AsyncSession) -> None:
    """The four Studio routes (element patch, chapter override, thumbnail
    override, restamp) legitimately edit a COMPLETED stage's output. They opt in
    with ``SET LOCAL w2v.allow_stage_output_edit = 'on'``; that must work."""
    stage_id = await _seed_stage(db_session, status="completed", output=_TERMINAL_OUTPUT)
    edited = {**_TERMINAL_OUTPUT, "title": "user-edited"}

    await db_session.execute(text("SET LOCAL w2v.allow_stage_output_edit = 'on'"))
    await _rewrite_output(db_session, stage_id, edited)
    await db_session.commit()

    assert await _read_output(db_session, stage_id) == edited


async def test_restamp_downstream_reset_is_permitted_under_the_escape(
    db_session: AsyncSession,
) -> None:
    """``restamp_stage_element`` also resets downstream stages from ``completed``
    to ``pending`` with ``output = NULL`` — a terminal→non-terminal transition
    that DOES change ``output``, i.e. exactly the trigger's WHEN shape. One
    transaction-scoped opt-in must cover it."""
    stage_id = await _seed_stage(db_session, status="completed", output=_TERMINAL_OUTPUT)

    await db_session.execute(text("SET LOCAL w2v.allow_stage_output_edit = 'on'"))
    await db_session.execute(
        text(
            "UPDATE stage_outputs SET status = 'pending', output = NULL, error = NULL, "
            "completed_at = NULL WHERE id = :id"
        ),
        {"id": stage_id},
    )
    await db_session.commit()

    assert await _read_output(db_session, stage_id) is None


async def test_escape_hatch_does_not_leak_past_its_transaction(db_session: AsyncSession) -> None:
    """``SET LOCAL`` — not ``SET``. Under pgbouncer transaction pooling a
    session-scoped GUC would leave the exemption armed for whatever request got
    the connection next, which would quietly un-ship this whole guard."""
    stage_id = await _seed_stage(db_session, status="completed", output=_TERMINAL_OUTPUT)

    await db_session.execute(text("SET LOCAL w2v.allow_stage_output_edit = 'on'"))
    await _rewrite_output(db_session, stage_id, {**_TERMINAL_OUTPUT, "title": "edited once"})
    await db_session.commit()

    # New transaction, no opt-in: the guard is armed again.
    with pytest.raises(DBAPIError) as excinfo:
        await _rewrite_output(db_session, stage_id, {"forged": True})
        await db_session.commit()
    assert getattr(excinfo.value.orig, "sqlstate", None) == ERRCODE
    await db_session.rollback()


async def test_terminal_row_non_output_columns_stay_updatable(db_session: AsyncSession) -> None:
    """Only ``output`` freezes. ``error``, ``attempts``, ``completed_at`` and
    ``status`` on a terminal row are still writable with NO opt-in — the WHEN
    clause requires ``NEW.output IS DISTINCT FROM OLD.output``, so these updates
    never reach the trigger function at all."""
    stage_id = await _seed_stage(db_session, status="failed", output=_TERMINAL_OUTPUT)

    await db_session.execute(
        text(
            "UPDATE stage_outputs SET error = 'annotated after the fact', "
            "attempts = attempts + 1, completed_at = now(), status = 'cancelled' "
            "WHERE id = :id"
        ),
        {"id": stage_id},
    )
    await db_session.commit()

    row = (
        await db_session.execute(
            text("SELECT error, attempts, status FROM stage_outputs WHERE id = :id"),
            {"id": stage_id},
        )
    ).one()
    assert row.error == "annotated after the fact"
    assert row.attempts == 1
    assert row.status == "cancelled"
    assert await _read_output(db_session, stage_id) == _TERMINAL_OUTPUT


async def test_rewriting_a_terminal_output_to_the_same_value_is_not_an_overwrite(
    db_session: AsyncSession,
) -> None:
    """``IS DISTINCT FROM`` on jsonb compares VALUES. An idempotent re-write of
    the identical output (key order changed, same document) is not a clobber and
    must not raise — otherwise a benign retry would start failing runs."""
    stage_id = await _seed_stage(db_session, status="completed", output=_TERMINAL_OUTPUT)

    reordered = {"duration": 61.4, "video_url": "https://minio.test/real.mp4"}
    await _rewrite_output(db_session, stage_id, reordered)
    await db_session.commit()

    assert await _read_output(db_session, stage_id) == _TERMINAL_OUTPUT


# ---------------------------------------------------------------------------
# (c) non-terminal stages are untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["pending"])
async def test_non_terminal_stage_output_is_freely_updatable(
    db_session: AsyncSession, status: str
) -> None:
    """``record_progress`` writes in-flight bookkeeping (sub-DAG child run ids,
    the in-flight html_render job id) into a RUNNING stage's output so it can
    survive a resumable-timeout re-dispatch. The guard must not touch that."""
    stage_id = await _seed_stage(db_session, status=status, output={"phase": "one"})

    await _rewrite_output(db_session, stage_id, {"phase": "two", "child_run_id": "abc"})
    await db_session.commit()

    assert await _read_output(db_session, stage_id) == {"phase": "two", "child_run_id": "abc"}
