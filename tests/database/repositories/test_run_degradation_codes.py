"""Which runs were degraded by code X — `RunRepository.degraded_codes_by_run`.

`degraded_run_ids` answers "was this run degraded?". Nine stable codes exist,
all nine are emitted, and nothing could ask WHICH — so "which runs shipped
without their music bed last week" had no answer short of loading every stage
blob and grepping it. This is the by-code sibling, at the same cost: one
aggregate query over `idx_stage_outputs_run` with `degraded_stage_clause()`
pushed into the WHERE, projecting ONLY the `degradations` subtree out of the
JSONB — never the multi-MB `output` payload.

Postgres 18 + psycopg 3.x throughout (no asyncpg, no SQLite): the JSONB subtree
projection and the falsey-value predicate are both evaluated in the server, so
these tests need the real database.

⚠️ Nothing here arms anything — no gate, no alert, no assertion that any code is
a failure. Exposure only.
"""

from __future__ import annotations

from uuid import uuid4, uuid7

import pytest
from sqlalchemy import event, text
from sqlalchemy.engine import Engine

from src.database.models import StageOutput
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository
from src.services.stages.shared.degradation import (
    CAPTIONS_MISSING,
    MASTERING_GATE_FAILED,
    SOUND_BED_MISSING,
    degradation,
)

pytestmark = pytest.mark.asyncio


async def _make_run(session):
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


async def _add_stage(session, run_id, stage_name, output, *, status="completed", started_at=None):
    session.add(
        StageOutput(
            id=uuid7(),
            run_id=run_id,
            stage_name=stage_name,
            status=status,
            output=output,
            started_at=started_at,
        )
    )
    await session.commit()


async def test_codes_come_back_per_run_in_recorded_order(db_session) -> None:
    """Two distinct degradations on one stage must stay two distinct facts, in
    the order the stage recorded them."""
    run = await _make_run(db_session)
    await _add_stage(
        db_session,
        run.id,
        "compile",
        {
            "degradations": [
                degradation(SOUND_BED_MISSING, "bed did not resolve"),
                degradation(CAPTIONS_MISSING, "no subtitles"),
            ]
        },
    )

    assert await RunRepository(db_session).degraded_codes_by_run([run.id]) == {
        run.id: [SOUND_BED_MISSING, CAPTIONS_MISSING]
    }


async def test_codes_from_several_stages_merge_in_stage_start_order(db_session) -> None:
    """A run's codes span its stages, ordered by when each stage STARTED — so the
    same run always reports the same list, not whatever order the planner
    happened to return rows in."""
    run = await _make_run(db_session)
    later = text("now() + interval '10 minutes'")
    await _add_stage(
        db_session,
        run.id,
        "encode_master",
        {"degradations": [degradation(MASTERING_GATE_FAILED, "gate_pass=false")]},
        started_at=later,
    )
    await _add_stage(
        db_session,
        run.id,
        "narrate",
        {"degradations": [degradation(SOUND_BED_MISSING, "no bed")]},
        started_at=text("now()"),
    )

    assert await RunRepository(db_session).degraded_codes_by_run([run.id]) == {
        run.id: [SOUND_BED_MISSING, MASTERING_GATE_FAILED]
    }


async def test_a_code_raised_by_two_stages_is_reported_once(db_session) -> None:
    """One code twice is one fact — de-duplicated at its FIRST position, which a
    `set` would also do while losing the position."""
    run = await _make_run(db_session)
    await _add_stage(
        db_session,
        run.id,
        "narrate",
        {"degradations": [degradation(SOUND_BED_MISSING, "a")]},
        started_at=text("now()"),
    )
    await _add_stage(
        db_session,
        run.id,
        "encode_master",
        {
            "degradations": [
                degradation(CAPTIONS_MISSING, "b"),
                degradation(SOUND_BED_MISSING, "c — same fact"),
            ]
        },
        started_at=text("now() + interval '5 minutes'"),
    )

    assert await RunRepository(db_session).degraded_codes_by_run([run.id]) == {
        run.id: [SOUND_BED_MISSING, CAPTIONS_MISSING]
    }


async def test_a_clean_run_is_absent_and_a_notes_only_run_manufactures_no_code(
    db_session,
) -> None:
    """FALSE-POSITIVE GUARD plus the additive seam.

    A clean run must not appear. A notes-only run IS degraded — `degraded_run_ids`
    still flags it — but the legacy prose channel has no stable identity, so it
    must not be given a code it never recorded. Those two together are what make
    this mapping's keys mean "degraded AND queryable" rather than "degraded".
    """
    repo = RunRepository(db_session)
    clean = await _make_run(db_session)
    await _add_stage(db_session, clean.id, "script_draft", {"script": "ok", "notes": []})
    noted = await _make_run(db_session)
    await _add_stage(db_session, noted.id, "media_discovery", {"notes": ["source unavailable"]})

    # A row whose `degradations` list is non-empty but carries NOTHING reportable
    # (no `code` key at all — hand-written, or a shape from a build that predates
    # the current entry). `degraded_stage_clause` matches it, so it would silently
    # acquire a `[]` entry and inflate `len(...)` past the real degraded-by-code
    # count unless it is dropped after the scan.
    codeless = await _make_run(db_session)
    await _add_stage(db_session, codeless.id, "compile", {"degradations": [{"detail": "no code"}]})

    assert await repo.degraded_codes_by_run([clean.id, noted.id, codeless.id]) == {}
    # ...while the boolean sibling still reports both degraded runs as degraded:
    # this method narrows the answer, it does not change it.
    assert await repo.degraded_run_ids([clean.id, noted.id, codeless.id]) == {
        noted.id,
        codeless.id,
    }


async def test_only_completed_stages_of_the_requested_runs_contribute(db_session) -> None:
    """Same WHERE as `degraded_stage_clause`: a non-completed stage is not a
    shipped degradation, and a run nobody asked about must not leak into the
    answer."""
    repo = RunRepository(db_session)
    asked = await _make_run(db_session)
    await _add_stage(
        db_session,
        asked.id,
        "compile",
        {"degradations": [degradation(CAPTIONS_MISSING, "d")]},
        status="failed",
    )
    other = await _make_run(db_session)
    await _add_stage(
        db_session,
        other.id,
        "compile",
        {"degradations": [degradation(MASTERING_GATE_FAILED, "d")]},
    )

    assert await repo.degraded_codes_by_run([asked.id]) == {}
    assert await repo.degraded_codes_by_run([]) == {}


async def test_the_output_blob_is_never_selected(db_session) -> None:
    """COST GUARD. `degraded_run_ids` exists because the list paths must not load
    stage payloads; its by-code sibling inherits that constraint. The SELECT list
    may reference `output` only to project the `degradations` subtree — never as
    a whole column, or a page of runs ships megabytes to answer a nine-value
    question."""
    run = await _make_run(db_session)
    await _add_stage(
        db_session,
        run.id,
        "compile",
        {
            "huge": "x" * 4096,
            "degradations": [degradation(CAPTIONS_MISSING, "d")],
        },
    )

    statements: list[str] = []

    def _record(conn, cursor, statement, parameters, context, executemany) -> None:
        if "stage_outputs" in statement.lower():
            statements.append(statement)

    event.listen(Engine, "before_cursor_execute", _record)
    try:
        result = await RunRepository(db_session).degraded_codes_by_run([run.id])
    finally:
        event.remove(Engine, "before_cursor_execute", _record)

    assert result == {run.id: [CAPTIONS_MISSING]}
    assert len(statements) == 1, f"expected ONE aggregate query, got {len(statements)}"
    select_list = statements[0].lower().split(" from ")[0]
    assert "degradations" in select_list, statements[0]
    assert "stage_outputs.output \n" not in select_list, statements[0]
    assert "stage_outputs.output," not in select_list, statements[0]
    assert "stage_outputs.output " not in select_list, statements[0]
