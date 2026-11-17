"""``StageStepRepository`` + the step ledger against REAL postgres.

Covers what the in-memory unit tests (``tests/services/stages/shared/
test_step_ledger.py``) cannot: the upsert, the cascade, the JSONB round trip, the
session-per-operation store, and — under a throwaway non-owner ``NOBYPASSRLS``
role — that the ``tenant_isolation`` policy actually isolates ledger rows.
"""

from __future__ import annotations

from uuid import UUID, uuid7

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from src.database.models import StageStepResult
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository
from src.database.repositories.stage_step_repository import StageStepRepository
from src.services.stages.shared.step_ledger import (
    SessionStepStore,
    StepLedger,
    step_ledger_factory,
)
from tests.conftest import require_createrole

pytestmark = pytest.mark.asyncio


async def _make_run(session, tenant: UUID | None = None) -> UUID:
    """project → deliverable → run for ``tenant``; committed. Returns the run id."""
    proj = await ProjectRepository(session).create_with_deliverables(
        tenant_id=tenant or uuid7(),
        owner_subject_id=uuid7(),
        type_="generic",
        title="step-ledger",
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
    return run.id


async def _row_count(session) -> int:
    return (await session.execute(text("SELECT count(*) FROM stage_step_results"))).scalar_one()


# ── repository ───────────────────────────────────────────────────────────────


async def test_get_returns_none_for_a_step_that_never_finished(db_session) -> None:
    run_id = await _make_run(db_session)

    assert await StageStepRepository(db_session).get(run_id, "narrate", "scene-1") is None


async def test_put_then_get_round_trips_the_jsonb_payload(db_session) -> None:
    run_id = await _make_run(db_session)
    repo = StageStepRepository(db_session)
    payload = {"value": {"url": "s3://a.wav", "seconds": 4.25, "tags": ["a", None], "ok": True}}

    await repo.put(run_id, "narrate", "scene-1", "hash-1", payload)
    await db_session.commit()

    row = await repo.get(run_id, "narrate", "scene-1")
    assert isinstance(row, StageStepResult)
    assert (row.run_id, row.stage_name, row.step_key) == (run_id, "narrate", "scene-1")
    assert row.input_hash == "hash-1"
    assert row.result == payload
    assert row.created_at is not None
    assert row.updated_at is not None


async def test_put_on_the_same_key_replaces_hash_and_result_in_place(db_session) -> None:
    run_id = await _make_run(db_session)
    repo = StageStepRepository(db_session)
    await repo.put(run_id, "narrate", "scene-1", "old-hash", {"value": "old"})
    await db_session.commit()
    first = await repo.get(run_id, "narrate", "scene-1")
    assert first is not None
    first_id, first_created, first_updated = first.id, first.created_at, first.updated_at

    await repo.put(run_id, "narrate", "scene-1", "new-hash", {"value": "new"})
    await db_session.commit()
    db_session.expire_all()

    second = await repo.get(run_id, "narrate", "scene-1")
    assert second is not None
    assert (second.input_hash, second.result) == ("new-hash", {"value": "new"})
    assert second.id == first_id, "the upsert must UPDATE the row, not insert a sibling"
    assert second.created_at == first_created
    assert second.updated_at > first_updated, "updated_at must be bumped by the upsert"
    assert await _row_count(db_session) == 1


async def test_rows_are_keyed_by_run_stage_and_step(db_session) -> None:
    run_a = await _make_run(db_session)
    run_b = await _make_run(db_session)
    repo = StageStepRepository(db_session)

    await repo.put(run_a, "narrate", "k", "h", {"value": "a/narrate/k"})
    await repo.put(run_a, "narrate", "k2", "h", {"value": "a/narrate/k2"})
    await repo.put(run_a, "compile", "k", "h", {"value": "a/compile/k"})
    await repo.put(run_b, "narrate", "k", "h", {"value": "b/narrate/k"})
    await db_session.commit()

    assert await _row_count(db_session) == 4
    got = await repo.get(run_b, "narrate", "k")
    assert got is not None
    assert got.result == {"value": "b/narrate/k"}
    assert await repo.get(run_b, "compile", "k") is None


async def test_deleting_the_run_cascades_its_ledger_rows(db_session) -> None:
    doomed = await _make_run(db_session)
    kept = await _make_run(db_session)
    repo = StageStepRepository(db_session)
    await repo.put(doomed, "narrate", "k", "h", {"value": 1})
    await repo.put(kept, "narrate", "k", "h", {"value": 2})
    await db_session.commit()

    await db_session.execute(text("DELETE FROM runs WHERE id = :id"), {"id": doomed})
    await db_session.commit()
    db_session.expire_all()

    assert await repo.get(doomed, "narrate", "k") is None
    assert await repo.get(kept, "narrate", "k") is not None
    assert await _row_count(db_session) == 1


async def test_put_does_not_commit_the_callers_session(db_session, db_session_factory) -> None:
    """Same convention as ``StageOutputRepository``: the caller owns the transaction."""
    run_id = await _make_run(db_session)
    await StageStepRepository(db_session).put(run_id, "narrate", "k", "h", {"value": 1})

    async with db_session_factory() as other:
        assert await StageStepRepository(other).get(run_id, "narrate", "k") is None
    await db_session.rollback()
    assert await _row_count(db_session) == 0


# ── ledger over the real store ───────────────────────────────────────────────


async def test_ledger_survives_a_process_boundary(db_session, db_session_factory) -> None:
    """The durable claim, end to end: a result stored by one ledger is served to a
    DIFFERENT ledger / store / session — which is all a re-entered stage on a new pod
    has in common with the one that died."""
    run_id = await _make_run(db_session)
    calls: list[str] = []

    async def _paid() -> dict:
        calls.append("paid")
        return {"url": "s3://clip.mp4", "seconds": 3.5}

    first = step_ledger_factory(db_session_factory, run_id=run_id, tenant_id=None)("render")
    assert await first.once("clip-1", "h1", _paid) == {"url": "s3://clip.mp4", "seconds": 3.5}

    reentered = StepLedger(SessionStepStore(db_session_factory, None), run_id, "render")
    assert await reentered.once("clip-1", "h1", _paid) == {"url": "s3://clip.mp4", "seconds": 3.5}
    assert calls == ["paid"], "the re-entered stage repeated a finished, paid sub-step"

    # A changed input is NOT served the old answer, and replaces the row.
    assert await reentered.once("clip-1", "h2", _paid) == {"url": "s3://clip.mp4", "seconds": 3.5}
    assert calls == ["paid", "paid"]
    row = await StageStepRepository(db_session).get(run_id, "render", "clip-1")
    assert row is not None
    assert row.input_hash == "h2"
    assert row.result == {"value": {"url": "s3://clip.mp4", "seconds": 3.5}}
    assert await _row_count(db_session) == 1


async def test_a_none_result_is_stored_and_served(db_session, db_session_factory) -> None:
    """``result`` is NOT NULL; the envelope is what lets a ``None`` result be a row."""
    run_id = await _make_run(db_session)
    calls: list[int] = []

    async def _returns_none() -> None:
        calls.append(1)

    ledger = StepLedger(SessionStepStore(db_session_factory, None), run_id, "probe")
    assert await ledger.once("k", "h", _returns_none) is None
    assert await ledger.once("k", "h", _returns_none) is None

    assert calls == [1]
    row = await StageStepRepository(db_session).get(run_id, "probe", "k")
    assert row is not None
    assert row.result == {"value": None}


async def test_ledger_write_for_a_run_that_does_not_exist_never_fails_the_stage(
    db_session, db_session_factory
) -> None:
    """A real DB error on the write path (FK violation) is absorbed: the step's
    result still comes back and nothing is stored."""
    ghost_run = uuid7()
    ledger = StepLedger(SessionStepStore(db_session_factory, None), ghost_run, "narrate")

    async def _paid() -> str:
        return "result"

    assert await ledger.once("k", "h", _paid) == "result"
    assert await _row_count(db_session) == 0


# ── RLS ──────────────────────────────────────────────────────────────────────


async def test_tenant_isolation_policy_isolates_ledger_rows(db_session_factory) -> None:
    """Under a non-owner NOBYPASSRLS role (the prod runtime posture): a tenant sees
    only its own runs' ledger rows, an unbound session sees none, and neither a read,
    an upsert nor a delete can reach another tenant's row."""
    role = f"rls_ledger_{uuid7().hex[:12]}"

    async with db_session_factory() as owner:
        schema = (await owner.execute(text("SELECT current_schema()"))).scalar_one()
        try:
            await owner.execute(text(f'CREATE ROLE "{role}" NOLOGIN NOSUPERUSER NOBYPASSRLS'))
        except ProgrammingError as exc:  # insufficient_privilege → no CREATEROLE
            await owner.rollback()
            require_createrole(exc)  # fail under CI, skip for a locally-restricted role
        await owner.execute(text(f'GRANT "{role}" TO CURRENT_USER'))
        await owner.execute(text(f'GRANT USAGE ON SCHEMA "{schema}" TO "{role}"'))
        await owner.execute(
            text(
                f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA "{schema}" TO "{role}"'
            )
        )
        await owner.commit()

    async def _as_tenant(session, tenant: UUID | None) -> None:
        await session.execute(text(f'SET LOCAL ROLE "{role}"'))
        if tenant is not None:
            await session.execute(
                text("SELECT set_config('app.current_tenant', :t, true)"), {"t": str(tenant)}
            )

    try:
        tenant_a, tenant_b = uuid7(), uuid7()
        async with db_session_factory() as owner:
            run_a = await _make_run(owner, tenant_a)
            run_b = await _make_run(owner, tenant_b)
            repo = StageStepRepository(owner)
            await repo.put(run_a, "narrate", "k", "h", {"value": "A"})
            await repo.put(run_b, "narrate", "k", "h", {"value": "B"})
            await owner.commit()

        # 1. Bound to A: A's row only — B's is invisible through the repository.
        async with db_session_factory() as s:
            await _as_tenant(s, tenant_a)
            repo = StageStepRepository(s)
            mine = await repo.get(run_a, "narrate", "k")
            assert mine is not None
            assert mine.result == {"value": "A"}
            assert await repo.get(run_b, "narrate", "k") is None
            assert await _row_count(s) == 1
            await s.rollback()

        # 2. Unbound: fail-closed, zero rows.
        async with db_session_factory() as s:
            await _as_tenant(s, None)
            assert await _row_count(s) == 0
            assert await StageStepRepository(s).get(run_a, "narrate", "k") is None
            await s.rollback()

        # 3. Bound to A, a NEW step on B's run is rejected by WITH CHECK.
        async with db_session_factory() as s:
            await _as_tenant(s, tenant_a)
            with pytest.raises(ProgrammingError) as caught:
                await StageStepRepository(s).put(run_b, "narrate", "new", "h", {"value": "evil"})
            assert "row-level security" in str(caught.value.orig).lower()
            await s.rollback()

        # 4. Bound to A, an upsert onto B's EXISTING key must not overwrite it either.
        async with db_session_factory() as s:
            await _as_tenant(s, tenant_a)
            with pytest.raises(ProgrammingError) as caught:
                await StageStepRepository(s).put(run_b, "narrate", "k", "h2", {"value": "evil"})
            assert "row-level security" in str(caught.value.orig).lower()
            await s.rollback()

        # 5. Bound to A, a DELETE cannot reach B's row.
        async with db_session_factory() as s:
            await _as_tenant(s, tenant_a)
            deleted = await s.execute(
                text("DELETE FROM stage_step_results WHERE run_id = :r"), {"r": run_b}
            )
            assert deleted.rowcount == 0
            await s.rollback()

        # 6. Bound to A, A's own writes go through (the policy is not just a wall).
        async with db_session_factory() as s:
            await _as_tenant(s, tenant_a)
            await StageStepRepository(s).put(run_a, "narrate", "k", "h2", {"value": "A2"})
            await s.commit()

        async with db_session_factory() as owner:
            repo = StageStepRepository(owner)
            a = await repo.get(run_a, "narrate", "k")
            b = await repo.get(run_b, "narrate", "k")
            assert a is not None
            assert b is not None
            assert (a.input_hash, a.result) == ("h2", {"value": "A2"})
            assert (b.input_hash, b.result) == ("h", {"value": "B"}), "tenant B's row was touched"
    finally:
        async with db_session_factory() as owner:
            for stmt in (
                f'REVOKE "{role}" FROM CURRENT_USER',
                f'DROP OWNED BY "{role}"',
                f'DROP ROLE IF EXISTS "{role}"',
            ):
                try:
                    await owner.execute(text(stmt))
                    await owner.commit()
                except ProgrammingError:
                    await owner.rollback()
