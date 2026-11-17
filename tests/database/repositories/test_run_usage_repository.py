import datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select, update

from src.database.models import Run
from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository
from src.database.repositories.run_usage_repository import RunUsageRepository

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


async def _tenant_of(session, run) -> UUID:
    from src.database.models import Deliverable, Project

    return (
        await session.execute(
            select(Project.tenant_id)
            .join(Deliverable, Deliverable.project_id == Project.id)
            .where(Deliverable.id == run.deliverable_id)
        )
    ).scalar_one()


async def _backdate_updated_at(session, run_id, *, minutes: int = 2) -> None:
    """Push a run's last state change past the sweep's cost-flush grace."""
    await session.execute(
        update(Run)
        .where(Run.id == run_id)
        .values(
            updated_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=minutes)
        )
    )
    await session.commit()


async def test_replace_and_list_for_run(db_session) -> None:
    run = await _setup_run(db_session)
    repo = RunUsageRepository(db_session)

    await repo.replace_for_run(
        run_id=run.id,
        rows=[
            {
                "stage_name": "analyze",
                "model": "deepseek-v3",
                "provider": "deepseek",
                "input_tokens": 1000,
                "output_tokens": 500,
                "cost_usd": "0.015000",
                "metadata": {"request_id": "abc123"},
            }
        ],
    )
    await db_session.commit()

    rows = await repo.list_for_run(run.id)
    assert len(rows) == 1
    row = rows[0]
    assert row.run_id == run.id
    assert row.stage_name == "analyze"
    assert row.model == "deepseek-v3"
    assert row.provider == "deepseek"
    assert row.input_tokens == 1000
    assert row.output_tokens == 500
    assert float(row.cost_usd) == pytest.approx(0.015, rel=1e-4)
    # The mirror keeps the row's metadata and stamps its own source on it (0105).
    assert row.metadata_ == {"request_id": "abc123", "source": "billing_mirror"}


async def test_mirrored_rows_are_attributed_to_the_run_subject(db_session) -> None:
    """P6 ORCH-3: a billing-mirrored row belongs to the subject pinned in the run's
    snapshot and has no tenant slot; tokens become units. Its funding is NULL: billing's
    per-run cost cannot tell platform-paid from tenant BYO, so labelling it (the old
    'platform') would mislabel BYO spend — only a row that states a taxonomy value
    keeps it."""
    run = await _setup_run(db_session)
    subject = uuid4()
    await db_session.execute(
        update(Run).where(Run.id == run.id).values(config_snapshot={"subject_id": str(subject)})
    )
    await RunUsageRepository(db_session).replace_for_run(
        run_id=run.id,
        rows=[
            {"stage_name": "a", "input_tokens": 10, "output_tokens": 5, "cost_usd": "0.1"},
            {"stage_name": "narrate", "input_tokens": None, "output_tokens": None},
            {"stage_name": "gpu", "funding": "local_gpu"},
            {"stage_name": "bogus", "funding": "platform"},
        ],
    )
    await db_session.commit()
    rows = {r.stage_name: r for r in await RunUsageRepository(db_session).list_for_run(run.id)}
    assert {r.subject_id for r in rows.values()} == {subject}
    assert {n: r.funding for n, r in rows.items()} == {
        "a": None,
        "narrate": None,
        "gpu": "local_gpu",
        "bogus": None,
    }
    assert {r.slot for r in rows.values()} == {None}
    assert (rows["a"].units, rows["a"].unit_kind) == (15, "tokens")
    assert (rows["narrate"].units, rows["narrate"].unit_kind) == (None, None)


async def test_mirrored_rows_of_a_run_without_a_subject_stay_unattributed(db_session) -> None:
    run = await _setup_run(db_session)  # config_snapshot={}
    await RunUsageRepository(db_session).replace_for_run(
        run_id=run.id, rows=[{"stage_name": "a", "input_tokens": 1, "output_tokens": 1}]
    )
    await db_session.commit()
    (row,) = await RunUsageRepository(db_session).list_for_run(run.id)
    assert row.subject_id is None and row.funding is None


async def test_total_sums_correctly(db_session) -> None:
    run = await _setup_run(db_session)
    repo = RunUsageRepository(db_session)

    await repo.replace_for_run(
        run_id=run.id,
        rows=[
            {
                "stage_name": "analyze",
                "model": "deepseek-v3",
                "provider": "deepseek",
                "input_tokens": 1000,
                "output_tokens": 200,
                "cost_usd": "0.010000",
            },
            {
                "stage_name": "script_draft",
                "model": "deepseek-v3",
                "provider": "deepseek",
                "input_tokens": 2000,
                "output_tokens": 800,
                "cost_usd": "0.020000",
            },
        ],
    )
    await db_session.commit()

    totals = await repo.total(run.id)
    assert totals["input_tokens"] == 3000
    assert totals["output_tokens"] == 1000
    assert totals["cost_usd"] == pytest.approx(0.03, rel=1e-4)
    # cost_usd should be a plain float
    assert isinstance(totals["cost_usd"], float)


async def test_total_returns_zeros_for_empty_run(db_session) -> None:
    run = await _setup_run(db_session)
    repo = RunUsageRepository(db_session)

    totals = await repo.total(run.id)
    assert totals["input_tokens"] == 0
    assert totals["output_tokens"] == 0
    assert totals["cost_usd"] == 0.0


async def _second_run(session, run):
    run2 = await RunRepository(session).create(
        deliverable_id=run.deliverable_id,
        blueprint_id="generic_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    await session.commit()
    return run2


def _usage(stage: str, inp: int, out: int) -> dict:
    return {
        "stage_name": stage,
        "model": "deepseek-v3",
        "provider": "deepseek",
        "input_tokens": inp,
        "output_tokens": out,
        "cost_usd": "0.001000",
    }


async def test_totals_for_runs_groups_per_run_and_omits_runs_without_usage(db_session) -> None:
    run_a = await _setup_run(db_session)
    run_b = await _second_run(db_session, run_a)
    run_empty = await _second_run(db_session, run_a)
    repo = RunUsageRepository(db_session)
    await repo.replace_for_run(
        run_id=run_a.id, rows=[_usage("analyze", 1000, 200), _usage("script_draft", 2000, 800)]
    )
    await repo.replace_for_run(run_id=run_b.id, rows=[_usage("analyze", 7, 3)])
    await db_session.commit()

    totals = await repo.totals_for_runs([run_a.id, run_b.id, run_empty.id, uuid4()])

    assert totals == {run_a.id: (3000, 1000), run_b.id: (7, 3)}
    # A run with no usage rows is ABSENT (callers default it to (0, 0)), not (0, 0).
    assert run_empty.id not in totals
    assert totals.get(run_empty.id, (0, 0)) == (0, 0)
    # Plain ints, not Decimal from SUM().
    assert all(type(v) is int for pair in totals.values() for v in pair)


async def test_totals_for_runs_empty_input_does_not_query(db_session, monkeypatch) -> None:
    repo = RunUsageRepository(db_session)

    async def _boom(*_a, **_k):
        raise AssertionError("totals_for_runs([]) must not query")

    monkeypatch.setattr(db_session, "execute", _boom)
    assert await repo.totals_for_runs([]) == {}


async def test_list_for_run_returns_empty_for_unknown_run(db_session) -> None:
    repo = RunUsageRepository(db_session)
    rows = await repo.list_for_run(uuid4())
    assert rows == []


async def test_replace_for_run_is_idempotent(db_session) -> None:
    run = await _setup_run(db_session)
    repo = RunUsageRepository(db_session)

    await repo.replace_for_run(
        run_id=run.id,
        rows=[
            {
                "stage_name": "a",
                "model": "m",
                "provider": "p",
                "input_tokens": 10,
                "output_tokens": 5,
                "cost_usd": "0.01",
            },
            {
                "stage_name": "b",
                "model": "m",
                "provider": "p",
                "input_tokens": 20,
                "output_tokens": 6,
                "cost_usd": "0.02",
            },
        ],
    )
    await db_session.commit()
    assert len(await repo.list_for_run(run.id)) == 2

    # Re-running replaces, not appends.
    await repo.replace_for_run(
        run_id=run.id,
        rows=[
            {
                "stage_name": "a",
                "model": "m",
                "provider": "p",
                "input_tokens": 99,
                "output_tokens": 1,
                "cost_usd": "0.50",
            },
        ],
    )
    await db_session.commit()
    rows = await repo.list_for_run(run.id)
    assert len(rows) == 1
    assert rows[0].input_tokens == 99
    assert float(rows[0].cost_usd) == pytest.approx(0.50, rel=1e-4)


async def test_run_ids_unsettled_usage_lists_terminal_runs_until_settled(db_session) -> None:
    run = await _setup_run(db_session)
    tenant_id = await _tenant_of(db_session, run)

    repo = RunUsageRepository(db_session)
    run_repo = RunRepository(db_session)

    # Pending run → not in the missing list (cost not final).
    assert await repo.run_ids_unsettled_usage(tenant_id) == []

    await run_repo.mark_completed(run.id)
    await db_session.commit()

    # Just-terminal run is held back by the cost-flush freshness guard.
    assert await repo.run_ids_unsettled_usage(tenant_id) == []

    # Backdate past the grace → terminal run with no usage now appears.
    await _backdate_updated_at(db_session, run.id)
    missing = await repo.run_ids_unsettled_usage(tenant_id)
    assert [r for r, _ in missing] == [run.id]

    # Mirrored rows alone do NOT settle it: a first mirror can be partial
    # (billing still flushing), so the run stays on the list until it is marked.
    await repo.replace_for_run(
        run_id=run.id,
        rows=[{"stage_name": "a", "input_tokens": 1, "output_tokens": 1, "cost_usd": "0.01"}],
    )
    await db_session.commit()
    assert [r for r, _ in await repo.run_ids_unsettled_usage(tenant_id)] == [run.id]

    # Reconciled (marked) → drops out.
    await repo.replace_for_run(
        run_id=run.id,
        rows=[{"stage_name": "a", "input_tokens": 1, "output_tokens": 1, "cost_usd": "0.02"}],
        mark_reconciled=True,
    )
    await db_session.commit()
    assert await repo.run_ids_unsettled_usage(tenant_id) == []


async def test_run_ids_unsettled_usage_includes_cancelled_runs(db_session) -> None:
    """A cancelled run spent real money on every stage that ran before the cancel.
    Excluding it from the sweep left that spend unmirrored — $0 forever."""
    run = await _setup_run(db_session)
    tenant_id = await _tenant_of(db_session, run)
    repo = RunUsageRepository(db_session)

    run.status = "cancelled"
    await db_session.commit()
    await _backdate_updated_at(db_session, run.id, minutes=66)
    assert [r for r, _ in await repo.run_ids_unsettled_usage(tenant_id)] == [run.id]


async def test_a_cancelled_run_waits_for_the_render_ceiling_before_the_sweep_settles_it(
    db_session,
) -> None:
    run = await _setup_run(db_session)
    tenant_id = await _tenant_of(db_session, run)
    repo = RunUsageRepository(db_session)
    run.status = "cancelled"
    await db_session.commit()
    await _backdate_updated_at(db_session, run.id, minutes=10)
    assert await repo.run_ids_unsettled_usage(tenant_id) == [], "a render may still be billing"
    await _backdate_updated_at(db_session, run.id, minutes=66)
    assert [r for r, _ in await repo.run_ids_unsettled_usage(tenant_id)] == [run.id]


async def test_a_duplicate_mirrored_row_is_impossible_but_local_rows_are_free(db_session) -> None:
    """uq_run_usage_mirrored_group (migration 0105) is the backstop for the mirror
    race: two billing-owned rows for one (run, stage, model, provider) cannot exist."""
    from uuid import uuid7

    from sqlalchemy.exc import IntegrityError

    from src.database.models import RunUsage
    from src.database.repositories.run_usage_repository import (
        BILLING_MIRROR_USAGE_SOURCE,
        IN_PROCESS_USAGE_SOURCE,
    )

    run = await _setup_run(db_session)

    def row(meta=None):
        return RunUsage(
            id=uuid7(), run_id=run.id, stage_name="s", model=None, provider=None,
            input_tokens=1, output_tokens=1, metadata_=meta,
        )  # fmt: skip

    # Local rows and pre-0105 source-less mirror rows are outside the index.
    db_session.add_all(
        [row({"source": IN_PROCESS_USAGE_SOURCE}), row({"source": IN_PROCESS_USAGE_SOURCE})]
    )
    db_session.add_all([row(), row()])
    db_session.add(row({"source": BILLING_MIRROR_USAGE_SOURCE}))
    await db_session.commit()
    db_session.add(row({"source": BILLING_MIRROR_USAGE_SOURCE}))
    with pytest.raises(IntegrityError, match="uq_run_usage_mirrored_group"):
        await db_session.commit()
    await db_session.rollback()


async def test_run_ids_unsettled_usage_excludes_marked_zero_cost_runs(db_session) -> None:
    # A zero-cost run writes no usage rows; the reconciled marker is what keeps it
    # from reappearing in the work-list (and being re-billed) on every load.
    run = await _setup_run(db_session)
    tenant_id = await _tenant_of(db_session, run)
    repo = RunUsageRepository(db_session)

    await RunRepository(db_session).mark_completed(run.id)
    await db_session.commit()
    await _backdate_updated_at(db_session, run.id)
    assert [r for r, _ in await repo.run_ids_unsettled_usage(tenant_id)] == [run.id]

    # Reconcile to ZERO rows but mark it → drops out despite having no rows.
    await repo.replace_for_run(run_id=run.id, rows=[], mark_reconciled=True)
    await db_session.commit()
    assert await repo.run_ids_unsettled_usage(tenant_id) == []


async def test_replace_for_run_coerces_null_token_and_cost_fields(db_session) -> None:
    # A non-LLM cost event (TTS/render/upload) legitimately carries explicit
    # null token/cost fields, which the reconciler forwards as None. Before the
    # fix int(None) / Decimal(str(None)) raised out of replace_for_run and the
    # /usage + run-viewer cost tile 500'd. Assert a null-field row persists as
    # zero instead of blowing up.
    run = await _setup_run(db_session)
    repo = RunUsageRepository(db_session)

    written = await repo.replace_for_run(
        run_id=run.id,
        rows=[
            {
                "stage_name": "narrate_tts",
                "model": None,
                "provider": "kokoro",
                "input_tokens": None,
                "output_tokens": None,
                "cost_usd": None,
            }
        ],
    )
    await db_session.commit()
    assert written == 1

    rows = await repo.list_for_run(run.id)
    assert len(rows) == 1
    assert rows[0].input_tokens == 0
    assert rows[0].output_tokens == 0
    assert float(rows[0].cost_usd) == pytest.approx(0.0, abs=1e-9)
