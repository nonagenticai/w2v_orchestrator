"""``runs.failure_class`` persistence — the column, its default, and its CHECKs.

``tests/models/test_failure_class_attribution.py`` proves the CLASSIFIER picks
the right answer; this file proves the answer actually lands in postgres, that a
caller with no evidence records ``unknown`` rather than a guess, and that the DB
refuses the two shapes that would make the column untrustworthy.
"""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.database.repositories.project_repository import ProjectRepository
from src.database.repositories.run_repository import RunRepository
from src.models.failure_class import (
    CUSTOMER_INPUT,
    EXTERNAL,
    OUR_DEFECT,
    UNKNOWN,
    attribute_exception,
)
from src.services.stages.upstream import classify

pytestmark = pytest.mark.asyncio


async def _setup_deliverable(session, tenant):
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
    return proj.deliverables[0].id


async def _fresh_run(session):
    did = await _setup_deliverable(session, uuid4())
    repo = RunRepository(session)
    run = await repo.create(
        deliverable_id=did,
        blueprint_id="generic_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    await repo.mark_started(run.id)
    await session.commit()
    return repo, run


def _status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://worker-management:8887/workers/x/message")
    response = httpx.Response(status, request=request, text="{}")
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


async def test_a_known_ours_failure_persists_our_defect(db_session) -> None:
    """The end-to-end ours path: a programming error reaches the DB as ``our_defect``."""
    repo, run = await _fresh_run(db_session)

    exc = AttributeError("'NoneType' object has no attribute 'clips'")
    assert await repo.mark_failed(run.id, str(exc), failure_class=attribute_exception(exc)) is True
    await db_session.commit()

    fresh = await repo.get(run.id)
    assert fresh is not None
    assert fresh.status == "failed"
    assert fresh.failure_class == OUR_DEFECT


async def test_a_known_customer_failure_persists_customer_input(db_session) -> None:
    """The end-to-end theirs path: WM's 402 reaches the DB as ``customer_input``."""
    repo, run = await _fresh_run(db_session)

    exc = classify(_status_error(402))
    assert await repo.mark_failed(run.id, str(exc), failure_class=attribute_exception(exc)) is True
    await db_session.commit()

    fresh = await repo.get(run.id)
    assert fresh is not None
    assert fresh.failure_class == CUSTOMER_INPUT


async def test_an_upstream_outage_persists_external(db_session) -> None:
    """A dependency 5xx must NOT be recorded as our bug — it has its own bucket."""
    repo, run = await _fresh_run(db_session)

    exc = classify(_status_error(503))
    await repo.mark_failed(run.id, str(exc), failure_class=attribute_exception(exc))
    await db_session.commit()

    fresh = await repo.get(run.id)
    assert fresh is not None
    assert fresh.failure_class == EXTERNAL
    assert fresh.failure_class != OUR_DEFECT


async def test_an_unrecognised_failure_persists_unknown_and_is_not_misattributed(
    db_session,
) -> None:
    """An unattributable failure records ``unknown`` — never one of the real buckets.

    Both halves are asserted: it IS ``unknown`` (so the row is not silently NULL,
    which would read as "never classified"), and it is not any of the three
    attributed values (so a widening of the classifier that starts guessing here
    fails this test rather than quietly re-bucketing production data).
    """
    repo, run = await _fresh_run(db_session)

    exc = RuntimeError("stage 'compile' returned no artifact")
    await repo.mark_failed(run.id, str(exc), failure_class=attribute_exception(exc))
    await db_session.commit()

    fresh = await repo.get(run.id)
    assert fresh is not None
    assert fresh.failure_class == UNKNOWN
    assert fresh.failure_class not in {OUR_DEFECT, CUSTOMER_INPUT, EXTERNAL}


async def test_mark_failed_defaults_to_unknown_rather_than_null(db_session) -> None:
    """A caller that passes no class records ``unknown``, not NULL and not a guess.

    This is what makes the rollout safe: the post-drain failure route and any
    future caller nobody remembered to update record "not attributed" instead of
    inheriting somebody else's bucket. NULL is reserved for rows that predate the
    column — a live writer must never produce one.
    """
    repo, run = await _fresh_run(db_session)

    await repo.mark_failed(run.id, "stage_failed_post_drain")
    await db_session.commit()

    fresh = await repo.get(run.id)
    assert fresh is not None
    assert fresh.failure_class == UNKNOWN, (
        "mark_failed's default must be 'unknown' (classified, unattributable), not "
        "NULL (never classified) — the two are different statements and aggregates "
        "over the column depend on telling them apart."
    )


async def test_a_successful_run_carries_no_failure_class(db_session) -> None:
    """``mark_completed`` must not write an attribution. Nothing failed."""
    repo, run = await _fresh_run(db_session)

    await repo.mark_completed(run.id)
    await db_session.commit()

    fresh = await repo.get(run.id)
    assert fresh is not None
    assert fresh.status == "completed"
    assert fresh.failure_class is None


async def test_the_database_rejects_a_class_outside_the_vocabulary(db_session) -> None:
    """``ck_runs_failure_class`` is real, so a typo cannot persist silently.

    Written as raw SQL on purpose: the repository's signature is typed, so only
    a direct write can reach the constraint — which is exactly what a future
    migration, an admin fix-up, or a hand-run backfill would be.
    """
    _repo, run = await _fresh_run(db_session)

    with pytest.raises(IntegrityError) as caught:
        await db_session.execute(
            text("UPDATE runs SET status = 'failed', failure_class = 'ours' WHERE id = :id"),
            {"id": run.id},
        )
        await db_session.flush()
    assert "ck_runs_failure_class" in str(caught.value)
    await db_session.rollback()


async def test_the_database_rejects_an_attribution_on_a_non_failed_run(db_session) -> None:
    """``ck_runs_failure_class_only_on_failed`` keeps aggregates honest.

    Without it a ``GROUP BY failure_class`` would silently count runs that
    succeeded, and the number nobody could reproduce would be worse than no
    number at all.
    """
    _repo, run = await _fresh_run(db_session)

    with pytest.raises(IntegrityError) as caught:
        await db_session.execute(
            text(
                "UPDATE runs SET status = 'completed', failure_class = 'our_defect' WHERE id = :id"
            ),
            {"id": run.id},
        )
        await db_session.flush()
    assert "ck_runs_failure_class_only_on_failed" in str(caught.value)
    await db_session.rollback()
