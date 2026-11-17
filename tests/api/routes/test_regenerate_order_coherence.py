"""A regenerate may not inherit narration authored for one order and be judged by another.

``regenerate`` freezes every stage before its cut point. For ``website_review`` that cut
sits BELOW ``website_review_record`` — a ``NarrationStageBase`` subclass listed in
``regenerate._NARRATION_PRODUCERS`` — so the child's spoken track is the parent's,
verbatim. The ordered duration is NOT frozen with it: ``build_config_snapshot`` reads
``deliverable.target_duration_seconds`` LIVE off the row on every run.

Measured in prod, one family, all four runs delivering the same film to within 0.088s
across three days::

    01a09f82-7179 (parent)  ordered 10  delivered 15.050  +50.50%
    01a0aa08-4282           ordered 10  delivered 15.138  +51.38%
    01a0aeae-db0d           ordered 50  delivered 15.138  -69.72%
    01a0b00d-f914           ordered 10  delivered 15.058  +50.58%

The two children 120 percentage points apart delivered 15.138s and 15.058s: the
differing input is the ORDER FIELD, not the film. ``website_review_compile`` recorded
``duration_out_of_tolerance`` ("delivered 15.1s against 50.0s ordered") and the run
completed anyway with ``output->>'degraded'`` NULL — the conformance verdict is a record,
not a veto.

The blueprint is not the problem: an ORIGINAL run of it ordered 50s delivered 46.737s
(-6.53%). Only the regenerate cannot reach a new order, because it never re-authors to
it. And -69.72% is the unrecoverable direction — no extend lever exists downstream of
the cut.

So the door refuses that one intersection, and BOTH halves of it are pinned here:
inheriting narration under an unchanged order stays legal, and changing the order with a
cut ABOVE the narrate stage (which does re-author) stays legal.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select

#: The prod family's two numbers.
PARENT_ORDER_SECONDS = 10
REQUESTED_ORDER_SECONDS = 50

#: Completed on the parent, in topological order, through the narrate stage. Cutting at
#: ``website_review_compile`` leaves every one of these INHERITED — including
#: ``website_review_record``, the narration.
_RAN_THROUGH_NARRATION = (
    "source_ingest",
    "site_capture",
    "media_discovery",
    "website_critique",
    "website_redesign",
    "website_review_script",
    "website_review_record",
)


async def _make_project(client, *, target_duration_seconds: int) -> tuple[UUID, UUID]:
    r = await client.post(
        "/api/projects",
        json={
            "type": "website_review",
            "title": "review",
            "brief": {"source_url": "https://example.com"},
            "deliverables": [
                {
                    "name": "d",
                    "format": "16x9",
                    "target_duration_seconds": target_duration_seconds,
                }
            ],
        },
    )
    assert r.status_code == 201, r.text
    project_id = UUID(r.json()["id"])
    detail = await client.get(f"/api/projects/{project_id}")
    return project_id, UUID(detail.json()["deliverables"][0]["id"])


async def _seed_parent_through_narration(
    db_session, project_id: UUID, deliv_id: UUID, *, completed: tuple[str, ...]
) -> UUID:
    """A paused parent shaped by the PRODUCTION seeder, completed through ``completed``.

    Hand-seeding only the interesting rows would not reproduce the real shape: a live
    parent carries a ``pending`` row for every stage that has not run yet
    (``insert_initial_stage_rows``), and those rows are what a narrow regenerate has to
    step around. So the seeder is called, not imitated.
    """
    from src.database.models import Deliverable, Project
    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository
    from src.services.blueprints.registry import get_registry
    from src.services.runs.run_setup import build_config_snapshot, insert_initial_stage_rows

    project = (
        await db_session.execute(select(Project).where(Project.id == project_id))
    ).scalar_one()
    deliverable = (
        await db_session.execute(select(Deliverable).where(Deliverable.id == deliv_id))
    ).scalar_one()
    snapshot = await build_config_snapshot(
        db_session,
        proj=project,
        deliv=deliverable,
        tenant_id=project.tenant_id,
        subject_id=uuid4(),
        config_overrides=None,
    )
    run_repo = RunRepository(db_session)
    run = await run_repo.create(
        deliverable_id=deliv_id,
        blueprint_id="website_review_v1",
        config_snapshot=snapshot,
        pause_after_stages=[],
    )
    # A regenerate is only offered on a parent that STOPPED: the route refuses a
    # pending/running one before any of this code is reached.
    await run_repo.mark_started(run.id)
    await run_repo.mark_paused(run.id, reason="checkpoint")
    so_repo = StageOutputRepository(db_session)
    await insert_initial_stage_rows(
        db_session,
        so_repo,
        run_id=run.id,
        blueprint=get_registry().resolve_by_id("website_review_v1"),
        preserved_outputs={},
    )
    await db_session.commit()
    for name in completed:
        await so_repo.mark_completed(run.id, name, {"stage": name})
    await db_session.commit()
    return run.id


async def _reorder_the_deliverable(db_session, deliv_id: UUID, seconds: int) -> None:
    """The customer edits the requested length between the two runs."""
    from sqlalchemy import update

    from src.database.models import Deliverable

    await db_session.execute(
        update(Deliverable)
        .where(Deliverable.id == deliv_id)
        .values(target_duration_seconds=seconds)
    )
    await db_session.commit()


async def _child_count(db_session, parent_id: UUID) -> int:
    from src.database.models import Run

    stmt = select(func.count()).select_from(Run).where(Run.parent_run_id == parent_id)
    return int((await db_session.execute(stmt)).scalar_one())


# ── the refusal ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_regenerate_that_inherits_narration_refuses_a_changed_order(
    client, db_session
) -> None:
    """The prod shape: ordered 10 -> re-ordered 50, cut BELOW the narrate stage.

    Without the guard this is a 201 and the child delivers the parent's 15.1s film
    against a 50s order (-69.7%), completing green.
    """
    project_id, deliv = await _make_project(client, target_duration_seconds=PARENT_ORDER_SECONDS)
    parent_id = await _seed_parent_through_narration(
        db_session, project_id, deliv, completed=_RAN_THROUGH_NARRATION
    )
    await _reorder_the_deliverable(db_session, deliv, REQUESTED_ORDER_SECONDS)

    r = await client.post(
        f"/api/runs/{parent_id}/regenerate", json={"fromStage": "website_review_compile"}
    )

    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["reason"] == "regenerate_order_mismatch"
    # It names the narration it would have inherited …
    assert detail["inherited_narration_stages"] == ["website_review_record"]
    # … and exactly which input moved, with both values, so the Studio can say what to
    # put back. `brief.*` did not move and must not be listed.
    assert detail["changed_duration_inputs"] == {
        "deliverable.target_duration_seconds": {
            "parent": float(PARENT_ORDER_SECONDS),
            "requested": float(REQUESTED_ORDER_SECONDS),
        }
    }
    # The escape hatch is stated, because a caller who MEANT to change the length needs
    # to know the full re-run is the way to get it.
    assert "omit 'from_stage'" in detail["message"]
    # And the whole point: no paid child run was created.
    assert await _child_count(db_session, parent_id) == 0


@pytest.mark.asyncio
async def test_the_refusal_fires_on_a_shortened_order_too(client, db_session) -> None:
    """Both directions, not just the unrecoverable one.

    A guard that only caught the order growing would pass the +50% rows of the measured
    family — three of the four — which are the same defect with the sign flipped.
    """
    project_id, deliv = await _make_project(client, target_duration_seconds=120)
    parent_id = await _seed_parent_through_narration(
        db_session, project_id, deliv, completed=_RAN_THROUGH_NARRATION
    )
    await _reorder_the_deliverable(db_session, deliv, 30)

    r = await client.post(
        f"/api/runs/{parent_id}/regenerate", json={"fromStage": "website_review_compile"}
    )

    assert r.status_code == 409, r.text
    assert r.json()["detail"]["changed_duration_inputs"] == {
        "deliverable.target_duration_seconds": {"parent": 120.0, "requested": 30.0}
    }


# ── the three flows that must stay legal ───────────────────────────────────────


@pytest.mark.asyncio
async def test_the_same_regenerate_is_allowed_when_the_order_did_not_move(
    client, db_session
) -> None:
    """Inheriting narration is the ORDINARY case — the guard must need BOTH conditions.

    A guard keyed on "narration inherited" alone would 409 every narrow regenerate in
    the product.
    """
    from src.database.repositories.stage_output_repository import StageOutputRepository

    project_id, deliv = await _make_project(client, target_duration_seconds=PARENT_ORDER_SECONDS)
    parent_id = await _seed_parent_through_narration(
        db_session, project_id, deliv, completed=_RAN_THROUGH_NARRATION
    )

    r = await client.post(
        f"/api/runs/{parent_id}/regenerate", json={"fromStage": "website_review_compile"}
    )

    assert r.status_code == 201, r.text
    # And it really was narrow — the narration WAS inherited, so this test is exercising
    # the guard's first condition rather than trivially missing it.
    child_rows = await StageOutputRepository(db_session).list_for_run(UUID(r.json()["id"]))
    child = {row.stage_name: row.status for row in child_rows}
    assert child["website_review_record"] == "completed", child
    assert child["website_review_compile"] == "pending", child


@pytest.mark.asyncio
async def test_a_cut_above_the_narrate_stage_is_allowed_after_an_order_change(
    client, db_session
) -> None:
    """Order changed + narration RE-AUTHORED is coherent, and is the fix path.

    ``website_review_record`` depends on ``website_review_script``, so cutting at the
    script puts the narrate stage in the re-run region: the child writes a new script
    to the new length. That is the flow that measured 50s -> 46.737s on an original run,
    and a guard keyed on "the order moved" alone would break it.
    """
    from src.database.repositories.stage_output_repository import StageOutputRepository

    project_id, deliv = await _make_project(client, target_duration_seconds=PARENT_ORDER_SECONDS)
    parent_id = await _seed_parent_through_narration(
        db_session, project_id, deliv, completed=_RAN_THROUGH_NARRATION
    )
    await _reorder_the_deliverable(db_session, deliv, REQUESTED_ORDER_SECONDS)

    r = await client.post(
        f"/api/runs/{parent_id}/regenerate", json={"fromStage": "website_review_script"}
    )

    assert r.status_code == 201, r.text
    child_rows = await StageOutputRepository(db_session).list_for_run(UUID(r.json()["id"]))
    child = {row.stage_name: row.status for row in child_rows}
    # The premise: narration is NOT inherited here, so there is nothing to be incoherent.
    assert child["website_review_record"] == "pending", child
    assert child["website_critique"] == "completed", child


@pytest.mark.asyncio
async def test_a_full_re_run_after_an_order_change_is_allowed(client, db_session) -> None:
    """The escape hatch the refusal message points at must actually work.

    ``POST /regenerate {}`` on a paused parent with no open annotations preserves
    nothing, so the narration is re-authored to the new length.
    """
    project_id, deliv = await _make_project(client, target_duration_seconds=PARENT_ORDER_SECONDS)
    parent_id = await _seed_parent_through_narration(
        db_session, project_id, deliv, completed=_RAN_THROUGH_NARRATION
    )
    await _reorder_the_deliverable(db_session, deliv, REQUESTED_ORDER_SECONDS)

    r = await client.post(f"/api/runs/{parent_id}/regenerate", json={})

    assert r.status_code == 201, r.text
    assert await _child_count(db_session, parent_id) == 1


@pytest.mark.asyncio
async def test_an_order_change_with_no_narration_completed_yet_is_allowed(
    client, db_session
) -> None:
    """A parent paused at an EARLY gate has no narration to inherit.

    The reviewer approves the critique, edits the length, and re-runs the redesign.
    Nothing was narrated, so nothing is frozen against the new order.
    """
    project_id, deliv = await _make_project(client, target_duration_seconds=PARENT_ORDER_SECONDS)
    parent_id = await _seed_parent_through_narration(
        db_session,
        project_id,
        deliv,
        completed=("source_ingest", "site_capture", "media_discovery", "website_critique"),
    )
    await _reorder_the_deliverable(db_session, deliv, REQUESTED_ORDER_SECONDS)

    r = await client.post(
        f"/api/runs/{parent_id}/regenerate", json={"fromStage": "website_redesign"}
    )

    assert r.status_code == 201, r.text


# ── the predicate itself ───────────────────────────────────────────────────────


def test_every_ordered_duration_path_is_read() -> None:
    """Each path in the union resolves, and a blueprint's key cannot silently drop out.

    The union exists because six resolvers across the fleet read six different keys; a
    path that stopped being read would make that blueprint's regenerate unguarded while
    every other test here stayed green.
    """
    from src.services.runs.regenerate import ORDERED_DURATION_PATHS, ordered_duration_inputs

    snapshot = {
        "target_duration_seconds": 1,
        "deliverable": {"target_duration_seconds": 2},
        "brief": {
            "target_duration_seconds": 3,
            "duration_seconds": 4,
            "duration_minutes": 5,
        },
        "direction": {"timing": {"target_seconds": 6}},
    }

    inputs = ordered_duration_inputs(snapshot)

    assert inputs == {
        "target_duration_seconds": 1.0,
        "deliverable.target_duration_seconds": 2.0,
        "brief.target_duration_seconds": 3.0,
        "brief.duration_seconds": 4.0,
        "brief.duration_minutes": 5.0,
        "direction.timing.target_seconds": 6.0,
    }
    assert set(inputs) == {".".join(path) for path in ORDERED_DURATION_PATHS}


def test_an_absent_order_does_not_compare_equal_to_a_stated_one() -> None:
    """ "No length stated" and "the blueprint's fallback" must not read as the same order.

    Mapping an absent key to a default would make an order APPEARING between parent and
    child invisible — and that is exactly a customer typing a duration into a project
    that had none.
    """
    from src.services.runs.regenerate import ordered_duration_changes

    changes = ordered_duration_changes({}, {"deliverable": {"target_duration_seconds": 50}})

    assert changes == {"deliverable.target_duration_seconds": (None, 50.0)}


def test_a_non_numeric_or_boolean_order_is_not_a_duration() -> None:
    """``bool`` is an ``int`` subclass, so ``True`` would otherwise read as a 1s order."""
    from src.services.runs.regenerate import ordered_duration_inputs

    assert ordered_duration_inputs({"deliverable": {"target_duration_seconds": True}}) == {}
    assert ordered_duration_inputs({"deliverable": {"target_duration_seconds": "soon"}}) == {}
    assert ordered_duration_inputs({"deliverable": {"target_duration_seconds": None}}) == {}
    # A numeric string IS a duration — JSONB round-trips have produced them.
    assert ordered_duration_inputs({"deliverable": {"target_duration_seconds": "30"}}) == {
        "deliverable.target_duration_seconds": 30.0
    }


def test_the_same_order_spelled_int_and_float_is_not_a_change() -> None:
    """``50`` from the request and ``50.0`` from JSONB are the same order.

    Comparing raw values rather than floats would 409 every regenerate of any run whose
    snapshot round-tripped through postgres.
    """
    from src.services.runs.regenerate import ordered_duration_changes

    assert (
        ordered_duration_changes(
            {"deliverable": {"target_duration_seconds": 50}},
            {"deliverable": {"target_duration_seconds": 50.0}},
        )
        == {}
    )


def test_the_guard_needs_both_conditions() -> None:
    """Directly, so the two halves are pinned without a DB round trip.

    Neither an inherited narration nor a moved order is, on its own, a defect.
    """
    from src.services.runs.regenerate import (
        RegenerateOrderMismatchError,
        assert_inherited_narration_matches_the_order,
    )

    ten = {"deliverable": {"target_duration_seconds": 10}}
    fifty = {"deliverable": {"target_duration_seconds": 50}}
    narration = {"website_review_record": {"clips": []}}

    # Narration inherited, order steady.
    assert_inherited_narration_matches_the_order(
        parent_snapshot=ten, child_snapshot=ten, preserved_outputs=narration
    )
    # Order moved, narration re-authored (a non-narration stage inherited).
    assert_inherited_narration_matches_the_order(
        parent_snapshot=ten,
        child_snapshot=fifty,
        preserved_outputs={"website_critique": {}},
    )
    # Both — and only both.
    with pytest.raises(RegenerateOrderMismatchError) as caught:
        assert_inherited_narration_matches_the_order(
            parent_snapshot=ten, child_snapshot=fifty, preserved_outputs=narration
        )
    assert caught.value.narration_stages == ["website_review_record"]
    assert caught.value.changes == {"deliverable.target_duration_seconds": (10.0, 50.0)}


def test_an_unset_side_is_not_a_contradiction() -> None:
    """A side that is merely ABSENT must not read as "the order changed".

    ``config_snapshot`` is not always a complete record of what was ordered — a parent
    seeded (or persisted) before an order field was ever written carries no key for it
    at all, while ``build_config_snapshot`` reads the child's value LIVE off the
    deliverable row regardless. ``<unset> -> 30`` is a FIRST order being placed where
    none existed, not a changed order contradicting a frozen narration: there is no
    prior order for the inherited narration to contradict.

    This was found breaking 4 pre-existing tests in
    ``tests/services/runs/test_child_checkpoint_restore.py`` whose parent fixture
    persists a bare ``config_snapshot={"tenant_id": ..., "subject_id": ...}`` (no
    ``deliverable`` key at all) while the child's snapshot is rebuilt live off a
    ``Deliverable(target_duration_seconds=30)`` row — exactly this shape, with no order
    ever actually changing.
    """
    from src.services.runs.regenerate import assert_inherited_narration_matches_the_order

    unset: dict = {}
    ten = {"deliverable": {"target_duration_seconds": 10}}
    thirty = {"deliverable": {"target_duration_seconds": 30}}
    narration = {"website_review_record": {"clips": []}}

    # Parent never recorded an order at all; child's is read live off the deliverable.
    assert_inherited_narration_matches_the_order(
        parent_snapshot=unset, child_snapshot=thirty, preserved_outputs=narration
    )
    # The mirror: parent had an order, the rebuilt child snapshot lost the key.
    assert_inherited_narration_matches_the_order(
        parent_snapshot=ten, child_snapshot=unset, preserved_outputs=narration
    )
    # Same value, no-op regardless.
    assert_inherited_narration_matches_the_order(
        parent_snapshot=ten, child_snapshot=ten, preserved_outputs=narration
    )
    # And the measured defect must still be refused: both sides SET and DIFFERENT.
    from src.services.runs.regenerate import RegenerateOrderMismatchError

    fifty = {"deliverable": {"target_duration_seconds": 50}}
    with pytest.raises(RegenerateOrderMismatchError):
        assert_inherited_narration_matches_the_order(
            parent_snapshot=ten, child_snapshot=fifty, preserved_outputs=narration
        )
