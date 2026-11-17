"""An EXPLICIT narrow ``regenerate`` must never silently widen into a full re-run.

Measured in prod 2026-09-16: ``POST /api/runs/{id}/regenerate {"fromStage":
"website_review_compile"}`` — which is supposed to re-run only
``{website_review_compile, growth_bundle}`` and inherit the rest — came back with
``website_redesign`` re-running and script/record/slides/visual_director all
``pending``. It had taken ``build_regenerate_state``'s "no notes, no target — re-run
all" fallback. A caller asking for a narrow re-run silently got a FULL one: fresh TTS,
fresh LLM, a DIFFERENT script, and peers bill per HTTP call on any 2xx, so the widening
bills like a whole new job and nothing anywhere says so.

Two independent holes produced it, and both are pinned here:

1. ``RegenerateRequest.from_stage`` had no camelCase alias, so ``fromStage`` was
   DROPPED to ``None`` — which is exactly the input that takes the re-run-all fallback.
2. Even with the target parsed, an empty preserved set fell through to a full re-run.
   It now REFUSES (``RegenerateScopeError`` → typed 409) rather than guessing why the
   parent's outputs were unreadable.

The implicit path (no ``from_stage`` at all) never refuses: it follows the per-stage
restart plan (``services/runs/restart_plan.py``), which promises nothing narrow.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

pytestmark = pytest.mark.asyncio


async def _make_project(client) -> tuple[UUID, UUID]:
    r = await client.post(
        "/api/projects",
        json={
            "type": "website_review",
            "title": "review",
            "brief": {"source_url": "https://example.com"},
            "deliverables": [{"name": "d", "format": "16x9", "target_duration_seconds": 60}],
        },
    )
    assert r.status_code == 201, r.text
    project_id = UUID(r.json()["id"])
    detail = await client.get(f"/api/projects/{project_id}")
    return project_id, UUID(detail.json()["deliverables"][0]["id"])


async def _seed_completed_parent_with_no_outputs(
    db_session,
    project_id: UUID,
    deliv_id: UUID,
    *,
    blueprint_id: str = "website_review_v1",
    with_resume_inputs: bool = False,
) -> UUID:
    """A COMPLETED parent whose ``stage_outputs`` read back EMPTY.

    This is the state the prod probe hit (it ran under a cross-tenant scope switch, so
    ``list_for_run`` returned nothing under the wrong RLS GUC). Modelled directly —
    the point under test is the orchestrator's RESPONSE to an unreadable parent, not
    the reason it was unreadable.
    """
    from src.database.models import Deliverable, Project
    from src.database.repositories.run_repository import RunRepository
    from src.services.runs.run_setup import build_config_snapshot

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
    if with_resume_inputs:
        # Without the live-input fingerprint the restart planner refuses to plan
        # (full re-run), the implicit path reuses nothing, and a test of the
        # implicit/explicit split cannot discriminate (it stayed green under a forced
        # `from_stage_explicit = True`).
        from src.services.blueprints.registry import get_registry
        from src.services.runs.resume_inputs import RESUME_INPUTS_KEY, build_resume_inputs

        snapshot[RESUME_INPUTS_KEY] = await build_resume_inputs(
            db_session,
            blueprint=get_registry().resolve_by_id(blueprint_id),
            project_id=project.id,
            config_snapshot=snapshot,
        )
    run_repo = RunRepository(db_session)
    run = await run_repo.create(
        deliverable_id=deliv_id,
        blueprint_id=blueprint_id,
        config_snapshot=snapshot,
        pause_after_stages=[],
    )
    if not with_resume_inputs:
        await run_repo.mark_completed(run.id)
    await db_session.commit()
    return run.id


async def _child_count(db_session, parent_id: UUID) -> int:
    from sqlalchemy import func

    from src.database.models import Run

    stmt = select(func.count()).select_from(Run).where(Run.parent_run_id == parent_id)
    return int((await db_session.execute(stmt)).scalar_one())


async def _child_snapshot(db_session, parent_id: UUID) -> dict:
    from src.database.models import Run

    stmt = select(Run.config_snapshot).where(Run.parent_run_id == parent_id)
    return dict((await db_session.execute(stmt)).scalar_one() or {})


async def test_explicit_from_stage_with_an_empty_preserved_set_is_refused(
    client, db_session
) -> None:
    project_id, deliv = await _make_project(client)
    parent_id = await _seed_completed_parent_with_no_outputs(db_session, project_id, deliv)

    r = await client.post(
        f"/api/runs/{parent_id}/regenerate", json={"from_stage": "website_review_compile"}
    )

    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["reason"] == "regenerate_scope_unavailable"
    # It NAMES the requested stage and gives the reason — the caller can act on it.
    assert detail["from_stage"] == "website_review_compile"
    assert detail["cause"] == "parent_outputs_unreadable"
    assert "website_review_compile" in detail["message"]
    # And, the whole point: no PAID child run was created.
    assert await _child_count(db_session, parent_id) == 0


async def test_the_camelcase_spelling_reaches_the_same_refusal(client, db_session) -> None:
    """``fromStage`` must be parsed, not dropped — otherwise the refusal is unreachable.

    Without the alias this request produces a 201 full re-run, which is the exact
    prod symptom.
    """
    project_id, deliv = await _make_project(client)
    parent_id = await _seed_completed_parent_with_no_outputs(db_session, project_id, deliv)

    r = await client.post(
        f"/api/runs/{parent_id}/regenerate", json={"fromStage": "website_review_compile"}
    )

    assert r.status_code == 409, r.text
    assert r.json()["detail"]["from_stage"] == "website_review_compile"
    assert await _child_count(db_session, parent_id) == 0


async def test_the_implicit_path_still_re_runs_everything(client, db_session) -> None:
    """No ``from_stage`` = no narrow promise, so the re-run-all fallback is UNCHANGED."""
    project_id, deliv = await _make_project(client)
    parent_id = await _seed_completed_parent_with_no_outputs(db_session, project_id, deliv)

    r = await client.post(f"/api/runs/{parent_id}/regenerate", json={})

    assert r.status_code == 201, r.text
    from src.database.models import Run
    from src.database.repositories.stage_output_repository import StageOutputRepository

    child = (
        await db_session.execute(select(Run).where(Run.id == UUID(r.json()["id"])))
    ).scalar_one()
    rows = await StageOutputRepository(db_session).list_for_run(child.id)
    # Nothing was preserved (there was nothing to preserve) — every stage re-runs.
    assert rows, "the implicit regenerate produced no stages at all"
    assert not [row for row in rows if row.status == "completed"]
    assert await _child_count(db_session, parent_id) == 1


async def test_an_explicit_root_stage_is_not_refused(client, db_session) -> None:
    """A DAG root's subtree IS the whole blueprint, so an empty preserved set is correct.

    The guard must fire on a SILENT widening, not on a caller who deliberately asked
    for a re-run of everything by naming the first stage. This is what keeps the
    refusal from being "any empty preserved set", which would break a legitimate call.
    """
    from src.services.blueprints.registry import get_registry

    # website_review_v1 has PARALLEL roots (source_ingest / site_capture / …), so no
    # stage of it covers the whole DAG. seo_audit_v1 is single-rooted at source_ingest.
    blueprint = get_registry().resolve_by_id("seo_audit_v1")
    order = blueprint.topological_order()
    root = "source_ingest"
    assert set(order) - ({root} | blueprint.descendants(root)) == set(), (
        f"'{root}' no longer covers seo_audit_v1 ({order}); this test's premise is gone"
    )

    r = await client.post(
        "/api/projects",
        json={
            "type": "seo_audit",
            "title": "audit",
            "brief": {"target_url": "https://example.com"},
            "deliverables": [{"name": "d", "format": "16x9", "target_duration_seconds": 60}],
        },
    )
    assert r.status_code == 201, r.text
    project_id = UUID(r.json()["id"])
    detail = await client.get(f"/api/projects/{project_id}")
    deliv = UUID(detail.json()["deliverables"][0]["id"])
    parent_id = await _seed_completed_parent_with_no_outputs(
        db_session, project_id, deliv, blueprint_id="seo_audit_v1"
    )

    r = await client.post(f"/api/runs/{parent_id}/regenerate", json={"fromStage": root})

    assert r.status_code == 201, r.text
    # The explicit target is recorded on the child (ai_image_gen reads it to stop
    # carrying stills forward when the user re-ran it on purpose).
    assert (await _child_snapshot(db_session, parent_id)).get("regenerate_from_stage") == root


# ── the guard must not fire where nothing was ever inherited ────────────────────


async def test_a_parent_that_completed_nothing_outside_the_subtree_is_not_refused(
    client, db_session
) -> None:
    """The CI regression (``test_runs.py::test_regenerate_from_stage_creates_child``).

    A reviewer paused at the first gate clicks "rerun this stage". The parent has
    legitimately completed only that stage — there is nothing outside the subtree to
    inherit, so nothing is being silently re-billed and the request must succeed.

    The first version of this guard used ``set(topological_order) - rerun_region`` as
    its baseline — what the BLUEPRINT says should exist — and 409'd this flow, telling
    the caller that 8 stages "should have been inherited" about 8 stages that never
    ran. The baseline is the PARENT'S OWN ROWS.
    """
    from src.database.repositories.stage_output_repository import StageOutputRepository

    project_id, deliv = await _make_project(client)
    parent_id = await _seed_completed_parent_with_no_outputs(db_session, project_id, deliv)
    # The ONLY completed stage is the one being regenerated. Its descendants are in the
    # re-run region; nothing sits outside it.
    so_repo = StageOutputRepository(db_session)
    await so_repo.upsert_pending(parent_id, "website_review_compile")
    await db_session.commit()
    await so_repo.mark_completed(parent_id, "website_review_compile", {"stage": "compile"})
    await db_session.commit()

    r = await client.post(
        f"/api/runs/{parent_id}/regenerate", json={"fromStage": "website_review_compile"}
    )

    assert r.status_code == 201, r.text


async def test_a_partial_inheritance_is_refused_not_only_a_total_one(client, db_session) -> None:
    """One preserved of two is still a silent re-bill of the other.

    The guard must not be "preserved is empty". A parent with TWO completed stages
    outside the re-run region, one of which carries a NULL output and so cannot be
    inherited, is about to re-run and re-bill that one — and it names WHICH.
    """
    from sqlalchemy import update

    from src.database.models import StageOutput
    from src.database.repositories.stage_output_repository import StageOutputRepository

    project_id, deliv = await _make_project(client)
    parent_id = await _seed_completed_parent_with_no_outputs(db_session, project_id, deliv)
    so_repo = StageOutputRepository(db_session)
    for name in ("website_critique", "website_redesign"):
        await so_repo.upsert_pending(parent_id, name)
    await db_session.commit()
    await so_repo.mark_completed(parent_id, "website_critique", {"stage": "website_critique"})
    await db_session.commit()
    # `website_redesign` is recorded COMPLETED with a NULL output — it can never be
    # preserved, so the child would silently re-run and re-bill it. Flipped by status
    # only: `mark_completed` would write an output, and `stage_outputs.output` is
    # write-once (a DB trigger rejects nulling it afterwards).
    await db_session.execute(
        update(StageOutput)
        .where(StageOutput.run_id == parent_id, StageOutput.stage_name == "website_redesign")
        .values(status="completed")
    )
    await db_session.commit()

    r = await client.post(
        f"/api/runs/{parent_id}/regenerate", json={"fromStage": "website_review_compile"}
    )

    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["cause"] == "preserved_outputs_incomplete"
    # It names the stage that would have been silently re-billed, and ONLY that one —
    # `website_critique` inherited fine and must not be listed.
    assert detail["dropped_stages"] == ["website_redesign"]
    assert await _child_count(db_session, parent_id) == 0


async def test_an_automatic_narrow_caller_is_warned_not_refused(client, db_session, caplog) -> None:
    """BLOCKER 5, the too-STRICT direction: only an EXPLICIT caller may be refused.

    ``structural_relief`` and ``_spawn_music_rewind_child`` pass a narrow ``from_stage``
    from an automatic loop with no human present. Refusing there could stall a run
    mid-pipeline, so the widening is logged at WARNING and the run proceeds.

    Pinned by calling ``build_regenerate_state`` directly with
    ``from_stage_explicit=False`` — the automatic callers' exact shape. If the
    explicit-only split is lost (``from_stage_explicit`` forced True), this raises.
    """
    import logging

    from src.database.models import Run
    from src.services.blueprints.registry import get_registry
    from src.services.runs.regenerate import build_regenerate_state

    project_id, deliv = await _make_project(client)
    parent_id = await _seed_completed_parent_with_no_outputs(db_session, project_id, deliv)
    parent = (await db_session.execute(select(Run).where(Run.id == parent_id))).scalar_one()

    with caplog.at_level(logging.WARNING, logger="src.services.runs.regenerate"):
        preserved, _notes, _ids = await build_regenerate_state(
            session=db_session,
            parent_run=parent,
            blueprint=get_registry().resolve_by_id("website_review_v1"),
            from_stage="website_review_compile",
            from_stage_explicit=False,
        )

    # It did NOT raise, and it DID say so.
    assert preserved == {}
    warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert warnings, "the automatic widening was silent — that is the whole defect"
    assert "regenerate scope widened silently" in warnings[0].getMessage()
    assert "website_review_compile" in warnings[0].getMessage()


async def test_the_same_call_from_an_explicit_caller_raises(client, db_session) -> None:
    """The other half of BLOCKER 5: identical inputs, explicit=True, must RAISE.

    Together with the test above this pins the split in BOTH directions — neither
    forcing ``from_stage_explicit`` True nor False leaves the pair green.
    """
    from src.database.models import Run
    from src.services.blueprints.registry import get_registry
    from src.services.runs.regenerate import RegenerateScopeError, build_regenerate_state

    project_id, deliv = await _make_project(client)
    parent_id = await _seed_completed_parent_with_no_outputs(db_session, project_id, deliv)
    parent = (await db_session.execute(select(Run).where(Run.id == parent_id))).scalar_one()

    with pytest.raises(RegenerateScopeError) as caught:
        await build_regenerate_state(
            session=db_session,
            parent_run=parent,
            blueprint=get_registry().resolve_by_id("website_review_v1"),
            from_stage="website_review_compile",
            from_stage_explicit=True,
        )
    assert caught.value.cause == "parent_outputs_unreadable"


# ── BLOCKER 5 at the ROUTE, where real users are ───────────────────────────────
#
# The direct-call pair above pins `build_regenerate_state`'s split. It does NOT pin
# the ROUTE's derivation of that flag: forcing `from_stage_explicit = True` in
# `lifecycle.py` left the whole suite green. These two drive the same widening parent
# through the HTTP door twice — once implicitly (the Studio's "Restart" of a failed
# run, which PLANS the cut itself — restart_plan.py) and once explicitly — and the two
# must answer differently.


async def _seed_widening_failed_parent(db_session, project_id: UUID, deliv_id: UUID) -> UUID:
    """A FAILED parent with a completed-but-unusable stage outside the re-run region."""
    from sqlalchemy import update

    from src.database.models import StageOutput
    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository

    parent_id = await _seed_completed_parent_with_no_outputs(
        db_session, project_id, deliv_id, with_resume_inputs=True
    )
    so_repo = StageOutputRepository(db_session)
    done = ("website_critique", "website_review_script")
    for name in (*done, "website_redesign", "website_review_record"):
        await so_repo.upsert_pending(parent_id, name)
    await db_session.commit()
    for name in done:
        await so_repo.mark_completed(parent_id, name, {"stage": name})
    await so_repo.mark_failed(parent_id, "website_review_record", "upstream 503")
    await RunRepository(db_session).mark_failed(parent_id, "stage failed")
    await db_session.commit()
    # Recorded COMPLETED with a NULL output — it can never be preserved, so the child
    # would silently re-run and re-bill it. Status-only flip: `stage_outputs.output` is
    # write-once (a DB trigger rejects nulling it after `mark_completed`).
    await db_session.execute(
        update(StageOutput)
        .where(StageOutput.run_id == parent_id, StageOutput.stage_name == "website_redesign")
        .values(status="completed")
    )
    await db_session.commit()
    return parent_id


async def test_the_route_refuses_the_explicit_request_on_a_widening_parent(
    client, db_session
) -> None:
    project_id, deliv = await _make_project(client)
    parent_id = await _seed_widening_failed_parent(db_session, project_id, deliv)

    r = await client.post(
        f"/api/runs/{parent_id}/regenerate", json={"fromStage": "website_review_record"}
    )

    assert r.status_code == 409, r.text
    assert r.json()["detail"]["cause"] == "preserved_outputs_incomplete"


async def test_the_route_allows_the_implicit_restart_of_the_same_parent(client, db_session) -> None:
    """The Studio's "Restart" — `POST /regenerate {}` — plans the cut internally.

    The caller promised nothing narrow, so the same widening must NOT refuse: the
    completed-but-outputless stage simply re-runs (``missing_output``). If the route
    stops deriving `from_stage_explicit` from `body.from_stage` (forcing it True), or
    the planner hands its cut back as a from_stage, a reviewer can no longer restart a
    failed run at all.
    """
    project_id, deliv = await _make_project(client)
    parent_id = await _seed_widening_failed_parent(db_session, project_id, deliv)

    r = await client.post(f"/api/runs/{parent_id}/regenerate", json={})

    assert r.status_code == 201, r.text
    assert await _child_count(db_session, parent_id) == 1
    # A DERIVED cut is not the caller's target: nothing is recorded.
    assert "regenerate_from_stage" not in await _child_snapshot(db_session, parent_id)


# ── F3: a parent shaped the way the ENGINE shapes one ──────────────────────────
#
# Every other parent in this file is hand-seeded: rows exist only for the stages the
# test cares about. A REAL parent is not like that. `run_engine.start_run` calls
# `insert_initial_stage_rows`, which writes one row per blueprint stage in topological
# order — `completed` for preserved ones and **`pending` for every stage that has not
# run yet**. So a real paused parent carries a `pending` row for most of its DAG.
#
# That difference is load-bearing and nothing here saw it: deleting
# `and so.status == "completed"` from `_detect_silent_widening` left all 8927 tests
# green, while on real traffic it makes EVERY narrow regenerate on a paused parent
# return 409 — the pending rows outside the re-run region are never in `preserved`, so
# `dropped` is never empty. Same class as the blueprint-baseline bug: the tests agreed
# with the code and neither agreed with production.
#
# This parent is therefore built by calling the PRODUCTION seeder, not by hand.


async def _seed_engine_shaped_parent(db_session, project_id: UUID, deliv_id: UUID) -> UUID:
    """A paused parent whose stage rows were written by `insert_initial_stage_rows`.

    Completed through `website_review_script`; `pending` rows present for everything
    after it, exactly as the engine leaves a run paused at a review gate.
    """
    from src.database.repositories.stage_output_repository import StageOutputRepository
    from src.services.blueprints.registry import get_registry
    from src.services.runs.run_setup import insert_initial_stage_rows

    parent_id = await _seed_completed_parent_with_no_outputs(db_session, project_id, deliv_id)
    blueprint = get_registry().resolve_by_id("website_review_v1")
    so_repo = StageOutputRepository(db_session)
    # THE PRODUCTION SEEDER — every blueprint stage gets a row.
    await insert_initial_stage_rows(
        db_session, so_repo, run_id=parent_id, blueprint=blueprint, preserved_outputs={}
    )
    await db_session.commit()
    ran = (
        "source_ingest",
        "site_capture",
        "media_discovery",
        "website_critique",
        "website_redesign",
        "website_review_script",
    )
    for name in ran:
        await so_repo.mark_completed(parent_id, name, {"stage": name})
    await db_session.commit()
    return parent_id


async def test_a_narrow_regenerate_succeeds_on_an_engine_shaped_parent(client, db_session) -> None:
    """The real-traffic case: pending rows everywhere, and the narrow re-run still works.

    Delete `and so.status == "completed"` from `_detect_silent_widening` and this 409s.
    """
    from src.database.repositories.stage_output_repository import StageOutputRepository

    project_id, deliv = await _make_project(client)
    parent_id = await _seed_engine_shaped_parent(db_session, project_id, deliv)

    # Precondition — prove the fixture really has the shape the defect needs, so this
    # test cannot quietly degrade into another hand-seeded one.
    rows = await StageOutputRepository(db_session).list_for_run(parent_id)
    statuses = {row.stage_name: row.status for row in rows}
    assert len(statuses) == 12, f"the engine seeder did not write every stage: {statuses}"
    pending_outside = [
        name
        for name, status in statuses.items()
        if status == "pending" and name not in {"website_review_compile", "growth_bundle"}
    ]
    assert pending_outside, f"no pending rows outside the re-run region: {statuses}"

    r = await client.post(
        f"/api/runs/{parent_id}/regenerate", json={"fromStage": "website_review_compile"}
    )

    assert r.status_code == 201, r.text
    # And it really was NARROW: the completed stages were inherited, not re-run.
    child_rows = await StageOutputRepository(db_session).list_for_run(UUID(r.json()["id"]))
    child = {row.stage_name: row.status for row in child_rows}
    assert child["website_redesign"] == "completed", child
    assert child["website_review_script"] == "completed", child
    assert child["website_review_compile"] == "pending", child
