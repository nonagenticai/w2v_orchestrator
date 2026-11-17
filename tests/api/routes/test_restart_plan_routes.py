"""Cancel → edit → Restart: reuse what the change does not touch, re-run the rest.

George, 2026-10-27: "why would someone cancel a job? … they want to reuse the
existing stuff they already wrote in the job and update something and then rerun
the job." The fixture is shaped like prod run 01a1770d (promo_v1): eight stages
finished, ``promo_render`` was mid-flight when the user cancelled, the rest never ran.

Rules pinned here (``services/runs/restart_plan.py``):

1. a cancelled run's content stays editable (element patch / restamp); a run the
   executor owns (pending / running) does not;
2. after an EDIT the edited stage is reused WITH the edit and only its dependents
   re-run; after a project INPUT change only the stages that recorded reading it
   (``config_reads``) and their dependents re-run;
3. a stage cancelled mid-flight is never reused, even with partial output;
4. review gates follow the reuse set: approved upstream gates stay approved, a gate
   on a re-run stage re-opens;
5. ``GET /runs/{id}/restart-plan`` previews exactly what ``POST /regenerate {}`` does.
"""

from __future__ import annotations

from uuid import UUID

import pytest
from sqlalchemy import select, update

pytestmark = pytest.mark.asyncio

#: What prod run 01a1770d had finished when it was cancelled.
_DONE = (
    "site_capture",
    "source_ingest",
    "promo_analyze",
    "promo_script",
    "promo_storyboard",
    "promo_narrate",
    "promo_html",
    "promo_slides",
)
#: Recorded reads (what the executor writes since 0102), shaped like the real ones.
_READS = {
    "site_capture": ["brief", "direction", "resolution", "type"],
    "source_ingest": ["brief", "precomputed_source_ingest"],
    "promo_analyze": ["brief", "title"],
    "promo_script": ["brief", "deliverable", "verbatim_items", "voice"],
    "promo_storyboard": ["brand", "brief"],
    "promo_narrate": ["voice"],
    "promo_html": ["brand", "theme"],
    "promo_slides": ["brand", "theme"],
}
_MID_FLIGHT = "promo_render"


async def _make_promo(client) -> tuple[UUID, UUID]:
    r = await client.post(
        "/api/projects",
        json={
            "type": "promo",
            "title": "promo",
            "brief": {"angle": "growth", "source_url": "https://example.com"},
            "deliverables": [{"name": "d", "format": "16x9", "captions_mode": "none"}],
        },
    )
    assert r.status_code == 201, r.text
    project_id = UUID(r.json()["id"])
    detail = await client.get(f"/api/projects/{project_id}")
    return project_id, UUID(detail.json()["deliverables"][0]["id"])


async def _seed_cancelled_promo(
    db_session,
    project_id: UUID,
    deliv_id: UUID,
    *,
    reads: dict[str, list[str]] | None = _READS,
    partial_output: bool = False,
    pending_gates: tuple[str, ...] = (),
) -> UUID:
    """A promo run that finished ``_DONE`` (both review gates approved — the ledger
    drained), was cancelled while ``promo_render`` was running, and never reached
    the rest. ``partial_output`` gives the mid-flight stage in-progress output."""
    from src.database.models import Deliverable, Project, StageOutput
    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository
    from src.services.blueprints.registry import get_registry
    from src.services.runs.resume_inputs import RESUME_INPUTS_KEY, build_resume_inputs
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
        subject_id=project.owner_subject_id,
        config_overrides=None,
    )
    blueprint = get_registry().resolve_by_id("promo_v1")
    snapshot[RESUME_INPUTS_KEY] = await build_resume_inputs(
        db_session, blueprint=blueprint, project_id=project.id, config_snapshot=snapshot
    )
    run_repo = RunRepository(db_session)
    run = await run_repo.create(
        deliverable_id=deliv_id,
        blueprint_id="promo_v1",
        config_snapshot=snapshot,
        pause_after_stages=list(pending_gates),
    )
    so_repo = StageOutputRepository(db_session)
    for spec in blueprint.stages:
        await so_repo.upsert_pending(run.id, spec.name)
    await db_session.commit()
    for name in _DONE:
        await so_repo.mark_completed(
            run.id, name, {"stage": name, "lines": [{"id": "L1", "text": "a"}]}
        )
        if reads is not None:
            await db_session.execute(
                update(StageOutput)
                .where(StageOutput.run_id == run.id, StageOutput.stage_name == name)
                .values(config_reads=reads[name])
            )
    await so_repo.mark_running(run.id, _MID_FLIGHT)
    if partial_output:
        await so_repo.record_progress(run.id, _MID_FLIGHT, {"stage": _MID_FLIGHT, "half": True})
    await run_repo.mark_started(run.id)
    await db_session.commit()
    assert await run_repo.mark_cancelled(run.id)
    await db_session.commit()
    return run.id


async def _restart(client, db_session, parent_id: UUID):
    from src.database.models import Run
    from src.database.repositories.stage_output_repository import StageOutputRepository

    r = await client.post(f"/api/runs/{parent_id}/regenerate", json={})
    assert r.status_code == 201, r.text
    db_session.expire_all()
    child = (
        await db_session.execute(select(Run).where(Run.id == UUID(r.json()["id"])))
    ).scalar_one()
    rows = await StageOutputRepository(db_session).list_for_run(child.id)
    return child, {row.stage_name: row for row in rows}


async def test_cancelled_run_restart_resumes_and_never_reuses_the_mid_flight_stage(
    client, db_session
) -> None:
    """Rule 3: the stage the cancel cut off — even with partial output — re-runs."""
    project_id, deliv = await _make_promo(client)
    parent_id = await _seed_cancelled_promo(db_session, project_id, deliv, partial_output=True)

    child, rows = await _restart(client, db_session, parent_id)
    for name in _DONE:
        assert rows[name].status == "completed", name
    assert rows[_MID_FLIGHT].status == "pending"
    assert rows[_MID_FLIGHT].output is None
    # Both gates were approved in the parent and their stages are reused: stay approved.
    assert not {"promo_script", "promo_storyboard"} & set(child.pause_after_stages or [])


async def test_editing_the_script_of_a_cancelled_run_reuses_it_and_re_runs_its_dependents(
    client, db_session
) -> None:
    """Rules 1, 2, 4 for an EDIT."""
    project_id, deliv = await _make_promo(client)
    parent_id = await _seed_cancelled_promo(db_session, project_id, deliv)

    edit = await client.patch(
        f"/api/runs/{parent_id}/stages/promo_script/elements/L1",
        json={"patch": {"text": "a better opening line"}},
    )
    assert edit.status_code == 200, edit.text

    child, rows = await _restart(client, db_session, parent_id)
    # The edited stage is inherited WITH the edit...
    assert rows["promo_script"].status == "completed"
    assert rows["promo_script"].output["lines"][0]["text"] == "a better opening line"
    # ...everything depending on it re-runs on it...
    for name in ("promo_storyboard", "promo_narrate", "promo_html", "promo_slides"):
        assert rows[name].status == "pending", name
    # ...and what it depends on is reused.
    for name in ("site_capture", "source_ingest", "promo_analyze"):
        assert rows[name].status == "completed", name
    # The script gate (reused) stays approved; the storyboard gate (re-run) re-opens.
    assert "promo_script" not in (child.pause_after_stages or [])
    assert "promo_storyboard" in (child.pause_after_stages or [])


async def test_changing_the_brand_of_a_cancelled_run_re_runs_only_its_readers(
    client, db_session
) -> None:
    """Rule 2 for a project INPUT (audit #6): the brand is read by the storyboard,
    the HTML and the slides — so those re-run (with their dependents) and the
    capture, ingest, analysis, script and narration are reused."""
    project_id, deliv = await _make_promo(client)
    parent_id = await _seed_cancelled_promo(db_session, project_id, deliv)
    patched = await client.patch(
        f"/api/projects/{project_id}", json={"brand": {"primary_color": "#ff0055"}}
    )
    assert patched.status_code == 200, patched.text

    child, rows = await _restart(client, db_session, parent_id)
    for name in ("promo_storyboard", "promo_html", "promo_slides"):
        assert rows[name].status == "pending", name
    for name in ("site_capture", "source_ingest", "promo_analyze", "promo_script", "promo_narrate"):
        assert rows[name].status == "completed", name
    assert "promo_storyboard" in (child.pause_after_stages or [])
    assert "promo_script" not in (child.pause_after_stages or [])


async def test_a_change_to_an_input_every_stage_reads_re_runs_everything(
    client, db_session
) -> None:
    """The brief is read by every authoring stage AND the capture — nothing survives."""
    project_id, deliv = await _make_promo(client)
    parent_id = await _seed_cancelled_promo(db_session, project_id, deliv)
    patched = await client.patch(
        f"/api/projects/{project_id}",
        json={"brief": {"angle": "retention", "source_url": "https://example.com"}},
    )
    assert patched.status_code == 200, patched.text

    _, rows = await _restart(client, db_session, parent_id)
    assert all(row.status == "pending" for row in rows.values()), {
        n: r.status for n, r in rows.items()
    }


async def test_restart_plan_preview_matches_what_restart_does(client, db_session) -> None:
    """Rule 5: the Studio shows this list before the user confirms."""
    project_id, deliv = await _make_promo(client)
    parent_id = await _seed_cancelled_promo(db_session, project_id, deliv)
    await client.patch(f"/api/projects/{project_id}", json={"brand": {"primary_color": "#000"}})

    preview = await client.get(f"/api/runs/{parent_id}/restart-plan")
    assert preview.status_code == 200, preview.text
    body = preview.json()
    assert body["mode"] == "partial"
    assert "brand" in body["changed_inputs"]
    planned = {s["stage_name"]: s for s in body["stages"]}
    assert planned["promo_html"]["reasons"][0]["code"] == "input_changed"
    assert planned["promo_html"]["reasons"][0]["inputs"] == ["brand"]
    assert planned[_MID_FLIGHT]["reasons"][0]["code"] == "cancelled"
    assert planned["promo_script"]["action"] == "reuse"

    _, rows = await _restart(client, db_session, parent_id)
    reused = {n for n, s in planned.items() if s["action"] == "reuse"}
    assert reused == {n for n, r in rows.items() if r.status == "completed"}


async def test_restart_plan_preview_refuses_a_paused_or_active_run(client, db_session) -> None:
    from src.database.models import Run

    project_id, deliv = await _make_promo(client)
    parent_id = await _seed_cancelled_promo(db_session, project_id, deliv)
    for status in ("paused", "running"):
        await db_session.execute(update(Run).where(Run.id == parent_id).values(status=status))
        await db_session.commit()
        r = await client.get(f"/api/runs/{parent_id}/restart-plan")
        assert r.status_code == 409, r.text
        assert r.json()["detail"]["code"] == "restart_plan_unavailable"


async def test_a_cancelled_run_stays_editable_but_an_active_one_does_not(
    client, db_session
) -> None:
    """Rule 1. ``pending`` is refused too: the dispatcher is about to own the rows."""
    from src.database.models import Run, StageOutput

    project_id, deliv = await _make_promo(client)
    parent_id = await _seed_cancelled_promo(db_session, project_id, deliv)

    ok = await client.post(
        f"/api/runs/{parent_id}/stages/promo_script/elements/L1/restamp",
        json={"value": {"text": "restamped"}},
    )
    assert ok.status_code == 200, ok.text
    db_session.expire_all()
    row = (
        await db_session.execute(
            select(StageOutput).where(
                StageOutput.run_id == parent_id, StageOutput.stage_name == "promo_script"
            )
        )
    ).scalar_one()
    assert row.edited_at is not None
    # The Studio learns what it may edit, and that it was edited, from the run detail.
    detail = await client.get(f"/api/runs/{parent_id}")
    assert detail.status_code == 200, detail.text
    by_stage = {s["stage_name"]: s for s in detail.json()["stage_outputs"]}
    assert by_stage["promo_script"]["element_paths"] == ["lines"]
    assert by_stage["promo_script"]["edited_at"] is not None
    assert by_stage["promo_analyze"]["edited_at"] is None
    assert by_stage[_MID_FLIGHT]["element_paths"] == []

    for status in ("pending", "running"):
        await db_session.execute(update(Run).where(Run.id == parent_id).values(status=status))
        await db_session.commit()
        patch_r = await client.patch(
            f"/api/runs/{parent_id}/stages/promo_script/elements/L1",
            json={"patch": {"text": "x"}},
        )
        assert patch_r.status_code == 409, (status, patch_r.text)
        restamp_r = await client.post(
            f"/api/runs/{parent_id}/stages/promo_script/elements/L1/restamp",
            json={"value": {"text": "x"}},
        )
        assert restamp_r.status_code == 409, (status, restamp_r.text)


async def test_completed_run_with_nothing_changed_regenerates_everything(
    client, db_session
) -> None:
    """A finished run with no edit and no input change: a new version is a fresh take."""
    from src.database.models import Run

    project_id, deliv = await _make_promo(client)
    parent_id = await _seed_cancelled_promo(db_session, project_id, deliv)
    await db_session.execute(update(Run).where(Run.id == parent_id).values(status="completed"))
    await db_session.commit()

    preview = await client.get(f"/api/runs/{parent_id}/restart-plan")
    assert preview.status_code == 200, preview.text
    assert preview.json()["mode"] == "full"
    assert preview.json()["full_reasons"][0]["code"] == "nothing_changed"


async def test_completed_run_after_an_edit_reuses_what_the_edit_does_not_touch(
    client, db_session
) -> None:
    from src.database.models import Run

    project_id, deliv = await _make_promo(client)
    parent_id = await _seed_cancelled_promo(db_session, project_id, deliv)
    await db_session.execute(update(Run).where(Run.id == parent_id).values(status="completed"))
    await db_session.commit()
    edit = await client.patch(
        f"/api/runs/{parent_id}/stages/promo_script/elements/L1",
        json={"patch": {"text": "edited"}},
    )
    assert edit.status_code == 200, edit.text

    _, rows = await _restart(client, db_session, parent_id)
    assert rows["promo_script"].status == "completed"
    assert rows["promo_analyze"].status == "completed"
    assert rows["promo_narrate"].status == "pending"


async def test_from_stage_on_a_run_cancelled_before_any_stage_ran_names_that_cause(
    client, db_session
) -> None:
    """Audit #1: a run cancelled while still pending has NO stage rows. Asking to
    re-run "from" a stage is refused with an accurate cause — nothing failed to
    read — and a plain Restart runs it from the beginning."""
    from src.database.repositories.run_repository import RunRepository

    _, deliv = await _make_promo(client)
    run_repo = RunRepository(db_session)
    run = await run_repo.create(
        deliverable_id=deliv, blueprint_id="promo_v1", config_snapshot={}, pause_after_stages=[]
    )
    await db_session.commit()
    assert await run_repo.mark_cancelled(run.id)
    await db_session.commit()

    r = await client.post(f"/api/runs/{run.id}/regenerate", json={"from_stage": "promo_script"})
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["cause"] == "parent_never_ran"
    assert "unreadable" not in detail["message"]
    assert "before any of its steps ran" in detail["message"]

    plain = await client.post(f"/api/runs/{run.id}/regenerate", json={})
    assert plain.status_code == 201, plain.text


async def test_start_run_with_a_cancelled_parent_is_accepted(client, db_session) -> None:
    """Audit #3: ``POST /deliverables/{id}/runs`` with ``parent_run_id`` shares
    ``_resolve_parent_state`` with regenerate, so a cancelled parent is a valid
    parent there too — deliberately: same "fix and run again" intent, and the
    explicit-list API keeps its explicit checkpoint list."""
    project_id, deliv = await _make_promo(client)
    parent_id = await _seed_cancelled_promo(db_session, project_id, deliv)

    r = await client.post(
        f"/api/deliverables/{deliv}/runs",
        json={"parent_run_id": str(parent_id), "pause_after_stages": []},
    )
    assert r.status_code == 201, r.text
