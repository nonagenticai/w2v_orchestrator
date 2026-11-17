"""A "Restart" of a FAILED or CANCELLED run resumes where it stopped, not from scratch.

Both are dead runs and the Studio offers Restart on both (``canRestartRun``). A
cancelled run's stages that were cut off mid-flight re-run; its completed ones are
reused unless a change affects them (``services/runs/restart_plan.py``; the
edit/input rules are pinned in ``test_restart_plan_routes.py``).

Prod: website_review run 01a0a045 failed in ``website_redesign``; its Restart child
01a0a054 re-executed ``website_critique`` (a new job id, done in 1s off the cache).
The next Restart after 01a0a054 failed in ``website_review_record`` re-ran critique,
redesign and script AND re-armed their review gates, so the reviewer re-approved
content that had not changed. The Studio's Restart is ``POST /regenerate {}``.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

pytestmark = pytest.mark.asyncio

_DONE = ("website_critique", "website_redesign", "website_review_script", "source_ingest")


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
    # A verbatim item and a storyboard panel exist BEFORE the parent runs, so the
    # unchanged-order case proves both round-trip through the persisted snapshot
    # equal, and the edit cases have something to edit.
    item = await client.post(
        f"/api/projects/{project_id}/verbatim-items", json={"text": "Book a free intake"}
    )
    assert item.status_code == 201, item.text
    panel = await client.post(
        f"/api/projects/{project_id}/storyboard/panels", json={"vo_text": "Opening line"}
    )
    assert panel.status_code == 201, panel.text
    return project_id, UUID(detail.json()["deliverables"][0]["id"])


async def _seed_failed_parent(
    db_session,
    project_id: UUID,
    deliv_id: UUID,
    *,
    blueprint_id: str = "website_review_v1",
    done: tuple[str, ...] = _DONE,
    failed: str = "website_review_record",
    with_resume_inputs: bool = True,
    cancelled: bool = False,
) -> UUID:
    """A run that finished ``done`` (review gates passed) and then stopped in ``failed``.

    It stops by FAILING there, or — ``cancelled=True`` — by being CANCELLED while that
    stage was still running. Its snapshot is assembled exactly as ``start_run``
    persists it: the config snapshot plus the live-input fingerprint
    (``with_resume_inputs=False`` models a parent started before the fingerprint
    shipped). Stage rows carry no recorded ``config_reads`` (a pre-0102 parent).
    """
    from src.database.models import Deliverable, Project
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
    # The parent's snapshot is built by the SAME assembler start_run persists, under
    # a DIFFERENT subject than the restarting client: identity keys must not count as
    # an order change.
    snapshot = await build_config_snapshot(
        db_session,
        proj=project,
        deliv=deliverable,
        tenant_id=project.tenant_id,
        subject_id=uuid4(),
        config_overrides=None,
    )
    if with_resume_inputs:
        snapshot[RESUME_INPUTS_KEY] = await build_resume_inputs(
            db_session,
            blueprint=get_registry().resolve_by_id(blueprint_id),
            project_id=project.id,
            config_snapshot=snapshot,
        )
    run_repo = RunRepository(db_session)
    # Both gates were approved: `resume_run` drained the ledger to [].
    run = await run_repo.create(
        deliverable_id=deliv_id,
        blueprint_id=blueprint_id,
        config_snapshot=snapshot,
        pause_after_stages=[],
    )
    so_repo = StageOutputRepository(db_session)
    for name in (*done, failed):
        await so_repo.upsert_pending(run.id, name)
    await db_session.commit()
    for name in done:
        await so_repo.mark_completed(run.id, name, {"stage": name})
    if cancelled:
        # The user cancelled a run that was stuck RUNNING (prod 01a1770d): the
        # cancel path sweeps every non-terminal stage row to ``cancelled``.
        await so_repo.mark_running(run.id, failed)
        assert await run_repo.mark_cancelled(run.id)
    else:
        await so_repo.mark_failed(run.id, failed, "upstream 503")
        await run_repo.mark_failed(run.id, "stage failed")
    await db_session.commit()
    return run.id


async def _child_rows(db_session, child_id: str):
    from src.database.models import Run
    from src.database.repositories.stage_output_repository import StageOutputRepository

    child = (await db_session.execute(select(Run).where(Run.id == UUID(child_id)))).scalar_one()
    rows = await StageOutputRepository(db_session).list_for_run(child.id)
    return child, {r.stage_name: r.status for r in rows}


async def test_restart_of_failed_run_preserves_finished_stages_and_gates(
    client, db_session
) -> None:
    project_id, deliv = await _make_project(client)
    parent_id = await _seed_failed_parent(db_session, project_id, deliv)

    r = await client.post(f"/api/runs/{parent_id}/regenerate", json={})
    assert r.status_code == 201, r.text
    child, stages = await _child_rows(db_session, r.json()["id"])

    for name in _DONE:
        assert stages[name] == "completed", f"{name} re-executed on Restart: {stages}"
    assert stages["website_review_record"] == "pending"
    assert stages["website_review_compile"] == "pending"
    # The gates the parent already passed are NOT re-armed on unchanged content.
    assert "website_redesign" not in (child.pause_after_stages or [])
    assert "website_review_script" not in (child.pause_after_stages or [])


async def test_restart_of_cancelled_run_is_accepted_and_preserves_finished_stages(
    client, db_session
) -> None:
    """Prod 2026-10-27: run 01a1770d was cancelled, then the Studio's Restart (offered
    for ``failed`` AND ``cancelled`` runs) got ``400 parent run must be
    paused/completed/failed (is cancelled)`` from ``_resolve_parent_state``."""
    project_id, deliv = await _make_project(client)
    parent_id = await _seed_failed_parent(db_session, project_id, deliv, cancelled=True)

    r = await client.post(f"/api/runs/{parent_id}/regenerate", json={})
    assert r.status_code == 201, r.text
    child, stages = await _child_rows(db_session, r.json()["id"])

    for name in _DONE:
        assert stages[name] == "completed", f"{name} re-executed on Restart: {stages}"
    assert stages["website_review_record"] == "pending"
    assert "website_redesign" not in (child.pause_after_stages or [])
    assert "website_review_script" not in (child.pause_after_stages or [])


async def test_rerun_stage_of_cancelled_run_is_accepted(client, db_session) -> None:
    project_id, deliv = await _make_project(client)
    parent_id = await _seed_failed_parent(db_session, project_id, deliv, cancelled=True)

    r = await client.post(
        f"/api/runs/{parent_id}/regenerate", json={"from_stage": "website_review_script"}
    )
    assert r.status_code == 201, r.text
    _, stages = await _child_rows(db_session, r.json()["id"])
    assert stages["website_critique"] == "completed", stages
    assert stages["website_review_script"] == "pending", stages


# --- a parent whose stages ran BEFORE per-stage input recording (config_reads NULL) --
# Nothing says which stages read the changed input, so every one of them re-runs.
# With recorded reads only the readers (and their dependents) re-run — that rule is
# pinned in test_restart_plan_routes.py.


async def test_restart_after_editing_the_brief_re_runs_everything(client, db_session) -> None:
    project_id, deliv = await _make_project(client)
    parent_id = await _seed_failed_parent(db_session, project_id, deliv)
    patched = await client.patch(
        f"/api/projects/{project_id}",
        json={"brief": {"source_url": "https://example.com/changed"}},
    )
    assert patched.status_code == 200, patched.text

    r = await client.post(f"/api/runs/{parent_id}/regenerate", json={})
    assert r.status_code == 201, r.text
    _, stages = await _child_rows(db_session, r.json()["id"])
    assert all(status == "pending" for status in stages.values()), stages


async def _assert_restart_re_runs_everything(client, db_session, parent_id: UUID) -> None:
    r = await client.post(f"/api/runs/{parent_id}/regenerate", json={})
    assert r.status_code == 201, r.text
    _, stages = await _child_rows(db_session, r.json()["id"])
    assert all(status == "pending" for status in stages.values()), stages


async def test_restart_after_changing_the_deliverable_duration_re_runs_everything(
    client, db_session
) -> None:
    """A 60s script must not be resumed into a 45s order (audit repro on #1012)."""
    project_id, deliv = await _make_project(client)
    parent_id = await _seed_failed_parent(db_session, project_id, deliv)
    patched = await client.patch(f"/api/deliverables/{deliv}", json={"target_duration_seconds": 45})
    assert patched.status_code == 200, patched.text

    await _assert_restart_re_runs_everything(client, db_session, parent_id)


async def test_restart_after_editing_a_verbatim_item_re_runs_everything(client, db_session) -> None:
    project_id, deliv = await _make_project(client)
    parent_id = await _seed_failed_parent(db_session, project_id, deliv)
    items = await client.get(f"/api/projects/{project_id}/verbatim-items")
    assert items.status_code == 200, items.text
    item_id = items.json()["items"][0]["id"]
    patched = await client.patch(
        f"/api/verbatim-items/{item_id}", json={"text": "Book a paid intake"}
    )
    assert patched.status_code == 200, patched.text

    await _assert_restart_re_runs_everything(client, db_session, parent_id)


async def test_restart_after_editing_a_storyboard_panel_re_runs_everything(
    client, db_session
) -> None:
    project_id, deliv = await _make_project(client)
    parent_id = await _seed_failed_parent(db_session, project_id, deliv)
    panels = await client.get(f"/api/projects/{project_id}/storyboard/panels")
    assert panels.status_code == 200, panels.text
    panel_id = panels.json()["items"][0]["id"]
    patched = await client.patch(
        f"/api/storyboard/panels/{panel_id}", json={"vo_text": "A different opening"}
    )
    assert patched.status_code == 200, patched.text

    await _assert_restart_re_runs_everything(client, db_session, parent_id)


async def test_restart_with_config_overrides_re_runs_everything(client, db_session) -> None:
    project_id, deliv = await _make_project(client)
    parent_id = await _seed_failed_parent(db_session, project_id, deliv)

    r = await client.post(
        f"/api/runs/{parent_id}/regenerate",
        json={"config_overrides": {"sound_effects": {"enabled": False}}},
    )
    assert r.status_code == 201, r.text
    _, stages = await _child_rows(db_session, r.json()["id"])
    assert stages["website_critique"] == "pending", stages


async def test_apply_edits_child_does_not_re_arm_passed_review_gates(client, db_session) -> None:
    """apply-edits re-runs only compile onward, so the redesign/script gates the
    parent passed stay passed (confirmed: this held before the Restart fix too)."""
    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository

    _, deliv = await _make_project(client)
    run_repo = RunRepository(db_session)
    parent = await run_repo.create(
        deliverable_id=deliv,
        blueprint_id="website_review_v1",
        config_snapshot={},
        pause_after_stages=[],
    )
    so_repo = StageOutputRepository(db_session)
    for name in _DONE:
        await so_repo.upsert_pending(parent.id, name)
    await db_session.commit()
    for name in _DONE:
        await so_repo.mark_completed(parent.id, name, {"lines": [{"id": "L1", "text": "a"}]})
    await run_repo.mark_completed(parent.id)
    await db_session.commit()

    r = await client.post(
        f"/api/runs/{parent.id}/apply-edits",
        json={"stage_name": "website_review_script", "edits": {}, "music": {"track": None}},
    )
    assert r.status_code == 201, r.text
    child, stages = await _child_rows(db_session, r.json()["id"])
    assert stages["website_redesign"] == "completed"
    assert not {"website_redesign", "website_review_script"} & set(child.pause_after_stages or [])


# --- live inputs a stage reads from the DB/object store, not the snapshot ---------

_REEL_DONE = ("source_ingest", "footage_probe", "shot_analysis")


async def _make_reel_project(client) -> tuple[UUID, UUID]:
    r = await client.post(
        "/api/projects",
        json={
            "type": "action_reel",
            "title": "kite reel",
            "brief": {"title": "kite session"},
            "deliverables": [{"name": "d", "format": "16x9", "target_duration_seconds": 30}],
        },
    )
    assert r.status_code == 201, r.text
    project_id = UUID(r.json()["id"])
    detail = await client.get(f"/api/projects/{project_id}")
    return project_id, UUID(detail.json()["deliverables"][0]["id"])


async def _add_clip(db_session, project_id: UUID, name: str) -> UUID:
    """What ``upload_video_source`` registers (its bytes live in the bucket)."""
    from src.database.models import VideoSource

    source_id = uuid4()
    db_session.add(
        VideoSource(
            id=source_id,
            project_id=project_id,
            filename=name,
            object_key=f"tenants/t/projects/{project_id}/video-sources/{source_id}/original.mp4",
            content_type="video/mp4",
            bytes=1234,
            preview_status="ready",
        )
    )
    await db_session.commit()
    return source_id


async def _failed_reel_parent(client, db_session) -> tuple[UUID, UUID]:
    project_id, deliv = await _make_reel_project(client)
    first_clip = await _add_clip(db_session, project_id, "a.mp4")
    await _add_clip(db_session, project_id, "b.mp4")
    parent_id = await _seed_failed_parent(
        db_session,
        project_id,
        deliv,
        blueprint_id="action_reel_v1",
        done=_REEL_DONE,
        failed="segment_select",
    )
    return parent_id, first_clip


async def test_action_reel_restart_with_unchanged_footage_resumes(client, db_session) -> None:
    """The control: without it the two footage cases below would pass vacuously."""
    parent_id, _ = await _failed_reel_parent(client, db_session)

    r = await client.post(f"/api/runs/{parent_id}/regenerate", json={})
    assert r.status_code == 201, r.text
    _, stages = await _child_rows(db_session, r.json()["id"])
    for name in _REEL_DONE:
        assert stages[name] == "completed", stages
    assert stages["segment_select"] == "pending"


async def _assert_footage_and_dependents_re_run(client, db_session, parent_id: UUID) -> None:
    """Changed footage re-runs ``footage_probe`` (it reads the clips LIVE) and every
    stage depending on it — and still reuses ``source_ingest``, which does not."""
    r = await client.post(f"/api/runs/{parent_id}/regenerate", json={})
    assert r.status_code == 201, r.text
    _, stages = await _child_rows(db_session, r.json()["id"])
    assert stages["footage_probe"] == "pending", stages
    assert stages["shot_analysis"] == "pending", stages
    assert stages["source_ingest"] == "completed", stages


async def test_action_reel_restart_after_uploading_a_clip_re_runs_the_footage_stages(
    client, db_session
) -> None:
    parent_id, _ = await _failed_reel_parent(client, db_session)
    parent = await _run(db_session, parent_id)
    await _add_clip(db_session, await _project_of(db_session, parent.deliverable_id), "c.mp4")

    await _assert_footage_and_dependents_re_run(client, db_session, parent_id)


async def test_action_reel_restart_after_deleting_a_clip_re_runs_the_footage_stages(
    client, db_session
) -> None:
    parent_id, first_clip = await _failed_reel_parent(client, db_session)
    parent = await _run(db_session, parent_id)
    project_id = await _project_of(db_session, parent.deliverable_id)
    deleted = await client.delete(f"/api/projects/{project_id}/video-sources/{first_clip}")
    assert deleted.status_code == 204, deleted.text

    await _assert_footage_and_dependents_re_run(client, db_session, parent_id)


async def test_action_reel_restart_after_a_clip_is_re_probed_re_runs_the_footage_stages(
    client, db_session
) -> None:
    """A derivative/probe landing moves ``updated_at`` — footage_probe reads the probe."""
    from src.database.models import VideoSource

    parent_id, first_clip = await _failed_reel_parent(client, db_session)
    row = (
        await db_session.execute(select(VideoSource).where(VideoSource.id == first_clip))
    ).scalar_one()
    row.probe = {"format": {"duration": "12.5"}}
    await db_session.commit()

    await _assert_footage_and_dependents_re_run(client, db_session, parent_id)


async def test_restart_of_a_parent_without_the_fingerprint_re_runs_everything(
    client, db_session
) -> None:
    """A run started before the fingerprint shipped cannot prove its inputs held."""
    project_id, deliv = await _make_project(client)
    parent_id = await _seed_failed_parent(db_session, project_id, deliv, with_resume_inputs=False)

    await _assert_restart_re_runs_everything(client, db_session, parent_id)


async def test_restart_of_a_run_grounded_in_an_uploaded_document_re_runs_everything(
    client, db_session
) -> None:
    """The document ref survives a same-filename re-upload that replaces its bytes."""
    project_id, deliv = await _make_project(client)
    patched = await client.patch(
        f"/api/projects/{project_id}",
        json={
            "brief": {
                "source_url": "https://example.com",
                "sources": [{"kind": "document", "ref": "artifacts/documents/t/upload/a.pdf"}],
            }
        },
    )
    assert patched.status_code == 200, patched.text
    parent_id = await _seed_failed_parent(db_session, project_id, deliv)

    await _assert_restart_re_runs_everything(client, db_session, parent_id)


async def test_restart_of_a_blueprint_off_the_allowlist_re_runs_everything(
    client, db_session, monkeypatch
) -> None:
    from src.services.runs import restart_plan

    project_id, deliv = await _make_project(client)
    parent_id = await _seed_failed_parent(db_session, project_id, deliv)
    monkeypatch.setattr(restart_plan, "RESUMABLE_BLUEPRINTS", frozenset({"action_reel_v1"}))

    await _assert_restart_re_runs_everything(client, db_session, parent_id)


async def test_start_run_persists_the_fingerprint_the_resume_check_rebuilds(
    client, db_session
) -> None:
    """Apples to apples: the parent's key comes from start_run itself, not a test helper."""
    from src.services.runs.resume_inputs import RESUME_INPUTS_KEY

    project_id, deliv = await _make_reel_project(client)
    clip = await _add_clip(db_session, project_id, "a.mp4")
    r = await client.post(f"/api/deliverables/{deliv}/runs", json={"pause_after_stages": []})
    assert r.status_code in (200, 201, 202), r.text
    run = await _run(db_session, UUID(r.json()["id"]))
    inputs = (run.config_snapshot or {})[RESUME_INPUTS_KEY]
    assert inputs["unresumable"] == []
    assert [row[0] for row in inputs["video_sources"]] == [str(clip)]


async def _run(db_session, run_id: UUID):
    from src.database.models import Run

    db_session.expire_all()
    return (await db_session.execute(select(Run).where(Run.id == run_id))).scalar_one()


async def _project_of(db_session, deliverable_id: UUID) -> UUID:
    from src.database.models import Deliverable

    deliverable = (
        await db_session.execute(select(Deliverable).where(Deliverable.id == deliverable_id))
    ).scalar_one()
    return deliverable.project_id
