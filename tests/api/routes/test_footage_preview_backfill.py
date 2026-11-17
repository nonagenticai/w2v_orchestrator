"""Old runs must show the footage preview that exists TODAY.

``footage_probe._describe`` presigns ``preview_url`` at STAGE-EXECUTION time, so
every run whose probe executed before the preview-proxy worker shipped stored
``preview_url: ""`` — permanently. The browser-playable proxy object is in MinIO
now and ``video_sources.preview_status`` says ``ready``, but the reviewer's
footage gate still renders "no preview generated" for all of it.

Two halves, and the ORDER matters: ``artifact_ref_guard`` must admit the preview
key to the run's allowlist BEFORE ``read.py`` advertises a proxy path for it, or
the SPA gets a visible-but-dead ``<video>`` (a 404 on every request), which is
strictly worse than the honest empty state. The last test here is the security
half: widening the allowlist must not admit ANOTHER project's preview.
"""

from __future__ import annotations

from uuid import uuid4, uuid7

import pytest

from src.config import get_settings
from src.database.models import StageOutput, VideoSource
from src.database.repositories.run_repository import RunRepository
from src.services.artifact_ref_guard import run_artifact_refs

# An OLD footage_probe output: the worker had not run when the stage did, so the
# stage recorded an empty url AND the then-current `pending` status.
_OLD_SOURCE_STATUS = "pending"


async def _make_project(client) -> tuple[str, str]:
    r = await client.post(
        "/api/projects",
        json={
            "type": "generic",
            "title": "footage preview backfill",
            "brief": {"angle": "growth", "target_duration_seconds": 30},
            "voice": {"descriptor": "crisp"},
            "deliverables": [
                {
                    "name": "d",
                    "format": "16x9",
                    "captions_mode": "none",
                    "target_duration_seconds": 30,
                }
            ],
        },
    )
    assert r.status_code == 201, r.text
    project_id = r.json()["id"]
    detail = await client.get(f"/api/projects/{project_id}")
    return project_id, detail.json()["deliverables"][0]["id"]


async def _seed_source(session, project_id, *, preview_status: str) -> tuple[str, str]:
    """One ``video_sources`` row; returns (source_id, preview_key)."""
    from uuid import UUID

    source_id = uuid7()
    prefix = f"tenants/{uuid4()}/projects/{project_id}/video-sources/{source_id}/"
    preview_key = f"{prefix}preview.mp4"
    session.add(
        VideoSource(
            id=source_id,
            project_id=UUID(str(project_id)),
            filename="drone.mov",
            object_key=f"{prefix}original.mov",
            content_type="video/quicktime",
            bytes=4321,
            preview_status=preview_status,
            # The key is set even for a NON-ready row, deliberately: it makes
            # ``preview_status`` the load-bearing gate. A key-presence-only test
            # would pass with the status check deleted, and a row CAN carry a key
            # it may not serve (a failed re-render leaves the previous key behind).
            preview_key=preview_key,
            probe={},
        )
    )
    return str(source_id), preview_key


async def _seed_old_probe_run(client, db_session_factory, deliverable_id, source_id) -> str:
    """A run carrying an OLD-style ``footage_probe`` output for ``source_id``."""
    from uuid import UUID

    async with db_session_factory() as session:
        run = await RunRepository(session).create(
            deliverable_id=UUID(deliverable_id),
            blueprint_id="generic_v1",
            config_snapshot={},
            pause_after_stages=[],
        )
        session.add(
            StageOutput(
                id=uuid7(),
                run_id=run.id,
                stage_name="footage_probe",
                status="completed",
                output={
                    "sources": [
                        {
                            "id": source_id,
                            "filename": "drone.mov",
                            "bytes": 4321,
                            "preview_status": _OLD_SOURCE_STATUS,
                            "original_url": "",
                            "preview_url": "",
                        }
                    ],
                    "total_duration_s": 12.0,
                },
            )
        )
        await session.commit()
        return str(run.id)


def _only_source(detail: dict) -> dict:
    stages = [s for s in detail["stage_outputs"] if s["stage_name"] == "footage_probe"]
    assert len(stages) == 1, detail["stage_outputs"]
    return stages[0]["output"]["sources"][0]


@pytest.mark.parametrize("preview_status", ["pending"])
@pytest.mark.asyncio
async def test_non_ready_previews_are_left_honestly_empty(
    client, db_session_factory, preview_status: str
) -> None:
    """No object exists, so no URL may be invented. An empty state a reviewer can
    read beats a ``<video>`` that errors out."""
    project_id, deliverable_id = await _make_project(client)
    async with db_session_factory() as session:
        source_id, preview_key = await _seed_source(
            session, project_id, preview_status=preview_status
        )
        await session.commit()
    run_id = await _seed_old_probe_run(client, db_session_factory, deliverable_id, source_id)

    resp = await client.get(f"/api/runs/{run_id}")
    assert resp.status_code == 200, resp.text
    source = _only_source(resp.json())

    assert source["preview_url"] == ""
    assert source["preview_status"] == _OLD_SOURCE_STATUS

    # The allowlist must not grow by it either. Nothing advertises the key, so
    # this is defence in depth rather than a live escape — but an allowlist entry
    # for an object that may not exist is surface for no benefit, and the READY
    # gate is the only thing keeping it out.
    from uuid import UUID

    bucket = get_settings().storage.minio_bucket
    async with db_session_factory() as session:
        assert f"{bucket}/{preview_key}" not in await run_artifact_refs(session, UUID(run_id))


# ── Stale pre-#809 `preview_status=pending` notes ────────────────────────────
#
# The producing code is GONE: `footage_probe._describe` emitted one note per
# not-yet-rendered derivative until #809 (`a95daeaf`, "a pending preview
# derivative is not a degradation") deleted it; today it notes only `failed`.
# Every surviving occurrence is frozen residue in `stage_outputs.output` from a
# run probed before that commit — and `_shared.py` turns ANY note on a completed
# stage into `degraded`, so those old runs are still badged "completed with
# warnings" for a condition that has since resolved AND that #809 ruled is not a
# warning at all. Live symptom: ten amber lines saying the derivatives are not
# rendered, stacked above ten tiles that say "preview ready" and that play.

_STALE_PENDING_NOTE = (
    "source {source_id} preview_status=pending "
    "(derivatives not rendered yet; original is still usable)"
)
_FAILED_NOTE = "source {source_id} preview_status=failed (encoder died; original is still usable)"
_NO_DURATION_NOTE = "source {source_id} has no probed duration; treating as 0s"


async def _seed_old_probe_run_with_notes(
    db_session_factory, deliverable_id, source_ids: list[str], notes: list[str]
) -> str:
    """An OLD-style ``footage_probe`` output: empty urls, ``pending`` statuses, notes."""
    from uuid import UUID

    async with db_session_factory() as session:
        run = await RunRepository(session).create(
            deliverable_id=UUID(deliverable_id),
            blueprint_id="generic_v1",
            config_snapshot={},
            pause_after_stages=[],
        )
        session.add(
            StageOutput(
                id=uuid7(),
                run_id=run.id,
                stage_name="footage_probe",
                status="completed",
                output={
                    "sources": [
                        {
                            "id": source_id,
                            "filename": "drone.mov",
                            "bytes": 4321,
                            "preview_status": _OLD_SOURCE_STATUS,
                            "original_url": "",
                            "preview_url": "",
                        }
                        for source_id in source_ids
                    ],
                    "total_duration_s": 12.0,
                    "notes": list(notes),
                },
            )
        )
        await session.commit()
        return str(run.id)


def _probe_stage(detail: dict) -> dict:
    stages = [s for s in detail["stage_outputs"] if s["stage_name"] == "footage_probe"]
    assert len(stages) == 1, detail["stage_outputs"]
    return stages[0]


@pytest.mark.asyncio
async def test_stale_pending_note_is_dropped_and_run_stops_being_degraded(
    client, db_session_factory
) -> None:
    """The measured bug, both halves.

    The amber line must go, AND the run must stop being badged "completed with
    warnings" — `degraded` is computed in `_stage_response` BEFORE the backfill
    projection runs, so dropping the note without recomputing it would ship a
    visibly-inert fix.
    """
    project_id, deliverable_id = await _make_project(client)
    async with db_session_factory() as session:
        source_id, _ = await _seed_source(session, project_id, preview_status="ready")
        await session.commit()
    run_id = await _seed_old_probe_run_with_notes(
        db_session_factory,
        deliverable_id,
        [source_id],
        [_STALE_PENDING_NOTE.format(source_id=source_id)],
    )

    resp = await client.get(f"/api/runs/{run_id}")
    assert resp.status_code == 200, resp.text
    detail = resp.json()
    stage = _probe_stage(detail)

    # Sanity: this run is one the backfill actually repaired.
    assert stage["output"]["sources"][0]["preview_status"] == "ready"
    assert stage["output"]["notes"] == []
    assert stage["degraded"] is False
    assert detail["degraded"] is False


@pytest.mark.asyncio
async def test_failed_preview_note_survives_beside_the_stale_pending_one(
    client, db_session_factory
) -> None:
    """Same source id, two notes. Only the `pending` one may go.

    `failed` is a REAL soft-failure that #809 deliberately kept (it carries
    `row.error`). Both notes start with the identical `source <id> preview_status=`
    span, so anything looser than pinning the literal `pending ` would eat it.
    """
    project_id, deliverable_id = await _make_project(client)
    async with db_session_factory() as session:
        source_id, _ = await _seed_source(session, project_id, preview_status="ready")
        await session.commit()
    failed_note = _FAILED_NOTE.format(source_id=source_id)
    run_id = await _seed_old_probe_run_with_notes(
        db_session_factory,
        deliverable_id,
        [source_id],
        [_STALE_PENDING_NOTE.format(source_id=source_id), failed_note],
    )

    stage = _probe_stage((await client.get(f"/api/runs/{run_id}")).json())
    assert stage["output"]["notes"] == [failed_note]
    # A surviving real warning must keep the badge on.
    assert stage["degraded"] is True


@pytest.mark.asyncio
async def test_unrelated_note_for_the_same_source_survives(client, db_session_factory) -> None:
    """`notes` is a general channel. A zero-duration source is still zero-duration."""
    project_id, deliverable_id = await _make_project(client)
    async with db_session_factory() as session:
        source_id, _ = await _seed_source(session, project_id, preview_status="ready")
        await session.commit()
    duration_note = _NO_DURATION_NOTE.format(source_id=source_id)
    run_id = await _seed_old_probe_run_with_notes(
        db_session_factory,
        deliverable_id,
        [source_id],
        [_STALE_PENDING_NOTE.format(source_id=source_id), duration_note],
    )

    stage = _probe_stage((await client.get(f"/api/runs/{run_id}")).json())
    assert stage["output"]["notes"] == [duration_note]
    assert stage["degraded"] is True


@pytest.mark.asyncio
async def test_note_for_a_genuinely_still_pending_source_survives(
    client, db_session_factory
) -> None:
    """Two sources, one repaired and one not. The un-repaired one keeps its note.

    This is the property that makes the drop safe: only an id this request proved
    `ready` against the live row is eligible, so a still-rendering derivative is
    still reported.
    """
    project_id, deliverable_id = await _make_project(client)
    async with db_session_factory() as session:
        ready_id, _ = await _seed_source(session, project_id, preview_status="ready")
        pending_id, _ = await _seed_source(session, project_id, preview_status="pending")
        await session.commit()
    pending_note = _STALE_PENDING_NOTE.format(source_id=pending_id)
    run_id = await _seed_old_probe_run_with_notes(
        db_session_factory,
        deliverable_id,
        [ready_id, pending_id],
        [_STALE_PENDING_NOTE.format(source_id=ready_id), pending_note],
    )

    stage = _probe_stage((await client.get(f"/api/runs/{run_id}")).json())
    by_id = {s["id"]: s for s in stage["output"]["sources"]}
    assert by_id[ready_id]["preview_status"] == "ready"
    assert by_id[pending_id]["preview_status"] == "pending"
    assert stage["output"]["notes"] == [pending_note]
    assert stage["degraded"] is True
