"""Route tests for storyboard panel image generation (spec 00, Phase 2).

The POST is 202 + a job; the work runs as a tracked background task and the
result lands on the PANEL. These cover the HTTP contract around that: the job
reaches ``completed`` and the panel carries the image, a stale
``expected_lock_version`` is a 409 BEFORE anything is paid for, another tenant
sees 404 on both the submit and the poll, and a job id belonging to a different
panel is not readable through this panel's path.

Plus the two properties that only bite AFTER the happy path works: a panel read
hands the browser a URL it can actually load (the stored one is a presigned
``minio.test`` URL — wrong host AND an expired signature), and a
content-addressed cache hit is visibly ``reused`` rather than indistinguishable
from a generator that produced nothing.

The proxy route that serves those rewritten URLs is covered separately in
``test_storyboard_panel_image_proxy.py``.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from src.clients.schemas.ai_content_generation import AcgConceptImageJob
from src.database.models import Project
from src.database.repositories.storyboard_repository import StoryboardPanelRepository
from src.services.storyboard.panel_image_jobs import PanelImageJobRunner
from src.services.tts_storage import set_storage_for_tests

pytestmark = pytest.mark.asyncio

# Default for the tests that do not assert on the url at all. Deliberately NOT
# under a project's own prefix, so no assertion here can pass by accident.
_IMAGE_URL = "https://minio.test/w2v-orchestrator/artifacts/somewhere/else.png"


def _owned_image_url(project_id: UUID, name: str = "asset-1") -> str:
    """The url a real generation stores for this project.

    ``StoryboardPanelImageService._namespace`` (``storyboard/{project_id}``) +
    ``STAGE_NAME`` + the content-addressed media id, as
    ``storage_service.store_stage_artifact`` lays it out.
    """
    return (
        "https://minio.test/w2v-orchestrator/"
        f"artifacts/storyboard/{project_id}/storyboard_panel_image/{name}.png"
    )


def _proxy_path(panel_id: str) -> str:
    """What the API must hand a browser instead of the stored in-cluster url."""
    return f"/api/storyboard/panels/{panel_id}/image/content"


def _make_client(*, image_url: str | None = _IMAGE_URL) -> AsyncMock:
    client = AsyncMock()
    client.create_concept_image = AsyncMock(return_value="job-1")
    client.wait_for_concept_image = AsyncMock(
        return_value=AcgConceptImageJob(
            job_id="job-1",
            asset_id="asset-1",
            status="completed",
            provider="local_image",
            image_url=image_url,
        )
    )
    client.download_concept_image = AsyncMock(return_value=b"\x89PNGfake")
    return client


async def _seed_project(db_session, *, tenant_id: UUID, subject_id: UUID) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="panel-image-routes",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    db_session.add(project)
    await db_session.commit()
    return project


async def _seed_panel(client, project_id: UUID) -> dict[str, Any]:
    resp = await client.post(
        f"/api/projects/{project_id}/storyboard/panels",
        json={"action_description": "Mara vaults the barrier", "shot_size": "MS"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _read_panel(client, project_id: UUID, panel_id: str) -> dict[str, Any]:
    """Re-read the panel off the list route (there is no by-id panel GET)."""
    resp = await client.get(f"/api/projects/{project_id}/storyboard/panels")
    assert resp.status_code == 200, resp.text
    rows = [p for p in resp.json()["items"] if p["id"] == panel_id]
    assert rows, f"panel {panel_id} vanished"
    return rows[0]


def _install_runner(app, acg: AsyncMock) -> PanelImageJobRunner:
    """Pre-seed the app's lazily-created runner with a fake ACG client."""
    runner = PanelImageJobRunner(
        session_factory=app.state.session_factory, redis=app.state.redis, client=acg
    )
    app.state.panel_image_jobs = runner
    return runner


async def test_generate_panel_image_end_to_end(app, client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id)
    runner = _install_runner(app, _make_client(image_url=_owned_image_url(project.id)))

    accepted = await client.post(f"/api/storyboard/panels/{panel['id']}/image", json={})
    assert accepted.status_code == 202, accepted.text
    job = accepted.json()
    assert job["status"] in {"queued", "running"}
    assert job["panel_id"] == panel["id"]

    await runner.wait_for(job["job_id"])

    polled = await client.get(f"/api/storyboard/panels/{panel['id']}/image/{job['job_id']}")
    assert polled.status_code == 200, polled.text
    body = polled.json()
    assert body["status"] == "completed", body
    assert body["media_ref"]["url"] == _proxy_path(panel["id"])
    assert body["gen"]["seed"] is not None

    # The durable outcome is the PANEL, not the job record.
    refreshed = await _read_panel(client, project.id, panel["id"])
    assert refreshed["media_ref"]["url"] == _proxy_path(panel["id"])
    assert refreshed["visual_type"] == "ai_image"
    assert refreshed["gen"]["seed"] == body["gen"]["seed"]


async def test_dead_downstream_url_surfaces_as_a_failed_job(app, client, db_session) -> None:
    """A generation that produced nothing usable must read as FAILED, never as a
    completed job pointing at a broken image."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id)
    runner = _install_runner(app, _make_client(image_url="none"))

    job = (await client.post(f"/api/storyboard/panels/{panel['id']}/image", json={})).json()
    await runner.wait_for(job["job_id"])

    body = (await client.get(f"/api/storyboard/panels/{panel['id']}/image/{job['job_id']}")).json()
    assert body["status"] == "failed"
    assert "image_url" in body["error"]
    assert (await _read_panel(client, project.id, panel["id"]))["media_ref"] == {}


async def test_stale_lock_version_is_409_before_anything_is_generated(
    app, client, db_session
) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id)
    acg = _make_client()
    _install_runner(app, acg)

    conflict = await client.post(
        f"/api/storyboard/panels/{panel['id']}/image",
        json={"expected_lock_version": panel["lock_version"] + 5},
    )

    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "panel was modified concurrently"
    acg.create_concept_image.assert_not_awaited()


async def test_matching_lock_version_is_accepted(app, client, db_session) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id)
    runner = _install_runner(app, _make_client())

    accepted = await client.post(
        f"/api/storyboard/panels/{panel['id']}/image",
        json={"expected_lock_version": panel["lock_version"]},
    )

    assert accepted.status_code == 202
    await runner.wait_for(accepted.json()["job_id"])


async def test_cross_tenant_panel_is_404_on_submit_and_poll(
    app, client, other_tenant_client, db_session
) -> None:
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id)
    runner = _install_runner(app, _make_client())
    job = (await client.post(f"/api/storyboard/panels/{panel['id']}/image", json={})).json()
    await runner.wait_for(job["job_id"])

    assert (
        await other_tenant_client.post(f"/api/storyboard/panels/{panel['id']}/image", json={})
    ).status_code == 404
    assert (
        await other_tenant_client.get(f"/api/storyboard/panels/{panel['id']}/image/{job['job_id']}")
    ).status_code == 404


async def test_job_id_of_another_panel_is_not_readable_here(app, client, db_session) -> None:
    """The job record lives in redis and is not an authorization artifact — the
    path's panel must own the job."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel_a = await _seed_panel(client, project.id)
    panel_b = await _seed_panel(client, project.id)
    runner = _install_runner(app, _make_client())
    job = (await client.post(f"/api/storyboard/panels/{panel_a['id']}/image", json={})).json()
    await runner.wait_for(job["job_id"])

    mismatched = await client.get(f"/api/storyboard/panels/{panel_b['id']}/image/{job['job_id']}")
    assert mismatched.status_code == 404


# ─── what the browser is handed on read ──────────────────────────────────
class _ResigningStorage:
    """Storage that re-mints an owned MinIO URL with a visibly fresh signature."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def resign_presigned_url(self, url: str) -> str | None:
        self.calls.append(url)
        if "minio.test" not in url:
            return None  # not ours — the real guard declines foreign URLs
        return f"{url.split('?', 1)[0]}?X-Amz-Signature=fresh"


@pytest.fixture
def resigning_storage():
    storage = _ResigningStorage()
    set_storage_for_tests(cast("Any", storage))
    try:
        yield storage
    finally:
        set_storage_for_tests(None)


async def test_panel_read_serves_a_url_the_browser_can_actually_load(
    app, client, db_session, resigning_storage
) -> None:
    """The stored ``media_ref.url`` is unusable to the SPA on TWO counts: its host
    is ``minio.test`` (Tailnet split-DNS — it does not resolve in a browser at
    all) and its signature is presigned for one hour while a board is read for
    days. The read path must therefore hand back the panel's own proxy path.

    Asserting merely that *a* url is present would pass with the dead in-cluster
    one, which is the actual bug. The re-sign that still matters — for the
    server-side PDF export, which fetches from inside the cluster — is proven in
    ``test_storyboard_export_routes.py``.
    """
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id)
    owned = _owned_image_url(project.id)
    expired = f"{owned}?X-Amz-Signature=expired&X-Amz-Date=20200101T000000Z"
    runner = _install_runner(app, _make_client(image_url=expired))
    job = (await client.post(f"/api/storyboard/panels/{panel['id']}/image", json={})).json()
    await runner.wait_for(job["job_id"])

    refreshed = await _read_panel(client, project.id, panel["id"])

    assert refreshed["media_ref"]["url"] == _proxy_path(panel["id"])
    assert "minio.test" not in refreshed["media_ref"]["url"], (
        "an in-cluster host was served — the SPA cannot resolve it and the image never loads"
    )
    assert "X-Amz-Signature" not in refreshed["media_ref"]["url"], (
        "a presigned signature was served — it is dead an hour after generation"
    )
    # The stored value is untouched: it is the durable object identity, and
    # rewriting it per read would just persist a URL that is stale again in 1h.
    row = await StoryboardPanelRepository(db_session).get(
        UUID(panel["id"]), tenant_id=UUID(client.tenant_id)
    )
    assert row is not None
    assert row.media_ref["url"] == expired


async def test_a_foreign_panel_image_url_is_left_alone(app, client, db_session) -> None:
    """An operator-pasted external image needs no signature; re-signing it would
    mint a pointer into OUR MinIO for an object that isn't there."""
    external = "https://example.com/art/reference.png"
    set_storage_for_tests(cast("Any", _ResigningStorage()))
    try:
        project = await _seed_project(
            db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
        )
        panel = await _seed_panel(client, project.id)
        runner = _install_runner(app, _make_client(image_url=external))
        job = (await client.post(f"/api/storyboard/panels/{panel['id']}/image", json={})).json()
        await runner.wait_for(job["job_id"])

        assert (await _read_panel(client, project.id, panel["id"]))["media_ref"]["url"] == external
    finally:
        set_storage_for_tests(None)


async def test_a_reused_generation_is_visible_as_reused(app, client, db_session) -> None:
    """A content-addressed cache hit returns in milliseconds. Without an explicit
    flag that is indistinguishable from a generator that silently produced
    nothing — which would make a hardware proof run worthless."""
    project = await _seed_project(
        db_session, tenant_id=UUID(client.tenant_id), subject_id=UUID(client.subject_id)
    )
    panel = await _seed_panel(client, project.id)
    acg = _make_client()
    runner = _install_runner(app, acg)
    first = (await client.post(f"/api/storyboard/panels/{panel['id']}/image", json={})).json()
    await runner.wait_for(first["job_id"])
    first_body = (
        await client.get(f"/api/storyboard/panels/{panel['id']}/image/{first['job_id']}")
    ).json()
    assert first_body["gen"]["reused"] is False, first_body

    # Now every stored object "exists", so the identical panel addresses the
    # SAME media id and must be served from storage without paying again.
    set_storage_for_tests(cast("Any", _AlwaysStoredStorage()))
    try:
        acg.create_concept_image.reset_mock()
        second = (await client.post(f"/api/storyboard/panels/{panel['id']}/image", json={})).json()
        await runner.wait_for(second["job_id"])
        second_body = (
            await client.get(f"/api/storyboard/panels/{panel['id']}/image/{second['job_id']}")
        ).json()
    finally:
        set_storage_for_tests(None)

    assert second_body["status"] == "completed", second_body
    assert second_body["gen"]["reused"] is True, (
        "a cache hit is indistinguishable from a broken generator without this flag"
    )
    acg.create_concept_image.assert_not_awaited()


class _AlwaysStoredStorage:
    """Storage whose objects always exist — every content-addressed lookup hits."""

    async def stage_artifact_url(self, run_id: str, stage_name: str, filename: str) -> str:
        return f"https://minio.test/w2v-orchestrator/artifacts/{run_id}/{stage_name}/{filename}"

    async def object_exists(self, url: str) -> bool:
        return True

    async def resign_presigned_url(self, url: str) -> str | None:
        return None
