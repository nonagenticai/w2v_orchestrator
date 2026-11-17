"""Studio v3 — /api/youtube publish route tests.

The upload route resolves a completed run's final video artifact and mints a
fresh presigned MinIO URL for it (the upload service fetches that directly).
These tests mock the downstream ``YouTubeUploadClient`` (set on
``app.state.youtube_client``), inject a fake storage backend (so presigning is
deterministic without a real MinIO), and assert the translation + run guards.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import httpx
import pytest

from src.services import tts_storage

pytestmark = pytest.mark.asyncio


class _FakeStorage:
    """Minimal storage stand-in: echoes a presigned URL from (bucket, key)."""

    presign_returns: str | None = "__echo__"

    async def initialize(self) -> None:
        return None

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        return f"artifacts/{job_id}/{stage_name}/{filename}"

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        if self.presign_returns != "__echo__":
            return self.presign_returns
        return f"http://minio.test:8389/{bucket}/{path}?X-Amz-Signature=test"

    async def resign_presigned_url(self, url: str) -> str | None:
        # Mirror StorageService: re-derive {bucket}/{key} from the path, re-sign.
        from urllib.parse import urlparse

        bucket, _, key = urlparse(url).path.lstrip("/").partition("/")
        if not key:
            return None
        return await self.get_presigned_url(key, bucket=bucket)


@pytest.fixture(autouse=True)
def _fake_storage() -> Iterator[_FakeStorage]:
    """Override the process-wide storage singleton with a deterministic fake."""
    fake = _FakeStorage()
    tts_storage.set_storage_for_tests(fake)
    try:
        yield fake
    finally:
        tts_storage.set_storage_for_tests(None)


_PROJECT_PAYLOAD = {
    "type": "generic",
    "title": "for publish",
    "brief": {"angle": "growth", "target_duration_seconds": 30},
    "voice": {"descriptor": "crisp"},
    "deliverables": [
        {"name": "d", "format": "16x9", "captions_mode": "none", "target_duration_seconds": 30}
    ],
}


async def _make_project(client) -> tuple[str, str]:
    r = await client.post("/api/projects", json=_PROJECT_PAYLOAD)
    assert r.status_code == 201, r.text
    project_id = r.json()["id"]
    detail = await client.get(f"/api/projects/{project_id}")
    return project_id, detail.json()["deliverables"][0]["id"]


async def _seed_completed_run(
    db_session_factory,
    deliverable_id: str,
    stage: str,
    output: dict,
    *,
    completed: bool = True,
) -> str:
    """Create a run on ``deliverable_id`` with one completed stage output."""
    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository

    async with db_session_factory() as session:
        run_repo = RunRepository(session)
        run = await run_repo.create(
            deliverable_id=UUID(deliverable_id),
            blueprint_id="lesson_v1",
            config_snapshot={},
            pause_after_stages=[],
        )
        so_repo = StageOutputRepository(session)
        await so_repo.upsert_pending(run.id, stage)
        await so_repo.mark_completed(run.id, stage, output)
        if completed:
            await run_repo.mark_completed(run.id)
        await session.commit()
        return str(run.id)


async def _seed_run_with_stages(
    db_session_factory,
    deliverable_id: str,
    stages: dict[str, dict],
    *,
    config_snapshot: dict | None = None,
    blueprint_id: str = "course_v1",
) -> str:
    """Create a completed run with several completed stage outputs."""
    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository

    async with db_session_factory() as session:
        run_repo = RunRepository(session)
        run = await run_repo.create(
            deliverable_id=UUID(deliverable_id),
            blueprint_id=blueprint_id,
            config_snapshot=config_snapshot or {},
            pause_after_stages=[],
        )
        so_repo = StageOutputRepository(session)
        for stage, output in stages.items():
            await so_repo.upsert_pending(run.id, stage)
            await so_repo.mark_completed(run.id, stage, output)
        await run_repo.mark_completed(run.id)
        await session.commit()
        return str(run.id)


_CHAPTERS_3 = [
    {"sequence": 1, "title": "Intro", "start_seconds": 0.0, "duration_seconds": 60.0},
    {"sequence": 2, "title": "Core", "start_seconds": 60.0, "duration_seconds": 40.0},
    {"sequence": 3, "title": "Wrap", "start_seconds": 100.0, "duration_seconds": 20.0},
]


def _mock_client(app, **overrides) -> AsyncMock:
    mock = AsyncMock()
    mock.start_upload.return_value = {
        "job_id": "job-123",
        "status": "pending",
        "youtube_url": None,
    }
    for k, v in overrides.items():
        getattr(mock, k).return_value = v
    app.state.youtube_client = mock
    return mock


async def test_publish_remints_fresh_presigned_url(client, app, db_session_factory):
    """A completed lesson run publishes via a freshly-minted presigned MinIO URL.

    The stored artifact URL's presign query is irrelevant — only its
    ``{bucket}/{object_key}`` identity is used to re-mint a fresh presigned URL.
    """
    _, deliv = await _make_project(client)
    run_id = await _seed_completed_run(
        db_session_factory,
        deliv,
        "lesson_compile",
        {"artifact_url": "http://minio.test:8389/w2v-compiled-videos/lesson-1.mp4?X=stale"},
    )
    mock = _mock_client(app)

    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "My Lesson"},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["job_id"] == "job-123"
    mock.start_upload.assert_awaited_once()
    payload = mock.start_upload.await_args.args[0]
    # (bucket, object_key) parsed from the stored URL → fresh presign (not the stale ?X=stale).
    assert payload["file_source"] == (
        "http://minio.test:8389/w2v-compiled-videos/lesson-1.mp4?X-Amz-Signature=test"
    )
    assert payload["file_source_type"] == "url"
    assert payload["title"] == "My Lesson"
    assert payload["privacy_status"] == "unlisted"
    assert "description" not in payload  # omitted when not provided


async def test_publish_502_when_presign_fails(client, app, db_session_factory, _fake_storage):
    """If the backend cannot presign (e.g. local backend) → 502, no upload started."""
    _fake_storage.presign_returns = None
    _, deliv = await _make_project(client)
    run_id = await _seed_completed_run(
        db_session_factory,
        deliv,
        "lesson_compile",
        {"artifact_url": "http://minio.test:8389/w2v-compiled-videos/lesson-1.mp4"},
    )
    mock = _mock_client(app)

    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "x"},
    )

    assert resp.status_code == 502, resp.text
    mock.start_upload.assert_not_awaited()


async def test_publish_prefers_course_stitch_over_compile(client, app, db_session_factory):
    """course_artifact_url wins over a lesson artifact_url on the same run."""
    _, deliv = await _make_project(client)
    run_id = await _seed_completed_run(
        db_session_factory,
        deliv,
        "course_stitch",
        {"course_artifact_url": "http://minio.test:8389/w2v-courses/course-9.mp4"},
    )
    mock = _mock_client(app)

    resp = await client.post(
        "/api/youtube/upload",
        json={
            "pipeline_job_id": run_id,
            "title": "My Course",
            "description": "desc",
            "privacy_status": "public",
        },
    )

    assert resp.status_code == 200, resp.text
    payload = mock.start_upload.await_args.args[0]
    assert payload["file_source"] == (
        "http://minio.test:8389/w2v-courses/course-9.mp4?X-Amz-Signature=test"
    )
    assert payload["description"] == "desc"


_MINIO = "http://minio.test:8389/w2v-compiled-videos"


@pytest.mark.parametrize("blueprint_id", ["showcase_v1"])
async def test_publish_uploads_the_delivered_artifact_not_the_compile(
    client, app, db_session_factory, blueprint_id
):
    """W11: the upload source is the blueprint's ``delivered_artifact_stage``.

    On showcase/demo_video the compile is an INTERMEDIATE: ``final_critique``
    re-masters it and owns the file the customer gets. The old hardcoded
    terminal-stage list picked ``demo_video_compile`` and published the
    un-mastered cut.
    """
    _, deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "demo_video_compile": {"artifact_url": f"{_MINIO}/cut.mp4"},
            "demo_video_encode_master": {"artifact_url": f"{_MINIO}/master.mp4"},
            "final_critique": {"artifact_url": f"{_MINIO}/delivered.mp4"},
        },
        blueprint_id=blueprint_id,
    )
    mock = _mock_client(app)

    resp = await client.post("/api/youtube/upload", json={"pipeline_job_id": run_id, "title": "t"})

    assert resp.status_code == 200, resp.text
    payload = mock.start_upload.await_args.args[0]
    assert payload["file_source"] == f"{_MINIO}/delivered.mp4?X-Amz-Signature=test"


async def test_publish_delivered_chain_falls_through_a_skipped_stage(
    client, app, db_session_factory
):
    """A skip-clean ``final_critique`` (``{}``) falls through to the master, exactly
    as ``GET /api/runs/{id}``'s ``delivered_artifact_url`` does — never to the cut."""
    _, deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "demo_video_compile": {"artifact_url": f"{_MINIO}/cut.mp4"},
            "demo_video_encode_master": {"artifact_url": f"{_MINIO}/master.mp4"},
            "final_critique": {},
        },
        blueprint_id="showcase_v1",
    )
    mock = _mock_client(app)

    resp = await client.post("/api/youtube/upload", json={"pipeline_job_id": run_id, "title": "t"})

    assert resp.status_code == 200, resp.text
    payload = mock.start_upload.await_args.args[0]
    assert payload["file_source"] == f"{_MINIO}/master.mp4?X-Amz-Signature=test"


async def test_publish_attaches_soft_cc_for_promo(client, app, db_session_factory):
    """A promo run with script + narrate gets a minted soft-CC caption_source."""
    _, deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "promo_script": {"lines": [{"id": "l1", "text": "Hello world"}]},
            "promo_narrate": {"clips": [{"line_id": "l1", "seconds": 2.0}]},
            "promo_render": {
                "artifact_url": "http://minio.test:8389/w2v-compiled-videos/promo-1.mp4"
            },
        },
    )
    mock = _mock_client(app)

    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "My Promo"},
    )

    assert resp.status_code == 200, resp.text
    payload = mock.start_upload.await_args.args[0]
    assert payload["caption_source_type"] == "url"
    assert payload["caption_language"] == "en"
    # Stored under the run's soft_cc stage, then re-presigned (fresh signature).
    assert "/soft_cc/" in payload["caption_source"]
    assert payload["caption_source"].endswith("?X-Amz-Signature=test")
    # The minted URL is echoed back to the client so the run view can offer the
    # .srt as a download (the upload service's own response doesn't carry it).
    assert resp.json()["caption_source"] == payload["caption_source"]


async def test_publish_skips_soft_cc_when_burned_in(client, app, db_session_factory):
    """A burned_in deliverable must NOT also get a soft-CC track (no double captions)."""
    _, deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "promo_script": {"lines": [{"id": "l1", "text": "Hello world"}]},
            "promo_narrate": {"clips": [{"line_id": "l1", "seconds": 2.0}]},
            "promo_render": {
                "artifact_url": "http://minio.test:8389/w2v-compiled-videos/promo-1.mp4"
            },
        },
        config_snapshot={"deliverable": {"captions_mode": "burned_in"}},
    )
    mock = _mock_client(app)

    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "My Promo"},
    )

    assert resp.status_code == 200, resp.text
    payload = mock.start_upload.await_args.args[0]
    # Captions are in the pixels — no soft-CC track attached, none echoed back.
    assert "caption_source" not in payload
    assert "caption_source" not in resp.json()


async def test_publish_attaches_soft_cc_for_course(client, app, db_session_factory):
    """A course mints soft-CC by concatenating its child lessons, offset by chapter start."""
    _, deliv = await _make_project(client)
    # Two child lesson runs, each carrying its own script + narrate.
    child1 = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "lesson_script": {"paragraphs": [{"id": "p1", "text": "Lesson one para"}]},
            "lesson_narrate": {"clips": [{"paragraph_id": "p1", "seconds": 3.0}]},
        },
    )
    child2 = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "lesson_script": {"paragraphs": [{"id": "p1", "text": "Lesson two para"}]},
            "lesson_narrate": {"clips": [{"paragraph_id": "p1", "seconds": 4.0}]},
        },
    )
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "course_compile_lessons": {
                "lessons": [
                    {"lesson_project_id": "proj-1", "lesson_run_id": child1},
                    {"lesson_project_id": "proj-2", "lesson_run_id": child2},
                ]
            },
            "course_stitch": {
                "course_artifact_url": "http://minio.test:8389/w2v-courses/course-9.mp4",
                "duration_seconds": 30.0,
                "chapters": [
                    {
                        "sequence": 1,
                        "title": "L1",
                        "start_seconds": 0.0,
                        "duration_seconds": 3.0,
                        "level": 0,
                        "lesson_project_id": "proj-1",
                    },
                    # a level-1 sub-chapter must be ignored (its lesson's paragraphs cover it)
                    {
                        "sequence": 2,
                        "title": "slide",
                        "start_seconds": 1.5,
                        "duration_seconds": 1.5,
                        "level": 1,
                        "lesson_project_id": "proj-1",
                    },
                    {
                        "sequence": 3,
                        "title": "L2",
                        "start_seconds": 3.0,
                        "duration_seconds": 4.0,
                        "level": 0,
                        "lesson_project_id": "proj-2",
                    },
                ],
            },
        },
    )
    mock = _mock_client(app)

    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "My Course"},
    )

    assert resp.status_code == 200, resp.text
    payload = mock.start_upload.await_args.args[0]
    assert payload["caption_source_type"] == "url"
    assert payload["caption_language"] == "en"
    assert "/soft_cc/" in payload["caption_source"]
    assert payload["caption_source"].endswith("?X-Amz-Signature=test")


async def test_publish_no_captions_for_course(client, app, db_session_factory):
    """A course run (no child lesson tracks) publishes without a caption_source."""
    _, deliv = await _make_project(client)
    run_id = await _seed_completed_run(
        db_session_factory,
        deliv,
        "course_stitch",
        {"course_artifact_url": "http://minio.test:8389/w2v-courses/course-9.mp4"},
    )
    mock = _mock_client(app)

    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "My Course"},
    )

    assert resp.status_code == 200, resp.text
    payload = mock.start_upload.await_args.args[0]
    assert "caption_source" not in payload
    # Nothing minted → nothing echoed to the client either.
    assert "caption_source" not in resp.json()


async def test_publish_course_appends_chapters_to_description(client, app, db_session_factory):
    """A course's per-lesson chapters become a YouTube description chapter block."""
    _, deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "course_stitch": {
                "course_artifact_url": "http://minio.test:8389/w2v-courses/c.mp4",
                "artifact_url": "http://minio.test:8389/w2v-courses/c.mp4",
                "chapters": _CHAPTERS_3,
            }
        },
    )
    mock = _mock_client(app)

    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "Course", "description": "Welcome."},
    )

    assert resp.status_code == 200, resp.text
    desc = mock.start_upload.await_args.args[0]["description"]
    assert desc.startswith("Welcome.")
    assert "Chapters:" in desc
    assert "0:00 Intro" in desc
    assert "1:00 Core" in desc
    assert "1:40 Wrap" in desc


async def test_publish_omits_chapter_block_when_too_few(client, app, db_session_factory):
    """Fewer than 3 chapters → no chapter block (YouTube wouldn't render it)."""
    _, deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "course_stitch": {
                "course_artifact_url": "http://minio.test:8389/w2v-courses/c.mp4",
                "chapters": _CHAPTERS_3[:2],
            }
        },
    )
    mock = _mock_client(app)

    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "Course", "description": "Welcome."},
    )

    assert resp.status_code == 200, resp.text
    desc = mock.start_upload.await_args.args[0]["description"]
    assert desc == "Welcome."
    assert "Chapters:" not in desc


async def test_publish_sets_thumbnail_source_from_course_thumbnail(client, app, db_session_factory):
    """A course_thumbnail cover is presigned and passed as the upload thumbnail."""
    _, deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "course_stitch": {
                "course_artifact_url": "http://minio.test:8389/w2v-courses/c.mp4",
                "chapters": _CHAPTERS_3,
            },
            "course_thumbnail": {
                "thumbnail_url": "http://minio.test:8389/w2v-orchestrator/cover.png"
            },
        },
    )
    mock = _mock_client(app)

    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "Course"},
    )

    assert resp.status_code == 200, resp.text
    payload = mock.start_upload.await_args.args[0]
    assert payload["thumbnail_source"] == (
        "http://minio.test:8389/w2v-orchestrator/cover.png?X-Amz-Signature=test"
    )
    assert payload["thumbnail_source_type"] == "url"


async def test_publish_no_thumbnail_when_stage_absent(client, app, db_session_factory):
    """No course_thumbnail output → no thumbnail keys in the payload."""
    _, deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "course_stitch": {
                "course_artifact_url": "http://minio.test:8389/w2v-courses/c.mp4",
                "chapters": _CHAPTERS_3,
            }
        },
    )
    mock = _mock_client(app)

    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "Course"},
    )

    assert resp.status_code == 200, resp.text
    payload = mock.start_upload.await_args.args[0]
    assert "thumbnail_source" not in payload


async def test_publish_includes_sub_chapters_in_description(client, app, db_session_factory):
    """Mid-lesson sub-chapters (level 1) are flattened into the YouTube chapter
    block, prefixed so the hierarchy reads — lesson titles head each segment,
    slide sub-chapters mark the transitions within."""
    chapters_with_subs = [
        {**_CHAPTERS_3[0], "level": 0, "lesson_project_id": "p1"},  # Intro @0:00
        {
            "sequence": 99,
            "title": "Sub A",
            "start_seconds": 30.0,
            "duration_seconds": 30.0,
            "level": 1,
            "parent_sequence": 1,
            "lesson_project_id": "p1",
        },  # -> "0:30 — Sub A"
        {**_CHAPTERS_3[1], "level": 0, "lesson_project_id": "p2"},  # Core @1:00
        {**_CHAPTERS_3[2], "level": 0, "lesson_project_id": "p3"},  # Wrap @1:40
    ]
    _, deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "course_stitch": {
                "course_artifact_url": "http://minio.test:8389/w2v-courses/c.mp4",
                "chapters": chapters_with_subs,
            }
        },
    )
    mock = _mock_client(app)
    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "Course", "description": "Welcome."},
    )
    assert resp.status_code == 200, resp.text
    desc = mock.start_upload.await_args.args[0]["description"]
    assert desc == "Welcome.\n\nChapters:\n0:00 Intro\n0:30 — Sub A\n1:00 Core\n1:40 Wrap"


async def test_publish_applies_slide_subchapter_override(client, app, db_session_factory):
    """An operator edit to a level-1 slide title (keyed by its ``chapter_key``,
    ``{lesson_project_id}#{ordinal}``) flows into the published description — the
    edited slide title replaces the auto one, still ``— ``-prefixed."""
    chapters_with_subs = [
        {**_CHAPTERS_3[0], "level": 0, "lesson_project_id": "p1", "chapter_key": "p1"},
        {
            "sequence": 99,
            "title": "Sub A",
            "start_seconds": 30.0,
            "duration_seconds": 30.0,
            "level": 1,
            "parent_sequence": 1,
            "lesson_project_id": "p1",
            "chapter_key": "p1#1",
        },
        {**_CHAPTERS_3[1], "level": 0, "lesson_project_id": "p2", "chapter_key": "p2"},
        {**_CHAPTERS_3[2], "level": 0, "lesson_project_id": "p3", "chapter_key": "p3"},
    ]
    _, deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "course_stitch": {
                "course_artifact_url": "http://minio.test:8389/w2v-courses/c.mp4",
                "chapters": chapters_with_subs,
                "chapter_title_overrides": {"p1#1": "Edited Slide"},
            }
        },
    )
    mock = _mock_client(app)
    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "Course", "description": "Welcome."},
    )
    assert resp.status_code == 200, resp.text
    desc = mock.start_upload.await_args.args[0]["description"]
    assert "Sub A" not in desc  # auto title replaced by the edit
    assert desc == "Welcome.\n\nChapters:\n0:00 Intro\n0:30 — Edited Slide\n1:00 Core\n1:40 Wrap"


async def test_publish_omits_hidden_sub_chapter_from_description(client, app, db_session_factory):
    """An operator-hidden level-1 sub-chapter (``hidden_chapter_keys``) is dropped
    from the published description, while its siblings + every lesson marker stay."""
    chapters_with_subs = [
        {**_CHAPTERS_3[0], "level": 0, "lesson_project_id": "p1", "chapter_key": "p1"},
        {
            "sequence": 98,
            "title": "Sub A",
            "start_seconds": 20.0,
            "duration_seconds": 20.0,
            "level": 1,
            "parent_sequence": 1,
            "lesson_project_id": "p1",
            "chapter_key": "p1#1",
        },  # -> "0:20 — Sub A" (kept)
        {
            "sequence": 99,
            "title": "Sub B",
            "start_seconds": 40.0,
            "duration_seconds": 20.0,
            "level": 1,
            "parent_sequence": 1,
            "lesson_project_id": "p1",
            "chapter_key": "p1#2",
        },  # hidden -> dropped
        {
            **_CHAPTERS_3[1],
            "level": 0,
            "lesson_project_id": "p2",
            "chapter_key": "p2",
        },  # Core @1:00
        {
            **_CHAPTERS_3[2],
            "level": 0,
            "lesson_project_id": "p3",
            "chapter_key": "p3",
        },  # Wrap @1:40
    ]
    _, deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "course_stitch": {
                "course_artifact_url": "http://minio.test:8389/w2v-courses/c.mp4",
                "chapters": chapters_with_subs,
                "hidden_chapter_keys": ["p1#2"],
            }
        },
    )
    mock = _mock_client(app)
    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "Course", "description": "Welcome."},
    )
    assert resp.status_code == 200, resp.text
    desc = mock.start_upload.await_args.args[0]["description"]
    assert "Sub B" not in desc  # hidden sub dropped
    assert desc == "Welcome.\n\nChapters:\n0:00 Intro\n0:20 — Sub A\n1:00 Core\n1:40 Wrap"


async def test_publish_collapses_lesson_marker_and_first_sub_chapter(
    client, app, db_session_factory
):
    """A lesson's first sub-chapter shares the lesson's timestamp (its
    lesson-relative offset is pinned to 0.0). The collision collapses to one
    chapter carrying the lesson title — not the slide title."""
    chapters = [
        {**_CHAPTERS_3[0], "level": 0, "lesson_project_id": "p1"},  # Intro @0:00
        {
            "sequence": 98,
            "title": "First Slide",
            "start_seconds": 0.0,  # collides with the Intro marker
            "duration_seconds": 30.0,
            "level": 1,
            "parent_sequence": 1,
            "lesson_project_id": "p1",
        },
        {
            "sequence": 99,
            "title": "Second Slide",
            "start_seconds": 30.0,
            "duration_seconds": 30.0,
            "level": 1,
            "parent_sequence": 1,
            "lesson_project_id": "p1",
        },
        {**_CHAPTERS_3[1], "level": 0, "lesson_project_id": "p2"},  # Core @1:00
        {**_CHAPTERS_3[2], "level": 0, "lesson_project_id": "p3"},  # Wrap @1:40
    ]
    _, deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "course_stitch": {
                "course_artifact_url": "http://minio.test:8389/w2v-courses/c.mp4",
                "chapters": chapters,
            }
        },
    )
    mock = _mock_client(app)
    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "Course", "description": "Welcome."},
    )
    assert resp.status_code == 200, resp.text
    desc = mock.start_upload.await_args.args[0]["description"]
    assert "First Slide" not in desc  # collapsed away — lesson title wins the slot
    assert desc.count("0:00") == 1
    assert desc == "Welcome.\n\nChapters:\n0:00 Intro\n0:30 — Second Slide\n1:00 Core\n1:40 Wrap"


async def test_publish_degrades_to_level0_when_sub_chapter_too_close(
    client, app, db_session_factory
):
    """A sub-chapter <10s after its lesson marker is the offending row and gets
    dropped (coalescing); with no other sub-chapters present, the result is the
    level-0-only list — never a partial, non-rendering chapter list."""
    chapters = [
        {**_CHAPTERS_3[0], "level": 0, "lesson_project_id": "p1"},  # Intro @0:00
        {
            "sequence": 99,
            "title": "Too Soon",
            "start_seconds": 5.0,  # 5s after the lesson marker — gap < 10s
            "duration_seconds": 55.0,
            "level": 1,
            "parent_sequence": 1,
            "lesson_project_id": "p1",
        },
        {**_CHAPTERS_3[1], "level": 0, "lesson_project_id": "p2"},
        {**_CHAPTERS_3[2], "level": 0, "lesson_project_id": "p3"},
    ]
    _, deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "course_stitch": {
                "course_artifact_url": "http://minio.test:8389/w2v-courses/c.mp4",
                "chapters": chapters,
            }
        },
    )
    mock = _mock_client(app)
    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "Course", "description": "Welcome."},
    )
    assert resp.status_code == 200, resp.text
    desc = mock.start_upload.await_args.args[0]["description"]
    assert "Too Soon" not in desc
    assert desc == "Welcome.\n\nChapters:\n0:00 Intro\n1:00 Core\n1:40 Wrap"


async def test_publish_coalesces_drops_only_offending_sub(client, app, db_session_factory):
    """Coalescing drops ONLY the too-close sub-chapter and keeps the qualifying
    ones — a single tightly-spaced slide no longer discards every sub-chapter (the
    old whole-list degrade)."""
    chapters = [
        {**_CHAPTERS_3[0], "level": 0, "lesson_project_id": "p1"},  # Intro @0:00
        {
            "sequence": 98,
            "title": "Too Soon",
            "start_seconds": 5.0,  # 5s after the lesson marker — offending, dropped
            "duration_seconds": 25.0,
            "level": 1,
            "parent_sequence": 1,
            "lesson_project_id": "p1",
        },
        {
            "sequence": 99,
            "title": "Good Sub",
            "start_seconds": 30.0,  # ≥10s after Intro AND ≥10s before Core — kept
            "duration_seconds": 30.0,
            "level": 1,
            "parent_sequence": 1,
            "lesson_project_id": "p1",
        },
        {**_CHAPTERS_3[1], "level": 0, "lesson_project_id": "p2"},  # Core @1:00
        {**_CHAPTERS_3[2], "level": 0, "lesson_project_id": "p3"},  # Wrap @1:40
    ]
    _, deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "course_stitch": {
                "course_artifact_url": "http://minio.test:8389/w2v-courses/c.mp4",
                "chapters": chapters,
                "duration_seconds": 120.0,
            }
        },
    )
    mock = _mock_client(app)
    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "Course", "description": "Welcome."},
    )
    assert resp.status_code == 200, resp.text
    desc = mock.start_upload.await_args.args[0]["description"]
    assert "Too Soon" not in desc  # only the offending sub is dropped
    assert "Good Sub" in desc  # the qualifying sub survives (old code lost it)
    assert desc == "Welcome.\n\nChapters:\n0:00 Intro\n0:30 — Good Sub\n1:00 Core\n1:40 Wrap"


async def test_publish_coalesce_drops_sub_crowding_next_lesson(client, app, db_session_factory):
    """A sub ≥10s after its own lesson marker but <10s BEFORE the next lesson is
    dropped — the mandatory lesson marker must keep its ≥10s lead-in (forward
    lookahead), so the lesson backbone is never pushed under the threshold."""
    chapters = [
        {**_CHAPTERS_3[0], "level": 0, "lesson_project_id": "p1"},  # Intro @0:00
        {
            "sequence": 99,
            "title": "Crowds Core",
            "start_seconds": 52.0,  # 52s after Intro (ok) but only 8s before Core — dropped
            "duration_seconds": 8.0,
            "level": 1,
            "parent_sequence": 1,
            "lesson_project_id": "p1",
        },
        {**_CHAPTERS_3[1], "level": 0, "lesson_project_id": "p2"},  # Core @1:00
        {**_CHAPTERS_3[2], "level": 0, "lesson_project_id": "p3"},  # Wrap @1:40
    ]
    _, deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "course_stitch": {
                "course_artifact_url": "http://minio.test:8389/w2v-courses/c.mp4",
                "chapters": chapters,
                "duration_seconds": 120.0,
            }
        },
    )
    mock = _mock_client(app)
    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "Course", "description": "Welcome."},
    )
    assert resp.status_code == 200, resp.text
    desc = mock.start_upload.await_args.args[0]["description"]
    assert "Crowds Core" not in desc
    assert desc == "Welcome.\n\nChapters:\n0:00 Intro\n1:00 Core\n1:40 Wrap"


async def test_publish_trims_trailing_sub_too_close_to_end(client, app, db_session_factory):
    """A trailing sub-chapter whose run-to-video-end duration is <10s is trimmed,
    and the prior lesson marker becomes the honest last chapter — the last-chapter
    ≥10s rule judges real playback length, not a short trailing slide."""
    chapters = [
        {**_CHAPTERS_3[0], "level": 0, "lesson_project_id": "p1"},  # Intro @0:00
        {**_CHAPTERS_3[1], "level": 0, "lesson_project_id": "p2"},  # Core @1:00
        {**_CHAPTERS_3[2], "level": 0, "lesson_project_id": "p3"},  # Wrap @1:40
        {
            "sequence": 99,
            "title": "Last Slide",
            "start_seconds": 115.0,  # only 5s before video end (120) — trimmed
            "duration_seconds": 5.0,
            "level": 1,
            "parent_sequence": 3,
            "lesson_project_id": "p3",
        },
    ]
    _, deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "course_stitch": {
                "course_artifact_url": "http://minio.test:8389/w2v-courses/c.mp4",
                "chapters": chapters,
                "duration_seconds": 120.0,
            }
        },
    )
    mock = _mock_client(app)
    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "Course", "description": "Welcome."},
    )
    assert resp.status_code == 200, resp.text
    desc = mock.start_upload.await_args.args[0]["description"]
    assert "Last Slide" not in desc
    assert desc == "Welcome.\n\nChapters:\n0:00 Intro\n1:00 Core\n1:40 Wrap"


async def test_publish_override_not_applied_to_sub_chapters(client, app, db_session_factory):
    """A level-0 title override (by lesson_project_id) renames the lesson marker
    but must NOT leak onto its sub-chapters — they carry the parent's
    lesson_project_id, but slide titles stay auto."""
    chapters = [
        {**_CHAPTERS_3[0], "level": 0, "lesson_project_id": "p1"},  # Intro @0:00
        {
            "sequence": 99,
            "title": "Slide Detail",
            "start_seconds": 30.0,
            "duration_seconds": 30.0,
            "level": 1,
            "parent_sequence": 1,
            "lesson_project_id": "p1",  # same parent key as the overridden lesson
        },
        {**_CHAPTERS_3[1], "level": 0, "lesson_project_id": "p2"},
        {**_CHAPTERS_3[2], "level": 0, "lesson_project_id": "p3"},
    ]
    _, deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "course_stitch": {
                "course_artifact_url": "http://minio.test:8389/w2v-courses/c.mp4",
                "chapters": chapters,
                "chapter_title_overrides": {"p1": "Renamed Lesson"},
            }
        },
    )
    mock = _mock_client(app)
    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "Course", "description": "Hi."},
    )
    assert resp.status_code == 200, resp.text
    desc = mock.start_upload.await_args.args[0]["description"]
    assert "0:00 Renamed Lesson" in desc
    assert "0:30 — Slide Detail" in desc
    assert desc.count("Renamed Lesson") == 1  # not stamped onto the sub-chapter


async def test_publish_clamps_chaptered_description_to_cap(client, app, db_session_factory):
    """A long base body + a chapter block stays within YouTube's 5000-char cap,
    truncating the base while keeping the chapter block intact."""
    chapters = [
        {**_CHAPTERS_3[0], "level": 0, "lesson_project_id": "p1"},
        {
            "sequence": 99,
            "title": "Sub A",
            "start_seconds": 30.0,
            "duration_seconds": 30.0,
            "level": 1,
            "parent_sequence": 1,
            "lesson_project_id": "p1",
        },
        {**_CHAPTERS_3[1], "level": 0, "lesson_project_id": "p2"},
        {**_CHAPTERS_3[2], "level": 0, "lesson_project_id": "p3"},
    ]
    _, deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "course_stitch": {
                "course_artifact_url": "http://minio.test:8389/w2v-courses/c.mp4",
                "chapters": chapters,
            }
        },
    )
    mock = _mock_client(app)
    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "Course", "description": "y" * 5000},
    )
    assert resp.status_code == 200, resp.text
    desc = mock.start_upload.await_args.args[0]["description"]
    assert len(desc) <= 5000
    assert desc.endswith("Chapters:\n0:00 Intro\n0:30 — Sub A\n1:00 Core\n1:40 Wrap")


async def test_publish_applies_chapter_title_overrides(client, app, db_session_factory):
    """An operator title override (by lesson_project_id) is reflected in the
    YouTube description."""
    chapters = [
        {**c, "level": 0, "lesson_project_id": pid}
        for c, pid in zip(_CHAPTERS_3, ["p1", "p2", "p3"], strict=True)
    ]
    _, deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "course_stitch": {
                "course_artifact_url": "http://minio.test:8389/w2v-courses/c.mp4",
                "chapters": chapters,
                "chapter_title_overrides": {"p2": "Deep Dive"},
            }
        },
    )
    mock = _mock_client(app)
    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "Course", "description": "Hi."},
    )
    assert resp.status_code == 200, resp.text
    desc = mock.start_upload.await_args.args[0]["description"]
    assert "1:00 Deep Dive" in desc
    assert "Core" not in desc


async def test_publish_thumbnail_override_wins(client, app, db_session_factory):
    """thumbnail_override_url beats the auto course_thumbnail cover."""
    _, deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "course_stitch": {
                "course_artifact_url": "http://minio.test:8389/w2v-courses/c.mp4",
                "chapters": _CHAPTERS_3,
                "thumbnail_override_url": "http://minio.test:8389/w2v-orchestrator/picked.png",
            },
            "course_thumbnail": {
                "thumbnail_url": "http://minio.test:8389/w2v-orchestrator/auto.png"
            },
        },
    )
    mock = _mock_client(app)
    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "Course"},
    )
    assert resp.status_code == 200, resp.text
    payload = mock.start_upload.await_args.args[0]
    assert payload["thumbnail_source"] == (
        "http://minio.test:8389/w2v-orchestrator/picked.png?X-Amz-Signature=test"
    )


async def test_publish_rejects_incomplete_run(client, app, db_session_factory):
    """A run that is not completed → 409, no downstream call."""
    _, deliv = await _make_project(client)
    run_id = await _seed_completed_run(
        db_session_factory,
        deliv,
        "lesson_compile",
        {"artifact_url": "http://minio.test:8389/b/k.mp4"},
        completed=False,
    )
    mock = _mock_client(app)

    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "x"},
    )

    assert resp.status_code == 409, resp.text
    mock.start_upload.assert_not_awaited()


async def test_publish_rejects_run_without_video(client, app, db_session_factory):
    """A completed run whose only artifact is audio → 422, no downstream call."""
    _, deliv = await _make_project(client)
    run_id = await _seed_completed_run(
        db_session_factory,
        deliv,
        "lesson_narrate",
        {"artifact_url": "http://minio.test:8389/w2v-audio/clip.wav"},
    )
    mock = _mock_client(app)

    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "x"},
    )

    assert resp.status_code == 422, resp.text
    mock.start_upload.assert_not_awaited()


async def test_publish_unknown_run_404(client, app):
    _mock_client(app)
    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": str(uuid4()), "title": "x"},
    )
    assert resp.status_code == 404, resp.text


async def test_publish_cross_tenant_run_404(client, app, db_session_factory, other_tenant_client):
    """A run owned by another tenant must not be publishable (tenant isolation)."""
    other_proj = await other_tenant_client.post(
        "/api/projects", json={**_PROJECT_PAYLOAD, "title": "theirs"}
    )
    assert other_proj.status_code == 201, other_proj.text
    deliv = other_proj.json()["deliverables"][0]["id"]
    run_id = await _seed_completed_run(
        db_session_factory,
        deliv,
        "lesson_compile",
        {"artifact_url": "http://minio.test:8389/b/k.mp4"},
    )
    mock = _mock_client(app)

    resp = await client.post(
        "/api/youtube/upload",
        json={"pipeline_job_id": run_id, "title": "x"},
    )
    assert resp.status_code == 404, resp.text
    mock.start_upload.assert_not_awaited()


async def test_jobs_proxy_passthrough(client, app):
    """Both job-read proxies forward the CALLER tenant explicitly as the owner scope.

    This used to assert ``get_upload_job("j1")`` / ``list_upload_jobs(limit=..,
    offset=..)`` — i.e. the owner scope rode the client's ambient ContextVar and
    the declared ``TenantIdDep`` was decorative. That ambient path no-ops when
    nothing is bound, which is how 35 of 48 prod rows got an ``owner_tenant_id``
    no caller can ever match. The tenant MUST be an explicit argument.
    """
    mock = _mock_client(app)
    mock.get_upload_job.return_value = {"job_id": "j1", "status": "completed"}
    mock.list_upload_jobs.return_value = {"jobs": [], "total": 0}

    one = await client.get("/api/youtube/upload/jobs/j1")
    assert one.status_code == 200, one.text
    assert one.json()["status"] == "completed"
    mock.get_upload_job.assert_awaited_once_with("j1", owner_tenant_id=client.tenant_id)

    many = await client.get("/api/youtube/upload/jobs?limit=10&offset=5")
    assert many.status_code == 200, many.text
    mock.list_upload_jobs.assert_awaited_once_with(
        limit=10, offset=5, owner_tenant_id=client.tenant_id
    )


async def test_jobs_proxy_translates_upstream_404(client, app):
    """A pruned/expired upstream job must come back as 404, not a 500.

    Regression: the client's raise_for_status bubbled an httpx.HTTPStatusError
    for a downstream 404 straight into a 500 (observed live, a poller hitting a
    gone job spammed ~4 500s/min).
    """
    mock = _mock_client(app)
    req = httpx.Request("GET", "http://youtube/api/upload/jobs/gone")
    resp = httpx.Response(404, request=req, json={"error": {"message": "not found"}})
    mock.get_upload_job.side_effect = httpx.HTTPStatusError("404", request=req, response=resp)

    r = await client.get("/api/youtube/upload/jobs/gone")
    assert r.status_code == 404, r.text


async def test_jobs_proxy_404_logs_at_warning_with_the_job_id(client, app, caplog):
    """The 404 branch must LOG, at a level prod actually emits.

    Prod runs ``LOG_LEVEL=WARNING``. This branch was silent at every level, so
    484 unreadable-row 404s went undiagnosed. An ``info`` here would be just as
    invisible as silence — hence the explicit ``levelno >= WARNING`` assertion,
    not merely "a record exists".
    """
    mock = _mock_client(app)
    req = httpx.Request("GET", "http://youtube/api/upload/jobs/j-silent")
    resp = httpx.Response(404, request=req, json={"error": {"message": "not found"}})
    mock.get_upload_job.side_effect = httpx.HTTPStatusError("404", request=req, response=resp)

    with caplog.at_level(logging.WARNING, logger="src.api.routes.youtube"):
        r = await client.get("/api/youtube/upload/jobs/j-silent")
    assert r.status_code == 404, r.text

    records = [rec for rec in caplog.records if rec.name == "src.api.routes.youtube"]
    assert records, "the 404 branch must emit a log record, not fail silently"
    assert len(records) == 1, f"expected exactly one record, got {records}"
    record = records[0]
    assert record.levelno >= logging.WARNING, (
        f"prod runs LOG_LEVEL=WARNING; a {record.levelname} record is invisible there"
    )
    message = record.getMessage()
    assert "j-silent" in message, f"the log must name the job_id, got: {message!r}"
    assert client.tenant_id in message, f"the log must name the tenant, got: {message!r}"


async def test_jobs_proxy_transient_upstream_is_503_with_retry_after(client, app):
    """A RETRYABLE upstream fault answers 503 + Retry-After, never a bare 500.

    This test used to be ``test_jobs_proxy_reraises_non_404`` and PINNED the defect:
    it asserted the raw ``httpx.HTTPStatusError`` escaped the handler, which in prod
    means an opaque 500 for every status except 404 — the exact "reads as an outage"
    failure the 404 branch was written to stop. The route handled ONE code and let
    everything else fall through undifferentiated.

    Now the ONE upstream policy (``stages/upstream.py::classify``) makes the call. A
    502 from a rolling upload-service is transient, so the poller is told to come
    back rather than being handed a server fault it cannot act on.
    """
    mock = _mock_client(app)
    req = httpx.Request("GET", "http://youtube/api/upload/jobs/boom")
    resp = httpx.Response(502, request=req, text="bad gateway")
    mock.get_upload_job.side_effect = httpx.HTTPStatusError("502", request=req, response=resp)

    r = await client.get("/api/youtube/upload/jobs/boom")
    assert r.status_code == 503, r.text
    assert r.headers.get("Retry-After") == "5"
    # The real upstream status + sentence survive into the detail, so the failure
    # stays diagnosable instead of collapsing to "Internal Server Error".
    assert "502" in r.json()["detail"]


async def test_jobs_proxy_rate_limit_is_transient_not_permanent(client, app):
    """A 429 quota block is a SCHEDULING signal — 503, not a permanent 502."""
    mock = _mock_client(app)
    req = httpx.Request("GET", "http://youtube/api/upload/jobs/slow")
    resp = httpx.Response(429, request=req, json={"error": {"message": "quota exceeded"}})
    mock.get_upload_job.side_effect = httpx.HTTPStatusError("429", request=req, response=resp)

    r = await client.get("/api/youtube/upload/jobs/slow")
    assert r.status_code == 503, r.text
    assert "quota exceeded" in r.json()["detail"]


async def test_jobs_proxy_permanent_upstream_is_502_not_the_upstream_401(client, app):
    """A revoked OAuth grant answers 502 — NOT a mirrored 401/403.

    Mirroring the upstream auth status back would read to the SPA as the CALLER's
    own session being invalid and log the studio user out over someone else's broken
    YouTube connection. ``classify`` puts 401/403 in ``PERMANENT_STATUSES``; the route
    renders that as a gateway fault and keeps the real sentence in the detail.
    """
    mock = _mock_client(app)
    req = httpx.Request("GET", "http://youtube/api/upload/jobs/revoked")
    resp = httpx.Response(
        403, request=req, json={"error": {"message": "youtube account not connected"}}
    )
    mock.get_upload_job.side_effect = httpx.HTTPStatusError("403", request=req, response=resp)

    r = await client.get("/api/youtube/upload/jobs/revoked")
    assert r.status_code == 502, r.text
    assert r.status_code not in (401, 403)
    assert "youtube account not connected" in r.json()["detail"]


async def test_start_upload_upstream_failure_is_not_a_bare_500(client, app, db_session_factory):
    """The POST publish path had NO upstream handler at all — every status was a 500."""
    _, deliv = await _make_project(client)
    run_id = await _seed_completed_run(
        db_session_factory,
        deliv,
        "lesson_compile",
        {"artifact_url": "http://minio.test:8389/w2v-compiled-videos/lesson-1.mp4"},
    )
    mock = _mock_client(app)
    req = httpx.Request("POST", "http://youtube/api/upload")
    resp = httpx.Response(402, request=req, json={"error": {"message": "youtube plan required"}})
    mock.start_upload.side_effect = httpx.HTTPStatusError("402", request=req, response=resp)

    r = await client.post("/api/youtube/upload", json={"pipeline_job_id": run_id, "title": "T"})
    assert r.status_code == 502, r.text
    assert "youtube plan required" in r.json()["detail"]


async def test_retract_proxy_passes_owner_tenant_explicitly(client, app):
    """Retraction is scoped to the caller tenant as an EXPLICIT argument."""
    mock = _mock_client(app)
    mock.retract_upload_job.return_value = {"job_id": "j1", "privacy_status": "private"}

    r = await client.post("/api/youtube/upload/jobs/j1/retract")
    assert r.status_code == 200, r.text
    assert r.json()["privacy_status"] == "private"
    mock.retract_upload_job.assert_awaited_once_with("j1", owner_tenant_id=client.tenant_id)


@pytest.mark.parametrize("upstream", [404])
async def test_retract_proxy_passes_domain_answers_through(client, app, upstream):
    mock = _mock_client(app)
    req = httpx.Request("POST", "http://youtube/api/upload/jobs/j1/retract")
    resp = httpx.Response(upstream, request=req, json={"detail": "x"})
    mock.retract_upload_job.side_effect = httpx.HTTPStatusError(
        str(upstream), request=req, response=resp
    )
    r = await client.post("/api/youtube/upload/jobs/j1/retract")
    assert r.status_code == upstream, r.text


async def test_retract_proxy_youtube_failure_is_not_a_bare_500(client, app):
    mock = _mock_client(app)
    req = httpx.Request("POST", "http://youtube/api/upload/jobs/j1/retract")
    resp = httpx.Response(502, request=req, json={"detail": {"error": "youtube_retract_failed"}})
    mock.retract_upload_job.side_effect = httpx.HTTPStatusError("502", request=req, response=resp)
    r = await client.post("/api/youtube/upload/jobs/j1/retract")
    assert r.status_code in (502, 503), r.text


async def test_set_privacy_proxy_passes_target_and_owner_tenant_explicitly(client, app):
    """The governed release forwards the target and scopes to the caller tenant explicitly."""
    mock = _mock_client(app)
    mock.set_upload_job_privacy.return_value = {"job_id": "j1", "privacy_status": "unlisted"}

    r = await client.post(
        "/api/youtube/upload/jobs/j1/privacy", json={"privacy_status": "unlisted"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["privacy_status"] == "unlisted"
    mock.set_upload_job_privacy.assert_awaited_once_with(
        "j1", "unlisted", owner_tenant_id=client.tenant_id
    )


async def test_set_privacy_proxy_refuses_private(client, app):
    """Lowering visibility is /retract's job: a private target never reaches upstream."""
    mock = _mock_client(app)
    r = await client.post("/api/youtube/upload/jobs/j1/privacy", json={"privacy_status": "private"})
    assert r.status_code == 422, r.text
    mock.set_upload_job_privacy.assert_not_called()


@pytest.mark.parametrize("upstream", [404])
async def test_set_privacy_proxy_passes_domain_answers_through(client, app, upstream):
    mock = _mock_client(app)
    req = httpx.Request("POST", "http://youtube/api/upload/jobs/j1/privacy")
    resp = httpx.Response(upstream, request=req, json={"detail": "x"})
    mock.set_upload_job_privacy.side_effect = httpx.HTTPStatusError(
        str(upstream), request=req, response=resp
    )
    r = await client.post("/api/youtube/upload/jobs/j1/privacy", json={"privacy_status": "public"})
    assert r.status_code == upstream, r.text


async def test_set_privacy_proxy_youtube_failure_is_not_a_bare_500(client, app):
    mock = _mock_client(app)
    req = httpx.Request("POST", "http://youtube/api/upload/jobs/j1/privacy")
    resp = httpx.Response(502, request=req, json={"detail": {"error": "youtube_privacy_failed"}})
    mock.set_upload_job_privacy.side_effect = httpx.HTTPStatusError(
        "502", request=req, response=resp
    )
    r = await client.post("/api/youtube/upload/jobs/j1/privacy", json={"privacy_status": "public"})
    assert r.status_code in (502, 503), r.text
