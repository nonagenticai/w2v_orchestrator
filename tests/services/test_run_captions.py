"""DB-coupled soft-CC orchestration — ``src.services.run_captions``.

The pure cue/SRT builders are covered in ``tests/services/test_captions.py``.
Here we exercise the parts that touch the database: ``srt_for_run`` resolving a
run's completed stage outputs into an SRT, including the stitched-course path
that fetches child lessons' ``lesson_script`` + ``lesson_narrate``.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from src.services.run_captions import srt_for_run

pytestmark = pytest.mark.asyncio


async def _seed_run_with_stages(
    db_session_factory,
    deliverable_id: str,
    stages: dict[str, dict],
) -> str:
    """Create a completed run with several completed stage outputs."""
    from src.database.repositories.run_repository import RunRepository
    from src.database.repositories.stage_output_repository import StageOutputRepository

    async with db_session_factory() as session:
        run_repo = RunRepository(session)
        run = await run_repo.create(
            deliverable_id=UUID(deliverable_id),
            blueprint_id="course_v1",
            config_snapshot={},
            pause_after_stages=[],
        )
        so_repo = StageOutputRepository(session)
        for stage, output in stages.items():
            await so_repo.upsert_pending(run.id, stage)
            await so_repo.mark_completed(run.id, stage, output)
        await run_repo.mark_completed(run.id)
        await session.commit()
        return str(run.id)


async def _make_project(client) -> str:
    r = await client.post(
        "/api/projects",
        json={
            "type": "generic",
            "title": "for captions",
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
    detail = await client.get(f"/api/projects/{r.json()['id']}")
    return detail.json()["deliverables"][0]["id"]


async def _srt(db_session_factory, run_id: str) -> str | None:
    """Open a fresh session and mint the run's SRT (committed seed data is visible)."""
    async with db_session_factory() as session:
        return await srt_for_run(session, UUID(run_id))


async def test_srt_for_run_promo(client, db_session_factory) -> None:
    """A promo run's script+narrate pair mints an SRT with the spoken text + timing."""
    deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {
            "promo_script": {
                "lines": [
                    {"id": "l1", "text": "Hello world"},
                    {"id": "l2", "text": "Second line"},
                ]
            },
            "promo_narrate": {
                "clips": [
                    {"line_id": "l1", "seconds": 2.0},
                    {"line_id": "l2", "seconds": 3.0},
                ]
            },
        },
    )

    srt = await _srt(db_session_factory, run_id)

    assert srt is not None
    assert "-->" in srt
    assert "Hello world" in srt
    assert "Second line" in srt
    # First cue [0, 2); second cue accumulates to [2, 5). SRT form HH:MM:SS,mmm.
    assert "00:00:00,000 --> 00:00:02,000" in srt
    assert "00:00:02,000 --> 00:00:05,000" in srt


async def test_srt_for_run_course(client, db_session_factory) -> None:
    """A stitched course mints an SRT by concatenating child lessons, offset by chapter start."""
    deliv = await _make_project(client)
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
                "course_artifact_url": "http://minio.test:8389/w2v-courses/c.mp4",
                "chapters": [
                    {"start_seconds": 0.0, "level": 0, "lesson_project_id": "proj-1"},
                    {"start_seconds": 3.0, "level": 0, "lesson_project_id": "proj-2"},
                ],
            },
        },
    )

    srt = await _srt(db_session_factory, run_id)

    assert srt is not None
    assert "Lesson one para" in srt
    assert "Lesson two para" in srt
    # Lesson one: [0, 3); lesson two offset by its chapter start (3s): [3, 7).
    assert "00:00:00,000 --> 00:00:03,000" in srt
    assert "00:00:03,000 --> 00:00:07,000" in srt


async def test_srt_for_run_none_for_audiobook(client, db_session_factory) -> None:
    """A run with no captionable narration (audiobook) yields no SRT."""
    deliv = await _make_project(client)
    run_id = await _seed_run_with_stages(
        db_session_factory,
        deliv,
        {"audiobook_stitch": {"artifact_url": "http://minio.test:8389/w2v-audiobooks/a.mp3"}},
    )

    assert await _srt(db_session_factory, run_id) is None
