"""Run-scoped soft-CC orchestration — the DB-coupled layer over ``captions``.

:mod:`src.services.captions` holds the *pure* caption builders (no I/O). This
module adds the parts that need the database: resolving a run's completed stage
outputs, and — for a stitched *course* — fetching each child lesson's
``lesson_script`` + ``lesson_narrate`` to assemble a course-wide track.

Both the YouTube publish path (``src/api/routes/youtube.py``) and the on-demand
caption-download route (``src/api/routes/runs.py``) consume these. The course
logic lives here rather than in either route because ``runs.py`` is the base
router (``youtube.py`` imports ``runs._tenant_owns_run``), so ``runs.py`` cannot
import from ``youtube.py`` without a cycle.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.database.repositories.stage_output_repository import StageOutputRepository
from src.services.captions import (
    CaptionCue,
    build_course_caption_cues,
    build_srt,
    caption_cues_for_run,
)


def completed_outputs(stage_outputs: list[Any]) -> dict[str, dict]:
    """Map ``stage_name -> output`` for completed stages with a non-empty output."""
    return {
        so.stage_name: so.output for so in stage_outputs if so.status == "completed" and so.output
    }


async def course_caption_cues(
    completed: dict[str, dict], session: AsyncSession
) -> list[CaptionCue]:
    """Build a course's soft-CC cues by concatenating its child lessons' tracks.

    A course run has no script+narrate pair of its own — its caption text+timing
    lives in the child lesson runs. ``course_compile_lessons`` maps each lesson
    project to its child ``lesson_run_id``; ``course_stitch.chapters`` carries the
    level-0 (per-lesson) markers in playback order, each with the lesson's stable
    ``lesson_project_id`` and its absolute ``start_seconds`` on the course
    timeline. For each lesson in order we fetch the child's ``lesson_script`` +
    ``lesson_narrate`` and shift its cues by the chapter offset. Returns ``[]`` for
    any non-course run (no recognised pair) — the caller falls back to no captions.
    """
    stitch = completed.get("course_stitch")
    compile_lessons = completed.get("course_compile_lessons")
    if not isinstance(stitch, dict) or not isinstance(compile_lessons, dict):
        return []
    run_by_pid: dict[str, str] = {}
    for lesson in compile_lessons.get("lessons") or []:
        pid = lesson.get("lesson_project_id")
        rid = lesson.get("lesson_run_id")
        if pid and rid:
            run_by_pid[str(pid)] = str(rid)
    if not run_by_pid:
        return []
    repo = StageOutputRepository(session)
    tracks: list[tuple[float, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for chapter in stitch.get("chapters") or []:
        if int(chapter.get("level") or 0) != 0:
            continue  # mid-lesson sub-chapters are covered by their lesson's paragraphs
        run_id = run_by_pid.get(str(chapter.get("lesson_project_id") or ""))
        if not run_id:
            continue
        script_so = await repo.get(UUID(run_id), "lesson_script")
        narrate_so = await repo.get(UUID(run_id), "lesson_narrate")
        if script_so is None or script_so.status != "completed" or not script_so.output:
            continue
        if narrate_so is None or narrate_so.status != "completed" or not narrate_so.output:
            continue
        paragraphs = script_so.output.get("paragraphs") or []
        clips = narrate_so.output.get("clips") or []
        tracks.append((float(chapter.get("start_seconds") or 0.0), paragraphs, clips))
    return build_course_caption_cues(tracks)


async def caption_cues_for_completed(
    completed: dict[str, dict], session: AsyncSession
) -> list[CaptionCue]:
    """Caption cues for a run's completed outputs.

    Tries the direct script+narrate pair (promo / lesson / website_review /
    generic) via the pure :func:`caption_cues_for_run`; falls back to the
    DB-coupled :func:`course_caption_cues` for a stitched course. Empty list when
    neither applies (audiobook / unrecognised).
    """
    cues = caption_cues_for_run(completed)
    if cues:
        return cues
    return await course_caption_cues(completed, session)


async def srt_for_run(session: AsyncSession, run_id: UUID) -> str | None:
    """Mint an SRT caption document for a run from its stage outputs (``None`` when N/A).

    Loads the run's completed stage outputs and builds an accurate caption track
    from narration timing. Returns ``None`` when the run has no captionable
    narration (audiobook / unrecognised) or yields a blank document — callers
    return 404 / publish without captions.
    """
    stage_outputs = await StageOutputRepository(session).list_for_run(run_id)
    completed = completed_outputs(stage_outputs)
    cues = await caption_cues_for_completed(completed, session)
    if not cues:
        return None
    srt = build_srt(cues)
    return srt if srt.strip() else None
