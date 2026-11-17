"""Shared helpers for the in-process full-DAG dispatch integration tests.

The archetype coverage tests (`test_in_process_dispatch.py` for generic/promo,
`test_in_process_lesson.py`, `test_in_process_website_review.py`,
`test_in_process_app_review.py`) all drive a real run through the in-process
dispatcher to a terminal status with every downstream client mocked at its
stage-module import site. This module holds the machinery they share:

- `_create_project` / `_dispatch_and_wait` — build a project + run and drive it.
- `_wm_client` / `_slide_service_mock` / `_tts_mock` / `_compile_mock` — the
  reusable downstream-client fakes (each an `AsyncMock` whose methods return the
  minimal payload the consuming stage reads).
- `_speech_rate_tts_mock` — the fake behind the autouse `_stub_speech_rate_tts`
  fixture in `conftest.py`; see that fixture for WHY it is autouse rather than
  per-test.
- `_completed_stage_names` — read the set of stages a run actually completed,
  for the anti-false-green DAG-spine assertion each archetype test makes.

Not a test module (no `test_` prefix ⇒ pytest does not collect it).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from unittest.mock import AsyncMock
from uuid import UUID

from w2v_common.clients import WMResponse

from src.clients.schemas.tts import TTSSynthesis

# ---------------------------------------------------------------------------
# Project + run drivers
# ---------------------------------------------------------------------------


async def _create_project(
    db_session_factory: Callable,
    *,
    tenant: UUID,
    subject: UUID,
    type_: str,
    brief: dict,
    voice: dict | None = None,
    brand: dict | None = None,
    theme: dict | None = None,
    deliverable: dict | None = None,
    parent_project_id: UUID | None = None,
) -> tuple[UUID, UUID]:
    """Create a project + single deliverable, returning (project_id, deliverable_id).

    `parent_project_id` is required for archetypes that must nest under a parent
    (e.g. `lesson` must point at a `course` — see ProjectRepository).
    """
    from src.database.repositories.project_repository import ProjectRepository

    async with db_session_factory() as session:
        proj = await ProjectRepository(session).create_with_deliverables(
            tenant_id=tenant,
            owner_subject_id=subject,
            type_=type_,
            title="x",
            brief=brief,
            brand=brand or {},
            voice=voice or {"descriptor": "crisp"},
            theme=theme or {},
            visibility="tenant",
            parent_project_id=parent_project_id,
            deliverables=[
                deliverable
                or {
                    "name": "d",
                    "format": "16x9",
                    "captions_mode": "none",
                    "target_duration_seconds": 30,
                }
            ],
        )
        await session.commit()
        return proj.id, proj.deliverables[0].id


async def _dispatch_and_wait(
    db_session_factory: Callable,
    redis_client,
    *,
    deliverable_id: UUID,
    tenant: UUID,
    subject: UUID,
    pause_after_stages: list[str] | None = None,
    max_polls: int = 60,
):
    """Start a run via RunEngine, dispatch it in-process, poll to terminal.

    Returns (run_id, final_run_row) where the row's status is in
    completed/failed/paused, or the last row seen if the poll budget is exhausted.
    """
    from src.database.repositories.run_repository import RunRepository
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher
    from src.services.run_engine import RunEngine

    engine = RunEngine(
        session_factory=db_session_factory,
        registry=get_registry(),
        dispatch=lambda rid: asyncio.sleep(0),
    )
    run = await engine.start_run(
        deliverable_id=deliverable_id,
        tenant_id=tenant,
        subject_id=subject,
        parent_run_id=None,
        pause_after_stages=pause_after_stages or [],
    )
    dispatcher = RunDispatcher(
        session_factory=db_session_factory,
        registry=get_registry(),
        redis=redis_client,
    )
    try:
        await dispatcher.dispatch(run.id)
        last = None
        for _ in range(max_polls):
            await asyncio.sleep(0.2)
            async with db_session_factory() as poll_session:
                last = await RunRepository(poll_session).get(run.id)
            if last is not None and last.status in ("completed", "failed", "paused"):
                break
    finally:
        await dispatcher.shutdown()
    return run.id, last


async def _dispatch_and_wait_with_run_engine(
    db_session_factory: Callable,
    redis_client,
    *,
    deliverable_id: UUID,
    tenant: UUID,
    subject: UUID,
    max_polls: int = 200,
):
    """Like `_dispatch_and_wait`, but for archetypes that spawn CHILD runs
    (course_compile_lessons). The injected RunEngine's `dispatch` callback is the
    REAL dispatcher (not a no-op), and it is bound onto the dispatcher via
    `bind_run_engine` so `course_compile_lessons` starts each child lesson run on
    the SAME in-process dispatcher; `await_runs` then sees the children terminate.
    `start_run` auto-dispatches the parent through the same callback, and
    `dispatch()` is idempotent, so no second dispatch call is needed.
    """
    from src.database.repositories.run_repository import RunRepository
    from src.services.blueprints.registry import get_registry
    from src.services.run_dispatcher import RunDispatcher
    from src.services.run_engine import RunEngine

    dispatcher = RunDispatcher(
        session_factory=db_session_factory,
        registry=get_registry(),
        redis=redis_client,
    )
    engine = RunEngine(
        session_factory=db_session_factory,
        registry=get_registry(),
        dispatch=dispatcher.dispatch,
    )
    dispatcher.bind_run_engine(engine)
    run = await engine.start_run(
        deliverable_id=deliverable_id,
        tenant_id=tenant,
        subject_id=subject,
        parent_run_id=None,
        pause_after_stages=[],
    )
    try:
        last = None
        for _ in range(max_polls):
            await asyncio.sleep(0.2)
            async with db_session_factory() as poll_session:
                last = await RunRepository(poll_session).get(run.id)
            if last is not None and last.status in ("completed", "failed", "paused"):
                break
    finally:
        await dispatcher.shutdown()
    return run.id, last


async def _completed_stage_names(db_session_factory: Callable, run_id: UUID) -> set[str]:
    """The set of stage names a run marked `completed`.

    Used for the anti-false-green assertion: a wrong patch target silently
    no-ops, and a weak `status == "completed"` check can still pass if the DAG
    short-circuited. Asserting the completed-stage set == the archetype's
    expected non-opt-in spine proves the DAG actually executed.
    """
    from src.database.repositories.stage_output_repository import StageOutputRepository

    async with db_session_factory() as session:
        rows = await StageOutputRepository(session).list_for_run(run_id)
    return {r.stage_name for r in rows if r.status == "completed"}


# ---------------------------------------------------------------------------
# Reusable downstream-client fakes
# ---------------------------------------------------------------------------


def _wm_client(output: dict) -> AsyncMock:
    """A fake WMClient whose .message returns a WMResponse wrapping `output`."""
    fake = AsyncMock()
    fake.message.return_value = WMResponse(
        output=output,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )
    return fake


def _slide_service_mock() -> AsyncMock:
    """A fake SlideServiceClient: resolve template -> start deck -> wait -> png."""
    fake = AsyncMock()
    fake.resolve_template_id.return_value = "tid-1"
    fake.start_deck.return_value = "job-1"
    fake.wait_for_deck.return_value = {}
    fake.get_slide_png.return_value = b"\x89PNG\r\n\x1a\n"
    return fake


def _tts_mock() -> AsyncMock:
    fake = AsyncMock()
    fake.synthesize.return_value = TTSSynthesis.model_validate(
        {
            "audio_bytes": b"\x00",
            "format": "wav",
            "duration_seconds": 5.0,
            "sample_rate": 24000,
            "provider": "test",
            "voice": "",
        }
    )
    return fake


#: The rate `_speech_rate_tts_mock` "speaks" at. Deliberately NOT
#: `time_budget.DEFAULT_WPM_NATURAL` (150.0): a stub that returns exactly the
#: hardcoded prior makes a measured/estimated mix-up invisible, because both
#: paths would produce the same seconds. 120 wpm is plausible for the live
#: kokoro voice (measured ~94-150 depending on content) and distinguishable.
_STUB_WPM = 120.0

#: Seconds a blank-but-not-empty line still costs. A real synthesis of e.g. "."
#: returns a short non-zero clip; returning 0.0 would let a zero-length total
#: pass a duration check it should not.
_STUB_FLOOR_SECONDS = 0.25


def _stub_synthesis(text: str) -> TTSSynthesis:
    """A TTSSynthesis whose duration is what `text` would really take at `_STUB_WPM`.

    Deterministic in the text alone, so a DAG test that measures the same script
    twice gets the same seconds both times and a duration assertion is stable.
    """
    words = len(str(text).split())
    seconds = max(words * 60.0 / _STUB_WPM, _STUB_FLOOR_SECONDS)
    return TTSSynthesis.model_validate(
        {
            "audio_bytes": b"\x00",
            "format": "wav",
            "duration_seconds": round(seconds, 3),
            "sample_rate": 24000,
            "provider": "test",
            "voice": "",
        }
    )


def _speech_rate_tts_mock() -> AsyncMock:
    """A fake TTSClient for the speech-rate MEASUREMENT path.

    ⚠️ This is a DIFFERENT seam from `_tts_mock`. The script stages measure every
    narration line by really synthesizing it (`speech_rate.measure_script_lines`,
    and the Phase-A/Phase-B probes `resolve_natural_wpm` /
    `resolve_script_sample_wpm`), through a `TTSClient` constructed at
    `src.services.stages.shared.speech_rate`'s OWN import site. Stubbing
    `primitives.narration.TTSClient` — which every archetype test does for the
    narrate/record stage — leaves that construction untouched, so the measurement
    reaches the real `tts.test` launchd unit on mini2 and the DAG test hangs
    until the poll budget expires with the run still `running`.

    Unlike `_tts_mock` (a flat 5.0s for every line) this returns a duration
    derived from the line's own word count, so the stage under test gets a
    plausible SPREAD of per-line seconds and the real measurement/extrapolation
    arithmetic is exercised rather than short-circuited.
    """
    fake = AsyncMock()

    async def _synthesize(*_args: object, text: str = "", **_kwargs: object) -> TTSSynthesis:
        return _stub_synthesis(text)

    fake.synthesize.side_effect = _synthesize
    return fake


def _compile_mock() -> AsyncMock:
    fake = AsyncMock()
    fake.compile.return_value = {
        "artifact_url": "http://a/x.mp4",
        "duration_seconds": 30.0,
    }
    return fake
