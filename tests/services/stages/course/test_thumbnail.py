"""Tests for the course_thumbnail stage handler."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.clients.schemas.ai_content_generation import AcgConceptImageJob
from src.services.runs.run_context import RunContext
from src.services.stages.base import StagePreconditionError, StageSkippedError
from src.services.stages.course.thumbnail import CourseThumbnailStage


def _make_client(*, job_id: str = "job-1", result: AcgConceptImageJob | None = None) -> AsyncMock:
    client = AsyncMock()
    client.create_concept_image = AsyncMock(return_value=job_id)
    client.wait_for_concept_image = AsyncMock(
        return_value=result
        or AcgConceptImageJob(
            job_id=job_id,
            status="completed",
            provider="local_image",
            image_url="https://minio.test/w2v-orchestrator/cover.png",
        )
    )
    return client


def _ctx(cfg: dict | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=cfg
        or {
            "title": "Postgres 18 Crash Course",
            "brief": {"audience": "DBAs", "outcome": "tune it"},
        },
    )


async def test_happy_path_absolute_url_passthrough() -> None:
    client = _make_client()
    stage = CourseThumbnailStage(client=client)

    out = await stage.run({}, ctx=_ctx())

    assert out.thumbnail_url == "https://minio.test/w2v-orchestrator/cover.png"
    # Prompt is built from the course title + brief audience/outcome.
    prompt, _cfg = client.create_concept_image.await_args.args
    assert "Postgres 18 Crash Course" in prompt
    assert "DBAs" in prompt
    assert "tune it" in prompt


async def test_relative_url_is_downloaded_and_stored(monkeypatch) -> None:
    client = _make_client(
        result=AcgConceptImageJob(
            job_id="job-1",
            status="completed",
            image_url="/api/concept-images/assets/cover-1/download",
        )
    )
    client.download_concept_image = AsyncMock(return_value=b"\x89PNGfake")
    captured: dict = {}

    async def _fake_store(*, run_id, stage_name, image_id, image_bytes, image_format):
        captured["stage"] = stage_name
        captured["bytes"] = image_bytes
        return "https://minio.test/w2v-orchestrator/artifacts/run/course_thumbnail/x.png"

    monkeypatch.setattr("src.services.stages.course.thumbnail.store_generated_image", _fake_store)
    stage = CourseThumbnailStage(client=client)

    out = await stage.run({}, ctx=_ctx())

    assert out.thumbnail_url is not None
    assert out.thumbnail_url.endswith("/course_thumbnail/x.png")
    assert out.notes == []
    client.download_concept_image.assert_awaited_once_with(
        "/api/concept-images/assets/cover-1/download"
    )
    assert captured["stage"] == "course_thumbnail"
    assert captured["bytes"] == b"\x89PNGfake"


async def test_skips_when_disabled_via_false() -> None:
    client = _make_client()
    stage = CourseThumbnailStage(client=client)
    with pytest.raises(StageSkippedError):
        await stage.run({}, ctx=_ctx({"course_thumbnail": False}))
    client.create_concept_image.assert_not_awaited()


async def test_skips_when_disabled_via_enabled_false() -> None:
    client = _make_client()
    stage = CourseThumbnailStage(client=client)
    with pytest.raises(StageSkippedError):
        await stage.run({}, ctx=_ctx({"course_thumbnail": {"enabled": False}}))
    client.create_concept_image.assert_not_awaited()


async def test_generation_failure_degrades_it_does_not_skip() -> None:
    """A FAILED cover must not masquerade as a cover the run never asked for.

    `StageSkippedError` means "this opt-in stage was not requested" and the DAG
    executor maps it to `skipped` — terminal, counted toward completion, invisible
    in run status. Routing a real failure through it made a paid, attempted, failed
    generation indistinguishable from a brief that never wanted a cover, so the
    course shipped a blank poster on a clean green run. It now completes with a
    note, which `runs/_shared.py` renders as `degraded`.
    """
    client = _make_client()
    client.create_concept_image = AsyncMock(side_effect=RuntimeError("image-gen down"))
    stage = CourseThumbnailStage(client=client)

    out = await stage.run({}, ctx=_ctx())

    assert out.thumbnail_url is None
    assert len(out.notes) == 1
    assert "image-gen down" in out.notes[0]


async def test_empty_image_url_degrades() -> None:
    client = _make_client(
        result=AcgConceptImageJob(job_id="job-1", status="failed", image_url="None")
    )
    stage = CourseThumbnailStage(client=client)

    out = await stage.run({}, ctx=_ctx())

    assert out.thumbnail_url is None
    assert out.notes


async def test_disabled_still_skips_not_degrades() -> None:
    """The one case `StageSkippedError` is actually for must keep using it."""
    client = _make_client()
    stage = CourseThumbnailStage(client=client)
    with pytest.raises(StageSkippedError):
        await stage.run({}, ctx=_ctx({"course_thumbnail": False}))
    client.create_concept_image.assert_not_awaited()


async def test_explicit_prompt_override() -> None:
    client = _make_client()
    stage = CourseThumbnailStage(client=client)
    cfg = {"title": "X", "course_thumbnail": {"prompt": "a hand-drawn owl"}}
    await stage.run({}, ctx=_ctx(cfg))
    prompt, _cfg = client.create_concept_image.await_args.args
    assert prompt == "a hand-drawn owl"


async def test_paid_call_carries_a_stable_idempotency_key() -> None:
    """The ONLY paid-generation call site that used to omit `idempotency_key`.

    A re-dispatch (resumable timeout) or a reaper re-claim re-runs this handler.
    Without a dedupe key the re-submit starts a SECOND billed GPU image job that
    ACG cannot dedupe, because nothing addresses the first.
    """
    ctx = _ctx()
    client_a = _make_client()
    client_b = _make_client()

    await CourseThumbnailStage(client=client_a).run({}, ctx=ctx)
    await CourseThumbnailStage(client=client_b).run({}, ctx=ctx)

    key_a = client_a.create_concept_image.await_args.kwargs["idempotency_key"]
    key_b = client_b.create_concept_image.await_args.kwargs["idempotency_key"]
    assert key_a, "paid generation was submitted with no idempotency key"
    assert key_a.startswith(f"{ctx.run_id}:course_thumbnail:")
    # Same run, same prompt, same job config → the SAME key, so the re-dispatch
    # addresses the original job instead of double-billing.
    assert key_a == key_b


async def test_a_different_prompt_gets_a_different_key() -> None:
    """The key must be content-addressed, not just per-(run, stage)."""
    client_a = _make_client()
    client_b = _make_client()
    ctx_a = _ctx({"title": "A", "course_thumbnail": {"prompt": "cover of A"}})
    ctx_b = RunContext(
        run_id=ctx_a.run_id,
        deliverable_id=ctx_a.deliverable_id,
        config_snapshot={"title": "B", "course_thumbnail": {"prompt": "cover of B"}},
    )

    await CourseThumbnailStage(client=client_a).run({}, ctx=ctx_a)
    await CourseThumbnailStage(client=client_b).run({}, ctx=ctx_b)

    assert (
        client_a.create_concept_image.await_args.kwargs["idempotency_key"]
        != client_b.create_concept_image.await_args.kwargs["idempotency_key"]
    )


async def test_an_already_stored_cover_short_circuits_the_paid_call(monkeypatch) -> None:
    """Resume check: a stored cover is reused, not paid for again."""
    stored = "https://minio.test/w2v-orchestrator/artifacts/run/course_thumbnail/c.png"

    async def _fake_find(run_id, stage_name, media_id, ext):
        assert stage_name == "course_thumbnail"
        assert ext == "png"
        return stored

    monkeypatch.setattr("src.services.stages.course.thumbnail.find_existing_media", _fake_find)
    client = _make_client()

    out = await CourseThumbnailStage(client=client).run({}, ctx=_ctx())

    client.create_concept_image.assert_not_awaited()
    assert out.thumbnail_url == stored
    assert out.notes == []


async def test_precondition_error_propagates_it_does_not_degrade() -> None:
    """Control-flow signals are NOT failures.

    `StagePreconditionError` is a `StageError` subclass, so the soft-degrade
    `except Exception` swallowed it into "cover generation failed" — the run engine
    never saw it. Siblings (`shared/script_doctor.py`, `audiobook/qa.py`,
    `shared/loop_engine.py`, `shared/footage_critique.py`) re-raise the PAIR.
    """
    client = _make_client()
    client.create_concept_image = AsyncMock(
        side_effect=StagePreconditionError("cover: required input absent")
    )
    stage = CourseThumbnailStage(client=client)

    with pytest.raises(StagePreconditionError):
        await stage.run({}, ctx=_ctx())
