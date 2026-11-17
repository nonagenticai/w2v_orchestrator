"""Title cards are content-addressed, so a resumed compile does not re-pay for them.

Before this, ``build_title_card_shots`` stored every card under a fresh
``uuid7()`` and sent no Idempotency-Key: a pod roll mid-compile (every merge rolls
the ``replicas: 1`` orchestrator) re-recorded both cards on video_production. The
fix mirrors ``ken_burns``: a deterministic media id over every input that shapes
the card's bytes, probed with ``find_existing_media`` before recording.

The storage fake below is a real (in-memory) object store — ``object_exists``
answers from what was actually STORED — so a hit can only come from a card the
previous call really persisted, never from a stub that always says yes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.shared.title_cards import (
    build_title_card_shots,
    title_card_idempotency_key,
    title_card_media_id,
)


class _ObjectStore:
    """In-memory stage-artifact store with a truthful existence probe."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def initialize(self) -> None:
        pass

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        path = f"artifacts/{job_id}/{stage_name}/{filename}"
        self.objects[path] = data
        return path

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        return f"https://minio.test/{path}"

    async def stage_artifact_url(self, job_id: str, stage_name: str, filename: str) -> str | None:
        return f"https://minio.test/artifacts/{job_id}/{stage_name}/{filename}"

    async def object_exists(self, url: str) -> bool | None:
        return url.removeprefix("https://minio.test/") in self.objects


@pytest.fixture
def store():
    s = _ObjectStore()
    tts_storage.set_storage_for_tests(s)
    yield s
    tts_storage.set_storage_for_tests(None)


def _video_fake() -> AsyncMock:
    v = AsyncMock()
    v.record_animation.return_value = "anim-1"
    v.wait_for_content_recording.return_value = {"duration_seconds": 3.0}
    v.get_recording_video.return_value = (b"\x00mp4", "video/mp4", None)
    return v


def _snapshot(title: str = "My Course") -> dict[str, Any]:
    return {
        "title": title,
        "brand": {"name": "Acme"},
        "deliverable": {"format": "16x9"},
        "title_card": {"enabled": True},
    }


@pytest.mark.asyncio
async def test_a_second_call_reuses_both_stored_cards(store: _ObjectStore) -> None:
    """THE resume: the second call (same run, same inputs) records NOTHING and returns
    the identical shots — same URL, same deterministic id."""
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot=_snapshot())

    first_video = _video_fake()
    first_hits: list[str] = []
    first = await build_title_card_shots(
        ctx, ordered_seconds=None, client=first_video, cache_hits=first_hits
    )
    assert first_video.record_animation.await_count == 2
    assert first_hits == []

    second_video = _video_fake()
    second_hits: list[str] = []
    second = await build_title_card_shots(
        ctx, ordered_seconds=None, client=second_video, cache_hits=second_hits
    )

    second_video.record_animation.assert_not_awaited()
    assert sorted(second_hits) == ["title-intro", "title-outro"]
    assert first == second
    assert first[0] is not None and first[1] is not None
    assert first[0]["video_url"] != first[1]["video_url"]
    UUID(first[0]["id"])  # still a UUID-shaped shot id


@pytest.mark.asyncio
async def test_an_edited_title_is_re_recorded_not_served_stale(store: _ObjectStore) -> None:
    """The hash covers the card's inputs: changing the intro title must miss the
    cache for the intro (re-record) while the unchanged outro is still reused."""
    run_id = uuid4()
    first_ctx = RunContext(run_id=run_id, deliverable_id=uuid4(), config_snapshot=_snapshot())
    first_intro, _ = await build_title_card_shots(
        first_ctx, ordered_seconds=None, client=_video_fake()
    )

    edited_ctx = RunContext(
        run_id=run_id, deliverable_id=uuid4(), config_snapshot=_snapshot(title="Renamed")
    )
    video = _video_fake()
    hits: list[str] = []
    intro, _ = await build_title_card_shots(
        edited_ctx, ordered_seconds=None, client=video, cache_hits=hits
    )

    assert video.record_animation.await_count == 1
    assert video.record_animation.await_args.kwargs["params"]["title"] == "Renamed"
    assert hits == ["title-outro"]
    assert intro is not None and first_intro is not None
    assert intro["video_url"] != first_intro["video_url"]


@pytest.mark.asyncio
async def test_record_carries_the_run_scoped_idempotency_key(store: _ObjectStore) -> None:
    """Key-shape pin: ``{run_id}:title_card:{media_id}`` — the ken_burns construction.
    A resume that lands while the dead pod's record is still in flight must attach to
    that job, so the key may contain nothing attempt-specific."""
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot=_snapshot())
    video = _video_fake()
    await build_title_card_shots(ctx, ordered_seconds=None, client=video)

    intro_call = video.record_animation.await_args_list[0].kwargs
    media_id = title_card_media_id(
        part_id=intro_call["part_id"],
        params=intro_call["params"],
        width=intro_call["width"],
        height=intro_call["height"],
        fps=intro_call["fps"],
        duration_ms=intro_call["duration_ms"],
    )
    assert intro_call["idempotency_key"] == f"{ctx.run_id}:title_card:{media_id}"
    assert intro_call["idempotency_key"] == title_card_idempotency_key(str(ctx.run_id), media_id)
    assert len(media_id) == 40
    # Two cards, two distinct keys — intro and outro never dedupe onto one job.
    keys = {c.kwargs["idempotency_key"] for c in video.record_animation.await_args_list}
    assert len(keys) == 2


def test_media_id_ignores_param_ordering_but_not_values() -> None:
    base = {"title": "A", "subtitle": "B"}
    common = {"part_id": "title-intro", "width": 1920, "height": 1080, "fps": 30}
    a = title_card_media_id(params=base, duration_ms=3000, **common)
    b = title_card_media_id(params={"subtitle": "B", "title": "A"}, duration_ms=3000, **common)
    c = title_card_media_id(params=base, duration_ms=4000, **common)
    assert a == b
    assert a != c


@pytest.mark.asyncio
async def test_a_shrunk_card_is_never_served_the_cached_full_length_clip(
    store: _ObjectStore,
) -> None:
    """The 20% cap makes a card's length depend on the ORDER, so the duration must be
    part of its cache identity. Same run, same text: a 3.0s intro is stored first, then
    the order drops to 20s (2.0s cards). The 2.0s card must be RECORDED at 2000ms and
    stored under its own address — never handed the 3.0s clip already sitting there."""
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot=_snapshot())

    long_video = _video_fake()
    long_intro, long_outro = await build_title_card_shots(
        ctx, ordered_seconds=60, client=long_video
    )
    assert long_intro is not None and long_outro is not None
    assert long_intro["seconds"] == 3.0

    short_video = _video_fake()
    hits: list[str] = []
    short_intro, short_outro = await build_title_card_shots(
        ctx, ordered_seconds=20, client=short_video, cache_hits=hits
    )

    assert hits == []  # the stored 3.0s clips were NOT reused
    assert short_video.record_animation.await_count == 2
    assert {c.kwargs["duration_ms"] for c in short_video.record_animation.await_args_list} == {2000}
    assert short_intro is not None and short_outro is not None
    assert short_intro["seconds"] == 2.0
    assert short_intro["video_url"] != long_intro["video_url"]
    assert short_outro["video_url"] != long_outro["video_url"]
    assert short_intro["id"] != long_intro["id"]

    # ...and each length is still reused against ITSELF on a resume.
    resumed = _video_fake()
    resumed_hits: list[str] = []
    again = await build_title_card_shots(
        ctx, ordered_seconds=20, client=resumed, cache_hits=resumed_hits
    )
    resumed.record_animation.assert_not_awaited()
    assert sorted(resumed_hits) == ["title-intro", "title-outro"]
    assert again == (short_intro, short_outro)


def test_media_id_differs_by_duration_alone() -> None:
    common: dict[str, Any] = {
        "part_id": "title-intro",
        "params": {"title": "A"},
        "width": 1920,
        "height": 1080,
        "fps": 30,
    }
    ids = {title_card_media_id(duration_ms=ms, **common) for ms in (1500, 2000, 3000)}
    assert len(ids) == 3
