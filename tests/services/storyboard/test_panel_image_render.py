"""Service-level tests for storyboard panel image generation (spec 00, Phase 2).

Real postgres (``db_session``), fake ``AIContentGenerationClient`` — no HTTP, no
GPU. These cover the properties that cost money or ship broken images when they
regress: the prompt actually carries the referenced assets' canonical
descriptors, the seed is pinned and recorded, a repeat generate REUSES instead of
re-paying, an ACG 200 with a dead ``image_url`` fails loudly, the license-gated
relative path is re-published to something fetchable, and submissions serialise
BOTH in-pod (the semaphore) and across replicas (the Redis lock) — the second is
the one that actually holds in prod, where the orchestrator runs two pods.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from src.clients.schemas.ai_content_generation import AcgConceptImageJob
from src.database.models import Project
from src.database.repositories.storyboard_repository import (
    StoryboardAssetRepository,
    StoryboardPanelRepository,
    StoryboardRepository,
)
from src.services.storyboard.panel_prompt import build_panel_prompt
from src.services.storyboard.panel_submit_lock import (
    SUBMIT_LOCK_KEY,
    PanelSubmitBusyError,
    PanelSubmitLock,
)
from src.services.storyboard.render import (
    PanelImageDownstreamError,
    PanelImageOptions,
    StoryboardPanelImageService,
)
from src.services.tts_storage import set_storage_for_tests

pytestmark = pytest.mark.asyncio

_REMOTE_URL = "https://minio.test/buckets/concept/asset-1.png"
_RELATIVE_URL = "/api/concept-images/assets/asset-1/download"
_REPUBLISHED = "https://minio.test/w2v-orchestrator/artifacts/storyboard/panel.png"


# ─── fakes ───────────────────────────────────────────────────────────────
def _make_client(*, image_url: str | None = _REMOTE_URL, job_id: str = "job-1") -> AsyncMock:
    client = AsyncMock()
    client.create_concept_image = AsyncMock(return_value=job_id)
    client.wait_for_concept_image = AsyncMock(
        return_value=AcgConceptImageJob(
            job_id=job_id,
            asset_id="asset-1",
            status="completed",
            provider="local_image",
            image_url=image_url,
        )
    )
    client.download_concept_image = AsyncMock(return_value=b"\x89PNGfake")
    return client


class _FakeStorageHit:
    """Storage whose objects always exist — every ``find_existing_media`` hits."""

    async def stage_artifact_url(self, run_id: str, stage_name: str, filename: str) -> str:
        return f"https://minio.test/w2v-orchestrator/artifacts/{run_id}/{stage_name}/{filename}"

    async def object_exists(self, url: str) -> bool:
        return True


# ─── seeding ─────────────────────────────────────────────────────────────
async def _seed_project(db_session, *, tenant_id: UUID, subject_id: UUID) -> Project:
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="panel-image",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    db_session.add(project)
    await db_session.commit()
    return project


async def _seed_panel(
    db_session,
    project: Project,
    *,
    tenant_id: UUID,
    with_assets: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """A panel plus (optionally) a seeded Character + Location it references."""
    storyboard = await StoryboardRepository(db_session).ensure_for_project(
        project.id, tenant_id=tenant_id
    )
    assert storyboard is not None  # None only on an archived project
    assets_repo = StoryboardAssetRepository(db_session)
    refs: dict[str, Any] = {}
    if with_assets:
        refs["character"] = await assets_repo.create(
            project_id=project.id,
            position=0,
            kind="character",
            name="Mara",
            canonical_descriptor="a wiry courier in a rust-red jacket, cropped black hair",
            seed=4242,
            style_token="ink-wash",
        )
        refs["location"] = await assets_repo.create(
            project_id=project.id,
            position=1,
            kind="location",
            name="Dock 9",
            canonical_descriptor="a rain-slick cargo dock under sodium lamps",
            seed=99,
        )
    panel = await StoryboardPanelRepository(db_session).create(
        project_id=project.id,
        storyboard_id=storyboard.id,
        position=0,
        shot_key="clip-1",
        action_description="Mara vaults the barrier and lands running",
        composition_notes="subject left third, deep space right",
        shot_size="MS",
        angle="low",
        lighting_mood="hard rim light, wet reflections",
        art_style="graphic-novel ink",
        characters=[str(refs["character"].id)] if with_assets else None,
        location_asset_id=refs["location"].id if with_assets else None,
    )
    await db_session.commit()
    return panel, refs


@pytest.fixture
def ids() -> tuple[UUID, UUID]:
    return uuid4(), uuid4()


@pytest.fixture
def storage_hit():
    """Install the always-hit storage fake and remove it afterwards."""
    set_storage_for_tests(cast("Any", _FakeStorageHit()))
    try:
        yield
    finally:
        set_storage_for_tests(None)


# ─── prompt + seed ───────────────────────────────────────────────────────
async def test_prompt_carries_asset_descriptors_and_pins_the_asset_seed(db_session, ids) -> None:
    """The prompt is composed from the panel's direction PLUS every referenced
    asset's canonical descriptor, and the seed is the character's pinned seed."""
    tenant_id, subject_id = ids
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    panel, refs = await _seed_panel(db_session, project, tenant_id=tenant_id)
    assets = {
        a.id: a
        for a in await StoryboardAssetRepository(db_session).list_for_project(
            project.id, tenant_id=tenant_id
        )
    }

    built = build_panel_prompt(panel, assets)

    assert "a wiry courier in a rust-red jacket" in built.prompt
    assert "a rain-slick cargo dock under sodium lamps" in built.prompt
    assert "Mara vaults the barrier" in built.prompt
    assert "MS low" in built.prompt
    assert "graphic-novel ink" in built.prompt
    # The CHARACTER's seed wins over the location's — the subject is the thing
    # that must stay on-model between panels.
    assert built.seed == refs["character"].seed == 4242
    assert built.style_token == "ink-wash"


async def test_seed_is_stable_without_any_seeded_asset(db_session, ids) -> None:
    """A panel referencing nothing still gets a REPRODUCIBLE seed (derived from
    its id), so a regenerate reproduces rather than re-rolls."""
    tenant_id, subject_id = ids
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    panel, _ = await _seed_panel(db_session, project, tenant_id=tenant_id, with_assets=False)

    first = build_panel_prompt(panel, {})
    second = build_panel_prompt(panel, {})

    assert first.seed == second.seed
    # int32, not uint32 — see test_derived_seeds_are_always_int32 below. This
    # assertion said `< 2**32` and passed while half of all panels 422'd.
    assert 0 <= first.seed <= 2_147_483_647


# ─── generation ──────────────────────────────────────────────────────────
async def test_generate_writes_media_ref_and_gen_with_the_seed(db_session, ids) -> None:
    """The durable outcome: media_ref points at the image and gen records the
    seed + prompt, so the generation is reproducible from the panel alone."""
    tenant_id, subject_id = ids
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    panel, refs = await _seed_panel(db_session, project, tenant_id=tenant_id)
    client = _make_client()

    result = await StoryboardPanelImageService(db_session, client=client).generate(
        panel.id, tenant_id=tenant_id
    )

    assert result.media_ref.model_dump() == {
        "type": "ai_image",
        "asset_id": "asset-1",
        "url": _REMOTE_URL,
    }
    assert result.gen.seed == refs["character"].seed
    assert "a wiry courier" in result.gen.prompt
    assert result.gen.provider == "local_image"
    row = await StoryboardPanelRepository(db_session).get(panel.id, tenant_id=tenant_id)
    assert row is not None
    assert row.visual_type == "ai_image"
    assert row.source == "ai"
    assert row.media_ref["url"] == _REMOTE_URL
    assert row.gen["seed"] == refs["character"].seed
    # The seed the panel records is the seed ACG was actually told to use.
    _prompt, config = client.create_concept_image.await_args.args
    assert config["extra_params"]["seed"] == refs["character"].seed


async def test_submit_carries_a_deterministic_idempotency_key(db_session, ids) -> None:
    """Without the key a retried submit starts a SECOND paid generation. The key
    is content-addressed, so it is identical across processes and retries."""
    tenant_id, subject_id = ids
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    panel, _ = await _seed_panel(db_session, project, tenant_id=tenant_id)

    first_client = _make_client()
    await StoryboardPanelImageService(db_session, client=first_client).generate(
        panel.id, tenant_id=tenant_id
    )
    second_client = _make_client()
    await StoryboardPanelImageService(db_session, client=second_client).generate(
        panel.id, tenant_id=tenant_id
    )

    # ``.get``, not ``[...]``: a dropped key must fail on the assertion below
    # (which names the consequence) rather than as a bare KeyError.
    first_key = first_client.create_concept_image.await_args.kwargs.get("idempotency_key")
    second_key = second_client.create_concept_image.await_args.kwargs.get("idempotency_key")
    assert first_key, "no idempotency key was sent — a retry would double-bill"
    assert first_key == second_key
    assert first_key.startswith(f"storyboard:{project.id}:")


async def test_repeat_generate_reuses_the_stored_image_instead_of_re_paying(
    db_session, ids, storage_hit
) -> None:
    """An unchanged panel regenerated addresses the same content-addressed media
    object, so ACG is never called and nothing is paid for."""
    tenant_id, subject_id = ids
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    panel, _ = await _seed_panel(db_session, project, tenant_id=tenant_id)
    client = _make_client()

    result = await StoryboardPanelImageService(db_session, client=client).generate(
        panel.id, tenant_id=tenant_id
    )

    assert result.reused is True
    assert result.gen.reused is True
    client.create_concept_image.assert_not_awaited()
    client.wait_for_concept_image.assert_not_awaited()


@pytest.mark.parametrize("dead", [None])
async def test_dead_image_url_fails_loudly(db_session, ids, dead) -> None:
    """ACG can 200 with a semantically empty url. Writing it onto the panel puts
    a broken image on the board behind a green status — fail instead."""
    tenant_id, subject_id = ids
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    panel, _ = await _seed_panel(db_session, project, tenant_id=tenant_id)
    client = _make_client(image_url=dead)

    with pytest.raises(PanelImageDownstreamError):
        await StoryboardPanelImageService(db_session, client=client).generate(
            panel.id, tenant_id=tenant_id
        )

    row = await StoryboardPanelRepository(db_session).get(panel.id, tenant_id=tenant_id)
    assert row is not None
    assert row.media_ref == {}, "a dead url must never reach media_ref"


async def test_relative_acg_path_is_republished_to_a_fetchable_url(
    db_session, ids, monkeypatch
) -> None:
    """ACG's download path is license-gated: the operator's browser cannot fetch
    it. The bytes are downloaded with the licensed client and re-stored, and the
    FETCHABLE url is what lands in media_ref."""
    tenant_id, subject_id = ids
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    panel, _ = await _seed_panel(db_session, project, tenant_id=tenant_id)
    client = _make_client(image_url=_RELATIVE_URL)
    captured: dict[str, Any] = {}

    async def _fake_store(*, run_id, stage_name, image_id, image_bytes, image_format):
        captured.update(
            run_id=run_id, stage_name=stage_name, image_bytes=image_bytes, fmt=image_format
        )
        return _REPUBLISHED

    monkeypatch.setattr("src.services.storyboard.render.store_generated_image", _fake_store)

    result = await StoryboardPanelImageService(db_session, client=client).generate(
        panel.id, tenant_id=tenant_id
    )

    assert result.media_ref.url == _REPUBLISHED
    assert result.media_ref.url is not None and not result.media_ref.url.startswith("/"), (
        "a license-gated relative path reached media_ref — the board would show a broken image"
    )
    client.download_concept_image.assert_awaited_once_with(_RELATIVE_URL)
    assert captured["image_bytes"] == b"\x89PNGfake"
    assert captured["run_id"] == f"storyboard/{project.id}"


def _png_of(width: int, height: int) -> bytes:
    """A PNG whose IHDR declares *width* x *height* (header only — enough to measure)."""
    return (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00"
    )


async def test_gen_records_the_measured_size_not_the_requested_one(
    db_session, ids, monkeypatch
) -> None:
    """``gen`` is the reproducibility record, so a size it states must be the
    size the image HAS.

    The generator overrides the request — a live panel asked for 1024x1024 and
    was delivered 1536x1536 — while ``_persist`` wrote ``job_config.size``, so the
    field asserted something the picture next to it contradicted. The republish
    path already holds the bytes, so the dimensions are measured from them.
    """
    tenant_id, subject_id = ids
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    panel, _ = await _seed_panel(db_session, project, tenant_id=tenant_id)
    client = _make_client(image_url=_RELATIVE_URL)
    client.download_concept_image = AsyncMock(return_value=_png_of(1536, 1536))

    async def _fake_store(*, run_id, stage_name, image_id, image_bytes, image_format):
        return _REPUBLISHED

    monkeypatch.setattr("src.services.storyboard.render.store_generated_image", _fake_store)

    result = await StoryboardPanelImageService(db_session, client=client).generate(
        panel.id, tenant_id=tenant_id, options=PanelImageOptions(size="1024x1024")
    )

    assert result.gen.size == "1536x1536"
    _prompt, config = client.create_concept_image.await_args.args
    assert config["size"] == "1024x1024"
    assert result.gen.size != config["size"], (
        "gen.size is the REQUESTED size again — the field is back to asserting "
        "something the delivered image contradicts"
    )
    row = await StoryboardPanelRepository(db_session).get(panel.id, tenant_id=tenant_id)
    assert row is not None
    assert row.gen["size"] == "1536x1536"


async def test_gen_size_is_none_when_the_bytes_never_pass_through_us(
    db_session, ids, monkeypatch
) -> None:
    """An absolute provider url is served straight through, so we never hold the
    pixels. Unknown is recorded as ``None`` — recording the request instead is
    exactly the lie this change removes.

    Also covers the corrupt-body path: a republish whose bytes carry no readable
    PNG header must not raise, it must simply leave the size unknown.
    """
    tenant_id, subject_id = ids
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)

    panel, _ = await _seed_panel(db_session, project, tenant_id=tenant_id)
    passthrough = await StoryboardPanelImageService(db_session, client=_make_client()).generate(
        panel.id, tenant_id=tenant_id, options=PanelImageOptions(size="1024x1024")
    )
    assert passthrough.media_ref.url == _REMOTE_URL
    assert passthrough.gen.size is None

    # A second project: panels are uniquely keyed (project_id, shot_key) and
    # ``_seed_panel`` always mints clip-1.
    other_project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    other, _ = await _seed_panel(db_session, other_project, tenant_id=tenant_id)
    client = _make_client(image_url=_RELATIVE_URL)
    client.download_concept_image = AsyncMock(return_value=b"\x89PNG-truncated")

    async def _fake_store(*, run_id, stage_name, image_id, image_bytes, image_format):
        return _REPUBLISHED

    monkeypatch.setattr("src.services.storyboard.render.store_generated_image", _fake_store)

    corrupt = await StoryboardPanelImageService(db_session, client=client).generate(
        other.id, tenant_id=tenant_id
    )
    assert corrupt.media_ref.url == _REPUBLISHED, "a short body must not fail the generation"
    assert corrupt.gen.size is None


async def test_republish_failure_fails_the_generation(db_session, ids) -> None:
    """Unlike the run stage (whose consumer has a fallback), a panel's consumer
    is a browser: an un-republished image is simply broken, so this is fatal."""
    tenant_id, subject_id = ids
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    panel, _ = await _seed_panel(db_session, project, tenant_id=tenant_id)
    client = _make_client(image_url=_RELATIVE_URL)
    client.download_concept_image = AsyncMock(side_effect=RuntimeError("acg down"))

    with pytest.raises(PanelImageDownstreamError):
        await StoryboardPanelImageService(db_session, client=client).generate(
            panel.id, tenant_id=tenant_id
        )


async def test_cross_tenant_panel_is_not_found(db_session, ids) -> None:
    tenant_id, subject_id = ids
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    panel, _ = await _seed_panel(db_session, project, tenant_id=tenant_id)

    from src.services.storyboard.render import PanelNotFoundError

    with pytest.raises(PanelNotFoundError):
        await StoryboardPanelImageService(db_session, client=_make_client()).generate(
            panel.id, tenant_id=uuid4()
        )


async def test_seed_override_changes_the_media_id_so_it_regenerates(db_session, ids) -> None:
    """An explicit re-roll must NOT hit the cache of the previous seed."""
    tenant_id, subject_id = ids
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    panel, _ = await _seed_panel(db_session, project, tenant_id=tenant_id)

    default_result = await StoryboardPanelImageService(db_session, client=_make_client()).generate(
        panel.id, tenant_id=tenant_id
    )
    rolled_result = await StoryboardPanelImageService(db_session, client=_make_client()).generate(
        panel.id, tenant_id=tenant_id, options=PanelImageOptions(seed=7)
    )

    assert rolled_result.gen.seed == 7
    # The content-addressed media id is the cache key; a different seed must
    # produce a different one or the re-roll would silently replay the old image.
    assert rolled_result.gen.media_id != default_result.gen.media_id


# ─── serialisation ───────────────────────────────────────────────────────
async def test_submissions_serialise(db_session, db_session_factory, ids) -> None:
    """The image runner behind ACG is single-threaded and already carries the
    live ai_image_gen stage. Two concurrent panel generates must never be in
    flight at ACG at the same time, or a board preview head-of-line blocks a
    real run.

    The overlap is forced, not hoped for: the FIRST submission parks inside ACG
    until the second one enters (or half a second passes). If the gate is open
    the second enters immediately and ``peak`` is 2; if it is closed the wait
    times out and ``peak`` stays 1. That removes the timing luck that made an
    earlier version of this test pass with the gate REMOVED — it only reddened
    when run alone, because under a loaded DB pool the two tasks happened not to
    overlap inside the 50ms sleep it used instead.
    """
    tenant_id, subject_id = ids
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    panel_a, _ = await _seed_panel(db_session, project, tenant_id=tenant_id)
    storyboard = await StoryboardRepository(db_session).ensure_for_project(
        project.id, tenant_id=tenant_id
    )
    assert storyboard is not None  # None only on an archived project
    panel_b = await StoryboardPanelRepository(db_session).create(
        project_id=project.id,
        storyboard_id=storyboard.id,
        position=1,
        shot_key="clip-2",
        action_description="the crate splits open",
    )
    await db_session.commit()

    in_flight = 0
    peak = 0
    second_entered = asyncio.Event()

    async def _parking_submit(*_args: Any, **_kwargs: Any) -> str:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            if in_flight == 1:
                # Hold the downstream open so a NON-serialised sibling has a
                # real, generous window to walk in behind us.
                with suppress(TimeoutError):
                    await asyncio.wait_for(second_entered.wait(), timeout=0.5)
            else:
                second_entered.set()
            return "job-1"
        finally:
            in_flight -= 1

    # Both sessions are opened BEFORE the gather: acquiring a pooled connection
    # can itself serialise the two tasks, which would make this test pass for
    # the wrong reason.
    async with db_session_factory() as session_a, db_session_factory() as session_b:

        async def _run(session: Any, panel_id: UUID) -> None:
            client = _make_client()
            client.create_concept_image = AsyncMock(side_effect=_parking_submit)
            await StoryboardPanelImageService(session, client=client).generate(
                panel_id, tenant_id=tenant_id
            )
            await session.commit()

        await asyncio.gather(_run(session_a, panel_a.id), _run(session_b, panel_b.id))

    assert peak == 1, f"{peak} concurrent ACG submissions — panel generation is not serialised"


# ─── cross-replica serialisation ─────────────────────────────────────────
async def test_cross_replica_lock_blocks_a_second_holder(redis_client) -> None:
    """The in-pod semaphore is per-process; with replicas:2 only this Redis key
    can stop two pods submitting at once. Prove a SECOND holder (the other
    replica) cannot enter while the first holds it, and can once it exits."""
    first = PanelSubmitLock(redis_client, poll=0.01, max_wait=0.2)
    second = PanelSubmitLock(redis_client, poll=0.01, max_wait=0.2)

    async with first:
        with pytest.raises(PanelSubmitBusyError):
            async with second:
                pytest.fail("two replicas held the panel submission slot at once")

    # Released on exit, so the next replica gets in.
    async with second:
        pass


async def test_generate_holds_the_cross_replica_lock_while_submitting(
    db_session, ids, redis_client
) -> None:
    """A generation running on THIS pod must exclude a submission from another
    pod. The 'other replica' is a second PanelSubmitLock over the same Redis —
    it must be unable to acquire while ACG is being called."""
    tenant_id, subject_id = ids
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    panel, _ = await _seed_panel(db_session, project, tenant_id=tenant_id)
    other_replica_locked_out = False

    async def _submit_while_other_replica_tries(*_a: Any, **_kw: Any) -> str:
        nonlocal other_replica_locked_out
        try:
            async with PanelSubmitLock(redis_client, poll=0.01, max_wait=0.15):
                pass
        except PanelSubmitBusyError:
            other_replica_locked_out = True
        return "job-1"

    client = _make_client()
    client.create_concept_image = AsyncMock(side_effect=_submit_while_other_replica_tries)

    await StoryboardPanelImageService(db_session, client=client, redis=redis_client).generate(
        panel.id, tenant_id=tenant_id
    )

    assert other_replica_locked_out, (
        "a second replica could submit concurrently — the gate is per-pod only"
    )
    # And the slot is handed back, not leaked until TTL.
    assert await redis_client.get(SUBMIT_LOCK_KEY) is None


async def test_lock_is_released_when_the_generation_fails(db_session, ids, redis_client) -> None:
    """A failed generation must not wedge every other panel behind a held slot."""
    tenant_id, subject_id = ids
    project = await _seed_project(db_session, tenant_id=tenant_id, subject_id=subject_id)
    panel, _ = await _seed_panel(db_session, project, tenant_id=tenant_id)

    with pytest.raises(PanelImageDownstreamError):
        await StoryboardPanelImageService(
            db_session, client=_make_client(image_url="none"), redis=redis_client
        ).generate(panel.id, tenant_id=tenant_id)

    assert await redis_client.get(SUBMIT_LOCK_KEY) is None
