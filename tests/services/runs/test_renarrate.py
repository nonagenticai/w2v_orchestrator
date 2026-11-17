"""Tests for single-line re-narrate (studio "Re-narrate this line").

A re-narrate spawns a child run that re-synthesizes EXACTLY ONE TTS clip, patches
that line's text on the stage the narrate handler reads (so the captions match the
new audio) plus that one clip entry on the narrate stage's preserved output, and
re-runs only the compile stage — every sibling clip's audio is reused verbatim.

The regression guard that matters most is ``test_renarrate_synthesizes_exactly_one_clip``:
cutting the DAG at the NARRATE stage (instead of at compile) would look surgical but
re-synthesize the entire narration, because ``find_existing_clip`` is keyed by run_id
and a regenerate always spawns a new child run.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import UUID, uuid7

import pytest

from src.clients.schemas.tts import TTSSynthesis
from src.database.models import Deliverable, Project, Run, StageOutput
from src.database.repositories.stage_output_repository import StageOutputRepository
from src.services import tts_storage
from src.services.blueprints.registry import get_registry
from src.services.run_engine import RunEngine
from src.services.runs import regenerate as regen
from src.services.runs.regenerate import (
    RenarrateError,
    build_renarrate_state,
)

PROMO_STAGES = (
    "promo_analyze",
    "promo_script",
    "promo_storyboard",
    "promo_image_prompts",
    "ai_image_gen",
    "promo_narrate",
    "promo_render",
)

# A 3-line script: re-narrating ONE line must leave the other two untouched.
LINES = [
    {"id": "L1", "text": "One."},
    {"id": "L2", "text": "Two."},
    {"id": "L3", "text": "Three."},
]
SHOTS = [
    {"id": "s1", "line_id": "L1", "seconds": 3},
    {"id": "s2", "line_id": "L2", "seconds": 4},
    {"id": "s3", "line_id": "L3", "seconds": 5},
]
CLIPS = [
    {"id": "c1", "line_id": "L1", "artifact_url": "https://minio.test/c1.wav", "seconds": 3.0},
    {"id": "c2", "line_id": "L2", "artifact_url": "https://minio.test/c2.wav", "seconds": 4.0},
    {"id": "c3", "line_id": "L3", "artifact_url": "https://minio.test/c3.wav", "seconds": 5.0},
]
CONFIG_VOICE = "af_heart"


class _FakeStorage:
    """Minimal clip-storage fake (no probe methods → find_existing_clip returns None)."""

    def __init__(self) -> None:
        self.stored: list[tuple[str, str, str]] = []

    async def initialize(self) -> None:
        pass

    async def store_stage_artifact(
        self, job_id: str, stage_name: str, data: bytes, filename: str
    ) -> str:
        self.stored.append((job_id, stage_name, filename))
        return f"artifacts/{job_id}/{stage_name}/{filename}"

    async def get_presigned_url(
        self, path: str, expires: int | None = None, bucket: str | None = None
    ) -> str | None:
        return f"https://minio.test/{path}"


@pytest.fixture
def fake_storage():
    storage = _FakeStorage()
    tts_storage.set_storage_for_tests(storage)
    yield storage
    tts_storage.set_storage_for_tests(None)


@pytest.fixture
def fake_tts(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch the TTSClient the re-narrate builder constructs; count synthesize calls."""
    client = AsyncMock()
    client.synthesize.return_value = TTSSynthesis(
        audio_bytes=b"\x01\x02\x03", sample_rate=0, duration_seconds=6.5, provider="", voice=""
    )
    monkeypatch.setattr(regen, "TTSClient", lambda: client)
    return client


async def _seed_promo_parent(
    db_session,
    *,
    audio_url_on: str | None = None,
) -> tuple[Project, Deliverable, Run]:
    """A completed 3-line promo run. ``audio_url_on`` stamps an uploaded voiceover
    onto that spine shot (the slice-2 upload path)."""
    tenant_id, subject_id = uuid7(), uuid7()
    project = Project(
        id=uuid7(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="promo",
        title="renarrate-parent",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    deliverable = Deliverable(
        id=uuid7(),
        project_id=project.id,
        name="d",
        format="16x9",
        captions_mode="burned_in",
        target_duration_seconds=30,
    )
    parent = Run(
        id=uuid7(),
        deliverable_id=deliverable.id,
        blueprint_id="promo_v1",
        status="completed",
        config_snapshot={
            "project_id": str(project.id),
            "tenant_id": str(tenant_id),
            "subject_id": str(subject_id),
            "title": project.title,
            "type": "promo",
            "brief": {},
            "brand": {},
            "voice": {"voice_id": CONFIG_VOICE, "descriptor": "warm"},
            "theme": {},
            "deliverable": {
                "id": str(deliverable.id),
                "name": deliverable.name,
                "format": deliverable.format,
                "captions_mode": deliverable.captions_mode,
                "target_duration_seconds": deliverable.target_duration_seconds,
            },
        },
        pause_after_stages=[],
    )
    # Explicit union (not Any): the SHOTS literals narrow to `str | int`, but a
    # voiceover override adds a float `audio_seconds`.
    shots: list[dict[str, str | int | float]] = [dict(s) for s in SHOTS]
    if audio_url_on is not None:
        for shot in shots:
            if shot["id"] == audio_url_on:
                shot["audio_url"] = "https://minio.test/voiceovers/t/clip_voiceover_override/u.mp3"
                shot["audio_seconds"] = 4.2
    outputs: dict[str, dict] = {
        "promo_script": {"lines": [dict(line) for line in LINES]},
        "promo_storyboard": {"shots": shots},
        "promo_narrate": {"clips": [dict(c) for c in CLIPS]},
    }
    rows = [
        StageOutput(
            id=uuid7(),
            run_id=parent.id,
            stage_name=name,
            status="completed",
            output=outputs.get(name, {"stage": name}),
        )
        for name in PROMO_STAGES
    ]
    db_session.add_all([project, deliverable, parent, *rows])
    await db_session.commit()
    return project, deliverable, parent


# ── Pure unit: stage resolution ────────────────────────────────────────────


# ── DB-backed: build_renarrate_state ───────────────────────────────────────


async def test_renarrate_synthesizes_exactly_one_clip(
    db_session, fake_storage: _FakeStorage, fake_tts: AsyncMock
) -> None:
    """THE regression guard: re-narrating 1 of 3 lines costs exactly 1 TTS call.

    Cutting the DAG at the narrate stage would re-synthesize all 3 (find_existing_clip
    is keyed by run_id, and the child run's id is new → every line misses).
    """
    _, _, parent = await _seed_promo_parent(db_session)

    preserved = await build_renarrate_state(
        session=db_session,
        parent_run=parent,
        blueprint=get_registry().resolve("promo"),
        spine_stage="promo_storyboard",
        element_id="s2",
        narration_text="Two, but better.",
    )

    assert fake_tts.synthesize.await_count == 1
    # ...and the one clip is stored under the PARENT run id (content-addressed id,
    # so it cannot collide with the parent's own clips) at the narrate stage.
    assert [(run_id, stage) for run_id, stage, _ in fake_storage.stored] == [
        (str(parent.id), "promo_narrate"),
        (str(parent.id), "promo_narrate"),  # wav + duration sidecar
    ]
    # Cut at COMPILE: narrate is preserved (patched), only promo_render re-runs.
    assert "promo_narrate" in preserved
    assert "promo_render" not in preserved


async def test_renarrate_patches_the_narrate_source_text_for_that_line_only(
    db_session, fake_storage: _FakeStorage, fake_tts: AsyncMock
) -> None:
    """Captions derive from the narrate source stage's text — it must track the audio."""
    _, _, parent = await _seed_promo_parent(db_session)

    preserved = await build_renarrate_state(
        session=db_session,
        parent_run=parent,
        blueprint=get_registry().resolve("promo"),
        spine_stage="promo_storyboard",
        element_id="s2",
        narration_text="Two, but better.",
    )

    lines = preserved["promo_script"]["lines"]
    assert [line["text"] for line in lines] == ["One.", "Two, but better.", "Three."]
    assert [line["id"] for line in lines] == ["L1", "L2", "L3"]
    # The spine itself is preserved verbatim (the text lives on the script, not here).
    assert preserved["promo_storyboard"]["shots"] == SHOTS


async def test_renarrate_replaces_only_the_targeted_clip_entry(
    db_session, fake_storage: _FakeStorage, fake_tts: AsyncMock
) -> None:
    _, _, parent = await _seed_promo_parent(db_session)

    preserved = await build_renarrate_state(
        session=db_session,
        parent_run=parent,
        blueprint=get_registry().resolve("promo"),
        spine_stage="promo_storyboard",
        element_id="s2",
        narration_text="Two, but better.",
    )

    clips = preserved["promo_narrate"]["clips"]
    assert [c["line_id"] for c in clips] == ["L1", "L2", "L3"]
    # Siblings byte-identical → their stored audio is reused by the recompile.
    assert clips[0] == CLIPS[0]
    assert clips[2] == CLIPS[2]
    # The target clip is fully replaced: new content-addressed id, new url, new duration.
    target = clips[1]
    assert target["id"] != CLIPS[1]["id"]
    assert target["artifact_url"] != CLIPS[1]["artifact_url"]
    assert target["artifact_url"].endswith(f"{target['id']}.wav")
    assert target["seconds"] == 6.5


async def test_renarrate_rejects_a_clip_with_an_uploaded_voiceover(
    db_session, fake_storage: _FakeStorage, fake_tts: AsyncMock
) -> None:
    """An uploaded voiceover wins over TTS in assemble_shots → the re-narrate would
    be silently inaudible. 422 instead (the route maps RenarrateError → 422)."""
    _, _, parent = await _seed_promo_parent(db_session, audio_url_on="s2")

    with pytest.raises(RenarrateError, match="Remove the uploaded voiceover"):
        await build_renarrate_state(
            session=db_session,
            parent_run=parent,
            blueprint=get_registry().resolve("promo"),
            spine_stage="promo_storyboard",
            element_id="s2",
            narration_text="Two, but better.",
        )
    fake_tts.synthesize.assert_not_awaited()


async def test_renarrate_rejects_unknown_clip_and_spineless_stage(
    db_session, fake_storage: _FakeStorage, fake_tts: AsyncMock
) -> None:
    _, _, parent = await _seed_promo_parent(db_session)
    blueprint = get_registry().resolve("promo")

    with pytest.raises(RenarrateError):
        await build_renarrate_state(
            session=db_session,
            parent_run=parent,
            blueprint=blueprint,
            spine_stage="promo_storyboard",
            element_id="does-not-exist",
            narration_text="x",
        )
    # A spine at/after the compile cut has no narration stage downstream.
    with pytest.raises(RenarrateError):
        await build_renarrate_state(
            session=db_session,
            parent_run=parent,
            blueprint=blueprint,
            spine_stage="promo_render",
            element_id="s2",
            narration_text="x",
        )
    fake_tts.synthesize.assert_not_awaited()


async def test_renarrate_closes_the_tts_client(
    db_session, fake_storage: _FakeStorage, fake_tts: AsyncMock
) -> None:
    """The builder constructs a ``TTSClient`` inline (a LicensedServiceClient that
    lazily opens a pooled httpx.AsyncClient). It must close it so a single-line
    re-narrate does not leak one connection pool per call — regression for the
    bare ``TTSClient()`` that was never torn down."""
    _, _, parent = await _seed_promo_parent(db_session)

    await build_renarrate_state(
        session=db_session,
        parent_run=parent,
        blueprint=get_registry().resolve("promo"),
        spine_stage="promo_storyboard",
        element_id="s2",
        narration_text="Two, but better.",
    )

    fake_tts.close.assert_awaited_once()


async def test_renarrate_closes_the_tts_client_even_when_synthesize_raises(
    db_session, fake_storage: _FakeStorage, fake_tts: AsyncMock
) -> None:
    """The close must run in a finally: a synthesis failure mid-re-narrate must
    not leak the pool either."""
    fake_tts.synthesize.side_effect = RuntimeError("tts down")
    _, _, parent = await _seed_promo_parent(db_session)

    with pytest.raises(RuntimeError, match="tts down"):
        await build_renarrate_state(
            session=db_session,
            parent_run=parent,
            blueprint=get_registry().resolve("promo"),
            spine_stage="promo_storyboard",
            element_id="s2",
            narration_text="Two, but better.",
        )

    fake_tts.close.assert_awaited_once()


# ── DB-backed: voice resolution ────────────────────────────────────────────


async def test_renarrate_forwards_an_explicit_voice(
    db_session, fake_storage: _FakeStorage, fake_tts: AsyncMock
) -> None:
    _, _, parent = await _seed_promo_parent(db_session)

    await build_renarrate_state(
        session=db_session,
        parent_run=parent,
        blueprint=get_registry().resolve("promo"),
        spine_stage="promo_storyboard",
        element_id="s2",
        voice="am_michael",
    )

    fake_tts.synthesize.assert_awaited_once_with(text="Two.", voice="am_michael")


async def test_renarrate_falls_back_to_the_configured_voice(
    db_session, fake_storage: _FakeStorage, fake_tts: AsyncMock
) -> None:
    """Voice omitted ⇒ the run's configured voice; text omitted ⇒ the current line."""
    _, _, parent = await _seed_promo_parent(db_session)

    await build_renarrate_state(
        session=db_session,
        parent_run=parent,
        blueprint=get_registry().resolve("promo"),
        spine_stage="promo_storyboard",
        element_id="s3",
        narration_text="Three, revised.",
    )

    fake_tts.synthesize.assert_awaited_once_with(text="Three, revised.", voice=CONFIG_VOICE)


# ── DB-backed: engine wiring ───────────────────────────────────────────────


async def test_run_engine_renarrate_reruns_only_compile(
    db_session, db_session_factory, fake_storage: _FakeStorage, fake_tts: AsyncMock
) -> None:
    project, deliverable, parent = await _seed_promo_parent(db_session)
    dispatched: list[UUID] = []

    async def _record_dispatch(run_id: UUID) -> None:
        dispatched.append(run_id)

    engine = RunEngine(
        session_factory=db_session_factory, registry=get_registry(), dispatch=_record_dispatch
    )
    child = await engine.start_run(
        deliverable_id=deliverable.id,
        tenant_id=project.tenant_id,
        subject_id=project.owner_subject_id,
        parent_run_id=parent.id,
        pause_after_stages=[],
        renarrate={
            "spine_stage": "promo_storyboard",
            "element_id": "s2",
            "narration_text": "Two, but better.",
            "voice": None,
        },
    )

    assert dispatched == [child.id]
    assert fake_tts.synthesize.await_count == 1
    rows = {
        so.stage_name: so for so in await StageOutputRepository(db_session).list_for_run(child.id)
    }
    # Every stage before the compile cut is seeded completed on the child; only the
    # compile stage is pending, so the DAG re-runs just the re-stitch.
    assert rows["promo_narrate"].status == "completed"
    assert rows["promo_render"].status == "pending"
    # ``StageOutput.output`` is nullable; a seeded-completed stage always carries one,
    # so narrow it explicitly rather than subscripting an optional.
    narrate_output = rows["promo_narrate"].output
    script_output = rows["promo_script"].output
    assert narrate_output is not None
    assert script_output is not None
    child_clips = narrate_output["clips"]
    assert child_clips[0] == CLIPS[0]
    assert child_clips[1]["id"] != CLIPS[1]["id"]
    assert script_output["lines"][1]["text"] == "Two, but better."


async def test_renarrate_binds_the_parent_runs_pinned_slots_and_correlation(
    db_session, fake_storage: _FakeStorage, fake_tts: AsyncMock
) -> None:
    """The re-narrate synth runs in REQUEST context: it must act as the parent run.

    Without the bind ``current_run_slot("tts")`` is ``None`` there, so a run that
    pinned a tenant TTS connection got ONE platform clip spliced into its narration,
    and its usage carried the request's correlation id instead of the run's. Every
    binding is restored afterwards so the rest of the request is untouched.
    """
    from w2v_common.auth.identity_context import (
        get_correlation_id_from_context,
        get_stage_from_context,
        get_w2v_project_id_from_context,
        set_correlation_context,
    )

    from src.auth.run_providers import current_run_slot

    project, _, parent = await _seed_promo_parent(db_session)
    connection_id = uuid7()
    parent.config_snapshot = {
        **parent.config_snapshot,
        "provider_slots": {
            "tts": {
                "source": "tenant",
                "kind": "elevenlabs",
                "connection_id": str(connection_id),
                "version": 3,
            }
        },
    }
    seen: dict[str, object] = {}

    async def _capture(**_kwargs: object) -> TTSSynthesis:
        seen["slot"] = current_run_slot("tts")
        seen["correlation"] = get_correlation_id_from_context()
        seen["stage"] = get_stage_from_context()
        seen["project"] = get_w2v_project_id_from_context()
        return TTSSynthesis(
            audio_bytes=b"\x01\x02\x03",
            sample_rate=0,
            duration_seconds=6.5,
            provider="",
            voice="",
        )

    fake_tts.synthesize.side_effect = _capture
    set_correlation_context("request-correlation", None)

    await build_renarrate_state(
        session=db_session,
        parent_run=parent,
        blueprint=get_registry().resolve("promo"),
        spine_stage="promo_storyboard",
        element_id="s2",
        narration_text="Two, but better.",
    )

    slot = seen["slot"]
    assert slot is not None
    assert getattr(slot, "source", None) == "tenant"
    assert getattr(slot, "connection_id", None) == connection_id
    assert getattr(slot, "version", None) == 3
    assert seen["correlation"] == str(parent.id)
    assert seen["stage"] == "promo_narrate"
    assert seen["project"] == str(project.id)
    # Restored: the request's own context is untouched after the synth.
    assert current_run_slot("tts") is None
    assert get_correlation_id_from_context() == "request-correlation"
    assert get_stage_from_context() == ""
    assert get_w2v_project_id_from_context() == ""
    set_correlation_context(None, None)
