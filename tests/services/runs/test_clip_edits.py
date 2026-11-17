"""DB-backed tests for per-clip edit re-stitch (studio timeline).

A clip edit spawns a child run that preserves every stage before the blueprint's
compile cut (``terminal_artifact_stage``) with each edited clip's override fields
merged onto the spine element, and re-runs only the compile stage onward —
reusing every generated clip asset (exactly the reorder story with a field-merge
instead of a permutation). These tests drive ``build_edit_state`` directly and
end-to-end through ``RunEngine.start_run(edit=...)`` against real postgres.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from src.api.schemas.runs import ApplyClipEdits
from src.database.models import Deliverable, Project, Run, StageOutput
from src.services.blueprints.registry import get_registry
from src.services.run_engine import RunEngine
from src.services.runs.elements import patch_elements_in_output
from src.services.runs.regenerate import _EDITABLE_CLIP_FIELDS, EditError, build_edit_state
from src.services.stages.shot_assembly import assemble_shots

pytestmark = pytest.mark.asyncio

GENERIC_STAGES = (
    "brief_intake",
    "script_draft",
    "storyboard_draft",
    "narrate",
    "video_render",
    "ai_image_gen",
    "ai_video_gen",
    "motion_graphics_gen",
    "source_ingest",
    "media_discovery",
)
SHOTS = [
    {"id": "s1", "seconds": 3, "description": "one"},
    {"id": "s2", "seconds": 4, "description": "two"},
    {"id": "s3", "seconds": 5, "description": "three", "clip_lead_seconds": 2.0},
]


async def _seed_parent(db_session) -> tuple[Project, Deliverable, Run]:
    tenant_id, subject_id = uuid4(), uuid4()
    project = Project(
        id=uuid4(),
        tenant_id=tenant_id,
        owner_subject_id=subject_id,
        visibility="tenant",
        type="generic",
        title="edit-parent",
        brief={},
        brand={},
        voice={},
        theme={},
    )
    deliverable = Deliverable(
        id=uuid4(),
        project_id=project.id,
        name="d",
        format="16x9",
        captions_mode="none",
        target_duration_seconds=30,
    )
    parent = Run(
        id=uuid4(),
        deliverable_id=deliverable.id,
        blueprint_id="generic_v1",
        status="completed",
        config_snapshot={
            "project_id": str(project.id),
            "tenant_id": str(tenant_id),
            "subject_id": str(subject_id),
            "title": project.title,
            "type": "generic",
            "brief": {},
            "brand": {},
            "voice": {},
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
    sos = []
    for name in GENERIC_STAGES:
        output = (
            {"shots": [dict(s) for s in SHOTS]} if name == "storyboard_draft" else {"stage": name}
        )
        sos.append(
            StageOutput(
                id=uuid4(), run_id=parent.id, stage_name=name, status="completed", output=output
            )
        )
    db_session.add_all([project, deliverable, parent, *sos])
    await db_session.commit()
    return project, deliverable, parent


async def test_build_edit_state_preserves_all_but_compile_and_patches_spine(db_session) -> None:
    _, _, parent = await _seed_parent(db_session)
    blueprint = get_registry().resolve("generic")

    preserved = await build_edit_state(
        session=db_session,
        parent_run=parent,
        blueprint=blueprint,
        spine_stage="storyboard_draft",
        edits={
            "s1": {"clip_lead_seconds": 1.5, "clip_anchor": "start"},
            "s2": {"caption": "hello world", "fit": "cover"},
            "s3": {"clip_lead_seconds": None},  # clear the seeded override
        },
    )

    # video_render is the compile cut → NOT preserved (child re-runs it).
    assert "video_render" not in preserved
    assert set(preserved) == set(GENERIC_STAGES) - {"video_render"}
    shots = {s["id"]: s for s in preserved["storyboard_draft"]["shots"]}
    # Edited clip carries the overrides; the caption/fit edit lands on s2;
    # the null clears the seeded lead on s3.
    assert shots["s1"]["clip_lead_seconds"] == 1.5
    assert shots["s1"]["clip_anchor"] == "start"
    assert shots["s2"]["caption"] == "hello world"
    assert shots["s2"]["fit"] == "cover"
    assert "clip_lead_seconds" not in shots["s3"]
    # A non-spine stage is preserved untouched.
    assert preserved["script_draft"] == {"stage": "script_draft"}


async def test_build_edit_state_patches_speed_and_transition_onto_spine(db_session) -> None:
    # WS-2: per-clip playback speed + entering-transition merge onto the spine shot
    # exactly like caption/fit — the same shape the route emits via
    # `ClipEdit.model_dump(exclude_unset=True)` (transition is a `{type, duration}`
    # dict once dumped).
    _, _, parent = await _seed_parent(db_session)
    blueprint = get_registry().resolve("generic")

    preserved = await build_edit_state(
        session=db_session,
        parent_run=parent,
        blueprint=blueprint,
        spine_stage="storyboard_draft",
        edits={
            "s1": {"speed": 1.5},
            "s2": {"transition": {"type": "fade", "duration": 0.75}},
        },
    )

    shots = {s["id"]: s for s in preserved["storyboard_draft"]["shots"]}
    # (a) speed lands on the edited clip.
    assert shots["s1"]["speed"] == 1.5
    # (b) transition lands as the compositor's `{type, duration}` dict.
    assert shots["s2"]["transition"] == {"type": "fade", "duration": 0.75}
    # Unedited clips carry neither key (additive / byte-identical default).
    assert "speed" not in shots["s3"]
    assert "transition" not in shots["s1"]


async def test_build_edit_state_patches_look_grade_and_overlays_onto_spine(db_session) -> None:
    # Per-clip colour grade + text overlays merge onto the spine shot exactly like
    # caption/fit/speed — the same shape the route emits via
    # `ClipEdit.model_dump(exclude_unset=True)` (adjust is a param dict; overlays a
    # list of overlay dicts once dumped).
    _, _, parent = await _seed_parent(db_session)
    blueprint = get_registry().resolve("generic")

    adjust = {
        "brightness": 0.2,
        "contrast": 1.1,
        "saturation": 1.3,
        "gamma": 1.0,
        "preset": "vivid",
    }
    overlays = [{"text": "Intro", "position": "top-left", "font_size": 60, "color": "black"}]
    preserved = await build_edit_state(
        session=db_session,
        parent_run=parent,
        blueprint=blueprint,
        spine_stage="storyboard_draft",
        edits={
            "s1": {"adjust": adjust},
            "s2": {"overlays": overlays},
        },
    )

    shots = {s["id"]: s for s in preserved["storyboard_draft"]["shots"]}
    # (a) grade lands on the edited clip.
    assert shots["s1"]["adjust"] == adjust
    # (b) overlays land as the compiler's list of overlay dicts.
    assert shots["s2"]["overlays"] == overlays
    # Unedited clips carry neither key (additive / byte-identical default).
    assert "adjust" not in shots["s3"]
    assert "overlays" not in shots["s1"]


async def test_build_edit_state_patches_audio_onto_spine(db_session) -> None:
    # Per-clip audio-level edit (gain/mute/fade) merges onto the spine shot exactly
    # like caption/fit/speed/adjust — the same shape the route emits via
    # `ClipEdit.model_dump(exclude_unset=True)` (a params dict once dumped).
    _, _, parent = await _seed_parent(db_session)
    blueprint = get_registry().resolve("generic")

    audio = {"gain_db": -20.0, "mute": False, "fade_in_s": 1.0, "fade_out_s": 0.0}
    preserved = await build_edit_state(
        session=db_session,
        parent_run=parent,
        blueprint=blueprint,
        spine_stage="storyboard_draft",
        edits={"s1": {"audio": audio}},
    )

    shots = {s["id"]: s for s in preserved["storyboard_draft"]["shots"]}
    # (a) audio-level edit lands on the edited clip.
    assert shots["s1"]["audio"] == audio
    # Unedited clips carry no audio key (additive / byte-identical default).
    assert "audio" not in shots["s2"]
    assert "audio" not in shots["s3"]


async def test_build_edit_state_patches_voiceover_onto_spine(db_session) -> None:
    # THE WHITELIST regression: `audio_url` / `audio_seconds` must be in
    # `_EDITABLE_CLIP_FIELDS`, or `patch_elements_in_output` SILENTLY DROPS them and
    # the uploaded voiceover never reaches the spine (the clip keeps its TTS take
    # and the upload is a no-op with a 200). Delete either entry ⇒ this test fails.
    _, _, parent = await _seed_parent(db_session)
    blueprint = get_registry().resolve("generic")

    vo_url = "http://minio.test/w2v-orchestrator/artifacts/voiceovers/t/vo/take.mp3"
    preserved = await build_edit_state(
        session=db_session,
        parent_run=parent,
        blueprint=blueprint,
        spine_stage="storyboard_draft",
        edits={"s1": {"audio_url": vo_url, "audio_seconds": 12.4}},
    )

    shots = {s["id"]: s for s in preserved["storyboard_draft"]["shots"]}
    assert shots["s1"]["audio_url"] == vo_url
    assert shots["s1"]["audio_seconds"] == 12.4
    # Unedited clips carry neither key (additive / byte-identical default).
    assert "audio_url" not in shots["s2"]
    assert "audio_seconds" not in shots["s3"]


async def test_null_audio_url_clears_the_voiceover_back_to_tts() -> None:
    # `null` in a patch is a field DELETE (elements.py), so clearing an uploaded
    # voiceover needs no special-casing: the shot loses the key and `assemble_shots`
    # falls back to the narration clip.
    out = {
        "shots": [{"id": "s1", "line_id": "l1", "seconds": 3.0, "audio_url": "http://vo/take.mp3"}]
    }
    new = patch_elements_in_output(
        out,
        "storyboard_draft",
        {"s1": {"audio_url": None, "audio_seconds": None}},
        allowed_fields=_EDITABLE_CLIP_FIELDS,
    )
    assert new is not None
    assert "audio_url" not in new["shots"][0]
    built = assemble_shots(
        storyboard_shots=new["shots"],
        images=[],
        narration_clips=[
            {"line_id": "l1", "artifact_url": "http://a/tts.wav", "seconds": 5.0},
        ],
    )[0]
    # Back to the TTS narration — url AND its measured duration.
    assert built["audio_url"] == "http://a/tts.wav"
    assert built["seconds"] == 5.0


async def test_apply_clip_edits_voiceover_round_trips_schema_into_shot_payload() -> None:
    # An ApplyClipEdits body carrying an uploaded voiceover round-trips through the
    # schema (snake_case spine keys via `model_dump(exclude_unset=True)`) and the
    # dumped keys are exactly the spine-shot keys `assemble_shots` reads — so the
    # uploaded take REPLACES the TTS narration and declares the shot's length.
    body = ApplyClipEdits.model_validate(
        {
            "stage_name": "storyboard_draft",
            "edits": {"s1": {"audio_url": "http://vo/take.mp3", "audio_seconds": 12.4}},
        }
    )
    dumped = {
        clip_id: edit.model_dump(exclude_unset=True, by_alias=False)
        for clip_id, edit in body.edits.items()
    }
    assert dumped["s1"] == {"audio_url": "http://vo/take.mp3", "audio_seconds": 12.4}
    shot = {"id": "s1", "line_id": "l1", "seconds": 3.0, **dumped["s1"]}
    built = assemble_shots(
        storyboard_shots=[shot],
        images=[],
        narration_clips=[{"line_id": "l1", "artifact_url": "http://a/tts.wav", "seconds": 3.0}],
    )[0]
    assert built["audio_url"] == "http://vo/take.mp3"
    assert built["seconds"] == 12.4


async def test_apply_clip_edits_look_round_trips_schema_into_shot_payload() -> None:
    # WS-3: an ApplyClipEdits body with adjust+overlays round-trips through the
    # schema and lands on the shot payload. Mirrors the route's dump
    # (`model_dump(exclude_unset=True, by_alias=False)` → snake_case spine keys),
    # then confirms `assemble_shots` attaches both onto the built compositor shot.
    body = ApplyClipEdits.model_validate(
        {
            "stage_name": "storyboard_draft",
            "edits": {
                "s1": {
                    "adjust": {"brightness": -0.1, "preset": "cinematic"},
                    "overlays": [{"text": "Lower third", "position": "bottom-left"}],
                }
            },
        }
    )
    dumped = {
        clip_id: edit.model_dump(exclude_unset=True, by_alias=False)
        for clip_id, edit in body.edits.items()
    }
    # `exclude_unset=True` recurses into nested models AND list items, so only the
    # fields the client actually sent survive — the compiler fills the rest from
    # the same schema defaults, so a partial grade never clobbers unsent knobs.
    assert dumped["s1"] == {
        "adjust": {"brightness": -0.1, "preset": "cinematic"},
        "overlays": [{"text": "Lower third", "position": "bottom-left"}],
    }
    # The dumped keys are the spine-shot keys, so they attach onto the built shot.
    shot = {"id": "s1", "line_id": "l1", "seconds": 3.0, **dumped["s1"]}
    built = assemble_shots(storyboard_shots=[shot], images=[], narration_clips=[])[0]
    assert built["adjust"] == dumped["s1"]["adjust"]
    assert built["overlays"] == dumped["s1"]["overlays"]


async def test_apply_clip_edits_audio_round_trips_schema_into_shot_payload() -> None:
    # WS-O: an ApplyClipEdits body with a per-clip audio-level edit round-trips
    # through the schema and lands on the shot payload. Mirrors the route's dump
    # (`model_dump(exclude_unset=True, by_alias=False)` → snake_case spine keys),
    # then confirms `assemble_shots` attaches it onto the built compositor shot.
    body = ApplyClipEdits.model_validate(
        {
            "stage_name": "storyboard_draft",
            "edits": {
                "s1": {"audio": {"gain_db": -20, "mute": False, "fade_in_s": 1.0}},
            },
        }
    )
    dumped = {
        clip_id: edit.model_dump(exclude_unset=True, by_alias=False)
        for clip_id, edit in body.edits.items()
    }
    # `exclude_unset=True` recurses into the nested model, so only the fields the
    # client actually sent survive — the compiler fills the rest from the same
    # schema defaults, so a partial audio edit never clobbers unsent knobs.
    assert dumped["s1"] == {"audio": {"gain_db": -20, "mute": False, "fade_in_s": 1.0}}
    # The dumped keys are the spine-shot keys, so they attach onto the built shot.
    shot = {"id": "s1", "line_id": "l1", "seconds": 3.0, **dumped["s1"]}
    built = assemble_shots(storyboard_shots=[shot], images=[], narration_clips=[])[0]
    assert built["audio"] == {"gain_db": -20, "mute": False, "fade_in_s": 1.0}
    # A shot with no `audio` key carries none (additive / byte-identical default).
    plain = assemble_shots(
        storyboard_shots=[{"id": "s2", "line_id": "l2", "seconds": 3.0}],
        images=[],
        narration_clips=[],
    )[0]
    assert "audio" not in plain


async def test_speed_survives_narration_duration_reconcile() -> None:
    # (c) Narration-sync guard. `_reconcile_seconds` overrides a shot's duration with
    # its narration clip's length; `speed` must ride alongside it untouched — the two
    # seams are orthogonal (the compositor retimes the visual, it does NOT shorten
    # `seconds`). Documents that a per-clip speed is not clobbered by the
    # narration-duration handling.
    shots = assemble_shots(
        storyboard_shots=[{"id": "s1", "line_id": "l1", "seconds": 3, "speed": 1.5}],
        images=[],
        narration_clips=[
            {"line_id": "l1", "artifact_url": "https://minio.test/a.wav", "seconds": 7.0}
        ],
    )

    built = shots[0]
    # Narration duration (7.0s) wins the shot duration over the storyboard's 3s…
    assert built["seconds"] == 7.0
    # …and the per-clip speed rides alongside it, uncorrupted.
    assert built["speed"] == 1.5
    assert built["audio_url"] == "https://minio.test/a.wav"


async def test_build_edit_state_empty_edits_preserves_spine_for_music_only(db_session) -> None:
    # A music-only re-stitch passes no clip edits: the spine is preserved verbatim
    # and only the compile stage re-runs (against the new config).
    _, _, parent = await _seed_parent(db_session)
    blueprint = get_registry().resolve("generic")

    preserved = await build_edit_state(
        session=db_session,
        parent_run=parent,
        blueprint=blueprint,
        spine_stage="storyboard_draft",
        edits={},
    )
    assert "video_render" not in preserved
    assert set(preserved) == set(GENERIC_STAGES) - {"video_render"}
    # Spine byte-identical to the parent (no patch applied).
    assert preserved["storyboard_draft"]["shots"] == [dict(s) for s in SHOTS]


async def test_build_edit_state_rejects_bad_inputs(db_session) -> None:
    _, _, parent = await _seed_parent(db_session)
    blueprint = get_registry().resolve("generic")

    with pytest.raises(EditError):  # spine == terminal, no downstream compile
        await build_edit_state(
            session=db_session,
            parent_run=parent,
            blueprint=blueprint,
            spine_stage="video_render",
            edits={"s1": {"clip_lead_seconds": 1.0}},
        )
    with pytest.raises(EditError):  # no edited id exists on the timeline
        await build_edit_state(
            session=db_session,
            parent_run=parent,
            blueprint=blueprint,
            spine_stage="storyboard_draft",
            edits={"sX": {"clip_lead_seconds": 1.0}},
        )


async def test_run_engine_edit_restitches_only_compile(db_session, db_session_factory) -> None:
    project, deliverable, parent = await _seed_parent(db_session)
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
        edit={
            "spine_stage": "storyboard_draft",
            "edits": {"s2": {"clip_lead_seconds": 3.0, "clip_anchor": "end"}},
        },
    )

    assert dispatched == [child.id]
    assert child.parent_run_id == parent.id

    child_sos = (
        (await db_session.execute(select(StageOutput).where(StageOutput.run_id == child.id)))
        .scalars()
        .all()
    )
    by_name = {so.stage_name: so for so in child_sos}
    # Only the compile stage is pending; everything upstream is preserved.
    assert by_name["video_render"].status == "pending"
    assert by_name["video_render"].output is None
    assert by_name["storyboard_draft"].status == "completed"
    edited = {s["id"]: s for s in by_name["storyboard_draft"].output["shots"]}
    assert edited["s2"]["clip_lead_seconds"] == 3.0
    assert edited["s2"]["clip_anchor"] == "end"
    assert by_name["narrate"].status == "completed"
    assert by_name["ai_image_gen"].status == "completed"
