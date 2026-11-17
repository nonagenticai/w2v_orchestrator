"""A music edit on a beat-cut reel must change the music the reel is CUT to.

The defect: a reviewer disliked an ``action_reel``'s bed, submitted a music edit,
and got a 201 plus a green child run with the identical soundtrack. Two reasons,
and fixing either one alone still ships a lie:

1. The override was written to ``config_snapshot["sound_effects"]``. No
   ``action_reel`` stage reads that key — ``ActionReelCompileStage`` overrides
   ``build_audio_bed``, so the shared ``sound_design.resolve`` path that consumes
   ``sound_effects`` for every other deliverable type is bypassed entirely.
2. The re-stitch cut at the blueprint's ``compile_cut_stage`` (``cut_plan``),
   which PRESERVES ``music_select`` — i.e. the old track's beat grid. Swapping the
   bed there would render a reel cut to a track it does not play, with every stage
   green and every shot stamped ``beat_locked``.

So the child run rewinds to ``music_select``. These tests pin both halves, plus
the regression that matters most: a non-reel deliverable's ``sound_effects`` path
is untouched.
"""

from __future__ import annotations

from uuid import UUID

import pytest
from sqlalchemy import select

from src.database.models import Run
from src.database.repositories.run_repository import RunRepository
from src.database.repositories.stage_output_repository import StageOutputRepository

pytestmark = pytest.mark.asyncio

# Everything strictly upstream of the grid owner on `action_reel`. These are the
# expensive stages (a KAG fan-out, a per-clip analyzer pass over 4K originals) and
# a music tweak must never re-bill them.
_FOOTAGE_BRANCH = ("source_ingest", "footage_probe", "shot_analysis", "segment_select")


async def _make_reel_project(client) -> str:
    r = await client.post(
        "/api/projects",
        json={
            "type": "action_reel",
            "title": "kite reel",
            "brief": {"title": "kite session", "tone": "high energy"},
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


async def _completed_reel_run(client, db_session) -> str:
    """A finished ``action_reel`` run with every stage's output recorded."""
    deliv = await _make_reel_project(client)
    run_repo = RunRepository(db_session)
    run = await run_repo.create(
        deliverable_id=UUID(deliv),
        blueprint_id="action_reel_v1",
        config_snapshot={"brief": {"title": "kite session"}},
        pause_after_stages=[],
    )
    so_repo = StageOutputRepository(db_session)
    outputs = {
        "source_ingest": {"grounding_text": "kite"},
        "footage_probe": {"sources": [{"id": "src-1", "original_url": "https://x/a.mp4"}]},
        "shot_analysis": {"segments": [{"id": "seg-1"}]},
        "segment_select": {"segments": [{"id": "seg-1", "source_id": "src-1"}]},
        "music_select": {"track_id": "corporate_uplift", "track_url": "u", "beats": [0.5, 1.0]},
        "cut_plan": {"shots": [{"id": "s1"}], "music_track_id": "corporate_uplift"},
        "graphics_design": {"cards": []},
        "narrate": {"clips": []},
        "compile": {"artifact_url": "s3://reels/a.mp4"},
        "render_manifest": {"manifest": {}},
    }
    for stage, output in outputs.items():
        await so_repo.upsert_pending(run.id, stage)
        await db_session.commit()
        await so_repo.mark_completed(run.id, stage, output)
        await db_session.commit()
    await run_repo.mark_completed(run.id)
    await db_session.commit()
    return str(run.id)


async def _child_of(db_session, response) -> Run:
    return (
        await db_session.execute(select(Run).where(Run.id == UUID(response.json()["id"])))
    ).scalar_one()


async def test_a_reel_music_edit_lands_where_music_select_reads_it(client, db_session) -> None:
    """The chosen track must reach ``brief``-equivalent pin territory — the top-level
    ``music`` block ``music_select._resolve_track`` reads — and NOT ``sound_effects``,
    which nothing in ``stages/action_reel/`` reads."""
    rid = await _completed_reel_run(client, db_session)

    r = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={"stage_name": "cut_plan", "edits": {}, "music": {"track": "jazz_lounge"}},
    )

    assert r.status_code == 201, r.text
    child = await _child_of(db_session, r)
    assert child.config_snapshot["music"]["track_id"] == "jazz_lounge"
    assert child.config_snapshot["music"]["enabled"] is True
    assert "sound_effects" not in child.config_snapshot, (
        "sound_effects is the sink NO action_reel stage reads — writing there is the "
        "silent no-op this fix exists for"
    )


async def test_a_reel_music_edit_rewinds_far_enough_to_re_derive_the_beat_grid(
    client, db_session
) -> None:
    """THE assertion. The child must NOT inherit the parent's ``music_select`` or
    ``cut_plan`` output: the grid has to be re-fetched for the new bed and every
    shot boundary re-snapped to it. It must still inherit the whole footage branch,
    or a music tweak silently re-bills a KAG crawl and an analyzer pass."""
    rid = await _completed_reel_run(client, db_session)

    r = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={"stage_name": "cut_plan", "edits": {}, "music": {"track": "jazz_lounge"}},
    )

    assert r.status_code == 201, r.text
    child = await _child_of(db_session, r)
    preserved = {
        so.stage_name
        for so in await StageOutputRepository(db_session).list_for_run(child.id)
        if so.status == "completed"
    }
    assert "music_select" not in preserved, (
        "preserving the grid owner is exactly the bug: the child would cut to the "
        "OLD track's beats while playing the new one"
    )
    assert "cut_plan" not in preserved, "the cut must be re-snapped to the new grid"
    assert "compile" not in preserved
    assert set(_FOOTAGE_BRANCH) <= preserved, (
        "the expensive footage branch must be reused, not re-billed"
    )


async def test_turning_a_reels_music_off_is_carried_as_an_explicit_off(client, db_session) -> None:
    """A falsy track is a request, not a typo. It rides as an explicit
    ``enabled: False`` so ``music_select`` can tell it apart from "no opinion" —
    which is what makes the silent reel honoured AND noted rather than ignored."""
    rid = await _completed_reel_run(client, db_session)

    r = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={"stage_name": "cut_plan", "edits": {}, "music": {"track": None}},
    )

    assert r.status_code == 201, r.text
    child = await _child_of(db_session, r)
    assert child.config_snapshot["music"] == {"enabled": False}


async def test_the_mix_knobs_ride_along_with_the_track(client, db_session) -> None:
    """``gain_db``/fades are honoured by ``compile``; ``duck_db`` cannot be and is
    reported there as an ignored field. Either way it must SURVIVE the route —
    dropping it here would make the report impossible."""
    rid = await _completed_reel_run(client, db_session)

    r = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={
            "stage_name": "cut_plan",
            "edits": {},
            "music": {
                "track": "jazz_lounge",
                "gain_db": -8.0,
                "fade_in_s": 0.5,
                "fade_out_s": 3.0,
                "duck_db": -4.0,
            },
        },
    )

    assert r.status_code == 201, r.text
    child = await _child_of(db_session, r)
    assert child.config_snapshot["music"] == {
        "enabled": True,
        "track_id": "jazz_lounge",
        "gain_db": -8.0,
        "fade_in_s": 0.5,
        "fade_out_s": 3.0,
        "duck_db": -4.0,
    }


async def test_a_reel_music_edit_carrying_clip_edits_is_refused_not_half_applied(
    client, db_session
) -> None:
    """``action_reel`` has no clip-spine stage at all, so the clip half of such a
    request cannot be honoured. Applying the music half silently would be the same
    class of lie as the no-op."""
    rid = await _completed_reel_run(client, db_session)

    r = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={
            "stage_name": "cut_plan",
            "edits": {"s1": {"clip_lead_seconds": 1.0}},
            "music": {"track": "jazz_lounge"},
        },
    )

    assert r.status_code == 422, r.text
    assert "per-clip timeline" in r.text


async def test_a_reel_run_with_no_music_change_still_takes_the_spine_gate(
    client, db_session
) -> None:
    """The rewind is opt-in on the presence of a music change. Without one,
    ``action_reel`` has no clip spine and the request is refused exactly as before."""
    rid = await _completed_reel_run(client, db_session)

    r = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={"stage_name": "cut_plan", "edits": {"s1": {"clip_lead_seconds": 1.0}}},
    )

    assert r.status_code == 422, r.text
    assert "does not carry a clip timeline" in r.text


async def test_a_non_reel_music_edit_still_uses_sound_effects_and_the_compile_cut(
    client, db_session
) -> None:
    """THE regression that matters most. ``sound_effects`` is the CORRECT sink for
    every deliverable type whose timeline does not depend on the bed — the shared
    ``sound_design.resolve`` consumes it — and a compile-cut re-stitch is the right,
    cheap answer there. Nothing about the reel path may leak into it: no ``music``
    block, and the spine's own stage output still preserved."""
    from tests.api.routes.test_runs import _make_project, _seed_stage_with_elements

    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(db_session, deliv, "storyboard_draft", [{"id": "s1"}])
    await RunRepository(db_session).mark_completed(UUID(rid))
    await db_session.commit()

    r = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={
            "stage_name": "storyboard_draft",
            "edits": {},
            "music": {"track": "jazz_lounge", "gain_db": -9.0, "duck_db": -5.0},
        },
    )

    assert r.status_code == 201, r.text
    child = await _child_of(db_session, r)
    sound = child.config_snapshot["sound_effects"]
    assert sound["music"] == "jazz_lounge"
    assert sound["enabled"] is True
    assert sound["music_gain_db"] == -9.0
    assert sound["duck_db"] == -5.0
    assert "music" not in child.config_snapshot, (
        "the reel-only override block must never appear on a deliverable whose "
        "audio bed comes from the shared sound_design resolver"
    )
    preserved = {
        so.stage_name
        for so in await StageOutputRepository(db_session).list_for_run(child.id)
        if so.status == "completed"
    }
    assert "storyboard_draft" in preserved, (
        "the ordinary path preserves every stage before the compile cut — a music "
        "tweak here must not re-run the spine"
    )


async def test_a_non_reel_music_off_is_an_explicit_off_in_sound_effects(client, db_session) -> None:
    """Music-off on a non-reel deliverable turns the bed OFF in ``sound_effects`` with
    an explicit ``music: False`` (a dropped key means "auto" and re-picks a bed), and
    leaves transitions / ambient alone — never the reel's top-level ``music`` block."""
    from tests.api.routes.test_runs import _make_project, _seed_stage_with_elements

    _, deliv = await _make_project(client)
    rid = await _seed_stage_with_elements(db_session, deliv, "storyboard_draft", [{"id": "s1"}])
    run = (await db_session.execute(select(Run).where(Run.id == UUID(rid)))).scalar_one()
    run.config_snapshot = {"sound_effects": {"enabled": True, "transitions": True, "music": "old"}}
    await RunRepository(db_session).mark_completed(UUID(rid))
    await db_session.commit()

    r = await client.post(
        f"/api/runs/{rid}/apply-edits",
        json={"stage_name": "storyboard_draft", "edits": {}, "music": {"track": ""}},
    )

    assert r.status_code == 201, r.text
    child = await _child_of(db_session, r)
    assert child.config_snapshot["sound_effects"] == {
        "enabled": True,
        "transitions": True,
        "music": False,
    }
    assert "music" not in child.config_snapshot
