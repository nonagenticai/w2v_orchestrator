"""Unit tests for ``resolve_audio_bed``'s ORCH-4.1 T1 music-supervisor fallback.

An explicit brief-authored ``music``/``ambient`` id must still WIN outright;
the deterministic ``music_supervisor`` (see ``test_music_supervisor.py`` for
its own unit tests) only fires when the brief named no track at all — a run
that opts sound design on but doesn't hand-pick anything must no longer ship
silent (06-sound-music.md §8 acceptance #1).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.shared.sound_design import resolve_audio_bed
from src.services.stages.sound_catalog import SoundLibrary


def _use_builtin_library(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake(*_args: object, **_kwargs: object) -> SoundLibrary:
        return SoundLibrary.builtin()

    monkeypatch.setattr("src.services.stages.shared.sound_design.load_sound_library", _fake)


def _fake_presign(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _presign(sfx_id: str) -> str:
        return f"https://minio.test/w2v-orchestrator/sound_library/shared/{sfx_id}.mp3?sig=x"

    monkeypatch.setattr("src.services.stages.shared.sound_design.presign_sound_asset", _presign)


def _ctx(cfg: dict) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"sound_effects": cfg},
        prior_run_outputs={},
    )


async def test_explicit_music_wins_over_supervisor_pick(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    # mood="upbeat" would make the supervisor pick corporate_uplift; the
    # authored id must win regardless.
    cfg = {"enabled": True, "music": "jazz_lounge", "mood": "upbeat"}
    bed = await resolve_audio_bed(_ctx(cfg), [{"seconds": 5.0}], cfg)
    assert bed is not None
    assert bed["music"]["url"].endswith("jazz_lounge.mp3?sig=x")


async def test_explicit_ambient_wins_over_supervisor_pick(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    # mood="calm" has no ambient table entry -> supervisor default is
    # room_tone; the authored office_hum id must win instead.
    cfg = {"enabled": True, "ambient": "office_hum", "mood": "calm"}
    bed = await resolve_audio_bed(_ctx(cfg), [{"seconds": 5.0}], cfg)
    assert bed is not None
    assert bed["ambient"][0]["url"].endswith("office_hum.mp3?sig=x")


async def test_no_named_music_falls_back_to_supervisor_pick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    cfg = {"enabled": True}
    bed = await resolve_audio_bed(_ctx(cfg), [{"seconds": 5.0}], cfg)
    assert bed is not None
    # module default: corporate_uplift, with the supervisor's default offset.
    assert bed["music"]["url"].endswith("corporate_uplift.mp3?sig=x")
    assert bed["music"]["music_offset_lu"] == 18.0


async def test_no_named_ambient_falls_back_to_supervisor_pick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    cfg = {"enabled": True}
    bed = await resolve_audio_bed(_ctx(cfg), [{"seconds": 5.0}], cfg)
    assert bed is not None
    assert bed["ambient"][0]["url"].endswith("room_tone.mp3?sig=x")
    assert bed["ambient"][0]["ambient_offset_lu"] == 30.0


async def test_music_true_resolves_to_default_bed_not_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BUG #2 regression: `music: true` (the boolean opt-in-default) must resolve
    to the supervisor's default music id, mirroring `ambient: true`.

    The pre-fix code did `music_id = ... else music_choice`, so a bare `True`
    became `music_id = True` ⇒ `is_valid("True")` fails ⇒ NO music bed (silence).
    Reverting the fix makes this fail: `bed["music"]` would be absent.
    """
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    cfg = {"enabled": True, "music": True}
    bed = await resolve_audio_bed(_ctx(cfg), [{"seconds": 5.0}], cfg)
    assert bed is not None
    assert "music" in bed
    # No mood ⇒ supervisor's default music id.
    assert bed["music"]["url"].endswith("corporate_uplift.mp3?sig=x")


async def test_music_false_is_not_treated_as_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    """`music: false` means "no bed" — NOT "pick one for me".

    With showcase now defaulting sound ON, this is the only way to ask for
    ambience without a music bed. A truthy-`or` fallback would silently hand
    that caller the supervisor's pick — the exact opposite of what they asked.
    """
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    cfg = {"enabled": True, "music": False, "ambient": True, "mood": "upbeat"}
    bed = await resolve_audio_bed(_ctx(cfg), [{"seconds": 5.0}], cfg)
    assert bed is not None
    assert "music" not in bed
    assert "ambient" in bed


async def test_ambient_false_is_not_treated_as_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    cfg = {"enabled": True, "music": "jazz_lounge", "ambient": False}
    bed = await resolve_audio_bed(_ctx(cfg), [{"seconds": 5.0}], cfg)
    assert bed is not None
    assert "ambient" not in bed


async def test_explicit_offset_override_flows_through_to_bed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    cfg = {"enabled": True, "music_offset_lu": 9.0, "ambient_offset_lu": 21.0}
    bed = await resolve_audio_bed(_ctx(cfg), [{"seconds": 5.0}], cfg)
    assert bed is not None
    assert bed["music"]["music_offset_lu"] == 9.0
    assert bed["ambient"][0]["ambient_offset_lu"] == 21.0


async def test_duck_open_lu_below_vo_reaches_both_music_and_ambient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The VO-relative duck key must land on BOTH cues — a payload where music
    # ducks VO-relative but ambient still ducks against the absolute threshold
    # would be worse than neither.
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    cfg = {"enabled": True}
    bed = await resolve_audio_bed(_ctx(cfg), [{"seconds": 5.0}], cfg)
    assert bed is not None
    assert bed["music"]["duck_open_lu_below_vo"] == 12.46
    assert bed["ambient"][0]["duck_open_lu_below_vo"] == 12.46


async def test_explicit_duck_open_override_flows_through_to_both_cues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    cfg = {"enabled": True, "duck_open_lu_below_vo": 15.0}
    bed = await resolve_audio_bed(_ctx(cfg), [{"seconds": 5.0}], cfg)
    assert bed is not None
    assert bed["music"]["duck_open_lu_below_vo"] == 15.0
    assert bed["ambient"][0]["duck_open_lu_below_vo"] == 15.0


def _ctx_unproven_vo(cfg: dict) -> RunContext:
    """A run where an earlier stage recorded AUDIO_UNNORMALIZED — at least one
    narration clip shipped at an unproven level."""
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"sound_effects": cfg},
        prior_run_outputs={
            "demo_video_narrate": {
                "clips": [],
                "degradations": [
                    {"code": "audio_unnormalized", "detail": "clip stored at an unknown level"}
                ],
            },
        },
    )


async def test_unproven_vo_level_omits_computed_offsets_and_duck_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The music/ambient offsets and the VO-relative duck key are back-solved
    from an ASSUMED -18 LUFS voiceover. When a prior stage recorded
    AUDIO_UNNORMALIZED that premise does not hold for this run, so the keys
    must be OMITTED entirely (never sent as 0.0/None) so the compiler falls
    back to measuring the real VO bus."""
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    cfg = {"enabled": True}
    bed = await resolve_audio_bed(_ctx_unproven_vo(cfg), [{"seconds": 5.0}], cfg)
    assert bed is not None
    assert "music_offset_lu" not in bed["music"]
    assert "duck_open_lu_below_vo" not in bed["music"]
    assert "ambient_offset_lu" not in bed["ambient"][0]
    assert "duck_open_lu_below_vo" not in bed["ambient"][0]
    # The music/ambient cues themselves still ship — this is a leveling
    # omission, not a dropped bed.
    assert "url" in bed["music"]
    assert "url" in bed["ambient"][0]


async def test_unproven_vo_level_does_not_override_an_explicit_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An author-set offset still ships even when the run's VO level could not
    be proven — the omission only ever applies to the COMPUTED default."""
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    cfg = {
        "enabled": True,
        "music_offset_lu": 9.0,
        "duck_open_lu_below_vo": 15.0,
    }
    bed = await resolve_audio_bed(_ctx_unproven_vo(cfg), [{"seconds": 5.0}], cfg)
    assert bed is not None
    assert bed["music"]["music_offset_lu"] == 9.0
    assert bed["music"]["duck_open_lu_below_vo"] == 15.0


async def test_a_proven_run_is_unaffected_by_the_unproven_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: a run with no AUDIO_UNNORMALIZED degradation anywhere
    keeps shipping the computed offsets exactly as before."""
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    cfg = {"enabled": True}
    bed = await resolve_audio_bed(_ctx(cfg), [{"seconds": 5.0}], cfg)
    assert bed is not None
    assert bed["music"]["music_offset_lu"] == 18.0
    assert bed["music"]["duck_open_lu_below_vo"] == 12.46


async def test_disabled_returns_none_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_builtin_library(monkeypatch)
    _fake_presign(monkeypatch)
    bed = await resolve_audio_bed(_ctx({}), [{"seconds": 5.0}], {})
    assert bed is None


async def test_pinned_tenant_beds_retired_mid_run_still_play(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P6 ORCH-4: the snapshot pinned tenant music + ambient ids; the tenant deleted
    both after the run started (RETIRED). The run's bed must not be dropped."""
    from src.services.stages.sound_catalog import BUILTIN_CATALOG, SoundAsset

    def _retired(cat: str) -> SoundAsset:
        return SoundAsset(
            key=f"sound_library/tenants/t1/{cat}/x.mp3",
            description="x",
            category=cat,  # ty: ignore[invalid-argument-type]  # why: literal per call
            duration_s=30.0,
            source="tenant",
            retired=True,
        )

    async def _fake(*_args: object, **_kwargs: object) -> SoundLibrary:
        return SoundLibrary(
            {**BUILTIN_CATALOG, "tenant_bed": _retired("music"), "tenant_amb": _retired("ambient")}
        )

    monkeypatch.setattr("src.services.stages.shared.sound_design.load_sound_library", _fake)
    _fake_presign(monkeypatch)
    cfg = {"enabled": True, "music": "tenant_bed", "ambient": "tenant_amb"}
    bed = await resolve_audio_bed(_ctx(cfg), [{"seconds": 5.0}], cfg)
    assert bed is not None
    assert bed["music"]["url"].endswith("tenant_bed.mp3?sig=x")
    assert bed["ambient"][0]["url"].endswith("tenant_amb.mp3?sig=x")
