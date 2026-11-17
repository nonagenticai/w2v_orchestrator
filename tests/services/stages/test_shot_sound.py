"""resolve_shot_sound_urls — compile-time id→presigned-URL resolution."""

from __future__ import annotations

import pytest

from src.services.stages import shot_sound


@pytest.fixture(autouse=True)
def _fake_presign(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(sfx_id: str) -> str | None:
        if sfx_id == "missing":
            return None
        return f"https://minio.test/w2v-orchestrator/sound_library/shared/{sfx_id}.mp3?sig=x"

    monkeypatch.setattr(shot_sound, "presign_sound_asset", fake)


async def test_transition_id_resolves_to_sound_cue() -> None:
    shots = [{"id": "s1", "transition_sfx_id": "whoosh_soft"}]
    out = await shot_sound.resolve_shot_sound_urls(shots)
    # No planned gain → the neutral -10dB default, the same level accents get.
    # Emitting no gain_db at all rendered every whoosh at FULL level.
    assert out[0]["transition_sfx"] == {
        "url": "https://minio.test/w2v-orchestrator/sound_library/shared/whoosh_soft.mp3?sig=x",
        "gain_db": -10.0,
    }
    # The raw planning keys are consumed (not forwarded on the wire).
    assert "transition_sfx_id" not in out[0]
    assert "transition_sfx_gain_db" not in out[0]


async def test_transition_gain_passes_through_to_sound_cue() -> None:
    shots = [{"id": "s1", "transition_sfx_id": "whoosh_soft", "transition_sfx_gain_db": -4.0}]
    out = await shot_sound.resolve_shot_sound_urls(shots)
    assert out[0]["transition_sfx"] == {
        "url": "https://minio.test/w2v-orchestrator/sound_library/shared/whoosh_soft.mp3?sig=x",
        "gain_db": -4.0,
    }
    assert "transition_sfx_gain_db" not in out[0]


async def test_transition_gain_key_never_leaks_when_url_fails() -> None:
    # The gain is popped UNCONDITIONALLY, so a dropped cue leaves no internal
    # planning key behind for the compiler to choke on.
    shots = [{"id": "s1", "transition_sfx_id": "missing", "transition_sfx_gain_db": -4.0}]
    out = await shot_sound.resolve_shot_sound_urls(shots)
    assert "transition_sfx" not in out[0]
    assert "transition_sfx_gain_db" not in out[0]


async def test_unresolvable_id_is_dropped() -> None:
    shots = [
        {
            "id": "s1",
            "sfx": [{"sfx_id": "missing", "at_seconds": 0.5}],
            "transition_sfx_id": "missing",
        }
    ]
    out = await shot_sound.resolve_shot_sound_urls(shots)
    assert "sfx" not in out[0]
    assert "transition_sfx" not in out[0]


async def test_presign_error_records_drop_reason_in_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A MinIO/sound-library blip drops the SFX bed. Without the notes sink that
    drop was recorded NOWHERE durable — the film shipped silently un-sounded."""

    async def boom(sfx_id: str) -> str | None:
        raise RuntimeError("minio unreachable")

    monkeypatch.setattr(shot_sound, "presign_sound_asset", boom)
    notes: list[str] = []
    shots = [{"id": "s1", "sfx": [{"sfx_id": "ding_positive", "at_seconds": 0.5}]}]
    out = await shot_sound.resolve_shot_sound_urls(shots, notes=notes)

    assert "sfx" not in out[0]  # behaviour unchanged: the cue is still dropped
    assert len(notes) == 1, notes
    note = notes[0]
    assert "ding_positive" in note  # names the cue that vanished
    assert "RuntimeError" in note  # the exception CLASS separates outage from bad id
    assert "minio unreachable" in note


async def test_unresolvable_id_records_drop_reason_in_notes() -> None:
    """`presign_sound_asset` returning None (unknown id / library not seeded) is the
    same fail-open as an exception and must also leave a record."""
    notes: list[str] = []
    shots = [{"id": "s1", "transition_sfx_id": "missing"}]
    out = await shot_sound.resolve_shot_sound_urls(shots, notes=notes)

    assert "transition_sfx" not in out[0]
    assert len(notes) == 1, notes
    assert "missing" in notes[0]
    assert "cue dropped" in notes[0]


async def test_notes_stay_empty_when_every_cue_resolves() -> None:
    """The sink auto-degrades a run, so a clean resolve must add nothing."""
    notes: list[str] = []
    shots = [
        {
            "id": "s1",
            "sfx": [{"sfx_id": "ding_positive", "at_seconds": 1.0}],
            "transition_sfx_id": "whoosh_soft",
        }
    ]
    await shot_sound.resolve_shot_sound_urls(shots, notes=notes)
    assert notes == []


async def test_repeated_failing_id_is_noted_once() -> None:
    """The presign cache dedupes, so one bad id across many shots is ONE note,
    not one per cue — a note per cue would drown the run detail."""
    notes: list[str] = []
    shots = [
        {"id": "s1", "sfx": [{"sfx_id": "missing", "at_seconds": 0.5}]},
        {"id": "s2", "sfx": [{"sfx_id": "missing", "at_seconds": 1.5}]},
        {"id": "s3", "transition_sfx_id": "missing"},
    ]
    await shot_sound.resolve_shot_sound_urls(shots, notes=notes)
    assert len(notes) == 1, notes


async def test_omitted_notes_sink_is_a_noop() -> None:
    """Backwards compatible: the existing call site passes no sink and still
    soft-fails exactly as before."""
    shots = [{"id": "s1", "sfx": [{"sfx_id": "missing", "at_seconds": 0.5}]}]
    out = await shot_sound.resolve_shot_sound_urls(shots)
    assert "sfx" not in out[0]
