"""Named-role casts: the wizard writes ``speaker="Mark"`` + ``voice.cast={"Mark": …}``.

Before the fix only the ``a``/``b`` slots resolved, so every named line fell back
to ``voice.voice_id`` — a two-person script read by one voice, silently.
"""

from __future__ import annotations

from src.services.stages.primitives.dialogue import dialogue_cast, voice_for_speaker

VOICE = {"voice_id": "af_heart", "cast": {"Mark": "am_adam", " Lisa ": "bf_emma"}}


def test_named_speakers_each_get_their_own_voice() -> None:
    assert voice_for_speaker({"speaker": "Mark"}, VOICE, {}) == "am_adam"
    assert voice_for_speaker({"speaker": "lisa"}, VOICE, {}) == "bf_emma"


def test_unknown_named_speaker_falls_back_to_single_voice() -> None:
    assert voice_for_speaker({"speaker": "Bob"}, VOICE, {}) == "af_heart"


def test_named_cast_needs_two_distinct_catalog_voices() -> None:
    one_voice = {"voice_id": "af_heart", "cast": {"Mark": "am_adam", "Lisa": "am_adam"}}
    assert dialogue_cast(one_voice, {}) is None
    bogus = {"voice_id": "af_heart", "cast": {"Mark": "am_adam", "Lisa": "not_a_voice"}}
    assert dialogue_cast(bogus, {}) is None
    assert voice_for_speaker({"speaker": "Mark"}, bogus, {}) == "af_heart"


def test_explicit_dialogue_disabled_wins_over_named_cast() -> None:
    assert dialogue_cast(VOICE, {"dialogue": {"enabled": False}}) is None


def test_a_b_two_hander_unchanged() -> None:
    brief = {"dialogue": {"enabled": True}}
    assert dialogue_cast({"voice_id": "af_heart"}, brief) == {"a": "af_heart", "b": "am_michael"}
    assert dialogue_cast({"voice_id": "af_heart"}, {}) is None
