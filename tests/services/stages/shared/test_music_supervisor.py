"""Unit tests for the deterministic music supervisor.

No DB, no WM — these are pure functions over a fake/stubbed SoundLibrary.
"""

from __future__ import annotations

from src.services.audio_premaster import TARGET_LUFS
from src.services.stages.shared.music_supervisor import (
    _DEFAULT_DUCK_OPEN_LU_BELOW_VO,
    BedChoice,
    pick_bed,
)
from src.services.stages.shared.provenance import ProvenanceValue
from src.services.stages.sound_catalog import SoundLibrary

# --- mood -> track mapping ---------------------------------------------------


# --- unknown mood -> default --------------------------------------------------


# the module default


# the module default (no mapping entry)


# --- authoritative validation: library beats the table ----------------------


class _StubLibrary:
    """A fake SoundLibrary that rejects every id — proves the table is only
    advisory and the (fake) library is authoritative, per acceptance #7 of
    06-sound-music.md."""

    def is_valid(self, sfx_id: str, *, category: str | None = None) -> bool:
        return False


def test_id_absent_from_library_is_dropped_not_returned() -> None:
    # Neither the mood-matched id nor the category default validates ->
    # the whole category resolves to None rather than a broken presign id.
    choice = pick_bed({}, _StubLibrary(), mood="upbeat")
    assert choice.music_id is None
    assert choice.ambient_id is None


class _OnlyDefaultLibrary:
    """Only the category defaults are valid — the mood-matched id for a
    non-default mood must be dropped and the default substituted."""

    def is_valid(self, sfx_id: str, *, category: str | None = None) -> bool:
        return sfx_id in {"corporate_uplift", "room_tone"}


class _EmptyLibrary:
    """Nothing validates at all -- both categories must resolve to None."""

    def is_valid(self, sfx_id: str, *, category: str | None = None) -> bool:
        return False


def test_default_also_invalid_returns_none_never_a_broken_id() -> None:
    choice = pick_bed({}, _EmptyLibrary(), mood="upbeat")
    assert choice.music_id is None
    assert choice.ambient_id is None


# --- offsets carry provenance -------------------------------------------------


def test_default_offsets_are_provenance_tracked_defaults() -> None:
    choice = pick_bed({}, SoundLibrary.builtin())
    assert isinstance(choice, BedChoice)
    assert isinstance(choice.music_offset_lu, ProvenanceValue)
    assert choice.music_offset_lu.value == 18.0
    assert choice.music_offset_lu.source == "default"
    assert choice.music_offset_lu.computed_by == "music_supervisor"
    assert isinstance(choice.ambient_offset_lu, ProvenanceValue)
    assert choice.ambient_offset_lu.value == 30.0
    assert choice.ambient_offset_lu.source == "default"


# --- VO-relative duck key -----------------------------------------------------


def test_duck_open_lu_below_vo_default_is_provenance_tracked() -> None:
    choice = pick_bed({}, SoundLibrary.builtin())
    assert isinstance(choice.duck_open_lu_below_vo, ProvenanceValue)
    assert choice.duck_open_lu_below_vo.value == 12.46
    assert choice.duck_open_lu_below_vo.source == "default"
    assert choice.duck_open_lu_below_vo.computed_by == "music_supervisor"


def test_duck_open_default_back_solves_the_shipped_absolute_threshold() -> None:
    # Pins the back-solve, NOT an independent derivation: the compiler computes
    # `10 ** ((vo_lufs - open_lu_below_vo) / 20)`, and every VO on this bus is
    # premastered to TARGET_LUFS = -18.0 (src/services/audio_premaster.py). At
    # that nominal point the derived threshold must reproduce the shipped
    # absolute literal `duck_threshold = 0.03` — so activating the VO-relative
    # key is behaviour-preserving at nominal and only changes OFF-nominal VO.
    # Deliberately reads the REAL constants, not literals: a future edit to
    # either the duck default or the premaster target that silently moves
    # nominal ducking must fail here.
    assert TARGET_LUFS == -18.0
    threshold = 10 ** ((TARGET_LUFS - _DEFAULT_DUCK_OPEN_LU_BELOW_VO) / 20)
    assert round(threshold, 4) == 0.03


# --- vo_level_unproven: computed offsets are omitted, overrides are not ------


def test_vo_level_unproven_nulls_the_computed_offsets() -> None:
    """The offsets/duck key are back-solved from an ASSUMED -18 LUFS voiceover.
    When the run's VO level is unproven that assumption does not hold, so the
    COMPUTED value must come back None (the caller omits the key) rather than
    the stale default."""
    choice = pick_bed({}, SoundLibrary.builtin(), vo_level_unproven=True)
    assert choice.music_offset_lu.value is None
    assert choice.ambient_offset_lu.value is None
    assert choice.duck_open_lu_below_vo.value is None
    # Source/computed_by survive for audit even though the value is nulled.
    assert choice.music_offset_lu.source == "default"
    assert choice.music_offset_lu.computed_by == "music_supervisor"


def test_vo_level_unproven_false_is_byte_identical_to_omitting_it() -> None:
    """Default parameter value keeps every existing caller's behaviour unchanged."""
    with_flag = pick_bed({}, SoundLibrary.builtin(), vo_level_unproven=False)
    without_flag = pick_bed({}, SoundLibrary.builtin())
    assert with_flag.music_offset_lu.value == without_flag.music_offset_lu.value == 18.0
    assert with_flag.duck_open_lu_below_vo.value == without_flag.duck_open_lu_below_vo.value
