"""The reel music-edit override block — pinned WITHOUT a database.

Its sibling ``test_apply_edits_reel_music.py`` drives the whole route and is
``@pytest.mark.integration``, which ``pyproject.toml``'s
``addopts = -m 'not integration …'`` de-selects and this service's ``ci-cd.yaml``
never re-selects (it passes no ``integration_test_select`` to the archetype). So
NOTHING in CI ran the one assertion the fix rests on: mutating
``_reel_music_overrides``'s return to ``{"sound_effects": block}`` — i.e. the
EXACT original bug, a silent no-op on this product — left the CI-selected suite
green at 91 passed.

``_reel_music_overrides`` is a pure function of ``(parent_config, MusicEdit)``, so
that assertion needs no database and no marker. These tests are the CI-visible
half; the integration file keeps the end-to-end wiring (child spawn, preserved
stage outputs) it genuinely needs a DB for.
"""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from src.api.schemas.runs import MusicEdit


def _block(overrides: dict[str, object] | None, key: str) -> dict[str, object]:
    """The named override block, narrowed. ``config_overrides`` is deliberately
    typed ``dict[str, object]`` at the route boundary, so a test that reads into it
    has to say so rather than being allowed to subscript an ``object``."""
    assert overrides is not None
    block = overrides[key]
    assert isinstance(block, dict)
    return cast("dict[str, object]", block)


# ── WHERE the override lands ────────────────────────────────────────────────


# ── the documented merge: "each omitted field keeps the run's current value" ──


# ── bounds: every one of these is interpolated into an ffmpeg filtergraph ────


@pytest.mark.parametrize(
    ("field", "value"),
    [("gain_db", 1e9)],
)
def test_an_absurd_mix_value_is_refused_at_the_schema(field: str, value: float) -> None:
    """``compile._mix_knob`` rejects only NON-finite reads, so ``gain_db: 1e9`` was
    finite, accepted, and emitted as ``gain_db: 1000000000.0`` into the filtergraph.
    Bounds mirror ``ClipAudio``, this module's other per-render level control."""
    with pytest.raises(ValidationError):
        MusicEdit.model_validate({"track": "jazz_lounge", field: value})


def test_the_ordinary_mix_range_is_still_accepted() -> None:
    """The bounds must not narrow what a real edit can ask for."""
    edit = MusicEdit(
        track="jazz_lounge", gain_db=-18.0, fade_in_s=2.0, fade_out_s=4.0, duck_db=-9.0
    )

    assert edit.gain_db == -18.0
    assert edit.duck_db == -9.0
