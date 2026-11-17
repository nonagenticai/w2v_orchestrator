"""Unit tests for the A0 resolver (intent + brief → PlanDocument).

No DB — pure functions. These import ``PlanDocument`` / ``KnobProvenance`` from
``src.api.schemas.intent`` (the parallel A0 schema PR); they resolve once both
land.
"""

from __future__ import annotations

from typing import Any

from src.services.planning.resolve_plan import resolve_plan


def _intent(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"preset": "default", "duration_seconds": 60}
    base.update(over)
    return base


# --- provenance ladder -----------------------------------------------------


def test_authored_gain_wins_as_override() -> None:
    plan = resolve_plan(
        _intent(),
        brief={"sound_effects": {"music_gain_db": -22}},
    )
    knob = plan.provenance["music_gain_db"]
    assert knob.source == "override"
    assert knob.value == -22
    assert plan.sound_effects["music_gain_db"] == -22


def test_mood_without_author_is_smart_auto() -> None:
    plan = resolve_plan(_intent(mood="energetic"), brief={})
    assert plan.provenance["music_gain_db"].source == "smart-auto"


def test_neutral_without_author_is_default_and_not_emitted() -> None:
    """A neutral, unauthored mix is `default`-sourced ⇒ it is NOT written into the
    plan's `sound_effects`, so it never pre-empts the pipeline's own sound default
    (e.g. the showcase sound-on default). The provenance still records it."""
    plan = resolve_plan(_intent(mood="neutral"), brief={})
    assert plan.provenance["music_gain_db"].source == "default"
    assert plan.sound_effects == {}


# --- make-or-break: lines gate on a resolvable package ---------------------


def test_no_package_yields_empty_lines() -> None:
    # THE make-or-break: no footage ⇒ lines stay [] so the generated pipeline
    # builds footage from target_url. Never emit authored lines here.
    plan = resolve_plan(
        _intent(must_show=["home", "editor"], must_say=["Hello.", "World."]),
        brief={},
    )
    assert plan.lines == []
    assert not plan.package_url


def test_bare_intent_does_not_force_defaults() -> None:
    """A minimal intent (no mood, no resolution, cast=auto) must NOT emit
    resolution / dialogue / voice / sound_effects — otherwise a default (1080p,
    two-voice, empty cast) would silently pre-empt the pipeline's own defaults."""
    plan = resolve_plan(_intent(), brief={})
    assert plan.resolution is None
    assert plan.dialogue == {}
    assert plan.voice == {}
    assert plan.sound_effects == {}
    # to_brief_patch therefore adds none of them.
    assert set(plan.to_brief_patch()) <= {"lines", "package_url"}


def test_authored_resolution_is_emitted() -> None:
    plan = resolve_plan(_intent(), brief={"resolution": "4k"})
    assert plan.resolution == "4k"
    assert plan.provenance["resolution"].source == "override"
    assert plan.to_brief_patch()["resolution"] == "4k"


def test_intent_cast_pins_round_trip() -> None:
    """An `intent.cast` {role: voice} map must reach `plan.voice["cast"]` (it was
    previously dropped — resolve_plan read a nonexistent `speakers` key)."""
    plan = resolve_plan(
        _intent(cast={"host": "af_heart", "guest": "am_onyx"}, package="minio://w2v/p.zip"),
        brief={},
    )
    assert plan.voice["cast"] == {"host": "af_heart", "guest": "am_onyx"}
    assert plan.provenance["voice_cast"].source == "smart-auto"


def test_brief_cast_overrides_intent_cast() -> None:
    plan = resolve_plan(
        _intent(cast={"host": "af_heart"}, package="minio://w2v/p.zip"),
        brief={"voice": {"cast": {"host": "am_onyx"}}},
    )
    assert plan.voice["cast"] == {"host": "am_onyx"}


def test_with_package_materializes_lines_verbatim() -> None:
    plan = resolve_plan(
        _intent(
            package="minio://w2v/pkg-123.zip",
            must_show=["home", "editor"],
            must_say=["Meet the editor.", "Ship in one click."],
        ),
        brief={},
    )
    assert plan.package_url == "minio://w2v/pkg-123.zip"
    assert [line.id for line in plan.lines] == ["clip-1", "clip-2"]
    assert [line.text for line in plan.lines] == ["Meet the editor.", "Ship in one click."]


def test_authored_package_url_in_brief_wins() -> None:
    plan = resolve_plan(
        _intent(must_show=["home"], must_say=["Hi."]),
        brief={"package_url": "minio://w2v/authored.zip"},
    )
    assert plan.package_url == "minio://w2v/authored.zip"
    assert plan.provenance["package_url"].source == "override"
    assert plan.lines  # footage present ⇒ lines materialized


# --- shape -----------------------------------------------------------------


def test_plan_carries_preset_name_and_input_hash() -> None:
    plan = resolve_plan(_intent(preset="explainer_calm"), brief={})
    assert plan.preset == "explainer_calm"
    assert isinstance(plan.input_hash, str) and plan.input_hash


def test_input_hash_is_stable_for_same_intent() -> None:
    a = resolve_plan(_intent(mood="calm"), brief={})
    b = resolve_plan(_intent(mood="calm"), brief={})
    assert a.input_hash == b.input_hash
