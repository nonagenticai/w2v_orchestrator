"""DemoVideoNarrateStage.synth_text_for — the SSML lowering hook (TTSB-3).

The load-bearing property is the default-OFF plain path: with
``NARRATION_SSML_ENABLED`` false the hook returns ``(text, False)`` — byte
identical to the pre-SSML behaviour (same TTS payload, same clip id). With the
flag on AND a matching voice_director directive carrying a pause / emphasis, the
hook lowers the line into ``<speak>…</speak>`` and reports ``is_ssml=True``.

``pause_seconds_for`` mirrors the same directive lookup (``_matching_directive``)
so the fit solver (``NarrationStageBase._fit_to_target`` / ``_fit_speed``) learns
about the fixed ``<break>`` silence baked into the SSML above WITHOUT re-parsing
the markup back out — see ``test_narration_fit.py`` for the solver-level proof
that this actually corrects the mis-attribution.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from uuid import uuid4

from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.narrate import DemoVideoNarrateStage
from src.services.stages.primitives.ssml import build_line_ssml

_LINE = {"id": "b1", "text": "Meet our new dashboard."}


def _ctx(directives: list[dict[str, Any]] | None = None) -> RunContext:
    prior: dict[str, Any] = {"demo_video_gen": {"lines": [_LINE]}}
    if directives is not None:
        prior["voice_director"] = {"directives": directives}
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"voice": {"voice_id": "af_heart"}},
        prior_run_outputs=prior,
    )


def _settings(*, narration_ssml_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(narration_ssml_enabled=narration_ssml_enabled)


def _patch_settings(*, narration_ssml_enabled: bool):
    return patch(
        "src.services.stages.demo_video.narrate.get_settings",
        return_value=_settings(narration_ssml_enabled=narration_ssml_enabled),
    )


def test_flag_off_returns_plain_even_with_a_directive() -> None:
    """Flag OFF ⇒ the plain-text path regardless of any voice_director directive."""
    ctx = _ctx([{"line_id": "b1", "pause_before_ms": 500, "emphasis_word_indices": [0]}])
    with _patch_settings(narration_ssml_enabled=False):
        text, is_ssml = DemoVideoNarrateStage().synth_text_for(_LINE, ctx)
    assert (text, is_ssml) == ("Meet our new dashboard.", False)


def test_flag_on_with_directive_lowers_to_ssml() -> None:
    """Flag ON + a matching directive with a pause + emphasis ⇒ (<speak>…</speak>, True)."""
    ctx = _ctx([{"line_id": "b1", "pause_before_ms": 500, "emphasis_word_indices": [0]}])
    with _patch_settings(narration_ssml_enabled=True):
        text, is_ssml = DemoVideoNarrateStage().synth_text_for(_LINE, ctx)
    assert is_ssml is True
    assert text.startswith("<speak>") and text.endswith("</speak>")
    assert text == build_line_ssml("Meet our new dashboard.", 500, [0])


def test_flag_on_no_matching_directive_returns_plain() -> None:
    """Flag ON but no directive for this line ⇒ plain path."""
    ctx = _ctx([{"line_id": "OTHER", "pause_before_ms": 500}])
    with _patch_settings(narration_ssml_enabled=True):
        text, is_ssml = DemoVideoNarrateStage().synth_text_for(_LINE, ctx)
    assert (text, is_ssml) == ("Meet our new dashboard.", False)


def test_flag_on_inert_directive_returns_plain() -> None:
    """Flag ON but the directive expresses nothing (no pause, no emphasis) ⇒
    build_line_ssml returns None ⇒ plain path (cache stays warm)."""
    ctx = _ctx([{"line_id": "b1", "pause_before_ms": 0, "emphasis_word_indices": []}])
    with _patch_settings(narration_ssml_enabled=True):
        text, is_ssml = DemoVideoNarrateStage().synth_text_for(_LINE, ctx)
    assert (text, is_ssml) == ("Meet our new dashboard.", False)


def test_flag_on_no_voice_director_stage_returns_plain() -> None:
    """Flag ON but voice_director never ran (no prior output) ⇒ plain path."""
    ctx = _ctx(directives=None)
    with _patch_settings(narration_ssml_enabled=True):
        text, is_ssml = DemoVideoNarrateStage().synth_text_for(_LINE, ctx)
    assert (text, is_ssml) == ("Meet our new dashboard.", False)


# ══ pause_seconds_for — feeds the fit solver the SAME pause synth_text_for used ═


def test_pause_seconds_for_flag_off_is_zero_even_with_a_directive() -> None:
    """Flag OFF ⇒ no SSML was built at all, so the fit solver must see 0.0."""
    ctx = _ctx([{"line_id": "b1", "pause_before_ms": 500, "emphasis_word_indices": [0]}])
    with _patch_settings(narration_ssml_enabled=False):
        assert DemoVideoNarrateStage().pause_seconds_for(_LINE, ctx) == 0.0


def test_pause_seconds_for_matches_the_directive_lowered_into_ssml() -> None:
    """Flag ON + a matching directive with pause_before_ms=500 ⇒ 0.5 seconds —
    the SAME 500 synth_text_for lowered into `<break time="500ms"/>`, read from
    the directive dict rather than re-parsed out of that markup."""
    ctx = _ctx([{"line_id": "b1", "pause_before_ms": 500, "emphasis_word_indices": [0]}])
    with _patch_settings(narration_ssml_enabled=True):
        assert DemoVideoNarrateStage().pause_seconds_for(_LINE, ctx) == 0.5


def test_pause_seconds_for_no_matching_directive_is_zero() -> None:
    ctx = _ctx([{"line_id": "OTHER", "pause_before_ms": 500}])
    with _patch_settings(narration_ssml_enabled=True):
        assert DemoVideoNarrateStage().pause_seconds_for(_LINE, ctx) == 0.0


def test_pause_seconds_for_no_pause_on_the_directive_is_zero() -> None:
    ctx = _ctx([{"line_id": "b1", "pause_before_ms": 0, "emphasis_word_indices": [0]}])
    with _patch_settings(narration_ssml_enabled=True):
        assert DemoVideoNarrateStage().pause_seconds_for(_LINE, ctx) == 0.0


def test_pause_seconds_for_no_voice_director_stage_is_zero() -> None:
    ctx = _ctx(directives=None)
    with _patch_settings(narration_ssml_enabled=True):
        assert DemoVideoNarrateStage().pause_seconds_for(_LINE, ctx) == 0.0


def test_pause_seconds_for_junk_pause_degrades_to_zero_not_a_crash() -> None:
    """A directive dict comes off an LLM-shaped stage output — not guaranteed
    numeric. Junk must degrade to "no pause" rather than raise or corrupt the
    fit-solver arithmetic (which divides by ``target - pause``)."""
    ctx = _ctx([{"line_id": "b1", "pause_before_ms": "soon"}])
    with _patch_settings(narration_ssml_enabled=True):
        assert DemoVideoNarrateStage().pause_seconds_for(_LINE, ctx) == 0.0
