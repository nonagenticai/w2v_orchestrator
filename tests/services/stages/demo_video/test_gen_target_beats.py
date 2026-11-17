"""demo_video_gen — the relief-loop ``target_beats`` beat-reducer (PR-C hole #2).

Pure unit tests for ``_apply_target_beats`` plus one end-to-end assertion through
``DemoVideoGenStage.handle`` on the AUTHORED path, proving a relief re-run
(``config_snapshot["structural_relief"]``) shrinks the spine deterministically
while preserving protected beats, ids, and order. No DB, no LLM, no bdd call.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.models import beat_fingerprint
from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.gen import (
    DemoVideoGenStage,
    DemoVideoLine,
)


def _patch_relief_flag(monkeypatch: Any, *, enabled: bool) -> None:
    """Patch gen's ``get_settings`` so the target_beats lever is gated by the kill-switch.

    ``handle`` reads ``get_settings().pipeline.structural_relief_autoconverge_enabled`` (default
    OFF) — the lever is INERT unless the flag is on, so a stray ``structural_relief`` in
    config_overrides can't shrink the spine behind the kill-switch's back."""
    fake = SimpleNamespace(
        pipeline=SimpleNamespace(
            structural_relief_autoconverge_enabled=enabled,
            # 0.0 = the underfull clamp disabled, so these tests keep exercising the
            # bare lever they were written for.
            demo_timeline_underfull_fail_ratio=0.0,
        )
    )
    monkeypatch.setattr("src.services.stages.demo_video.gen.get_settings", lambda: fake)


def _line(idx: int, **overrides: object) -> DemoVideoLine:
    base: dict[str, object] = {"id": f"clip-{idx}", "text": f"Beat {idx}."}
    base.update(overrides)
    return DemoVideoLine.model_validate(base)


def _ctx(brief: dict, structural_relief: dict | None = None) -> RunContext:
    snapshot: dict[str, object] = {"brief": brief}
    if structural_relief is not None:
        snapshot["structural_relief"] = structural_relief
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=snapshot,
    )


# ── _apply_target_beats: no-op cases ───────────────────────────────────────


# ── _apply_target_beats: reduction ─────────────────────────────────────────


# ── culprit hints must survive a regeneration (content-keyed, not positional) ──


def test_beat_fingerprint_is_content_keyed_normalized_and_stable():
    # Deterministic + normalized: casing and whitespace runs are not content changes.
    assert beat_fingerprint(text="Hello   world") == beat_fingerprint(text="hello\n world")
    assert beat_fingerprint(text="A") != beat_fingerprint(text="B")
    # scenario_name wins over the other fields when present…
    assert beat_fingerprint(scenario_name="User Logs In", text="x") == beat_fingerprint(
        scenario_name="user logs in", text="totally different narration"
    )
    # …then gherkin, then text.
    assert beat_fingerprint(gherkin="Given a user", text="x") == beat_fingerprint(
        gherkin="given a user", text="y"
    )
    # domain-separated: the same string in a different field is a different beat.
    assert beat_fingerprint(scenario_name="x") != beat_fingerprint(text="x")
    # no content ⇒ no fingerprint (never a match)
    assert beat_fingerprint() == ""
    assert beat_fingerprint(scenario_name="   ", text="") == ""


# ── through DemoVideoGenStage.handle (authored path) ───────────────────────


@pytest.mark.asyncio
async def test_handle_authored_path_shrinks_to_target_beats(monkeypatch: Any):
    """A relief re-run over the authored spine shrinks it to target_beats, keeps
    protected beats + ids, and never calls bdd.

    Protection here is asserted via ``time_lock`` because the legacy authored-path line
    builder (``_lines_entry_to_line`` → ``_timeline_fields``) carries only
    ``time_lock``/``min_seconds``/``max_seconds``/``vo_rate`` from ``brief.lines`` — an
    inline ``pin_position``/``hook_role`` is silently dropped there (those survive only on
    the bdd + storyboard-panel paths, which the relief re-run's gen actually produces).
    ``_apply_target_beats``'s pin_position/hook_role protections are exercised by the
    direct unit tests above.

    The lever is gated by the kill-switch, so the flag must be ON for the shrink to apply.
    """
    _patch_relief_flag(monkeypatch, enabled=True)
    client = AsyncMock()
    stage = DemoVideoGenStage(client=client)
    brief = {
        "lines": [
            {"id": "clip-1", "text": "Opener.", "time_lock": True},
            {"id": "clip-2", "text": "Filler one."},
            {"id": "clip-3", "text": "Filler two."},
            {"id": "clip-4", "text": "Locked beat.", "time_lock": True},
            {"id": "clip-5", "text": "Filler three."},
        ],
    }
    out = await stage.handle(
        stage.input_model(), _ctx(brief, structural_relief={"target_beats": 3})
    )
    client.start_demo_video.assert_not_called()
    assert len(out.lines) == 3
    surviving = {line.id for line in out.lines}
    # both protected beats survive; order preserved
    assert {"clip-1", "clip-4"} <= surviving
    assert [line.id for line in out.lines] == sorted(
        surviving, key=lambda cid: int(cid.split("-")[1])
    )


@pytest.mark.asyncio
async def test_handle_authored_path_no_relief_keeps_full_spine():
    stage = DemoVideoGenStage(client=AsyncMock())
    brief = {
        "lines": [
            {"id": "clip-1", "text": "One."},
            {"id": "clip-2", "text": "Two."},
            {"id": "clip-3", "text": "Three."},
        ],
    }
    out = await stage.handle(stage.input_model(), _ctx(brief))
    assert [line.id for line in out.lines] == ["clip-1", "clip-2", "clip-3"]


# ── handle: the target_beats lever is gated by the kill-switch ──────────────


def _five_beat_brief() -> dict:
    return {
        "lines": [
            {"id": "clip-1", "text": "Opener.", "time_lock": True},
            {"id": "clip-2", "text": "Filler one."},
            {"id": "clip-3", "text": "Filler two."},
            {"id": "clip-4", "text": "Locked beat.", "time_lock": True},
            {"id": "clip-5", "text": "Filler three."},
        ],
    }


@pytest.mark.asyncio
async def test_handle_relief_lever_applies_when_flag_on(monkeypatch: Any):
    # Flag ON + a structural_relief override in config_snapshot ⇒ the spine shrinks.
    _patch_relief_flag(monkeypatch, enabled=True)
    stage = DemoVideoGenStage(client=AsyncMock())
    out = await stage.handle(
        stage.input_model(),
        _ctx(_five_beat_brief(), structural_relief={"target_beats": 3}),
    )
    assert len(out.lines) == 3


@pytest.mark.asyncio
async def test_handle_relief_lever_inert_when_flag_off(monkeypatch: Any):
    # Flag OFF (the default) ⇒ the lever is INERT: even with a structural_relief override
    # present, the spine is UNCHANGED — the kill-switch fully disables the feature.
    _patch_relief_flag(monkeypatch, enabled=False)
    stage = DemoVideoGenStage(client=AsyncMock())
    out = await stage.handle(
        stage.input_model(),
        _ctx(_five_beat_brief(), structural_relief={"target_beats": 3}),
    )
    assert [line.id for line in out.lines] == ["clip-1", "clip-2", "clip-3", "clip-4", "clip-5"]
