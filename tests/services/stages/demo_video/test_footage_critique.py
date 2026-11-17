"""demo_video_footage_critique — subclass wiring + terminal-artifact propagation.

Covers the demo_video subclass hooks (shot-window timeline, starting artifact) and
the DAG-level landmine: downstream (render_manifest / the finalizer's terminal probe)
must read the POST-LOOP artifact when the critique ran and compile's artifact when it
skipped-clean.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.services.blueprints.demo_video import DEMO_VIDEO_BLUEPRINT
from src.services.runs.run_context import RunContext
from src.services.stages.base import StagePreconditionError
from src.services.stages.demo_video.footage_critique import DemoVideoFootageCritiqueStage


def _stage() -> DemoVideoFootageCritiqueStage:
    return DemoVideoFootageCritiqueStage(
        client=AsyncMock(), analyzer=AsyncMock(), compile_client=AsyncMock()
    )


def _ctx(prior: dict[str, Any], config: dict[str, Any] | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=config or {},
        prior_run_outputs=prior,
    )


# ── shot windows: cumulative timeline off the spine + real narration durations ─
def test_shot_windows_are_cumulative_over_narration_durations() -> None:
    ctx = _ctx(
        {
            "demo_video_gen": {"lines": [{"id": "clip-1"}, {"id": "clip-2"}, {"id": "clip-3"}]},
            "demo_video_narrate": {
                "clips": [
                    {"line_id": "clip-1", "seconds": 4.0},
                    {"line_id": "clip-2", "seconds": 3.0},
                    # clip-3 has no narration clip → 0-length window, timeline unchanged.
                ]
            },
        }
    )
    windows = _stage().shot_windows(ctx)
    assert windows == [
        {"shot_id": "clip-1", "t0": 0.0, "t1": 4.0},
        {"shot_id": "clip-2", "t0": 4.0, "t1": 7.0},
        {"shot_id": "clip-3", "t0": 7.0, "t1": 7.0},
    ]


def test_shot_windows_skips_malformed_lines_without_raising() -> None:
    # shot_windows runs BEFORE the loop's soft-degrade guard, so a malformed spine
    # line / narrate clip must be SKIPPED, never raised (a hard-fail here would
    # violate the soft gate). A non-Mapping line, an id-less line, and a non-Mapping
    # clip are all dropped; the good line still gets its window.
    ctx = _ctx(
        {
            "demo_video_gen": {"lines": ["not-a-mapping", {"no_id": True}, {"id": "clip-ok"}]},
            "demo_video_narrate": {"clips": ["junk", {"line_id": "clip-ok", "seconds": 5.0}]},
        }
    )
    windows = _stage().shot_windows(ctx)
    assert windows == [{"shot_id": "clip-ok", "t0": 0.0, "t1": 5.0}]


# ── starting artifact comes from demo_video_compile ──────────────────────────
def test_starting_artifact_reads_compile_output() -> None:
    ctx = _ctx(
        {"demo_video_compile": {"artifact_url": "http://v/orig.mp4", "duration_seconds": 9.5}}
    )
    assert _stage().starting_artifact(ctx) == ("http://v/orig.mp4", 9.5)


def test_starting_artifact_missing_raises_precondition() -> None:
    with pytest.raises(StagePreconditionError):
        _stage().starting_artifact(_ctx({"demo_video_compile": {}}))


def test_base_plans_reads_footage_direction_output() -> None:
    ctx = _ctx({"demo_video_footage_direction": {"plans": [{"line_id": "clip-1", "zoom": "auto"}]}})
    plans = _stage().base_plans(ctx)
    assert [p["line_id"] for p in plans] == ["clip-1"]


# ── terminal-artifact propagation: render_manifest reads critique-else-compile ─


# ── DAG registration in both blueprints ──────────────────────────────────────
@pytest.mark.parametrize("blueprint", [DEMO_VIDEO_BLUEPRINT])
def test_critique_stage_registered_and_wired(blueprint: Any) -> None:
    # The critique stage depends on compile (needs its artifact) …
    spec = blueprint.stage("demo_video_footage_critique")
    assert "demo_video_compile" in spec.depends_on
    # … the terminal artifact stays compile (reorder cut + finalizer probe) …
    assert blueprint.terminal_artifact_stage == "demo_video_compile"
    # … the S19 encode/mux master hangs off critique (observes its post-loop artifact) …
    assert "demo_video_footage_critique" in blueprint.stage("demo_video_encode_master").depends_on
    # … the A11 ship gate now hangs off the encode master …
    assert "demo_video_encode_master" in blueprint.stage("final_critique").depends_on
    # … and render_manifest hangs off the ship gate, which carries the accepted-best
    # artifact through to the deliverable (final_critique → encode_master →
    # footage_critique → compile).
    assert "final_critique" in blueprint.stage("render_manifest").depends_on


# ── re-render swaps the footage-direction output and stamps a resume-stable key ─
async def test_rerender_swaps_direction_and_stamps_pass_ordinal() -> None:
    ctx = _ctx(
        {
            "demo_video_footage_direction": {"plans": [{"line_id": "line-1"}]},
            "demo_video_compile": {"artifact_url": "http://v/orig.mp4", "duration_seconds": 10.0},
        }
    )
    nudged = [{"line_id": "line-1", "zoom": "auto"}]

    fake_handle = AsyncMock(
        return_value=SimpleNamespace(artifact_url="http://v/re.mp4", duration_seconds=12.0)
    )
    fake_stage_cls = MagicMock(return_value=SimpleNamespace(handle=fake_handle))
    with patch(
        "src.services.stages.demo_video.footage_critique.DemoVideoCompileStage", fake_stage_cls
    ):
        result = await _stage().rerender(ctx, nudged, 1)

    assert result == ("http://v/re.mp4", 12.0)
    # The nudged plan is swapped into a REPLACED ctx (never mutating the original).
    assert fake_handle.await_args is not None
    new_ctx = fake_handle.await_args.args[1]
    assert new_ctx.prior_run_outputs["demo_video_footage_direction"] == {"plans": nudged}
    assert ctx.prior_run_outputs["demo_video_footage_direction"] == {
        "plans": [{"line_id": "line-1"}]
    }
    # REVERT-PROOF: the compile discriminator is the loop-position pass ordinal, NOT a
    # hash of the nudged plan. Reverting the override to
    # ``render_idempotency_discriminator(plans)`` makes this a ``critique:<sha256…>``
    # string that fails this exact-match — pinning the resume-stable scheme.
    assert new_ctx.render_idempotency_discriminator == "critique:pass1"
