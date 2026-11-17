"""promo_footage_critique — subclass wiring + terminal-artifact pass-through.

Covers the promo subclass hooks (shot-window timeline off the storyboard spine +
narration durations, starting artifact from promo_render, base plans from
promo_footage_direction, the canonical-compile re-render) and the pass-through
path: with ``skip_when_unrequested=False`` and no ``direction.footage.critique``,
``handle`` delivers promo_render's artifact unchanged (no analyzer/WM call).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.services.runs.run_context import RunContext
from src.services.stages.promo.footage_critique import PromoFootageCritiqueStage
from src.services.stages.shared.footage_critique import FootageCritiqueInput


def _stage() -> PromoFootageCritiqueStage:
    return PromoFootageCritiqueStage(
        client=AsyncMock(), analyzer=AsyncMock(), compile_client=AsyncMock()
    )


def _ctx(prior: dict[str, Any], config: dict[str, Any] | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot=config or {},
        prior_run_outputs=prior,
    )


# ── shot windows: cumulative timeline off the storyboard spine + narration ─────


def test_shot_windows_skips_malformed_shots_without_raising() -> None:
    # shot_windows runs BEFORE the loop's soft-degrade guard, so a malformed
    # storyboard shot / narrate clip must be SKIPPED, never raised (a hard-fail here
    # would violate the soft gate). A non-Mapping shot, a line_id-less shot, and a
    # non-Mapping clip are all dropped; the good shot still gets its window.
    ctx = _ctx(
        {
            "promo_storyboard": {
                "shots": ["not-a-mapping", {"no_line_id": True}, {"line_id": "line-ok"}]
            },
            "promo_narrate": {"clips": ["junk", {"line_id": "line-ok", "seconds": 5.0}]},
        }
    )
    windows = _stage().shot_windows(ctx)
    assert windows == [{"shot_id": "line-ok", "t0": 0.0, "t1": 5.0}]


# ── starting artifact comes from promo_render ─────────────────────────────────
def test_starting_artifact_reads_promo_render_output() -> None:
    ctx = _ctx({"promo_render": {"artifact_url": "http://v/orig.mp4", "duration_seconds": 9.5}})
    assert _stage().starting_artifact(ctx) == ("http://v/orig.mp4", 9.5)


# ── base plans come from promo_footage_direction ──────────────────────────────


def test_base_plans_non_list_is_empty() -> None:
    ctx = _ctx({"promo_footage_direction": {"plans": "not-a-list"}})
    assert _stage().base_plans(ctx) == []


# ── re-render swaps the footage-direction output and drives the compile stage ──
async def test_rerender_swaps_direction_and_drives_promo_render() -> None:
    ctx = _ctx(
        {
            "promo_footage_direction": {"plans": [{"line_id": "line-1"}]},
            "promo_render": {"artifact_url": "http://v/orig.mp4", "duration_seconds": 10.0},
        }
    )
    nudged = [{"line_id": "line-1", "zoom": "auto"}]

    fake_handle = AsyncMock(
        return_value=SimpleNamespace(artifact_url="http://v/re.mp4", duration_seconds=12.0)
    )
    fake_stage_cls = MagicMock(return_value=SimpleNamespace(handle=fake_handle))
    with patch("src.services.stages.promo.footage_critique.PromoRenderStage", fake_stage_cls):
        result = await _stage().rerender(ctx, nudged, 1)

    assert result == ("http://v/re.mp4", 12.0)
    # The nudged plan is swapped into a REPLACED ctx (never mutating the original).
    assert fake_handle.await_args is not None
    new_ctx = fake_handle.await_args.args[1]
    assert new_ctx.prior_run_outputs["promo_footage_direction"] == {"plans": nudged}
    assert ctx.prior_run_outputs["promo_footage_direction"] == {"plans": [{"line_id": "line-1"}]}
    # REVERT-PROOF: the compile discriminator is the loop-position pass ordinal, NOT a
    # hash of the nudged plan. Reverting the override to
    # ``render_idempotency_discriminator(plans)`` makes this a ``critique:<sha256…>``
    # string that fails this exact-match — pinning the resume-stable scheme.
    assert new_ctx.render_idempotency_discriminator == "critique:pass1"


# ── pass-through: no direction.footage.critique ⇒ deliver promo_render unchanged ─
async def test_handle_passes_through_when_critique_not_requested() -> None:
    # skip_when_unrequested=False ⇒ an absent critique block does NOT skip clean; it
    # delivers the current promo_render artifact unchanged (byte-identical), with no
    # analyzer / WM call. Exercises the WS-A base pass-through hook.
    analyzer = AsyncMock()
    wm = AsyncMock()
    stage = PromoFootageCritiqueStage(client=wm, analyzer=analyzer, compile_client=AsyncMock())
    ctx = _ctx(
        {"promo_render": {"artifact_url": "http://v/orig.mp4", "duration_seconds": 8.0}},
        config={},  # no direction.footage.critique
    )
    out = await stage.handle(FootageCritiqueInput(), ctx)
    assert out.artifact_url == "http://v/orig.mp4"
    assert out.duration_seconds == 8.0
    assert out.critique.reason == "critique_not_requested"
    analyzer.assert_not_awaited()
    wm.assert_not_awaited()
