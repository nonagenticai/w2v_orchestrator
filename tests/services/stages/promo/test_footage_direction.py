"""promo_footage_direction — authoritative shot-plan validation + skip + has_click.

Mirrors the demo_video footage-direction tests against promo's data sources: the
footage lines come from ``promo_storyboard`` (visual_type == "footage"), text from
``promo_script``, durations from ``promo_narrate``, and — the load-bearing bit —
``has_click`` is derived from the bound demo screen's captured ``focus_track`` via
the SAME line→screen map (``promo_visual_director`` assignments) ``promo_render``
uses. Opt-in skip-clean is owned by the shared base (no ``direction.footage`` ⇒
``StageSkippedError``), so a footage line whose screen captured NO click has its
zoom forced off and any callout dropped.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.promo.footage_direction import PromoFootageDirectionStage


def _resp(obj: object) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _ctx(*, footage: bool = True) -> RunContext:
    """Two footage lines: l1 → screen s1 (captured a click) and l2 → screen s2 (no
    click); plus a non-footage slide line l0 that must never be planned over."""
    direction: dict[str, object] = {"footage": {"auto_zoom": True}} if footage else {}
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"direction": direction} if direction else {},
        prior_run_outputs={
            "promo_storyboard": {
                "shots": [
                    {"line_id": "l0", "visual_type": "slide"},
                    {"line_id": "l1", "visual_type": "footage"},
                    {"line_id": "l2", "visual_type": "footage"},
                ]
            },
            "promo_script": {
                "lines": [
                    {"id": "l0", "text": "Intro slide.", "speaker": "a"},
                    {"id": "l1", "text": "See this leaderboard.", "speaker": "b"},
                    {"id": "l2", "text": "And the settings page.", "speaker": "b"},
                ]
            },
            "promo_narrate": {
                "clips": [
                    {"id": "n1", "line_id": "l1", "seconds": 4.0},
                    {"id": "n2", "line_id": "l2", "seconds": 3.0},
                ]
            },
            "promo_visual_director": {
                "assignments": [
                    {"id": "a1", "line_id": "l1", "screen_id": "s1"},
                    {"id": "a2", "line_id": "l2", "screen_id": "s2"},
                ]
            },
            "site_capture": {
                "screens": [
                    # s1 captured a real click (non-empty focus_track) → has_click.
                    {
                        "id": "s1",
                        "focus_track": [{"t": 0.5, "bbox": {"x": 10, "y": 20, "w": 100, "h": 40}}],
                    },
                    # s2 recorded no click target (no focus_track) → has_click False.
                    {"id": "s2"},
                ]
            },
        },
    )


async def test_valid_plan_survives_validate_footage_lines_only() -> None:
    fake = AsyncMock()
    # The model also plans over l0 (a slide line, not in the footage set) → dropped.
    fake.message.return_value = _resp(
        {
            "plans": [
                {"line_id": "l0", "zoom": "auto"},  # not a footage line → dropped
                {"line_id": "l1", "zoom": "auto", "transition_signal": "cut"},
            ]
        }
    )
    out = await PromoFootageDirectionStage(client=fake).run({}, ctx=_ctx())
    assert [p.line_id for p in out.plans] == ["l1"]
    p = out.plans[0]
    assert (p.zoom, p.transition_signal) == ("auto", "cut")


async def test_malformed_fields_coerced_to_defaults() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "plans": [
                {
                    "line_id": "l1",
                    "zoom": "SUPERZOOM",  # bad → auto
                    "reframe": "wobble",  # bad → follow-cursor
                    "transition_signal": "explode",  # bad → none
                },
            ]
        }
    )
    out = await PromoFootageDirectionStage(client=fake).run({}, ctx=_ctx())
    p = out.plans[0]
    assert (p.zoom, p.reframe, p.transition_signal) == ("auto", "follow-cursor", "none")


async def test_has_click_from_focus_track_gates_zoom_and_callout() -> None:
    # The model tries to zoom + callout BOTH lines. l1's screen captured a click so
    # its zoom/callout survive; l2's screen has no focus_track → zoom forced off and
    # callout dropped (the no-target invariant, driven purely by has_click).
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "plans": [
                {"line_id": "l1", "zoom": "auto", "callout": {"type": "label", "label": "here"}},
                {"line_id": "l2", "zoom": "auto", "callout": {"type": "label", "label": "there"}},
            ]
        }
    )
    out = await PromoFootageDirectionStage(client=fake).run({}, ctx=_ctx())
    by = {p.line_id: p for p in out.plans}
    l1 = by["l1"]
    assert l1.zoom == "auto"  # has_click True → zoom survives
    assert l1.callout is not None
    assert l1.callout.type == "label"
    l2 = by["l2"]
    assert l2.zoom == "off"  # no captured click ⇒ forced off
    assert l2.callout is None  # no captured click ⇒ dropped
