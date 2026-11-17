from uuid import uuid4

import pytest

from src.services.runs.run_context import RunContext
from src.services.stages.promo.storyboard import (
    PromoStoryboardStage,
    is_single_capture,
    plan_visual_types,
    requested_accent_types,
)


def test_plan_visual_types_opens_title_closes_cta_slide() -> None:
    plan = plan_visual_types(6, has_footage=True)
    assert len(plan) == 6
    assert plan[0] == "slide"  # opening title card
    assert plan[-1] == "slide"  # closing CTA
    assert "footage" in plan  # footage-led interior
    assert set(plan) <= {"footage", "slide", "html", "ai_image"}


def test_plan_visual_types_collapses_footage_without_source() -> None:
    plan = plan_visual_types(6, has_footage=False)
    assert "footage" not in plan
    assert plan[0] == "slide" and plan[-1] == "slide"


def test_plan_visual_types_demo_sync_interior_all_footage() -> None:
    # Frame-synced demo: every interior line is footage (a keyed demo screenshot),
    # only the opening title + closing CTA stay slides.
    plan = plan_visual_types(8, has_footage=True, demo_sync=True)
    assert plan[0] == "slide" and plan[-1] == "slide"
    assert plan[1:-1] == ["footage"] * 6
    assert "ai_image" not in plan and "html" not in plan


@pytest.mark.asyncio
async def test_storyboard_demo_sync_when_screens_present() -> None:
    stage = PromoStoryboardStage()
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {
                "source_url": "https://commonpractice.nonagentic.ai/demo",
                "site_capture": {"screens": [{"selector": ".nav", "capture_id": "home"}]},
            }
        },
        prior_run_outputs={
            "promo_script": {
                "lines": [{"id": f"l{i}", "text": f"line {i}", "seconds": 2.0} for i in range(8)],
                "total_seconds": 16.0,
            },
        },
    )

    out = await stage.run({}, ctx=ctx)
    types = [s.visual_type for s in out.shots]
    assert types[0] == "slide" and types[-1] == "slide"
    assert types[1:-1] == ["footage"] * 6  # all-footage interior for the demo sync


def test_plan_visual_types_degenerate_sizes() -> None:
    assert plan_visual_types(0, has_footage=True) == []
    assert plan_visual_types(1, has_footage=True) == ["slide"]
    assert plan_visual_types(2, has_footage=True) == ["slide", "slide"]


@pytest.mark.asyncio
async def test_storyboard_one_shot_per_line_with_brand_style() -> None:
    stage = PromoStoryboardStage()
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brand": {"font": "Inter"},
            "theme": {"background": "#0B0E14", "accent": "#1F6FEB"},
        },
        prior_run_outputs={
            "promo_script": {
                "lines": [
                    {"id": "l1", "text": "Hook.", "seconds": 3.0},
                    {"id": "l2", "text": "CTA.", "seconds": 2.0},
                ],
                "total_seconds": 5.0,
            },
        },
    )

    out = await stage.run({}, ctx=ctx)

    assert len(out.shots) == 2
    assert out.shots[0].line_id == "l1"
    assert out.shots[1].line_id == "l2"
    assert all(s.id for s in out.shots)
    assert out.shots[0].seconds == 3.0
    style = out.shots[0].visual_style
    assert "Inter" in style
    assert "#0B0E14" in style
    assert "#1F6FEB" in style


@pytest.mark.asyncio
async def test_storyboard_assigns_visual_plan_with_footage_when_source_url() -> None:
    stage = PromoStoryboardStage()
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": {"source_url": "https://commonpractice.nonagentic.ai/"}},
        prior_run_outputs={
            "promo_script": {
                "lines": [{"id": f"l{i}", "text": f"line {i}", "seconds": 2.0} for i in range(6)],
                "total_seconds": 12.0,
            },
        },
    )

    out = await stage.run({}, ctx=ctx)

    types = [s.visual_type for s in out.shots]
    assert types[0] == "slide" and types[-1] == "slide"
    assert "footage" in types  # source_url present ⇒ footage retained


@pytest.mark.asyncio
async def test_storyboard_no_footage_type_without_source_url() -> None:
    stage = PromoStoryboardStage()
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},  # no brief.source_url
        prior_run_outputs={
            "promo_script": {
                "lines": [{"id": f"l{i}", "text": f"l{i}", "seconds": 2.0} for i in range(6)],
                "total_seconds": 12.0,
            },
        },
    )

    out = await stage.run({}, ctx=ctx)

    assert "footage" not in [s.visual_type for s in out.shots]


@pytest.mark.asyncio
async def test_storyboard_raises_without_script() -> None:
    stage = PromoStoryboardStage()
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    with pytest.raises(RuntimeError, match="promo_script"):
        await stage.run({}, ctx=ctx)


@pytest.mark.asyncio
async def test_storyboard_uses_safe_defaults_when_brand_missing() -> None:
    stage = PromoStoryboardStage()
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={
            "promo_script": {
                "lines": [{"id": "l1", "text": "x", "seconds": 1.0}],
                "total_seconds": 1.0,
            },
        },
    )

    out = await stage.run({}, ctx=ctx)

    assert len(out.shots) == 1
    assert "sans-serif" in out.shots[0].visual_style


# --- single-capture footage cap (merqi 2026-10-05: a bare homepage source_url
# turned ~80% of a promo into one scroll clip) ---------------------------------


def _footage_runs(types: list[str]) -> list[int]:
    return [i for i in range(1, len(types)) if types[i] == types[i - 1] == "footage"]


@pytest.mark.parametrize("n", [4])
def test_single_capture_caps_footage_one_in_four_never_adjacent(n: int) -> None:
    plan = plan_visual_types(n, has_footage=True, single_capture=True)
    assert plan.count("footage") <= n // 4
    assert _footage_runs(plan) == []
    assert plan[0] == "slide" and plan[-1] == "slide"


def test_single_capture_keeps_some_footage_when_long_enough() -> None:
    plan = plan_visual_types(12, has_footage=True, single_capture=True)
    assert plan.count("footage") == 3


def test_single_capture_demoted_lines_take_requested_accents() -> None:
    plan = plan_visual_types(
        12, has_footage=True, single_capture=True, accents=("motion_graphics", "ai_image")
    )
    assert plan.count("motion_graphics") >= 3
    assert plan.count("ai_image") >= 3
    assert plan.count("footage") <= 3


def test_single_capture_without_requested_stages_demotes_to_slides() -> None:
    plan = plan_visual_types(12, has_footage=True, single_capture=True)
    assert "motion_graphics" not in plan
    assert plan.count("slide") >= 12 - 3 - 1  # all but footage + the pattern's one ai_image


def test_multi_clip_capture_keeps_footage_led_interior() -> None:
    plan = plan_visual_types(12, has_footage=True, single_capture=False)
    assert plan.count("footage") > 12 // 4


def test_is_single_capture_shapes() -> None:
    assert is_single_capture({"source_url": "https://merqi.nl"})
    assert is_single_capture({"site_capture": {"paths": [""]}})
    assert not is_single_capture({"site_capture": {"paths": ["", "/pricing"]}})
    assert not is_single_capture({"site_capture": {"screens": [{"selector": ".x"}]}})
    assert not is_single_capture(
        {"site_capture": {"paths": [{"path": "/demo", "interactions": [{"action": "click"}]}]}}
    )


def test_requested_accent_types() -> None:
    assert requested_accent_types({}) == ("slide",)
    assert requested_accent_types({"ai_image_gen": {"prompts_from_stage": "x"}}) == ("ai_image",)
    assert requested_accent_types({"motion_graphics": {"template_id": "stat"}}) == (
        "motion_graphics",
    )
    # A motion block with neither template nor html skips the stage -> not an accent.
    assert requested_accent_types({"motion_graphics": {"items": []}}) == ("slide",)


@pytest.mark.asyncio
async def test_storyboard_bare_source_url_is_not_mostly_footage() -> None:
    """The merqi brief shape: bare homepage source_url + rich opt-in blocks."""
    stage = PromoStoryboardStage()
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"source_url": "https://merqi.nl"},
            "motion_graphics": {"template_id": "kinetic_typography"},
            "ai_image_gen": {"prompts_from_stage": "promo_image_prompts"},
        },
        prior_run_outputs={
            "promo_script": {
                "lines": [{"id": f"l{i}", "text": f"line {i}", "seconds": 4.0} for i in range(14)],
                "total_seconds": 56.0,
            },
        },
    )

    out = await stage.run({}, ctx=ctx)
    types = [s.visual_type for s in out.shots]
    assert types.count("footage") <= 14 // 4
    assert _footage_runs(types) == []
    assert "motion_graphics" in types and "ai_image" in types
