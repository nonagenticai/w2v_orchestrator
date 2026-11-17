from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest

from src.clients.schemas.app_inspector import (
    InspectionResult,
    NavigationEdge,
    ScreenSummary,
)
from src.services.runs.run_context import RunContext
from src.services.stages.app_review.inspect import AppInspectStage
from src.services.stages.upstream import classify


def _ctx_for(target: str, competitors: list[str] | None = None) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {
                "target_url": target,
                "competitor_urls": competitors or [],
            }
        },
    )


@pytest.mark.asyncio
async def test_inspect_collects_facts_from_target_and_competitors() -> None:
    fake = AsyncMock()
    fake.inspect_app.side_effect = [
        InspectionResult(
            estimated_purpose="A task manager",
            screens=[
                ScreenSummary(
                    screen_id="s1",
                    screen_type="home",
                    headings=["Today", "Upcoming"],
                    accessibility_score=90,
                    screenshot_path="http://stub/ss.png",
                ),
                ScreenSummary(
                    screen_id="s2",
                    screen_type="settings",
                    headings=["Account"],
                    accessibility_score=85,
                ),
            ],
            navigation_graph=[NavigationEdge(from_screen="home", to_screen="settings")],
            screens_discovered=2,
            overall_accessibility_score=88,
        ),
        InspectionResult(
            estimated_purpose="Rival app",
            screens=[ScreenSummary(screen_id="c1", screen_type="landing")],
            screens_discovered=1,
            overall_accessibility_score=70,
        ),
    ]
    stage = AppInspectStage(client=fake)

    out = await stage.run({}, ctx=_ctx_for("http://target", ["http://competitor"]))

    assert out.target_url == "http://target"
    assert out.screenshot_url == "http://stub/ss.png"
    assert any(f.kind == "purpose" for f in out.facts)
    assert any(f.kind == "screen" for f in out.facts)
    assert any(f.kind == "accessibility" for f in out.facts)
    assert any(f.kind == "screenshot" for f in out.facts)
    assert all(f.id for f in out.facts)
    assert len(out.facts) >= 4
    assert len(out.competitors) == 1
    assert out.competitors[0]["url"] == "http://competitor"
    assert out.competitors[0]["purpose"] == "Rival app"
    assert out.competitors[0]["screen_count"] == 1


@pytest.mark.asyncio
async def test_inspect_continues_when_competitor_fetch_fails() -> None:
    fake = AsyncMock()

    async def _inspect(url: str) -> InspectionResult:
        if url == "http://target":
            return InspectionResult(
                estimated_purpose="A task manager",
                screens=[ScreenSummary(screen_id="s1", screen_type="home")],
                screens_discovered=1,
                overall_accessibility_score=88,
            )
        raise RuntimeError("competitor down")

    fake.inspect_app.side_effect = _inspect
    stage = AppInspectStage(client=fake)

    out = await stage.run({}, ctx=_ctx_for("http://target", ["http://competitor"]))

    assert out.competitors == []
    assert out.target_url == "http://target"
    assert out.facts


@pytest.mark.asyncio
async def test_inspect_continues_when_a_competitor_body_fails_validation() -> None:
    """A schema-drifted competitor body must not fail the PAID target inspection.

    ``AppInspectorClient._fetch_result`` ends in ``InspectionResult.model_validate(raw)``,
    and pydantic's ``ValidationError`` derives from ``ValueError`` — not from
    ``HTTPError``/``OSError``/``TimeoutError``/``RuntimeError``. Before this was added to
    the per-competitor ``except``, one malformed competitor body escaped ``stage_gather``
    and failed the whole ``app_inspect`` stage *after* the target had already succeeded.

    The error raised here is a REAL ``ValidationError`` produced by validating a bad body,
    not a synthetic stand-in — so the test still holds if the model's shape changes.
    """
    fake = AsyncMock()

    async def _inspect(url: str) -> InspectionResult:
        if url == "http://target":
            return InspectionResult(
                estimated_purpose="A task manager",
                screens=[ScreenSummary(screen_id="s1", screen_type="home")],
                screens_discovered=1,
                overall_accessibility_score=88,
            )
        # Exactly what the client does with a drifted /result body.
        InspectionResult.model_validate({"screens": "not-a-list"})
        raise AssertionError("model_validate should have raised")

    fake.inspect_app.side_effect = _inspect
    stage = AppInspectStage(client=fake)

    out = await stage.run({}, ctx=_ctx_for("http://target", ["http://competitor"]))

    assert out.competitors == []
    assert out.target_url == "http://target"
    assert out.facts


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [402])
async def test_inspect_continues_when_a_competitor_upstream_is_classified(status: int) -> None:
    """A classified upstream failure must degrade like any other competitor failure.

    ``AppInspectorClient._fetch_result`` now routes every non-409 status through
    ``stages/upstream.classify``, so what reaches this stage is a
    ``Permanent``/``TransientUpstreamError`` — and those subclass plain ``Exception``,
    matching NONE of ``httpx.HTTPError`` / ``OSError`` / ``TimeoutError`` /
    ``RuntimeError`` / ``ValidationError``. Without them in the per-competitor
    ``except``, one competitor 503 escapes ``stage_gather`` and fails the whole PAID
    stage after the target has already succeeded — the same regression the
    ``ValidationError`` case above documents, one class later.

    The error is produced by the REAL ``classify`` on a REAL ``HTTPStatusError``, not a
    synthetic stand-in, so the test still holds if the classification changes shape.
    """
    fake = AsyncMock()

    async def _inspect(url: str) -> InspectionResult:
        if url == "http://target":
            return InspectionResult(
                estimated_purpose="A task manager",
                screens=[ScreenSummary(screen_id="s1", screen_type="home")],
                screens_discovered=1,
                overall_accessibility_score=88,
            )
        request = httpx.Request("GET", "http://app-inspector.test/api/inspect/x/result")
        response = httpx.Response(status, text='{"detail":"upstream said so"}', request=request)
        raise classify(httpx.HTTPStatusError("boom", request=request, response=response))

    fake.inspect_app.side_effect = _inspect
    stage = AppInspectStage(client=fake)

    out = await stage.run({}, ctx=_ctx_for("http://target", ["http://competitor"]))

    assert out.competitors == []
    assert out.target_url == "http://target"
    assert out.facts


@pytest.mark.asyncio
async def test_local_screenshot_path_is_not_surfaced() -> None:
    fake = AsyncMock()
    fake.inspect_app.side_effect = [
        InspectionResult(
            estimated_purpose="A task manager",
            screens=[
                ScreenSummary(
                    screen_id="s1",
                    screen_type="home",
                    screenshot_path="/var/data/screens/home.png",
                ),
            ],
            screens_discovered=1,
            overall_accessibility_score=88,
        ),
    ]
    stage = AppInspectStage(client=fake)

    out = await stage.run({}, ctx=_ctx_for("http://target"))

    assert out.screenshot_url == ""
    assert not any(f.kind == "screenshot" for f in out.facts)
    assert any(f.kind == "screen" for f in out.facts)


@pytest.mark.asyncio
async def test_inspect_raises_without_target_url() -> None:
    fake = AsyncMock()
    stage = AppInspectStage(client=fake)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={"brief": {}})

    with pytest.raises(RuntimeError, match="target_url"):
        await stage.run({}, ctx=ctx)


# --------------------------------------------------------------------------- #
# app_inspector #92 — a screenshot-less inspection must be VISIBLE, not fatal
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_an_artifact_degraded_inspection_marks_the_run_degraded_without_failing() -> None:
    """The gate, driven through the REAL stage entrypoint.

    Before ``artifacts_degraded``, an inspection that discovered screens and
    captured NOTHING was indistinguishable, on every field this stage reads, from
    a good one: facts non-empty, status "completed", structure well-formed. The
    run finished clean green and the defect surfaced only downstream as a video
    full of text cards.

    Both halves matter and neither alone is the fix:
      * non-empty ``notes`` renders the run ``degraded``
        (``api/routes/runs/_shared.py``) — the signal that did not exist;
      * the stage still COMPLETES. ``app_inspect`` legitimately degrades on
        missing input (194 completed / 62 failed measured on the live DB), so
        raising here would turn a degraded-but-watchable review into a dead run —
        worse than the bug.
    """
    fake = AsyncMock()
    fake.inspect_app.return_value = InspectionResult(
        estimated_purpose="A task manager",
        screens=[
            ScreenSummary(
                screen_id="s1",
                screen_type="home",
                # URL-shaped and non-empty: a path to a file that was never written.
                screenshot_path="http://stub/never-written.png",
                screenshot_status="failed",
            ),
        ],
        screens_discovered=1,
        screenshots_captured=0,
        artifacts_degraded=True,
    )
    stage = AppInspectStage(client=fake)

    out = await stage.run({}, ctx=_ctx_for("http://target"))

    # ...still completed, with usable facts.
    assert out.target_url == "http://target"
    assert any(f.kind == "screen" for f in out.facts)
    # ...but the degrade is on the record instead of invisible.
    assert out.artifacts_degraded is True
    assert out.screenshots_captured == 0
    assert out.notes, (
        "an artifacts_degraded inspection produced NO note, so the run renders "
        "clean green — exactly the invisibility this gate exists to remove"
    )
    assert "short on artifacts" in out.notes[0]
    assert "1 screen(s) report a FAILED capture" in out.notes[0]
    # And the dead reference is not passed downstream as a usable visual.
    assert out.screenshot_url == ""


@pytest.mark.asyncio
async def test_a_healthy_inspection_carries_no_notes() -> None:
    """Fence the note against firing on every run — a permanently-degraded run is
    the same as no signal at all."""
    fake = AsyncMock()
    fake.inspect_app.return_value = InspectionResult(
        estimated_purpose="A task manager",
        screens=[
            ScreenSummary(
                screen_id="s1",
                screen_type="home",
                screenshot_path="http://stub/ss.png",
                screenshot_status="captured",
            ),
        ],
        screens_discovered=1,
        screenshots_captured=1,
        artifacts_degraded=False,
    )
    stage = AppInspectStage(client=fake)

    out = await stage.run({}, ctx=_ctx_for("http://target"))

    assert out.notes == []
    assert out.artifacts_degraded is False
    assert out.screenshot_url == "http://stub/ss.png"
