"""A FAILED scenario's clip must not ship the timeout wait that failed it.

Replays the shape of prod run ``01a0b007-9c2c-716a-883f-beefd14f4fd5`` (FTR job
``01a0b008-fefd-72f4-9159-27673003a09d``): ``clip-1`` failed on
``Timed out 5000ms waiting for expect(locator).toContainText`` and recorded 14.56s
(9.64s when the same scenario passed the run before); ``clip-5`` passed at 9.64s.
``record`` harvested both at their FILE length, and ``reconcile_timeline`` pinned the
beats to it, so the film carried the dead 5s tail.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.services.stages.demo_video.record import (
    DemoVideoRecordOutput,
    DemoVideoRecordStage,
)
from src.services.stages.shared.degradation import FAILED_SCENARIO_FOOTAGE

from .test_stages import _ctx, _passing_analyzer, _qa_settings

FEATURE = "React TodoMVC Walkthrough"
# Verbatim prod error, ANSI colouring included.
TIMEOUT_ERROR = (
    "Error: \x1b[31mTimed out 5000ms waiting for \x1b[39m\x1b[2mexpect(\x1b[22m\x1b[31m"
    "locator\x1b[39m\x1b[2m).\x1b[22mtoContainText\x1b[2m(\x1b[22m\x1b[32mexpected\x1b[39m"
    "\x1b[2m)\x1b[22m\n\nLocator: locator('#todo-count').first()\nExpected string: "
    '\x1b[32m"1"\x1b[39m\nReceived: <element(s) not found>'
)
# Prod clip-1's focus track: three typing events, the last at t=0.536s.
TYPING_TRACK = [
    {"t": t, "bbox": {"x": 730.0, "y": 349.2, "w": 1100.0, "h": 131.2}, "action": "type"}
    for t in (0.251, 0.337, 0.536)
]


def _clip(key: str, scenario: str, status: str, duration_ms: float, **extra: Any) -> dict:
    return {
        "clip_key": key,
        "feature": FEATURE,
        "scenario": scenario,
        "status": status,
        "duration_ms": duration_ms,
        "width": 2560,
        "height": 1440,
        "focus_track": TYPING_TRACK,
        "cursor_track": [],
        "viewport": {"w": 2560, "h": 1440},
        **extra,
    }


def _result(clips: list[dict], errors: dict[str, str]) -> dict[str, Any]:
    return {
        "status": "completed",
        "summary": {"total": len(clips), "passed": 1, "failed": len(clips) - 1},
        "test_results": [
            {
                "name": c["scenario"],
                "feature": FEATURE,
                "status": c["status"],
                "error": errors.get(c["clip_key"], ""),
                "duration_ms": c["duration_ms"] + 300.0,
            }
            for c in clips
        ],
        "scenario_clips": clips,
    }


async def _record(result: dict[str, Any], lines: list[str]) -> DemoVideoRecordOutput:
    client = AsyncMock()
    client.start_execution.return_value = "ftr-job"
    client.wait_for_execution.return_value = {}
    client.get_job_result.return_value = result
    client.get_clip_video.return_value = b"mp4"
    stage = DemoVideoRecordStage(client=client, analyzer=_passing_analyzer())
    gen = {"bdd_job_id": "bdd-1", "lines": [{"id": line} for line in lines]}
    with (
        patch(
            "src.services.stages.demo_video.record.get_settings",
            return_value=_qa_settings(),
        ),
        patch(
            "src.services.stages.demo_video.record.store_capture_clip",
            new=AsyncMock(side_effect=lambda **kw: f"https://minio/{kw['clip_id']}.mp4"),
        ),
    ):
        return await stage.handle(
            stage.input_model(),
            _ctx({"target_url": "https://todomvc.example"}, prior={"demo_video_gen": gen}),
        )


def _prod_replay() -> dict[str, Any]:
    return _result(
        [
            _clip("clip-1", "Add a new todo item", "failed", 14560.0),
            _clip("clip-5", "Verify todo input clears after adding", "passed", 9640.0),
        ],
        errors={"clip-1": TIMEOUT_ERROR},
    )


@pytest.mark.asyncio
async def test_failed_clip_loses_its_timeout_tail_and_passing_clip_is_untouched():
    out = await _record(_prod_replay(), ["clip-1", "clip-5"])
    by_id = {c.id: c for c in out.clips}

    failed, passed = by_id["clip-1"], by_id["clip-5"]
    # 14.56 - 5.000 = 9.56s: the typing (last action t=0.536s) is kept, the wait is not.
    assert failed.seconds == pytest.approx(9.56)
    assert failed.recorded_seconds == pytest.approx(14.56)
    assert failed.scenario_status == "failed"
    assert passed.seconds == pytest.approx(9.64)
    assert passed.recorded_seconds is None
    assert passed.scenario_status == "passed"

    codes = [d for d in out.degradations if d["code"] == FAILED_SCENARIO_FOOTAGE]
    assert len(codes) == 1
    assert "clip clip-1" in codes[0]["detail"]
    assert "14.560s -> 9.560s" in codes[0]["detail"]


@pytest.mark.asyncio
async def test_an_action_inside_the_would_be_tail_keeps_the_clip_whole():
    """A non-terminal (soft) wait: the scenario kept acting, so nothing is provably dead."""
    late = [*TYPING_TRACK, {"t": 12.0, "bbox": {"x": 1, "y": 1, "w": 1, "h": 1}, "action": "click"}]
    result = _result(
        [_clip("clip-1", "Add a new todo item", "failed", 14560.0, focus_track=late)],
        errors={"clip-1": TIMEOUT_ERROR},
    )
    out = await _record(result, ["clip-1"])
    assert out.clips[0].seconds == pytest.approx(14.56)
    assert out.clips[0].recorded_seconds is None
    assert [d["code"] for d in out.degradations] == [FAILED_SCENARIO_FOOTAGE]
    assert "kept whole" in out.degradations[0]["detail"]


@pytest.mark.asyncio
async def test_failure_without_a_wait_timeout_is_kept_whole_but_recorded():
    result = _result(
        [_clip("clip-1", "Add a new todo item", "failed", 14560.0)],
        errors={"clip-1": 'Error: expect(received).toBe(expected)\nExpected: "2"'},
    )
    out = await _record(result, ["clip-1"])
    assert out.clips[0].seconds == pytest.approx(14.56)
    assert "no expired Playwright wait" in out.degradations[0]["detail"]


# FTR's error for a scenario demoted ONLY for a bdd soft-fail: the annotation
# description is `JSON.stringify({action, selector, label, error})`
# (bdd_playwright_service scaffold/lib/recording_helpers.ts `softFail`), joined by
# frontend_test_runner_service execution_engine.py `_extract_tests_from_suite`.
SOFT_FAIL_ERROR = (
    'soft-failed selector(s): {"action":"click","selector":"#add-todo","label":"Add",'
    '"error":"TimeoutError: locator.click: Timeout 2500ms exceeded.\\nCall log:\\n'
    "  - waiting for locator('#add-todo')\"}"
)


@pytest.mark.asyncio
async def test_a_soft_fail_demotion_is_not_trimmed():
    """The scenario kept running past its 2.5s soft miss — its clip is complete."""
    result = _result(
        [_clip("clip-1", "Add a new todo item", "failed", 14560.0)],
        errors={"clip-1": SOFT_FAIL_ERROR},
    )
    out = await _record(result, ["clip-1"])
    assert out.clips[0].seconds == pytest.approx(14.56)
    assert out.clips[0].recorded_seconds is None
    assert "no expired Playwright wait" in out.degradations[0]["detail"]


@pytest.mark.asyncio
async def test_a_timeout_not_shorter_than_the_clip_keeps_it_whole():
    result = _result(
        [_clip("clip-1", "Add a new todo item", "failed", 5000.0, focus_track=[])],
        errors={"clip-1": TIMEOUT_ERROR},
    )
    out = await _record(result, ["clip-1"])
    assert out.clips[0].seconds == pytest.approx(5.0)
    assert out.clips[0].recorded_seconds is None
    assert "is not shorter than" in out.degradations[0]["detail"]


@pytest.mark.asyncio
async def test_a_cursor_action_inside_the_would_be_tail_keeps_the_clip_whole():
    cursor = [{"t": 11.2, "x": 770.0, "y": 540.0, "click": True}]
    result = _result(
        [_clip("clip-1", "Add a new todo item", "failed", 14560.0, cursor_track=cursor)],
        errors={"clip-1": TIMEOUT_ERROR},
    )
    out = await _record(result, ["clip-1"])
    assert out.clips[0].seconds == pytest.approx(14.56)
    assert "t=11.200s" in out.degradations[0]["detail"]


@pytest.mark.asyncio
async def test_a_flaky_scenario_is_never_trimmed():
    """Its kept video is the PASSING retry; the error can describe the failed attempt."""
    result = _result(
        [_clip("clip-1", "Add a new todo item", "flaky", 9640.0)],
        errors={"clip-1": TIMEOUT_ERROR},
    )
    out = await _record(result, ["clip-1"])
    assert out.clips[0].seconds == pytest.approx(9.64)
    assert out.degradations == []
