"""website_review_script — authored-shotlist (A2) path.

When the operator authors a shotlist, it IS the BEFORE act (verbatim, deterministic
``shot-NNN`` ids), and the LLM writes ONLY the AFTER (redesign) act. The fully-AI
two-act path is exercised in test_script.py and must stay unchanged here.
"""

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.website_review.script import WebsiteReviewScriptStage

# Mixed authored BEFORE act: url, slide, url.
_SHOTLIST = [
    {
        "kind": "url",
        "url": "https://acme.test/",
        "narration": "Here's the landing page the moment you arrive.",
        "directives": {"hold_ms": 3000},
    },
    {
        "kind": "slide",
        "narration": "Three problems jump out right away.",
        "slide": {"template": "bullets", "title": "Problems", "bullets": ["Slow", "Cluttered"]},
    },
    {
        "kind": "url",
        "url": "https://acme.test/pricing",
        "narration": "And the pricing page buries the plans below the fold.",
    },
]


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


def _ctx(shotlist: list[dict] | None = None) -> RunContext:
    brief: dict[str, Any] = {"duration_seconds": 60}
    if shotlist is not None:
        brief["shotlist"] = shotlist
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": brief, "voice": {"descriptor": "warm"}},
        prior_run_outputs={
            "website_critique": {
                "url": "https://acme.test",
                "overall_score": 70.0,
                "summary": "Slow and unclear.",
                "findings": [
                    {
                        "id": "F1",
                        "category": "performance",
                        "severity": "high",
                        "title": "Slow LCP",
                        "recommendation": "Optimize images.",
                    }
                ],
            },
            "website_redesign": {"style": "modern", "visuals": []},
        },
    )


def _after_lines() -> WMResponse:
    # The LLM tags everything "before" (the shipped gotcha) — the shotlist path
    # must still pin every generated line to "after" without the pivot heuristic.
    return _resp(
        {
            "lines": [
                {"text": "Here is the new modern redesign.", "seconds": 4, "phase": "before"},
                {"text": "Did it fix the slow images?", "seconds": 3, "phase": "before"},
                {"text": "Yes, the hero now lazy-loads.", "seconds": 4, "phase": "before"},
            ]
        }
    )


@pytest.mark.asyncio
async def test_before_lines_authored_verbatim_with_shot_ids() -> None:
    fake = AsyncMock()
    fake.message.return_value = _after_lines()
    out = await WebsiteReviewScriptStage(client=fake).run({}, ctx=_ctx(_SHOTLIST))

    # 3 authored before lines + 3 AI after lines, in order.
    assert len(out.lines) == 6
    before = out.lines[:3]
    after = out.lines[3:]
    assert [line.id for line in before] == ["shot-000", "shot-001", "shot-002"]
    assert [line.text for line in before] == [s["narration"] for s in _SHOTLIST]
    assert all(line.phase == "before" for line in before)
    # Every generated line is pinned "after" — even though the LLM tagged them
    # "before" — because the boundary is known (no _assign_phases pivot).
    assert all(line.phase == "after" for line in after)
    assert [line.text for line in after] == [
        "Here is the new modern redesign.",
        "Did it fix the slow images?",
        "Yes, the hero now lazy-loads.",
    ]


@pytest.mark.asyncio
async def test_prompt_is_after_only_and_carries_authored_before() -> None:
    prompt = WebsiteReviewScriptStage(client=AsyncMock()).build_prompt(_ctx(_SHOTLIST))
    assert "ALREADY AUTHORED" in prompt
    assert "ONLY the AFTER act" in prompt
    # The authored before narration is passed as read-only context.
    assert "Here's the landing page the moment you arrive." in prompt
    # The two-act authoring instructions are NOT used on the shotlist path.
    assert "Structure the script as TWO acts" not in prompt


@pytest.mark.asyncio
async def test_speakers_derived_over_combined_acts() -> None:
    # With dialogue enabled, speaker is derived from question role across the whole
    # combined script (authored before + AI after).
    fake = AsyncMock()
    fake.message.return_value = _after_lines()
    ctx = _ctx(_SHOTLIST)
    ctx.config_snapshot["brief"]["dialogue"] = {"enabled": True}
    out = await WebsiteReviewScriptStage(client=fake).run({}, ctx=ctx)
    # before: statement, statement, statement → b; after: statement(b), question(a), statement(b)
    assert [line.speaker for line in out.lines] == ["b", "b", "b", "b", "a", "b"]


@pytest.mark.asyncio
async def test_legacy_two_act_path_unchanged_without_shotlist() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "lines": [
                {"text": "The current site is slow.", "seconds": 5, "phase": "before"},
                {"text": "Here is the new redesign.", "seconds": 5, "phase": "after"},
            ]
        }
    )
    out = await WebsiteReviewScriptStage(client=fake).run({}, ctx=_ctx())
    # No authored before lines injected; ids are LLM-minted (not shot-NNN), and the
    # content pivot drives the phases.
    assert len(out.lines) == 2
    assert not any(line.id.startswith("shot-") for line in out.lines)
    assert [line.phase for line in out.lines] == ["before", "after"]
    prompt = WebsiteReviewScriptStage(client=AsyncMock()).build_prompt(_ctx())
    assert "Structure the script as TWO acts" in prompt
