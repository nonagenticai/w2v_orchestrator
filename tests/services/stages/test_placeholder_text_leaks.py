"""Placeholder text must never reach a customer's video.

Regression for a delivered promo that (1) closed on a spoken "None" — the script
prompt said ``End with this CTA verbatim: "None"`` because the caller sent no
call_to_action — and (2) showed a "Common Practice" eyebrow, a hardcoded house
brand, on every slide of a customer's video. promo_review noticed the "None"
segment, scored cta_presence 2/5, and the run published anyway.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services import tts_storage
from src.services.runs.run_context import RunContext
from src.services.stages.base import raised_failure_flags
from src.services.stages.generic.script_draft import ScriptDraftStage
from src.services.stages.promo.analyze import PromoAnalyzeStage
from src.services.stages.promo.review import (
    PromoReviewOutput,
    PromoReviewStage,
    placeholder_findings,
)
from src.services.stages.promo.script import PromoScriptStage
from src.services.stages.promo.slides import PromoSlidesStage
from src.services.stages.shared.placeholder_text import (
    is_placeholder_text,
    slide_anchor,
    text_or_empty,
)


def _resp(obj: Any) -> WMResponse:
    return WMResponse(
        output=obj, output_type="object", output_schema_used=None, metadata=None, usage_metrics=None
    )


# --------------------------------------------------------------------------- #
# vocabulary
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", [None])
def test_placeholder_values_are_recognised(value: object) -> None:
    assert is_placeholder_text(value)
    assert text_or_empty(value) == ""


@pytest.mark.parametrize("value", ["Get started today."])
def test_real_content_is_not_a_placeholder(value: object) -> None:
    assert not is_placeholder_text(value)
    assert text_or_empty(value) == str(value).strip()


# --------------------------------------------------------------------------- #
# Fix 1 — the script prompt never interpolates an absent CTA (or other field)
# --------------------------------------------------------------------------- #

_ANALYZE = {"hook_candidates": [{"text": "Tired of X?"}], "pain_points": [{"text": "slow"}]}


def _script_prompt(brief: dict, cfg: dict | None = None, voice: dict | None = None) -> str:
    cfg = {"brief": brief, **(cfg or {})}
    return PromoScriptStage._build_prompt(cfg, brief, voice or {}, _ANALYZE, [])


def test_script_prompt_without_cta_has_no_none_and_no_cta_line() -> None:
    prompt = _script_prompt({"value_proposition": None}, cfg={"title": None})
    assert "None" not in prompt
    assert "CTA verbatim" not in prompt
    assert "Title:" not in prompt
    assert "Convey this value" not in prompt
    assert "Voice:" not in prompt
    # The close is a natural takeaway, explicitly not a placeholder.
    assert "natural closing line" in prompt


def test_script_prompt_with_cta_keeps_the_verbatim_instruction() -> None:
    prompt = _script_prompt(
        {"call_to_action": "Book a demo.", "value_proposition": "Ship faster"},
        cfg={"title": "Acme"},
        voice={"descriptor": "warm"},
    )
    assert 'End with this CTA verbatim: "Book a demo."' in prompt
    assert "Title: Acme" in prompt
    assert "Voice: warm" in prompt
    assert "Ship faster" in prompt
    assert "natural closing line" not in prompt


def test_analyze_prompt_omits_absent_brief_fields() -> None:
    prompt = PromoAnalyzeStage._build_prompt({"title": "Acme"}, {"audience": "ops teams"}, {})
    assert "None" not in prompt
    assert "CTA:" not in prompt
    assert "Duration:" not in prompt
    assert "Title: Acme" in prompt
    assert "Audience: ops teams" in prompt


# --------------------------------------------------------------------------- #
# Fix 2 — slide anchor: never a hardcoded brand
# --------------------------------------------------------------------------- #


def test_slide_anchor_fallback_order() -> None:
    full = {
        "deliverable": {"title": "Launch film", "name": "marketing-video"},
        "brand": {"name": "Acme"},
        "title": "Acme promo",
    }
    assert slide_anchor(full) == "Launch film"
    assert slide_anchor({**full, "deliverable": {"name": "marketing-video"}}) == "Acme"
    assert slide_anchor({"deliverable": {"name": "x"}, "brand": {}, "title": "Acme promo"}) == (
        "Acme promo"
    )
    # Nothing usable → no eyebrow at all, never a house brand or an internal label.
    assert slide_anchor({"deliverable": {"name": "marketing-video"}, "title": "Untitled"}) == ""
    assert slide_anchor({}) == ""


class _FakeStorage:
    async def initialize(self) -> None:
        pass

    async def store_stage_artifact(self, job_id: str, stage_name: str, data: bytes, filename: str):
        return f"artifacts/{job_id}/{stage_name}/{filename}"

    async def get_presigned_url(self, path: str, expires: int | None = None, bucket=None):
        return f"https://minio.test/{path}"


@pytest.mark.asyncio
async def test_promo_slides_unbranded_run_has_no_common_practice_eyebrow() -> None:
    tts_storage.set_storage_for_tests(_FakeStorage())
    try:
        fake = AsyncMock()
        fake.resolve_template_id.side_effect = lambda name: f"tpl-{name}"
        fake.start_deck.return_value = "deck-1"
        fake.wait_for_deck.return_value = {"status": "completed", "slide_count": 2}
        fake.get_slide_png.side_effect = [b"\x89PNG-0", b"\x89PNG-1"]
        ctx = RunContext(
            run_id=uuid4(),
            deliverable_id=uuid4(),
            config_snapshot={"deliverable": {"name": "marketing-video"}, "resolution": "1080p"},
            prior_run_outputs={
                "promo_script": {
                    "lines": [{"id": "l1", "text": "Ship faster."}, {"id": "l2", "text": "Try it."}]
                }
            },
        )
        await PromoSlidesStage(client=fake).run({}, ctx=ctx)
    finally:
        tts_storage.set_storage_for_tests(None)

    payload = fake.start_deck.await_args.args[0]
    assert "Common Practice" not in repr(payload)
    assert "marketing-video" not in repr(payload)
    assert payload["title"] == ""
    assert all(s["content"]["eyebrow"] == "" for s in payload["slides"])


# --------------------------------------------------------------------------- #
# Fix 3 — placeholder lines are dropped at parse time (every script stage)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_script_stage_drops_a_trailing_none_line() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {"lines": [{"text": "Ship faster.", "seconds": 2.0}, {"text": "None", "seconds": 1.0}]}
    )
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={},
        prior_run_outputs={
            "brief_intake": {"title": "T", "angle": "a", "voice_descriptor": "crisp"}
        },
    )
    out = await ScriptDraftStage(client=fake).run({}, ctx=ctx)
    assert [line.text for line in out.lines] == ["Ship faster."]


# --------------------------------------------------------------------------- #
# Fix 4 — promo_review hard-fails on placeholder text
# --------------------------------------------------------------------------- #


def _review_ctx(lines: list[dict]) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": {}},
        prior_run_outputs={
            "promo_script": {"lines": lines},
            "promo_render": {"artifact_url": "http://stub/a.mp4"},
        },
    )


def test_placeholder_findings_deterministic_and_structured_only() -> None:
    assert placeholder_findings([{"text": "Real."}]) == []
    assert placeholder_findings([{"text": "None"}])
    assert placeholder_findings([], reviewer_saw_placeholder=True, reviewer_evidence="None")


@pytest.mark.asyncio
async def test_review_hard_fails_on_a_none_line() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "items": [
                {"criterion": "clarity", "score": 4, "rationale": "clear"},
                {"criterion": "cta_presence", "score": 2, "rationale": "final 'None' segment"},
            ],
            "summary": "ok",
            "placeholder_text_visible": True,
        }
    )
    lines = [{"id": "l1", "text": "Ship faster.", "seconds": 2.0}, {"id": "l2", "text": "None"}]
    out = await PromoReviewStage(client=fake).run({}, ctx=_review_ctx(lines))
    assert out.placeholder_text_found is True
    assert any("placeholder" in n for n in out.notes)
    # The executor's failure-flag gate is what turns this into a failed stage.
    dumped = out.model_dump()
    assert raised_failure_flags(dumped, PromoReviewOutput.fails_when_flagged) == (
        "placeholder_text_found",
    )


@pytest.mark.asyncio
async def test_review_of_a_clean_promo_does_not_fail() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "items": [
                {"criterion": "clarity", "score": 4, "rationale": "none of it is confusing"},
                {"criterion": "cta_presence", "score": 5, "rationale": "CTA present, not 'None'"},
            ],
            "summary": "No placeholder text ('None', 'null') found.",
            "placeholder_text_visible": False,
        }
    )
    out = await PromoReviewStage(client=fake).run(
        {}, ctx=_review_ctx([{"id": "l1", "text": "Book a demo.", "seconds": 2.0}])
    )
    assert out.placeholder_text_found is False
    assert raised_failure_flags(out.model_dump(), PromoReviewOutput.fails_when_flagged) == ()


@pytest.mark.asyncio
async def test_review_cta_less_brief_is_not_flagged_and_not_rendered_as_none() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {"items": [{"criterion": "cta_presence", "score": 3, "rationale": "soft close"}]}
    )
    ctx = _review_ctx([{"id": "l1", "text": "Ship faster.", "seconds": 2.0}])
    ctx.config_snapshot["brief"] = {"call_to_action": None, "audience": "ops", "notes": ""}
    out = await PromoReviewStage(client=fake).run({}, ctx=ctx)
    assert out.placeholder_text_found is False
    prompt = fake.message.await_args.kwargs["message"]
    brief_line = next(ln for ln in prompt.splitlines() if ln.startswith("Brief:"))
    assert "None" not in brief_line
    assert "call_to_action" not in prompt
    assert "'audience': 'ops'" in prompt
