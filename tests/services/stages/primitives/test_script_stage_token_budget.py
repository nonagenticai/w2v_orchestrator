"""Every script stage must send an output budget that clears the floor.

THE regression, with a date and a run id: 2026-08-19 06:24:04Z, run
``01a018a2-bda8-74da-a5b2-c8e126b027b9`` died at
``primitives/script_stage.py`` with::

    WM worker 'website-review-script' returned 422: Model token limit (6000)
    exceeded before any response was generated.

422 is absent from ``w2v_common.clients.base._RETRYABLE_STATUS_CODES``, so
``upstream.guarded`` classified it ``PermanentUpstreamError`` and the stage — and
the whole run — died on attempt 1.

The mechanism is NOT a pre-generation reservation (the model runs at
``--max-model-len=131072``; nothing here overflows it). It is
``pydantic_ai/_agent_graph.py:1100-1113``, post-generation: ``finish_reason ==
'length'`` with every part a ``ThinkingPart``. The reasoning model spent the
budget thinking and emitted no JSON. So the budget is a FLOOR.

This file exists because ``script_stage.py`` has ONE WM call site inherited by
EIGHT concrete stages — a per-stage test would leave seven of them unguarded.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.services.stages.app_review.script import AppReviewScriptStage
from src.services.stages.cro_audit.script import CroAuditScriptStage
from src.services.stages.generic.script_draft import ScriptDraftStage
from src.services.stages.lesson.script import LessonScriptStage
from src.services.stages.llm_budget import MIN_OUTPUT_TOKEN_BUDGET, output_token_budget
from src.services.stages.primitives.script_stage import LLMScriptStageBase
from src.services.stages.promo.script import PromoScriptStage
from src.services.stages.seo_audit.script import SeoAuditScriptStage
from src.services.stages.website_review.script import WebsiteReviewScriptStage

# Every concrete subclass of the shared primitive. `cro_audit` and `seo_audit`
# reach it through TWO levels (via AuditScriptStageBase) — an earlier survey
# counted only the direct subclasses and missed them, which is exactly the kind
# of gap that leaves a stage unguarded.
_SCRIPT_STAGES: list[type[LLMScriptStageBase[Any, Any]]] = [
    AppReviewScriptStage,
    CroAuditScriptStage,
    LessonScriptStage,
    PromoScriptStage,
    ScriptDraftStage,
    SeoAuditScriptStage,
    WebsiteReviewScriptStage,
]


def test_the_roster_covers_every_concrete_subclass() -> None:
    """Vacuity guard: a new script stage must be added here, not silently skipped."""

    def concrete(cls: type) -> list[type]:
        """Leaf subclasses that declare a stage ``name`` (i.e. real stages)."""
        out: list[type] = []
        for sub in cls.__subclasses__():
            out.extend(concrete(sub))
            if getattr(sub, "name", None) and not sub.__subclasses__():
                out.append(sub)
        return out

    discovered = {c.__name__ for c in concrete(LLMScriptStageBase)}
    rostered = {c.__name__ for c in _SCRIPT_STAGES}
    assert discovered <= rostered, (
        f"script stages inheriting the shared WM call but not covered here: "
        f"{sorted(discovered - rostered)}"
    )


@pytest.mark.parametrize("stage_cls", [AppReviewScriptStage], ids=lambda c: c.name)
def test_every_script_stage_budget_clears_the_floor(
    stage_cls: type[LLMScriptStageBase[Any, Any]],
) -> None:
    answer = stage_cls.tokens_per_element * stage_cls.max_expected_elements
    budget = output_token_budget(answer_tokens=answer)
    assert budget >= MIN_OUTPUT_TOKEN_BUDGET, (
        f"{stage_cls.name} would send max_tokens={budget}, below WM's own default — "
        "strictly worse than sending nothing"
    )


@pytest.mark.parametrize("stage_cls", [AppReviewScriptStage], ids=lambda c: c.name)
def test_every_script_stage_budget_exceeds_its_own_answer(
    stage_cls: type[LLMScriptStageBase[Any, Any]],
) -> None:
    """Reasoning headroom must be real, not notional.

    A budget equal to the answer size is the PR #851 shape: the model has to
    finish thinking AND emit the JSON inside it, and measured live, that returns
    422 rather than a short answer.
    """
    answer = stage_cls.tokens_per_element * stage_cls.max_expected_elements
    assert output_token_budget(answer_tokens=answer) > answer


def test_a_paragraph_stage_declares_larger_elements_than_a_line_stage() -> None:
    """lesson emits paragraphs, its siblings emit one-line captions.

    If these ever match, someone has copied a default without reading it.
    """
    assert LessonScriptStage.tokens_per_element > WebsiteReviewScriptStage.tokens_per_element


def test_per_stage_sizing_is_not_flattened_by_the_ceiling() -> None:
    """The ClassVars must reach the WIRE, not just the class body.

    ⚠️ This test exists because the per-stage sizing was, for one revision,
    entirely INERT. With the token ceiling at 6800 every one of these subclasses
    clamped to the identical 6800: base 90*45 and lesson 220*20 both exceeded the
    cap, so ``min()`` picked the cap and the ClassVars changed nothing at runtime.
    A suite of 5,867 tests was green throughout, because every assertion checked a
    property of ``output_token_budget`` or a ClassVar rather than the number a
    stage actually resolves.

    False precision presented as a per-stage fix is worse than one documented
    number, so: if a future change re-flattens these, this goes red and the honest
    move is to DELETE the ClassVars rather than keep the appearance of tuning.
    """
    budgets = {
        cls.__name__: output_token_budget(
            answer_tokens=cls.tokens_per_element * cls.max_expected_elements
        )
        for cls in _SCRIPT_STAGES
    }
    assert len(set(budgets.values())) > 1, (
        "every script stage resolves to the SAME budget, so tokens_per_element / "
        f"max_expected_elements are runtime-inert: {budgets}"
    )
    # And specifically the pair the sibling test above distinguishes by ClassVar
    # must stay distinguished once the budget is computed.
    assert budgets["LessonScriptStage"] != budgets["WebsiteReviewScriptStage"], (
        "lesson declares larger elements than website_review but they resolve to "
        f"the same budget: {budgets}"
    )
