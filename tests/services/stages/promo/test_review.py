from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from w2v_common.clients import WMResponse

from src.services.runs.run_context import RunContext
from src.services.stages.base import StageError
from src.services.stages.promo.review import (
    PromoReviewItem,
    PromoReviewStage,
    reviewer_quality_notes,
)


def _resp(obj: Any) -> WMResponse:
    # WM returns the structured output already parsed (output is the dict
    # itself, not a JSON string) when output_schema is passed.
    return WMResponse(
        output=obj,
        output_type="object",
        output_schema_used=None,
        metadata=None,
        usage_metrics=None,
    )


_VALID_REVIEW = {
    "items": [{"criterion": "clarity", "score": 4, "rationale": "clear"}],
    "summary": "ok",
}
# Valid object, but the single item is missing the required `criterion` field —
# the exact shape that hard-failed promo E2E run c19324ea on 2026-06-01.
_MISSING_CRITERION = {"items": [{"score": 4, "rationale": "clear"}], "summary": "ok"}


# promo_render's own compile URL, and the (possibly re-rendered) delivered URL that
# the terminal promo_footage_critique loop hands off. When the critique block wasn't
# requested the critique passes RENDER_URL through byte-identically.
RENDER_URL = "http://stub/a.mp4"
CRITIQUE_URL = "http://stub/best.mp4"


def _ctx(*, include_critique: bool = True) -> RunContext:
    prior: dict[str, Any] = {
        "promo_script": {"lines": [{"id": "l1", "text": "x", "seconds": 1.0}]},
        "promo_render": {
            "artifact_url": RENDER_URL,
            "duration_seconds": 1.0,
            "captions_burned": True,
        },
    }
    if include_critique:
        # The delivered artifact is the footage-critique terminal output.
        prior["promo_footage_critique"] = {
            "artifact_url": CRITIQUE_URL,
            "duration_seconds": 1.0,
        }
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": {"call_to_action": "Get it.", "audience": "saas"}},
        prior_run_outputs=prior,
    )


@pytest.mark.asyncio
async def test_review_scores_per_criterion() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "items": [
                {"criterion": "clarity", "score": 4, "rationale": "clear"},
                {"criterion": "hook_strength", "score": 5, "rationale": "great"},
            ],
            "summary": "Solid promo.",
        }
    )
    stage = PromoReviewStage(client=fake)

    out = await stage.run({}, ctx=_ctx())

    assert len(out.items) == 2
    assert all(i.id for i in out.items)
    assert out.items[0].score == 4
    assert out.items[1].score == 5
    assert out.overall_score == pytest.approx(4.5)
    assert out.summary == "Solid promo."


@pytest.mark.asyncio
async def test_review_raises_on_malformed_llm() -> None:
    fake = AsyncMock()
    fake.message.return_value = _resp({})
    stage = PromoReviewStage(client=fake)

    with pytest.raises(StageError):
        await stage.run({}, ctx=_ctx())


# low, but the run survives


@pytest.mark.asyncio
async def test_review_scores_delivered_artifact_with_critique_fallback() -> None:
    """The review must describe the DELIVERED artifact: the footage-critique loop's
    best URL when present, else promo_render's exact URL (byte-identical pass-through
    when `direction.footage.critique` was absent)."""
    # (1) critique present → review scores the critique's delivered URL, not render's.
    fake = AsyncMock()
    fake.message.return_value = _resp(_VALID_REVIEW)
    stage = PromoReviewStage(client=fake)
    await stage.run({}, ctx=_ctx(include_critique=True))
    prompt = fake.message.call_args.kwargs["message"]
    assert CRITIQUE_URL in prompt
    assert RENDER_URL not in prompt

    # (2) critique absent → falls back to promo_render's exact URL, unchanged.
    fake2 = AsyncMock()
    fake2.message.return_value = _resp(_VALID_REVIEW)
    stage2 = PromoReviewStage(client=fake2)
    await stage2.run({}, ctx=_ctx(include_critique=False))
    prompt2 = fake2.message.call_args.kwargs["message"]
    assert RENDER_URL in prompt2


def _item(criterion: str, score: int, rationale: str = "because") -> PromoReviewItem:
    return PromoReviewItem(id=str(uuid4()), criterion=criterion, score=score, rationale=rationale)


def test_reviewer_quality_notes_silent_on_a_differentiated_review() -> None:
    """A healthy review emits no notes at all — the run stays un-degraded."""
    items = [_item("clarity", 4), _item("pacing", 2), _item("cta_presence", 5)]

    assert reviewer_quality_notes(items) == []


def test_reviewer_quality_notes_flags_uniform_scores() -> None:
    """A uniform 4/5 across every criterion has a healthy-looking overall_score
    (4.0) but carries zero information — the ported score-discrimination check."""
    items = [_item(c, 4) for c in ("clarity", "hook_strength", "cta_presence", "pacing")]

    notes = reviewer_quality_notes(items)

    assert len(notes) == 1
    assert "no score discrimination" in notes[0]
    assert "all 4 criteria scored 4" in notes[0]


def test_reviewer_quality_notes_ignores_flatness_of_a_single_criterion() -> None:
    """One criterion cannot be 'undifferentiated' — no false positive."""
    assert reviewer_quality_notes([_item("clarity", 3)]) == []


@pytest.mark.parametrize("rationale", [""])
def test_reviewer_quality_notes_flags_unjustified_scores(rationale: str) -> None:
    """The ported na_check: scores handed out with no justification."""
    items = [_item("clarity", 2, rationale), _item("pacing", 5, rationale)]

    notes = reviewer_quality_notes(items)

    assert any("rationales are empty or N/A" in n for n in notes)


def test_reviewer_quality_notes_tolerates_a_minority_of_na_rationales() -> None:
    """One thin rationale out of three is sloppy, not degenerate."""
    items = [_item("clarity", 2, "n/a"), _item("pacing", 5, "brisk"), _item("cta_presence", 3)]

    assert reviewer_quality_notes(items) == []


def test_reviewer_quality_notes_handles_no_items() -> None:
    assert reviewer_quality_notes([]) == []


@pytest.mark.asyncio
async def test_review_degrades_run_on_degenerate_reviewer() -> None:
    """A degenerate review must DEGRADE (notes -> `degraded` run), never fail:
    the promo may be fine, it is the score that cannot be trusted."""
    fake = AsyncMock()
    fake.message.return_value = _resp(
        {
            "items": [
                {"criterion": "clarity", "score": 4, "rationale": "N/A"},
                {"criterion": "hook_strength", "score": 4, "rationale": "N/A"},
                {"criterion": "pacing", "score": 4, "rationale": "N/A"},
            ],
            "summary": "Fine.",
        }
    )
    stage = PromoReviewStage(client=fake)

    out = await stage.run({}, ctx=_ctx())

    assert out.overall_score == pytest.approx(4.0)  # looks healthy...
    assert len(out.notes) == 2  # ...but both degeneracy checks fired
    assert any("no score discrimination" in n for n in out.notes)
    assert any("rationales are empty or N/A" in n for n in out.notes)


@pytest.mark.asyncio
async def test_review_raises_without_render_or_script() -> None:
    fake = AsyncMock()
    stage = PromoReviewStage(client=fake)
    ctx = RunContext(run_id=uuid4(), deliverable_id=uuid4(), config_snapshot={})

    with pytest.raises(RuntimeError, match="render"):
        await stage.run({}, ctx=ctx)


@pytest.mark.asyncio
async def test_review_reprompts_when_item_missing_criterion() -> None:
    """A one-off item missing `criterion` re-prompts within the retry budget
    rather than failing the whole run (the regression behind promo E2E
    c19324ea — a non-deterministic LLM dropped a per-item field)."""
    fake = AsyncMock()
    fake.message = AsyncMock(side_effect=[_resp(_MISSING_CRITERION), _resp(_VALID_REVIEW)])
    stage = PromoReviewStage(client=fake)

    out = await stage.run({}, ctx=_ctx())

    assert fake.message.await_count == 2  # re-prompted once, then succeeded
    assert [i.criterion for i in out.items] == ["clarity"]
    assert out.overall_score == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_review_fails_loud_after_retries_on_persistent_bad_shape() -> None:
    """A persistently malformed shape still fails loud (no synthetic fallback)
    once the retry budget is spent."""
    fake = AsyncMock()
    fake.message = AsyncMock(return_value=_resp(_MISSING_CRITERION))
    stage = PromoReviewStage(client=fake)

    with pytest.raises(StageError, match="criterion"):
        await stage.run({}, ctx=_ctx())
    assert fake.message.await_count == 2  # llm_retries=1 → 2 attempts, all bad


# ── an empty review is never a pass (prod run 01a17e6f) ─────────────────────

_EMPTY_REVIEW = {"items": [], "summary": "No promo content provided to review."}


@pytest.mark.asyncio
async def test_an_empty_review_is_reprompted_and_then_scored() -> None:
    fake = AsyncMock()
    fake.message.side_effect = [_resp(_EMPTY_REVIEW), _resp(_VALID_REVIEW)]
    stage = PromoReviewStage(client=fake)
    stage._llm_retries = 1
    out = await stage.run({}, ctx=_ctx())
    assert fake.message.await_count == 2
    assert [i.criterion for i in out.items] == ["clarity"]
    assert out.degradations == []


@pytest.mark.asyncio
async def test_a_review_that_stays_empty_is_flagged_with_a_code_not_passed() -> None:
    from src.services.stages.shared.degradation import ensure_degradation_codes, is_degraded

    fake = AsyncMock()
    fake.message.return_value = _resp(_EMPTY_REVIEW)
    stage = PromoReviewStage(client=fake)
    stage._llm_retries = 1
    out = await stage.run({}, ctx=_ctx())
    assert fake.message.await_count == 2
    assert out.items == []
    assert [d["code"] for d in out.degradations] == ["promo_review_empty"]
    assert "Rerun this step" in out.degradations[0]["detail"]
    assert out.summary.startswith("Not reviewed")
    dumped = out.model_dump(mode="json")
    assert is_degraded(dumped)
    # A real code, so the executor never stamps "Unclassified note" on it.
    assert not ensure_degradation_codes("promo_review", dumped)


@pytest.mark.asyncio
async def test_a_script_with_no_narration_fails_instead_of_being_reviewed() -> None:
    from src.services.stages.base import StagePreconditionError

    fake = AsyncMock()
    fake.message.return_value = _resp(_VALID_REVIEW)
    ctx = _ctx()
    ctx.prior_run_outputs["promo_script"] = {"lines": [{"id": "l1", "text": "  "}]}
    with pytest.raises(StagePreconditionError, match="no narration to review"):
        await PromoReviewStage(client=fake).run({}, ctx=ctx)
    fake.message.assert_not_awaited()


def test_the_prompt_says_the_script_is_the_content() -> None:
    prompt = PromoReviewStage._build_prompt(
        {"lines": [{"text": "Report a bottleneck"}, {"text": ""}]}, {}, "http://x/a.mp4"
    )
    assert "content is its narration script" in prompt
    assert "1. Report a bottleneck\nOutput URL:" in prompt


def test_the_prompt_never_carries_the_site_login_or_render_settings() -> None:
    """Run 01a17e6f's real brief shape went into the prompt whole: the sign-in
    (username, sealed password token, OIDC URL), the shotlist, sources, render
    settings. The reviewer then said "No promo content provided"."""
    brief = {
        "audience": "senior management",
        "call_to_action": "Register for early access",
        "cta_url": "https://cp.example/",
        "value_proposition": "Kaizen for knowledge work",
        "source_url": "https://app.example/",
        "sources": [{"ref": "https://app.example/", "kind": "url", "label": ""}],
        "shotlist": [{"kind": "url", "url": "https://app.example/", "narration": "home"}],
        "resolution": "4k",
        "motion_graphics": {"enabled": True},
        "auth_info": {
            "type": "basic",
            "username": "person@example.com",
            "password": {"$sealed": "gAAAAAB-not-real"},  # pragma: allowlist secret
            "login_url": "https://auth.example/realms/x/auth?state=abc",
        },
    }
    script = {"lines": [{"id": f"shot-00{i}", "text": f"Line {i}"} for i in range(5)]}
    prompt = PromoReviewStage._build_prompt(script, brief, "http://x/a.mp4")
    for leaked in ("person@example.com", "$sealed", "gAAAAAB", "auth.example", "shotlist", "4k"):
        assert leaked not in prompt, leaked
    assert all(f"{i + 1}. Line {i}" in prompt for i in range(5))
    assert "Register for early access" in prompt
    assert "Kaizen for knowledge work" in prompt
