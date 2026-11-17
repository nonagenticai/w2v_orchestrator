"""The ORDERED duration reaches bdd_playwright, which AUTHORS the length.

Measured live 2026-09-10: run ``01a08afc`` delivered **47.662s against a 30s order
(159%)** and run ``01a08b0d`` **25.400s (85%)**. Narration-to-picture fit was ~100.2%
in both — the error was entirely structural. Each recorded beat yields ~23.8s, so 1
beat is 85% and 2 beats are 159%, and **no integer beat count lands inside the +/-10%
conformance band**. Every downstream lever (beat count, footage trim, ``vo_rate``) was
therefore arithmetically incapable or destructive.

The order used to reach bdd only as a scenario-COUNT ceiling
(``max_scenarios_per_feature``). bdd writes the narration text AND paces the dwell that
records over it, and had never heard the ordered length. These tests pin that it now
does — and, harder, that a run with NO order sends NO key: demo_video has no blueprint
fallback duration, so a fabricated default would have bdd budget narration against an
order nobody placed.

No DB, no LLM, no live bdd — the client is a mock and the response goes through the
real ``BddJobResult`` model.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.clients.schemas.bdd_playwright import BddJobResult
from src.services.runs.run_context import RunContext
from src.services.stages.demo_video.gen import DemoVideoGenStage

_URL = "https://app.example"
_ORDERED = 30.0


def _ctx(**sections: Any) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={"brief": {"target_url": _URL}, **sections},
        prior_run_outputs={},
    )


def _client() -> AsyncMock:
    client = AsyncMock()
    client.start_demo_video.return_value = "bdd-job-1"
    client.wait_for_job.return_value = {}
    client.get_job_result.return_value = BddJobResult.model_validate(
        {
            "job_id": "bdd-job-1",
            "status": "completed",
            "features_generated": 1,
            "narration_script": {
                "parts": [
                    {
                        "part_id": "clip-1",
                        "scenario_name": "Login",
                        "text": "First we log in.",
                        "original_gherkin": "Scenario: Login",
                        "estimated_duration_seconds": 3.2,
                    }
                ]
            },
        }
    )
    return client


async def _sent(ctx: RunContext) -> dict[str, Any]:
    """The payload the stage actually handed to ``start_demo_video``."""
    client = _client()
    stage = DemoVideoGenStage(client=client)
    await stage.handle(stage.input_model(), ctx)
    return dict(client.start_demo_video.call_args.args[0])


async def test_the_ordered_duration_is_sent_to_the_generator() -> None:
    """Run 01a08afc's 30s order now reaches the thing that authors the length."""
    sent = await _sent(_ctx(deliverable={"target_duration_seconds": _ORDERED}))

    assert sent["target_duration_seconds"] == pytest.approx(_ORDERED)


async def test_the_explicit_timing_override_is_what_is_sent() -> None:
    """demo_video's OWN target ladder, not the deliverable rung alone.

    ``direction.timing.target_seconds`` is the override ``reconcile_timeline`` and
    the delivered-duration conformance record both honour first. Sending the
    deliverable instead would have bdd budget a 45s film against a 30s order —
    reintroducing the very mismatch this key exists to close, one layer down.
    """
    sent = await _sent(
        _ctx(
            deliverable={"target_duration_seconds": _ORDERED},
            direction={"timing": {"mode": "auto", "target_seconds": 45}},
        )
    )

    assert sent["target_duration_seconds"] == pytest.approx(45.0)


async def test_no_order_sends_no_key_rather_than_a_fabricated_default() -> None:
    """⛔ THE FAILURE MODE THIS GUARDS.

    demo_video has no blueprint fallback duration — ``reconcile_timeline`` raises
    ``StageSkippedError`` rather than invent one, and ``_target_seconds`` returns
    ``None``. Any default here (30s, 60s, the promo constant) would make bdd budget
    narration and dwell against an order nobody placed, and every such run would
    then report a conformance verdict against a phantom number. Absent order ⇒
    absent key ⇒ bdd applies its own pacing exactly as it does today.
    """
    for ctx in (
        _ctx(),
        _ctx(deliverable={"format": "16x9"}),
        _ctx(direction={"timing": {"mode": "auto"}}),
        _ctx(deliverable={}),
        # Corrupt config is not a 1-second order.
        _ctx(deliverable={"target_duration_seconds": True}),
        _ctx(deliverable={"target_duration_seconds": 0}),
    ):
        sent = await _sent(ctx)
        assert "target_duration_seconds" not in sent, sent


async def test_the_scenario_count_ceiling_is_unchanged() -> None:
    """The new key is ADDITIVE — the existing count derivation must not move.

    With no measured per-beat yield the order cannot be converted into a count at
    all, so the default ceiling stands (``_DEFAULT_MAX_SCENARIOS``). Sending the
    order down must not have quietly become a second sizing input.
    """
    with_order = await _sent(_ctx(deliverable={"target_duration_seconds": _ORDERED}))
    without = await _sent(_ctx())

    assert with_order["max_scenarios_per_feature"] == without["max_scenarios_per_feature"] == 5


async def test_an_explicit_author_scenario_count_still_wins() -> None:
    """An author who named a number still gets it, order or no order."""
    ctx = RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"target_url": _URL, "max_scenarios_per_feature": 3},
            "deliverable": {"target_duration_seconds": _ORDERED},
        },
        prior_run_outputs={},
    )
    sent = await _sent(ctx)

    assert sent["max_scenarios_per_feature"] == 3
    assert sent["target_duration_seconds"] == pytest.approx(_ORDERED)


async def test_the_rest_of_the_payload_is_byte_identical() -> None:
    """Nothing else about the generation request moved."""
    sent = await _sent(_ctx(deliverable={"target_duration_seconds": _ORDERED}))

    assert sent["url"] == _URL
    assert sent["feed_to_video"] is False
    assert sent["narration"] == {"enabled": True, "tone": "tutorial", "llm_polish": True}
    assert set(sent) == {
        "url",
        "max_scenarios_per_feature",
        "narration",
        "feed_to_video",
        "target_duration_seconds",
        # The word budget rides with the ordered duration (see
        # `test_gen_sends_word_budget`): a duration alone is not actionable by the
        # thing that AUTHORS the narration. `narration_words_per_scenario` is NOT here
        # because this ctx has no authored scenario count, and the derived count is a
        # ceiling rather than a quota — an absent split is deliberate.
        "narration_words_target",
    }


# ════════════════════════════════════════════════════════════════════════════════
# O3 — THE SCENARIO FLOOR (A/V quality wave 13)
#
# `max_scenarios_per_feature` is a CEILING, NOT A QUOTA — `_scenarios_for_target`'s
# own docstring says so and names the cost: on run 01a03011 the derivation asked for
# 15, bdd generated 5, and the film totalled 27.26s of a 60s order (45%). Raising a
# ceiling permits scenarios; it does not ask for them. `min_scenarios` is the bound
# that does.
#
# ⚠️ CONTRACT: the key name `min_scenarios` is agreed with bdd_playwright_service.
# Renaming it breaks a sibling service.
# ════════════════════════════════════════════════════════════════════════════════

_MEASURED = {"structural_relief": {"measured_seconds_per_beat": 10.0}}


async def _sent_over_a_five_beat_spine(ctx: RunContext) -> dict[str, Any]:
    """The payload, for a bdd result carrying FIVE narration parts.

    ⚠️ The single-part `_client()` cannot be used for the floor tests. The stage runs
    `_assert_plan_can_reach_floor` on the spine bdd ACTUALLY returned, and a one-beat
    spine at 10.0s/beat cannot reach the 15.00s underfull floor of a 30s order — so the
    stage refuses before the payload is ever built, and the test would be asserting
    against an exception instead of a bound. Five beats clear the floor at every yield
    used below, which keeps these tests about `min_scenarios` and nothing else.
    """
    client = _client()
    client.get_job_result.return_value = BddJobResult.model_validate(
        {
            "job_id": "bdd-job-1",
            "status": "completed",
            "features_generated": 1,
            "narration_script": {
                "parts": [
                    {
                        "part_id": f"clip-{i}",
                        "scenario_name": f"Step {i}",
                        "text": "First we log in.",
                        "original_gherkin": f"Scenario: Step {i}",
                        "estimated_duration_seconds": 3.2,
                    }
                    for i in range(1, 6)
                ]
            },
        }
    )
    stage = DemoVideoGenStage(client=client)
    await stage.handle(stage.input_model(), ctx)
    return dict(client.start_demo_video.call_args.args[0])


async def test_the_scenario_floor_is_sent_under_the_contracted_key() -> None:
    """A 30s order at a measured 10.0s/beat needs at least 3 scenarios to reach it."""
    sent = await _sent_over_a_five_beat_spine(
        _ctx(deliverable={"target_duration_seconds": _ORDERED}, **_MEASURED)
    )
    assert sent["min_scenarios"] == 3
    # The floor never exceeds the ceiling it ships beside — an incoherent request.
    assert sent["min_scenarios"] <= sent["max_scenarios_per_feature"]


async def test_an_unmeasured_yield_sends_no_floor_rather_than_a_fabricated_one() -> None:
    """The `_SECONDS_PER_SCENARIO = 4.0` sin, refused. With no measurement the ordered
    duration cannot be converted into a count at all, so the key is OMITTED — a floor
    derived from a fiction is worse than no floor.
    """
    sent = await _sent_over_a_five_beat_spine(
        _ctx(deliverable={"target_duration_seconds": _ORDERED})
    )
    assert "min_scenarios" not in sent


async def test_no_order_sends_no_floor() -> None:
    """No ordered duration ⇒ no duration to reach ⇒ nothing to demand."""
    sent = await _sent_over_a_five_beat_spine(_ctx(**_MEASURED))
    assert "min_scenarios" not in sent


async def test_an_authored_scenario_count_is_not_second_guessed() -> None:
    """An author who pinned `max_scenarios_per_feature` named BOTH bounds. Manufacturing
    a floor underneath it would override a number they meant.
    """
    sent = await _sent_over_a_five_beat_spine(
        _ctx(
            brief={"target_url": _URL, "max_scenarios_per_feature": 2},
            deliverable={"target_duration_seconds": _ORDERED},
            **_MEASURED,
        )
    )
    assert sent["max_scenarios_per_feature"] == 2
    assert "min_scenarios" not in sent
