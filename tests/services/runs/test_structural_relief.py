"""Unit tests for the PR-C structural-relief post-completion spawn hook.

Drives ``maybe_spawn_structural_relief_run`` directly with a stubbed run engine + fake
stage rows, so every decision branch, the idempotency no-op, and the
never-break-completion guarantee are proven deterministically (no DB, no LLM).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest
from sqlalchemy.exc import IntegrityError

from src.models import beat_fingerprint
from src.services.runs.structural_relief import (
    maybe_spawn_structural_relief_run,
)

# The beats/floor feasibility maths moved OUT of the spawner into the pure policy module
# so ``reconcile_timeline``'s degrade note can reach the same decision (it used to forecast
# the spawn from ``decide_structural_relief`` alone and therefore lied on an unconvergeable
# spine). Same functions, same behaviour, one home — these tests follow them there.
from src.services.runs.structural_relief_policy import (
    next_target_beats as _next_target_beats,
)
from src.services.runs.structural_relief_policy import (
    protected_seconds_floor as _protected_seconds_floor,
)


def _spine(beats: int) -> list[dict[str, Any]]:
    """The parent's post-prepass spine as reconcile_timeline persists it — the beat
    OBJECTS the flagged ordinals index into."""
    return [{"id": f"clip-{i}", "text": f"Beat {i}.", "scenario_name": ""} for i in range(beats)]


def _stage_rows(
    *,
    outcome: str,
    requested: bool,
    overrun: float,
    beats: int = 3,
    target: float = 20.0,
    protected: int = 0,
    beat_min_seconds: list[float] | None = None,
    lines: list[dict[str, Any]] | None = None,
) -> list[Any]:
    beat_rows: list[dict[str, Any]] = [{"id": f"clip-{i}"} for i in range(beats)]
    if beat_min_seconds is not None:
        for row, mins in zip(beat_rows, beat_min_seconds, strict=True):
            row["min_seconds"] = mins
    manifest = {
        "run": {
            "convergence_outcome": outcome,
            "structural_relief_requested": requested,
            "overrun_seconds": overrun,
            "target_seconds": target,
            "protected_beat_count": protected,
            "infeasible_beat_ids": ["clip-0", "clip-1"],
        },
        "beats": beat_rows,
    }
    if lines is None:
        lines = _spine(beats)
    reconcile_output: dict[str, Any] = {"timeline_manifest": manifest, "lines": lines}
    return [
        SimpleNamespace(stage_name="demo_video_gen", output={"lines": []}),
        SimpleNamespace(stage_name="reconcile_timeline", output=reconcile_output),
    ]


def _snapshot(
    *, pass_number: int = 0, prior: float | None = None, relief_on: bool = True
) -> dict[str, Any]:
    snap: dict[str, Any] = {"tenant_id": str(uuid7()), "subject_id": str(uuid7())}
    if relief_on:
        # The relief opt-in the spawner requires — the same block reconcile arms on.
        snap["direction"] = {"timing": {"mode": "auto", "relief": {"mode": "auto"}}}
    if pass_number or prior is not None:
        relief: dict[str, Any] = {"pass": pass_number}
        if prior is not None:
            relief["prior_overrun"] = prior
        snap["structural_relief"] = relief
    return snap


def _engine(**kw: Any) -> Any:
    return SimpleNamespace(start_run=AsyncMock(**kw))


async def _run(engine: Any, snapshot: dict[str, Any], rows: list[Any]) -> Any:
    return await maybe_spawn_structural_relief_run(
        run_engine=engine,
        parent_run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot=snapshot,
        stage_rows=rows,
    )


def _patch_settings(
    monkeypatch: Any,
    *,
    enabled: bool = True,
    max_passes: int = 2,
    underfull_ratio: float = 0.0,
) -> None:
    # ``underfull_ratio`` defaults to 0.0 = the underfull clamp DISABLED, exactly as the
    # production setting disables both it and reconcile_timeline's gate. Every test that
    # predates the clamp therefore keeps testing what it tested; the clamp's own tests
    # pass a real ratio.
    fake = SimpleNamespace(
        pipeline=SimpleNamespace(
            structural_relief_autoconverge_enabled=enabled,
            structural_relief_max_passes=max_passes,
            demo_timeline_underfull_fail_ratio=underfull_ratio,
        )
    )
    monkeypatch.setattr("src.services.runs.structural_relief.get_settings", lambda: fake)


@pytest.fixture(autouse=True)
def _enable_autoconverge(monkeypatch: Any) -> None:
    """Kill-switch ON for every test here except where a test overrides it — the hook
    returns early when `structural_relief_autoconverge_enabled` is False (the default)."""
    _patch_settings(monkeypatch, enabled=True)


@pytest.mark.asyncio
async def test_spawns_child_on_cant_converge_first_pass() -> None:
    child = SimpleNamespace(id=uuid7())
    engine = _engine(return_value=child)
    result = await _run(
        engine, _snapshot(), _stage_rows(outcome="cant_converge", requested=True, overrun=13.53)
    )
    assert result == child.id
    engine.start_run.assert_awaited_once()
    kwargs = engine.start_run.call_args.kwargs
    assert kwargs["from_stage"] == "demo_video_gen"
    assert kwargs["structural_relief_pass"] == 1
    assert kwargs["pause_after_stages"] == []
    relief = kwargs["config_overrides"]["structural_relief"]
    assert relief["pass"] == 1
    assert relief["prior_overrun"] == pytest.approx(13.53)
    # target_beats must SHRINK the 3-beat spine (else the re-run can't converge)
    assert 1 <= relief["target_beats"] < 3
    # the over-budget culprits are threaded so gen cuts THEM first — the positional ids
    # for backwards compatibility, AND the content fingerprints that survive the child's
    # bdd regeneration (the ids do not: a fresh suite renumbers every clip-<n>).
    assert relief["over_budget_ids"] == ["clip-0", "clip-1"]
    assert relief["over_budget_fingerprints"] == [
        beat_fingerprint(text="Beat 0."),
        beat_fingerprint(text="Beat 1."),
    ]


@pytest.mark.asyncio
async def test_fingerprints_are_empty_when_parent_spine_is_unavailable() -> None:
    # An old/partial reconcile_timeline output with no ``lines``: there is nothing to
    # fingerprint against, so the hint degrades to the ids alone — never a crash, and
    # never a bogus fingerprint that could match an unrelated child beat.
    child = SimpleNamespace(id=uuid7())
    engine = _engine(return_value=child)
    rows = _stage_rows(outcome="cant_converge", requested=True, overrun=13.53, lines=[])
    assert await _run(engine, _snapshot(), rows) == child.id
    relief = engine.start_run.call_args.kwargs["config_overrides"]["structural_relief"]
    assert relief["over_budget_ids"] == ["clip-0", "clip-1"]
    assert relief["over_budget_fingerprints"] == []


@pytest.mark.asyncio
async def test_spawns_next_pass_when_overrun_decreased() -> None:
    child = SimpleNamespace(id=uuid7())
    engine = _engine(return_value=child)
    # pass 1 already ran with prior overrun 13.5; now down to 8.0 → progress → spawn pass 2
    result = await _run(
        engine,
        _snapshot(pass_number=1, prior=13.5),
        _stage_rows(outcome="cant_converge", requested=True, overrun=8.0),
    )
    assert result == child.id
    assert engine.start_run.call_args.kwargs["structural_relief_pass"] == 2


@pytest.mark.parametrize(
    ("outcome", "requested"),
    [("converged", True)],
)
@pytest.mark.asyncio
async def test_no_spawn_when_not_cant_converge(outcome: str, requested: bool) -> None:
    engine = _engine()
    result = await _run(
        engine, _snapshot(), _stage_rows(outcome=outcome, requested=requested, overrun=13.53)
    )
    assert result is None
    engine.start_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_spawns_cant_converge_even_when_not_requested() -> None:
    """WS-C: a footage-dominated MIXED spine classifies ``cant_converge`` yet leaves
    ``structural_relief_requested`` False (rerun_stages empty — only elastic VO beats
    were flagged). The spawn must still fire: the guard trusts ``convergence_outcome``
    alone, so this no longer starves the case orch #688 rewrote the classifier to catch.
    """
    child = SimpleNamespace(id=uuid7())
    engine = _engine(return_value=child)
    result = await _run(
        engine,
        _snapshot(),
        _stage_rows(outcome="cant_converge", requested=False, overrun=13.53),
    )
    assert result == child.id
    engine.start_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_spawn_at_cap() -> None:
    engine = _engine()
    result = await _run(
        engine,
        _snapshot(pass_number=2, prior=20.0),  # pass 2 >= max_passes 2
        _stage_rows(outcome="cant_converge", requested=True, overrun=13.53),
    )
    assert result is None
    engine.start_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_spawn_on_flat_overrun() -> None:
    engine = _engine()
    # prior 13.5, now 13.53 → within the 0.5s epsilon → no real progress → stop
    result = await _run(
        engine,
        _snapshot(pass_number=1, prior=13.5),
        _stage_rows(outcome="cant_converge", requested=True, overrun=13.53),
    )
    assert result is None
    engine.start_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_manifest_no_spawn() -> None:
    engine = _engine()
    rows = [SimpleNamespace(stage_name="demo_video_gen", output={"lines": []})]
    result = await _run(engine, _snapshot(), rows)
    assert result is None
    engine.start_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_integrity_error_is_swallowed_as_noop() -> None:
    # A concurrent replica / the reconciler backstop already spawned this (parent, pass):
    # the unique partial index rejects the duplicate INSERT. Must be a no-op, never a raise.
    engine = _engine(side_effect=IntegrityError("INSERT", {}, Exception("duplicate key")))
    result = await _run(
        engine, _snapshot(), _stage_rows(outcome="cant_converge", requested=True, overrun=13.53)
    )
    assert result is None


@pytest.mark.asyncio
async def test_arbitrary_spawn_error_never_breaks_completion() -> None:
    # The hook runs inside the completion path; ANY error must be swallowed so the parent
    # run stays terminal. A raise here would strand/duplicate a completed run.
    engine = _engine(side_effect=RuntimeError("boom"))
    result = await _run(
        engine, _snapshot(), _stage_rows(outcome="cant_converge", requested=True, overrun=13.53)
    )
    assert result is None


@pytest.mark.asyncio
async def test_missing_tenant_or_subject_skips_spawn() -> None:
    engine = _engine()
    snap = {
        "structural_relief": {"pass": 0},
        "direction": _snapshot()["direction"],
    }  # no tenant/subject
    result = await _run(
        engine, snap, _stage_rows(outcome="cant_converge", requested=True, overrun=13.53)
    )
    assert result is None
    engine.start_run.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "direction",
    [None],
)
async def test_no_spawn_when_relief_is_not_opted_in(direction: dict[str, Any] | None) -> None:
    # PR #1078 defect: reconcile builds a manifest on a relief-OFF run to carry footage
    # out-points, and the kill-switch is ON in prod, so "manifest present" alone bought a
    # PAID child. Even a manifest that says ``cant_converge`` with a real overrun must be
    # refused unless THIS run opted into relief.
    engine = _engine(return_value=SimpleNamespace(id=uuid7()))
    snap = _snapshot(relief_on=False)
    if direction is not None:
        snap["direction"] = direction
    result = await _run(
        engine, snap, _stage_rows(outcome="cant_converge", requested=True, overrun=13.53)
    )
    assert result is None
    engine.start_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_spawn_when_killswitch_off(monkeypatch: Any) -> None:
    # KILL-SWITCH (default) OFF → the hook never spawns a paid child, even on a
    # cant_converge run that would otherwise qualify.
    _patch_settings(monkeypatch, enabled=False)
    engine = _engine()
    result = await _run(
        engine, _snapshot(), _stage_rows(outcome="cant_converge", requested=True, overrun=13.53)
    )
    assert result is None
    engine.start_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_spawn_when_spine_cannot_shrink_single_beat() -> None:
    # A cant_converge run whose spine is a SINGLE beat. `_next_target_beats` floors at 1, so
    # target_beats (1) >= beat_count (1): a child would re-render an identical spine and burn a
    # paid pass for nothing. The no-shrink guard must skip the spawn even though the policy
    # decision is SPAWN (footage-pinned, requested, under cap, first pass).
    engine = _engine()
    result = await _run(
        engine,
        _snapshot(),
        _stage_rows(outcome="cant_converge", requested=True, overrun=13.53, beats=1),
    )
    assert result is None
    engine.start_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_spawn_when_overrun_is_zero() -> None:
    # A cant_converge run reporting overrun_seconds == 0: there is nothing to relieve, so the
    # no-shrink guard (`overrun <= 0`) must skip the spawn even though the policy said SPAWN.
    engine = _engine()
    result = await _run(
        engine,
        _snapshot(),
        _stage_rows(outcome="cant_converge", requested=True, overrun=0.0),
    )
    assert result is None
    engine.start_run.assert_not_awaited()


def test_next_target_beats_ignores_protected_beat_count() -> None:
    # INVERTED (was `test_next_target_beats_clamps_up_to_protected_floor`). The old clamp
    # raised the target to `protected_beat_count` on the false premise that gen cannot cut
    # those beats. gen's protected set (`gen._apply_target_beats`: index 0 / hook_role /
    # unculprit pin_position|time_lock) does NOT include footage-pinned or claim-bearing
    # beats, so the clamp only ever asked for a spine too long to converge. A 5-beat spine
    # with a 40s overrun against a 10s target must be allowed to cut all the way to 1
    # regardless of `protected_beat_count`.
    run = {"target_seconds": 10.0, "overrun_seconds": 40.0, "protected_beat_count": 4}
    assert _next_target_beats(run, 5) == 1


def test_next_target_beats_zero_beat_count_returns_zero() -> None:
    # An empty spine cannot target more beats than exist — even the protected floor cannot
    # invent beats. `n == 0` returns 0 (the old `n <= 1 → 1` wrongly claimed a beat here).
    assert _next_target_beats({"protected_beat_count": 3}, 0) == 0


def test_next_target_beats_matches_legacy_compute() -> None:
    # The pure overrun-driven compute: a 3-beat, 13.53s-overrun spine cuts to a strictly
    # smaller, >=1 beat count.
    run = {"target_seconds": 20.0, "overrun_seconds": 13.53}
    assert 1 <= _next_target_beats(run, 3) < 3


@pytest.mark.asyncio
async def test_spawns_deep_cut_on_footage_dominated_spine() -> None:
    """The arbitration case the old protected-floor clamp burned a paid render on.

    12 beats, 10 of them footage-pinned (`protected_beat_count=10`), Σmin 90s against a 60s
    target. The clamp raised target_beats from the computed 8 up to 10, so gen dropped only
    two cheap VO beats, the child still rendered ~90s and did not converge — one wasted PAID
    render. With the clamp gone the lever cuts deep: `gen._apply_target_beats` may drop
    footage-pinned beats, so a target BELOW the protected count is exactly what converges.
    """
    child = SimpleNamespace(id=uuid7())
    engine = _engine(return_value=child)
    result = await _run(
        engine,
        _snapshot(),
        _stage_rows(
            outcome="cant_converge",
            requested=True,
            overrun=30.0,
            beats=12,
            target=60.0,
            protected=10,
        ),
    )
    assert result == child.id
    relief = engine.start_run.call_args.kwargs["config_overrides"]["structural_relief"]
    assert relief["target_beats"] == 8  # ceil(30 / (90/12)) = 4 cut from 12
    assert relief["target_beats"] < 10  # BELOW the old protected-beat clamp


@pytest.mark.asyncio
async def test_no_spawn_when_no_beat_subset_can_fit_the_target() -> None:
    """Money guard: a render that provably cannot converge must never be billed.

    Three beats whose per-beat `min_seconds` are each 30s against a 20s target. Whatever
    subset gen keeps, the cheapest possible 1-beat spine still costs 30s > 20s, so the child
    cannot converge — skip the paid spawn even though the policy decision is SPAWN.
    """
    engine = _engine()
    result = await _run(
        engine,
        _snapshot(),
        _stage_rows(
            outcome="cant_converge",
            requested=True,
            overrun=70.0,
            beats=3,
            target=20.0,
            beat_min_seconds=[30.0, 30.0, 30.0],
        ),
    )
    assert result is None
    engine.start_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_spawns_when_a_cheap_subset_still_fits_the_target() -> None:
    # The mirror of the guard: the same shape but with two cheap beats, so a 1-beat spine
    # costs 5s against the 20s target and CAN converge. The guard must not over-fire.
    child = SimpleNamespace(id=uuid7())
    engine = _engine(return_value=child)
    result = await _run(
        engine,
        _snapshot(),
        _stage_rows(
            outcome="cant_converge",
            requested=True,
            overrun=13.53,
            beats=3,
            target=20.0,
            beat_min_seconds=[5.0, 5.0, 23.53],
        ),
    )
    assert result == child.id
    engine.start_run.assert_awaited_once()


def test_protected_seconds_floor_sums_the_k_smallest_mins() -> None:
    beats = [
        {"id": "a", "min_seconds": 9.0},
        {"id": "b", "min_seconds": 1.0},
        {"id": "c", "min_seconds": 4.0},
    ]
    assert _protected_seconds_floor(beats, 2) == pytest.approx(5.0)


def test_protected_seconds_floor_is_none_on_a_thin_manifest() -> None:
    # An old manifest with no per-beat min_seconds (or all zeros) cannot support the
    # judgement — return None so the caller SPAWNS rather than suppressing relief on
    # missing data.
    assert _protected_seconds_floor([{"id": "a"}, {"id": "b"}], 2) is None
    assert _protected_seconds_floor([{"id": "a", "min_seconds": 0.0}], 1) is None
    assert _protected_seconds_floor([], 1) is None
    assert _protected_seconds_floor([{"id": "a", "min_seconds": 5.0}], 0) is None


@pytest.mark.asyncio
async def test_max_passes_zero_disables_spawn(monkeypatch: Any) -> None:
    # structural_relief_max_passes=0 disables spawning even with the kill-switch on.
    _patch_settings(monkeypatch, enabled=True, max_passes=0)
    engine = _engine()
    result = await _run(
        engine, _snapshot(), _stage_rows(outcome="cant_converge", requested=True, overrun=13.53)
    )
    assert result is None
    engine.start_run.assert_not_awaited()
