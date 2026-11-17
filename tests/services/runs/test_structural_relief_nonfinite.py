"""A non-finite number in the manifest must never decide a PAID render.

The relief lever spawns child renders, and billing in this fleet is append-only with no
refund path — so every number it reads is a money decision. Those numbers come off a
jsonb ``timeline_manifest``/``config_snapshot`` whose ultimate source is the unvalidated
``config_overrides`` dict on the run-create API. ``json.loads("1e400")`` is ``inf``, and
no pydantic model in the path sets ``allow_inf_nan=False``.

Two failure directions are pinned, and they are opposite:

* **A false REFUSAL.** ``math.isclose(inf, inf)`` is True, so a single ``inf`` beat row
  qualified as footage-PINNED and made ``measured_seconds_per_beat`` return ``inf``. At
  that "yield" no beat count can land inside the conformance band, the enumeration comes
  back empty, and the shape reports ``STOP_UNDERFULL`` — a confident "no legal sizing
  exists" derived from a number nobody measured, blocking a spawn the real footage
  supports.

* **A false ACCEPTANCE.** ``nan``/``inf`` sailed through every ``<= 0`` guard (each
  comparison against ``nan`` is False) into ``math.ceil``, which raises mid-decision —
  and where it did not raise, a corrupt overrun silently satisfied the progress test
  that gates the next paid pass.

The answer in both directions is REFUSAL TO JUDGE, not a verdict: ``None`` from
``relief_shape_from_manifest`` ("cannot promise a spawn") and ``None`` from the spawner
("no spawn"). Neither asserts the render is impossible — this repo has shipped a false
impossibility claim before, and a corrupt input must not manufacture another.
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest

from src.services.runs.structural_relief import maybe_spawn_structural_relief_run
from src.services.runs.structural_relief_policy import (
    decide_relief_shape,
    next_target_beats,
    protected_seconds_floor,
    relief_shape_from_manifest,
)

NAN = float("nan")
INF = float("inf")

_FLOOR_RATIO = 0.5
#: 6 beats x 7.00s measured against a 30.00s order with a 12.00s overrun: 4 beats =
#: 28.00s lands inside the +/-10% band, so a legal relief cut EXISTS and the shape
#: spawns. That is the point — a guard is only observable where the answer is not
#: already "stop".
_PER_BEAT_SECONDS = 7.0
_BEAT_COUNT = 6
_TARGET_SECONDS = 30.0


def _beat(seconds: float, index: int = 0) -> dict[str, Any]:
    """A footage-PINNED manifest row: ``min == max`` is the measured-truth discriminator."""
    return {
        "id": f"clip-{index}",
        "min_seconds": seconds,
        "max_seconds": seconds,
        "seconds": seconds,
    }


def _beats(*, corrupt_first: bool = False) -> list[dict[str, Any]]:
    rows = [_beat(_PER_BEAT_SECONDS, i) for i in range(_BEAT_COUNT)]
    if corrupt_first:
        rows[0] = _beat(INF, 0)
    return rows


def _run_block(**over: Any) -> dict[str, Any]:
    run: dict[str, Any] = {
        "target_seconds": _TARGET_SECONDS,
        "overrun_seconds": 12.0,
        "convergence_outcome": "cant_converge",
        "structural_relief_requested": True,
        "protected_beat_count": 0,
        "infeasible_beat_ids": ["clip-0", "clip-1"],
    }
    run.update(over)
    return run


def test_an_inf_beat_row_no_longer_fakes_a_stop_underfull() -> None:
    """One corrupt row must not veto a spawn the REAL footage supports.

    Same spine twice, the second with beat 0's measured length replaced by ``inf``. The
    ``inf`` row is dropped BEFORE the average now, so both decisions are made on the
    7.00s the recorder actually produced and land on the same verdict and the same count.
    """
    clean = decide_relief_shape(run=_run_block(), beats=_beats(), floor_ratio=_FLOOR_RATIO)
    corrupt = decide_relief_shape(
        run=_run_block(), beats=_beats(corrupt_first=True), floor_ratio=_FLOOR_RATIO
    )

    assert clean.verdict == "SPAWN"
    assert clean.measured_seconds_per_beat == _PER_BEAT_SECONDS
    # The corrupt row contributes NOTHING: same measured yield, same verdict, same count.
    assert corrupt.measured_seconds_per_beat == _PER_BEAT_SECONDS
    assert corrupt.verdict == "SPAWN"
    assert corrupt.target_beats == clean.target_beats


@pytest.mark.parametrize("bad", [NAN])
def test_a_nonfinite_target_never_manufactures_an_out_of_band_refusal(bad: float) -> None:
    """``decide_relief_shape`` reads an unreadable target as ABSENT, not as a number.

    With ``inf`` passed through, ``target_seconds > 0.0`` is True, ``band_seconds``
    returns the "no band" ``(0.0, 0.0)``, ``beat_counts_within_band`` empties — and an
    empty enumeration is precisely what licenses ``STOP_UNDERFULL``. The refusal would
    then cite counts it never really examined, against an order that was never really
    read. "I could not measure" must not become "it cannot be done".
    """
    shape = decide_relief_shape(
        run=_run_block(target_seconds=bad), beats=_beats(), floor_ratio=_FLOOR_RATIO
    )
    assert shape.verdict != "STOP_UNDERFULL"
    assert shape.target_seconds == 0.0  # "no ordered duration", the absent-key value


@pytest.mark.parametrize("bad", [NAN])
@pytest.mark.parametrize("key", ["target_seconds"])
def test_a_nonfinite_manifest_refuses_cleanly_instead_of_raising(key: str, bad: float) -> None:
    """The reachable path, end to end: ``config_overrides`` -> manifest -> here.

    ``reconcile_timeline`` calls this in-process on the manifest it has just built.
    Before the guard, ``nan`` reached ``math.ceil`` (``ValueError``) and ``inf`` reached
    it as ``OverflowError``, INSIDE the completion path of a run already paid for.
    ``None`` is this module's existing "cannot support the judgement" answer, which the
    caller reads as "cannot promise a spawn" — never as "will spawn", and never as an
    assertion that no spawn could work.
    """
    manifest = {"run": _run_block(**{key: bad}), "beats": _beats()}
    assert relief_shape_from_manifest(manifest, floor_ratio=_FLOOR_RATIO) is None


def test_a_finite_manifest_still_gets_a_shape() -> None:
    """The no-behaviour-change half: the guard must not refuse a healthy manifest."""
    shape = relief_shape_from_manifest(
        {"run": _run_block(), "beats": _beats()}, floor_ratio=_FLOOR_RATIO
    )
    assert shape is not None
    assert shape.verdict == "SPAWN"
    assert shape.measured_seconds_per_beat == _PER_BEAT_SECONDS


# ── the three readers that run BEFORE (or beside) the coercion above ─────────────────
#
# Found by sweeping {nan, ±inf} through every duration input of this module and asserting
# on what came back, rather than by reading the diff: the guards added at
# ``decide_relief_shape``/``relief_shape_from_manifest`` do NOT cover these three, because
# ``next_target_beats`` is evaluated as an ARGUMENT to the first of them and the beat-row
# readers were never coerced at all.


@pytest.mark.parametrize("bad", [NAN])
def test_a_nonfinite_overrun_does_not_crash_the_cut_sizing(bad: float) -> None:
    """The ``math.ceil`` crash, still live one frame above the guard that was added for it.

    ``decide_relief_shape`` coerces its own ``overrun``/``target`` — but it computes
    ``target_beats = next_target_beats(run, ...)`` FIRST, and that function re-read the
    raw run block. With ``overrun_seconds = inf`` the average fallback computes
    ``sigma_min = target + inf = inf``, ``avg_min = inf``, and ``ceil(inf / inf)`` is
    ``ceil(nan)`` → ``ValueError: cannot convert float NaN to integer``, raised inside the
    completion path of a run already paid for. Read as ABSENT, the ``overrun <= 0`` branch
    owns it: cut nothing, shrink by the standard one beat.
    """
    run = _run_block(overrun_seconds=bad)
    # A ``nan`` beat row is what forces the AVERAGE fallback (the cheapest-first sweep
    # refuses an unreadable row), which is the branch holding the ceil.
    rows = _beats()
    rows[0] = _beat(NAN, 0)
    assert next_target_beats(run, _BEAT_COUNT, rows) == _BEAT_COUNT - 1
    # …and the whole decision completes rather than raising.
    shape = decide_relief_shape(run=run, beats=rows, floor_ratio=_FLOOR_RATIO)
    assert shape.verdict == "STOP_CANNOT_SHRINK"  # no overrun to judge ⇒ nothing to shrink


@pytest.mark.parametrize("bad", [NAN])
def test_a_nonfinite_beat_row_never_moves_the_convergence_floor(bad: float) -> None:
    """``protected_seconds_floor`` is compared against the target to withhold a PAID render.

    All three non-finite values corrupt that comparison, and two of them fail OPEN:
    ``nan > target`` is False and ``-inf`` drags the sum below any target, so a corrupt row
    would license a render the real footage cannot deliver; ``inf`` fails the other way and
    manufactures an unconvergeable verdict. Dropping the row leaves the floor exactly where
    the measured beats put it.
    """
    rows = _beats()
    rows[0] = _beat(bad, 0)
    floor = protected_seconds_floor(rows, 3)
    assert floor is not None and math.isfinite(floor)
    # The 5 surviving 7.00s rows: the three smallest sum to 21.00s, unchanged by the row.
    assert floor == protected_seconds_floor(_beats()[1:], 3) == 21.0

    # Nothing measurable left at all ⇒ UNMEASURED (``None``), which this module's caller
    # documents as "fall back to spawning" — a judgement withheld, not one asserted.
    assert protected_seconds_floor([_beat(bad, 0)], 1) is None


def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.services.runs.structural_relief.get_settings",
        lambda: SimpleNamespace(
            pipeline=SimpleNamespace(
                structural_relief_autoconverge_enabled=True,
                structural_relief_max_passes=2,
                demo_timeline_underfull_fail_ratio=_FLOOR_RATIO,
            )
        ),
    )


def _stage_rows(run_block: dict[str, Any]) -> list[Any]:
    manifest = {"run": run_block, "beats": _beats()}
    lines = [
        {"id": f"clip-{i}", "text": f"Beat {i}.", "scenario_name": ""} for i in range(_BEAT_COUNT)
    ]
    return [
        SimpleNamespace(stage_name="demo_video_gen", output={"lines": []}),
        SimpleNamespace(
            stage_name="reconcile_timeline",
            output={"timeline_manifest": manifest, "lines": lines},
        ),
    ]


@pytest.mark.asyncio
async def test_spawner_bills_a_child_on_a_healthy_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control. Without this the refusal tests below prove nothing — a spawner that
    never spawns would pass them all."""
    _patch_settings(monkeypatch)
    child = SimpleNamespace(id=uuid7())
    engine = SimpleNamespace(start_run=AsyncMock(return_value=child))
    spawned = await maybe_spawn_structural_relief_run(
        run_engine=engine,  # ty: ignore[invalid-argument-type]
        parent_run_id=uuid7(),
        deliverable_id=uuid7(),
        config_snapshot={
            "tenant_id": str(uuid7()),
            "subject_id": str(uuid7()),
            "direction": {"timing": {"mode": "auto", "relief": {"mode": "auto"}}},
            "structural_relief": {"pass": 0, "prior_overrun": 20.0},
        },
        stage_rows=_stage_rows(_run_block()),
    )
    assert spawned == child.id
    engine.start_run.assert_awaited_once()


@pytest.mark.parametrize("bad", [NAN])
@pytest.mark.parametrize(
    ("location", "key"),
    [("run", "target_seconds")],
)
@pytest.mark.asyncio
async def test_spawner_refuses_to_bill_a_child_on_a_nonfinite_number(
    location: str,
    key: str,
    bad: float,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No spawn, no raise, and the reason is on the record.

    ``maybe_spawn_structural_relief_run`` runs on the parent's COMPLETION path and
    swallows every exception, so a crash here is invisible: the child silently never
    appears and nothing says why. The refusal is explicit instead — and it withholds a
    paid render rather than claiming the render is impossible.
    """
    _patch_settings(monkeypatch)
    run_block = _run_block()
    relief_cfg: dict[str, Any] = {"pass": 0, "prior_overrun": 20.0}
    if location == "run":
        run_block[key] = bad
    else:
        relief_cfg[key] = bad

    engine = SimpleNamespace(start_run=AsyncMock())
    with caplog.at_level(logging.WARNING):
        spawned = await maybe_spawn_structural_relief_run(
            run_engine=engine,  # ty: ignore[invalid-argument-type]
            parent_run_id=uuid7(),
            deliverable_id=uuid7(),
            config_snapshot={
                "tenant_id": str(uuid7()),
                "subject_id": str(uuid7()),
                "direction": {"timing": {"mode": "auto", "relief": {"mode": "auto"}}},
                "structural_relief": relief_cfg,
            },
            stage_rows=_stage_rows(run_block),
        )

    assert spawned is None
    engine.start_run.assert_not_awaited()
    assert any("non-finite" in record.getMessage() for record in caplog.records), (
        "the refusal must say WHY — a silent no-spawn is indistinguishable from a "
        "healthy 'nothing to do'"
    )
