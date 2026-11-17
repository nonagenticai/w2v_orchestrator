"""A non-finite ORDER is NO order — the stage refuses it exactly as it refuses an absent one.

This is the boundary of an input-reachable crash on a billing-relevant path, and no part
of it needs a corrupt database:

1. ``POST /api/runs`` takes ``config_overrides: dict`` with no validation and no
   ``allow_inf_nan=False`` anywhere in the stack.
2. ``json.loads("1e400")`` is a legal JSON number and evaluates to ``inf``; jsonb
   round-trips it whole, and ``direction`` is copied through verbatim.
3. ``_resolve_timing_config`` read it with a bare ``float(raw_target)`` and stamped it on
   ``_TimingConfig.target``.
4. From there it reaches ``manifest["run"]["target_seconds"]`` and, in the SAME process,
   ``relief_shape_from_manifest`` -> ``plan_beats`` -> ``math.ceil``, which raises
   ``ValueError`` (``nan``) / ``OverflowError`` (``inf``).

Every ``<= 0.0`` guard on that path was blind to it: each comparison against ``nan`` is
False, so ``nan`` passed every one of them.

What is pinned here:

* The stage SKIPS — ``StageSkippedError``, the identical refusal it already gives when
  there is no target at all. Not a crash, and above all not a substituted default: a
  clamp to a stock duration would manufacture an order nobody placed, which is the exact
  defect this arithmetic exists to delete.
* The ladder is unchanged — an unreadable explicit override is never quietly demoted to
  the deliverable's target (that would swap one order for a different one).
* A finite order still resolves byte-identically.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

NAN = float("nan")
INF = float("inf")


def _ctx(config_snapshot: dict[str, Any]) -> Any:
    """``_resolve_timing_config`` is pure over ``ctx.config_snapshot``."""
    from types import SimpleNamespace

    return cast("Any", SimpleNamespace(config_snapshot=config_snapshot))


_FLOOR_RATIO = 0.5
#: 6 beats x 7.00s measured against a 30.00s order overrunning by 12.00s: a 4-beat spine
#: is 28.00s, inside the +/-10% band, so a legal relief cut EXISTS and the shape SPAWNS.
#: The guard is only observable where the honest answer is not already "stop".
_SPAWNABLE_BEATS = [
    {"id": f"clip-{i}", "min_seconds": 7.0, "max_seconds": 7.0, "seconds": 7.0} for i in range(6)
]


def _manifest(target_seconds: float) -> dict[str, Any]:
    return {
        "run": {
            "target_seconds": target_seconds,
            "overrun_seconds": 12.0,
            "convergence_outcome": "cant_converge",
            "structural_relief_requested": True,
            "protected_beat_count": 0,
            "infeasible_beat_ids": ["clip-0", "clip-1"],
        },
        "beats": _SPAWNABLE_BEATS,
    }


def _note_for(manifest: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> str:
    """``_cant_converge_note`` touches no ``self`` state, so no stage wiring is needed."""
    from types import SimpleNamespace

    from src.services.stages.demo_video.reconcile_timeline import ReconcileTimelineStage

    monkeypatch.setattr(
        "src.services.stages.demo_video.reconcile_timeline.get_settings",
        lambda: SimpleNamespace(
            pipeline=SimpleNamespace(
                structural_relief_autoconverge_enabled=True,
                structural_relief_max_passes=2,
                demo_timeline_underfull_fail_ratio=_FLOOR_RATIO,
            )
        ),
    )
    stage = object.__new__(ReconcileTimelineStage)
    return stage._cant_converge_note(
        _ctx({"structural_relief": {"pass": 0}}),
        overrun_seconds=12.0,
        rerun_stages=["demo_video_gen"],
        structural_relief_requested=True,
        manifest=manifest,
    )


def test_a_spawnable_manifest_does_promise_a_re_render(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CONTROL. Without it the fail-closed test below proves nothing — a note that
    never promises a re-render would pass it for free."""
    note = _note_for(_manifest(30.0), monkeypatch)
    assert "re-render will be attempted" in note
    assert "no automatic re-render will follow" not in note


@pytest.mark.parametrize("bad", [NAN])
def test_an_unreadable_order_never_promises_a_paid_re_render(
    bad: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FAIL-CLOSED at the caller: ``None`` from the shape is NOT a licence to spawn.

    ``shape_allows_spawn = shape is not None and shape.verdict == "SPAWN"`` — the
    ``is not None`` half is the whole point and is what this pins. ``decide_structural
    _relief`` still answers ``SPAWN`` here (kill-switch armed, pass 0 of 2, a real
    overrun), so the ONLY thing standing between an unreadable order and a promise of a
    PAID child render is that conjunct. Written as ``shape is None or shape.verdict ==
    "SPAWN"``, or with the ``None`` coerced to a shape, the corrupt order would buy a
    render.

    Note what the refusal says: "no automatic re-render will follow" — a statement about
    what this system will DO, and the same wording a capped or unshrinkable run gets. It
    does not tell the user the render is impossible.
    """
    note = _note_for(_manifest(bad), monkeypatch)
    assert "no automatic re-render will follow" in note
    assert "re-render will be attempted" not in note
    # …and it still routes the user to the lever that does work.
    assert "re-run demo_video_gen" in note


# ════════════════════════════════════════════════════════════════════════════════
# O9 — THE SAME BOUNDARY, APPLIED TO THE TUNABLES (A/V quality wave 13)
#
# Until wave 13 `_resolve_timing_config` guarded exactly ONE field —
# `target_seconds` — while its four siblings were read with a bare `float()`/`int()`
# off the same unvalidated `config_overrides` blob. Every test above proves the
# guarded field; none proved the unguarded ones, which is why they stayed unguarded
# through three passes over this function.
# ════════════════════════════════════════════════════════════════════════════════


def _timing(**knobs: object) -> Any:
    return _ctx({"direction": {"timing": {"mode": "auto", "target_seconds": 30.0, **knobs}}})
