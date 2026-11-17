"""Guard: the LLM-stage timeout ladder stays coherent (outer > inner).

Every LLM text stage runs its WM call under ``stage_call`` with the shared
``llm_budget_seconds()`` per-attempt budget, wrapped by the blueprint's outer
``TimeoutMiddleware(default_timeout_seconds)``. If the outer ceiling drops at or
below the inner budget, the outer wrapper cancels a still-working pgx call before
``stage_call``'s own (clean) timeout can fire — the exact inverted ladder that
terminal-failed slow-but-working runs. This test walks every registered blueprint
and asserts that invariant for every stage that actually uses the LLM budget, so
a future stage (or a lowered outer) can't silently re-invert it.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys

import pytest
from pydantic import ValidationError
from w2v_common.stages.execution import _validate_retries

import src.services.blueprints as blueprints_pkg
from src.config import (
    LLM_CLIENT_TIMEOUT_DELTA_S,
    LLM_DOWNSTREAM_FLOOR_S,
    LLM_MIN_CLIENT_SHARE,
    LLM_STAGE_BASE_CEILING_S,
    STAGE_CALL_MAX_RETRIES,
    PipelineSettings,
    client_timeout_for,
)
from src.services.blueprints.base import StageSpec
from src.services.stages.llm_budget import llm_budget_seconds, llm_client_timeout_seconds

# Outer must exceed the inner WORST-CASE budget by at least this margin so
# stage_call's StageTimeoutError fires before the outer TimeoutMiddleware.
_MIN_OUTER_MARGIN_SECONDS = 30.0

# Extra slack for a MULTI-CALL (evaluator-optimizer) stage on top of its
# calls x attempts x budget product — the loop also does deterministic work
# (analyzer jobs, merges, a re-compile) between the LLM windows.
_LOOP_OUTER_MARGIN_SECONDS = 100.0

# ── KNOWN UNENFORCED INVARIANT: the multi-call class ─────────────────────────
# ⚠️ This is a BACKLOG, not a guard. Read the whole block before adding to it.
#
# The ladder guard below is per-STAGE_CALL: it sees the shared budget and the
# outer ceiling, but it cannot see how MANY sequential ``stage_call`` windows a
# handler opens. A stage driven by ``run_evaluator_optimizer``
# (src/services/stages/shared/loop_engine.py) opens THREE — critic pass 1
# (:229) -> optimizer (:238) -> critic pass 2 (:245), ``max_passes`` hard-clamped
# to 2 at :182 — and two handlers open a fourth window outside the loop. So the
# worst case for these stages is ``calls x attempts x budget``, and every pinned
# ceiling below is under it.
#
# That is not cosmetic. A TimeoutMiddleware cut arrives as ``CancelledError``, a
# **BaseException**, which these stages' soft-degrade ``except Exception``
# handlers CANNOT catch — so an outer-first fire HARD-FAILS a paid run instead of
# degrading. Only a clean inner ``StageTimeoutError`` degrades correctly.
#
# WHY IT IS NOT FIXED HERE. The shortfall is PRE-EXISTING and fleet-wide: at the
# old 660s budget, 3 x 2 x 660 = 3960s already exceeded a 2500s pin. The
# 660 -> 950 budget raise makes it worse but did not create it. Closing it means
# ceilings of 5800-7700s, which would let a genuinely hung PAID stage hold for up
# to 2.1h and would break the premise of ``reconciler_max_age_seconds = 7200``
# (src/config.py:774, "covers the slowest blueprint ~75 min"). That needs its own
# PR and its own blast-radius pass, not a rider on a budget change.
#
# WHAT IS ENFORCED. ``test_loop_stage_call_backlog_is_current`` below detects loop
# stages from the handler's MRO source, so a NEW loop stage FAILS on omission
# rather than passing invisibly, a stale entry FAILS, and an entry whose ceiling
# has since been raised to cover FAILS so it gets deleted. Following the
# ``_RUNTIME_DEPS_ONLY_STAGES`` convention, every entry carries a mandatory
# reason: the file:line that proves the call count. A future PR empties this dict.
_LOOP_STAGE_CALLS_BACKLOG: dict[str, tuple[int, str]] = {
    # NOTE: `script_doctor` is deliberately ABSENT — it is the one loop stage whose
    # pin (5800s, demo_video.py / showcase.py) already covers 3 x 2 x 950 + 100. The
    # `covered` assertion below is what keeps that claim honest.
    "claim_finish": (
        3,
        "loop only — run_evaluator_optimizer at stages/demo_video/claim_finish.py:582",
    ),
    "retention_plan": (
        3,
        "loop only — run_evaluator_optimizer at stages/demo_video/retention_plan.py:408",
    ),
    "reconcile_timeline": (
        4,
        "loop (run_evaluator_optimizer in stages/demo_video/reconcile_timeline.py) PLUS "
        "one stage_call window in _apply_structural_prepass (the guarded structural "
        "timing-director call) that runs BEFORE the loop. It was a BARE self._wm.message "
        "bounded only by WMClient.timeout; it is now stage_call(guarded(...)) with the "
        "shared budget + llm_retries, which is what this entry's `attempts` factor "
        "already assumed.",
    ),
    "final_critique": (
        4,
        "loop (stages/demo_video/final_critique.py:1449) PLUS the initial "
        "_reevaluate -> _critic stage_call (:1787) that runs before _auto_redrive",
    ),
    "audiobook_qa": (
        3,
        "loop only — run_evaluator_optimizer at stages/audiobook/qa.py:880 (the "
        "_wm.message at :715 is nested inside the loop's critic closure)",
    ),
    "demo_video_footage_critique": (
        3,
        "loop only — run_evaluator_optimizer at stages/shared/footage_critique.py:719 "
        "(the batched _wm.message at :573 is nested inside critic_fn)",
    ),
    "promo_footage_critique": (
        3,
        "same handler as demo_video_footage_critique — "
        "run_evaluator_optimizer at stages/shared/footage_critique.py:719",
    ),
    "demo_video_encode_master": (
        3,
        "loop only — run_evaluator_optimizer at stages/demo_video/encode_master.py:563 "
        "(the :450 call passes critic_fn=None => zero awaits)",
    ),
}

# Stages whose handler_factory legitimately REFUSES to build from an empty deps
# mapping because it needs live run-time deps (run_engine / redis /
# session_factory). They are not on the LLM budget ladder, so excluding them is
# correct — but the exclusion is PINNED here rather than inferred from "it threw
# something".
#
# This used to be a blanket ``except Exception: continue``. Under it, any stage
# that started raising for a REAL reason (a broken import, a renamed setting, a
# constructor bug) quietly deleted itself from the invariant set and the ladder
# guard kept passing on a shrinking population — coverage erosion with zero
# signal. Now anything that raises and is not listed here fails the test loudly.
#
# Adding an entry is a deliberate act: confirm the stage genuinely needs run-time
# deps, do not use it to silence a handler that is actually broken.
_RUNTIME_DEPS_ONLY_STAGES: frozenset[tuple[str, str]] = frozenset(
    {
        ("course", "course_compile_lessons"),
    }
)


def _all_registered_blueprints() -> list:
    # Import every blueprint module so it registers itself, then read the
    # registry — new blueprints are covered automatically.
    for mod in pkgutil.iter_modules(blueprints_pkg.__path__):
        if mod.name not in ("base", "registry"):
            importlib.import_module(f"src.services.blueprints.{mod.name}")
    from src.services.blueprints.registry import get_registry

    return get_registry().all()


def _llm_stages() -> list[tuple[str, str, int]]:
    """(project_type, stage_name, outer_timeout) — the handler-carrying walk, projected."""
    return [(pt, name, outer) for pt, name, outer, _ in _ladder_stages()]


def _ladder_stages() -> list[tuple[str, str, int, object]]:
    """(project_type, stage_name, outer_timeout) for LLM budget stages.

    A stage is "on the LLM ladder" iff its handler builds a ``_wm`` client whose
    timeout equals the shared CLIENT rung — i.e. it went through
    ``llm_client_timeout_seconds()``.

    ⚠️ This used to match ``llm_budget_seconds()`` — the per-attempt STAGE budget.
    That was the same number, and that identity was the defect: a client handed the
    budget the stage clock is already counting down can never raise its retryable
    ``httpx.ReadTimeout`` (the stage clock starts first and wins every tie). Now the
    client rung sits strictly below the budget, so detect on the rung the handlers
    actually pass.

    Handlers that cannot be built from an empty deps mapping are excluded ONLY if
    they are pinned in ``_RUNTIME_DEPS_ONLY_STAGES``. Any other construction
    failure is reported loudly instead of silently shrinking the invariant set.
    """
    client_rung = llm_client_timeout_seconds()
    found: list[tuple[str, str, int, object]] = []
    broken: list[str] = []
    excluded_seen: set[tuple[str, str]] = set()
    for bp in _all_registered_blueprints():
        for spec in bp.stages:
            key = (bp.project_type, spec.name)
            try:
                handler = spec.handler_factory({})
            except Exception as exc:  # noqa: BLE001 -- classified below, never swallowed
                if key in _RUNTIME_DEPS_ONLY_STAGES:
                    excluded_seen.add(key)
                    continue
                broken.append(
                    f"{bp.project_type}:{spec.name} handler_factory raised "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            wm = getattr(handler, "_wm", None)
            wm_timeout = getattr(wm, "timeout", None) if wm is not None else None
            if isinstance(wm_timeout, (int, float)) and wm_timeout == client_rung:
                # None on the spec => inherit the runtime base ceiling; resolve it
                # to the shipped default so the ladder invariant is checked on the
                # value the executor would actually use.
                outer = (
                    spec.default_timeout_seconds
                    if spec.default_timeout_seconds is not None
                    else LLM_STAGE_BASE_CEILING_S
                )
                found.append((bp.project_type, spec.name, outer, handler))
    if broken:
        pytest.fail(
            "stage handler(s) failed to construct, so the LLM timeout-ladder guard "
            "could not inspect them — a stage that cannot be built silently drops "
            "out of the invariant set. Fix the handler; only add it to "
            "_RUNTIME_DEPS_ONLY_STAGES if it genuinely requires live run-time "
            "deps:\n  " + "\n  ".join(broken),
            pytrace=False,
        )
    stale = _RUNTIME_DEPS_ONLY_STAGES - excluded_seen
    if stale:
        pytest.fail(
            "_RUNTIME_DEPS_ONLY_STAGES has entries that no longer raise (they now "
            "build fine, or the stage was renamed/removed). A stale exclusion list "
            "is how a real breakage gets pre-authorised — remove: "
            f"{sorted(stale)}",
            pytrace=False,
        )
    return found


def test_llm_ladder_outer_exceeds_inner_budget() -> None:
    # Built at test time (not collection time): constructing handlers reads
    # get_settings(), which needs the env the conftest fixtures provide.
    stages = _llm_stages()
    # Detection sanity: if this ever returns nothing the guard would be vacuous.
    assert stages, "no LLM-budget stages detected — the ladder guard would be vacuous"

    budget = llm_budget_seconds()
    # ATTEMPTS-AWARE. A stage is `llm_retries + 1` SEQUENTIAL stage_call attempts, and
    # the outer ceiling is a per-STAGE TimeoutMiddleware budget covering all of them —
    # so the thing the outer has to clear is `budget * attempts`, never one attempt.
    # Checking one attempt is how the shipped 660 x 2 = 1320 under a 900 ceiling passed:
    # attempt 2 could never finish and the blunt outer abort fired on every retry. That
    # abort is a CancelledError (BaseException), which the stages' soft-degrade
    # `except Exception` cannot catch — so an outer-first fire HARD-FAILS a paid run
    # instead of degrading. Only a clean inner StageTimeoutError degrades correctly.
    attempts = PipelineSettings().llm_retries + 1
    required = budget * attempts + _MIN_OUTER_MARGIN_SECONDS
    violations = [
        f"{project_type}:{stage_name} outer={outer}s < budget {budget}s x {attempts} "
        f"attempts + {_MIN_OUTER_MARGIN_SECONDS}s margin = {required}s"
        for project_type, stage_name, outer in stages
        if outer < required
    ]
    assert not violations, (
        "Inverted LLM timeout ladder — the outer TimeoutMiddleware would cut off the "
        "WM call before stage_call's own timeout fires:\n  " + "\n  ".join(violations)
    )


def _is_loop_stage(handler: object) -> bool:
    """True iff the handler's own module drives ``run_evaluator_optimizer``.

    Read off the SOURCE, not off a hand-kept list, so a newly-added loop stage is
    detected the moment it lands — that is what makes the mapping below fail on
    omission instead of silently not covering it.
    """
    # Walk the MRO, not just type(handler).__module__: the footage-critique family
    # subclasses a SHARED loop handler (stages/shared/footage_critique.py) from a thin
    # per-family module, so reading only the leaf module misses the loop entirely.
    for klass in type(handler).__mro__:
        module = sys.modules.get(klass.__module__)
        if module is None:
            continue
        try:
            src = inspect.getsource(module)
        except OSError, TypeError:  # pragma: no cover -- builtins have no source
            continue
        if "run_evaluator_optimizer(" in src:
            return True
    return False


def test_loop_stage_call_backlog_is_current() -> None:
    """The KNOWN-unenforced multi-call shortfall stays visible, complete and honest.

    This does NOT assert the ceilings are correct — they are not, deliberately (see
    ``_LOOP_STAGE_CALLS_BACKLOG``). It asserts the backlog cannot rot in either
    direction: a NEW loop stage must be counted rather than silently absent, a
    removed/renamed one must be deleted, and an entry whose ceiling has since been
    raised to cover its loop must be deleted too — that is how the list drains.
    """
    stages = _ladder_stages()
    assert stages, "no LLM-budget stages detected — this guard would be vacuous"

    budget = llm_budget_seconds()
    attempts = PipelineSettings().llm_retries + 1

    unmapped: list[str] = []
    covered: list[str] = []
    seen: set[str] = set()
    for project_type, stage_name, outer, handler in stages:
        if not _is_loop_stage(handler):
            continue
        entry = _LOOP_STAGE_CALLS_BACKLOG.get(stage_name)
        if entry is None:
            required_if_new = 3 * attempts * budget + _LOOP_OUTER_MARGIN_SECONDS
            if outer < required_if_new:
                unmapped.append(f"{project_type}:{stage_name} outer={outer}s")
            continue
        calls, _reason = entry
        seen.add(stage_name)
        required = calls * attempts * budget + _LOOP_OUTER_MARGIN_SECONDS
        if outer >= required:
            covered.append(
                f"{project_type}:{stage_name} outer={outer}s now covers {calls} calls x "
                f"{attempts} attempts x {budget}s + {_LOOP_OUTER_MARGIN_SECONDS}s "
                f"= {required}s"
            )

    assert not unmapped, (
        "these stages drive run_evaluator_optimizer, carry no _LOOP_STAGE_CALLS_BACKLOG "
        "entry, and their ceiling does not cover even a bare 3-call loop — so the "
        "shortfall would be invisible. Count the SEQUENTIAL stage_call windows the handler "
        "opens (the loop alone is 3: critic -> optimizer -> critic) and add an entry with "
        f"the file:line that proves it: {sorted(unmapped)}"
    )
    assert not covered, (
        "these _LOOP_STAGE_CALLS_BACKLOG entries are FIXED — their pinned ceiling now "
        "covers the whole loop, so leaving them listed overstates the outstanding debt. "
        f"Delete them: {sorted(covered)}"
    )
    stale = set(_LOOP_STAGE_CALLS_BACKLOG) - seen
    assert not stale, (
        "_LOOP_STAGE_CALLS_BACKLOG has entries for stages that are no longer detected as "
        "loop stages (renamed, removed, or they stopped using run_evaluator_optimizer). A "
        f"stale backlog overstates the debt — remove: {sorted(stale)}"
    )


def test_base_ceiling_default_is_the_inherit_sentinel() -> None:
    """LLM stages carry the ``None`` inherit-sentinel, not a duplicated 700 literal.

    The base ceiling now lives in a single env-configurable place
    (``PipelineSettings.llm_stage_base_ceiling_s``, default == module
    ``LLM_STAGE_BASE_CEILING_S``); the DAG executor resolves a ``None`` spec to it.
    Re-hardcoding a bare number on the spec default would silently make the GitOps
    knob inert again — this guards against that.
    """
    assert StageSpec.__dataclass_fields__["default_timeout_seconds"].default is None
    assert PipelineSettings().llm_stage_base_ceiling_s == LLM_STAGE_BASE_CEILING_S


def test_default_budget_sits_inside_the_ladder_window() -> None:
    # The shipped default must satisfy floor <= budget < ceiling; if a future
    # slow-vLLM raise pushes the budget up it must move the ceiling too.
    settings = PipelineSettings()
    assert (
        settings.llm_downstream_floor_s
        <= settings.llm_budget_seconds
        < (settings.llm_stage_base_ceiling_s)
    )


def test_ceiling_and_budget_raised_together_accepted() -> None:
    # A genuine slow-vLLM raise: lift BOTH the ceiling and the budget. The ceiling must
    # clear budget x attempts, not merely the budget -- 880 x 2 = 1760, so 1800.
    settings = PipelineSettings(llm_stage_base_ceiling_s=1800, llm_budget_seconds=880.0)
    assert settings.llm_budget_seconds == 880.0
    assert settings.llm_stage_base_ceiling_s == 1800


def test_budget_at_or_above_raised_ceiling_rejected() -> None:
    # Raising the budget to/above even a RAISED ceiling still inverts the ladder.
    with pytest.raises(ValidationError):
        PipelineSettings(llm_stage_base_ceiling_s=900, llm_budget_seconds=900.0)


def test_ceiling_must_cover_every_attempt_not_just_one() -> None:
    """The bug this file did not catch for as long as it existed.

    A stage is ``llm_retries + 1`` sequential ``stage_call`` attempts and the ceiling is
    the OUTER per-stage TimeoutMiddleware budget, so it has to cover all of them. The
    shipped config was 660 x 2 = 1320 against a live ceiling of 900: attempt 2 could
    never finish, and the blunt outer abort this whole ladder exists to prevent fired on
    every retry. The old validator accepted it because it only checked one attempt.
    """
    # One attempt fits (880 < 900) but two do not (1760 >= 900) -> must be REJECTED.
    with pytest.raises(ValidationError, match="x 2 attempts"):
        PipelineSettings(llm_stage_base_ceiling_s=900, llm_budget_seconds=880.0, llm_retries=1)
    # The SAME budget and ceiling with the retry removed is fine -- proving the
    # rejection is about the attempt count and not about the two numbers alone.
    ok = PipelineSettings(llm_stage_base_ceiling_s=900, llm_budget_seconds=880.0, llm_retries=0)
    assert ok.llm_budget_seconds == 880.0


def test_shipped_default_ladder_covers_all_attempts() -> None:
    """The shipped defaults must satisfy the product rule, not just the window.

    Mutation-check for this file: drop ``LLM_STAGE_BASE_CEILING_S`` back to 900 (or
    raise ``llm_budget_seconds`` past half the ceiling) and this goes red.
    """
    s = PipelineSettings()
    attempts = s.llm_retries + 1
    assert s.llm_budget_seconds * attempts < s.llm_stage_base_ceiling_s, (
        f"{s.llm_budget_seconds} x {attempts} attempts does not fit under the outer "
        f"ceiling {s.llm_stage_base_ceiling_s}"
    )


def test_lowered_floor_via_env_accepted() -> None:
    # The floor is configurable too; a budget above a lowered floor is valid.
    settings = PipelineSettings(llm_downstream_floor_s=500.0, llm_budget_seconds=550.0)
    assert settings.llm_budget_seconds == 550.0


@pytest.mark.parametrize(
    "bad_budget",
    [LLM_DOWNSTREAM_FLOOR_S - 1.0],
)
def test_budget_outside_window_is_rejected_at_boot(bad_budget: float) -> None:
    with pytest.raises(ValidationError):
        PipelineSettings(llm_budget_seconds=bad_budget)


def test_budget_at_the_floor_plus_the_client_delta_is_accepted() -> None:
    """The lower bound moved onto the CLIENT rung, and this test moved with it.

    It used to assert that ``budget == llm_downstream_floor_s`` boots. That was only
    coherent while the client rung did not exist: the client is what actually holds the
    socket, so IT is the thing that must not cut a still-working call off before the
    downstream read budget. With ``client = budget - delta``, a budget sitting exactly ON
    the floor puts the client BELOW it — the very failure the floor exists to prevent.
    The smallest legal budget is therefore ``floor + delta``, and that is what is pinned.
    """
    smallest = LLM_DOWNSTREAM_FLOOR_S + LLM_CLIENT_TIMEOUT_DELTA_S
    settings = PipelineSettings(llm_budget_seconds=smallest)
    assert settings.llm_budget_seconds == smallest
    assert settings.llm_client_timeout_seconds == LLM_DOWNSTREAM_FLOOR_S

    # ...and one second under it is rejected, so the boundary is a real edge and not a
    # number this test happens to agree with.
    with pytest.raises(ValidationError):
        PipelineSettings(llm_budget_seconds=smallest - 1.0)


# ---------------------------------------------------------------------------
# The CLIENT rung — strictly below the per-attempt timeout it is derived FROM.
# ---------------------------------------------------------------------------
#
# ⚠️ FIRST, THE CORRECTION. The rung shipped with this justification:
#
#     WMClient's read timeout and stage_call's per-attempt budget were both 950.0;
#     the stage clock starts first; so httpx.ReadTimeout is never raised and the
#     in-stage retry was DEAD CODE — segment_select could only ever die terminally.
#
# The last clause is FALSE, and ``test_stage_call_retries_its_own_per_attempt_timeout``
# below proves it against the real w2v_common implementation rather than a paraphrase.
# ``stage_call`` catches its OWN ``TimeoutError`` and retries it; only the final attempt
# raises ``StageTimeoutError``. The evidence quoted at the time — "all 2 attempts timed
# out" — is stage_call reporting that it retried twice.
#
# So the rung is defence in depth (a typed, classifiable ``httpx.ReadTimeout``; a socket
# bound for any call path not wrapped in ``stage_call``/``asyncio.timeout``), NOT a
# resurrected retry — and because it is not load-bearing, the slack it takes out of
# GENERATION time must be small. Hence an absolute delta, not a fraction.


def test_stage_call_retries_its_own_per_attempt_timeout() -> None:
    """The premise-check. A per-attempt timeout is RETRIED, not terminal.

    This is asserted against the deployed ``w2v_common`` source, so if that behaviour
    ever changes the comment block above stops being true loudly instead of quietly.
    """
    import inspect as _inspect

    from w2v_common.stages.execution import stage_call

    src = _inspect.getsource(stage_call)
    assert "except TimeoutError" in src, (
        "stage_call no longer catches its own TimeoutError — the whole rationale block "
        "above (and in src/config.py) depends on it"
    )
    # The terminal raise is GUARDED by the last-attempt check; without that guard a
    # timeout really would be terminal on attempt 1.
    body = src[src.index("except TimeoutError") :]
    assert "if attempt == attempts:" in body[: body.index("raise StageTimeoutError")], (
        "StageTimeoutError is no longer gated on the LAST attempt"
    )


@pytest.mark.asyncio
async def test_stage_call_retry_on_timeout_is_observable_not_just_textual() -> None:
    """The same fact, executed rather than read: attempt 1 times out, attempt 2 returns."""
    import asyncio as _asyncio

    from w2v_common.stages.execution import stage_call

    calls = {"n": 0}

    async def _flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            await _asyncio.sleep(10.0)  # blows the 1s per-attempt budget
        return "ok"

    result = await stage_call(
        _flaky,
        timeout_seconds=1.0,
        retries=1,
        backoff_initial=0.0,
        backoff_jitter=0.0,
        stage_name="premise_check",
    )
    assert result == "ok"
    assert calls["n"] == 2, "a timed-out attempt was not retried"


# ── the derivation ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("effective", [1.0])
def test_client_timeout_for_is_strictly_below_and_at_least_half_its_input(
    effective: float,
) -> None:
    """``t * min_share <= client_timeout_for(t) < t`` for every positive ``t``.

    The upper half is the ordering invariant. The lower half is what stops the delta
    inverting on a SUBDIVIDED slice: a flat 20s off a 30s judge batch would leave 10s and
    make the client the binding constraint — the fraction's mistake, mirrored.
    """
    rung = client_timeout_for(effective)
    assert rung < effective
    assert rung >= effective * LLM_MIN_CLIENT_SHARE


def test_client_timeout_for_rejects_a_non_positive_input() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        client_timeout_for(0.0)


def test_the_absolute_delta_dominates_at_every_realistic_stage_budget() -> None:
    """At stage scale the rung is ``t - delta``; the half-share only guards tiny slices.

    Mutation-check: turn the delta back into a 0.9 fraction and the 950s row goes red at
    855 instead of 930.
    """
    for effective in (600.0, 950.0, 1200.0, 3600.0):
        assert client_timeout_for(effective) == effective - LLM_CLIENT_TIMEOUT_DELTA_S
    # Below 2 * delta the share takes over, and it is the LARGER of the two.
    assert client_timeout_for(30.0) == 15.0


def test_the_delta_does_not_steal_meaningful_generation_time() -> None:
    """DEFECT 1. The slack must cover PROPAGATION, not a slice of the model's output.

    A 0.9 fraction put the rung at 855s under a 950s budget, so a call needing 855-950s
    — which used to SUCCEED — was aborted and retried; ``stage_call`` hands attempt 2 a
    FRESH full budget, so the same prompt burns 855s again and the stage dies at ~1711s
    instead of returning at ~900s. The deployed local model was just swapped for a much
    slower one, i.e. that band is where calls now sit.

    Two things are pinned: the slack is a small ABSOLUTE number, and it is small relative
    to the budget. Mutation-check: restore ``fraction=0.9`` (slack 95s) and both fail.
    """
    s = PipelineSettings()
    slack = s.llm_budget_seconds - s.llm_client_timeout_seconds
    assert slack == LLM_CLIENT_TIMEOUT_DELTA_S == 20.0
    assert slack <= 30.0, f"the client rung forfeits {slack}s of generation time"
    assert slack / s.llm_budget_seconds < 0.05, (
        f"the client rung forfeits {slack / s.llm_budget_seconds:.1%} of the per-attempt "
        "budget; a fraction-shaped rung is what this bound exists to reject"
    )
    # The concrete regression: a call in the old dead band still completes.
    assert s.llm_client_timeout_seconds > 900.0


def test_a_raised_budget_does_not_raise_the_slack() -> None:
    """The delta is ABSOLUTE, so a slower model does not forfeit proportionally more.

    Under the 0.9 fraction, raising the budget to 1900s for a slower model would have
    thrown away 190s per attempt. Mutation-check: reinstate the fraction and the equality
    below becomes 1710.0 != 1880.0.
    """
    s = PipelineSettings(llm_budget_seconds=1900.0, llm_stage_base_ceiling_s=3900)
    assert s.llm_client_timeout_seconds == 1900.0 - LLM_CLIENT_TIMEOUT_DELTA_S
    assert s.llm_budget_seconds - s.llm_client_timeout_seconds == LLM_CLIENT_TIMEOUT_DELTA_S


def test_client_rung_is_strictly_below_the_per_attempt_budget() -> None:
    """The ordering invariant on the shipped defaults.

    Mutation-check: set ``llm_client_timeout_delta_s=0.0`` (the two rungs equal, which is
    exactly what shipped originally) and this goes red — via the Field constraint.
    """
    s = PipelineSettings()
    assert s.llm_client_timeout_seconds < s.llm_budget_seconds


def test_full_ladder_ordering_holds_on_the_shipped_defaults() -> None:
    """floor <= client rung < per-attempt budget, and budget * attempts <= outer ceiling."""
    s = PipelineSettings()
    attempts = s.llm_retries + 1
    assert s.llm_downstream_floor_s <= s.llm_client_timeout_seconds < s.llm_budget_seconds
    assert s.llm_budget_seconds * attempts <= s.llm_stage_base_ceiling_s


def test_client_rung_tracks_a_raised_budget() -> None:
    # The rung is DERIVED from the budget, so raising the budget for a slower model can
    # never leave the client stranded at the old (now equal-or-higher) value.
    s = PipelineSettings(llm_budget_seconds=1200.0, llm_stage_base_ceiling_s=2500)
    assert s.llm_client_timeout_seconds == pytest.approx(1200.0 - s.llm_client_timeout_delta_s)
    assert s.llm_client_timeout_seconds < s.llm_budget_seconds


# ── BOTH bounds, at boot ────────────────────────────────────────────────────


def test_a_zero_delta_is_rejected_at_boot() -> None:
    # delta == 0 makes the two rungs identical. That is the original bug; it must be
    # impossible to configure, not merely discouraged.
    with pytest.raises(ValidationError):
        PipelineSettings(llm_client_timeout_delta_s=0.0)


def test_a_negative_delta_is_rejected_at_boot() -> None:
    with pytest.raises(ValidationError):
        PipelineSettings(llm_client_timeout_delta_s=-5.0)


def test_a_client_rung_below_the_downstream_floor_is_rejected_at_boot() -> None:
    """DEFECT 3. The validator had NO lower bound, so it was asymmetric.

    It rejected ``client >= budget`` and said nothing about ``client`` being far too
    SMALL: ``PIPELINE_LLM_CLIENT_TIMEOUT_FRACTION=0.5`` booted clean at 475s — a config
    that starts up happily and then cuts every still-working call off 125s under the
    downstream read budget, i.e. it boots straight into a broken ladder, the exact class
    the ladder exists to prevent.

    The equivalent under the delta shape is a large delta. Mutation-check: delete the
    LOWER-bound branch in ``_validate_llm_timeout_ladder`` and this goes red.
    """
    with pytest.raises(ValidationError, match="below the downstream read budget"):
        PipelineSettings(llm_client_timeout_delta_s=400.0)  # 950 - 400 = 550 < 600

    # And the ORIGINAL reported configuration, translated: half the budget.
    with pytest.raises(ValidationError, match="below the downstream read budget"):
        PipelineSettings(llm_client_timeout_delta_s=475.0)  # 950 - 475 = 475, the 0.5 case


def test_the_lower_bound_moves_with_a_lowered_floor() -> None:
    # All three knobs are env-configurable, so the guard must be relative, not a literal.
    s = PipelineSettings(llm_downstream_floor_s=400.0, llm_client_timeout_delta_s=400.0)
    assert s.llm_client_timeout_seconds == 550.0
    with pytest.raises(ValidationError, match="below the downstream read budget"):
        PipelineSettings(llm_downstream_floor_s=600.0, llm_client_timeout_delta_s=400.0)


def test_llm_retries_is_refused_above_the_stage_call_cap_at_config_load() -> None:
    """``PIPELINE_LLM_RETRIES`` above the library cap must die at BOOT, not mid-run.

    ``stage_call`` runs ``w2v_common.stages.execution._validate_retries``, which raises
    ``ValueError`` above ``_MAX_RETRIES`` INSIDE the stage handler -- at runtime, in
    production, across the ~40 sites that pass ``retries=self._llm_retries``. The field
    bound moves that to config load.

    Note the assertion is on the CAP, not on the ladder: pydantic runs field constraints
    before the ``mode="after"`` ``_validate_llm_timeout_ladder``, so an over-cap value is
    rejected while NAMING the retry limit, instead of failing for the ladder's unrelated
    `budget x attempts` reason (whose error text invites raising the ceiling in lockstep
    -- which is exactly what would make >cap reachable).
    """
    over = STAGE_CALL_MAX_RETRIES + 1
    with pytest.raises(ValidationError, match="less than or equal to"):
        PipelineSettings(llm_retries=over)
    with pytest.raises(ValidationError, match="greater than or equal to"):
        PipelineSettings(llm_retries=-1)


def test_llm_retries_bound_agrees_with_the_library_validator() -> None:
    """The config bound and ``stage_call``'s own check must accept/reject the SAME set.

    Deliberately exercises the library's real ``_validate_retries`` rather than comparing
    the field's ``le`` to the imported constant (which the import makes tautological).
    If someone replaces the imported bound with a hardcoded number that drifts from the
    library, one of these two halves goes red.
    """
    at_cap = STAGE_CALL_MAX_RETRIES
    over_cap = STAGE_CALL_MAX_RETRIES + 1
    # Smallest budget the ladder's OTHER rules allow, so the ceiling needed to hold
    # `at_cap + 1` attempts stays as small as possible and this test isolates the cap.
    budget = LLM_DOWNSTREAM_FLOOR_S + LLM_CLIENT_TIMEOUT_DELTA_S

    # AT the cap: the library accepts it, so the config must too. A ladder wide enough
    # to hold `at_cap + 1` attempts isolates this to the retry bound.
    assert _validate_retries(at_cap) == at_cap
    ok = PipelineSettings(
        llm_retries=at_cap,
        llm_budget_seconds=budget,
        llm_stage_base_ceiling_s=int(budget * (at_cap + 1)) + 100,
    )
    assert ok.llm_retries == at_cap

    # OVER the cap: the library rejects it, so the config must reject it FIRST.
    with pytest.raises(ValueError, match="retries must be <="):
        _validate_retries(over_cap)
    with pytest.raises(ValidationError):
        PipelineSettings(
            llm_retries=over_cap,
            llm_budget_seconds=budget,
            llm_stage_base_ceiling_s=int(budget * (over_cap + 1)) + 100,
        )
